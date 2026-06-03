// Arm extension lens — "did the straight punch reach full extension?".
//
// The verdict is bend, and bend only: r[f] is the elbow straightness, so a
// peak r ≥ threshold means the arm locked out. But bend is only trustworthy
// when the punch travels across the image plane — a straight thrown toward or
// away from the camera foreshortens and reads wrong. So a single AXIALITY GATE
// fronts the verdict: punches whose forearm points too close to the camera
// axis are skipped (can't judge), the rest are scored on bend alone.
//
// Per-frame metrics (both arms, every frame):
//
//   r[f]     = |shoulder→wrist| / (|shoulder→elbow| + |elbow→wrist|)
//   reach[f] = |shoulder→wrist| / |shoulder_mid → hip_mid|     (Euclidean torso)
//
// Domain of r:
//   1.00  arm dead straight (elbow on the shoulder→wrist line)
//   0.95  ~25° bend at the elbow
//   0.71  90° bend
//   →0    arm fully folded
//
// reach is no longer a gate — it only picks the peak (most-extended) frame
// inside the punch window, the moment bend is read at.
//
// Per labelled punch (straights only — jab/cross head/body):
//   peak          = r[f] at the peak (max-reach) frame in the punch window
//   peak_axiality = axiality (0 = flat across image / sideways, 1 = down the
//                   camera axis) from the trained temporal model's held-out
//                   per-punch prediction, joined by punch_uuid
//   predicted     = skip (axial) / fail (bent) / pass — see rescorePunch
//
// Wrist source: prefer the v6 cache (pose_cache_v6/, Apple Vision skeleton
// with glove wrists baked in at joints 9/10 for gloved rounds — same model
// the iOS app runs live). When the v6 meta says wrist_replaced_with_glove,
// the wrist conf at joints 9/10 IS the glove conf, gated by minGloveConf;
// otherwise it's the Vision wrist gated by minPoseConf. Legacy raw-glove
// sidecar (pose.gloveWrists) is still honored as a fallback for rounds
// that pre-date v6. Anatomical side maps from (hand, stance) the same way
// guard_drop / step_punch_sync do.
//
// Compares predicted vs the labeler's rule_extension verdict when
// available — same agree/disagree pattern as the hip_rotation lens.

import { J } from "../skeleton.js";
import { gloveXY, gloveConf } from "../pose-loader.js";
// Axiality is the sideways gate: the trained temporal model's held-out
// per-punch prediction (predictions_axiality_*.json, loaded by axiality_model.js
// and joined by punch_uuid). 0 = flat across the image (side-on), 1 = down the
// camera axis (foreshortened) — same scale and meaning the gate always used.
import { ensureAxialityModel, axialityForPunch } from "./axiality_model.js";

const DEFAULTS = {
  threshold:        0.95,     // pass if peak ratio ≥ this (geometric straightness)
  // Axiality gate — the only gate. Forearm axiality (0 = flat across the
  // image / sideways, 1 = pointing down the camera axis) measures how
  // foreshortened the punching forearm is. Bend is only trustworthy when the
  // punch travels across the image plane, so punches with axiality ABOVE this
  // cut are skipped (too close to the camera axis to judge). The default
  // ≈0.707 keeps the forearm within ±45° of the image plane (the [45°,135°]
  // off-camera-axis band); the signal is magnitude-only so that band collapses
  // to a single upper cut. Lower = stricter (demand more side-on).
  axialityGate:     true,
  axialityMax:      Math.SQRT1_2,   // ≈0.7071 = cos 45° = sin 45°
  minGloveConf:     0.20,
  minPoseConf:      0.20,
  // Anatomical shoulder correction. COCO labels the shoulder kp at the
  // acromion (top of the shoulder); the glenohumeral joint center is a
  // bit below it. Shifting the shoulder anchor by α × (hip_mid −
  // shoulder_mid), where the offset vector is the round-median (so
  // per-frame hip motion doesn't jitter it), reduces the false bend the
  // acromion-anchor introduces on actually-straight punches.
  shoulderCorrect:      false,
  shoulderCorrectAlpha: 0.20,
  // Only straights have a "should be extended" goal — hooks/uppercuts
  // are supposed to be bent at the elbow.
  appliesTo: new Set([
    "jab_head", "jab_body",
    "cross_head", "cross_body",
  ]),
};

const COLORS = {
  pass:        "#5fd97a",
  fail:        "#e85a5a",
  unclear:     "#f5b945",
  ratioGuide:  "rgba(255,255,255,0.45)",
  poseArm:     "rgba(255,200,80,0.85)",
  gloveArm:    "#d68bff",
  agree:       "#5fd97a",
  disagree:    "#e85a5a",
};

// (hand, stance) → anatomical side, mirroring guard_drop.GUARD_JOINTS logic.
const SIDE_FOR = {
  lead: { orthodox: "L", southpaw: "R" },
  rear: { orthodox: "R", southpaw: "L" },
};

const JOINTS_FOR_SIDE = {
  L: { shoulder: J.L_SHOULDER, elbow: J.L_ELBOW, wrist: J.L_WRIST, gloveSide: 0 },
  R: { shoulder: J.R_SHOULDER, elbow: J.R_ELBOW, wrist: J.R_WRIST, gloveSide: 1 },
};

let host;
let cfg = { ...DEFAULTS };
let signals = null;
let lastPose = null;
let latestState = null;   // live state, for the async model-ready recompute

// v6 cache is the canonical wrist source — it bakes in the glove substitution
// the iOS app does live. Fall back to state.pose for rounds that pre-date v6.
function pickPose(state) {
  return state.poseV6 || state.pose;
}

export const ArmExtensionRule = {
  id: "arm_extension",
  label: "Arm extension (straights)",

  requires(slot) {
    // v6 cache alone is sufficient — it has Vision + glove substitution baked
    // in. For legacy rounds without v6, we still accept raw Vision/YOLO +
    // a glove sidecar so the lens can do the substitution itself.
    return !!slot?.vision_glove
      || (!!(slot?.vision || slot?.yolo) && !!slot?.glove);
  },

  skeletonStyle() {
    return {
      boneColor: "rgba(255,255,255,0.22)",
      boneWidth: 1.5,
      jointRadius: 3,
      // Highlight the joints we use for the ratio, EXCEPT wrists — this lens
      // draws its own wrist markers (square = glove, ring = pose) so letting
      // the base renderer also paint an amber confidence dot at the same
      // spot creates visual noise. Same approach as the wrist_swap lens.
      hideJoints: new Set([J.L_WRIST, J.R_WRIST]),
      highlightJoints: new Set([
        J.L_SHOULDER, J.R_SHOULDER, J.L_ELBOW, J.R_ELBOW,
      ]),
    };
  },

  mount(_host, state) {
    host = _host;
    cfg = { ...DEFAULTS };
    latestState = state;
    // Kick the shared axiality-model loader; onModelReady fires once the
    // per-punch predictions land and recomputes peak_axiality + the verdicts.
    ensureAxialityModel(state, onModelReady);
    signals = computeAll(state, cfg);
    lastPose = pickPose(state);

    host.innerHTML = renderTemplate(signals, cfg);
    renderPunchTable();
    renderAggregate();

    // Threshold slider
    const slider = host.querySelector("#ae-threshold");
    const out    = host.querySelector("#ae-threshold-out");
    if (slider) {
      slider.addEventListener("input", () => {
        cfg.threshold = Number(slider.value);
        out.textContent = cfg.threshold.toFixed(2);
        // Re-score punches against the new threshold — fast, no recompute of
        // r[f]. Routes through rescorePunch so the axiality gate still applies.
        for (const p of signals.punches) rescorePunch(p, cfg);
        renderPunchTable();
        renderAggregate();
        window.__viewerRedraw?.();
      });
    }

    // Shoulder correction toggle + α slider — both require a full recompute
    // of signals (per-frame ratio depends on the corrected shoulder).
    const recomputeAndRefresh = () => {
      signals = computeAll(state, cfg);
      renderPunchTable();
      renderAggregate();
      updateCorrStatus();
      updateGateStatus();
      updateAxialStatus();
      window.__viewerRedraw?.();
    };

    // Confidence-gate status — how many frames pass the current gates
    const updateGateStatus = () => {
      const el = host.querySelector("#ae-gate-status");
      if (!el) return;
      const N = signals.ratioL.length;
      let lOk = 0, rOk = 0;
      for (let f = 0; f < N; f++) {
        if (Number.isFinite(signals.ratioL[f])) lOk++;
        if (Number.isFinite(signals.ratioR[f])) rOk++;
      }
      const pct = (n) => `${(100 * n / Math.max(N, 1)).toFixed(0)}%`;
      el.textContent = `Valid frames: L ${pct(lOk)} (${lOk}/${N}) · R ${pct(rOk)} (${rOk}/${N})`;
    };

    // Pose / glove confidence-gate sliders
    const wireGate = (sliderId, outId, cfgKey) => {
      const s = host.querySelector("#" + sliderId);
      const o = host.querySelector("#" + outId);
      if (!s) return;
      s.addEventListener("input", () => {
        cfg[cfgKey] = Number(s.value);
        o.textContent = cfg[cfgKey].toFixed(2);
        recomputeAndRefresh();
      });
    };
    wireGate("ae-pose-gate",  "ae-pose-gate-out",  "minPoseConf");
    wireGate("ae-glove-gate", "ae-glove-gate-out", "minGloveConf");

    // Axiality gate — the only gate. Both the toggle and the slider only move
    // the verdict (peak_axiality is already computed per punch in computeAll),
    // so they just re-score; no perFrameRatio recompute needed. updateAxialStatus
    // is module-level (shared with onModelReady).
    const axialToggle = host.querySelector("#ae-axial-toggle");
    const axialSlider = host.querySelector("#ae-axial-max");
    const axialOut    = host.querySelector("#ae-axial-max-out");
    const axialRow    = host.querySelector("#ae-axial-slider-row");
    if (axialToggle) {
      axialToggle.addEventListener("change", () => {
        cfg.axialityGate = axialToggle.checked;
        if (axialSlider) axialSlider.disabled = !cfg.axialityGate;
        if (axialRow) axialRow.style.opacity = cfg.axialityGate ? 1 : 0.5;
        for (const p of signals.punches) rescorePunch(p, cfg);
        renderPunchTable();
        renderAggregate();
        updateAxialStatus();
        window.__viewerRedraw?.();
      });
    }
    if (axialSlider) {
      axialSlider.addEventListener("input", () => {
        cfg.axialityMax = Number(axialSlider.value);
        axialOut.textContent = cfg.axialityMax.toFixed(2);
        if (!cfg.axialityGate) return;
        for (const p of signals.punches) rescorePunch(p, cfg);
        renderPunchTable();
        renderAggregate();
        updateAxialStatus();
        window.__viewerRedraw?.();
      });
    }
    updateAxialStatus();

    const updateCorrStatus = () => {
      const el = host.querySelector("#ae-corr-status");
      if (!el) return;
      if (!cfg.shoulderCorrect) { el.textContent = "—"; return; }
      const b = signals.bodyAxis;
      if (!b) { el.textContent = "Correction enabled but no valid hip frames in this round — falls back to raw shoulder."; return; }
      const mag = Math.hypot(b.dx, b.dy);
      el.textContent = `Body axis: dx=${b.dx.toFixed(1)} dy=${b.dy.toFixed(1)} px  (|offset| = ${mag.toFixed(1)} px, from ${b.nFrames} frames)`;
    };
    const corrToggle = host.querySelector("#ae-corr-toggle");
    const corrAlpha  = host.querySelector("#ae-corr-alpha");
    const corrAlphaOut = host.querySelector("#ae-corr-alpha-out");
    const corrSliderRow = host.querySelector("#ae-corr-slider-row");
    if (corrToggle) {
      corrToggle.addEventListener("change", () => {
        cfg.shoulderCorrect = corrToggle.checked;
        if (corrAlpha) corrAlpha.disabled = !cfg.shoulderCorrect;
        if (corrSliderRow) corrSliderRow.style.opacity = cfg.shoulderCorrect ? 1 : 0.5;
        recomputeAndRefresh();
      });
    }
    if (corrAlpha) {
      corrAlpha.addEventListener("input", () => {
        cfg.shoulderCorrectAlpha = Number(corrAlpha.value);
        corrAlphaOut.textContent = cfg.shoulderCorrectAlpha.toFixed(2);
        if (cfg.shoulderCorrect) recomputeAndRefresh();
      });
    }
    updateCorrStatus();
    updateGateStatus();

    // Click a punch row to seek. Same scrubber-dispatch trick the other
    // lenses use — keeps the seek path single-sourced through the existing
    // input handler instead of inventing a new API.
    host.addEventListener("click", (ev) => {
      const tr = ev.target.closest("tr[data-frame]");
      if (!tr) return;
      const f = Number(tr.dataset.frame);
      const slider = document.getElementById("scrubber");
      if (!slider) return;
      slider.value = String(f);
      slider.dispatchEvent(new Event("input"));
    });
  },

  draw(ctx, state) {
    latestState = state;
    const pose = pickPose(state);
    if (pose !== lastPose) {
      // Pose swapped (e.g. compare-mode toggle) — recompute.
      signals = computeAll(state, cfg);
      lastPose = pose;
    }
    const s = state.renderScale || 1;
    const f = state.frame;

    // Draw both arms with their current ratio overlay.
    drawArmRatio(ctx, pose, f, "L", signals.ratioL[f], cfg, s, signals.bodyAxis);
    drawArmRatio(ctx, pose, f, "R", signals.ratioR[f], cfg, s, signals.bodyAxis);

    // HUD — when the playhead is inside a labelled punch window, show the
    // GT verdict + our prediction so you can eyeball agreement on-video.
    const active = activePunchAt(signals.punches, f);
    if (active) drawVerdictHud(ctx, active, s, pose);
  },

  update(state) {
    latestState = state;
    ensureAxialityModel(state, onModelReady);
    const pose = pickPose(state);
    if (pose !== lastPose) {
      signals = computeAll(state, cfg);
      lastPose = pose;
    }
    const f = state.frame;
    setMetric("ae-l-ratio", signals.ratioL[f], cfg);
    setMetric("ae-r-ratio", signals.ratioR[f], cfg);
    setText("ae-l-bend", formatBend(signals.bendL[f]));
    setText("ae-r-bend", formatBend(signals.bendR[f]));
    // Axiality is now per-punch (the model scores a whole punch, not a frame),
    // so the inspector shows the active punch's model value on its arm only.
    const ap = activePunchAt(signals.punches, f);
    setAxial("ae-l-axial", ap && ap.side === "L" ? ap.peak_axiality : NaN, cfg);
    setAxial("ae-r-axial", ap && ap.side === "R" ? ap.peak_axiality : NaN, cfg);
    setText("ae-l-source", signals.sourceL[f] || "—");
    setText("ae-r-source", signals.sourceR[f] || "—");
  },
};

// ─── compute ───────────────────────────────────────────────────────────────

function computeAll(state, cfg) {
  const pose = pickPose(state);
  const N = pose.n_frames;
  const fps = pose.fps;
  // Source-video time offset — added to the label's cache-relative
  // timestamp so the per-punch table matches the playback timebar
  // (which uses absolute source time).
  const startSec = pose.start_sec || 0;

  // Body-axis offset for the optional anatomical shoulder correction.
  // We use the round-median of (hip_mid − shoulder_mid) so that per-frame
  // hip motion during a punch doesn't move the corrected shoulder around.
  const bodyAxis = cfg.shoulderCorrect ? bodyAxisOffset(pose, cfg) : null;

  // Reach denominator: per-frame Euclidean torso (|sh_mid → hip_mid|). Torso
  // scales with the boxer's distance to camera the same way the arm does, so
  // dividing by it makes reach depth-motion invariant — round-median arm
  // length (the old denominator) broke whenever the boxer stepped in or out.
  // Same normalizer the stance_width rule uses, just per-frame.
  const torsoEuclid = torsoEuclidPerFrame(pose, cfg);

  const { ratio: ratioL, reach: reachL, bendDeg: bendL, source: sourceL } = perFrameRatio(pose, "L", cfg, bodyAxis, torsoEuclid);
  const { ratio: ratioR, reach: reachR, bendDeg: bendR, source: sourceR } = perFrameRatio(pose, "R", cfg, bodyAxis, torsoEuclid);

  // Labelled punches — filtered to straights, with optional rule_extension
  // verdict for agreement scoring.
  const detections = (state.labels?.detections || []).filter(d =>
    cfg.appliesTo.has(d.punch_type)
  );
  const punches = detections.map((d, idx) => {
    const sf = Math.max(0, d.start_frame);
    const ef = Math.min(N - 1, d.end_frame);

    const stance = (d.stance === "southpaw" || d.stance === "orthodox")
      ? d.stance : "orthodox";
    const side = SIDE_FOR[d.hand]?.[stance] || "L";
    const ratioArr = side === "L" ? ratioL : ratioR;
    const reachArr = side === "L" ? reachL : reachR;
    const bendArr  = side === "L" ? bendL : bendR;
    const srcArr   = side === "L" ? sourceL : sourceR;

    // Pick the peak frame as the most-extended one in the window — that's
    // the moment of "full extension" that drives the verdict. Prefer max
    // reach (|sh→wr|/arm_length); fall back to max r (straightness) when
    // arm_length is unknown so reach is NaN.
    let peakReachWin = -Infinity;
    let peakRWin = -Infinity, peakFrame = sf, gloveFrames = 0, validFrames = 0;
    let peakFrameByR = sf, peakFrameByReach = sf;
    for (let f = sf; f <= ef; f++) {
      const r = ratioArr[f];
      if (!Number.isFinite(r)) continue;
      validFrames++;
      if (srcArr[f] === "glove") gloveFrames++;
      if (r > peakRWin) { peakRWin = r; peakFrameByR = f; }
      const rea = reachArr[f];
      if (Number.isFinite(rea) && rea > peakReachWin) { peakReachWin = rea; peakFrameByReach = f; }
    }
    const peakValid = Number.isFinite(peakRWin);
    peakFrame = (peakReachWin > -Infinity) ? peakFrameByReach : peakFrameByR;
    const peak = peakValid ? ratioArr[peakFrame] : NaN;
    const peakReach = peakValid ? reachArr[peakFrame] : NaN;

    // Axiality from the trained temporal model, joined by punch_uuid: held-out
    // for direction-labeled straights, forward-inferred for the rest (score_all
    // writes both). 0 = forearm flat across the image (side-on, bend
    // trustworthy), 1 = down the camera axis (foreshortened). null when the
    // model has no entry for this punch (no pose cache) → NaN → gated as
    // axial_unknown. This is the only gate — see rescorePunch.
    const peak_axiality = axialityForPunch(d.punch_uuid)?.predAxiality ?? NaN;

    const label = d.rule_extension === "pass" || d.rule_extension === "fail"
      ? d.rule_extension : null;

    // Per-joint confidence at the peak frame — what fed the verdict.
    // Wrist conf reflects whichever source was actually used: if the
    // glove won the source pick that frame, use the glove conf; else
    // the pose wrist conf.
    const joints = JOINTS_FOR_SIDE[side];
    let shConf = NaN, elConf = NaN, wrConf = NaN;
    if (peakValid) {
      shConf = pose.conf[peakFrame * 17 + joints.shoulder];
      elConf = pose.conf[peakFrame * 17 + joints.elbow];
      if (srcArr[peakFrame] === "glove" && pose.gloveWrists) {
        wrConf = gloveConf(pose.gloveWrists, peakFrame, joints.gloveSide);
      } else {
        wrConf = pose.conf[peakFrame * 17 + joints.wrist];
      }
    }

    const p = {
      idx,
      timestamp: d.timestamp,
      t_abs: startSec + (Number.isFinite(d.timestamp) ? d.timestamp : 0),
      hand: d.hand,
      stance,
      side,
      punch_type: d.punch_type,
      start_frame: sf,
      end_frame: ef,
      land_frame: peakValid ? peakFrame : sf,
      peak: peakValid ? peak : NaN,
      peak_bend_deg: peakValid ? bendArr[peakFrame] : NaN,
      peak_reach: peakValid ? peakReach : NaN,
      peak_axiality,
      glove_coverage: validFrames ? gloveFrames / validFrames : 0,
      peak_sh_conf: shConf,
      peak_el_conf: elConf,
      peak_wr_conf: wrConf,
      peak_wr_source: peakValid ? (srcArr[peakFrame] || "—") : "—",
      label,
    };
    // Single source of truth for predicted/reason — keeps re-score handlers
    // (threshold sliders, gate toggles) consistent with first-pass scoring.
    rescorePunch(p, cfg);
    return p;
  });

  return {
    ratioL, ratioR, bendL, bendR,
    sourceL, sourceR, punches, fps, bodyAxis,
  };
}

// The axiality sidecar loads async. When it lands, peak_axiality (read from the
// model in computeAll) must be recomputed, then the table / aggregate / gate
// status refreshed. ensureAxialityModel's guard makes this fire once; resetting
// lastPose keeps the next draw from recomputing redundantly.
function onModelReady() {
  if (!host || !latestState) return;
  signals = computeAll(latestState, cfg);
  lastPose = pickPose(latestState);
  renderPunchTable();
  renderAggregate();
  updateAxialStatus();
  window.__viewerRedraw?.();
}

// Gate status line: how many punches the axiality gate skipped as too head-on,
// and how many the model never scored. Module-level so onModelReady and the
// gate toggle/slider handlers all share it.
function updateAxialStatus() {
  const el = host?.querySelector("#ae-axial-status");
  if (!el) return;
  if (!cfg.axialityGate) { el.textContent = "Disabled — every punch scored on bend regardless of foreshortening."; return; }
  const N = signals.punches.length;
  const skipped = signals.punches.filter(p => p.reason === "axial").length;
  const unknown = signals.punches.filter(p => p.reason === "axial_unknown").length;
  el.textContent = `${skipped}/${N} too head-on (skipped) · ${unknown} not scored by the model.`;
}

function bodyAxisOffset(pose, cfg) {
  // Median (hip_mid − shoulder_mid) across frames where all four joints
  // have decent confidence. Component-wise median keeps it stable against
  // single bad frames. Returns null if too few frames qualify.
  const N = pose.n_frames;
  const dxs = [], dys = [];
  const confGate = cfg.minPoseConf;
  for (let f = 0; f < N; f++) {
    const cLs = pose.conf[f * 17 + J.L_SHOULDER];
    const cRs = pose.conf[f * 17 + J.R_SHOULDER];
    const cLh = pose.conf[f * 17 + J.L_HIP];
    const cRh = pose.conf[f * 17 + J.R_HIP];
    if (cLs < confGate || cRs < confGate || cLh < confGate || cRh < confGate) continue;
    const lsx = pose.skeleton[(f * 17 + J.L_SHOULDER) * 2];
    const lsy = pose.skeleton[(f * 17 + J.L_SHOULDER) * 2 + 1];
    const rsx = pose.skeleton[(f * 17 + J.R_SHOULDER) * 2];
    const rsy = pose.skeleton[(f * 17 + J.R_SHOULDER) * 2 + 1];
    const lhx = pose.skeleton[(f * 17 + J.L_HIP) * 2];
    const lhy = pose.skeleton[(f * 17 + J.L_HIP) * 2 + 1];
    const rhx = pose.skeleton[(f * 17 + J.R_HIP) * 2];
    const rhy = pose.skeleton[(f * 17 + J.R_HIP) * 2 + 1];
    if (![lsx, lsy, rsx, rsy, lhx, lhy, rhx, rhy].every(Number.isFinite)) continue;
    const shMidX = 0.5 * (lsx + rsx), shMidY = 0.5 * (lsy + rsy);
    const hipMidX = 0.5 * (lhx + rhx), hipMidY = 0.5 * (lhy + rhy);
    dxs.push(hipMidX - shMidX);
    dys.push(hipMidY - shMidY);
  }
  if (dxs.length < 5) return null;
  const median = arr => {
    const s = [...arr].sort((a, b) => a - b);
    return s[Math.floor(s.length / 2)];
  };
  return { dx: median(dxs), dy: median(dys), nFrames: dxs.length };
}

// Resolve the shoulder position for a given frame, applying the anatomical
// correction when enabled. Single source of truth — used by both
// perFrameRatio (for the metric) and drawArmRatio (for the visual).
function shoulderXY(pose, frame, joints, cfg, bodyAxis) {
  const raw_x = pose.skeleton[(frame * 17 + joints.shoulder) * 2];
  const raw_y = pose.skeleton[(frame * 17 + joints.shoulder) * 2 + 1];
  if (!cfg.shoulderCorrect || !bodyAxis) {
    return { x: raw_x, y: raw_y, raw_x, raw_y };
  }
  const a = cfg.shoulderCorrectAlpha;
  return {
    x: raw_x + a * bodyAxis.dx,
    y: raw_y + a * bodyAxis.dy,
    raw_x, raw_y,
  };
}

function perFrameRatio(pose, side, cfg, bodyAxis, torsoEuclid) {
  const N = pose.n_frames;
  const joints = JOINTS_FOR_SIDE[side];
  const ratio  = new Float32Array(N);
  const reach  = new Float32Array(N);
  const bendDeg = new Float32Array(N);
  const source = new Array(N);
  const RAD_TO_DEG = 180 / Math.PI;
  for (let f = 0; f < N; f++) {
    const w = wristXY(pose, f, joints, cfg);
    if (!w) { ratio[f] = NaN; reach[f] = NaN; bendDeg[f] = NaN; source[f] = null; continue; }

    const sc = pose.conf[f * 17 + joints.shoulder];
    const ec = pose.conf[f * 17 + joints.elbow];
    if (sc < cfg.minPoseConf || ec < cfg.minPoseConf) {
      ratio[f] = NaN; reach[f] = NaN; bendDeg[f] = NaN; source[f] = null; continue;
    }
    const sh = shoulderXY(pose, f, joints, cfg, bodyAxis);
    const sx = sh.x, sy = sh.y;
    const ex = pose.skeleton[(f * 17 + joints.elbow) * 2];
    const ey = pose.skeleton[(f * 17 + joints.elbow) * 2 + 1];

    const ue = Math.hypot(sx - ex, sy - ey);          // shoulder→elbow
    const fa = Math.hypot(ex - w.x, ey - w.y);        // elbow→wrist
    const sw = Math.hypot(sx - w.x, sy - w.y);        // shoulder→wrist
    const path = ue + fa;
    if (path < 1e-3 || ue < 1e-3 || fa < 1e-3) {
      ratio[f] = NaN; reach[f] = NaN; bendDeg[f] = NaN; source[f] = null; continue;
    }
    // Bounded [0,1]. Clamp tiny float overshoots that can happen when the
    // wrist is collinear with shoulder–elbow.
    ratio[f] = Math.min(1, sw / path);
    // Reach: |shoulder→wrist| / euclidean torso (per-frame). Torso scales
    // with the boxer's distance to camera the same way the arm does, so
    // depth motion cancels out. Typical full extension lands ≈1.8–2.5
    // torsos; depth-foreshortened punches drop below ~1.0 even though r
    // still reads near 1.0. NaN when the torso conf gate (both shoulders
    // + both hips) rejected this frame.
    const t = torsoEuclid[f];
    reach[f] = (Number.isFinite(t) && t > 0) ? (sw / t) : NaN;

    // Exact elbow angle from law of cosines — handles uneven upper-arm /
    // forearm lengths (the sin(θ/2) approximation in ratio_to_bend_deg
    // assumes equal segments). Bend = 180° − elbow_angle.
    const cosElbow = Math.max(-1, Math.min(1, (ue*ue + fa*fa - sw*sw) / (2*ue*fa)));
    bendDeg[f] = 180 - Math.acos(cosElbow) * RAD_TO_DEG;
    source[f] = w.source;
  }
  return { ratio, reach, bendDeg, source };
}

// Per-frame Euclidean torso length, |shoulder_mid → hip_mid|. Returns a
// Float32Array of length N with NaN for frames where any of the four
// torso joints (both shoulders + both hips) fails the conf gate. Drives
// the reach metric — same normalizer convention stance_width uses, just
// computed every frame instead of round-medianed.
function torsoEuclidPerFrame(pose, cfg) {
  const N = pose.n_frames;
  const out = new Float32Array(N);
  const gate = cfg.minPoseConf;
  for (let f = 0; f < N; f++) {
    const cLs = pose.conf[f * 17 + J.L_SHOULDER];
    const cRs = pose.conf[f * 17 + J.R_SHOULDER];
    const cLh = pose.conf[f * 17 + J.L_HIP];
    const cRh = pose.conf[f * 17 + J.R_HIP];
    if (cLs < gate || cRs < gate || cLh < gate || cRh < gate) {
      out[f] = NaN; continue;
    }
    const lsx = pose.skeleton[(f * 17 + J.L_SHOULDER) * 2];
    const lsy = pose.skeleton[(f * 17 + J.L_SHOULDER) * 2 + 1];
    const rsx = pose.skeleton[(f * 17 + J.R_SHOULDER) * 2];
    const rsy = pose.skeleton[(f * 17 + J.R_SHOULDER) * 2 + 1];
    const lhx = pose.skeleton[(f * 17 + J.L_HIP) * 2];
    const lhy = pose.skeleton[(f * 17 + J.L_HIP) * 2 + 1];
    const rhx = pose.skeleton[(f * 17 + J.R_HIP) * 2];
    const rhy = pose.skeleton[(f * 17 + J.R_HIP) * 2 + 1];
    if (![lsx, lsy, rsx, rsy, lhx, lhy, rhx, rhy].every(Number.isFinite)) {
      out[f] = NaN; continue;
    }
    const shMidX = 0.5 * (lsx + rsx), shMidY = 0.5 * (lsy + rsy);
    const hipMidX = 0.5 * (lhx + rhx), hipMidY = 0.5 * (lhy + rhy);
    out[f] = Math.hypot(shMidX - hipMidX, shMidY - hipMidY);
  }
  return out;
}

function wristXY(pose, frame, joints, cfg) {
  // Legacy raw-glove sidecar path — only set on pre-v6 rounds. When present
  // we honor it for backwards compatibility, but new rounds use v6 below.
  const g = pose.gloveWrists;
  if (g) {
    const [gx, gy] = gloveXY(g, frame, joints.gloveSide);
    const gc       = gloveConf(g, frame, joints.gloveSide);
    if (gc >= cfg.minGloveConf && Number.isFinite(gx) && Number.isFinite(gy)) {
      return { x: gx, y: gy, source: "glove" };
    }
  }
  // v6 path (also the legacy pose-only fallback). When the meta says
  // wrist_replaced_with_glove, joints 9/10 are the glove wrist already and
  // conf at those indices is the glove conf — gate with minGloveConf and
  // treat below-threshold as no detection (matches the production contract:
  // NO Vision fallback when the glove model was running). When the meta
  // says wrists are pure Vision, use minPoseConf like any other joint.
  const px = pose.skeleton[(frame * 17 + joints.wrist) * 2];
  const py = pose.skeleton[(frame * 17 + joints.wrist) * 2 + 1];
  const pc = pose.conf[frame * 17 + joints.wrist];
  const isGloveBaked = pose.meta?.wrist_replaced_with_glove === true;
  const gate = isGloveBaked ? cfg.minGloveConf : cfg.minPoseConf;
  if (pc < gate || !Number.isFinite(px)) return null;
  return { x: px, y: py, source: isGloveBaked ? "glove" : "pose" };
}

// ─── render ────────────────────────────────────────────────────────────────

function renderTemplate(sig, cfg) {
  const hasLabels = sig.punches.some(p => p.label !== null);
  return `
    <h2>Arm extension (straights)</h2>
    <p class="hint">
      <code>r = |shoulder→wrist| / (|shoulder→elbow| + |elbow→wrist|)</code>.
      1.00 = dead straight, 0.95 ≈ 25° bend, 0.71 = 90° bend, 0 = folded.
    </p>

    <h3>Legend</h3>
    <ul class="hint" style="list-style:none;padding-left:0;margin:0 0 12px 0;line-height:1.7">
      <li><span style="display:inline-block;width:24px;height:3px;background:${COLORS.pass};vertical-align:middle"></span>
        &nbsp;arm bones (sh→el→wr) when ratio ≥ threshold</li>
      <li><span style="display:inline-block;width:24px;height:3px;background:${COLORS.fail};vertical-align:middle"></span>
        &nbsp;arm bones when ratio &lt; threshold</li>
      <li><span style="display:inline-block;width:10px;height:10px;background:${COLORS.pass};border-radius:50%;vertical-align:middle"></span>
        &nbsp;<b>shoulder</b> + <b>elbow</b> markers — the three corners that drive the ratio</li>
      <li><span style="display:inline-block;width:18px;height:18px;background:${COLORS.pass};border:2px solid white;border-radius:50%;vertical-align:middle;position:relative">
        <span style="position:absolute;left:5px;top:5px;width:6px;height:6px;background:rgba(0,0,0,0.55);border-radius:50%"></span></span>
        &nbsp;<b>corrected shoulder</b> (target marker) — only shown when anatomical correction is on; smaller hollow ring shows the raw acromion, dashed line shows the shift</li>
      <li><span style="display:inline-block;width:16px;height:16px;border:2px solid ${COLORS.pass};border-top-left-radius:16px;border-top-right-radius:0;border-bottom-left-radius:0;border-bottom-right-radius:0;border-bottom:none;border-right:none;vertical-align:middle"></span>
        &nbsp;arc at the elbow shows the <b>interior angle</b>; the number next to it is the <b>bend</b> in °</li>
      <li><span style="display:inline-block;width:24px;height:1px;background:${COLORS.ratioGuide};border-top:1px dashed ${COLORS.ratioGuide};vertical-align:middle"></span>
        &nbsp;dashed shoulder→wrist guide (the "if fully extended" line)</li>
      <li><span style="display:inline-block;width:12px;height:12px;background:rgba(0,0,0,0.55);border:2px solid ${COLORS.gloveArm};vertical-align:middle"></span>
        &nbsp;wrist from the <b>glove detector</b> (conf ≥ ${cfg.minGloveConf})</li>
      <li><span style="display:inline-block;width:14px;height:14px;border:2px solid ${COLORS.poseArm};border-radius:50%;vertical-align:middle"></span>
        &nbsp;wrist from the <b>pose model</b> (fallback when glove missing/low-conf)</li>
      <li><span style="color:${COLORS.pass};font-family:monospace">0.97</span>
        / <span style="color:${COLORS.fail};font-family:monospace">0.82</span>
        &nbsp;ratio readout next to each wrist, colored by pass/fail vs threshold</li>
    </ul>

    <h3>Live ratio</h3>
    <div class="metric-grid">
      <div class="metric">
        <div class="metric-label">L arm</div>
        <div class="metric-val" id="ae-l-ratio">—</div>
        <div class="metric-sub">
          <span id="ae-l-bend">—</span> ·
          axial <span id="ae-l-axial">—</span> ·
          <span id="ae-l-source">—</span>
        </div>
      </div>
      <div class="metric">
        <div class="metric-label">R arm</div>
        <div class="metric-val" id="ae-r-ratio">—</div>
        <div class="metric-sub">
          <span id="ae-r-bend">—</span> ·
          axial <span id="ae-r-axial">—</span> ·
          <span id="ae-r-source">—</span>
        </div>
      </div>
    </div>

    <h3>Pass threshold (shape)</h3>
    <div class="slider-row">
      <input type="range" id="ae-threshold" min="0.70" max="1.00" step="0.01" value="${cfg.threshold}" />
      <output id="ae-threshold-out">${cfg.threshold.toFixed(2)}</output>
      <span class="muted small">peak ratio r — geometric straightness</span>
    </div>

    <h3>Axiality gate (sideways only)</h3>
    <p class="hint">
      Bend is read in 2D, so it's only trustworthy when the punch travels
      across the image plane. <b>Axiality</b> (0 = flat across the image /
      side-on, 1 = pointing straight down the camera axis) measures that
      foreshortening — now the <b>trained temporal model</b>'s held-out
      per-punch prediction (joined by punch_uuid), not a per-frame geometric
      estimate. Punches whose predicted axiality is <em>above</em> the cut are
      too head-on to judge and get a <b>skip</b>. The signal is magnitude-only,
      so one upper cut covers both toward- and away-camera punches; the default
      ≈0.71 keeps the forearm within ±45° of the image plane. The model only
      scores <em>labelled</em> straights it held out, so punches it never scored
      fall through as <b>not scored</b>.
    </p>
    <label style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
      <input type="checkbox" id="ae-axial-toggle" ${cfg.axialityGate ? 'checked' : ''} />
      <span>Enable axiality gate</span>
    </label>
    <div class="slider-row" style="opacity:${cfg.axialityGate ? 1 : 0.5}" id="ae-axial-slider-row">
      <input type="range" id="ae-axial-max" min="0.30" max="1.00" step="0.01"
        value="${cfg.axialityMax}" ${cfg.axialityGate ? '' : 'disabled'} />
      <output id="ae-axial-max-out">${cfg.axialityMax.toFixed(2)}</output>
      <span class="muted small">max peak axiality (lower = stricter / demand more side-on; 0.71 ≈ 45°)</span>
    </div>
    <p class="hint muted small" id="ae-axial-status" style="margin:4px 0 0 0">—</p>

    <h3>Confidence gates</h3>
    <p class="hint">
      Frames where the relevant pose joints (shoulder/elbow/wrist/hip) or the
      glove detection fall below these confidences are rejected from the
      ratio computation. Raising the pose gate kills frames with sketchy
      pose tracking; raising the glove gate makes the lens fall back to
      the pose wrist more often.
    </p>
    <div class="slider-row">
      <input type="range" id="ae-pose-gate" min="0.05" max="0.95" step="0.05" value="${cfg.minPoseConf}" />
      <output id="ae-pose-gate-out">${cfg.minPoseConf.toFixed(2)}</output>
      <span class="muted small">pose conf — shoulder, elbow, wrist-fallback, hip</span>
    </div>
    <div class="slider-row">
      <input type="range" id="ae-glove-gate" min="0.05" max="0.95" step="0.05" value="${cfg.minGloveConf}" />
      <output id="ae-glove-gate-out">${cfg.minGloveConf.toFixed(2)}</output>
      <span class="muted small">glove conf — v6: below = no wrist detection (no Vision fallback); legacy sidecar: below = fall back to pose wrist</span>
    </div>
    <p class="hint muted small" id="ae-gate-status" style="margin:4px 0 0 0">—</p>

    <h3>Anatomical shoulder correction</h3>
    <p class="hint">
      COCO labels the shoulder at the acromion (top of the shoulder), not the
      glenohumeral joint center. When on, the shoulder anchor is shifted by
      α × <code>(hip_mid − shoulder_mid)</code>, using the round-median offset
      so per-frame hip motion during a punch doesn't move the anchor. The raw
      acromion is shown as a hollow ring connected by a dashed line.
    </p>
    <label style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
      <input type="checkbox" id="ae-corr-toggle" ${cfg.shoulderCorrect ? 'checked' : ''} />
      <span>Enable correction</span>
    </label>
    <div class="slider-row" style="opacity:${cfg.shoulderCorrect ? 1 : 0.5}" id="ae-corr-slider-row">
      <input type="range" id="ae-corr-alpha" min="0.00" max="0.35" step="0.01"
        value="${cfg.shoulderCorrectAlpha}" ${cfg.shoulderCorrect ? '' : 'disabled'} />
      <output id="ae-corr-alpha-out">${cfg.shoulderCorrectAlpha.toFixed(2)}</output>
      <span class="muted small">α — fraction of torso-axis offset (~0.20 ≈ 5–7 cm down)</span>
    </div>
    <p class="hint muted small" id="ae-corr-status" style="margin:4px 0 0 0">—</p>

    <h3>Per-punch (straights only)</h3>
    <p class="hint">
      Peak = max ratio inside the punch window. Clickable rows seek to the
      peak frame. ${hasLabels
        ? "Verdict shows labeler vs predicted; ✓ when they agree."
        : "<span class='muted'>No <code>rule_extension</code> labels found — predicted only.</span>"}
    </p>
    <div id="ae-table-host"></div>

    <h3>Aggregate</h3>
    <div id="ae-aggregate" class="metric-grid"></div>
  `;
}

function pill(value, kind) {
  // kind: "gt" | "pred" — same color vocabulary as the canvas HUD
  if (value !== "pass" && value !== "fail" && value !== "unclear" && value !== "skip") {
    return `<span class="ae-pill ae-pill-empty" title="no label">—</span>`;
  }
  let col;
  if (value === "pass")      col = COLORS.pass;
  else if (value === "fail") col = COLORS.fail;
  else if (value === "skip") col = "#7ec8ff"; // blue-ish for depth-skipped
  else                       col = COLORS.unclear;
  return `<span class="ae-pill" style="background:${col}1f;color:${col};border:1px solid ${col}66">${value}</span>`;
}

function confCell(value) {
  // Same color vocabulary as the base skeleton renderer's confColor()
  // (skeleton.js): green ≥ 0.5, amber ≥ 0.2, red < 0.2, em-dash when NaN.
  if (!Number.isFinite(value)) {
    return `<td class="muted" style="font-variant-numeric:tabular-nums">—</td>`;
  }
  const col = value >= 0.5 ? COLORS.pass
            : value >= 0.2 ? COLORS.unclear
            : COLORS.fail;
  return `<td style="color:${col};font-variant-numeric:tabular-nums">${value.toFixed(2)}</td>`;
}

function renderPunchTable() {
  const hasAnyLabel = signals.punches.some(p => p.label);
  const tbody = signals.punches.length
    ? signals.punches.map(p => {
        const tsStr   = Number.isFinite(p.t_abs) ? p.t_abs.toFixed(2) : "—";
        const predCell = pill(p.predicted, "pred");
        // Agreement marker against GT — only meaningful when GT exists
        let match = "";
        if (p.label && p.predicted !== "unclear") {
          match = p.label === p.predicted
            ? `<span style="color:${COLORS.agree}" title="GT ${p.label} · agrees">✓</span>`
            : `<span style="color:${COLORS.disagree}" title="GT ${p.label} · disagrees">✗</span>`;
        }
        const bendStr = Number.isFinite(p.peak_bend_deg)
          ? `${p.peak_bend_deg.toFixed(1)}°` : "—";
        // Axiality cell — the gate signal. Green = side-on enough to trust
        // bend (≤ cut), red = too head-on (skipped). Greyed when the gate is
        // off.
        let axialCell;
        if (!Number.isFinite(p.peak_axiality)) {
          axialCell = `<td class="muted">—</td>`;
        } else {
          const col = !cfg.axialityGate ? "var(--muted, #888)"
                    : (p.peak_axiality <= cfg.axialityMax) ? COLORS.pass
                    : COLORS.fail;
          axialCell = `<td style="color:${col};font-variant-numeric:tabular-nums">${p.peak_axiality.toFixed(2)}</td>`;
        }
        // Reach cell — context only now (no longer a gate), always muted.
        const reachCell = Number.isFinite(p.peak_reach)
          ? `<td class="muted" style="font-variant-numeric:tabular-nums">${p.peak_reach.toFixed(2)}</td>`
          : `<td class="muted">—</td>`;
        const reasonCode = p.reason || "—";
        const reasonText = (p.reason_text || "").replace(/"/g, '&quot;');
        const reasonCell = `<td class="muted small" title="${reasonText}" style="white-space:nowrap">${reasonCode}</td>`;
        return `<tr data-frame="${p.land_frame}" style="cursor:pointer">
          <td>${tsStr}s</td>
          <td>${p.punch_type}</td>
          <td>${predCell}</td>
          <td style="text-align:center">${match}</td>
          ${reasonCell}
          <td style="font-variant-numeric:tabular-nums" class="muted">${bendStr}</td>
          ${axialCell}
          ${reachCell}
          ${confCell(p.peak_sh_conf)}
          ${confCell(p.peak_el_conf)}
          ${confCell(p.peak_wr_conf)}
        </tr>`;
      }).join("")
    : `<tr><td colspan="11" class="muted">no labeled straights in this round</td></tr>`;

  const tableHost = host.querySelector("#ae-table-host");
  if (tableHost) {
    const gtNote = hasAnyLabel
      ? ""
      : `<p class="hint muted" style="margin:0 0 8px 0">No <code>rule_extension</code> GT verdicts attached to this round (Sheet labels missing or no match) — match column will be blank.</p>`;
    tableHost.innerHTML = `
      ${gtNote}
      <style>
        .ae-pill {
          display:inline-block; padding:1px 8px; border-radius:10px;
          font-size:12px; font-weight:600; letter-spacing:0.02em;
          font-family: inherit;
        }
        .ae-pill-empty {
          background:transparent; color:var(--muted, #888);
          border:1px dashed currentColor;
        }
      </style>
      <table class="rule-table">
        <thead><tr>
          <th>t</th><th>type</th><th>pred</th><th title="agrees with GT verdict">vs GT</th>
          <th title="which gate decided this punch — hover for the numbers that fired it (e.g. r=0.87 < 0.95)">why</th>
          <th>bend</th>
          <th title="axiality model's per-punch prediction (0 = side-on, 1 = down the camera axis) — the gate. Above the cut → skipped as too head-on to judge bend">axiality</th>
          <th title="peak reach = |sh→wr| / euclidean torso at peak frame — context only, no longer a gate (units: torsos)">reach</th>
          <th title="shoulder confidence at peak frame">sh</th>
          <th title="elbow confidence at peak frame">el</th>
          <th title="wrist confidence at peak frame — glove if used, pose if fallback">wr</th>
        </tr></thead>
        <tbody>${tbody}</tbody>
      </table>`;
  }
}

function renderAggregate() {
  const host_ = host.querySelector("#ae-aggregate");
  if (!host_) return;
  const ps = signals.punches.filter(p => Number.isFinite(p.peak));
  if (!ps.length) {
    host_.innerHTML = `<div class="metric muted">no scorable punches</div>`;
    return;
  }
  const meanPeak = ps.reduce((s, p) => s + p.peak, 0) / ps.length;
  const passed = ps.filter(p => p.predicted === "pass").length;
  const skipped = signals.punches.filter(p => p.predicted === "skip").length;
  // Agreement is only meaningful on punches the gate actually scored
  // (pass/fail) — skipped punches have no comparable verdict.
  const labelled = ps.filter(p => p.label && p.predicted !== "skip");
  const agree = labelled.filter(p => p.label === p.predicted).length;
  const agreePct = labelled.length
    ? `${Math.round(100 * agree / labelled.length)}%`
    : "—";
  host_.innerHTML = `
    <div class="metric">
      <div class="metric-label">scored</div>
      <div class="metric-val">${ps.length}</div>
      <div class="metric-sub">of ${signals.punches.length} labelled straights</div>
    </div>
    <div class="metric">
      <div class="metric-label">predicted pass</div>
      <div class="metric-val">${passed}</div>
      <div class="metric-sub">${Math.round(100 * passed / ps.length)}% · r ≥ ${cfg.threshold.toFixed(2)}</div>
    </div>
    <div class="metric">
      <div class="metric-label">skipped (axial)</div>
      <div class="metric-val">${skipped}</div>
      <div class="metric-sub">too head-on to judge</div>
    </div>
    <div class="metric">
      <div class="metric-label">mean peak r</div>
      <div class="metric-val">${meanPeak.toFixed(3)}</div>
    </div>
    <div class="metric">
      <div class="metric-label">label agreement</div>
      <div class="metric-val">${agreePct}</div>
      <div class="metric-sub">${agree} / ${labelled.length} match (scored only)</div>
    </div>
  `;
}

// ─── draw ──────────────────────────────────────────────────────────────────

function drawArmGhost(ctx, pose, frame, side, cfg, scale) {
  // The lens couldn't score this arm at this frame. Don't go silent —
  // draw a faded marker at whichever joints we have positions for, and
  // label the failure reason so the user sees "the lens is here, but
  // joint X is too low conf to score". Uses raw acromion (not the
  // corrected anchor) since corrected shoulder needs the gate to pass.
  const joints = JOINTS_FOR_SIDE[side];
  const sx = pose.skeleton[(frame * 17 + joints.shoulder) * 2];
  const sy = pose.skeleton[(frame * 17 + joints.shoulder) * 2 + 1];
  const ex = pose.skeleton[(frame * 17 + joints.elbow) * 2];
  const ey = pose.skeleton[(frame * 17 + joints.elbow) * 2 + 1];
  const wx = pose.skeleton[(frame * 17 + joints.wrist) * 2];
  const wy = pose.skeleton[(frame * 17 + joints.wrist) * 2 + 1];
  const sc = pose.conf[frame * 17 + joints.shoulder];
  const ec = pose.conf[frame * 17 + joints.elbow];
  const wc = pose.conf[frame * 17 + joints.wrist];
  const gloveOK = pose.gloveWrists && (() => {
    const gc = gloveConf(pose.gloveWrists, frame, joints.gloveSide);
    const [gx] = gloveXY(pose.gloveWrists, frame, joints.gloveSide);
    return gc >= cfg.minGloveConf && Number.isFinite(gx);
  })();

  const tooLow = [];
  if (sc < cfg.minPoseConf) tooLow.push(`sh ${sc.toFixed(2)}`);
  if (ec < cfg.minPoseConf) tooLow.push(`el ${ec.toFixed(2)}`);
  if (!gloveOK && wc < cfg.minPoseConf) tooLow.push(`wr ${wc.toFixed(2)}`);
  if (!tooLow.length) tooLow.push("no wrist signal");

  ctx.save();
  ctx.fillStyle = "rgba(255,255,255,0.35)";
  ctx.strokeStyle = "rgba(255,255,255,0.35)";
  ctx.lineWidth = 1.5 * scale;
  ctx.setLineDash([3 * scale, 3 * scale]);
  // Bones — only if both ends are finite
  if (Number.isFinite(sx) && Number.isFinite(ex)) {
    ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();
  }
  if (Number.isFinite(ex) && Number.isFinite(wx)) {
    ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(wx, wy); ctx.stroke();
  }
  ctx.setLineDash([]);
  // Dots at whichever joints are finite
  for (const [x, y] of [[sx, sy], [ex, ey], [wx, wy]]) {
    if (!Number.isFinite(x)) continue;
    ctx.beginPath(); ctx.arc(x, y, 4 * scale, 0, Math.PI * 2); ctx.fill();
  }
  // Label near the shoulder (most stable joint) saying why we're hiding
  if (Number.isFinite(sx)) {
    const txt = `${side}: ${tooLow.join(", ")}`;
    ctx.font = `${12 * scale}px ui-monospace, monospace`;
    const tw = ctx.measureText(txt).width;
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(sx + 10 * scale, sy - 9 * scale, tw + 8 * scale, 18 * scale);
    ctx.fillStyle = "rgba(255,200,80,0.95)";
    ctx.fillText(txt, sx + 14 * scale, sy + 4 * scale);
  }
  ctx.restore();
}

function rescorePunch(p, cfg) {
  // Single source of truth for the verdict — also called from computeAll
  // so reason + predicted always stay in sync. The verdict is BEND ALONE;
  // axiality is the only gate (it just decides whether bend can be trusted).
  //   p.predicted   "pass" | "fail" | "skip" | "unclear"
  //   p.reason      short code (one of the branches below)
  //   p.reason_text human-readable with the actual numbers that fired it
  if (!Number.isFinite(p.peak)) {
    p.predicted = "unclear";
    p.reason = "no_peak";
    p.reason_text = "no valid peak frame (joints below conf gates throughout window)";
    return;
  }
  // Axiality gate — the only gate. Bend is read in 2D, so it's only
  // trustworthy when the forearm travels across the image plane. Axiality is
  // magnitude-only (0 = flat/side-on, 1 = down the camera axis), so a single
  // upper cut covers both toward- and away-camera foreshortening. Punches
  // above the cut are too head-on to judge → skip.
  if (cfg.axialityGate) {
    if (!Number.isFinite(p.peak_axiality)) {
      p.predicted = "skip";
      p.reason = "axial_unknown";
      p.reason_text = "the axiality model didn't score this punch (not in its held-out labelled set)";
      return;
    }
    if (p.peak_axiality > cfg.axialityMax) {
      p.predicted = "skip";
      p.reason = "axial";
      const offDeg = Math.acos(Math.max(0, Math.min(1, p.peak_axiality))) * 180 / Math.PI;
      p.reason_text = `foreshortened: axiality ${p.peak_axiality.toFixed(2)} > ${cfg.axialityMax.toFixed(2)} (forearm ~${offDeg.toFixed(0)}° from camera axis, need ≥45°)`;
      return;
    }
  }
  // Verdict = elbow straightness alone.
  if (p.peak < cfg.threshold) {
    p.predicted = "fail";
    p.reason = "bent";
    const bend = Number.isFinite(p.peak_bend_deg) ? `${p.peak_bend_deg.toFixed(0)}°` : "—";
    p.reason_text = `elbow bent: r=${p.peak.toFixed(2)} < ${cfg.threshold.toFixed(2)} (bend ${bend})`;
    return;
  }
  p.predicted = "pass";
  p.reason = "ok";
  const axPart = (cfg.axialityGate && Number.isFinite(p.peak_axiality))
    ? `, axiality ${p.peak_axiality.toFixed(2)} ≤ ${cfg.axialityMax.toFixed(2)}`
    : "";
  p.reason_text = `r=${p.peak.toFixed(2)} ≥ ${cfg.threshold.toFixed(2)}${axPart}`;
}

function activePunchAt(punches, frame) {
  for (const p of punches) {
    if (frame >= p.start_frame && frame <= p.end_frame) return p;
  }
  return null;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y,     x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x,     y + h, r);
  ctx.arcTo(x,     y + h, x,     y,     r);
  ctx.arcTo(x,     y,     x + w, y,     r);
  ctx.closePath();
}

function drawVerdictHud(ctx, punch, scale, pose) {
  const colorFor = v => v === "pass" ? COLORS.pass
                       : v === "fail" ? COLORS.fail
                       : COLORS.unclear;
  const labelTxt = punch.label ? `GT:   ${punch.label}` : "GT:   —";
  const predTxt  = `pred: ${punch.predicted}`;
  const peakTxt  = Number.isFinite(punch.peak)
    ? `peak r ${punch.peak.toFixed(3)}`
      + (Number.isFinite(punch.peak_bend_deg) ? `  ·  bend ${punch.peak_bend_deg.toFixed(1)}°` : "")
      + (Number.isFinite(punch.peak_axiality) ? `  ·  ax ${punch.peak_axiality.toFixed(2)}` : "")
    : null;
  const hand     = `${punch.hand} ${punch.punch_type}`;
  const agreeSym = punch.label && punch.predicted !== "unclear"
    ? (punch.label === punch.predicted ? "  ✓" : "  ✗") : "";

  const labelCol = punch.label ? colorFor(punch.label) : "rgba(255,255,255,0.55)";
  const predCol  = colorFor(punch.predicted);
  const agreeCol = punch.label === punch.predicted ? COLORS.pass : COLORS.fail;

  const fontPx = 15 * scale;
  const lineH  = 22 * scale;
  const padX   = 14 * scale;
  const padY   = 10 * scale;
  const x0 = 24 * scale, y0 = 24 * scale;

  ctx.save();
  ctx.font = `bold ${fontPx}px ui-monospace, "SF Mono", Menlo, monospace`;
  const lines = [hand, labelTxt + agreeSym, predTxt];
  if (peakTxt) lines.push(peakTxt);
  const w = Math.max(...lines.map(t => ctx.measureText(t).width)) + 2 * padX;
  const h = padY * 2 + lineH * lines.length;

  ctx.fillStyle = "rgba(0,0,0,0.78)";
  roundRect(ctx, x0, y0, w, h, 10 * scale);
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.lineWidth = 1 * scale;
  ctx.stroke();

  let y = y0 + padY + fontPx;
  ctx.fillStyle = "rgba(255,255,255,0.92)";
  ctx.fillText(hand, x0 + padX, y);
  y += lineH;
  ctx.fillStyle = labelCol;
  ctx.fillText(labelTxt, x0 + padX, y);
  if (agreeSym) {
    ctx.fillStyle = agreeCol;
    ctx.fillText(agreeSym, x0 + padX + ctx.measureText(labelTxt).width, y);
  }
  y += lineH;
  ctx.fillStyle = predCol;
  ctx.fillText(predTxt, x0 + padX, y);
  if (peakTxt) {
    y += lineH;
    ctx.font = `${fontPx}px ui-monospace, "SF Mono", Menlo, monospace`;
    ctx.fillStyle = "rgba(255,255,255,0.72)";
    ctx.fillText(peakTxt, x0 + padX, y);
  }
  ctx.restore();
}

function drawArmRatio(ctx, pose, frame, side, ratio, cfg, scale, bodyAxis) {
  const joints = JOINTS_FOR_SIDE[side];
  // Skip when the per-frame ratio couldn't be computed for this arm — that
  // means shoulder or elbow conf was below the gate (or the wrist was), so
  // any positions we'd plot are unreliable. Without this gate, the lens
  // happily painted bones/arc at whatever stale coords the pose model
  // returned, which looked like a lagging skeleton on rounds with poor
  // pose tracking. BUT we still want to tell the user *why* this side
  // disappeared — draw a faded ghost with a label showing which joint
  // failed the conf gate.
  if (!Number.isFinite(ratio)) {
    drawArmGhost(ctx, pose, frame, side, cfg, scale);
    return;
  }
  const sh = shoulderXY(pose, frame, joints, cfg, bodyAxis);
  const sx = sh.x, sy = sh.y;
  const ex = pose.skeleton[(frame * 17 + joints.elbow) * 2];
  const ey = pose.skeleton[(frame * 17 + joints.elbow) * 2 + 1];
  const w = wristXY(pose, frame, joints, cfg);
  if (!w) return;
  if (!Number.isFinite(sx) || !Number.isFinite(ex)) return;

  const color = ratio >= cfg.threshold ? COLORS.pass : COLORS.fail;

  ctx.save();
  ctx.lineWidth = 4 * scale;
  ctx.strokeStyle = color;
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.lineTo(ex, ey);
  ctx.lineTo(w.x, w.y);
  ctx.stroke();

  // Direct shoulder→wrist guide (dashed) — visualises the "if extended" line
  ctx.strokeStyle = COLORS.ratioGuide;
  ctx.lineWidth = 1.5 * scale;
  ctx.setLineDash([5 * scale, 4 * scale]);
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.lineTo(w.x, w.y);
  ctx.stroke();
  ctx.setLineDash([]);

  // Mark the three corners we measure against — shoulder and elbow get
  // solid dots so the user can see exactly which joints feed the ratio /
  // bend computation. The wrist gets its own glove-or-pose marker below.
  ctx.fillStyle = color;
  ctx.strokeStyle = "rgba(0,0,0,0.55)";
  ctx.lineWidth = 1.5 * scale;
  const cornerR = 5 * scale;
  // Elbow dot — always rendered the same way
  ctx.beginPath();
  ctx.arc(ex, ey, cornerR, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  // Shoulder dot — when the anatomical correction is active and actually
  // shifted the anchor, paint a much more prominent marker at the
  // CORRECTED position (the joint center we now believe in) and a
  // smaller "ghost" hollow ring at the raw acromion, connected by a
  // dashed correction vector. Without correction (or when α=0), keep the
  // single small dot at the acromion.
  const corrActive = cfg.shoulderCorrect && bodyAxis
                     && (sh.raw_x !== sx || sh.raw_y !== sy);
  if (corrActive) {
    // Dashed correction vector raw → corrected (arrow-ish — just a line
    // is enough; the relative dot weights tell direction).
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.6)";
    ctx.lineWidth = 1.5 * scale;
    ctx.setLineDash([4 * scale, 3 * scale]);
    ctx.beginPath();
    ctx.moveTo(sh.raw_x, sh.raw_y);
    ctx.lineTo(sx, sy);
    ctx.stroke();
    ctx.setLineDash([]);
    // Raw acromion — small hollow white ring + small "raw" tick label
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 2 * scale;
    ctx.beginPath();
    ctx.arc(sh.raw_x, sh.raw_y, 5 * scale, 0, Math.PI * 2);
    ctx.stroke();
    // Corrected joint center — big dot, white outer ring for contrast.
    // This is the "where we now think the shoulder is" indicator.
    const bigR = 10 * scale;
    ctx.fillStyle = color;
    ctx.lineWidth = 2.5 * scale;
    ctx.strokeStyle = "rgba(255,255,255,0.95)";
    ctx.beginPath();
    ctx.arc(sx, sy, bigR, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    // Inner darker dot so the marker reads as a target reticle
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.beginPath();
    ctx.arc(sx, sy, bigR * 0.35, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  } else {
    // No correction → single small dot at the raw shoulder position.
    ctx.beginPath();
    ctx.arc(sx, sy, cornerR, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  // Angle arc at the elbow — visualises the interior angle (elbow_angle).
  // bend = 180° − elbow_angle, so when the arc is a near-flat sweep, the
  // arm is straight; when it's a quarter-circle, the elbow is at 90°.
  // Drawn from the elbow→shoulder ray to the elbow→wrist ray, going
  // whichever direction is shorter (the interior).
  if (Number.isFinite(ratio)) {
    const aSh = Math.atan2(sy - ey, sx - ex);
    const aWr = Math.atan2(w.y - ey, w.x - ex);
    let diff = aWr - aSh;
    while (diff >  Math.PI) diff -= 2 * Math.PI;
    while (diff < -Math.PI) diff += 2 * Math.PI;
    const arcR = 28 * scale;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5 * scale;
    ctx.beginPath();
    if (diff >= 0) ctx.arc(ex, ey, arcR, aSh, aSh + diff, false);
    else            ctx.arc(ex, ey, arcR, aSh, aSh + diff, true);
    ctx.stroke();

    // Bend label inside the arc on the bisector — exact angle from
    // law of cosines, computed the same way the per-frame array uses.
    const ue = Math.hypot(sx - ex, sy - ey);
    const fa = Math.hypot(ex - w.x, ey - w.y);
    const sw2 = Math.hypot(sx - w.x, sy - w.y);
    let bendDeg = NaN;
    if (ue > 1e-3 && fa > 1e-3) {
      const cosElbow = Math.max(-1, Math.min(1,
        (ue*ue + fa*fa - sw2*sw2) / (2*ue*fa)));
      bendDeg = 180 - Math.acos(cosElbow) * (180 / Math.PI);
    }
    if (Number.isFinite(bendDeg)) {
      const bisector = aSh + diff / 2;
      const lx = ex + Math.cos(bisector) * (arcR + 18 * scale);
      const ly = ey + Math.sin(bisector) * (arcR + 18 * scale);
      ctx.font = `${12 * scale}px ui-monospace, monospace`;
      const txt = `${bendDeg.toFixed(0)}°`;
      const tw = ctx.measureText(txt).width;
      ctx.fillStyle = "rgba(0,0,0,0.7)";
      ctx.fillRect(lx - tw/2 - 4*scale, ly - 8*scale, tw + 8*scale, 16*scale);
      ctx.fillStyle = color;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(txt, lx, ly);
      ctx.textAlign = "start";
      ctx.textBaseline = "alphabetic";
    }
    ctx.restore();
  }

  // Wrist marker — square if glove, ring if pose. Same shape vocabulary as
  // the wrist_swap lens.
  if (w.source === "glove") {
    const s_ = 8 * scale;
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(w.x - s_, w.y - s_, s_ * 2, s_ * 2);
    ctx.strokeStyle = COLORS.gloveArm;
    ctx.lineWidth = 3 * scale;
    ctx.strokeRect(w.x - s_, w.y - s_, s_ * 2, s_ * 2);
  } else {
    ctx.strokeStyle = COLORS.poseArm;
    ctx.lineWidth = 3 * scale;
    ctx.beginPath();
    ctx.arc(w.x, w.y, 9 * scale, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Ratio readout next to the wrist
  if (Number.isFinite(ratio)) {
    ctx.font = `${14 * scale}px ui-monospace, monospace`;
    ctx.fillStyle = "rgba(0,0,0,0.7)";
    ctx.fillRect(w.x + 12 * scale, w.y - 16 * scale, 56 * scale, 22 * scale);
    ctx.fillStyle = color;
    ctx.fillText(ratio.toFixed(2), w.x + 18 * scale, w.y + 1 * scale);
  }
  ctx.restore();
}

// ─── DOM helpers ───────────────────────────────────────────────────────────

function setText(id, value) {
  const el = host?.querySelector("#" + id);
  if (el) el.textContent = value;
}

function formatBend(deg) {
  if (!Number.isFinite(deg)) return "—";
  return `${deg.toFixed(1)}° bend`;
}

function setMetric(id, ratio, cfg) {
  const el = host?.querySelector("#" + id);
  if (!el) return;
  if (!Number.isFinite(ratio)) {
    el.textContent = "—";
    el.style.color = "";
    return;
  }
  el.textContent = ratio.toFixed(3);
  el.style.color = ratio >= cfg.threshold ? COLORS.pass : COLORS.fail;
}

function setAxial(id, ax, cfg) {
  const el = host?.querySelector("#" + id);
  if (!el) return;
  if (!Number.isFinite(ax)) { el.textContent = "—"; el.style.color = ""; return; }
  el.textContent = ax.toFixed(2);
  if (!cfg.axialityGate) { el.style.color = "var(--muted, #888)"; return; }
  // Green when side-on enough to trust bend (≤ cut), red when too head-on.
  el.style.color = ax <= cfg.axialityMax ? COLORS.pass : COLORS.fail;
}

