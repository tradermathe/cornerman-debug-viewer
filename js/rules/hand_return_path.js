// Hand return path lens — "did the fist come straight back to guard?".
//
// rule_hand_ushape: after the punch lands, the fist should travel back to
// the guard along roughly the same line it went out on. The classic
// violation is the U-shape — the hand drops low and loops back up to the
// guard. Watch the retract, not the throw: the throw can be clean and the
// return still wrong.
//
// Per labelled punch (straights only — jab/cross head/body):
//
//   peak  = most-extended frame in the punch window (max reach =
//           |shoulder→wrist| / torso, same pick as arm_extension) — the
//           start of the return.
//   B     = return end: first frame after peak where the wrist is back
//           inside the guard radius (wrist→nose euclidean ≤ reGuardDist
//           torsos). The search is capped by the next punch on the SAME
//           hand (combos), maxReturnSec, and end of cache; if the wrist
//           never re-guards inside the cap, B falls back to the
//           closest-approach frame (annotated, still scored).
//   sag   = max perpendicular deviation of the return wrist track BELOW
//           the straight chord peak→B, normalized by the window-median
//           torso. Measured in the SHOULDER FRAME (wrist minus per-frame
//           shoulder-line midpoint) so a knee bend / bob that lowers the
//           whole body during the return doesn't masquerade as a dip.
//           A straight return reads ~0; a U reads the depth of the U.
//           Verdict is sag alone: fail if sag ≥ sagFail.
//
// AXIALITY GATE — same cut as arm_extension, joined by punch_uuid: the
// return arc is only visible when the punch travels across the image
// plane, so punches the temporal model calls too head-on are skipped
// (can't judge a path you can't see).
//
// COVERAGE GATE — the sag max over a gappy track can miss the dip
// (glove occlusion → NaN wrist, no Vision fallback). Windows with less
// than minCoverage valid interior frames score "unclear" instead of
// pretending the path was clean.
//
// Wrist source: identical contract to arm_extension — v6 cache preferred
// (glove wrists baked in at joints 9/10, conf gated by minGloveConf),
// legacy raw-glove sidecar honored, pose wrist otherwise.
//
// Compares predicted vs the labeler's rule_hand_ushape verdict when
// available — same agree/disagree pattern as arm_extension.

import { J } from "../skeleton.js";
import { gloveXY, gloveConf } from "../pose-loader.js";
import { ensureAxialityModel, axialityForPunch } from "./axiality_model.js";

const DEFAULTS = {
  sagFail:      0.20,   // fail if max sag ≥ this (torsos)
  reGuardDist:  0.60,   // wrist→nose euclidean ≤ this (torsos) = back at guard
  maxReturnSec: 1.0,    // search cap after the peak frame
  minCoverage:  0.60,   // min fraction of valid wrist frames inside the return window
  axialityGate: true,
  axialityMax:  Math.SQRT1_2,   // ≈0.7071 = cos 45° — same cut as arm_extension
  minGloveConf: 0.20,
  minPoseConf:  0.20,
  // Only straights have a "back along the same line" goal — hooks and
  // uppercuts arc by design.
  appliesTo: new Set([
    "jab_head", "jab_body",
    "cross_head", "cross_body",
  ]),
};

const COLORS = {
  pass:      "#5fd97a",
  fail:      "#e85a5a",
  unclear:   "#f5b945",
  skip:      "#7ec8ff",
  outPath:   "rgba(255,255,255,0.35)",
  chord:     "rgba(255,255,255,0.55)",
  guardRing: "rgba(126,200,255,0.45)",
  sagMark:   "#ff8bd2",
  agree:     "#5fd97a",
  disagree:  "#e85a5a",
};

// (hand, stance) → anatomical side, mirroring arm_extension / guard_drop.
const SIDE_FOR = {
  lead: { orthodox: "L", southpaw: "R" },
  rear: { orthodox: "R", southpaw: "L" },
};

const JOINTS_FOR_SIDE = {
  L: { shoulder: J.L_SHOULDER, wrist: J.L_WRIST, gloveSide: 0 },
  R: { shoulder: J.R_SHOULDER, wrist: J.R_WRIST, gloveSide: 1 },
};

let host;
let cfg = { ...DEFAULTS };
let signals = null;
let lastPose = null;
let lastDetections = null;
let latestState = null;

// Loop playback — step through the straights one at a time, video looping
// within each one. Mirrors hip_rotation_review.
let videoEl = null;
let loopWindow = null;       // {start_frame, end_frame} the video loops within
let activeIdx = -1;          // index into signals.punches currently looped
let timeupdateHandler = null;
let keydownHandler = null;

// v6 cache is the canonical wrist source — same pick as arm_extension.
function pickPose(state) {
  return state.poseV6 || state.pose;
}

export const HandReturnPathRule = {
  id: "hand_return_path",
  label: "Hand return path (straights)",

  requires(slot) {
    return !!slot?.vision_glove
      || (!!(slot?.vision || slot?.yolo) && !!slot?.glove);
  },

  skeletonStyle() {
    return {
      boneColor: "rgba(255,255,255,0.22)",
      boneWidth: 1.5,
      jointRadius: 3,
      // The lens draws its own wrist trail — keep the base renderer's
      // wrist dots out of the way, highlight the guard anchor instead.
      hideJoints: new Set([J.L_WRIST, J.R_WRIST]),
      highlightJoints: new Set([J.NOSE]),
    };
  },

  mount(_host, state) {
    host = _host;
    cfg = { ...DEFAULTS };
    latestState = state;
    ensureAxialityModel(state, onModelReady);
    signals = computeAll(state, cfg);
    lastPose = pickPose(state);
    lastDetections = state.labels?.detections || null;

    host.innerHTML = renderTemplate(signals, cfg);
    renderPunchTable();
    renderAggregate();

    const recomputeAndRefresh = () => {
      signals = computeAll(latestState, cfg);
      renderPunchTable();
      renderAggregate();
      updateAxialStatus();
      syncActiveLoop();
      window.__viewerRedraw?.();
    };
    const rescoreAndRefresh = () => {
      for (const p of signals.punches) rescorePunch(p, cfg);
      renderPunchTable();
      renderAggregate();
      updateAxialStatus();
      syncActiveLoop();
      window.__viewerRedraw?.();
    };

    // sagFail only moves the verdict — rescore, no window recompute.
    wireSlider("hrp-sag", "hrp-sag-out", "sagFail", rescoreAndRefresh);
    // These reshape the return window — full recompute.
    wireSlider("hrp-reguard", "hrp-reguard-out", "reGuardDist", recomputeAndRefresh);
    wireSlider("hrp-maxret",  "hrp-maxret-out",  "maxReturnSec", recomputeAndRefresh, 1);
    wireSlider("hrp-cov",     "hrp-cov-out",     "minCoverage", rescoreAndRefresh);
    wireSlider("hrp-pose-gate",  "hrp-pose-gate-out",  "minPoseConf", recomputeAndRefresh);
    wireSlider("hrp-glove-gate", "hrp-glove-gate-out", "minGloveConf", recomputeAndRefresh);

    // Axiality gate toggle + cut — verdict-only, like arm_extension.
    const axialToggle = host.querySelector("#hrp-axial-toggle");
    const axialSlider = host.querySelector("#hrp-axial-max");
    const axialRow    = host.querySelector("#hrp-axial-slider-row");
    if (axialToggle) {
      axialToggle.addEventListener("change", () => {
        cfg.axialityGate = axialToggle.checked;
        if (axialSlider) axialSlider.disabled = !cfg.axialityGate;
        if (axialRow) axialRow.style.opacity = cfg.axialityGate ? 1 : 0.5;
        rescoreAndRefresh();
      });
    }
    wireSlider("hrp-axial-max", "hrp-axial-max-out", "axialityMax", rescoreAndRefresh);
    updateAxialStatus();

    // Nav buttons — step through the straights, looping each in place.
    host.querySelector("#hrp-prev")?.addEventListener("click",
      () => seekToPunch(activeIdx - 1, latestState));
    host.querySelector("#hrp-next")?.addEventListener("click",
      () => seekToPunch(activeIdx + 1, latestState));
    host.querySelector("#hrp-mute")?.addEventListener("click", toggleMute);

    // Click a punch row to jump to (and loop) that straight.
    host.addEventListener("click", (ev) => {
      const tr = ev.target.closest("tr[data-idx]");
      if (!tr) return;
      seekToPunch(Number(tr.dataset.idx), latestState);
    });

    // Loop the active straight's window + N/P/M keys, mirroring
    // hip_rotation_review. Start on the first straight.
    videoEl = document.getElementById("video");
    installTimeupdateLoop();
    installKeyHandlers();
    updateMuteButton();
    if (signals.punches.length) seekToPunch(0, latestState);
    else updateCounter();
  },

  draw(ctx, state) {
    latestState = state;
    maybeRecompute(state);
    const pose = pickPose(state);
    const s = state.renderScale || 1;
    const f = state.frame;

    // Guard radius ring around the nose — the "back at guard" target.
    const nc = pose.conf[f * 17 + J.NOSE];
    const t  = signals.torso[f];
    if (nc >= cfg.minPoseConf && Number.isFinite(t) && t > 0) {
      const nx = pose.skeleton[(f * 17 + J.NOSE) * 2];
      const ny = pose.skeleton[(f * 17 + J.NOSE) * 2 + 1];
      if (Number.isFinite(nx)) {
        ctx.save();
        ctx.strokeStyle = COLORS.guardRing;
        ctx.lineWidth = 1.5 * s;
        ctx.setLineDash([4 * s, 4 * s]);
        ctx.beginPath();
        ctx.arc(nx, ny, cfg.reGuardDist * t, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
    }

    const active = activePunchAt(signals.punches, f);
    if (active) drawReturnPath(ctx, active, s);
  },

  update(state) {
    latestState = state;
    ensureAxialityModel(state, onModelReady);
    maybeRecompute(state);

    // If the user scrubbed out of the active straight, hop the loop to
    // whichever straight the cursor now sits inside — same pattern as
    // hip_rotation_review.
    const f = state.frame;
    const punches = signals.punches;
    if (punches.length) {
      const TOL = 5;
      const active = activeIdx >= 0 ? punches[activeIdx] : null;
      const aw = active ? loopWindowFor(active) : null;
      const nearActive = aw && f >= aw.start_frame - TOL && f <= aw.end_frame + TOL;
      if (!nearActive) {
        const inside = punches.findIndex(p => {
          const w = loopWindowFor(p);
          return f >= w.start_frame && f <= w.end_frame;
        });
        if (inside !== -1 && inside !== activeIdx) {
          activeIdx = inside;
          loopWindow = loopWindowFor(punches[inside]);
          updateCounter();
          updateActiveRow();
        }
      }
    }

    setNoseDist("hrp-l-dist", signals.arms.L.noseDist[f], cfg);
    setNoseDist("hrp-r-dist", signals.arms.R.noseDist[f], cfg);
    const ap = activePunchAt(signals.punches, f);
    setText("hrp-active", ap
      ? `${ap.punch_type} · sag ${Number.isFinite(ap.sag) ? ap.sag.toFixed(2) + "t" : "—"} · ${ap.predicted}`
      : "—");
  },

  unmount() {
    if (videoEl && timeupdateHandler) videoEl.removeEventListener("timeupdate", timeupdateHandler);
    if (keydownHandler) document.removeEventListener("keydown", keydownHandler, true);
    timeupdateHandler = null;
    keydownHandler = null;
    loopWindow = null;
    activeIdx = -1;
  },
};

function maybeRecompute(state) {
  const pose = pickPose(state);
  const dets = state.labels?.detections || null;
  if (pose !== lastPose || dets !== lastDetections) {
    signals = computeAll(state, cfg);
    lastPose = pose;
    lastDetections = dets;
    // New round (or relabeled) — rebuild the table and reset the loop to
    // the first straight so N/P walks the new round cleanly.
    renderPunchTable();
    renderAggregate();
    if (signals.punches.length) seekToPunch(0, state);
    else { activeIdx = -1; loopWindow = null; updateCounter(); }
  }
}

// ─── compute ───────────────────────────────────────────────────────────────

function computeAll(state, cfg) {
  const pose = pickPose(state);
  const N = pose.n_frames;
  const fps = pose.fps;
  const startSec = pose.start_sec || 0;

  const torso = torsoEuclidPerFrame(pose, cfg);
  const midShoulder = midShoulderPerFrame(pose, cfg);
  const arms = {
    L: perFrameArm(pose, "L", cfg, torso),
    R: perFrameArm(pose, "R", cfg, torso),
  };

  // Every detection gets a side — the return-window cap needs the next
  // punch on the same hand even when that next punch isn't a straight.
  const all = (state.labels?.detections || []).map(d => {
    const stance = (d.stance === "southpaw" || d.stance === "orthodox")
      ? d.stance : "orthodox";
    return { d, stance, side: SIDE_FOR[d.hand]?.[stance] || "L" };
  });

  const punches = all
    .filter(({ d }) => cfg.appliesTo.has(d.punch_type))
    .map(({ d, stance, side }, idx) =>
      buildPunch(d, stance, side, idx, { arms, torso, midShoulder, all, N, fps, startSec, cfg }));

  return { arms, torso, midShoulder, punches, fps };
}

function buildPunch(d, stance, side, idx, ctx) {
  const { arms, torso, midShoulder, all, N, fps, startSec, cfg } = ctx;
  const arm = arms[side];
  const sf = Math.max(0, d.start_frame);
  const ef = Math.min(N - 1, d.end_frame);

  // Peak = most-extended frame in the punch window. Prefer reach
  // (|sh→wr|/torso); fall back to wrist→nose distance when the shoulder
  // or torso gates killed reach.
  let peakFrame = -1, best = -Infinity;
  for (let f = sf; f <= ef; f++) {
    const r = arm.reach[f];
    if (Number.isFinite(r) && r > best) { best = r; peakFrame = f; }
  }
  if (peakFrame < 0) {
    best = -Infinity;
    for (let f = sf; f <= ef; f++) {
      const nd = arm.noseDist[f];
      if (Number.isFinite(nd) && nd > best) { best = nd; peakFrame = f; }
    }
  }

  const peak_axiality = axialityForPunch(d.punch_uuid)?.predAxiality ?? NaN;
  const label = d.rule_hand_ushape === "pass" || d.rule_hand_ushape === "fail"
    ? d.rule_hand_ushape : null;

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
    peak_frame: peakFrame,
    peak_valid: peakFrame >= 0,
    b_frame: -1,
    sag_frame: -1,
    re_guarded: false,
    has_return: false,
    return_sec: NaN,
    sag: NaN,
    coverage: NaN,
    closest_dist: NaN,
    peak_axiality,
    label,
  };
  if (!p.peak_valid) { rescorePunch(p, cfg); return p; }

  // Search cap: maxReturnSec, end of cache, and the next punch thrown
  // with the SAME hand (in a combo that hand is busy again — the return,
  // whatever its shape, is over).
  let cap = Math.min(N - 1, peakFrame + Math.round(cfg.maxReturnSec * fps));
  for (const { d: od, side: oside } of all) {
    if (oside !== side) continue;
    if (od.start_frame > peakFrame && od.start_frame - 1 < cap) {
      cap = od.start_frame - 1;
    }
  }

  // B = first re-guard frame, else closest approach inside the cap.
  let bFrame = -1, minDist = Infinity, minDistFrame = -1;
  for (let f = peakFrame + 1; f <= cap; f++) {
    const dist = arm.noseDist[f];
    if (!Number.isFinite(dist)) continue;
    if (dist < minDist) { minDist = dist; minDistFrame = f; }
    if (dist <= cfg.reGuardDist) { bFrame = f; break; }
  }
  p.re_guarded = bFrame >= 0;
  if (bFrame < 0) bFrame = minDistFrame;
  if (bFrame < 0) { rescorePunch(p, cfg); return p; }  // no valid frames after peak

  p.b_frame = bFrame;
  p.has_return = true;
  p.return_sec = (bFrame - peakFrame) / fps;
  p.closest_dist = p.re_guarded ? arm.noseDist[bFrame] : minDist;

  // Sag: max deviation below the chord A(peak)→B, over interior frames —
  // measured in the SHOULDER FRAME. Each wrist sample has the per-frame
  // shoulder-line midpoint subtracted, so whole-body vertical motion during
  // the return (knee bend, bob, a step down) is cancelled: it moves the
  // wrist in the image but not relative to the body, and must not read as a
  // U. A genuine loop still curves relative to the shoulders. We only
  // subtract a translation, so +y is still image-down and the normal logic
  // is unchanged. Mid-shoulder is finite at A and B by construction (peak is
  // picked via reach, B via nose distance — both already require valid
  // shoulders/torso).
  const { msx, msy } = midShoulder;
  const Ax = arm.wx[peakFrame] - msx[peakFrame], Ay = arm.wy[peakFrame] - msy[peakFrame];
  const Bx = arm.wx[bFrame]   - msx[bFrame],     By = arm.wy[bFrame]   - msy[bFrame];
  const dx = Bx - Ax, dy = By - Ay;
  const len = Math.hypot(dx, dy);
  // Chord normal pointing "down" in image coords (+y). Degenerate chord
  // (A≈B — punch barely left the guard) falls back to straight-down.
  let nx = 0, ny = 1;
  if (len >= 1e-3) {
    nx = -dy / len; ny = dx / len;
    if (ny < 0) { nx = -nx; ny = -ny; }
  }
  let sagPx = 0, sagFrame = -1, nValid = 0;
  const nTotal = Math.max(0, bFrame - peakFrame - 1);
  for (let f = peakFrame + 1; f < bFrame; f++) {
    const x = arm.wx[f] - msx[f], y = arm.wy[f] - msy[f];
    if (!Number.isFinite(x)) continue;   // wrist OR shoulder missing → skip
    nValid++;
    const s = (x - Ax) * nx + (y - Ay) * ny;
    if (s > sagPx) { sagPx = s; sagFrame = f; }
  }
  p.coverage = nTotal > 0 ? nValid / nTotal : 1;
  p.sag_frame = sagFrame;

  const tVals = [];
  for (let f = peakFrame; f <= bFrame; f++) {
    if (Number.isFinite(torso[f])) tVals.push(torso[f]);
  }
  const tMed = median(tVals);
  p.sag = (Number.isFinite(tMed) && tMed > 0) ? sagPx / tMed : NaN;

  rescorePunch(p, cfg);
  return p;
}

function rescorePunch(p, cfg) {
  // Single source of truth for predicted/reason — same contract as
  // arm_extension.rescorePunch.
  if (!p.peak_valid) {
    p.predicted = "unclear";
    p.reason = "no_peak";
    p.reason_text = "no valid peak frame (wrist/shoulder below conf gates throughout the punch window)";
    return;
  }
  if (cfg.axialityGate) {
    if (!Number.isFinite(p.peak_axiality)) {
      p.predicted = "skip";
      p.reason = "axial_unknown";
      p.reason_text = "the axiality model didn't score this punch — can't tell if the return arc is visible";
      return;
    }
    if (p.peak_axiality > cfg.axialityMax) {
      p.predicted = "skip";
      p.reason = "axial";
      p.reason_text = `axiality ${p.peak_axiality.toFixed(2)} > ${cfg.axialityMax.toFixed(2)} — too head-on, return arc not visible in 2D`;
      return;
    }
  }
  if (!p.has_return) {
    p.predicted = "unclear";
    p.reason = "no_return";
    p.reason_text = "no valid wrist frames after the peak inside the search window";
    return;
  }
  if (p.coverage < cfg.minCoverage) {
    p.predicted = "unclear";
    p.reason = "wrist_gaps";
    p.reason_text = `only ${Math.round(100 * p.coverage)}% of return frames have a wrist — gaps could hide the dip`;
    return;
  }
  if (!Number.isFinite(p.sag)) {
    p.predicted = "unclear";
    p.reason = "no_torso";
    p.reason_text = "no valid torso in the return window — can't normalize the sag";
    return;
  }
  const tail = p.re_guarded
    ? `re-guarded in ${p.return_sec.toFixed(2)}s`
    : `never re-guarded inside the cap (closest approach ${p.closest_dist.toFixed(2)}t)`;
  if (p.sag >= cfg.sagFail) {
    p.predicted = "fail";
    p.reason = "sag";
    p.reason_text = `max sag ${p.sag.toFixed(2)}t ≥ ${cfg.sagFail.toFixed(2)} — U-shaped return; ${tail}`;
  } else {
    p.predicted = "pass";
    p.reason = "straight";
    p.reason_text = `max sag ${p.sag.toFixed(2)}t < ${cfg.sagFail.toFixed(2)} — return tracks the chord; ${tail}`;
  }
}

// Per-frame wrist track + reach + wrist→nose distance for one side.
function perFrameArm(pose, side, cfg, torso) {
  const N = pose.n_frames;
  const joints = JOINTS_FOR_SIDE[side];
  const wx = new Float32Array(N).fill(NaN);
  const wy = new Float32Array(N).fill(NaN);
  const reach = new Float32Array(N).fill(NaN);
  const noseDist = new Float32Array(N).fill(NaN);
  const source = new Array(N).fill(null);
  for (let f = 0; f < N; f++) {
    const w = wristXY(pose, f, joints, cfg);
    if (!w) continue;
    wx[f] = w.x; wy[f] = w.y; source[f] = w.source;
    const t = torso[f];
    const tOk = Number.isFinite(t) && t > 0;
    const sc = pose.conf[f * 17 + joints.shoulder];
    if (sc >= cfg.minPoseConf && tOk) {
      const sx = pose.skeleton[(f * 17 + joints.shoulder) * 2];
      const sy = pose.skeleton[(f * 17 + joints.shoulder) * 2 + 1];
      if (Number.isFinite(sx)) reach[f] = Math.hypot(w.x - sx, w.y - sy) / t;
    }
    const nc = pose.conf[f * 17 + J.NOSE];
    if (nc >= cfg.minPoseConf && tOk) {
      const nx = pose.skeleton[(f * 17 + J.NOSE) * 2];
      const ny = pose.skeleton[(f * 17 + J.NOSE) * 2 + 1];
      if (Number.isFinite(nx)) noseDist[f] = Math.hypot(w.x - nx, w.y - ny) / t;
    }
  }
  return { wx, wy, reach, noseDist, source };
}

// Same wrist contract as arm_extension: legacy glove sidecar first, then
// v6 (glove baked in, minGloveConf, no Vision fallback) / pose wrist.
function wristXY(pose, frame, joints, cfg) {
  const g = pose.gloveWrists;
  if (g) {
    const [gx, gy] = gloveXY(g, frame, joints.gloveSide);
    const gc       = gloveConf(g, frame, joints.gloveSide);
    if (gc >= cfg.minGloveConf && Number.isFinite(gx) && Number.isFinite(gy)) {
      return { x: gx, y: gy, source: "glove" };
    }
  }
  const px = pose.skeleton[(frame * 17 + joints.wrist) * 2];
  const py = pose.skeleton[(frame * 17 + joints.wrist) * 2 + 1];
  const pc = pose.conf[frame * 17 + joints.wrist];
  const isGloveBaked = pose.meta?.wrist_replaced_with_glove === true;
  const gate = isGloveBaked ? cfg.minGloveConf : cfg.minPoseConf;
  if (pc < gate || !Number.isFinite(px)) return null;
  return { x: px, y: py, source: isGloveBaked ? "glove" : "pose" };
}

// Per-frame shoulder-line midpoint (L/R shoulder average). NaN when either
// shoulder is below the conf gate. This is the reference the sag is measured
// against: subtracting it cancels whole-body vertical motion (knee bend, bob,
// a step down) during the return, which moves the wrist in the image but not
// relative to the body — so it must not read as a U.
function midShoulderPerFrame(pose, cfg) {
  const N = pose.n_frames;
  const msx = new Float32Array(N).fill(NaN);
  const msy = new Float32Array(N).fill(NaN);
  const gate = cfg.minPoseConf;
  for (let f = 0; f < N; f++) {
    const cL = pose.conf[f * 17 + J.L_SHOULDER];
    const cR = pose.conf[f * 17 + J.R_SHOULDER];
    if (cL < gate || cR < gate) continue;
    const lx = pose.skeleton[(f * 17 + J.L_SHOULDER) * 2];
    const ly = pose.skeleton[(f * 17 + J.L_SHOULDER) * 2 + 1];
    const rx = pose.skeleton[(f * 17 + J.R_SHOULDER) * 2];
    const ry = pose.skeleton[(f * 17 + J.R_SHOULDER) * 2 + 1];
    if (!Number.isFinite(lx) || !Number.isFinite(rx)) continue;
    msx[f] = 0.5 * (lx + rx);
    msy[f] = 0.5 * (ly + ry);
  }
  return { msx, msy };
}

// Per-frame euclidean torso |shoulder_mid → hip_mid| — same normalizer
// arm_extension / stance_width use.
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
    out[f] = Math.hypot(
      0.5 * (lsx + rsx) - 0.5 * (lhx + rhx),
      0.5 * (lsy + rsy) - 0.5 * (lhy + rhy),
    );
  }
  return out;
}

function median(arr) {
  if (!arr.length) return NaN;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

function activePunchAt(punches, frame) {
  // Active through the return, not just the labelled window — the lens'
  // whole point is what happens after the peak.
  return punches.find(p =>
    frame >= p.start_frame
    && frame <= Math.max(p.end_frame, p.b_frame >= 0 ? p.b_frame : p.end_frame)
  ) || null;
}

// The axiality sidecar loads async — recompute the join when it lands.
function onModelReady() {
  if (!host || !latestState) return;
  signals = computeAll(latestState, cfg);
  lastPose = pickPose(latestState);
  lastDetections = latestState.labels?.detections || null;
  renderPunchTable();
  renderAggregate();
  updateAxialStatus();
  window.__viewerRedraw?.();
}

function updateAxialStatus() {
  const el = host?.querySelector("#hrp-axial-status");
  if (!el) return;
  if (!cfg.axialityGate) { el.textContent = "Disabled — every punch scored regardless of foreshortening."; return; }
  const N = signals.punches.length;
  const skipped = signals.punches.filter(p => p.reason === "axial").length;
  const unknown = signals.punches.filter(p => p.reason === "axial_unknown").length;
  el.textContent = `${skipped}/${N} too head-on (skipped) · ${unknown} not scored by the model.`;
}

// ─── loop playback (mirrors hip_rotation_review) ─────────────────────────────

// Window the video loops within for a punch: the throw plus the full
// return (peak → re-guard / closest approach), so you watch the retract.
function loopWindowFor(p) {
  const end = Math.max(p.end_frame, p.b_frame >= 0 ? p.b_frame : p.end_frame);
  return { start_frame: p.start_frame, end_frame: end };
}

function seekToPunch(idx, state) {
  const punches = signals?.punches || [];
  if (!punches.length || !state) return;
  if (idx < 0) idx = 0;
  if (idx >= punches.length) idx = punches.length - 1;
  activeIdx = idx;
  loopWindow = loopWindowFor(punches[idx]);
  if (videoEl && state.fps) {
    videoEl.currentTime = (state.start_sec || 0) + loopWindow.start_frame / state.fps;
    if (videoEl.paused) {
      const pr = videoEl.play();
      if (pr && typeof pr.catch === "function") pr.catch(() => {});
    }
  }
  updateCounter();
  updateActiveRow();
}

function installTimeupdateLoop() {
  if (!videoEl) return;
  if (timeupdateHandler) videoEl.removeEventListener("timeupdate", timeupdateHandler);
  timeupdateHandler = () => {
    if (latestState?.rule?.id !== "hand_return_path") return;
    if (!loopWindow || !latestState.fps) return;
    const endTime = (latestState.start_sec || 0) + (loopWindow.end_frame + 0.5) / latestState.fps;
    if (videoEl.currentTime > endTime) {
      videoEl.currentTime = (latestState.start_sec || 0) + loopWindow.start_frame / latestState.fps;
    }
  };
  videoEl.addEventListener("timeupdate", timeupdateHandler);
}

function installKeyHandlers() {
  // Capture phase so we run before the viewer's bubble-phase keydown
  // listener and can clamp out-of-loop frame steps before they fire.
  if (keydownHandler) document.removeEventListener("keydown", keydownHandler, true);
  keydownHandler = (e) => {
    if (latestState?.rule?.id !== "hand_return_path") return;
    const tag = e.target?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA") return;

    if (e.key === "n" || e.key === "N") { e.preventDefault(); seekToPunch(activeIdx + 1, latestState); return; }
    if (e.key === "p" || e.key === "P") { e.preventDefault(); seekToPunch(activeIdx - 1, latestState); return; }
    if (e.key === "m" || e.key === "M") { e.preventDefault(); toggleMute(); return; }

    if (!loopWindow) return;
    let delta = 0;
    if      (e.key === "ArrowLeft")  delta = -1;
    else if (e.key === "ArrowRight") delta = +1;
    else if (e.key === "[")          delta = -10;
    else if (e.key === "]")          delta = +10;
    else return;
    const f = latestState.frame;
    if (f < loopWindow.start_frame || f > loopWindow.end_frame) return;
    const target = f + delta;
    if (target < loopWindow.start_frame || target > loopWindow.end_frame) {
      e.preventDefault();
      e.stopImmediatePropagation();
    }
  };
  document.addEventListener("keydown", keydownHandler, true);
}

function toggleMute() {
  if (!videoEl) return;
  videoEl.muted = !videoEl.muted;
  updateMuteButton();
}

function updateMuteButton() {
  const btn = host?.querySelector("#hrp-mute");
  if (btn && videoEl) btn.textContent = videoEl.muted ? "unmute (M)" : "mute (M)";
}

function updateCounter() {
  const el = host?.querySelector("#hrp-counter");
  if (!el) return;
  const n = signals?.punches?.length || 0;
  el.textContent = n
    ? `${Math.min(activeIdx + 1, n)} / ${n} straights`
    : "no straights in this round";
}

// Keep activeIdx / loopWindow valid after a recompute (slider change) so the
// loop bounds follow the new return window without yanking playback.
function syncActiveLoop() {
  const punches = signals?.punches || [];
  if (!punches.length) { activeIdx = -1; loopWindow = null; updateCounter(); return; }
  if (activeIdx >= punches.length) activeIdx = punches.length - 1;
  if (activeIdx >= 0) loopWindow = loopWindowFor(punches[activeIdx]);
  updateCounter();
}

function updateActiveRow() {
  if (!host) return;
  host.querySelectorAll("tr[data-idx]").forEach(tr => {
    const on = Number(tr.dataset.idx) === activeIdx;
    tr.style.background = on ? "rgba(255,210,74,0.14)" : "";
    tr.style.fontWeight = on ? "600" : "normal";
  });
}

// ─── draw ──────────────────────────────────────────────────────────────────

function drawReturnPath(ctx, p, scale) {
  if (!p.peak_valid || !p.has_return) return;
  const arm = signals.arms[p.side];
  const col = COLORS[p.predicted] || COLORS.unclear;

  // Out path (start→peak), faint — context for "same line back".
  drawTrail(ctx, arm, p.start_frame, p.peak_frame, COLORS.outPath, 1.5, [3, 3], scale);
  // Chord A→B, dashed — the "if it came straight back" line.
  const Ax = arm.wx[p.peak_frame], Ay = arm.wy[p.peak_frame];
  const Bx = arm.wx[p.b_frame],   By = arm.wy[p.b_frame];
  ctx.save();
  ctx.strokeStyle = COLORS.chord;
  ctx.lineWidth = 1.5 * scale;
  ctx.setLineDash([6 * scale, 4 * scale]);
  ctx.beginPath(); ctx.moveTo(Ax, Ay); ctx.lineTo(Bx, By); ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
  // Return path (peak→B), verdict-colored.
  drawTrail(ctx, arm, p.peak_frame, p.b_frame, col, 3, null, scale);

  ctx.save();
  // Endpoints: A = peak (filled), B = re-guard / closest approach (ring).
  ctx.fillStyle = col;
  ctx.beginPath(); ctx.arc(Ax, Ay, 5 * scale, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = col;
  ctx.lineWidth = 2 * scale;
  ctx.beginPath(); ctx.arc(Bx, By, 6 * scale, 0, Math.PI * 2); ctx.stroke();

  // Max-sag point + readout.
  if (p.sag_frame >= 0 && Number.isFinite(p.sag)) {
    const sx = arm.wx[p.sag_frame], sy = arm.wy[p.sag_frame];
    if (Number.isFinite(sx)) {
      ctx.strokeStyle = COLORS.sagMark;
      ctx.lineWidth = 2 * scale;
      const r = 6 * scale;
      ctx.beginPath();
      ctx.moveTo(sx - r, sy - r); ctx.lineTo(sx + r, sy + r);
      ctx.moveTo(sx - r, sy + r); ctx.lineTo(sx + r, sy - r);
      ctx.stroke();
      const txt = `sag ${p.sag.toFixed(2)}t · ${p.predicted}${p.label ? ` (GT ${p.label})` : ""}`;
      ctx.font = `${12 * scale}px ui-monospace, monospace`;
      const tw = ctx.measureText(txt).width;
      ctx.fillStyle = "rgba(0,0,0,0.65)";
      ctx.fillRect(sx + 10 * scale, sy - 9 * scale, tw + 8 * scale, 18 * scale);
      ctx.fillStyle = col;
      ctx.fillText(txt, sx + 14 * scale, sy + 4 * scale);
    }
  }
  ctx.restore();
}

function drawTrail(ctx, arm, f0, f1, color, width, dash, scale) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width * scale;
  if (dash) ctx.setLineDash(dash.map(d => d * scale));
  ctx.beginPath();
  let pen = false;
  for (let f = f0; f <= f1; f++) {
    const x = arm.wx[f], y = arm.wy[f];
    if (!Number.isFinite(x)) { pen = false; continue; }
    if (!pen) { ctx.moveTo(x, y); pen = true; }
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();
}

// ─── render ────────────────────────────────────────────────────────────────

function renderTemplate(sig, cfg) {
  const hasLabels = sig.punches.some(p => p.label !== null);
  return `
    <h2>Hand return path (straights)</h2>
    <p class="hint">
      After the peak, the wrist should travel back to the guard along the
      straight chord peak→re-guard. <b>sag</b> = max deviation below that
      chord, in torsos — ~0 for a straight return, the depth of the U for
      a looping one. Measured relative to the shoulder line, so bending the
      knees / bobbing mid-return doesn't count as a dip.
    </p>

    <h3>Legend</h3>
    <ul class="hint" style="list-style:none;padding-left:0;margin:0 0 12px 0;line-height:1.7">
      <li><span style="display:inline-block;width:24px;height:1px;border-top:2px dashed ${COLORS.outPath};vertical-align:middle"></span>
        &nbsp;out path (punch start → peak), context only</li>
      <li><span style="display:inline-block;width:24px;height:1px;border-top:2px dashed ${COLORS.chord};vertical-align:middle"></span>
        &nbsp;chord peak → re-guard — the "straight back" reference</li>
      <li><span style="display:inline-block;width:24px;height:3px;background:${COLORS.pass};vertical-align:middle"></span>
        / <span style="display:inline-block;width:24px;height:3px;background:${COLORS.fail};vertical-align:middle"></span>
        &nbsp;actual return path, colored by verdict</li>
      <li><span style="color:${COLORS.sagMark};font-weight:700">✕</span>
        &nbsp;max-sag point with the sag readout</li>
      <li><span style="display:inline-block;width:14px;height:14px;border:2px dashed ${COLORS.guardRing};border-radius:50%;vertical-align:middle"></span>
        &nbsp;guard radius around the nose (re-guard target)</li>
    </ul>

    <h3>Live wrist → nose (torsos)</h3>
    <div class="metric-grid">
      <div class="metric">
        <div class="metric-label">L wrist</div>
        <div class="metric-val" id="hrp-l-dist">—</div>
      </div>
      <div class="metric">
        <div class="metric-label">R wrist</div>
        <div class="metric-val" id="hrp-r-dist">—</div>
      </div>
      <div class="metric">
        <div class="metric-label">active punch</div>
        <div class="metric-val" id="hrp-active" style="font-size:14px">—</div>
      </div>
    </div>

    <h3>Sag fail threshold</h3>
    <div class="slider-row">
      <input type="range" id="hrp-sag" min="0.05" max="0.50" step="0.01" value="${cfg.sagFail}" />
      <output id="hrp-sag-out">${cfg.sagFail.toFixed(2)}</output>
      <span class="muted small">max sag (torsos) — at or above = U-shaped return</span>
    </div>

    <h3>Return window</h3>
    <p class="hint">
      The return runs from the peak to the first frame the wrist is back
      inside the guard radius, capped by the next punch on the same hand,
      the time cap, and end of cache. No re-guard inside the cap → the
      closest-approach frame ends the window (annotated in "why").
    </p>
    <div class="slider-row">
      <input type="range" id="hrp-reguard" min="0.30" max="1.00" step="0.05" value="${cfg.reGuardDist}" />
      <output id="hrp-reguard-out">${cfg.reGuardDist.toFixed(2)}</output>
      <span class="muted small">guard radius — wrist→nose distance (torsos) counting as "back"</span>
    </div>
    <div class="slider-row">
      <input type="range" id="hrp-maxret" min="0.3" max="2.0" step="0.1" value="${cfg.maxReturnSec}" />
      <output id="hrp-maxret-out">${cfg.maxReturnSec.toFixed(1)}</output>
      <span class="muted small">time cap after the peak (s)</span>
    </div>

    <h3>Axiality gate (sideways only)</h3>
    <p class="hint">
      The return arc is read in 2D, so it's only visible when the punch
      travels across the image plane — same gate as arm_extension, same
      trained temporal model, joined by punch_uuid. Punches above the cut
      are too head-on and get a <b>skip</b>.
    </p>
    <label style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
      <input type="checkbox" id="hrp-axial-toggle" ${cfg.axialityGate ? 'checked' : ''} />
      <span>Enable axiality gate</span>
    </label>
    <div class="slider-row" style="opacity:${cfg.axialityGate ? 1 : 0.5}" id="hrp-axial-slider-row">
      <input type="range" id="hrp-axial-max" min="0.30" max="1.00" step="0.01"
        value="${cfg.axialityMax}" ${cfg.axialityGate ? '' : 'disabled'} />
      <output id="hrp-axial-max-out">${cfg.axialityMax.toFixed(2)}</output>
      <span class="muted small">max axiality (lower = stricter / demand more side-on; 0.71 ≈ 45°)</span>
    </div>
    <p class="hint muted small" id="hrp-axial-status" style="margin:4px 0 0 0">—</p>

    <h3>Coverage + confidence gates</h3>
    <p class="hint">
      A gappy wrist track can hide the dip — windows with too few valid
      return frames score <b>unclear</b> instead of pretending the path
      was clean. Conf gates match the other lenses (v6: glove conf below
      the gate = no wrist, no Vision fallback).
    </p>
    <div class="slider-row">
      <input type="range" id="hrp-cov" min="0.20" max="1.00" step="0.05" value="${cfg.minCoverage}" />
      <output id="hrp-cov-out">${cfg.minCoverage.toFixed(2)}</output>
      <span class="muted small">min valid-frame fraction inside the return window</span>
    </div>
    <div class="slider-row">
      <input type="range" id="hrp-pose-gate" min="0.05" max="0.95" step="0.05" value="${cfg.minPoseConf}" />
      <output id="hrp-pose-gate-out">${cfg.minPoseConf.toFixed(2)}</output>
      <span class="muted small">pose conf — nose, shoulder, wrist-fallback, hip</span>
    </div>
    <div class="slider-row">
      <input type="range" id="hrp-glove-gate" min="0.05" max="0.95" step="0.05" value="${cfg.minGloveConf}" />
      <output id="hrp-glove-gate-out">${cfg.minGloveConf.toFixed(2)}</output>
      <span class="muted small">glove conf — below = no wrist detection on v6 rounds</span>
    </div>

    <h3>Per-punch (straights only)</h3>
    <p class="hint">
      The video loops within each straight (throw → return). Step with the
      buttons or <code>N</code>/<code>P</code>, <code>M</code> mutes, or
      click a row to jump to it. ${hasLabels
        ? "Verdict shows labeler vs predicted; ✓ when they agree."
        : "<span class='muted'>No <code>rule_hand_ushape</code> labels found — predicted only.</span>"}
    </p>
    <div class="ol-nav" style="display:flex;gap:8px;align-items:center;margin:6px 0 10px">
      <button id="hrp-prev" class="orient-btn-action secondary" style="padding:6px 10px">⏮ prev (P)</button>
      <button id="hrp-next" class="orient-btn-action secondary" style="padding:6px 10px">next (N) ⏭</button>
      <button id="hrp-mute" class="orient-btn-action secondary" style="padding:6px 10px">mute (M)</button>
      <span id="hrp-counter" style="margin-left:6px;color:#888;font-size:12px"></span>
    </div>
    <div id="hrp-table-host"></div>

    <h3>Aggregate</h3>
    <div id="hrp-aggregate" class="metric-grid"></div>
  `;
}

function pill(value) {
  if (value !== "pass" && value !== "fail" && value !== "unclear" && value !== "skip") {
    return `<span class="hrp-pill hrp-pill-empty" title="no label">—</span>`;
  }
  const col = COLORS[value] || COLORS.unclear;
  return `<span class="hrp-pill" style="background:${col}1f;color:${col};border:1px solid ${col}66">${value}</span>`;
}

function renderPunchTable() {
  const hasAnyLabel = signals.punches.some(p => p.label);
  const tbody = signals.punches.length
    ? signals.punches.map(p => {
        const tsStr = Number.isFinite(p.t_abs) ? p.t_abs.toFixed(2) : "—";
        let match = "";
        if (p.label && (p.predicted === "pass" || p.predicted === "fail")) {
          match = p.label === p.predicted
            ? `<span style="color:${COLORS.agree}" title="GT ${p.label} · agrees">✓</span>`
            : `<span style="color:${COLORS.disagree}" title="GT ${p.label} · disagrees">✗</span>`;
        }
        const sagCell = Number.isFinite(p.sag)
          ? `<td style="color:${p.sag >= cfg.sagFail ? COLORS.fail : COLORS.pass};font-variant-numeric:tabular-nums">${p.sag.toFixed(2)}</td>`
          : `<td class="muted">—</td>`;
        const retCell = Number.isFinite(p.return_sec)
          ? `<td class="muted" style="font-variant-numeric:tabular-nums">${p.return_sec.toFixed(2)}s${p.re_guarded ? "" : " ⃠"}</td>`
          : `<td class="muted">—</td>`;
        let axialCell;
        if (!Number.isFinite(p.peak_axiality)) {
          axialCell = `<td class="muted">—</td>`;
        } else {
          const col = !cfg.axialityGate ? "var(--muted, #888)"
                    : (p.peak_axiality <= cfg.axialityMax) ? COLORS.pass
                    : COLORS.fail;
          axialCell = `<td style="color:${col};font-variant-numeric:tabular-nums">${p.peak_axiality.toFixed(2)}</td>`;
        }
        const covCell = Number.isFinite(p.coverage)
          ? `<td style="color:${p.coverage >= cfg.minCoverage ? COLORS.pass : COLORS.fail};font-variant-numeric:tabular-nums">${Math.round(100 * p.coverage)}%</td>`
          : `<td class="muted">—</td>`;
        const reasonText = (p.reason_text || "").replace(/"/g, '&quot;');
        return `<tr data-idx="${p.idx}" style="cursor:pointer">
          <td>${tsStr}s</td>
          <td>${p.punch_type}</td>
          <td>${pill(p.predicted)}</td>
          <td style="text-align:center">${match}</td>
          <td class="muted small" title="${reasonText}" style="white-space:nowrap">${p.reason || "—"}</td>
          ${sagCell}
          ${retCell}
          ${axialCell}
          ${covCell}
        </tr>`;
      }).join("")
    : `<tr><td colspan="9" class="muted">no labeled straights in this round</td></tr>`;

  const tableHost = host.querySelector("#hrp-table-host");
  if (!tableHost) return;
  const gtNote = hasAnyLabel
    ? ""
    : `<p class="hint muted" style="margin:0 0 8px 0">No <code>rule_hand_ushape</code> GT verdicts attached to this round (Sheet labels missing or no match) — match column will be blank.</p>`;
  tableHost.innerHTML = `
    ${gtNote}
    <style>
      .hrp-pill {
        display:inline-block; padding:1px 8px; border-radius:10px;
        font-size:12px; font-weight:600; letter-spacing:0.02em;
        font-family: inherit;
      }
      .hrp-pill-empty {
        background:transparent; color:var(--muted, #888);
        border:1px dashed currentColor;
      }
    </style>
    <table class="rule-table">
      <thead><tr>
        <th>t</th><th>type</th><th>pred</th><th title="agrees with GT verdict">vs GT</th>
        <th title="which gate decided this punch — hover for the numbers that fired it">why</th>
        <th title="max sag below the peak→re-guard chord (torsos)">sag</th>
        <th title="peak → return end; ⃠ = never re-guarded inside the cap (closest approach used)">ret</th>
        <th title="axiality model's per-punch prediction (0 = side-on, 1 = down the camera axis) — above the cut = arc not visible, skipped">axiality</th>
        <th title="valid wrist frames inside the return window">cov</th>
      </tr></thead>
      <tbody>${tbody}</tbody>
    </table>`;
  updateActiveRow();
}

function renderAggregate() {
  const host_ = host.querySelector("#hrp-aggregate");
  if (!host_) return;
  const scored = signals.punches.filter(p => p.predicted === "pass" || p.predicted === "fail");
  if (!scored.length) {
    host_.innerHTML = `<div class="metric muted">no scorable punches</div>`;
    return;
  }
  const passed = scored.filter(p => p.predicted === "pass").length;
  const skipped = signals.punches.filter(p => p.predicted === "skip").length;
  const meanSag = scored.reduce((s, p) => s + p.sag, 0) / scored.length;
  const labelled = scored.filter(p => p.label);
  const agree = labelled.filter(p => p.label === p.predicted).length;
  const agreePct = labelled.length
    ? `${Math.round(100 * agree / labelled.length)}%` : "—";
  host_.innerHTML = `
    <div class="metric">
      <div class="metric-label">scored</div>
      <div class="metric-val">${scored.length}</div>
      <div class="metric-sub">of ${signals.punches.length} labelled straights</div>
    </div>
    <div class="metric">
      <div class="metric-label">predicted pass</div>
      <div class="metric-val">${passed}</div>
      <div class="metric-sub">${Math.round(100 * passed / scored.length)}% · sag &lt; ${cfg.sagFail.toFixed(2)}</div>
    </div>
    <div class="metric">
      <div class="metric-label">skipped (axial)</div>
      <div class="metric-val">${skipped}</div>
      <div class="metric-sub">return arc not visible</div>
    </div>
    <div class="metric">
      <div class="metric-label">mean sag</div>
      <div class="metric-val">${meanSag.toFixed(3)}</div>
      <div class="metric-sub">torsos (scored only)</div>
    </div>
    <div class="metric">
      <div class="metric-label">label agreement</div>
      <div class="metric-val">${agreePct}</div>
      <div class="metric-sub">${agree} / ${labelled.length} match (scored only)</div>
    </div>
  `;
}

// ─── small DOM helpers ─────────────────────────────────────────────────────

function wireSlider(sliderId, outId, cfgKey, onChange, digits = 2) {
  const s = host.querySelector("#" + sliderId);
  const o = host.querySelector("#" + outId);
  if (!s) return;
  s.addEventListener("input", () => {
    cfg[cfgKey] = Number(s.value);
    if (o) o.textContent = cfg[cfgKey].toFixed(digits);
    onChange();
  });
}

function setText(id, txt) {
  const el = host?.querySelector("#" + id);
  if (el) el.textContent = txt;
}

function setNoseDist(id, v, cfg) {
  const el = host?.querySelector("#" + id);
  if (!el) return;
  if (!Number.isFinite(v)) { el.textContent = "—"; el.style.color = ""; return; }
  el.textContent = v.toFixed(2);
  el.style.color = v <= cfg.reGuardDist ? COLORS.pass : "";
}
