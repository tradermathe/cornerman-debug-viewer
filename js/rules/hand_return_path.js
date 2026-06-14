// Hand return path lens — "did the fist come straight back to guard?".
//
// rule_hand_ushape: after the punch lands, the fist should travel back to
// the guard along roughly the same line it went out on. The classic
// violation is the U-shape — the hand drops low and loops back up to the
// guard. Watch the retract, not the throw: the throw can be clean and the
// return still wrong.
//
// Per labelled straight (jab/cross, head + body):
//
//   peak  = most-extended frame in the punch window (max reach =
//           |shoulder→wrist| / torso, same pick as arm_extension) — the
//           start of the return.
//   window= [peak, cap], cap = min(peak + maxReturnSec, next same-hand
//           punch − 1, cache end). NOT re-guard — the verdict reads the
//           recovery off the trajectory, so it only needs a roomy window,
//           not a precise end. Re-guard is computed for the readout / loop
//           bound only.
//   drop  = the U-dip's PROMINENCE, against the SAME-SIDE SHOULDER so
//           whole-body vertical motion cancels. Per frame (smoothed),
//           offset = wrist_y − shoulder_y (bigger = fist lower). The low is
//           the biggest offset over the window; the recovery is the highest
//           the fist climbs back AFTER the low. A real U needs BOTH halves:
//             descent  = offset_low − offset_peak
//             recovery = offset_low − min(offset after low)
//             drop     = max(0, min(descent, recovery)) / torso
//           Equivalently drop = (offset_low − max(offset_peak, recovery_h))
//           / torso. A monotonic descent to a low guard never climbs back
//           → recovery ~0 → pass. A body shot's fist is low at peak and
//           only rises → descent ~0 → pass. A knee bend (even continuous)
//           moves wrist+shoulder together → offset flat → pass. Fail if
//           drop ≥ dropFail. Vertical-only: catches a dropped hand, not a
//           purely-sideways loop (the rarer fault). If a combo chops the
//           window before any recovery shows → unclear, not a guess.
//
// AXIALITY GATE — optional (togglable), joined by punch_uuid. Less load-
// bearing than for the old 2D-arc metric since a vertical drop reads even
// on a head-on punch; left in so head-on punches can be excluded if wanted.
//
// COVERAGE GATE — the low over a gappy track can miss the dip (glove
// occlusion → NaN wrist, no Vision fallback). Windows with less than
// minCoverage valid frames score "unclear" instead of guessing.
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
  dropFail:     0.20,   // fail if U-dip prominence ≥ this (torsos)
  reGuardDist:  0.60,   // wrist→nose euclidean ≤ this (torsos) = back at guard (cosmetic)
  maxReturnSec: 1.0,    // window cap after the peak frame
  smoothSec:    0.08,   // moving-average window on the offset signal
  minCoverage:  0.60,   // min fraction of valid wrist frames inside the return window
  axialityGate: true,
  axialityMax:  Math.SQRT1_2,   // ≈0.7071 = cos 45° — same cut as arm_extension
  minGloveConf: 0.20,
  minPoseConf:  0.20,
  // Straights only (hooks/uppercuts arc by design). Body shots are back
  // in: the drop metric measures the fist sinking BELOW its extension
  // height, and a body shot's fist is already low at peak and only rises
  // afterward, so it reads ~0 instead of false-failing the way the old
  // chord-sag did.
  appliesTo: new Set([
    "jab_head", "jab_body",
    "cross_head", "cross_body",
  ]),
};

const COLORS = {
  pass:        "#5fd97a",
  fail:        "#e85a5a",
  unclear:     "#f5b945",
  skip:        "#7ec8ff",
  outPath:     "rgba(255,255,255,0.35)",
  guardRing:   "rgba(126,200,255,0.45)",
  shoulderPk:  "rgba(255,255,255,0.30)",   // shoulder height at peak (frozen)
  shoulderNow: "rgba(255,255,255,0.70)",   // shoulder height this frame (live)
  baseLine:    "rgba(126,200,255,0.65)",   // extension height carried to the low frame
  wristNow:    "#7ec8ff",                  // live wrist dot
  lowMark:     "#ff8bd2",                  // detected-low marker
  agree:       "#5fd97a",
  disagree:    "#e85a5a",
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

    // dropFail only moves the verdict — rescore, no window recompute.
    wireSlider("hrp-drop", "hrp-drop-out", "dropFail", rescoreAndRefresh);
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
    if (active) drawReturnPath(ctx, active, s, f);
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

    // Live wrist-below-shoulder offset (torsos) for the active side.
    let offVal = NaN;
    if (ap) {
      const arm = signals.arms[ap.side];
      const t = signals.torso[f];
      if (Number.isFinite(arm.wy[f]) && Number.isFinite(arm.shy[f]) && t > 0) {
        offVal = (arm.wy[f] - arm.shy[f]) / t;
      }
    }
    const offEl = host?.querySelector("#hrp-off");
    if (offEl) offEl.textContent = Number.isFinite(offVal)
      ? (offVal >= 0 ? "+" : "") + offVal.toFixed(2) : "—";

    setText("hrp-active", ap
      ? `${ap.punch_type} · drop ${Number.isFinite(ap.drop) ? ap.drop.toFixed(2) + "t" : "—"} · ${ap.predicted}`
      : "—");

    drawSparkline(ap, state);
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
      buildPunch(d, stance, side, idx, { arms, torso, all, N, fps, startSec, cfg }));

  return { arms, torso, punches, fps };
}

function buildPunch(d, stance, side, idx, ctx) {
  const { arms, torso, all, N, fps, startSec, cfg } = ctx;
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
    cap: -1,
    low_frame: -1,
    base_offset: NaN,
    torso_med: NaN,
    re_guarded: false,
    has_return: false,
    chopped: false,
    return_sec: NaN,
    drop: NaN,
    coverage: NaN,
    closest_dist: NaN,
    peak_axiality,
    label,
  };
  if (!p.peak_valid) { rescorePunch(p, cfg); return p; }

  // Window cap (generous): maxReturnSec, the next same-hand punch (that hand
  // is busy again), end of cache. NOT the re-guard heuristic — the metric
  // reads recovery from the trajectory, so it only needs a roomy window, not
  // a precise end. capByPunch tracks whether a combo cut it short.
  const maxRetCap = Math.min(N - 1, peakFrame + Math.round(cfg.maxReturnSec * fps));
  let cap = maxRetCap, capByPunch = false;
  for (const { d: od, side: oside } of all) {
    if (oside !== side) continue;
    if (od.start_frame > peakFrame && od.start_frame - 1 < cap) {
      cap = od.start_frame - 1; capByPunch = true;
    }
  }
  p.cap = cap;
  if (cap <= peakFrame) { rescorePunch(p, cfg); return p; }   // no return window
  p.has_return = true;

  // Re-guard — COSMETIC ONLY now (readout + loop bound), never the verdict
  // anchor. First wrist-in-guard frame, else closest approach.
  let bFrame = -1, minDist = Infinity, minDistFrame = -1;
  for (let f = peakFrame + 1; f <= cap; f++) {
    const dist = arm.noseDist[f];
    if (!Number.isFinite(dist)) continue;
    if (dist < minDist) { minDist = dist; minDistFrame = f; }
    if (dist <= cfg.reGuardDist) { bFrame = f; break; }
  }
  p.re_guarded = bFrame >= 0;
  p.b_frame = bFrame >= 0 ? bFrame : (minDistFrame >= 0 ? minDistFrame : cap);
  p.return_sec = (p.b_frame - peakFrame) / fps;
  p.closest_dist = p.re_guarded ? arm.noseDist[p.b_frame] : minDist;

  // Smoothed shoulder-relative offset (wrist_y − shoulder_y; bigger = fist
  // lower). Smoothing stops one jittery glove frame faking a dip / recovery.
  const half = Math.max(0, Math.round((cfg.smoothSec * fps - 1) / 2));
  const offS = smoothOffset(arm, peakFrame, cap, half);
  const offPeak = offS[peakFrame];

  // Low = lowest fist (max offset) over the return; coverage tally.
  let lowFrame = peakFrame, offLow = Number.isFinite(offPeak) ? offPeak : -Infinity;
  let nValid = 0; const nTotal = cap - peakFrame;
  for (let f = peakFrame + 1; f <= cap; f++) {
    const o = offS[f];
    if (!Number.isFinite(o)) continue;
    nValid++;
    if (o > offLow) { offLow = o; lowFrame = f; }
  }
  p.coverage = nTotal > 0 ? nValid / nTotal : 1;
  p.low_frame = lowFrame;

  // Recovery = how far the fist climbs back AFTER the low (min offset after).
  let recoHigh = offLow;
  for (let f = lowFrame + 1; f <= cap; f++) {
    const o = offS[f];
    if (Number.isFinite(o) && o < recoHigh) recoHigh = o;
  }
  // Chopped: a combo cut the window before any recovery could be observed →
  // can't tell whether it would have climbed back.
  p.chopped = capByPunch && (cap - lowFrame) <= Math.max(1, half);

  // Prominence: it's a U only if the fist BOTH fell below extension AND
  // climbed back. Score the smaller of the two (= dip prominence). The
  // baseline is the higher (lower-fist) of {peak, recovery}, so a monotonic
  // descent to a low guard — or a body shot rising from a low peak — reads ~0.
  const baseOff = Math.max(Number.isFinite(offPeak) ? offPeak : -Infinity, recoHigh);
  p.base_offset = baseOff;
  const uPx = Math.max(0, offLow - baseOff);

  const tVals = [];
  for (let f = peakFrame; f <= cap; f++) {
    if (Number.isFinite(torso[f])) tVals.push(torso[f]);
  }
  const tMed = median(tVals);
  p.torso_med = tMed;
  p.drop = (Number.isFinite(tMed) && tMed > 0 && nValid > 0 && Number.isFinite(baseOff))
    ? uPx / tMed
    : NaN;

  rescorePunch(p, cfg);
  return p;
}

// NaN-aware moving average of the shoulder-relative offset (wrist_y −
// shoulder_y) over [lo, hi], half-window `half`. Frames missing wrist or
// shoulder are skipped; output NaN where no neighbour was valid.
function smoothOffset(arm, lo, hi, half) {
  const out = new Float32Array(arm.wy.length).fill(NaN);
  for (let f = lo; f <= hi; f++) {
    let s = 0, n = 0;
    const a = Math.max(lo, f - half), b = Math.min(hi, f + half);
    for (let k = a; k <= b; k++) {
      const wy = arm.wy[k], sy = arm.shy[k];
      if (Number.isFinite(wy) && Number.isFinite(sy)) { s += wy - sy; n++; }
    }
    if (n > 0) out[f] = s / n;
  }
  return out;
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
  if (p.chopped) {
    p.predicted = "unclear";
    p.reason = "chopped";
    p.reason_text = "the next same-hand punch cut the return before the fist recovered — can't tell if it would have climbed back";
    return;
  }
  if (!Number.isFinite(p.drop)) {
    p.predicted = "unclear";
    p.reason = "no_torso";
    p.reason_text = "no valid torso / shoulder in the return window — can't measure the drop";
    return;
  }
  const tail = p.re_guarded
    ? `re-guarded in ${p.return_sec.toFixed(2)}s`
    : `never re-guarded inside the cap (closest approach ${p.closest_dist.toFixed(2)}t)`;
  if (p.drop >= cfg.dropFail) {
    p.predicted = "fail";
    p.reason = "u_dip";
    p.reason_text = `fist dipped ${p.drop.toFixed(2)}t below where it sits and climbed back ≥ ${cfg.dropFail.toFixed(2)} — U-shaped return; ${tail}`;
  } else {
    p.predicted = "pass";
    p.reason = "no_dip";
    p.reason_text = `U-dip ${p.drop.toFixed(2)}t < ${cfg.dropFail.toFixed(2)} — no drop-and-recover; ${tail}`;
  }
}

// Per-frame wrist track + reach + wrist→nose distance for one side.
function perFrameArm(pose, side, cfg, torso) {
  const N = pose.n_frames;
  const joints = JOINTS_FOR_SIDE[side];
  const wx = new Float32Array(N).fill(NaN);
  const wy = new Float32Array(N).fill(NaN);
  const shx = new Float32Array(N).fill(NaN);   // same-side shoulder, conf-gated
  const shy = new Float32Array(N).fill(NaN);
  const reach = new Float32Array(N).fill(NaN);
  const noseDist = new Float32Array(N).fill(NaN);
  const source = new Array(N).fill(null);
  for (let f = 0; f < N; f++) {
    // Same-side shoulder (drop reference) — gated on shoulder conf alone so
    // the overlay lines survive even when torso is briefly missing.
    const sc = pose.conf[f * 17 + joints.shoulder];
    let sx = NaN, sy = NaN;
    if (sc >= cfg.minPoseConf) {
      sx = pose.skeleton[(f * 17 + joints.shoulder) * 2];
      sy = pose.skeleton[(f * 17 + joints.shoulder) * 2 + 1];
      if (Number.isFinite(sx)) { shx[f] = sx; shy[f] = sy; }
    }
    const w = wristXY(pose, f, joints, cfg);
    if (!w) continue;
    wx[f] = w.x; wy[f] = w.y; source[f] = w.source;
    const t = torso[f];
    const tOk = Number.isFinite(t) && t > 0;
    if (Number.isFinite(shx[f]) && tOk) {
      reach[f] = Math.hypot(w.x - shx[f], w.y - shy[f]) / t;
    }
    const nc = pose.conf[f * 17 + J.NOSE];
    if (nc >= cfg.minPoseConf && tOk) {
      const nx = pose.skeleton[(f * 17 + J.NOSE) * 2];
      const ny = pose.skeleton[(f * 17 + J.NOSE) * 2 + 1];
      if (Number.isFinite(nx)) noseDist[f] = Math.hypot(w.x - nx, w.y - ny) / t;
    }
  }
  return { wx, wy, shx, shy, reach, noseDist, source };
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
    && frame <= Math.max(p.end_frame, p.cap >= 0 ? p.cap : p.end_frame)
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
  // Loop the whole metric window (through cap), so the low — which can fall
  // after re-guard — is on screen.
  const end = Math.max(p.end_frame, p.cap >= 0 ? p.cap : p.end_frame);
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
  drawSparkline(punches[idx], latestState);
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

// Four-line overlay (all in pose-pixel space, which the canvas is in):
//   1. shoulder at peak  — frozen, faint    (where the body was)
//   2. shoulder now      — live              (gap 1↔2 = body sank, ignored)
//   3. detected low      — frozen, verdict   (lowest the fist got)
//   + a caliper, frozen at the low frame, from the prominence baseline
//     (shoulder-at-low + max(peak, recovery) offset) down to the low — that
//     gap IS the U-dip. + the live wrist dot.
function drawReturnPath(ctx, p, scale, frame) {
  if (!p.peak_valid || !p.has_return) return;
  const arm = signals.arms[p.side];
  const col = COLORS[p.predicted] || COLORS.unclear;
  const W = ctx.canvas.width;
  const peak = p.peak_frame, low = p.low_frame, end = p.cap >= 0 ? p.cap : p.b_frame;

  // Faint context: the actual wrist trail through the return window.
  drawTrail(ctx, arm, peak, end, COLORS.outPath, 1.5, [3, 3], scale);

  // 1. shoulder at peak (frozen) · 2. shoulder now (live).
  hline(ctx, arm.shy[peak], W, COLORS.shoulderPk, 1 * scale, [6 * scale, 5 * scale]);
  hline(ctx, arm.shy[frame], W, COLORS.shoulderNow, 1.2 * scale, null);

  // 3. + caliper, frozen at the low frame.
  if (low >= 0 && Number.isFinite(arm.wy[low]) && Number.isFinite(arm.shy[low]) && Number.isFinite(p.base_offset)) {
    const wxLow = arm.wx[low], wyLow = arm.wy[low];
    const baseY = arm.shy[low] + p.base_offset;    // prominence baseline at the low frame's shoulder

    hline(ctx, wyLow, W, col, 1.2 * scale, [6 * scale, 5 * scale]);

    ctx.save();
    // baseline tick around the low point
    ctx.strokeStyle = COLORS.baseLine;
    ctx.lineWidth = 1.5 * scale;
    ctx.setLineDash([4 * scale, 3 * scale]);
    ctx.beginPath();
    ctx.moveTo(wxLow - 42 * scale, baseY); ctx.lineTo(wxLow + 42 * scale, baseY);
    ctx.stroke();
    ctx.setLineDash([]);

    // caliper from baseline down to the low
    const cx = wxLow - 28 * scale;
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.5 * scale;
    ctx.beginPath();
    ctx.moveTo(cx, baseY); ctx.lineTo(cx, wyLow);
    ctx.moveTo(cx - 4 * scale, baseY); ctx.lineTo(cx + 4 * scale, baseY);
    ctx.moveTo(cx - 4 * scale, wyLow); ctx.lineTo(cx + 4 * scale, wyLow);
    ctx.stroke();

    // low marker ✕
    ctx.strokeStyle = COLORS.lowMark;
    ctx.lineWidth = 2 * scale;
    const r = 6 * scale;
    ctx.beginPath();
    ctx.moveTo(wxLow - r, wyLow - r); ctx.lineTo(wxLow + r, wyLow + r);
    ctx.moveTo(wxLow - r, wyLow + r); ctx.lineTo(wxLow + r, wyLow - r);
    ctx.stroke();

    // readout
    const txt = `drop ${Number.isFinite(p.drop) ? p.drop.toFixed(2) : "—"}t · ${p.predicted}${p.label ? ` (GT ${p.label})` : ""}`;
    ctx.font = `${12 * scale}px ui-monospace, monospace`;
    const tw = ctx.measureText(txt).width;
    const ly = (baseY + wyLow) / 2;
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(cx - tw - 14 * scale, ly - 9 * scale, tw + 8 * scale, 18 * scale);
    ctx.fillStyle = col;
    ctx.fillText(txt, cx - tw - 10 * scale, ly + 4 * scale);
    ctx.restore();
  }

  // Live wrist dot.
  if (Number.isFinite(arm.wx[frame])) {
    ctx.save();
    ctx.fillStyle = COLORS.wristNow;
    ctx.beginPath(); ctx.arc(arm.wx[frame], arm.wy[frame], 5 * scale, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }
}

// Full-width horizontal reference line at pose-y.
function hline(ctx, y, W, color, width, dash) {
  if (!Number.isFinite(y)) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash || []);
  ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
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

// Sidebar sparkline of the offset signal (wrist below shoulder, torsos) over
// the active punch's return window — so spikes (tracking glitches) and real
// dip-and-recover shapes are both visible at a glance.
function drawSparkline(p, state) {
  const cv = host?.querySelector("#hrp-spark");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#111"; ctx.fillRect(0, 0, W, H);

  if (!p || !p.peak_valid || !p.has_return || p.cap < 0 || !(p.torso_med > 0)) {
    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText("no active punch", 10, H / 2);
    return;
  }
  const arm = signals.arms[p.side];
  const peak = p.peak_frame, cap = p.cap, tm = p.torso_med;
  const half = Math.max(0, Math.round((cfg.smoothSec * (signals.fps || 30) - 1) / 2));
  const offS = smoothOffset(arm, peak, cap, half);

  const baseV = p.base_offset / tm;
  const thrV  = baseV + cfg.dropFail;
  let vmin = Math.min(0, baseV), vmax = Math.max(thrV, 0);
  for (let f = peak; f <= cap; f++) {
    const o = offS[f]; if (!Number.isFinite(o)) continue;
    const v = o / tm; if (v < vmin) vmin = v; if (v > vmax) vmax = v;
  }
  vmin -= 0.05; vmax += 0.05;
  const padX = 6, padT = 6, padB = 6;
  const X = f => padX + (cap > peak ? (f - peak) / (cap - peak) : 0) * (W - 2 * padX);
  const Y = v => padT + (vmax > vmin ? (v - vmin) / (vmax - vmin) : 0.5) * (H - padT - padB);
  const col = COLORS[p.predicted] || COLORS.unclear;

  const hl = (v, color, dash) => {
    ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash(dash || []);
    ctx.beginPath(); ctx.moveTo(padX, Y(v)); ctx.lineTo(W - padX, Y(v)); ctx.stroke();
    ctx.setLineDash([]);
  };
  hl(0, "rgba(255,255,255,0.18)");                  // shoulder height
  hl(baseV, COLORS.baseLine, [4, 3]);               // prominence baseline
  hl(thrV, COLORS.fail, [2, 3]);                    // fail threshold

  // current-frame cursor
  if (state && Number.isFinite(state.frame) && state.frame >= peak && state.frame <= cap) {
    ctx.strokeStyle = "rgba(255,255,255,0.5)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(state.frame), padT); ctx.lineTo(X(state.frame), H - padB); ctx.stroke();
  }

  // offset curve
  ctx.strokeStyle = col; ctx.lineWidth = 1.6; ctx.beginPath();
  let pen = false;
  for (let f = peak; f <= cap; f++) {
    const o = offS[f];
    if (!Number.isFinite(o)) { pen = false; continue; }
    const px = X(f), py = Y(o / tm);
    if (!pen) { ctx.moveTo(px, py); pen = true; } else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // detected-low ✕
  if (p.low_frame >= 0 && Number.isFinite(offS[p.low_frame])) {
    const px = X(p.low_frame), py = Y(offS[p.low_frame] / tm), r = 4;
    ctx.strokeStyle = COLORS.lowMark; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(px - r, py - r); ctx.lineTo(px + r, py + r);
    ctx.moveTo(px - r, py + r); ctx.lineTo(px + r, py - r);
    ctx.stroke();
  }
}

// ─── render ────────────────────────────────────────────────────────────────

function renderTemplate(sig, cfg) {
  const hasLabels = sig.punches.some(p => p.label !== null);
  return `
    <h2>Hand return path (straights)</h2>
    <p class="hint">
      <b>drop</b> = the U-dip's prominence vs the same-side shoulder, in
      torsos: the fist must BOTH fall below its extension height AND climb
      back. We score the smaller half, so a clean retrace to a low guard
      (falls, never climbs) and a body shot (low at peak, only rises) both
      read ~0 — and because it's wrist−shoulder every frame, bending the
      knees (even continuously after impact) cancels too.
    </p>

    <h3>Legend</h3>
    <ul class="hint" style="list-style:none;padding-left:0;margin:0 0 12px 0;line-height:1.7">
      <li><span style="display:inline-block;width:24px;height:1px;border-top:2px dashed ${COLORS.shoulderPk};vertical-align:middle"></span>
        &nbsp;shoulder height at peak (frozen)</li>
      <li><span style="display:inline-block;width:24px;height:2px;background:${COLORS.shoulderNow};vertical-align:middle"></span>
        &nbsp;shoulder height now — gap from the line above = body sank (ignored)</li>
      <li><span style="display:inline-block;width:24px;height:1px;border-top:2px dashed ${COLORS.baseLine};vertical-align:middle"></span>
        &nbsp;prominence baseline — higher of extension / recovery</li>
      <li><span style="color:${COLORS.lowMark};font-weight:700">✕</span>
        &nbsp;detected low + verdict-colored caliper = the U-dip</li>
      <li><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${COLORS.wristNow};vertical-align:middle"></span>
        &nbsp;live wrist &nbsp;·&nbsp;
        <span style="display:inline-block;width:14px;height:14px;border:2px dashed ${COLORS.guardRing};border-radius:50%;vertical-align:middle"></span>
        &nbsp;guard radius (re-guard target)</li>
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
        <div class="metric-label">wrist ↓ shoulder</div>
        <div class="metric-val" id="hrp-off">—</div>
      </div>
      <div class="metric">
        <div class="metric-label">active punch</div>
        <div class="metric-val" id="hrp-active" style="font-size:14px">—</div>
      </div>
    </div>

    <h3>Offset trace · active punch</h3>
    <canvas id="hrp-spark" width="520" height="120"
      style="width:100%;height:120px;background:#111;border:1px solid #2a2a2a;border-radius:4px;display:block"></canvas>
    <p class="hint muted small" style="margin:4px 0 0 0">
      wrist below shoulder (torsos) across the return. Flat ≈ hand stayed up;
      a spike that returns = the dip. <span style="color:${COLORS.baseLine}">blue</span> = prominence
      baseline, <span style="color:${COLORS.fail}">red</span> = fail threshold,
      <span style="color:${COLORS.lowMark}">✕</span> = detected low. A single sharp
      spike is usually a wrist-tracking glitch, not a real drop.
    </p>

    <h3>Drop fail threshold</h3>
    <div class="slider-row">
      <input type="range" id="hrp-drop" min="0.05" max="0.50" step="0.01" value="${cfg.dropFail}" />
      <output id="hrp-drop-out">${cfg.dropFail.toFixed(2)}</output>
      <span class="muted small">drop below extension height (torsos) — at or above = dropped hand</span>
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
        const dropCell = Number.isFinite(p.drop)
          ? `<td style="color:${p.drop >= cfg.dropFail ? COLORS.fail : COLORS.pass};font-variant-numeric:tabular-nums">${p.drop.toFixed(2)}</td>`
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
          ${dropCell}
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
        <th title="how far the fist dropped below its extension height, vs the same-side shoulder (torsos)">drop</th>
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
  const meanDrop = scored.reduce((s, p) => s + p.drop, 0) / scored.length;
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
      <div class="metric-sub">${Math.round(100 * passed / scored.length)}% · drop &lt; ${cfg.dropFail.toFixed(2)}</div>
    </div>
    <div class="metric">
      <div class="metric-label">skipped (axial)</div>
      <div class="metric-val">${skipped}</div>
      <div class="metric-sub">head-on, gated out</div>
    </div>
    <div class="metric">
      <div class="metric-label">mean drop</div>
      <div class="metric-val">${meanDrop.toFixed(3)}</div>
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
