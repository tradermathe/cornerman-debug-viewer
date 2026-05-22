// Hip rotation review — walk through every labelled rotation-applicable punch
// (cross/hook/uppercut, head/body) and read off how much the hips swung
// during the punch window. Mirrors straights_review.js: one punch at a time,
// video loops within the search window, N/P keys (or buttons) step to next/
// prev, M mutes.
//
// Signal:
//   gap[f] = (L_hip.x − R_hip.x) / torso_height                  (signed, smoothed)
//          - sign flips when the hip line crosses the camera axis, which
//            lets us measure crossover rotations at full magnitude
//          - frames with either hip confidence < cfg.minHipConf are NaN'd
//            so brief occlusions / L-R swaps don't poison the signal
//   W_est  = quantile(|gap|, 0.99) across the video               (running max across rounds)
//
// Per-punch compute:
//   search window = [start − searchPreSec, end + searchPostSec]
//                   asymmetric: load-up happens before the labelled start;
//                   the labelled end already runs past peak rotation.
//   peak / trough     = max / min of signed gap inside the window
//   peak_θ / trough_θ = arcsin(peak / W_est), arcsin(trough / W_est)    ∈ [−90°, 90°]
//   rotation_deg      = peak_θ − trough_θ                              ∈ [0°, 180°]
//
// Gate: skip if min(|gap|)/W_est > noiseRatioMin across the window —
// catches "hips parked saturated near ±W the whole punch" (arcsin noise
// floor too high). Crossover rotations pass through zero, so they're
// never flagged as noise.
//
// Per punch:
//   score = clamp(rotation_deg / scoreTargetDeg × 100, 0, 100)
//   tier  = pass (≥ solidRotationDeg) / warn / fail (<minRotationDeg)
// The tier is a debug-friendly label; the score is the load-bearing
// number that flows to the app. Session score = mean(score) over
// applicable + evaluable (non-skipped) punches. Session severity:
//   sessionScore ≥ scoreHideAbove → don't show the FormIssue card
//   ≥ scoreLowAbove               → low
//   ≥ scoreMedAbove               → medium
//   else                          → high

import { J, torsoHeight, drawSkeleton } from "../skeleton.js";

// W_est is per-VIDEO, not per-round. As the user visits more rounds of the
// same video we take the running max so the estimate gets more accurate (the
// max best approximates the boxer's true hip width at broadside). Module-
// level so it survives lens remounts and round-switches.
const wEstByVideo = new Map();   // videoStem → number

const DEFAULTS = {
  gapSmoothSeconds:       0.083,
  // Asymmetric search padding. Load-up often happens BEFORE the labelled
  // punch start, so we extend the search window back by searchPreSec to
  // catch it. The labelled end already runs until the hand returns, which
  // is past the furthest rotation point — so we don't pad after.
  searchPreSec:           0.3,
  searchPostSec:          0.0,

  // Gate: min(|signed gap|) / W_est > noiseRatioMin → skip. Catches
  // "hips parked broadside the whole punch" (gap signal saturated near
  // ±W; can't tell rotation from noise).
  noiseRatioMin:          0.95,
  wEstQuantile:           0.99,
  // Per-punch debug tier labels (the load-bearing number is the score
  // below — these are just coarse spot-check labels).
  //   rotation < minRotationDeg                     → fail
  //   minRotationDeg ≤ rotation < solidRotationDeg  → warn
  //   rotation ≥ solidRotationDeg                   → pass
  minRotationDeg:         15,
  solidRotationDeg:       40,

  // Per-punch score (0–100). Linear in rotation_deg, capped at 100 once
  // the boxer hits the target. Anchored at 40° because everything below
  // 40° is considered imperfect; 25–40° is "not a massive issue but
  // also not perfect", and <25° starts feeling like a real penalty.
  //   rotation 40°+ → 100  (perfect)
  //   rotation 30°  →  75
  //   rotation 25°  →  62
  //   rotation 15°  →  37
  //   rotation 0°   →   0  (no rotation at all)
  //
  // Session score = mean(per-punch score) across applicable + evaluable
  // (non-skipped) punches. The session-level severity for the app's
  // FormIssue card is derived from the session score:
  //   ≥ scoreHideAbove        → don't show the card (boxer is fine)
  //   scoreLowAbove–HideAbove → low
  //   scoreMedAbove–LowAbove  → medium
  //   < scoreMedAbove         → high
  // Cutoffs are policy and will be re-tuned once real session reports
  // ship; current values are sensible defaults.
  scoreTargetDeg:         40,
  scoreHideAbove:         90,
  scoreLowAbove:          80,
  scoreMedAbove:          65,
  // Minimum hip-keypoint confidence to include a frame in the gap
  // signal. Below this on either hip, the frame becomes NaN and the
  // moving average skips it. Defends against pose-detector L/R swaps
  // (which become "high-noise frames" the model can't confidently
  // commit to) corrupting the signed gap signal.
  minHipConf:             0.30,
  // If fewer than this fraction of the search window has valid gap
  // data (after the hip-confidence gate above), skip the punch — the
  // surviving frames probably don't sample the full motion and the
  // recovered rotation will be misleading. Empirically:
  //   <30% validity → 86% are fails with median ~0° rotation (spurious)
  //   <50% validity → 50%+ fails, mostly spurious
  //   <70% validity → 2× more fail-skips than pass-skips at this gate
  //                   (still mostly catching bad-data fails, costs ~5%)
  //   <80% validity → ~1.4× ratio, gate starts losing legit verdicts
  // 0.70 sits at the knee where each new skip is still mostly catching
  // bad data, not erasing real ones.
  minValidFrac:           0.7,

  // Rule only applies to punches with rotation expectation (jab + body
  // shots excluded — matches hip_rotation.js).
  appliesTo: new Set([
    "cross_head", "cross_body",
    "lead_hook_head", "lead_uppercut_head",
    "rear_uppercut_head", "rear_hook_head",
  ]),
};

const COLOR_HIP        = "#a78bfa";  // purple — current-frame hip line
const COLOR_HIP_PEAK   = "#ffd24a";  // amber  — hip line at peak (widest) frame
const COLOR_HIP_TROUGH = "#ff7e3a";  // orange — hip line at trough (narrowest) frame
const COLOR_PASS       = "#5fd97a";
const COLOR_WARN       = "#f5b945";  // amber — between fail and pass
const COLOR_FAIL       = "#e85a5a";
const COLOR_SKIP       = "#7ec8ff";
const COLOR_UNCLEAR    = "#f5b945";

let host = null;
let videoEl = null;
let timeupdateHandler = null;
let keydownHandler = null;
let loopWindow = null;
let activeIdx = -1;
let punches = [];
let signals = null;           // {gap, punches, fps, wEst, sessionScore, sessionSeverity, sessionN, ...}
let lastDetectionsRef = null;
let lastStemForReset = null;
let lastPose = null;
let latestState = null;
let cfg = { ...DEFAULTS };

// ─── helpers ──────────────────────────────────────────────────────────────

// Format a signed number with explicit sign for display. Helps readers
// see direction at a glance now that values can be negative.
function signed3(v) {
  if (!Number.isFinite(v)) return "—";
  return (v >= 0 ? "+" : "") + v.toFixed(3);
}
function signedDeg1(v) {
  if (!Number.isFinite(v)) return "—";
  return (v >= 0 ? "+" : "") + v.toFixed(1) + "°";
}
function signedDeg0(v) {
  if (!Number.isFinite(v)) return "—";
  return (v >= 0 ? "+" : "") + v.toFixed(0) + "°";
}

function colorFor(predicted) {
  if (predicted === "pass") return COLOR_PASS;
  if (predicted === "warn") return COLOR_WARN;
  if (predicted === "fail") return COLOR_FAIL;
  if (predicted === "skip") return COLOR_SKIP;
  return COLOR_UNCLEAR;
}

// Per-punch score (0–100), linear in rotation_deg, capped at the target.
// Returns NaN if rotation_deg isn't finite (skip/no data).
function rotationScore(rotationDeg, targetDeg) {
  if (!Number.isFinite(rotationDeg)) return NaN;
  return Math.max(0, Math.min(100, (rotationDeg / targetDeg) * 100));
}

// Session score → severity (or null for "don't show the card").
function severityFor(sessionScore, cfg) {
  if (!Number.isFinite(sessionScore)) return null;
  if (sessionScore >= cfg.scoreHideAbove) return null;
  if (sessionScore >= cfg.scoreLowAbove)  return "low";
  if (sessionScore >= cfg.scoreMedAbove)  return "medium";
  return "high";
}

// Map session severity → display color. Keeps the per-punch palette but
// remaps to fit the high/medium/low naming.
function severityColor(severity) {
  if (severity === "high")   return COLOR_FAIL;
  if (severity === "medium") return COLOR_WARN;
  if (severity === "low")    return COLOR_PASS;
  return COLOR_SKIP;
}

// NaN-aware moving average. Frames with NaN (e.g., low-confidence pose
// gated out) are skipped; the average uses only the valid neighbours in
// the window. If no valid frames exist in the window, output is NaN.
function movingAvg(arr, w) {
  if (w <= 1) return arr;
  const n = arr.length;
  const out = new Float32Array(n);
  const half = Math.floor(w / 2);
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - half), hi = Math.min(n - 1, i + half);
    let s = 0, count = 0;
    for (let k = lo; k <= hi; k++) {
      if (Number.isFinite(arr[k])) { s += arr[k]; count++; }
    }
    out[i] = count > 0 ? s / count : NaN;
  }
  return out;
}


// ─── compute ──────────────────────────────────────────────────────────────

function computeAll(state, cfg) {
  const pose = state.pose;
  const fps = pose.fps || 30;
  const N = pose.n_frames;

  // Signed gap = (L_hip.x − R_hip.x) / torso_height, smoothed. Sign carries
  // direction: positive means L is to the right of R in image, negative is
  // swapped. The sign flips when the hip line crosses being aligned with the
  // camera axis — that's how we detect crossover rotations (load on one
  // side → drive past perpendicular → end on the other) that unsigned gap
  // would undercount by reflecting the swing back on itself.
  //
  // Frames where either hip's confidence is below cfg.minHipConf become
  // NaN — the moving average then skips them. This is the defense against
  // pose-detector L/R swaps, which tend to coincide with low-confidence
  // moments (occlusion, fast motion blur). A confident swap (rare) still
  // gets through, but the moving average dampens isolated swaps.
  const gapRaw = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const cL = pose.conf[i * 17 + J.L_HIP];
    const cR = pose.conf[i * 17 + J.R_HIP];
    if (cL < cfg.minHipConf || cR < cfg.minHipConf) {
      gapRaw[i] = NaN;
      continue;
    }
    const lx = pose.skeleton[(i * 17 + J.L_HIP) * 2];
    const rx = pose.skeleton[(i * 17 + J.R_HIP) * 2];
    const th = Math.max(1e-6, torsoHeight(pose, i));
    gapRaw[i] = (lx - rx) / th;
  }
  const smoothFrames = Math.max(1, Math.round(cfg.gapSmoothSeconds * fps));
  const gap = movingAvg(gapRaw, smoothFrames);

  // W_est: per-VIDEO estimate of this boxer's true hip-width / torso-height
  // ratio. The signed gap sweeps between −W and +W (over a full rotation),
  // so the max of |signed gap| approaches W. Per-round we take a high
  // quantile of |signed gap| (default 99th %ile). Across rounds of the same
  // video we take the running max, so the estimate gets stronger as the
  // user visits more rounds.
  const absGapForW = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    absGapForW[i] = Number.isFinite(gap[i]) ? Math.abs(gap[i]) : NaN;
  }
  const localWEst = quantile(absGapForW, cfg.wEstQuantile);
  const stem = state.cacheBasename || null;
  let wEst = localWEst;
  if (stem && Number.isFinite(localWEst)) {
    const cached = wEstByVideo.get(stem);
    wEst = (cached != null && cached > localWEst) ? cached : localWEst;
    wEstByVideo.set(stem, wEst);
  }
  const wEstFromCache = (stem && wEstByVideo.get(stem) !== localWEst && wEst > localWEst);

  // Frame in THIS round whose |gap| is closest to W_est — used by the
  // sidebar snapshot so the user can sanity-check that it really IS a
  // broadside-hip frame.
  let wEstFrame = 0, wEstBestDiff = Infinity;
  for (let f = 0; f < gap.length; f++) {
    if (!Number.isFinite(gap[f])) continue;
    const diff = Math.abs(Math.abs(gap[f]) - wEst);
    if (diff < wEstBestDiff) { wEstBestDiff = diff; wEstFrame = f; }
  }
  const wEstFrameGap = gap[wEstFrame];

  const detections = (state.labels?.detections || [])
    .filter(d => cfg.appliesTo.has(d.punch_type));

  const preFrames = Math.max(0, Math.round(cfg.searchPreSec * fps));
  const postFrames = Math.max(0, Math.round(cfg.searchPostSec * fps));

  const out = detections.map(d => {
    const sf = Math.max(0, d.start_frame);
    const ef = Math.min(N - 1, d.end_frame);
    const ss = Math.max(0, sf - preFrames);
    const se = Math.min(N - 1, ef + postFrames);

    // Scan the search window for peak (max signed gap), trough (min
    // signed gap), and the smallest |signed gap| seen — that last one
    // feeds the noise-zone gate. NaN frames (low pose confidence) are
    // skipped so a brief occlusion doesn't poison the extremes. Also
    // tally validity for the low-validity gate below.
    let peak = -Infinity, peakAt = ss;
    let trough = Infinity, troughAt = ss;
    let minAbs = Infinity;
    let nValid = 0;
    const nWindow = se - ss + 1;
    for (let f = ss; f <= se; f++) {
      const g = gap[f];
      if (!Number.isFinite(g)) continue;
      nValid++;
      if (g > peak)   { peak = g;   peakAt = f; }
      if (g < trough) { trough = g; troughAt = f; }
      const ag = Math.abs(g);
      if (ag < minAbs) minAbs = ag;
    }
    if (nValid === 0) { peak = NaN; trough = NaN; minAbs = NaN; }
    const validFrac = nWindow > 0 ? nValid / nWindow : 0;

    const stance = d.stance?.toLowerCase?.() || null;

    // Recover hip-line angles. Signed gap / W_est lives in [−1, 1] so
    // arcsin returns angles in [−90°, 90°]. peak − trough captures the
    // full swing INCLUDING crossover rotations (where peak and trough
    // sit on opposite signs).
    const peakRatio = wEst > 0 && Number.isFinite(peak)
      ? Math.max(-1, Math.min(1, peak / wEst)) : NaN;
    const troughRatio = wEst > 0 && Number.isFinite(trough)
      ? Math.max(-1, Math.min(1, trough / wEst)) : NaN;
    const minAbsRatio = wEst > 0 && Number.isFinite(minAbs)
      ? Math.min(1, minAbs / wEst) : NaN;
    const peakTheta = Number.isFinite(peakRatio) ? Math.asin(peakRatio) * 180 / Math.PI : NaN;
    const troughTheta = Number.isFinite(troughRatio) ? Math.asin(troughRatio) * 180 / Math.PI : NaN;
    const rotationDeg = Number.isFinite(peakTheta) && Number.isFinite(troughTheta)
      ? peakTheta - troughTheta : NaN;
    const score = rotationScore(rotationDeg, cfg.scoreTargetDeg);

    // Gates: (1) low-validity → too few frames had valid pose data,
    // the surviving frames probably miss part of the motion (e.g. the
    // drive phase blurs the hips, confidence drops, and we end up
    // sampling only the loaded phase). (2) noise-zone — |signed gap|
    // stayed saturated near ±W throughout, arcsin too noisy.
    // can't trust the recovered angle. Crossover rotations DON'T trigger
    // this: signed gap passes through zero on its way from +W to −W (or
    // back), so min(|gap|) ≈ 0, well below the threshold.
    let predicted;
    if (!Number.isFinite(minAbsRatio) || !Number.isFinite(rotationDeg)) {
      predicted = "skip";              // W_est unavailable or no valid frames
    } else if (validFrac < cfg.minValidFrac) {
      predicted = "skip";              // too sparse — likely missing the drive
    } else if (minAbsRatio > cfg.noiseRatioMin) {
      predicted = "skip";              // sin-flat zone (one-sided saturation)
    } else if (rotationDeg >= cfg.solidRotationDeg) {
      predicted = "pass";              // solid rotation
    } else if (rotationDeg >= cfg.minRotationDeg) {
      predicted = "warn";              // not enough rotation
    } else {
      predicted = "fail";              // real problem
    }

    const label = d.rule_hip_rotation === "pass" || d.rule_hip_rotation === "fail"
      ? d.rule_hip_rotation : null;

    return {
      timestamp: d.timestamp,
      hand: d.hand,
      stance,
      punch_type: d.punch_type,
      start_frame: sf,
      end_frame: ef,
      search_start: ss,
      search_end: se,
      land_frame: peakAt,
      peak_gap: peak,
      peak_frame: peakAt,
      trough_gap: trough,
      trough_frame: troughAt,
      n_valid: nValid,
      n_window: nWindow,
      valid_frac: validFrac,
      peak_ratio: peakRatio,
      trough_ratio: troughRatio,
      min_abs_ratio: minAbsRatio,
      peak_theta: peakTheta,
      trough_theta: troughTheta,
      rotation_deg: rotationDeg,
      score,
      predicted,
      label,
    };
  });

  // Session-level aggregate: mean per-punch score across applicable +
  // evaluable (non-skipped) punches. Drives the FormIssue severity.
  let sumScore = 0, nEval = 0;
  for (const p of out) {
    if (p.predicted === "skip") continue;
    if (!Number.isFinite(p.score)) continue;
    sumScore += p.score;
    nEval++;
  }
  const sessionScore = nEval > 0 ? sumScore / nEval : NaN;
  const sessionSeverity = severityFor(sessionScore, cfg);

  return { gap, punches: out, fps, wEst, wEstFrame, wEstFrameGap, wEstFromCache,
           sessionScore, sessionSeverity, sessionN: nEval };
}

// 99th percentile (or whichever quantile) of a Float32Array, ignoring
// non-finite values. Used to estimate this boxer's true hip width.
function quantile(arr, q) {
  const sorted = [];
  for (let i = 0; i < arr.length; i++) {
    if (Number.isFinite(arr[i])) sorted.push(arr[i]);
  }
  if (!sorted.length) return NaN;
  sorted.sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(q * (sorted.length - 1))));
  return sorted[idx];
}

// ─── data plumbing ────────────────────────────────────────────────────────

function rebuildPunches(state) {
  const stem = state.cacheBasename || "";
  const dets = state.labels?.detections;
  const stemChanged = stem !== lastStemForReset;
  const detsChanged = dets !== lastDetectionsRef;
  const poseChanged = state.pose !== lastPose;
  if (!stemChanged && !detsChanged && !poseChanged && punches.length) {
    rebuildSidebar(state);
    return;
  }
  lastStemForReset = stem;
  lastDetectionsRef = dets;
  lastPose = state.pose;

  signals = computeAll(state, cfg);
  const next = signals.punches.sort((a, b) => a.start_frame - b.start_frame);

  const hadNone = punches.length === 0;
  punches = next;
  if (stemChanged || (hadNone && next.length)) {
    if (next.length) seekToPunch(0, state);
    else { loopWindow = null; activeIdx = -1; }
  } else if (activeIdx >= next.length) {
    activeIdx = next.length - 1;
  }
  renderPunchTable();  // punches changed → rebuild the table once
  rebuildSidebar(state);
}

function seekToPunch(idx, state) {
  if (!punches.length) return;
  if (idx < 0) idx = 0;
  if (idx >= punches.length) idx = punches.length - 1;
  const p = punches[idx];
  activeIdx = idx;
  // Loop the FULL search window (label window + asymmetric padding:
  // searchPreSec before, searchPostSec after) so the user actually sees
  // the frames the verdict was computed over — including any wind-up
  // that sits before the labelled start.
  loopWindow = { start_frame: p.search_start, end_frame: p.search_end };
  if (videoEl && state.fps) {
    videoEl.currentTime = (state.start_sec || 0) + p.search_start / state.fps;
    if (videoEl.paused) {
      const promise = videoEl.play();
      if (promise && typeof promise.catch === "function") promise.catch(() => {});
    }
  }
  rebuildSidebar(state);
}

function installTimeupdateLoop(state) {
  if (!videoEl) return;
  if (timeupdateHandler) videoEl.removeEventListener("timeupdate", timeupdateHandler);
  timeupdateHandler = () => {
    if (state.rule?.id !== "hip_rotation_review") return;
    if (!loopWindow || !state.fps) return;
    const endTime = (state.start_sec || 0) + (loopWindow.end_frame + 0.5) / state.fps;
    if (videoEl.currentTime > endTime) {
      const startTime = (state.start_sec || 0) + loopWindow.start_frame / state.fps;
      videoEl.currentTime = startTime;
    }
  };
  videoEl.addEventListener("timeupdate", timeupdateHandler);
}

function installKeyHandlers(state) {
  // Use the CAPTURE phase so we run before the viewer's bubble-phase
  // keydown listener. This lets us block out-of-loop arrow/bracket steps
  // before seekToFrame ever fires.
  if (keydownHandler) document.removeEventListener("keydown", keydownHandler, true);
  keydownHandler = (e) => {
    if (state.rule?.id !== "hip_rotation_review") return;
    const tag = e.target?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA") return;

    if (e.key === "n" || e.key === "N") { e.preventDefault(); seekToPunch(activeIdx + 1, state); return; }
    if (e.key === "p" || e.key === "P") { e.preventDefault(); seekToPunch(activeIdx - 1, state); return; }
    if (e.key === "m" || e.key === "M") { e.preventDefault(); toggleMute(); return; }

    // Frame-stepping clamp: when the cursor sits inside the active loop
    // window, arrows / brackets can't push it past the loop boundaries.
    // Outside the loop (user scrubbed away manually), keys behave normally.
    if (!loopWindow) return;
    let delta = 0;
    if      (e.key === "ArrowLeft")  delta = -1;
    else if (e.key === "ArrowRight") delta = +1;
    else if (e.key === "[")          delta = -10;
    else if (e.key === "]")          delta = +10;
    else return;

    const f = state.frame;
    if (f < loopWindow.start_frame || f > loopWindow.end_frame) return;
    const target = f + delta;
    if (target < loopWindow.start_frame || target > loopWindow.end_frame) {
      // Block the viewer's handler so the cursor doesn't escape the loop.
      e.preventDefault();
      e.stopImmediatePropagation();
    }
  };
  document.addEventListener("keydown", keydownHandler, true);
}

// ─── sidebar ──────────────────────────────────────────────────────────────

function buildSidebarSkeleton() {
  if (!host) return;
  host.innerHTML = `
    <h2>Hip rotation review</h2>
    <p class="hint">
      Walk through every cross / hook / uppercut. Gap is <em>signed</em>
      (<code>L.x − R.x</code>), so the sign flips when the hip line crosses
      the camera axis — letting us measure crossover rotations at full
      magnitude. Per video we estimate <code>W_est</code> (the boxer's true
      hip/torso ratio at broadside) as the 99th percentile of
      <code>|signed gap|</code> across all rounds. Per punch we recover the
      hip-line angle <code>θ = arcsin(gap/W_est) ∈ [−90°, 90°]</code> at
      peak and trough, then convert to a 0–100 score
      <code>= clamp(rotation_deg / ${cfg.scoreTargetDeg}° × 100, 0, 100)</code>.
      The session score (mean over evaluable punches) drives the app's
      <span style="color:${COLOR_FAIL}">high</span> /
      <span style="color:${COLOR_WARN}">medium</span> /
      <span style="color:${COLOR_PASS}">low</span> / hide severity.
      Per-punch we also tag a <code>pass/warn/fail</code> tier
      (<code>&lt;${cfg.minRotationDeg}°</code> fail,
      <code>${cfg.minRotationDeg}–${cfg.solidRotationDeg}°</code> warn,
      <code>≥${cfg.solidRotationDeg}°</code> pass) for spot-checking. Skip when
      <code>min |gap|/W_est > ${cfg.noiseRatioMin}</code> (hips stayed
      saturated near ±W throughout — arcsin noise floor too high). Frames
      where either hip's confidence is below
      <code>${cfg.minHipConf}</code> are dropped from the signal.
    </p>
    <p class="hint" style="margin-top:6px">
      Canvas overlay:
      <span style="color:${COLOR_HIP}">●</span> current-frame hip line,
      <span style="color:${COLOR_HIP_PEAK}">●</span> hip line at peak frame (widest gap),
      <span style="color:${COLOR_HIP_TROUGH}">●</span> hip line at trough frame (narrowest gap).
      The two ghost lines are the evidence behind the verdict.
    </p>
    <div class="ol-nav" style="display:flex; gap:8px; align-items:center; margin:10px 0 8px;">
      <button id="hrr-prev" class="orient-btn-action secondary" style="padding:6px 10px;">⏮ prev (P)</button>
      <button id="hrr-next" class="orient-btn-action secondary" style="padding:6px 10px;">next (N) ⏭</button>
      <button id="hrr-mute" class="orient-btn-action secondary" style="padding:6px 10px;">mute (M)</button>
      <span id="hrr-counter" style="margin-left:6px; color:#888; font-size:12px;"></span>
    </div>
    <div style="margin:0 0 8px; font-size:12px; color:#888; font-family:ui-monospace,monospace;">
      <span id="hrr-w-est"></span>
    </div>
    <div id="hrr-w-snapshot-wrap" style="margin:0 0 14px; display:none;">
      <canvas id="hrr-w-snapshot" width="240" height="160"
              style="display:block; background:#111; border:1px solid #2a2a2a; border-radius:4px;"></canvas>
      <div id="hrr-w-snapshot-cap" style="margin-top:4px; font-size:11px; color:#888; font-family:ui-monospace,monospace;"></div>
    </div>
    <div id="hrr-state" class="hint" style="line-height:1.7;"></div>
    <div id="hrr-summary" class="hint" style="margin-top:14px; padding-top:10px; border-top:1px solid #2a2a2a;"></div>
    <div id="hrr-table-wrap" style="margin-top:10px; max-height:360px; overflow-y:auto;"></div>
    <p class="hint" style="margin-top:14px; font-size:11px;">
      Loops within the punch window. Score target
      <code>${cfg.scoreTargetDeg}°</code>; severity cutoffs
      <code>≥${cfg.scoreHideAbove}</code> hide,
      <code>${cfg.scoreLowAbove}+</code> low,
      <code>${cfg.scoreMedAbove}+</code> medium,
      <code>else</code> high. Tweak in <code>hip_rotation_review.js</code>.
    </p>
  `;
  host.querySelector("#hrr-prev")?.addEventListener("click",
    () => seekToPunch(activeIdx - 1, latestState));
  host.querySelector("#hrr-next")?.addEventListener("click",
    () => seekToPunch(activeIdx + 1, latestState));
  host.querySelector("#hrr-mute")?.addEventListener("click", toggleMute);
  updateMuteButton();
  updateWEstReadout();
  renderPunchTable();
}

function updateWEstReadout() {
  const el = host?.querySelector("#hrr-w-est");
  if (el && signals && Number.isFinite(signals.wEst)) {
    el.textContent = `W_est = ${signals.wEst.toFixed(3)} (hip/torso, 99th %ile across video)`;
  }
}

// Capture the W_est candidate frame (video frame at signals.wEstFrame plus
// the skeleton at that frame) into the sidebar snapshot canvas so the user
// can sanity-check that the frame W_est was sampled from really IS a
// broadside-hip frame. Uses the viewer's shared thumb-video element to seek
// without disturbing main playback — same trick the scrubber-hover preview
// in viewer.js uses. The seek is async; we re-trigger when state changes.
let lastSnapshotKey = null;   // dedupe by (stem, frame) to avoid re-seek loops
function captureWEstSnapshot(state, signals) {
  if (!host || !signals || !state.pose) return;
  const wrap = host.querySelector("#hrr-w-snapshot-wrap");
  const canvas = host.querySelector("#hrr-w-snapshot");
  const cap = host.querySelector("#hrr-w-snapshot-cap");
  if (!wrap || !canvas || !cap) return;
  if (!Number.isFinite(signals.wEst) || !Number.isFinite(signals.wEstFrame)) {
    wrap.style.display = "none"; return;
  }
  wrap.style.display = "block";

  const thumbVideo = document.getElementById("thumb-video");
  if (!thumbVideo || !thumbVideo.src) {
    cap.textContent = "video not loaded";
    return;
  }
  const frame = signals.wEstFrame;
  const stem = state.cacheBasename || "";
  const key = `${stem}:${state.cacheRound}:${frame}`;
  if (key === lastSnapshotKey) return;
  lastSnapshotKey = key;

  const fps = state.fps || state.pose.fps || 30;
  const startSec = state.start_sec || 0;
  const targetTime = startSec + (frame + 0.5) / fps;

  const draw = () => {
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    const vw = thumbVideo.videoWidth || state.pose.width || 16;
    const vh = thumbVideo.videoHeight || state.pose.height || 9;
    const scale = Math.min(W / vw, H / vh);
    const dw = vw * scale, dh = vh * scale;
    const dx = (W - dw) / 2, dy = (H - dh) / 2;
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, W, H);
    try { ctx.drawImage(thumbVideo, dx, dy, dw, dh); } catch { /* not ready */ }
    ctx.save();
    ctx.translate(dx, dy);
    ctx.scale(scale, scale);
    drawSkeleton(ctx, state.pose, frame, {
      boneColor: "rgba(255,255,255,0.45)",
      boneWidth: 2,
      jointRadius: 3,
    });
    // Emphasised hip line in amber — the actual segment whose length IS W_est.
    const lc = state.pose.conf[frame * 17 + J.L_HIP];
    const rc = state.pose.conf[frame * 17 + J.R_HIP];
    if (lc >= 0.05 && rc >= 0.05) {
      const lx = state.pose.skeleton[(frame * 17 + J.L_HIP) * 2];
      const ly = state.pose.skeleton[(frame * 17 + J.L_HIP) * 2 + 1];
      const rx = state.pose.skeleton[(frame * 17 + J.R_HIP) * 2];
      const ry = state.pose.skeleton[(frame * 17 + J.R_HIP) * 2 + 1];
      ctx.strokeStyle = COLOR_HIP_PEAK;
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(lx, ly); ctx.lineTo(rx, ry);
      ctx.stroke();
      ctx.fillStyle = COLOR_HIP_PEAK;
      ctx.beginPath(); ctx.arc(lx, ly, 4, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(rx, ry, 4, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();

    const cachedNote = signals.wEstFromCache ? " (W_est from another round)" : "";
    cap.textContent =
      `W_est frame: f${frame} · t=${(startSec + frame / fps).toFixed(2)}s · gap=${signed3(signals.wEstFrameGap)}${cachedNote}`;
  };

  // If thumbVideo already at our target (or close), draw immediately.
  if (Math.abs(thumbVideo.currentTime - targetTime) < 0.05 && thumbVideo.readyState >= 2) {
    draw();
    return;
  }
  // Otherwise seek and draw on completion. Use a one-shot listener so we
  // don't interfere with the scrubber-hover thumbnail's seeked handler.
  const onSeeked = () => {
    thumbVideo.removeEventListener("seeked", onSeeked);
    draw();
  };
  thumbVideo.addEventListener("seeked", onSeeked);
  thumbVideo.currentTime = targetTime;
}

// Build the per-punch summary table. Called once after the skeleton is
// up, and again whenever the punches array changes (from rebuildPunches).
function renderPunchTable() {
  if (!host) return;
  const container = host.querySelector("#hrr-table-wrap");
  if (!container) return;
  if (!punches.length) { container.innerHTML = ""; return; }

  const rows = punches.map((p, i) => {
    const predCol = colorFor(p.predicted);
    const typeStr = (p.punch_type || "?").replace(/_/g, " ");
    const tStr = Number.isFinite(p.timestamp) ? p.timestamp.toFixed(2) + "s" : "—";
    const metricStr = Number.isFinite(p.rotation_deg) ? p.rotation_deg.toFixed(1) + "°" : "—";
    const scoreStr = Number.isFinite(p.score) ? p.score.toFixed(0) : "—";
    return `<tr data-idx="${i}" style="border-bottom:1px solid rgba(255,255,255,0.04);">
      <td style="padding:4px 6px; text-align:right; color:#888;">${i + 1}</td>
      <td style="padding:4px 6px; color:#aaa; font-family:ui-monospace, monospace;">${tStr}</td>
      <td style="padding:4px 6px;">${typeStr}</td>
      <td style="padding:4px 6px; text-align:right; font-family:ui-monospace, monospace; color:${predCol};">${metricStr}</td>
      <td style="padding:4px 6px; text-align:right; font-family:ui-monospace, monospace; color:${predCol};">${scoreStr}</td>
      <td style="padding:4px 6px;">${pill(p.predicted, predCol)}</td>
    </tr>`;
  }).join("");

  container.innerHTML = `
    <table style="width:100%; border-collapse:collapse; font-size:12px;">
      <thead>
        <tr style="text-align:left; color:#888; border-bottom:1px solid #2a2a2a;">
          <th style="padding:4px 6px; text-align:right; font-weight:600;">#</th>
          <th style="padding:4px 6px; font-weight:600;">t</th>
          <th style="padding:4px 6px; font-weight:600;">type</th>
          <th style="padding:4px 6px; text-align:right; font-weight:600;">rotation</th>
          <th style="padding:4px 6px; text-align:right; font-weight:600;">score</th>
          <th style="padding:4px 6px; font-weight:600;">verdict</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;

  container.querySelectorAll("tr[data-idx]").forEach(tr => {
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => {
      const idx = parseInt(tr.getAttribute("data-idx"), 10);
      seekToPunch(idx, latestState);
    });
    tr.addEventListener("mouseenter", () => {
      const idx = parseInt(tr.getAttribute("data-idx"), 10);
      if (idx !== activeIdx) tr.style.background = "rgba(255,255,255,0.04)";
    });
    tr.addEventListener("mouseleave", () => {
      const idx = parseInt(tr.getAttribute("data-idx"), 10);
      if (idx !== activeIdx) tr.style.background = "";
    });
  });

  updateActiveRow();
}

// Highlight the row matching activeIdx. Cheap — just toggles styles, no
// re-render. Called every rebuildSidebar tick.
function updateActiveRow() {
  if (!host) return;
  host.querySelectorAll("tr[data-idx]").forEach(tr => {
    const idx = parseInt(tr.getAttribute("data-idx"), 10);
    if (idx === activeIdx) {
      tr.style.background = "rgba(255,210,74,0.14)";
      tr.style.fontWeight = "600";
    } else {
      tr.style.background = "";
      tr.style.fontWeight = "normal";
    }
  });
}

function toggleMute() {
  if (!videoEl) return;
  videoEl.muted = !videoEl.muted;
  updateMuteButton();
}

function updateMuteButton() {
  const btn = host?.querySelector("#hrr-mute");
  if (!btn || !videoEl) return;
  btn.textContent = videoEl.muted ? "unmute (M)" : "mute (M)";
}

function pill(text, color) {
  return `<span style="display:inline-block;padding:1px 8px;border-radius:10px;
    font-size:12px;font-weight:600;letter-spacing:0.02em;
    background:${color}1f;color:${color};border:1px solid ${color}66">${text}</span>`;
}

function rebuildSidebar(state) {
  if (!host) return;
  latestState = state;
  if (!host.querySelector("#hrr-state")) buildSidebarSkeleton();

  updateActiveRow();
  updateWEstReadout();
  captureWEstSnapshot(state, signals);

  const counter = host.querySelector("#hrr-counter");
  if (counter) {
    counter.textContent = punches.length
      ? `${activeIdx + 1} / ${punches.length}`
      : "no rotation-applicable punches in this round";
  }

  // Aggregate summary across all punches in the round.
  const summary = host.querySelector("#hrr-summary");
  if (summary) {
    if (!punches.length) {
      summary.innerHTML = "";
    } else {
      const counts = { pass: 0, warn: 0, fail: 0, skip: 0, unclear: 0 };
      let agree = 0, withGt = 0;
      for (const p of punches) {
        counts[p.predicted] = (counts[p.predicted] || 0) + 1;
        if (p.label) {
          // GT is binary pass/fail; collapse our 3 tiers to that for agreement.
          const predGT = p.predicted === "pass" ? "pass"
                       : (p.predicted === "warn" || p.predicted === "fail") ? "fail"
                       : null;
          if (predGT) {
            withGt++;
            if (p.label === predGT) agree++;
          }
        }
      }
      const agreeStr = withGt
        ? `${agree}/${withGt} agree (${Math.round(100 * agree / withGt)}%)`
        : `no GT verdicts`;
      const sScore = signals?.sessionScore;
      const sSev = signals?.sessionSeverity;
      const sN = signals?.sessionN || 0;
      const scoreStr = Number.isFinite(sScore) ? sScore.toFixed(0) : "—";
      const sevLabel = sSev === null ? "hide (boxer ok)" : (sSev || "—");
      const sevCol = sSev ? severityColor(sSev) : COLOR_PASS;
      summary.innerHTML =
        `<b>Round score:</b> `
        + `${pill(`${scoreStr} / 100`, sevCol)} `
        + `${pill(sevLabel, sevCol)} `
        + `<span class="muted">over ${sN} evaluable punch${sN === 1 ? "" : "es"}</span>`
        + `<br><span class="muted">Tier counts:</span> `
        + `${pill(`${counts.pass} pass`, COLOR_PASS)} `
        + `${pill(`${counts.warn} warn`, COLOR_WARN)} `
        + `${pill(`${counts.fail} fail`, COLOR_FAIL)} `
        + `${pill(`${counts.skip} skip`, COLOR_SKIP)} `
        + `· ${agreeStr}`;
    }
  }

  const el = host.querySelector("#hrr-state");
  if (!el) return;

  if (!punches.length) {
    el.innerHTML = state.labels?.detections
      ? `<span class="muted">No rotation-applicable punches in this round (cross/hook/uppercut).</span>`
      : `<span class="muted">Waiting for label data…</span>`;
    return;
  }

  const p = punches[activeIdx];
  if (!p) { el.innerHTML = ""; return; }

  const noiseZone = Number.isFinite(p.min_abs_ratio) && p.min_abs_ratio > cfg.noiseRatioMin;
  const mRatioStr = Number.isFinite(p.min_abs_ratio) ? p.min_abs_ratio.toFixed(3) : "—";

  // Verdict line. GT is binary pass/fail; warn collapses to fail for ✓/✗.
  const predCol = colorFor(p.predicted);
  const predBinary = p.predicted === "pass" ? "pass"
                   : (p.predicted === "warn" || p.predicted === "fail") ? "fail"
                   : null;
  const verdictLine = `<span style="color:${predCol}">predicted:</span> ${pill(p.predicted, predCol)}`
    + (p.label
        ? ` · <span class="muted">GT:</span> ${pill(p.label, colorFor(p.label))}`
          + (predBinary && p.label === predBinary
              ? ` <span style="color:${COLOR_PASS}">✓</span>`
              : ` <span style="color:${COLOR_FAIL}">✗</span>`)
        : ` · <span class="muted">no GT</span>`);

  const validFrac = Number.isFinite(p.valid_frac) ? p.valid_frac : 0;
  const lowValid = validFrac < cfg.minValidFrac;
  const validStr = `${p.n_valid}/${p.n_window} (${(100*validFrac).toFixed(0)}%)`;

  const lines = [
    `<b>${p.punch_type.replace(/_/g, " ")}</b> · <code>${p.hand}</code> · stance <code>${p.stance || "?"}</code>`,
    `label window <code>${p.start_frame}-${p.end_frame}</code> · search (looped) <code>${p.search_start}-${p.search_end}</code> (−${cfg.searchPreSec.toFixed(2)}s / +${cfg.searchPostSec.toFixed(2)}s)`,
    "",
    `<b>1. Gates</b>`,
    `valid frames = <code>${validStr}</code>`
      + (lowValid
          ? ` · ${pill("LOW VALIDITY → skip", COLOR_SKIP)} (thr ${(cfg.minValidFrac*100).toFixed(0)}%)`
          : ` · ${pill("OK", COLOR_PASS)}`),
    `min |gap| / W_est = <code>${mRatioStr}</code>`
      + (noiseZone ? ` · ${pill("NOISE ZONE → skip", COLOR_SKIP)} (thr ${cfg.noiseRatioMin})`
                   : ` · ${pill("OK", COLOR_PASS)}`),
    "",
  ];

  const wEstStr = Number.isFinite(signals.wEst) ? signals.wEst.toFixed(3) : "—";
  const pRatio = signed3(p.peak_ratio);
  const tRatio = signed3(p.trough_ratio);
  const pTheta = signedDeg1(p.peak_theta);
  const tTheta = signedDeg1(p.trough_theta);
  const rotDeg = Number.isFinite(p.rotation_deg) ? p.rotation_deg.toFixed(1) + "°" : "—";
  const rotCol = noiseZone ? COLOR_SKIP : colorFor(p.predicted);
  const thrTxt = `<code>&lt;${cfg.minRotationDeg}°</code> fail · `
               + `<code>${cfg.minRotationDeg}–${cfg.solidRotationDeg}°</code> warn · `
               + `<code>≥${cfg.solidRotationDeg}°</code> pass`;
  const scoreStr = Number.isFinite(p.score) ? p.score.toFixed(0) : "—";
  lines.push(
    `<b>2. Rotation (degrees)</b>`,
    `<code>W_est = ${wEstStr}</code> (99th %ile of |signed gap| / torso across video)`,
    `peak <code>${pRatio}</code> = sin(<code>${pTheta}</code>) @frame <code>${p.peak_frame}</code>`,
    `trough <code>${tRatio}</code> = sin(<code>${tTheta}</code>) @frame <code>${p.trough_frame}</code>`,
    `rotation = peak θ − trough θ = <code style="color:${rotCol}">${rotDeg}</code>`,
    `score = clamp(rotation / ${cfg.scoreTargetDeg}° × 100, 0, 100) = <code style="color:${rotCol}">${scoreStr} / 100</code>`,
    `tiers: ${thrTxt}`,
  );

  lines.push(
    "",
    `<b>3. Verdict</b>`,
    verdictLine,
  );
  el.innerHTML = lines.join("<br>");
}

// ─── draw ─────────────────────────────────────────────────────────────────

function drawHipLine(ctx, pose, frame, scale, opts = {}) {
  const color = opts.color || COLOR_HIP;
  const alpha = opts.alpha ?? 0.85;
  const lineWidth = opts.lineWidth ?? 3 * scale;
  const dotRadius = opts.dotRadius ?? 5 * scale;
  const label = opts.label || null;

  const lc = pose.conf[frame * 17 + J.L_HIP];
  const rc = pose.conf[frame * 17 + J.R_HIP];
  if (lc < 0.05 || rc < 0.05) return null;
  const lx = pose.skeleton[(frame * 17 + J.L_HIP) * 2];
  const ly = pose.skeleton[(frame * 17 + J.L_HIP) * 2 + 1];
  const rx = pose.skeleton[(frame * 17 + J.R_HIP) * 2];
  const ry = pose.skeleton[(frame * 17 + J.R_HIP) * 2 + 1];

  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.globalAlpha = alpha;
  if (opts.dashed) ctx.setLineDash([6 * scale, 4 * scale]);
  ctx.beginPath();
  ctx.moveTo(lx, ly); ctx.lineTo(rx, ry);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath(); ctx.arc(lx, ly, dotRadius, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(rx, ry, dotRadius, 0, Math.PI * 2); ctx.fill();

  if (label) {
    // Small label above the midpoint of the line.
    const mx = (lx + rx) / 2;
    const my = (ly + ry) / 2;
    const fontPx = Math.round(11 * scale);
    ctx.globalAlpha = 1;
    ctx.font = `bold ${fontPx}px ui-monospace, "SF Mono", Menlo, monospace`;
    const tw = ctx.measureText(label).width;
    const padX = 4 * scale, padY = 3 * scale;
    const h = fontPx + 2 * padY;
    ctx.fillStyle = "rgba(0,0,0,0.78)";
    ctx.fillRect(mx - tw / 2 - padX, my - h - 6 * scale, tw + 2 * padX, h);
    ctx.fillStyle = color;
    ctx.textBaseline = "top";
    ctx.fillText(label, mx - tw / 2, my - h - 6 * scale + padY);
  }
  ctx.restore();
  return { lx, ly, rx, ry };
}

function drawCanvas(ctx, state) {
  if (!punches.length || activeIdx < 0) return;
  const p = punches[activeIdx];
  const f = state.frame;
  const s = state.renderScale || 1;
  const pose = state.pose;

  // Ghost hip lines at the two extreme frames inside the search window —
  // peak and trough of the gap signal. Drawn without inline labels (they
  // overlap each other when the two ghost lines are close together);
  // their gap/θ values are shown in the corner legend instead.
  drawHipLine(ctx, pose, p.trough_frame, s, {
    color: COLOR_HIP_TROUGH,
    alpha: 0.55,
    lineWidth: 2.5 * s,
    dotRadius: 4 * s,
    dashed: true,
  });
  drawHipLine(ctx, pose, p.peak_frame, s, {
    color: COLOR_HIP_PEAK,
    alpha: 0.55,
    lineWidth: 2.5 * s,
    dotRadius: 4 * s,
    dashed: true,
  });

  // Current-frame hip line (purple, solid, full opacity) — drawn last so
  // it stays on top.
  drawHipLine(ctx, pose, f, s);

  // Corner legend for the peak/trough ghost lines (top-right).
  drawCornerLabels(ctx, p, s);

  drawHud(ctx, p, s);
}

// Top-right corner legend showing peak and trough gap → θ. Replaces the
// inline midpoint labels which overlapped when the two ghost hip lines
// sat close to each other.
function drawCornerLabels(ctx, p, s) {
  if (!Number.isFinite(p.peak_gap) || !Number.isFinite(p.trough_gap)) return;
  const peakStr = Number.isFinite(p.peak_theta)
    ? `peak  gap ${signed3(p.peak_gap)} → θ ${signedDeg0(p.peak_theta)}`
    : `peak  gap ${signed3(p.peak_gap)}`;
  const troughStr = Number.isFinite(p.trough_theta)
    ? `trough  gap ${signed3(p.trough_gap)} → θ ${signedDeg0(p.trough_theta)}`
    : `trough  gap ${signed3(p.trough_gap)}`;

  const fontPx = Math.round(12 * s);
  const lineH  = Math.round(20 * s);
  const padX   = 10 * s;
  const padY   = 8 * s;
  const margin = 24 * s;
  const swatch = 8 * s;
  const swatchGap = 6 * s;

  ctx.save();
  ctx.font = `bold ${fontPx}px ui-monospace, "SF Mono", Menlo, monospace`;
  const peakW = ctx.measureText(peakStr).width;
  const troughW = ctx.measureText(troughStr).width;
  const textW = Math.max(peakW, troughW);
  const w = textW + 2 * padX + swatch + swatchGap;
  const h = padY * 2 + lineH * 2 - (lineH - fontPx);
  const x0 = ctx.canvas.width - w - margin;
  const y0 = margin;

  // Background pill.
  ctx.fillStyle = "rgba(0,0,0,0.78)";
  const r = 10 * s;
  ctx.beginPath();
  ctx.moveTo(x0 + r, y0);
  ctx.arcTo(x0 + w, y0,     x0 + w, y0 + h, r);
  ctx.arcTo(x0 + w, y0 + h, x0,     y0 + h, r);
  ctx.arcTo(x0,     y0 + h, x0,     y0,     r);
  ctx.arcTo(x0,     y0,     x0 + w, y0,     r);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.lineWidth = 1 * s;
  ctx.stroke();

  ctx.textBaseline = "alphabetic";
  let y = y0 + padY + fontPx;

  ctx.fillStyle = COLOR_HIP_PEAK;
  ctx.fillRect(x0 + padX, y - fontPx + 2 * s, swatch, fontPx - 3 * s);
  ctx.fillText(peakStr, x0 + padX + swatch + swatchGap, y);
  y += lineH;

  ctx.fillStyle = COLOR_HIP_TROUGH;
  ctx.fillRect(x0 + padX, y - fontPx + 2 * s, swatch, fontPx - 3 * s);
  ctx.fillText(troughStr, x0 + padX + swatch + swatchGap, y);

  ctx.restore();
}

function drawHud(ctx, p, s) {
  const predCol = colorFor(p.predicted);
  const noiseZone = Number.isFinite(p.min_abs_ratio) && p.min_abs_ratio > cfg.noiseRatioMin;

  const pTheta = signedDeg0(p.peak_theta);
  const tTheta = signedDeg0(p.trough_theta);
  const rotDeg = Number.isFinite(p.rotation_deg) ? `${p.rotation_deg.toFixed(1)}°` : "—";
  const scoreStr = Number.isFinite(p.score) ? `${p.score.toFixed(0)}` : "—";
  const peakTxt = `peak θ ${pTheta}  ·  trough θ ${tTheta}`;
  const metricTxt = `rotation ${rotDeg}  ·  score ${scoreStr}/100`;
  const metricCol = noiseZone ? COLOR_SKIP : colorFor(p.predicted);

  const mRatioStr = Number.isFinite(p.min_abs_ratio) ? p.min_abs_ratio.toFixed(3) : "—";
  const gateTxt = `min|gap|/W ${mRatioStr}  ·  thr ${cfg.noiseRatioMin}`;
  const gateCol = noiseZone ? COLOR_SKIP : COLOR_PASS;

  const titleTxt = `${p.hand} ${p.punch_type.replace(/_/g, " ")}  ·  ${p.stance || "?"}`;
  const predTxt  = `pred: ${p.predicted}`;
  const gtTxt    = p.label ? `GT: ${p.label}` : "GT: —";
  // GT is binary pass/fail; warn collapses to fail for the ✓/✗ check.
  const predBinary = p.predicted === "pass" ? "pass"
                   : (p.predicted === "warn" || p.predicted === "fail") ? "fail"
                   : null;
  const agreeSym = p.label && predBinary
    ? (p.label === predBinary ? "  ✓" : "  ✗") : "";

  const fontPx = 15 * s;
  const lineH  = 22 * s;
  const padX   = 14 * s;
  const padY   = 10 * s;
  const x0 = 24 * s, y0 = 24 * s;

  const lines = [titleTxt, gateTxt, peakTxt, metricTxt, predTxt, gtTxt + agreeSym];

  ctx.save();
  ctx.font = `bold ${fontPx}px ui-monospace, "SF Mono", Menlo, monospace`;
  const w = Math.max(...lines.map(t => ctx.measureText(t).width)) + 2 * padX;
  const h = padY * 2 + lineH * lines.length;

  ctx.fillStyle = "rgba(0,0,0,0.78)";
  const r = 10 * s;
  ctx.beginPath();
  ctx.moveTo(x0 + r, y0);
  ctx.arcTo(x0 + w, y0,     x0 + w, y0 + h, r);
  ctx.arcTo(x0 + w, y0 + h, x0,     y0 + h, r);
  ctx.arcTo(x0,     y0 + h, x0,     y0,     r);
  ctx.arcTo(x0,     y0,     x0 + w, y0,     r);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.lineWidth = 1 * s;
  ctx.stroke();

  let y = y0 + padY + fontPx;
  ctx.fillStyle = "rgba(255,255,255,0.92)"; ctx.fillText(titleTxt, x0 + padX, y); y += lineH;
  ctx.fillStyle = gateCol;                  ctx.fillText(gateTxt, x0 + padX, y); y += lineH;
  ctx.fillStyle = "rgba(255,255,255,0.78)"; ctx.fillText(peakTxt, x0 + padX, y); y += lineH;
  ctx.fillStyle = metricCol;                ctx.fillText(metricTxt, x0 + padX, y); y += lineH;
  ctx.fillStyle = predCol;                  ctx.fillText(predTxt, x0 + padX, y); y += lineH;
  ctx.fillStyle = p.label ? colorFor(p.label) : "rgba(255,255,255,0.55)";
  ctx.fillText(gtTxt, x0 + padX, y);
  if (agreeSym) {
    ctx.fillStyle = p.label === predBinary ? COLOR_PASS : COLOR_FAIL;
    ctx.fillText(agreeSym, x0 + padX + ctx.measureText(gtTxt).width, y);
  }
  ctx.restore();
}

// ─── lens contract ────────────────────────────────────────────────────────

export const HipRotationReviewRule = {
  id: "hip_rotation_review",
  label: "Hip rotation review (loop)",

  // Apple Vision is the production pose source — it's what the iOS app
  // runs and what the model is calibrated against. YOLO is for parity
  // tooling, not for verdicts. Refuse rounds that only have YOLO.
  requires(slot) {
    return !!slot?.vision;
  },

  skeletonStyle() {
    return {
      boneColor: "rgba(255,255,255,0.22)",
      boneWidth: 1.5,
      jointRadius: 3,
      highlightJoints: new Set([J.L_HIP, J.R_HIP, J.L_SHOULDER, J.R_SHOULDER]),
    };
  },

  mount(_host, state) {
    host = _host;
    videoEl = document.getElementById("video");
    cfg = { ...DEFAULTS };
    lastPose = null;
    lastDetectionsRef = null;
    lastStemForReset = null;
    rebuildPunches(state);
    installTimeupdateLoop(state);
    installKeyHandlers(state);
  },

  update(state) {
    rebuildPunches(state);
    // If the user scrubbed away from the active punch, hop to whichever
    // rotation-applicable punch the cursor is now inside. Compares against
    // the search window (label window + asymmetric pre/post padding) since
    // that's what the loop plays. Small extra tolerance handles the race
    // between the viewer's frame update and the snap-back handler.
    const ACTIVE_TOLERANCE_FRAMES = 5;
    if (punches.length) {
      const f = state.frame;
      const active = activeIdx >= 0 ? punches[activeIdx] : null;
      const nearActive = active &&
        f >= active.search_start - ACTIVE_TOLERANCE_FRAMES &&
        f <= active.search_end   + ACTIVE_TOLERANCE_FRAMES;
      if (!nearActive) {
        const inside = punches.findIndex(p =>
          f >= p.search_start && f <= p.search_end);
        if (inside !== -1 && inside !== activeIdx) {
          activeIdx = inside;
          const p = punches[inside];
          loopWindow = { start_frame: p.search_start, end_frame: p.search_end };
          rebuildSidebar(state);
        }
      }
    }
  },

  draw(ctx, state) {
    drawCanvas(ctx, state);
  },

  unmount() {
    if (videoEl && timeupdateHandler) {
      videoEl.removeEventListener("timeupdate", timeupdateHandler);
    }
    if (keydownHandler) {
      document.removeEventListener("keydown", keydownHandler, true);
    }
    timeupdateHandler = null;
    keydownHandler = null;
    loopWindow = null;
    lastSnapshotKey = null;
  },
};
