// Hip rotation review — walk through every labelled rotation-applicable punch
// (cross/hook/uppercut, head/body), apply the orientation gate, and read off
// how much the hips swung during the punch window. Mirrors
// straights_review.js: one punch at a time, video loops within [start, end],
// N/P keys (or buttons) step to next/prev.
//
// Why the orientation gate matters here: the rule's signal is
//   gap[f] = |L_hip − R_hip| / torso_height
// When the boxer is too frontal/back-on (|facing| close to 0° or ±180°), the
// hip line collapses on the image and rotation creates depth-only motion that
// doesn't lift the gap. We tightly gate to broadside (|facing| ≈ 90°) so the
// hip line sits on the STEEP wings of sin(θ), where range ≈ rotation amount.
//
// Per-punch compute (no round-level baseline — too unreliable when the boxer
// moves, switches stance, or chains combos):
//   * gap[f] smoothed with moving-average over cfg.gapSmoothSeconds
//   * search window = [start − searchPreSec, end + searchPostSec]
//                     asymmetric: load-up often happens before the labeled
//                     start, but the labeled end already extends past the
//                     furthest rotation point (until the hand returns) so
//                     we don't need padding on the back side.
//   * peak  = max(gap[search])
//   * trough = min(gap[search])
//   * range = peak − trough     ← THE signal: how much the hips swung
//   * predicted = skip if orientation gate fails
//                 pass if range ≥ min_range
//                 fail otherwise

import { J, torsoHeight, drawSkeleton } from "../skeleton.js";
import { STANCE_FITS } from "./orientation_lens.js";

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
  orientationGate:        true,
  // Tiered orientation gate. INNER band = boxer broadside to camera, hip
  // line on the steep wings of sin(θ) — the cleanest regime. OUTER band =
  // boxer angled toward / away from camera, hip line closer to sin's peak
  // where each degree of rotation moves the gap less. Empirically the
  // metric still works in the outer band, just with a different (usually
  // smaller) range value for the same rotation amount. Set per-band
  // thresholds independently. Punches outside both bands → skip.
  orientationInnerMinAbsDeg: 60,
  orientationInnerMaxAbsDeg: 120,
  orientationOuterMinAbsDeg: 30,
  orientationOuterMaxAbsDeg: 150,
  // Verdict thresholds on (max gap − min gap) inside the search window.
  // Range, not delta-from-round-median: we don't trust the round median
  // when boxers move, switch guards, or chain combos.
  minRangeInner:          0.10,
  minRangeOuter:          0.10,    // tune from the table verdicts

  // ── Mode toggle ────────────────────────────────────────────────────
  // "range" = compare raw range (max gap − min gap) against minRangeInner /
  //           minRangeOuter per band. Default — works for all videos.
  // "ratio" = use W_est (99th percentile of gap across the round, ≈ this
  //           boxer's true hip-width / torso-height ratio) to convert gap
  //           into an implied hip-rotation angle in degrees. Skip punches
  //           where the trough gap stays close to W (hips parked broadside,
  //           arcsin too noisy). Requires the round to contain at least
  //           one near-broadside frame for W_est to be accurate.
  mode:                   "range",  // "range" | "ratio"
  wEstQuantile:           0.99,
  noiseRatioMin:          0.95,     // skip if trough/W_est > this
  minRotationDeg:         15,       // pass threshold in ratio mode
  // Rule only applies to punches with rotation expectation (jab + body shots
  // excluded — matches hip_rotation.js).
  appliesTo: new Set([
    "cross_head", "cross_body",
    "lead_hook_head", "lead_uppercut_head",
    "rear_uppercut_head", "rear_hook_head",
  ]),
};

const MIN_ANKLE_CONF = 0.30;
const COLOR_HIP        = "#a78bfa";  // purple — current-frame hip line
const COLOR_HIP_PEAK   = "#ffd24a";  // amber  — hip line at peak (widest) frame
const COLOR_HIP_TROUGH = "#ff7e3a";  // orange — hip line at trough (narrowest) frame
const COLOR_PRED       = "#3ad9e0";  // cyan   — predicted facing
const COLOR_PASS       = "#5fd97a";
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
let signals = null;           // {gap, punches, fps}
let lastDetectionsRef = null;
let lastStemForReset = null;
let lastPose = null;
let latestState = null;
let cfg = { ...DEFAULTS };

// ─── helpers ──────────────────────────────────────────────────────────────

function wrap180(deg) { return ((deg + 180) % 360 + 360) % 360 - 180; }

function colorFor(predicted) {
  if (predicted === "pass") return COLOR_PASS;
  if (predicted === "fail") return COLOR_FAIL;
  if (predicted === "skip") return COLOR_SKIP;
  return COLOR_UNCLEAR;
}

function movingAvg(arr, w) {
  if (w <= 1) return arr;
  const n = arr.length;
  const out = new Float32Array(n);
  const half = Math.floor(w / 2);
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - half), hi = Math.min(n - 1, i + half);
    let s = 0;
    for (let k = lo; k <= hi; k++) s += arr[k];
    out[i] = s / (hi - lo + 1);
  }
  return out;
}

// Window-median ankle-arrow → predicted facing direction (same algorithm
// arm_extension / 07_punch_directions use). Returns null when fewer than
// 3 frames clear the ankle conf gate.
function medianPredictedFacing(pose, sf, ef, stance) {
  if (!stance) return null;
  const fit = STANCE_FITS[stance];
  if (!fit) return null;
  const dxs = [], dys = [];
  for (let f = sf; f <= ef; f++) {
    const cL = pose.conf[f * 17 + J.L_ANKLE];
    const cR = pose.conf[f * 17 + J.R_ANKLE];
    if (cL < MIN_ANKLE_CONF || cR < MIN_ANKLE_CONF) continue;
    const lx = pose.skeleton[(f * 17 + J.L_ANKLE) * 2];
    const ly = pose.skeleton[(f * 17 + J.L_ANKLE) * 2 + 1];
    const rx = pose.skeleton[(f * 17 + J.R_ANKLE) * 2];
    const ry = pose.skeleton[(f * 17 + J.R_ANKLE) * 2 + 1];
    if (![lx, ly, rx, ry].every(Number.isFinite)) continue;
    const orthodox = stance !== "southpaw";
    dxs.push(orthodox ? (lx - rx) : (rx - lx));
    dys.push(orthodox ? (ly - ry) : (ry - ly));
  }
  if (dxs.length < 3) return null;
  dxs.sort((a, b) => a - b);
  dys.sort((a, b) => a - b);
  const mid = Math.floor(dxs.length / 2);
  const mdx = dxs.length % 2 ? dxs[mid] : 0.5 * (dxs[mid - 1] + dxs[mid]);
  const mdy = dys.length % 2 ? dys[mid] : 0.5 * (dys[mid - 1] + dys[mid]);
  if (mdx * mdx + mdy * mdy < 1e-6) return null;
  const arrowDeg = Math.atan2(mdy, mdx) * 180 / Math.PI;
  return wrap180(fit.sign * arrowDeg + fit.offset_deg);
}

function hipMidAt(pose, f) {
  const cLH = pose.conf[f * 17 + J.L_HIP];
  const cRH = pose.conf[f * 17 + J.R_HIP];
  if (cLH < 0.2 || cRH < 0.2) return null;
  const lhx = pose.skeleton[(f * 17 + J.L_HIP) * 2];
  const lhy = pose.skeleton[(f * 17 + J.L_HIP) * 2 + 1];
  const rhx = pose.skeleton[(f * 17 + J.R_HIP) * 2];
  const rhy = pose.skeleton[(f * 17 + J.R_HIP) * 2 + 1];
  if (![lhx, lhy, rhx, rhy].every(Number.isFinite)) return null;
  return { x: 0.5 * (lhx + rhx), y: 0.5 * (lhy + rhy) };
}

function drawArrowhead(ctx, x, y, angle, size) {
  const a = 0.45;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x - size * Math.cos(angle - a), y - size * Math.sin(angle - a));
  ctx.lineTo(x - size * Math.cos(angle + a), y - size * Math.sin(angle + a));
  ctx.closePath();
  ctx.fill();
}

// ─── compute ──────────────────────────────────────────────────────────────

function computeAll(state, cfg) {
  const pose = state.pose;
  const fps = pose.fps || 30;
  const N = pose.n_frames;

  // gap[f] = |L_hip − R_hip| / torso_height, smoothed.
  const gapRaw = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const lx = pose.skeleton[(i * 17 + J.L_HIP) * 2];
    const ly = pose.skeleton[(i * 17 + J.L_HIP) * 2 + 1];
    const rx = pose.skeleton[(i * 17 + J.R_HIP) * 2];
    const ry = pose.skeleton[(i * 17 + J.R_HIP) * 2 + 1];
    const th = Math.max(1e-6, torsoHeight(pose, i));
    gapRaw[i] = Math.hypot(lx - rx, ly - ry) / th;
  }
  const smoothFrames = Math.max(1, Math.round(cfg.gapSmoothSeconds * fps));
  const gap = movingAvg(gapRaw, smoothFrames);

  // W_est: per-VIDEO estimate of this boxer's true hip-width / torso-height
  // ratio. Per-round we take a high quantile of the gap signal (default 99th
  // percentile) — the max visible gap approaches W·sin(90°) = W. Across rounds
  // of the same video we take the running max, so the estimate gets stronger
  // as the user visits more rounds. Stored in module-level wEstByVideo.
  const localWEst = quantile(gap, cfg.wEstQuantile);
  const stem = state.cacheBasename || null;
  let wEst = localWEst;
  if (stem && Number.isFinite(localWEst)) {
    const cached = wEstByVideo.get(stem);
    wEst = (cached != null && cached > localWEst) ? cached : localWEst;
    wEstByVideo.set(stem, wEst);
  }
  const wEstFromCache = (stem && wEstByVideo.get(stem) !== localWEst && wEst > localWEst);

  // Frame in THIS round whose gap is closest to W_est — used by the sidebar
  // snapshot so the user can sanity-check whether it's actually a broadside
  // frame. Falls back to the round's argmax-gap when W_est came from another
  // round.
  let wEstFrame = 0, wEstBestDiff = Infinity;
  for (let f = 0; f < gap.length; f++) {
    const diff = Math.abs(gap[f] - wEst);
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

    let peak = gap[ss], peakAt = ss, trough = gap[ss], troughAt = ss;
    for (let f = ss; f <= se; f++) {
      const g = gap[f];
      if (g > peak)   { peak = g;   peakAt = f; }
      if (g < trough) { trough = g; troughAt = f; }
    }
    const range = peak - trough;

    // Orientation gate — window-median predicted facing from ankle arrow.
    const stance = d.stance?.toLowerCase?.() || null;
    const orientationDeg = medianPredictedFacing(pose, sf, ef, stance);
    // Classify into a facing band. "inner" = broadside (steep wings of sin).
    // "outer" = angled toward/away from camera (closer to sin peak).
    // null = outside both bands → skip.
    let band = null;
    if (orientationDeg != null) {
      const a = Math.abs(orientationDeg);
      if (a >= cfg.orientationInnerMinAbsDeg && a <= cfg.orientationInnerMaxAbsDeg) {
        band = "inner";
      } else if (a >= cfg.orientationOuterMinAbsDeg && a <= cfg.orientationOuterMaxAbsDeg) {
        band = "outer";
      }
    }
    const threshold = band === "inner" ? cfg.minRangeInner
                    : band === "outer" ? cfg.minRangeOuter
                    : null;

    // Ratio-mode derived quantities. Always computed so the UI can show
    // them; only USED for the verdict when cfg.mode === "ratio".
    const peakRatio = wEst > 0 ? Math.min(1, peak / wEst) : NaN;
    const troughRatio = wEst > 0 ? Math.min(1, trough / wEst) : NaN;
    const peakTheta = Number.isFinite(peakRatio) ? Math.asin(peakRatio) * 180 / Math.PI : NaN;
    const troughTheta = Number.isFinite(troughRatio) ? Math.asin(troughRatio) * 180 / Math.PI : NaN;
    const rotationDeg = Number.isFinite(peakTheta) && Number.isFinite(troughTheta)
      ? peakTheta - troughTheta : NaN;

    let predicted;
    if (cfg.orientationGate && (orientationDeg == null || band == null)) {
      predicted = "skip";
    } else if (cfg.mode === "ratio") {
      // Ratio-mode: skip if hips parked at broadside (arcsin too noisy),
      // else compare implied rotation angle (in degrees) to minRotationDeg.
      if (Number.isFinite(troughRatio) && troughRatio > cfg.noiseRatioMin) {
        predicted = "skip";
      } else if (Number.isFinite(rotationDeg) && rotationDeg >= cfg.minRotationDeg) {
        predicted = "pass";
      } else {
        predicted = "fail";
      }
    } else if (range >= threshold) {
      // Range-mode: original peak-trough vs per-band threshold.
      predicted = "pass";
    } else {
      predicted = "fail";
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
      range,
      peak_ratio: peakRatio,
      trough_ratio: troughRatio,
      peak_theta: peakTheta,
      trough_theta: troughTheta,
      rotation_deg: rotationDeg,
      orientation_deg: orientationDeg,
      band,
      threshold,
      predicted,
      label,
    };
  });

  return { gap, punches: out, fps, wEst, wEstFrame, wEstFrameGap, wEstFromCache };
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
  if (keydownHandler) document.removeEventListener("keydown", keydownHandler);
  keydownHandler = (e) => {
    if (state.rule?.id !== "hip_rotation_review") return;
    const tag = e.target?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    if (e.key === "n" || e.key === "N") { e.preventDefault(); seekToPunch(activeIdx + 1, state); }
    else if (e.key === "p" || e.key === "P") { e.preventDefault(); seekToPunch(activeIdx - 1, state); }
    else if (e.key === "m" || e.key === "M") { e.preventDefault(); toggleMute(); }
  };
  document.addEventListener("keydown", keydownHandler);
}

// ─── sidebar ──────────────────────────────────────────────────────────────

function buildSidebarSkeleton() {
  if (!host) return;
  host.innerHTML = `
    <h2>Hip rotation review</h2>
    <p class="hint">
      Walk through every cross / hook / uppercut. Each punch is classified by
      the orientation gate into one of two bands:
      <b>inner</b> (broadside, <code>|facing| ∈ [${cfg.orientationInnerMinAbsDeg}°, ${cfg.orientationInnerMaxAbsDeg}°]</code>) or
      <b>outer</b> (camera-directed, <code>[${cfg.orientationOuterMinAbsDeg}°-${cfg.orientationInnerMinAbsDeg}°] ∪ [${cfg.orientationInnerMaxAbsDeg}°-${cfg.orientationOuterMaxAbsDeg}°]</code>).
      Outside both bands → <i>skip</i>. Inside, the verdict is
      <code>max(gap) − min(gap) ≥ minRange</code> (per-band threshold) — pure
      local swing, no round-level baseline.
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
    <div class="ol-nav" style="display:flex; gap:8px; align-items:center; margin:0 0 8px;">
      <span style="color:#888; font-size:12px;">mode:</span>
      <button id="hrr-mode-range" class="orient-btn-action secondary" style="padding:4px 10px; font-size:12px;">range</button>
      <button id="hrr-mode-ratio" class="orient-btn-action secondary" style="padding:4px 10px; font-size:12px;">ratio (W_est)</button>
      <span id="hrr-w-est" style="margin-left:6px; color:#888; font-size:12px; font-family:ui-monospace,monospace;"></span>
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
      Loops within the punch window. Per-band thresholds
      (<code>minRangeInner=${cfg.minRangeInner.toFixed(2)}</code>,
      <code>minRangeOuter=${cfg.minRangeOuter.toFixed(2)}</code>) and search
      padding come from the defaults; tweak in <code>hip_rotation_review.js</code>.
    </p>
  `;
  host.querySelector("#hrr-prev")?.addEventListener("click",
    () => seekToPunch(activeIdx - 1, latestState));
  host.querySelector("#hrr-next")?.addEventListener("click",
    () => seekToPunch(activeIdx + 1, latestState));
  host.querySelector("#hrr-mute")?.addEventListener("click", toggleMute);
  host.querySelector("#hrr-mode-range")?.addEventListener("click", () => setMode("range"));
  host.querySelector("#hrr-mode-ratio")?.addEventListener("click", () => setMode("ratio"));
  updateMuteButton();
  updateModeButtons();
  renderPunchTable();
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
      `W_est frame: f${frame} · t=${(startSec + frame / fps).toFixed(2)}s · gap=${signals.wEstFrameGap.toFixed(3)}${cachedNote}`;
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

function setMode(newMode) {
  if (cfg.mode === newMode) return;
  cfg.mode = newMode;
  // Re-score every punch under the new mode.
  if (latestState) {
    lastPose = null;  // force computeAll to re-run
    rebuildPunches(latestState);
  }
  updateModeButtons();
}

function updateModeButtons() {
  const rangeBtn = host?.querySelector("#hrr-mode-range");
  const ratioBtn = host?.querySelector("#hrr-mode-ratio");
  if (!rangeBtn || !ratioBtn) return;
  const active = cfg.mode || "range";
  for (const [btn, name] of [[rangeBtn, "range"], [ratioBtn, "ratio"]]) {
    if (name === active) {
      btn.style.background = "rgba(255,210,74,0.18)";
      btn.style.borderColor = COLOR_HIP_PEAK;
      btn.style.color = COLOR_HIP_PEAK;
      btn.style.fontWeight = "700";
    } else {
      btn.style.background = "";
      btn.style.borderColor = "";
      btn.style.color = "";
      btn.style.fontWeight = "normal";
    }
  }
  // Refresh W_est readout.
  const wEl = host?.querySelector("#hrr-w-est");
  if (wEl && signals && Number.isFinite(signals.wEst)) {
    wEl.textContent = `W_est = ${signals.wEst.toFixed(3)} (hip/torso)`;
  }
}

// Build the per-punch summary table. Called once after the skeleton is
// up, and again whenever the punches array changes (from rebuildPunches).
function renderPunchTable() {
  if (!host) return;
  const container = host.querySelector("#hrr-table-wrap");
  if (!container) return;
  if (!punches.length) { container.innerHTML = ""; return; }

  const isRatio = cfg.mode === "ratio";
  const metricCol = isRatio ? "rotation" : "range";

  const rows = punches.map((p, i) => {
    const predCol = colorFor(p.predicted);
    const typeStr = (p.punch_type || "?").replace(/_/g, " ");
    const tStr = Number.isFinite(p.timestamp) ? p.timestamp.toFixed(2) + "s" : "—";
    const bandStr = p.band || "—";
    const bandCol = p.band === "inner" ? COLOR_PASS
                  : p.band === "outer" ? COLOR_SKIP
                  : "#888";
    const metricStr = isRatio
      ? (Number.isFinite(p.rotation_deg) ? p.rotation_deg.toFixed(1) + "°" : "—")
      : p.range.toFixed(3);
    return `<tr data-idx="${i}" style="border-bottom:1px solid rgba(255,255,255,0.04);">
      <td style="padding:4px 6px; text-align:right; color:#888;">${i + 1}</td>
      <td style="padding:4px 6px; color:#aaa; font-family:ui-monospace, monospace;">${tStr}</td>
      <td style="padding:4px 6px;">${typeStr}</td>
      <td style="padding:4px 6px; font-size:11px; color:${bandCol};">${bandStr}</td>
      <td style="padding:4px 6px; text-align:right; font-family:ui-monospace, monospace; color:${predCol};">${metricStr}</td>
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
          <th style="padding:4px 6px; font-weight:600;">band</th>
          <th style="padding:4px 6px; text-align:right; font-weight:600;">${metricCol}</th>
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
  updateModeButtons();
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
      const counts = { pass: 0, fail: 0, skip: 0, unclear: 0 };
      let agree = 0, withGt = 0;
      for (const p of punches) {
        counts[p.predicted] = (counts[p.predicted] || 0) + 1;
        if (p.label) {
          withGt++;
          if (p.label === p.predicted) agree++;
        }
      }
      const agreeStr = withGt
        ? `${agree}/${withGt} agree (${Math.round(100 * agree / withGt)}%)`
        : `no GT verdicts`;
      summary.innerHTML =
        `<b>Round summary:</b> `
        + `${pill(`${counts.pass} pass`, COLOR_PASS)} `
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

  // 1) Orientation block
  const oDeg = p.orientation_deg;
  let orientLine;
  if (oDeg == null) {
    orientLine = `<span style="color:${COLOR_PRED}">orientation:</span> <code>—</code> · ${pill("UNKNOWN", COLOR_UNCLEAR)}`;
  } else {
    const a = Math.abs(oDeg);
    const status = p.band === "inner" ? pill("INNER ✓", COLOR_PASS)
                 : p.band === "outer" ? pill("OUTER ✓", COLOR_SKIP)
                 : pill("OUT OF GATE ✗", COLOR_FAIL);
    orientLine = `<span style="color:${COLOR_PRED}">orientation:</span> `
      + `<code>${oDeg.toFixed(1)}°</code> · |a|=<code>${a.toFixed(1)}°</code> · ${status}`;
  }

  // 2) Rotation data — content varies by mode.
  const rangeCol = (p.threshold != null && p.range >= p.threshold) ? COLOR_PASS : COLOR_FAIL;
  const thrStr = p.threshold != null ? p.threshold.toFixed(2) : "—";

  // 3) Verdict
  const predCol = colorFor(p.predicted);
  const verdictLine = `<span style="color:${predCol}">predicted:</span> ${pill(p.predicted, predCol)}`
    + (p.label
        ? ` · <span class="muted">GT:</span> ${pill(p.label, colorFor(p.label))}`
          + (p.label === p.predicted
              ? ` <span style="color:${COLOR_PASS}">✓</span>`
              : ` <span style="color:${COLOR_FAIL}">✗</span>`)
        : ` · <span class="muted">no GT</span>`);

  const lines = [
    `<b>${p.punch_type.replace(/_/g, " ")}</b> · <code>${p.hand}</code> · stance <code>${p.stance || "?"}</code>`,
    `label window <code>${p.start_frame}-${p.end_frame}</code> · search (looped) <code>${p.search_start}-${p.search_end}</code> (−${cfg.searchPreSec.toFixed(2)}s / +${cfg.searchPostSec.toFixed(2)}s)`,
    "",
    `<b>1. Orientation gate</b>`,
    orientLine,
    "",
  ];

  if (cfg.mode === "ratio") {
    // Ratio-mode signal display.
    const wEstStr = Number.isFinite(signals.wEst) ? signals.wEst.toFixed(3) : "—";
    const pRatio = Number.isFinite(p.peak_ratio) ? p.peak_ratio.toFixed(3) : "—";
    const tRatio = Number.isFinite(p.trough_ratio) ? p.trough_ratio.toFixed(3) : "—";
    const pTheta = Number.isFinite(p.peak_theta) ? p.peak_theta.toFixed(1) + "°" : "—";
    const tTheta = Number.isFinite(p.trough_theta) ? p.trough_theta.toFixed(1) + "°" : "—";
    const rotDeg = Number.isFinite(p.rotation_deg) ? p.rotation_deg.toFixed(1) + "°" : "—";
    const rotPass = Number.isFinite(p.rotation_deg) && p.rotation_deg >= cfg.minRotationDeg;
    const noiseZone = Number.isFinite(p.trough_ratio) && p.trough_ratio > cfg.noiseRatioMin;
    const rotCol = noiseZone ? COLOR_SKIP : (rotPass ? COLOR_PASS : COLOR_FAIL);
    lines.push(
      `<b>2. Hip / torso ratio → angle</b>`,
      `<code>W_est = ${wEstStr}</code> (99th %ile of gap/torso across round)`,
      `peak <code>${pRatio}</code> = sin(<code>${pTheta}</code>) @frame <code>${p.peak_frame}</code>`,
      `trough <code>${tRatio}</code> = sin(<code>${tTheta}</code>) @frame <code>${p.trough_frame}</code>`,
      `rotation = <code style="color:${rotCol}">${rotDeg}</code> (threshold <code>${cfg.minRotationDeg}°</code>)`
        + (noiseZone ? ` · ${pill("NOISE ZONE", COLOR_SKIP)} trough/W > ${cfg.noiseRatioMin}` : ""),
    );
  } else {
    // Range-mode signal display (original).
    lines.push(
      `<b>2. Hip rotation swing (local)</b>`,
      `peak gap <code>${p.peak_gap.toFixed(3)}</code> @frame <code>${p.peak_frame}</code> · trough <code>${p.trough_gap.toFixed(3)}</code> @frame <code>${p.trough_frame}</code>`,
      `range = peak − trough = <code style="color:${rangeCol}">${p.range.toFixed(3)}</code> (${p.band || "—"} band threshold <code>${thrStr}</code>)`,
    );
  }

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
  // these are the two frames whose gap difference IS the range we're
  // testing. Drawn semi-transparent / dashed so they sit behind the
  // current-frame line. Labels show gap value, and (in ratio mode) the
  // implied angle θ recovered via arcsin(gap/W_est).
  const ratioMode = cfg.mode === "ratio";
  const troughLabel = ratioMode && Number.isFinite(p.trough_theta)
    ? `trough · ${p.trough_gap.toFixed(3)} → ${p.trough_theta.toFixed(0)}°`
    : `trough · ${p.trough_gap.toFixed(3)}`;
  const peakLabel = ratioMode && Number.isFinite(p.peak_theta)
    ? `peak · ${p.peak_gap.toFixed(3)} → ${p.peak_theta.toFixed(0)}°`
    : `peak · ${p.peak_gap.toFixed(3)}`;
  drawHipLine(ctx, pose, p.trough_frame, s, {
    color: COLOR_HIP_TROUGH,
    alpha: 0.55,
    lineWidth: 2.5 * s,
    dotRadius: 4 * s,
    dashed: true,
    label: troughLabel,
  });
  drawHipLine(ctx, pose, p.peak_frame, s, {
    color: COLOR_HIP_PEAK,
    alpha: 0.55,
    lineWidth: 2.5 * s,
    dotRadius: 4 * s,
    dashed: true,
    label: peakLabel,
  });

  // Current-frame hip line (purple, solid, full opacity) — drawn last so
  // it stays on top.
  drawHipLine(ctx, pose, f, s);

  // Predicted facing arrow (cyan) from hip midpoint — fixed across the window.
  const hip = hipMidAt(pose, f);
  if (hip && p.orientation_deg != null) {
    const imgAngle = (90 - p.orientation_deg) * Math.PI / 180;
    const len = 80 * s;
    const x1 = hip.x + len * Math.cos(imgAngle);
    const y1 = hip.y + len * Math.sin(imgAngle);
    ctx.save();
    ctx.strokeStyle = COLOR_PRED;
    ctx.fillStyle = COLOR_PRED;
    ctx.lineWidth = 3.5 * s;
    ctx.beginPath(); ctx.moveTo(hip.x, hip.y); ctx.lineTo(x1, y1); ctx.stroke();
    drawArrowhead(ctx, x1, y1, imgAngle, 14 * s);
    ctx.restore();
  }

  drawHud(ctx, p, s);
}

function drawHud(ctx, p, s) {
  const predCol = colorFor(p.predicted);
  const bandTag = p.band ? `[${p.band}]` : "[out]";
  const orientStatus = p.orientation_deg == null
    ? "orientation: —"
    : `orientation: ${p.orientation_deg.toFixed(0)}° ${bandTag}`;
  const orientCol = p.orientation_deg == null ? COLOR_UNCLEAR
                   : p.band === "inner" ? COLOR_PASS
                   : p.band === "outer" ? COLOR_SKIP
                   : COLOR_FAIL;
  const thrStr = p.threshold != null ? p.threshold.toFixed(2) : "—";
  const ratioMode = cfg.mode === "ratio";

  // Two info lines depend on mode: peak/trough display and the metric/threshold line.
  let peakTxt, metricTxt, metricCol;
  if (ratioMode) {
    const pTheta = Number.isFinite(p.peak_theta) ? `${p.peak_theta.toFixed(0)}°` : "—";
    const tTheta = Number.isFinite(p.trough_theta) ? `${p.trough_theta.toFixed(0)}°` : "—";
    const rotDeg = Number.isFinite(p.rotation_deg) ? `${p.rotation_deg.toFixed(1)}°` : "—";
    peakTxt = `peak θ ${pTheta}  ·  trough θ ${tTheta}`;
    metricTxt = `rotation ${rotDeg}  ·  thr ${cfg.minRotationDeg}°`;
    const noiseZone = Number.isFinite(p.trough_ratio) && p.trough_ratio > cfg.noiseRatioMin;
    metricCol = noiseZone
      ? COLOR_SKIP
      : (Number.isFinite(p.rotation_deg) && p.rotation_deg >= cfg.minRotationDeg
          ? COLOR_PASS : COLOR_FAIL);
  } else {
    peakTxt = `peak ${p.peak_gap.toFixed(3)}  ·  trough ${p.trough_gap.toFixed(3)}`;
    metricTxt = `range ${p.range.toFixed(3)}  ·  thr ${thrStr}`;
    metricCol = (p.threshold != null && p.range >= p.threshold) ? COLOR_PASS : COLOR_FAIL;
  }

  const titleTxt = `${p.hand} ${p.punch_type.replace(/_/g, " ")}  ·  ${p.stance || "?"}`;
  const predTxt  = `pred: ${p.predicted}`;
  const gtTxt    = p.label ? `GT: ${p.label}` : "GT: —";
  const agreeSym = p.label && p.predicted !== "skip" && p.predicted !== "unclear"
    ? (p.label === p.predicted ? "  ✓" : "  ✗") : "";

  const fontPx = 15 * s;
  const lineH  = 22 * s;
  const padX   = 14 * s;
  const padY   = 10 * s;
  const x0 = 24 * s, y0 = 24 * s;

  const lines = [titleTxt, orientStatus, peakTxt, metricTxt, predTxt, gtTxt + agreeSym];

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
  ctx.fillStyle = orientCol;                ctx.fillText(orientStatus, x0 + padX, y); y += lineH;
  ctx.fillStyle = "rgba(255,255,255,0.78)"; ctx.fillText(peakTxt, x0 + padX, y); y += lineH;
  ctx.fillStyle = metricCol;
  ctx.fillText(metricTxt, x0 + padX, y); y += lineH;
  ctx.fillStyle = predCol;                  ctx.fillText(predTxt, x0 + padX, y); y += lineH;
  ctx.fillStyle = p.label ? colorFor(p.label) : "rgba(255,255,255,0.55)";
  ctx.fillText(gtTxt, x0 + padX, y);
  if (agreeSym) {
    ctx.fillStyle = p.label === p.predicted ? COLOR_PASS : COLOR_FAIL;
    ctx.fillText(agreeSym, x0 + padX + ctx.measureText(gtTxt).width, y);
  }
  ctx.restore();
}

// ─── lens contract ────────────────────────────────────────────────────────

export const HipRotationReviewRule = {
  id: "hip_rotation_review",
  label: "Hip rotation review (loop + orientation)",

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
      document.removeEventListener("keydown", keydownHandler);
    }
    timeupdateHandler = null;
    keydownHandler = null;
    loopWindow = null;
    lastSnapshotKey = null;
  },
};
