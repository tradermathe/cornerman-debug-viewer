// Stance-width rule lens.
//
// Recomputes the stance_width rule in the browser from the loaded pose
// cache — unlike the on-device lens, which only displays the sidecar the
// phone uploaded. This is the workbench for iterating on the rule: today
// it is a line-for-line port of the shipped implementation
// (cornerman-ios StanceWidthRule.swift v4, itself parity-tested against
// cornerman_rules.rules.stance_width); tweaks/experiments go here first.
//
// Pipeline (identical to the app):
//   confidence gate → temporal cleanup → knee/ankle sanity →
//   sep_ratio (ankle separation / torso height) < narrow_threshold →
//   bridge short gaps → violation cleanup → counts, severity, clips.
//
// The compute core is exported standalone (plain arrays in, plain object
// out) so a node parity script can drive it against the Python rule.

import { J } from "../skeleton.js";

// Defaults match rules_config.json → rules.stance_width v4 and
// StanceWidthConfig.default in the Swift port.
export const DEFAULT_CONFIG = {
  narrowThreshold: 0.5,
  minConfidence: 0.5,
  // COCO-17 indices; ankles must be the last two entries.
  requiredJoints: [J.L_SHOULDER, J.R_SHOULDER, J.L_HIP, J.R_HIP, J.L_ANKLE, J.R_ANKLE],
  minValidSeconds: 0.17,
  minViolationSeconds: 0.10,
  bridgeGapSeconds: 0.30,
  severityThresholds: { mild: 0.03, moderate: 0.05, severe: 0.10 },
  coachCueNarrow: "Widen your stance, feet are too close together",
  coachCueGood: "Good stance width",
  clipPaddingSeconds: 0.5,
};

// ── rule core (ports of RuleFiltering / SkeletonGeometry / RuleClips) ──────
//
// All functions take `skeleton` as a flat (nFrames*17*2) array and `conf`
// as a flat (nFrames*17) array — the viewer's pose-cache layout.

function confidenceGate(conf, nFrames, joints, threshold) {
  const mask = new Array(nFrames);
  for (let f = 0; f < nFrames; f++) {
    let ok = true;
    for (const j of joints) {
      if (!(conf[f * 17 + j] > threshold)) { ok = false; break; }
    }
    mask[f] = ok;
  }
  return mask;
}

// Zero out any contiguous true-run shorter than minFrames. Shared by
// temporal cleanup (valid mask) and violation cleanup (violation mask).
function cleanupShortRuns(mask, minFrames) {
  if (minFrames <= 1) return mask.slice();
  const result = mask.slice();
  const n = result.length;
  let i = 0;
  while (i < n) {
    if (result[i]) {
      const runStart = i;
      while (i < n && result[i]) i++;
      if (i - runStart < minFrames) {
        for (let k = runStart; k < i; k++) result[k] = false;
      }
    } else {
      i++;
    }
  }
  return result;
}

// Fill false-gaps shorter than maxGapFrames that are bookended by true on
// both sides. Leading/trailing gaps are never filled.
function bridgeShortGaps(mask, maxGapFrames) {
  if (maxGapFrames <= 0) return mask.slice();
  const result = mask.slice();
  const n = result.length;
  let i = 0;
  while (i < n) {
    if (!result[i]) {
      const gapStart = i;
      while (i < n && !result[i]) i++;
      const gapEnd = i;
      if (gapStart > 0 && gapEnd < n && gapEnd - gapStart < maxGapFrames) {
        for (let k = gapStart; k < gapEnd; k++) result[k] = true;
      }
    } else {
      i++;
    }
  }
  return result;
}

// Invalidate frames where an ankle sits above its knee (top-left-origin
// image coords, so ankles should have LARGER y) — classic keypoint flip.
function kneeAnkleSanity(skeleton, validMask) {
  const result = validMask.slice();
  for (let f = 0; f < result.length; f++) {
    if (!result[f]) continue;
    const base = f * 17;
    const lKneeY  = skeleton[(base + J.L_KNEE)  * 2 + 1];
    const rKneeY  = skeleton[(base + J.R_KNEE)  * 2 + 1];
    const lAnkleY = skeleton[(base + J.L_ANKLE) * 2 + 1];
    const rAnkleY = skeleton[(base + J.R_ANKLE) * 2 + 1];
    if (lAnkleY < lKneeY || rAnkleY < rKneeY) result[f] = false;
  }
  return result;
}

function frameTorsoHeight(skeleton, f) {
  const base = f * 17;
  const sx = 0.5 * (skeleton[(base + J.L_SHOULDER) * 2]     + skeleton[(base + J.R_SHOULDER) * 2]);
  const sy = 0.5 * (skeleton[(base + J.L_SHOULDER) * 2 + 1] + skeleton[(base + J.R_SHOULDER) * 2 + 1]);
  const hx = 0.5 * (skeleton[(base + J.L_HIP) * 2]          + skeleton[(base + J.R_HIP) * 2]);
  const hy = 0.5 * (skeleton[(base + J.L_HIP) * 2 + 1]      + skeleton[(base + J.R_HIP) * 2 + 1]);
  return Math.hypot(sx - hx, sy - hy);
}

function frameSepRatio(skeleton, f, jointA, jointB) {
  const th = frameTorsoHeight(skeleton, f);
  if (th < 1e-6) return 0.0;
  const base = f * 17;
  const ax = skeleton[(base + jointA) * 2], ay = skeleton[(base + jointA) * 2 + 1];
  const bx = skeleton[(base + jointB) * 2], by = skeleton[(base + jointB) * 2 + 1];
  return Math.hypot(ax - bx, ay - by) / th;
}

function classifySeverity(ratio, t) {
  if (ratio >= t.severe)   return "severe";
  if (ratio >= t.moderate) return "moderate";
  if (ratio >= t.mild)     return "mild";
  return "none";
}

function maskToClips(violationMask, fps, paddingSeconds) {
  const n = violationMask.length;
  if (n === 0 || fps <= 0) return [];
  const videoDuration = n / fps;
  const round2 = x => Math.round(x * 100) / 100;
  const clips = [];
  let i = 0;
  while (i < n) {
    if (violationMask[i]) {
      const start = i;
      while (i < n && violationMask[i]) i++;
      const eventStart = start / fps;
      const eventEnd   = i / fps;
      clips.push({
        start_time: round2(Math.max(0, eventStart - paddingSeconds)),
        end_time:   round2(Math.min(videoDuration, eventEnd + paddingSeconds)),
        timestamp:  round2((eventStart + eventEnd) * 0.5),
      });
    } else {
      i++;
    }
  }
  return clips;
}

const round4 = x => Math.round(x * 10000) / 10000;

function medianSorted(xs) {
  const n = xs.length;
  if (n === 0) return 0;
  if (n % 2 === 1) return xs[(n - 1) / 2];
  return 0.5 * (xs[n / 2 - 1] + xs[n / 2]);
}

// Full rule. Returns { result, debug } shaped like the Swift detectDebug:
// result mirrors RuleResult, debug carries the per-frame intermediates.
//
// `sepOverride` (experiment hook): per-frame sep-ratio array used instead of
// the computed one — everything else (masks, counting, clips) is identical.
// Omitted ⇒ exact port of the shipped rule.
export function detectStanceWidth(skeleton, conf, nFrames, fps, config = DEFAULT_CONFIG, sepOverride = null) {
  const minValidFrames     = Math.max(1, Math.trunc(config.minValidSeconds     * fps));
  const minViolationFrames = Math.max(1, Math.trunc(config.minViolationSeconds * fps));
  const bridgeGapFrames    = Math.max(0, Math.trunc(config.bridgeGapSeconds    * fps));

  if (nFrames === 0) {
    return {
      result: emptyResult("No skeleton data to analyze", 0),
      debug: { validMask: [], violationMask: [], sepRatios: [] },
    };
  }

  const confMask     = confidenceGate(conf, nFrames, config.requiredJoints, config.minConfidence);
  const temporalMask = cleanupShortRuns(confMask, minValidFrames);
  const validMask    = kneeAnkleSanity(skeleton, temporalMask);

  let validCount = 0;
  for (const v of validMask) if (v) validCount++;

  const lAnkle = config.requiredJoints[config.requiredJoints.length - 2];
  const rAnkle = config.requiredJoints[config.requiredJoints.length - 1];

  let sepRatios;
  if (sepOverride) {
    sepRatios = sepOverride.slice();
  } else {
    sepRatios = new Array(nFrames).fill(NaN);
    for (let f = 0; f < nFrames; f++) {
      if (frameTorsoHeight(skeleton, f) > 1e-6) {
        sepRatios[f] = frameSepRatio(skeleton, f, lAnkle, rAnkle);
      }
    }
  }

  if (validCount < minValidFrames) {
    return {
      result: emptyResult("Insufficient confident pose data to analyze stance", validCount),
      debug: { validMask, violationMask: new Array(nFrames).fill(false), sepRatios },
    };
  }

  const violationRaw = new Array(nFrames).fill(false);
  for (let f = 0; f < nFrames; f++) {
    if (validMask[f] && Number.isFinite(sepRatios[f]) && sepRatios[f] < config.narrowThreshold) {
      violationRaw[f] = true;
    }
  }

  const violationBridged = bridgeShortGaps(violationRaw, bridgeGapFrames);
  const violationFinal   = cleanupShortRuns(violationBridged, minViolationFrames);

  let narrowCount = 0;
  for (let f = 0; f < nFrames; f++) {
    if (validMask[f] && violationFinal[f]) narrowCount++;
  }
  const violationRatio = narrowCount / validCount;
  const severity = classifySeverity(violationRatio, config.severityThresholds);

  const validRatios = [];
  for (let f = 0; f < nFrames; f++) {
    if (validMask[f] && Number.isFinite(sepRatios[f])) validRatios.push(sepRatios[f]);
  }
  const meanSepRatio = validRatios.length
    ? validRatios.reduce((a, b) => a + b, 0) / validRatios.length : 0;
  const medianSepRatio = medianSorted(validRatios.slice().sort((a, b) => a - b));

  return {
    result: {
      rule_id: "stance_width",
      version: "4",
      violation_ratio: round4(violationRatio),
      severity,
      coach_cue: severity === "none" ? config.coachCueGood : config.coachCueNarrow,
      valid_frames: validCount,
      violation_frames: narrowCount,
      clips: maskToClips(violationFinal, fps, config.clipPaddingSeconds),
      extras: {
        mean_sep_ratio:   round4(meanSepRatio),
        median_sep_ratio: round4(medianSepRatio),
        narrow_count: narrowCount,
        ok_count: validCount - narrowCount,
      },
    },
    debug: { validMask, violationMask: violationFinal, sepRatios },
  };
}

function emptyResult(cue, validFrames) {
  return {
    rule_id: "stance_width",
    version: "4",
    violation_ratio: 0.0,
    severity: "none",
    coach_cue: cue,
    valid_frames: validFrames,
    violation_frames: 0,
    clips: [],
    extras: {},
  };
}

// ── lens UI ────────────────────────────────────────────────────────────────

const COLOR_VIOLATION  = "#ff5d6c";
const COLOR_VALID      = "#7adf7a";
const COLOR_INVALID    = "#888";
const COLOR_FRAME_MARK = "#3ad9e0";
const COLOR_DX         = "#7ec8ff";  // horizontal ankle component
const COLOR_DY         = "#ffd95c";  // vertical ankle component (depth proxy)
const COLOR_CORRECTED  = "#e08aff";  // foreshortening-corrected sep / variant

// Foreshortening-correction experiment: when the smoothed Δy/Δx ratio says
// the stance line is depth-aligned, the measured sep underestimates the true
// width — boost it before the rule logic runs. Ratio is smoothed with a
// rolling MEDIAN so single-frame spikes (heel raises, steps) can't flip the
// gate; only sustained depth-alignment does.
const CORR = {
  smoothSeconds: 0.5,  // half-window of the rolling median
  // 0.75 = knee of the label-eval sweep: fixes 26/54 foreshortened false
  // flags while erasing only 2/27 genuine side-on flags. (2.0 was near-zero
  // collateral but only fixed 13/54.)
  ratioGate: 0.75,     // smoothed Δy/Δx above this ⇒ depth-aligned
  boost: 1.5,          // multiply sep by this when gated (+50%)
  ratioCap: 99,        // dx≈0 ⇒ ratio explodes; cap keeps the median sane
  minWindowValid: 3,   // need at least this many finite ratios in the window
};

let host;
// Memoized round computation, keyed on the pose object + fps.
let cache = { pose: null, fps: 0, out: null };

// v6 cache is the canonical pose source (mirrors guard_drop.js pickPose).
function pickPose(state) {
  return state.poseV6 || state.pose;
}

// Lens-side diagnostic, NOT part of the ported rule: split the ankle
// separation into its horizontal/vertical image components (torso-
// normalized, same scale as sep ratio). With the camera above ankle
// height, depth maps to image-vertical — so a vertical-leaning ankle
// line suggests the stance is depth-aligned and sep ratio undercounts.
function computeDxDy(pose) {
  const n = pose.n_frames;
  const dx = new Array(n).fill(NaN);
  const dy = new Array(n).fill(NaN);
  for (let f = 0; f < n; f++) {
    const th = frameTorsoHeight(pose.skeleton, f);
    if (!(th > 1e-6)) continue;
    const base = f * 17;
    dx[f] = Math.abs(pose.skeleton[(base + J.L_ANKLE) * 2]     - pose.skeleton[(base + J.R_ANKLE) * 2])     / th;
    dy[f] = Math.abs(pose.skeleton[(base + J.L_ANKLE) * 2 + 1] - pose.skeleton[(base + J.R_ANKLE) * 2 + 1]) / th;
  }
  return { dx, dy };
}

// NaN-aware rolling median; windows with too few finite values stay NaN.
function rollingMedian(xs, halfWin, minValid) {
  const n = xs.length;
  const out = new Array(n).fill(NaN);
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - halfWin), hi = Math.min(n - 1, i + halfWin);
    const vals = [];
    for (let k = lo; k <= hi; k++) {
      if (Number.isFinite(xs[k])) vals.push(xs[k]);
    }
    if (vals.length < minValid) continue;
    vals.sort((a, b) => a - b);
    out[i] = vals.length % 2 ? vals[(vals.length - 1) / 2]
                             : 0.5 * (vals[vals.length / 2 - 1] + vals[vals.length / 2]);
  }
  return out;
}

function computeFor(state) {
  const pose = pickPose(state);
  if (!pose) return null;
  if (cache.pose !== pose || cache.fps !== state.fps) {
    const out = detectStanceWidth(pose.skeleton, pose.conf, pose.n_frames, state.fps);
    const dxdy = computeDxDy(pose);

    const rawRatio = dxdy.dx.map((dxv, f) =>
      Number.isFinite(dxv) && Number.isFinite(dxdy.dy[f])
        ? Math.min(dxdy.dy[f] / Math.max(dxv, 1e-6), CORR.ratioCap)
        : NaN);
    const halfWin = Math.max(1, Math.round(CORR.smoothSeconds * state.fps));
    const smoothRatio = rollingMedian(rawRatio, halfWin, CORR.minWindowValid);

    const sepCorr = out.debug.sepRatios.map((r, f) =>
      Number.isFinite(r) && smoothRatio[f] > CORR.ratioGate ? r * CORR.boost : r);
    const outCorr = detectStanceWidth(
      pose.skeleton, pose.conf, pose.n_frames, state.fps, DEFAULT_CONFIG, sepCorr);

    cache = { pose, fps: state.fps, out, outCorr, dxdy, smoothRatio, sepCorr };
  }
  return cache;
}

function fmt(n, digits = 3) {
  return Number.isFinite(n) ? n.toFixed(digits) : "—";
}

function severityColor(sev) {
  switch (sev) {
    case "severe":   return "#ff5d6c";
    case "moderate": return "#ffa64a";
    case "mild":     return "#ffd24a";
    case "none":     return "#7adf7a";
    default:         return "#888";
  }
}

export const StanceWidthLensRule = {
  id: "stance_width_lens",
  label: "Stance width (rule workbench)",

  skeletonStyle() {
    return {
      boneColor: "rgba(255,255,255,0.25)",
      boneWidth: 1.5,
      jointRadius: 3,
      highlightJoints: new Set([
        J.L_SHOULDER, J.R_SHOULDER, J.L_HIP, J.R_HIP,
        J.L_KNEE, J.R_KNEE, J.L_ANKLE, J.R_ANKLE,
      ]),
    };
  },

  mount(_host) {
    host = _host;
    cache = { pose: null, fps: 0, out: null };
    host.innerHTML = `
      <h2>Stance width</h2>
      <p class="hint">
        Recomputes the shipped stance_width rule (v4, exact port of the
        iOS implementation) from the loaded pose cache. Ankle separation /
        torso height &lt; ${DEFAULT_CONFIG.narrowThreshold} on a valid frame
        ⇒ violation. Bottom strip:
        <span style="color:${COLOR_VALID}">valid</span> ·
        <span style="color:${COLOR_VIOLATION}">violation</span> ·
        <span style="color:${COLOR_INVALID}">filtered</span>.
      </p>
      <div id="sw-round"></div>
      <h3>Current frame</h3>
      <div id="sw-frame" style="font-size:13px; line-height:1.6"></div>
      <h3>Sep ratio over time</h3>
      <canvas id="sw-trace" width="320" height="110"></canvas>
      <h3>Ankle <span style="color:${COLOR_DX}">Δx</span> / <span style="color:${COLOR_DY}">Δy</span> over time</h3>
      <p class="hint" style="margin-top:0">Torso-normalized image components of the
        ankle separation. Δy ≈ depth proxy: when the stance line points at the
        camera, Δx collapses and Δy carries the (foreshortened) width.</p>
      <canvas id="sw-dxdy" width="320" height="110"></canvas>
      <div id="sw-clips"></div>
    `;
  },

  update(state) {
    if (!host) return;
    const c = computeFor(state);
    if (!c) {
      host.querySelector("#sw-round").innerHTML =
        `<p class="muted">No pose cache loaded.</p>`;
      return;
    }
    const { result: r, debug: d } = c.out;
    const rc = c.outCorr.result;
    const f = state.frame;

    const verdictCol = (title, res, color) => `
      <div style="flex:1">
        <div style="font-size:11px; color:${color}; font-weight:700; text-transform:uppercase">${title}</div>
        <span style="display:inline-block; margin:3px 0; padding:1px 8px; border-radius:10px; background:${severityColor(res.severity)}; color:#000; font-size:11px; font-weight:700; text-transform:uppercase">${res.severity}</span><br>
        <code>${(res.violation_ratio * 100).toFixed(2)}%</code>
        <span class="muted">(${res.violation_frames} / ${res.valid_frames})</span><br>
        <span class="muted" style="font-size:11px">${res.clips.length} clip${res.clips.length === 1 ? "" : "s"} ·
        med sep <code>${fmt(res.extras.median_sep_ratio, 2)}</code></span>
      </div>`;

    host.querySelector("#sw-round").innerHTML = `
      <h3 style="margin:10px 0 6px; font-size:14px">Round verdict</h3>
      <div style="display:flex; gap:12px; font-size:13px; line-height:1.5">
        ${verdictCol("stock (app)", r, "#aaa")}
        ${verdictCol("corrected", rc, COLOR_CORRECTED)}
      </div>
      <p class="hint" style="margin:6px 0 0">corrected = sep ×${CORR.boost} where the
        rolling-median (±${CORR.smoothSeconds}s) Δy/Δx exceeds ${CORR.ratioGate}.</p>
    `;

    const inValid = !!d.validMask[f];
    const inViolation = !!d.violationMask[f];
    const frameState = inViolation ? "VIOLATION" : inValid ? "valid (ok)" : "filtered out";
    const frameColor = inViolation ? COLOR_VIOLATION : inValid ? COLOR_VALID : COLOR_INVALID;
    const corrViolation = !!c.outCorr.debug.violationMask[f];
    const corrState = corrViolation ? "VIOLATION" : inValid ? "valid (ok)" : "filtered out";
    const corrColor = corrViolation ? COLOR_VIOLATION : inValid ? COLOR_VALID : COLOR_INVALID;
    const boosted = c.sepCorr[f] !== d.sepRatios[f]
      && Number.isFinite(c.sepCorr[f]);
    const dx = c.dxdy.dx[f], dy = c.dxdy.dy[f];
    const tilt = Math.atan2(dy, dx) * 180 / Math.PI;
    host.querySelector("#sw-frame").innerHTML = `
      <strong>frame ${f}:</strong>
      stock <span style="color:${frameColor}; font-weight:600">${frameState}</span> ·
      corrected <span style="color:${corrColor}; font-weight:600">${corrState}</span><br>
      sep: <code>${fmt(d.sepRatios[f])}</code>
      → corrected: <code style="color:${boosted ? COLOR_CORRECTED : "inherit"}">${fmt(c.sepCorr[f])}</code>
      <span class="muted">(threshold ${DEFAULT_CONFIG.narrowThreshold})</span><br>
      <span style="color:${COLOR_DX}">Δx: <code>${fmt(dx)}</code></span> ·
      <span style="color:${COLOR_DY}">Δy: <code>${fmt(dy)}</code></span> ·
      Δy/Δx: <code>${fmt(dy / dx, 2)}</code> ·
      smoothed: <code>${fmt(c.smoothRatio[f], 2)}</code> ·
      tilt: <code>${fmt(tilt, 1)}°</code>
    `;

    const clipList = (title, clips, color) => clips.length
      ? `<h3 style="color:${color}">${clips.length} clip${clips.length === 1 ? "" : "s"} — ${title}</h3>
         <ul style="margin:4px 0 0 16px; padding:0; font-size:12px">
           ${clips.map(cl => `<li>${cl.start_time.toFixed(2)}s → ${cl.end_time.toFixed(2)}s (mid ${cl.timestamp.toFixed(2)}s)</li>`).join("")}
         </ul>`
      : `<h3 style="color:${color}">0 clips — ${title}</h3>`;
    host.querySelector("#sw-clips").innerHTML =
      clipList("stock", r.clips, "#aaa") + clipList("corrected", rc.clips, COLOR_CORRECTED);

    drawTrace(host.querySelector("#sw-trace"), c, f);
    drawDxDyTrace(host.querySelector("#sw-dxdy"), c, f);
  },

  draw(ctx, state) {
    const c = computeFor(state);
    if (!c) return;
    const out = c.out;
    const f = state.frame;
    const pose = pickPose(state);
    const s = state.renderScale || 1;

    // Ankle-to-ankle line, colored by the frame's rule state.
    const base = f * 17;
    const lx = pose.skeleton[(base + J.L_ANKLE) * 2], ly = pose.skeleton[(base + J.L_ANKLE) * 2 + 1];
    const rx = pose.skeleton[(base + J.R_ANKLE) * 2], ry = pose.skeleton[(base + J.R_ANKLE) * 2 + 1];
    if ([lx, ly, rx, ry].every(Number.isFinite)) {
      const color = out.debug.violationMask[f] ? COLOR_VIOLATION
                  : out.debug.validMask[f]     ? COLOR_VALID
                                               : COLOR_INVALID;
      ctx.save();
      // Dashed right-triangle legs: the horizontal (Δx) and vertical (Δy)
      // image components of the ankle separation, color-matched to the
      // Δx/Δy trace in the sidebar.
      ctx.lineWidth = 2 * s;
      ctx.setLineDash([4 * s, 4 * s]);
      ctx.strokeStyle = COLOR_DX;
      ctx.beginPath();
      ctx.moveTo(lx, ly);
      ctx.lineTo(rx, ly);
      ctx.stroke();
      ctx.strokeStyle = COLOR_DY;
      ctx.beginPath();
      ctx.moveTo(rx, ly);
      ctx.lineTo(rx, ry);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 3 * s;
      ctx.beginPath();
      ctx.moveTo(lx, ly);
      ctx.lineTo(rx, ry);
      ctx.stroke();
      ctx.restore();
    }

    // Numbers live in a corner HUD, not around the feet — the overlay there
    // stays just the ankle line + dashed component legs.
    {
      const dx = c.dxdy.dx[f], dy = c.dxdy.dy[f];
      const sep = out.debug.sepRatios[f];
      const tilt = Math.atan2(dy, dx) * 180 / Math.PI;
      const sepColor = out.debug.violationMask[f] ? COLOR_VIOLATION
                     : out.debug.validMask[f]     ? COLOR_VALID
                                                  : COLOR_INVALID;
      const sepC = c.sepCorr[f];
      const smooth = c.smoothRatio[f];
      const fsz = Math.round(13 * s);
      const lineH = fsz + 4 * s;
      const lines = [
        [`sep   ${Number.isFinite(sep) ? sep.toFixed(3) : "—"}`, sepColor],
        [`sep*  ${Number.isFinite(sepC) ? sepC.toFixed(3) : "—"}`, COLOR_CORRECTED],
        [`Δx    ${Number.isFinite(dx) ? dx.toFixed(3) : "—"}`, COLOR_DX],
        [`Δy    ${Number.isFinite(dy) ? dy.toFixed(3) : "—"}`, COLOR_DY],
        [`Δy/Δx ${Number.isFinite(dy / dx) ? (dy / dx).toFixed(2) : "—"}`, "#fff"],
        [`r̄     ${Number.isFinite(smooth) ? smooth.toFixed(2) : "—"}`, "#fff"],
        [`tilt  ${Number.isFinite(tilt) ? tilt.toFixed(1) + "°" : "—"}`, "#fff"],
      ];
      const padX = 10 * s, padY = 8 * s;
      const boxW = 110 * s;
      const boxH = lines.length * lineH + padY * 2 - 4 * s;
      const bx = ctx.canvas.width - boxW - 10 * s;
      const by = 10 * s;

      ctx.save();
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      ctx.beginPath();
      ctx.roundRect(bx, by, boxW, boxH, 6 * s);
      ctx.fill();
      ctx.font = `${fsz}px ui-monospace, monospace`;
      ctx.textBaseline = "top";
      lines.forEach(([text, color], i) => {
        ctx.fillStyle = color;
        ctx.fillText(text, bx + padX, by + padY + i * lineH);
      });
      ctx.restore();
    }

    drawStateStrip(ctx, state, c);
  },
};

// Bottom-of-canvas per-frame state strips (same idiom as ondevice_lens.js):
// stock rule on the bottom, corrected variant stacked just above it.
function drawStateStrip(ctx, state, c) {
  const { validMask, violationMask } = c.out.debug;
  const corrViolation = c.outCorr.debug.violationMask;
  const N = validMask.length;
  if (!N) return;
  const W = ctx.canvas.width;
  const H = ctx.canvas.height;
  const s = state.renderScale || 1;
  const stripH = 8 * s;
  const stockY = H - stripH - 4 * s;
  const corrY  = stockY - stripH - 2 * s;

  ctx.save();
  for (let f = 0; f < N; f++) {
    const x = (f / Math.max(1, N - 1)) * W;
    const w = Math.max(1, W / Math.max(1, N - 1));
    ctx.globalAlpha = 0.75;
    if (violationMask[f])  ctx.fillStyle = COLOR_VIOLATION;
    else if (validMask[f]) ctx.fillStyle = COLOR_VALID;
    else                   ctx.fillStyle = COLOR_INVALID;
    ctx.fillRect(x, stockY, w + 0.5, stripH);
    if (corrViolation[f])  ctx.fillStyle = COLOR_VIOLATION;
    else if (validMask[f]) ctx.fillStyle = COLOR_VALID;
    else                   ctx.fillStyle = COLOR_INVALID;
    ctx.fillRect(x, corrY, w + 0.5, stripH);
  }
  // Tiny labels so the two strips stay identifiable.
  ctx.globalAlpha = 1;
  ctx.font = `${Math.round(9 * s)}px sans-serif`;
  ctx.fillStyle = "rgba(255,255,255,0.8)";
  ctx.fillText("corr",  4 * s, corrY  + stripH - 1.5 * s);
  ctx.fillText("stock", 4 * s, stockY + stripH - 1.5 * s);

  const fx = (state.frame / Math.max(1, N - 1)) * W;
  ctx.strokeStyle = COLOR_FRAME_MARK;
  ctx.lineWidth = 2 * s;
  ctx.beginPath();
  ctx.moveTo(fx, corrY - 2 * s);
  ctx.lineTo(fx, stockY + stripH + 2 * s);
  ctx.stroke();
  ctx.restore();
}

// Sep-ratio sparkline over the full round: threshold line, valid/violation
// coloring, corrected-sep overlay on boosted frames, current-frame marker.
function drawTrace(canvas, c, frame) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const { sepRatios, validMask, violationMask } = c.out.debug;
  const N = sepRatios.length;
  if (!N) return;

  let maxR = DEFAULT_CONFIG.narrowThreshold * 1.5;
  for (const r of sepRatios) if (Number.isFinite(r) && r > maxR) maxR = r;
  for (const r of c.sepCorr) if (Number.isFinite(r) && r > maxR) maxR = r;
  const yOf = r => H - 4 - (r / maxR) * (H - 12);
  const xOf = f => (f / Math.max(1, N - 1)) * W;

  // Threshold line.
  ctx.strokeStyle = "rgba(255,255,255,0.35)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(0, yOf(DEFAULT_CONFIG.narrowThreshold));
  ctx.lineTo(W, yOf(DEFAULT_CONFIG.narrowThreshold));
  ctx.stroke();
  ctx.setLineDash([]);

  // Per-frame dots, colored by state. Dots instead of a polyline so the
  // filtered-out gaps stay visible as gaps. Boosted frames get a second
  // dot at the corrected value so the lift is visible as a parallel band.
  for (let f = 0; f < N; f++) {
    const r = sepRatios[f];
    if (!Number.isFinite(r)) continue;
    ctx.fillStyle = violationMask[f] ? COLOR_VIOLATION
                  : validMask[f]     ? COLOR_VALID
                                     : "rgba(136,136,136,0.5)";
    ctx.fillRect(xOf(f) - 0.5, yOf(r) - 1, 2, 2);
    if (c.sepCorr[f] !== r && Number.isFinite(c.sepCorr[f])) {
      ctx.fillStyle = COLOR_CORRECTED;
      ctx.fillRect(xOf(f) - 0.5, yOf(c.sepCorr[f]) - 1, 2, 2);
    }
  }

  // Current-frame marker.
  ctx.strokeStyle = COLOR_FRAME_MARK;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(xOf(frame), 0);
  ctx.lineTo(xOf(frame), H);
  ctx.stroke();
}

// Δx / Δy components over the full round, same x-axis and y-scale idiom as
// the sep-ratio trace so dips line up visually between the two canvases.
function drawDxDyTrace(canvas, c, frame) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const { dx, dy } = c.dxdy;
  const N = dx.length;
  if (!N) return;

  let maxV = DEFAULT_CONFIG.narrowThreshold * 1.5;
  for (let f = 0; f < N; f++) {
    if (Number.isFinite(dx[f]) && dx[f] > maxV) maxV = dx[f];
    if (Number.isFinite(dy[f]) && dy[f] > maxV) maxV = dy[f];
  }
  const yOf = v => H - 4 - (v / maxV) * (H - 12);
  const xOf = f => (f / Math.max(1, N - 1)) * W;

  for (let f = 0; f < N; f++) {
    if (Number.isFinite(dx[f])) {
      ctx.fillStyle = COLOR_DX;
      ctx.fillRect(xOf(f) - 0.5, yOf(dx[f]) - 1, 2, 2);
    }
    if (Number.isFinite(dy[f])) {
      ctx.fillStyle = COLOR_DY;
      ctx.fillRect(xOf(f) - 0.5, yOf(dy[f]) - 1, 2, 2);
    }
  }

  ctx.strokeStyle = COLOR_FRAME_MARK;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(xOf(frame), 0);
  ctx.lineTo(xOf(frame), H);
  ctx.stroke();
}
