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
export function detectStanceWidth(skeleton, conf, nFrames, fps, config = DEFAULT_CONFIG) {
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

  const sepRatios = new Array(nFrames).fill(NaN);
  for (let f = 0; f < nFrames; f++) {
    if (frameTorsoHeight(skeleton, f) > 1e-6) {
      sepRatios[f] = frameSepRatio(skeleton, f, lAnkle, rAnkle);
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

let host;
// Memoized round computation, keyed on the pose object + fps.
let cache = { pose: null, fps: 0, out: null };

// v6 cache is the canonical pose source (mirrors guard_drop.js pickPose).
function pickPose(state) {
  return state.poseV6 || state.pose;
}

function computeFor(state) {
  const pose = pickPose(state);
  if (!pose) return null;
  if (cache.pose !== pose || cache.fps !== state.fps) {
    cache = {
      pose,
      fps: state.fps,
      out: detectStanceWidth(pose.skeleton, pose.conf, pose.n_frames, state.fps),
    };
  }
  return cache.out;
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
      <div id="sw-clips"></div>
    `;
  },

  update(state) {
    if (!host) return;
    const out = computeFor(state);
    if (!out) {
      host.querySelector("#sw-round").innerHTML =
        `<p class="muted">No pose cache loaded.</p>`;
      return;
    }
    const { result: r, debug: d } = out;
    const f = state.frame;

    host.querySelector("#sw-round").innerHTML = `
      <h3 style="margin:10px 0 6px; font-size:14px">Round verdict
        <span style="display:inline-block; padding:1px 8px; border-radius:10px; background:${severityColor(r.severity)}; color:#000; font-size:11px; font-weight:700; text-transform:uppercase">${r.severity}</span>
      </h3>
      <div style="font-size:13px; line-height:1.6">
        violation ratio: <code>${(r.violation_ratio * 100).toFixed(2)}%</code>
        <span class="muted">(${r.violation_frames} / ${r.valid_frames} valid frames)</span><br>
        mean sep ratio: <code>${fmt(r.extras.mean_sep_ratio)}</code> ·
        median: <code>${fmt(r.extras.median_sep_ratio)}</code><br>
        <em style="color:#ccc">${r.coach_cue}</em>
      </div>
    `;

    const inValid = !!d.validMask[f];
    const inViolation = !!d.violationMask[f];
    const frameState = inViolation ? "VIOLATION" : inValid ? "valid (ok)" : "filtered out";
    const frameColor = inViolation ? COLOR_VIOLATION : inValid ? COLOR_VALID : COLOR_INVALID;
    host.querySelector("#sw-frame").innerHTML = `
      <strong>frame ${f}:</strong>
      <span style="color:${frameColor}; font-weight:600">${frameState}</span><br>
      sep ratio: <code>${fmt(d.sepRatios[f])}</code>
      <span class="muted">(threshold ${DEFAULT_CONFIG.narrowThreshold})</span>
    `;

    host.querySelector("#sw-clips").innerHTML = r.clips.length
      ? `<h3>${r.clips.length} violation clip${r.clips.length === 1 ? "" : "s"}</h3>
         <ul style="margin:4px 0 0 16px; padding:0; font-size:12px">
           ${r.clips.map(c => `<li>${c.start_time.toFixed(2)}s → ${c.end_time.toFixed(2)}s (mid ${c.timestamp.toFixed(2)}s)</li>`).join("")}
         </ul>`
      : `<h3>Violation clips</h3><p class="muted" style="font-size:12px">none</p>`;

    drawTrace(host.querySelector("#sw-trace"), out, f);
  },

  draw(ctx, state) {
    const out = computeFor(state);
    if (!out) return;
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
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 3 * s;
      ctx.beginPath();
      ctx.moveTo(lx, ly);
      ctx.lineTo(rx, ry);
      ctx.stroke();
      const ratio = out.debug.sepRatios[f];
      if (Number.isFinite(ratio)) {
        ctx.font = `${Math.round(12 * s)}px sans-serif`;
        ctx.fillText(ratio.toFixed(3), (lx + rx) / 2 + 8 * s, (ly + ry) / 2 - 8 * s);
      }
      ctx.restore();
    }

    drawStateStrip(ctx, state, out);
  },
};

// Bottom-of-canvas per-frame state strip (same idiom as ondevice_lens.js).
function drawStateStrip(ctx, state, out) {
  const { validMask, violationMask } = out.debug;
  const N = validMask.length;
  if (!N) return;
  const W = ctx.canvas.width;
  const H = ctx.canvas.height;
  const s = state.renderScale || 1;
  const stripH = 8 * s;
  const stripY = H - stripH - 4 * s;

  ctx.save();
  for (let f = 0; f < N; f++) {
    const x = (f / Math.max(1, N - 1)) * W;
    const w = Math.max(1, W / Math.max(1, N - 1));
    if (violationMask[f])  ctx.fillStyle = COLOR_VIOLATION;
    else if (validMask[f]) ctx.fillStyle = COLOR_VALID;
    else                   ctx.fillStyle = COLOR_INVALID;
    ctx.globalAlpha = 0.75;
    ctx.fillRect(x, stripY, w + 0.5, stripH);
  }
  const fx = (state.frame / Math.max(1, N - 1)) * W;
  ctx.globalAlpha = 1;
  ctx.strokeStyle = COLOR_FRAME_MARK;
  ctx.lineWidth = 2 * s;
  ctx.beginPath();
  ctx.moveTo(fx, stripY - 2 * s);
  ctx.lineTo(fx, stripY + stripH + 2 * s);
  ctx.stroke();
  ctx.restore();
}

// Sep-ratio sparkline over the full round: threshold line, valid/violation
// coloring, current-frame marker.
function drawTrace(canvas, out, frame) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const { sepRatios, validMask, violationMask } = out.debug;
  const N = sepRatios.length;
  if (!N) return;

  let maxR = DEFAULT_CONFIG.narrowThreshold * 1.5;
  for (const r of sepRatios) if (Number.isFinite(r) && r > maxR) maxR = r;
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
  // filtered-out gaps stay visible as gaps.
  for (let f = 0; f < N; f++) {
    const r = sepRatios[f];
    if (!Number.isFinite(r)) continue;
    ctx.fillStyle = violationMask[f] ? COLOR_VIOLATION
                  : validMask[f]     ? COLOR_VALID
                                     : "rgba(136,136,136,0.5)";
    ctx.fillRect(xOf(f) - 0.5, yOf(r) - 1, 2, 2);
  }

  // Current-frame marker.
  ctx.strokeStyle = COLOR_FRAME_MARK;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(xOf(frame), 0);
  ctx.lineTo(xOf(frame), H);
  ctx.stroke();
}
