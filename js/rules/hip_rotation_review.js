// Hip rotation review — walk through every labelled rotation-applicable punch
// (cross/hook/uppercut, head/body), apply the orientation gate, and review the
// gap-delta verdict side by side with the canvas overlay. Mirrors
// straights_review.js: one punch at a time, video loops within [start, end],
// N/P keys (or buttons) step to next/prev.
//
// Why the orientation gate matters here: the rule's signal is
//   gap[f] = |L_hip − R_hip| / torso_height
// When the boxer is too frontal/back-on (|facing| close to 0° or ±180°), the
// hip line collapses on the image and rotation creates depth-only motion that
// doesn't lift the gap. So we predict "skip" outside the sideways band — same
// gate the arm-extension / straights review uses, same defaults.
//
// Per-punch compute:
//   * gap[f] smoothed with moving-average over cfg.gapSmoothSeconds
//   * baseline = median(gap) across the whole round
//   * search window = [start − search_window, end + search_window]
//   * peak_gap   = max(gap[search])
//   * delta      = peak_gap − baseline
//   * predicted  = skip if orientation gate fails
//                  pass if delta ≥ min_delta
//                  fail otherwise

import { J, torsoHeight } from "../skeleton.js";
import { STANCE_FITS } from "./orientation_lens.js";

const DEFAULTS = {
  gapSmoothSeconds:       0.083,
  searchWindowSec:        0.4,
  minDelta:               0.05,
  orientationGate:        true,
  orientationMinAbsDeg:   60,
  orientationMaxAbsDeg:   150,
  // Rule only applies to punches with rotation expectation (jab + body shots
  // excluded — matches hip_rotation.js).
  appliesTo: new Set([
    "cross_head", "cross_body",
    "lead_hook_head", "lead_uppercut_head",
    "rear_uppercut_head", "rear_hook_head",
  ]),
};

const MIN_ANKLE_CONF = 0.30;
const COLOR_HIP     = "#a78bfa";  // purple — hip line
const COLOR_PRED    = "#3ad9e0";  // cyan   — predicted facing
const COLOR_PASS    = "#5fd97a";
const COLOR_FAIL    = "#e85a5a";
const COLOR_SKIP    = "#7ec8ff";
const COLOR_UNCLEAR = "#f5b945";

let host = null;
let videoEl = null;
let timeupdateHandler = null;
let keydownHandler = null;
let loopWindow = null;
let activeIdx = -1;
let punches = [];
let signals = null;           // {gap, baseline, fps}
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

function median(arr) {
  const s = Array.from(arr).sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
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
  const baseline = median(gap);

  const detections = (state.labels?.detections || [])
    .filter(d => cfg.appliesTo.has(d.punch_type));

  const searchFrames = Math.max(2, Math.round(cfg.searchWindowSec * fps));

  const out = detections.map(d => {
    const sf = Math.max(0, d.start_frame);
    const ef = Math.min(N - 1, d.end_frame);
    const ss = Math.max(0, sf - searchFrames);
    const se = Math.min(N - 1, ef + searchFrames);

    let peak = gap[ss], peakAt = ss, trough = gap[ss];
    for (let f = ss; f <= se; f++) {
      const g = gap[f];
      if (g > peak)   { peak = g; peakAt = f; }
      if (g < trough)  trough = g;
    }
    const delta = peak - baseline;

    // Orientation gate — window-median predicted facing from ankle arrow.
    const stance = d.stance?.toLowerCase?.() || null;
    const orientationDeg = medianPredictedFacing(pose, sf, ef, stance);
    const orientationSideways = orientationDeg != null && (() => {
      const a = Math.abs(orientationDeg);
      return a >= cfg.orientationMinAbsDeg && a <= cfg.orientationMaxAbsDeg;
    })();

    let predicted;
    if (cfg.orientationGate && orientationDeg == null) {
      predicted = "skip";
    } else if (cfg.orientationGate && !orientationSideways) {
      predicted = "skip";
    } else if (delta >= cfg.minDelta) {
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
      trough_gap: trough,
      range_gap: peak - trough,
      delta,
      orientation_deg: orientationDeg,
      orientation_sideways: orientationSideways,
      predicted,
      label,
    };
  });

  return { gap, baseline, punches: out, fps };
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
  rebuildSidebar(state);
}

function seekToPunch(idx, state) {
  if (!punches.length) return;
  if (idx < 0) idx = 0;
  if (idx >= punches.length) idx = punches.length - 1;
  const p = punches[idx];
  activeIdx = idx;
  loopWindow = { start_frame: p.start_frame, end_frame: p.end_frame };
  if (videoEl && state.fps) {
    videoEl.currentTime = (state.start_sec || 0) + p.start_frame / state.fps;
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
  };
  document.addEventListener("keydown", keydownHandler);
}

// ─── sidebar ──────────────────────────────────────────────────────────────

function buildSidebarSkeleton() {
  if (!host) return;
  host.innerHTML = `
    <h2>Hip rotation review</h2>
    <p class="hint">
      Walk through every cross / hook / uppercut. Each punch first checks the
      <b>orientation gate</b> (fighter must be sideways,
      |angle| in <code>[${cfg.orientationMinAbsDeg}°, ${cfg.orientationMaxAbsDeg}°]</code>) —
      too frontal and the hip line collapses on screen, so the rule predicts
      <i>skip</i>. Otherwise scores <code>peak_gap − baseline ≥ min_delta</code>.
    </p>
    <div class="ol-nav" style="display:flex; gap:8px; align-items:center; margin:10px 0 14px;">
      <button id="hrr-prev" class="orient-btn-action secondary" style="padding:6px 10px;">⏮ prev (P)</button>
      <button id="hrr-next" class="orient-btn-action secondary" style="padding:6px 10px;">next (N) ⏭</button>
      <span id="hrr-counter" style="margin-left:6px; color:#888; font-size:12px;"></span>
    </div>
    <div id="hrr-state" class="hint" style="line-height:1.7;"></div>
    <div id="hrr-summary" class="hint" style="margin-top:14px; padding-top:10px; border-top:1px solid #2a2a2a;"></div>
    <p class="hint" style="margin-top:14px; font-size:11px;">
      Loops within the punch window. Baseline / threshold come from the
      hip-rotation defaults; tweak in <code>hip_rotation_review.js</code>.
    </p>
  `;
  host.querySelector("#hrr-prev")?.addEventListener("click",
    () => seekToPunch(activeIdx - 1, latestState));
  host.querySelector("#hrr-next")?.addEventListener("click",
    () => seekToPunch(activeIdx + 1, latestState));
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
        + `· ${agreeStr}`
        + ` · baseline <code>${signals.baseline.toFixed(3)}</code>`;
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
    const status = p.orientation_sideways
      ? pill("SIDEWAYS ✓", COLOR_PASS)
      : pill("NOT SIDEWAYS ✗", COLOR_FAIL);
    orientLine = `<span style="color:${COLOR_PRED}">orientation:</span> `
      + `<code>${oDeg.toFixed(1)}°</code> · |a|=<code>${a.toFixed(1)}°</code> `
      + `· gate [${cfg.orientationMinAbsDeg}°, ${cfg.orientationMaxAbsDeg}°] · ${status}`;
  }

  // 2) Rotation data
  const deltaCol = (p.delta >= cfg.minDelta) ? COLOR_PASS : COLOR_FAIL;
  const deltaSign = p.delta >= 0 ? "+" : "";

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
    `frames <code>${p.start_frame}-${p.end_frame}</code> (${p.end_frame - p.start_frame + 1} frames) · peak@<code>${p.land_frame}</code>`,
    "",
    `<b>1. Orientation gate</b>`,
    orientLine,
    "",
    `<b>2. Hip rotation signal</b>`,
    `peak gap <code>${p.peak_gap.toFixed(3)}</code> · trough <code>${p.trough_gap.toFixed(3)}</code> · range <code>${p.range_gap.toFixed(3)}</code>`,
    `baseline <code>${signals.baseline.toFixed(3)}</code> (round median)`,
    `Δ = <code style="color:${deltaCol}">${deltaSign}${p.delta.toFixed(3)}</code> (threshold <code>${cfg.minDelta.toFixed(2)}</code>)`,
    "",
    `<b>3. Verdict</b>`,
    verdictLine,
  ];
  el.innerHTML = lines.join("<br>");
}

// ─── draw ─────────────────────────────────────────────────────────────────

function drawHipLine(ctx, pose, frame, scale) {
  const lc = pose.conf[frame * 17 + J.L_HIP];
  const rc = pose.conf[frame * 17 + J.R_HIP];
  if (lc < 0.05 || rc < 0.05) return;
  const lx = pose.skeleton[(frame * 17 + J.L_HIP) * 2];
  const ly = pose.skeleton[(frame * 17 + J.L_HIP) * 2 + 1];
  const rx = pose.skeleton[(frame * 17 + J.R_HIP) * 2];
  const ry = pose.skeleton[(frame * 17 + J.R_HIP) * 2 + 1];
  ctx.save();
  ctx.strokeStyle = COLOR_HIP;
  ctx.lineWidth = 3 * scale;
  ctx.globalAlpha = 0.85;
  ctx.beginPath();
  ctx.moveTo(lx, ly); ctx.lineTo(rx, ry);
  ctx.stroke();
  ctx.fillStyle = COLOR_HIP;
  ctx.beginPath(); ctx.arc(lx, ly, 5 * scale, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(rx, ry, 5 * scale, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}

function drawCanvas(ctx, state) {
  if (!punches.length || activeIdx < 0) return;
  const p = punches[activeIdx];
  const f = state.frame;
  const s = state.renderScale || 1;
  const pose = state.pose;

  // Hip line (purple) — the rule's input signal.
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
  const orientStatus = p.orientation_deg == null
    ? "orientation: —"
    : (p.orientation_sideways
        ? `orientation: ${p.orientation_deg.toFixed(0)}° sideways ✓`
        : `orientation: ${p.orientation_deg.toFixed(0)}° not sideways ✗`);
  const orientCol = p.orientation_deg == null ? COLOR_UNCLEAR
                   : (p.orientation_sideways ? COLOR_PASS : COLOR_FAIL);
  const deltaSign = p.delta >= 0 ? "+" : "";
  const deltaTxt = `Δ gap ${deltaSign}${p.delta.toFixed(3)}  ·  thr ${cfg.minDelta.toFixed(2)}`;
  const peakTxt = `peak ${p.peak_gap.toFixed(3)}  ·  base ${signals.baseline.toFixed(3)}`;
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

  const lines = [titleTxt, orientStatus, peakTxt, deltaTxt, predTxt, gtTxt + agreeSym];

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
  ctx.fillStyle = p.delta >= cfg.minDelta ? COLOR_PASS : COLOR_FAIL;
  ctx.fillText(deltaTxt, x0 + padX, y); y += lineH;
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
    // rotation-applicable punch the cursor is now inside. Same 15-frame
    // tolerance straights_review uses to avoid flip-flop with the loop snap.
    const ACTIVE_TOLERANCE_FRAMES = 15;
    if (punches.length) {
      const f = state.frame;
      const active = activeIdx >= 0 ? punches[activeIdx] : null;
      const nearActive = active &&
        f >= active.start_frame - ACTIVE_TOLERANCE_FRAMES &&
        f <= active.end_frame   + ACTIVE_TOLERANCE_FRAMES;
      if (!nearActive) {
        const inside = punches.findIndex(p =>
          f >= p.start_frame && f <= p.end_frame);
        if (inside !== -1 && inside !== activeIdx) {
          activeIdx = inside;
          const p = punches[inside];
          loopWindow = { start_frame: p.start_frame, end_frame: p.end_frame };
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
  },
};
