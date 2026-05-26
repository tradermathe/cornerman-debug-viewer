// On-device analysis lens.
//
// Reads the per-frame analysis sidecar the iOS app uploaded
// (sessions/{id}/round_{N}_ondevice_analysis.json) and renders the exact
// numbers the phone produced — no recomputation. The viewer is a window
// into the on-device pipeline, not an independent implementation.
//
// Data source: state.analysis, populated by viewer.js's Firebase load path.
// Format documented in js/ondevice-loader.js.
//
// Visual idiom mirrors orientation_lens.js: cyan facing arrow drawn from
// the hip midpoint shows the current-frame angle. The sidebar shows the
// frame value + round stats. New vs the ankle-fit lens: this angle comes
// from the trained on-device LogReg, not from an ankle-direction fit.
// Stance-width state overlays as a thin strip at the bottom of the video.
//
// New rule ports drop in here as additional sidebar sections.

import { J } from "../skeleton.js";

const COLOR_PRED      = "#3ad9e0";  // cyan — on-device orientation arrow
const COLOR_VIOLATION = "#ff5d6c";
const COLOR_VALID     = "#7adf7a";
const COLOR_INVALID   = "#888";
const COLOR_FRAME_MARK = "#3ad9e0";
const MIN_HIP_CONF = 0.20;

const SPARKLINE_WINDOW = 60; // frames either side of current

let host;

function fmt(n, digits = 1) {
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

function renderSidebar(state) {
  if (!host) return;
  const el = host.querySelector("#ondev-state");
  if (!el) return;

  const a = state.analysis;
  if (!a) {
    el.innerHTML = `<span class="muted">No on-device analysis loaded — pick a Firebase session above.</span>`;
    return;
  }

  const f = state.frame;
  const N = a.n_frames;
  const fps = a.fps;

  // ── Orientation ─────────────────────────────────────────────────────
  const angle = a.orientation.angles[f];
  const conf  = a.orientation.confidences[f];
  const validFrames = a.orientation.validFrames;

  // Round-wide stats: circular mean + observed min/max so the user can see
  // the actual angle range, not just the mean.
  let sumCos = 0, sumSin = 0, sumConf = 0;
  let angMin = Infinity, angMax = -Infinity;
  for (let i = 0; i < N; i++) {
    const ai = a.orientation.angles[i];
    const ci = a.orientation.confidences[i];
    if (Number.isFinite(ai) && Number.isFinite(ci)) {
      const r = ai * Math.PI / 180;
      sumCos += ci * Math.cos(r);
      sumSin += ci * Math.sin(r);
      sumConf += ci;
      if (ai < angMin) angMin = ai;
      if (ai > angMax) angMax = ai;
    }
  }
  const meanAngle = sumConf > 0 ? Math.atan2(sumSin, sumCos) * 180 / Math.PI : NaN;
  const hasRange = Number.isFinite(angMin) && Number.isFinite(angMax);

  // Sparkline of angles around the current frame. Y-axis auto-fits to the
  // visible window's actual range (with a small min span so dead-still
  // rounds don't look like a hairline), padded by ~10% on each side.
  const startF = Math.max(0, f - SPARKLINE_WINDOW);
  const endF   = Math.min(N, f + SPARKLINE_WINDOW + 1);
  const sparkW = 240, sparkH = 60;

  let winMin = Infinity, winMax = -Infinity;
  for (let i = startF; i < endF; i++) {
    const ai = a.orientation.angles[i];
    if (Number.isFinite(ai)) {
      if (ai < winMin) winMin = ai;
      if (ai > winMax) winMax = ai;
    }
  }
  let yLo, yHi;
  if (!Number.isFinite(winMin)) { yLo = -5; yHi = 5; }
  else {
    const span = Math.max(10, winMax - winMin); // never zoom closer than ±5°
    const pad = span * 0.10;
    const mid = (winMax + winMin) / 2;
    yLo = mid - span / 2 - pad;
    yHi = mid + span / 2 + pad;
  }
  const yPx = (deg) => sparkH - ((deg - yLo) / (yHi - yLo)) * sparkH;

  const pts = [];
  for (let i = startF; i < endF; i++) {
    const ai = a.orientation.angles[i];
    if (!Number.isFinite(ai)) continue;
    const x = ((i - startF) / Math.max(1, endF - startF - 1)) * sparkW;
    pts.push(`${x.toFixed(1)},${yPx(ai).toFixed(1)}`);
  }
  const polyline = pts.length
    ? `<polyline points="${pts.join(" ")}" fill="none" stroke="#3ad9e0" stroke-width="1.5"/>`
    : "";
  const frameX = ((f - startF) / Math.max(1, endF - startF - 1)) * sparkW;
  // Zero line if it's inside the visible y-range.
  const zeroLine = (yLo <= 0 && yHi >= 0)
    ? `<line x1="0" y1="${yPx(0)}" x2="${sparkW}" y2="${yPx(0)}" stroke="#444" stroke-dasharray="2 4" />`
    : "";
  const sparkSvg = `
    <svg width="${sparkW}" height="${sparkH}" style="background:#1c1c20; border-radius:4px; margin-top:6px">
      ${zeroLine}
      ${polyline}
      <line x1="${frameX}" y1="0" x2="${frameX}" y2="${sparkH}" stroke="${COLOR_FRAME_MARK}" stroke-width="1" />
      <text x="2" y="10" fill="#666" font-size="9" font-family="monospace">${yHi.toFixed(0)}°</text>
      <text x="2" y="${sparkH - 2}" fill="#666" font-size="9" font-family="monospace">${yLo.toFixed(0)}°</text>
    </svg>`;

  // ── Stance width ─────────────────────────────────────────────────────
  const sw = a.rules.stance_width;
  let stanceHtml;
  if (sw) {
    const inValid = sw.validMask ? !!sw.validMask[f] : false;
    const inViolation = sw.violationMask ? !!sw.violationMask[f] : false;
    const sepRatio = sw.sepRatios ? sw.sepRatios[f] : NaN;
    const frameState = inViolation ? "VIOLATION"
                     : inValid     ? "valid (ok)"
                                   : "filtered out";
    const frameColor = inViolation ? COLOR_VIOLATION
                     : inValid     ? COLOR_VALID
                                   : COLOR_INVALID;

    const clipsHtml = sw.clips.length
      ? `<details><summary>${sw.clips.length} clip${sw.clips.length === 1 ? "" : "s"}</summary>
           <ul style="margin:4px 0 0 16px; padding:0; font-size:11px">
             ${sw.clips.map(c => `<li>${fmt(c.start_time, 2)}s → ${fmt(c.end_time, 2)}s (mid ${fmt(c.timestamp, 2)}s)</li>`).join("")}
           </ul></details>`
      : `<span class="muted">no clips</span>`;

    stanceHtml = `
      <h3 style="margin:18px 0 6px; font-size:14px">Stance width
        <span style="display:inline-block; padding:1px 8px; border-radius:10px; background:${severityColor(sw.severity)}; color:#000; font-size:11px; font-weight:700; text-transform:uppercase">${sw.severity}</span>
      </h3>
      <div style="font-size:13px; line-height:1.6">
        violation ratio: <code>${(sw.violationRatio * 100).toFixed(2)}%</code><br>
        <span class="muted">${sw.violationFrames} / ${sw.validFrames} valid frames flagged</span><br>
        <em style="color:#ccc">${sw.coachCue || ""}</em><br>
        <br>
        <strong>frame ${f}:</strong>
        <span style="color:${frameColor}; font-weight:600">${frameState}</span><br>
        sep ratio: <code>${fmt(sepRatio, 3)}</code><br>
        ${clipsHtml}
      </div>`;
  } else {
    stanceHtml = `<p class="muted">No stance_width rule in this sidecar.</p>`;
  }

  const rangeLine = hasRange
    ? `<br>round range: <code>${fmt(angMin, 1)}°</code> → <code>${fmt(angMax, 1)}°</code> <span class="muted">(span ${(angMax - angMin).toFixed(1)}°)</span>`
    : "";
  el.innerHTML = `
    <h3 style="margin:6px 0; font-size:14px">Orientation</h3>
    <div style="font-size:13px; line-height:1.6">
      <strong>frame ${f}:</strong> angle <code>${fmt(angle, 1)}°</code>, conf <code>${fmt(conf, 2)}</code><br>
      round mean: <code>${fmt(meanAngle, 1)}°</code>
      <span class="muted">(${validFrames} / ${N} valid frames, ${(100 * validFrames / Math.max(1, N)).toFixed(1)}%)</span>
      ${rangeLine}
      ${sparkSvg}
      <div style="display:flex; justify-content:space-between; font-size:10px; color:#888; margin-top:2px">
        <span>frame ${startF}</span>
        <span>frame ${endF - 1}</span>
      </div>
    </div>
    ${stanceHtml}
  `;
}

// Hip midpoint — same shape as orientation_lens.hipMid so the visual
// stays identical (arrow anchor + minimum-confidence gate).
function hipMid(pose, f) {
  const cLH = pose.conf[f * 17 + J.L_HIP];
  const cRH = pose.conf[f * 17 + J.R_HIP];
  if (cLH < MIN_HIP_CONF || cRH < MIN_HIP_CONF) return null;
  const lhx = pose.skeleton[(f * 17 + J.L_HIP) * 2];
  const lhy = pose.skeleton[(f * 17 + J.L_HIP) * 2 + 1];
  const rhx = pose.skeleton[(f * 17 + J.R_HIP) * 2];
  const rhy = pose.skeleton[(f * 17 + J.R_HIP) * 2 + 1];
  if (![lhx, lhy, rhx, rhy].every(Number.isFinite)) return null;
  return { x: 0.5 * (lhx + rhx), y: 0.5 * (lhy + rhy) };
}

// Filled arrowhead at (x, y) pointing along angle_rad. Lifted verbatim
// from orientation_lens.js so the visual matches.
function drawArrowhead(ctx, x, y, angle_rad, size) {
  const a = 0.45;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x - size * Math.cos(angle_rad - a), y - size * Math.sin(angle_rad - a));
  ctx.lineTo(x - size * Math.cos(angle_rad + a), y - size * Math.sin(angle_rad + a));
  ctx.closePath();
  ctx.fill();
}

// On-device orientation arrow from hip midpoint. Same coordinate convention
// as orientation_lens.js: image_angle = 90° − orientation_deg.
function drawOrientationArrow(ctx, state) {
  const a = state.analysis;
  if (!a) return;
  const f = state.frame;
  const angle = a.orientation.angles[f];
  if (!Number.isFinite(angle)) return;
  const hip = hipMid(state.pose, f);
  if (!hip) return;

  const s = state.renderScale || 1;
  const imgAngle = (90 - angle) * Math.PI / 180;
  const len = 70 * s;
  const x1 = hip.x + len * Math.cos(imgAngle);
  const y1 = hip.y + len * Math.sin(imgAngle);

  ctx.save();
  ctx.strokeStyle = COLOR_PRED;
  ctx.fillStyle   = COLOR_PRED;
  ctx.lineWidth = 3.5 * s;
  ctx.beginPath();
  ctx.moveTo(hip.x, hip.y);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  drawArrowhead(ctx, x1, y1, imgAngle, 14 * s);
  ctx.restore();
}

// Bottom-of-canvas per-frame state strip: one pixel column per ~10 frames.
function drawStateStrip(ctx, state) {
  const a = state.analysis;
  const sw = a?.rules?.stance_width;
  if (!sw || !sw.validMask || !sw.violationMask) return;

  const W = ctx.canvas.width;
  const H = ctx.canvas.height;
  const s = state.renderScale || 1;
  const stripH = 8 * s;
  const stripY = H - stripH - 4 * s;

  const N = a.n_frames;
  ctx.save();
  for (let f = 0; f < N; f++) {
    const x = (f / Math.max(1, N - 1)) * W;
    const w = Math.max(1, W / Math.max(1, N - 1));
    if (sw.violationMask[f])      ctx.fillStyle = COLOR_VIOLATION;
    else if (sw.validMask[f])     ctx.fillStyle = COLOR_VALID;
    else                           ctx.fillStyle = COLOR_INVALID;
    ctx.globalAlpha = 0.75;
    ctx.fillRect(x, stripY, w + 0.5, stripH);
  }

  // Current-frame indicator.
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

export const OnDeviceLensRule = {
  id: "ondevice",
  label: "On-device analysis (from phone)",

  mount(_host, state) {
    host = _host;
    host.innerHTML = `
      <h2>On-device analysis</h2>
      <p class="hint">
        Renders the exact per-frame numbers the iOS app produced —
        orientation classifier + ported rules. Pick a Firebase session
        above to load the sidecar; this lens is dumb display, no
        computation.
      </p>
      <div id="ondev-state" class="hint" style="line-height:1.55"></div>
      <p class="hint" style="margin-top:14px; font-size:11px">
        <span style="color:${COLOR_PRED}">●</span> on-device facing arrow (from hip midpoint).<br>
        Bottom-of-video strip:
        <span style="color:${COLOR_VIOLATION}">●</span> stance violation,
        <span style="color:${COLOR_VALID}">●</span> valid (no violation),
        <span style="color:${COLOR_INVALID}">●</span> filtered out.
      </p>
    `;
    renderSidebar(state);
  },

  update(state) {
    renderSidebar(state);
  },

  draw(ctx, state) {
    drawOrientationArrow(ctx, state);
    drawStateStrip(ctx, state);
  },
};
