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
// Today this shows:
//   • Orientation: current-frame angle + confidence, sparkline of recent
//     angles, round mean.
//   • Stance width: severity badge, violation ratio, coach cue, current
//     frame state (violation / valid-ok / filtered), per-frame sep ratio.
//   • Canvas strip at the bottom: per-frame state (red/green/gray).
//
// New rule ports drop in here as additional sidebar sections.

const COLOR_VIOLATION = "#ff5d6c";
const COLOR_VALID     = "#7adf7a";
const COLOR_INVALID   = "#888";
const COLOR_FRAME_MARK = "#3ad9e0";

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

  // Mean angle over the round (circular mean weighted by confidence).
  let sumCos = 0, sumSin = 0, sumConf = 0;
  for (let i = 0; i < N; i++) {
    const ai = a.orientation.angles[i];
    const ci = a.orientation.confidences[i];
    if (Number.isFinite(ai) && Number.isFinite(ci)) {
      const r = ai * Math.PI / 180;
      sumCos += ci * Math.cos(r);
      sumSin += ci * Math.sin(r);
      sumConf += ci;
    }
  }
  const meanAngle = sumConf > 0 ? Math.atan2(sumSin, sumCos) * 180 / Math.PI : NaN;

  // Sparkline of recent angles (-180..180 → 0..1 → y px).
  const startF = Math.max(0, f - SPARKLINE_WINDOW);
  const endF   = Math.min(N, f + SPARKLINE_WINDOW + 1);
  const sparkW = 240, sparkH = 60;
  const pts = [];
  for (let i = startF; i < endF; i++) {
    const ai = a.orientation.angles[i];
    if (!Number.isFinite(ai)) continue;
    const x = ((i - startF) / Math.max(1, endF - startF - 1)) * sparkW;
    const y = sparkH - ((ai + 180) / 360) * sparkH;
    pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  }
  const polyline = pts.length ? `<polyline points="${pts.join(" ")}" fill="none" stroke="#3ad9e0" stroke-width="1.5"/>` : "";
  const frameX = ((f - startF) / Math.max(1, endF - startF - 1)) * sparkW;
  const sparkSvg = `
    <svg width="${sparkW}" height="${sparkH}" style="background:#1c1c20; border-radius:4px; margin-top:6px">
      <line x1="0" y1="${sparkH/2}" x2="${sparkW}" y2="${sparkH/2}" stroke="#444" stroke-dasharray="2 4" />
      ${polyline}
      <line x1="${frameX}" y1="0" x2="${frameX}" y2="${sparkH}" stroke="${COLOR_FRAME_MARK}" stroke-width="1" />
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

  el.innerHTML = `
    <h3 style="margin:6px 0; font-size:14px">Orientation</h3>
    <div style="font-size:13px; line-height:1.6">
      <strong>frame ${f}:</strong> angle <code>${fmt(angle, 1)}°</code>, conf <code>${fmt(conf, 2)}</code><br>
      round mean: <code>${fmt(meanAngle, 1)}°</code>
      <span class="muted">(${validFrames} / ${N} valid frames, ${(100 * validFrames / Math.max(1, N)).toFixed(1)}%)</span>
      ${sparkSvg}
      <div style="display:flex; justify-content:space-between; font-size:10px; color:#888; margin-top:2px">
        <span>frame ${startF}</span>
        <span>frame ${endF - 1}</span>
      </div>
    </div>
    ${stanceHtml}
  `;
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
    drawStateStrip(ctx, state);
  },
};
