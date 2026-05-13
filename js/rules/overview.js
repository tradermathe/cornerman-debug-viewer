// Overview panel — no rule lens, just shows the raw per-joint state at the
// current frame. Useful when you want to eyeball Apple Vision's confidence
// distribution before picking a specific rule.
//
// Also paints a labeler-style punch label in the top-left of the video when
// the current frame falls inside a labelled (or model-detected) punch window.
// Source order: state.labels (ground truth) > state.punches (ST-GCN) > none.

import { JOINT_NAMES, confColor } from "../skeleton.js";

let host;

export const OverviewRule = {
  id: "overview",
  label: "Overview (no lens)",

  mount(_host, state) {
    host = _host;
    host.innerHTML = `
      <h2>Per-joint state</h2>
      <div id="ov-source" class="hint" style="margin-bottom:8px"></div>
      <p class="hint">Confidence is colour-coded: green ≥ 0.5, amber ≥ 0.2, red below.
      A zero means Apple Vision didn't detect that joint at all (different from
      YOLO, which usually returns a low-conf guess).</p>
      <table class="joint-table">
        <thead><tr><th>#</th><th>Joint</th><th>x</th><th>y</th><th>conf</th></tr></thead>
        <tbody id="joint-tbody"></tbody>
      </table>
    `;
    renderSourceLine(state);
  },

  update(state) {
    const tbody = host.querySelector("#joint-tbody");
    const rows = [];
    for (let j = 0; j < 17; j++) {
      const x = state.pose.skeleton[(state.frame * 17 + j) * 2];
      const y = state.pose.skeleton[(state.frame * 17 + j) * 2 + 1];
      const c = state.pose.conf[state.frame * 17 + j];
      rows.push(
        `<tr>
          <td class="muted">${j}</td>
          <td>${JOINT_NAMES[j]}</td>
          <td class="num">${x.toFixed(0)}</td>
          <td class="num">${y.toFixed(0)}</td>
          <td class="num" style="color:${confColor(c)}">${c.toFixed(2)}</td>
        </tr>`
      );
    }
    tbody.innerHTML = rows.join("");
    renderSourceLine(state);
  },

  // Top-left punch label, mimicking the labeler's HUD. Stacks one block per
  // active punch when multiple windows overlap (fast combos).
  draw(ctx, state) {
    const aps = activePunches(state, state.frame);
    if (aps.length === 0) return;
    const s = state.renderScale || 1;
    drawPunchHudStack(ctx, aps, state, s);
  },
};

// Returns every punch (label or ST-GCN detection) whose [start_frame,
// end_frame] window contains `frame`. Labels win over punches when both
// exist — we never mix sources in the same stack to avoid double-labelling.
// Ordered by start_frame so the earliest-starting punch is on top.
function activePunches(state, frame) {
  const src = pickSource(state);
  if (!src) return [];
  const hits = [];
  for (const d of src.detections) {
    if (frame >= d.start_frame && frame <= d.end_frame) {
      hits.push({ ...d, _source: src.kind });
    }
  }
  hits.sort((a, b) => a.start_frame - b.start_frame);
  return hits;
}

function pickSource(state) {
  if (state.labels?.detections?.length) {
    return { detections: state.labels.detections, kind: "labels", meta: state.labels };
  }
  if (state.punches?.detections?.length) {
    return { detections: state.punches.detections, kind: "punches", meta: state.punches };
  }
  return null;
}

function renderSourceLine(state) {
  const el = host?.querySelector("#ov-source");
  if (!el) return;
  const src = pickSource(state);
  // Surface label-fetch errors even when no labels are picked, so the user
  // knows WHY the lens isn't using GT.
  const labelErr = state.labels?.error;
  if (!src) {
    const errHtml = labelErr
      ? `<span class="muted">Labels: <span class="bad">${labelErr}</span> for <code>${state.labels.cacheBasename || "?"}</code>.</span><br>`
      : "";
    el.innerHTML = errHtml +
      `<span class="muted">No punches loaded — drop a <code>*_punches.json</code> next to the cache for ST-GCN detections.</span>`;
    return;
  }
  if (src.kind === "labels") {
    const time = new Date(src.meta.fetched_at).toLocaleTimeString();
    const cached = src.meta.from_cache ? " (cached)" : "";
    const conf = src.meta.match_confidence || "?";
    el.innerHTML =
      `<span class="role-lead">Ground truth</span> · ${src.detections.length} labels · ` +
      `live @ ${time}${cached} · ` +
      `auto-matched (${conf}) → <code>${src.meta.source_video}</code>`;
  } else {
    const errLine = labelErr
      ? `<br><span class="muted">Labels: <span class="bad">${labelErr}</span> — using ST-GCN.</span>`
      : "";
    el.innerHTML =
      `<span class="role-rear">ST-GCN punches</span> · ${src.detections.length} detected` + errLine;
  }
}

// Top-left HUD: big punch labels like the labeler tool. Stacks one block per
// active punch vertically when multiple windows overlap (fast combos / paired
// hand events). Only drawn when at least one detection's window contains the
// current frame.
function drawPunchHudStack(ctx, aps, state, scale) {
  const margin = 12 * scale;
  const gap    = 6 * scale;     // gap between stacked blocks
  let cursorY  = margin;
  for (const ap of aps) {
    const h = drawPunchHudBlock(ctx, ap, margin, cursorY, scale);
    cursorY += h + gap;
  }
}

// Draws a single block at (x, y) and returns its height so the stacker can
// place the next one below it.
function drawPunchHudBlock(ctx, ap, x, y, scale) {
  const padX = 12 * scale;
  const padY = 8 * scale;
  const bigSize = Math.round(20 * scale);
  const smallSize = Math.round(11 * scale);

  const labelText = (ap.punch_type || "?").replace(/_/g, " ").toUpperCase();
  const handText = ap.hand ? ap.hand.toUpperCase() : null;
  const stanceText = ap.stance ? ap.stance.toUpperCase() : null;
  const sourceTag = ap._source === "labels" ? "GT" : "ST-GCN";
  const labelerTag = ap.labeler ? ` · ${ap.labeler}` : "";
  const metaLine = [handText, stanceText].filter(Boolean).join(" · ");
  const tagLine = `${sourceTag}${labelerTag}`;

  ctx.save();
  // Measure widths to size the rounded background.
  ctx.font = `bold ${bigSize}px ui-monospace, "SF Mono", monospace`;
  const lw = ctx.measureText(labelText).width;
  ctx.font = `${smallSize}px ui-monospace, "SF Mono", monospace`;
  const mw = metaLine ? ctx.measureText(metaLine).width : 0;
  const tw = ctx.measureText(tagLine).width;
  const innerW = Math.max(lw, mw, tw);
  const lineGap = 2 * scale;
  const innerH = bigSize + (metaLine ? smallSize + lineGap : 0) + smallSize + lineGap;
  const boxW = innerW + padX * 2;
  const boxH = innerH + padY * 2;

  // Background pill — roundRect when available, plain rect otherwise.
  ctx.fillStyle = "rgba(0,0,0,0.72)";
  if (typeof ctx.roundRect === "function") {
    ctx.beginPath();
    ctx.roundRect(x, y, boxW, boxH, 8 * scale);
    ctx.fill();
  } else {
    ctx.fillRect(x, y, boxW, boxH);
  }
  // Left accent bar — green for GT, amber for ST-GCN.
  ctx.fillStyle = ap._source === "labels" ? "#5fd97a" : "#f5b945";
  ctx.fillRect(x, y, 4 * scale, boxH);

  // Big label.
  let cursorY = y + padY;
  ctx.fillStyle = "#fff";
  ctx.font = `bold ${bigSize}px ui-monospace, "SF Mono", monospace`;
  ctx.textBaseline = "top";
  ctx.fillText(labelText, x + padX, cursorY);
  cursorY += bigSize + lineGap;

  // Meta line.
  if (metaLine) {
    ctx.fillStyle = "#e6e9ef";
    ctx.font = `${smallSize}px ui-monospace, "SF Mono", monospace`;
    ctx.fillText(metaLine, x + padX, cursorY);
    cursorY += smallSize + lineGap;
  }

  // Source tag.
  ctx.fillStyle = "#8a93a3";
  ctx.font = `${smallSize}px ui-monospace, "SF Mono", monospace`;
  ctx.fillText(tagLine, x + padX, cursorY);

  ctx.restore();
  return boxH;
}
