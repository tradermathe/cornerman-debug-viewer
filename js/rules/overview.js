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

  // Top-left punch label, mimicking the labeler's HUD.
  draw(ctx, state) {
    const ap = activePunch(state, state.frame);
    if (!ap) return;
    const s = state.renderScale || 1;
    drawPunchHud(ctx, ap, state, s);
  },
};

// Returns the punch (label or ST-GCN detection) that contains `frame` within
// its [start_frame, end_frame] window, or null. Labels win over punches when
// both contain the frame.
function activePunch(state, frame) {
  const src = pickSource(state);
  if (!src) return null;
  for (const d of src.detections) {
    if (frame >= d.start_frame && frame <= d.end_frame) {
      return { ...d, _source: src.kind };
    }
  }
  return null;
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

// Top-left HUD: big punch label like the labeler tool. Renders only inside a
// detection's window — falls back to invisible the rest of the time so it
// doesn't compete with other lenses' overlays.
function drawPunchHud(ctx, ap, state, scale) {
  const margin = 12 * scale;
  // Two stacked lines: big label + small meta (hand / source / labeler).
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
  ctx.font = `bold ${bigSize}px ui-monospace, "SF Mono", monospace`;
  const lw = ctx.measureText(labelText).width;
  ctx.font = `${smallSize}px ui-monospace, "SF Mono", monospace`;
  const mw = ctx.measureText(metaLine).width;
  const tw = ctx.measureText(tagLine).width;
  const boxW = Math.max(lw, mw, tw) + margin * 2;
  const boxH = bigSize + smallSize * 2 + margin * 1.4;

  // Background pill — use roundRect when available, plain rect otherwise.
  ctx.fillStyle = "rgba(0,0,0,0.72)";
  if (typeof ctx.roundRect === "function") {
    ctx.beginPath();
    ctx.roundRect(margin, margin, boxW, boxH, 8 * scale);
    ctx.fill();
  } else {
    ctx.fillRect(margin, margin, boxW, boxH);
  }
  // Left accent bar — green for GT, amber for ST-GCN.
  ctx.fillStyle = ap._source === "labels" ? "#5fd97a" : "#f5b945";
  ctx.fillRect(margin, margin, 4 * scale, boxH);

  // Big label.
  ctx.fillStyle = "#fff";
  ctx.font = `bold ${bigSize}px ui-monospace, "SF Mono", monospace`;
  ctx.textBaseline = "top";
  ctx.fillText(labelText, margin + margin, margin + margin * 0.4);

  // Meta line.
  if (metaLine) {
    ctx.fillStyle = "#e6e9ef";
    ctx.font = `${smallSize}px ui-monospace, "SF Mono", monospace`;
    ctx.fillText(metaLine, margin + margin, margin + margin * 0.4 + bigSize + 2);
  }

  // Source tag.
  ctx.fillStyle = "#8a93a3";
  ctx.font = `${smallSize}px ui-monospace, "SF Mono", monospace`;
  ctx.fillText(tagLine, margin + margin,
    margin + margin * 0.4 + bigSize + smallSize + 4);

  ctx.restore();
}
