// Overview panel — no rule lens, just shows the raw per-joint state at the
// current frame. Useful when you want to eyeball Apple Vision's confidence
// distribution before picking a specific rule.
//
// Also paints a labeler-style punch label in the top-left of the video when
// the current frame falls inside a labelled (or model-detected) punch window.
// Source order: state.labels (ground truth) > state.punches (ST-GCN) > none.

import { JOINT_NAMES, confColor } from "../skeleton.js";

let host;

// ── Skeleton/engine picker so the per-joint table can show any loaded engine
//    (Vision / YOLO / RTMPose / v6), not just the primary. ──
const ENGINE_LABEL = {
  apple_vision_2d: "Vision", yolo_pose: "YOLO", rtmpose_body17: "RTMPose",
  apple_vision_2d_combined: "v6 (combined)",
};
function engLabel(p) {
  const e = (p && p.engine) || "";
  return ENGINE_LABEL[e] || (e.startsWith("apple_vision_2d+glove") ? "v6" : (e || "pose"));
}
function ovSources(state) {
  const seen = new Set(), out = [];
  const add = (p, suffix) => {
    if (!p || seen.has(p)) return;
    seen.add(p); out.push({ pose: p, label: engLabel(p) + (suffix || "") });
  };
  add(state.pose); add(state.poseSecondary); add(state.poseRtm);
  add(state.poseV6, " v6"); add(state.poseCombined, " comb");
  return out;
}
let ovSel = null;   // selected source label (persists across rounds)
// Map the viewer's current frame to the selected pose's frame by ABSOLUTE pts
// (same alignment engine_compare uses); falls back to the start_sec+fps model.
function ovFrame(p, state) {
  if (p === state.pose) return state.frame;
  const ap = state.pose.pts, bp = p.pts, f = state.frame;
  if (ap && bp && bp.length && f < ap.length && ap[f] === ap[f]) {
    const t = ap[f]; let lo = 0, hi = bp.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (bp[m] < t) lo = m + 1; else hi = m; }
    let best = lo;
    if (lo > 0 && Math.abs(bp[lo - 1] - t) <= Math.abs(bp[lo] - t)) best = lo - 1;
    return best;
  }
  const t = (state.pose.start_sec || 0) + f / state.pose.fps;
  const sf = Math.round((t - (p.start_sec || 0)) * p.fps);
  return (sf >= 0 && sf < p.n_frames) ? sf : 0;
}

export const OverviewRule = {
  id: "overview",
  label: "Overview (no lens)",

  mount(_host, state) {
    host = _host;
    const srcs = ovSources(state);
    if (!srcs.find(s => s.label === ovSel)) ovSel = srcs[0]?.label || null;
    const opts = srcs.map(s =>
      `<option value="${s.label}"${s.label === ovSel ? " selected" : ""}>${s.label}</option>`).join("");
    host.innerHTML = `
      <h2>Per-joint state</h2>
      <label class="hint" style="display:block;margin-bottom:6px">Skeleton:
        <select id="ov-engine" style="background:#1c1c1c;color:#eee;border:1px solid #444;border-radius:4px;padding:1px 5px">${opts || "<option>—</option>"}</select></label>
      <div id="ov-source" class="hint" style="margin-bottom:8px"></div>
      <p class="hint">Confidence is colour-coded: green ≥ 0.5, amber ≥ 0.2, red below.
      A zero means the engine didn't detect that joint (Vision emits 0; YOLO/RTMPose
      usually return a low-conf guess). Non-primary engines are mapped to this frame by PTS.</p>
      <table class="joint-table">
        <thead><tr><th>#</th><th>Joint</th><th>x</th><th>y</th><th>conf</th></tr></thead>
        <tbody id="joint-tbody"></tbody>
      </table>
    `;
    const sel = host.querySelector("#ov-engine");
    if (sel) sel.addEventListener("change", () => { ovSel = sel.value; this.update(state); });
    renderSourceLine(state);
  },

  update(state) {
    const srcs = ovSources(state);
    const src = srcs.find(s => s.label === ovSel) || srcs[0];
    const tbody = host.querySelector("#joint-tbody");
    if (!src) { if (tbody) tbody.innerHTML = ""; renderSourceLine(state); return; }
    const pose = src.pose, fr = ovFrame(pose, state);
    const rows = [];
    for (let j = 0; j < 17; j++) {
      const x = pose.skeleton[(fr * 17 + j) * 2];
      const y = pose.skeleton[(fr * 17 + j) * 2 + 1];
      const c = pose.conf[fr * 17 + j];
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
//
// Exported so other lenses (e.g. hip_rotation) can paint the same
// labeler-style HUD without duplicating logic.
export function activePunches(state, frame) {
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
export function drawPunchHudStack(ctx, aps, state, scale) {
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
