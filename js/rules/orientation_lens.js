// Orientation lens — visualises the ankle-arrow → facing-direction hypothesis.
//
// Draws THREE things over the video:
//
//   1. Raw ankle line  (amber)  — the segment between both ankles, extended
//      faintly across the frame so the axis is readable even when feet are
//      close. Back→front arrowhead by stance (orthodox = left foot front).
//
//   2. PREDICTED facing arrow  (cyan)  — computed as
//          predicted_gt = sign * arrow_image_angle + offset
//      with per-stance (sign, offset) fitted by boxing_ai/orientation_model/
//      06_ankle_arrow.py against existing orientation labels. Only drawn when
//      the current frame falls inside a labelled punch (stance is taken from
//      that punch — that's the only place we know orthodox vs southpaw).
//      Drawn from hip midpoint.
//
//   3. GT orientation arrow  (red)  — the labelled facing direction for this
//      exact (round, frame) if a row exists in the Orientation Labels sheet.
//      Drawn from hip midpoint, dashed so it visually distinguishes from the
//      prediction even when they overlap.
//
// Coordinate convention: GT label angle convention is "rotation from
// facing-camera, positive = toward viewer's right". In image space (y down),
// that maps to `image_angle = 90° - gt`. The arrow we draw at GT or PRED
// points in that image direction.

import { J } from "../skeleton.js";
import { fetchOrientationForStem } from "../sheet-labels.js";

// Per-stance fit from 06_ankle_arrow.py (2026-05-21, n=183, MAE 41° orthodox /
// 13° southpaw). Treat these as starting values — refit when label count grows.
const STANCE_FITS = {
  orthodox: { sign: -1, offset_deg: 58.8 },
  southpaw: { sign: -1, offset_deg: 100.6 },
};

const MIN_ANKLE_CONF = 0.30;
const COLOR_RAW   = "#ffd24a";  // amber — raw ankle line
const COLOR_PRED  = "#3ad9e0";  // cyan  — predicted facing direction
const COLOR_GT    = "#ff5d6c";  // red   — ground-truth facing label

let host;

// Per-video orientation-label cache so we don't refetch on every redraw.
// Keyed by cacheBasename (= labeler's video stem); value is the Map returned
// by fetchOrientationForStem.byKey.
let lastFetchedStem = null;
let labelMap = null;       // Map<`${round}:${frame}`, {label, labeler, ts}>
let fetchError = null;
let fetchInfo = "";        // human-readable summary for the sidebar

function wrap180(deg) {
  return ((deg + 180) % 360 + 360) % 360 - 180;
}

// Returns the active punch (if any) at the current frame, picking the
// earliest by start_frame when fast combos overlap. Stance is stable within
// a round, so the first hit gives the right stance regardless.
function activePunch(state) {
  const dets = state.labels?.detections;
  if (!Array.isArray(dets) || dets.length === 0) return null;
  const f = state.frame;
  let best = null;
  for (const d of dets) {
    if (f < d.start_frame || f > d.end_frame) continue;
    if (best == null || d.start_frame < best.start_frame) best = d;
  }
  return best;
}

function ankleArrow(pose, f, stance) {
  const cL = pose.conf[f * 17 + J.L_ANKLE];
  const cR = pose.conf[f * 17 + J.R_ANKLE];
  if (cL < MIN_ANKLE_CONF || cR < MIN_ANKLE_CONF) return null;
  const lx = pose.skeleton[(f * 17 + J.L_ANKLE) * 2];
  const ly = pose.skeleton[(f * 17 + J.L_ANKLE) * 2 + 1];
  const rx = pose.skeleton[(f * 17 + J.R_ANKLE) * 2];
  const ry = pose.skeleton[(f * 17 + J.R_ANKLE) * 2 + 1];
  if (![lx, ly, rx, ry].every(Number.isFinite)) return null;
  // Front foot by stance. When stance is unknown we still want to draw the
  // raw line; arbitrarily treat L as "front" so the arrow has a direction.
  const orthodoxFront = stance !== "southpaw";
  const fx = orthodoxFront ? lx : rx;
  const fy = orthodoxFront ? ly : ry;
  const bx = orthodoxFront ? rx : lx;
  const by = orthodoxFront ? ry : ly;
  const dx = fx - bx, dy = fy - by;
  if (dx * dx + dy * dy < 1e-6) return null;
  return { lx, ly, rx, ry, fx, fy, bx, by, dx, dy,
           angle_deg: Math.atan2(dy, dx) * 180 / Math.PI };
}

function hipMid(pose, f) {
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

// Draw a line that extends past both endpoints to the canvas edges.
function drawExtendedLine(ctx, p0, p1, W, H) {
  const dx = p1.x - p0.x, dy = p1.y - p0.y;
  if (Math.abs(dx) < 1e-6 && Math.abs(dy) < 1e-6) return;
  let x0, y0, x1, y1;
  if (Math.abs(dx) >= Math.abs(dy)) {
    const slope = dy / dx;
    x0 = 0; y0 = p0.y + (x0 - p0.x) * slope;
    x1 = W; y1 = p0.y + (x1 - p0.x) * slope;
  } else {
    const slope = dx / dy;
    y0 = 0; x0 = p0.x + (y0 - p0.y) * slope;
    y1 = H; x1 = p0.x + (y1 - p0.y) * slope;
  }
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
}

// Filled arrowhead at (x, y) pointing along angle_rad.
function drawArrowhead(ctx, x, y, angle_rad, size) {
  const a = 0.45;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x - size * Math.cos(angle_rad - a), y - size * Math.sin(angle_rad - a));
  ctx.lineTo(x - size * Math.cos(angle_rad + a), y - size * Math.sin(angle_rad + a));
  ctx.closePath();
  ctx.fill();
}

// Fire-and-forget fetch when cache identity changes. Module-level state keeps
// the result around between mount/update calls without re-hitting the sheet.
function refreshLabelsIfNeeded(state) {
  const stem = state.cacheBasename;
  if (!stem || stem === lastFetchedStem) return;
  lastFetchedStem = stem;
  labelMap = null; fetchError = null;
  fetchInfo = `fetching orientation labels for "${stem}"…`;
  renderSidebar(state);
  fetchOrientationForStem(stem).then(res => {
    if (state.cacheBasename !== stem) return;  // stale (video switched)
    if (res.error) { fetchError = res.error; labelMap = new Map(); }
    else {
      labelMap = res.byKey;
      fetchInfo = `${labelMap.size} labels in this video (${res.countForVideo} rows incl. skips).`;
    }
    renderSidebar(state);
  });
}

function gtForCurrentFrame(state) {
  if (!labelMap || state.cacheRound == null) return null;
  const entry = labelMap.get(`${state.cacheRound}:${state.frame}`);
  return entry || null;
}

function renderSidebar(state) {
  if (!host) return;
  const el = host.querySelector("#ol-state");
  if (!el) return;

  const f = state.frame;
  const cL = state.pose.conf[f * 17 + J.L_ANKLE];
  const cR = state.pose.conf[f * 17 + J.R_ANKLE];
  const punch = activePunch(state);
  const stance = punch?.stance || null;
  const fit = stance ? STANCE_FITS[stance.toLowerCase?.()] : null;
  const arr = ankleArrow(state.pose, f, stance);

  const lines = [];
  if (cL < MIN_ANKLE_CONF || cR < MIN_ANKLE_CONF) {
    lines.push(`<span class="muted">Ankles below ${MIN_ANKLE_CONF.toFixed(2)} conf (L=${cL.toFixed(2)}, R=${cR.toFixed(2)}) — nothing drawn.</span>`);
  } else {
    lines.push(`L_ankle conf <code>${cL.toFixed(2)}</code> · R_ankle conf <code>${cR.toFixed(2)}</code>`);
  }
  if (arr) lines.push(`ankle arrow (image): <code>${arr.angle_deg.toFixed(1)}°</code>`);

  if (punch) {
    const pred = (fit && arr)
      ? wrap180(fit.sign * arr.angle_deg + fit.offset_deg)
      : null;
    lines.push(`active punch: <code>${punch.punch_type || "?"}</code> · stance <code>${punch.stance || "?"}</code>`);
    if (pred != null) {
      lines.push(`<span style="color:${COLOR_PRED}">predicted facing:</span> <code>${pred.toFixed(1)}°</code> &nbsp;(sign ${fit.sign}, offset ${fit.offset_deg.toFixed(1)}°)`);
    } else if (!fit) {
      lines.push(`<span class="muted">no fit for stance "${punch.stance}"</span>`);
    }
  } else {
    lines.push(`<span class="muted">not in a labelled punch window — no stance, no prediction.</span>`);
  }

  const gt = gtForCurrentFrame(state);
  if (gt) {
    lines.push(`<span style="color:${COLOR_GT}">GT facing label:</span> <code>${gt.label.toFixed(1)}°</code> &nbsp;by ${gt.labeler || "?"}`);
    if (punch && arr && fit) {
      const pred = wrap180(fit.sign * arr.angle_deg + fit.offset_deg);
      const err = Math.abs(wrap180(pred - gt.label));
      lines.push(`error: <code>${err.toFixed(1)}°</code>`);
    }
  } else if (labelMap) {
    lines.push(`<span class="muted">no GT label at (round ${state.cacheRound}, frame ${state.frame}).</span>`);
  }

  if (fetchError) lines.push(`<span class="bad">label fetch error: ${fetchError}</span>`);
  else if (fetchInfo) lines.push(`<span class="muted">${fetchInfo}</span>`);

  el.innerHTML = lines.join("<br>");
}

export const OrientationLensRule = {
  id: "orientation_lens",
  label: "Orientation lens (ankle arrow)",

  mount(_host, state) {
    host = _host;
    host.innerHTML = `
      <h2>Orientation lens</h2>
      <p class="hint">
        Three overlays:
        <span style="color:${COLOR_RAW}">●</span> raw ankle line (back→front by stance),
        <span style="color:${COLOR_PRED}">●</span> predicted facing (only inside a labelled punch),
        <span style="color:${COLOR_GT}">●</span> GT label (only when this frame is in the Orientation Labels sheet).
      </p>
      <div id="ol-state" class="hint" style="line-height:1.55"></div>
      <p class="hint" style="margin-top:14px; font-size:11px;">
        Per-stance fit baked from <code>06_ankle_arrow.py</code> 2026-05-21:
        orthodox sign −1 / offset +58.8°, southpaw sign −1 / offset +100.6°.
      </p>
    `;
    refreshLabelsIfNeeded(state);
    renderSidebar(state);
  },

  update(state) {
    refreshLabelsIfNeeded(state);
    renderSidebar(state);
  },

  draw(ctx, state) {
    const W = ctx.canvas.width;
    const H = ctx.canvas.height;
    const s = state.renderScale || 1;
    const f = state.frame;

    const punch = activePunch(state);
    const stance = punch?.stance?.toLowerCase?.() || null;
    const fit = stance ? STANCE_FITS[stance] : null;
    const arr = ankleArrow(state.pose, f, stance);
    if (!arr) return;

    ctx.save();

    // (1) Raw ankle line — extended faintly + solid segment + arrowhead at front.
    ctx.strokeStyle = COLOR_RAW;
    ctx.globalAlpha = 0.35;
    ctx.lineWidth = 1.5 * s;
    drawExtendedLine(ctx, { x: arr.bx, y: arr.by }, { x: arr.fx, y: arr.fy }, W, H);
    ctx.globalAlpha = 1;
    ctx.lineWidth = 3 * s;
    ctx.beginPath(); ctx.moveTo(arr.bx, arr.by); ctx.lineTo(arr.fx, arr.fy); ctx.stroke();
    ctx.fillStyle = COLOR_RAW;
    drawArrowhead(ctx, arr.fx, arr.fy,
                  Math.atan2(arr.dy, arr.dx), 12 * s);
    // Small dots at each ankle.
    for (const [x, y] of [[arr.lx, arr.ly], [arr.rx, arr.ry]]) {
      ctx.beginPath(); ctx.arc(x, y, 3.5 * s, 0, Math.PI * 2); ctx.fill();
    }

    // (2) Predicted facing arrow from hip midpoint (cyan).
    const hip = hipMid(state.pose, f);
    if (hip && fit) {
      const predGt = wrap180(fit.sign * arr.angle_deg + fit.offset_deg);
      const imgAngle = (90 - predGt) * Math.PI / 180;
      const len = 70 * s;
      const x1 = hip.x + len * Math.cos(imgAngle);
      const y1 = hip.y + len * Math.sin(imgAngle);
      ctx.strokeStyle = COLOR_PRED;
      ctx.fillStyle = COLOR_PRED;
      ctx.lineWidth = 3.5 * s;
      ctx.beginPath(); ctx.moveTo(hip.x, hip.y); ctx.lineTo(x1, y1); ctx.stroke();
      drawArrowhead(ctx, x1, y1, imgAngle, 14 * s);
    }

    // (3) GT label arrow from hip midpoint (red, dashed).
    const gt = gtForCurrentFrame(state);
    if (hip && gt) {
      const imgAngle = (90 - gt.label) * Math.PI / 180;
      const len = 70 * s;
      const x1 = hip.x + len * Math.cos(imgAngle);
      const y1 = hip.y + len * Math.sin(imgAngle);
      ctx.strokeStyle = COLOR_GT;
      ctx.fillStyle = COLOR_GT;
      ctx.lineWidth = 3.5 * s;
      ctx.setLineDash([6 * s, 4 * s]);
      ctx.beginPath(); ctx.moveTo(hip.x, hip.y); ctx.lineTo(x1, y1); ctx.stroke();
      ctx.setLineDash([]);
      drawArrowhead(ctx, x1, y1, imgAngle, 14 * s);
    }

    ctx.restore();
  },
};
