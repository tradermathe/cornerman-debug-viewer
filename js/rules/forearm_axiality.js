// Forearm axiality lens. Reads the 2D pose cache (COCO-17, pixels) and scores,
// per frame and per arm, how aligned a punch is with the camera axis: 0 means
// the forearm lies flat across the image (a hook), 1 means it points down the
// lens (a straight thrown toward or away from camera). The signal is pure
// foreshortening of the rigid forearm bone read against the torso ruler. See
// forearm_axiality_core.js for the math and test/forearm_axiality_oracle.mjs
// for the synthetic projection test that certifies it.
//
// Magnitude only (no toward vs away sign) and straights only, by design. The
// flat-reference circle on the canvas is the key overlay: the wrist sitting on
// the rim means flat (low axiality), pulled in toward the elbow means
// foreshortened (high axiality).

import { J } from "../skeleton.js";
import { handForLabel } from "../sheet-labels.js";
import {
  forearmLengths,
  axialityFromRatio,
  percentileWithIndex,
  runningMedian,
  mid,
} from "./forearm_axiality_core.js";

// Straight punch labels (head + body). Matches arm_extension.js appliesTo.
const STRAIGHTS = new Set(["jab_head", "jab_body", "cross_head", "cross_body"]);

// (hand, stance) -> anatomical side. Mirrors arm_extension.js SIDE_FOR.
const SIDE_FOR = {
  lead: { orthodox: "L", southpaw: "R" },
  rear: { orthodox: "R", southpaw: "L" },
};
const SIDE_J = {
  L: { sh: J.L_SHOULDER, el: J.L_ELBOW, wr: J.L_WRIST },
  R: { sh: J.R_SHOULDER, el: J.R_ELBOW, wr: J.R_WRIST },
};

const MIN_CONF = 0.30;          // elbow / wrist confidence floor for a valid frame
const TORSO_CONF = 0.20;        // shoulder / hip floor for a trustworthy ruler
const DEPTH_LEAN_FACTOR = 0.80; // torso below this * running median = depth-lean flag
export const APEX_HALF = 2;     // apex aggregation window is apex +/- this many frames
const F0_PCT = 90;              // flat forearm reference percentile

// ── module state ────────────────────────────────────────────────────────────
let host = null;
let videoEl = null;
let tlCanvas = null;
let tlCtx = null;
let tlClickHandler = null;
let arms = null;        // { L: sideData, R: sideData }
let punches = [];       // straights with apex + axiality_punch
let lastPose = null;
let lastLabels = null;
let latestState = null;

// ── small helpers ───────────────────────────────────────────────────────────
function pt(pose, f, j) {
  const i = (f * 17 + j) * 2;
  return [pose.skeleton[i], pose.skeleton[i + 1]];
}
function cf(pose, f, j) { return pose.conf[f * 17 + j]; }
function imp(pose, f, j) { return pose.imputed ? pose.imputed[f * 17 + j] : 0; }
function fmt(x) { return Number.isFinite(x) ? x.toFixed(3) : "—"; }
function fmt0(x) { return Number.isFinite(x) ? String(Math.round(x)) : "—"; }
function lerp(a, b, t) { return a + (b - a) * t; }

// Color ramp for axiality: blue (across, 0) -> green (mid) -> orange (toward, 1).
function axColor(ax) {
  if (!Number.isFinite(ax)) return "rgba(180,180,180,0.9)";
  const x = Math.max(0, Math.min(1, ax));
  let c0, c1, t;
  if (x < 0.5) { c0 = [74, 168, 255]; c1 = [122, 250, 154]; t = x / 0.5; }
  else { c0 = [122, 250, 154]; c1 = [255, 107, 74]; t = (x - 0.5) / 0.5; }
  return `rgb(${Math.round(lerp(c0[0], c1[0], t))},${Math.round(lerp(c0[1], c1[1], t))},${Math.round(lerp(c0[2], c1[2], t))})`;
}

// ── compute ─────────────────────────────────────────────────────────────────
// Exported so the arm-extension lens can gate on the exact same axiality the
// standalone lens shows (raw Vision elbow/wrist, MIN_CONF 0.30, F0 = 90th pct).
export function computeSide(pose, side) {
  const n = pose.n_frames;
  const sj = SIDE_J[side];
  const forearm = new Float64Array(n).fill(NaN);
  const torso = new Float64Array(n).fill(NaN);
  const ratio = new Float64Array(n).fill(NaN);
  const valid = new Uint8Array(n);
  const eConf = new Float64Array(n);
  const wConf = new Float64Array(n);

  for (let f = 0; f < n; f++) {
    const el = pt(pose, f, sj.el), wr = pt(pose, f, sj.wr);
    const midSh = mid(pt(pose, f, J.L_SHOULDER), pt(pose, f, J.R_SHOULDER));
    const midHip = mid(pt(pose, f, J.L_HIP), pt(pose, f, J.R_HIP));
    const L = forearmLengths(el, wr, midSh, midHip);
    forearm[f] = L.forearm; torso[f] = L.torso; ratio[f] = L.ratio;
    const ec = cf(pose, f, sj.el), wc = cf(pose, f, sj.wr);
    eConf[f] = ec; wConf[f] = wc;

    const armOk = ec >= MIN_CONF && wc >= MIN_CONF
      && !imp(pose, f, sj.el) && !imp(pose, f, sj.wr)
      && Number.isFinite(L.forearm);
    const torsoOk = Number.isFinite(L.torso) && L.torso > 1e-3
      && cf(pose, f, J.L_SHOULDER) >= TORSO_CONF && cf(pose, f, J.R_SHOULDER) >= TORSO_CONF
      && cf(pose, f, J.L_HIP) >= TORSO_CONF && cf(pose, f, J.R_HIP) >= TORSO_CONF;
    valid[f] = (armOk && torsoOk) ? 1 : 0;
  }

  // F0 = 90th percentile of ratio over valid frames (the flat forearm length).
  const ratioValid = new Float64Array(n).fill(NaN);
  for (let f = 0; f < n; f++) if (valid[f]) ratioValid[f] = ratio[f];
  const F0 = percentileWithIndex(ratioValid, F0_PCT);

  const B = new Float64Array(n).fill(NaN);
  const axiality = new Float64Array(n).fill(NaN);
  const clamp = new Uint8Array(n);
  const lean = new Uint8Array(n);
  const medTorso = runningMedian(torso, 31);
  for (let f = 0; f < n; f++) {
    const a = axialityFromRatio(ratio[f], F0.value);
    B[f] = a.B; axiality[f] = a.axiality;
    if (valid[f] && F0.value > 0 && Number.isFinite(ratio[f]) && ratio[f] / F0.value > 1) clamp[f] = 1;
    if (Number.isFinite(torso[f]) && Number.isFinite(medTorso[f]) && medTorso[f] > 0
      && torso[f] < DEPTH_LEAN_FACTOR * medTorso[f]) lean[f] = 1;
  }
  return { forearm, torso, ratio, B, axiality, valid, clamp, lean, eConf, wConf, F0 };
}

function apexFrame(side, sf, ef) {
  const a = arms[side];
  const lo = Math.max(0, sf | 0), hi = Math.min(a.forearm.length - 1, ef | 0);
  let best = -1, bv = -Infinity, anyBest = -1, anyBv = -Infinity;
  for (let f = lo; f <= hi; f++) {
    const v = a.forearm[f];
    if (!Number.isFinite(v)) continue;
    if (v > anyBv) { anyBv = v; anyBest = f; }
    if (a.valid[f] && v > bv) { bv = v; best = f; }
  }
  return best >= 0 ? best : anyBest;
}

function medianAxiality(side, apex) {
  const a = arms[side];
  const vals = [];
  for (let f = apex - APEX_HALF; f <= apex + APEX_HALF; f++) {
    if (f < 0 || f >= a.axiality.length) continue;
    if (a.valid[f] && Number.isFinite(a.axiality[f])) vals.push(a.axiality[f]);
  }
  if (!vals.length) return Number.isFinite(a.axiality[apex]) ? a.axiality[apex] : NaN;
  vals.sort((x, y) => x - y);
  const m = vals.length >> 1;
  return vals.length % 2 ? vals[m] : (vals[m - 1] + vals[m]) / 2;
}

function buildPunches(state) {
  const dets = state.labels?.detections || [];
  punches = dets
    .filter(d => STRAIGHTS.has(d.punch_type))
    .map(d => {
      const hand = d.hand || handForLabel(d.punch_type);
      const stance = (d.stance === "southpaw" || d.stance === "orthodox") ? d.stance : "orthodox";
      const side = (hand && SIDE_FOR[hand]?.[stance]) || "L";
      const apex = apexFrame(side, d.start_frame, d.end_frame);
      const a = arms[side];
      return {
        start_frame: d.start_frame, end_frame: d.end_frame,
        side, hand, punch_type: d.punch_type, apex,
        ratioApex: apex >= 0 ? a.ratio[apex] : NaN,
        F0: a.F0.value,
        BApex: apex >= 0 ? a.B[apex] : NaN,
        axialityPunch: apex >= 0 ? medianAxiality(side, apex) : NaN,
        dir: d.punch_direction || d.direction || null,
      };
    })
    .sort((x, y) => x.start_frame - y.start_frame);
}

function recompute(state) {
  const poseChanged = state.pose !== lastPose;
  const labelsChanged = state.labels !== lastLabels;
  if (!poseChanged && !labelsChanged && arms) return;
  if (poseChanged || !arms) {
    arms = state.pose
      ? { L: computeSide(state.pose, "L"), R: computeSide(state.pose, "R") }
      : null;
  }
  lastPose = state.pose;
  lastLabels = state.labels;
  if (arms) buildPunches(state);
  else punches = [];
}

// ── arm selection ───────────────────────────────────────────────────────────
function activePunch(f) {
  if (!punches.length) return null;
  const inside = punches.find(p => f >= p.start_frame && f <= p.end_frame);
  if (inside) return inside;
  let best = null, bd = Infinity;
  for (const p of punches) {
    const d = Math.abs((p.apex >= 0 ? p.apex : p.start_frame) - f);
    if (d < bd) { bd = d; best = p; }
  }
  return best;
}

// Which arm the canvas + timeline display. The active punch's arm when there is
// one; otherwise the arm doing more (larger ratio) at this frame, so the
// overlay stays useful outside labeled windows.
function displaySide(f) {
  const p = activePunch(f);
  if (p) return p.side;
  if (!arms) return "R";
  const rl = arms.L.ratio[f], rr = arms.R.ratio[f];
  if (Number.isFinite(rr) && (!Number.isFinite(rl) || rr >= rl)) return "R";
  return "L";
}

// ── canvas draw ─────────────────────────────────────────────────────────────
function drawCanvas(ctx, state) {
  const pose = state.pose;
  if (!pose || !arms) return;
  const f = state.frame, s = state.renderScale || 1;
  const side = displaySide(f);
  const sj = SIDE_J[side], a = arms[side];
  const el = pt(pose, f, sj.el), wr = pt(pose, f, sj.wr);
  const midSh = mid(pt(pose, f, J.L_SHOULDER), pt(pose, f, J.R_SHOULDER));
  const midHip = mid(pt(pose, f, J.L_HIP), pt(pose, f, J.R_HIP));
  const torsoLen = a.torso[f], F0 = a.F0.value, ax = a.axiality[f];

  // Torso ruler (mid-shoulder to mid-hip): the denominator, drawn so it is visible.
  if (Number.isFinite(midSh[0]) && Number.isFinite(midHip[0])) {
    ctx.save();
    ctx.strokeStyle = "rgba(120,200,255,0.55)";
    ctx.lineWidth = 2 * s;
    ctx.setLineDash([6 * s, 4 * s]);
    ctx.beginPath(); ctx.moveTo(midSh[0], midSh[1]); ctx.lineTo(midHip[0], midHip[1]); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(120,200,255,0.85)";
    for (const p of [midSh, midHip]) { ctx.beginPath(); ctx.arc(p[0], p[1], 3 * s, 0, Math.PI * 2); ctx.fill(); }
    ctx.restore();
  }

  // Flat-reference circle: centered on the elbow, radius F0 * torso(t). The
  // wrist would sit on this rim if the forearm were flat at the current scale;
  // it actually sits at fraction B of the radius (pulled inward = foreshortened
  // = axial). This is the overlay that lets the value be judged by eye.
  if (Number.isFinite(el[0]) && Number.isFinite(torsoLen) && F0 > 0) {
    const radius = F0 * torsoLen;
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.lineWidth = 1.5 * s;
    ctx.setLineDash([4 * s, 4 * s]);
    ctx.beginPath(); ctx.arc(el[0], el[1], radius, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);
    if (Number.isFinite(wr[0])) {
      const dx = wr[0] - el[0], dy = wr[1] - el[1];
      const L = Math.hypot(dx, dy) || 1;
      const rimX = el[0] + (dx / L) * radius, rimY = el[1] + (dy / L) * radius;
      ctx.strokeStyle = "rgba(255,255,255,0.30)";
      ctx.lineWidth = 1 * s;
      ctx.beginPath(); ctx.moveTo(el[0], el[1]); ctx.lineTo(rimX, rimY); ctx.stroke();
      ctx.fillStyle = "rgba(255,255,255,0.55)";
      ctx.beginPath(); ctx.arc(rimX, rimY, 3.5 * s, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
  }

  // The forearm itself, bold, colored by axiality.
  if (Number.isFinite(el[0]) && Number.isFinite(wr[0])) {
    const col = axColor(ax);
    ctx.save();
    ctx.strokeStyle = col; ctx.lineWidth = 5 * s; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(el[0], el[1]); ctx.lineTo(wr[0], wr[1]); ctx.stroke();
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(wr[0], wr[1], 5 * s, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(el[0], el[1], 4 * s, 0, Math.PI * 2); ctx.fill();
    const p = activePunch(f);
    ctx.font = `bold ${13 * s}px ui-monospace, "SF Mono", Menlo, monospace`;
    ctx.fillStyle = "rgba(255,255,255,0.92)";
    ctx.fillText(p ? `${p.hand || side} ${p.punch_type}` : `${side} arm`, el[0] + 8 * s, el[1] - 8 * s);
    ctx.restore();
  }

  drawHud(ctx, state, side);
}

function drawHud(ctx, state, side) {
  const f = state.frame, s = state.renderScale || 1, a = arms[side];
  const ax = a.axiality[f];
  const p = activePunch(f);
  const inApex = p && p.apex >= 0 && Math.abs(f - p.apex) <= APEX_HALF;
  const flags = [];
  if (a.lean[f]) flags.push("depth-lean");
  if (a.clamp[f]) flags.push("clamp");
  if (!a.valid[f]) flags.push("low-conf");

  const title = p ? `${p.hand || side} ${p.punch_type}${inApex ? "  · APEX" : ""}` : `${side} arm (no punch)`;
  const lines = [
    title,
    `axiality ${fmt(ax)}`,
    "",  // gauge row
    `forearm ${fmt0(a.forearm[f])}px · torso ${fmt0(a.torso[f])}px`,
    `ratio ${fmt(a.ratio[f])} · F0 ${fmt(a.F0.value)} · B ${fmt(a.B[f])}`,
    `el ${fmt(a.eConf[f])} · wr ${fmt(a.wConf[f])}`,
    flags.length ? `flags: ${flags.join(", ")}` : "flags: none",
  ];

  const fontPx = 14 * s, lineH = 20 * s, padX = 14 * s, padY = 10 * s;
  const x0 = 24 * s, y0 = 24 * s;
  ctx.save();
  ctx.font = `${fontPx}px ui-monospace, "SF Mono", Menlo, monospace`;
  const w = Math.max(...lines.map(t => ctx.measureText(t).width), 150 * s) + 2 * padX;
  const h = padY * 2 + lineH * lines.length;
  ctx.fillStyle = "rgba(0,0,0,0.78)";
  const r = 10 * s;
  ctx.beginPath();
  ctx.moveTo(x0 + r, y0);
  ctx.arcTo(x0 + w, y0, x0 + w, y0 + h, r);
  ctx.arcTo(x0 + w, y0 + h, x0, y0 + h, r);
  ctx.arcTo(x0, y0 + h, x0, y0, r);
  ctx.arcTo(x0, y0, x0 + w, y0, r);
  ctx.closePath(); ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.18)"; ctx.lineWidth = 1 * s; ctx.stroke();

  let y = y0 + padY + fontPx;
  for (let i = 0; i < lines.length; i++) {
    if (i === 2) {
      // gauge bar in the reserved row
      const gx = x0 + padX, gy = y - fontPx * 0.8, gw = w - 2 * padX, gh = 9 * s;
      ctx.fillStyle = "rgba(255,255,255,0.15)";
      ctx.fillRect(gx, gy, gw, gh);
      if (Number.isFinite(ax)) {
        ctx.fillStyle = axColor(ax);
        ctx.fillRect(gx, gy, gw * Math.max(0, Math.min(1, ax)), gh);
      }
      ctx.strokeStyle = "rgba(255,255,255,0.25)"; ctx.lineWidth = 1 * s;
      ctx.strokeRect(gx, gy, gw, gh);
      y += lineH; continue;
    }
    ctx.fillStyle = i === 1 ? axColor(ax)
      : i === 0 ? "rgba(255,255,255,0.92)"
        : (i === lines.length - 1 && flags.length) ? "#f5b945" : "rgba(255,255,255,0.78)";
    ctx.fillText(lines[i], x0 + padX, y);
    y += lineH;
  }
  ctx.restore();
}

// ── timeline (the main validation panel) ────────────────────────────────────
function renderTimeline(state) {
  if (!tlCtx || !arms || !state.pose) return;
  const n = state.pose.n_frames;
  const cssW = tlCanvas.clientWidth || tlCanvas.parentElement?.clientWidth || 600;
  const Hcss = 120;
  const dpr = window.devicePixelRatio || 1;
  tlCanvas.width = Math.round(cssW * dpr);
  tlCanvas.height = Math.round(Hcss * dpr);
  const ctx = tlCtx;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = cssW, H = Hcss;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0c0f14"; ctx.fillRect(0, 0, W, H);

  const f = state.frame, side = displaySide(f), a = arms[side];
  const padT = 6, padB = 16, plotH = H - padT - padB;
  const xOf = fr => (n > 1 ? (fr / (n - 1)) * W : 0);
  const yOf = v => padT + (1 - Math.max(0, Math.min(1, v))) * plotH;

  // Punch windows + shaded apex windows.
  for (const p of punches) {
    const x0 = xOf(p.start_frame), x1 = xOf(p.end_frame);
    ctx.fillStyle = p.side === side ? "rgba(120,200,255,0.10)" : "rgba(120,120,140,0.05)";
    ctx.fillRect(x0, padT, Math.max(1, x1 - x0), plotH);
    if (p.apex >= 0) {
      const a0 = xOf(p.apex - APEX_HALF), a1 = xOf(p.apex + APEX_HALF);
      ctx.fillStyle = p.side === side ? "rgba(255,180,80,0.22)" : "rgba(255,180,80,0.08)";
      ctx.fillRect(a0, padT, Math.max(1, a1 - a0), plotH);
    }
  }

  // ratio trace (amber, thin), normalized so a flat arm (ratio ~ F0) sits high.
  const rscale = a.F0.value > 0 ? 1 / (1.15 * a.F0.value) : 1;
  ctx.strokeStyle = "rgba(245,185,69,0.7)"; ctx.lineWidth = 1;
  ctx.beginPath();
  let started = false;
  for (let fr = 0; fr < n; fr++) {
    const v = a.ratio[fr];
    if (!Number.isFinite(v)) { started = false; continue; }
    const x = xOf(fr), y = yOf(v * rscale);
    if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // axiality trace (cyan, bold) over valid frames.
  ctx.strokeStyle = "#3ad9e0"; ctx.lineWidth = 2;
  ctx.beginPath();
  started = false;
  for (let fr = 0; fr < n; fr++) {
    const v = a.axiality[fr];
    if (!Number.isFinite(v) || !a.valid[fr]) { started = false; continue; }
    const x = xOf(fr), y = yOf(v);
    if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Flagged / dropped frames ticked along the bottom edge.
  for (let fr = 0; fr < n; fr++) {
    let col = null;
    if (a.lean[fr]) col = "#e0653b";
    else if (a.clamp[fr]) col = "#c9a227";
    else if (!a.valid[fr]) col = "rgba(150,150,150,0.5)";
    if (col) { ctx.fillStyle = col; ctx.fillRect(xOf(fr), H - padB + 2, 1, 4); }
  }

  // Aggregated axiality_punch printed at each apex (this arm only).
  ctx.font = "10px ui-monospace, Menlo, monospace";
  for (const p of punches) {
    if (p.side !== side || !Number.isFinite(p.axialityPunch) || p.apex < 0) continue;
    const x = xOf(p.apex), y = yOf(p.axialityPunch);
    ctx.fillStyle = "#fff";
    ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.fillText(p.axialityPunch.toFixed(2), Math.min(W - 22, x + 3), Math.max(10, y - 3));
  }

  // Cursor at the current frame.
  const cx = xOf(f);
  ctx.strokeStyle = "rgba(255,255,255,0.85)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(cx, padT); ctx.lineTo(cx, H - padB); ctx.stroke();

  ctx.fillStyle = "rgba(255,255,255,0.5)";
  ctx.fillText(`${side} arm · axiality (cyan 0..1) · ratio/F0 (amber) · click to seek`, 6, H - 4);
}

// ── sidebar ─────────────────────────────────────────────────────────────────
function buildSidebar() {
  if (!host) return;
  host.innerHTML = `
    <h2>Forearm axiality</h2>
    <p class="hint">
      How aligned each punch is with the camera axis, from forearm
      foreshortening: <b>0</b> = flat across the image (a hook), <b>1</b> =
      down the lens (a straight toward or away). Magnitude only, straights only.
      The dashed <b>circle</b> is where the wrist would sit if the forearm were
      flat; the wrist pulled inward means foreshortened (axial).
    </p>
    <div id="fa-f0" class="hint" style="line-height:1.7;margin:8px 0;"></div>
    <canvas id="fa-timeline" style="width:100%;height:120px;display:block;border-radius:6px;border:1px solid #222;"></canvas>
    <div id="fa-table" style="margin-top:12px;"></div>
    <p class="hint" style="margin-top:12px;font-size:11px;">
      Off-center boxers carry a perspective error the ratio cannot remove (a
      straight thrown from the edge of frame can still read long). Trust the
      separation between toward and across straights, not the absolute value.
    </p>
  `;
  tlCanvas = host.querySelector("#fa-timeline");
  tlCtx = tlCanvas.getContext("2d");
  if (tlClickHandler) tlCanvas.removeEventListener("click", tlClickHandler);
  tlClickHandler = (ev) => {
    if (!latestState?.pose) return;
    const rect = tlCanvas.getBoundingClientRect();
    const frac = (ev.clientX - rect.left) / Math.max(1, rect.width);
    const n = latestState.pose.n_frames;
    const fr = Math.max(0, Math.min(n - 1, Math.round(frac * (n - 1))));
    if (videoEl && latestState.fps) videoEl.currentTime = (latestState.start_sec || 0) + fr / latestState.fps;
  };
  tlCanvas.addEventListener("click", tlClickHandler);
}

function renderSidebar(state) {
  if (!host) return;
  if (!host.querySelector("#fa-f0")) buildSidebar();

  const f0el = host.querySelector("#fa-f0");
  if (f0el && arms) {
    f0el.innerHTML =
      `flat reference <b>F0</b> (90th pct of ratio):<br>`
      + `L <code>${fmt(arms.L.F0.value)}</code> @f${arms.L.F0.index} (n${arms.L.F0.n}) · `
      + `R <code>${fmt(arms.R.F0.value)}</code> @f${arms.R.F0.index} (n${arms.R.F0.n})`;
  }

  const tbl = host.querySelector("#fa-table");
  if (tbl) {
    if (!punches.length) {
      tbl.innerHTML = state.labels?.detections
        ? `<span class="muted">No straights labeled in this round.</span>`
        : `<span class="muted">Waiting for label data…</span>`;
    } else {
      const rows = punches.map((p, i) => `
        <tr data-i="${i}" style="cursor:pointer">
          <td>${p.hand || p.side}</td>
          <td>${p.punch_type}</td>
          <td style="text-align:right">${p.apex}</td>
          <td style="text-align:right">${fmt(p.ratioApex)}</td>
          <td style="text-align:right;color:${axColor(p.axialityPunch)};font-weight:600">${fmt(p.axialityPunch)}</td>
          <td>${p.dir || "—"}</td>
        </tr>`).join("");
      tbl.innerHTML = `
        <table style="width:100%;border-collapse:collapse;font-size:12px">
          <thead><tr style="color:#888;text-align:left">
            <th>arm</th><th>type</th><th style="text-align:right">apex</th>
            <th style="text-align:right">ratio</th><th style="text-align:right">axiality</th><th>dir</th>
          </tr></thead><tbody>${rows}</tbody>
        </table>`;
      tbl.querySelectorAll("tr[data-i]").forEach(tr => {
        tr.addEventListener("click", () => {
          const p = punches[Number(tr.dataset.i)];
          if (p && p.apex >= 0 && videoEl && latestState?.fps) {
            videoEl.currentTime = (latestState.start_sec || 0) + p.apex / latestState.fps;
          }
        });
      });
    }
  }

  renderTimeline(state);
}

// ── lens contract ───────────────────────────────────────────────────────────
export const ForearmAxialityRule = {
  id: "forearm_axiality",
  label: "Forearm axiality (foreshortening)",

  skeletonStyle(state) {
    // Highlight the measured arm's elbow + wrist so the eye goes to the bone
    // under test.
    if (!arms || !state.pose) return {};
    const sj = SIDE_J[displaySide(state.frame)];
    return { highlightJoints: new Set([sj.el, sj.wr]) };
  },

  mount(_host, state) {
    host = _host;
    videoEl = document.getElementById("video");
    lastPose = null; lastLabels = null;
    recompute(state);
    buildSidebar();
    renderSidebar(state);
  },

  update(state) {
    latestState = state;
    recompute(state);
    renderSidebar(state);
  },

  draw(ctx, state) {
    latestState = state;
    drawCanvas(ctx, state);
  },

  unmount() {
    if (tlClickHandler && tlCanvas) tlCanvas.removeEventListener("click", tlClickHandler);
    tlClickHandler = null; tlCanvas = null; tlCtx = null;
  },
};
