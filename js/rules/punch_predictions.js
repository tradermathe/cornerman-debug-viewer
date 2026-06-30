// Punch predictions (axiality model). A focused scorecard for the trained
// temporal axiality model: for every labeled straight in this round, show the
// model's predicted angle vs the labeled truth, in degrees.
//
// Angle here is the off-axis angle = arccos(axiality), folded to [0°, 90°]:
//   0°  = straight down the lens (a straight thrown toward OR away from camera)
//   90° = flat across the image (side-on)
// The model is magnitude only — it can't tell toward from away — so you read
// left/right (toward/away) yourself from the video. That's the deliberate
// limit the user signed off on ("just show what we have, I can figure out the
// right or left").
//
// Data join: this round's straights ⨝ the model's per-punch axiality. Two
// sources, in priority order:
//   1. Sheet labels (state.labels.detections) ⨝ predictions_axiality_*.json by
//      punch_uuid — carries direction truth, so model-vs-truth grading shows.
//   2. Model detections (state.punches.detections) with inline `axiality` — for
//      unlabeled videos with no sheet/sidecar; model axiality only, no truth.
// No pose geometry — this lens is only the learned model's guess (vs truth when
// a label exists).

import { handForLabel } from "../sheet-labels.js";
import {
  ensureAxialityModel,
  axialityForPunch,
  axialityModelMeta,
  axialityModelError,
  axialityBucketName,
} from "./axiality_model.js";

// Straight punch labels (head + body). Matches arm_extension.js. The sheet uses
// "jab_head"/"cross_head"/…; the on-device punch classifier emits bare
// "jab"/"cross" — accept both so model-detected straights count too.
const STRAIGHTS = new Set(["jab_head", "jab_body", "cross_head", "cross_body"]);
function isStraight(pt) {
  return STRAIGHTS.has(pt)
    || (typeof pt === "string" && (pt.startsWith("jab") || pt.startsWith("cross")));
}

// Snap levels = |cos| of [90,67.5,45,22.5,0]° — to bucket an inline axiality that
// arrives without a precomputed bucket. Mirrors AxialityFeatures.levels.
const LEVELS = [0.0, 0.3826834323650898, 0.7071067811865476, 0.9238795325112867, 1.0];
function bucketOf(ax) {
  if (!Number.isFinite(ax)) return null;
  const x = Math.max(0, Math.min(1, ax));
  let best = 0, bd = Infinity;
  for (let i = 0; i < LEVELS.length; i++) {
    const d = Math.abs(x - LEVELS[i]);
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

// ── module state ────────────────────────────────────────────────────────────
let host = null;
let videoEl = null;
let punches = [];        // this round's straights, in time order (uuid join key carried)
let lastLabels = null;
let lastPunchesRef = null;
let modelSource = false; // true when built from model detections (no sheet labels)
let latestState = null;

// Per-punch prediction: the sheet+sidecar join by uuid (carries ground truth),
// else the inline model axiality (no truth — model-detected/unlabeled rounds).
// Mirror of the arm_extension fallback: axialityForPunch(uuid) ?? d.axiality.
function predFor(p) {
  const mp = axialityForPunch(p.punch_uuid);
  if (mp) return mp;
  if (Number.isFinite(p.axiality)) {
    return {
      predAxiality: p.axiality,
      predBucket: Number.isInteger(p.axiality_bucket) ? p.axiality_bucket : bucketOf(p.axiality),
      gtAxiality: null, gtBucket: null, source: "inline",
    };
  }
  return null;
}

// ── small helpers ─────────────────────────────────────────────────────────--
function lerp(a, b, t) { return a + (b - a) * t; }

// Off-axis angle in degrees = arccos(axiality): 0° = down the lens, 90° = side-on.
function fmtDeg(ax) {
  if (!Number.isFinite(ax)) return "—";
  return `${Math.round(Math.acos(Math.max(0, Math.min(1, ax))) * 180 / Math.PI)}°`;
}
function deg(ax) {
  return Number.isFinite(ax) ? Math.acos(Math.max(0, Math.min(1, ax))) * 180 / Math.PI : NaN;
}

// Color ramp for axiality: blue (across, 0) -> green (mid) -> orange (toward, 1).
function axColor(ax) {
  if (!Number.isFinite(ax)) return "rgba(180,180,180,0.9)";
  const x = Math.max(0, Math.min(1, ax));
  let c0, c1, t;
  if (x < 0.5) { c0 = [74, 168, 255]; c1 = [122, 250, 154]; t = x / 0.5; }
  else { c0 = [122, 250, 154]; c1 = [255, 107, 74]; t = (x - 0.5) / 0.5; }
  return `rgb(${Math.round(lerp(c0[0], c1[0], t))},${Math.round(lerp(c0[1], c1[1], t))},${Math.round(lerp(c0[2], c1[2], t))})`;
}

// Bucket agreement between model and truth. Exact = ✓, within one bucket = ±1
// (the model's headline metric is ±1 accuracy), off by two or more = ✗.
function agree(predB, gtB) {
  if (!Number.isInteger(predB) || !Number.isInteger(gtB)) return { mark: "—", color: "#888" };
  const d = Math.abs(predB - gtB);
  if (d === 0) return { mark: "✓", color: "#7afa9a" };
  if (d === 1) return { mark: "±1", color: "#f5b945" };
  return { mark: "✗", color: "#ff6b4a" };
}

// ── compute ───────────────────────────────────────────────────────────────--
function buildPunches(state) {
  // Prefer sheet labels (carry direction truth); fall back to the model's own
  // detections (state.punches) for unlabeled videos — those carry inline axiality.
  const sheet = state.labels?.detections || [];
  const model = state.punches?.detections || [];
  modelSource = sheet.length === 0 && model.length > 0;
  const dets = sheet.length ? sheet : model;
  punches = dets
    .filter(d => isStraight(d.punch_type))
    .map(d => ({
      hand: d.hand || handForLabel(d.punch_type),
      punch_type: d.punch_type,
      timestamp: d.timestamp,            // cache-relative seconds (punch center) for seek
      start_frame: d.start_frame,
      end_frame: d.end_frame,
      punch_uuid: d.punch_uuid || null,  // join key to the temporal model's preds
      axiality: d.axiality != null ? Number(d.axiality) : undefined,         // inline (model)
      axiality_bucket: Number.isInteger(d.axiality_bucket) ? d.axiality_bucket : undefined,
    }))
    .sort((a, b) => a.start_frame - b.start_frame);
}

function recompute(state) {
  if (state.labels === lastLabels && state.punches === lastPunchesRef && punches.length) return;
  lastLabels = state.labels;
  lastPunchesRef = state.punches;
  buildPunches(state);
}

// The straight whose labeled window contains the current frame, if any.
function activePunch(f) {
  if (!punches.length) return null;
  return punches.find(p => f >= p.start_frame && f <= p.end_frame) || null;
}

// Fires once when the sidecar finishes loading (async) so the table + canvas
// refresh without waiting for the next frame tick.
function onModelReady() {
  if (latestState) renderSidebar(latestState);
  if (typeof window !== "undefined" && window.__viewerRedraw) window.__viewerRedraw();
}

// ── status + summary ────────────────────────────────────────────────────────
function statusHtml() {
  if (modelSource) {
    return `<span class="muted">No sheet labels for this round — showing the model's `
      + `axiality on <b>${punches.length}</b> model-detected straight(s) `
      + `(<code>*_punches.json</code>). No direction truth to grade against.</span>`;
  }
  const err = axialityModelError();
  if (err) return `<span class="muted">temporal model: load error — ${err}</span>`;
  const m = axialityModelMeta();
  if (!m) {
    return `<span class="muted">No <code>predictions_axiality_*.json</code> in this folder — `
      + `run <code>train_axiality_temporal.py</code> (it writes one next to the caches).</span>`;
  }
  const ens = m.metrics && m.metrics["TCN_FA (ours, 3-seed ensemble)"];
  const pm1 = ens && Number.isFinite(ens.overall_pm1) ? ` &plusmn;1 ${ens.overall_pm1.toFixed(2)}` : "";
  const ex = ens && Number.isFinite(ens.overall_exact) ? ` · exact ${ens.overall_exact.toFixed(2)}` : "";
  const c = m.counts;
  const cov = c
    ? `${c.total} scored <span class="muted">(${c.oof} held-out + ${c.infer} inferred)</span>`
    : `${m.n} punches scored`;
  const cvLine = (pm1 || ex)
    ? `<span class="muted">held-out CV (labeled subset):${pm1}${ex}</span><br>`
    : "";
  const stamp = m.exportedAt ? `${m.exportedAt} · ` : "";
  return `temporal model <code>${m.model}</code> · ${cov}<br>`
    + cvLine
    + `<span class="muted">${stamp}${m.file}</span>`;
}

function summaryHtml() {
  if (!punches.length) return "";
  let scored = 0, withGt = 0, exact = 0, pm1 = 0, sumAbsDeg = 0;
  for (const p of punches) {
    const mp = predFor(p);
    if (!mp) continue;
    scored++;
    if (!Number.isInteger(mp.gtBucket)) continue;   // forward-inferred: no truth to grade
    withGt++;
    const d = Math.abs(mp.predBucket - mp.gtBucket);
    if (d === 0) exact++;
    if (d <= 1) pm1++;
    const dd = Math.abs(deg(mp.predAxiality) - deg(mp.gtAxiality));
    if (Number.isFinite(dd)) sumAbsDeg += dd;
  }
  if (!scored) {
    return `<span class="muted">${punches.length} straight(s) in this round — none scored by the model `
      + `(no axiality for these punches).</span>`;
  }
  const unscored = punches.length - scored;
  const unscoredTxt = unscored ? ` · ${unscored} unscored` : "";
  const gradeLine = withGt
    ? `<span class="muted">vs truth (${withGt} labeled): bucket exact ${exact}/${withGt} · `
      + `&plusmn;1 ${pm1}/${withGt} · mean |&Delta;angle| ${Math.round(sumAbsDeg / withGt)}°</span>`
    : `<span class="muted">no direction labels in this round — predictions shown without truth.</span>`;
  return `this round: <b>${scored}</b>/${punches.length} scored${unscoredTxt}<br>` + gradeLine;
}

// ── canvas HUD ──────────────────────────────────────────────────────────────
// Compact overlay for the straight under the playhead: model angle vs truth
// angle, so the prediction reads right on the video while scrubbing.
function drawHud(ctx, state) {
  const p = activePunch(state.frame);
  if (!p) return;
  const mp = predFor(p);
  const s = state.renderScale || 1;
  const lines = [
    `${p.hand || "?"} ${p.punch_type}`,
    mp ? `model ${fmtDeg(mp.predAxiality)} · ${axialityBucketName(mp.predBucket)}` : "model —",
    mp ? `truth ${fmtDeg(mp.gtAxiality)} · ${axialityBucketName(mp.gtBucket)}` : "truth —",
  ];
  const colors = [
    "rgba(255,255,255,0.92)",
    mp ? axColor(mp.predAxiality) : "rgba(255,255,255,0.5)",
    mp ? axColor(mp.gtAxiality) : "rgba(255,255,255,0.5)",
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
    ctx.fillStyle = colors[i];
    ctx.fillText(lines[i], x0 + padX, y);
    y += lineH;
  }
  ctx.restore();
}

// ── sidebar ───────────────────────────────────────────────────────────────--
function buildSidebar() {
  if (!host) return;
  host.innerHTML = `
    <h2>Punch predictions (axiality)</h2>
    <p class="hint">
      For each straight, the trained temporal model's predicted <b>angle</b> vs
      the labeled <b>truth</b>, in degrees. Angle is how far off the camera axis
      the punch points: <b>0°</b> = straight down the lens (toward or away),
      <b>90°</b> = flat across the image (side-on). Magnitude only — the model
      can't tell toward from away, so read left/right yourself. <b>✓</b> = same
      bucket, <b>±1</b> = one bucket off, <b>✗</b> = two or more.
      <span class="muted">A <b>—</b> truth = no direction label yet; the model
      still predicts it (forward inference). A fully <b>—</b> row = no pose cache
      for that punch.</span>
    </p>
    <div id="pp-status" class="hint" style="line-height:1.7;margin:8px 0;"></div>
    <div id="pp-summary" class="hint" style="line-height:1.7;margin:8px 0;"></div>
    <div id="pp-table" style="margin-top:12px;"></div>
  `;
}

function renderSidebar(state) {
  if (!host) return;
  if (!host.querySelector("#pp-status")) buildSidebar();

  const statusEl = host.querySelector("#pp-status");
  if (statusEl) statusEl.innerHTML = statusHtml();

  const summaryEl = host.querySelector("#pp-summary");
  if (summaryEl) summaryEl.innerHTML = summaryHtml();

  const tbl = host.querySelector("#pp-table");
  if (!tbl) return;
  if (!punches.length) {
    const hasModel = (state.punches?.detections || []).length > 0;
    tbl.innerHTML = (state.labels?.detections || hasModel)
      ? `<span class="muted">No straights detected in this round.</span>`
      : `<span class="muted">Waiting for label or model data…</span>`;
    return;
  }

  const f = state.frame;
  const rows = punches.map((p, i) => {
    const mp = predFor(p);
    const active = f >= p.start_frame && f <= p.end_frame;
    let modelCell, truthCell, agreeCell;
    if (mp) {
      modelCell = `<span style="color:${axColor(mp.predAxiality)};font-weight:600">${fmtDeg(mp.predAxiality)}</span>`
        + ` <span class="muted">${axialityBucketName(mp.predBucket)}</span>`;
      truthCell = `<span style="color:${axColor(mp.gtAxiality)};font-weight:600">${fmtDeg(mp.gtAxiality)}</span>`
        + ` <span class="muted">${axialityBucketName(mp.gtBucket)}</span>`;
      const a = agree(mp.predBucket, mp.gtBucket);
      const dd = Math.round(Math.abs(deg(mp.predAxiality) - deg(mp.gtAxiality)));
      agreeCell = `<span style="color:${a.color};font-weight:600">${a.mark}</span> <span class="muted">${dd}°</span>`;
    } else {
      modelCell = `<span class="muted">—</span>`;
      truthCell = `<span class="muted">—</span>`;
      agreeCell = `<span class="muted">—</span>`;
    }
    const bg = active ? ' style="cursor:pointer;background:rgba(120,200,255,0.12)"' : ' style="cursor:pointer"';
    return `
      <tr data-i="${i}"${bg}>
        <td>${p.hand || "?"}</td>
        <td>${p.punch_type}</td>
        <td style="text-align:right">${modelCell}</td>
        <td style="text-align:right">${truthCell}</td>
        <td style="text-align:right">${agreeCell}</td>
      </tr>`;
  }).join("");

  tbl.innerHTML = `
    <table style="width:100%;border-collapse:collapse;font-size:12px">
      <thead><tr style="color:#888;text-align:left">
        <th>arm</th><th>type</th>
        <th style="text-align:right" title="the model's predicted off-axis angle = arccos(axiality); 0° = down the lens (toward/away), 90° = side-on">model</th>
        <th style="text-align:right" title="the labeled off-axis angle, same convention">truth</th>
        <th style="text-align:right" title="bucket agreement (✓ exact / ±1 one off / ✗ two+) and the angle gap in degrees">Δ</th>
      </tr></thead><tbody>${rows}</tbody>
    </table>`;

  tbl.querySelectorAll("tr[data-i]").forEach(tr => {
    tr.addEventListener("click", () => {
      const p = punches[Number(tr.dataset.i)];
      if (p && videoEl && Number.isFinite(p.timestamp)) {
        videoEl.currentTime = (latestState?.start_sec || 0) + p.timestamp;
      }
    });
  });
}

// ── lens contract ───────────────────────────────────────────────────────────
export const PunchPredictionsRule = {
  id: "punch_predictions",
  label: "Punch predictions (axiality)",

  mount(_host, state) {
    host = _host;
    videoEl = document.getElementById("video");
    lastLabels = null;
    latestState = state;
    recompute(state);
    ensureAxialityModel(state, onModelReady);
    buildSidebar();
    renderSidebar(state);
  },

  update(state) {
    latestState = state;
    recompute(state);
    ensureAxialityModel(state, onModelReady);
    renderSidebar(state);
  },

  draw(ctx, state) {
    latestState = state;
    drawHud(ctx, state);
  },
};
