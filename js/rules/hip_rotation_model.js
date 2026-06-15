// Hip rotation MODEL review — walk through every punch the temporal TCN scored
// and read its held-out 1–4 rating against the labelled 1–4 ground truth.
//
// This is the learned-model counterpart to hip_rotation_review.js (which is a
// hand-crafted geometric heuristic). Here there is no per-frame signal to
// recompute: train_hip_rotation_temporal.py already produced an honest
// cross-validated (out-of-fold) rating per punch and dumped it next to the
// Vision caches as:
//
//   predictions_hip_rotation_temporal.json
//   { "kind": "hip_rotation_temporal", "model": "...", "exported_at": "...",
//     "levels": [0, 1/3, 2/3, 1], "names": ["1 (none)",..,"4 (full)"],
//     "metrics": { ...cv_metrics... },
//     "punches": { "<punch_uuid>": { pred_rating:1-4, pred_score:0-1,
//                                    gt_rating:1-4 } } }
//
// The viewer auto-collects every predictions_*.json into state.predictionFiles;
// we pick the newest predictions_hip_rotation_*.json, parse it once, and join
// to this round's detections by punch_uuid. Punches the model never scored
// (no entry) are skipped — every entry carries both pred and GT, since the OOF
// dump only covers labelled punches.

import { J, drawSkeleton } from "../skeleton.js";

const NAME_RE = /^predictions_hip_rotation_.*\.json$/i;

// 1..4 rating palette: none → full.
const RATING_COLORS = {
  1: "#e85a5a",   // red    — no rotation
  2: "#f5b945",   // amber  — slight
  3: "#9bd96a",   // light green — good
  4: "#5fd97a",   // green  — full
};
const COLOR_HIP   = "#a78bfa";
const COLOR_OK    = "#5fd97a";
const COLOR_NEAR  = "#f5b945";   // off by 1
const COLOR_MISS  = "#e85a5a";   // off by 2+
const COLOR_MUTED = "#888";

// ─── sidecar load (idempotent, mirrors axiality_model.js) ───────────────────

let preds = null;       // Map<uuid, {predRating, predScore, gtRating}>
let pmeta = null;       // { names, levels, metrics, model, exportedAt, n }
let loadErr = null;
let filesRef = null, filesSize = -1, token = 0;

async function materialize(value) {
  if (value instanceof File) return value;
  if (typeof value?.getFile === "function") {
    try { return await value.getFile(); } catch { return null; }
  }
  return null;
}

function ensurePreds(state, onReady) {
  const files = state?.predictionFiles;
  const size = files ? files.size : 0;
  if (files === filesRef && size === filesSize) return;
  filesRef = files; filesSize = size;
  preds = null; pmeta = null; loadErr = null;
  const myToken = ++token;
  if (!files || size === 0) return;

  (async () => {
    const cands = [];
    for (const [name, value] of files) {
      if (!NAME_RE.test(name)) continue;
      const file = await materialize(value);
      if (file) cands.push({ name, file });
    }
    if (!cands.length) return;
    cands.sort((a, b) => (b.file.lastModified || 0) - (a.file.lastModified || 0));
    let nextPreds = null, nextMeta = null, nextErr = null;
    try {
      const parsed = JSON.parse(await cands[0].file.text());
      const m = new Map();
      for (const [uuid, v] of Object.entries(parsed.punches || {})) {
        m.set(uuid, {
          predRating: v.pred_rating, predScore: v.pred_score, gtRating: v.gt_rating,
        });
      }
      nextPreds = m;
      nextMeta = {
        names: parsed.names || ["1", "2", "3", "4"],
        levels: parsed.levels || null,
        metrics: parsed.metrics || null,
        model: parsed.model || "?",
        exportedAt: parsed.exported_at || null,
        file: cands[0].name, n: m.size,
      };
    } catch (e) {
      nextErr = e.message;
    }
    if (myToken !== token) return;
    preds = nextPreds; pmeta = nextMeta; loadErr = nextErr;
    if (typeof onReady === "function") onReady();
  })();
}

// ─── module state ───────────────────────────────────────────────────────────

let host = null;
let videoEl = null;
let timeupdateHandler = null;
let keydownHandler = null;
let loopWindow = null;
let activeIdx = -1;
let punches = [];           // this round's detections that have a model entry
let lastDetectionsRef = null;
let lastStemForReset = null;
let latestState = null;

const LOOP_PRE_SEC  = 0.3;
const LOOP_POST_SEC = 0.2;

// ─── helpers ────────────────────────────────────────────────────────────────

function agreeColor(gt, pred) {
  if (!Number.isFinite(gt) || !Number.isFinite(pred)) return COLOR_MUTED;
  const d = Math.abs(gt - pred);
  if (d === 0) return COLOR_OK;
  if (d === 1) return COLOR_NEAR;
  return COLOR_MISS;
}

function agreeSym(gt, pred) {
  if (!Number.isFinite(gt) || !Number.isFinite(pred)) return "";
  const d = Math.abs(gt - pred);
  if (d === 0) return "✓";
  if (d === 1) return "≈";
  return "✗";
}

function pill(text, color) {
  return `<span style="display:inline-block;padding:1px 8px;border-radius:10px;
    font-size:12px;font-weight:600;letter-spacing:0.02em;
    background:${color}1f;color:${color};border:1px solid ${color}66">${text}</span>`;
}

function ratingPill(r) {
  if (!Number.isFinite(r)) return `<span style="color:#666">—</span>`;
  return pill(String(r), RATING_COLORS[r] || COLOR_MUTED);
}

// ─── build this round's punch list ──────────────────────────────────────────

function rebuildPunches(state) {
  ensurePreds(state, () => { rebuildPunches(state); });

  const stem = state.cacheBasename || "";
  const dets = state.labels?.detections;
  const stemChanged = stem !== lastStemForReset;
  const detsChanged = dets !== lastDetectionsRef;
  if (!stemChanged && !detsChanged && punches.length) {
    rebuildSidebar(state);
    return;
  }
  lastStemForReset = stem;
  lastDetectionsRef = dets;

  const N = state.pose?.n_frames ?? 0;
  const next = (dets || [])
    .map(d => {
      const e = preds && d.punch_uuid ? preds.get(d.punch_uuid) : null;
      if (!e) return null;
      return {
        timestamp: d.timestamp,
        hand: d.hand,
        stance: d.stance?.toLowerCase?.() || null,
        punch_type: d.punch_type,
        start_frame: Math.max(0, d.start_frame),
        end_frame: Math.min(N > 0 ? N - 1 : d.end_frame, d.end_frame),
        gt: e.gtRating,
        pred: e.predRating,
        score: e.predScore,
      };
    })
    .filter(Boolean)
    .sort((a, b) => a.start_frame - b.start_frame);

  const hadNone = punches.length === 0;
  punches = next;
  if (stemChanged || (hadNone && next.length)) {
    if (next.length) seekToPunch(0, state);
    else { loopWindow = null; activeIdx = -1; }
  } else if (activeIdx >= next.length) {
    activeIdx = next.length - 1;
  }
  renderPunchTable();
  rebuildSidebar(state);
}

function loopFor(p, state) {
  const fps = state.fps || state.pose?.fps || 30;
  const N = state.pose?.n_frames ?? (p.end_frame + 1);
  const pre = Math.round(LOOP_PRE_SEC * fps);
  const post = Math.round(LOOP_POST_SEC * fps);
  return {
    start_frame: Math.max(0, p.start_frame - pre),
    end_frame: Math.min(N - 1, p.end_frame + post),
  };
}

function seekToPunch(idx, state) {
  if (!punches.length) return;
  idx = Math.max(0, Math.min(punches.length - 1, idx));
  const p = punches[idx];
  activeIdx = idx;
  loopWindow = loopFor(p, state);
  if (videoEl && state.fps) {
    videoEl.currentTime = (state.start_sec || 0) + loopWindow.start_frame / state.fps;
    if (videoEl.paused) {
      const pr = videoEl.play();
      if (pr && typeof pr.catch === "function") pr.catch(() => {});
    }
  }
  rebuildSidebar(state);
}

function installTimeupdateLoop(state) {
  if (!videoEl) return;
  if (timeupdateHandler) videoEl.removeEventListener("timeupdate", timeupdateHandler);
  timeupdateHandler = () => {
    if (state.rule?.id !== "hip_rotation_model") return;
    if (!loopWindow || !state.fps) return;
    const endTime = (state.start_sec || 0) + (loopWindow.end_frame + 0.5) / state.fps;
    if (videoEl.currentTime > endTime) {
      videoEl.currentTime = (state.start_sec || 0) + loopWindow.start_frame / state.fps;
    }
  };
  videoEl.addEventListener("timeupdate", timeupdateHandler);
}

function installKeyHandlers(state) {
  if (keydownHandler) document.removeEventListener("keydown", keydownHandler, true);
  keydownHandler = (e) => {
    if (state.rule?.id !== "hip_rotation_model") return;
    const tag = e.target?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    if (e.key === "n" || e.key === "N") { e.preventDefault(); seekToPunch(activeIdx + 1, latestState); return; }
    if (e.key === "p" || e.key === "P") { e.preventDefault(); seekToPunch(activeIdx - 1, latestState); return; }
    if (e.key === "m" || e.key === "M") { e.preventDefault(); toggleMute(); return; }

    if (!loopWindow) return;
    let delta = 0;
    if      (e.key === "ArrowLeft")  delta = -1;
    else if (e.key === "ArrowRight") delta = +1;
    else if (e.key === "[")          delta = -10;
    else if (e.key === "]")          delta = +10;
    else return;
    const f = latestState?.frame ?? 0;
    if (f < loopWindow.start_frame || f > loopWindow.end_frame) return;
    const target = f + delta;
    if (target < loopWindow.start_frame || target > loopWindow.end_frame) {
      e.preventDefault();
      e.stopImmediatePropagation();
    }
  };
  document.addEventListener("keydown", keydownHandler, true);
}

// ─── sidebar ────────────────────────────────────────────────────────────────

function buildSidebarSkeleton() {
  if (!host) return;
  host.innerHTML = `
    <h2>Hip rotation model (held-out)</h2>
    <p class="hint">
      The temporal TCN's <b>cross-validated 1–4 rating</b> for each punch,
      against the labelled ground truth. These are <em>out-of-fold</em>
      predictions (each punch scored by a model that never trained on its
      video), so this is the honest generalization — the same numbers the CV
      metrics report. Loaded from
      <code>predictions_hip_rotation_temporal.json</code>; join is by
      <code>punch_uuid</code>. Ratings:
      ${ratingPill(1)} none · ${ratingPill(2)} slight ·
      ${ratingPill(3)} good · ${ratingPill(4)} full. Agreement:
      <span style="color:${COLOR_OK}">✓ exact</span> ·
      <span style="color:${COLOR_NEAR}">≈ off by 1</span> ·
      <span style="color:${COLOR_MISS}">✗ off by 2+</span>.
    </p>
    <div id="hrm-card" class="hint" style="margin:8px 0; padding:8px 10px; border:1px solid #2a2a2a; border-radius:6px; font-family:ui-monospace,monospace; font-size:12px;"></div>
    <div class="ol-nav" style="display:flex; gap:8px; align-items:center; margin:10px 0 8px;">
      <button id="hrm-prev" class="orient-btn-action secondary" style="padding:6px 10px;">⏮ prev (P)</button>
      <button id="hrm-next" class="orient-btn-action secondary" style="padding:6px 10px;">next (N) ⏭</button>
      <button id="hrm-mute" class="orient-btn-action secondary" style="padding:6px 10px;">mute (M)</button>
      <span id="hrm-counter" style="margin-left:6px; color:#888; font-size:12px;"></span>
    </div>
    <div id="hrm-state" class="hint" style="line-height:1.7;"></div>
    <div id="hrm-summary" class="hint" style="margin-top:14px; padding-top:10px; border-top:1px solid #2a2a2a;"></div>
    <div id="hrm-table-wrap" style="margin-top:10px; max-height:360px; overflow-y:auto;"></div>
  `;
  host.querySelector("#hrm-prev")?.addEventListener("click", () => seekToPunch(activeIdx - 1, latestState));
  host.querySelector("#hrm-next")?.addEventListener("click", () => seekToPunch(activeIdx + 1, latestState));
  host.querySelector("#hrm-mute")?.addEventListener("click", toggleMute);
  updateMuteButton();
  renderModelCard();
  renderPunchTable();
}

// Top "model card": where the sidecar came from + the headline CV metrics.
function renderModelCard() {
  const el = host?.querySelector("#hrm-card");
  if (!el) return;
  if (loadErr) { el.innerHTML = `<span style="color:${COLOR_MISS}">sidecar parse error: ${loadErr}</span>`; return; }
  if (!pmeta) {
    el.innerHTML = `<span class="muted">No <code>predictions_hip_rotation_*.json</code> in the connected folder yet.</span>`;
    return;
  }
  const ens = pmeta.metrics?.["TCN (ours, 3-seed ensemble)"];
  let metricLine = "";
  if (ens) {
    const f = (k) => Number.isFinite(ens[k]) ? ens[k].toFixed(3) : "—";
    metricLine = `<br>held-out: exact <b>${f("exact")}</b> · ±1 <b>${f("pm1")}</b> · `
               + `QWK <b>${f("qwk")}</b> · macro-recall <b>${f("macro_recall")}</b>`;
  }
  const np = pmeta.metrics?.n_punches, nv = pmeta.metrics?.n_videos;
  el.innerHTML =
    `<b>${pmeta.model}</b> · ${pmeta.n} punches scored`
    + (np ? ` · ${np} / ${nv} vids` : "")
    + (pmeta.exportedAt ? `<br><span class="muted">exported ${pmeta.exportedAt}</span>` : "")
    + metricLine;
}

function renderPunchTable() {
  if (!host) return;
  const container = host.querySelector("#hrm-table-wrap");
  if (!container) return;
  if (!punches.length) { container.innerHTML = ""; return; }

  const rows = punches.map((p, i) => {
    const typeStr = (p.punch_type || "?").replace(/_/g, " ");
    const tStr = Number.isFinite(p.timestamp) ? p.timestamp.toFixed(2) + "s" : "—";
    const scoreStr = Number.isFinite(p.score) ? p.score.toFixed(2) : "—";
    const col = agreeColor(p.gt, p.pred);
    const sym = agreeSym(p.gt, p.pred);
    return `<tr data-idx="${i}" style="border-bottom:1px solid rgba(255,255,255,0.04);">
      <td style="padding:4px 6px; text-align:right; color:#888;">${i + 1}</td>
      <td style="padding:4px 6px; color:#aaa; font-family:ui-monospace,monospace;">${tStr}</td>
      <td style="padding:4px 6px;">${typeStr}</td>
      <td style="padding:4px 6px; text-align:center;">${ratingPill(p.gt)}</td>
      <td style="padding:4px 6px; text-align:center;">${ratingPill(p.pred)}</td>
      <td style="padding:4px 6px; text-align:right; font-family:ui-monospace,monospace; color:#aaa;">${scoreStr}</td>
      <td style="padding:4px 6px; text-align:center; color:${col}; font-weight:700;">${sym}</td>
    </tr>`;
  }).join("");

  container.innerHTML = `
    <table style="width:100%; border-collapse:collapse; font-size:12px;">
      <thead>
        <tr style="text-align:left; color:#888; border-bottom:1px solid #2a2a2a;">
          <th style="padding:4px 6px; text-align:right; font-weight:600;">#</th>
          <th style="padding:4px 6px; font-weight:600;">t</th>
          <th style="padding:4px 6px; font-weight:600;">type</th>
          <th style="padding:4px 6px; text-align:center; font-weight:600;">GT</th>
          <th style="padding:4px 6px; text-align:center; font-weight:600;">pred</th>
          <th style="padding:4px 6px; text-align:right; font-weight:600;">score</th>
          <th style="padding:4px 6px; text-align:center; font-weight:600;">✓</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;

  container.querySelectorAll("tr[data-idx]").forEach(tr => {
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => seekToPunch(parseInt(tr.getAttribute("data-idx"), 10), latestState));
    tr.addEventListener("mouseenter", () => {
      const idx = parseInt(tr.getAttribute("data-idx"), 10);
      if (idx !== activeIdx) tr.style.background = "rgba(255,255,255,0.04)";
    });
    tr.addEventListener("mouseleave", () => {
      const idx = parseInt(tr.getAttribute("data-idx"), 10);
      if (idx !== activeIdx) tr.style.background = "";
    });
  });
  updateActiveRow();
}

function updateActiveRow() {
  if (!host) return;
  host.querySelectorAll("tr[data-idx]").forEach(tr => {
    const idx = parseInt(tr.getAttribute("data-idx"), 10);
    if (idx === activeIdx) { tr.style.background = "rgba(255,210,74,0.14)"; tr.style.fontWeight = "600"; }
    else { tr.style.background = ""; tr.style.fontWeight = "normal"; }
  });
}

function toggleMute() {
  if (!videoEl) return;
  videoEl.muted = !videoEl.muted;
  updateMuteButton();
}

function updateMuteButton() {
  const btn = host?.querySelector("#hrm-mute");
  if (!btn || !videoEl) return;
  btn.textContent = videoEl.muted ? "unmute (M)" : "mute (M)";
}

function rebuildSidebar(state) {
  if (!host) return;
  latestState = state;
  if (!host.querySelector("#hrm-state")) buildSidebarSkeleton();

  renderModelCard();
  updateActiveRow();

  const counter = host.querySelector("#hrm-counter");
  if (counter) {
    counter.textContent = punches.length
      ? `${activeIdx + 1} / ${punches.length}`
      : "no model-scored punches in this round";
  }

  // Round summary: exact / ±1 agreement + mean |err| across this round.
  const summary = host.querySelector("#hrm-summary");
  if (summary) {
    if (!punches.length) {
      summary.innerHTML = "";
    } else {
      let exact = 0, near = 0, sumAbs = 0, n = 0;
      for (const p of punches) {
        if (!Number.isFinite(p.gt) || !Number.isFinite(p.pred)) continue;
        const d = Math.abs(p.gt - p.pred);
        if (d === 0) exact++;
        if (d <= 1) near++;
        sumAbs += d; n++;
      }
      const pct = (x) => n ? Math.round(100 * x / n) : 0;
      summary.innerHTML = n
        ? `<b>Round agreement (${n} punch${n === 1 ? "" : "es"}):</b> `
          + `${pill(`${exact}/${n} exact (${pct(exact)}%)`, COLOR_OK)} `
          + `${pill(`${near}/${n} ±1 (${pct(near)}%)`, COLOR_NEAR)} `
          + `<span class="muted">· mean |err| ${(sumAbs / n).toFixed(2)}</span>`
        : "";
    }
  }

  const el = host.querySelector("#hrm-state");
  if (!el) return;

  if (!punches.length) {
    el.innerHTML = !preds
      ? `<span class="muted">Waiting for the predictions sidecar…</span>`
      : state.labels?.detections
        ? `<span class="muted">No model-scored punches in this round (none of its labelled punches are in the sidecar).</span>`
        : `<span class="muted">Waiting for label data…</span>`;
    return;
  }

  const p = punches[activeIdx];
  if (!p) { el.innerHTML = ""; return; }
  const col = agreeColor(p.gt, p.pred);
  const sym = agreeSym(p.gt, p.pred);
  const scoreStr = Number.isFinite(p.score) ? p.score.toFixed(3) : "—";
  el.innerHTML = [
    `<b>${p.punch_type.replace(/_/g, " ")}</b> · <code>${p.hand}</code> · stance <code>${p.stance || "?"}</code>`,
    `label window <code>${p.start_frame}-${p.end_frame}</code>`,
    "",
    `<b>GT:</b> ${ratingPill(p.gt)}  &nbsp; <b>pred:</b> ${ratingPill(p.pred)} `
      + `<span style="color:${col}; font-weight:700">${sym}</span>`,
    `pred score (0–1) = <code>${scoreStr}</code>  → snapped to rating <code>${p.pred}</code>`,
  ].join("<br>");
}

// ─── draw ─────────────────────────────────────────────────────────────────

function drawHud(ctx, p, s) {
  const col = agreeColor(p.gt, p.pred);
  const titleTxt = `${p.hand} ${p.punch_type.replace(/_/g, " ")}  ·  ${p.stance || "?"}`;
  const gtTxt    = `GT: ${Number.isFinite(p.gt) ? p.gt : "—"}`;
  const predTxt  = `pred: ${Number.isFinite(p.pred) ? p.pred : "—"}  ${agreeSym(p.gt, p.pred)}`;
  const scoreTxt = `score ${Number.isFinite(p.score) ? p.score.toFixed(2) : "—"}/1.0`;
  const lines = [titleTxt, gtTxt, predTxt, scoreTxt];

  const fontPx = 15 * s, lineH = 22 * s, padX = 14 * s, padY = 10 * s;
  const x0 = 24 * s, y0 = 24 * s;
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
  ctx.fillStyle = RATING_COLORS[p.gt] || COLOR_MUTED; ctx.fillText(gtTxt, x0 + padX, y); y += lineH;
  ctx.fillStyle = col; ctx.fillText(predTxt, x0 + padX, y); y += lineH;
  ctx.fillStyle = "rgba(255,255,255,0.7)"; ctx.fillText(scoreTxt, x0 + padX, y);
  ctx.restore();
}

function drawCanvas(ctx, state) {
  if (!punches.length || activeIdx < 0) return;
  const p = punches[activeIdx];
  const s = state.renderScale || 1;
  const pose = state.pose;
  if (!pose) return;
  const f = state.frame;

  // Current-frame hip line for visual context (the model has no per-frame
  // signal to draw — the rating is per-punch).
  const lc = pose.conf[f * 17 + J.L_HIP];
  const rc = pose.conf[f * 17 + J.R_HIP];
  if (lc >= 0.05 && rc >= 0.05) {
    const lx = pose.skeleton[(f * 17 + J.L_HIP) * 2];
    const ly = pose.skeleton[(f * 17 + J.L_HIP) * 2 + 1];
    const rx = pose.skeleton[(f * 17 + J.R_HIP) * 2];
    const ry = pose.skeleton[(f * 17 + J.R_HIP) * 2 + 1];
    ctx.save();
    ctx.strokeStyle = COLOR_HIP; ctx.fillStyle = COLOR_HIP;
    ctx.lineWidth = 3 * s; ctx.globalAlpha = 0.85;
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(rx, ry); ctx.stroke();
    ctx.beginPath(); ctx.arc(lx, ly, 5 * s, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(rx, ry, 5 * s, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }

  drawHud(ctx, p, s);
}

// ─── lens contract ──────────────────────────────────────────────────────────

export const HipRotationModelRule = {
  id: "hip_rotation_model",
  label: "Hip rotation model (GT vs pred)",

  // Apple Vision is the production pose source the model was trained on.
  requires(slot) {
    return !!slot?.vision;
  },

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
    lastDetectionsRef = null;
    lastStemForReset = null;
    rebuildPunches(state);
    installTimeupdateLoop(state);
    installKeyHandlers(state);
  },

  update(state) {
    rebuildPunches(state);
    const TOL = 5;
    if (punches.length) {
      const f = state.frame;
      const active = activeIdx >= 0 ? punches[activeIdx] : null;
      const lw = loopWindow;
      const nearActive = active && lw &&
        f >= lw.start_frame - TOL && f <= lw.end_frame + TOL;
      if (!nearActive) {
        const inside = punches.findIndex(p => {
          const w = loopFor(p, state);
          return f >= w.start_frame && f <= w.end_frame;
        });
        if (inside !== -1 && inside !== activeIdx) {
          activeIdx = inside;
          loopWindow = loopFor(punches[inside], state);
          rebuildSidebar(state);
        }
      }
    }
  },

  draw(ctx, state) {
    drawCanvas(ctx, state);
  },

  unmount() {
    if (videoEl && timeupdateHandler) videoEl.removeEventListener("timeupdate", timeupdateHandler);
    if (keydownHandler) document.removeEventListener("keydown", keydownHandler, true);
    timeupdateHandler = null;
    keydownHandler = null;
    loopWindow = null;
  },
};
