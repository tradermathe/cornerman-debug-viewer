// BlazePose-33 inspector — the "completely check out BlazePose" lens.
//
// Draws the FULL 33-joint skeleton (body + feet + hands) on the video and, in
// the side panel, lists per-joint visibility / presence / z (image + world-m)
// for the current frame. Reads state.blaze33 — the full (N,33,8) production
// cache loaded in viewer.js — NOT the COCO-17 remap that engine_compare uses,
// so nothing BlazePose returns is dropped (feet, z, world-3D, presence).
//
// state.blaze33.data channels per joint:
//   0 x   1 y   2 z   3 x_world_m   4 y_world_m   5 z_world_m   6 visibility   7 presence
// x,y are image-normalised (0..1); we de-normalise to video px for the overlay.

import { SCHEMAS, GROUPS } from "./_skeleton_schemas.js";

const SCH = SCHEMAS.blazepose33;
const NAMES = SCH.names, TAGS = SCH.point_tags, EDGES = SCH.edges;

const X = 0, Y = 1, Z = 2, ZW = 5, VIS = 6, PRES = 7, CH = 8, J = 33;

// Region → colour (blazepose33 only uses body / feet / hand_anchor / hand_other).
const TAG_COLOR = {
  body: "#5fd1ff", feet: "#5fd97a", hand_anchor: "#ff8a3c", hand_other: "#e040fb",
};
const BIG = new Set(["left_wrist", "right_wrist", "left_ankle", "right_ankle", "nose"]);

let host = null;
let rowEls = [];                 // 33 <tr>, built once in mount()
// Per-joint overlay visibility is the real gate — each joint has its own
// checkbox in the table. The region checkboxes are bulk select/deselect over a
// region's joints; default-on regions seed the initial selection.
const DEFAULT_ON_TAGS = new Set(GROUPS.filter(g => g[2]).map(g => g[0]));  // body+feet+hand_anchor
let jointOn = new Set();
for (let j = 0; j < J; j++) if (DEFAULT_ON_TAGS.has(TAGS[j])) jointOn.add(j);
let regionCbs = {};              // tag → region checkbox, for two-way sync
let THR = 0;                     // visibility gate (0 = draw every joint)
let showLabels = false;
let hoverJoint = -1;             // table-hovered joint → ringed in the overlay

function b33(state) { return state.blaze33 || null; }
function forceRedraw() { window.__viewerRedraw?.(); }

// Map the displayed instant to this cache's frame. It's the same extraction as
// the COCO-17 blazepose engine, so usually frame-aligned with the video; fall
// back to PTS/time mapping when BlazePose isn't the primary timeline.
function frameOf(b, state) {
  const aligned = b.n_frames === state.n_frames
    && Math.abs((b.fps || 0) - (state.fps || 0)) < 0.01
    && Math.abs((b.start_sec || 0) - (state.start_sec || 0)) < 1e-3;
  if (aligned) return Math.min(Math.max(state.frame, 0), b.n_frames - 1);
  const v = (typeof document !== "undefined") && document.getElementById("video");
  const t = (v && isFinite(v.currentTime)) ? v.currentTime
    : (state.start_sec || 0) + state.frame / (state.fps || 30);
  if (b.pts && b.pts.length) {
    let lo = 0, hi = b.pts.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (b.pts[m] < t) lo = m + 1; else hi = m; }
    if (lo > 0 && Math.abs(b.pts[lo - 1] - t) <= Math.abs(b.pts[lo] - t)) lo--;
    return lo;
  }
  const f = Math.round((t - (b.start_sec || 0)) * (b.fps || 30));
  return (f >= 0 && f < b.n_frames) ? f : null;
}

// 0..1 → red→amber→green for the vis/pres cells; grey for NaN/undetected.
function grade(v) {
  if (v !== v) return "#666";
  const h = Math.max(0, Math.min(1, v)) * 120;
  return `hsl(${h.toFixed(0)} 70% 55%)`;
}
function fmt(v, d = 2) { return (v !== v) ? "—" : v.toFixed(d); }

function setJoint(j, on) { if (on) jointOn.add(j); else jointOn.delete(j); }

// Reflect jointOn into the per-joint checkboxes (+ dim rows that are off).
function syncRowChecks() {
  for (let j = 0; j < J; j++) {
    const tr = rowEls[j];
    if (!tr || !tr.__check) continue;
    tr.__check.checked = jointOn.has(j);
    tr.style.opacity = jointOn.has(j) ? "" : "0.4";
  }
}

// A region box is checked when all its joints are on, indeterminate when some.
function syncRegionChecks() {
  for (const tag in regionCbs) {
    let on = 0, total = 0;
    for (let j = 0; j < J; j++) if (TAGS[j] === tag) { total++; if (jointOn.has(j)) on++; }
    const cb = regionCbs[tag];
    cb.checked = total > 0 && on === total;
    cb.indeterminate = on > 0 && on < total;
  }
}

// Fill per-joint cells for the current frame (no DOM rebuild → smooth playback).
function renderValues(state) {
  const b = b33(state);
  const fl = host?.querySelector("#bp-frame");
  if (!b) {
    if (fl) fl.innerHTML = `<span class="muted" style="color:var(--bad)">No BlazePose-33 cache loaded for this round.</span>`;
    return;
  }
  const f = frameOf(b, state);
  if (fl) {
    fl.innerHTML = f == null ? `<span class="muted">no aligned frame</span>`
      : `frame <b>${f}</b> / ${b.n_frames - 1} · 33 joints · ${b.width}×${b.height}`;
  }
  const base = f == null ? -1 : f * J * CH;
  for (let j = 0; j < J; j++) {
    const tr = rowEls[j];
    if (!tr) continue;
    const cells = tr.__cells;
    if (base < 0) { cells.vis.textContent = cells.pres.textContent = cells.z.textContent = cells.zw.textContent = "—"; continue; }
    const o = base + j * CH;
    const vis = b.data[o + VIS], pres = b.data[o + PRES], z = b.data[o + Z], zw = b.data[o + ZW];
    cells.vis.textContent = fmt(vis);  cells.vis.style.color = grade(vis);
    cells.pres.textContent = fmt(pres); cells.pres.style.color = grade(pres);
    cells.z.textContent = fmt(z, 3);
    cells.zw.textContent = fmt(zw, 3);
    tr.style.background = (j === hoverJoint) ? "#2c2c1a" : "";
  }
}

export const BlazePoseInspectorRule = {
  id: "blazepose_inspector",
  label: "BlazePose inspector (33-joint)",

  // Only meaningful when a BlazePose cache exists for the round.
  requires(slot) { return !!(slot && slot.blazepose); },

  // We draw the full 33-joint skeleton ourselves — suppress the base renderer.
  skeletonStyle() {
    return { boneColor: "rgba(0,0,0,0)", jointRadius: 0, minConf: Infinity, showImputed: false };
  },

  mount(_host, state) {
    host = _host;
    rowEls = [];
    host.innerHTML = `
      <h2>BlazePose-33 inspector</h2>
      <p class="hint">Full 33-joint skeleton + per-joint <b>visibility</b>,
        <b>presence</b> and <b>z</b> (image &amp; world-metres) for the current frame.</p>
      <div class="toggles" id="bp-toggles" style="display:flex;gap:12px;flex-wrap:wrap;font-size:12px;margin-bottom:6px">
        <strong style="color:#aaa">Show:</strong>
      </div>
      <label class="slider-row" style="display:block;font-size:13px;margin:4px 0">
        visibility ≥ <output id="bp-thr-out">${THR.toFixed(2)}</output>
        <input type="range" id="bp-thr" min="0" max="1" step="0.05" value="${THR}">
      </label>
      <label style="display:flex;gap:6px;align-items:center;font-size:12px;margin:2px 0 8px;cursor:pointer">
        <input type="checkbox" id="bp-labels" ${showLabels ? "checked" : ""}> joint-index labels on overlay
      </label>
      <div id="bp-frame" style="font-size:13px;margin-bottom:4px">—</div>
      <div style="max-height:46vh;overflow:auto;border:1px solid #2a2a2a;border-radius:6px">
      <table style="width:100%;border-collapse:collapse;font-size:11px;font-family:ui-monospace,'SF Mono',monospace">
        <thead><tr style="position:sticky;top:0;background:#1a1a1a;color:#aaa;text-align:right">
          <th style="text-align:left;padding:3px 6px">joint</th>
          <th style="padding:3px 6px">vis</th><th style="padding:3px 6px">pres</th>
          <th style="padding:3px 6px">z</th><th style="padding:3px 6px">z·m</th>
        </tr></thead>
        <tbody id="bp-tbody"></tbody>
      </table></div>
      <p class="hint" style="margin-top:8px">vis = visible-vs-occluded · pres = in-frame-vs-cropped (per-joint, BlazePose's
        only confidence signals). z = image-space depth (hip-relative, x-scaled, − toward camera);
        z·m = world depth in metres. Tick a joint's box to show/hide just that joint on
        the overlay; region boxes + all/none bulk-toggle. Hover a row to ring that joint.</p>
    `;

    // Region checkboxes = bulk select/deselect every joint in that region;
    // the per-joint checkboxes in the table give exact control.
    const tg = host.querySelector("#bp-toggles");
    const present = new Set(TAGS);
    for (const [tag, label] of GROUPS) {
      if (!present.has(tag)) continue;
      const lab = document.createElement("label");
      lab.style.cssText = "display:flex;gap:4px;align-items:center;cursor:pointer";
      const cb = document.createElement("input");
      cb.type = "checkbox"; cb.dataset.tag = tag;
      cb.addEventListener("change", () => {
        const on = cb.checked;
        for (let j = 0; j < J; j++) if (TAGS[j] === tag) setJoint(j, on);
        cb.indeterminate = false;
        syncRowChecks();
        forceRedraw();
      });
      regionCbs[tag] = cb;
      lab.appendChild(cb);
      lab.appendChild(document.createTextNode(label));
      tg.appendChild(lab);
    }
    // all / none over every joint.
    const mkBtn = (text, fn) => {
      const b = document.createElement("button");
      b.textContent = text;
      b.style.cssText = "font-size:11px;padding:1px 8px;cursor:pointer;background:#222;color:#ccc;border:1px solid #3a3a3a;border-radius:4px";
      b.addEventListener("click", fn);
      return b;
    };
    tg.appendChild(mkBtn("all", () => { for (let j = 0; j < J; j++) jointOn.add(j); syncRowChecks(); syncRegionChecks(); forceRedraw(); }));
    tg.appendChild(mkBtn("none", () => { jointOn.clear(); syncRowChecks(); syncRegionChecks(); forceRedraw(); }));

    // Build the 33 rows ONCE; updates only set text/colour afterwards.
    const tbody = host.querySelector("#bp-tbody");
    for (let j = 0; j < J; j++) {
      const tr = document.createElement("tr");
      tr.dataset.j = String(j);
      tr.style.borderTop = "1px solid #232323";
      const dot = TAG_COLOR[TAGS[j]] || "#5fd1ff";
      const nameTd = document.createElement("td");
      nameTd.style.cssText = "text-align:left;padding:2px 6px;white-space:nowrap";
      nameTd.innerHTML = `<span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${dot};margin-right:5px;vertical-align:middle"></span>${j} ${NAMES[j]}`;
      // Per-joint show/hide checkbox — exact control over what's drawn.
      const jc = document.createElement("input");
      jc.type = "checkbox";
      jc.checked = jointOn.has(j);
      jc.style.cssText = "margin-right:6px;vertical-align:middle;cursor:pointer";
      jc.addEventListener("change", () => {
        setJoint(j, jc.checked);
        tr.style.opacity = jc.checked ? "" : "0.4";
        syncRegionChecks();
        forceRedraw();
      });
      nameTd.prepend(jc);
      tr.__check = jc;
      tr.style.opacity = jc.checked ? "" : "0.4";
      const mk = () => { const td = document.createElement("td"); td.style.cssText = "text-align:right;padding:2px 6px"; td.textContent = "—"; return td; };
      const vis = mk(), pres = mk(), z = mk(), zw = mk();
      tr.append(nameTd, vis, pres, z, zw);
      tr.__cells = { vis, pres, z, zw };
      tr.addEventListener("mouseenter", () => { hoverJoint = j; forceRedraw(); renderValues(state); });
      tr.addEventListener("mouseleave", () => { hoverJoint = -1; forceRedraw(); renderValues(state); });
      tbody.appendChild(tr);
      rowEls[j] = tr;
    }

    const thr = host.querySelector("#bp-thr"), thrOut = host.querySelector("#bp-thr-out");
    thr.addEventListener("input", () => { THR = parseFloat(thr.value); thrOut.textContent = THR.toFixed(2); forceRedraw(); });
    host.querySelector("#bp-labels").addEventListener("change", e => { showLabels = e.target.checked; forceRedraw(); });

    syncRowChecks();
    syncRegionChecks();
    renderValues(state);
  },

  update(state) {
    if (host) renderValues(state);
  },

  draw(ctx, state) {
    const b = b33(state);
    if (!b) return;
    const f = frameOf(b, state);
    if (f == null) return;
    const s = state.renderScale || 1;
    const w = b.width || state.pose?.width || 1;
    const h = b.height || state.pose?.height || 1;
    const base = f * J * CH;
    const px = j => b.data[base + j * CH + X] * w;
    const py = j => b.data[base + j * CH + Y] * h;
    const vis = j => b.data[base + j * CH + VIS];

    // Bones.
    ctx.lineWidth = 2 * s;
    for (const [a, bb, tag] of EDGES) {
      if (!jointOn.has(a) || !jointOn.has(bb)) continue;
      const va = vis(a), vb = vis(bb);
      if (!(va >= THR) || !(vb >= THR)) continue;
      const ax = px(a), ay = py(a), bx = px(bb), by = py(bb);
      if (ax !== ax || ay !== ay || bx !== bx || by !== by) continue;  // NaN guard
      ctx.strokeStyle = TAG_COLOR[tag] || "#5fd1ff";
      ctx.globalAlpha = 0.5;
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Joints — opacity tracks visibility so weak/occluded joints read as faint.
    ctx.font = `${Math.round(10 * s)}px ui-monospace, "SF Mono", monospace`;
    ctx.textBaseline = "middle";
    for (let j = 0; j < J; j++) {
      if (!jointOn.has(j)) continue;
      const vj = vis(j);
      if (!(vj >= THR)) continue;
      const x = px(j), y = py(j);
      if (x !== x || y !== y) continue;
      const r = (BIG.has(NAMES[j]) ? 7 : TAGS[j] === "hand_other" ? 2.5 : 4) * s;
      ctx.globalAlpha = 0.35 + 0.65 * Math.max(0, Math.min(1, vj));
      ctx.fillStyle = TAG_COLOR[TAGS[j]] || "#5fd1ff";
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
      if (j === hoverJoint) {
        ctx.globalAlpha = 1; ctx.strokeStyle = "#ffe05a"; ctx.lineWidth = 2.5 * s;
        ctx.beginPath(); ctx.arc(x, y, r + 4 * s, 0, Math.PI * 2); ctx.stroke();
      }
      if (showLabels) {
        ctx.globalAlpha = 1; ctx.fillStyle = "#fff";
        ctx.fillText(String(j), x + r + 2 * s, y);
      }
    }
    ctx.globalAlpha = 1;
  },
};
