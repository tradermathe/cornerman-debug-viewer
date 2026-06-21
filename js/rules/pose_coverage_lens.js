// Pose coverage lens — quantitative pose-quality summary for the loaded round.
//
// For every loaded engine (primary, secondary, v6) it reports, over the WHOLE
// round, per body region:
//   • uptime%  — fraction of frames the region's joints are detected at conf ≥ slider
//   • mean conf — average confidence of that region's joints WHEN detected (conf > 0)
//
// This is the production-skeleton (COCO-17) analog of the BlazePose-vs-RTMPose
// bake-off stats. The diagnostic move: drag the confidence slider up — uptime
// that HOLDS = solid tracking; uptime that COLLAPSES = the engine was guessing
// at low confidence. Wrists (the punch signal) and Ankles (stance) are the rows
// that matter most.
//
// Cross-engine caveat: Apple Vision returns 0 when a joint isn't detected,
// while YOLO returns a low-confidence guess — so at low thresholds YOLO's
// uptime looks inflated. Compare engines with the slider raised.

import { JOINT_NAMES } from "../skeleton.js";

// Region groupings over COCO-17. Wrists/Ankles flagged as the key rows.
const REGIONS = [
  ["Head",      [0, 1, 2, 3, 4], false],
  ["Shoulders", [5, 6],          false],
  ["Elbows",    [7, 8],          false],
  ["Wrists",    [9, 10],         true],
  ["Hips",      [11, 12],        false],
  ["Knees",     [13, 14],        false],
  ["Ankles",    [15, 16],        true],
];
const ALL17 = Array.from({ length: 17 }, (_, j) => j);

const C_GREEN = "#5fd97a", C_AMBER = "#f5b945", C_RED = "#e85a5a";

function engineName(e) {
  return e === "yolo_pose" ? "YOLO"
    : e === "apple_vision_2d" ? "Vision"
    : e === "vision_combined" ? "Vision (combined)"
    : (e || "pose");
}

function upColor(u) { return u >= 0.8 ? C_GREEN : u >= 0.5 ? C_AMBER : C_RED; }

// Collect the distinct loaded pose sources (primary, secondary, v6), labelled.
function collect(state) {
  const seen = new Set(), out = [];
  const add = (p, tag) => {
    if (!p || seen.has(p)) return;
    seen.add(p);
    out.push({ pose: p, name: engineName(p.engine) + (tag ? ` ${tag}` : "") });
  };
  add(state.pose);
  add(state.poseSecondary);
  add(state.poseV6, "v6");
  return out;
}

// Whole-round stats for one pose at threshold thr.
function statsForPose(pose, thr) {
  const n = pose.n_frames, conf = pose.conf;
  const up = new Float64Array(17), cSum = new Float64Array(17), cN = new Int32Array(17);
  for (let f = 0; f < n; f++) {
    const base = f * 17;
    for (let j = 0; j < 17; j++) {
      const c = conf[base + j];
      if (c >= thr) up[j]++;
      if (c > 0) { cSum[j] += c; cN[j]++; }
    }
  }
  const jointUptime = new Float64Array(17);
  for (let j = 0; j < 17; j++) jointUptime[j] = n ? up[j] / n : 0;

  const agg = (idxs) => {
    let u = 0, cs = 0, cn = 0;
    for (const j of idxs) { u += jointUptime[j]; cs += cSum[j]; cn += cN[j]; }
    return { uptime: u / idxs.length, conf: cn ? cs / cn : 0 };
  };
  const region = {};
  for (const [name, idxs] of REGIONS) region[name] = agg(idxs);
  region.__all = agg(ALL17);
  return { jointUptime, region, n };
}

let host;
let THR = 0.2;
let memo = { poses: null, thr: -1, computed: null };
let last = { srcs: [], frame: 0, fps: 30 };

function ensureComputed(srcs) {
  const same = memo.poses && memo.poses.length === srcs.length
    && memo.poses.every((p, i) => p === srcs[i].pose);
  if (same && memo.thr === THR) return;
  memo.poses = srcs.map(s => s.pose);
  memo.thr = THR;
  memo.computed = srcs.map(s => ({ name: s.name, engine: s.pose.engine, stats: statsForPose(s.pose, THR) }));
  renderTable();
  renderJoints();
}

function pct(u) { return `<span style="color:${upColor(u)}">${(u * 100).toFixed(0)}%</span>`; }
function confTxt(c) { return c ? `<span class="muted">${c.toFixed(2)}</span>` : `<span class="muted">—</span>`; }

function renderTable() {
  const el = host && host.querySelector("#pc-table");
  if (!el) return;
  const cols = memo.computed || [];
  if (!cols.length) { el.innerHTML = `<p class="muted">No pose cache loaded.</p>`; return; }
  const head = `<tr><th style="text-align:left">Region</th>${cols.map(c =>
    `<th>${c.name}<br><span class="muted" style="font-weight:400">uptime · conf</span></th>`).join("")}</tr>`;
  const rowFor = (label, key, strong) => {
    const cells = cols.map(c => {
      const r = key === "__all" ? c.stats.region.__all : c.stats.region[key];
      return `<td>${pct(r.uptime)} · ${confTxt(r.conf)}</td>`;
    }).join("");
    const w = strong ? "font-weight:700" : "";
    return `<tr style="${w}"><td style="text-align:left;${w}">${label}</td>${cells}</tr>`;
  };
  const rows = REGIONS.map(([name, , key]) => rowFor(name, name, key)).join("");
  const allRow = rowFor("All 17", "__all", false);
  el.innerHTML = `<table class="joint-table" style="width:100%">
    <thead>${head}</thead>
    <tbody>${rows}<tr><td colspan="${cols.length + 1}"><hr style="border:0;border-top:1px solid var(--border);margin:2px 0"></td></tr>${allRow}</tbody>
  </table>`;
}

function renderJoints() {
  const el = host && host.querySelector("#pc-joints");
  if (!el) return;
  const cols = memo.computed || [];
  if (!cols.length) { el.innerHTML = ""; return; }
  const head = `<tr><th style="text-align:left">#</th><th style="text-align:left">Joint</th>${cols.map(c =>
    `<th>${c.name}</th>`).join("")}</tr>`;
  let body = "";
  for (let j = 0; j < 17; j++) {
    const cells = cols.map(c => `<td>${pct(c.stats.jointUptime[j])}</td>`).join("");
    body += `<tr><td class="muted">${j}</td><td>${JOINT_NAMES[j]}</td>${cells}</tr>`;
  }
  el.innerHTML = `<table class="joint-table" style="width:100%"><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

// Per-frame line: how many of 17 joints each engine has at conf ≥ THR right now.
function renderFrameLine() {
  const el = host && host.querySelector("#pc-frame");
  if (!el) return;
  const { srcs, frame } = last;
  if (!srcs.length) { el.innerHTML = ""; return; }
  const parts = srcs.map(s => {
    const base = frame * 17, conf = s.pose.conf;
    let k = 0;
    for (let j = 0; j < 17; j++) if (conf[base + j] >= THR) k++;
    const col = upColor(k / 17);
    return `${s.name} <span style="color:${col};font-weight:600">${k}/17</span>`;
  }).join(" · ");
  el.innerHTML = `<strong>frame ${frame}:</strong> ${parts} <span class="muted">(joints ≥ ${THR.toFixed(2)})</span>`;
}

export const PoseCoverageLensRule = {
  id: "pose_coverage",
  label: "Pose coverage (uptime + confidence)",

  mount(_host, state) {
    host = _host;
    memo = { poses: null, thr: -1, computed: null };
    const srcs = collect(state);
    const dur = state.pose ? (state.pose.n_frames / (state.fps || 30)) : 0;
    host.innerHTML = `
      <h2>Pose coverage</h2>
      <p class="hint">
        Whole-round detection <b>uptime%</b> and <b>mean confidence</b> per region, for each
        loaded engine. Drag the threshold: uptime that <b>holds</b> = solid tracking; uptime that
        <b>collapses</b> = low-confidence guessing. <b>Wrists</b> (punch signal) and <b>Ankles</b>
        (stance) matter most.
      </p>
      <p class="hint" style="color:var(--text-muted,#888)">
        Caveat: Vision returns 0 for an undetected joint; YOLO returns a low-conf guess — so at low
        thresholds YOLO's uptime looks inflated. Compare engines with the slider raised.
      </p>
      <label class="slider-row" style="display:block;font-size:13px;margin:6px 0">
        confidence ≥ <output id="pc-thr-out">${THR.toFixed(2)}</output>
        <input type="range" id="pc-thr" min="0" max="1" step="0.05" value="${THR}">
      </label>
      <div class="muted small" style="margin-bottom:8px">
        ${srcs.length} engine${srcs.length === 1 ? "" : "s"} · ${state.pose ? state.pose.n_frames : 0} frames · ${dur.toFixed(1)}s
      </div>
      <div id="pc-table"></div>
      <h3 style="margin-top:12px">Current frame</h3>
      <div id="pc-frame" style="font-size:13px;line-height:1.6"></div>
      <details style="margin-top:10px">
        <summary style="cursor:pointer;font-size:13px">Per-joint uptime</summary>
        <div id="pc-joints" style="margin-top:6px"></div>
      </details>
    `;
    const slider = host.querySelector("#pc-thr"), out = host.querySelector("#pc-thr-out");
    slider.addEventListener("input", () => {
      THR = parseFloat(slider.value);
      out.textContent = THR.toFixed(2);
      ensureComputed(last.srcs.length ? last.srcs : srcs);
      renderFrameLine();
    });
    last = { srcs, frame: state.frame || 0, fps: state.fps || 30 };
    ensureComputed(srcs);
    renderFrameLine();
  },

  update(state) {
    if (!host) return;
    const srcs = collect(state);
    last = { srcs, frame: state.frame || 0, fps: state.fps || 30 };
    ensureComputed(srcs);
    renderFrameLine();
  },
};
