// Shoulder broadside gate lens.
//
// The elbow-flare metric is only valid when we can SEE the axis the elbow
// flares along — the shoulder line — i.e. when the shoulder line is broadside
// to the camera (the coronal plane is roughly fronto-parallel). This lens
// computes that gate per frame and shows whether the frame QUALIFIES.
//
// It is deliberately NOT a hip / "facing" gate: flare is a shoulder/upper-arm
// quantity, so a bladed stance whose shoulders are broadside still qualifies,
// and a square stance turned sideways still gets rejected. We only ask one
// thing: is the shoulder line across the view, or pointing at the camera?
//
// Two estimators, shown side by side so we can judge how much to trust z:
//
//   3D angle (uses z):  φ3d = asin(|unit(R_sh−L_sh) · unit(cam − shoulder_mid)|)
//     0° = shoulder line perpendicular to the camera ray (broadside, good),
//     90° = shoulder line pointing straight at the camera (foreshortened).
//     Uses pose3d.xyz + camMatrices. z is mediocre, so it's rolling-median
//     smoothed over time (orientation is slow).
//
//   2D proxy (z-free):  φproxy = acos(clamp( (shoulderW/torso)_f / p95 , 0,1))
//     Per-person calibrated: p95 of shoulderW/torso over the round ≈ the
//     fighter's broadest (most-frontal) shoulder span. No depth needed.
//
// QUALIFIES = chosen estimator's angle < threshold. Frames past threshold are
// dropped (the metric simply can't be measured there), never guessed.

import { J } from "../skeleton.js";
import { J3 } from "../skeleton-3d.js";

const cfg = {
  thresholdDeg: 35,       // shoulder line within this of broadside ⇒ qualifies
  smoothSeconds: 0.5,     // rolling-median half-window for the noisy 3D angle
  minConfidence: 0.5,
  mode: "auto",           // "auto" | "3d" | "proxy"  (auto = 3d when available)
};

const REQUIRED_2D = [J.L_SHOULDER, J.R_SHOULDER, J.L_HIP, J.R_HIP];

const COLOR_PASS    = "#7adf7a";
const COLOR_REJECT  = "#ff5d6c";
const COLOR_INVALID = "#888";
const COLOR_FRAME   = "#3ad9e0";
const COLOR_3D      = "#7ec8ff";
const COLOR_PROXY   = "#ffd95c";

// ── geometry ────────────────────────────────────────────────────────────────

function torsoHeight2d(sk, f) {
  const b = f * 17;
  const sx = 0.5 * (sk[(b + J.L_SHOULDER) * 2]     + sk[(b + J.R_SHOULDER) * 2]);
  const sy = 0.5 * (sk[(b + J.L_SHOULDER) * 2 + 1] + sk[(b + J.R_SHOULDER) * 2 + 1]);
  const hx = 0.5 * (sk[(b + J.L_HIP) * 2]          + sk[(b + J.R_HIP) * 2]);
  const hy = 0.5 * (sk[(b + J.L_HIP) * 2 + 1]      + sk[(b + J.R_HIP) * 2 + 1]);
  return Math.hypot(sx - hx, sy - hy);
}

function shoulderWidth2d(sk, f) {
  const b = f * 17;
  return Math.hypot(
    sk[(b + J.L_SHOULDER) * 2]     - sk[(b + J.R_SHOULDER) * 2],
    sk[(b + J.L_SHOULDER) * 2 + 1] - sk[(b + J.R_SHOULDER) * 2 + 1],
  );
}

function valid2d(conf, f) {
  for (const j of REQUIRED_2D) if (!(conf[f * 17 + j] > cfg.minConfidence)) return false;
  return true;
}

function xyz3(pose3d, f, j) {
  const b = (f * 17 + j) * 3;
  return [pose3d.xyz[b], pose3d.xyz[b + 1], pose3d.xyz[b + 2]];
}

// Camera position in body-frame metres (row-major 4x4 translation column).
function camPos(pose3d, f) {
  if (!pose3d.camMatrices) return null;
  const m = f * 16;
  const c = [pose3d.camMatrices[m + 3], pose3d.camMatrices[m + 7], pose3d.camMatrices[m + 11]];
  return Number.isFinite(c[0]) ? c : null;
}

// φ3d in degrees, or NaN if 3D unavailable for this frame.
function broadside3d(pose3d, f) {
  const L = xyz3(pose3d, f, J3.L_SHOULDER);
  const R = xyz3(pose3d, f, J3.R_SHOULDER);
  const C = camPos(pose3d, f);
  if (!C || !Number.isFinite(L[0]) || !Number.isFinite(R[0])) return NaN;
  const sh = [R[0] - L[0], R[1] - L[1], R[2] - L[2]];
  const shLen = Math.hypot(sh[0], sh[1], sh[2]);
  if (!(shLen > 1e-6)) return NaN;
  const mid = [(L[0] + R[0]) / 2, (L[1] + R[1]) / 2, (L[2] + R[2]) / 2];
  const ray = [C[0] - mid[0], C[1] - mid[1], C[2] - mid[2]];
  const rayLen = Math.hypot(ray[0], ray[1], ray[2]);
  if (!(rayLen > 1e-6)) return NaN;
  let dot = (sh[0] * ray[0] + sh[1] * ray[1] + sh[2] * ray[2]) / (shLen * rayLen);
  dot = Math.min(1, Math.abs(dot));
  return Math.asin(dot) * 180 / Math.PI;
}

function rollingMedian(xs, halfWin) {
  if (halfWin <= 0) return xs.slice();
  const n = xs.length, out = new Array(n).fill(NaN);
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - halfWin), hi = Math.min(n - 1, i + halfWin);
    const v = [];
    for (let k = lo; k <= hi; k++) if (Number.isFinite(xs[k])) v.push(xs[k]);
    if (!v.length) continue;
    v.sort((a, b) => a - b);
    const m = v.length >> 1;
    out[i] = v.length % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
  }
  return out;
}

function percentile(xs, p) {
  const v = xs.filter(Number.isFinite).sort((a, b) => a - b);
  if (!v.length) return NaN;
  return v[Math.min(v.length - 1, Math.max(0, Math.floor(p * (v.length - 1))))];
}

// ── per-pose memo ───────────────────────────────────────────────────────────

let cache = { pose: null, pose3d: null, fps: 0, smooth: -1 };

function pickPose(state) { return state.poseV6 || state.pose; }

function compute(state) {
  const pose = pickPose(state);
  if (!pose) return null;
  const pose3d = state.pose3d || null;
  const halfWin = Math.max(0, Math.round(cfg.smoothSeconds * state.fps));
  if (cache.pose === pose && cache.pose3d === pose3d && cache.fps === state.fps && cache.smooth === halfWin) {
    return cache;
  }

  const n = pose.n_frames, sk = pose.skeleton, conf = pose.conf;
  const ratio = new Array(n).fill(NaN);   // shoulderW / torso (2D)
  const v2d = new Array(n).fill(false);
  const phi3d = new Array(n).fill(NaN);
  const has3d = !!pose3d;

  for (let f = 0; f < n; f++) {
    const th = torsoHeight2d(sk, f);
    if (th > 1e-6) { ratio[f] = shoulderWidth2d(sk, f) / th; v2d[f] = valid2d(conf, f); }
    if (has3d) phi3d[f] = broadside3d(pose3d, f);
  }

  const p95 = percentile(ratio.filter((r, f) => v2d[f]), 0.95);
  const phiProxy = ratio.map(r =>
    Number.isFinite(r) && Number.isFinite(p95) && p95 > 1e-6
      ? Math.acos(Math.min(1, Math.max(0, r / p95))) * 180 / Math.PI : NaN);
  const phi3dSmooth = rollingMedian(phi3d, halfWin);

  cache = { pose, pose3d, fps: state.fps, smooth: halfWin, n, ratio, v2d, p95, phiProxy, phi3d, phi3dSmooth, has3d };
  return cache;
}

function activeMode(c) {
  if (cfg.mode === "3d") return "3d";
  if (cfg.mode === "proxy") return "proxy";
  return c.has3d ? "3d" : "proxy";   // auto
}

// "pass" | "reject" | "invalid" for a frame under the active mode.
function frameState(c, f) {
  const mode = activeMode(c);
  if (mode === "3d") {
    const a = c.phi3dSmooth[f];
    if (!Number.isFinite(a)) return "invalid";
    return a < cfg.thresholdDeg ? "pass" : "reject";
  }
  if (!c.v2d[f] || !Number.isFinite(c.phiProxy[f])) return "invalid";
  return c.phiProxy[f] < cfg.thresholdDeg ? "pass" : "reject";
}

function stateColor(s) { return s === "pass" ? COLOR_PASS : s === "reject" ? COLOR_REJECT : COLOR_INVALID; }
function fmt(x, d = 1) { return Number.isFinite(x) ? x.toFixed(d) : "—"; }

function coverage(c) {
  let pass = 0, considered = 0;
  for (let f = 0; f < c.n; f++) {
    const s = frameState(c, f);
    if (s === "invalid") continue;
    considered++; if (s === "pass") pass++;
  }
  return { pass, considered, pct: considered ? 100 * pass / considered : 0 };
}

// ── lens ────────────────────────────────────────────────────────────────────

let host;

function refresh() { document.getElementById("video")?.dispatchEvent(new Event("seeked")); }

export const ShoulderGateRule = {
  id: "shoulder_gate_lens",
  label: "Shoulder gate (broadside)",

  skeletonStyle() {
    return {
      boneColor: "rgba(255,255,255,0.22)",
      boneWidth: 1.5,
      jointRadius: 3,
      highlightJoints: new Set([J.L_SHOULDER, J.R_SHOULDER]),
    };
  },

  mount(_host) {
    host = _host;
    cache = { pose: null, pose3d: null, fps: 0, smooth: -1 };
    host.innerHTML = `
      <h2>Shoulder gate — broadside</h2>
      <p class="hint">
        Is the shoulder line across the view (<span style="color:${COLOR_PASS}">qualifies</span>)
        or pointing at the camera (<span style="color:${COLOR_REJECT}">rejected</span>)? This is
        the validity gate for elbow-flare — shoulders only, no hips, so a bladed stance with
        broadside shoulders still passes.
        <span style="color:${COLOR_3D}">3D angle</span> uses z;
        <span style="color:${COLOR_PROXY}">2D proxy</span> is z-free (shoulderW/torso vs its p95).
      </p>

      <label class="slider-row" style="display:block; font-size:12px; margin-top:6px">
        broadside threshold = <output id="sg-thr-out">35</output>°
        <input type="range" id="sg-thr" min="10" max="70" step="1" value="35"></label>
      <label class="slider-row" style="display:block; font-size:12px">
        3D smoothing = <output id="sg-sm-out">0.50</output>s
        <input type="range" id="sg-sm" min="0" max="1.5" step="0.05" value="0.5"></label>
      <label class="slider-row" style="display:block; font-size:12px">gate on:
        <select id="sg-mode" style="font-size:12px">
          <option value="auto">auto (3D if present)</option>
          <option value="3d">3D angle (z)</option>
          <option value="proxy">2D proxy (z-free)</option>
        </select></label>

      <h3>Round</h3>
      <div id="sg-round" style="font-size:13px; line-height:1.6"></div>

      <h3>Current frame</h3>
      <div id="sg-frame" style="font-size:13px; line-height:1.6"></div>

      <h3><span style="color:${COLOR_3D}">3D</span> / <span style="color:${COLOR_PROXY}">proxy</span> angle over time</h3>
      <canvas id="sg-trace" width="320" height="120"></canvas>
    `;
    mountTimeline();

    const thr = host.querySelector("#sg-thr"), thrOut = host.querySelector("#sg-thr-out");
    thr.addEventListener("input", () => { cfg.thresholdDeg = parseInt(thr.value); thrOut.textContent = thr.value; refresh(); });
    const sm = host.querySelector("#sg-sm"), smOut = host.querySelector("#sg-sm-out");
    sm.addEventListener("input", () => { cfg.smoothSeconds = parseFloat(sm.value); smOut.textContent = cfg.smoothSeconds.toFixed(2); refresh(); });
    const mode = host.querySelector("#sg-mode");
    mode.value = cfg.mode;
    mode.addEventListener("change", () => { cfg.mode = mode.value; refresh(); });
  },

  update(state) {
    if (!host || !state) return;
    const c = compute(state);
    if (!c) { host.querySelector("#sg-round").innerHTML = `<p class="muted">No pose cache loaded.</p>`; return; }
    const f = state.frame;
    const cov = coverage(c);
    const mode = activeMode(c);

    host.querySelector("#sg-round").innerHTML = `
      <div>gate: <code>${mode === "3d" ? "3D angle (z)" : "2D proxy"}</code>
        ${cfg.mode === "auto" ? `<span class="muted">(auto)</span>` : ""}
        ${c.has3d ? "" : `<span style="color:${COLOR_REJECT}"> · no 3D cache</span>`}</div>
      <div>qualifies <code>${cov.pct.toFixed(1)}%</code>
        <span class="muted">(${cov.pass}/${cov.considered} frames, thr ${cfg.thresholdDeg}°)</span></div>
      <div class="muted" style="font-size:12px">p95 shoulderW/torso = <code>${fmt(c.p95, 2)}</code></div>`;

    const s = frameState(c, f);
    host.querySelector("#sg-frame").innerHTML = `
      <strong>frame ${f}:</strong>
      <span style="color:${stateColor(s)}; font-weight:700; text-transform:uppercase">${s === "pass" ? "QUALIFIES" : s}</span><br>
      <span style="color:${COLOR_3D}">φ3d</span> <code>${fmt(c.phi3dSmooth[f])}</code>°
        <span class="muted">(raw ${fmt(c.phi3d[f])}°)</span> ·
      <span style="color:${COLOR_PROXY}">φproxy</span> <code>${fmt(c.phiProxy[f])}</code>°<br>
      <span class="muted">shoulderW/torso <code>${fmt(c.ratio[f], 2)}</code></span>`;

    drawTrace(host.querySelector("#sg-trace"), c, f);
    drawTimeline(document.getElementById("sg-timeline"), c, f);
  },

  draw(ctx, state) {
    const c = compute(state);
    if (!c) return;
    const pose = pickPose(state);
    const f = state.frame, s = state.renderScale || 1, b = f * 17;
    const lx = pose.skeleton[(b + J.L_SHOULDER) * 2], ly = pose.skeleton[(b + J.L_SHOULDER) * 2 + 1];
    const rx = pose.skeleton[(b + J.R_SHOULDER) * 2], ry = pose.skeleton[(b + J.R_SHOULDER) * 2 + 1];

    if ([lx, ly, rx, ry].every(Number.isFinite)) {
      const color = stateColor(frameState(c, f));
      ctx.save();
      ctx.strokeStyle = color; ctx.lineWidth = 4 * s; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(rx, ry); ctx.stroke();
      ctx.fillStyle = color;
      for (const [x, y] of [[lx, ly], [rx, ry]]) { ctx.beginPath(); ctx.arc(x, y, 4 * s, 0, Math.PI * 2); ctx.fill(); }
      ctx.restore();
    }

    const mode = activeMode(c);
    const fsz = Math.round(13 * s), lineH = fsz + 4 * s;
    const lines = [
      [`gate ${mode === "3d" ? "3D" : "proxy"}`, "#fff"],
      [`φ3d   ${fmt(c.phi3dSmooth[f])}`, COLOR_3D],
      [`φprx  ${fmt(c.phiProxy[f])}`, COLOR_PROXY],
      [`thr   ${cfg.thresholdDeg}`, "#fff"],
      [frameState(c, f) === "pass" ? "QUALIFIES" : frameState(c, f).toUpperCase(), stateColor(frameState(c, f))],
    ];
    const padX = 10 * s, padY = 8 * s, boxW = 132 * s;
    const boxH = lines.length * lineH + padY * 2 - 4 * s;
    const bx = ctx.canvas.width - boxW - 10 * s, by = 10 * s;
    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.beginPath(); ctx.roundRect(bx, by, boxW, boxH, 6 * s); ctx.fill();
    ctx.font = `${fsz}px ui-monospace, monospace`; ctx.textBaseline = "top";
    lines.forEach(([t, col], i) => { ctx.fillStyle = col; ctx.fillText(t, bx + padX, by + padY + i * lineH); });
    ctx.restore();
  },
};

// ── below-video timeline (3D / proxy tracks, click to seek) ─────────────────

const TL_LABEL_W = 56;

function mountTimeline() {
  const slot = document.getElementById("stage-extras");
  if (!slot) return;
  slot.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.style.cssText = "margin-top:12px;padding:10px 12px;background:var(--bg-card);border:1px solid var(--border);border-radius:8px";
  const label = document.createElement("div");
  label.className = "muted small";
  label.style.cssText = "margin-bottom:6px";
  label.textContent = "Shoulder gate — qualifies (green) / rejected (red) / no data (grey) · click to seek";
  wrap.appendChild(label);
  const canvas = document.createElement("canvas");
  canvas.id = "sg-timeline";
  canvas.style.cssText = "display:block;width:100%;height:70px";
  canvas.width = 800; canvas.height = 70;
  wrap.appendChild(canvas);
  slot.appendChild(wrap);

  canvas.addEventListener("click", e => {
    const N = cache?.n; if (!N) return;
    const rect = canvas.getBoundingClientRect();
    const ratio = (e.clientX - rect.left - TL_LABEL_W) / Math.max(1, rect.width - TL_LABEL_W - 4);
    const slider = document.getElementById("scrubber");
    if (!slider) return;
    slider.value = Math.max(0, Math.min(N - 1, Math.round(ratio * (N - 1))));
    slider.dispatchEvent(new Event("input"));
  });
}

// Color a frame for a given estimator's angle array (independent of active mode).
function angleState(angle, validOk) {
  if (!Number.isFinite(angle) || validOk === false) return "invalid";
  return angle < cfg.thresholdDeg ? "pass" : "reject";
}

function drawTimeline(canvas, c, frame) {
  if (!canvas) return;
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const cssW = Math.max(1, canvas.getBoundingClientRect().width);
  const cssH = Math.max(1, canvas.getBoundingClientRect().height);
  if (canvas.width !== Math.round(cssW * dpr))  canvas.width  = Math.round(cssW * dpr);
  if (canvas.height !== Math.round(cssH * dpr)) canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = cssW, H = cssH;
  ctx.clearRect(0, 0, W, H);
  const N = c.n; if (!N) return;

  const xOf = f => TL_LABEL_W + (f / Math.max(1, N - 1)) * (W - TL_LABEL_W - 4);
  const colW = Math.max(1, (W - TL_LABEL_W - 4) / Math.max(1, N - 1));
  const tracks = [
    { label: "3D",    color: COLOR_3D,    get: f => angleState(c.phi3dSmooth[f], true) },
    { label: "proxy", color: COLOR_PROXY, get: f => angleState(c.phiProxy[f], c.v2d[f]) },
  ];
  const gap = 6, top = 4;
  const trackH = Math.floor((H - top * 2 - gap) / tracks.length);
  ctx.font = "10px ui-monospace, monospace";
  tracks.forEach((t, i) => {
    const y = top + i * (trackH + gap);
    ctx.fillStyle = t.color; ctx.fillText(t.label, 6, y + trackH / 2 + 3);
    for (let f = 0; f < N; f++) {
      ctx.fillStyle = stateColor(t.get(f)); ctx.globalAlpha = 0.9;
      ctx.fillRect(xOf(f), y, colW + 0.5, trackH);
    }
    ctx.globalAlpha = 1;
  });
  ctx.strokeStyle = COLOR_FRAME; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(xOf(frame), 1); ctx.lineTo(xOf(frame), H - 1); ctx.stroke();
}

function drawTrace(canvas, c, frame) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const N = c.n; if (!N) return;
  const maxV = 90;
  const yOf = v => H - 4 - (v / maxV) * (H - 12);
  const xOf = f => (f / Math.max(1, N - 1)) * W;

  // threshold line
  ctx.strokeStyle = "rgba(255,255,255,0.45)"; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(0, yOf(cfg.thresholdDeg)); ctx.lineTo(W, yOf(cfg.thresholdDeg)); ctx.stroke();
  ctx.setLineDash([]);

  const dots = (arr, color) => {
    for (let f = 0; f < N; f++) { const v = arr[f]; if (!Number.isFinite(v)) continue; ctx.fillStyle = color; ctx.fillRect(xOf(f) - 0.5, yOf(v) - 1, 2, 2); }
  };
  dots(c.phiProxy, COLOR_PROXY);
  dots(c.phi3dSmooth, COLOR_3D);

  ctx.strokeStyle = COLOR_FRAME; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(xOf(frame), 0); ctx.lineTo(xOf(frame), H); ctx.stroke();
}
