// Shoulder broadside gate + elbow flare (2D, z-free).
//
// Validity gate for elbow flare: is the shoulder line broadside to the camera
// (so the flare axis is visible)? Shoulders only — no hips — so a bladed stance
// with broadside shoulders still passes, and a square stance turned sideways
// gets rejected. Gate signal is the z-free foreshortening proxy: per-person
// calibrate the broadest shoulder span (p95 of shoulderW/torso ≈ fully square),
// and a frame's turn angle is acos(ratio / p95). Pass when within `threshold`.
//
// Elbow flare (just a number for now): |x_shoulder − x_elbow| / torso, per
// side, on every frame. Known limit: on bladed frames a forward-reaching guard
// leaks into this horizontal number — demo-grade, not production.
//
// Hook exclusion: during a hook the throwing hand's elbow abducts on purpose,
// so that side's flare is excluded within the hook's frame window.

import { J } from "../skeleton.js";
import { activeDetections } from "./_detections.js";

const cfg = {
  thresholdDeg: 35,       // shoulder within this many degrees of broadside ⇒ qualifies
  minConfidence: 0.5,
};

const REQUIRED_2D = [J.L_SHOULDER, J.R_SHOULDER, J.L_HIP, J.R_HIP];

const COLOR_PASS    = "#7adf7a";
const COLOR_REJECT  = "#ff5d6c";
const COLOR_INVALID = "#888";
const COLOR_FRAME   = "#3ad9e0";
const COLOR_PROXY   = "#ffd95c";
const COLOR_FLARE   = "#c0a7ff";
const COLOR_HOOK    = "#ff9e64";

// Hook exclusion: punch `hand` is boxer-relative (lead/rear); map to anatomical
// L/R via stance (default orthodox), the same convention arm_extension uses.
const SIDE_FOR = { lead: { orthodox: "L", southpaw: "R" }, rear: { orthodox: "R", southpaw: "L" } };
const isHook = t => /hook/i.test(t || "");

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

function percentile(xs, p) {
  const v = xs.filter(Number.isFinite).sort((a, b) => a - b);
  if (!v.length) return NaN;
  return v[Math.min(v.length - 1, Math.max(0, Math.floor(p * (v.length - 1))))];
}

// ── per-pose memo ───────────────────────────────────────────────────────────

let cache = { pose: null, dets: null, fps: 0 };

function pickPose(state) { return state.poseV6 || state.pose; }

function compute(state) {
  const pose = pickPose(state);
  if (!pose) return null;
  const dets = activeDetections(state);
  if (cache.pose === pose && cache.dets === dets && cache.fps === state.fps) return cache;

  const n = pose.n_frames, sk = pose.skeleton, conf = pose.conf;
  const ratio = new Array(n).fill(NaN);   // shoulderW / torso
  const v2d = new Array(n).fill(false);
  // Raw horizontal elbow flare, |x_shoulder − x_elbow| / torso, per side, all frames.
  const flareL = new Array(n).fill(NaN);
  const flareR = new Array(n).fill(NaN);

  for (let f = 0; f < n; f++) {
    const th = torsoHeight2d(sk, f);
    if (th > 1e-6) {
      ratio[f] = shoulderWidth2d(sk, f) / th;
      v2d[f] = valid2d(conf, f);
      const b = f * 17;
      flareL[f] = Math.abs(sk[(b + J.L_SHOULDER) * 2] - sk[(b + J.L_ELBOW) * 2]) / th;
      flareR[f] = Math.abs(sk[(b + J.R_SHOULDER) * 2] - sk[(b + J.R_ELBOW) * 2]) / th;
    }
  }

  // Per-side hook exclusion: mark frames where THAT hand is throwing a hook.
  const hookL = new Array(n).fill(false), hookR = new Array(n).fill(false);
  if (dets) {
    const roundStance = state.analysis?.ankleOrientation?.stance;
    for (const d of dets) {
      if (!isHook(d.punch_type)) continue;
      const stance = (d.stance === "southpaw" || d.stance === "orthodox") ? d.stance
                   : (roundStance === "southpaw" || roundStance === "orthodox") ? roundStance
                   : "orthodox";
      const side = SIDE_FOR[d.hand]?.[stance];
      if (!side) continue;
      const s = Math.max(0, Math.round(d.start_frame));
      const e = Math.min(n - 1, Math.round(d.end_frame));
      const arr = side === "L" ? hookL : hookR;
      for (let f = s; f <= e; f++) arr[f] = true;
    }
  }

  const p95 = percentile(ratio.filter((r, f) => v2d[f]), 0.95);
  const phiProxy = ratio.map(r =>
    Number.isFinite(r) && Number.isFinite(p95) && p95 > 1e-6
      ? Math.acos(Math.min(1, Math.max(0, r / p95))) * 180 / Math.PI : NaN);

  cache = { pose, dets, fps: state.fps, n, ratio, v2d, p95, phiProxy, flareL, flareR, hookL, hookR };
  return cache;
}

// "pass" | "reject" | "invalid" for the current frame.
function frameState(c, f) {
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
    cache = { pose: null, dets: null, fps: 0 };
    host.innerHTML = `
      <h2>Shoulder gate — broadside + elbow flare</h2>
      <p class="hint">
        Is the shoulder line across the view (<span style="color:${COLOR_PASS}">qualifies</span>)
        or pointing at the camera (<span style="color:${COLOR_REJECT}">rejected</span>)? Gate is the
        z-free proxy — <span style="color:${COLOR_PROXY}">shoulderW/torso vs its p95</span>. Elbow
        <span style="color:${COLOR_FLARE}">flare</span> = |Δx shoulder→elbow|/torso per side (raw).
        A hand's flare is <span style="color:${COLOR_HOOK}">excluded</span> while it throws a hook.
      </p>

      <label class="slider-row" style="display:block; font-size:12px; margin-top:6px">
        broadside threshold = <output id="sg-thr-out">35</output>°
        <input type="range" id="sg-thr" min="10" max="70" step="1" value="35"></label>

      <h3>Round</h3>
      <div id="sg-round" style="font-size:13px; line-height:1.6"></div>

      <h3>Current frame</h3>
      <div id="sg-frame" style="font-size:13px; line-height:1.6"></div>

      <h3><span style="color:${COLOR_PROXY}">broadside angle</span> over time</h3>
      <canvas id="sg-trace" width="320" height="120"></canvas>
    `;
    mountTimeline();

    const thr = host.querySelector("#sg-thr"), thrOut = host.querySelector("#sg-thr-out");
    thr.addEventListener("input", () => { cfg.thresholdDeg = parseInt(thr.value); thrOut.textContent = thr.value; refresh(); });
  },

  update(state) {
    if (!host || !state) return;
    const c = compute(state);
    if (!c) { host.querySelector("#sg-round").innerHTML = `<p class="muted">No pose cache loaded.</p>`; return; }
    const f = state.frame;
    const cov = coverage(c);
    let hkCntL = 0, hkCntR = 0;
    for (let i = 0; i < c.n; i++) { if (c.hookL[i]) hkCntL++; if (c.hookR[i]) hkCntR++; }

    host.querySelector("#sg-round").innerHTML = `
      <div>qualifies <code>${cov.pct.toFixed(1)}%</code>
        <span class="muted">(${cov.pass}/${cov.considered} frames, thr ${cfg.thresholdDeg}°)</span></div>
      <div class="muted" style="font-size:12px">p95 shoulderW/torso = <code>${fmt(c.p95, 2)}</code>
        · <span style="color:${COLOR_HOOK}">hook-excluded</span>: L ${hkCntL} · R ${hkCntR} frames
        ${c.dets ? "" : "<span class=\"muted\">(no punch data)</span>"}</div>`;

    const hookTag = h => h ? ` <span style="color:${COLOR_HOOK}">⟂hook excl</span>` : "";
    const s = frameState(c, f);
    host.querySelector("#sg-frame").innerHTML = `
      <strong>frame ${f}:</strong>
      <span style="color:${stateColor(s)}; font-weight:700; text-transform:uppercase">${s === "pass" ? "QUALIFIES" : s}</span> ·
      <span style="color:${COLOR_PROXY}">φproxy</span> <code>${fmt(c.phiProxy[f])}</code>°<br>
      <span style="color:${COLOR_FLARE}">elbow flare (|Δx|/torso)</span>:
        L <code>${fmt(c.flareL[f], 2)}</code>${hookTag(c.hookL[f])} ·
        R <code>${fmt(c.flareR[f], 2)}</code>${hookTag(c.hookR[f])}<br>
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

    const fsz = Math.round(13 * s), lineH = fsz + 4 * s;
    const lines = [
      [`φproxy ${fmt(c.phiProxy[f])}`, COLOR_PROXY],
      [`flare ${fmt(c.flareL[f], 2)}${c.hookL[f] ? "h" : ""}/${fmt(c.flareR[f], 2)}${c.hookR[f] ? "h" : ""}`, COLOR_FLARE],
      [`thr   ${cfg.thresholdDeg}`, "#fff"],
      [frameState(c, f) === "pass" ? "QUALIFIES" : frameState(c, f).toUpperCase(), stateColor(frameState(c, f))],
    ];
    if (c.hookL[f] || c.hookR[f]) {
      lines.push([`hook excl ${c.hookL[f] ? "L" : ""}${c.hookR[f] ? "R" : ""}`, COLOR_HOOK]);
    }
    const padX = 10 * s, padY = 8 * s, boxW = 138 * s;
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

// ── below-video timeline (gate track + hook markers, click to seek) ─────────

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
  label.textContent = "Shoulder gate — qualifies (green) / rejected (red) / no data (grey) · hook windows below · click to seek";
  wrap.appendChild(label);
  const canvas = document.createElement("canvas");
  canvas.id = "sg-timeline";
  canvas.style.cssText = "display:block;width:100%;height:64px";
  canvas.width = 800; canvas.height = 64;
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
  const gap = 6, top = 4, hookH = 14;
  const gateH = H - top * 2 - gap - hookH;

  // gate track
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillStyle = COLOR_PASS; ctx.fillText("gate", 6, top + gateH / 2 + 3);
  for (let f = 0; f < N; f++) {
    ctx.fillStyle = stateColor(frameState(c, f)); ctx.globalAlpha = 0.9;
    ctx.fillRect(xOf(f), top, colW + 0.5, gateH);
  }
  ctx.globalAlpha = 1;

  // hook windows (L above, R below within the strip)
  const hookY = top + gateH + gap;
  ctx.fillStyle = COLOR_HOOK; ctx.fillText("hook", 6, hookY + hookH / 2 + 3);
  for (let f = 0; f < N; f++) {
    if (c.hookL[f]) { ctx.fillStyle = COLOR_HOOK; ctx.fillRect(xOf(f), hookY, colW + 0.5, hookH / 2); }
    if (c.hookR[f]) { ctx.fillStyle = COLOR_HOOK; ctx.fillRect(xOf(f), hookY + hookH / 2, colW + 0.5, hookH / 2); }
  }

  ctx.strokeStyle = COLOR_FRAME; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(xOf(frame), 1); ctx.lineTo(xOf(frame), H - 1); ctx.stroke();
}

// Broadside-angle sparkline over the round: threshold line + current-frame marker.
function drawTrace(canvas, c, frame) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const N = c.n; if (!N) return;
  const maxV = 90;
  const yOf = v => H - 4 - (v / maxV) * (H - 12);
  const xOf = f => (f / Math.max(1, N - 1)) * W;

  ctx.strokeStyle = "rgba(255,255,255,0.45)"; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(0, yOf(cfg.thresholdDeg)); ctx.lineTo(W, yOf(cfg.thresholdDeg)); ctx.stroke();
  ctx.setLineDash([]);

  for (let f = 0; f < N; f++) {
    const v = c.phiProxy[f];
    if (!Number.isFinite(v)) continue;
    ctx.fillStyle = COLOR_PROXY;
    ctx.fillRect(xOf(f) - 0.5, yOf(v) - 1, 2, 2);
  }

  ctx.strokeStyle = COLOR_FRAME; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(xOf(frame), 0); ctx.lineTo(xOf(frame), H); ctx.stroke();
}
