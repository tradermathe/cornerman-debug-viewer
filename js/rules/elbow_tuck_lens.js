// Elbow-tuck (width) lens — exploration workbench.
//
// First pass at a "are the elbows tucked?" rule. From a FRONTAL view a flared
// elbow shows up as the elbow sitting far to the side of the shoulder, so the
// signal is the horizontal (image-x) gap between shoulder and elbow, per side,
// normalized by torso height (the house unit, same as sep_ratio):
//
//   flare = |x_shoulder − x_elbow| / torso_height
//
// This lens is NOT a ported rule yet — it has no temporal cleanup / clips. The
// point is to read the flare threshold off real footage before committing to a
// rule. Drag the threshold and watch the on-video brackets + timeline light up.
//
// Caveat baked into the UI: this only means "flare" from the front. When the
// boxer turns side-on the shoulder→elbow x-gap stops measuring tuck, so a
// frontal-view gate (shoulder width / torso) is provided — default OFF so this
// stays "just the width" until we decide how to gate.

import { J } from "../skeleton.js";
import { activeDetections } from "./_detections.js";

// ── tunables (driven by the sidebar sliders) ────────────────────────────────
const cfg = {
  flareThreshold: 0.30,   // flare above this on a valid frame ⇒ flagged
  frontalGate: 0.0,       // shoulderWidth/torso below this ⇒ side-on, excluded (0 = off)
  minConfidence: 0.5,
};

// Joints that must be confident for a frame to count (both arms + torso).
const REQUIRED = [J.L_SHOULDER, J.R_SHOULDER, J.L_ELBOW, J.R_ELBOW, J.L_HIP, J.R_HIP];

const COLOR_FLAG    = "#ff5d6c";  // red — flare > 0.3
const COLOR_WARN    = "#ff9e64";  // orange — flare 0.2–0.3
const COLOR_OK      = "#7adf7a";  // green — flare < 0.2 (tucked)
const COLOR_INVALID = "#888";     // filtered / side-on
const COLOR_HOOK    = "#2e8b57";  // dark green — excluded (hand throwing a hook)
const COLOR_L       = "#7ec8ff";  // left side accents
const COLOR_R       = "#ffd95c";  // right side accents
const COLOR_FRAME   = "#3ad9e0";  // current-frame marker
const COLOR_FRONTAL = "#b48cff";  // frontal-ness strip

// Hook exclusion: punch `hand` is boxer-relative (lead/rear); map to anatomical
// L/R via stance (default orthodox), the same convention arm_extension uses.
const SIDE_FOR = { lead: { orthodox: "L", southpaw: "R" }, rear: { orthodox: "R", southpaw: "L" } };
const isHook = t => /hook/i.test(t || "");

// ── metric core ─────────────────────────────────────────────────────────────

function frameTorsoHeight(skeleton, f) {
  const base = f * 17;
  const sx = 0.5 * (skeleton[(base + J.L_SHOULDER) * 2]     + skeleton[(base + J.R_SHOULDER) * 2]);
  const sy = 0.5 * (skeleton[(base + J.L_SHOULDER) * 2 + 1] + skeleton[(base + J.R_SHOULDER) * 2 + 1]);
  const hx = 0.5 * (skeleton[(base + J.L_HIP) * 2]          + skeleton[(base + J.R_HIP) * 2]);
  const hy = 0.5 * (skeleton[(base + J.L_HIP) * 2 + 1]      + skeleton[(base + J.R_HIP) * 2 + 1]);
  return Math.hypot(sx - hx, sy - hy);
}

function frameValid(conf, f) {
  for (const j of REQUIRED) {
    if (!(conf[f * 17 + j] > cfg.minConfidence)) return false;
  }
  return true;
}

// pose-keyed memo of the per-frame arrays (slider changes don't touch these).
let metricCache = { pose: null, fps: 0 };

function pickPose(state) {
  return state.poseV6 || state.pose;
}

function computeMetrics(state) {
  const pose = pickPose(state);
  if (!pose) return null;
  const dets = activeDetections(state);
  if (metricCache.pose === pose && metricCache.dets === dets && metricCache.fps === state.fps) return metricCache;

  const n = pose.n_frames;
  const sk = pose.skeleton, conf = pose.conf;
  const flareL = new Array(n).fill(NaN);
  const flareR = new Array(n).fill(NaN);
  const frontal = new Array(n).fill(NaN);
  const valid = new Array(n).fill(false);

  for (let f = 0; f < n; f++) {
    const th = frameTorsoHeight(sk, f);
    if (!(th > 1e-6)) continue;
    const base = f * 17;
    const lShX = sk[(base + J.L_SHOULDER) * 2], rShX = sk[(base + J.R_SHOULDER) * 2];
    const lElX = sk[(base + J.L_ELBOW) * 2],    rElX = sk[(base + J.R_ELBOW) * 2];
    flareL[f] = Math.abs(lShX - lElX) / th;
    flareR[f] = Math.abs(rShX - rElX) / th;
    frontal[f] = Math.abs(lShX - rShX) / th;
    valid[f] = frameValid(conf, f);
  }

  // Per-side hook exclusion: frames where THAT hand is throwing a hook.
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
      for (let g = s; g <= e; g++) arr[g] = true;
    }
  }

  metricCache = { pose, dets, fps: state.fps, n, flareL, flareR, frontal, valid, hookL, hookR };
  return metricCache;
}

// Per-frame frame-state for one side, given current slider values.
//   "flag" | "ok" | "filtered" | "hook"
function sideState(c, side, f) {
  const flare = side === "L" ? c.flareL[f] : c.flareR[f];
  if (!c.valid[f] || !Number.isFinite(flare)) return "filtered";
  if (c.frontal[f] < cfg.frontalGate) return "filtered";   // side-on
  if ((side === "L" ? c.hookL : c.hookR)[f]) return "hook"; // throwing a hook
  return flare > cfg.flareThreshold ? "flag" : "ok";
}

function stateColor(s) {
  return s === "flag" ? COLOR_FLAG : s === "hook" ? COLOR_HOOK
       : s === "ok" ? COLOR_OK : COLOR_INVALID;
}

// Per-side flare colour: dark green if hooked, grey if filtered, else banded by
// magnitude — tucked (<0.2) green, 0.2–0.3 orange, >0.3 red.
function flareColorAt(c, side, f) {
  const st = sideState(c, side, f);
  if (st === "filtered") return COLOR_INVALID;
  if (st === "hook") return COLOR_HOOK;
  const v = side === "L" ? c.flareL[f] : c.flareR[f];
  if (v > 0.3) return COLOR_FLAG;
  if (v > 0.2) return COLOR_WARN;
  return COLOR_OK;
}

// ── round rollup (cheap, recomputed on every slider move) ───────────────────

function rollup(c) {
  let consideredL = 0, flagL = 0, consideredR = 0, flagR = 0, flagEither = 0, considered = 0;
  const flaresL = [], flaresR = [];
  for (let f = 0; f < c.n; f++) {
    const sl = sideState(c, "L", f), sr = sideState(c, "R", f);
    const exL = sl === "filtered" || sl === "hook";   // hooks are excluded too
    const exR = sr === "filtered" || sr === "hook";
    if (!exL) { consideredL++; if (sl === "flag") flagL++; flaresL.push(c.flareL[f]); }
    if (!exR) { consideredR++; if (sr === "flag") flagR++; flaresR.push(c.flareR[f]); }
    if (!exL || !exR) {
      considered++;
      if (sl === "flag" || sr === "flag") flagEither++;
    }
  }
  return {
    consideredL, flagL, consideredR, flagR, considered, flagEither,
    medL: median(flaresL), medR: median(flaresR),
    pctL: consideredL ? (100 * flagL / consideredL) : 0,
    pctR: consideredR ? (100 * flagR / consideredR) : 0,
    pctEither: considered ? (100 * flagEither / considered) : 0,
  };
}

function median(xs) {
  const v = xs.filter(Number.isFinite).sort((a, b) => a - b);
  if (!v.length) return NaN;
  const m = Math.floor(v.length / 2);
  return v.length % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

function fmt(n, d = 3) { return Number.isFinite(n) ? n.toFixed(d) : "—"; }

// ── lens ────────────────────────────────────────────────────────────────────

let host;

export const ElbowTuckRule = {
  id: "elbow_tuck_lens",
  label: "Elbow tuck (width)",

  skeletonStyle() {
    return {
      boneColor: "rgba(255,255,255,0.25)",
      boneWidth: 1.5,
      jointRadius: 3,
      highlightJoints: new Set([J.L_SHOULDER, J.R_SHOULDER, J.L_ELBOW, J.R_ELBOW]),
    };
  },

  mount(_host) {
    host = _host;
    metricCache = { pose: null, fps: 0 };
    host.innerHTML = `
      <h2>Elbow tuck — width</h2>
      <p class="hint">
        Horizontal shoulder→elbow gap / torso height, per side. A flared elbow
        sits far to the side of the shoulder ⇒ large gap. Drag the threshold to
        read the tuck cutoff off the footage. Flare colour:
        <span style="color:${COLOR_OK}">&lt;0.2</span> ·
        <span style="color:${COLOR_WARN}">0.2–0.3</span> ·
        <span style="color:${COLOR_FLAG}">&gt;0.3</span> ·
        <span style="color:${COLOR_HOOK}">hook (excluded)</span>.
        Frontal gate is OFF by default — this only measures tuck from the front.
      </p>

      <label class="slider-row" style="display:block; font-size:12px; margin-top:6px">
        flare threshold = <output id="et-thr-out">0.30</output>
        <input type="range" id="et-thr" min="0.05" max="0.60" step="0.01" value="0.30"></label>
      <label class="slider-row" style="display:block; font-size:12px">
        frontal gate (shoulderW/torso, 0=off) = <output id="et-fg-out">0.00</output>
        <input type="range" id="et-fg" min="0.0" max="0.6" step="0.01" value="0.0"></label>

      <h3>Round</h3>
      <div id="et-round" style="font-size:13px; line-height:1.6"></div>

      <h3>Current frame</h3>
      <div id="et-frame" style="font-size:13px; line-height:1.6"></div>

      <h3><span style="color:${COLOR_L}">L</span> / <span style="color:${COLOR_R}">R</span> flare over time</h3>
      <canvas id="et-trace" width="320" height="120"></canvas>
    `;
    mountStageTimeline();

    const wire = (id, key, dec, out) => {
      const s = host.querySelector(id), o = host.querySelector(out);
      s.addEventListener("input", () => {
        cfg[key] = parseFloat(s.value);
        if (o) o.textContent = cfg[key].toFixed(dec);
        refresh();   // re-run the viewer's redraw → recolors brackets + sidebar
      });
    };
    wire("#et-thr", "flareThreshold", 2, "#et-thr-out");
    wire("#et-fg", "frontalGate", 2, "#et-fg-out");
  },

  update(state) {
    if (!host || !state) return;
    const c = computeMetrics(state);
    if (!c) {
      host.querySelector("#et-round").innerHTML = `<p class="muted">No pose cache loaded.</p>`;
      return;
    }
    const f = state.frame;
    const r = rollup(c);

    host.querySelector("#et-round").innerHTML = `
      <div style="display:flex; gap:14px">
        <div><span style="color:${COLOR_L}">L</span> flared
          <code>${r.pctL.toFixed(1)}%</code>
          <span class="muted">(${r.flagL}/${r.consideredL}) · med <code>${fmt(r.medL, 2)}</code></span></div>
        <div><span style="color:${COLOR_R}">R</span> flared
          <code>${r.pctR.toFixed(1)}%</code>
          <span class="muted">(${r.flagR}/${r.consideredR}) · med <code>${fmt(r.medR, 2)}</code></span></div>
      </div>
      <div style="margin-top:2px">either side
        <code>${r.pctEither.toFixed(1)}%</code>
        <span class="muted">(${r.flagEither}/${r.considered} frames over thr ${cfg.flareThreshold.toFixed(2)})</span></div>`;

    const sl = sideState(c, "L", f), sr = sideState(c, "R", f);
    host.querySelector("#et-frame").innerHTML = `
      <strong>frame ${f}:</strong>
      <span style="color:${COLOR_L}">L</span> <span style="color:${stateColor(sl)}; font-weight:600">${sl}</span>
      <code style="color:${flareColorAt(c, "L", f)}">${fmt(c.flareL[f], 2)}</code> ·
      <span style="color:${COLOR_R}">R</span> <span style="color:${stateColor(sr)}; font-weight:600">${sr}</span>
      <code style="color:${flareColorAt(c, "R", f)}">${fmt(c.flareR[f], 2)}</code><br>
      <span class="muted">frontal (shoulderW/torso): <code>${fmt(c.frontal[f], 2)}</code>${
        cfg.frontalGate > 0 ? ` · gate ${cfg.frontalGate.toFixed(2)}` : ""}</span>`;

    drawTrace(host.querySelector("#et-trace"), c, f);
    drawTimeline(document.getElementById("et-timeline"), c, f);
  },

  draw(ctx, state) {
    const c = computeMetrics(state);
    if (!c) return;
    const pose = pickPose(state);
    const f = state.frame;
    const s = state.renderScale || 1;
    const base = f * 17;

    // Per-side shoulder→elbow bracket: plumb line down from the shoulder, then
    // a horizontal run out to the elbow's x. The horizontal run IS the metric.
    const drawBracket = (shJ, elJ, accent) => {
      const shX = pose.skeleton[(base + shJ) * 2], shY = pose.skeleton[(base + shJ) * 2 + 1];
      const elX = pose.skeleton[(base + elJ) * 2], elY = pose.skeleton[(base + elJ) * 2 + 1];
      if (![shX, shY, elX, elY].every(Number.isFinite)) return;
      const side = shJ === J.L_SHOULDER ? "L" : "R";
      const color = flareColorAt(c, side, f);
      ctx.save();
      // plumb line (shoulder x, down to elbow height)
      ctx.strokeStyle = accent;
      ctx.lineWidth = 1.5 * s;
      ctx.setLineDash([4 * s, 4 * s]);
      ctx.beginPath(); ctx.moveTo(shX, shY); ctx.lineTo(shX, elY); ctx.stroke();
      ctx.setLineDash([]);
      // horizontal gap = the flare metric, colored by tucked/flared
      ctx.strokeStyle = color;
      ctx.lineWidth = 3 * s;
      ctx.beginPath(); ctx.moveTo(shX, elY); ctx.lineTo(elX, elY); ctx.stroke();
      // little caps
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(elX, elY, 3.5 * s, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    };
    drawBracket(J.L_SHOULDER, J.L_ELBOW, COLOR_L);
    drawBracket(J.R_SHOULDER, J.R_ELBOW, COLOR_R);

    // corner HUD
    const fsz = Math.round(13 * s), lineH = fsz + 4 * s;
    const lines = [
      [`L flare ${fmt(c.flareL[f], 2)}${c.hookL[f] ? " hook" : ""}`, flareColorAt(c, "L", f)],
      [`R flare ${fmt(c.flareR[f], 2)}${c.hookR[f] ? " hook" : ""}`, flareColorAt(c, "R", f)],
      [`frontal ${fmt(c.frontal[f], 2)}`, COLOR_FRONTAL],
      [`thr     ${cfg.flareThreshold.toFixed(2)}`, "#fff"],
    ];
    const padX = 10 * s, padY = 8 * s, boxW = 130 * s;
    const boxH = lines.length * lineH + padY * 2 - 4 * s;
    const bx = ctx.canvas.width - boxW - 10 * s, by = 10 * s;
    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.beginPath(); ctx.roundRect(bx, by, boxW, boxH, 6 * s); ctx.fill();
    ctx.font = `${fsz}px ui-monospace, monospace`;
    ctx.textBaseline = "top";
    lines.forEach(([t, col], i) => { ctx.fillStyle = col; ctx.fillText(t, bx + padX, by + padY + i * lineH); });
    ctx.restore();
  },
};

// ── below-video timeline (L / R / frontal strips, click to seek) ────────────

// Force the viewer to re-run draw()+update() (both bound to the video's
// "seeked" event) so slider changes recolor the on-video brackets and refresh
// the sidebar immediately, without waiting for the next frame step.
function refresh() {
  document.getElementById("video")?.dispatchEvent(new Event("seeked"));
}

const TL_LABEL_W = 56;

function mountStageTimeline() {
  const slot = document.getElementById("stage-extras");
  if (!slot) return;
  slot.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.style.cssText = "margin-top:12px;padding:10px 12px;background:var(--bg-card);border:1px solid var(--border);border-radius:8px";
  const label = document.createElement("div");
  label.className = "muted small";
  label.style.cssText = "margin-bottom:6px";
  label.textContent = "Elbow flare — L / R flagged · frontal-ness (click to seek)";
  wrap.appendChild(label);
  const canvas = document.createElement("canvas");
  canvas.id = "et-timeline";
  canvas.style.cssText = "display:block;width:100%;height:84px";
  canvas.width = 800; canvas.height = 84;
  wrap.appendChild(canvas);
  slot.appendChild(wrap);

  canvas.addEventListener("click", e => {
    const N = metricCache?.n;
    if (!N) return;
    const rect = canvas.getBoundingClientRect();
    const ratio = (e.clientX - rect.left - TL_LABEL_W) / Math.max(1, rect.width - TL_LABEL_W - 4);
    seekHack(Math.max(0, Math.min(N - 1, Math.round(ratio * (N - 1)))));
  });
}

function seekHack(f) {
  const slider = document.getElementById("scrubber");
  if (!slider) return;
  slider.value = f;
  slider.dispatchEvent(new Event("input"));
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
  const N = c.n;
  if (!N) return;

  const xOf = f => TL_LABEL_W + (f / Math.max(1, N - 1)) * (W - TL_LABEL_W - 4);
  const colW = Math.max(1, (W - TL_LABEL_W - 4) / Math.max(1, N - 1));

  // frontal-ness max for the bottom strip
  let maxFrontal = 0.6;
  for (let f = 0; f < N; f++) if (Number.isFinite(c.frontal[f]) && c.frontal[f] > maxFrontal) maxFrontal = c.frontal[f];

  const tracks = [{ label: "L", side: "L" }, { label: "R", side: "R" }];
  const gap = 6, top = 4, stripH = 16;
  const trackH = Math.floor((H - top * 2 - gap * tracks.length - stripH) / tracks.length);
  ctx.font = "10px ui-monospace, monospace";

  tracks.forEach((t, i) => {
    const y = top + i * (trackH + gap);
    ctx.fillStyle = i === 0 ? COLOR_L : COLOR_R;
    ctx.fillText(t.label, 6, y + trackH / 2 + 3);
    for (let f = 0; f < N; f++) {
      ctx.fillStyle = flareColorAt(c, t.side, f);
      ctx.globalAlpha = 0.9;
      ctx.fillRect(xOf(f), y, colW + 0.5, trackH);
    }
    ctx.globalAlpha = 1;
  });

  // frontal-ness strip at the bottom: height ∝ shoulderW/torso, dim under gate
  const stripY = top + tracks.length * (trackH + gap);
  ctx.fillStyle = COLOR_FRONTAL;
  ctx.fillText("front", 6, stripY + stripH / 2 + 3);
  for (let f = 0; f < N; f++) {
    const v = c.frontal[f];
    if (!Number.isFinite(v)) continue;
    const h = Math.min(1, v / maxFrontal) * stripH;
    ctx.globalAlpha = (cfg.frontalGate > 0 && v < cfg.frontalGate) ? 0.25 : 0.8;
    ctx.fillRect(xOf(f), stripY + (stripH - h), colW + 0.5, h);
  }
  ctx.globalAlpha = 1;
  if (cfg.frontalGate > 0) {
    const gy = stripY + stripH - Math.min(1, cfg.frontalGate / maxFrontal) * stripH;
    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(TL_LABEL_W, gy); ctx.lineTo(W - 4, gy); ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.strokeStyle = COLOR_FRAME;
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(xOf(frame), 1); ctx.lineTo(xOf(frame), H - 1); ctx.stroke();
}

// L / R flare sparklines with the live threshold line + current-frame marker.
function drawTrace(canvas, c, frame) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const N = c.n;
  if (!N) return;

  let maxV = cfg.flareThreshold * 1.4;
  for (let f = 0; f < N; f++) {
    if (Number.isFinite(c.flareL[f]) && c.flareL[f] > maxV) maxV = c.flareL[f];
    if (Number.isFinite(c.flareR[f]) && c.flareR[f] > maxV) maxV = c.flareR[f];
  }
  const yOf = v => H - 4 - (v / maxV) * (H - 12);
  const xOf = f => (f / Math.max(1, N - 1)) * W;

  // threshold line
  ctx.strokeStyle = "rgba(255,255,255,0.45)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(0, yOf(cfg.flareThreshold)); ctx.lineTo(W, yOf(cfg.flareThreshold)); ctx.stroke();
  ctx.setLineDash([]);

  const dots = (arr, color) => {
    for (let f = 0; f < N; f++) {
      const v = arr[f];
      if (!Number.isFinite(v)) continue;
      ctx.fillStyle = color;
      ctx.fillRect(xOf(f) - 0.5, yOf(v) - 1, 2, 2);
    }
  };
  dots(c.flareL, COLOR_L);
  dots(c.flareR, COLOR_R);

  ctx.strokeStyle = COLOR_FRAME;
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(xOf(frame), 0); ctx.lineTo(xOf(frame), H); ctx.stroke();
}
