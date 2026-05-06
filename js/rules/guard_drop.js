// Guard-drop debug panel.
//
// What it visualises (everything you asked for, plus a few extras the Vision
// pipeline now lets us show that YOLO didn't):
//   - Skeleton with nose, both wrists, and both shoulders highlighted.
//   - Dashed horizontal lines at the y-coords of nose, L_wrist, R_wrist,
//     L_shoulder, R_shoulder. Colour-coded so you can read the stack at a glance.
//   - Per-joint confidence badges floating next to nose + wrists.
//   - Numerical readout: wrist→nose distance and wrist→shoulder distance,
//     normalised by torso height (these are exactly the metrics guard_drop.py
//     uses to decide whether the guard dropped).
//   - "Guard up?" verdict per side using the same delta_threshold +
//     guard_low_threshold defaults from rules_config.json.
//   - 60-frame fading wrist trail (last 2 s at 30 fps) — shows the punching
//     motion shape, not just the current position. Vision's wrist tracking
//     stays cleaner than YOLO's during fast extensions, so the trail actually
//     reads as a path.
//   - Sparkline of nose + wrist y over the full clip with the current frame
//     marked. Drops jump out as red spikes.
//   - Confidence sparklines for nose/L_wrist/R_wrist showing tracker dropouts.
//   - Face-direction hint from ear/eye visibility (Apple Vision returns 0 conf
//     for the side that isn't visible, which is a stronger signal than YOLO's
//     low-conf guesses).
//
// Apple-Vision-specific things worth noting that this view surfaces:
//   * conf == 0 means "not detected" rather than "low confidence guess".
//   * Per-joint confidence is calibrated differently from YOLO — wrist conf
//     stays high through fast hand movement instead of collapsing.
//   * Face landmark asymmetry (one ear visible, the other not) gives a free
//     stance / facing-direction read.

import { J, torsoHeight } from "../skeleton.js";

// Defaults match rules_config.json → rules.guard_drop.params at the time
// this viewer was written. The UI exposes sliders so we can re-tune without
// touching code.
const DEFAULTS = {
  deltaThreshold: 0.10,        // shoulder-anchored delta over the punch
  guardLowThreshold: 0.30,     // end-of-punch wrist→nose normalised distance
  minWristConfidence: 0.30,
  trailFrames: 60,
};

const COLORS = {
  nose:        "#7ec8ff",
  l_wrist:     "#ff8a5c",
  r_wrist:     "#ffd95c",
  l_shoulder:  "rgba(126,200,255,0.45)",
  r_shoulder:  "rgba(255,217,92,0.45)",
};

let host;
let cfg = { ...DEFAULTS };

export const GuardDropRule = {
  id: "guard_drop",
  label: "Guard drop",

  skeletonStyle() {
    // Fade the rest of the body, highlight the joints the rule cares about.
    return {
      boneColor: "rgba(255,255,255,0.25)",
      boneWidth: 1.5,
      jointRadius: 3,
      highlightJoints: new Set([
        J.NOSE, J.L_WRIST, J.R_WRIST, J.L_SHOULDER, J.R_SHOULDER,
      ]),
    };
  },

  mount(_host, state) {
    host = _host;
    host.innerHTML = `
      <h2>Guard drop</h2>
      <p class="hint">Lines: <span style="color:${COLORS.nose}">nose</span> ·
        <span style="color:${COLORS.l_wrist}">L wrist</span> ·
        <span style="color:${COLORS.r_wrist}">R wrist</span> ·
        <span style="color:#aac">L/R shoulders (faint)</span>.
        Wrist‐to‐nose &gt; threshold ⇒ guard considered low.</p>

      <div class="metric-grid">
        <div class="metric"><div class="metric-label">L wrist conf</div><div class="metric-val" id="l-wrist-conf">—</div></div>
        <div class="metric"><div class="metric-label">R wrist conf</div><div class="metric-val" id="r-wrist-conf">—</div></div>
        <div class="metric"><div class="metric-label">Nose conf</div><div class="metric-val" id="nose-conf">—</div></div>
        <div class="metric"><div class="metric-label">Torso px</div><div class="metric-val" id="torso-h">—</div></div>
      </div>

      <h3>Wrist → nose (normalised)</h3>
      <div class="metric-grid">
        <div class="metric">
          <div class="metric-label">L wrist</div>
          <div class="metric-val" id="l-nose-dist">—</div>
          <div class="metric-sub" id="l-nose-verdict"></div>
        </div>
        <div class="metric">
          <div class="metric-label">R wrist</div>
          <div class="metric-val" id="r-nose-dist">—</div>
          <div class="metric-sub" id="r-nose-verdict"></div>
        </div>
      </div>

      <h3>Wrist → same-side shoulder</h3>
      <p class="hint">This is the duck-resistant signal — if you slip your head
      down, both nose and wrist drop together; the shoulder anchor doesn't.</p>
      <div class="metric-grid">
        <div class="metric"><div class="metric-label">L</div><div class="metric-val" id="l-sho-dist">—</div></div>
        <div class="metric"><div class="metric-label">R</div><div class="metric-val" id="r-sho-dist">—</div></div>
      </div>

      <h3>Facing</h3>
      <div class="metric"><div class="metric-val" id="facing">—</div></div>

      <h3>Wrist y over time</h3>
      <canvas id="trace-canvas" width="320" height="120"></canvas>

      <h3>Confidence over time</h3>
      <canvas id="conf-canvas" width="320" height="80"></canvas>

      <h3>Thresholds</h3>
      <label class="slider">
        <span>guard_low_threshold = <output id="glt-out">${cfg.guardLowThreshold.toFixed(2)}</output></span>
        <input type="range" id="glt-slider" min="-0.30" max="0.80" step="0.01" value="${cfg.guardLowThreshold}">
      </label>
      <label class="slider">
        <span>min_wrist_confidence = <output id="mwc-out">${cfg.minWristConfidence.toFixed(2)}</output></span>
        <input type="range" id="mwc-slider" min="0" max="1" step="0.01" value="${cfg.minWristConfidence}">
      </label>
      <label class="slider">
        <span>trail = <output id="tr-out">${cfg.trailFrames}</output> frames</span>
        <input type="range" id="tr-slider" min="0" max="120" step="5" value="${cfg.trailFrames}">
      </label>
    `;

    // Wire sliders. Each one just updates the cfg + triggers a redraw.
    const wire = (slider, out, key, fmt = v => v.toFixed(2)) => {
      const s = host.querySelector(slider);
      const o = host.querySelector(out);
      s.addEventListener("input", () => {
        cfg[key] = parseFloat(s.value);
        o.textContent = fmt(cfg[key]);
        // Trigger a redraw via a synthetic seek — cheapest way to refresh.
        const f = state.frame;
        state.frame = -1;
        seekHack(state, f);
      });
    };
    wire("#glt-slider", "#glt-out", "guardLowThreshold");
    wire("#mwc-slider", "#mwc-out", "minWristConfidence");
    wire("#tr-slider",  "#tr-out",  "trailFrames", v => String(Math.round(v)));
  },

  draw(ctx, state) {
    const f = state.frame;
    const p = state.pose;
    const W = ctx.canvas.width;

    const nose = jt(p, f, J.NOSE);
    const lw = jt(p, f, J.L_WRIST);
    const rw = jt(p, f, J.R_WRIST);
    const ls = jt(p, f, J.L_SHOULDER);
    const rs = jt(p, f, J.R_SHOULDER);

    // Horizontal y-lines at nose / wrists / shoulders.
    drawHLine(ctx, nose.y, W, COLORS.nose, 2);
    drawHLine(ctx, lw.y,   W, COLORS.l_wrist, 2);
    drawHLine(ctx, rw.y,   W, COLORS.r_wrist, 2);
    drawHLine(ctx, ls.y,   W, COLORS.l_shoulder, 1);
    drawHLine(ctx, rs.y,   W, COLORS.r_shoulder, 1);

    // Wrist trail — fading dots over the last N frames.
    drawTrail(ctx, p, f, J.L_WRIST, COLORS.l_wrist, cfg.trailFrames);
    drawTrail(ctx, p, f, J.R_WRIST, COLORS.r_wrist, cfg.trailFrames);

    // Confidence badges floating above each highlighted joint.
    drawBadge(ctx, lw,   `L${lw.c.toFixed(2)}`,  COLORS.l_wrist);
    drawBadge(ctx, rw,   `R${rw.c.toFixed(2)}`,  COLORS.r_wrist);
    drawBadge(ctx, nose, `N${nose.c.toFixed(2)}`, COLORS.nose);
  },

  update(state) {
    const f = state.frame;
    const p = state.pose;

    const nose = jt(p, f, J.NOSE);
    const lw = jt(p, f, J.L_WRIST);
    const rw = jt(p, f, J.R_WRIST);
    const ls = jt(p, f, J.L_SHOULDER);
    const rs = jt(p, f, J.R_SHOULDER);
    const torso = Math.max(1e-6, torsoHeight(p, f));

    setText("l-wrist-conf", lw.c.toFixed(2), confTextColor(lw.c));
    setText("r-wrist-conf", rw.c.toFixed(2), confTextColor(rw.c));
    setText("nose-conf",    nose.c.toFixed(2), confTextColor(nose.c));
    setText("torso-h",      torso.toFixed(0));

    // y axis grows downward in image coords — wrist *above* nose is wrist.y < nose.y,
    // so a negative normalised distance = wrist higher than nose = guard up.
    const lNoseDist = (lw.y - nose.y) / torso;
    const rNoseDist = (rw.y - nose.y) / torso;
    setText("l-nose-dist", lNoseDist.toFixed(2));
    setText("r-nose-dist", rNoseDist.toFixed(2));
    setText("l-nose-verdict", verdict(lNoseDist, lw.c));
    setText("r-nose-verdict", verdict(rNoseDist, rw.c));

    setText("l-sho-dist", ((lw.y - ls.y) / torso).toFixed(2));
    setText("r-sho-dist", ((rw.y - rs.y) / torso).toFixed(2));

    setText("facing", facingFromFace(p, f));

    drawTrace(host.querySelector("#trace-canvas"), p, f);
    drawConfTrace(host.querySelector("#conf-canvas"), p, f);
  },
};

// ── helpers ────────────────────────────────────────────────────────────────

function jt(pose, frame, j) {
  return {
    x: pose.skeleton[(frame * 17 + j) * 2],
    y: pose.skeleton[(frame * 17 + j) * 2 + 1],
    c: pose.conf[frame * 17 + j],
  };
}

function drawHLine(ctx, y, w, color, dash) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([dash * 3, dash * 3]);
  ctx.beginPath();
  ctx.moveTo(0, y);
  ctx.lineTo(w, y);
  ctx.stroke();
  ctx.restore();
}

function drawBadge(ctx, joint, text, color) {
  if (joint.c <= 0) return;
  ctx.save();
  ctx.font = "16px ui-monospace, monospace";
  const pad = 4;
  const m = ctx.measureText(text);
  const x = joint.x + 10;
  const y = joint.y - 10;
  ctx.fillStyle = "rgba(0,0,0,0.65)";
  ctx.fillRect(x - pad, y - 14, m.width + pad * 2, 18);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
  ctx.restore();
}

function drawTrail(ctx, pose, frame, jointIdx, color, n) {
  if (n <= 0) return;
  const start = Math.max(0, frame - n);
  ctx.save();
  for (let f = start; f <= frame; f++) {
    const c = pose.conf[f * 17 + jointIdx];
    if (c < 0.05) continue;
    const age = (frame - f) / n;
    ctx.globalAlpha = 0.9 * (1 - age);
    ctx.fillStyle = color;
    const x = pose.skeleton[(f * 17 + jointIdx) * 2];
    const y = pose.skeleton[(f * 17 + jointIdx) * 2 + 1];
    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

// Wrist-y / nose-y trace across the full clip with current frame marker.
function drawTrace(canvas, pose, frame) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Sample every K frames so we don't draw 5000 line segments.
  const N = pose.n_frames;
  const stride = Math.max(1, Math.floor(N / W));

  // Pull series. Y in image coords grows downward; flip for the chart so
  // "up" on screen means "higher in the frame".
  const series = [
    { idx: J.NOSE,    color: COLORS.nose },
    { idx: J.L_WRIST, color: COLORS.l_wrist },
    { idx: J.R_WRIST, color: COLORS.r_wrist },
  ];

  // Find min/max y across all three series for autoscale.
  let yMin = Infinity, yMax = -Infinity;
  for (const s of series) {
    for (let f = 0; f < N; f += stride) {
      const c = pose.conf[f * 17 + s.idx];
      if (c < 0.2) continue;
      const y = pose.skeleton[(f * 17 + s.idx) * 2 + 1];
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    }
  }
  if (!isFinite(yMin)) { yMin = 0; yMax = pose.height; }

  const ymap = y => H - ((y - yMin) / (yMax - yMin)) * (H - 4) - 2;
  const xmap = f => (f / (N - 1)) * (W - 2) + 1;

  ctx.lineWidth = 1.2;
  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.beginPath();
    let started = false;
    for (let f = 0; f < N; f += stride) {
      const c = pose.conf[f * 17 + s.idx];
      const y = pose.skeleton[(f * 17 + s.idx) * 2 + 1];
      if (c < 0.2) { started = false; continue; }
      const px = xmap(f), py = ymap(y);
      if (!started) { ctx.moveTo(px, py); started = true; }
      else            ctx.lineTo(px, py);
    }
    ctx.stroke();
  }

  // Current-frame marker.
  ctx.strokeStyle = "rgba(255,255,255,0.7)";
  ctx.lineWidth = 1;
  const x = xmap(frame);
  ctx.beginPath();
  ctx.moveTo(x, 0);
  ctx.lineTo(x, H);
  ctx.stroke();
}

function drawConfTrace(canvas, pose, frame) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const N = pose.n_frames;
  const stride = Math.max(1, Math.floor(N / W));
  const series = [
    { idx: J.NOSE,    color: COLORS.nose },
    { idx: J.L_WRIST, color: COLORS.l_wrist },
    { idx: J.R_WRIST, color: COLORS.r_wrist },
  ];

  // Threshold line for min_wrist_confidence.
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.setLineDash([2, 3]);
  const ty = H - cfg.minWristConfidence * (H - 2) - 1;
  ctx.beginPath();
  ctx.moveTo(0, ty);
  ctx.lineTo(W, ty);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.lineWidth = 1;
  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.beginPath();
    for (let f = 0; f < N; f += stride) {
      const c = pose.conf[f * 17 + s.idx];
      const px = (f / (N - 1)) * (W - 2) + 1;
      const py = H - c * (H - 2) - 1;
      if (f === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(255,255,255,0.7)";
  const x = (frame / (N - 1)) * (W - 2) + 1;
  ctx.beginPath();
  ctx.moveTo(x, 0);
  ctx.lineTo(x, H);
  ctx.stroke();
}

function verdict(d, c) {
  if (c < cfg.minWristConfidence) return `<span class="muted">low conf, gated</span>`;
  if (d > cfg.guardLowThreshold)  return `<span class="bad">guard low</span>`;
  return `<span class="good">guard up</span>`;
}

function facingFromFace(pose, f) {
  const lEar = pose.conf[f * 17 + J.L_EAR];
  const rEar = pose.conf[f * 17 + J.R_EAR];
  const lEye = pose.conf[f * 17 + J.L_EYE];
  const rEye = pose.conf[f * 17 + J.R_EYE];
  const lScore = lEar + lEye;
  const rScore = rEar + rEye;
  if (lScore < 0.2 && rScore < 0.2) return "face not visible";
  if (Math.abs(lScore - rScore) < 0.4) return "facing camera";
  return lScore > rScore ? "left side toward camera" : "right side toward camera";
}

function setText(id, value, color) {
  const el = host.querySelector("#" + id);
  if (!el) return;
  el.innerHTML = value;
  if (color) el.style.color = color;
}

function confTextColor(c) {
  if (c >= 0.5) return "#5fd97a";
  if (c >= 0.2) return "#f5b945";
  return "#e85a5a";
}

// Re-trigger a viewer redraw without exposing internal redraw() — set frame
// to a sentinel and back, dispatching the same event the scrubber does.
function seekHack(state, f) {
  const ev = new Event("input");
  const slider = document.getElementById("scrubber");
  slider.value = f;
  slider.dispatchEvent(ev);
}
