// Guard-height lens — a stripped-down view of just the vertical guard stack:
//
//   - Solid line at nose height.
//   - Dashed "target" line just below the nose — where the wrists should sit.
//   - One line per wrist at its current height.
//
// How "just below nose" is configured: the target line is
//     target_y = nose_y + targetOffset * torso_height
// where torso_height is the shoulder→hip distance (same normaliser
// guard_drop.py uses), so the offset scales with the boxer's size and
// distance from the camera instead of being a fixed pixel count.
// Default targetOffset = 0.10 (~chin height); tune it with the slider.

import { J, torsoHeight } from "../skeleton.js";

const DEFAULTS = {
  targetOffset: 0.10,          // fraction of torso height below the nose
  minWristConfidence: 0.30,
};

const COLORS = {
  nose:    "#7ec8ff",
  target:  "#5fd97a",
  l_wrist: "#ff8a5c",
  r_wrist: "#ffd95c",
};

let host;
let cfg = { ...DEFAULTS };

export const GuardHeightRule = {
  id: "guard_height",
  label: "Guard height",

  skeletonStyle() {
    return {
      boneColor: "rgba(255,255,255,0.25)",
      boneWidth: 1.5,
      jointRadius: 3,
      highlightJoints: new Set([J.NOSE, J.L_WRIST, J.R_WRIST]),
    };
  },

  mount(_host, state) {
    host = _host;
    host.innerHTML = `
      <h2>Guard height</h2>
      <p class="hint">Lines: <span style="color:${COLORS.nose}">nose (solid)</span> ·
        <span style="color:${COLORS.target}">wrist target (dashed)</span> ·
        <span style="color:${COLORS.l_wrist}">L wrist</span> ·
        <span style="color:${COLORS.r_wrist}">R wrist</span>.
        Target = nose_y + offset × torso height (shoulder→hip), so it scales
        with the boxer instead of being a pixel constant.</p>

      <h3>Wrist vs target (torso units, − = above)</h3>
      <div class="metric-grid">
        <div class="metric">
          <div class="metric-label">L wrist</div>
          <div class="metric-val" id="l-target-dist">—</div>
          <div class="metric-sub" id="l-target-verdict"></div>
        </div>
        <div class="metric">
          <div class="metric-label">R wrist</div>
          <div class="metric-val" id="r-target-dist">—</div>
          <div class="metric-sub" id="r-target-verdict"></div>
        </div>
      </div>

      <h3>Config</h3>
      <label class="slider">
        <span>target_offset = <output id="to-out">${cfg.targetOffset.toFixed(2)}</output> × torso below nose</span>
        <input type="range" id="to-slider" min="0" max="0.40" step="0.01" value="${cfg.targetOffset}">
      </label>
      <label class="slider">
        <span>min_wrist_confidence = <output id="mwc-out">${cfg.minWristConfidence.toFixed(2)}</output></span>
        <input type="range" id="mwc-slider" min="0" max="1" step="0.01" value="${cfg.minWristConfidence}">
      </label>
    `;

    const wire = (slider, out, key) => {
      const s = host.querySelector(slider);
      const o = host.querySelector(out);
      s.addEventListener("input", () => {
        cfg[key] = parseFloat(s.value);
        o.textContent = cfg[key].toFixed(2);
        seekHack(state, state.frame);
      });
    };
    wire("#to-slider", "#to-out", "targetOffset");
    wire("#mwc-slider", "#mwc-out", "minWristConfidence");
  },

  draw(ctx, state) {
    const f = state.frame;
    const p = pickPose(state);
    const W = ctx.canvas.width;
    const s = state.renderScale || 1;

    const nose = jt(p, f, J.NOSE);
    const lw = jt(p, f, J.L_WRIST);
    const rw = jt(p, f, J.R_WRIST);
    const torso = Math.max(1e-6, torsoHeight(p, f));
    const targetY = nose.y + cfg.targetOffset * torso;

    drawHLine(ctx, nose.y,  W, COLORS.nose,    2 * s, null);
    drawHLine(ctx, targetY, W, COLORS.target,  2 * s, 3 * s);
    drawHLine(ctx, lw.y,    W, COLORS.l_wrist, 2 * s, 3 * s);
    drawHLine(ctx, rw.y,    W, COLORS.r_wrist, 2 * s, 3 * s);
  },

  update(state) {
    const f = state.frame;
    const p = pickPose(state);

    const nose = jt(p, f, J.NOSE);
    const lw = jt(p, f, J.L_WRIST);
    const rw = jt(p, f, J.R_WRIST);
    const torso = Math.max(1e-6, torsoHeight(p, f));
    const targetY = nose.y + cfg.targetOffset * torso;

    const lDist = (lw.y - targetY) / torso;
    const rDist = (rw.y - targetY) / torso;
    setText("l-target-dist", lDist.toFixed(2));
    setText("r-target-dist", rDist.toFixed(2));
    setText("l-target-verdict", verdict(lDist, lw.c));
    setText("r-target-verdict", verdict(rDist, rw.c));
  },
};

// ── helpers (mirrors guard_drop.js) ────────────────────────────────────────

function pickPose(state) {
  return state.poseV6 || state.pose;
}

function jt(pose, frame, j) {
  return {
    x: pose.skeleton[(frame * 17 + j) * 2],
    y: pose.skeleton[(frame * 17 + j) * 2 + 1],
    c: pose.conf[frame * 17 + j],
  };
}

// dashUnit null ⇒ solid line.
function drawHLine(ctx, y, w, color, lineWidth, dashUnit) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  if (dashUnit) ctx.setLineDash([dashUnit * 2, dashUnit * 2]);
  ctx.beginPath();
  ctx.moveTo(0, y);
  ctx.lineTo(w, y);
  ctx.stroke();
  ctx.restore();
}

function verdict(d, c) {
  if (c < cfg.minWristConfidence) return `<span class="muted">low conf, gated</span>`;
  if (d > 0) return `<span class="bad">below target</span>`;
  return `<span class="good">at/above target</span>`;
}

function setText(id, value) {
  const el = host.querySelector("#" + id);
  if (el) el.innerHTML = value;
}

function seekHack(state, f) {
  const slider = document.getElementById("scrubber");
  slider.value = f;
  slider.dispatchEvent(new Event("input"));
}
