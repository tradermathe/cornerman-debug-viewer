// Orientation lens — visualizes the line through the two ankles.
//
// First step toward testing whether the ankle-line angle (with normalization)
// can predict the boxer's facing direction. For now this only draws the line;
// angle/normalization/prediction come later.

import { J } from "../skeleton.js";

const MIN_CONF = 0.3;        // skip frames where an ankle is poorly detected
const LINE_COLOR = "#ffd24a"; // amber, distinct from the skeleton greens/reds

let host;

export const OrientationLensRule = {
  id: "orientation_lens",
  label: "Orientation lens (ankle line)",

  mount(_host, _state) {
    host = _host;
    host.innerHTML = `
      <h2>Orientation lens</h2>
      <p class="hint">Draws the line through both ankles (extended across the
      frame). First step toward testing whether ankle geometry, normalized,
      can predict facing direction. No angle / prediction yet.</p>
      <div id="ol-state" class="hint"></div>
    `;
  },

  update(state) {
    const el = host?.querySelector("#ol-state");
    if (!el) return;
    const f = state.frame;
    const cL = state.pose.conf[f * 17 + J.L_ANKLE];
    const cR = state.pose.conf[f * 17 + J.R_ANKLE];
    const ok = cL >= MIN_CONF && cR >= MIN_CONF;
    el.innerHTML = ok
      ? `L_ankle conf <code>${cL.toFixed(2)}</code> · R_ankle conf <code>${cR.toFixed(2)}</code>`
      : `<span class="muted">Hidden — at least one ankle below ${MIN_CONF.toFixed(2)} confidence (L=${cL.toFixed(2)}, R=${cR.toFixed(2)}).</span>`;
  },

  draw(ctx, state) {
    const f = state.frame;
    const cL = state.pose.conf[f * 17 + J.L_ANKLE];
    const cR = state.pose.conf[f * 17 + J.R_ANKLE];
    if (cL < MIN_CONF || cR < MIN_CONF) return;

    const lx = state.pose.skeleton[(f * 17 + J.L_ANKLE) * 2];
    const ly = state.pose.skeleton[(f * 17 + J.L_ANKLE) * 2 + 1];
    const rx = state.pose.skeleton[(f * 17 + J.R_ANKLE) * 2];
    const ry = state.pose.skeleton[(f * 17 + J.R_ANKLE) * 2 + 1];

    // Extend the segment to the canvas edges so the orientation axis is
    // visible regardless of how close the feet are.
    const W = ctx.canvas.width;
    const H = ctx.canvas.height;
    const dx = rx - lx;
    const dy = ry - ly;
    let x0, y0, x1, y1;
    if (Math.abs(dx) < 1e-6 && Math.abs(dy) < 1e-6) {
      return; // ankles coincident — degenerate, nothing to draw
    }
    if (Math.abs(dx) >= Math.abs(dy)) {
      // mostly horizontal — extend across full width
      const slope = dy / dx;
      x0 = 0;          y0 = ly + (x0 - lx) * slope;
      x1 = W;          y1 = ly + (x1 - lx) * slope;
    } else {
      // mostly vertical — extend across full height
      const slope = dx / dy;
      y0 = 0;          x0 = lx + (y0 - ly) * slope;
      y1 = H;          x1 = lx + (y1 - ly) * slope;
    }

    const s = state.renderScale || 1;
    ctx.save();

    // Extended line (thin, semi-transparent) so it doesn't dominate the frame.
    ctx.strokeStyle = LINE_COLOR;
    ctx.globalAlpha = 0.45;
    ctx.lineWidth = 1.5 * s;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();

    // Solid segment between the actual ankle points.
    ctx.globalAlpha = 1;
    ctx.lineWidth = 3 * s;
    ctx.beginPath();
    ctx.moveTo(lx, ly);
    ctx.lineTo(rx, ry);
    ctx.stroke();

    // Mark the two endpoints.
    ctx.fillStyle = LINE_COLOR;
    for (const [x, y] of [[lx, ly], [rx, ry]]) {
      ctx.beginPath();
      ctx.arc(x, y, 4 * s, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  },
};
