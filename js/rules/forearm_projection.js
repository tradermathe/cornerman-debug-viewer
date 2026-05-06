// Forearm-projected wrist lens.
//
// The YOLO wrist anchors inconsistently when gloves are on — sometimes the
// model picks the middle of the glove, sometimes the cuff. The elbow is more
// stable (no glove ambiguity), and the forearm length is roughly constant
// per person within a round. So a more robust "fist" estimate is:
//
//   fist = elbow + normalise(wrist - elbow) × median_forearm_length
//
// We trust the elbow (position) and the forearm direction (jiggly anchor →
// small angular noise because the arm is ~30 cm long), but throw away the
// magnitude. Direction × stable length = wrist anchored at a consistent
// distance from the elbow regardless of where YOLO landed on the glove.
//
// This lens draws BOTH the raw wrist and the projected one so we can scrub
// through actual punches and see whether the projected version looks
// cleaner. If it does, the same logic gets promoted into guard_drop.py.

import { J } from "../skeleton.js";

const COLORS = {
  rawL:  "#ff8a5c",                       // orange — raw L wrist
  rawR:  "#ffd95c",                       // yellow — raw R wrist
  projL: "#5fd97a",                       // green  — projected L wrist
  projR: "#7ec8ff",                       // blue   — projected R wrist
  forearmL: "rgba(255,138,92,0.55)",
  forearmR: "rgba(255,217,92,0.55)",
};

// Frames below this confidence don't contribute to the median forearm length.
// Higher gives a tighter estimate, lower gives more samples — 0.5 is a
// reasonable balance for YOLO-Pose.
const MIN_LEN_CONF = 0.5;

let host;
let stats = { lenL: 0, lenR: 0, nL: 0, nR: 0 };

export const ForearmProjectionRule = {
  id: "forearm_projection",
  label: "Forearm-projected wrist",

  skeletonStyle() {
    // Fade the rest, highlight elbows + wrists.
    return {
      boneColor: "rgba(255,255,255,0.25)",
      boneWidth: 1.5,
      jointRadius: 3,
      highlightJoints: new Set([J.L_ELBOW, J.R_ELBOW, J.L_WRIST, J.R_WRIST]),
    };
  },

  mount(_host, state) {
    host = _host;

    // Compute median forearm length per side from high-confidence frames.
    // Median (not mean) is robust to the same outliers we're trying to fix.
    const left  = forearmLengths(state.pose, J.L_ELBOW, J.L_WRIST);
    const right = forearmLengths(state.pose, J.R_ELBOW, J.R_WRIST);
    stats = {
      lenL: median(left),
      lenR: median(right),
      nL: left.length,
      nR: right.length,
    };

    host.innerHTML = `
      <h2>Forearm-projected wrist</h2>
      <p class="hint">
        Solid dot = <span style="color:${COLORS.rawL}">raw L wrist</span> /
        <span style="color:${COLORS.rawR}">raw R wrist</span>.
        Hollow dot = <span style="color:${COLORS.projL}">projected L</span> /
        <span style="color:${COLORS.projR}">projected R</span>
        (elbow + unit(elbow→wrist) × median forearm length).
      </p>

      <h3>Forearm length (median)</h3>
      <div class="metric-grid">
        <div class="metric">
          <div class="metric-label">L</div>
          <div class="metric-val">${stats.lenL.toFixed(0)} px</div>
          <div class="metric-sub">from ${stats.nL} frames ≥ ${MIN_LEN_CONF} conf</div>
        </div>
        <div class="metric">
          <div class="metric-label">R</div>
          <div class="metric-val">${stats.lenR.toFixed(0)} px</div>
          <div class="metric-sub">from ${stats.nR} frames ≥ ${MIN_LEN_CONF} conf</div>
        </div>
      </div>

      <h3>Per-frame</h3>
      <p class="hint">How far the raw wrist is from where the forearm vector says
      it should be. Big numbers = YOLO probably picked the wrong point on the glove.</p>
      <div class="metric-grid">
        <div class="metric">
          <div class="metric-label">L Δ raw vs proj</div>
          <div class="metric-val" id="fp-l-diff">—</div>
          <div class="metric-sub" id="fp-l-pct">—</div>
        </div>
        <div class="metric">
          <div class="metric-label">R Δ raw vs proj</div>
          <div class="metric-val" id="fp-r-diff">—</div>
          <div class="metric-sub" id="fp-r-pct">—</div>
        </div>
      </div>

      <h3>Raw vs projected y</h3>
      <p class="hint">Y position drives guard-drop. If projected stays steadier
      across the round than raw, the new logic is worth promoting.</p>
      <div class="metric-grid">
        <div class="metric">
          <div class="metric-label">L raw y / proj y</div>
          <div class="metric-val" id="fp-l-y">—</div>
        </div>
        <div class="metric">
          <div class="metric-label">R raw y / proj y</div>
          <div class="metric-val" id="fp-r-y">—</div>
        </div>
      </div>
    `;
  },

  draw(ctx, state) {
    drawProjection(ctx, state.pose, state.frame,
                   J.L_ELBOW, J.L_WRIST, stats.lenL,
                   COLORS.rawL, COLORS.projL, COLORS.forearmL);
    drawProjection(ctx, state.pose, state.frame,
                   J.R_ELBOW, J.R_WRIST, stats.lenR,
                   COLORS.rawR, COLORS.projR, COLORS.forearmR);
  },

  update(state) {
    const f = state.frame;
    const p = state.pose;

    const L = projectFrame(p, f, J.L_ELBOW, J.L_WRIST, stats.lenL);
    const R = projectFrame(p, f, J.R_ELBOW, J.R_WRIST, stats.lenR);

    setText("fp-l-diff", L ? `${L.diff.toFixed(0)} px` : "—");
    setText("fp-r-diff", R ? `${R.diff.toFixed(0)} px` : "—");
    setText("fp-l-pct",
      L && stats.lenL ? `${(100 * L.diff / stats.lenL).toFixed(0)}% of forearm` : "—");
    setText("fp-r-pct",
      R && stats.lenR ? `${(100 * R.diff / stats.lenR).toFixed(0)}% of forearm` : "—");
    setText("fp-l-y", L ? `${L.rawY.toFixed(0)} / ${L.projY.toFixed(0)}` : "—");
    setText("fp-r-y", R ? `${R.rawY.toFixed(0)} / ${R.projY.toFixed(0)}` : "—");
  },
};

// ── helpers ────────────────────────────────────────────────────────────────

function forearmLengths(pose, elbowJ, wristJ) {
  const lens = [];
  for (let f = 0; f < pose.n_frames; f++) {
    const ec = pose.conf[f * 17 + elbowJ];
    const wc = pose.conf[f * 17 + wristJ];
    if (ec < MIN_LEN_CONF || wc < MIN_LEN_CONF) continue;
    const ex = pose.skeleton[(f * 17 + elbowJ) * 2];
    const ey = pose.skeleton[(f * 17 + elbowJ) * 2 + 1];
    const wx = pose.skeleton[(f * 17 + wristJ) * 2];
    const wy = pose.skeleton[(f * 17 + wristJ) * 2 + 1];
    lens.push(Math.hypot(wx - ex, wy - ey));
  }
  return lens;
}

function median(arr) {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

// Returns { rawX, rawY, projX, projY, diff } or null if data missing.
function projectFrame(pose, frame, elbowJ, wristJ, forearmLen) {
  if (forearmLen <= 0) return null;
  const ex = pose.skeleton[(frame * 17 + elbowJ) * 2];
  const ey = pose.skeleton[(frame * 17 + elbowJ) * 2 + 1];
  const wx = pose.skeleton[(frame * 17 + wristJ) * 2];
  const wy = pose.skeleton[(frame * 17 + wristJ) * 2 + 1];
  const ec = pose.conf[frame * 17 + elbowJ];
  const wc = pose.conf[frame * 17 + wristJ];
  if (ec < 0.05 || wc < 0.05) return null;

  const dx = wx - ex, dy = wy - ey;
  const len = Math.hypot(dx, dy);
  if (len === 0) return null;
  const projX = ex + (dx / len) * forearmLen;
  const projY = ey + (dy / len) * forearmLen;
  const diff = Math.hypot(wx - projX, wy - projY);
  return { rawX: wx, rawY: wy, projX, projY, diff };
}

function drawProjection(ctx, pose, frame, elbowJ, wristJ, forearmLen,
                        rawColor, projColor, lineColor) {
  const r = projectFrame(pose, frame, elbowJ, wristJ, forearmLen);
  if (!r) return;
  const ex = pose.skeleton[(frame * 17 + elbowJ) * 2];
  const ey = pose.skeleton[(frame * 17 + elbowJ) * 2 + 1];

  // Forearm line elbow → projected fist (the new logic's forearm).
  ctx.save();
  ctx.strokeStyle = lineColor;
  ctx.lineWidth = 2.5;
  ctx.setLineDash([7, 4]);
  ctx.beginPath();
  ctx.moveTo(ex, ey);
  ctx.lineTo(r.projX, r.projY);
  ctx.stroke();
  ctx.restore();

  // Raw wrist (filled) — what guard_drop.py currently uses.
  ctx.fillStyle = rawColor;
  ctx.beginPath();
  ctx.arc(r.rawX, r.rawY, 7, 0, Math.PI * 2);
  ctx.fill();

  // Projected wrist (hollow ring) — what the new logic would use.
  ctx.strokeStyle = projColor;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(r.projX, r.projY, 9, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = "rgba(0,0,0,0.55)";
  ctx.beginPath();
  ctx.arc(r.projX, r.projY, 5, 0, Math.PI * 2);
  ctx.fill();
}

function setText(id, value) {
  const el = host?.querySelector("#" + id);
  if (el) el.textContent = value;
}
