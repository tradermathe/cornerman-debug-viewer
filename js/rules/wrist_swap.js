// Wrist-swap lens — compares Apple Vision wrists against the dedicated
// glove-model wrists for the same Vision skeleton.
//
// Vision provides the full 17-joint skeleton; the glove detector emits a
// wrist-only sidecar (L_wrist, R_wrist) aligned to Vision's frame timing
// 1:1. This lens lets you toggle which source's wrists get rendered on top
// of the otherwise-unchanged Vision skeleton — exactly the substitution
// the production iOS app will eventually do at frame time.
//
// Requires:
//   - vision skeleton cache (slot.vision)
//   - glove-wrist sidecar attached to the pose object as pose.gloveWrists
//     (loaded by viewer.js's loadFromIndex when slot.glove exists)
//
// Three modes:
//   "vision"  — render only Vision's wrists (baseline behavior)
//   "glove"   — render only Glove model's wrists (the production substitution)
//   "both"    — render both, with a connecting line so divergence is obvious

import { J } from "../skeleton.js";

const VISION_COL = "#5fd1ff";
const GLOVE_COL = "#ffaa3c";
const DIVERGENCE_COL = "rgba(255,255,255,0.6)";

let host;
let mode = "both";

function visionPose(state) {
  if (state.pose?.engine === "apple_vision_2d" && state.pose.gloveWrists) return state.pose;
  if (state.poseSecondary?.engine === "apple_vision_2d" && state.poseSecondary.gloveWrists) return state.poseSecondary;
  return null;
}

export const WristSwapRule = {
  id: "wrist_swap",
  label: "Wrist swap (Vision vs Glove)",

  requires(slot) {
    // Needs Vision (skeleton) AND glove (wrist sidecar).
    return !!slot?.vision && !!slot?.glove;
  },

  // Suppress just the wrists in the base skeleton renderer — we'll redraw
  // them ourselves in the active mode's color. Keeps the rest of the skeleton
  // (shoulders, elbows, hips, knees…) untouched.
  skeletonStyle() {
    // Hide Vision's wrist joints (and the forearm edges touching them) so
    // the rule's draw() can paint its own wrist markers in the active mode's
    // color without conflicting with the base renderer.
    return { hideJoints: new Set([J.L_WRIST, J.R_WRIST]) };
  },

  mount(_host, state) {
    host = _host;
    const ok = !!visionPose(state);
    host.innerHTML = `
      <h2>Wrist source — Vision vs Glove model</h2>
      ${ok ? "" : `<p class="hint" style="color:var(--bad)">
        No glove-wrist cache for this round. Make sure the cache folder
        contains <code>&lt;base&gt;_glove_r{N}.npy</code> + <code>_meta.json</code>
        alongside the Vision cache.
      </p>`}
      <p class="hint">
        Same Vision skeleton, swap wrist source.
        <span style="color:${VISION_COL}">cyan</span> = Apple Vision wrist,
        <span style="color:${GLOVE_COL}">orange</span> = glove-model wrist.
        In <b>Both</b> mode, a faint white line connects the pair when they disagree.
      </p>
      <div class="rule-pick" style="display:flex;gap:6px;flex-wrap:wrap;margin:8px 0 14px">
        <button data-mode="vision" class="ws-mode">Vision wrists</button>
        <button data-mode="glove"  class="ws-mode">Glove wrists</button>
        <button data-mode="both"   class="ws-mode">Both side-by-side</button>
      </div>
      <h3>Per-frame Δ (px)</h3>
      <p class="hint">Distance between Vision's wrist and the glove model's wrist this frame.
        Empty when one side is missing. Conf shown as <code>V / G</code>.</p>
      <div class="metric-grid">
        <div class="metric"><div class="metric-label">L wrist</div><div class="metric-val" id="ws-l">—</div><div class="metric-sub" id="ws-l-conf">—</div></div>
        <div class="metric"><div class="metric-label">R wrist</div><div class="metric-val" id="ws-r">—</div><div class="metric-sub" id="ws-r-conf">—</div></div>
      </div>
      <h3>Wrist coverage (this round)</h3>
      <p class="hint">% of frames where each source has a usable wrist (conf > 0.2).</p>
      <div class="metric-grid" id="ws-cov">
        <div class="metric"><div class="metric-label">L · Vision</div><div class="metric-val" id="ws-cov-vl">—</div></div>
        <div class="metric"><div class="metric-label">L · Glove</div><div class="metric-val"  id="ws-cov-gl">—</div></div>
        <div class="metric"><div class="metric-label">R · Vision</div><div class="metric-val" id="ws-cov-vr">—</div></div>
        <div class="metric"><div class="metric-label">R · Glove</div><div class="metric-val"  id="ws-cov-gr">—</div></div>
      </div>
      <p class="metric-sub" id="ws-recovered">—</p>
    `;

    // Wire mode buttons
    for (const btn of host.querySelectorAll(".ws-mode")) {
      btn.style.cursor = "pointer";
      btn.style.padding = "5px 10px";
      btn.style.borderRadius = "4px";
      btn.style.border = "1px solid var(--muted, #555)";
      btn.style.background = "transparent";
      btn.style.color = "inherit";
      btn.addEventListener("click", () => {
        mode = btn.dataset.mode;
        for (const b of host.querySelectorAll(".ws-mode")) {
          b.style.background = (b.dataset.mode === mode) ? "var(--accent, #4da6ff)" : "transparent";
          b.style.color      = (b.dataset.mode === mode) ? "white" : "inherit";
        }
        // request a redraw — viewer auto-refreshes on next frame; force one now
        if (state.requestDraw) state.requestDraw();
      });
      if (btn.dataset.mode === mode) {
        btn.style.background = "var(--accent, #4da6ff)";
        btn.style.color = "white";
      }
    }

    // Pre-compute coverage stats once
    const vp = visionPose(state);
    if (vp) {
      const N = vp.n_frames;
      let vl=0, vr=0, gl=0, gr=0, recL=0, recR=0;
      for (let f = 0; f < N; f++) {
        const cvl = vp.conf[f * 17 + J.L_WRIST];
        const cvr = vp.conf[f * 17 + J.R_WRIST];
        const cgl = vp.gloveWrists.conf[f * 2 + 0];
        const cgr = vp.gloveWrists.conf[f * 2 + 1];
        const okvl = cvl > 0.2, okvr = cvr > 0.2;
        const okgl = cgl > 0.2 && !isNaN(vp.gloveWrists.wrists[(f*2+0)*2]);
        const okgr = cgr > 0.2 && !isNaN(vp.gloveWrists.wrists[(f*2+1)*2]);
        if (okvl) vl++; if (okvr) vr++; if (okgl) gl++; if (okgr) gr++;
        if (!okvl && okgl) recL++;
        if (!okvr && okgr) recR++;
      }
      const pct = (n) => (100 * n / N).toFixed(1) + "%";
      setText("ws-cov-vl", pct(vl));
      setText("ws-cov-vr", pct(vr));
      setText("ws-cov-gl", pct(gl));
      setText("ws-cov-gr", pct(gr));
      setText("ws-recovered", `Recovered (Vision missed → Glove found): L ${recL} frames · R ${recR} frames`);
    }
  },

  update(state) {
    const vp = visionPose(state);
    if (!vp) {
      setText("ws-l", "—"); setText("ws-r", "—");
      setText("ws-l-conf", "—"); setText("ws-r-conf", "—");
      return;
    }
    const f = state.frame;
    const upd = (id, side, jointIdx) => {
      const vx = vp.skeleton[(f * 17 + jointIdx) * 2];
      const vy = vp.skeleton[(f * 17 + jointIdx) * 2 + 1];
      const vc = vp.conf[f * 17 + jointIdx];
      const gx = vp.gloveWrists.wrists[(f * 2 + side) * 2];
      const gy = vp.gloveWrists.wrists[(f * 2 + side) * 2 + 1];
      const gc = vp.gloveWrists.conf[f * 2 + side];
      const vOk = vc > 0.05;
      const gOk = gc > 0.05 && !isNaN(gx);
      setHTML(`${id}-conf`,
        `<span style="color:${VISION_COL}">V ${vOk?vc.toFixed(2):"—"}</span> · ` +
        `<span style="color:${GLOVE_COL}">G ${gOk?gc.toFixed(2):"—"}</span>`);
      if (!vOk || !gOk) { setText(id, "—"); return; }
      setText(id, `${Math.hypot(vx-gx, vy-gy).toFixed(0)} px`);
    };
    upd("ws-l", 0, J.L_WRIST);
    upd("ws-r", 1, J.R_WRIST);
  },

  draw(ctx, state) {
    const vp = visionPose(state);
    if (!vp) return;
    const s = state.renderScale || 1;
    const f = state.frame;
    const drawDot = (x, y, color, r=8) => {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, r * s, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 2 * s;
      ctx.strokeStyle = "rgba(0,0,0,0.7)";
      ctx.stroke();
    };
    const drawForearm = (elbowJ, wristXY, color) => {
      const ec = vp.conf[f * 17 + elbowJ];
      if (ec < 0.2) return;
      const ex = vp.skeleton[(f * 17 + elbowJ) * 2];
      const ey = vp.skeleton[(f * 17 + elbowJ) * 2 + 1];
      ctx.strokeStyle = color;
      ctx.lineWidth = 2 * s;
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(wristXY[0], wristXY[1]);
      ctx.stroke();
    };

    for (const [side, jointIdx, elbowJ] of [
      [0, J.L_WRIST, J.L_ELBOW],
      [1, J.R_WRIST, J.R_ELBOW],
    ]) {
      const vx = vp.skeleton[(f * 17 + jointIdx) * 2];
      const vy = vp.skeleton[(f * 17 + jointIdx) * 2 + 1];
      const vc = vp.conf[f * 17 + jointIdx];
      const gx = vp.gloveWrists.wrists[(f * 2 + side) * 2];
      const gy = vp.gloveWrists.wrists[(f * 2 + side) * 2 + 1];
      const gc = vp.gloveWrists.conf[f * 2 + side];
      const vOk = vc > 0.2;
      const gOk = gc > 0.2 && !isNaN(gx);

      if (mode === "vision" || mode === "both") {
        if (vOk) {
          drawForearm(elbowJ, [vx, vy], "rgba(95,209,255,0.65)");
          drawDot(vx, vy, VISION_COL, 8);
        }
      }
      if (mode === "glove" || mode === "both") {
        if (gOk) {
          drawForearm(elbowJ, [gx, gy], "rgba(255,170,60,0.65)");
          drawDot(gx, gy, GLOVE_COL, 8);
        }
      }
      if (mode === "both" && vOk && gOk) {
        const d = Math.hypot(vx - gx, vy - gy);
        if (d > 4) {
          ctx.strokeStyle = DIVERGENCE_COL;
          ctx.lineWidth = 1.5 * s;
          ctx.setLineDash([4 * s, 3 * s]);
          ctx.beginPath();
          ctx.moveTo(vx, vy);
          ctx.lineTo(gx, gy);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    }
  },
};

function setText(id, value) {
  const el = host?.querySelector("#" + id);
  if (el) el.textContent = value;
}

function setHTML(id, html) {
  const el = host?.querySelector("#" + id);
  if (el) el.innerHTML = html;
}
