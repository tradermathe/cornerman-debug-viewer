// 3D skeleton inspector — big 3D view beside the video, 50/50.
//
// Purpose: eyeball whether the Apple 3D pose is any good. Puts the full
// 17-joint 3D skeleton in a large panel next to the video (each ~half the
// stage), so you can orbit it freely while the video plays as ground truth.
//
// Interaction (inside the 3D panel):
//   drag   → orbit (yaw + pitch)
//   wheel  → zoom
//   Reset  → default 3/4 view
//   Spin   → slow auto-rotate, so depth reads at a glance
//
// Layout: on mount we wrap the video and a fresh 3D canvas in a flex row
// inside #stage. Lenses have no unmount hook, so we restore the original
// layout via a one-shot listener on #rule-select when you switch away.

import { J3, drawSkeleton3D, makeView, project, EDGES_3D } from "../skeleton-3d.js";

const DEFAULT_VIEW = { yawRad: 0.5, pitchRad: 0.18, zoom: null };

let host;
let view = { ...DEFAULT_VIEW };
let dragging = null;
let spin = false, spinHandle = null;
let splitState = null;        // { stage, vw, vwCss, split }

function currentState() { return host?._lensState || null; }

export const Pose3DInspectRule = {
  id: "pose3d_inspect",
  label: "3D skeleton inspector",

  requires(slot) { return !!slot?.vision3d; },

  // Keep the video clean — it's the ground-truth reference for the 3D panel.
  skeletonStyle() { return { boneColor: "rgba(0,0,0,0)", jointRadius: 0, minConf: Infinity }; },

  mount(_host, state) {
    host = _host;
    view = { ...DEFAULT_VIEW };
    const have3d = !!state.pose3d;

    host.innerHTML = `
      <h2>3D skeleton inspector</h2>
      ${have3d ? `
        <p class="hint">Video (left) is ground truth; orbit the 3D skeleton (right)
          to judge whether the depth is any good. <b>Drag</b> to orbit · <b>wheel</b> to zoom.</p>
        <div class="controls" style="flex-wrap:wrap; gap:8px">
          <button type="button" id="p3d-reset">Reset view</button>
          <label style="display:inline-flex; gap:6px; align-items:center; color:var(--fg-muted); font-size:12px">
            <input type="checkbox" id="p3d-spin"> Auto-spin
          </label>
        </div>
        <div id="p3d-info" class="muted small" style="font-family:ui-monospace,monospace; margin-top:8px; line-height:1.6">—</div>
      ` : `
        <p class="hint" style="color:var(--bad)">No 3D cache for this round.
          Load a round that has <code>&lt;base&gt;_vision3d_r{N}.npy</code>.</p>
      `}
    `;

    if (!have3d) { stopSpin(); teardownSplit(); return; }

    const right = buildSplit();
    if (!right) { host.querySelector("#p3d-info").textContent = "could not build split layout"; return; }
    wireCanvas(right.querySelector("#p3d-canvas"));
    installCleanupOnSwitch();

    host.querySelector("#p3d-reset").addEventListener("click", () => {
      view = { ...DEFAULT_VIEW }; drawNow(currentState());
    });
    const spinChk = host.querySelector("#p3d-spin");
    spinChk.checked = spin;
    spinChk.addEventListener("change", () => spinChk.checked ? startSpin() : stopSpin());

    drawNow(state);
  },

  update(state) {
    if (!state.pose3d) return;
    drawNow(state);
  },
};

// ── stage split (video | 3D), with restore ─────────────────────────────────

function buildSplit() {
  if (document.getElementById("p3d-split")) {
    return document.getElementById("p3d-right");   // already built (re-mount)
  }
  const stage = document.getElementById("stage");
  const vw = stage?.querySelector(".video-wrap");
  if (!stage || !vw) return null;

  const split = document.createElement("div");
  split.id = "p3d-split";
  split.style.cssText = "display:flex; gap:12px; align-items:flex-start; width:100%";

  const right = document.createElement("div");
  right.id = "p3d-right";
  right.style.cssText = "flex:1 1 50%; min-width:0; position:relative; background:#0b1018;" +
    "border:1px solid var(--border); border-radius:6px; overflow:hidden;" +
    "aspect-ratio:var(--video-ratio,16/9); max-height:75vh";
  const cv = document.createElement("canvas");
  cv.id = "p3d-canvas";
  cv.style.cssText = "position:absolute; inset:0; width:100%; height:100%; cursor:grab; touch-action:none";
  right.appendChild(cv);

  const vwCss = vw.style.cssText;
  stage.insertBefore(split, vw);
  split.appendChild(vw);
  split.appendChild(right);
  vw.style.flex = "1 1 50%";
  vw.style.maxWidth = "none";
  vw.style.margin = "0";
  vw.style.minWidth = "0";

  splitState = { stage, vw, vwCss, split };
  return right;
}

function teardownSplit() {
  if (!splitState) return;
  const { stage, vw, vwCss, split } = splitState;
  if (split.parentNode === stage) {
    stage.insertBefore(vw, split);
    vw.style.cssText = vwCss;
    split.remove();
  }
  splitState = null;
}

function installCleanupOnSwitch() {
  const sel = document.getElementById("rule-select");
  if (!sel || sel._p3dCleanup) return;
  const handler = () => {
    if (sel.value === "pose3d_inspect") return;
    stopSpin();
    teardownSplit();
    sel.removeEventListener("change", handler);
    sel._p3dCleanup = false;
    document.getElementById("video")?.dispatchEvent(new Event("seeked"));  // re-fit overlay
  };
  sel.addEventListener("change", handler);
  sel._p3dCleanup = true;
}

// ── interaction ─────────────────────────────────────────────────────────────

function wireCanvas(cv) {
  cv.addEventListener("pointerdown", e => {
    cv.setPointerCapture(e.pointerId);
    dragging = { px: e.clientX, py: e.clientY, yaw: view.yawRad, pitch: view.pitchRad };
    cv.style.cursor = "grabbing";
  });
  cv.addEventListener("pointermove", e => {
    if (!dragging) return;
    view.yawRad = dragging.yaw + (e.clientX - dragging.px) * (Math.PI / 220);
    view.pitchRad = clamp(dragging.pitch + (e.clientY - dragging.py) * (Math.PI / 220),
      -Math.PI / 2 + 0.05, Math.PI / 2 - 0.05);
    drawNow(currentState());
  });
  const end = () => { dragging = null; cv.style.cursor = "grab"; };
  cv.addEventListener("pointerup", end);
  cv.addEventListener("pointercancel", end);
  cv.addEventListener("wheel", e => {
    e.preventDefault();
    view.zoom = (view.zoom ?? defaultZoom(cv)) * Math.exp(-e.deltaY * 0.001);
    drawNow(currentState());
  }, { passive: false });
}

function startSpin() {
  if (spin) return;
  spin = true;
  const tick = () => {
    if (!spin) return;
    view.yawRad += 0.012;
    drawNow(currentState());
    spinHandle = requestAnimationFrame(tick);
  };
  spinHandle = requestAnimationFrame(tick);
}
function stopSpin() {
  spin = false;
  if (spinHandle) cancelAnimationFrame(spinHandle);
  spinHandle = null;
}

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function defaultZoom(cv) { return (cv.height * 0.7) / 1.7; }

// ── render ──────────────────────────────────────────────────────────────────

function drawNow(state) {
  if (state) host._lensState = state;
  state = state || currentState();
  const cv = document.getElementById("p3d-canvas");
  if (!cv || !state?.pose3d) return;

  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const rect = cv.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width * dpr)), h = Math.max(1, Math.round(rect.height * dpr));
  if (cv.width !== w) cv.width = w;
  if (cv.height !== h) cv.height = h;

  const ctx = cv.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#0b1018";
  ctx.fillRect(0, 0, w, h);

  drawFloor(ctx, cv, state);

  const v = makeView({ width: w, height: h, yawRad: view.yawRad, pitchRad: view.pitchRad, zoom: view.zoom });
  drawSkeleton3D(ctx, state.pose3d.xyz, state.frame, v, {
    boneColor: "rgba(180,220,255,0.9)",
    jointColor: "#9fdfff",
    highlightJoints: new Set([J3.L_ELBOW, J3.R_ELBOW, J3.L_WRIST, J3.R_WRIST]),
    highlightColor: "#ffd966",
    jointRadius: 4 * dpr,
    boneWidth: 2.2 * dpr,
  });
  drawCompass(ctx, v, dpr);

  ctx.fillStyle = "rgba(180,200,220,0.85)";
  ctx.font = `${12 * dpr}px ui-monospace, monospace`;
  ctx.textAlign = "left";
  ctx.fillText(`frame ${state.frame}${spin ? "  · spinning" : ""}`, 10 * dpr, h - 10 * dpr);

  const info = host?.querySelector("#p3d-info");
  if (info) info.textContent = `yaw ${(view.yawRad * 180 / Math.PI).toFixed(0)}° · pitch ${(view.pitchRad * 180 / Math.PI).toFixed(0)}° · frame ${state.frame}`;
}

function drawFloor(ctx, cv, state) {
  const base = state.frame * 17 * 3;
  const la = state.pose3d.xyz[base + J3.L_ANKLE * 3 + 1];
  const ra = state.pose3d.xyz[base + J3.R_ANKLE * 3 + 1];
  const y0 = Number.isFinite(la) && Number.isFinite(ra) ? Math.min(la, ra) : -1.0;
  const v = makeView({ width: cv.width, height: cv.height, yawRad: view.yawRad, pitchRad: view.pitchRad, zoom: view.zoom, perspectiveStrength: 0.4 });
  ctx.strokeStyle = "rgba(95,125,165,0.22)";
  ctx.lineWidth = 1;
  const span = 1.5, step = 0.25;
  for (let g = -span; g <= span + 1e-6; g += step) {
    const [ax, ay] = project(v, g, y0, -span), [bx, by] = project(v, g, y0, span);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    const [cx, cy] = project(v, -span, y0, g), [dx, dy] = project(v, span, y0, g);
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(dx, dy); ctx.stroke();
  }
}

function drawCompass(ctx, v, dpr) {
  const cx = 34 * dpr, cy = 34 * dpr, len = 22 * dpr;
  const tip = (x, y, z) => project({ ...v, cx, cy, zoom: len, perspectiveStrength: 0 }, x, y, z);
  for (const a of [
    { p: tip(1, 0, 0), c: "#ff7a7a", t: "X" },
    { p: tip(0, 1, 0), c: "#7afa9a", t: "Y" },
    { p: tip(0, 0, 1), c: "#7ec8ff", t: "Z" },
  ]) {
    ctx.strokeStyle = a.c; ctx.lineWidth = 2 * dpr;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(a.p[0], a.p[1]); ctx.stroke();
    ctx.fillStyle = a.c; ctx.font = `${11 * dpr}px ui-monospace, monospace`;
    ctx.fillText(a.t, a.p[0] + 3 * dpr, a.p[1] - 3 * dpr);
  }
}
