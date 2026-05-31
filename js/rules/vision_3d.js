// Vision 3D lens — renders the Apple-native 17-joint 3D skeleton beside
// the video so you can visually check whether the 3D output matches the
// pose in the source frame.
//
// Requires the cache folder to contain `<base>_vision3d_r{N}.npy` (and
// optionally the matching `_cam.npy`). If absent, the lens explains how
// to populate it. The 2D skeleton on the video is suppressed so the visual
// reference is just the source frame itself.
//
// Interaction:
//   - drag inside the 3D canvas → orbit the view (yaw + pitch)
//   - wheel inside the 3D canvas → zoom
//   - "Reset" button → straight-on body-frame view
//   - "Apply 2D mask" toggle → dim joints whose 2D analogue had low
//      confidence (the bag-occlusion gating signal from the notebook).
//
// The 2D-mask map is derived from `state.pose` (the primary 2D engine):
//   12 shared joints     → direct 2D confidence
//   3D-only torso joints → mean of their 2D anatomical neighbours
//
// See: feedback_wrist_tracking_gloves.md and the comparison notebook scaffold.

import { J as J2 } from "../skeleton.js";
import {
  J3, JOINT_NAMES_3D, drawSkeleton3D, makeView, project, EDGES_3D,
} from "../skeleton-3d.js";

// 3D joint -> list of COCO-17 (2D) joint indices whose mean confidence we use
// as the 3D joint's gate.
const CONF_NEIGHBOURS = {
  [J3.TOP_HEAD]:        [J2.NOSE, J2.L_EYE, J2.R_EYE, J2.L_EAR, J2.R_EAR],
  [J3.CENTER_HEAD]:     [J2.NOSE, J2.L_EYE, J2.R_EYE, J2.L_EAR, J2.R_EAR],
  [J3.CENTER_SHOULDER]: [J2.L_SHOULDER, J2.R_SHOULDER],
  [J3.L_SHOULDER]:      [J2.L_SHOULDER],
  [J3.R_SHOULDER]:      [J2.R_SHOULDER],
  [J3.L_ELBOW]:         [J2.L_ELBOW],
  [J3.R_ELBOW]:         [J2.R_ELBOW],
  [J3.L_WRIST]:         [J2.L_WRIST],
  [J3.R_WRIST]:         [J2.R_WRIST],
  [J3.SPINE]:           [J2.L_SHOULDER, J2.R_SHOULDER, J2.L_HIP, J2.R_HIP],
  [J3.ROOT]:            [J2.L_HIP, J2.R_HIP],
  [J3.L_HIP]:           [J2.L_HIP],
  [J3.R_HIP]:           [J2.R_HIP],
  [J3.L_KNEE]:          [J2.L_KNEE],
  [J3.R_KNEE]:          [J2.R_KNEE],
  [J3.L_ANKLE]:         [J2.L_ANKLE],
  [J3.R_ANKLE]:         [J2.R_ANKLE],
};

// Module-level state. The lens is a singleton in the registry, so this is
// safe — it's overwritten every time mount() runs.
let host;
let canvas;
let view = { yawRad: 0, pitchRad: 0, zoom: null };
let dragging = null;       // { px, py, yaw, pitch } at drag start
let useMask = true;        // 2D-confidence mask toggle
let overlayOnVideo = true; // draw projected 3D skeleton on the video canvas
let infoBox = null;        // text element under the canvas
let confMaskCache = { token: null, mask: null };  // tied to pose identity
let armReachCache = { token: null, reach: null }; // theoretical max reach per side

const PUNCH_EXT_GATE = 0.75;  // min reach-fraction before we call an arm a punch
                              // (3D reach caps ~0.8 even on a straight arm, so
                              //  0.75 ≈ "near full extension"; tune to taste)

export const Vision3DRule = {
  id: "vision_3d",
  label: "Vision 3D skeleton (vs video)",

  // This lens requires the experimental 3D cache. Used by the viewer's
  // lens-aware filter to hide videos/rounds that don't have 3D data.
  requires(slot) { return !!slot?.vision3d; },

  // Hide the default 2D skeleton — the video itself is the visual reference,
  // and overlaying a 2D skeleton on top would just be noise.
  skeletonStyle() {
    return { boneColor: "rgba(0,0,0,0)", jointRadius: 0, minConf: Infinity };
  },

  mount(_host, state) {
    host = _host;
    const pose3d = state.pose3d;
    const have3d = !!pose3d;

    host.innerHTML = `
      <h2>Vision 3D skeleton</h2>
      ${have3d ? "" : `
        <p class="hint" style="color:var(--bad)">
          No 3D cache for this round. Run <code>PoseExtract3D</code>
          (see <code>cornerman-pose-cli-3d/</code>) to produce
          <code>&lt;base&gt;_vision3d_r{N}.npy</code> and
          <code>&lt;base&gt;_vision3d_r{N}_cam.npy</code>, then reload.
        </p>
      `}

      <canvas id="v3d-canvas" width="340" height="380"
        style="display:block; width:100%; height:auto; background:#0b1018;
               border:1px solid var(--border); border-radius:6px; cursor:grab;
               touch-action:none;"></canvas>

      <div class="controls" style="margin-top:8px; flex-wrap:wrap;">
        <button type="button" id="v3d-reset" title="Straight-on view">Reset view</button>
        <label style="display:inline-flex; gap:6px; align-items:center;
                      color:var(--fg-muted); font-size:12px;">
          <input type="checkbox" id="v3d-mask" ${useMask ? "checked" : ""}>
          Apply 2D-conf mask
        </label>
        <label style="display:inline-flex; gap:6px; align-items:center;
                      color:var(--fg-muted); font-size:12px;"
               title="Draws the 3D-derived skeleton on top of the video using Apple's pointInImage(). Needs _proj.npy.">
          <input type="checkbox" id="v3d-overlay" ${overlayOnVideo ? "checked" : ""}>
          Overlay on video
        </label>
      </div>

      <p class="hint">
        Drag inside the canvas to orbit · wheel to zoom. "Apply 2D-conf mask"
        dims any 3D joint whose 2D analogue is low-confidence (bag-occlusion gate).
        "Overlay on video" projects the 3D skeleton onto the source frame so you
        can compare directly to where the boxer's joints actually are.
      </p>

      <div id="v3d-info" class="muted small"
           style="font-family:ui-monospace,'SF Mono',monospace; margin-top:8px;
                  line-height:1.6;">—</div>
    `;

    canvas = host.querySelector("#v3d-canvas");
    infoBox = host.querySelector("#v3d-info");
    const resetBtn = host.querySelector("#v3d-reset");
    const maskChk = host.querySelector("#v3d-mask");
    const overlayChk = host.querySelector("#v3d-overlay");

    if (have3d) {
      // Default view: straight-on from in front of the boxer.
      view = { yawRad: 0, pitchRad: 0, zoom: null };
      wireInteractions();
      resetBtn.addEventListener("click", () => {
        view = { yawRad: 0, pitchRad: 0, zoom: null };
        drawNow(state);
      });
      maskChk.addEventListener("change", () => {
        useMask = maskChk.checked;
        drawNow(state);
      });
      overlayChk.addEventListener("change", () => {
        overlayOnVideo = overlayChk.checked;
        // The overlay lives on the main video canvas — ask the viewer to
        // repaint so draw() reflects the new state immediately.
        if (typeof window.__viewerRedraw === "function") window.__viewerRedraw();
        drawNow(state);
      });
      drawNow(state);
    } else {
      // Clear the canvas to a neutral message.
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#161a21";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "rgba(180,200,220,0.6)";
      ctx.font = "13px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("no 3D cache for this round",
                   canvas.width / 2, canvas.height / 2);
    }
  },

  update(state) {
    if (state.pose3d) drawNow(state);
  },

  // Overlay the 3D-derived skeleton on the video canvas, in image-space
  // coordinates from Apple's pointInImage(). Drawn ONLY when the user
  // toggles "Overlay on video"; otherwise the video stays clean and the
  // lens lives entirely in the side canvas.
  //
  // Requires the _proj.npy sidecar; if absent (older 3D caches) the toggle
  // is effectively a no-op and the info line says so.
  draw(ctx, state) {
    if (!overlayOnVideo) return;
    const pose3d = state.pose3d;
    if (!pose3d || !pose3d.projection) return;

    const f = state.frame;
    const N = 17;
    const W = ctx.canvas.width;
    const H = ctx.canvas.height;
    const proj = pose3d.projection;
    const s = state.renderScale || 1;

    const confMask = useMask ? get2dDerivedMask(state) : null;
    const trust = j => !confMask || confMask[f * N + j] >= 0.3;

    // Pre-resolve joint positions in pixel coords (or null if missing).
    const xy = new Array(N);
    for (let j = 0; j < N; j++) {
      const x = proj[(f * N + j) * 2 + 0];
      const y = proj[(f * N + j) * 2 + 1];
      if (!Number.isFinite(x) || !Number.isFinite(y)) { xy[j] = null; continue; }
      xy[j] = [x * W, y * H];
    }

    // Bones.
    ctx.lineWidth = 3 * s;
    for (const [a, b] of EDGES_3D) {
      const A = xy[a], B = xy[b];
      if (!A || !B) continue;
      const dim = !trust(a) || !trust(b);
      ctx.strokeStyle = dim ? "rgba(180, 220, 255, 0.20)" : "rgba(180, 220, 255, 0.95)";
      ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke();
    }
    // Joints.
    for (let j = 0; j < N; j++) {
      if (!xy[j]) continue;
      const dim = !trust(j);
      const isWrist = (j === J3.L_WRIST || j === J3.R_WRIST);
      const r = (isWrist ? 7 : 5) * s;
      ctx.fillStyle = isWrist
        ? (dim ? "rgba(255, 217, 102, 0.30)" : "#ffd966")
        : (dim ? "rgba(159, 223, 255, 0.25)" : "#9fdfff");
      ctx.beginPath(); ctx.arc(xy[j][0], xy[j][1], r, 0, Math.PI * 2); ctx.fill();
    }

    // Predicted punch direction: arrow from shoulder→wrist on the video, plus
    // a toward/away/across text tag. The on-video arrow foreshortens for
    // camera-axis punches, so the words carry the depth call.
    const pd = punchDirection(state, f);
    if (pd && pd.frac > PUNCH_EXT_GATE) {
      const Sx = proj[(f * N + pd.s) * 2 + 0], Sy = proj[(f * N + pd.s) * 2 + 1];
      const Wx = proj[(f * N + pd.w) * 2 + 0], Wy = proj[(f * N + pd.w) * 2 + 1];
      if ([Sx, Sy, Wx, Wy].every(Number.isFinite)) {
        const sxp = Sx * W, syp = Sy * H, wxp = Wx * W, wyp = Wy * H;
        const dx = wxp - sxp, dy = wyp - syp, len = Math.hypot(dx, dy);
        const col = punchDirColor(pd.azDeg);
        ctx.lineWidth = 4 * s; ctx.strokeStyle = col; ctx.fillStyle = col;
        let lx = wxp, ly = wyp;
        if (len > 1) {
          const ux = dx / len, uy = dy / len;
          lx = wxp + ux * 30 * s * Math.min(1.4, pd.frac);
          ly = wyp + uy * 30 * s * Math.min(1.4, pd.frac);
          ctx.beginPath(); ctx.moveTo(sxp, syp); ctx.lineTo(lx, ly); ctx.stroke();
          const ang = Math.atan2(uy, ux), ah = 10 * s;
          ctx.beginPath(); ctx.moveTo(lx, ly);
          ctx.lineTo(lx - ah * Math.cos(ang - 0.4), ly - ah * Math.sin(ang - 0.4));
          ctx.lineTo(lx - ah * Math.cos(ang + 0.4), ly - ah * Math.sin(ang + 0.4));
          ctx.closePath(); ctx.fill();
        } else {
          ctx.beginPath(); ctx.arc(wxp, wyp, 8 * s, 0, Math.PI * 2); ctx.stroke();
        }
        ctx.font = `${12 * s}px ui-monospace, monospace`;
        ctx.textAlign = "left";
        ctx.fillText(
          `${pd.hand} ${punchDirLabel(pd.azDeg)} (az ${pd.azDeg.toFixed(0)}°)`,
          lx + 8 * s, ly - 6 * s,
        );
      }
    }
  },
};

function wireInteractions() {
  canvas.addEventListener("pointerdown", e => {
    canvas.setPointerCapture(e.pointerId);
    dragging = {
      px: e.clientX, py: e.clientY,
      yaw: view.yawRad, pitch: view.pitchRad,
    };
    canvas.style.cursor = "grabbing";
  });
  canvas.addEventListener("pointermove", e => {
    if (!dragging) return;
    const dx = e.clientX - dragging.px;
    const dy = e.clientY - dragging.py;
    // 200 px drag = full ~180° rotation. Plenty for inspection.
    view.yawRad   = dragging.yaw   + dx * (Math.PI / 200);
    view.pitchRad = clamp(
      dragging.pitch + dy * (Math.PI / 200),
      -Math.PI / 2 + 0.05, Math.PI / 2 - 0.05,
    );
    drawNow(currentState());
  });
  const endDrag = () => {
    dragging = null;
    canvas.style.cursor = "grab";
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    const factor = Math.exp(-e.deltaY * 0.001);
    view.zoom = (view.zoom ?? defaultZoom()) * factor;
    drawNow(currentState());
  }, { passive: false });
}

// `currentState` is a small indirection so the pointermove handler — which
// captures `state` by closure at mount time — sees the up-to-date frame.
// We pull from the singleton `state.rule`-bound viewer via the global the
// viewer manages; simpler approach is to grab from a closure-captured ref
// at mount(). For now, the registry hands us state every update() — and the
// pointermove redraw only needs `view`, `state.pose3d`, and `state.frame`,
// all of which are stable references via the host's data attribute below.
function currentState() {
  return host?._lensState || {};
}

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

function defaultZoom() {
  // Matches the default in makeView() — keep in sync.
  return (canvas.height * 0.7) / 1.7;
}

function drawNow(state) {
  // Stash the latest state on the host so pointer handlers can find it after
  // mount() returns. (`state` is the same object the viewer mutates in place,
  // so re-reading frame/pose3d each draw stays current.)
  if (host) host._lensState = state;
  if (!canvas || !state?.pose3d) return;

  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#0b1018";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Floor grid for depth cue: a small grid at y = ankle height.
  drawFloorGrid(ctx, state);

  const v = makeView({
    width: canvas.width,
    height: canvas.height,
    yawRad: view.yawRad,
    pitchRad: view.pitchRad,
    zoom: view.zoom,
    perspectiveStrength: 0,
  });

  const confMask = useMask ? get2dDerivedMask(state) : null;
  const frame = state.frame;

  drawSkeleton3D(ctx, state.pose3d.xyz, frame, v, {
    confMask,
    maskThreshold: 0.3,
    boneColor: "rgba(180, 220, 255, 0.85)",
    dimColor:  "rgba(180, 220, 255, 0.18)",
    jointColor: "#9fdfff",
    dimJointColor: "rgba(159, 223, 255, 0.20)",
    highlightJoints: new Set([J3.L_WRIST, J3.R_WRIST]),
    highlightColor: "#ffd966",
    jointRadius: 4,
    boneWidth: 2,
  });

  // Compass — body-axis arrows in the top-left so you know which way is
  // forward (+Z = boxer's forward, away from the camera in body-frame).
  drawCompass(ctx, v);

  // Bird's-eye punch-direction dial (bottom-right), camera marked at bottom.
  drawTopDownDial(ctx, punchDirection(state, frame));

  updateInfo(state, confMask);
}

function drawFloorGrid(ctx, state) {
  const { pose3d } = state;
  const N = 17;
  const base = state.frame * N * 3;
  // Estimate floor height from ankle Y at this frame.
  const la = pose3d.xyz[base + J3.L_ANKLE * 3 + 1];
  const ra = pose3d.xyz[base + J3.R_ANKLE * 3 + 1];
  const y0 = Number.isFinite(la) && Number.isFinite(ra) ? Math.min(la, ra)
           : -1.0;
  const v = makeView({
    width: canvas.width, height: canvas.height,
    yawRad: view.yawRad, pitchRad: view.pitchRad,
    zoom: view.zoom, perspectiveStrength: 0.4,
  });
  ctx.strokeStyle = "rgba(95, 125, 165, 0.25)";
  ctx.lineWidth = 1;
  const step = 0.25;        // 25 cm grid
  const span = 1.5;         // ±1.5 m from root
  for (let g = -span; g <= span + 1e-6; g += step) {
    // Lines along Z (front/back) at fixed X
    const [ax, ay] = project(v, g, y0, -span);
    const [bx, by] = project(v, g, y0,  span);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    // Lines along X (left/right) at fixed Z
    const [cx, cy] = project(v, -span, y0, g);
    const [dx, dy] = project(v,  span, y0, g);
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(dx, dy); ctx.stroke();
  }
}

function drawCompass(ctx, v) {
  // Small 3-axis indicator in the top-left of the canvas, in body-axis
  // colours: X red, Y green, Z blue. Apply the same rotation as the main
  // view so the arrows reflect what "forward" currently looks like.
  const cx = 32, cy = 32, len = 22;
  const tip = (x, y, z) => {
    const [sx, sy] = project(
      { ...v, cx, cy, zoom: len, perspectiveStrength: 0 },
      x, y, z
    );
    return [sx, sy];
  };
  const labels = [
    { p: tip(1, 0, 0), color: "#ff7a7a", text: "X" },
    { p: tip(0, 1, 0), color: "#7afa9a", text: "Y" },
    { p: tip(0, 0, 1), color: "#7ec8ff", text: "Z" },
  ];
  for (const a of labels) {
    ctx.strokeStyle = a.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(a.p[0], a.p[1]);
    ctx.stroke();
    ctx.fillStyle = a.color;
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText(a.text, a.p[0] + 3, a.p[1] - 3);
  }
}

// ── Punch direction (camera-referenced) ─────────────────────────────────────
// Decompose the extended arm's shoulder→wrist vector against the REAL camera
// direction (from cameraOriginMatrix), not the body +Z axis — so azimuth is
// 0° = straight at the camera, ±90° = across the frame, ±180° = away. That
// toward/away axis is exactly what flat 2D footage foreshortens away.
function camPosBody(pose3d, f) {
  if (!pose3d.camMatrices) return null;
  // Row-major 4x4 translation column = camera position in body-frame metres.
  const m = f * 16;
  const c = [pose3d.camMatrices[m + 3], pose3d.camMatrices[m + 7], pose3d.camMatrices[m + 11]];
  return Number.isFinite(c[0]) ? c : null;
}
function jointXYZ3(pose3d, f, j) {
  const b = (f * 17 + j) * 3;
  return [pose3d.xyz[b], pose3d.xyz[b + 1], pose3d.xyz[b + 2]];
}
function punchDirection(state, f) {
  const pose3d = state.pose3d;
  if (!pose3d) return null;
  const C = camPosBody(pose3d, f);
  if (!C) return null;
  const reach = getArmReach(state);
  // Pick the more-extended arm as "the punch".
  let best = null;
  for (const sd of [
    { hand: "L", s: J3.L_SHOULDER, w: J3.L_WRIST, reach: reach?.L },
    { hand: "R", s: J3.R_SHOULDER, w: J3.R_WRIST, reach: reach?.R },
  ]) {
    const S = jointXYZ3(pose3d, f, sd.s);
    const Wp = jointXYZ3(pose3d, f, sd.w);
    if (!Number.isFinite(S[0]) || !Number.isFinite(Wp[0])) continue;
    const p = [Wp[0] - S[0], Wp[1] - S[1], Wp[2] - S[2]];
    const ext = Math.hypot(p[0], p[1], p[2]);
    const frac = sd.reach ? ext / sd.reach : ext;
    if (!best || frac > best.frac) best = { hand: sd.hand, s: sd.s, w: sd.w, S, p, frac };
  }
  if (!best) return null;
  const S = best.S, p = best.p;
  // Horizontal toward-camera unit vector (body x/z plane) anchored at the
  // shoulder, plus its lateral perpendicular.
  const th = [C[0] - S[0], C[2] - S[2]];
  const tl = Math.hypot(th[0], th[1]) || 1;
  const t = [th[0] / tl, th[1] / tl];
  const r = [t[1], -t[0]];
  const aTow = p[0] * t[0] + p[2] * t[1];
  const aLat = p[0] * r[0] + p[2] * r[1];
  // az: 0 = toward camera, ±90 = across, ±180 = away. toward/away uses the real
  // camera position (robust); the across-left/right SIGN depends on Apple's
  // X-handedness (its +Z label is posterior, not anterior) so it may be
  // mirrored — confirm against one clip and flip aLat's sign if so.
  const azDeg = Math.atan2(aLat, aTow) * 180 / Math.PI;
  const elDeg = Math.atan2(p[1], Math.hypot(p[0], p[2])) * 180 / Math.PI;
  return { hand: best.hand, s: best.s, w: best.w, frac: best.frac, azDeg, elDeg };
}
function punchDirLabel(az) {
  const a = Math.abs(az);
  if (a <= 35) return "TOWARD camera";
  if (a >= 145) return "AWAY from camera";
  return az > 0 ? "ACROSS →" : "ACROSS ←";
}
function punchDirColor(az) {
  const a = Math.abs(az);
  if (a <= 35) return "#ff6b4a";   // toward
  if (a >= 145) return "#4aa8ff";  // away
  return "#7afa9a";                // across
}
// Bird's-eye azimuth dial in the side canvas: camera at the bottom, "away" at
// the top, so a toward-camera punch points down. Makes the depth axis legible
// where the on-video arrow (foreshortened) can't.
function drawTopDownDial(ctx, pd) {
  const R = 34, cx = canvas.width - R - 14, cy = canvas.height - R - 14;
  ctx.save();
  ctx.fillStyle = "rgba(11,16,24,0.88)";
  ctx.strokeStyle = "rgba(95,125,165,0.5)";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.fillStyle = "rgba(180,200,220,0.8)";
  ctx.font = "9px ui-monospace, monospace";
  ctx.textAlign = "center";
  ctx.fillText("▲ cam", cx, cy + R - 3);
  ctx.fillText("away", cx, cy - R + 9);
  ctx.fillStyle = "#9fdfff";
  ctx.beginPath(); ctx.arc(cx, cy, 2.5, 0, Math.PI * 2); ctx.fill();
  if (pd && pd.frac > PUNCH_EXT_GATE && Number.isFinite(pd.azDeg)) {
    const a = pd.azDeg * Math.PI / 180;
    const tx = cx + Math.sin(a) * (R - 6), ty = cy + Math.cos(a) * (R - 6);
    const col = punchDirColor(pd.azDeg);
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(tx, ty); ctx.stroke();
    const ang = Math.atan2(ty - cy, tx - cx), ah = 7;
    ctx.beginPath(); ctx.moveTo(tx, ty);
    ctx.lineTo(tx - ah * Math.cos(ang - 0.4), ty - ah * Math.sin(ang - 0.4));
    ctx.lineTo(tx - ah * Math.cos(ang + 0.4), ty - ah * Math.sin(ang + 0.4));
    ctx.closePath(); ctx.fill();
  }
  ctx.restore();
}

function updateInfo(state, confMask) {
  if (!infoBox) return;
  const pose3d = state.pose3d;
  const f = state.frame;
  const N = 17;
  const base = f * N * 3;

  // Per-frame body yaw from cameraOriginMatrix (atan2(-R[2,0], R[0,0])).
  let bodyYawDeg = null;
  if (pose3d.camMatrices) {
    const m = f * 16;
    // Row-major 4x4: row r, col c at m + r*4 + c.
    const r20 = pose3d.camMatrices[m + 2 * 4 + 0];
    const r00 = pose3d.camMatrices[m + 0 * 4 + 0];
    bodyYawDeg = Math.atan2(-r20, r00) * (180 / Math.PI);
  }

  // Shoulder-to-wrist extension for both arms, in metres, and as a percentage
  // of the theoretical maximum reach (upper arm + forearm bone lengths from
  // the same skeleton). The bone lengths are fixed by Apple's parametric body
  // model — so "reach %" measures how much of full extension the model is
  // willing to show. Empirically caps around 80% even when the arm is visibly
  // straight in the source video.
  const ext = side => {
    const sIdx = side === "L" ? J3.L_SHOULDER : J3.R_SHOULDER;
    const wIdx = side === "L" ? J3.L_WRIST    : J3.R_WRIST;
    const dx = pose3d.xyz[base + wIdx * 3 + 0] - pose3d.xyz[base + sIdx * 3 + 0];
    const dy = pose3d.xyz[base + wIdx * 3 + 1] - pose3d.xyz[base + sIdx * 3 + 1];
    const dz = pose3d.xyz[base + wIdx * 3 + 2] - pose3d.xyz[base + sIdx * 3 + 2];
    return Math.hypot(dx, dy, dz);
  };
  const reach = getArmReach(state);
  const lExt = ext("L"), rExt = ext("R");
  const lPct = reach ? (lExt / reach.L * 100) : null;
  const rPct = reach ? (rExt / reach.R * 100) : null;

  // Counts of masked joints at this frame.
  let trustedCount = 17, totalCount = 17;
  if (confMask) {
    trustedCount = 0;
    for (let j = 0; j < N; j++) {
      if (confMask[f * N + j] >= 0.3) trustedCount++;
    }
  }

  const fmt = v => Number.isFinite(v) ? v.toFixed(2) : "—";
  const pct = v => v == null ? "" : ` (${v.toFixed(0)}%)`;
  const noProj = !pose3d.projection
    ? ` · <span style="color:var(--bad)">no _proj.npy — re-extract for overlay</span>`
    : "";
  const pd = punchDirection(state, f);
  const pdLine = pd && pd.frac > PUNCH_EXT_GATE
    ? `<br>punch <b>${pd.hand}</b>: <span style="color:${punchDirColor(pd.azDeg)}">${punchDirLabel(pd.azDeg)}</span> · az ${pd.azDeg.toFixed(0)}° · el ${pd.elDeg.toFixed(0)}° · ext ${(pd.frac * 100).toFixed(0)}%`
    : `<br>punch dir: <span class="muted">(no arm extended)</span>`;
  infoBox.innerHTML =
    `frame ${f}   ·   L ext ${fmt(lExt)} m${pct(lPct)}   ·   R ext ${fmt(rExt)} m${pct(rPct)}<br>` +
    (bodyYawDeg !== null
      ? `body yaw vs camera: ${bodyYawDeg.toFixed(1)}°   ·   `
      : `(no cam matrix)   ·   `) +
    `trusted joints: ${trustedCount}/${totalCount}` +
    (confMask ? ` (2D gate @0.3)` : ` (gate off)`) +
    noProj + pdLine;
}

// Per-arm theoretical max reach = upper arm + forearm bone lengths.
// Apple's body model is parametric so these are essentially constant across
// the whole round — we just average over the first 30 detected frames to be
// safe. Cached per pose-source so frame scrubbing doesn't recompute it.
function getArmReach(state) {
  const pose3d = state.pose3d;
  if (!pose3d) return null;
  const token = pose3d.source;
  if (armReachCache.token === token && armReachCache.reach) return armReachCache.reach;
  const N = 17;
  const samples = Math.min(60, pose3d.n_frames);
  const sums = { L_upper: 0, L_fore: 0, R_upper: 0, R_fore: 0 };
  let count = 0;
  for (let f = 0; f < samples; f++) {
    const i0 = f * N;
    const lSh = i0 + J3.L_SHOULDER;
    const lEl = i0 + J3.L_ELBOW;
    const lWr = i0 + J3.L_WRIST;
    const rSh = i0 + J3.R_SHOULDER;
    const rEl = i0 + J3.R_ELBOW;
    const rWr = i0 + J3.R_WRIST;
    const dist = (a, b) => {
      const ax = pose3d.xyz[a*3 + 0], ay = pose3d.xyz[a*3 + 1], az = pose3d.xyz[a*3 + 2];
      const bx = pose3d.xyz[b*3 + 0], by = pose3d.xyz[b*3 + 1], bz = pose3d.xyz[b*3 + 2];
      return Math.hypot(ax - bx, ay - by, az - bz);
    };
    const lu = dist(lSh, lEl), lf = dist(lEl, lWr);
    const ru = dist(rSh, rEl), rf = dist(rEl, rWr);
    if (!Number.isFinite(lu) || !Number.isFinite(lf)) continue;
    sums.L_upper += lu; sums.L_fore += lf;
    sums.R_upper += ru; sums.R_fore += rf;
    count++;
  }
  if (count === 0) return null;
  const reach = {
    L: (sums.L_upper + sums.L_fore) / count,
    R: (sums.R_upper + sums.R_fore) / count,
  };
  armReachCache = { token, reach };
  return reach;
}

// Build a (n_frames, 17) Float32Array of per-joint confidence for the 3D
// layout, derived from the active 2D pose. Cached on identity so swapping
// frames doesn't rebuild it each redraw.
function get2dDerivedMask(state) {
  const pose2d = state.pose;          // primary 2D engine
  const pose3d = state.pose3d;
  if (!pose2d || !pose3d) return null;
  // Token: the pair of source filenames is enough to invalidate when a new
  // round loads.
  const token = `${pose2d.source}::${pose3d.source}`;
  if (confMaskCache.token === token && confMaskCache.mask) {
    return confMaskCache.mask;
  }
  const N3 = 17;
  const N2 = 17;
  const nFrames = Math.min(pose2d.n_frames, pose3d.n_frames);
  const mask = new Float32Array(pose3d.n_frames * N3);
  for (let f = 0; f < nFrames; f++) {
    for (let j3 = 0; j3 < N3; j3++) {
      const srcs = CONF_NEIGHBOURS[j3];
      let sum = 0;
      for (const j2 of srcs) sum += pose2d.conf[f * N2 + j2];
      mask[f * N3 + j3] = sum / srcs.length;
    }
  }
  // Frames beyond pose2d's range stay at 0 (effectively masked out).
  confMaskCache = { token, mask };
  return mask;
}
