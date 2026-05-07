// Engine compare lens — shows the source video twice, side by side, with
// YOLO drawn on the main stage and Apple Vision drawn on a mirrored video
// inside the lens panel. Each engine's skeleton stays on its own pane so
// the small (~1 frame) seek-vs-walk timing difference between the two
// extractions doesn't show up as a "ghost" overlay disagreement.
//
// Visual encoding:
//   YOLO   — orange/yellow joints + warm bones (main stage)
//   Vision — cyan/green joints + cool bones    (lens-side mirror)

import { J } from "../skeleton.js";

const Y_BONE = "rgba(255,170,80,0.55)";
const V_BONE = "rgba(110,200,255,0.55)";
const Y_WRIST = "#ff8a3c";
const V_WRIST = "#5fd1ff";
const HIGHLIGHT = new Set([J.NOSE, J.L_WRIST, J.R_WRIST, J.L_ELBOW, J.R_ELBOW]);

let host;
let sideVideo = null;
let sideCanvas = null;
let mainVideo = null;
let syncHandlers = null;

export const EngineCompareRule = {
  id: "engine_compare",
  label: "Engine compare (YOLO vs Vision)",

  // Suppress the base skeleton renderer — we draw YOLO ourselves on the main
  // canvas (so it's a single-engine overlay rather than two on top).
  skeletonStyle() {
    return { boneColor: "rgba(0,0,0,0)", jointRadius: 0, minConf: Infinity };
  },

  mount(_host, state) {
    host = _host;
    teardownSync();
    const hasBoth = !!state.poseSecondary;
    host.innerHTML = `
      <h2>Engine compare (YOLO vs Vision)</h2>
      <p class="hint">
        Main stage shows <span style="color:${Y_WRIST}"><b>YOLO</b></span>.
        Pane below shows the same source with
        <span style="color:${V_WRIST}"><b>Apple Vision</b></span>.
        Each skeleton stays on its own pane so the seek-vs-walk frame timing
        difference doesn't read as overlay disagreement.
      </p>

      ${hasBoth ? `
        <div class="ec-side-wrap">
          <video id="ec-side-video" muted playsinline preload="auto"></video>
          <canvas id="ec-side-canvas"></canvas>
        </div>
      ` : `
        <p class="hint" style="color:var(--bad)">
          No second engine for this round. Make sure the cache folder
          contains both <code>&lt;base&gt;_yolo_r{N}.*</code> and
          <code>&lt;base&gt;_vision_r{N}.*</code> files.
        </p>
      `}

      <h3>Per-frame Δ (raw px)</h3>
      <p class="hint">Distance between YOLO and Vision detection of each joint at this frame.</p>
      <div class="metric-grid">
        <div class="metric"><div class="metric-label">L wrist</div><div class="metric-val" id="ec-l-wrist">—</div></div>
        <div class="metric"><div class="metric-label">R wrist</div><div class="metric-val" id="ec-r-wrist">—</div></div>
        <div class="metric"><div class="metric-label">Nose</div><div class="metric-val" id="ec-nose">—</div></div>
        <div class="metric"><div class="metric-label">L elbow</div><div class="metric-val" id="ec-l-elbow">—</div></div>
        <div class="metric"><div class="metric-label">R elbow</div><div class="metric-val" id="ec-r-elbow">—</div></div>
      </div>

      <h3>Wrist y over time</h3>
      <p class="hint">Both engines plotted side-by-side. Spikes that appear in
      one but not the other are the model disagreeing — usually means a glove
      anchor jump for YOLO.</p>
      <canvas id="ec-trace-l" width="320" height="100"></canvas>
      <div class="metric-sub" style="text-align:center">L wrist y</div>
      <canvas id="ec-trace-r" width="320" height="100"></canvas>
      <div class="metric-sub" style="text-align:center">R wrist y</div>
    `;

    if (hasBoth) setupSidePane(state);
  },

  draw(ctx, state) {
    // Main canvas: YOLO only.
    const s = state.renderScale || 1;
    drawEngineSkeleton(ctx, state.pose, state.frame, {
      boneColor: Y_BONE, boneWidth: 2 * s,
      jointColor: Y_WRIST, jointRadius: 4 * s,
      wristColor: Y_WRIST, wristRadius: 9 * s,
      strokeWidth: 2 * s,
    });

    // Side canvas: Vision only, drawn at the secondary pose's frame (which
    // we resolve by VIDEO TIME, not frame index — matters when start_sec/fps
    // differ between engines after the NTSC drift fix).
    if (sideCanvas && state.poseSecondary) {
      const sCtx = sideCanvas.getContext("2d");
      sCtx.clearRect(0, 0, sideCanvas.width, sideCanvas.height);
      const sf = secondaryFrame(state);
      if (sf != null) {
        const cssW = sideCanvas.getBoundingClientRect().width || sideCanvas.width;
        const ss = sideCanvas.width / Math.max(1, cssW);
        drawEngineSkeleton(sCtx, state.poseSecondary, sf, {
          boneColor: V_BONE, boneWidth: 2 * ss,
          jointColor: V_WRIST, jointRadius: 4 * ss,
          wristColor: V_WRIST, wristRadius: 9 * ss,
          strokeWidth: 2 * ss,
        });
      }
    }
  },

  update(state) {
    // Mirror main video state into the side pane every redraw — covers
    // keyboard scrub, scrubber drag, and play/pause without needing every
    // event hooked up.
    if (sideVideo && mainVideo) {
      if (sideVideo.src !== mainVideo.src) sideVideo.src = mainVideo.src;
      const dt = Math.abs(sideVideo.currentTime - mainVideo.currentTime);
      if (dt > 0.04) sideVideo.currentTime = mainVideo.currentTime;
      if (mainVideo.paused !== sideVideo.paused) {
        if (mainVideo.paused) sideVideo.pause();
        else sideVideo.play().catch(() => {});
      }
    }

    const f = state.frame;
    const a = state.pose;
    const b = state.poseSecondary;
    const sf = b ? secondaryFrame(state) : null;
    const setJointDiff = (id, j) => {
      if (!b) return setText(id, "—");
      if (sf == null) return setText(id, "out of range");
      const ax = a.skeleton[(f * 17 + j) * 2];
      const ay = a.skeleton[(f * 17 + j) * 2 + 1];
      const ac = a.conf[f * 17 + j];
      const bx = b.skeleton[(sf * 17 + j) * 2];
      const by = b.skeleton[(sf * 17 + j) * 2 + 1];
      const bc = b.conf[sf * 17 + j];
      if (ac < 0.05 || bc < 0.05) {
        setText(id, "low conf");
        return;
      }
      const d = Math.hypot(ax - bx, ay - by);
      setText(id, `${d.toFixed(0)} px`);
    };
    setJointDiff("ec-l-wrist", J.L_WRIST);
    setJointDiff("ec-r-wrist", J.R_WRIST);
    setJointDiff("ec-nose",    J.NOSE);
    setJointDiff("ec-l-elbow", J.L_ELBOW);
    setJointDiff("ec-r-elbow", J.R_ELBOW);

    if (b) {
      drawWristTrace(host.querySelector("#ec-trace-l"), a, b, J.L_WRIST, f);
      drawWristTrace(host.querySelector("#ec-trace-r"), a, b, J.R_WRIST, f);
    }
  },
};

function setupSidePane(state) {
  sideVideo = host.querySelector("#ec-side-video");
  sideCanvas = host.querySelector("#ec-side-canvas");
  mainVideo = document.getElementById("video");

  // Internal canvas resolution = source video's, so we draw with raw pixel
  // skeleton coords just like the main canvas. CSS scales both together.
  const w = state.poseSecondary.width  || mainVideo.videoWidth  || 16;
  const h = state.poseSecondary.height || mainVideo.videoHeight || 9;
  sideCanvas.width = w;
  sideCanvas.height = h;

  // Mirror aspect ratio so portrait videos don't blow up.
  const wrap = host.querySelector(".ec-side-wrap");
  if (wrap) wrap.style.setProperty("--video-ratio", `${w} / ${h}`);

  // Boot the side video at the main's current state.
  sideVideo.src = mainVideo.src;
  sideVideo.currentTime = mainVideo.currentTime;
  if (!mainVideo.paused) sideVideo.play().catch(() => {});

  // Belt-and-braces sync — update() runs each redraw and is the main path,
  // but these events let the side pane catch up faster on big jumps.
  syncHandlers = {
    seeked:    () => { sideVideo.currentTime = mainVideo.currentTime; },
    play:      () => sideVideo.play().catch(() => {}),
    pause:     () => sideVideo.pause(),
    ratechange: () => { sideVideo.playbackRate = mainVideo.playbackRate; },
  };
  for (const [evt, h] of Object.entries(syncHandlers)) {
    mainVideo.addEventListener(evt, h);
  }
}

function teardownSync() {
  if (syncHandlers && mainVideo) {
    for (const [evt, h] of Object.entries(syncHandlers)) {
      mainVideo.removeEventListener(evt, h);
    }
  }
  syncHandlers = null;
  sideVideo = null;
  sideCanvas = null;
  mainVideo = null;
}

// Map the primary's current frame to the secondary's frame index by VIDEO
// TIME, not by frame index. Required because the two engines can have
// (slightly) different start_sec or fps after the NTSC-drift fix —
// secondary's frame N no longer necessarily covers the same instant as
// primary's frame N. Returns null when video time falls outside the
// secondary cache's range.
function secondaryFrame(state) {
  const a = state.pose, b = state.poseSecondary;
  const t = (a.start_sec || 0) + state.frame / a.fps;
  const sf = Math.round((t - (b.start_sec || 0)) * b.fps);
  return (sf >= 0 && sf < b.n_frames) ? sf : null;
}

function drawEngineSkeleton(ctx, pose, frame, style) {
  const EDGES = [
    [5,7],[7,9],[6,8],[8,10],
    [5,6],[5,11],[6,12],[11,12],
    [11,13],[13,15],[12,14],[14,16],
    [0,1],[0,2],[1,3],[2,4],
  ];
  ctx.lineWidth = style.boneWidth;
  ctx.strokeStyle = style.boneColor;
  for (const [a, b] of EDGES) {
    const ca = pose.conf[frame * 17 + a];
    const cb = pose.conf[frame * 17 + b];
    if (ca < 0.05 || cb < 0.05) continue;
    const ax = pose.skeleton[(frame * 17 + a) * 2];
    const ay = pose.skeleton[(frame * 17 + a) * 2 + 1];
    const bx = pose.skeleton[(frame * 17 + b) * 2];
    const by = pose.skeleton[(frame * 17 + b) * 2 + 1];
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }
  for (let j = 0; j < 17; j++) {
    const c = pose.conf[frame * 17 + j];
    if (c < 0.05) continue;
    const x = pose.skeleton[(frame * 17 + j) * 2];
    const y = pose.skeleton[(frame * 17 + j) * 2 + 1];
    const isWrist = j === J.L_WRIST || j === J.R_WRIST;
    const isHi = HIGHLIGHT.has(j);
    const r = isWrist ? style.wristRadius : (isHi ? style.jointRadius * 1.5 : style.jointRadius);
    ctx.fillStyle = isWrist ? style.wristColor : style.jointColor;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    if (isWrist) {
      ctx.lineWidth = style.strokeWidth ?? 2;
      ctx.strokeStyle = "rgba(0,0,0,0.65)";
      ctx.stroke();
    }
  }
}

function drawWristTrace(canvas, a, b, jointIdx, frame) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const N = a.n_frames;
  const stride = Math.max(1, Math.floor(N / W));

  let yMin = Infinity, yMax = -Infinity;
  const sample = (p) => {
    for (let f = 0; f < N; f += stride) {
      const c = p.conf[f * 17 + jointIdx];
      if (c < 0.2) continue;
      const y = p.skeleton[(f * 17 + jointIdx) * 2 + 1];
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    }
  };
  sample(a); sample(b);
  if (!isFinite(yMin)) { yMin = 0; yMax = 1; }

  const ymap = y => H - ((y - yMin) / (yMax - yMin)) * (H - 4) - 2;
  const xmap = f => (f / (N - 1)) * (W - 2) + 1;

  const drawSeries = (p, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    let started = false;
    for (let f = 0; f < N; f += stride) {
      const c = p.conf[f * 17 + jointIdx];
      const y = p.skeleton[(f * 17 + jointIdx) * 2 + 1];
      if (c < 0.2) { started = false; continue; }
      const px = xmap(f), py = ymap(y);
      if (!started) { ctx.moveTo(px, py); started = true; }
      else            ctx.lineTo(px, py);
    }
    ctx.stroke();
  };
  drawSeries(a, Y_WRIST);
  drawSeries(b, V_WRIST);

  ctx.strokeStyle = "rgba(255,255,255,0.7)";
  ctx.lineWidth = 1;
  const x = xmap(frame);
  ctx.beginPath();
  ctx.moveTo(x, 0);
  ctx.lineTo(x, H);
  ctx.stroke();
}

function setText(id, value) {
  const el = host?.querySelector("#" + id);
  if (el) el.textContent = value;
}
