// Engine compare lens — overlays YOLO and Apple Vision skeletons on the
// same frame so we can eyeball where they disagree, especially around the
// wrists when gloves are on.
//
// Requires the cache folder to contain BOTH `<base>_yolo_r{N}.*` and
// `<base>_vision_r{N}.*` for the loaded round. If only one engine is
// present, the lens shows a "no comparison data" message and falls back
// to drawing just the primary skeleton.
//
// Visual encoding:
//   YOLO   — orange/yellow joints + warm bones
//   Vision — cyan/green joints + cool bones
//   wrists drawn larger so the divergence is easy to read

import { J } from "../skeleton.js";

const Y_BONE = "rgba(255,170,80,0.55)";   // YOLO bones
const V_BONE = "rgba(110,200,255,0.55)";  // Vision bones
const Y_WRIST = "#ff8a3c";
const V_WRIST = "#5fd1ff";
const HIGHLIGHT = new Set([J.NOSE, J.L_WRIST, J.R_WRIST, J.L_ELBOW, J.R_ELBOW]);

let host;

export const EngineCompareRule = {
  id: "engine_compare",
  label: "Engine compare (YOLO vs Vision)",

  // Suppress the base skeleton renderer — we draw both ourselves.
  skeletonStyle() {
    return { boneColor: "rgba(0,0,0,0)", jointRadius: 0, minConf: Infinity };
  },

  mount(_host, state) {
    host = _host;
    const hasBoth = !!state.poseSecondary;
    host.innerHTML = `
      <h2>Engine compare (YOLO vs Vision)</h2>
      ${hasBoth ? "" : `
        <p class="hint" style="color:var(--bad)">
          No second engine for this round. Make sure the cache folder
          contains both <code>&lt;base&gt;_yolo_r{N}.*</code> and
          <code>&lt;base&gt;_vision_r{N}.*</code> files.
        </p>
      `}
      <p class="hint">
        <span style="color:${Y_WRIST}">orange/yellow</span> = YOLO,
        <span style="color:${V_WRIST}">cyan/green</span> = Apple Vision.
        Wrists drawn extra-large so disagreement reads at a glance.
      </p>

      <h3>Per-frame Δ (raw px)</h3>
      <p class="hint">Distance between YOLO and Vision detection of each joint at this frame, and each engine's confidence (Y / V).</p>
      <div class="metric-grid">
        <div class="metric"><div class="metric-label">L wrist</div><div class="metric-val" id="ec-l-wrist">—</div><div class="metric-sub" id="ec-l-wrist-conf">—</div></div>
        <div class="metric"><div class="metric-label">R wrist</div><div class="metric-val" id="ec-r-wrist">—</div><div class="metric-sub" id="ec-r-wrist-conf">—</div></div>
        <div class="metric"><div class="metric-label">Nose</div><div class="metric-val" id="ec-nose">—</div><div class="metric-sub" id="ec-nose-conf">—</div></div>
        <div class="metric"><div class="metric-label">L elbow</div><div class="metric-val" id="ec-l-elbow">—</div><div class="metric-sub" id="ec-l-elbow-conf">—</div></div>
        <div class="metric"><div class="metric-label">R elbow</div><div class="metric-val" id="ec-r-elbow">—</div><div class="metric-sub" id="ec-r-elbow-conf">—</div></div>
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
  },

  draw(ctx, state) {
    const s = state.renderScale || 1;
    // Color-code by actual engine name rather than primary/secondary
    // order — primary used to always be YOLO, but the viewer now prefers
    // Vision when both exist, so we'd otherwise paint Vision in YOLO's
    // colors. YOLO always = orange/yellow; Vision always = cyan/green,
    // regardless of which one happens to be primary this load.
    const stylesFor = pose => (pose.engine === "yolo_pose")
      ? { boneColor: Y_BONE, jointColor: Y_WRIST, wristColor: Y_WRIST }
      : { boneColor: V_BONE, jointColor: V_WRIST, wristColor: V_WRIST };

    drawEngineSkeleton(ctx, state.pose, state.frame, {
      ...stylesFor(state.pose),
      boneWidth: 2 * s, jointRadius: 4 * s, wristRadius: 9 * s, strokeWidth: 2 * s,
    });
    if (state.poseSecondary) {
      const sf = secondaryFrame(state);
      if (sf != null) {
        drawEngineSkeleton(ctx, state.poseSecondary, sf, {
          ...stylesFor(state.poseSecondary),
          boneWidth: 2 * s, jointRadius: 4 * s, wristRadius: 9 * s, strokeWidth: 2 * s,
        });
      }
    }
  },

  update(state) {
    const f = state.frame;
    const a = state.pose;
    const b = state.poseSecondary;
    const sf = b ? secondaryFrame(state) : null;
    // Engine-aware labels. Primary used to always be YOLO; now Primary is
    // Vision when both engines are present, so we resolve the conf
    // letters/colors from each pose's engine tag.
    const tagFor = pose => pose.engine === "yolo_pose"
      ? { letter: "Y", color: Y_WRIST }
      : { letter: "V", color: V_WRIST };
    const setJointDiff = (id, j) => {
      const confId = `${id}-conf`;
      if (!b) { setText(id, "—"); setText(confId, "—"); return; }
      if (sf == null) { setText(id, "out of range"); setText(confId, "—"); return; }
      const ax = a.skeleton[(f * 17 + j) * 2];
      const ay = a.skeleton[(f * 17 + j) * 2 + 1];
      const ac = a.conf[f * 17 + j];
      const bx = b.skeleton[(sf * 17 + j) * 2];
      const by = b.skeleton[(sf * 17 + j) * 2 + 1];
      const bc = b.conf[sf * 17 + j];
      const aTag = tagFor(a);
      const bTag = tagFor(b);
      setHTML(confId,
        `<span style="color:${aTag.color}">${aTag.letter} ${ac.toFixed(2)}</span> · ` +
        `<span style="color:${bTag.color}">${bTag.letter} ${bc.toFixed(2)}</span>`);
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

// Draw a single engine's skeleton with custom colours. Largely a stripped-down
// fork of skeleton.js drawSkeleton so we can colour everything one engine at a
// time without fighting the highlight-set logic.
// 0.3 gate: low enough that genuinely visible joints with reduced conf still
// render (Vision's mean conf is ~0.6, well above this), high enough that
// off-screen "guesses" Apple Vision keeps emitting at conf 0.1-0.2 don't
// produce ghost bones into thin air. Tuned against the close-up cuts in
// "3 BODY SHOT COMBOS" where the knees aren't in frame.
const DRAW_CONF_GATE = 0.3;

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
    if (ca < DRAW_CONF_GATE || cb < DRAW_CONF_GATE) continue;
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
    if (c < DRAW_CONF_GATE) continue;
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

  // Autoscale across both engines.
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
  // Engine-aware colors — primary may be Vision now, so pick by engine tag.
  const colorFor = pose => pose.engine === "yolo_pose" ? Y_WRIST : V_WRIST;
  drawSeries(a, colorFor(a));
  drawSeries(b, colorFor(b));

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

function setHTML(id, html) {
  const el = host?.querySelector("#" + id);
  if (el) el.innerHTML = html;
}
