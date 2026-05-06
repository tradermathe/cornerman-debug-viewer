// Main viewer. Owns the <video> + <canvas> sync, frame navigation, scrubber,
// and the rule-panel host. Rule panels register themselves via rules/registry.js
// and get a chance to (a) draw extra overlay graphics each frame and (b) own
// a side-panel DOM area that they refresh.

import { loadPose } from "./pose-loader.js";
import { drawSkeleton } from "./skeleton.js";
import { RULES } from "./rules/registry.js";

const els = {
  videoFile:   document.getElementById("video-file"),
  poseFile:    document.getElementById("pose-file"),
  loadStatus:  document.getElementById("load-status"),
  pickerCard:  document.getElementById("picker-card"),
  viewer:      document.getElementById("viewer"),
  video:       document.getElementById("video"),
  canvas:      document.getElementById("overlay"),
  stage:       document.getElementById("stage"),
  prevFrame:   document.getElementById("prev-frame"),
  nextFrame:   document.getElementById("next-frame"),
  playPause:   document.getElementById("play-pause"),
  speedSel:    document.getElementById("speed"),
  frameLabel:  document.getElementById("frame-label"),
  scrubber:    document.getElementById("scrubber"),
  ruleSel:     document.getElementById("rule-select"),
  ruleHost:    document.getElementById("rule-panel"),
  meta:        document.getElementById("meta"),
};

const state = {
  pose: null,
  videoUrl: null,
  videoFileName: null,
  fps: 30,
  n_frames: 0,
  frame: 0,
  rule: null,    // active rule module
  raf: null,
};

// ── File loading ────────────────────────────────────────────────────────────
els.videoFile.addEventListener("change", onPick);
els.poseFile.addEventListener("change", onPick);

function onPick() {
  const v = els.videoFile.files[0];
  const ps = els.poseFile.files;
  const haveNpy  = Array.from(ps).some(f => f.name.endsWith(".npy"));
  const haveJson = Array.from(ps).some(f => f.name.endsWith(".json"));
  if (!v || !haveNpy || !haveJson) {
    const missing = [];
    if (!v) missing.push("video");
    if (!haveNpy) missing.push(".npy");
    if (!haveJson) missing.push("_meta.json");
    els.loadStatus.textContent = missing.length === 3
      ? "" : `Still need: ${missing.join(", ")}`;
    return;
  }
  els.loadStatus.textContent = `Loading ${v.name} + ${ps.length} pose file(s)…`;
  // Pose loader needs video dimensions to de-normalise coords, so load video
  // first, then read videoWidth/videoHeight, then load the cache.
  loadVideo(v)
    .then(() => loadPose(ps, {
      width: els.video.videoWidth,
      height: els.video.videoHeight,
    }))
    .then(start)
    .catch(err => {
      console.error(err);
      els.loadStatus.textContent = `Error: ${err.message}`;
    });
}

function loadVideo(file) {
  return new Promise((resolve, reject) => {
    if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
    state.videoUrl = URL.createObjectURL(file);
    state.videoFileName = file.name;
    els.video.src = state.videoUrl;
    els.video.onloadedmetadata = () => resolve();
    els.video.onerror = () => reject(new Error("Video failed to load"));
  });
}

function start(pose) {
  state.pose = pose;
  state.fps = pose.fps;
  state.n_frames = pose.n_frames;
  state.frame = 0;

  els.pickerCard.classList.add("collapsed");
  els.viewer.hidden = false;
  els.scrubber.max = pose.n_frames - 1;
  els.scrubber.value = 0;

  fitCanvasToVideo();

  populateRuleSelect();
  setRule(els.ruleSel.value);

  // Mismatch warning — common when the user picks a clipped cache against
  // the full source video, or vice versa.
  const vidFrames = Math.round(els.video.duration * pose.fps);
  let metaText = `${pose.engine} · ${pose.width}×${pose.height} · ` +
                 `${pose.fps.toFixed(1)} fps · ${pose.n_frames} frames`;
  if (Math.abs(vidFrames - pose.n_frames) > 5) {
    metaText += ` · ⚠ video has ~${vidFrames} frames (mismatch)`;
  }
  els.meta.textContent = metaText;
  els.loadStatus.textContent = "";

  seekToFrame(0);
}

function fitCanvasToVideo() {
  // Canvas internal resolution = video native resolution, so we can draw using
  // raw skeleton pixel coords. CSS sizes both elements together.
  els.canvas.width = els.video.videoWidth || state.pose.width;
  els.canvas.height = els.video.videoHeight || state.pose.height;
}

window.addEventListener("resize", () => {
  // CSS handles visual sizing; nothing to redo internally. Repaint anyway.
  redraw();
});

// ── Rule panels ─────────────────────────────────────────────────────────────
function populateRuleSelect() {
  els.ruleSel.innerHTML = "";
  for (const r of RULES) {
    const o = document.createElement("option");
    o.value = r.id;
    o.textContent = r.label;
    els.ruleSel.appendChild(o);
  }
}

els.ruleSel.addEventListener("change", () => setRule(els.ruleSel.value));

function setRule(id) {
  const rule = RULES.find(r => r.id === id);
  if (!rule) return;
  state.rule = rule;
  els.ruleHost.innerHTML = "";
  rule.mount(els.ruleHost, state);
  redraw();
}

// ── Frame navigation ────────────────────────────────────────────────────────
els.prevFrame.addEventListener("click", () => seekToFrame(state.frame - 1));
els.nextFrame.addEventListener("click", () => seekToFrame(state.frame + 1));
els.playPause.addEventListener("click", togglePlay);
els.scrubber.addEventListener("input", e => seekToFrame(parseInt(e.target.value)));
els.speedSel.addEventListener("change", () => {
  els.video.playbackRate = parseFloat(els.speedSel.value);
});

document.addEventListener("keydown", e => {
  if (els.viewer.hidden) return;
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  switch (e.key) {
    case "ArrowLeft":  seekToFrame(state.frame - 1); e.preventDefault(); break;
    case "ArrowRight": seekToFrame(state.frame + 1); e.preventDefault(); break;
    case "[":          seekToFrame(state.frame - 10); break;
    case "]":          seekToFrame(state.frame + 10); break;
    case " ":          togglePlay(); e.preventDefault(); break;
  }
});

function seekToFrame(f) {
  if (!state.pose) return;
  f = Math.max(0, Math.min(state.n_frames - 1, Math.round(f)));
  state.frame = f;
  els.scrubber.value = f;
  // Seek to the *middle* of the frame's time slot to avoid landing on the
  // boundary and getting the previous frame back.
  els.video.currentTime = (f + 0.5) / state.fps;
  // canvas redraw happens on the seeked event so video+overlay stay in sync.
}

els.video.addEventListener("seeked", redraw);
els.video.addEventListener("timeupdate", () => {
  if (els.video.paused) return;
  const f = Math.floor(els.video.currentTime * state.fps);
  if (f !== state.frame) {
    state.frame = Math.min(f, state.n_frames - 1);
    els.scrubber.value = state.frame;
    redraw();
  }
});

function togglePlay() {
  if (els.video.paused) { els.video.play(); els.playPause.textContent = "⏸"; }
  else                  { els.video.pause(); els.playPause.textContent = "▶"; }
}

els.video.addEventListener("play",  () => els.playPause.textContent = "⏸");
els.video.addEventListener("pause", () => els.playPause.textContent = "▶");

function redraw() {
  if (!state.pose) return;
  const ctx = els.canvas.getContext("2d");
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);

  // Let the active rule influence the base skeleton style, then draw it.
  const baseStyle = state.rule?.skeletonStyle?.(state) || {};
  drawSkeleton(ctx, state.pose, state.frame, baseStyle);

  // Then the rule paints its own decorations on top.
  state.rule?.draw?.(ctx, state);

  // And refreshes its side panel.
  state.rule?.update?.(state);

  els.frameLabel.textContent =
    `frame ${state.frame} / ${state.n_frames - 1}   ·   ` +
    `t=${(state.frame / state.fps).toFixed(2)}s`;
}
