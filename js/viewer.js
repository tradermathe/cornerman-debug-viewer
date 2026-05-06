// Main viewer. Owns the <video> + <canvas> sync, frame navigation, scrubber,
// and the rule-panel host. Rule panels register themselves via rules/registry.js
// and get a chance to (a) draw extra overlay graphics each frame and (b) own
// a side-panel DOM area that they refresh.

import { loadPose } from "./pose-loader.js";
import { drawSkeleton } from "./skeleton.js";
import { RULES } from "./rules/registry.js";

const els = {
  videoFile:    document.getElementById("video-file"),
  poseFile:     document.getElementById("pose-file"),
  cacheFolder:  document.getElementById("cache-folder"),
  cacheStatus:  document.getElementById("cache-status"),
  cacheSection: document.getElementById("cache-section"),
  roundSel:     document.getElementById("round-select"),
  loadStatus:   document.getElementById("load-status"),
  pickerCard:   document.getElementById("picker-card"),
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
  thumbVideo:  document.getElementById("thumb-video"),
  thumbTip:    document.getElementById("thumb-tooltip"),
  thumbCanvas: document.getElementById("thumb-canvas"),
  thumbLabel:  document.getElementById("thumb-label"),
};

// Cache index built from the folder picker:
//   Map<videoBasename, Map<roundN, { yolo?: {npy,meta}, vision?: {npy,meta} }>>
// Survives across video picks within one page session.
let cacheIndex = null;

// Monotonically increasing token. Bumped on every load attempt so an in-flight
// load can detect that a newer pick has superseded it and bail out before
// overwriting state (avoids the "I picked a new video but the old one keeps
// loading on top" race).
let currentLoadToken = 0;

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
els.cacheFolder.addEventListener("change", onCacheFolder);
els.videoFile.addEventListener("change", onVideoPick);
els.roundSel.addEventListener("change", onRoundPick);
els.poseFile.addEventListener("change", onManualPose);

// Folder picker: index every `<base>_<engine>_r<N>.npy` and matching
// `<base>_<engine>_r<N>_meta.json`. `engine` is `yolo` or `vision`. Each
// round entry tracks both engines independently so the viewer can offer a
// YOLO-vs-Vision compare lens whenever both are present.
function onCacheFolder(e) {
  const files = Array.from(e.target.files || []);
  cacheIndex = new Map();
  for (const f of files) {
    if (f.name.endsWith(".bak.npy")) continue;
    const m = f.name.match(/^(.+?)_(yolo|vision)_r(\d+)(_meta)?\.(npy|json)$/);
    if (!m) continue;
    const [, base, engine, roundStr, isMeta, ext] = m;
    const round = parseInt(roundStr);
    if (ext === "json" && !isMeta) continue;  // some random json next door

    if (!cacheIndex.has(base)) cacheIndex.set(base, new Map());
    const rounds = cacheIndex.get(base);
    if (!rounds.has(round)) rounds.set(round, {});
    const roundSlot = rounds.get(round);
    if (!roundSlot[engine]) roundSlot[engine] = {};
    const engineSlot = roundSlot[engine];
    if (ext === "npy")  engineSlot.npy = f;
    if (ext === "json") engineSlot.meta = f;
  }

  // Drop incomplete pairs (need both .npy + .json per engine) so we never
  // offer something we can't actually load. A round is kept if at least one
  // engine has a complete pair.
  for (const [base, rounds] of cacheIndex) {
    for (const [round, slot] of rounds) {
      for (const eng of ["yolo", "vision"]) {
        if (slot[eng] && (!slot[eng].npy || !slot[eng].meta)) delete slot[eng];
      }
      if (!slot.yolo && !slot.vision) rounds.delete(round);
    }
    if (rounds.size === 0) cacheIndex.delete(base);
  }

  const nVideos = cacheIndex.size;
  let nRounds = 0, nVision = 0;
  for (const rounds of cacheIndex.values()) {
    for (const slot of rounds.values()) {
      nRounds++;
      if (slot.vision) nVision++;
    }
  }
  if (nRounds) {
    const visionNote = nVision ? `, ${nVision} with Apple Vision` : "";
    els.cacheStatus.textContent =
      `— ${nRounds} rounds across ${nVideos} videos${visionNote}`;
    if (els.cacheSection) els.cacheSection.open = false;
  } else {
    els.cacheStatus.textContent =
      "— no `_<engine>_r{N}.npy + _meta.json` pairs found in that folder";
  }

  // Re-evaluate any already-picked video against the new index.
  if (els.videoFile.files[0]) onVideoPick();
}

function onVideoPick() {
  const v = els.videoFile.files[0];
  if (!v) return;

  if (!cacheIndex) {
    // No cache index yet — still swap the video so the user sees their
    // pick reflected, just without skeleton.
    loadVideoOnly(v, "Pick a cache folder above, or open the manual file picker.");
    populateRoundSelect(null);
    return;
  }
  const base = videoBasename(v.name);
  const rounds = cacheIndex.get(base);
  if (!rounds || rounds.size === 0) {
    // Same idea — show the new video so it's obvious the pick worked,
    // and surface the matching error.
    loadVideoOnly(v,
      `⚠ No cache match for "${base}". Use the manual file pick below, ` +
      `or rename the video to match a cache basename.`);
    populateRoundSelect(null);
    return;
  }
  populateRoundSelect(rounds);
  // Always auto-load the first round. (The dropdown defaults to r0 so
  // clicking r0 doesn't fire `change`; the dropdown stays interactive so
  // you can still pick r1, r2, …)
  const first = [...rounds.keys()].sort((a, b) => a - b)[0];
  els.roundSel.value = String(first);
  loadFromIndex(v, rounds.get(first));
}

// Swap the video src without loading any pose. Clears any leftover skeleton
// from the previous round so the screen is honest about the missing data.
function loadVideoOnly(videoFile, errMessage) {
  const token = ++currentLoadToken;
  els.loadStatus.textContent = `Loading ${videoFile.name}…`;
  loadVideo(videoFile)
    .then(() => {
      if (token !== currentLoadToken) return;
      state.pose = null;
      state.poseSecondary = null;
      state.n_frames = 0;
      state.frame = 0;
      els.scrubber.max = 0;
      els.scrubber.value = 0;
      fitCanvasToVideo();
      // Clear the canvas explicitly — redraw() early-returns when no pose,
      // which would leave the prior skeleton ghosting on screen.
      const ctx = els.canvas.getContext("2d");
      ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
      els.viewer.hidden = false;
      els.frameLabel.textContent = "no pose data loaded";
      els.meta.textContent = `${videoFile.name} · ${els.video.videoWidth}×${els.video.videoHeight}`;
      els.loadStatus.textContent = errMessage;
    })
    .catch(err => {
      if (token !== currentLoadToken) return;
      els.loadStatus.textContent = `Error: ${err.message}`;
    });
}

function onRoundPick() {
  const v = els.videoFile.files[0];
  const r = parseInt(els.roundSel.value);
  if (!v || isNaN(r)) return;
  const base = videoBasename(v.name);
  const pair = cacheIndex?.get(base)?.get(r);
  if (pair) loadFromIndex(v, pair);
}

function onManualPose() {
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
  loadFromFiles(v, ps);
}

function loadFromIndex(videoFile, slot) {
  // `slot` may have `yolo` and/or `vision`. Primary pose is YOLO when
  // present (matches the rules engine), Vision is loaded as a sibling
  // for the compare lens. If only one engine is present, that one is
  // primary and there's no comparison.
  const primary = slot.yolo || slot.vision;
  const secondary = (slot.yolo && slot.vision) ? slot.vision : null;
  const status =
    `Loading ${videoFile.name}${secondary ? " (yolo + vision)" : ""}…`;
  els.loadStatus.textContent = status;

  const token = ++currentLoadToken;
  loadVideo(videoFile)
    .then(async () => {
      if (token !== currentLoadToken) return;
      const size = { width: els.video.videoWidth, height: els.video.videoHeight };
      const posePrimary = await loadPose([primary.npy, primary.meta], size);
      if (token !== currentLoadToken) return;
      let poseSecondary = null;
      if (secondary) {
        poseSecondary = await loadPose([secondary.npy, secondary.meta], size);
        if (token !== currentLoadToken) return;
        // Stamp engine name so the lens can label them correctly even if the
        // file source said "yolo_pose" for both (loader currently hard-codes
        // engine name).
        poseSecondary.engine = "apple_vision_2d";
      }
      posePrimary.engine = slot.yolo ? "yolo_pose" : "apple_vision_2d";
      start(posePrimary, poseSecondary);
    })
    .catch(err => {
      if (token !== currentLoadToken) return;
      console.error(err);
      els.loadStatus.textContent = `Error: ${err.message}`;
    });
}

function loadFromFiles(videoFile, poseFiles) {
  // Manual file picker — single engine only.
  const names = Array.from(poseFiles).map(f => f.name).join(" + ");
  els.loadStatus.textContent = `Loading ${videoFile.name} + ${names}…`;
  const token = ++currentLoadToken;
  loadVideo(videoFile)
    .then(() => {
      if (token !== currentLoadToken) return null;
      return loadPose(poseFiles, {
        width: els.video.videoWidth,
        height: els.video.videoHeight,
      });
    })
    .then(p => {
      if (p == null || token !== currentLoadToken) return;
      start(p, null);
    })
    .catch(err => {
      if (token !== currentLoadToken) return;
      console.error(err);
      els.loadStatus.textContent = `Error: ${err.message}`;
    });
}

function populateRoundSelect(rounds) {
  els.roundSel.innerHTML = "";
  if (!rounds || rounds.size === 0) {
    els.roundSel.innerHTML = `<option value="">—</option>`;
    els.roundSel.disabled = true;
    return;
  }
  const sorted = [...rounds.keys()].sort((a, b) => a - b);
  for (const r of sorted) {
    const o = document.createElement("option");
    o.value = String(r);
    o.textContent = `r${r}`;
    els.roundSel.appendChild(o);
  }
  els.roundSel.disabled = sorted.length < 2;
}

// Strip extension; the cache files were named after the source video so
// `<videoBasename>` should be the prefix of `<videoBasename>_yolo_r0.npy`.
function videoBasename(name) {
  return name.replace(/\.[^.]+$/, "");
}

function loadVideo(file) {
  return new Promise((resolve, reject) => {
    if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
    state.videoUrl = URL.createObjectURL(file);
    state.videoFileName = file.name;
    els.video.src = state.videoUrl;
    // The thumb-video shares the source so we can seek-and-snapshot it on
    // scrubber hover without disturbing the main playback.
    els.thumbVideo.src = state.videoUrl;
    els.video.onloadedmetadata = () => resolve();
    els.video.onerror = () => reject(new Error("Video failed to load"));
  });
}

function start(pose, poseSecondary = null) {
  state.pose = pose;
  state.poseSecondary = poseSecondary;   // optional second engine for compare
  state.fps = pose.fps;
  state.n_frames = pose.n_frames;
  state.start_sec = pose.start_sec || 0;
  state.frame = 0;

  els.pickerCard.classList.add("loaded");
  els.viewer.hidden = false;
  els.scrubber.max = pose.n_frames - 1;
  els.scrubber.value = 0;

  fitCanvasToVideo();

  populateRuleSelect();
  setRule(els.ruleSel.value);

  // Cache covers [start_sec, start_sec + n_frames/fps]. The official
  // round begins at round_start_sec which can be later than start_sec
  // when a pre-buffer is included — that's why frame 0 of the cache
  // shows footage BEFORE the round, not the first punch.
  const startSec = state.start_sec;
  const endSec = startSec + pose.n_frames / pose.fps;
  const clipRange = startSec || endSec !== els.video.duration
    ? ` · clip ${startSec.toFixed(1)}–${endSec.toFixed(1)}s of ${els.video.duration.toFixed(1)}s video`
    : "";
  const preBuffer = pose.pre_buffer_sec > 0.01
    ? ` · ${pose.pre_buffer_sec.toFixed(1)}s pre-buffer (round starts ${pose.round_start_sec.toFixed(1)}s)`
    : "";
  els.meta.textContent =
    `${pose.engine} · ${pose.width}×${pose.height} · ` +
    `${pose.fps.toFixed(1)} fps · ${pose.n_frames} frames${clipRange}${preBuffer}`;
  els.loadStatus.textContent = "";

  // Cache the pre-buffer frame count so the frame-label can mark when the
  // round officially starts.
  state.pre_buffer_frames = Math.round(pose.pre_buffer_sec * pose.fps);

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
  // Frame N of the cache lives at video time start_sec + (N + 0.5)/fps —
  // the +0.5 lands in the middle of the frame's time slot so we don't end
  // up on a boundary and get the previous video frame back.
  els.video.currentTime = state.start_sec + (f + 0.5) / state.fps;
  // canvas redraw happens on the seeked event so video+overlay stay in sync.
}

els.video.addEventListener("seeked", redraw);

// During playback `timeupdate` only fires ~4–66 Hz and isn't aligned with
// rendered frames, so the skeleton drifts visibly. requestVideoFrameCallback
// fires once per displayed frame with the exact mediaTime — frame-accurate.
// Falls back to requestAnimationFrame polling on browsers without rVFC.
const hasRvfc = "requestVideoFrameCallback" in HTMLVideoElement.prototype;
let playbackHandle = null;

function syncFromVideoTime(t_video) {
  const f = Math.floor((t_video - state.start_sec) * state.fps);
  if (f !== state.frame) {
    state.frame = Math.max(0, Math.min(f, state.n_frames - 1));
    els.scrubber.value = state.frame;
    redraw();
  }
}

function rvfcTick(_now, metadata) {
  syncFromVideoTime(metadata.mediaTime);
  if (!els.video.paused) {
    playbackHandle = els.video.requestVideoFrameCallback(rvfcTick);
  }
}

function rafTick() {
  if (els.video.paused) { playbackHandle = null; return; }
  syncFromVideoTime(els.video.currentTime);
  playbackHandle = requestAnimationFrame(rafTick);
}

els.video.addEventListener("play", () => {
  if (hasRvfc) {
    playbackHandle = els.video.requestVideoFrameCallback(rvfcTick);
  } else {
    playbackHandle = requestAnimationFrame(rafTick);
  }
});
els.video.addEventListener("pause", () => {
  if (playbackHandle == null) return;
  if (hasRvfc && els.video.cancelVideoFrameCallback) {
    els.video.cancelVideoFrameCallback(playbackHandle);
  } else {
    cancelAnimationFrame(playbackHandle);
  }
  playbackHandle = null;
});

function togglePlay() {
  if (els.video.paused) { els.video.play(); els.playPause.textContent = "⏸"; }
  else                  { els.video.pause(); els.playPause.textContent = "▶"; }
}

els.video.addEventListener("play",  () => els.playPause.textContent = "⏸");
els.video.addEventListener("pause", () => els.playPause.textContent = "▶");

// ── Scrubber hover thumbnail ────────────────────────────────────────────────
// Hovering the scrubber should preview the frame at that timeline position
// without disturbing main playback. We seek a hidden duplicate of the video
// (els.thumbVideo) and draw its current frame + the skeleton at that frame
// into a small canvas tooltip near the cursor.
//
// mousemove fires far faster than the video can seek, so we keep only the
// latest target frame. When a seek finishes, if the target moved we kick off
// another seek.
let thumbTarget = null;        // frame the user is currently hovering
let thumbDrawn = -1;           // frame currently visible in the canvas
let thumbSeekInFlight = false;

els.scrubber.addEventListener("mousemove", e => {
  if (!state.pose) return;
  const rect = els.scrubber.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  const f = Math.floor(ratio * (state.n_frames - 1));
  thumbTarget = f;
  positionThumb(e.clientX, rect.top);
  els.thumbTip.hidden = false;
  // If the requested frame is already drawn, skip the seek.
  if (f === thumbDrawn) return;
  if (!thumbSeekInFlight) seekThumb();
});

els.scrubber.addEventListener("mouseleave", () => {
  els.thumbTip.hidden = true;
  thumbTarget = null;
});

function seekThumb() {
  if (thumbTarget == null || !els.thumbVideo.src) return;
  const target = thumbTarget;
  thumbSeekInFlight = true;
  els.thumbVideo.currentTime = state.start_sec + (target + 0.5) / state.fps;
  els.thumbVideo.onseeked = () => {
    drawThumb(target);
    thumbDrawn = target;
    thumbSeekInFlight = false;
    // If the user moved while we were seeking, chase the new target.
    if (thumbTarget != null && thumbTarget !== target) seekThumb();
  };
}

function drawThumb(frame) {
  const ctx = els.thumbCanvas.getContext("2d");
  const W = els.thumbCanvas.width, H = els.thumbCanvas.height;
  const vw = els.thumbVideo.videoWidth || state.pose.width;
  const vh = els.thumbVideo.videoHeight || state.pose.height;

  // Letterbox the frame inside the canvas.
  const scale = Math.min(W / vw, H / vh);
  const dw = vw * scale, dh = vh * scale;
  const dx = (W - dw) / 2, dy = (H - dh) / 2;

  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, W, H);
  ctx.drawImage(els.thumbVideo, dx, dy, dw, dh);

  // Skeleton at the hovered frame, scaled to match the letterboxed video.
  ctx.save();
  ctx.translate(dx, dy);
  ctx.scale(scale, scale);
  drawSkeleton(ctx, state.pose, frame, state.rule?.skeletonStyle?.(state) || {});
  ctx.restore();

  els.thumbLabel.textContent =
    `f${frame}  ·  t=${(state.start_sec + frame / state.fps).toFixed(2)}s`;
}

function positionThumb(cursorX, scrubberTop) {
  const tip = els.thumbTip;
  const tw = tip.offsetWidth || 250;
  const th = tip.offsetHeight || 160;
  const left = Math.min(window.innerWidth - tw - 8, Math.max(8, cursorX - tw / 2));
  const top = Math.max(8, scrubberTop - th - 10);
  tip.style.left = left + "px";
  tip.style.top = top + "px";
}

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

  // Show frame within the cache, video time (= start_sec + frame/fps),
  // and a "before round" marker when we're inside the pre-buffer.
  const t_video = state.start_sec + state.frame / state.fps;
  const before = state.pre_buffer_frames && state.frame < state.pre_buffer_frames
    ? "  ·  ⏪ pre-buffer (before round)" : "";
  els.frameLabel.textContent =
    `frame ${state.frame} / ${state.n_frames - 1}   ·   ` +
    `t=${t_video.toFixed(2)}s${before}`;
}
