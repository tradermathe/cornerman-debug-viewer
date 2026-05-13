// Main viewer. Owns the <video> + <canvas> sync, frame navigation, scrubber,
// and the rule-panel host. Rule panels register themselves via rules/registry.js
// and get a chance to (a) draw extra overlay graphics each frame and (b) own
// a side-panel DOM area that they refresh.

// Bump this on every push so the user can tell whether the new code is
// actually live or whether GitHub Pages / their browser is still serving
// a cached copy. Format: YYYY-MM-DD.N where N restarts at 1 each day.
const BUILD = "2026-05-13.16";
{
  const el = document.getElementById("build-tag");
  if (el) el.textContent = `build ${BUILD}`;
}

import { loadPose } from "./pose-loader.js";
import { loadPunches } from "./punches-loader.js";
import { fetchLiveLabels } from "./sheet-labels.js";
import { drawSkeleton } from "./skeleton.js";
import { RULES } from "./rules/registry.js";
import * as drive from "./drive-folder.js";

const els = {
  videoFile:    document.getElementById("video-file"),
  videoPick:    document.getElementById("video-pick"),
  poseFile:     document.getElementById("pose-file"),
  cacheFolder:  document.getElementById("cache-folder"),
  cacheClear:   document.getElementById("cache-clear"),
  cacheStatus:  document.getElementById("cache-status"),
  cacheSection: document.getElementById("cache-section"),
  driveConnect: document.getElementById("drive-connect"),
  driveDisconnect: document.getElementById("drive-disconnect"),
  driveStatus:  document.getElementById("drive-status"),
  driveSection: document.getElementById("drive-section"),
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

// Cache index built from the folder picker (or the Drive folder walker):
//   Map<videoBasename, Map<roundN, { yolo?: {npy,meta,punches?}, vision?: ... }>>
// Slot values are EITHER File objects (manual picker) OR
// FileSystemFileHandle objects (Drive folder). loadFromIndex calls
// drive.toFile() on values when it's actually time to load.
// Survives across video picks within one page session.
let cacheIndex = null;

// Drive-folder state: a separate index of video filename -> FileSystemFileHandle
// built by the Drive folder walker. Populated when a Drive folder is
// connected; consulted to populate the video dropdown.
let driveVideos = null;
let driveHandle = null;

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
els.cacheClear.addEventListener("click", onCacheClear);
els.videoFile.addEventListener("change", onVideoPick);
els.roundSel.addEventListener("change", onRoundPick);
els.poseFile.addEventListener("change", onManualPose);
if (els.driveConnect)    els.driveConnect.addEventListener("click", onDriveConnect);
if (els.driveDisconnect) els.driveDisconnect.addEventListener("click", onDriveDisconnect);
if (els.videoPick)       els.videoPick.addEventListener("change", onDriveVideoPick);

// On boot, hide the Drive section entirely if the API isn't there (Safari /
// Firefox today). Otherwise try to silently restore the last folder handle.
initDriveSection();
async function initDriveSection() {
  if (!els.driveSection) return;
  if (!drive.isSupported()) {
    els.driveSection.hidden = true;
    return;
  }
  const restored = await drive.tryRestore();
  if (!restored) {
    setDriveStatus("idle");
    return;
  }
  driveHandle = restored.handle;
  if (restored.permission === "granted") {
    await refreshDriveFolder();
  } else {
    setDriveStatus("needs-permission", restored.handle.name);
  }
}

function setDriveStatus(state, name) {
  if (!els.driveStatus) return;
  switch (state) {
    case "idle":
      els.driveStatus.textContent = "— not connected";
      if (els.driveConnect)    els.driveConnect.textContent = "Connect Drive folder";
      if (els.driveDisconnect) els.driveDisconnect.hidden = true;
      break;
    case "needs-permission":
      els.driveStatus.innerHTML = `— need permission for <code>${name || "folder"}</code>`;
      if (els.driveConnect)    els.driveConnect.textContent = "Reconnect";
      if (els.driveDisconnect) els.driveDisconnect.hidden = false;
      break;
    case "scanning":
      els.driveStatus.innerHTML = `— scanning <code>${name || ""}</code>…`;
      if (els.driveDisconnect) els.driveDisconnect.hidden = false;
      break;
    case "connected": {
      const nRounds = countRounds(cacheIndex);
      const nVideos = driveVideos?.size || 0;
      els.driveStatus.innerHTML =
        `— connected to <code>${name || "folder"}</code> · ${nVideos} videos · ${nRounds} round caches`;
      if (els.driveConnect)    els.driveConnect.textContent = "Pick a different folder";
      if (els.driveDisconnect) els.driveDisconnect.hidden = false;
      break;
    }
    case "denied":
      els.driveStatus.innerHTML = `— permission denied for <code>${name || "folder"}</code>`;
      if (els.driveConnect)    els.driveConnect.textContent = "Reconnect";
      if (els.driveDisconnect) els.driveDisconnect.hidden = false;
      break;
  }
}

function countRounds(idx) {
  let n = 0;
  for (const rounds of idx?.values() || []) n += rounds.size;
  return n;
}

async function onDriveConnect() {
  // Two distinct flows: (1) we already have a handle but need permission, and
  // (2) the user wants to pick a (different) folder. Either way ends with a
  // valid handle + read permission, then a re-scan.
  try {
    if (driveHandle) {
      const perm = await drive.requestPermission(driveHandle);
      if (perm === "granted") {
        await refreshDriveFolder();
        return;
      }
      // Fall through to pick a new folder if denied.
    }
    const handle = await drive.pickFolder();
    driveHandle = handle;
    await refreshDriveFolder();
  } catch (err) {
    if (err?.name === "AbortError") return;     // user cancelled picker
    console.error("Drive folder connect failed:", err);
    els.loadStatus.textContent = `Drive folder: ${err.message}`;
  }
}

async function onDriveDisconnect() {
  driveHandle = null;
  driveVideos = null;
  // Clear only Drive-sourced entries — the manual cache picker may have added
  // its own entries we don't want to drop. For now we don't distinguish, so
  // clearing the whole index is the safe behaviour; pick again to rebuild.
  cacheIndex = null;
  await drive.forget();
  populateDriveVideoSelect();
  populateRoundSelect(null);
  refreshCacheStatus();
  setDriveStatus("idle");
}

async function refreshDriveFolder() {
  if (!driveHandle) return;
  setDriveStatus("scanning", driveHandle.name);
  try {
    const { videos, cacheIndex: idx } = await drive.walk(driveHandle);
    driveVideos = videos;
    // Merge Drive-sourced entries on top of anything from the manual picker —
    // Drive wins (more likely fresh).
    if (!cacheIndex) cacheIndex = new Map();
    for (const [base, rounds] of idx) {
      if (!cacheIndex.has(base)) cacheIndex.set(base, new Map());
      const merged = cacheIndex.get(base);
      for (const [round, slot] of rounds) merged.set(round, slot);
    }
    populateDriveVideoSelect();
    refreshCacheStatus();
    setDriveStatus("connected", driveHandle.name);
  } catch (err) {
    console.error("Drive folder walk failed:", err);
    setDriveStatus("denied", driveHandle.name);
    els.loadStatus.textContent = `Drive folder walk failed: ${err.message}`;
  }
}

function populateDriveVideoSelect() {
  if (!els.videoPick) return;
  const sel = els.videoPick;
  sel.innerHTML = "";
  if (!driveVideos || driveVideos.size === 0) {
    sel.innerHTML = `<option value="">— connect a Drive folder to populate —</option>`;
    sel.disabled = true;
    return;
  }
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = `— pick a video from ${driveHandle?.name || "Drive folder"} —`;
  sel.appendChild(placeholder);
  // Sort: videos that have at least one matching cache float to the top.
  const items = [...driveVideos.entries()].map(([name, h]) => {
    const base = videoBasename(name);
    const matched = !!cacheIndex?.get(base)?.size;
    return { name, h, matched };
  });
  items.sort((a, b) => {
    if (a.matched !== b.matched) return a.matched ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
  for (const it of items) {
    const o = document.createElement("option");
    o.value = it.name;
    o.textContent = it.matched ? it.name : `${it.name} (no cache)`;
    sel.appendChild(o);
  }
  sel.disabled = false;
}

async function onDriveVideoPick() {
  const name = els.videoPick.value;
  if (!name) return;
  const handle = driveVideos?.get(name);
  if (!handle) return;
  // Materialize the video file from its handle and feed the same code path
  // the manual <input type=file> uses.
  let file;
  try { file = await handle.getFile(); }
  catch (err) {
    els.loadStatus.textContent = `Couldn't open ${name}: ${err.message}`;
    return;
  }
  // Stuff the same File-like into the videoFile input via a DataTransfer so
  // the existing onVideoPick path runs unchanged.
  try {
    const dt = new DataTransfer();
    dt.items.add(file);
    els.videoFile.files = dt.files;
  } catch {
    /* DataTransfer assignment isn't strictly required — onVideoPick reads
       from els.videoFile.files, and if assignment fails we still have the
       file in scope. Fall through to direct pose load below. */
  }
  onVideoPick();
}

// Folder picker: index every `<base>_<engine>_r<N>.npy` and matching
// `<base>_<engine>_r<N>_meta.json`. `engine` is `yolo` or `vision`. Each
// round entry tracks both engines independently so the viewer can offer a
// YOLO-vs-Vision compare lens whenever both are present.
//
// Picks MERGE into the existing index (so the cross-platform workflow is
// "pick yolo_pose_cache, then pick apple_vision_pose_cache"); use the
// Clear button to start over. Re-picking a folder that already contributed
// files just upserts those files — same files in same slot, no growth.
function onCacheFolder(e) {
  const files = Array.from(e.target.files || []);
  if (!cacheIndex) cacheIndex = new Map();
  for (const f of files) {
    if (f.name.endsWith(".bak.npy")) continue;
    // Three sibling patterns we recognize per round + engine:
    //   <base>_<engine>_r<N>.npy           — pose data
    //   <base>_<engine>_r<N>_meta.json     — pose metadata
    //   <base>_<engine>_r<N>_punches.json  — ST-GCN detections (optional;
    //                                        produced by dump_punches.py)
    // GT labels are pulled live from the Sheet at load time — no sidecar.
    const m = f.name.match(
      /^(.+?)_(yolo|vision)_r(\d+)(_meta|_punches)?\.(npy|json)$/
    );
    if (!m) continue;
    const [, base, engine, roundStr, suffix, ext] = m;
    const round = parseInt(roundStr);
    if (ext === "json" && !suffix) continue;

    if (!cacheIndex.has(base)) cacheIndex.set(base, new Map());
    const rounds = cacheIndex.get(base);
    if (!rounds.has(round)) rounds.set(round, {});
    const roundSlot = rounds.get(round);
    if (!roundSlot[engine]) roundSlot[engine] = {};
    const engineSlot = roundSlot[engine];
    if (ext === "npy")                     engineSlot.npy = f;
    else if (suffix === "_meta")           engineSlot.meta = f;
    else if (suffix === "_punches")        engineSlot.punches = f;
  }

  // Drop incomplete pose pairs (need .npy + _meta.json per engine). Punches
  // are optional — their absence is fine, we just fall back to the heuristic.
  for (const [base, rounds] of cacheIndex) {
    for (const [round, slot] of rounds) {
      for (const eng of ["yolo", "vision"]) {
        if (slot[eng] && (!slot[eng].npy || !slot[eng].meta)) delete slot[eng];
      }
      if (!slot.yolo && !slot.vision) rounds.delete(round);
    }
    if (rounds.size === 0) cacheIndex.delete(base);
  }

  refreshCacheStatus();

  // Reset the input so picking the same folder again still fires `change`
  // (otherwise the second click is a no-op and the user thinks merging is
  // broken).
  els.cacheFolder.value = "";

  // Re-evaluate any already-picked video against the updated index.
  if (els.videoFile.files[0]) onVideoPick();
}

function onCacheClear() {
  cacheIndex = null;
  refreshCacheStatus();
  populateRoundSelect(null);
  if (els.videoFile.files[0]) onVideoPick();
}

function refreshCacheStatus() {
  const nVideos = cacheIndex?.size || 0;
  let nRounds = 0, nYolo = 0, nVision = 0;
  for (const rounds of cacheIndex?.values() || []) {
    for (const slot of rounds.values()) {
      nRounds++;
      if (slot.yolo)   nYolo++;
      if (slot.vision) nVision++;
    }
  }
  if (nRounds) {
    const parts = [];
    if (nYolo)   parts.push(`${nYolo} YOLO`);
    if (nVision) parts.push(`${nVision} Apple Vision`);
    els.cacheStatus.textContent =
      `— ${nRounds} rounds across ${nVideos} videos (${parts.join(" + ")})`;
    if (els.cacheSection) els.cacheSection.open = false;
    els.cacheClear.hidden = false;
  } else {
    els.cacheStatus.textContent = cacheIndex
      ? "— no `_<engine>_r{N}.npy + _meta.json` pairs found in that folder"
      : "— pick once per session";
    els.cacheClear.hidden = true;
  }
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
  // `slot` may have `yolo` and/or `vision`. Primary pose is APPLE VISION
  // when present — it's what the production iOS app runs on, and its
  // clean `conf = 0` for non-detected joints is what the facing-direction
  // lens (and any rule that consults face-confidence asymmetry) needs.
  // YOLO is loaded as the secondary so the engine-compare lens still has
  // both. If only one engine exists, that one is primary.
  const primary = slot.vision || slot.yolo;
  const secondary = (slot.vision && slot.yolo) ? slot.yolo : null;
  const status =
    `Loading ${videoFile.name}${secondary ? " (vision + yolo)" : ""}…`;
  els.loadStatus.textContent = status;

  const token = ++currentLoadToken;
  loadVideo(videoFile)
    .then(async () => {
      if (token !== currentLoadToken) return;
      const size = { width: els.video.videoWidth, height: els.video.videoHeight };
      // Each slot value may be a File OR a FileSystemFileHandle (Drive folder).
      // drive.toFile() returns a File from either.
      const primaryNpy  = await drive.toFile(primary.npy);
      const primaryMeta = await drive.toFile(primary.meta);
      const posePrimary = await loadPose([primaryNpy, primaryMeta], size);
      if (token !== currentLoadToken) return;
      // Engine tags reflect which slot each pose came from. Primary is
      // Vision when both engines are present (see the slot-pick above),
      // so when there's both we know primary = vision and secondary = yolo;
      // otherwise primary is whichever single engine exists for this round.
      const primaryEngine = (slot.vision && primary === slot.vision)
        ? "apple_vision_2d"
        : "yolo_pose";
      posePrimary.engine = primaryEngine;
      let poseSecondary = null;
      if (secondary) {
        const secNpy  = await drive.toFile(secondary.npy);
        const secMeta = await drive.toFile(secondary.meta);
        poseSecondary = await loadPose([secNpy, secMeta], size);
        if (token !== currentLoadToken) return;
        poseSecondary.engine =
          primaryEngine === "apple_vision_2d" ? "yolo_pose" : "apple_vision_2d";
      }
      // Optional sibling: ST-GCN punch detections for the primary engine.
      let punches = null;
      if (primary.punches) {
        try {
          const punchFile = await drive.toFile(primary.punches);
          punches = await loadPunches(punchFile);
        } catch (err) {
          console.warn("punches load failed:", err.message);
        }
      }
      if (token !== currentLoadToken) return;
      start(posePrimary, poseSecondary, punches);

      // Live GT labels: derive a basename from the cache filename, then hit
      // the Sheet. Best-effort — failure just leaves state.labels null so
      // the lens falls back to ST-GCN / heuristic.
      tryLiveLabels({
        cacheBasename: stripCacheSuffix(primary.npy.name),
        cacheStartSec: posePrimary.start_sec || 0,
        fps: posePrimary.fps,
        nFrames: posePrimary.n_frames,
        token,
      });
    })
    .catch(err => {
      if (token !== currentLoadToken) return;
      console.error(err);
      els.loadStatus.textContent = `Error: ${err.message}`;
    });
}

function loadFromFiles(videoFile, poseFiles) {
  // Manual file picker — single engine only. Pose loader takes the .npy and
  // the _meta.json; a third file ending in _punches.json is consumed as the
  // ST-GCN-detection source. GT labels are pulled live from the Sheet using
  // the cache basename as the source-video hint.
  const all = Array.from(poseFiles);
  const punchFile = all.find(f => /_punches\.json$/i.test(f.name));
  const poseOnly  = all.filter(f => f !== punchFile);
  const npyFile   = all.find(f => /\.npy$/i.test(f.name));
  const names = all.map(f => f.name).join(" + ");
  els.loadStatus.textContent = `Loading ${videoFile.name} + ${names}…`;
  const token = ++currentLoadToken;
  loadVideo(videoFile)
    .then(async () => {
      if (token !== currentLoadToken) return null;
      const pose = await loadPose(poseOnly, {
        width: els.video.videoWidth,
        height: els.video.videoHeight,
      });
      let punches = null;
      if (punchFile) {
        try { punches = await loadPunches(punchFile); }
        catch (err) { console.warn("punches load failed:", err.message); }
      }
      return { pose, punches };
    })
    .then(loaded => {
      if (loaded == null || token !== currentLoadToken) return;
      start(loaded.pose, null, loaded.punches);
      // Live GT labels.
      tryLiveLabels({
        cacheBasename: npyFile ? stripCacheSuffix(npyFile.name) : null,
        cacheStartSec: loaded.pose.start_sec || 0,
        fps: loaded.pose.fps,
        nFrames: loaded.pose.n_frames,
        token,
      });
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

// Strip the cache-shape tail from a .npy filename so what's left is the
// basename that points at the source video. `30 MIN…_h264_vision_r0.npy`
// → `30 MIN…_h264`. Anything that doesn't match the convention is returned
// extension-stripped so we can still try a fuzzy match.
function stripCacheSuffix(fileName) {
  if (!fileName) return null;
  return fileName
    .replace(/\.npy$/i, "")
    .replace(/_(yolo|vision)_r\d+$/i, "");
}

// Fire a best-effort live-label fetch. Doesn't block UI. On success, sets
// state.labels and remounts the active lens so the new data shows up.
async function tryLiveLabels({ cacheBasename, cacheStartSec, fps, nFrames, token }) {
  if (!cacheBasename) return;
  let live;
  try {
    live = await fetchLiveLabels({ cacheBasename, cacheStartSec, fps, nFrames });
  } catch (err) {
    console.warn("live label fetch threw:", err);
    return;
  }
  if (token !== currentLoadToken) return;
  if (live.error) {
    // Stash a minimal state.labels so the source pill can explain WHY there
    // are no labels (auto-match failed, network failed, etc).
    state.labels = { error: live.error, cacheBasename, detections: [] };
  } else {
    state.labels = live;
  }
  if (state.rule) state.rule.mount(els.ruleHost, state);
  redraw();
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

function start(pose, poseSecondary = null, punches = null) {
  state.pose = pose;
  state.poseSecondary = poseSecondary;   // optional second engine for compare
  state.punches = punches;               // optional ST-GCN detections
  state.labels = null;                   // populated asynchronously by tryLiveLabels()
  state.fps = pose.fps;
  state.n_frames = pose.n_frames;
  state.start_sec = pose.start_sec || 0;
  // YOLO extraction (yolo_pose_extraction.ipynb) floors the start frame:
  //   start_frame = int(actual_start_sec * fps); cap.set(POS_FRAMES, start_frame)
  // and walks N frames from there. The .npy data therefore lives on
  // source frames [start_frame_floor .. start_frame_floor + n_frames).
  // If we set video.currentTime = start_sec + (f+0.5)/fps directly, when
  // (start_sec * fps) has a fractional part >= 0.5, the +0.5 epsilon walks
  // us past the floored frame's time slot into the next source frame —
  // skeleton ends up drawn one frame late. Snap to the floored frame's
  // timeline instead so seek and data agree.
  state.start_frame = Math.floor(state.start_sec * state.fps);
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
  const w = els.video.videoWidth || state.pose?.width || 16;
  const h = els.video.videoHeight || state.pose?.height || 9;
  els.canvas.width = w;
  els.canvas.height = h;
  // Tell the .video-wrap CSS the aspect ratio so portrait videos don't
  // explode vertically — the wrap caps at 75vh and uses this ratio to
  // pick a sane width.
  const wrap = document.querySelector(".video-wrap");
  if (wrap) wrap.style.setProperty("--video-ratio", `${w} / ${h}`);
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
  // Cache frame N lives on source frame (start_frame + N). The +0.5
  // epsilon lands in the middle of that source frame's time slot so the
  // browser doesn't pick the previous frame on a boundary.
  els.video.currentTime = (state.start_frame + f + 0.5) / state.fps;
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
  const f = Math.floor(t_video * state.fps) - state.start_frame;
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
  els.thumbVideo.currentTime = (state.start_frame + target + 0.5) / state.fps;
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
  const ctx = els.canvas.getContext("2d");
  // Always clear so the previous frame doesn't ghost when there's no pose.
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  if (!state.pose) return;

  // Canvas internal resolution is the source video's, but the rendered CSS
  // box is often much smaller (portrait 1080×1920 capped at 75vh ≈ 149×264).
  // 2px-wide bones at 1080 internal → 0.3px on screen, i.e. invisible. Push
  // a render-scale into state so the active rule can multiply its drawing
  // dimensions accordingly. Default skeleton dims also get scaled below.
  const cssW = els.canvas.getBoundingClientRect().width || els.canvas.width;
  state.renderScale = els.canvas.width / Math.max(1, cssW);

  // Let the active rule influence the base skeleton style, then draw it.
  // Apply renderScale to width-style fields so lines/dots stay legible at
  // any rendered size. Lenses can override either by passing absolute or by
  // reading state.renderScale themselves.
  const baseStyle = state.rule?.skeletonStyle?.(state) || {};
  const scaled = {
    ...baseStyle,
    boneWidth:   (baseStyle.boneWidth   ?? 2) * state.renderScale,
    jointRadius: (baseStyle.jointRadius ?? 4) * state.renderScale,
  };
  drawSkeleton(ctx, state.pose, state.frame, scaled);

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
