// Main viewer. Owns the <video> + <canvas> sync, frame navigation, scrubber,
// and the rule-panel host. Rule panels register themselves via rules/registry.js
// and get a chance to (a) draw extra overlay graphics each frame and (b) own
// a side-panel DOM area that they refresh.

// Bump this on every push so the user can tell whether the new code is
// actually live or whether GitHub Pages / their browser is still serving
// a cached copy. Format: YYYY-MM-DD.N where N restarts at 1 each day.
const BUILD = "2026-06-27.1";
{
  const el = document.getElementById("build-tag");
  if (el) el.textContent = `build ${BUILD}`;
}

import { loadPose, loadGloveWrists, loadPtsArray, loadBlaze33 } from "./pose-loader.js";
import { loadPose3D } from "./pose-3d-loader.js";
import { loadPunches } from "./punches-loader.js";
import { fetchLiveLabels } from "./sheet-labels.js";
import { drawSkeleton } from "./skeleton.js";
import { RULES } from "./rules/registry.js";
import * as drive from "./drive-folder.js";
import * as firebaseSource from "./firebase-source.js";
import { loadOnDeviceSkeleton, loadOnDeviceAnalysis } from "./ondevice-loader.js";

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
  muteToggle:  document.getElementById("mute-toggle"),
  exportBtn:   document.getElementById("export-clip"),
  speedSel:    document.getElementById("speed"),
  frameLabel:  document.getElementById("frame-label"),
  scrubber:    document.getElementById("scrubber"),
  ruleSel:     document.getElementById("rule-select"),
  ruleHost:    document.getElementById("rule-panel"),
  videoInfo:   document.getElementById("video-info"),
  viName:      document.getElementById("vi-name"),
  viDrive:     document.getElementById("vi-drive"),
  meta:        document.getElementById("meta"),
  thumbVideo:  document.getElementById("thumb-video"),
  thumbTip:    document.getElementById("thumb-tooltip"),
  thumbCanvas: document.getElementById("thumb-canvas"),
  thumbLabel:  document.getElementById("thumb-label"),
  stageExtras: document.getElementById("stage-extras"),
  // Firebase picker (on-device sessions).
  fbSection:     document.getElementById("firebase-section"),
  fbSessionId:   document.getElementById("fb-session-id"),
  fbRound:       document.getElementById("fb-round"),
  fbLoad:        document.getElementById("fb-load"),
  fbListRecent:  document.getElementById("fb-list-recent"),
  fbRecent:      document.getElementById("fb-recent-sessions"),
  fbStatus:      document.getElementById("fb-status"),
  fbSignin:      document.getElementById("fb-signin"),
  fbSignout:     document.getElementById("fb-signout"),
  fbUser:        document.getElementById("fb-user"),
  odVideo:       document.getElementById("od-video"),
  odSkeleton:    document.getElementById("od-skeleton"),
  odAnalysis:    document.getElementById("od-analysis"),
  odRound:       document.getElementById("od-round"),
  odLoad:        document.getElementById("od-load"),
  odStatus:      document.getElementById("od-status"),
};

// Cache index built from the folder picker (or the Drive folder walker):
//   Map<videoBasename, Map<roundN, { yolo?: {npy,meta,punches?}, vision?: ... }>>
// Slot values are EITHER File objects (manual picker) OR
// FileSystemFileHandle objects (Drive folder). loadFromIndex calls
// drive.toFile() on values when it's actually time to load.
// Survives across video picks within one page session.
//
// Engine slots: `yolo`, `vision`, `vision3d`, `glove`, `vision_glove` are
// recognized by filename. `vision_combined` is a synthetic tag we apply when
// a `_vision_` file lives inside a folder whose name starts with `pose_cache_v`
// (the older "Apple Vision skeleton with glove-model wrists baked in" cache
// built by glove_wrist_cache_build.ipynb §8). `vision_glove` is the v6 cache
// (pose_cache_v6/, files named `<stem>_vision_glove_r<N>`) — same shape but
// the filename advertises the combination, and the meta carries per-round
// glove-presence info consumed by the round_v6 lens.
const ENGINE_TAGS = ["yolo", "vision", "vision3d", "rtmpose", "movenet", "yolo11", "blazepose", "glove", "vision_combined", "vision_glove"];
const SKELETON_ENGINES = ["yolo", "vision", "vision3d", "rtmpose", "movenet", "yolo11", "blazepose", "vision_combined", "vision_glove"];
const COMBINED_DIR_RE = /^pose_cache_v/i;
function classifyEngine(engine, parentDir) {
  if (engine === "vision" && parentDir && COMBINED_DIR_RE.test(parentDir)) {
    return "vision_combined";
  }
  return engine;
}
let cacheIndex = null;

// Drive-folder state: a separate index of video filename -> FileSystemFileHandle
// built by the Drive folder walker. Populated when a Drive folder is
// connected; consulted to populate the video dropdown.
let driveVideos = null;
let driveHandle = null;

// Punch-classifier predictions dumps found anywhere in the Drive folder or
// the manual cache-folder pick. Keyed by filename, value is a File OR
// FileSystemFileHandle (same dual-shape trick the cache index uses). Exposed
// to lenses via `state.predictionFiles` so the punch_classifier lens can
// auto-load without a manual file picker.
let predictionFiles = new Map();

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
  analysis: null, // populated by the Firebase load path; consumed by ondevice_lens
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
if (els.fbLoad)          els.fbLoad.addEventListener("click", onFirebaseLoad);
if (els.fbListRecent)    els.fbListRecent.addEventListener("click", onFirebaseListRecent);
if (els.fbRecent)        els.fbRecent.addEventListener("change", onFirebaseRecentPick);
if (els.odLoad)          els.odLoad.addEventListener("click", onLocalOnDeviceLoad);
if (els.fbSignin)        els.fbSignin.addEventListener("click", onFirebaseSignIn);
if (els.fbSignout)       els.fbSignout.addEventListener("click", () => firebaseSource.signOutViewer());

// Reflect Google auth state in the Firebase panel (button visibility + uid).
firebaseSource.onAuthChange((user) => {
  if (!els.fbUser) return;
  if (user) {
    els.fbUser.textContent = `— signed in as ${user.email}`;
    if (els.fbSignin) els.fbSignin.hidden = true;
    if (els.fbSignout) els.fbSignout.hidden = false;
  } else {
    els.fbUser.textContent = "— sign in with the account used on the phone (or the debug admin)";
    if (els.fbSignin) els.fbSignin.hidden = false;
    if (els.fbSignout) els.fbSignout.hidden = true;
  }
});

// If we just came back from a redirect sign-in (popup fallback), finish it —
// onAuthChange above then updates the panel.
firebaseSource.completeRedirectSignIn();

async function onFirebaseSignIn() {
  try {
    await firebaseSource.signIn();
  } catch (err) {
    console.error("[firebase sign-in]", err);
    els.fbStatus.textContent = `— sign-in failed: ${err.message}`;
  }
}

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
  predictionFiles = new Map();
  await drive.forget();
  populateDriveVideoSelect();
  populateRoundSelect(null);
  refreshCacheStatus();
  setDriveStatus("idle");
  notifyPredictionFilesChanged();
}

async function refreshDriveFolder() {
  if (!driveHandle) return;
  setDriveStatus("scanning", driveHandle.name);
  try {
    const { videos, cacheIndex: idx, predictions } = await drive.walk(driveHandle);
    driveVideos = videos;
    // Merge Drive-sourced entries on top of anything from the manual picker —
    // Drive wins (more likely fresh).
    if (!cacheIndex) cacheIndex = new Map();
    for (const [base, rounds] of idx) {
      if (!cacheIndex.has(base)) cacheIndex.set(base, new Map());
      const merged = cacheIndex.get(base);
      for (const [round, slot] of rounds) merged.set(round, slot);
    }
    // Replace any prior Drive-sourced predictions; merge with anything the
    // manual cache-folder picker may have added (those are File objects,
    // Drive's are handles — both work via drive.toFile()).
    if (predictions) {
      for (const [name, handle] of predictions) predictionFiles.set(name, handle);
    }
    populateDriveVideoSelect();
    refreshCacheStatus();
    setDriveStatus("connected", driveHandle.name);
    notifyPredictionFilesChanged();
  } catch (err) {
    console.error("Drive folder walk failed:", err);
    setDriveStatus("denied", driveHandle.name);
    els.loadStatus.textContent = `Drive folder walk failed: ${err.message}`;
  }
}

function populateDriveVideoSelect() {
  if (!els.videoPick) return;
  const sel = els.videoPick;
  // Preserve current selection across re-population (e.g. when the lens
  // changes) so the user doesn't lose the video they're inspecting.
  const previousValue = sel.value;
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
  // Lens-aware filter: only show videos whose cached rounds satisfy the
  // active lens's requires(). Two states per video:
  //   matched = lens-compatible cache exists for at least one round
  //   anyCache = some cache exists, but not for this lens (kept visible
  //              with a "(no <lens> cache)" tag so the user can see the
  //              video isn't forgotten — they just need a different lens)
  const items = [...driveVideos.entries()].map(([name, h]) => {
    const base = videoBasename(name);
    const rounds = cacheIndex?.get(base);
    const anyCache = !!rounds?.size;
    const matched = videoMatchesActiveLens(base);
    return { name, h, matched, anyCache };
  });
  // Sort: lens-matched first, then any-cache, then nothing.
  items.sort((a, b) => {
    if (a.matched !== b.matched) return a.matched ? -1 : 1;
    if (a.anyCache !== b.anyCache) return a.anyCache ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
  for (const it of items) {
    if (!it.matched) continue;   // hide non-matching videos entirely
    const o = document.createElement("option");
    o.value = it.name;
    o.textContent = it.name;
    sel.appendChild(o);
  }
  // Restore previous selection if it's still selectable, else clear.
  if (previousValue && [...sel.options].some(o => o.value === previousValue && !o.disabled)) {
    sel.value = previousValue;
  }
  sel.disabled = false;
}

// True if a round-slot satisfies the active lens's requirements.
// Default predicate (when a rule doesn't declare `requires`): at least one
// 2D engine present. This keeps every existing 2D rule working without a
// per-rule code change.
function slotMatchesActiveLens(slot) {
  const req = state.rule?.requires;
  if (!req) return !!(slot?.yolo || slot?.vision || slot?.vision_glove || slot?.rtmpose
                      || slot?.movenet || slot?.yolo11 || slot?.blazepose);
  try { return !!req(slot); }
  catch { return false; }
}

function videoMatchesActiveLens(base) {
  const rounds = cacheIndex?.get(base);
  if (!rounds) return false;
  for (const slot of rounds.values()) {
    if (slotMatchesActiveLens(slot)) return true;
  }
  return false;
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
  let predictionsTouched = false;
  for (const f of files) {
    if (/^predictions_.*\.json$/i.test(f.name)) {
      predictionFiles.set(f.name, f);
      predictionsTouched = true;
      continue;
    }
    if (f.name.endsWith(".bak.npy")) continue;
    // Three sibling patterns we recognize per round + engine:
    //   <base>_<engine>_r<N>.npy           — pose data
    //   <base>_<engine>_r<N>_meta.json     — pose metadata
    //   <base>_<engine>_r<N>_punches.json  — ST-GCN detections (optional;
    //                                        produced by dump_punches.py)
    // GT labels are pulled live from the Sheet at load time — no sidecar.
    // `vision3d` is the experimental Apple 3D engine; pairs an `.npy`
    // with optional `_cam.npy` (per-frame cameraOriginMatrix) and `_proj.npy`
    // (image-space projection via pointInImage) sidecars alongside the usual
    // `_meta.json`. `_punches` only applies to 2D engines.
    // Longer tokens first — `vision_glove` would otherwise be eaten as
    // `vision` with a base ending in `_vision`.
    const m = f.name.match(
      /^(.+?)_(vision_glove|vision3d|vision|yolo11|yolo|rtmpose|movenet|blazepose|glove)_r(\d+)(_meta|_punches|_cam|_proj|_pts)?\.(npy|json)$/
    );
    if (!m) continue;
    const [, base, rawEngine, roundStr, suffix, ext] = m;
    const round = parseInt(roundStr);
    if (ext === "json" && !suffix) continue;
    // webkitRelativePath is "pickedFolder/.../file.npy" — the immediate parent
    // directory is what disambiguates `pose_cache_v5/foo_vision_r0.npy` (the
    // combined cache) from `apple_vision_pose_cache/foo_vision_r0.npy` (raw).
    const relPath = f.webkitRelativePath || "";
    const segs = relPath.split("/");
    const parentDir = segs.length >= 2 ? segs[segs.length - 2] : "";
    const engine = classifyEngine(rawEngine, parentDir);

    if (!cacheIndex.has(base)) cacheIndex.set(base, new Map());
    const rounds = cacheIndex.get(base);
    if (!rounds.has(round)) rounds.set(round, {});
    const roundSlot = rounds.get(round);
    if (!roundSlot[engine]) roundSlot[engine] = {};
    const engineSlot = roundSlot[engine];
    if (ext === "npy" && suffix === "_cam")        engineSlot.cam = f;
    else if (ext === "npy" && suffix === "_proj")  engineSlot.proj = f;
    else if (ext === "npy" && suffix === "_pts")   engineSlot.pts = f;
    else if (ext === "npy")                        engineSlot.npy = f;
    else if (suffix === "_meta")                   engineSlot.meta = f;
    else if (suffix === "_punches")                engineSlot.punches = f;
  }

  // Drop incomplete pose pairs (need .npy + _meta.json per engine). Punches
  // and cam-matrix sidecars are optional — their absence is fine.
  for (const [base, rounds] of cacheIndex) {
    for (const [round, slot] of rounds) {
      for (const eng of ENGINE_TAGS) {
        if (slot[eng] && (!slot[eng].npy || !slot[eng].meta)) delete slot[eng];
      }
      if (!SKELETON_ENGINES.some(eng => slot[eng])) rounds.delete(round);
    }
    if (rounds.size === 0) cacheIndex.delete(base);
  }

  refreshCacheStatus();

  // Reset the input so picking the same folder again still fires `change`
  // (otherwise the second click is a no-op and the user thinks merging is
  // broken).
  els.cacheFolder.value = "";

  if (predictionsTouched) notifyPredictionFilesChanged();

  // Re-evaluate any already-picked video against the updated index.
  if (els.videoFile.files[0]) onVideoPick();
}

// Push the latest prediction-file map onto state and re-mount the active
// lens if it cares. Lens reads state.predictionFiles inside mount().
function notifyPredictionFilesChanged() {
  state.predictionFiles = predictionFiles;
  if (state.pose && state.rule) {
    state.rule.mount(els.ruleHost, state);
    redraw();
  }
}

function onCacheClear() {
  cacheIndex = null;
  refreshCacheStatus();
  populateRoundSelect(null);
  if (els.videoFile.files[0]) onVideoPick();
}

function refreshCacheStatus() {
  const nVideos = cacheIndex?.size || 0;
  let nRounds = 0, nYolo = 0, nVision = 0, nVision3D = 0, nCombined = 0, nV6 = 0;
  for (const rounds of cacheIndex?.values() || []) {
    for (const slot of rounds.values()) {
      nRounds++;
      if (slot.yolo)             nYolo++;
      if (slot.vision)           nVision++;
      if (slot.vision3d)         nVision3D++;
      if (slot.vision_combined)  nCombined++;
      if (slot.vision_glove)     nV6++;
    }
  }
  if (nRounds) {
    const parts = [];
    if (nYolo)      parts.push(`${nYolo} YOLO`);
    if (nVision)    parts.push(`${nVision} Apple Vision`);
    if (nVision3D)  parts.push(`${nVision3D} Vision 3D`);
    if (nCombined)  parts.push(`${nCombined} Vision+glove`);
    if (nV6)        parts.push(`${nV6} v6`);
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
      state.poseCombined = null;
      state.poseV6 = null;
      state.pose3d = null;
      state.engines = [];
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

// ── Firebase (on-device sessions) ───────────────────────────────────────────
//
// Parallel data source to Drive/cache folder. Fetches video + skeleton +
// on-device analysis sidecar from Firebase Storage for a (sessionId,
// roundNumber) and feeds them into start() like any other load path.
// State.analysis is set so the on-device lens can render.

async function onFirebaseLoad() {
  const sessionId = els.fbSessionId.value.trim();
  const roundN = parseInt(els.fbRound.value, 10);
  if (!sessionId || Number.isNaN(roundN)) {
    els.fbStatus.textContent = "— enter a session id and round number";
    return;
  }

  els.fbStatus.textContent = `— fetching ${sessionId} r${roundN}…`;
  els.loadStatus.textContent = `Loading ${sessionId} / round ${roundN} from Firebase…`;
  const token = ++currentLoadToken;

  try {
    const blobs = await firebaseSource.fetchRoundBlobs(sessionId, roundN);
    if (token !== currentLoadToken) return;
    await startOnDeviceRound(blobs, sessionId, roundN, els.fbStatus, token);
  } catch (err) {
    if (token !== currentLoadToken) return;
    console.error("[firebase load]", err);
    els.fbStatus.textContent = `— error: ${err.message}`;
    els.loadStatus.textContent = `Firebase load failed: ${err.message}`;
  }
}

// Shared by the Firebase loader and the local on-device picker: take the
// three blobs (video required, analysis optional) and feed them into start()
// the same way. `idLabel`/`roundN` are identity hints for lenses + status.
async function startOnDeviceRound(blobs, idLabel, roundN, statusEl, token) {
  if (!blobs.videoBlob) {
    throw new Error(
      "round has no uploaded video yet (skeleton/analysis only) — retry once the video finishes uploading",
    );
  }
  // Wrap the video blob in a File so loadVideo's pattern matches existing
  // call sites (it only uses the Blob interface, but giving it a name keeps
  // state.videoFileName meaningful).
  const videoFile = new File([blobs.videoBlob], `${idLabel}_r${roundN}.mp4`, { type: "video/mp4" });
  await loadVideo(videoFile);
  if (token !== currentLoadToken) return;

  const pose = await loadOnDeviceSkeleton(blobs.skeletonBlob);
  if (token !== currentLoadToken) return;
  pose.engine = pose.engine || "apple_vision_2d";

  let analysis = null;
  if (blobs.analysisBlob) {
    try {
      analysis = await loadOnDeviceAnalysis(blobs.analysisBlob);
    } catch (err) {
      console.error("[on-device load] analysis parse failed:", err);
    }
  }
  if (token !== currentLoadToken) return;

  // Identity hints used by some lenses (orientation_lens reads cacheBasename
  // to pull Sheet labels; on-device lens doesn't need them but we set them
  // anyway for consistency).
  state.cacheBasename = idLabel;
  state.cacheRound = roundN;

  // Pull punches out of the analysis sidecar so the existing punch rendering
  // pipeline (3rd start() arg) lights up just like it does for training-cache
  // loads.
  const punches = analysis?.punches ?? null;
  start(pose, null, punches, null, null, null, analysis);

  const punchNote = punches ? `· ${punches.detections.length} punches` : "";
  const sidecarNote = analysis
    ? `with on-device analysis (${Object.keys(analysis.rules).length} rules${punchNote})`
    : `no analysis sidecar`;
  if (statusEl) statusEl.textContent = `— loaded ${idLabel} r${roundN}, ${sidecarNote}`;
  els.loadStatus.textContent = "";
}

// Local-file twin of onFirebaseLoad: read the on-device round straight from
// disk (video + skeleton + optional analysis sidecar). Sidesteps Storage
// owner-scope auth entirely — used for rounds pulled off the phone.
async function onLocalOnDeviceLoad() {
  const videoFile = els.odVideo.files?.[0];
  const skeletonFile = els.odSkeleton.files?.[0];
  const analysisFile = els.odAnalysis.files?.[0] ?? null;
  if (!videoFile || !skeletonFile) {
    els.odStatus.textContent = "— pick at least a video + skeleton JSON";
    return;
  }
  const roundN = parseInt(els.odRound.value, 10) || 0;
  const idLabel = videoFile.name.replace(/\.mp4$/i, "");
  els.odStatus.textContent = `— loading ${videoFile.name}…`;
  els.loadStatus.textContent = "Loading on-device round from local files…";
  const token = ++currentLoadToken;
  try {
    await startOnDeviceRound(
      { videoBlob: videoFile, skeletonBlob: skeletonFile, analysisBlob: analysisFile },
      idLabel, roundN, els.odStatus, token,
    );
  } catch (err) {
    if (token !== currentLoadToken) return;
    console.error("[local on-device load]", err);
    els.odStatus.textContent = `— error: ${err.message}`;
    els.loadStatus.textContent = `Local load failed: ${err.message}`;
  }
}

async function onFirebaseListRecent() {
  els.fbStatus.textContent = "— fetching session list…";
  try {
    const sessions = await firebaseSource.listRecentSessions(20);
    els.fbRecent.innerHTML = '<option value="">— pick a recent session —</option>';
    for (const s of sessions) {
      const opt = document.createElement("option");
      opt.value = s.sessionId;
      // Stash the available rounds so onFirebaseRecentPick can default the
      // round input to a value that actually exists for this session.
      opt.dataset.rounds = JSON.stringify(s.rounds);
      const rounds = s.rounds.length ? `r${s.rounds.join(",")}` : "(no rounds)";
      opt.textContent = `${s.sessionId} (${rounds})`;
      els.fbRecent.appendChild(opt);
    }
    els.fbRecent.hidden = sessions.length === 0;
    els.fbStatus.textContent = sessions.length
      ? `— ${sessions.length} recent sessions`
      : `— no sessions found`;
  } catch (err) {
    console.error("[firebase list]", err);
    els.fbStatus.textContent = `— list error: ${err.message}`;
  }
}

function onFirebaseRecentPick() {
  const sel = els.fbRecent;
  const sessionId = sel.value;
  if (!sessionId) return;
  els.fbSessionId.value = sessionId;
  // Default the round to the first one this session actually has — sessions
  // that started at round 1 (or higher) would 404 if we left the input at 0.
  const opt = sel.options[sel.selectedIndex];
  const rounds = opt?.dataset?.rounds ? JSON.parse(opt.dataset.rounds) : [];
  if (rounds.length > 0) {
    els.fbRound.value = String(rounds[0]);
  }
  onFirebaseLoad();
}

function loadFromIndex(videoFile, slot) {
  // BlazePose is REQUIRED for every lens (user request 2026-06-24). If this
  // round has no BlazePose cache, show the video with a clear "missing
  // BlazePose" message and NO skeleton — never silently fall back to Apple
  // Vision. loadVideoOnly clears any leftover skeleton so the screen is honest
  // about the gap.
  if (!slot.blazepose) {
    loadVideoOnly(videoFile,
      `⚠ No BlazePose cache for this round — the viewer requires BlazePose for ` +
      `every lens (Apple Vision fallback is disabled). Extract a _blazepose_ ` +
      `cache for this video/round.`);
    return;
  }
  // `slot` may also have `yolo`, `vision`, and/or `vision_glove`. The `primary`
  // variable below is the SIDECAR ANCHOR (glove wrists, punches, 3D, v6) and
  // keeps the Vision-first priority — but it is NOT the skeleton the rule
  // lenses run on. The rule-facing primary is forced to BlazePose just before
  // start() (see below); Vision is demoted to the secondary slot there so the
  // compare lenses and wrist_swap can still reach it.
  const primary = slot.vision || slot.yolo || slot.vision_glove || slot.rtmpose
                || slot.movenet || slot.yolo11 || slot.blazepose;
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
      // If the v6 cache was the only thing in the slot, the meta records
      // the actual engine (`apple_vision_2d` for ungloved or
      // `apple_vision_2d+glove_v6` for gloved) — use it verbatim.
      let primaryEngine;
      if (slot.vision && primary === slot.vision)            primaryEngine = "apple_vision_2d";
      else if (slot.yolo && primary === slot.yolo)           primaryEngine = "yolo_pose";
      else if (slot.vision_glove && primary === slot.vision_glove)
        primaryEngine = posePrimary.meta?.engine || "apple_vision_2d+glove_v6";
      else if (slot.rtmpose && primary === slot.rtmpose)     primaryEngine = "rtmpose_body17";
      else if (slot.movenet && primary === slot.movenet)     primaryEngine = "movenet";
      else if (slot.yolo11 && primary === slot.yolo11)       primaryEngine = "yolo11_pose";
      else if (slot.blazepose && primary === slot.blazepose) primaryEngine = "blazepose";
      else                                                    primaryEngine = "unknown";
      posePrimary.engine = primaryEngine;
      // Per-frame PTS sidecar → exact cross-engine time alignment (engine_compare).
      if (primary.pts) posePrimary.pts = await loadPtsArray(await drive.toFile(primary.pts));
      let poseSecondary = null;
      if (secondary) {
        const secNpy  = await drive.toFile(secondary.npy);
        const secMeta = await drive.toFile(secondary.meta);
        poseSecondary = await loadPose([secNpy, secMeta], size);
        if (token !== currentLoadToken) return;
        poseSecondary.engine =
          primaryEngine === "apple_vision_2d" ? "yolo_pose" : "apple_vision_2d";
        if (secondary.pts) poseSecondary.pts = await loadPtsArray(await drive.toFile(secondary.pts));
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
      // Optional: Apple Vision 3D cache lives in slot.vision3d. Loaded
      // entirely independently of the 2D engines — different layout
      // (17-joint Apple-native), different coordinate space (body-frame
      // metres). The Vision 3D lens consumes it; 2D rules ignore it.
      let pose3d = null;
      if (slot.vision3d) {
        try {
          const v3 = slot.vision3d;
          pose3d = await loadPose3D({
            npy:  await drive.toFile(v3.npy),
            meta: await drive.toFile(v3.meta),
            cam:  v3.cam  ? await drive.toFile(v3.cam)  : null,
            proj: v3.proj ? await drive.toFile(v3.proj) : null,
          });
        } catch (err) {
          console.warn("3D pose load failed:", err.message);
        }
      }
      // Optional: glove-wrist sidecar — attached to the vision pose object
      // (or primary, if no vision). Frame timing matches the matching
      // vision cache 1:1, so the wrist-swap lens can index it directly.
      if (slot.glove) {
        try {
          const gNpy  = await drive.toFile(slot.glove.npy);
          const gMeta = await drive.toFile(slot.glove.meta);
          const glove = await loadGloveWrists(gNpy, gMeta, size);
          // Attach to whichever pose is the Vision pose; fall back to primary.
          if (poseSecondary && poseSecondary.engine === "apple_vision_2d") {
            poseSecondary.gloveWrists = glove;
          } else {
            posePrimary.gloveWrists = glove;
          }
        } catch (err) {
          console.warn("glove wrists load failed:", err.message);
        }
      }
      // Optional: combined vision+glove cache (pose_cache_v*/). Same shape as
      // a raw vision cache but with wrists 9/10 replaced where the glove
      // model was confident. Carried separately so combined_compare can
      // overlay it against the raw vision pose.
      let poseCombined = null;
      if (slot.vision_combined) {
        try {
          const cNpy  = await drive.toFile(slot.vision_combined.npy);
          const cMeta = await drive.toFile(slot.vision_combined.meta);
          poseCombined = await loadPose([cNpy, cMeta], size);
          poseCombined.engine = "apple_vision_2d_combined";
        } catch (err) {
          console.warn("combined pose load failed:", err.message);
        }
      }
      // Optional: v6 cache (pose_cache_v6/). Production-shape Apple Vision
      // skeleton with glove wrists baked in for gloved rounds. The round_v6
      // lens reads this directly so it doesn't redo the substitution; engine
      // / presence / wrist_replaced flags live in the meta JSON.
      let poseV6 = null;
      if (slot.vision_glove) {
        try {
          const vgNpy  = await drive.toFile(slot.vision_glove.npy);
          const vgMeta = await drive.toFile(slot.vision_glove.meta);
          poseV6 = await loadPose([vgNpy, vgMeta], size);
          // Reflect whichever engine the meta records — pure Vision for
          // ungloved rounds, vision+glove for gloved rounds.
          poseV6.engine = poseV6.meta?.engine || "apple_vision_2d+glove_v6";
        } catch (err) {
          console.warn("v6 pose load failed:", err.message);
        }
      }
      // Optional: RTMPose Body-17 cache (rtmpose_pose_cache/). Same COCO-17
      // (N,17,3) normalized layout as YOLO/Vision, aligned to the same rounds —
      // loaded as a separate engine so engine_compare can pick it vs Vision.
      let poseRtm = null;
      if (slot.rtmpose) {
        try {
          const rNpy  = await drive.toFile(slot.rtmpose.npy);
          const rMeta = await drive.toFile(slot.rtmpose.meta);
          poseRtm = await loadPose([rNpy, rMeta], size);
          poseRtm.engine = "rtmpose_body17";
          if (slot.rtmpose.pts) poseRtm.pts = await loadPtsArray(await drive.toFile(slot.rtmpose.pts));
        } catch (err) {
          console.warn("rtmpose pose load failed:", err.message);
        }
      }
      // Bake-off engines (COCO-17): MoveNet / YOLO11 / BlazePose (coco17 remap).
      // Loaded as independent engines so the multi-skeleton compare lens can
      // overlay any of them. blazepose33 (feet) is NOT loaded here — non-coco17.
      const loadEngine = async (s, tag) => {
        if (!s) return null;
        try {
          const p = await loadPose([await drive.toFile(s.npy), await drive.toFile(s.meta)], size);
          p.engine = tag;
          if (s.pts) p.pts = await loadPtsArray(await drive.toFile(s.pts));
          return p;
        } catch (err) { console.warn(`${tag} load failed:`, err.message); return null; }
      };
      const poseMovenet = await loadEngine(slot.movenet, "movenet");
      const poseYolo11  = await loadEngine(slot.yolo11, "yolo11_pose");
      const poseBlaze   = await loadEngine(slot.blazepose, "blazepose");

      // Full BlazePose-33 (all joints + z + visibility + presence) for the
      // dedicated inspector lens. The COCO-17 remap above (poseBlaze) feeds
      // engine_compare; this keeps everything the engine_compare path drops.
      let blaze33 = null;
      if (slot.blazepose) {
        try {
          blaze33 = await loadBlaze33(await drive.toFile(slot.blazepose.npy),
                                      await drive.toFile(slot.blazepose.meta), size);
          if (slot.blazepose.pts) blaze33.pts = await loadPtsArray(await drive.toFile(slot.blazepose.pts));
        } catch (err) { console.warn("blaze33 load failed:", err.message); }
      }

      if (token !== currentLoadToken) return;

      // ── BlazePose is the REQUIRED working skeleton for every rule lens ─────
      // (user request 2026-06-24): the primary handed to start() must be
      // BlazePose so every lens that reads state.pose runs on it. poseBlaze is
      // already the 33→COCO-17 remap (pose-loader), so each lens keeps the EXACT
      // joint indices it used for Vision — a 17-joint lens still sees 17, an
      // ankles-only lens still sees ankles. The rounds-without-BlazePose case is
      // handled loudly at the top of loadFromIndex; reaching here with no blaze
      // pose means the cache file was present but failed to load — fail loudly
      // rather than silently rendering Apple Vision.
      const rulePrimary = posePrimary.engine === "blazepose" ? posePrimary : poseBlaze;
      if (!rulePrimary) {
        throw new Error(
          `BlazePose cache for this round failed to load (${primary.npy.name}). ` +
          `The viewer requires BlazePose for every lens — Apple Vision fallback is disabled.`
        );
      }
      // Keep the original non-BlazePose primary (Vision/YOLO/…, with its glove /
      // punch sidecars) reachable for the compare lenses and the Vision-anchored
      // wrist_swap lens: demote it to the secondary slot when free, else to an
      // extra engine.
      let ruleSecondary = poseSecondary;
      let extraEngines = [poseMovenet, poseYolo11, poseBlaze].filter(Boolean);
      if (rulePrimary !== posePrimary) {
        if (!ruleSecondary) ruleSecondary = posePrimary;
        else extraEngines.push(posePrimary);
      }
      extraEngines = extraEngines.filter(p => p !== rulePrimary);
      start(rulePrimary, ruleSecondary, punches, pose3d, poseCombined, poseV6, null, poseRtm,
            extraEngines, blaze33);

      // Expose cache identity on state so lenses that key by (stem, round, frame)
      // — e.g. orientation_lens looking up orientation GT labels — can find it
      // without re-parsing filenames themselves. Round comes from the `_rN`
      // suffix; cacheBasename is the source-video stem (suffix stripped).
      // Longer engine tokens first so `_vision_glove_` doesn't get truncated
      // at `_vision_`.
      const npyName = primary.npy.name;
      state.cacheBasename = stripCacheSuffix(npyName);
      const rndMatch = /_(?:vision_glove|vision|yolo11|yolo|rtmpose|movenet|blazepose)_r(\d+)\.npy$/i.exec(npyName);
      state.cacheRound = rndMatch ? parseInt(rndMatch[1], 10) : null;

      // Live GT labels: derive a basename from the cache filename, then hit
      // the Sheet. Best-effort — failure just leaves state.labels null so
      // the lens falls back to ST-GCN / heuristic.
      tryLiveLabels({
        cacheBasename: state.cacheBasename,
        cacheStartSec: rulePrimary.start_sec || 0,
        fps: rulePrimary.fps,
        nFrames: rulePrimary.n_frames,
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
      // Mirror the cache-identity wiring from loadFromIndex so lenses that
      // need (stem, round, frame) work in the manual-picker path too.
      if (npyFile) {
        state.cacheBasename = stripCacheSuffix(npyFile.name);
        const rndMatch = /_(?:vision_glove|vision|yolo11|yolo|rtmpose|movenet|blazepose)_r(\d+)\.npy$/i.exec(npyFile.name);
        state.cacheRound = rndMatch ? parseInt(rndMatch[1], 10) : null;
      }
      // Live GT labels.
      tryLiveLabels({
        cacheBasename: state.cacheBasename || null,
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
  // start()→setRule() rebuilds this dropdown on every load, so the user's
  // pick has to survive the rebuild or it visually snaps back to r0.
  const prev = els.roundSel.value;
  els.roundSel.innerHTML = "";
  if (!rounds || rounds.size === 0) {
    els.roundSel.innerHTML = `<option value="">—</option>`;
    els.roundSel.disabled = true;
    return;
  }
  // Lens-aware: rounds without a cache for the active lens are still listed
  // (so the user can see r0/r1/r2/… all exist) but disabled so they can't
  // be picked when the lens won't render anything for them.
  const sorted = [...rounds.keys()].sort((a, b) => a - b);
  let enabledCount = 0;
  for (const r of sorted) {
    const o = document.createElement("option");
    o.value = String(r);
    const slot = rounds.get(r);
    const ok = slotMatchesActiveLens(slot);
    if (!ok) o.disabled = true;
    o.textContent = ok ? `r${r}` : `r${r} (no cache for lens)`;
    els.roundSel.appendChild(o);
    if (ok) enabledCount++;
  }
  els.roundSel.disabled = enabledCount < 2;
  if (prev && [...els.roundSel.options].some(o => o.value === prev && !o.disabled)) {
    els.roundSel.value = prev;
  }
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
    // Strip any recognised cache-shape tail. Engines are
    // yolo/vision/vision3d/glove/vision_combined/vision_glove per ENGINE_TAGS;
    // anything else falls through and we hand the raw stem to the
    // auto-matcher's fuzzy logic. Longer tokens listed first so
    // `_vision_glove_` matches as a unit instead of being truncated to
    // `_vision_`.
    .replace(/_(vision_glove|vision_combined|vision3d|vision|yolo11|yolo|rtmpose|movenet|blazepose|glove)_r\d+$/i, "");
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
    updateVideoInfo();
    els.video.src = state.videoUrl;
    // The thumb-video shares the source so we can seek-and-snapshot it on
    // scrubber hover without disturbing the main playback.
    els.thumbVideo.src = state.videoUrl;
    els.video.onloadedmetadata = () => resolve();
    els.video.onerror = () => reject(new Error("Video failed to load"));
  });
}

// Surface the current source video's filename plus a Drive *search* link in
// the side panel. We only have local file handles (File System Access API),
// not Drive file IDs, so a name-search query is the best deep link we can
// build without Drive API + OAuth. yt-dlp may have substituted unicode
// look-alikes (｜ ⧸) into the filename; those round-trip fine through the
// search query.
function updateVideoInfo() {
  if (!els.videoInfo) return;
  const name = state.videoFileName;
  if (!name) { els.videoInfo.hidden = true; return; }
  els.viName.textContent = name;
  const stem = name.replace(/\.[^.]+$/, "");
  els.viDrive.href = `https://drive.google.com/drive/search?q=${encodeURIComponent(stem)}`;
  els.videoInfo.hidden = false;
}

// Short human label for an engine tag — used by the multi-skeleton compare lens.
function engineDisplayLabel(tag) {
  const t = tag || "";
  if (t === "apple_vision_2d") return "Vision";
  if (t === "apple_vision_2d_combined") return "Vision-comb";
  if (t.startsWith("apple_vision_2d+glove")) return "v6";
  if (t === "yolo_pose") return "YOLO";
  if (t === "yolo11_pose") return "YOLO11";
  if (t === "rtmpose_body17") return "RTMPose";
  if (t === "movenet") return "MoveNet";
  if (t === "blazepose") return "BlazePose";
  return t || "pose";
}

function start(pose, poseSecondary = null, punches = null, pose3d = null, poseCombined = null, poseV6 = null, analysis = null, poseRtm = null, extraEngines = [], blaze33 = null) {
  state.pose = pose;
  state.blaze33 = blaze33;               // optional full BlazePose-33 (inspector lens)
  state.poseSecondary = poseSecondary;   // optional second engine for compare
  state.poseCombined = poseCombined;     // optional vision+glove combined cache
  state.poseV6 = poseV6;                 // optional pose_cache_v6 (vision+glove_v6)
  state.poseRtm = poseRtm;               // optional RTMPose Body-17 (rtmpose_pose_cache)
  // Every COCO-17 skeleton engine loaded for this round, deduped by engine tag —
  // what the multi-skeleton compare lens overlays. Primary first, then the rest.
  state.engines = [];
  {
    const seenEng = new Set();
    for (const p of [pose, poseSecondary, poseCombined, poseV6, poseRtm, ...extraEngines]) {
      if (!p || !p.skeleton || seenEng.has(p.engine)) continue;
      seenEng.add(p.engine);
      state.engines.push({ key: p.engine, label: engineDisplayLabel(p.engine), pose: p });
    }
  }
  state.punches = punches;               // optional ST-GCN detections
  state.pose3d = pose3d;                 // optional Apple Vision 3D (separate layout)
  state.analysis = analysis;             // optional on-device analysis sidecar (Firebase load path)
  state.labels = null;                   // populated asynchronously by tryLiveLabels()
  state.orientationLabels = null;        // populated by orientation lens on demand
  state.predictionFiles = predictionFiles; // punch-classifier dumps from Drive / cache folder
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

// ── Ctrl+scroll to zoom the video + overlay ──────────────────────────────────
// Scales the video and its skeleton overlay together via a shared CSS transform
// (transform-origin 0 0, so identical transforms stay aligned). Ctrl+wheel —
// which is also what a macOS trackpad pinch dispatches — zooms toward the
// cursor; scrolling back out clamps to 1× and recentres. Double-click resets.
const videoWrap = document.querySelector(".video-wrap");
const zoom = { scale: 1, tx: 0, ty: 0 };
const ZOOM_MAX = 8;

function applyZoom() {
  const t = `translate(${zoom.tx}px, ${zoom.ty}px) scale(${zoom.scale})`;
  els.video.style.transform = t;
  els.canvas.style.transform = t;
  // Hint that the frame is draggable once it's zoomed in.
  if (videoWrap) videoWrap.style.cursor = zoom.scale > 1 ? "grab" : "";
}

// Keep the frame covering the wrap (no black gaps) after any tx/ty change.
function clampPan(rect) {
  zoom.tx = Math.max(rect.width  * (1 - zoom.scale), Math.min(0, zoom.tx));
  zoom.ty = Math.max(rect.height * (1 - zoom.scale), Math.min(0, zoom.ty));
}

function resetZoom() {
  zoom.scale = 1; zoom.tx = 0; zoom.ty = 0;
  applyZoom();
}

if (videoWrap) {
  els.video.style.transformOrigin = "0 0";
  els.canvas.style.transformOrigin = "0 0";
  videoWrap.addEventListener("wheel", e => {
    if (!e.ctrlKey) return;            // only ctrl+scroll / pinch zooms
    e.preventDefault();
    const rect = videoWrap.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    // Content point currently under the cursor, in unscaled wrap coords.
    const cx = (mx - zoom.tx) / zoom.scale;
    const cy = (my - zoom.ty) / zoom.scale;
    // Exponential step feels uniform across zoom levels.
    const next = zoom.scale * Math.exp(-e.deltaY * 0.0015);
    zoom.scale = Math.max(1, Math.min(ZOOM_MAX, next));
    // Keep that content point pinned under the cursor.
    zoom.tx = mx - cx * zoom.scale;
    zoom.ty = my - cy * zoom.scale;
    clampPan(rect);
    applyZoom();
  }, { passive: false });
  videoWrap.addEventListener("dblclick", resetZoom);

  // Hold left mouse + drag to pan the zoomed-in frame around.
  let drag = null;
  videoWrap.addEventListener("mousedown", e => {
    if (e.button !== 0 || zoom.scale <= 1) return;
    e.preventDefault();
    drag = { x: e.clientX, y: e.clientY, tx: zoom.tx, ty: zoom.ty };
    videoWrap.style.cursor = "grabbing";
  });
  window.addEventListener("mousemove", e => {
    if (!drag) return;
    zoom.tx = drag.tx + (e.clientX - drag.x);
    zoom.ty = drag.ty + (e.clientY - drag.y);
    clampPan(videoWrap.getBoundingClientRect());
    applyZoom();
  });
  window.addEventListener("mouseup", () => {
    if (!drag) return;
    drag = null;
    applyZoom();                       // restores the "grab" cursor
  });
}

// ── Rule panels ─────────────────────────────────────────────────────────────
function populateRuleSelect() {
  // Preserve current selection across the rebuild so loading a new video
  // doesn't snap the lens back to the first one. The RULES list is static
  // so the previous id will always still exist.
  const prev = els.ruleSel.value;
  els.ruleSel.innerHTML = "";
  for (const r of RULES) {
    const o = document.createElement("option");
    o.value = r.id;
    o.textContent = r.label;
    els.ruleSel.appendChild(o);
  }
  if (prev && [...els.ruleSel.options].some(o => o.value === prev)) {
    els.ruleSel.value = prev;
  }
}

els.ruleSel.addEventListener("change", () => setRule(els.ruleSel.value));

function setRule(id) {
  const rule = RULES.find(r => r.id === id);
  if (!rule) return;
  state.rule = rule;
  els.ruleHost.innerHTML = "";
  // Clear the stage-wide extras slot so a lens that added a full-width
  // canvas (e.g. punch_classifier's timeline) doesn't leak into the next
  // lens. Lenses that want it back put their elements back in mount().
  if (els.stageExtras) els.stageExtras.innerHTML = "";
  rule.mount(els.ruleHost, state);
  // Lens may change which videos/rounds are valid — refresh the dropdowns
  // so they reflect the new lens's requirements. We do NOT auto-swap the
  // currently-loaded video; the lens's own mount() shows an empty/hint
  // state if the loaded clip can't be rendered.
  populateDriveVideoSelect();
  const v = els.videoFile.files[0];
  if (v) populateRoundSelect(cacheIndex?.get(videoBasename(v.name)));
  redraw();
}

// Expose the viewer's redraw to lenses that need to repaint the main canvas
// in response to their own controls (e.g. the Vision 3D lens's "Overlay on
// video" toggle). The lens calls window.__viewerRedraw() — small hack vs.
// inventing a richer rule API.
window.__viewerRedraw = () => redraw();

// ── Frame navigation ────────────────────────────────────────────────────────
els.prevFrame.addEventListener("click", () => seekToFrame(state.frame - 1));
els.nextFrame.addEventListener("click", () => seekToFrame(state.frame + 1));
els.playPause.addEventListener("click", togglePlay);
els.muteToggle.addEventListener("click", toggleMute);
els.scrubber.addEventListener("input", e => seekToFrame(parseInt(e.target.value)));
els.speedSel.addEventListener("change", () => {
  els.video.playbackRate = parseFloat(els.speedSel.value);
});
// Setting els.video.src (new video OR new round) resets playbackRate to 1.0,
// but the speed dropdown UI still shows whatever the user last picked. Re-apply
// the selected rate every time metadata loads so the displayed speed matches
// reality.
els.video.addEventListener("loadedmetadata", () => {
  els.video.playbackRate = parseFloat(els.speedSel.value);
  resetZoom();
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
    case "m": case "M": toggleMute(); break;
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
  if (recording) recTick();
  if (!els.video.paused) {
    playbackHandle = els.video.requestVideoFrameCallback(rvfcTick);
  }
}

function rafTick() {
  if (els.video.paused) { playbackHandle = null; return; }
  syncFromVideoTime(els.video.currentTime);
  if (recording) recTick();
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

function toggleMute() {
  els.video.muted = !els.video.muted;
  els.muteToggle.textContent = els.video.muted ? "🔇" : "🔊";
}
els.video.addEventListener("volumechange", () => {
  els.muteToggle.textContent = els.video.muted ? "🔇" : "🔊";
});

// ── Export: record video + current-lens overlay to a .webm ──────────────────
// redraw() already paints the overlay canvas (skeleton + active lens) frame-
// accurately. To export, we composite the <video> frame and that overlay onto
// an offscreen canvas on every displayed frame during a real-time playback
// pass (see the rvfc/raf tick hooks) and pipe it through MediaRecorder. The
// overlay's internal resolution equals the video's, so the two layers align
// 1:1 at native resolution.
let recording = false;
let recCanvas = null, recCtx = null, recorder = null, recChunks = null;
let recStopFrame = 0, recPrevRate = 1;

function compositeRecFrame() {
  if (!recCtx) return;
  recCtx.drawImage(els.video,  0, 0, recCanvas.width, recCanvas.height);
  recCtx.drawImage(els.canvas, 0, 0, recCanvas.width, recCanvas.height);
}

// Called from the playback ticks while recording: grab the frame, then stop
// once playback has reached the requested end frame.
function recTick() {
  compositeRecFrame();
  if (state.frame >= recStopFrame) finishRecording();
}

function pickRecMime() {
  // Prefer mp4 — Safari (and recent Chrome) record H.264 mp4 directly. Fall
  // back to webm on browsers that can't. The download extension follows the
  // mime actually chosen (see recorder.onstop).
  const types = [
    "video/mp4;codecs=avc1",
    "video/mp4",
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];
  return types.find(t => MediaRecorder.isTypeSupported(t)) || "";
}

function finishRecording() {
  if (!recording) return;
  recording = false;
  els.video.pause();
  els.video.playbackRate = recPrevRate;
  if (recorder && recorder.state !== "inactive") recorder.stop();
}

// Record cache frames [startFrame, endFrame] at 1x, compositing video+overlay.
function startRecording(startFrame, endFrame) {
  if (recording) return;
  if (!state.pose || !els.video.src) { alert("Load a video + pose cache first."); return; }
  const last = state.n_frames - 1;
  startFrame = Math.max(0, Math.min(last, Math.round(startFrame)));
  endFrame   = Math.max(startFrame, Math.min(last, Math.round(endFrame)));
  recStopFrame = endFrame;

  recCanvas = document.createElement("canvas");
  recCanvas.width  = els.video.videoWidth;
  recCanvas.height = els.video.videoHeight;
  recCtx = recCanvas.getContext("2d");

  recChunks = [];
  recorder = new MediaRecorder(recCanvas.captureStream(state.fps), { mimeType: pickRecMime() });
  recorder.ondataavailable = e => { if (e.data.size) recChunks.push(e.data); };

  const lensId = state.rule?.id || "raw";
  recorder.onstop = () => {
    const blob = new Blob(recChunks, { type: recorder.mimeType });
    const ext = recorder.mimeType.includes("mp4") ? "mp4" : "webm";
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${lensId}_overlay.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
    els.exportBtn.disabled = false;
    els.exportBtn.textContent = "⬇︎ Export";
  };

  recPrevRate = els.video.playbackRate;
  els.video.playbackRate = 1;
  els.exportBtn.disabled = true;
  els.exportBtn.textContent = "● Recording…";

  // Natural end is a safety net when endFrame is the last frame.
  els.video.addEventListener("ended", finishRecording, { once: true });

  seekToFrame(startFrame);
  els.video.addEventListener("seeked", () => {
    recording = true;
    compositeRecFrame();   // seed the stream with the first frame
    recorder.start();
    els.video.play();
  }, { once: true });
}

// ── Export dialog ───────────────────────────────────────────────────────────
// Time fields accept "m:ss(.sss)" OR plain seconds, in the same clock the frame
// label shows (start_sec + frame/fps). "Use current" copies the scrubbed point.
function parseTimeInput(str) {
  const s = String(str).trim();
  if (!s) return NaN;
  if (s.includes(":")) {
    const parts = s.split(":");
    if (parts.length > 3) return NaN;
    let v = 0;
    for (const p of parts) {
      const n = Number(p);
      if (!Number.isFinite(n) || n < 0) return NaN;
      v = v * 60 + n;
    }
    return v;
  }
  const n = Number(s);
  return Number.isFinite(n) && n >= 0 ? n : NaN;
}

function currentVideoTime() {
  return state.start_sec + state.frame / state.fps;
}

let exportDialog = null;
function buildExportDialog() {
  const dlg = document.createElement("dialog");
  dlg.id = "export-dialog";
  dlg.innerHTML = `
    <form class="export-form" method="dialog">
      <h3>Export overlay clip</h3>
      <label class="opt">
        <input type="radio" name="exp-range" value="whole" checked>
        <span>Whole video <span class="muted" id="exp-whole-range"></span></span>
      </label>
      <label class="opt">
        <input type="radio" name="exp-range" value="custom">
        <span>Custom range</span>
      </label>
      <div class="export-range" hidden>
        <div class="rng-row">
          <label for="exp-start">Start</label>
          <input id="exp-start" type="text" placeholder="0:00" autocomplete="off">
          <button type="button" id="exp-start-now">Use current</button>
        </div>
        <div class="rng-row">
          <label for="exp-end">End</label>
          <input id="exp-end" type="text" placeholder="1:30" autocomplete="off">
          <button type="button" id="exp-end-now">Use current</button>
        </div>
        <p class="hint">Enter <code>m:ss</code> (e.g. <code>1:30</code>) or seconds (e.g. <code>90</code>).
          Tip: scrub the video, then click <b>Use current</b>.</p>
      </div>
      <div class="export-err" id="exp-error"></div>
      <div class="export-actions">
        <button type="button" id="exp-cancel">Cancel</button>
        <button type="button" id="exp-go" class="primary">Export</button>
      </div>
    </form>`;
  document.body.appendChild(dlg);

  const rangeBox = dlg.querySelector(".export-range");
  const startIn  = dlg.querySelector("#exp-start");
  const endIn    = dlg.querySelector("#exp-end");
  const errEl    = dlg.querySelector("#exp-error");
  const radios   = dlg.querySelectorAll('input[name="exp-range"]');
  const isCustom = () => dlg.querySelector('input[name="exp-range"]:checked').value === "custom";

  radios.forEach(r => r.addEventListener("change", () => {
    rangeBox.hidden = !isCustom();
    errEl.textContent = "";
  }));
  dlg.querySelector("#exp-start-now").addEventListener("click", () => {
    startIn.value = fmtClock(currentVideoTime());
  });
  dlg.querySelector("#exp-end-now").addEventListener("click", () => {
    endIn.value = fmtClock(currentVideoTime());
  });
  dlg.querySelector("#exp-cancel").addEventListener("click", () => dlg.close());

  dlg.querySelector("#exp-go").addEventListener("click", () => {
    const last = state.n_frames - 1;
    let startFrame = 0, endFrame = last;
    if (isCustom()) {
      const ts = parseTimeInput(startIn.value);
      const te = parseTimeInput(endIn.value);
      if (Number.isNaN(ts) || Number.isNaN(te)) {
        errEl.textContent = "Enter both times as m:ss or seconds."; return;
      }
      if (te <= ts) { errEl.textContent = "End must be after start."; return; }
      startFrame = Math.round((ts - state.start_sec) * state.fps);
      endFrame   = Math.round((te - state.start_sec) * state.fps);
      if (endFrame < 0 || startFrame > last) {
        errEl.textContent = "That range is outside this clip."; return;
      }
    }
    dlg.close();
    startRecording(startFrame, endFrame);
  });

  return dlg;
}

function openExportDialog() {
  if (recording) return;
  if (!state.pose || !els.video.src) { alert("Load a video + pose cache first."); return; }
  if (!exportDialog) exportDialog = buildExportDialog();
  const last = state.n_frames - 1;
  exportDialog.querySelector("#exp-whole-range").textContent =
    `(${fmtClock(state.start_sec)} – ${fmtClock(state.start_sec + last / state.fps)})`;
  exportDialog.querySelector("#exp-error").textContent = "";
  exportDialog.querySelector('input[name="exp-range"][value="whole"]').checked = true;
  exportDialog.querySelector(".export-range").hidden = true;
  exportDialog.showModal();
}

els.exportBtn.addEventListener("click", openExportDialog);

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
    `f${frame}  ·  t=${fmtClock(state.start_sec + frame / state.fps)}`;
}

// Sheet/labeler clock format: MM:SS.sss (zero-padded), matching the
// `start_sec`/`end_sec` strings in the Box Labeled Data sheet (e.g. `04:27.306`).
function fmtClock(seconds) {
  const s = Math.max(0, Number(seconds) || 0);
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${String(m).padStart(2, "0")}:${r.toFixed(3).padStart(6, "0")}`;
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
    `t=${fmtClock(t_video)}${before}`;
}
