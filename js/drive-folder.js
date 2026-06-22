// Persistent Drive-folder integration via the File System Access API.
//
// Idea: you already have Drive for Desktop syncing your boxing folder to a
// real local path (`~/Library/CloudStorage/...` on Mac, `G:\My Drive\...` on
// Windows). One-time we ask the browser for a directory handle to that
// folder; we stash it in IndexedDB; on every subsequent visit we restore the
// handle, re-request permission (browser typically auto-grants for handles
// already granted), and walk the folder to populate the cache index plus a
// video catalogue — so neither the cache files nor the source videos have
// to be picked manually.
//
// All file access is read-only. Files never leave the user's machine; the
// "Drive" part of the name is just where they happen to live.
//
// Chrome and Edge support this API today. On Safari/Firefox, isSupported()
// returns false and the existing <input type=file> upload pickers remain
// the only path — those keep working exactly as before.

const DB_NAME = "cornerman-viewer";
const DB_VERSION = 1;
const STORE_NAME = "handles";
const HANDLE_KEY = "drive-folder";

const VIDEO_EXTENSIONS = /\.(mp4|mov|m4v|webm)$/i;
// Same pattern that viewer.js's <input> picker uses — kept in sync so a
// Drive-folder walk indexes exactly the same files a manual folder pick would.
// (GT labels are pulled live from the Sheet at load time; no sidecar.)
//
// Six engine tags now: `yolo` and `vision` are the COCO-17 2D engines;
// `vision3d` is the experimental Apple Vision 3D engine producing
// (N, 17, 4) body-frame metres + an optional `_cam.npy` sidecar with the
// per-frame cameraOriginMatrix; `glove` is a wrist-only sidecar — shape
// (N, 2, 3) holding (x, y, conf) for [L_wrist, R_wrist] from the trained
// glove detector, time-aligned with the matching vision cache, used by the
// wrist-swap lens to replace Vision's wrists at render time.
// `vision_combined` is a synthetic tag we apply to `_vision_` files that
// live inside a `pose_cache_v*/` folder — the production-shaped cache built
// by glove_wrist_cache_build.ipynb §8 with glove wrists baked into joints 9/10.
// `vision_glove` is the v6 cache (pose_cache_v6/) — Apple Vision skeleton
// with glove-detector wrists already baked in for gloved rounds (or pure
// Vision for ungloved rounds), shape (N, 17, 3). The meta sidecar carries
// the per-round glove-presence decision and engine versions so the round_v6
// lens can render exactly what the iOS app sees.
// The 3D / glove files share the same `_r{N}` and `_meta.json` conventions
// so the slot machinery below is reused as-is.
//
// Engine alternation MUST list longer tokens first — `vision_glove` would
// otherwise be eaten as `(vision)` with base ending in `_vision`, then
// `_glove_r…` would fail to match.
// Bake-off engines (movenet/yolo11/blazepose) are COCO-17 like yolo/rtmpose.
// `yolo11` MUST precede `yolo` (longer-first) so `_yolo11_` isn't eaten as
// `_yolo`. `blazepose33` is intentionally absent — that 33-joint cache is for
// the schema-aware skeleton_compare lens, not the COCO-17 loaders.
const CACHE_FILE_RE =
  /^(.+?)_(vision_glove|vision3d|vision|yolo11|yolo|rtmpose|movenet|blazepose|glove)_r(\d+)(_meta|_punches|_cam|_proj|_pts)?\.(npy|json)$/;
// Punch-classifier predictions dump (one file per training run, schema in
// js/rules/punch_classifier.js). Walker captures these and exposes them as
// `state.predictionFiles` so the lens auto-loads without a file picker.
const PREDICTIONS_FILE_RE = /^predictions_.*\.json$/i;
const COMBINED_DIR_RE = /^pose_cache_v/i;
function classifyEngine(engine, _parentDirName) {
  // `vision_glove` and `vision_combined` are filename-distinguished now (the
  // v6 builder writes `_vision_glove_` directly), so the parent-folder hint
  // for `vision_combined` only matters for the older pose_cache_v*/ files
  // that still use the bare `_vision_` naming. v6 files match unambiguously.
  if (engine === "vision" && _parentDirName && COMBINED_DIR_RE.test(_parentDirName)) {
    return "vision_combined";
  }
  return engine;
}

// ── IndexedDB plumbing (tiny manual wrapper to avoid pulling in idb-keyval) ──

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function putHandle(handle) {
  const db = await openDb();
  await new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).put(handle, HANDLE_KEY);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
  db.close();
}

async function getHandle() {
  const db = await openDb();
  try {
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const req = tx.objectStore(STORE_NAME).get(HANDLE_KEY);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
  } finally {
    db.close();
  }
}

async function deleteHandle() {
  const db = await openDb();
  await new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).delete(HANDLE_KEY);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
  db.close();
}

// ── Public API ──────────────────────────────────────────────────────────────

export function isSupported() {
  return typeof window !== "undefined" &&
    typeof window.showDirectoryPicker === "function";
}

export async function pickFolder() {
  if (!isSupported()) {
    throw new Error("File System Access API is not available in this browser.");
  }
  const handle = await window.showDirectoryPicker({
    id: "cornerman-cache",
    mode: "read",
    startIn: "documents",
  });
  await putHandle(handle);
  return handle;
}

// Try to restore a previously-granted handle from IndexedDB. Does NOT
// re-prompt the user — that has to be triggered by a user gesture. Returns
// { handle, permission } or null if nothing stored.
export async function tryRestore() {
  if (!isSupported()) return null;
  let handle;
  try { handle = await getHandle(); } catch { return null; }
  if (!handle) return null;
  let permission;
  try {
    permission = await handle.queryPermission({ mode: "read" });
  } catch {
    // Cross-origin or invalidated handle — drop it.
    await deleteHandle();
    return null;
  }
  return { handle, permission };
}

// Requires a user gesture (button click). Browser may auto-grant if the
// handle has been used before in this origin.
export async function requestPermission(handle) {
  try {
    return await handle.requestPermission({ mode: "read" });
  } catch (err) {
    console.warn("Drive folder permission request failed:", err);
    return "denied";
  }
}

export async function forget() {
  await deleteHandle();
}

// Recursively walk the directory. Returns the same `cacheIndex` shape that
// viewer.js builds from <input type=file webkitdirectory> picks — but with
// FileSystemFileHandle values instead of File objects. viewer.js calls
// `toFile()` on values when it's actually time to load.
//
// Also returns a `videos` Map keyed by exact filename → FileSystemFileHandle,
// so the video dropdown can be sourced from the same walk.
export async function walk(rootHandle) {
  const videos = new Map();
  const cacheIndex = new Map();
  // Keyed by filename so the lens can surface every found file if it wants;
  // viewer.js picks one (most-recently-modified) by default.
  const predictions = new Map();

  async function visit(dirHandle) {
    for await (const entry of dirHandle.values()) {
      if (entry.kind === "directory") {
        await visit(entry);
        continue;
      }
      const name = entry.name;
      if (VIDEO_EXTENSIONS.test(name)) {
        videos.set(name, entry);
        continue;
      }
      if (PREDICTIONS_FILE_RE.test(name)) {
        predictions.set(name, entry);
        continue;
      }
      if (name.endsWith(".bak.npy")) continue;
      const m = name.match(CACHE_FILE_RE);
      if (!m) continue;
      const [, base, rawEngine, roundStr, suffix, ext] = m;
      const round = parseInt(roundStr);
      if (ext === "json" && !suffix) continue;
      // Parent-folder hint disambiguates `pose_cache_v5/foo_vision_r0.npy`
      // (combined cache with glove wrists baked in) from `apple_vision_pose_cache/`
      // (raw vision). Same filename pattern; only the folder differs.
      const engine = classifyEngine(rawEngine, dirHandle.name);
      // The 3D loader expects `npy` (data) and `_cam.npy` (camera matrices)
      // to live in the same engine slot. The cam .npy isn't required —
      // viewer is happy without it — but if present, stash it as `cam`.
      // `_punches` only applies to 2D engines.

      if (!cacheIndex.has(base)) cacheIndex.set(base, new Map());
      const rounds = cacheIndex.get(base);
      if (!rounds.has(round)) rounds.set(round, {});
      const slot = rounds.get(round);
      if (!slot[engine]) slot[engine] = {};
      const engineSlot = slot[engine];
      if (ext === "npy" && suffix === "_cam")        engineSlot.cam = entry;
      else if (ext === "npy" && suffix === "_proj")  engineSlot.proj = entry;
      else if (ext === "npy" && suffix === "_pts")   engineSlot.pts = entry;
      else if (ext === "npy")                        engineSlot.npy = entry;
      else if (suffix === "_meta")                   engineSlot.meta = entry;
      else if (suffix === "_punches")                engineSlot.punches = entry;
    }
  }

  await visit(rootHandle);

  // Same completeness filter the manual picker applies: drop engines that
  // don't have both .npy + _meta.json; drop rounds that lost ALL skeleton
  // engines. The 3D engine is treated the same way — it needs its own
  // .npy + meta. The `glove` sidecar follows the same npy+meta requirement,
  // but does NOT count as a skeleton engine — it's only useful when paired
  // with vision (it replaces Vision's wrists at render time).
  // `vision_combined` and `vision_glove` ARE skeleton engines — both are
  // full COCO-17 poses, just with wrists 9/10 replaced by glove predictions
  // on confident frames (vision_combined = legacy pose_cache_v5/, vision_glove
  // = pose_cache_v6/).
  for (const [base, rounds] of cacheIndex) {
    for (const [round, slot] of rounds) {
      for (const eng of ["yolo", "vision", "vision3d", "rtmpose", "movenet", "yolo11", "blazepose", "glove", "vision_combined", "vision_glove"]) {
        if (slot[eng] && (!slot[eng].npy || !slot[eng].meta)) delete slot[eng];
      }
      // glove alone is useless — needs a skeleton engine to overlay on
      if (!slot.yolo && !slot.vision && !slot.vision3d && !slot.rtmpose && !slot.movenet
          && !slot.yolo11 && !slot.blazepose && !slot.vision_combined && !slot.vision_glove) {
        rounds.delete(round);
      }
    }
    if (rounds.size === 0) cacheIndex.delete(base);
  }

  return { videos, cacheIndex, predictions };
}

// Materialize a slot value into a File. Lets viewer.js stay agnostic about
// whether a slot came from a manual picker (already File) or a Drive walk
// (FileSystemFileHandle).
export async function toFile(value) {
  if (value == null) return null;
  if (value instanceof File) return value;
  if (typeof value.getFile === "function") return value.getFile();
  throw new Error("Unknown slot value: " + value);
}
