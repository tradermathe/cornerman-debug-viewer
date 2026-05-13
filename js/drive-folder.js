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
const CACHE_FILE_RE =
  /^(.+?)_(yolo|vision)_r(\d+)(_meta|_punches|_labels)?\.(npy|json)$/;

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
      if (name.endsWith(".bak.npy")) continue;
      const m = name.match(CACHE_FILE_RE);
      if (!m) continue;
      const [, base, engine, roundStr, suffix, ext] = m;
      const round = parseInt(roundStr);
      if (ext === "json" && !suffix) continue;

      if (!cacheIndex.has(base)) cacheIndex.set(base, new Map());
      const rounds = cacheIndex.get(base);
      if (!rounds.has(round)) rounds.set(round, {});
      const slot = rounds.get(round);
      if (!slot[engine]) slot[engine] = {};
      const engineSlot = slot[engine];
      if (ext === "npy")              engineSlot.npy = entry;
      else if (suffix === "_meta")    engineSlot.meta = entry;
      else if (suffix === "_punches") engineSlot.punches = entry;
      else if (suffix === "_labels")  engineSlot.labels = entry;
    }
  }

  await visit(rootHandle);

  // Same completeness filter the manual picker applies: drop engines that
  // don't have both .npy + _meta.json; drop rounds that lost both engines.
  for (const [base, rounds] of cacheIndex) {
    for (const [round, slot] of rounds) {
      for (const eng of ["yolo", "vision"]) {
        if (slot[eng] && (!slot[eng].npy || !slot[eng].meta)) delete slot[eng];
      }
      if (!slot.yolo && !slot.vision) rounds.delete(round);
    }
    if (rounds.size === 0) cacheIndex.delete(base);
  }

  return { videos, cacheIndex };
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
