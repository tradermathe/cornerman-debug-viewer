// Firebase Web SDK wrapper for the debug viewer.
//
// Loads sessions recorded by the Cornerman iOS app: pulls the video,
// skeleton JSON, and on-device analysis sidecar from Firebase Storage
// for a given (sessionId, roundNumber).
//
// All assets live behind RTDB rules that require `auth != null`, so we
// sign in anonymously at module load. The web Firebase config is public
// by design (browser can't keep secrets) — same values the iOS app
// embeds in src/services/firebase.ts.

import { initializeApp } from "https://www.gstatic.com/firebasejs/12.11.0/firebase-app.js";
import { getAuth, signInAnonymously } from "https://www.gstatic.com/firebasejs/12.11.0/firebase-auth.js";
import { getStorage, ref as storageRef, getBlob } from "https://www.gstatic.com/firebasejs/12.11.0/firebase-storage.js";
import { getDatabase, ref as dbRef, get, query, orderByKey, limitToLast } from "https://www.gstatic.com/firebasejs/12.11.0/firebase-database.js";

const firebaseConfig = {
  apiKey: "AIzaSyC6B7rqq7ACJllABmPD8K6dEmYl2B3X3uY",
  authDomain: "mycorner-bee6a.firebaseapp.com",
  databaseURL: "https://mycorner-bee6a-default-rtdb.firebaseio.com",
  projectId: "mycorner-bee6a",
  storageBucket: "mycorner-bee6a.firebasestorage.app",
  messagingSenderId: "1015897745773",
  appId: "1:1015897745773:web:7203acf289bef8b2a39264",
};

let app = null;
let auth = null;
let storage = null;
let db = null;
let authPromise = null;

// Idempotent — calling twice is fine.
export function init() {
  if (app) return authPromise;
  app = initializeApp(firebaseConfig);
  auth = getAuth(app);
  storage = getStorage(app);
  db = getDatabase(app);
  authPromise = signInAnonymously(auth).then(
    () => { console.log("[firebase-source] anon sign-in OK, uid:", auth.currentUser?.uid); },
    (err) => {
      console.error("[firebase-source] anon sign-in failed:", err);
      throw err;
    }
  );
  return authPromise;
}

// List the N most recent sessions (by sessionId, which is timestamp-prefixed).
// Returns an array of { sessionId, meta, rounds } sorted newest-first.
export async function listRecentSessions(limit = 20) {
  await init();
  const snap = await get(query(dbRef(db, "sessions"), orderByKey(), limitToLast(limit)));
  const out = [];
  snap.forEach((sessionSnap) => {
    const sessionId = sessionSnap.key;
    const data = sessionSnap.val() || {};
    const rounds = data.rounds ? Object.keys(data.rounds).sort((a, b) => Number(a) - Number(b)) : [];
    out.push({ sessionId, meta: data.meta || {}, rounds });
  });
  out.reverse(); // newest first
  return out;
}

// Read the per-round RTDB metadata. Used to verify a round has finished
// uploading before we try to fetch the blobs.
export async function getRoundMeta(sessionId, roundNumber) {
  await init();
  const snap = await get(dbRef(db, `sessions/${sessionId}/rounds/${roundNumber}`));
  return snap.val();
}

// Fetch all three blobs for a round. Storage paths match the iOS app's
// upload conventions:
//   sessions/{id}/round_{N}.mp4
//   sessions/{id}/round_{N}_skeleton.json
//   sessions/{id}/round_{N}_ondevice_analysis.json
//
// Returns the blobs (caller wraps them in URL.createObjectURL / parses JSON).
// Analysis sidecar may not exist on older sessions — returns null there
// rather than throwing so the viewer can still load video + skeleton.
export async function fetchRoundBlobs(sessionId, roundNumber) {
  await init();
  const base = `sessions/${sessionId}/round_${roundNumber}`;
  const videoPath    = `${base}.mp4`;
  const skeletonPath = `${base}_skeleton.json`;
  const analysisPath = `${base}_ondevice_analysis.json`;

  const [videoBlob, skeletonBlob, analysisBlob] = await Promise.all([
    getBlob(storageRef(storage, videoPath))
      .catch((err) => { throw new Error(`video fetch failed at ${videoPath}: ${err.message}`); }),
    getBlob(storageRef(storage, skeletonPath))
      .catch((err) => { throw new Error(`skeleton fetch failed at ${skeletonPath}: ${err.message}`); }),
    getBlob(storageRef(storage, analysisPath))
      .catch((err) => {
        console.warn(`[firebase-source] no on-device analysis at ${analysisPath}: ${err.message}`);
        return null;
      }),
  ]);

  return {
    videoBlob,
    skeletonBlob,
    analysisBlob, // may be null on pre-port sessions
    paths: { videoPath, skeletonPath, analysisPath },
  };
}
