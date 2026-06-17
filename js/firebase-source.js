// Firebase Web SDK wrapper for the debug viewer.
//
// Loads sessions recorded by the Cornerman iOS app: pulls the video,
// skeleton JSON, and on-device analysis sidecar from Firebase Storage
// for a given (sessionId, roundNumber).
//
// Session media + RTDB are owner-scoped (storage.rules / database.rules.json
// in cornerman-ios check ownerUid against the caller's uid). Anonymous sign-in
// no longer works — its fresh uid never matches the device's real account. So
// the viewer signs in with Google: sign in with the SAME account used on the
// phone to read your own sessions, or the debug-admin account (matheee.wieme,
// uid G5ZEk…) granted read-all in the deployed rules. The web Firebase config
// is public by design — same values the iOS app embeds in firebase.ts.

import { initializeApp } from "https://www.gstatic.com/firebasejs/12.11.0/firebase-app.js";
import { getAuth, GoogleAuthProvider, signInWithPopup, signInWithRedirect, getRedirectResult, signOut, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/12.11.0/firebase-auth.js";
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

// Idempotent — calling twice is fine. Initialises the SDK but does NOT sign
// in; reads require an explicit Google sign-in (see signIn()).
export function init() {
  if (app) return;
  app = initializeApp(firebaseConfig);
  auth = getAuth(app);
  storage = getStorage(app);
  db = getDatabase(app);
}

// Google sign-in. Try the popup first (no page reload); fall back to a
// full-page redirect when the popup is blocked — common in Safari and any
// browser with a popup blocker, since Firebase's popup does async work that
// can lose the click gesture. The redirect resolves on return via
// completeRedirectSignIn(), called once at startup.
export async function signIn() {
  init();
  const provider = new GoogleAuthProvider();
  try {
    const cred = await signInWithPopup(auth, provider);
    console.log("[firebase-source] signed in (popup):", cred.user.email, cred.user.uid);
    return cred.user;
  } catch (err) {
    const fallback = [
      "auth/popup-blocked",
      "auth/popup-closed-by-user",
      "auth/cancelled-popup-request",
      "auth/operation-not-supported-in-this-environment",
    ];
    if (fallback.includes(err?.code)) {
      console.warn(`[firebase-source] popup unavailable (${err.code}), redirecting…`);
      await signInWithRedirect(auth, provider); // navigates away; completes on return
      return null;
    }
    throw err;
  }
}

// Complete a redirect-based sign-in if the page just came back from one.
// No-op on a normal load. Call once at startup.
export async function completeRedirectSignIn() {
  init();
  try {
    const res = await getRedirectResult(auth);
    if (res?.user) {
      console.log("[firebase-source] signed in (redirect):", res.user.email, res.user.uid);
    }
    return res?.user ?? null;
  } catch (err) {
    console.error("[firebase-source] redirect sign-in failed:", err);
    return null;
  }
}

export async function signOutViewer() {
  init();
  await signOut(auth);
}

export function currentUser() {
  return auth?.currentUser ?? null;
}

// Notify the caller on every auth-state change (and once on subscribe).
export function onAuthChange(cb) {
  init();
  return onAuthStateChanged(auth, cb);
}

// Throw a clear, actionable error if the caller tries to read before signing in.
function requireAuth() {
  init();
  if (!auth.currentUser) {
    throw new Error("not signed in — click “Sign in with Google” first");
  }
}

// List the N most recent sessions (by sessionId, which is timestamp-prefixed).
// Returns an array of { sessionId, meta, rounds } sorted newest-first.
export async function listRecentSessions(limit = 20) {
  requireAuth();
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
  requireAuth();
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
  requireAuth();
  const base = `sessions/${sessionId}/round_${roundNumber}`;
  const videoPath    = `${base}.mp4`;
  const skeletonPath = `${base}_skeleton.json`;
  const analysisPath = `${base}_ondevice_analysis.json`;

  const [videoBlob, skeletonBlob, analysisBlob] = await Promise.all([
    // Video may not have finished uploading (it's a large fire-and-forget
    // upload) — non-fatal so a skeleton+analysis-only round still loads.
    getBlob(storageRef(storage, videoPath))
      .catch((err) => {
        console.warn(`[firebase-source] no video at ${videoPath}: ${err.message}`);
        return null;
      }),
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
