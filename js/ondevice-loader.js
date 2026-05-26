// Parsers for the iOS app's on-device data formats.
//
// Two formats:
//   • Skeleton JSON — written by extractPose / stopLiveCapture. Same
//     base64 wire format the backend's process_queue.py decodes.
//   • On-device analysis sidecar — written by analyzeAndUploadResults.
//     Schema documented in OrientationClassifier comments + RoundAnalyzer.
//
// Both formats use base64-encoded little-endian binary blobs for the
// per-frame arrays so the JSON stays compact. Browsers can decode base64
// via atob() and reinterpret the bytes as Float32Array / Uint8Array.

const N_JOINTS = 17;

// Decode a base64-encoded LE float32 array.
function base64ToFloat32(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

// Decode a base64-encoded uint8 array — used for boolean masks (0 / 1 per frame).
function base64ToUint8(b64) {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
}

// Parse the skeleton JSON the iOS app uploads. Returns an object shaped
// the same way the existing pose-loader output is shaped, so the viewer's
// rendering code (drawSkeleton, etc.) just works.
//
// `videoSize` may be omitted — the JSON already carries width/height in
// pixels (top-left origin) so we don't need to renormalise.
export async function loadOnDeviceSkeleton(jsonBlob) {
  const text = await jsonBlob.text();
  const payload = JSON.parse(text);

  if (payload.engine !== "apple_vision_2d") {
    console.warn(
      `[ondevice-loader] unexpected skeleton engine "${payload.engine}", continuing anyway`
    );
  }

  const skeleton = base64ToFloat32(payload.skeleton_b64);
  const conf = base64ToFloat32(payload.conf_b64);
  const nFrames = Number(payload.n_frames);

  const expSkel = nFrames * N_JOINTS * 2;
  const expConf = nFrames * N_JOINTS;
  if (skeleton.length !== expSkel) {
    throw new Error(
      `skeleton length ${skeleton.length} != expected ${expSkel} for n_frames=${nFrames}`
    );
  }
  if (conf.length !== expConf) {
    throw new Error(
      `conf length ${conf.length} != expected ${expConf} for n_frames=${nFrames}`
    );
  }

  return {
    skeleton, conf,
    fps: Number(payload.fps),
    width: Number(payload.width),
    height: Number(payload.height),
    n_frames: nFrames,
    detected_frames: Number(payload.detected_frames ?? nFrames),
    engine: payload.engine || "apple_vision_2d",
    source: "firebase (on-device)",
    normalised: false,           // iOS writes pixel coords, not [0,1]
    start_sec: 0,
    round_start_sec: 0,
    pre_buffer_sec: 0,
    meta: payload,               // keep the raw payload for any lens that needs extras
  };
}

// Parse the on-device analysis sidecar. Returns an object the new
// `ondevice_lens.js` consumes directly:
//   {
//     n_frames, fps,
//     orientation: { angles, confidences, validFrames },
//     rules: {
//       stance_width: {
//         ruleId, version, severity, violationRatio, coachCue,
//         validFrames, violationFrames, clips, extras,
//         validMask, violationMask, sepRatios
//       }
//     }
//   }
// All per-frame arrays are typed (Float32Array or Uint8Array).
export async function loadOnDeviceAnalysis(jsonBlob) {
  const text = await jsonBlob.text();
  const payload = JSON.parse(text);

  if (payload.engine !== "ondevice_v1") {
    console.warn(
      `[ondevice-loader] unexpected analysis engine "${payload.engine}"`
    );
  }
  const nFrames = Number(payload.n_frames);

  const orient = payload.orientation || {};
  const orientation = {
    version: orient.version || "v1",
    deprecated: orient.deprecated === true,
    validFrames: Number(orient.valid_frames ?? 0),
    angles: orient.angles_b64 ? base64ToFloat32(orient.angles_b64) : new Float32Array(nFrames),
    confidences: orient.confidences_b64 ? base64ToFloat32(orient.confidences_b64) : new Float32Array(nFrames),
  };

  // Trusted ankle-direction + per-stance fit orientation. Always present
  // on sidecars written from 2026-05-26 onwards; null on older sessions.
  const ank = payload.ankle_orientation || null;
  const ankleOrientation = ank ? {
    version: ank.version || "v1",
    stance: ank.stance || "orthodox",
    validFrames: Number(ank.valid_frames ?? 0),
    angles: ank.angles_b64 ? base64ToFloat32(ank.angles_b64) : new Float32Array(nFrames),
    confidences: ank.confidences_b64 ? base64ToFloat32(ank.confidences_b64) : new Float32Array(nFrames),
  } : null;

  const rulesIn = payload.rules || {};
  const rules = {};
  for (const [ruleId, r] of Object.entries(rulesIn)) {
    rules[ruleId] = {
      ruleId,
      version: r.version || "unknown",
      severity: r.severity || "none",
      violationRatio: Number(r.violation_ratio ?? 0),
      coachCue: r.coach_cue || "",
      validFrames: Number(r.valid_frames ?? 0),
      violationFrames: Number(r.violation_frames ?? 0),
      clips: Array.isArray(r.clips) ? r.clips : [],
      extras: r.extras || {},
      validMask: r.valid_mask_b64 ? base64ToUint8(r.valid_mask_b64) : null,
      violationMask: r.violation_mask_b64 ? base64ToUint8(r.violation_mask_b64) : null,
      sepRatios: r.sep_ratios_b64 ? base64ToFloat32(r.sep_ratios_b64) : null,
    };
  }

  return {
    n_frames: nFrames,
    fps: Number(payload.fps),
    orientation,        // deprecated LogReg — kept for comparison
    ankleOrientation,   // trusted ankle+correction (may be null on old sessions)
    rules,
    raw: payload, // for debugging
  };
}
