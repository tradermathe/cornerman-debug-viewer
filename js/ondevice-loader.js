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
      // stance_width v5: corrected sep (post foreshortening boost — the
      // values the violation decision actually used) + smoothed |dy|/|dx|.
      // null on v4 sidecars.
      sepRatiosCorrected: r.sep_ratios_corrected_b64 ? base64ToFloat32(r.sep_ratios_corrected_b64) : null,
      axisRatioSmoothed: r.axis_ratio_smoothed_b64 ? base64ToFloat32(r.axis_ratio_smoothed_b64) : null,
      // pivot_rate v1: per-punch orientation samples + ratchet flags
      // (mirrors the Python rule's details.per_punch), the per-frame
      // ankle-orientation arrays the rule scored from (NaN where ankles
      // unusable), and the short-circuit reason. null on other rules.
      skipReason: r.skip_reason ?? null,
      perPunch: Array.isArray(r.per_punch) ? r.per_punch : null,
      orientationAngles: r.orientation_angle_b64 ? base64ToFloat32(r.orientation_angle_b64) : null,
      orientationConfs: r.orientation_conf_b64 ? base64ToFloat32(r.orientation_conf_b64) : null,
    };
  }

  // ST-GCN punch classifier output. Written by iOS sidecar from
  // `runPunchClassifier` (RoundAnalyzer.swift). Shape mirrors the backend's
  // `detect_punches()` so this slots into the existing `loadPunches()`
  // output shape downstream — the viewer can treat firebase-loaded punches
  // and training-cache-loaded punches identically.
  const fps = Number(payload.fps);
  const punchesIn = payload.punches || {};
  const punches = Array.isArray(punchesIn.detections) ? {
    source: "ondevice_stgcn",
    schema_version: 1,
    fps,
    detections: punchesIn.detections.map((d, i) => {
      const start = Number(d.start_time);
      const end = Number(d.end_time);
      return {
        idx: i,
        timestamp: Number(d.timestamp ?? (start + end) / 2),
        start_time: start,
        end_time: end,
        start_frame: Math.round(start * fps),
        end_frame: Math.round(end * fps),
        hand: d.hand || "?",                  // "lead" | "rear"
        punch_type: d.punch_type || "?",
        category: d.category || null,         // "jab" | "cross" | "power"
        n_frames: Number(d.n_frames) || 0,
        // Inline axiality from the on-device AxialityScorer (same metric as the
        // trained model's predAxiality: 0 = side-on, 1 = down the camera axis).
        // The geometric lenses' axiality gate falls back to this when there's
        // no punch_uuid to join the model by (on-device rounds have none).
        axiality: d.axiality != null ? Number(d.axiality) : undefined,
      };
    }),
    total_punches: Number(punchesIn.total_punches ?? 0),
    punches_per_minute: Number(punchesIn.punches_per_minute ?? 0),
    breakdown: punchesIn.punch_breakdown || {},
    breakdown_detailed: punchesIn.punch_breakdown_detailed || {},
  } : null;

  return {
    n_frames: nFrames,
    fps,
    orientation,        // deprecated LogReg — kept for comparison
    ankleOrientation,   // trusted ankle+correction (may be null on old sessions)
    rules,
    punches,            // ST-GCN detections (null when sidecar has no punches block)
    raw: payload, // for debugging
  };
}
