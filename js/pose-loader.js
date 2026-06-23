// Loads YOLO-Pose Drive cache: a `<round>.npy` of shape (N, 17, 3) holding
// (x, y, conf) per joint per frame in COCO-17 order, plus a sibling
// `<round>_meta.json` with at least { fps, layout: "coco17" }.
//
// Coords in the .npy are normalised to [0, 1]; we de-normalise to pixels using
// the loaded video's natural dimensions (so the video must be loaded first).
//
// **NaN handling (must match the classifier's training-time view):**
// As of 2026-05-27 the v14j Vision classifier replaced `nan_to_num(nan=0.0)`
// with linear interpolation of NaN runs (`interpolate_nan_runs` in the
// training notebook). This pose-loader mirrors that — NaN positions get
// interpolated between flanking valid frames, with carry-forward /
// carry-back at clip boundaries. The `imputed` mask flags which joints
// were originally NaN so lenses can render them with a magenta indicator
// around their now-interpolated position (skeleton.js handles the
// rendering).
//
// Previous behaviour (nan_to_num→0) is documented in git history for
// the v6/v17j models that were trained against that policy.
//
// Output shape (consumed by the rest of the viewer):
//   { skeleton: Float32Array(n*17*2),  // (frame, joint, xy) row-major
//     conf:     Float32Array(n*17),    // (frame, joint)     row-major
//     imputed:  Uint8Array(n*17),      // 1 = was NaN in cache, now 0,0,0
//     fps, width, height, n_frames, engine: "yolo_pose", source }

const N_JOINTS = 17;

export async function loadPose(files, videoSize) {
  const arr = Array.from(files || []);
  const npy  = arr.find(f => f.name.toLowerCase().endsWith(".npy"));
  const meta = arr.find(f => f.name.toLowerCase().endsWith(".json"));

  if (!npy || !meta) {
    throw new Error(
      "Pick the .npy and its sibling _meta.json together (multi-select)."
    );
  }
  return loadNpy(npy, meta, videoSize);
}

async function loadNpy(npyFile, metaFile, videoSize) {
  const meta = JSON.parse(await metaFile.text());
  const fps = Number(meta.fps);
  // The cache typically includes a pre-buffer of pre_buffer_sec (≈1.5s)
  // BEFORE the official round start, so cache frame 0 lives at
  // `actual_start_sec`, NOT `start_sec`. Some newer caches don't write
  // pre-buffer fields and the round begins right at `start_sec`. Trust
  // actual_start_sec when present; otherwise fall back.
  const start_sec = Number(meta.actual_start_sec ?? meta.start_sec ?? 0);
  const round_start_sec = Number(meta.start_sec ?? start_sec);
  const pre_buffer_sec = Math.max(0, round_start_sec - start_sec);
  const layout = meta.layout || "coco17";
  if (layout !== "coco17" && layout !== "blazepose33") {
    throw new Error(`Unsupported layout '${layout}' — only coco17 and blazepose33 are wired up.`);
  }

  const { data, shape, dtype } = parseNpy(await npyFile.arrayBuffer());
  if (dtype !== "<f4") {
    throw new Error(`Expected float32 LE in .npy, got dtype '${dtype}'.`);
  }

  // BlazePose-33 production cache (blazepose_pose_cache/, shape (N,33,8) =
  // [x, y, z, x_world_m, y_world_m, z_world_m, visibility, presence]) is read
  // THROUGH a 33→COCO-17 remap so the rest of the viewer — and the
  // engine-compare lens — sees an ordinary COCO-17 engine. x,y are already
  // image-normalised like coco17; `conf` is BlazePose's per-joint `visibility`
  // (channel 6), its cleanest occlusion signal. The extra 33-joint data (feet,
  // world-3D) rides the schema-aware skeleton_compare lens, not this path.
  const isBlaze = layout === "blazepose33" ||
                  (shape.length === 3 && shape[1] === 33 && shape[2] === 8);
  if (isBlaze) {
    if (shape.length !== 3 || shape[1] !== 33 || shape[2] !== 8) {
      throw new Error(
        `Expected .npy shape (N, 33, 8) for blazepose33 cache, got (${shape.join(", ")}).`
      );
    }
  } else if (shape.length !== 3 || shape[1] !== 17 || shape[2] !== 3) {
    throw new Error(
      `Expected .npy shape (N, 17, 3) for coco17 cache, got (${shape.join(", ")}).`
    );
  }

  // COCO-17 joint j ← BlazePose-33 source index (mouth/hands/feet-extra dropped).
  const BLAZE_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
  const J_IN    = isBlaze ? 33 : 17;
  const CH      = isBlaze ? 8  : 3;
  const CONF_CH = isBlaze ? 6  : 2;   // BlazePose visibility vs coco17 conf
  // Base offset into `data` for the source joint backing COCO joint j at frame f.
  const srcBase = (f, j) => (f * J_IN + (isBlaze ? BLAZE_TO_COCO[j] : j)) * CH;

  const n_frames = shape[0];
  const w = videoSize?.width || meta.width || 1;
  const h = videoSize?.height || meta.height || 1;

  // Detect normalised (0..1) vs pixel coords by probing the first ~20 frames.
  // The cached files are normalised; the check is cheap insurance.
  let maxXY = 0;
  const probeFrames = Math.min(20, n_frames);
  for (let f = 0; f < probeFrames; f++) {
    for (let j = 0; j < N_JOINTS; j++) {
      const b = srcBase(f, j);
      if (data[b]   > maxXY) maxXY = data[b];
      if (data[b+1] > maxXY) maxXY = data[b+1];
    }
  }
  const normalised = maxXY <= 1.5;
  const sx = normalised ? w : 1;
  const sy = normalised ? h : 1;

  // Split (N, 17, 3) → flat skeleton (N*17*2) + flat conf (N*17).
  //
  // **NaN handling (must match the classifier's training-time pipeline):**
  // As of 2026-05-27, the v14j Vision classifier replaced `nan_to_num(nan=0.0)`
  // with linear interpolation across NaN runs — see `interpolate_nan_runs`
  // in the 5-class training notebook. The viewer must mirror that to honor
  // the "debug must mirror target" principle (memory:
  // feedback_debug_must_mirror_target.md). NaN runs are interpolated
  // per-(joint, channel) between flanking valid frames; boundary runs
  // carry-forward / carry-back.
  //
  // `imputed[f*17+j]` flags joints whose original cache value was NaN.
  // Lenses use it to draw a magenta indicator around the joint at its
  // INTERPOLATED position (no longer at (0,0)), so the user can see
  // exactly which joints are inferred vs real-detected.
  const skeleton = new Float32Array(n_frames * N_JOINTS * 2);
  const conf = new Float32Array(n_frames * N_JOINTS);
  const imputed = new Uint8Array(n_frames * N_JOINTS);
  let n_imputed = 0;

  // Pass 1: copy raw data into flat arrays, preserving NaN. Track which
  // (frame, joint) pairs were NaN so the imputed mask is built from the
  // RAW data (we'll interpolate next, but we want to remember what was
  // originally missing).
  for (let f = 0; f < n_frames; f++) {
    for (let j = 0; j < N_JOINTS; j++) {
      const base = srcBase(f, j);
      const rx = data[base + 0];
      const ry = data[base + 1];
      const rc = data[base + CONF_CH];
      // `x !== x` is the standard NaN check that works on raw Float32Array
      // reads without a function call. Faster than Number.isNaN inside this
      // hot loop.
      const wasNan = (rx !== rx) || (ry !== ry) || (rc !== rc);
      if (wasNan) {
        imputed[f * N_JOINTS + j] = 1;
        n_imputed++;
      }
      skeleton[(f * N_JOINTS + j) * 2 + 0] = rx * sx;
      skeleton[(f * N_JOINTS + j) * 2 + 1] = ry * sy;
      conf[f * N_JOINTS + j] = rc;
    }
  }

  // Pass 2: interpolate NaN runs in each (joint, channel) trajectory.
  // Linear interp between flanking valid frames, nearest-valid carry at
  // clip boundaries. Same algorithm as `interpolate_nan_runs` in the
  // training notebook (np.interp with clamp-to-endpoint).
  interpolateNanTrajectories(skeleton, n_frames, N_JOINTS, 2);
  interpolateNanTrajectories(conf,     n_frames, N_JOINTS, 1);

  return {
    skeleton, conf, imputed, n_imputed, fps,
    start_sec,           // video time of cache frame 0 (incl. pre-buffer)
    round_start_sec,     // video time the official round begins
    pre_buffer_sec,      // round_start_sec - start_sec, for the meta line
    width: w, height: h, n_frames,
    engine: "yolo_pose",
    source: npyFile.name,
    normalised,
    meta,                // parsed _meta.json — lenses read extras (e.g. wrist_run)
  };
}

// Linear-interpolate NaN runs in a flat (n_frames × n_joints × n_channels)
// array. Operates per (joint, channel) trajectory: walk the time axis,
// replace each NaN frame with linear interpolation between the nearest
// valid frames on either side, or carry-forward / carry-back at clip
// boundaries. If the whole trajectory is NaN, fill with zeros (shouldn't
// happen on Vision in practice).
//
// Mirrors the training notebook's `interpolate_nan_runs` (which uses
// np.interp's clamp-to-endpoint behaviour for out-of-range x). The
// `arr[idx] !== arr[idx]` check is the standard NaN test that works
// without a function call inside the hot loop.
function interpolateNanTrajectories(arr, n_frames, n_joints, n_channels) {
  const stride = n_joints * n_channels;
  // Reusable buffer for valid frame indices on this trajectory.
  const valid = new Int32Array(n_frames);
  for (let j = 0; j < n_joints; j++) {
    for (let c = 0; c < n_channels; c++) {
      const off = j * n_channels + c;
      // First sweep: collect valid frame indices for this trajectory.
      let v = 0;
      for (let f = 0; f < n_frames; f++) {
        const x = arr[f * stride + off];
        if (x === x) valid[v++] = f;   // x === x is false for NaN
      }
      if (v === n_frames) continue;     // no NaN — nothing to do
      if (v === 0) {
        // All-NaN trajectory: zero it. Safe fallback so downstream maths
        // doesn't propagate NaN.
        for (let f = 0; f < n_frames; f++) arr[f * stride + off] = 0;
        continue;
      }
      // Second sweep: walk frames, interpolate NaN against the precomputed
      // valid index list. `vi` is a moving pointer into `valid`.
      let vi = 0;
      for (let f = 0; f < n_frames; f++) {
        const idx = f * stride + off;
        if (arr[idx] === arr[idx]) continue;   // not NaN
        // Advance vi to the first valid index > f.
        while (vi < v && valid[vi] <= f) vi++;
        const nextF = vi < v ? valid[vi] : -1;
        const prevF = vi > 0 ? valid[vi - 1] : -1;
        if (prevF >= 0 && nextF >= 0) {
          const t = (f - prevF) / (nextF - prevF);
          const pv = arr[prevF * stride + off];
          const nv = arr[nextF * stride + off];
          arr[idx] = (1 - t) * pv + t * nv;
        } else if (prevF >= 0) {
          arr[idx] = arr[prevF * stride + off];
        } else {
          arr[idx] = arr[nextF * stride + off];
        }
      }
    }
  }
}

// ── .npy parser ────────────────────────────────────────────────────────────
// Implements numpy's NPY format v1/v2/v3 (subset we need).
// https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
function parseNpy(buffer) {
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  // Magic: \x93NUMPY
  if (bytes[0] !== 0x93 ||
      String.fromCharCode(...bytes.slice(1, 6)) !== "NUMPY") {
    throw new Error("Not a .npy file (magic mismatch)");
  }
  const major = bytes[6];
  let headerLen, dataOffset;
  if (major === 1) {
    headerLen = view.getUint16(8, true);
    dataOffset = 10 + headerLen;
  } else if (major === 2 || major === 3) {
    headerLen = view.getUint32(8, true);
    dataOffset = 12 + headerLen;
  } else {
    throw new Error(`Unsupported .npy major version ${major}`);
  }

  const headerStr = new TextDecoder("ascii")
    .decode(bytes.slice(major === 1 ? 10 : 12, dataOffset))
    .trim().replace(/,\s*}$/, "}");

  const dtype = (headerStr.match(/'descr'\s*:\s*'([^']+)'/) || [])[1];
  const fortran = /'fortran_order'\s*:\s*True/.test(headerStr);
  const shapeStr = (headerStr.match(/'shape'\s*:\s*\(([^)]*)\)/) || [])[1] || "";
  const shape = shapeStr
    .split(",").map(s => s.trim()).filter(Boolean).map(Number);

  if (fortran) throw new Error("Fortran-order .npy not supported.");

  const total = shape.reduce((a, b) => a * b, 1);
  const slice = buffer.slice(dataOffset);

  let data;
  switch (dtype) {
    case "<f4": data = new Float32Array(slice, 0, total); break;
    case "<f8": data = new Float64Array(slice, 0, total); break;
    case "<i4": data = new Int32Array(slice, 0, total); break;
    default:
      throw new Error(`Unsupported .npy dtype '${dtype}'.`);
  }

  return { data, shape, dtype };
}

// Load a per-frame PTS sidecar (<base>_<engine>_r<N>_pts.npy → (N,) float32,
// seconds) for deterministic cross-engine time alignment. Returns a
// Float32Array, or null if absent / wrong dtype.
export async function loadPtsArray(file) {
  if (!file) return null;
  try {
    const { data, dtype } = parseNpy(await file.arrayBuffer());
    return dtype === "<f4" ? data : null;
  } catch (e) {
    console.warn("pts sidecar load failed:", e.message);
    return null;
  }
}

export function jointXY(pose, frame, joint) {
  const i = (frame * N_JOINTS + joint) * 2;
  return [pose.skeleton[i], pose.skeleton[i + 1]];
}

export function jointConf(pose, frame, joint) {
  return pose.conf[frame * N_JOINTS + joint];
}

// Glove-wrist sidecar loader. Cache shape is (N, 2, 3) — [L_wrist, R_wrist]
// each storing (x_norm, y_norm, conf). NaN x/y means the glove model didn't
// produce that wrist for this frame. Frame timing exactly matches the matching
// vision cache (same actual_start_sec, fps, n_frames) so the index lines up
// 1:1 with the vision pose object the rule attaches it to.
//
// Returns:
//   { wrists: Float32Array(n*2*2),   // [f, side, xy] row-major, pixel coords
//     conf:   Float32Array(n*2),     // [f, side]
//     fps, start_sec, n_frames, width, height }
export async function loadGloveWrists(npyFile, metaFile, videoSize) {
  const meta = JSON.parse(await metaFile.text());
  const fps = Number(meta.fps);
  const start_sec = Number(meta.actual_start_sec ?? meta.start_sec ?? 0);

  const { data, shape, dtype } = parseNpy(await npyFile.arrayBuffer());
  if (shape.length !== 3 || shape[1] !== 2 || shape[2] !== 3) {
    throw new Error(
      `Expected glove cache shape (N, 2, 3), got (${shape.join(", ")}).`
    );
  }
  if (dtype !== "<f4") {
    throw new Error(`Expected float32 LE in glove .npy, got dtype '${dtype}'.`);
  }
  const n_frames = shape[0];
  const w = videoSize?.width  || meta.width  || 1;
  const h = videoSize?.height || meta.height || 1;

  const wrists = new Float32Array(n_frames * 2 * 2);
  const conf   = new Float32Array(n_frames * 2);
  for (let f = 0; f < n_frames; f++) {
    for (let s = 0; s < 2; s++) {
      const base = (f * 2 + s) * 3;
      wrists[(f * 2 + s) * 2 + 0] = data[base + 0] * w;
      wrists[(f * 2 + s) * 2 + 1] = data[base + 1] * h;
      conf[f * 2 + s] = data[base + 2];
    }
  }
  return { wrists, conf, fps, start_sec, n_frames, width: w, height: h };
}

// Lookup helpers for the glove cache. side: 0=L, 1=R.
export function gloveXY(g, frame, side) {
  const i = (frame * 2 + side) * 2;
  return [g.wrists[i], g.wrists[i + 1]];
}
export function gloveConf(g, frame, side) {
  return g.conf[frame * 2 + side];
}
