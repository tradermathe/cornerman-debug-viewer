// Loads Apple Vision 3D pose cache: `<base>_vision3d_r{N}.npy` of shape
// (N, 17, 4) holding (x_m, y_m, z_m, conf) per joint per frame in Apple's
// NATIVE 17-joint order (NOT COCO-17 — see skeleton-3d.js for the order).
// Coordinates are body-frame metres (root-anchored).
//
// Optional siblings:
//   `<base>_vision3d_r{N}_cam.npy`  — (N, 4, 4) cameraOriginMatrix per frame
//   `<base>_vision3d_r{N}_proj.npy` — (N, 17, 2) image-space projection
//                                     via Apple's pointInImage(); normalised
//                                     0..1 with TOP-LEFT origin (same as 2D
//                                     pipeline), so the viewer can overlay
//                                     the 3D-derived skeleton directly on
//                                     the source video.
//
// Output shape consumed by the viewer:
//   { xyz:  Float32Array(N*17*3),   // (frame, joint, xyz) row-major, metres
//     conf: Float32Array(N*17),     // (frame, joint), observation-level
//                                   //   replicated per joint — see scaffold
//     proj: Float32Array(N*17*2) | null,        // image-space, normalised
//     camMatrices: Float32Array(N*16) | null,   // (frame, 4, 4) row-major
//     fps, n_frames, engine: "apple_vision_3d", source }

const N_JOINTS_3D = 17;

export async function loadPose3D({ npy, meta, cam, proj }) {
  if (!npy || !meta) {
    throw new Error("3D loader needs the .npy and the _meta.json together.");
  }
  const m = JSON.parse(await meta.text());
  const fps = Number(m.fps);
  const start_sec = Number(m.actual_start_sec ?? m.start_sec ?? 0);
  const round_start_sec = Number(m.start_sec ?? start_sec);
  const pre_buffer_sec = Math.max(0, round_start_sec - start_sec);
  const layout = m.layout || "apple_vision_3d_17";
  if (layout !== "apple_vision_3d_17") {
    throw new Error(
      `Unsupported 3D layout '${layout}' — expected 'apple_vision_3d_17'.`
    );
  }

  const { data, shape, dtype } = parseNpy(await npy.arrayBuffer());
  if (shape.length !== 3 || shape[1] !== N_JOINTS_3D || shape[2] !== 4) {
    throw new Error(
      `Expected 3D .npy shape (N, 17, 4), got (${shape.join(", ")}).`
    );
  }
  if (dtype !== "<f4") {
    throw new Error(`Expected float32 LE in 3D .npy, got dtype '${dtype}'.`);
  }
  const n_frames = shape[0];

  // Split (N, 17, 4) → flat xyz (N*17*3) + flat conf (N*17).
  const xyz = new Float32Array(n_frames * N_JOINTS_3D * 3);
  const conf = new Float32Array(n_frames * N_JOINTS_3D);
  for (let f = 0; f < n_frames; f++) {
    for (let j = 0; j < N_JOINTS_3D; j++) {
      const base = (f * N_JOINTS_3D + j) * 4;
      xyz[(f * N_JOINTS_3D + j) * 3 + 0] = data[base + 0];
      xyz[(f * N_JOINTS_3D + j) * 3 + 1] = data[base + 1];
      xyz[(f * N_JOINTS_3D + j) * 3 + 2] = data[base + 2];
      conf[f * N_JOINTS_3D + j] = data[base + 3];
    }
  }

  // Optional camera-origin matrices — flat Float32Array of length N*16.
  let camMatrices = null;
  if (cam) {
    const parsed = parseNpy(await cam.arrayBuffer());
    if (parsed.shape.length !== 3 ||
        parsed.shape[0] !== n_frames ||
        parsed.shape[1] !== 4 ||
        parsed.shape[2] !== 4) {
      console.warn(
        `cameraOriginMatrix .npy has shape (${parsed.shape.join(", ")}); ` +
        `expected (${n_frames}, 4, 4) — ignoring.`
      );
    } else if (parsed.dtype !== "<f4") {
      console.warn(`cameraOriginMatrix has dtype '${parsed.dtype}'; expected '<f4' — ignoring.`);
    } else {
      // Already row-major per the Swift CLI's packRowMajor().
      camMatrices = parsed.data;
    }
  }

  // Optional 2D image-space projection — flat Float32Array of length N*17*2.
  // Same coord convention as the 2D cache (normalised 0..1, top-left origin)
  // so the viewer can scale by videoWidth/videoHeight and draw directly.
  let projection = null;
  if (proj) {
    const parsed = parseNpy(await proj.arrayBuffer());
    if (parsed.shape.length !== 3 ||
        parsed.shape[0] !== n_frames ||
        parsed.shape[1] !== N_JOINTS_3D ||
        parsed.shape[2] !== 2) {
      console.warn(
        `_proj.npy has shape (${parsed.shape.join(", ")}); ` +
        `expected (${n_frames}, ${N_JOINTS_3D}, 2) — ignoring.`
      );
    } else if (parsed.dtype !== "<f4") {
      console.warn(`_proj.npy has dtype '${parsed.dtype}'; expected '<f4' — ignoring.`);
    } else {
      projection = parsed.data;
    }
  }

  return {
    xyz, conf, camMatrices,
    projection,
    fps, n_frames,
    start_sec, round_start_sec, pre_buffer_sec,
    engine: "apple_vision_3d",
    source: npy.name,
    metaRaw: m,
  };
}

// ── .npy parser ────────────────────────────────────────────────────────────
// Duplicated from pose-loader.js intentionally — keeps the 3D loader entirely
// self-contained so it can be removed cleanly if the experiment doesn't pan
// out. (See feedback_duplicate_for_experiments.md.)
function parseNpy(buffer) {
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

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
    default: throw new Error(`Unsupported .npy dtype '${dtype}'.`);
  }
  return { data, shape, dtype };
}
