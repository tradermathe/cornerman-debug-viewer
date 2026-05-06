// Loads YOLO-Pose Drive cache: a `<round>.npy` of shape (N, 17, 3) holding
// (x, y, conf) per joint per frame in COCO-17 order, plus a sibling
// `<round>_meta.json` with at least { fps, layout: "coco17" }.
//
// Coords in the .npy are normalised to [0, 1]; we de-normalise to pixels using
// the loaded video's natural dimensions (so the video must be loaded first).
//
// Output shape (consumed by the rest of the viewer):
//   { skeleton: Float32Array(n*17*2),  // (frame, joint, xy) row-major
//     conf:     Float32Array(n*17),    // (frame, joint)     row-major
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
  // start_sec is the round's offset inside the source video — without it
  // the overlay plays at t=0 of the video while the skeleton is for some
  // later slice, and they go out of sync.
  const start_sec = Number(meta.start_sec ?? meta.actual_start_sec ?? 0);
  const layout = meta.layout || "coco17";
  if (layout !== "coco17") {
    throw new Error(`Unsupported layout '${layout}' — only coco17 is wired up.`);
  }

  const { data, shape, dtype } = parseNpy(await npyFile.arrayBuffer());
  if (shape.length !== 3 || shape[1] !== 17 || shape[2] !== 3) {
    throw new Error(
      `Expected .npy shape (N, 17, 3) for coco17 cache, got (${shape.join(", ")}).`
    );
  }
  if (dtype !== "<f4") {
    throw new Error(`Expected float32 LE in .npy, got dtype '${dtype}'.`);
  }

  const n_frames = shape[0];
  const w = videoSize?.width || meta.width || 1;
  const h = videoSize?.height || meta.height || 1;

  // Detect normalised (0..1) vs pixel coords by probing the first ~20 frames.
  // The cached files are normalised; the check is cheap insurance.
  let maxXY = 0;
  const probe = Math.min(20 * 17 * 3, data.length);
  for (let i = 0; i < probe; i += 3) {
    if (data[i]   > maxXY) maxXY = data[i];
    if (data[i+1] > maxXY) maxXY = data[i+1];
  }
  const normalised = maxXY <= 1.5;
  const sx = normalised ? w : 1;
  const sy = normalised ? h : 1;

  // Split (N, 17, 3) → flat skeleton (N*17*2) + flat conf (N*17).
  const skeleton = new Float32Array(n_frames * N_JOINTS * 2);
  const conf = new Float32Array(n_frames * N_JOINTS);
  for (let f = 0; f < n_frames; f++) {
    for (let j = 0; j < N_JOINTS; j++) {
      const base = (f * 17 + j) * 3;
      skeleton[(f * N_JOINTS + j) * 2 + 0] = data[base + 0] * sx;
      skeleton[(f * N_JOINTS + j) * 2 + 1] = data[base + 1] * sy;
      conf[f * N_JOINTS + j] = data[base + 2];
    }
  }

  return {
    skeleton, conf, fps, start_sec,
    width: w, height: h, n_frames,
    engine: "yolo_pose",
    source: npyFile.name,
    normalised,
  };
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

export function jointXY(pose, frame, joint) {
  const i = (frame * N_JOINTS + joint) * 2;
  return [pose.skeleton[i], pose.skeleton[i + 1]];
}

export function jointConf(pose, frame, joint) {
  return pose.conf[frame * N_JOINTS + joint];
}
