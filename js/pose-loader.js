// Loads Apple Vision pose JSON in either of the two formats the project produces:
//  1. Parity / Mac-side format — straight-array `skeleton_frames` and `confidences`
//     (produced by apple_vision_pose CLI; see Mathe_Test_vision.json).
//  2. Production iOS format — base64 float32 LE in `skeleton_b64` / `conf_b64`
//     (uploaded by the cornerman-vision-pose Expo module; see CLAUDE.md).
//
// Output shape is the same regardless of input:
//   { skeleton: Float32Array(n*17*2), conf: Float32Array(n*17),
//     fps, width, height, n_frames, engine, source }
//
// Skeleton layout is (frame, joint, xy) flattened row-major. Helpers below
// expose joint(frame, joint) and conf(frame, joint) views.

const N_JOINTS = 17;

export async function loadPoseFromFile(file) {
  const text = await file.text();
  const obj = JSON.parse(text);

  const fps = Number(obj.fps);
  const width = Number(obj.width);
  const height = Number(obj.height);
  const engine = obj.engine || "apple_vision_2d";

  let skeleton, conf, n_frames;

  if (typeof obj.skeleton_b64 === "string" && typeof obj.conf_b64 === "string") {
    // Production iOS format — base64 float32 LE.
    skeleton = decodeFloat32LE(obj.skeleton_b64);
    conf = decodeFloat32LE(obj.conf_b64);
    n_frames = Number(obj.n_frames);
  } else if (Array.isArray(obj.skeleton_frames) && Array.isArray(obj.confidences)) {
    // Parity / Mac-side format — straight nested arrays.
    n_frames = obj.skeleton_frames.length;
    skeleton = new Float32Array(n_frames * N_JOINTS * 2);
    conf = new Float32Array(n_frames * N_JOINTS);
    for (let f = 0; f < n_frames; f++) {
      const fr = obj.skeleton_frames[f];
      const cr = obj.confidences[f];
      for (let j = 0; j < N_JOINTS; j++) {
        skeleton[(f * N_JOINTS + j) * 2 + 0] = fr[j][0];
        skeleton[(f * N_JOINTS + j) * 2 + 1] = fr[j][1];
        conf[f * N_JOINTS + j] = cr[j];
      }
    }
  } else {
    throw new Error(
      "Unrecognized pose JSON. Expected either skeleton_b64/conf_b64 " +
      "(production format) or skeleton_frames/confidences (parity format)."
    );
  }

  if (skeleton.length !== n_frames * N_JOINTS * 2) {
    throw new Error(
      `skeleton length ${skeleton.length} doesn't match n_frames=${n_frames} * 17 * 2`
    );
  }

  return {
    skeleton, conf, fps, width, height, n_frames, engine,
    source: file.name,
  };
}

function decodeFloat32LE(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  // Float32Array reads using the host endianness, which is little-endian on
  // every platform we ship to (x86_64, arm64). If we ever needed to support
  // a big-endian host we'd have to byteswap here.
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4);
}

// Joint accessor: returns [x, y] for the given frame/joint.
export function jointXY(pose, frame, joint) {
  const i = (frame * N_JOINTS + joint) * 2;
  return [pose.skeleton[i], pose.skeleton[i + 1]];
}

export function jointConf(pose, frame, joint) {
  return pose.conf[frame * N_JOINTS + joint];
}
