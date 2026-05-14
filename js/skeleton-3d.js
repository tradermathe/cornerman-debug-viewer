// Apple Vision 3D joint topology — DIFFERENT from COCO-17 in skeleton.js.
// 17 joints, no face landmarks (no nose/eyes/ears), but adds spine/root/
// centerShoulder/topHead/centerHead. This file is intentionally separate
// from skeleton.js so the 2D pipeline's COCO-17 invariants stay untouched.

export const J3 = {
  TOP_HEAD: 0,
  CENTER_HEAD: 1,
  CENTER_SHOULDER: 2,
  L_SHOULDER: 3, R_SHOULDER: 4,
  L_ELBOW: 5, R_ELBOW: 6,
  L_WRIST: 7, R_WRIST: 8,
  SPINE: 9,
  ROOT: 10,
  L_HIP: 11, R_HIP: 12,
  L_KNEE: 13, R_KNEE: 14,
  L_ANKLE: 15, R_ANKLE: 16,
};

export const JOINT_NAMES_3D = [
  "topHead", "centerHead", "centerShoulder",
  "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
  "L_wrist", "R_wrist",
  "spine", "root",
  "L_hip", "R_hip",
  "L_knee", "R_knee",
  "L_ankle", "R_ankle",
];

export const EDGES_3D = [
  // head / neck
  [J3.TOP_HEAD, J3.CENTER_HEAD],
  [J3.CENTER_HEAD, J3.CENTER_SHOULDER],
  // shoulders + torso skeleton
  [J3.CENTER_SHOULDER, J3.L_SHOULDER],
  [J3.CENTER_SHOULDER, J3.R_SHOULDER],
  [J3.CENTER_SHOULDER, J3.SPINE],
  [J3.SPINE, J3.ROOT],
  // arms
  [J3.L_SHOULDER, J3.L_ELBOW], [J3.L_ELBOW, J3.L_WRIST],
  [J3.R_SHOULDER, J3.R_ELBOW], [J3.R_ELBOW, J3.R_WRIST],
  // hips
  [J3.ROOT, J3.L_HIP], [J3.ROOT, J3.R_HIP],
  [J3.L_HIP, J3.R_HIP],
  // legs
  [J3.L_HIP, J3.L_KNEE], [J3.L_KNEE, J3.L_ANKLE],
  [J3.R_HIP, J3.R_KNEE], [J3.R_KNEE, J3.R_ANKLE],
];

// ── 3D → 2D projection ────────────────────────────────────────────────────
// Apple body-frame: +X = boxer's left, +Y = up, +Z = forward (out of chest).
// To make the default canvas view feel "natural" (boxer standing upright,
// facing the viewer), we map: world-X → screen-X, world-Y → -screen-Y (flip
// because canvas Y grows downward), and apply a user-controlled yaw/pitch
// around the body's vertical axis on top.

// Compose a rotation matrix (3x3) from yaw (around Y) and pitch (around X).
// Order: pitch then yaw — pitch first lets the user "tip the head down" without
// the yaw axis tilting with them, which feels more like camera orbit.
function rotMatrix(yawRad, pitchRad) {
  const cy = Math.cos(yawRad),   sy = Math.sin(yawRad);
  const cp = Math.cos(pitchRad), sp = Math.sin(pitchRad);
  // R_y(yaw) * R_x(pitch):
  return [
    cy,        sy * sp,  sy * cp,
    0,         cp,      -sp,
    -sy,       cy * sp,  cy * cp,
  ];
}

function rotApply(R, x, y, z) {
  return [
    R[0] * x + R[1] * y + R[2] * z,
    R[3] * x + R[4] * y + R[5] * z,
    R[6] * x + R[7] * y + R[8] * z,
  ];
}

// Project a (x, y, z) body-frame metre point into canvas pixel coords.
// `view` is the result of makeView(); the returned [sx, sy, z] has the
// depth value preserved so the caller can sort or depth-shade.
export function project(view, x, y, z) {
  const [rx, ry, rz] = rotApply(view.R, x, y, z);
  // Weak perspective: shrink with depth modestly so wrists popping out
  // toward the viewer feel a bit bigger. Tunable; near-orthographic
  // (perspectiveStrength → 0) is fine too.
  const f = view.zoom / (1 + view.perspectiveStrength * rz);
  const sx = view.cx + rx * f;
  const sy = view.cy - ry * f;  // flip Y for canvas
  return [sx, sy, rz];
}

// Build a view object that the caller can reuse across a draw.
//
// `perspectiveStrength` defaults to 0 (orthographic). Reason: Apple's 3D model
// already under-extends limbs (rigid parametric skeleton + regression-toward-
// mean prior), so adding perspective foreshortening on top would further
// compress forward-extended arms — making the "dulled" effect worse than it
// already is. Orthographic gives an honest view of the metres-in-body-frame
// data; bump to e.g. 0.3 if you want a more "camera-like" feel.
export function makeView({ width, height, yawRad = 0, pitchRad = 0,
                          zoom = null, perspectiveStrength = 0 }) {
  return {
    width, height,
    cx: width / 2,
    cy: height / 2,
    R: rotMatrix(yawRad, pitchRad),
    // Default zoom assumes a ~1.7 m tall human and shows it ~70% of canvas height.
    zoom: zoom ?? (height * 0.7) / 1.7,
    perspectiveStrength,
    yawRad, pitchRad,
  };
}

// Draw the 17-joint skeleton at `frame` into `ctx`, using `view` for
// projection. `xyz` is the flat (N*17*3) array; `confMask` is an optional
// (N*17) Float32Array — if provided, joints with mask < threshold are dimmed
// (used to apply the 2D-derived confidence gate the scaffold described).
export function drawSkeleton3D(ctx, xyz, frame, view, {
  confMask = null,
  maskThreshold = 0.3,
  boneColor = "rgba(180, 220, 255, 0.85)",
  dimColor  = "rgba(180, 220, 255, 0.18)",
  jointColor = "#9fdfff",
  dimJointColor = "rgba(159, 223, 255, 0.25)",
  highlightJoints = null,   // Set<number>
  highlightColor = "#ffd966",
  jointRadius = 4,
  boneWidth = 2,
} = {}) {
  const N = 17;
  const base = frame * N * 3;

  // Pre-project all joints once.
  const xy = new Array(N);
  for (let j = 0; j < N; j++) {
    const x = xyz[base + j * 3 + 0];
    const y = xyz[base + j * 3 + 1];
    const z = xyz[base + j * 3 + 2];
    if (!Number.isFinite(x)) { xy[j] = null; continue; }
    xy[j] = project(view, x, y, z);
  }

  const trust = j => !confMask || confMask[frame * N + j] >= maskThreshold;

  // Bones first.
  ctx.lineWidth = boneWidth;
  for (const [a, b] of EDGES_3D) {
    const A = xy[a], B = xy[b];
    if (!A || !B) continue;
    const dim = !trust(a) || !trust(b);
    ctx.strokeStyle = dim ? dimColor : boneColor;
    ctx.beginPath();
    ctx.moveTo(A[0], A[1]);
    ctx.lineTo(B[0], B[1]);
    ctx.stroke();
  }

  // Joints on top.
  for (let j = 0; j < N; j++) {
    if (!xy[j]) continue;
    const dim = !trust(j);
    const hi = highlightJoints && highlightJoints.has(j);
    ctx.fillStyle = hi ? highlightColor : (dim ? dimJointColor : jointColor);
    const r = hi ? jointRadius * 1.7 : jointRadius;
    ctx.beginPath();
    ctx.arc(xy[j][0], xy[j][1], r, 0, Math.PI * 2);
    ctx.fill();
  }
}
