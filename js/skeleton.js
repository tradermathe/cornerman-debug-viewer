// COCO-17 joint indices and bone topology, mirroring the rules engine
// (cornerman_rules/utils/skeleton.py). Both YOLO-Pose and the on-device
// Apple Vision module write joints in this order.

export const J = {
  NOSE: 0,
  L_EYE: 1, R_EYE: 2, L_EAR: 3, R_EAR: 4,
  L_SHOULDER: 5, R_SHOULDER: 6,
  L_ELBOW: 7, R_ELBOW: 8,
  L_WRIST: 9, R_WRIST: 10,
  L_HIP: 11, R_HIP: 12,
  L_KNEE: 13, R_KNEE: 14,
  L_ANKLE: 15, R_ANKLE: 16,
};

export const JOINT_NAMES = [
  "nose", "L_eye", "R_eye", "L_ear", "R_ear",
  "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
  "L_wrist", "R_wrist", "L_hip", "R_hip",
  "L_knee", "R_knee", "L_ankle", "R_ankle",
];

// Edges drawn as bones. Grouped roughly by region for readability; the renderer
// just iterates the flat list.
export const EDGES = [
  // arms
  [J.L_SHOULDER, J.L_ELBOW], [J.L_ELBOW, J.L_WRIST],
  [J.R_SHOULDER, J.R_ELBOW], [J.R_ELBOW, J.R_WRIST],
  // shoulders + torso
  [J.L_SHOULDER, J.R_SHOULDER],
  [J.L_SHOULDER, J.L_HIP], [J.R_SHOULDER, J.R_HIP],
  [J.L_HIP, J.R_HIP],
  // legs
  [J.L_HIP, J.L_KNEE], [J.L_KNEE, J.L_ANKLE],
  [J.R_HIP, J.R_KNEE], [J.R_KNEE, J.R_ANKLE],
  // face
  [J.NOSE, J.L_EYE], [J.NOSE, J.R_EYE],
  [J.L_EYE, J.L_EAR], [J.R_EYE, J.R_EAR],
];

// Used by the rules engine to normalize lengths. Mirrors torso_height() in
// cornerman_rules/utils/skeleton.py: vertical distance between shoulder
// midpoint and hip midpoint.
export function torsoHeight(pose, frame) {
  const lsy = pose.skeleton[(frame * 17 + J.L_SHOULDER) * 2 + 1];
  const rsy = pose.skeleton[(frame * 17 + J.R_SHOULDER) * 2 + 1];
  const lhy = pose.skeleton[(frame * 17 + J.L_HIP) * 2 + 1];
  const rhy = pose.skeleton[(frame * 17 + J.R_HIP) * 2 + 1];
  return Math.abs((lsy + rsy) / 2 - (lhy + rhy) / 2);
}

// Color a joint marker by confidence — green ≥ 0.5, amber ≥ 0.2, red below.
// Zero confidence (Vision didn't detect this joint) renders transparent so the
// overlay doesn't lie about a joint that isn't actually there.
export function confColor(c) {
  if (c <= 0) return "rgba(0,0,0,0)";
  if (c >= 0.5) return "#5fd97a";
  if (c >= 0.2) return "#f5b945";
  return "#e85a5a";
}

// Draw the bone skeleton + joint dots. `style` lets a rule panel dim/recolor
// things without rewriting the renderer (e.g. fade non-relevant joints).
export function drawSkeleton(ctx, pose, frame, style = {}) {
  const {
    boneColor = "rgba(255,255,255,0.65)",
    boneWidth = 2,
    jointRadius = 4,
    minConf = 0.05,            // hide joints/edges below this conf
    highlightJoints = null,    // Set of joint indices to draw larger
  } = style;

  ctx.lineWidth = boneWidth;
  ctx.strokeStyle = boneColor;

  // Bones — only drawn if both endpoints are above minConf (otherwise they
  // ghost-connect to (0,0) when a joint wasn't detected).
  for (const [a, b] of EDGES) {
    const ca = pose.conf[frame * 17 + a];
    const cb = pose.conf[frame * 17 + b];
    if (ca < minConf || cb < minConf) continue;
    const ax = pose.skeleton[(frame * 17 + a) * 2];
    const ay = pose.skeleton[(frame * 17 + a) * 2 + 1];
    const bx = pose.skeleton[(frame * 17 + b) * 2];
    const by = pose.skeleton[(frame * 17 + b) * 2 + 1];
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }

  // Joints
  for (let j = 0; j < 17; j++) {
    const c = pose.conf[frame * 17 + j];
    if (c < minConf) continue;
    const x = pose.skeleton[(frame * 17 + j) * 2];
    const y = pose.skeleton[(frame * 17 + j) * 2 + 1];
    const isHi = highlightJoints && highlightJoints.has(j);
    const r = isHi ? jointRadius * 2.2 : jointRadius;
    ctx.fillStyle = confColor(c);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    if (isHi) {
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "rgba(255,255,255,0.9)";
      ctx.stroke();
    }
  }
}
