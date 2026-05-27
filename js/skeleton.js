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
//
// `showImputed` (default true): when a joint was NaN in the raw cache and
// got imputed to (0,0,0) by pose-loader's nan_to_num, draw it as a magenta
// "X" at (0,0) PLUS a dashed line from its real anchor to that phantom
// point. This mirrors exactly what the classifier sees (the model is fed
// (0,0)-imputed wrists with full velocity-spike bone vectors) — debugging
// the classifier honestly requires seeing the same thing. Lenses that
// genuinely want the strict-NaN view (rules-engine debug) can opt out by
// passing `showImputed: false`.
export function drawSkeleton(ctx, pose, frame, style = {}) {
  const {
    boneColor = "rgba(255,255,255,0.65)",
    boneWidth = 2,
    jointRadius = 4,
    minConf = 0.05,            // hide joints/edges below this conf
    highlightJoints = null,    // Set of joint indices to draw larger
    hideJoints = null,         // Set of joint indices to skip entirely
                               //   (and any edge touching them)
    showImputed = true,        // render NaN-imputed joints at their phantom (0,0)
  } = style;

  ctx.lineWidth = boneWidth;
  ctx.strokeStyle = boneColor;

  const imputed = pose.imputed;   // may be undefined on older pose objects

  // Bones — only drawn if both endpoints are above minConf (otherwise they
  // ghost-connect to (0,0) when a joint wasn't detected). Edges that touch
  // a hidden joint are skipped so the wrist-swap lens (which redraws wrists
  // separately) doesn't show a forearm into thin air.
  for (const [a, b] of EDGES) {
    if (hideJoints && (hideJoints.has(a) || hideJoints.has(b))) continue;
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
    if (hideJoints && hideJoints.has(j)) continue;
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

  // Imputed joints — render last so the magenta marker sits on top of the
  // (0,0) corner regardless of whatever else got drawn there. Each marker
  // is a 14×14 X at the phantom point with a "j:<index>" label, so the
  // user can see WHICH joints got imputed without parsing pixel coords.
  if (showImputed && imputed) {
    ctx.save();
    ctx.strokeStyle = "#e040fb";    // magenta = same hue the punch-classifier lens uses for "mistype"
    ctx.fillStyle   = "#e040fb";
    ctx.lineWidth = 2;
    const sz = 7;
    let row = 0;
    ctx.font = `${Math.round(jointRadius * 3)}px ui-monospace, "SF Mono", monospace`;
    for (let j = 0; j < 17; j++) {
      if (!imputed[frame * 17 + j]) continue;
      if (hideJoints && hideJoints.has(j)) continue;
      // X marker at (0,0)
      ctx.beginPath();
      ctx.moveTo(-sz, -sz); ctx.lineTo(sz, sz);
      ctx.moveTo(-sz,  sz); ctx.lineTo(sz, -sz);
      ctx.stroke();
      // Labels stacked vertically next to the X so multiple imputed joints
      // don't overdraw each other.
      ctx.fillText(`j${j}`, sz + 4, sz + 2 + row * (jointRadius * 3));
      row++;
    }
    ctx.restore();
  }
}
