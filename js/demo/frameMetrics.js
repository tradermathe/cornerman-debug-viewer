// Per-frame geometric form checks, ported verbatim from the debug-viewer lenses
// guard_height.js and elbow_tuck_lens.js. Same geometry the rules read off; run
// here from the pose so the demo can show live per-frame guard/flare values.

import { J, torsoHeight } from "../skeleton.js";

const GUARD_TARGET_OFFSET = 0.30;  // fraction of torso below the nose (guard_height.js)
const GUARD_MIN_CONF = 0.30;
const FLARE_MIN_CONF = 0.50;       // elbow_tuck_lens REQUIRED-joint threshold

const x = (p, f, j) => p.skeleton[(f * 17 + j) * 2];
const y = (p, f, j) => p.skeleton[(f * 17 + j) * 2 + 1];
const c = (p, f, j) => p.conf[f * 17 + j];

// Guard height for one wrist: signed distance below the nose-relative target in
// torso units (dist > 0 ⇒ dropped). null when gated on confidence.
export function guardHeight(p, f, wristJ) {
  const torso = torsoHeight(p, f);
  const noseY = y(p, f, J.NOSE), wristY = y(p, f, wristJ);
  if (!(c(p, f, J.NOSE) >= GUARD_MIN_CONF) || !(torso > 1) ||
      !(c(p, f, wristJ) >= GUARD_MIN_CONF) || !Number.isFinite(noseY) || !Number.isFinite(wristY)) return null;
  const dist = (wristY - (noseY + GUARD_TARGET_OFFSET * torso)) / torso;
  return { dist, dropped: dist > 0 };
}

// Elbow flare for one arm: |shoulderX − elbowX| / torso (Euclidean shoulder→hip).
// null when the required torso/arm joints aren't confident.
export function elbowFlare(p, f, shJ, elJ) {
  const sx = 0.5 * (x(p, f, J.L_SHOULDER) + x(p, f, J.R_SHOULDER));
  const sy = 0.5 * (y(p, f, J.L_SHOULDER) + y(p, f, J.R_SHOULDER));
  const hx = 0.5 * (x(p, f, J.L_HIP) + x(p, f, J.R_HIP));
  const hy = 0.5 * (y(p, f, J.L_HIP) + y(p, f, J.R_HIP));
  const torso = Math.hypot(sx - hx, sy - hy);
  const required = [J.L_SHOULDER, J.R_SHOULDER, J.L_ELBOW, J.R_ELBOW, J.L_HIP, J.R_HIP];
  if (!(torso > 1e-6) || !required.every((j) => c(p, f, j) > FLARE_MIN_CONF)) return null;
  const flare = Math.abs(x(p, f, shJ) - x(p, f, elJ)) / torso;
  return { flare, status: flare > 0.3 ? "flared" : flare > 0.2 ? "out" : "tucked" };
}
