// Synthetic oracle for the forearm axiality metric. Acceptance gate: this must
// pass before any real video is trusted.
//
// It builds a three point arm (shoulder/elbow/wrist) plus a torso ruler in 3D
// at a known out of plane angle, projects them through a pinhole camera to
// pixels (with W != H on purpose), and runs the SAME forearmAxiality function
// the lens ships. Then it asserts:
//   1. axiality tracks sin(angle) across an angle sweep.
//   2. at a fixed angle, axiality stays in a tight band as we sweep camera
//      distance, off center offset, focal length, body scale, and the forearm's
//      in plane orientation.
// A slope where there should be a flat line means a bug:
//   drift with distance / scale  -> ratio normalization is wrong,
//   drift with focal length      -> projection or normalization is wrong,
//   drift with in plane phi      -> lengths were measured in anisotropic
//                                   (non pixel) coordinates, the (W, H) scaling
//                                   was skipped,
//   drift with off center        -> real perspective residual (allowed a wider
//                                   band, this is the looming the metric cannot
//                                   fully remove).
//
// Run: node test/forearm_axiality_oracle.mjs

import { forearmAxiality, forearmLengths } from "../js/rules/forearm_axiality_core.js";

// Pinhole projection. Camera at the origin looking down +Z (Z > 0 is in
// front). f is in pixel units. Principal point at image center (W/2, H/2).
// W != H so the metric only reads as isotropic if lengths are computed in
// pixels (the whole point of the isotropy check).
function project(P, cam) {
  return [cam.f * P[0] / P[2] + cam.W / 2, cam.f * P[1] / P[2] + cam.H / 2];
}

// Build a boxer and project to 2D pixels.
//   theta : forearm angle out of the image plane (0 = flat across image,
//           PI/2 = pointing straight down the lens). axiality should read
//           sin(theta).
//   phi   : in plane rotation of the (flat) forearm direction about the view
//           axis. A correct pixel metric is invariant to phi.
//   Z0    : boxer depth (metres). Xb, Yb : world offset of the boxer.
//   L     : true forearm length. T : true torso length. scale : body scale.
function scene({ theta, phi = 0, Z0 = 6, Xb = 0, Yb = 0,
                 L = 0.30, T = 0.55, f = 1400, W = 1080, H = 1920, scale = 1 }) {
  const cam = { f, W, H };
  const k = scale;
  // Torso ruler: vertical, fronto parallel, both ends at depth Z0.
  const midHip3 = [Xb * k, (Yb - T / 2) * k, Z0 * k];
  const midSh3  = [Xb * k, (Yb + T / 2) * k, Z0 * k];
  // Elbow at shoulder height and the boxer's lateral position.
  const elbow3  = [Xb * k, (Yb + T / 2) * k, Z0 * k];
  // Flat forearm vector points along +X; tilt out of plane by theta about the
  // vertical axis (toward camera is -Z); then rotate in plane by phi about the
  // view axis.
  const ct = Math.cos(theta), st = Math.sin(theta);
  let fx = L * ct, fy = 0; const fz = -L * st;
  const cp = Math.cos(phi), sp = Math.sin(phi);
  const rx = fx * cp - fy * sp;
  const ry = fx * sp + fy * cp;
  const wrist3 = [elbow3[0] + rx * k, elbow3[1] + ry * k, elbow3[2] + fz * k];
  return {
    elbow: project(elbow3, cam),
    wrist: project(wrist3, cam),
    midSh: project(midSh3, cam),
    midHip: project(midHip3, cam),
  };
}

// F0 (the flat reference) is the ratio of the same geometry held flat
// (theta = 0), mimicking the per clip high percentile of a flat armed frame.
function axAt(opts) {
  const flat = scene({ ...opts, theta: 0 });
  const F0 = forearmLengths(flat.elbow, flat.wrist, flat.midSh, flat.midHip).ratio;
  const s = scene(opts);
  const r = forearmAxiality(s.elbow, s.wrist, s.midSh, s.midHip, F0);
  return { ...r, F0 };
}

let failures = 0, total = 0;
function check(name, got, want, tol) {
  total++;
  const ok = Number.isFinite(got) && Math.abs(got - want) <= tol;
  if (!ok) failures++;
  const d = Number.isFinite(got) ? (got - want).toFixed(4) : "NaN";
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name.padEnd(14)} got ${String(got.toFixed ? got.toFixed(4) : got).padStart(8)}  want ${want.toFixed(4)}  d ${d}  (tol ${tol})`);
}

const D = Math.PI / 180;
const A45 = Math.sin(45 * D); // 0.7071

console.log("# 1. angle sweep: axiality should track sin(theta)");
for (let deg = 0; deg <= 70; deg += 10) {
  check(`theta=${deg}`, axAt({ theta: deg * D }).axiality, Math.sin(deg * D), 0.08);
}

// Distance carries a small perspective droop (closer boxer = more near end
// magnification = axiality reads a touch low). Bounded at ~0.06 down to 3 m,
// asymptoting to the orthographic 0.707 far away. Tol covers the close end.
console.log("\n# 2a. distance sweep at theta=45 (axiality stays ~0.707)");
for (const Z0 of [3, 4, 5, 6, 8, 10]) {
  check(`Z0=${Z0}`, axAt({ theta: 45 * D, Z0 }).axiality, A45, 0.07);
}

// Off center is the metric's weak axis: a boxer well off the optical axis,
// combined with the toward camera tilt, gets a SIGNED perspective error the
// ratio cannot remove (this is the "straight to camera punch still looks long"
// effect). We ASSERT only the realistic centered range (boxer within ~0.5 m of
// the optical axis, the usual framing) and CHARACTERIZE the extremes as data,
// not as a pass/fail, because that drift is physics, not a bug.
console.log("\n# 2b. off center sweep at theta=45 (assert centered range only)");
for (const Xb of [-0.5, -0.25, 0, 0.25, 0.5]) {
  check(`Xb=${Xb}`, axAt({ theta: 45 * D, Xb }).axiality, A45, 0.12);
}
console.log("   characterize (no assert): perspective residual at large offset");
for (const Xb of [-1.5, -1.0, 1.0, 1.5]) {
  const got = axAt({ theta: 45 * D, Xb }).axiality;
  console.log(`     Xb=${String(Xb).padStart(5)}  axiality ${got.toFixed(4)}  (true 0.7071, d ${(got - A45).toFixed(4)})`);
}

console.log("\n# 2c. focal sweep at theta=45 (f cancels in the ratio)");
for (const f of [800, 1200, 1600, 2000, 2400]) {
  check(`f=${f}`, axAt({ theta: 45 * D, f }).axiality, A45, 0.03);
}

console.log("\n# 2d. body scale sweep at theta=45 (scale cancels in the ratio)");
for (const scale of [0.5, 0.75, 1, 1.5, 2]) {
  check(`scale=${scale}`, axAt({ theta: 45 * D, scale }).axiality, A45, 0.03);
}

// Isotropy gate: a correct PIXEL metric barely moves as the flat forearm
// rotates in plane. Any large slope here would mean lengths were measured in
// anisotropic normalized coordinates (the (W, H) scaling was skipped). The
// small residual is perspective from the elbow sitting above the principal
// point, not anisotropy.
console.log("\n# 2e. in plane orientation sweep at theta=45 (isotropy: invariant to phi)");
for (const deg of [0, 30, 60, 90, 120, 150]) {
  check(`phi=${deg}`, axAt({ theta: 45 * D, phi: deg * D }).axiality, A45, 0.07);
}

console.log(`\n${failures ? "FAILED" : "OK"}: ${total - failures}/${total} checks passed`);
process.exit(failures ? 1 : 0);
