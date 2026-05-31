// Forearm axiality core math. No DOM, no imports: this module is shared by
// the lens (forearm_axiality.js) and the Node oracle test
// (test/forearm_axiality_oracle.mjs), so the numbers the test certifies are
// exactly the numbers the viewer ships.
//
// Signal: the forearm (elbow to wrist) is a fixed length bone, so its on
// screen length shrinks only as it rotates out of the image plane. We read
// that shrink against a flat reference (the torso ruler) and turn it into an
// angle. axiality 0 = forearm flat across the image (a hook), 1 = forearm
// pointing down the lens (a straight thrown toward or away from the camera).
//
// All inputs are pixel coordinates (isotropic). The pose cache is normalized
// [0, 1]; the viewer's pose loader already multiplies by (W, H) before these
// functions see a point, so do NOT scale again here.

export function dist(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

export function mid(a, b) {
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
}

// Per frame, per arm: the rigid bone length (forearm), the straight line
// torso ruler (mid shoulder to mid hip, NOT the vertical drop), and their
// scale free ratio. The ratio kills distance to camera and body size because
// both lengths sit at roughly the same depth and scale together.
export function forearmLengths(elbow, wrist, midSh, midHip) {
  const forearm = dist(elbow, wrist);
  const torso = dist(midSh, midHip);
  const ratio = torso > 0 ? forearm / torso : NaN;
  return { forearm, torso, ratio };
}

// B = clamp(ratio / F0, 0, 1) is the foreshortening fraction (how much of the
// flat length still shows). axiality = sqrt(1 - B*B) is the sine of the out of
// plane angle. F0 is the flat forearm reference for this arm (the high
// percentile of ratio over the clip): the flattest, most in plane frames give
// the largest ratio, so that high percentile is the flat length.
export function axialityFromRatio(ratio, F0) {
  if (!(F0 > 0) || !Number.isFinite(ratio)) return { B: NaN, axiality: NaN };
  let B = ratio / F0;
  if (B < 0) B = 0;
  if (B > 1) B = 1;
  return { B, axiality: Math.sqrt(1 - B * B) };
}

// Full chain from four pixel points plus the clip's F0 to the axiality and
// every intermediate (so the viewer can show which step broke).
export function forearmAxiality(elbow, wrist, midSh, midHip, F0) {
  const { forearm, torso, ratio } = forearmLengths(elbow, wrist, midSh, midHip);
  const { B, axiality } = axialityFromRatio(ratio, F0);
  return { forearm, torso, ratio, B, axiality };
}

// Linear interpolated percentile of a numeric array, NaNs dropped, p in
// [0, 100]. Returns { value, index, n } where index is the frame whose value
// is closest to the percentile (so F0's reference frame can be inspected) and
// n is how many valid samples fed it. A wrong F0 biases every axiality, so the
// caller surfaces both value and index.
export function percentileWithIndex(values, p) {
  const idx = [];
  for (let i = 0; i < values.length; i++) {
    if (Number.isFinite(values[i])) idx.push(i);
  }
  if (!idx.length) return { value: NaN, index: -1, n: 0 };
  idx.sort((a, b) => values[a] - values[b]);
  const rank = (p / 100) * (idx.length - 1);
  const lo = Math.floor(rank), hi = Math.ceil(rank);
  const value = values[idx[lo]] * (1 - (rank - lo)) + values[idx[hi]] * (rank - lo);
  let best = idx[0], bestErr = Infinity;
  for (const i of idx) {
    const e = Math.abs(values[i] - value);
    if (e < bestErr) { bestErr = e; best = i; }
  }
  return { value, index: best, n: idx.length };
}

// Running median over a centered window (odd length), NaNs ignored. Used to
// flag depth lean frames where the torso ruler collapses (the boxer leans in
// or away) and the denominator can no longer be trusted.
export function runningMedian(values, win) {
  const half = Math.floor(win / 2);
  const out = new Float64Array(values.length).fill(NaN);
  const buf = [];
  for (let i = 0; i < values.length; i++) {
    buf.length = 0;
    for (let k = i - half; k <= i + half; k++) {
      if (k >= 0 && k < values.length && Number.isFinite(values[k])) buf.push(values[k]);
    }
    if (!buf.length) continue;
    buf.sort((a, b) => a - b);
    const m = buf.length >> 1;
    out[i] = buf.length % 2 ? buf[m] : (buf[m - 1] + buf[m]) / 2;
  }
  return out;
}
