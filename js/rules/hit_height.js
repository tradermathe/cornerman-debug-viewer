// Hit-height lens — "did the punch land at a legal, useful height?".
//
// Single-boxer (shadowboxing) footage has no opponent, so we judge the fist
// against an IMAGINARY opponent: someone the same size as the boxer, standing
// in a normal boxing stance, on the same floor. The reference is drawn on the
// overlay as a ghost stance skeleton, and the zone lines are read straight off
// that skeleton's landmarks so the lines and the body always agree.
//
// The reference is built from the REAL skeleton we see — not standard body
// ratios:
//   - Every segment (shin, thigh, torso, head, shoulder/hip width, arms) is the
//     clip-median of the boxer's own measured limb length. So the ghost is the
//     real height of the person on screen.
//   - The feet are placed where we actually see them: the current frame's real
//     ankle positions (clip-median fallback when a foot is out of frame).
//   - The legs are the real measured leg length, drawn up from those real feet —
//     bent slightly into an athletic stance, not bolt upright.
//
// Limb lengths are measured as Euclidean segment lengths, so they don't shrink
// when the boxer ducks (a bent knee is the same bone length). The floor is the
// planted feet. Together that means a duck lowers the fist, not the target.
//
// Two on-target zones, read off the stance skeleton:
//   HEAD   slightly below the chin → slightly above the crown
//   BODY   solar plexus → belt
// Everything else flagged: over the head, the shoulder/neck dead zone (chin →
// solar plexus), below the belt.
//
// Landing frame = most-extended frame in the punch window (max |shoulder→wrist|).
// Side maps from (hand, stance) like guard_drop / arm_extension; wrist source
// prefers the v6 glove-baked wrist, then a legacy glove sidecar, then pose.

import { J } from "../skeleton.js";
import { gloveXY, gloveConf } from "../pose-loader.js";

const DEFAULTS = {
  // Head zone margins (× standing height) around the stance crown / chin.
  headTopMargin: 0.04,   // head zone extends this far ABOVE the crown
  chinMargin:    0.06,   // head zone extends this far BELOW the chin
  minWristConfidence:  0.20,
  minAnchorConfidence: 0.20,
};

// Stance shaping (applied to the boxer's REAL measured dimensions — these are
// pose angles / sub-segment ratios, not body-size assumptions).
const SOLAR_OF_TORSO = 0.65;  // solar plexus ≈ 65% up the torso from the hips
const HEAD_BLOCK_K   = 1.50;  // shoulder→crown ≈ 1.5 × the vertical nose rise (real)
const LEAN_TAN       = 0.16;  // forward lean ≈ 9° from the hips up (boxing stance)
// Fallback only: torso (shoulder_mid→hip_mid) ÷ standing height. Used to place
// the floor / height when the legs & feet are too unreliable to measure
// directly (e.g. a side-on clip where the lower legs drop out).
const TORSO_FRACTION = 0.29;

const COLORS = {
  ok:       "#5fd97a",
  flag:     "#e85a5a",
  bandOk:   "rgba(95,217,122,0.08)",
  bandFlag: "rgba(232,90,90,0.09)",
  line:     "rgba(255,255,255,0.55)",
  ghost:    "#5fd0e6",
};

const SIDE_FOR = {
  lead: { orthodox: "L", southpaw: "R" },
  rear: { orthodox: "R", southpaw: "L" },
};
const JOINTS_FOR_SIDE = {
  L: { shoulder: J.L_SHOULDER, wrist: J.L_WRIST, gloveSide: 0 },
  R: { shoulder: J.R_SHOULDER, wrist: J.R_WRIST, gloveSide: 1 },
};

let host;
let cfg = { ...DEFAULTS };
let ref = null;        // cached measured reference (segment medians + feet)
let refPose = null;

export const HitHeightRule = {
  id: "hit_height",
  label: "Hit height",

  skeletonStyle() {
    return {
      boneColor: "rgba(255,255,255,0.25)",
      boneWidth: 1.5,
      jointRadius: 3,
      highlightJoints: new Set([J.L_WRIST, J.R_WRIST]),
    };
  },

  mount(_host, state) {
    host = _host;
    host.innerHTML = `
      <h2>Hit height</h2>
      <p class="hint">Where the punching fist peaks, against a same-size opponent
        standing in a normal stance (the <b style="color:${COLORS.ghost}">ghost
        skeleton</b> on the overlay). It's built from the boxer's own measured
        limb lengths and stands on the real feet, so it's the true height of the
        person on screen and camera-distance insensitive.
        <b style="color:${COLORS.ok}">Head</b> &amp;
        <b style="color:${COLORS.ok}">body</b> are on-target;
        <b style="color:${COLORS.flag}">over the head</b>,
        <b style="color:${COLORS.flag}">shoulder height</b>, and
        <b style="color:${COLORS.flag}">below the belt</b> are flagged.</p>

      <h3>Standing reference</h3>
      <div class="metric-grid">
        <div class="metric"><div class="metric-label">Height px</div><div class="metric-val" id="hh-H">—</div></div>
        <div class="metric"><div class="metric-label">Leg px</div><div class="metric-val" id="hh-leg">—</div></div>
        <div class="metric"><div class="metric-label">Feet</div><div class="metric-val" id="hh-floor">—</div></div>
      </div>
      <div class="metric">
        <div class="metric-label">Active punch — fist height</div>
        <div class="metric-val" id="hh-live-height">—</div>
        <div class="metric-sub" id="hh-live-zone"></div>
      </div>

      <h3>Per-punch</h3>
      <div id="hh-summary" class="hint"></div>
      <div id="hh-table"></div>

      <h3>Tuning</h3>
      <label class="slider">
        <span>head zone above crown = <output id="hh-o1">${cfg.headTopMargin.toFixed(2)}</output></span>
        <input type="range" id="hh-s1" min="0" max="0.15" step="0.01" value="${cfg.headTopMargin}">
      </label>
      <label class="slider">
        <span>head zone below chin = <output id="hh-o2">${cfg.chinMargin.toFixed(2)}</output></span>
        <input type="range" id="hh-s2" min="0" max="0.15" step="0.01" value="${cfg.chinMargin}">
      </label>
      <label class="slider">
        <span>min wrist confidence = <output id="hh-o3">${cfg.minWristConfidence.toFixed(2)}</output></span>
        <input type="range" id="hh-s3" min="0" max="1" step="0.01" value="${cfg.minWristConfidence}">
      </label>
    `;

    const wire = (slider, out, key) => {
      const s = host.querySelector(slider);
      const o = host.querySelector(out);
      s.addEventListener("input", () => {
        cfg[key] = parseFloat(s.value);
        o.textContent = cfg[key].toFixed(2);
        renderTable(state);
        seekHack(state, state.frame);
      });
    };
    wire("#hh-s1", "#hh-o1", "headTopMargin");
    wire("#hh-s2", "#hh-o2", "chinMargin");
    wire("#hh-s3", "#hh-o3", "minWristConfidence");

    host.addEventListener("click", (ev) => {
      const tr = ev.target.closest("tr[data-frame]");
      if (!tr) return;
      const slider = document.getElementById("scrubber");
      if (!slider) return;
      slider.value = tr.dataset.frame;
      slider.dispatchEvent(new Event("input"));
    });

    renderTable(state);
  },

  draw(ctx, state) {
    const p = pickPose(state);
    const r = getReference(p);
    if (!r) return;
    const S = stanceAt(p, state.frame, r);
    if (!S) return;
    const W = ctx.canvas.width;
    const s = state.renderScale || 1;
    const B = boundaries(S);

    fillBand(ctx, -1e4,        B.overHead, W, COLORS.bandFlag); // over head
    fillBand(ctx, B.overHead,  B.head,     W, COLORS.bandOk);   // head
    fillBand(ctx, B.head,      B.bodyTop,  W, COLORS.bandFlag); // shoulder/neck
    fillBand(ctx, B.bodyTop,   B.belt,     W, COLORS.bandOk);   // body
    fillBand(ctx, B.belt,      1e4,        W, COLORS.bandFlag); // below belt

    drawStanceSkeleton(ctx, S, s);

    labeledLine(ctx, B.overHead, W, "over head ↑", s);
    labeledLine(ctx, B.head,     W, "chin",         s);
    labeledLine(ctx, B.bodyTop,  W, "solar plexus", s);
    labeledLine(ctx, B.belt,     W, "belt ↓",       s);

    const punch = activePunch(state);
    if (!punch) return;
    const w = wristXY(p, state.frame, JOINTS_FOR_SIDE[sideFor(punch)], cfg);
    if (!w) return;
    const z = zoneFor(w.y, B);
    const col = z.flag ? COLORS.flag : COLORS.ok;
    ctx.save();
    ctx.strokeStyle = col;
    ctx.lineWidth = 2.5 * s;
    ctx.beginPath();
    ctx.arc(w.x, w.y, 9 * s, 0, Math.PI * 2);
    ctx.stroke();
    drawTag(ctx, w.x + 12 * s, w.y, z.label, col, s);
    ctx.restore();
  },

  update(state) {
    const p = pickPose(state);
    const r = getReference(p);
    const S = r ? stanceAt(p, state.frame, r) : null;
    setText("hh-H", S ? S.H.toFixed(0) : "—");
    setText("hh-leg", S ? S.legLen.toFixed(0) + (r.legSource === "estimated" ? " est" : "") : "—");
    setText("hh-floor", S ? S.floorSource : `<span class="muted">unknown</span>`);

    const punch = activePunch(state);
    if (!S || !punch) {
      setText("hh-live-height", "—");
      setText("hh-live-zone", punch ? "" : `<span class="muted">no punch at this frame</span>`);
      return;
    }
    const w = wristXY(p, state.frame, JOINTS_FOR_SIDE[sideFor(punch)], cfg);
    if (!w) {
      setText("hh-live-height", "—");
      setText("hh-live-zone", `<span class="muted">no wrist signal</span>`);
      return;
    }
    const B = boundaries(S);
    const z = zoneFor(w.y, B);
    const frac = (S.floorY - w.y) / S.H;
    setText("hh-live-height", `${frac.toFixed(2)} H`);
    setText("hh-live-zone", `<span class="${z.flag ? "bad" : "good"}">${z.label}</span> · ${w.source}`);
  },
};

// ─── reference (real measured dimensions) ────────────────────────────────────

function getReference(pose) {
  if (pose === refPose) return ref;
  ref = buildReference(pose);
  refPose = pose;
  return ref;
}

// Measure the boxer's real dimensions from the clip and SUM them for standing
// height: legs (ankle→knee + knee→hip, both sides), torso, and a head block.
// Summing real parts keeps the legs the right length instead of backing them
// out as a remainder. Medians so jitter / dropouts don't wobble it; torso-ratio
// fallbacks only when a part is too rarely seen to trust.
function buildReference(pose) {
  const N = pose.n_frames;
  const g = cfg.minAnchorConfidence;
  const min = Math.max(8, 0.1 * N);
  const shin = [], thigh = [], torsoA = [], shoulderW = [], hipW = [], upperArm = [], foreArm = [], noseRise = [];
  const floors = [], feetL = [], feetR = [], hips = [], centers = [], noseDX = [];

  const xy = (f, j) => [pose.skeleton[(f * 17 + j) * 2], pose.skeleton[(f * 17 + j) * 2 + 1]];
  const ok = (f, j) => pose.conf[f * 17 + j] >= g;
  const seg = (f, a, b) => { const [ax, ay] = xy(f, a), [bx, by] = xy(f, b); return Math.hypot(ax - bx, ay - by); };

  for (let f = 0; f < N; f++) {
    if (ok(f, J.L_KNEE) && ok(f, J.L_ANKLE)) shin.push(seg(f, J.L_KNEE, J.L_ANKLE));
    if (ok(f, J.R_KNEE) && ok(f, J.R_ANKLE)) shin.push(seg(f, J.R_KNEE, J.R_ANKLE));
    if (ok(f, J.L_HIP) && ok(f, J.L_KNEE)) thigh.push(seg(f, J.L_HIP, J.L_KNEE));
    if (ok(f, J.R_HIP) && ok(f, J.R_KNEE)) thigh.push(seg(f, J.R_HIP, J.R_KNEE));
    if (ok(f, J.L_SHOULDER) && ok(f, J.L_ELBOW)) upperArm.push(seg(f, J.L_SHOULDER, J.L_ELBOW));
    if (ok(f, J.R_SHOULDER) && ok(f, J.R_ELBOW)) upperArm.push(seg(f, J.R_SHOULDER, J.R_ELBOW));
    if (ok(f, J.L_ELBOW) && ok(f, J.L_WRIST)) foreArm.push(seg(f, J.L_ELBOW, J.L_WRIST));
    if (ok(f, J.R_ELBOW) && ok(f, J.R_WRIST)) foreArm.push(seg(f, J.R_ELBOW, J.R_WRIST));

    if (ok(f, J.L_SHOULDER) && ok(f, J.R_SHOULDER) && ok(f, J.L_HIP) && ok(f, J.R_HIP)) {
      const [lsx, lsy] = xy(f, J.L_SHOULDER), [rsx, rsy] = xy(f, J.R_SHOULDER);
      const [lhx, lhy] = xy(f, J.L_HIP),      [rhx, rhy] = xy(f, J.R_HIP);
      const smx = (lsx + rsx) / 2, smy = (lsy + rsy) / 2;
      const hmx = (lhx + rhx) / 2, hmy = (lhy + rhy) / 2;
      torsoA.push(Math.hypot(smx - hmx, smy - hmy));
      shoulderW.push(Math.hypot(lsx - rsx, lsy - rsy));
      hipW.push(Math.hypot(lhx - rhx, lhy - rhy));
      hips.push({ x: hmx, y: hmy });
      centers.push((smx + hmx) / 2);
      if (ok(f, J.NOSE)) {
        const [nx, ny] = xy(f, J.NOSE);
        noseDX.push(nx - hmx);                 // nose ahead of the hips → facing/lean direction
        if (smy > ny) noseRise.push(smy - ny); // vertical nose rise above the shoulders
      }
    }
    let foot = -Infinity;
    if (ok(f, J.L_ANKLE)) { const [x, y] = xy(f, J.L_ANKLE); foot = Math.max(foot, y); feetL.push({ x, y }); }
    if (ok(f, J.R_ANKLE)) { const [x, y] = xy(f, J.R_ANKLE); foot = Math.max(foot, y); feetR.push({ x, y }); }
    if (foot > -Infinity) floors.push(foot);
  }

  if (!torsoA.length) return null;
  const med = a => a.length ? median(a) : 0;
  const medPt = arr => ({ x: median(arr.map(p => p.x)), y: median(arr.map(p => p.y)) });
  const torso = med(torsoA);
  const hip = hips.length ? medPt(hips) : null;

  // Head block (shoulder→crown) from the real VERTICAL nose rise — robust to the
  // sideways lean that inflates a Euclidean shoulder→nose length.
  const headBlock = noseRise.length >= min ? HEAD_BLOCK_K * median(noseRise) : 0.33 * torso;
  const neck = 0.45 * headBlock, headLen = 0.55 * headBlock;

  // Legs from the real measured segments (avg L/R via pooled medians). Fall back
  // to a torso-ratio leg only when the lower body is too rarely seen to trust.
  const legsReliable = shin.length >= min && thigh.length >= min;
  let legLen, legSource;
  if (legsReliable) { legLen = med(shin) + med(thigh); legSource = "measured"; }
  else { legLen = Math.max(0.2 * torso, torso / TORSO_FRACTION - torso - headBlock); legSource = "estimated"; }

  // Floor: median lowest ankle when seen often enough; else hips + leg length.
  let floorY, floorSource;
  if (floors.length >= min) { floorY = median(floors); floorSource = "ankles"; }
  else { floorY = (hip ? hip.y : 0) + legLen; floorSource = "estimated"; }

  // Store the body shape as RATIOS to torso (clip-stable) plus a reference torso.
  // The absolute scale is taken per-frame from the fighter's current torso
  // (torsoAt), so the ghost tracks the fighter moving toward / away from the
  // camera instead of being frozen at one clip-wide size.
  return {
    torsoRef: torso,
    legR:       legLen / torso,
    shinR:      (shin.length >= min ? med(shin) : 0.5 * legLen) / torso,  // for the knee→ankle fallback
    headR:      headBlock / torso,
    shoulderWR: med(shoulderW) / torso,
    hipWR:      med(hipW) / torso,
    upperArmR:  (upperArm.length ? med(upperArm) : 0.6 * torso) / torso,
    foreArmR:   (foreArm.length  ? med(foreArm)  : 0.55 * torso) / torso,
    legSource, floorSource,
    floorY,                 // clip-median fallback floor (when this frame's feet are missing)
    feetL: feetL.length ? medPt(feetL) : null,
    feetR: feetR.length ? medPt(feetR) : null,
    hip,
    centerX: centers.length ? median(centers) : pose.width / 2,
    forwardSign: noseDX.length ? (Math.sign(median(noseDX)) || 1) : 1,
  };
}

// The fighter's apparent scale THIS frame: torso (shoulder-centre→hip-centre),
// median-smoothed over a small window so detection jitter doesn't pulse the
// ghost, and clamped near the clip median so a badly foreshortened frame can't
// blow it up or collapse it. Torso is a rigid segment, so it tracks distance
// but stays steady through ducks / leans. Falls back to the clip torso.
function torsoAt(pose, frame, r) {
  const g = cfg.minAnchorConfidence;
  const W = 5;
  const vals = [];
  for (let f = Math.max(0, frame - W); f <= Math.min(pose.n_frames - 1, frame + W); f++) {
    const i = f * 17;
    if (pose.conf[i + J.L_SHOULDER] >= g && pose.conf[i + J.R_SHOULDER] >= g &&
        pose.conf[i + J.L_HIP] >= g && pose.conf[i + J.R_HIP] >= g) {
      const smx = (pose.skeleton[(i + J.L_SHOULDER) * 2] + pose.skeleton[(i + J.R_SHOULDER) * 2]) / 2;
      const smy = (pose.skeleton[(i + J.L_SHOULDER) * 2 + 1] + pose.skeleton[(i + J.R_SHOULDER) * 2 + 1]) / 2;
      const hmx = (pose.skeleton[(i + J.L_HIP) * 2] + pose.skeleton[(i + J.R_HIP) * 2]) / 2;
      const hmy = (pose.skeleton[(i + J.L_HIP) * 2 + 1] + pose.skeleton[(i + J.R_HIP) * 2 + 1]) / 2;
      vals.push(Math.hypot(smx - hmx, smy - hmy));
    }
  }
  if (!vals.length) return r.torsoRef;
  return Math.max(0.6 * r.torsoRef, Math.min(1.6 * r.torsoRef, median(vals)));
}

// Where to plant the stance this frame: the real feet when this frame's ankles
// are confident (and the clip floor came from ankles), else the stable clip
// floor centred under the boxer's hips.
function frameAnchor(pose, frame, r, t) {
  const g = cfg.minAnchorConfidence;
  const i = frame * 17;
  const pt = j => ({ x: pose.skeleton[(i + j) * 2], y: pose.skeleton[(i + j) * 2 + 1] });
  const shin = r.shinR * t;
  const stanceWidth = r.hipWR * t * 1.8;

  // Per foot, best available evidence: the ankle, else the knee extended down by
  // one shin length, else nothing for this side.
  const foot = (ankleJ, kneeJ) => {
    if (pose.conf[i + ankleJ] >= g) return { p: pt(ankleJ), ankle: true };
    if (pose.conf[i + kneeJ]  >= g) { const k = pt(kneeJ); return { p: { x: k.x, y: k.y + shin }, ankle: false }; }
    return null;
  };
  const L = foot(J.L_ANKLE, J.L_KNEE), R = foot(J.R_ANKLE, J.R_KNEE);

  const hipsOk = pose.conf[i + J.L_HIP] >= g && pose.conf[i + J.R_HIP] >= g;
  let centerX;
  if (hipsOk) centerX = (pt(J.L_HIP).x + pt(J.R_HIP).x) / 2;
  else if (L || R) centerX = ((L || R).p.x + (R || L).p.x) / 2;
  else centerX = r.centerX;

  if (L || R) {
    const floorY = Math.max(L ? L.p.y : -Infinity, R ? R.p.y : -Infinity);
    const Lp = L ? L.p : { x: centerX - stanceWidth / 2, y: floorY };
    const Rp = R ? R.p : { x: centerX + stanceWidth / 2, y: floorY };
    const source = (L && L.ankle) || (R && R.ankle) ? "ankles" : "knees";
    return { L: Lp, R: Rp, floorY, centerX, source };
  }
  // Neither ankle nor knee this frame — fall back to the clip floor under the hips.
  const floorY = r.floorY;
  return {
    L: { x: centerX - stanceWidth / 2, y: floorY },
    R: { x: centerX + stanceWidth / 2, y: floorY },
    floorY, centerX, source: r.floorSource,
  };
}

// Stand the real-sized skeleton up on the feet → joint pixel coords + landmark
// Ys. Sized to the fighter's CURRENT-frame torso (clip-stable proportions ×
// per-frame scale) so it tracks him moving toward / away from the camera.
function stanceAt(pose, frame, r) {
  const t = torsoAt(pose, frame, r);
  const s = {
    torso:     t,
    neck:      0.45 * r.headR * t,
    headLen:   0.55 * r.headR * t,
    shoulderW: r.shoulderWR * t,
    hipW:      r.hipWR * t,
    upperArm:  r.upperArmR * t,
    foreArm:   r.foreArmR * t,
  };
  const legLen = r.legR * t;
  const a = frameAnchor(pose, frame, r, t);
  const floorY = a.floorY, centerX = a.centerX;

  const hipY = floorY - legLen;
  const kneeY = floorY - 0.5 * legLen;
  const shoulderY = hipY - s.torso;
  const chinY = shoulderY - s.neck;
  const crownY = chinY - s.headLen;
  const solarY = hipY - SOLAR_OF_TORSO * s.torso;
  const beltY = hipY;
  const H = floorY - crownY;

  // Forward lean: tilt everything above the hips forward (head leans most),
  // pivoting at the hip line. Legs / feet stay planted.
  const lean = (x, y) => x + (y < hipY ? r.forwardSign * LEAN_TAN * (hipY - y) : 0);

  const hw = s.hipW / 2, sw = s.shoulderW / 2;
  // Stand tall on a normal-width stance. The feet are drawn at a normal width
  // under the body (≈ shoulder width) — purely cosmetic, since no zone depends on
  // foot x — while the GROUND height (floorY) still comes from the real feet. The
  // legs stay roughly vertical, so hipY = floorY − legLen is the true standing hip
  // height no matter how wide / sunk the boxer's actual stance is.
  const Lank = { x: centerX - sw, y: floorY };
  const Rank = { x: centerX + sw, y: floorY };
  const joints = {
    Lank, Rank,
    Lkne: { x: (Lank.x + centerX - hw) / 2, y: kneeY }, Rkne: { x: (Rank.x + centerX + hw) / 2, y: kneeY },
    Lhip: { x: centerX - hw, y: hipY }, Rhip: { x: centerX + hw, y: hipY },
    pelvis: { x: centerX, y: hipY },
    chest: { x: lean(centerX, solarY), y: solarY },
    neck: { x: lean(centerX, shoulderY), y: shoulderY },
    Lsh: { x: lean(centerX - sw, shoulderY), y: shoulderY }, Rsh: { x: lean(centerX + sw, shoulderY), y: shoulderY },
    Lel: { x: lean(centerX - sw * 0.7, shoulderY + 0.55 * s.upperArm), y: shoulderY + 0.55 * s.upperArm },
    Rel: { x: lean(centerX + sw * 0.7, shoulderY + 0.55 * s.upperArm), y: shoulderY + 0.55 * s.upperArm },
    Lwr: { x: lean(centerX - sw * 0.45, chinY + 0.15 * s.headLen), y: chinY + 0.15 * s.headLen },   // gloves up by the cheeks
    Rwr: { x: lean(centerX + sw * 0.45, chinY + 0.15 * s.headLen), y: chinY + 0.15 * s.headLen },
    chin: { x: lean(centerX, chinY), y: chinY },
  };
  const headCenterY = (chinY + crownY) / 2;
  const headX = lean(centerX, headCenterY);
  return { joints, crownY, chinY, solarY, beltY, floorY, centerX, headX, H, legLen, floorSource: a.source, headLen: s.headLen };
}

const STANCE_BONES = [
  ["chin", "neck"], ["neck", "chest"], ["chest", "pelvis"],
  ["neck", "Lsh"], ["neck", "Rsh"],
  ["Lsh", "Lel"], ["Lel", "Lwr"], ["Rsh", "Rel"], ["Rel", "Rwr"],
  ["pelvis", "Lhip"], ["pelvis", "Rhip"],
  ["Lhip", "Lkne"], ["Lkne", "Lank"], ["Rhip", "Rkne"], ["Rkne", "Rank"],
];

function boundaries(S) {
  return {
    overHead: S.crownY - cfg.headTopMargin * S.H,
    head:     S.chinY  + cfg.chinMargin    * S.H,
    bodyTop:  S.solarY,
    belt:     S.beltY,
  };
}

function zoneFor(y, B) {
  if (y < B.overHead)  return { key: "over_head",  label: "over the head",   flag: true };
  if (y < B.head)      return { key: "head",       label: "head",            flag: false };
  if (y < B.bodyTop)   return { key: "shoulder",   label: "shoulder height", flag: true };
  if (y <= B.belt)     return { key: "body",       label: "body / stomach",  flag: false };
  return                      { key: "below_belt", label: "below the belt",  flag: true };
}

// ─── per-punch ───────────────────────────────────────────────────────────────

function computePunches(state) {
  const p = pickPose(state);
  const r = getReference(p);
  const N = p.n_frames;
  const dets = (state.labels?.detections || []).filter(d => !skipType(d));
  return dets.map((d, idx) => {
    const sf = Math.max(0, d.start_frame);
    const ef = Math.min(N - 1, d.end_frame);
    const joints = JOINTS_FOR_SIDE[sideFor(d)];

    let bestReach = -Infinity, landFrame = sf;
    for (let f = sf; f <= ef; f++) {
      const w = wristXY(p, f, joints, cfg);
      if (!w) continue;
      const sx = p.skeleton[(f * 17 + joints.shoulder) * 2];
      const sy = p.skeleton[(f * 17 + joints.shoulder) * 2 + 1];
      const reach = Math.hypot(w.x - sx, w.y - sy);
      if (reach > bestReach) { bestReach = reach; landFrame = f; }
    }

    const w = wristXY(p, landFrame, joints, cfg);
    const S = r ? stanceAt(p, landFrame, r) : null;
    let frac = NaN, zone = null;
    if (S && w) {
      zone = zoneFor(w.y, boundaries(S));
      frac = (S.floorY - w.y) / S.H;
    }
    return {
      idx,
      land_frame: landFrame,
      timestamp: d.timestamp,
      punch_type: d.punch_type || "?",
      frac,
      zone,
    };
  });
}

// ─── wrist / side ────────────────────────────────────────────────────────────

function wristXY(pose, frame, joints, cfg) {
  const g = pose.gloveWrists;
  if (g) {
    const [gx, gy] = gloveXY(g, frame, joints.gloveSide);
    const gc = gloveConf(g, frame, joints.gloveSide);
    if (gc >= cfg.minWristConfidence && Number.isFinite(gx) && Number.isFinite(gy)) {
      return { x: gx, y: gy, source: "glove" };
    }
  }
  const px = pose.skeleton[(frame * 17 + joints.wrist) * 2];
  const py = pose.skeleton[(frame * 17 + joints.wrist) * 2 + 1];
  const pc = pose.conf[frame * 17 + joints.wrist];
  if (pc < cfg.minWristConfidence || !Number.isFinite(px)) return null;
  const baked = pose.meta?.wrist_replaced_with_glove === true;
  return { x: px, y: py, source: baked ? "glove" : "pose" };
}

function sideFor(d) {
  const stance = (d.stance === "southpaw" || d.stance === "orthodox") ? d.stance : "orthodox";
  return SIDE_FOR[d.hand]?.[stance] || "L";
}

// Only straights (jab/cross) are scored. Uppercuts travel up to the chin/body by
// design, and hooks land with a bent arm, so the peak-reach impact frame is
// unreliable for both — skip them entirely for now.
function skipType(d) {
  return /uppercut|hook/i.test(d.punch_type || "");
}

function activePunch(state) {
  const f = state.frame;
  const dets = state.labels?.detections || [];
  return dets.find(d => f >= d.start_frame && f <= d.end_frame && !skipType(d)) || null;
}

// ─── render ──────────────────────────────────────────────────────────────────

function renderTable(state) {
  const tableEl = host.querySelector("#hh-table");
  const sumEl = host.querySelector("#hh-summary");
  if (!tableEl) return;

  const punches = computePunches(state);
  if (!punches.length) {
    sumEl.textContent = "No labelled punches loaded.";
    tableEl.innerHTML = "";
    return;
  }
  const scored = punches.filter(p => p.zone);
  const flagged = scored.filter(p => p.zone.flag).length;
  sumEl.innerHTML = scored.length
    ? `<b>${flagged}</b> / ${scored.length} punches flagged off-target` +
      (scored.length < punches.length ? ` · ${punches.length - scored.length} unscored` : "")
    : "No punches could be scored (no standing reference).";

  const rows = punches.map(p => {
    const t = Number.isFinite(p.timestamp) ? p.timestamp.toFixed(2) + "s" : "—";
    if (!p.zone) {
      return `<tr data-frame="${p.land_frame}">
        <td>${p.idx + 1}</td><td>${t}</td><td>${p.punch_type}</td>
        <td colspan="2" class="muted">unscored</td></tr>`;
    }
    const col = p.zone.flag ? COLORS.flag : COLORS.ok;
    return `<tr data-frame="${p.land_frame}" style="cursor:pointer">
      <td>${p.idx + 1}</td><td>${t}</td><td>${p.punch_type}</td>
      <td style="text-align:right">${p.frac.toFixed(2)}</td>
      <td style="color:${col};font-weight:600">${p.zone.label}</td></tr>`;
  }).join("");

  tableEl.innerHTML = `
    <table class="joint-table">
      <thead><tr><th>#</th><th>t</th><th>type</th><th>ht</th><th>zone</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// ─── canvas helpers ──────────────────────────────────────────────────────────

function drawStanceSkeleton(ctx, S, s) {
  ctx.save();
  ctx.globalAlpha = 0.85;
  ctx.strokeStyle = COLORS.ghost;
  ctx.fillStyle = COLORS.ghost;
  ctx.lineWidth = 2.5 * s;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  for (const [a, b] of STANCE_BONES) {
    const A = S.joints[a], B = S.joints[b];
    ctx.beginPath();
    ctx.moveTo(A.x, A.y);
    ctx.lineTo(B.x, B.y);
    ctx.stroke();
  }
  // Head (chin → crown), leaned forward with the upper body.
  ctx.beginPath();
  ctx.arc(S.headX, (S.chinY + S.crownY) / 2, (S.chinY - S.crownY) / 2, 0, Math.PI * 2);
  ctx.stroke();

  for (const name in S.joints) {
    const P = S.joints[name];
    ctx.beginPath();
    ctx.arc(P.x, P.y, 3 * s, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function fillBand(ctx, y0, y1, w, color) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.fillRect(0, y0, w, y1 - y0);
  ctx.restore();
}

function labeledLine(ctx, y, w, text, scale) {
  ctx.save();
  ctx.strokeStyle = COLORS.line;
  ctx.lineWidth = 1 * scale;
  ctx.setLineDash([5 * scale, 4 * scale]);
  ctx.beginPath();
  ctx.moveTo(0, y);
  ctx.lineTo(w, y);
  ctx.stroke();
  ctx.setLineDash([]);
  const fontSize = Math.round(11 * scale);
  ctx.font = `${fontSize}px ui-monospace, monospace`;
  const m = ctx.measureText(text);
  ctx.fillStyle = "rgba(0,0,0,0.6)";
  ctx.fillRect(2 * scale, y - fontSize - 1 * scale, m.width + 6 * scale, fontSize + 4 * scale);
  ctx.fillStyle = COLORS.line;
  ctx.fillText(text, 5 * scale, y - 3 * scale);
  ctx.restore();
}

function drawTag(ctx, x, y, text, color, scale) {
  const fontSize = Math.round(12 * scale);
  ctx.font = `${fontSize}px ui-monospace, monospace`;
  const pad = 3 * scale;
  const m = ctx.measureText(text);
  ctx.fillStyle = "rgba(0,0,0,0.7)";
  ctx.fillRect(x - pad, y - fontSize, m.width + pad * 2, fontSize + 4 * scale);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

// ─── misc ────────────────────────────────────────────────────────────────────

function pickPose(state) {
  return state.poseV6 || state.pose;
}

function median(arr) {
  const a = arr.slice().sort((x, y) => x - y);
  const n = a.length;
  return n % 2 ? a[(n - 1) / 2] : (a[n / 2 - 1] + a[n / 2]) / 2;
}

function setText(id, value, color) {
  const el = host.querySelector("#" + id);
  if (!el) return;
  el.innerHTML = value;
  if (color) el.style.color = color;
}

function seekHack(state, f) {
  const slider = document.getElementById("scrubber");
  if (!slider) return;
  slider.value = f;
  slider.dispatchEvent(new Event("input"));
}
