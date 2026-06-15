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
  chinMargin:    0.03,   // head zone extends this far BELOW the chin
  minWristConfidence:  0.20,
  minAnchorConfidence: 0.20,
};

// Stance shaping (applied to the boxer's REAL measured segment lengths — these
// are pose angles / sub-segment ratios, not body-size assumptions).
const KNEE_BEND      = 0.95;  // legs ~95% extended → slight athletic sink
const SOLAR_OF_TORSO = 0.55;  // solar plexus ≈ 55% up the torso from the hips
const NECK_OF_S2N    = 0.50;  // shoulder→chin ≈ 0.5 × shoulder→nose (real)
const HEAD_OF_S2N    = 1.30;  // chin→crown   ≈ 1.3 × shoulder→nose (real)

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
    setText("hh-leg", r ? (r.seg.shin + r.seg.thigh).toFixed(0) : "—");
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

// Clip-median of every limb length the boxer actually shows, plus where the feet
// typically are. Medians so jitter / dropouts in a few frames don't wobble it.
function buildReference(pose) {
  const N = pose.n_frames;
  const g = cfg.minAnchorConfidence;
  const acc = { shin: [], thigh: [], torso: [], shoulderW: [], hipW: [], s2n: [], upperArm: [], foreArm: [] };
  const feetL = [], feetR = [], hips = [];

  const xy = (f, j) => [pose.skeleton[(f * 17 + j) * 2], pose.skeleton[(f * 17 + j) * 2 + 1]];
  const ok = (f, j) => pose.conf[f * 17 + j] >= g;
  const seg = (f, a, b) => { const [ax, ay] = xy(f, a), [bx, by] = xy(f, b); return Math.hypot(ax - bx, ay - by); };

  for (let f = 0; f < N; f++) {
    if (ok(f, J.L_KNEE) && ok(f, J.L_ANKLE)) acc.shin.push(seg(f, J.L_KNEE, J.L_ANKLE));
    if (ok(f, J.R_KNEE) && ok(f, J.R_ANKLE)) acc.shin.push(seg(f, J.R_KNEE, J.R_ANKLE));
    if (ok(f, J.L_HIP) && ok(f, J.L_KNEE)) acc.thigh.push(seg(f, J.L_HIP, J.L_KNEE));
    if (ok(f, J.R_HIP) && ok(f, J.R_KNEE)) acc.thigh.push(seg(f, J.R_HIP, J.R_KNEE));
    if (ok(f, J.L_SHOULDER) && ok(f, J.L_ELBOW)) acc.upperArm.push(seg(f, J.L_SHOULDER, J.L_ELBOW));
    if (ok(f, J.R_SHOULDER) && ok(f, J.R_ELBOW)) acc.upperArm.push(seg(f, J.R_SHOULDER, J.R_ELBOW));
    if (ok(f, J.L_ELBOW) && ok(f, J.L_WRIST)) acc.foreArm.push(seg(f, J.L_ELBOW, J.L_WRIST));
    if (ok(f, J.R_ELBOW) && ok(f, J.R_WRIST)) acc.foreArm.push(seg(f, J.R_ELBOW, J.R_WRIST));

    if (ok(f, J.L_SHOULDER) && ok(f, J.R_SHOULDER) && ok(f, J.L_HIP) && ok(f, J.R_HIP)) {
      const [lsx, lsy] = xy(f, J.L_SHOULDER), [rsx, rsy] = xy(f, J.R_SHOULDER);
      const [lhx, lhy] = xy(f, J.L_HIP),      [rhx, rhy] = xy(f, J.R_HIP);
      const smx = (lsx + rsx) / 2, smy = (lsy + rsy) / 2;
      const hmx = (lhx + rhx) / 2, hmy = (lhy + rhy) / 2;
      acc.torso.push(Math.hypot(smx - hmx, smy - hmy));
      acc.shoulderW.push(Math.hypot(lsx - rsx, lsy - rsy));
      acc.hipW.push(Math.hypot(lhx - rhx, lhy - rhy));
      hips.push({ x: hmx, y: hmy });
      if (ok(f, J.NOSE)) {
        const [nx, ny] = xy(f, J.NOSE);
        acc.s2n.push(Math.hypot(smx - nx, smy - ny));
      }
    }
    if (ok(f, J.L_ANKLE)) { const [x, y] = xy(f, J.L_ANKLE); feetL.push({ x, y }); }
    if (ok(f, J.R_ANKLE)) { const [x, y] = xy(f, J.R_ANKLE); feetR.push({ x, y }); }
  }

  if (!acc.torso.length) return null;
  const med = a => a.length ? median(a) : 0;
  const torso = med(acc.torso);
  const s2n = acc.s2n.length ? med(acc.s2n) : 0.45 * torso;   // head from real nose; fall back to torso
  const seg2 = {
    shin:      acc.shin.length  ? med(acc.shin)  : 0.85 * torso,
    thigh:     acc.thigh.length ? med(acc.thigh) : 0.85 * torso,
    torso,
    shoulderW: med(acc.shoulderW),
    hipW:      med(acc.hipW),
    upperArm:  acc.upperArm.length ? med(acc.upperArm) : 0.6 * torso,
    foreArm:   acc.foreArm.length  ? med(acc.foreArm)  : 0.55 * torso,
    neck:    NECK_OF_S2N * s2n,
    headLen: HEAD_OF_S2N * s2n,
  };

  const medPt = arr => ({ x: median(arr.map(p => p.x)), y: median(arr.map(p => p.y)) });
  const haveFeet = feetL.length || feetR.length;
  const stanceWidth = seg2.hipW * 1.8;   // feet a touch wider than the hips

  return {
    seg: seg2,
    stanceWidth,
    feetL: feetL.length ? medPt(feetL) : null,
    feetR: feetR.length ? medPt(feetR) : null,
    hip:   hips.length  ? medPt(hips)  : null,
    haveFeet,
  };
}

// Real feet for this frame: where we actually see them, with a clip-median
// fallback per foot, and a hip-derived estimate when the feet are never visible.
function frameFeet(pose, frame, r) {
  const g = cfg.minAnchorConfidence;
  const i = frame * 17;
  const pt = j => ({ x: pose.skeleton[(i + j) * 2], y: pose.skeleton[(i + j) * 2 + 1] });
  let L = pose.conf[i + J.L_ANKLE] >= g ? pt(J.L_ANKLE) : (r.feetL || null);
  let R = pose.conf[i + J.R_ANKLE] >= g ? pt(J.R_ANKLE) : (r.feetR || null);
  let source = "ankles";

  if (!L && !R) {
    if (!r.hip) return null;
    const floorY = r.hip.y + (r.seg.shin + r.seg.thigh);   // estimate floor under the hips
    L = { x: r.hip.x - r.stanceWidth / 2, y: floorY };
    R = { x: r.hip.x + r.stanceWidth / 2, y: floorY };
    return { L, R, source: "hip-estimate" };
  }
  if (!L) { L = { x: R.x - r.stanceWidth, y: R.y }; source = "1 foot"; }
  if (!R) { R = { x: L.x + r.stanceWidth, y: L.y }; source = "1 foot"; }
  return { L, R, source };
}

// Stand the real-sized skeleton up on the real feet → joint pixel coords +
// landmark Ys. Knees slightly bent (athletic stance), guard up.
function stanceAt(pose, frame, r) {
  const feet = frameFeet(pose, frame, r);
  if (!feet) return null;
  const s = r.seg;
  const floorY = Math.max(feet.L.y, feet.R.y);
  const centerX = (feet.L.x + feet.R.x) / 2;

  const kneeY = floorY - KNEE_BEND * s.shin;
  const hipY = kneeY - KNEE_BEND * s.thigh;
  const shoulderY = hipY - s.torso;
  const chinY = shoulderY - s.neck;
  const crownY = chinY - s.headLen;
  const solarY = hipY - SOLAR_OF_TORSO * s.torso;
  const beltY = hipY;
  const H = floorY - crownY;

  const hw = s.hipW / 2, sw = s.shoulderW / 2;
  const joints = {
    Lank: feet.L, Rank: feet.R,
    Lkne: { x: feet.L.x, y: kneeY }, Rkne: { x: feet.R.x, y: kneeY },
    Lhip: { x: centerX - hw, y: hipY }, Rhip: { x: centerX + hw, y: hipY },
    pelvis: { x: centerX, y: hipY },
    chest: { x: centerX, y: solarY },
    neck: { x: centerX, y: shoulderY },
    Lsh: { x: centerX - sw, y: shoulderY }, Rsh: { x: centerX + sw, y: shoulderY },
    Lel: { x: centerX - sw * 0.7, y: shoulderY + 0.55 * s.upperArm },
    Rel: { x: centerX + sw * 0.7, y: shoulderY + 0.55 * s.upperArm },
    Lwr: { x: centerX - sw * 0.45, y: chinY + 0.15 * s.headLen },   // gloves up by the cheeks
    Rwr: { x: centerX + sw * 0.45, y: chinY + 0.15 * s.headLen },
    chin: { x: centerX, y: chinY },
  };
  return { joints, crownY, chinY, solarY, beltY, floorY, centerX, H, floorSource: feet.source, headLen: s.headLen };
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
  const dets = state.labels?.detections || [];
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

function activePunch(state) {
  const f = state.frame;
  const dets = state.labels?.detections || [];
  return dets.find(d => f >= d.start_frame && f <= d.end_frame) || null;
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
  // Head (chin → crown).
  ctx.beginPath();
  ctx.arc(S.centerX, (S.chinY + S.crownY) / 2, (S.chinY - S.crownY) / 2, 0, Math.PI * 2);
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
