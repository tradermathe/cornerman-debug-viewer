// Hit-height lens — "did the punch land at a legal, useful height?".
//
// Single-boxer (shadowboxing) footage has no opponent, so we judge the fist
// against an IMAGINARY opponent: someone the same size as the boxer, standing
// in a normal boxing stance, on the same floor. The boxer's own ducks / leans /
// crouches must NOT move the target — a duck lowers your fist, it doesn't lower
// the other guy's head. So the reference is fixed for the clip, not per-frame.
//
// The reference is drawn on the overlay as a ghost stance skeleton — knees bent,
// slight crouch, hands up in guard (NOT bolt upright) — and the zone lines are
// read straight off that skeleton's own landmarks, so the lines and the body
// always agree.
//
// Building the standing reference (camera-distance insensitive, posture-proof):
//   - Scale: torso length (shoulder_mid → hip_mid). A rigid segment — it doesn't
//     change when the knees bend, so it survives ducks. Standing height
//     H = torso / TORSO_FRACTION (torso ≈ 0.29 of standing height). Clip median
//     so a few bad frames don't skew it; tracks apparent size ⇒ camera distance
//     cancels.
//   - Floor: the ankle line (feet stay planted through a duck). Clip median of
//     the lowest confident ankle. Falls back to a hip-based estimate when the
//     feet are cropped out of frame.
//   - Center: clip-median hip-midpoint x — where the ghost stance is drawn.
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
  // Head zone margins (× standing height) around the stance skeleton's crown /
  // chin. Body zone uses the skeleton's solar-plexus / belt landmarks directly.
  headTopMargin: 0.04,   // head zone extends this far ABOVE the crown
  chinMargin:    0.03,   // head zone extends this far BELOW the chin
  // Torso (shoulder_mid→hip_mid) as a fraction of standing height.
  // Standing height H = torso / this. Lower ⇒ taller reconstruction.
  torsoFraction: 0.29,
  minWristConfidence:  0.20,
  minAnchorConfidence: 0.20,
};

// Canonical boxing stance, in fractions of standing height H above the floor
// (f) and x-offset from center in units of H (dx, +right). Knees bent, slight
// crouch (crown ≈ 0.95 H, not 1.0), feet apart + staggered, hands up in guard.
const CANON = {
  joints: {
    chin:   { f: 0.83, dx: 0.00 },
    neck:   { f: 0.80, dx: 0.00 },
    Lsh:    { f: 0.78, dx: -0.11 }, Rsh: { f: 0.78, dx: 0.11 },
    Lel:    { f: 0.60, dx: -0.08 }, Rel: { f: 0.60, dx: 0.08 },  // elbows tucked to the ribs
    Lwr:    { f: 0.82, dx: -0.06 }, Rwr: { f: 0.82, dx: 0.06 },  // gloves up by the cheeks
    chest:  { f: 0.60, dx: 0.00 },                                // solar plexus
    pelvis: { f: 0.52, dx: 0.00 },
    Lhip:   { f: 0.48, dx: -0.07 }, Rhip: { f: 0.48, dx: 0.07 },
    // Knees bent + pushed out, ankles tucked back under — a sunk, athletic
    // stance (not straight legs). Slight L/R asymmetry = staggered feet.
    Lkne:   { f: 0.28, dx: -0.16 }, Rkne: { f: 0.28, dx: 0.15 },
    Lank:   { f: 0.02, dx: -0.11 }, Rank: { f: 0.02, dx: 0.13 },
  },
  bones: [
    ["chin", "neck"], ["neck", "chest"], ["chest", "pelvis"],
    ["neck", "Lsh"], ["neck", "Rsh"],
    ["Lsh", "Lel"], ["Lel", "Lwr"], ["Rsh", "Rel"], ["Rel", "Rwr"],
    ["pelvis", "Lhip"], ["pelvis", "Rhip"],
    ["Lhip", "Lkne"], ["Lkne", "Lank"], ["Rhip", "Rkne"], ["Rkne", "Rank"],
  ],
  crownFrac: 0.95, chinFrac: 0.83, solarPlexusFrac: 0.60, beltFrac: 0.50,
  headCenterFrac: 0.89, headRadiusFrac: 0.06,
};

const COLORS = {
  ok:       "#5fd97a",
  flag:     "#e85a5a",
  bandOk:   "rgba(95,217,122,0.08)",
  bandFlag: "rgba(232,90,90,0.09)",
  line:     "rgba(255,255,255,0.55)",
  ghost:    "#5fd0e6",   // the reference opponent
};

const SIDE_FOR = {
  lead: { orthodox: "L", southpaw: "R" },
  rear: { orthodox: "R", southpaw: "L" },
};
const JOINTS_FOR_SIDE = {
  L: { shoulder: J.L_SHOULDER, wrist: J.L_WRIST, gloveSide: 0 },
  R: { shoulder: J.R_SHOULDER, wrist: J.R_WRIST, gloveSide: 1 },
};

const HIP_ABOVE_FLOOR = 0.53;   // hip joint ≈ 0.53 H above floor (feet-cropped fallback)

let host;
let cfg = { ...DEFAULTS };
let ref = null;        // cached standing reference {H, floorY, centerX, floorSource, torso}
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
        skeleton</b> on the overlay). The boxer's own ducks don't move the zones —
        the reference is anchored to the floor and the boxer's standing height
        (torso-derived), so it's camera-distance insensitive.
        <b style="color:${COLORS.ok}">Head</b> &amp;
        <b style="color:${COLORS.ok}">body</b> are on-target;
        <b style="color:${COLORS.flag}">over the head</b>,
        <b style="color:${COLORS.flag}">shoulder height</b>, and
        <b style="color:${COLORS.flag}">below the belt</b> are flagged.</p>

      <h3>Standing reference</h3>
      <div class="metric-grid">
        <div class="metric"><div class="metric-label">Height px</div><div class="metric-val" id="hh-H">—</div></div>
        <div class="metric"><div class="metric-label">Torso px</div><div class="metric-val" id="hh-torso">—</div></div>
        <div class="metric"><div class="metric-label">Floor</div><div class="metric-val" id="hh-floor">—</div></div>
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
        <span>torso ÷ standing height = <output id="hh-o3">${cfg.torsoFraction.toFixed(3)}</output></span>
        <input type="range" id="hh-s3" min="0.22" max="0.36" step="0.005" value="${cfg.torsoFraction}">
      </label>
      <label class="slider">
        <span>min wrist confidence = <output id="hh-o4">${cfg.minWristConfidence.toFixed(2)}</output></span>
        <input type="range" id="hh-s4" min="0" max="1" step="0.01" value="${cfg.minWristConfidence}">
      </label>
    `;

    const wire = (slider, out, key, fmt = v => v.toFixed(2)) => {
      const s = host.querySelector(slider);
      const o = host.querySelector(out);
      s.addEventListener("input", () => {
        cfg[key] = parseFloat(s.value);
        o.textContent = fmt(cfg[key]);
        refPose = null;             // torsoFraction affects H → rebuild ref
        renderTable(state);
        seekHack(state, state.frame);
      });
    };
    wire("#hh-s1", "#hh-o1", "headTopMargin");
    wire("#hh-s2", "#hh-o2", "chinMargin");
    wire("#hh-s3", "#hh-o3", "torsoFraction", v => v.toFixed(3));
    wire("#hh-s4", "#hh-o4", "minWristConfidence");

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
    const W = ctx.canvas.width;
    const s = state.renderScale || 1;
    const B = boundaries(r);

    // Fixed zone bands — the standing opponent's target map, same every frame.
    fillBand(ctx, -1e4,        B.overHead, W, COLORS.bandFlag); // over head
    fillBand(ctx, B.overHead,  B.head,     W, COLORS.bandOk);   // head
    fillBand(ctx, B.head,      B.bodyTop,  W, COLORS.bandFlag); // shoulder/neck
    fillBand(ctx, B.bodyTop,   B.belt,     W, COLORS.bandOk);   // body
    fillBand(ctx, B.belt,      1e4,        W, COLORS.bandFlag); // below belt

    drawStanceSkeleton(ctx, r, s);

    labeledLine(ctx, B.overHead, W, "over head ↑", s);
    labeledLine(ctx, B.head,     W, "chin",         s);
    labeledLine(ctx, B.bodyTop,  W, "solar plexus", s);
    labeledLine(ctx, B.belt,     W, "belt ↓",       s);

    // Mark the punching fist if the current frame is inside a punch window.
    const punch = activePunch(state);
    if (!punch) return;
    const w = wristXY(p, state.frame, JOINTS_FOR_SIDE[sideFor(punch)], cfg);
    if (!w) return;
    const z = zoneFor((r.floorY - w.y) / r.H);
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
    setText("hh-H", r ? r.H.toFixed(0) : "—");
    setText("hh-torso", r ? r.torso.toFixed(0) : "—");
    setText("hh-floor", r ? r.floorSource : `<span class="muted">unknown</span>`);

    const punch = activePunch(state);
    if (!r || !punch) {
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
    const frac = (r.floorY - w.y) / r.H;
    const z = zoneFor(frac);
    setText("hh-live-height", `${frac.toFixed(2)} H`);
    setText("hh-live-zone", `<span class="${z.flag ? "bad" : "good"}">${z.label}</span> · ${w.source}`);
  },
};

// ─── reference ───────────────────────────────────────────────────────────────

function getReference(pose) {
  if (pose === refPose) return ref;
  ref = buildReference(pose);
  refPose = pose;
  return ref;
}

// Clip-stable standing reference: median torso → standing height, median ankle
// line → floor, median hip x → center. Medians so ducks / dropouts in a few
// frames don't shift it.
function buildReference(pose) {
  const N = pose.n_frames;
  const g = cfg.minAnchorConfidence;
  const torsos = [], floors = [], hipYs = [], centerXs = [];

  for (let f = 0; f < N; f++) {
    const i = f * 17;
    const lsc = pose.conf[i + J.L_SHOULDER], rsc = pose.conf[i + J.R_SHOULDER];
    const lhc = pose.conf[i + J.L_HIP],      rhc = pose.conf[i + J.R_HIP];
    if (lsc >= g && rsc >= g && lhc >= g && rhc >= g) {
      const shx = (pose.skeleton[(i + J.L_SHOULDER) * 2]     + pose.skeleton[(i + J.R_SHOULDER) * 2])     / 2;
      const shy = (pose.skeleton[(i + J.L_SHOULDER) * 2 + 1] + pose.skeleton[(i + J.R_SHOULDER) * 2 + 1]) / 2;
      const hpx = (pose.skeleton[(i + J.L_HIP) * 2]     + pose.skeleton[(i + J.R_HIP) * 2])     / 2;
      const hpy = (pose.skeleton[(i + J.L_HIP) * 2 + 1] + pose.skeleton[(i + J.R_HIP) * 2 + 1]) / 2;
      torsos.push(Math.hypot(shx - hpx, shy - hpy));
      hipYs.push(hpy);
      centerXs.push((shx + hpx) / 2);
    }
    const laC = pose.conf[i + J.L_ANKLE], raC = pose.conf[i + J.R_ANKLE];
    let foot = -Infinity;
    if (laC >= g) foot = Math.max(foot, pose.skeleton[(i + J.L_ANKLE) * 2 + 1]);
    if (raC >= g) foot = Math.max(foot, pose.skeleton[(i + J.R_ANKLE) * 2 + 1]);
    if (foot > -Infinity) floors.push(foot);
  }

  if (!torsos.length) return null;
  const torso = median(torsos);
  const H = torso / cfg.torsoFraction;
  const centerX = median(centerXs);

  let floorY, floorSource;
  if (floors.length) {
    floorY = median(floors);
    floorSource = "ankles";
  } else if (hipYs.length) {
    floorY = median(hipYs) + HIP_ABOVE_FLOOR * H;
    floorSource = "hip-estimate";
  } else {
    return null;
  }
  return { H, floorY, centerX, floorSource, torso };
}

// Zone thresholds as fractions of H, read off the stance skeleton's landmarks.
function zoneThresholds() {
  return {
    overHead:   CANON.crownFrac + cfg.headTopMargin,
    headBottom: CANON.chinFrac  - cfg.chinMargin,
    bodyTop:    CANON.solarPlexusFrac,
    belt:       CANON.beltFrac,
  };
}

function boundaries(r) {
  const T = zoneThresholds();
  return {
    overHead: r.floorY - T.overHead   * r.H,
    head:     r.floorY - T.headBottom * r.H,
    bodyTop:  r.floorY - T.bodyTop    * r.H,
    belt:     r.floorY - T.belt       * r.H,
  };
}

function zoneFor(frac) {
  const T = zoneThresholds();
  if (frac > T.overHead)   return { key: "over_head",  label: "over the head",   flag: true };
  if (frac >= T.headBottom)return { key: "head",       label: "head",            flag: false };
  if (frac >= T.bodyTop)   return { key: "shoulder",   label: "shoulder height", flag: true };
  if (frac >= T.belt)      return { key: "body",       label: "body / stomach",  flag: false };
  return                          { key: "below_belt", label: "below the belt",  flag: true };
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
    let frac = NaN, zone = null;
    if (r && w) {
      frac = (r.floorY - w.y) / r.H;
      zone = zoneFor(frac);
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

// The reference opponent: a canonical boxing stance scaled to H, anchored at the
// floor and the boxer's median center x. Drawn translucent so the boxer's own
// skeleton stays readable on top.
function drawStanceSkeleton(ctx, r, s) {
  const pt = (name) => {
    const j = CANON.joints[name];
    return { x: r.centerX + j.dx * r.H, y: r.floorY - j.f * r.H };
  };
  ctx.save();
  ctx.globalAlpha = 0.85;
  ctx.strokeStyle = COLORS.ghost;
  ctx.fillStyle = COLORS.ghost;
  ctx.lineWidth = 2.5 * s;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  for (const [a, b] of CANON.bones) {
    const A = pt(a), B = pt(b);
    ctx.beginPath();
    ctx.moveTo(A.x, A.y);
    ctx.lineTo(B.x, B.y);
    ctx.stroke();
  }

  // Head (chin → crown).
  ctx.beginPath();
  ctx.arc(r.centerX, r.floorY - CANON.headCenterFrac * r.H, CANON.headRadiusFrac * r.H, 0, Math.PI * 2);
  ctx.stroke();

  // Joint dots.
  for (const name in CANON.joints) {
    const P = pt(name);
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
