// Hit-height lens — "did the punch land at a legal, useful height?".
//
// Single-boxer (shadowboxing) footage has no opponent, so "hit height" is the
// height the PUNCHING FIST reaches at peak extension. A good punch lands at the
// head or the body (stomach / solar plexus). We flag three off-target bands:
//
//   over the head   — fist peaks above the top of the skull (missed high / illegal)
//   shoulder height — the dead zone between head and body (neck / shoulders)
//   below the belt  — fist drops past the hip line (foul / wasted low)
//
// Vertical ordering of the zones (image y grows downward):
//
//   y < overHeadCut           OVER THE HEAD   flag (too high)
//   overHeadCut … headFloor   HEAD            ok
//   headFloor  … bodyCeil     SHOULDER        flag (dead zone)
//   bodyCeil   … beltLine     BODY / STOMACH  ok
//   y ≥ beltLine              BELOW THE BELT  flag (too low)
//
// Camera-distance insensitive: every boundary is anchored to the boxer's own
// landmarks in the SAME frame as the fist — nose for the skull, the shoulder
// midpoint for head/shoulder split, the hip midpoint for the belt — and the
// fuzzy offsets are scaled by torso height (shoulder_mid → hip_mid). It's all
// ratios within one frame, so apparent size (how far the camera is, where the
// boxer stands) cancels out. The torso unit is the boxer's standing-height
// proxy: it tracks the body, not the pixels.
//
// Landing frame = the most-extended frame inside the punch window (max
// |shoulder→wrist| / torso), the same "peak" the arm_extension lens reads at.
// Punching side maps from (hand, stance) exactly like guard_drop / arm_extension.
// Wrist source prefers the v6 glove-baked wrist (the iOS app's live source),
// falls back to a legacy glove sidecar, then the pose wrist.

import { J, torsoHeight } from "../skeleton.js";
import { gloveXY, gloveConf } from "../pose-loader.js";

const DEFAULTS = {
  // Boundary offsets, all in torso units (shoulder_mid → hip_mid vertical).
  // Anchored to the joint named in each comment so they track the body.
  headTopAboveNose:    0.45,   // skull top ≈ nose − this·torso   → over-head cut
  chinAboveShoulder:   0.20,   // chin ≈ shoulder − this·torso    → head / shoulder split
  chestBelowShoulder:  0.35,   // top of body ≈ shoulder + this·torso → shoulder / body split
  beltBelowHip:        0.00,   // belt ≈ hip + this·torso         → body / below-belt split
  minWristConfidence:  0.20,
  minAnchorConfidence: 0.20,   // nose / shoulders / hips must clear this to judge
};

const COLORS = {
  ok:        "#5fd97a",
  flag:      "#e85a5a",
  bandOk:    "rgba(95,217,122,0.07)",
  bandFlag:  "rgba(232,90,90,0.09)",
  line:      "rgba(255,255,255,0.55)",
  fist:      "#ffd95c",
};

// (hand, stance) → anatomical side. Same table arm_extension / guard_drop use.
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

export const HitHeightRule = {
  id: "hit_height",
  label: "Hit height",

  skeletonStyle() {
    return {
      boneColor: "rgba(255,255,255,0.25)",
      boneWidth: 1.5,
      jointRadius: 3,
      highlightJoints: new Set([
        J.NOSE, J.L_SHOULDER, J.R_SHOULDER, J.L_HIP, J.R_HIP,
        J.L_WRIST, J.R_WRIST,
      ]),
    };
  },

  mount(_host, state) {
    host = _host;
    host.innerHTML = `
      <h2>Hit height</h2>
      <p class="hint">Where the punching fist peaks, relative to the boxer's own
        body. Head &amp; body are on-target; <b style="color:${COLORS.flag}">over the
        head</b>, <b style="color:${COLORS.flag}">shoulder height</b>, and
        <b style="color:${COLORS.flag}">below the belt</b> are flagged. All
        boundaries are torso-normalised, so the read is camera-distance
        insensitive.</p>

      <h3>Live (current frame)</h3>
      <div class="metric-grid">
        <div class="metric"><div class="metric-label">Torso px</div><div class="metric-val" id="hh-torso">—</div></div>
        <div class="metric"><div class="metric-label">Anchors</div><div class="metric-val" id="hh-anchors">—</div></div>
      </div>
      <div class="metric">
        <div class="metric-label">Active punch — fist height</div>
        <div class="metric-val" id="hh-live-height">—</div>
        <div class="metric-sub" id="hh-live-zone"></div>
      </div>

      <h3>Per-punch</h3>
      <div id="hh-summary" class="hint"></div>
      <div id="hh-table"></div>

      <h3>Boundaries <span class="hint">(torso units)</span></h3>
      <label class="slider">
        <span>skull top above nose = <output id="hh-o1">${cfg.headTopAboveNose.toFixed(2)}</output></span>
        <input type="range" id="hh-s1" min="0" max="1" step="0.01" value="${cfg.headTopAboveNose}">
      </label>
      <label class="slider">
        <span>chin above shoulder = <output id="hh-o2">${cfg.chinAboveShoulder.toFixed(2)}</output></span>
        <input type="range" id="hh-s2" min="0" max="0.8" step="0.01" value="${cfg.chinAboveShoulder}">
      </label>
      <label class="slider">
        <span>body top below shoulder = <output id="hh-o3">${cfg.chestBelowShoulder.toFixed(2)}</output></span>
        <input type="range" id="hh-s3" min="0" max="1" step="0.01" value="${cfg.chestBelowShoulder}">
      </label>
      <label class="slider">
        <span>belt below hip = <output id="hh-o4">${cfg.beltBelowHip.toFixed(2)}</output></span>
        <input type="range" id="hh-s4" min="-0.3" max="0.5" step="0.01" value="${cfg.beltBelowHip}">
      </label>
      <label class="slider">
        <span>min wrist confidence = <output id="hh-o5">${cfg.minWristConfidence.toFixed(2)}</output></span>
        <input type="range" id="hh-s5" min="0" max="1" step="0.01" value="${cfg.minWristConfidence}">
      </label>
    `;

    const wire = (slider, out, key) => {
      const s = host.querySelector(slider);
      const o = host.querySelector(out);
      s.addEventListener("input", () => {
        cfg[key] = parseFloat(s.value);
        o.textContent = cfg[key].toFixed(2);
        renderTable(state);
        seekHack(state, state.frame);   // refresh overlay + live readout
      });
    };
    wire("#hh-s1", "#hh-o1", "headTopAboveNose");
    wire("#hh-s2", "#hh-o2", "chinAboveShoulder");
    wire("#hh-s3", "#hh-o3", "chestBelowShoulder");
    wire("#hh-s4", "#hh-o4", "beltBelowHip");
    wire("#hh-s5", "#hh-o5", "minWristConfidence");

    // Click a punch row to seek to its landing frame.
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
    const f = state.frame;
    const W = ctx.canvas.width;
    const s = state.renderScale || 1;

    const a = anchors(p, f);
    if (!a) return;
    const B = boundaries(a);

    // Faint zone bands across the frame, anchored to this frame's anatomy.
    fillBand(ctx, -1e4,         B.overHeadCut, W, COLORS.bandFlag); // over head
    fillBand(ctx, B.overHeadCut, B.headFloor,  W, COLORS.bandOk);   // head
    fillBand(ctx, B.headFloor,   B.bodyCeil,   W, COLORS.bandFlag); // shoulder
    fillBand(ctx, B.bodyCeil,    B.beltLine,   W, COLORS.bandOk);   // body
    fillBand(ctx, B.beltLine,    1e4,          W, COLORS.bandFlag); // below belt

    // Boundary lines with labels.
    labeledLine(ctx, B.overHeadCut, W, "over head ↑", s);
    labeledLine(ctx, B.headFloor,   W, "head / shoulder", s);
    labeledLine(ctx, B.bodyCeil,    W, "shoulder / body", s);
    labeledLine(ctx, B.beltLine,    W, "belt ↓", s);

    // If the current frame is inside a punch window, mark the punching fist.
    const punch = activePunch(state);
    if (!punch) return;
    const side = sideFor(punch);
    const w = wristXY(p, f, JOINTS_FOR_SIDE[side], cfg);
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
    const f = state.frame;
    const a = anchors(p, f);

    setText("hh-torso", a ? a.torso.toFixed(0) : "—");
    setText("hh-anchors", a
      ? `nose ${a.nc.toFixed(2)} · sho ${a.sc.toFixed(2)} · hip ${a.hc.toFixed(2)}`
      : `<span class="muted">low confidence</span>`);

    const punch = activePunch(state);
    if (!a || !punch) {
      setText("hh-live-height", "—");
      setText("hh-live-zone", punch ? "" : `<span class="muted">no punch at this frame</span>`);
      return;
    }
    const side = sideFor(punch);
    const w = wristXY(p, f, JOINTS_FOR_SIDE[side], cfg);
    if (!w) {
      setText("hh-live-height", "—");
      setText("hh-live-zone", `<span class="muted">no wrist signal</span>`);
      return;
    }
    const B = boundaries(a);
    const z = zoneFor(w.y, B);
    const h = (a.shoulderY - w.y) / a.torso;   // +above shoulders, −below
    setText("hh-live-height", `${h >= 0 ? "+" : ""}${h.toFixed(2)} torso`);
    setText("hh-live-zone",
      `<span class="${z.flag ? "bad" : "good"}">${z.label}</span> · ${w.source}`);
  },
};

// ─── compute ─────────────────────────────────────────────────────────────────

function computePunches(state) {
  const p = pickPose(state);
  const N = p.n_frames;
  const detections = state.labels?.detections || [];
  return detections.map((d, idx) => {
    const sf = Math.max(0, d.start_frame);
    const ef = Math.min(N - 1, d.end_frame);
    const side = sideFor(d);
    const joints = JOINTS_FOR_SIDE[side];

    // Landing = most-extended frame in the window (max |sh→wr| / torso).
    let bestReach = -Infinity, landFrame = sf;
    for (let f = sf; f <= ef; f++) {
      const a = anchors(p, f);
      const w = wristXY(p, f, joints, cfg);
      if (!a || !w) continue;
      const sx = p.skeleton[(f * 17 + joints.shoulder) * 2];
      const sy = p.skeleton[(f * 17 + joints.shoulder) * 2 + 1];
      const reach = Math.hypot(w.x - sx, w.y - sy) / a.torso;
      if (reach > bestReach) { bestReach = reach; landFrame = f; }
    }

    const a = anchors(p, landFrame);
    const w = a ? wristXY(p, landFrame, joints, cfg) : null;
    let zone = null, height = NaN;
    if (a && w) {
      zone = zoneFor(w.y, boundaries(a));
      height = (a.shoulderY - w.y) / a.torso;
    }
    return {
      idx,
      land_frame: landFrame,
      timestamp: d.timestamp,
      punch_type: d.punch_type || "?",
      hand: d.hand || "?",
      side,
      height,
      zone,
    };
  });
}

// ─── geometry ────────────────────────────────────────────────────────────────

// Body anchors at a frame, or null if any anchor is too low-confidence to trust.
function anchors(pose, frame) {
  const i = frame * 17;
  const nc = pose.conf[i + J.NOSE];
  const lsc = pose.conf[i + J.L_SHOULDER], rsc = pose.conf[i + J.R_SHOULDER];
  const lhc = pose.conf[i + J.L_HIP],      rhc = pose.conf[i + J.R_HIP];
  const sc = Math.min(lsc, rsc), hc = Math.min(lhc, rhc);
  const g = cfg.minAnchorConfidence;
  if (nc < g || sc < g || hc < g) return null;

  const noseY = pose.skeleton[(i + J.NOSE) * 2 + 1];
  const shoulderY = (pose.skeleton[(i + J.L_SHOULDER) * 2 + 1] +
                     pose.skeleton[(i + J.R_SHOULDER) * 2 + 1]) / 2;
  const hipY = (pose.skeleton[(i + J.L_HIP) * 2 + 1] +
                pose.skeleton[(i + J.R_HIP) * 2 + 1]) / 2;
  const torso = Math.max(1e-6, torsoHeight(pose, frame));
  return { noseY, shoulderY, hipY, torso, nc, sc, hc };
}

function boundaries(a) {
  return {
    overHeadCut: a.noseY     - cfg.headTopAboveNose   * a.torso,
    headFloor:   a.shoulderY - cfg.chinAboveShoulder  * a.torso,
    bodyCeil:    a.shoulderY + cfg.chestBelowShoulder * a.torso,
    beltLine:    a.hipY      + cfg.beltBelowHip        * a.torso,
  };
}

function zoneFor(y, B) {
  if (y < B.overHeadCut) return { key: "over_head", label: "over the head",  flag: true };
  if (y < B.headFloor)   return { key: "head",      label: "head",           flag: false };
  if (y < B.bodyCeil)    return { key: "shoulder",  label: "shoulder height", flag: true };
  if (y < B.beltLine)    return { key: "body",      label: "body / stomach", flag: false };
  return                        { key: "below_belt", label: "below the belt", flag: true };
}

// Prefer v6 glove-baked wrist (iOS live source), then legacy glove sidecar,
// then the pose wrist. Mirrors arm_extension.wristXY, gated by confidence.
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
      (scored.length < punches.length ? ` · ${punches.length - scored.length} unscored (low conf)` : "")
    : "No punches could be scored (low confidence).";

  const rows = punches.map(p => {
    const t = Number.isFinite(p.timestamp) ? p.timestamp.toFixed(2) + "s" : "—";
    if (!p.zone) {
      return `<tr data-frame="${p.land_frame}">
        <td>${p.idx + 1}</td><td>${t}</td><td>${p.punch_type}</td>
        <td colspan="2" class="muted">low conf</td></tr>`;
    }
    const col = p.zone.flag ? COLORS.flag : COLORS.ok;
    const h = `${p.height >= 0 ? "+" : ""}${p.height.toFixed(2)}`;
    return `<tr data-frame="${p.land_frame}" style="cursor:pointer">
      <td>${p.idx + 1}</td><td>${t}</td><td>${p.punch_type}</td>
      <td style="text-align:right">${h}</td>
      <td style="color:${col};font-weight:600">${p.zone.label}</td></tr>`;
  }).join("");

  tableEl.innerHTML = `
    <table class="joint-table">
      <thead><tr><th>#</th><th>t</th><th>type</th><th>ht</th><th>zone</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// ─── canvas helpers ──────────────────────────────────────────────────────────

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
  ctx.fillStyle = "rgba(0,0,0,0.6)";
  const m = ctx.measureText(text);
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

// ─── misc helpers ────────────────────────────────────────────────────────────

function pickPose(state) {
  return state.poseV6 || state.pose;
}

function setText(id, value, color) {
  const el = host.querySelector("#" + id);
  if (!el) return;
  el.innerHTML = value;
  if (color) el.style.color = color;
}

// Re-trigger a viewer redraw without exposing internal redraw() — set the
// scrubber and dispatch the same event the scrubber handler listens for.
function seekHack(state, f) {
  const slider = document.getElementById("scrubber");
  if (!slider) return;
  slider.value = f;
  slider.dispatchEvent(new Event("input"));
}
