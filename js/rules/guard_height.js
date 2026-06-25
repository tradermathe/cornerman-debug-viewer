// Guard-height lens — "is a hand dropping below guard while not punching?".
//
// Two jobs:
//   1. VISUALISE the vertical guard stack on the overlay:
//        - Solid line at nose height.
//        - Dashed "target" line just below the nose — where the wrists should sit.
//        - One line per wrist at its current height, tinted red when it's flagged low.
//   2. FLAG too-low hands across the clip, EXCLUDING punch windows, and roll the
//      result up into a per-hand "guard low %".
//
// How "just below nose" is configured: the target line is
//     target_y = nose_y + targetOffset * torso_height
// where torso_height is the shoulder→hip distance (same normaliser
// guard_drop.py uses), so the offset scales with the boxer's size and
// distance from the camera instead of being a fixed pixel count.
// Default targetOffset = 0.25: with glove tops at nose height the wrist
// joint sits roughly a glove-length lower (0.10 was glove-top height, not
// wrist height). Kept just above guard_drop's guard_low_threshold (0.30)
// so "at target" here still counts as guard-up there. Tune with the slider.
//
// Punch exclusion is PER-HAND, not per-frame: a wrist is only ignored while
// THAT wrist is the one throwing (mapped from the detection's hand+stance via
// SIDE_FOR, same as hit_height / guard_drop). This keeps the extended punching
// fist from being flagged as "low", while still catching the classic mistake of
// the guard hand dropping during the other hand's punch (e.g. the left dropping
// on a right cross). A small ± frame pad covers the hand still returning home.

import { J, torsoHeight } from "../skeleton.js";
import { activeDetections } from "./_detections.js";

const DEFAULTS = {
  targetOffset: 0.25,          // fraction of torso height below the nose
  minWristConfidence: 0.30,
  punchPad: 3,                 // frames of pad added each side of a punch window
};

const COLORS = {
  nose:    "#7ec8ff",
  target:  "#5fd97a",
  l_wrist: "#ff8a5c",
  r_wrist: "#ffd95c",
  low:     "#e85a5a",
  up:      "#5fd97a",
  muted:   "rgba(255,255,255,0.45)",
};

// Per-frame, per-hand status codes (also drive the timeline colours).
const ST = { GATED: 0, EXCLUDED: 1, UP: 2, LOW: 3 };

// Detection hand + stance → anatomical wrist side. Orthodox leads with the
// left; southpaw leads with the right. Mirrors guard_drop.py's GUARD_JOINTS.
const SIDE_FOR = {
  lead: { orthodox: "L", southpaw: "R" },
  rear: { orthodox: "R", southpaw: "L" },
};

let host;
let cfg = { ...DEFAULTS };
let cache = null;   // memoised whole-clip flagging (see getData)

export const GuardHeightRule = {
  id: "guard_height",
  label: "Guard height",

  skeletonStyle() {
    return {
      boneColor: "rgba(255,255,255,0.25)",
      boneWidth: 1.5,
      jointRadius: 3,
      highlightJoints: new Set([J.NOSE, J.L_WRIST, J.R_WRIST]),
    };
  },

  mount(_host, state) {
    host = _host;
    host.innerHTML = `
      <h2>Guard height</h2>
      <p class="hint">Lines: <span style="color:${COLORS.nose}">nose (solid)</span> ·
        <span style="color:${COLORS.target}">wrist target (dashed)</span> ·
        <span style="color:${COLORS.l_wrist}">L wrist</span> ·
        <span style="color:${COLORS.r_wrist}">R wrist</span>.
        Target = nose_y + offset × torso height (shoulder→hip), so it scales
        with the boxer instead of being a pixel constant. A wrist below the
        target while it isn't punching is flagged <span style="color:${COLORS.low}">low</span>.</p>

      <h3>Guard low % (punches excluded)</h3>
      <div class="metric-grid">
        <div class="metric">
          <div class="metric-label">L wrist</div>
          <div class="metric-val" id="gh-l-low">—</div>
        </div>
        <div class="metric">
          <div class="metric-label">R wrist</div>
          <div class="metric-val" id="gh-r-low">—</div>
        </div>
        <div class="metric">
          <div class="metric-label">Both</div>
          <div class="metric-val" id="gh-tot-low">—</div>
        </div>
      </div>
      <p class="hint" id="gh-note"></p>

      <h3>Guard timeline</h3>
      <p class="hint">Per frame, per hand:
        <span style="color:${COLORS.up}">guard up</span> ·
        <span style="color:${COLORS.low}">low</span> ·
        <span style="color:${COLORS.muted}">punching (excluded)</span> ·
        <span style="color:#f5b945">low conf</span>. Click to seek.</p>
      <canvas id="gh-timeline" width="320" height="48" style="cursor:pointer;width:100%"></canvas>

      <h3>Wrist vs target (torso units, − = above)</h3>
      <div class="metric-grid">
        <div class="metric">
          <div class="metric-label">L wrist</div>
          <div class="metric-val" id="l-target-dist">—</div>
          <div class="metric-sub" id="l-target-verdict"></div>
        </div>
        <div class="metric">
          <div class="metric-label">R wrist</div>
          <div class="metric-val" id="r-target-dist">—</div>
          <div class="metric-sub" id="r-target-verdict"></div>
        </div>
      </div>

      <h3>Config</h3>
      <label class="slider">
        <span>target_offset = <output id="to-out">${cfg.targetOffset.toFixed(2)}</output> × torso below nose</span>
        <input type="range" id="to-slider" min="0" max="0.40" step="0.01" value="${cfg.targetOffset}">
      </label>
      <label class="slider">
        <span>min_wrist_confidence = <output id="mwc-out">${cfg.minWristConfidence.toFixed(2)}</output></span>
        <input type="range" id="mwc-slider" min="0" max="1" step="0.01" value="${cfg.minWristConfidence}">
      </label>
      <label class="slider">
        <span>punch_pad = <output id="pp-out">${cfg.punchPad}</output> frames each side</span>
        <input type="range" id="pp-slider" min="0" max="15" step="1" value="${cfg.punchPad}">
      </label>
    `;

    const wire = (slider, out, key, fmt = v => v.toFixed(2)) => {
      const s = host.querySelector(slider);
      const o = host.querySelector(out);
      s.addEventListener("input", () => {
        cfg[key] = parseFloat(s.value);
        o.textContent = fmt(cfg[key]);
        seekHack(state, state.frame);   // cfg change → getData re-keys + recomputes
      });
    };
    wire("#to-slider", "#to-out", "targetOffset");
    wire("#mwc-slider", "#mwc-out", "minWristConfidence");
    wire("#pp-slider", "#pp-out", "punchPad", v => String(Math.round(v)));

    // Click the timeline to jump to that frame.
    host.querySelector("#gh-timeline").addEventListener("click", (ev) => {
      if (!cache) return;
      const rect = ev.currentTarget.getBoundingClientRect();
      const frac = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
      const f = Math.round(frac * (cache.N - 1));
      const sc = document.getElementById("scrubber");
      if (sc) { sc.value = f; sc.dispatchEvent(new Event("input")); }
    });
  },

  draw(ctx, state) {
    const f = state.frame;
    const p = pickPose(state);
    const W = ctx.canvas.width;
    const s = state.renderScale || 1;
    const data = getData(state);

    const nose = jt(p, f, J.NOSE);
    const lw = jt(p, f, J.L_WRIST);
    const rw = jt(p, f, J.R_WRIST);
    const torso = Math.max(1e-6, torsoHeight(p, f));
    const targetY = nose.y + cfg.targetOffset * torso;

    drawHLine(ctx, nose.y,  W, COLORS.nose,   2 * s, null);
    drawHLine(ctx, targetY, W, COLORS.target, 2 * s, 3 * s);
    drawHLine(ctx, lw.y,    W, lineColor(data.L[f], COLORS.l_wrist), 2 * s, 3 * s);
    drawHLine(ctx, rw.y,    W, lineColor(data.R[f], COLORS.r_wrist), 2 * s, 3 * s);

    drawTag(ctx, lw.x + 8 * s, lw.y, "L " + statusLabel(data.L[f]), tagColor(data.L[f], COLORS.l_wrist), s);
    drawTag(ctx, rw.x + 8 * s, rw.y, "R " + statusLabel(data.R[f]), tagColor(data.R[f], COLORS.r_wrist), s);
  },

  update(state) {
    const f = state.frame;
    const p = pickPose(state);
    const data = getData(state);

    const nose = jt(p, f, J.NOSE);
    const lw = jt(p, f, J.L_WRIST);
    const rw = jt(p, f, J.R_WRIST);
    const torso = Math.max(1e-6, torsoHeight(p, f));
    const targetY = nose.y + cfg.targetOffset * torso;

    setText("l-target-dist", ((lw.y - targetY) / torso).toFixed(2));
    setText("r-target-dist", ((rw.y - targetY) / torso).toFixed(2));
    setText("l-target-verdict", verdictFor(data.L[f]));
    setText("r-target-verdict", verdictFor(data.R[f]));

    const fmt = (st) => st.elig ? `${Math.round(100 * st.low / st.elig)}% <span class="metric-sub">(${st.low}/${st.elig})</span>` : "—";
    setText("gh-l-low", fmt(data.stats.L));
    setText("gh-r-low", fmt(data.stats.R));
    const tElig = data.stats.L.elig + data.stats.R.elig;
    const tLow = data.stats.L.low + data.stats.R.low;
    setText("gh-tot-low", tElig ? `${Math.round(100 * tLow / tElig)}%` : "—");
    setText("gh-note", data.detCount
      ? `${data.detCount} punches excluded per-hand (±${Math.round(cfg.punchPad)}f pad).`
      : `No punches loaded — nothing excluded, so this is raw per-frame flagging.`);

    drawTimeline(host.querySelector("#gh-timeline"), data, f);
  },
};

// ── whole-clip flagging (memoised) ──────────────────────────────────────────

// Recompute only when the pose, the punch source, or a slider actually changes.
// update() + draw() both call this every frame, so the memo matters.
function getData(state) {
  const p = pickPose(state);
  const dets = activeDetections(state);
  const sig = [p, dets, cfg.targetOffset, cfg.minWristConfidence, cfg.punchPad];
  if (cache && cache.sig.length === sig.length && cache.sig.every((v, i) => v === sig[i])) {
    return cache;
  }
  cache = compute(p, dets);
  cache.sig = sig;
  return cache;
}

function compute(p, dets) {
  const N = p.n_frames;
  const pad = Math.round(cfg.punchPad);

  // Per-hand punch masks: a wrist is excluded only while IT is the punching hand.
  const inPunch = { L: new Uint8Array(N), R: new Uint8Array(N) };
  let detCount = 0;
  for (const d of (dets || [])) {
    const side = sideFor(d);
    if (!side) continue;
    let sf = d.start_frame, ef = d.end_frame;
    if (!Number.isFinite(sf) || !Number.isFinite(ef)) continue;
    detCount++;
    sf = Math.max(0, Math.round(sf) - pad);
    ef = Math.min(N - 1, Math.round(ef) + pad);
    const mask = inPunch[side];
    for (let f = sf; f <= ef; f++) mask[f] = 1;
  }

  const L = new Uint8Array(N), R = new Uint8Array(N);
  const stats = { L: { elig: 0, low: 0 }, R: { elig: 0, low: 0 } };

  for (let f = 0; f < N; f++) {
    const i = f * 17;
    const noseY = p.skeleton[(i + J.NOSE) * 2 + 1];
    const noseC = p.conf[i + J.NOSE];
    const torso = torsoHeight(p, f);
    const target = noseY + cfg.targetOffset * torso;
    // Frame-level validity: need a trustworthy nose + a sane torso for the target.
    const frameOk = noseC >= cfg.minWristConfidence && torso > 1 && Number.isFinite(noseY);

    for (const side of ["L", "R"]) {
      const jIdx = side === "L" ? J.L_WRIST : J.R_WRIST;
      const wy = p.skeleton[(i + jIdx) * 2 + 1];
      const wc = p.conf[i + jIdx];
      const arr = side === "L" ? L : R;

      if (inPunch[side][f]) { arr[f] = ST.EXCLUDED; continue; }
      if (!frameOk || wc < cfg.minWristConfidence || !Number.isFinite(wy)) { arr[f] = ST.GATED; continue; }

      const low = (wy - target) / torso > 0;   // y grows downward → below target = low
      arr[f] = low ? ST.LOW : ST.UP;
      stats[side].elig++;
      if (low) stats[side].low++;
    }
  }

  return { N, L, R, stats, detCount };
}

function sideFor(d) {
  const stance = (d.stance === "southpaw" || d.stance === "orthodox") ? d.stance : "orthodox";
  return SIDE_FOR[d.hand]?.[stance] || null;
}

// ── status → display ─────────────────────────────────────────────────────────

function verdictFor(status) {
  switch (status) {
    case ST.EXCLUDED: return `<span class="muted">punching (excluded)</span>`;
    case ST.GATED:    return `<span class="muted">low conf, gated</span>`;
    case ST.LOW:      return `<span class="bad">below target</span>`;
    default:          return `<span class="good">at/above target</span>`;
  }
}

function statusLabel(status) {
  switch (status) {
    case ST.EXCLUDED: return "punch";
    case ST.GATED:    return "low conf";
    case ST.LOW:      return "LOW";
    default:          return "up";
  }
}

function lineColor(status, base) {
  return status === ST.LOW ? COLORS.low : base;
}

function tagColor(status, base) {
  if (status === ST.LOW) return COLORS.low;
  if (status === ST.UP)  return COLORS.up;
  return COLORS.muted;
}

function timelineColor(status) {
  switch (status) {
    case ST.LOW:      return COLORS.low;
    case ST.UP:       return COLORS.up;
    case ST.EXCLUDED: return "rgba(255,255,255,0.14)";
    default:          return "rgba(245,185,69,0.22)";   // gated / low conf
  }
}

// ── canvas helpers ───────────────────────────────────────────────────────────

function pickPose(state) {
  return state.poseV6 || state.pose;
}

function jt(pose, frame, j) {
  return {
    x: pose.skeleton[(frame * 17 + j) * 2],
    y: pose.skeleton[(frame * 17 + j) * 2 + 1],
    c: pose.conf[frame * 17 + j],
  };
}

// dashUnit null ⇒ solid line.
function drawHLine(ctx, y, w, color, lineWidth, dashUnit) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  if (dashUnit) ctx.setLineDash([dashUnit * 2, dashUnit * 2]);
  ctx.beginPath();
  ctx.moveTo(0, y);
  ctx.lineTo(w, y);
  ctx.stroke();
  ctx.restore();
}

function drawTag(ctx, x, y, text, color, scale) {
  const fontSize = Math.round(12 * scale);
  ctx.save();
  ctx.font = `${fontSize}px ui-monospace, monospace`;
  const pad = 3 * scale;
  const m = ctx.measureText(text);
  ctx.fillStyle = "rgba(0,0,0,0.65)";
  ctx.fillRect(x - pad, y - fontSize, m.width + pad * 2, fontSize + 4 * scale);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
  ctx.restore();
}

// Two-row strip (L over R) of per-frame guard status across the whole clip.
function drawTimeline(canvas, data, frame) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const N = data.N;
  const rowH = Math.floor((H - 2) / 2);
  const xToFrame = x => Math.round((x / (W - 1)) * (N - 1));

  for (let x = 0; x < W; x++) {
    const f = xToFrame(x);
    ctx.fillStyle = timelineColor(data.L[f]);
    ctx.fillRect(x, 1, 1, rowH);
    ctx.fillStyle = timelineColor(data.R[f]);
    ctx.fillRect(x, 1 + rowH, 1, rowH);
  }

  ctx.fillStyle = "rgba(255,255,255,0.7)";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText("L", 2, 11);
  ctx.fillText("R", 2, 1 + rowH + 11);

  const cx = Math.round((frame / Math.max(1, N - 1)) * (W - 1));
  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx + 0.5, 0);
  ctx.lineTo(cx + 0.5, H);
  ctx.stroke();
}

function setText(id, value) {
  const el = host.querySelector("#" + id);
  if (el) el.innerHTML = value;
}

function seekHack(state, f) {
  const slider = document.getElementById("scrubber");
  if (!slider) return;
  slider.value = f;
  slider.dispatchEvent(new Event("input"));
}
