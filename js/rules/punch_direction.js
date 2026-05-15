// Punch direction lens — for each labelled punch, visualise the wrist's
// 2D trajectory during the punch window. The same data that feeds the
// arm_extension "travel" column, but presented spatially: an arrow on the
// canvas from where the wrist started to where it peaked, plus a per-
// punch table of direction angles and magnitudes.
//
// What we draw, per labelled punch:
//   • start dot at the wrist position when the punch window opens
//   • peak  dot at the wrist position where |sh→wr| was largest
//   • arrow from start → peak, colored by hand (lead / rear)
//
// Per-punch metrics:
//   direction_deg = atan2(dy, dx) in degrees  (0° = →, 90° = ↓, ±180° = ←, -90° = ↑)
//   magnitude_px  = |peak − start|
//   travel_norm   = magnitude_px / arm_length
//
// Wrist source: glove conf ≥ 0.20, pose fallback — same as the wrist_swap
// and arm_extension lenses.

import { J } from "../skeleton.js";
import { gloveXY, gloveConf } from "../pose-loader.js";

const DEFAULTS = {
  minGloveConf: 0.20,
  minPoseConf:  0.20,
  // Show only straight punches by default (matches arm_extension's scope);
  // toggle in the panel to include hooks / uppercuts.
  straightsOnly: true,
  showAll: false,   // when on, overlay every punch's arrow at once
};

const COLORS = {
  lead:        "#5fd1ff",
  rear:        "#ffaa3c",
  active:      "#5fd97a",
  bg:          "rgba(0,0,0,0.55)",
  dimLead:     "rgba(95,209,255,0.30)",
  dimRear:     "rgba(255,170,60,0.30)",
};

const SIDE_FOR = {
  lead: { orthodox: "L", southpaw: "R" },
  rear: { orthodox: "R", southpaw: "L" },
};
const JOINTS_FOR_SIDE = {
  L: { shoulder: J.L_SHOULDER, elbow: J.L_ELBOW, wrist: J.L_WRIST, gloveSide: 0 },
  R: { shoulder: J.R_SHOULDER, elbow: J.R_ELBOW, wrist: J.R_WRIST, gloveSide: 1 },
};
const STRAIGHTS = new Set(["jab_head", "jab_body", "cross_head", "cross_body"]);

let host;
let cfg = { ...DEFAULTS };
let signals = null;
let lastPose = null;

export const PunchDirectionRule = {
  id: "punch_direction",
  label: "Punch direction (wrist trajectory)",

  requires(slot) {
    return !!(slot?.vision || slot?.yolo) && !!slot?.glove;
  },

  skeletonStyle() {
    return {
      boneColor: "rgba(255,255,255,0.20)",
      boneWidth: 1.5,
      jointRadius: 3,
      hideJoints: new Set([J.L_WRIST, J.R_WRIST]),
    };
  },

  mount(_host, state) {
    host = _host;
    cfg = { ...DEFAULTS };
    signals = computeAll(state, cfg);
    lastPose = state.pose;

    host.innerHTML = renderTemplate(signals, cfg);
    renderPunchTable();

    const wireToggle = (id, key) => {
      const t = host.querySelector("#" + id);
      if (!t) return;
      t.addEventListener("change", () => {
        cfg[key] = t.checked;
        signals = computeAll(state, cfg);
        renderPunchTable();
        state.requestDraw?.();
      });
    };
    wireToggle("pd-straights-only", "straightsOnly");
    wireToggle("pd-show-all",       "showAll");

    host.addEventListener("click", (ev) => {
      const tr = ev.target.closest("tr[data-frame]");
      if (!tr) return;
      const f = Number(tr.dataset.frame);
      const slider = document.getElementById("scrubber");
      if (!slider) return;
      slider.value = String(f);
      slider.dispatchEvent(new Event("input"));
    });
  },

  draw(ctx, state) {
    if (state.pose !== lastPose) {
      signals = computeAll(state, cfg);
      lastPose = state.pose;
    }
    const s = state.renderScale || 1;
    const f = state.frame;

    if (cfg.showAll) {
      // Draw every punch's arrow dimmed
      for (const p of signals.punches) {
        const active = (f >= p.start_frame && f <= p.end_frame);
        const dim = !active;
        drawPunchArrow(ctx, p, dim, s);
      }
    } else {
      // Only the active punch
      const active = activePunchAt(signals.punches, f);
      if (active) drawPunchArrow(ctx, active, false, s);
    }
  },

  update(state) {
    if (state.pose !== lastPose) {
      signals = computeAll(state, cfg);
      lastPose = state.pose;
    }
    const f = state.frame;
    const active = activePunchAt(signals.punches, f);
    if (active && Number.isFinite(active.direction_deg)) {
      setText("pd-active-summary",
        `${active.hand} ${active.punch_type} · ${active.direction_deg.toFixed(0)}° · ` +
        `${active.magnitude_px.toFixed(0)} px (${(active.travel_norm * 100).toFixed(0)}% of arm length)`);
    } else {
      setText("pd-active-summary", "— playhead is not inside a labelled punch window —");
    }
  },
};

// ─── compute ───────────────────────────────────────────────────────────────

function computeAll(state, cfg) {
  const pose = state.pose;
  const N = pose.n_frames;
  const fps = pose.fps;
  const startSec = pose.start_sec || 0;

  // arm length per side (round-stable)
  const armLengthL = armLengthFor(pose, "L", cfg);
  const armLengthR = armLengthFor(pose, "R", cfg);

  const detections = (state.labels?.detections || []).filter(d => {
    if (cfg.straightsOnly) return STRAIGHTS.has(d.punch_type);
    return true;
  });

  const punches = detections.map((d, idx) => {
    const sf = Math.max(0, d.start_frame);
    const ef = Math.min(N - 1, d.end_frame);
    const stance = (d.stance === "southpaw" || d.stance === "orthodox") ? d.stance : "orthodox";
    const side = SIDE_FOR[d.hand]?.[stance] || "L";
    const joints = JOINTS_FOR_SIDE[side];
    const armLength = side === "L" ? armLengthL : armLengthR;

    // Wrist position at every valid frame in the window, plus sw to find peak
    let startXY = null;        // first valid frame's wrist
    let peakXY = null;          // wrist where |sh→wr| was largest
    let peakSw = -Infinity, peakFrame = sf;
    for (let f = sf; f <= ef; f++) {
      const w = wristXY(pose, f, joints, cfg);
      if (!w) continue;
      const sc = pose.conf[f * 17 + joints.shoulder];
      if (sc < cfg.minPoseConf) continue;
      const sx = pose.skeleton[(f * 17 + joints.shoulder) * 2];
      const sy = pose.skeleton[(f * 17 + joints.shoulder) * 2 + 1];
      const sw = Math.hypot(sx - w.x, sy - w.y);
      if (!startXY) startXY = { x: w.x, y: w.y, frame: f };
      if (sw > peakSw) { peakSw = sw; peakXY = { x: w.x, y: w.y, frame: f }; peakFrame = f; }
    }

    if (!startXY || !peakXY) {
      return {
        idx, timestamp: d.timestamp, t_abs: startSec + (Number.isFinite(d.timestamp) ? d.timestamp : 0),
        hand: d.hand, stance, side, punch_type: d.punch_type,
        start_frame: sf, end_frame: ef, land_frame: peakFrame,
        startXY: null, peakXY: null,
        direction_deg: NaN, magnitude_px: NaN, travel_norm: NaN,
        armLength,
      };
    }
    const dx = peakXY.x - startXY.x;
    const dy = peakXY.y - startXY.y;
    const magnitude_px = Math.hypot(dx, dy);
    const direction_deg = Math.atan2(dy, dx) * 180 / Math.PI;
    const travel_norm = (armLength && armLength > 0) ? magnitude_px / armLength : NaN;
    return {
      idx, timestamp: d.timestamp, t_abs: startSec + (Number.isFinite(d.timestamp) ? d.timestamp : 0),
      hand: d.hand, stance, side, punch_type: d.punch_type,
      start_frame: sf, end_frame: ef, land_frame: peakFrame,
      startXY, peakXY,
      direction_deg, magnitude_px, travel_norm,
      armLength,
    };
  });

  return { punches, fps, armLengthL, armLengthR };
}

function armLengthFor(pose, side, cfg) {
  const N = pose.n_frames;
  const joints = JOINTS_FOR_SIDE[side];
  const ueArr = [], faArr = [];
  for (let f = 0; f < N; f++) {
    const sc = pose.conf[f * 17 + joints.shoulder];
    const ec = pose.conf[f * 17 + joints.elbow];
    if (sc >= cfg.minPoseConf && ec >= cfg.minPoseConf) {
      const sx = pose.skeleton[(f * 17 + joints.shoulder) * 2];
      const sy = pose.skeleton[(f * 17 + joints.shoulder) * 2 + 1];
      const ex = pose.skeleton[(f * 17 + joints.elbow) * 2];
      const ey = pose.skeleton[(f * 17 + joints.elbow) * 2 + 1];
      const ue = Math.hypot(sx - ex, sy - ey);
      if (Number.isFinite(ue)) ueArr.push(ue);
    }
    if (ec >= cfg.minPoseConf) {
      const w = wristXY(pose, f, joints, cfg);
      if (w) {
        const ex = pose.skeleton[(f * 17 + joints.elbow) * 2];
        const ey = pose.skeleton[(f * 17 + joints.elbow) * 2 + 1];
        const fa = Math.hypot(ex - w.x, ey - w.y);
        if (Number.isFinite(fa)) faArr.push(fa);
      }
    }
  }
  if (ueArr.length < 10 || faArr.length < 10) return null;
  const median = arr => { const s = [...arr].sort((a,b)=>a-b); return s[Math.floor(s.length / 2)]; };
  return median(ueArr) + median(faArr);
}

function wristXY(pose, frame, joints, cfg) {
  const g = pose.gloveWrists;
  if (g) {
    const [gx, gy] = gloveXY(g, frame, joints.gloveSide);
    const gc = gloveConf(g, frame, joints.gloveSide);
    if (gc >= cfg.minGloveConf && Number.isFinite(gx) && Number.isFinite(gy)) {
      return { x: gx, y: gy, source: "glove" };
    }
  }
  const px = pose.skeleton[(frame * 17 + joints.wrist) * 2];
  const py = pose.skeleton[(frame * 17 + joints.wrist) * 2 + 1];
  const pc = pose.conf[frame * 17 + joints.wrist];
  if (pc < cfg.minPoseConf || !Number.isFinite(px)) return null;
  return { x: px, y: py, source: "pose" };
}

function activePunchAt(punches, frame) {
  for (const p of punches) {
    if (frame >= p.start_frame && frame <= p.end_frame) return p;
  }
  return null;
}

// ─── render ────────────────────────────────────────────────────────────────

function renderTemplate(sig, cfg) {
  return `
    <h2>Punch direction</h2>
    <p class="hint">
      For each labelled punch, draws an arrow on the canvas from where the
      wrist started (when the punch window opens) to where the wrist
      <em>peaked</em> from the shoulder. Lets you see the 2D direction of
      the punch — and instantly spot camera-facing punches as tiny arrows.
    </p>

    <h3>Legend</h3>
    <ul class="hint" style="list-style:none;padding-left:0;margin:0 0 12px 0;line-height:1.7">
      <li><span style="display:inline-block;width:24px;height:3px;background:${COLORS.lead};vertical-align:middle"></span>
        &nbsp;lead-hand punch arrow</li>
      <li><span style="display:inline-block;width:24px;height:3px;background:${COLORS.rear};vertical-align:middle"></span>
        &nbsp;rear-hand punch arrow</li>
      <li>📐 angle in image coords: <code>0°</code> = right, <code>90°</code> = down, <code>±180°</code> = left, <code>-90°</code> = up</li>
      <li>📏 magnitude in pixels + as % of arm length (the round-median <code>|sh→el|+|el→wr|</code>)</li>
    </ul>

    <label style="display:flex;align-items:center;gap:8px;margin:4px 0">
      <input type="checkbox" id="pd-straights-only" ${cfg.straightsOnly ? 'checked' : ''} />
      <span>Straights only (jab / cross, head + body)</span>
    </label>
    <label style="display:flex;align-items:center;gap:8px;margin:4px 0 12px 0">
      <input type="checkbox" id="pd-show-all" ${cfg.showAll ? 'checked' : ''} />
      <span>Show every punch's arrow simultaneously (dimmed when not active)</span>
    </label>

    <h3>Active punch</h3>
    <p class="hint" id="pd-active-summary">— playhead is not inside a labelled punch window —</p>

    <h3>Per-punch</h3>
    <p class="hint">
      Clickable rows seek to the peak frame. Tiny magnitudes are the depth-
      facing punches you can't score with 2D geometry.
    </p>
    <div id="pd-table-host"></div>
  `;
}

function renderPunchTable() {
  const tbody = signals.punches.length
    ? signals.punches.map(p => {
        const tsStr = Number.isFinite(p.t_abs) ? p.t_abs.toFixed(2) : "—";
        const dirStr = Number.isFinite(p.direction_deg) ? `${p.direction_deg.toFixed(0)}°` : "—";
        const magStr = Number.isFinite(p.magnitude_px) ? `${p.magnitude_px.toFixed(0)} px` : "—";
        const travelStr = Number.isFinite(p.travel_norm) ? `${(p.travel_norm * 100).toFixed(0)}%` : "—";
        const handCol = p.hand === "lead" ? COLORS.lead : COLORS.rear;
        return `<tr data-frame="${p.land_frame}" style="cursor:pointer">
          <td>${tsStr}s</td>
          <td>${p.punch_type}</td>
          <td><span style="color:${handCol}">${p.hand}</span></td>
          <td style="font-variant-numeric:tabular-nums">${dirStr}</td>
          <td style="font-variant-numeric:tabular-nums">${magStr}</td>
          <td style="font-variant-numeric:tabular-nums">${travelStr}</td>
        </tr>`;
      }).join("")
    : `<tr><td colspan="6" class="muted">no punches in this round (or none match the straights-only filter)</td></tr>`;
  const tableHost = host.querySelector("#pd-table-host");
  if (!tableHost) return;
  tableHost.innerHTML = `
    <table class="rule-table">
      <thead><tr>
        <th>t</th><th>type</th><th>hand</th>
        <th title="direction in image coords (atan2 of dy, dx)">dir</th>
        <th title="|wrist[peak] − wrist[start]|">mag</th>
        <th title="magnitude as a fraction of arm length">travel</th>
      </tr></thead>
      <tbody>${tbody}</tbody>
    </table>`;
}

// ─── draw ──────────────────────────────────────────────────────────────────

function drawPunchArrow(ctx, p, dim, scale) {
  if (!p.startXY || !p.peakXY) return;
  const handCol = p.hand === "lead"
    ? (dim ? COLORS.dimLead : COLORS.lead)
    : (dim ? COLORS.dimRear : COLORS.rear);

  ctx.save();
  ctx.strokeStyle = handCol;
  ctx.fillStyle = handCol;
  ctx.lineWidth = (dim ? 1.5 : 3) * scale;

  // Line from start → peak
  ctx.beginPath();
  ctx.moveTo(p.startXY.x, p.startXY.y);
  ctx.lineTo(p.peakXY.x, p.peakXY.y);
  ctx.stroke();

  // Arrowhead at the peak
  const dx = p.peakXY.x - p.startXY.x;
  const dy = p.peakXY.y - p.startXY.y;
  const len = Math.hypot(dx, dy);
  if (len > 6 * scale) {
    const ux = dx / len, uy = dy / len;
    const headLen = Math.min(18 * scale, len * 0.35);
    const headWidth = headLen * 0.55;
    // perpendicular
    const px = -uy, py = ux;
    const tipX = p.peakXY.x, tipY = p.peakXY.y;
    const baseX = tipX - ux * headLen, baseY = tipY - uy * headLen;
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(baseX + px * headWidth, baseY + py * headWidth);
    ctx.lineTo(baseX - px * headWidth, baseY - py * headWidth);
    ctx.closePath();
    ctx.fill();
  }

  // Start dot (hollow ring) + peak dot (filled)
  ctx.beginPath();
  ctx.arc(p.startXY.x, p.startXY.y, 5 * scale, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(p.peakXY.x, p.peakXY.y, 4 * scale, 0, Math.PI * 2);
  ctx.fill();

  if (!dim) {
    // Label near the midpoint: "165°  ·  120 px"
    const midX = (p.startXY.x + p.peakXY.x) / 2;
    const midY = (p.startXY.y + p.peakXY.y) / 2;
    const txt = Number.isFinite(p.direction_deg)
      ? `${p.direction_deg.toFixed(0)}° · ${p.magnitude_px.toFixed(0)}px`
      : "—";
    ctx.font = `${12 * scale}px ui-monospace, monospace`;
    const tw = ctx.measureText(txt).width;
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(midX - tw/2 - 4 * scale, midY - 9 * scale, tw + 8 * scale, 18 * scale);
    ctx.fillStyle = handCol;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(txt, midX, midY);
    ctx.textAlign = "start";
    ctx.textBaseline = "alphabetic";
  }
  ctx.restore();
}

function setText(id, value) {
  const el = host?.querySelector("#" + id);
  if (el) el.textContent = value;
}
