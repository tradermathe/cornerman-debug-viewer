// Facing direction lens — estimates the boxer's rotation around the
// vertical body axis (the angle their facing line makes with the
// camera-pointing line). Lets other rules gate on "is this camera
// angle workable for me right now" instead of silently false-flagging
// in the projection dead zones we discovered while shelving hip
// rotation.
//
// What "facing_angle" means here
//   0°    boxer facing camera (chest toward the camera)
//   +90°  boxer's right side toward camera (left flank away)
//   -90°  boxer's left side toward camera  (right flank away)
//   ±180° boxer facing away (we can't usually distinguish that from
//         ±90° on skeleton alone; we don't try)
//
// Composition
//   sign      from face-joint confidence asymmetry. Apple Vision
//             returns conf 0 for joints it didn't detect, so a missing
//             ear on one side is an unusually strong signal compared to
//             other estimators. We combine L/R ear + L/R eye conf.
//   magnitude from hip-line and shoulder-line length, normalised by
//             their per-round maximum: a value of 1.0 means the line is
//             at its widest seen this round (boxer at most-square), and
//             arccos(line_ratio) gives an angle off-square. We take the
//             stronger of the two (whichever isn't occluded that frame).
//
// All visible bias bands (sweet-zone, dead-zone) are configurable.

import { J, torsoHeight } from "../skeleton.js";

const DEFAULTS = {
  // Where rotation-based rules can be trusted vs where they hit
  // projection dead zones. These are signed degree thresholds applied
  // to |facing_angle|.
  sweetZoneMinDeg: 25,    // |angle| above this is workable
  sweetZoneMaxDeg: 65,    // |angle| up to this is workable
  smoothFrames: 5,        // moving average on the per-frame angle
  minFaceConf: 0.10,      // ear/eye sums below this → trust line length only
  earConfWeight: 1.0,     // relative weight of ear asymmetry in the sign
  eyeConfWeight: 0.5,     // relative weight of eye asymmetry in the sign
};

const COLORS = {
  arrow:    "#a78bfa",
  arrowDim: "rgba(167,139,250,0.35)",
  sweet:    "#5fd97a",
  warning:  "#f5b945",
  dead:     "#e85a5a",
  current:  "rgba(255,255,255,0.85)",
};

let host;
let cfg = { ...DEFAULTS };
let signals = null;
let lastPose = null;

export const FacingDirectionRule = {
  id: "facing_direction",
  label: "Facing direction",

  skeletonStyle() {
    return {
      boneColor: "rgba(255,255,255,0.22)",
      boneWidth: 1.5,
      jointRadius: 3,
      highlightJoints: new Set([
        J.NOSE, J.L_EAR, J.R_EAR, J.L_EYE, J.R_EYE,
        J.L_SHOULDER, J.R_SHOULDER, J.L_HIP, J.R_HIP,
      ]),
    };
  },

  mount(_host, state) {
    host = _host;
    host.innerHTML = template();
    signals = computeAll(state, cfg);
    lastPose = state.pose;

    wireSlider(state, "#fd-smin", "sweetZoneMinDeg", v => `${v.toFixed(0)}°`);
    wireSlider(state, "#fd-smax", "sweetZoneMaxDeg", v => `${v.toFixed(0)}°`);
    wireSlider(state, "#fd-smooth", "smoothFrames",  v => `${v.toFixed(0)} frames`);

    const canvas = host.querySelector("#fd-angle-canvas");
    if (canvas) {
      canvas.style.cursor = "pointer";
      canvas.addEventListener("click", evt => {
        const rect = canvas.getBoundingClientRect();
        const frac = (evt.clientX - rect.left) / rect.width;
        const N = state.pose.n_frames;
        seekHack(Math.max(0, Math.min(N - 1, Math.round(frac * (N - 1)))));
      });
    }

    renderSummary();
  },

  update(state) {
    if (state.pose !== lastPose) {
      signals = computeAll(state, cfg);
      lastPose = state.pose;
      renderSummary();
    }
    const f = state.frame;
    const a = signals.facingAngle[f];
    const z = zoneFor(a, cfg);
    setText("fd-angle",   formatAngle(a));
    setText("fd-zone",    z.label);
    setHtml("fd-zone-pill", `<span style="background:${z.color}33;color:${z.color};padding:2px 8px;border-radius:6px;font-weight:600">${z.label}</span>`);
    setText("fd-sign",    formatSign(signals.signDeg[f]));
    setText("fd-magn",    formatAngle(signals.magnitudeDeg[f], false));
    setText("fd-face-asym", signals.faceAsymRaw[f].toFixed(2));
    setText("fd-hip-ratio", signals.hipRatio[f].toFixed(2));
    setText("fd-sho-ratio", signals.shoulderRatio[f].toFixed(2));
    drawAngleTrace(host.querySelector("#fd-angle-canvas"), signals, f, cfg);
  },

  draw(ctx, state) {
    const f = state.frame;
    const p = state.pose;
    const s = state.renderScale || 1;
    drawFacingArrow(ctx, p, f, signals.facingAngle[f], s);
  },
};

// ── DOM ────────────────────────────────────────────────────────────────────

function template() {
  return `
    <h2>Facing direction</h2>
    <p class="hint"><b>This tracks the body, not the face.</b> Magnitude
      comes from hip and shoulder line lengths (the wider one each
      frame); sign comes from face-joint confidence asymmetry (which
      ear/eye is visible). Body and face usually rotate together but
      can diverge — a boxer can keep their face square to the target
      while rotating their hips. For gating rotation-based rules,
      body angle is what matters.</p>
    <p class="hint">0° = body facing the camera, ±90° = body sideways.
      Rotation-based rules (hip rotation, shoulder rotation) hit
      projection dead zones near 0° and near ±90° and should abstain
      there. <b>Known limit:</b> fully sideways usually reads as 70–80°
      rather than 90° — the 2D estimator never lets the hip line truly
      collapse to zero, so <code>arccos(line/max)</code> tops out short
      of π/2. Treat 70–80° as "effectively sideways."</p>

    <h3>Now</h3>
    <div class="metric-grid">
      <div class="metric">
        <div class="metric-label">facing_angle</div>
        <div class="metric-val" id="fd-angle">—</div>
        <div class="metric-sub" id="fd-zone-pill"></div>
      </div>
      <div class="metric">
        <div class="metric-label">zone</div>
        <div class="metric-val" id="fd-zone">—</div>
        <div class="metric-sub muted">sweet = rotation rules trustable</div>
      </div>
      <div class="metric"><div class="metric-label">sign source</div><div class="metric-val" id="fd-sign">—</div></div>
      <div class="metric"><div class="metric-label">magnitude</div><div class="metric-val" id="fd-magn">—</div></div>
    </div>
    <p class="hint" style="margin-top:4px">
      face asym <span id="fd-face-asym">—</span> · hip-line ratio <span id="fd-hip-ratio">—</span> · shoulder-line ratio <span id="fd-sho-ratio">—</span>
    </p>

    <h3>Facing angle over the round</h3>
    <p class="hint">Y axis = signed degrees. Green band = sweet zone
      (rotation rules are reliable). Red bands top + bottom = dead zones
      (near 0° = facing camera, near ±90° = sideways). Click to seek.</p>
    <canvas id="fd-angle-canvas" width="320" height="200"></canvas>

    <h3>Round summary</h3>
    <div id="fd-summary"></div>

    <h3>Thresholds</h3>
    <label class="slider">
      <span>sweet_zone_min = <output id="fd-smin-out">${cfg.sweetZoneMinDeg.toFixed(0)}°</output></span>
      <input type="range" id="fd-smin" min="0" max="45" step="1" value="${cfg.sweetZoneMinDeg}">
      <span class="muted small">|angle| below this = "facing camera" dead zone.</span>
    </label>
    <label class="slider">
      <span>sweet_zone_max = <output id="fd-smax-out">${cfg.sweetZoneMaxDeg.toFixed(0)}°</output></span>
      <input type="range" id="fd-smax" min="45" max="90" step="1" value="${cfg.sweetZoneMaxDeg}">
      <span class="muted small">|angle| above this = "sideways" dead zone.</span>
    </label>
    <label class="slider">
      <span>smoothing = <output id="fd-smooth-out">${cfg.smoothFrames.toFixed(0)} frames</output></span>
      <input type="range" id="fd-smooth" min="0" max="15" step="1" value="${cfg.smoothFrames}">
      <span class="muted small">Moving-average window on the per-frame angle. Kills face-joint jitter.</span>
    </label>
  `;
}

function wireSlider(state, sel, key, fmt) {
  const s = host.querySelector(sel);
  const out = host.querySelector(sel + "-out");
  if (!s || !out) return;
  s.addEventListener("input", () => {
    cfg[key] = parseFloat(s.value);
    out.textContent = fmt(cfg[key]);
    signals = computeAll(state, cfg);
    renderSummary();
    seekHack(state.frame);
  });
}

function renderSummary() {
  const el = host?.querySelector("#fd-summary");
  if (!el || !signals) return;
  // Time spent in each zone.
  let sweet = 0, deadFront = 0, deadSide = 0, ambig = 0;
  for (let i = 0; i < signals.facingAngle.length; i++) {
    const z = zoneFor(signals.facingAngle[i], cfg);
    if (z.kind === "sweet")     sweet++;
    else if (z.kind === "front") deadFront++;
    else if (z.kind === "side")  deadSide++;
    else ambig++;
  }
  const N = signals.facingAngle.length;
  const pct = n => `${Math.round(100 * n / N)}%`;
  el.innerHTML = `
    <table class="sps-tbl">
      <thead><tr><th>Zone</th><th>Frames</th><th>%</th><th>Meaning for rules</th></tr></thead>
      <tbody>
        <tr><td><span style="color:${COLORS.sweet}">sweet</span></td><td>${sweet}</td><td>${pct(sweet)}</td><td>Rotation rules trustable</td></tr>
        <tr><td><span style="color:${COLORS.dead}">facing-camera dead zone</span></td><td>${deadFront}</td><td>${pct(deadFront)}</td><td>Hip/shoulder rotation signal vanishes</td></tr>
        <tr><td><span style="color:${COLORS.dead}">sideways dead zone</span></td><td>${deadSide}</td><td>${pct(deadSide)}</td><td>One hip occludes the other; same problem in reverse</td></tr>
        <tr><td><span style="color:${COLORS.warning}">ambiguous (no sign)</span></td><td>${ambig}</td><td>${pct(ambig)}</td><td>Face joints too low-confidence to tell direction</td></tr>
      </tbody>
    </table>
  `;
}

// ── Compute ────────────────────────────────────────────────────────────────

function computeAll(state, cfg) {
  const pose = state.pose;
  const N = pose.n_frames;

  // Per-frame raw signals.
  const faceAsymRaw = new Float32Array(N);   // (L - R) / (L + R), signed [-1, +1]
  const hipLine    = new Float32Array(N);
  const shoLine    = new Float32Array(N);

  for (let f = 0; f < N; f++) {
    const cLEar = pose.conf[f * 17 + J.L_EAR];
    const cREar = pose.conf[f * 17 + J.R_EAR];
    const cLEye = pose.conf[f * 17 + J.L_EYE];
    const cREye = pose.conf[f * 17 + J.R_EYE];
    const earL = cfg.earConfWeight * cLEar;
    const earR = cfg.earConfWeight * cREar;
    const eyeL = cfg.eyeConfWeight * cLEye;
    const eyeR = cfg.eyeConfWeight * cREye;
    const sumL = earL + eyeL;
    const sumR = earR + eyeR;
    const total = sumL + sumR;
    faceAsymRaw[f] = total > 1e-6 ? (sumL - sumR) / total : 0;

    const th = Math.max(1e-6, torsoHeight(pose, f));
    const lhx = pose.skeleton[(f * 17 + J.L_HIP) * 2];
    const lhy = pose.skeleton[(f * 17 + J.L_HIP) * 2 + 1];
    const rhx = pose.skeleton[(f * 17 + J.R_HIP) * 2];
    const rhy = pose.skeleton[(f * 17 + J.R_HIP) * 2 + 1];
    hipLine[f] = Math.hypot(lhx - rhx, lhy - rhy) / th;

    const lsx = pose.skeleton[(f * 17 + J.L_SHOULDER) * 2];
    const lsy = pose.skeleton[(f * 17 + J.L_SHOULDER) * 2 + 1];
    const rsx = pose.skeleton[(f * 17 + J.R_SHOULDER) * 2];
    const rsy = pose.skeleton[(f * 17 + J.R_SHOULDER) * 2 + 1];
    shoLine[f] = Math.hypot(lsx - rsx, lsy - rsy) / th;
  }

  // Per-round 95th-percentile (the most-square frame the boxer hit
  // this round) for each line — used as the "facing camera" anchor.
  // We DON'T rescale against an observed min. That sounds tempting
  // (saturate at 90° at sideways), but breaks on all-front footage:
  // when min ≈ max, tiny noise gets amplified into garbage angles.
  // Trade-off: with this raw arccos, fully-sideways tops out around
  // 70-80° because 2D pose estimators never let the hip line truly
  // collapse — the occluded hip still gets a non-zero best-guess
  // position. We accept that and just remember: 70-80° here is
  // actually fully sideways.
  const hipMax = percentile(hipLine, 0.95);
  const shoMax = percentile(shoLine, 0.95);

  const hipRatio = new Float32Array(N);
  const shoulderRatio = new Float32Array(N);
  const magnitudeDeg = new Float32Array(N);
  const signDeg = new Float32Array(N);

  for (let f = 0; f < N; f++) {
    hipRatio[f]      = hipMax > 0 ? Math.min(1, hipLine[f] / hipMax) : 0;
    shoulderRatio[f] = shoMax > 0 ? Math.min(1, shoLine[f] / shoMax) : 0;
    // Use whichever line is wider — it's the less-occluded one this
    // frame. arccos returns [0, π/2] for ratio in [0, 1].
    const ratio = Math.max(hipRatio[f], shoulderRatio[f]);
    magnitudeDeg[f] = Math.acos(Math.min(1, Math.max(0, ratio))) * 180 / Math.PI;
    // Sign: from face-joint asymmetry. Positive asym = L side more
    // visible = boxer rotated such that their left flank is toward
    // camera.
    signDeg[f] = faceAsymRaw[f];
  }

  // Smooth + combine.
  const smoothed = movingAvg(magnitudeDeg, cfg.smoothFrames);
  const smoothedSign = movingAvg(signDeg,  cfg.smoothFrames);
  const facingAngle = new Float32Array(N);
  for (let f = 0; f < N; f++) {
    const m = smoothed[f];
    const sRaw = smoothedSign[f];
    const signMag = Math.abs(sRaw);
    let sign = 0;
    if (signMag > 0.05) sign = Math.sign(sRaw);
    facingAngle[f] = sign * m;
  }

  return {
    faceAsymRaw, hipLine, shoLine,
    hipRatio, shoulderRatio,
    magnitudeDeg: smoothed,
    signDeg: smoothedSign,
    facingAngle,
    hipMax, shoMax,
  };
}

function movingAvg(arr, w) {
  if (w <= 1) return arr;
  const n = arr.length;
  const out = new Float32Array(n);
  const half = Math.floor(w / 2);
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - half), hi = Math.min(n - 1, i + half);
    let s = 0;
    for (let k = lo; k <= hi; k++) s += arr[k];
    out[i] = s / (hi - lo + 1);
  }
  return out;
}

function percentile(arr, p) {
  const s = Array.from(arr).filter(v => v > 0).sort((a, b) => a - b);
  if (!s.length) return 0;
  return s[Math.min(s.length - 1, Math.floor(s.length * p))];
}

function zoneFor(angle, cfg) {
  const a = Math.abs(angle);
  if (a < cfg.sweetZoneMinDeg) return { kind: "front", label: "facing camera",  color: COLORS.dead };
  if (a > cfg.sweetZoneMaxDeg) return { kind: "side",  label: "near sideways",  color: COLORS.dead };
  return { kind: "sweet", label: "sweet zone", color: COLORS.sweet };
}

// ── Drawing ────────────────────────────────────────────────────────────────

function drawFacingArrow(ctx, pose, frame, angleDeg, scale) {
  // Anchor at hip midpoint (well below the face) and draw a small, thin
  // compass-style arrow indicating which way the boxer is rotated.
  // Numeric label goes in the canvas's top-right corner so it never
  // overlaps the boxer.
  const lhx = pose.skeleton[(frame * 17 + J.L_HIP) * 2];
  const lhy = pose.skeleton[(frame * 17 + J.L_HIP) * 2 + 1];
  const rhx = pose.skeleton[(frame * 17 + J.R_HIP) * 2];
  const rhy = pose.skeleton[(frame * 17 + J.R_HIP) * 2 + 1];
  const lhc = pose.conf[frame * 17 + J.L_HIP];
  const rhc = pose.conf[frame * 17 + J.R_HIP];

  const haveHips = lhc >= 0.05 && rhc >= 0.05;
  if (haveHips) {
    const cx = (lhx + rhx) / 2;
    const cy = (lhy + rhy) / 2;
    // Arrow length scales with the (rescaled) hip line itself so it
    // doesn't dwarf small boxers in the frame.
    const hipPx = Math.hypot(lhx - rhx, lhy - rhy);
    const len = Math.max(20 * scale, hipPx * 0.7);

    const rad = angleDeg * Math.PI / 180;
    const dx = Math.sin(rad);
    const intoScreen = Math.cos(rad);

    ctx.save();
    // Small zone-colored dot under the hips. Subtle, doesn't obscure
    // the skeleton.
    const zone = zoneFor(angleDeg, cfg);
    ctx.fillStyle = zone.color;
    ctx.globalAlpha = 0.55;
    ctx.beginPath();
    ctx.arc(cx, cy, 4 * scale, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;

    // Thin arrow body, anchored at hip midpoint.
    const tipX = cx + dx * len;
    const tipY = cy;     // keep horizontal — pure rotation indicator
    ctx.strokeStyle = COLORS.arrow;
    ctx.lineWidth = 2 * scale;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();

    // Small arrowhead.
    const ah = 6 * scale;
    const angle = Math.atan2(tipY - cy, tipX - cx);
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(tipX - ah * Math.cos(angle - 0.4), tipY - ah * Math.sin(angle - 0.4));
    ctx.lineTo(tipX - ah * Math.cos(angle + 0.4), tipY - ah * Math.sin(angle + 0.4));
    ctx.closePath();
    ctx.fillStyle = COLORS.arrow;
    ctx.fill();
    ctx.restore();
  }

  // Top-right numeric label — independent of the boxer's position so it
  // never overlaps the body.
  drawCornerLabel(ctx, angleDeg, scale);
}

function drawCornerLabel(ctx, angleDeg, scale) {
  const zone = zoneFor(angleDeg, cfg);
  const angleTxt = formatAngle(angleDeg);
  const zoneTxt = zone.label;
  ctx.save();
  const fontPx = Math.round(13 * scale);
  const subPx = Math.round(10 * scale);
  ctx.font = `bold ${fontPx}px ui-monospace, monospace`;
  const w1 = ctx.measureText(angleTxt).width;
  ctx.font = `${subPx}px ui-monospace, monospace`;
  const w2 = ctx.measureText(zoneTxt).width;
  const padX = 8 * scale, padY = 6 * scale;
  const boxW = Math.max(w1, w2) + padX * 2;
  const boxH = fontPx + subPx + padY * 2 + 2 * scale;
  const x = ctx.canvas.width - boxW - 12 * scale;
  const y = 12 * scale;
  ctx.fillStyle = "rgba(0,0,0,0.72)";
  if (typeof ctx.roundRect === "function") {
    ctx.beginPath();
    ctx.roundRect(x, y, boxW, boxH, 6 * scale);
    ctx.fill();
  } else {
    ctx.fillRect(x, y, boxW, boxH);
  }
  ctx.fillStyle = zone.color;
  ctx.fillRect(x, y, 4 * scale, boxH);
  ctx.fillStyle = "#fff";
  ctx.textBaseline = "top";
  ctx.font = `bold ${fontPx}px ui-monospace, monospace`;
  ctx.fillText(angleTxt, x + padX, y + padY);
  ctx.fillStyle = zone.color;
  ctx.font = `${subPx}px ui-monospace, monospace`;
  ctx.fillText(zoneTxt, x + padX, y + padY + fontPx + 2 * scale);
  ctx.restore();
}

function drawAngleTrace(canvas, signals, frame, cfg) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const N = signals.facingAngle.length;
  const stride = Math.max(1, Math.floor(N / W));

  const yMin = -95, yMax = 95;
  const ymap = v => H - ((v - yMin) / (yMax - yMin)) * (H - 8) - 4;
  const xmap = f => (f / Math.max(1, N - 1)) * (W - 2) + 1;

  // Dead-zone bands.
  const ySweetMin = ymap(cfg.sweetZoneMinDeg);
  const ySweetMax = ymap(cfg.sweetZoneMaxDeg);
  const ySweetMinNeg = ymap(-cfg.sweetZoneMinDeg);
  const ySweetMaxNeg = ymap(-cfg.sweetZoneMaxDeg);
  // Front dead zone (-min .. +min)
  ctx.fillStyle = "rgba(232,90,90,0.12)";
  ctx.fillRect(0, ySweetMin, W, ySweetMinNeg - ySweetMin);
  // Side dead zones
  ctx.fillRect(0, 0,            W, ySweetMax);
  ctx.fillRect(0, ySweetMaxNeg, W, H - ySweetMaxNeg);
  // Sweet bands (above sweet_min, below sweet_max)
  ctx.fillStyle = "rgba(95,217,122,0.12)";
  ctx.fillRect(0, ySweetMax, W, ySweetMin - ySweetMax);
  ctx.fillRect(0, ySweetMinNeg, W, ySweetMaxNeg - ySweetMinNeg);

  // Gridlines at 0, ±30, ±60, ±90.
  ctx.strokeStyle = "rgba(255,255,255,0.15)";
  ctx.lineWidth = 1;
  for (const v of [-90, -60, -30, 0, 30, 60, 90]) {
    const y = ymap(v);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText(`${v >= 0 ? "+" : ""}${v}°`, 4, y - 2);
  }

  // Facing angle line.
  ctx.strokeStyle = COLORS.arrow;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let f = 0; f < N; f += stride) {
    const px = xmap(f), py = ymap(signals.facingAngle[f]);
    if (f === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // Current frame.
  ctx.strokeStyle = COLORS.current;
  ctx.lineWidth = 1;
  const x = xmap(frame);
  ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
}

// ── Utilities ──────────────────────────────────────────────────────────────

function formatAngle(deg, withSign = true) {
  if (deg == null || Number.isNaN(deg)) return "—";
  const s = deg >= 0 ? "+" : "";
  return withSign ? `${s}${deg.toFixed(1)}°` : `${Math.abs(deg).toFixed(1)}°`;
}

function formatSign(raw) {
  if (raw == null) return "—";
  if (Math.abs(raw) < 0.05) return "ambiguous";
  return raw > 0 ? "L side toward cam" : "R side toward cam";
}

function setText(id, value) {
  const el = host.querySelector("#" + id);
  if (el) el.textContent = value;
}
function setHtml(id, value) {
  const el = host.querySelector("#" + id);
  if (el) el.innerHTML = value;
}
function seekHack(f) {
  const slider = document.getElementById("scrubber");
  if (!slider) return;
  slider.value = f;
  slider.dispatchEvent(new Event("input"));
}
