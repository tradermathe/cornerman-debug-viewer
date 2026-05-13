// Step detector — standalone lens for inspecting *only* the foot-step half
// of the step+punch-sync rule. Use this when "no step" is showing up
// suspiciously and you want to see what the algorithm sees.
//
// Mirrors the algorithm in cornerman_rules/rules/step_punch_sync.py and the
// step+punch-sync lens, so what you see here is what the rule sees:
//
//   1. Per-frame ankle velocity (torso-heights per frame), smoothed over a
//      configurable window.
//   2. A "step" = a contiguous run above step_high. The peak of that run
//      is the step's onset frame.
//   3. The "plant" = first frame after the peak where velocity stays below
//      step_low for 2+ frames. If no such plant exists in the cache, the
//      step is dropped — that's the brittle bit.
//
// Visuals:
//   * Overlay: each ankle dot colour-coded by its velocity bucket
//     (planted / transitioning / stepping) + a label with the live
//     velocity number. PEAK and PLANT frames get persistent labels with
//     frame/ms offsets so you can scrub to them.
//   * Side panel: live current-frame metrics, ankle-velocity sparkline
//     with both threshold lines, full step-event table.
//   * Threshold sliders re-run detection on the fly.

import { J, torsoHeight } from "../skeleton.js";

const DEFAULTS = {
  highVelThreshold: 0.020,    // ankle is clearly moving (torso/frame)
  lowVelThreshold:  0.005,    // ankle is "planted"
  velSmoothSeconds: 0.083,    // moving average window
  plantFramesRequired: 2,     // consecutive sub-low frames to call it planted
  labelMarginMs:    120,      // how far before/after a step we keep the label
};

const COLORS = {
  stepping:     "#ef4444",    // red — above HIGH
  transit:      "#facc15",    // amber — between HIGH and LOW
  planted:      "#5fd97a",    // green — below LOW
  ankleLabel:   "#ffffff",
  peakAccent:   "#ef4444",
  plantAccent:  "#5fd97a",
};

let host;
let cfg = { ...DEFAULTS };
let signals = null;
let lastPose = null;

export const StepDetectorRule = {
  id: "step_detector",
  label: "Step detector",

  skeletonStyle() {
    return {
      boneColor: "rgba(255,255,255,0.22)",
      boneWidth: 1.5,
      jointRadius: 3,
      highlightJoints: new Set([
        J.L_ANKLE, J.R_ANKLE, J.L_KNEE, J.R_KNEE, J.L_HIP, J.R_HIP,
      ]),
    };
  },

  mount(_host, state) {
    host = _host;
    host.innerHTML = template();
    signals = computeAll(state, cfg);
    lastPose = state.pose;

    wireSlider(state, "#sd-high",   "highVelThreshold",     v => v.toFixed(3));
    wireSlider(state, "#sd-low",    "lowVelThreshold",      v => v.toFixed(3));
    wireSlider(state, "#sd-smooth", "velSmoothSeconds",     v => `${(v*1000).toFixed(0)} ms`);
    wireSlider(state, "#sd-plantn", "plantFramesRequired",  v => `${v.toFixed(0)} frames`);

    // Click anywhere on the velocity sparkline to seek to that frame.
    const canvas = host.querySelector("#sd-vel-canvas");
    if (canvas) {
      canvas.style.cursor = "pointer";
      canvas.addEventListener("click", evt => {
        const rect = canvas.getBoundingClientRect();
        const px = evt.clientX - rect.left;
        const frac = px / rect.width;
        const N = state.pose.n_frames;
        const target = Math.max(0, Math.min(N - 1, Math.round(frac * (N - 1))));
        seekHackSimple(target);
      });
    }

    renderEventsTable();
  },

  update(state) {
    if (state.pose !== lastPose) {
      signals = computeAll(state, cfg);
      lastPose = state.pose;
      renderEventsTable();
    }

    const f = state.frame;
    const vL = signals.velL[f];
    const vR = signals.velR[f];
    setText("sd-vel-l", vL.toFixed(4), bucketColor(vL, cfg));
    setText("sd-vel-r", vR.toFixed(4), bucketColor(vR, cfg));
    setText("sd-state-l", describeFootState(vL, cfg));
    setText("sd-state-r", describeFootState(vR, cfg));
    setText("sd-fps", `${signals.fps.toFixed(1)} fps · smoothing ≈ ${signals.smoothFrames}f`);

    renderStatus(signals, f, cfg);

    // Highlight the active row in the events table if any.
    host.querySelectorAll("tr[data-event-idx]").forEach(tr => {
      const idx = parseInt(tr.getAttribute("data-event-idx"), 10);
      const ev = signals.events[idx];
      const active = ev && f >= ev.peak && f <= ev.plant;
      tr.classList.toggle("active", !!active);
    });

    drawVelTrace(host.querySelector("#sd-vel-canvas"), signals, f, cfg);
  },

  draw(ctx, state) {
    const f = state.frame;
    const p = state.pose;
    const s = state.renderScale || 1;
    const marginFrames = Math.max(1, Math.round(cfg.labelMarginMs * signals.fps / 1000));

    // Per ankle: velocity-bucket ring + live number, plus a thick
    // step-progress ring during the peak→plant span and PEAK/PLANT
    // labels at those exact frames.
    drawAnkle(ctx, p, f, J.L_ANKLE, "L", signals.velL,
      signals.events.filter(e => e.ankle === J.L_ANKLE), marginFrames, signals.fps, s, cfg);
    drawAnkle(ctx, p, f, J.R_ANKLE, "R", signals.velR,
      signals.events.filter(e => e.ankle === J.R_ANKLE), marginFrames, signals.fps, s, cfg);

    // Video overlay banner — only when we're inside a detected step.
    const active = activeStepAt(signals, f);
    if (active && (active.where === "peak" || active.where === "plant" || active.where === "inside")) {
      drawTopBanner(ctx, active, s);
    }
  },
};

// Big top-right banner on the video itself so the detection is obvious
// without having to look at the side panel.
function drawTopBanner(ctx, { ev, where }, scale) {
  const ankle = ev.ankle === J.L_ANKLE ? "L" : "R";
  const color = ev.ankle === J.L_ANKLE ? "#7ec8ff" : "#ff8a5c";
  const title = `STEP — ${ankle} ankle`;
  const sub = where === "peak"  ? `PEAK frame · vel ${ev.peakVel.toFixed(3)}`
            : where === "plant" ? `PLANT frame · dwell ${ev.plant - ev.peak}f`
            : `inside step · peak f${ev.peak} → plant f${ev.plant}`;

  ctx.save();
  const padX = 12 * scale;
  const padY = 10 * scale;
  const titleSize = Math.round(15 * scale);
  const subSize = Math.round(11 * scale);
  ctx.font = `bold ${titleSize}px ui-monospace, "SF Mono", monospace`;
  const tw = ctx.measureText(title).width;
  ctx.font = `${subSize}px ui-monospace, "SF Mono", monospace`;
  const sw = ctx.measureText(sub).width;
  const w = Math.max(tw, sw) + padX * 2;
  const h = titleSize + subSize + padY * 2 + 4 * scale;
  const x = ctx.canvas.width - w - 12 * scale;
  const y = 12 * scale;

  ctx.fillStyle = "rgba(0,0,0,0.78)";
  if (typeof ctx.roundRect === "function") {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 8 * scale);
    ctx.fill();
  } else {
    ctx.fillRect(x, y, w, h);
  }
  ctx.fillStyle = color;
  ctx.fillRect(x, y, 4 * scale, h);

  ctx.fillStyle = "#fff";
  ctx.font = `bold ${titleSize}px ui-monospace, "SF Mono", monospace`;
  ctx.textBaseline = "top";
  ctx.fillText(title, x + padX, y + padY);
  ctx.fillStyle = "#e6e9ef";
  ctx.font = `${subSize}px ui-monospace, "SF Mono", monospace`;
  ctx.fillText(sub, x + padX, y + padY + titleSize + 4 * scale);
  ctx.restore();
}

// ── DOM ────────────────────────────────────────────────────────────────────

function template() {
  return `
    <h2>Step detector</h2>
    <p class="hint">Velocity = ankle displacement between consecutive frames,
      normalized by torso height. A <b>step</b> is a contiguous run above
      <code>step_high</code>; the <b>plant</b> is the first frame after the
      peak where velocity stays below <code>step_low</code> for
      <code>plant_frames</code> frames. <span id="sd-fps" class="muted small">—</span></p>

    <h3>Now</h3>
    <div id="sd-step-status">—</div>

    <h3>Detected step events <span class="muted small" id="sd-events-count">(0)</span></h3>
    <div id="sd-events-table"></div>

    <h3>Current frame</h3>
    <div class="metric-grid">
      <div class="metric">
        <div class="metric-label">L ankle vel</div>
        <div class="metric-val" id="sd-vel-l">—</div>
        <div class="metric-sub muted" id="sd-state-l">—</div>
      </div>
      <div class="metric">
        <div class="metric-label">R ankle vel</div>
        <div class="metric-val" id="sd-vel-r">—</div>
        <div class="metric-sub muted" id="sd-state-r">—</div>
      </div>
    </div>

    <h3>Velocity (full clip)</h3>
    <p class="hint">L = blue, R = orange. Dashed lines = step_high (top) and
      step_low (bottom). The shaded bands span every detected step
      (peak → plant). Click anywhere to seek.</p>
    <canvas id="sd-vel-canvas" width="320" height="140"></canvas>

    <h3>Thresholds</h3>
    <label class="slider">
      <span>step_high = <output id="sd-high-out">${cfg.highVelThreshold.toFixed(3)}</output> <span class="muted small">torso/frame</span></span>
      <input type="range" id="sd-high" min="0.005" max="0.05" step="0.001" value="${cfg.highVelThreshold}">
      <span class="muted small">Onset gate. Ankle velocity must exceed this for at least one frame for a step to start. Higher = stricter (only obvious steps).</span>
    </label>
    <label class="slider">
      <span>step_low = <output id="sd-low-out">${cfg.lowVelThreshold.toFixed(3)}</output> <span class="muted small">torso/frame</span></span>
      <input type="range" id="sd-low" min="0.001" max="0.020" step="0.001" value="${cfg.lowVelThreshold}">
      <span class="muted small">Plant gate. After the peak, velocity must drop below this (for <code>plant_frames</code> frames) to register the plant. Lower = stricter (boxer must really stand still).</span>
    </label>
    <label class="slider">
      <span>smoothing = <output id="sd-smooth-out">${(cfg.velSmoothSeconds*1000).toFixed(0)} ms</output></span>
      <input type="range" id="sd-smooth" min="0" max="0.250" step="0.01" value="${cfg.velSmoothSeconds}">
      <span class="muted small">Moving-average window applied to the raw velocity. Filters jitter, but too wide blurs short spikes.</span>
    </label>
    <label class="slider">
      <span>plant_frames = <output id="sd-plantn-out">${cfg.plantFramesRequired.toFixed(0)} frames</output></span>
      <input type="range" id="sd-plantn" min="1" max="6" step="1" value="${cfg.plantFramesRequired}">
      <span class="muted small">How many consecutive sub-<code>step_low</code> frames the algorithm needs to call the foot "planted".</span>
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
    renderEventsTable();
    seekHack(state, state.frame);
  });
}

function renderEventsTable() {
  if (!signals) return;
  const rows = signals.events.map((e, i) => {
    const ankleLabel = e.ankle === J.L_ANKLE ? "L" : "R";
    const ankleCls = e.ankle === J.L_ANKLE ? "role-lead" : "role-rear";
    return `<tr data-seek="${e.peak}" data-event-idx="${i}">
      <td>${(e.peak / signals.fps).toFixed(2)}s</td>
      <td><span class="${ankleCls}">${ankleLabel}</span></td>
      <td>f${e.peak} → f${e.plant}</td>
      <td>${e.plant - e.peak}f</td>
      <td>${e.peakVel.toFixed(3)}</td>
    </tr>`;
  }).join("");
  host.querySelector("#sd-events-table").innerHTML = `
    <table class="sps-tbl">
      <thead><tr><th>t</th><th>Foot</th><th>Peak→Plant</th><th>Δ</th><th>Peak vel</th></tr></thead>
      <tbody>${rows || `<tr><td colspan="5" class="muted">no steps detected — try lowering step_high or raising step_low</td></tr>`}</tbody>
    </table>
  `;
  setText("sd-events-count", `(${signals.events.length} total)`);
  host.querySelectorAll("tr[data-seek]").forEach(tr => {
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => {
      seekHackSimple(parseInt(tr.getAttribute("data-seek"), 10));
    });
  });
}

// Big "STEP DETECTED" banner at the top of the panel, plus a relative
// position read-out telling the user where the cursor sits within the
// step's [peak, plant] span.
function renderStatus(signals, f, cfg) {
  const el = host.querySelector("#sd-step-status");
  if (!el) return;
  const active = activeStepAt(signals, f);
  if (!active) {
    el.innerHTML = `<div class="sd-banner sd-banner-idle">No active step at this frame</div>`;
    return;
  }
  const { ev, where } = active;
  const ankle = ev.ankle === J.L_ANKLE ? "L" : "R";
  const sideClass = ev.ankle === J.L_ANKLE ? "sd-banner-l" : "sd-banner-r";
  // Where in the span are we?
  let pos;
  if (where === "peak")  pos = `at PEAK frame`;
  else if (where === "plant") pos = `at PLANT frame`;
  else if (where === "inside") {
    const dwell = ev.plant - ev.peak;
    const into = f - ev.peak;
    pos = `inside step, frame ${into + 1} of ${dwell + 1}`;
  } else if (where === "approach") pos = `step about to start`;
  else if (where === "decay") pos = `step just ended`;
  el.innerHTML = `
    <div class="sd-banner ${sideClass}">
      STEP DETECTED · ${ankle} ankle
      <div class="sd-banner-sub">
        peak <b>f${ev.peak}</b> → plant <b>f${ev.plant}</b>
        · peak vel <b>${ev.peakVel.toFixed(3)}</b>
        · dwell <b>${ev.plant - ev.peak}f</b>
        · <i>${pos}</i>
      </div>
    </div>
  `;
}

// Returns { ev, where } where `where` is one of:
//   peak       — current frame is the peak
//   inside     — between peak and plant (exclusive of endpoints)
//   plant      — current frame is the plant
//   approach   — within labelMarginMs frames BEFORE the peak
//   decay      — within labelMarginMs frames AFTER the plant
// Or null when no step is anywhere near.
function activeStepAt(signals, f) {
  const margin = Math.max(1, Math.round(cfg.labelMarginMs * signals.fps / 1000));
  for (const ev of signals.events) {
    if (f === ev.peak)              return { ev, where: "peak" };
    if (f === ev.plant)             return { ev, where: "plant" };
    if (f > ev.peak && f < ev.plant) return { ev, where: "inside" };
    if (f >= ev.peak - margin && f < ev.peak)  return { ev, where: "approach" };
    if (f > ev.plant && f <= ev.plant + margin) return { ev, where: "decay" };
  }
  return null;
}

// ── Compute ────────────────────────────────────────────────────────────────

function computeAll(state, cfg) {
  const pose = state.pose;
  const fps = pose.fps || 30;
  const N = pose.n_frames;
  const smoothFrames = Math.max(1, Math.round(cfg.velSmoothSeconds * fps));
  const velL = ankleVel(pose, J.L_ANKLE, smoothFrames);
  const velR = ankleVel(pose, J.R_ANKLE, smoothFrames);
  const stepsL = findSteps(velL, cfg, J.L_ANKLE);
  const stepsR = findSteps(velR, cfg, J.R_ANKLE);
  const events = [...stepsL, ...stepsR].sort((a, b) => a.peak - b.peak);
  return { velL, velR, events, fps, smoothFrames };
}

function ankleVel(pose, ankleIdx, smoothFrames) {
  const N = pose.n_frames;
  const v = new Float32Array(N);
  for (let i = 1; i < N; i++) {
    const dx = pose.skeleton[(i * 17 + ankleIdx) * 2]     - pose.skeleton[((i-1) * 17 + ankleIdx) * 2];
    const dy = pose.skeleton[(i * 17 + ankleIdx) * 2 + 1] - pose.skeleton[((i-1) * 17 + ankleIdx) * 2 + 1];
    const th = Math.max(1e-6, torsoHeight(pose, i));
    v[i] = Math.hypot(dx, dy) / th;
  }
  return movingAvg(v, smoothFrames);
}

function movingAvg(arr, w) {
  if (w <= 1) return arr;
  const n = arr.length;
  const out = new Float32Array(n);
  const half = Math.floor(w / 2);
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - half);
    const hi = Math.min(n - 1, i + half);
    let s = 0;
    for (let k = lo; k <= hi; k++) s += arr[k];
    out[i] = s / (hi - lo + 1);
  }
  return out;
}

function findSteps(vel, cfg, ankleIdx) {
  const events = [];
  const N = vel.length;
  const high = cfg.highVelThreshold;
  const low  = cfg.lowVelThreshold;
  const need = Math.max(1, Math.round(cfg.plantFramesRequired));
  let i = 1;
  while (i < N) {
    if (vel[i] > high) {
      const runStart = i;
      while (i < N && vel[i] > high) i++;
      const runEnd = i;
      let peak = runStart, pv = vel[runStart];
      for (let k = runStart; k < runEnd; k++) {
        if (vel[k] > pv) { pv = vel[k]; peak = k; }
      }
      let plant = -1;
      // Need `need` consecutive frames below `low` (starting at j).
      for (let j = peak; j <= N - need; j++) {
        let ok = true;
        for (let q = 0; q < need; q++) {
          if (vel[j + q] >= low) { ok = false; break; }
        }
        if (ok) { plant = j; break; }
      }
      if (plant >= 0) events.push({ ankle: ankleIdx, peak, plant, peakVel: pv });
    } else {
      i++;
    }
  }
  return events;
}

// ── Per-frame helpers ──────────────────────────────────────────────────────

function bucketColor(v, cfg) {
  if (v > cfg.highVelThreshold) return COLORS.stepping;
  if (v < cfg.lowVelThreshold)  return COLORS.planted;
  return COLORS.transit;
}

function describeFootState(v, cfg) {
  if (v > cfg.highVelThreshold) return "stepping";
  if (v < cfg.lowVelThreshold)  return "planted";
  return "transitioning";
}

function describeContext(signals, frame, cfg) {
  // What's happening relative to step events at this frame?
  const margin = Math.max(1, Math.round(cfg.labelMarginMs * signals.fps / 1000));
  for (const e of signals.events) {
    if (Math.abs(frame - e.peak) <= 1)
      return `at PEAK (${e.ankle === J.L_ANKLE ? "L" : "R"}, vel ${e.peakVel.toFixed(3)})`;
    if (Math.abs(frame - e.plant) <= 1)
      return `at PLANT (${e.ankle === J.L_ANKLE ? "L" : "R"}, dwell ${(e.plant - e.peak)} frames)`;
    if (frame >= e.peak - margin && frame <= e.plant + margin) {
      const off = e.plant - frame;
      const ms = (off * 1000 / signals.fps).toFixed(0);
      return `inside step window (plant ${off >= 0 ? "in +" : ""}${ms} ms, ${e.ankle === J.L_ANKLE ? "L" : "R"})`;
    }
  }
  return "no step nearby";
}

// ── Drawing ────────────────────────────────────────────────────────────────

function drawAnkle(ctx, pose, frame, jointIdx, label, vel, events, marginFrames, fps, scale, cfg) {
  const c = pose.conf[frame * 17 + jointIdx];
  if (c < 0.05) return;
  const x = pose.skeleton[(frame * 17 + jointIdx) * 2];
  const y = pose.skeleton[(frame * 17 + jointIdx) * 2 + 1];
  const v = vel[frame] || 0;
  const ringColor = bucketColor(v, cfg);

  // Are we inside a detected step's full peak→plant span?
  let activeSpan = null;
  let nearActive = null;
  for (const e of events) {
    if (frame >= e.peak && frame <= e.plant) { activeSpan = e; break; }
    if (frame >= e.peak - marginFrames && frame <= e.plant + marginFrames) {
      nearActive = e;
    }
  }

  ctx.save();
  // Thick "step in progress" outer ring that lights up for every frame
  // between peak and plant — the unambiguous "I detected a step here" signal.
  if (activeSpan) {
    const accent = jointIdx === J.L_ANKLE ? "#7ec8ff" : "#ff8a5c";
    ctx.strokeStyle = accent;
    ctx.lineWidth = 4 * scale;
    ctx.shadowColor = accent;
    ctx.shadowBlur = 8 * scale;
    ctx.beginPath();
    ctx.arc(x, y, 16 * scale, 0, Math.PI * 2);
    ctx.stroke();
    ctx.shadowBlur = 0;
  }
  // Inner velocity-bucket ring — same as before.
  ctx.strokeStyle = ringColor;
  ctx.lineWidth = 2.2 * scale;
  ctx.beginPath();
  ctx.arc(x, y, 10 * scale, 0, Math.PI * 2);
  ctx.stroke();

  // Live velocity number to the right of the ankle.
  const fontPx = Math.round(11 * scale);
  ctx.font = `bold ${fontPx}px ui-monospace, "SF Mono", monospace`;
  const txt = `${label} ${v.toFixed(3)}`;
  const txtW = ctx.measureText(txt).width;
  const tx = x + 18 * scale;
  const ty = y + 4 * scale;
  ctx.fillStyle = "rgba(0,0,0,0.65)";
  ctx.fillRect(tx - 2, ty - fontPx, txtW + 6, fontPx + 4);
  ctx.fillStyle = ringColor;
  ctx.fillText(txt, tx + 2, ty);
  ctx.restore();

  // PEAK / PLANT markers (only at exactly those frames, hard-edged so the
  // user can tell them apart from the progress ring).
  const ev = activeSpan || nearActive;
  if (!ev) return;
  if (frame === ev.peak) {
    drawEventMarker(ctx, x, y, "PEAK", offsetText(ev.peak, frame, fps),
      COLORS.peakAccent, -22, scale);
  }
  if (frame === ev.plant) {
    drawEventMarker(ctx, x, y, "PLANT", offsetText(ev.plant, frame, fps),
      COLORS.plantAccent, +32, scale);
  }
}

function drawEventMarker(ctx, ankleX, ankleY, label, offset, color, dy, scale) {
  ctx.save();
  const fontPx = Math.round(11 * scale);
  ctx.font = `bold ${fontPx}px ui-monospace, "SF Mono", monospace`;
  const text = `${label} ${offset}`;
  const w = ctx.measureText(text).width;
  const pad = 4 * scale;
  const x = ankleX - w / 2 - pad;
  const y = ankleY + dy * scale;
  ctx.fillStyle = "rgba(0,0,0,0.78)";
  if (typeof ctx.roundRect === "function") {
    ctx.beginPath();
    ctx.roundRect(x, y - fontPx, w + pad * 2, fontPx + 4, 4 * scale);
    ctx.fill();
  } else {
    ctx.fillRect(x, y - fontPx, w + pad * 2, fontPx + 4);
  }
  ctx.fillStyle = color;
  ctx.fillText(text, x + pad, y);
  ctx.restore();
}

function offsetText(target, current, fps) {
  if (target == null) return "?";
  const delta = (target - current) * 1000 / fps;
  if (Math.abs(delta) < 1000 / fps / 2) return "now";
  if (delta > 0) return `in +${delta.toFixed(0)} ms`;
  return `${(-delta).toFixed(0)} ms ago`;
}

function drawVelTrace(canvas, signals, frame, cfg) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const N = signals.velL.length;
  const stride = Math.max(1, Math.floor(N / W));
  let yMax = Math.max(cfg.highVelThreshold * 2, 0.02);
  for (let f = 0; f < N; f += stride) {
    if (signals.velL[f] > yMax) yMax = signals.velL[f];
    if (signals.velR[f] > yMax) yMax = signals.velR[f];
  }
  const ymap = v => H - (v / yMax) * (H - 8) - 6;
  const xmap = f => (f / Math.max(1, N - 1)) * (W - 2) + 1;

  // Step spans — translucent full-height bands across [peak, plant].
  // L (blue) and R (orange) are tinted differently so overlapping bands stay
  // legible; the current-frame line is rendered on top.
  for (const e of signals.events) {
    const x1 = xmap(e.peak);
    const x2 = Math.max(x1 + 2, xmap(e.plant));
    ctx.fillStyle = (e.ankle === J.L_ANKLE)
      ? "rgba(126,200,255,0.18)"
      : "rgba(255,138,92,0.18)";
    ctx.fillRect(x1, 0, x2 - x1, H);
  }

  // Threshold lines.
  ctx.strokeStyle = "rgba(255,255,255,0.28)";
  ctx.setLineDash([3, 3]);
  ctx.lineWidth = 1;
  const yh = ymap(cfg.highVelThreshold), yl = ymap(cfg.lowVelThreshold);
  ctx.beginPath(); ctx.moveTo(0, yh); ctx.lineTo(W, yh); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, yl); ctx.lineTo(W, yl); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(255,255,255,0.7)";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(`step_high ${cfg.highVelThreshold.toFixed(3)}`, 4, yh - 2);
  ctx.fillText(`step_low ${cfg.lowVelThreshold.toFixed(3)}`, 4, yl + 10);

  drawLine(ctx, signals.velL, stride, xmap, ymap, "#7ec8ff");   // L = blue
  drawLine(ctx, signals.velR, stride, xmap, ymap, "#ff8a5c");   // R = orange

  // Peak + plant markers along the top edge so each span has clear bookends.
  for (const e of signals.events) {
    const color = (e.ankle === J.L_ANKLE) ? "#7ec8ff" : "#ff8a5c";
    ctx.fillStyle = color;
    const px = xmap(e.peak);
    const pl = xmap(e.plant);
    ctx.fillRect(px - 1, 0, 3, 6);   // peak — top
    ctx.fillRect(pl - 1, H - 6, 3, 6); // plant — bottom
  }

  // Current frame line.
  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.lineWidth = 1;
  const x = xmap(frame);
  ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
}

function drawLine(ctx, arr, stride, xmap, ymap, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (let f = 0; f < arr.length; f += stride) {
    const px = xmap(f), py = ymap(arr[f]);
    if (f === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();
}

// ── Tiny utilities ─────────────────────────────────────────────────────────

function setText(id, value, color) {
  const el = host.querySelector("#" + id);
  if (!el) return;
  el.innerHTML = value;
  if (color) el.style.color = color;
}

function seekHack(state, f) {
  const ev = new Event("input");
  const slider = document.getElementById("scrubber");
  if (!slider) return;
  slider.value = f;
  slider.dispatchEvent(ev);
}

function seekHackSimple(f) {
  seekHack(null, f);
}
