// Step-and-punch sync debug panel.
//
// What it visualises:
//   - Per-frame ankle velocity (torso-normalised) for both ankles, plotted as
//     two sparklines across the full clip with the step-high and step-low
//     threshold lines drawn in.
//   - Detected step events (peak + plant) on both ankles, marked on the
//     velocity sparkline and listed in a table.
//   - Per-frame wrist→same-side-shoulder distance (torso-normalised) as a
//     "candidate punch land" signal. Local maxima above a min-prominence
//     threshold are treated as punch-land candidates and listed.
//   - For each step plant, the nearest candidate punch land in a configurable
//     search window — and the resulting gap_ms. Out-of-sync rows highlighted.
//   - Overlay: ankles ringed (lead = green, rear = amber) plus shoulder→wrist
//     lines so you can see the extension signal we're measuring. When the
//     current frame is the plant frame for a detected step or the peak
//     frame for a detected wrist extension, that joint gets a thicker ring
//     and a label.
//
// Why no ST-GCN punch detection here: the GH-pages viewer is browser-only and
// runs off a pose cache; we can't run the trained ST-GCN model client-side.
// Wrist-extension peaks are an approximation that works well enough on
// stepping straights (jab/cross) — the punch lands exactly when the wrist is
// furthest from the shoulder. Use this to scrub the video and confirm whether
// the engine-side rule's plant/land picks line up with what you see.
//
// Mirrors the algorithm in cornerman_rules/rules/step_punch_sync.py so the
// numbers on screen agree with what the rule will compute when the same cache
// is run through the rules engine.

import { J, torsoHeight } from "../skeleton.js";

const DEFAULTS = {
  stance: "orthodox",          // orthodox = lead is L; southpaw = lead is R
  highVelThreshold: 0.020,     // ankle vel above which a "step" is in progress
  lowVelThreshold: 0.005,      // ankle vel below which the foot is "planted"
  velSmoothSeconds: 0.083,     // moving average over this many seconds
  searchWindowSec: 0.35,       // search around each step for paired wrist extension
  syncToleranceMs: 100,        // |gap_ms| above this counts as out of sync
  minWristExtension: 0.5,      // wrist→shoulder norm distance to consider a peak
};

const COLORS = {
  lead:    "#5fd97a",          // lead ankle (green)
  rear:    "#f5b945",          // rear ankle (amber)
  lWrist:  "#ff8a5c",
  rWrist:  "#ffd95c",
  lExt:    "#ff8a5c",
  rExt:    "#ffd95c",
  current: "rgba(255,255,255,0.8)",
  oos:     "#e85a5a",
  good:    "#5fd97a",
};

// Module-scoped state. mount() is called once when the user picks this rule;
// remount() resets it. Computed signals are cached so update()/draw() is cheap.
let host;
let cfg = { ...DEFAULTS };
let signals = null;            // { velL, velR, extL, extR, steps, extPeaks, paired }
let lastPose = null;           // detects pose changes between mount/update

export const StepPunchSyncRule = {
  id: "step_punch_sync",
  label: "Step + punch sync",

  skeletonStyle() {
    return {
      boneColor: "rgba(255,255,255,0.25)",
      boneWidth: 1.5,
      jointRadius: 3,
      highlightJoints: new Set([
        J.L_ANKLE, J.R_ANKLE,
        J.L_WRIST, J.R_WRIST,
        J.L_SHOULDER, J.R_SHOULDER,
      ]),
    };
  },

  mount(_host, state) {
    host = _host;
    host.innerHTML = template();
    signals = computeAll(state.pose, cfg);
    lastPose = state.pose;

    // Stance + threshold controls.
    host.querySelector("#sps-stance").value = cfg.stance;
    host.querySelector("#sps-stance").addEventListener("change", e => {
      cfg.stance = e.target.value;
      signals = computeAll(state.pose, cfg);    // pairing depends on lead foot
      renderTables();
      seekHack(state, state.frame);
    });

    wireSlider(state, "#sps-high",     "highVelThreshold", v => v.toFixed(3));
    wireSlider(state, "#sps-low",      "lowVelThreshold",  v => v.toFixed(3));
    wireSlider(state, "#sps-sync",     "syncToleranceMs",  v => `${v.toFixed(0)} ms`);
    wireSlider(state, "#sps-search",   "searchWindowSec",  v => `${(v*1000).toFixed(0)} ms`);
    wireSlider(state, "#sps-extmin",   "minWristExtension",v => v.toFixed(2));

    renderTables();
  },

  update(state) {
    if (state.pose !== lastPose) {
      // pose was swapped under us (engine compare, fresh round, etc.)
      signals = computeAll(state.pose, cfg);
      lastPose = state.pose;
      renderTables();
    }

    const f = state.frame;
    const p = state.pose;

    const leadIdx = cfg.stance === "orthodox" ? J.L_ANKLE : J.R_ANKLE;
    const rearIdx = cfg.stance === "orthodox" ? J.R_ANKLE : J.L_ANKLE;
    const vLead = leadIdx === J.L_ANKLE ? signals.velL[f] : signals.velR[f];
    const vRear = leadIdx === J.L_ANKLE ? signals.velR[f] : signals.velL[f];

    setText("sps-vel-lead", vLead.toFixed(4), velColor(vLead, cfg));
    setText("sps-vel-rear", vRear.toFixed(4), velColor(vRear, cfg));
    setText("sps-ext-l",    signals.extL[f].toFixed(2));
    setText("sps-ext-r",    signals.extR[f].toFixed(2));

    setText("sps-step-state", stepStateAt(signals, f, leadIdx, cfg));

    // Active step / extension peak — drives the LAND/PLANT label drawing.
    setText("sps-active",    activeContextAt(signals, f, cfg));

    drawVelTrace(host.querySelector("#sps-vel-canvas"), signals, f, cfg);
    drawExtTrace(host.querySelector("#sps-ext-canvas"), signals, f, cfg);
  },

  draw(ctx, state) {
    const f = state.frame;
    const p = state.pose;
    const s = state.renderScale || 1;

    const leadIdx = cfg.stance === "orthodox" ? J.L_ANKLE : J.R_ANKLE;
    const rearIdx = cfg.stance === "orthodox" ? J.R_ANKLE : J.L_ANKLE;

    // Shoulder→wrist segments (both sides) — what we use to detect extension.
    drawSeg(ctx, p, f, J.L_SHOULDER, J.L_WRIST, COLORS.lExt, s);
    drawSeg(ctx, p, f, J.R_SHOULDER, J.R_WRIST, COLORS.rExt, s);

    // Lead + rear ankle rings (always on).
    drawAnkleRing(ctx, p, f, leadIdx, COLORS.lead, "LEAD", s);
    drawAnkleRing(ctx, p, f, rearIdx, COLORS.rear, "rear", s);

    // If current frame is within ±1 frame of a step plant for either ankle,
    // emphasize that ankle and label it PLANT.
    for (const ev of signals.steps) {
      if (Math.abs(f - ev.plant) <= 1) {
        const color = (ev.ankle === leadIdx) ? COLORS.lead : COLORS.rear;
        emphasizeJoint(ctx, p, f, ev.ankle, color, "PLANT", 18, s);
      }
    }

    // If current frame is the peak of a wrist extension, ring that wrist.
    for (const peak of signals.extPeaks) {
      if (Math.abs(f - peak.frame) <= 1) {
        const wristJ = peak.side === "L" ? J.L_WRIST : J.R_WRIST;
        const color = peak.side === "L" ? COLORS.lExt : COLORS.rExt;
        emphasizeJoint(ctx, p, f, wristJ, color, "EXT", 16, s);
      }
    }

    // Status banner top-left.
    drawBanner(ctx, contextBannerAt(signals, f, cfg), s);
  },
};

// ── DOM ────────────────────────────────────────────────────────────────────

function template() {
  return `
    <h2>Step + punch sync</h2>
    <p class="hint">Coach cue: "step and punch land together". When you step
      on a punch, the foot plant frame and the moment of full extension
      should be within a few frames of each other.</p>

    <h3>Stance</h3>
    <select id="sps-stance">
      <option value="orthodox">Orthodox (lead = L)</option>
      <option value="southpaw">Southpaw (lead = R)</option>
    </select>

    <h3>Current frame</h3>
    <div class="metric-grid">
      <div class="metric"><div class="metric-label">Lead ankle vel</div><div class="metric-val" id="sps-vel-lead">—</div></div>
      <div class="metric"><div class="metric-label">Rear ankle vel</div><div class="metric-val" id="sps-vel-rear">—</div></div>
      <div class="metric"><div class="metric-label">L wrist ext</div><div class="metric-val" id="sps-ext-l">—</div></div>
      <div class="metric"><div class="metric-label">R wrist ext</div><div class="metric-val" id="sps-ext-r">—</div></div>
    </div>
    <p class="hint" style="margin-top:4px"><span id="sps-step-state" class="muted">—</span> · <span id="sps-active" class="muted">—</span></p>

    <h3>Ankle velocity (full clip)</h3>
    <p class="hint">Green = lead ankle, amber = rear. Dashed lines = step
      thresholds. Vertical ticks = detected step plants.</p>
    <canvas id="sps-vel-canvas" width="320" height="120"></canvas>

    <h3>Wrist → shoulder distance</h3>
    <p class="hint">Local maxima above min-extension are treated as candidate
      punch-land frames. Vertical ticks = peaks.</p>
    <canvas id="sps-ext-canvas" width="320" height="80"></canvas>

    <h3>Paired step → punch</h3>
    <p class="hint">For each step plant, the nearest wrist extension peak
      within the search window. <b>Only lead-foot steps</b> are scored.</p>
    <div id="sps-pair-table"></div>

    <h3>All step events</h3>
    <div id="sps-step-table"></div>

    <h3>Thresholds</h3>
    <label class="slider">
      <span>step_high = <output id="sps-high-out">${cfg.highVelThreshold.toFixed(3)}</output></span>
      <input type="range" id="sps-high" min="0.005" max="0.05" step="0.001" value="${cfg.highVelThreshold}">
    </label>
    <label class="slider">
      <span>step_low = <output id="sps-low-out">${cfg.lowVelThreshold.toFixed(3)}</output></span>
      <input type="range" id="sps-low" min="0.001" max="0.02" step="0.001" value="${cfg.lowVelThreshold}">
    </label>
    <label class="slider">
      <span>sync_tolerance = <output id="sps-sync-out">${cfg.syncToleranceMs.toFixed(0)} ms</output></span>
      <input type="range" id="sps-sync" min="20" max="300" step="10" value="${cfg.syncToleranceMs}">
    </label>
    <label class="slider">
      <span>search_window = <output id="sps-search-out">${(cfg.searchWindowSec*1000).toFixed(0)} ms</output></span>
      <input type="range" id="sps-search" min="0.10" max="0.80" step="0.05" value="${cfg.searchWindowSec}">
    </label>
    <label class="slider">
      <span>min_wrist_extension = <output id="sps-extmin-out">${cfg.minWristExtension.toFixed(2)}</output></span>
      <input type="range" id="sps-extmin" min="0.20" max="1.20" step="0.05" value="${cfg.minWristExtension}">
    </label>
  `;
}

function wireSlider(state, sel, key, fmt) {
  const s = host.querySelector(sel);
  const out = host.querySelector(sel + "-out");
  s.addEventListener("input", () => {
    cfg[key] = parseFloat(s.value);
    out.textContent = fmt(cfg[key]);
    // Recompute step events + pairings — they all depend on these.
    signals = computeAll(state.pose, cfg);
    renderTables();
    seekHack(state, state.frame);
  });
}

function renderTables() {
  if (!signals) return;
  const leadIdx = cfg.stance === "orthodox" ? J.L_ANKLE : J.R_ANKLE;
  const leadLabel = leadIdx === J.L_ANKLE ? "L" : "R";

  // Paired table — one row per step plant.
  const paired = signals.paired;
  const pairRows = paired.map((p, i) => {
    const ankleLbl = p.step.ankle === J.L_ANKLE ? "L" : "R";
    const role = p.step.ankle === leadIdx ? "lead" : "rear";
    const cls = role === "lead" ? "scored" : "unscored";
    let gapHtml = "—";
    if (p.peak) {
      const gapCls = role === "lead"
        ? (Math.abs(p.gap_ms) <= cfg.syncToleranceMs ? "good" : "bad")
        : "muted";
      const sign = p.gap_ms >= 0 ? "+" : "";
      gapHtml = `<span class="${gapCls}">${sign}${p.gap_ms.toFixed(0)} ms</span>`;
    }
    const ts = (p.step.plant / signals.fps).toFixed(2);
    const peakInfo = p.peak
      ? `${p.peak.side}·ext ${p.peak.value.toFixed(2)} @ f${p.peak.frame}`
      : `<span class="muted">no peak in window</span>`;
    return `
      <tr class="${cls}" data-seek="${p.step.plant}">
        <td>${ts}s</td>
        <td><span class="role-${role}">${ankleLbl}·${role}</span></td>
        <td>f${p.step.plant}</td>
        <td>${peakInfo}</td>
        <td>${gapHtml}</td>
      </tr>`;
  }).join("");

  const scored = paired.filter(p => p.step.ankle === leadIdx).length;
  const oos    = paired.filter(p =>
    p.step.ankle === leadIdx && p.peak && Math.abs(p.gap_ms) > cfg.syncToleranceMs
  ).length;

  host.querySelector("#sps-pair-table").innerHTML = `
    <p class="hint" style="margin:0 0 6px"><b>${oos}/${scored}</b> lead-foot
      steps out of sync · sync tol ±${cfg.syncToleranceMs.toFixed(0)} ms · lead ${leadLabel}</p>
    <table class="sps-tbl">
      <thead><tr><th>t</th><th>Foot</th><th>Plant</th><th>Paired peak</th><th>Gap</th></tr></thead>
      <tbody>${pairRows || `<tr><td colspan="5" class="muted">no step events detected — try lowering step_high.</td></tr>`}</tbody>
    </table>
  `;

  // All step events.
  const stepRows = signals.steps.map(ev => {
    const ankleLbl = ev.ankle === J.L_ANKLE ? "L" : "R";
    const role = ev.ankle === leadIdx ? "lead" : "rear";
    return `<tr data-seek="${ev.plant}">
      <td>${(ev.plant / signals.fps).toFixed(2)}s</td>
      <td><span class="role-${role}">${ankleLbl}·${role}</span></td>
      <td>peak f${ev.peak} → plant f${ev.plant}</td>
      <td>${ev.peakVel.toFixed(3)}</td>
    </tr>`;
  }).join("");
  host.querySelector("#sps-step-table").innerHTML = `
    <table class="sps-tbl">
      <thead><tr><th>t</th><th>Foot</th><th>Peak→Plant</th><th>Peak vel</th></tr></thead>
      <tbody>${stepRows || `<tr><td colspan="4" class="muted">none</td></tr>`}</tbody>
    </table>
  `;

  // Wire click-to-seek on every row.
  host.querySelectorAll("tr[data-seek]").forEach(tr => {
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => {
      const f = parseInt(tr.getAttribute("data-seek"), 10);
      seekHackFromTable(f);
    });
  });
}

// ── Compute ────────────────────────────────────────────────────────────────

function computeAll(pose, cfg) {
  const N = pose.n_frames;
  const fps = pose.fps || 30;

  const velL = ankleVel(pose, J.L_ANKLE, cfg.velSmoothSeconds, fps);
  const velR = ankleVel(pose, J.R_ANKLE, cfg.velSmoothSeconds, fps);
  const extL = wristExt(pose, J.L_WRIST, J.L_SHOULDER);
  const extR = wristExt(pose, J.R_WRIST, J.R_SHOULDER);

  const stepsL = findSteps(velL, cfg.highVelThreshold, cfg.lowVelThreshold, J.L_ANKLE);
  const stepsR = findSteps(velR, cfg.highVelThreshold, cfg.lowVelThreshold, J.R_ANKLE);
  const steps = [...stepsL, ...stepsR].sort((a, b) => a.plant - b.plant);

  const peaksL = findPeaks(extL, cfg.minWristExtension, "L");
  const peaksR = findPeaks(extR, cfg.minWristExtension, "R");
  const extPeaks = [...peaksL, ...peaksR].sort((a, b) => a.frame - b.frame);

  // Pair each step with the nearest wrist-extension peak within search window.
  const searchFrames = Math.max(2, Math.round(cfg.searchWindowSec * fps));
  const paired = steps.map(step => {
    let best = null;
    for (const peak of extPeaks) {
      const dist = Math.abs(peak.frame - step.plant);
      if (dist > searchFrames) continue;
      if (!best || dist < Math.abs(best.frame - step.plant)) best = peak;
    }
    const gap_ms = best ? (best.frame - step.plant) * 1000 / fps : null;
    return { step, peak: best, gap_ms };
  });

  return { velL, velR, extL, extR, steps, extPeaks, paired, fps };
}

function ankleVel(pose, ankleIdx, smoothSec, fps) {
  const N = pose.n_frames;
  const v = new Float32Array(N);
  for (let i = 1; i < N; i++) {
    const dx = pose.skeleton[(i * 17 + ankleIdx) * 2]     - pose.skeleton[((i-1) * 17 + ankleIdx) * 2];
    const dy = pose.skeleton[(i * 17 + ankleIdx) * 2 + 1] - pose.skeleton[((i-1) * 17 + ankleIdx) * 2 + 1];
    const dist = Math.hypot(dx, dy);
    const th = Math.max(1e-6, torsoHeight(pose, i));
    v[i] = dist / th;
  }
  const w = Math.max(1, Math.round(smoothSec * fps));
  return movingAvg(v, w);
}

function wristExt(pose, wristIdx, shoulderIdx) {
  const N = pose.n_frames;
  const e = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const wx = pose.skeleton[(i * 17 + wristIdx) * 2];
    const wy = pose.skeleton[(i * 17 + wristIdx) * 2 + 1];
    const sx = pose.skeleton[(i * 17 + shoulderIdx) * 2];
    const sy = pose.skeleton[(i * 17 + shoulderIdx) * 2 + 1];
    const wc = pose.conf[i * 17 + wristIdx];
    if (wc < 0.05) { e[i] = 0; continue; }
    const dx = wx - sx, dy = wy - sy;
    const th = Math.max(1e-6, torsoHeight(pose, i));
    e[i] = Math.hypot(dx, dy) / th;
  }
  return e;
}

function movingAvg(arr, w) {
  if (w <= 1) return arr;
  const n = arr.length;
  const out = new Float32Array(n);
  let sum = 0;
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

// Step = contiguous run above HIGH; plant = first frame after the peak where
// velocity sits below LOW for at least 2 frames.
function findSteps(vel, high, low, ankleIdx) {
  const events = [];
  const N = vel.length;
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
      for (let j = peak; j < N - 1; j++) {
        if (vel[j] < low && vel[j + 1] < low) { plant = j; break; }
      }
      if (plant >= 0) events.push({ ankle: ankleIdx, peak, plant, peakVel: pv });
    } else {
      i++;
    }
  }
  return events;
}

// Local maxima above a min-prominence threshold. We don't need fancy
// prominence — just a magnitude floor + simple rising/falling test.
function findPeaks(arr, minVal, side) {
  const peaks = [];
  const N = arr.length;
  for (let i = 2; i < N - 2; i++) {
    if (arr[i] < minVal) continue;
    if (arr[i] >= arr[i-1] && arr[i] >= arr[i-2] &&
        arr[i] >= arr[i+1] && arr[i] >= arr[i+2]) {
      // Suppress same-peak duplicates — collapse runs of equal values.
      if (peaks.length && i - peaks[peaks.length - 1].frame < 8) continue;
      peaks.push({ frame: i, value: arr[i], side });
    }
  }
  return peaks;
}

// ── Per-frame label helpers ────────────────────────────────────────────────

function stepStateAt(signals, f, leadIdx, cfg) {
  const v = leadIdx === J.L_ANKLE ? signals.velL[f] : signals.velR[f];
  if (v > cfg.highVelThreshold) return `lead foot stepping (vel ${v.toFixed(4)})`;
  if (v < cfg.lowVelThreshold)  return `lead foot planted (vel ${v.toFixed(4)})`;
  return `lead foot mid-transition (vel ${v.toFixed(4)})`;
}

function activeContextAt(signals, f, cfg) {
  for (const ev of signals.steps) {
    if (Math.abs(f - ev.plant) <= 1) {
      const lbl = ev.ankle === J.L_ANKLE ? "L" : "R";
      return `at PLANT (${lbl} ankle, peak vel ${ev.peakVel.toFixed(3)})`;
    }
  }
  for (const peak of signals.extPeaks) {
    if (Math.abs(f - peak.frame) <= 1) {
      return `at EXT peak (${peak.side} wrist, ext ${peak.value.toFixed(2)})`;
    }
  }
  return "—";
}

function contextBannerAt(signals, f, cfg) {
  // Show "punch land + step plant gap" when both are happening near each
  // other — that's the moment the rule actually scores.
  for (const pair of signals.paired) {
    if (Math.abs(f - pair.step.plant) <= 1 && pair.peak) {
      const sign = pair.gap_ms >= 0 ? "+" : "";
      const inSync = Math.abs(pair.gap_ms) <= cfg.syncToleranceMs;
      return {
        text: `gap ${sign}${pair.gap_ms.toFixed(0)} ms`,
        color: inSync ? COLORS.good : COLORS.oos,
      };
    }
  }
  return null;
}

// ── Drawing ────────────────────────────────────────────────────────────────

function drawSeg(ctx, pose, frame, a, b, color, scale) {
  const ax = pose.skeleton[(frame * 17 + a) * 2];
  const ay = pose.skeleton[(frame * 17 + a) * 2 + 1];
  const bx = pose.skeleton[(frame * 17 + b) * 2];
  const by = pose.skeleton[(frame * 17 + b) * 2 + 1];
  const ac = pose.conf[frame * 17 + a];
  const bc = pose.conf[frame * 17 + b];
  if (ac < 0.05 || bc < 0.05) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2 * scale;
  ctx.globalAlpha = 0.85;
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(bx, by);
  ctx.stroke();
  ctx.restore();
}

function drawAnkleRing(ctx, pose, frame, jointIdx, color, label, scale) {
  const x = pose.skeleton[(frame * 17 + jointIdx) * 2];
  const y = pose.skeleton[(frame * 17 + jointIdx) * 2 + 1];
  const c = pose.conf[frame * 17 + jointIdx];
  if (c < 0.05) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2 * scale;
  ctx.beginPath();
  ctx.arc(x, y, 9 * scale, 0, Math.PI * 2);
  ctx.stroke();
  // Label with dark shadow for legibility on bright frames.
  const fontPx = Math.round(11 * scale);
  ctx.font = `bold ${fontPx}px ui-monospace, monospace`;
  ctx.fillStyle = "rgba(0,0,0,0.65)";
  ctx.fillText(label, x + 12 * scale + 1, y + 4 * scale + 1);
  ctx.fillStyle = color;
  ctx.fillText(label, x + 12 * scale, y + 4 * scale);
  ctx.restore();
}

function emphasizeJoint(ctx, pose, frame, jointIdx, color, label, radius, scale) {
  const x = pose.skeleton[(frame * 17 + jointIdx) * 2];
  const y = pose.skeleton[(frame * 17 + jointIdx) * 2 + 1];
  const c = pose.conf[frame * 17 + jointIdx];
  if (c < 0.05) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3 * scale;
  ctx.beginPath();
  ctx.arc(x, y, radius * scale, 0, Math.PI * 2);
  ctx.stroke();
  const fontPx = Math.round(12 * scale);
  ctx.font = `bold ${fontPx}px ui-monospace, monospace`;
  const tw = ctx.measureText(label).width;
  ctx.fillStyle = "rgba(0,0,0,0.75)";
  ctx.fillRect(x + (radius + 4) * scale - 2, y - fontPx - 2, tw + 6, fontPx + 4);
  ctx.fillStyle = color;
  ctx.fillText(label, x + (radius + 4) * scale, y - 4);
  ctx.restore();
}

function drawBanner(ctx, banner, scale) {
  if (!banner) return;
  ctx.save();
  const fontPx = Math.round(13 * scale);
  ctx.font = `bold ${fontPx}px ui-monospace, monospace`;
  const pad = 6 * scale;
  const tw = ctx.measureText(banner.text).width;
  ctx.fillStyle = "rgba(0,0,0,0.65)";
  ctx.fillRect(8 * scale, 8 * scale, tw + pad * 2, fontPx + pad * 2);
  ctx.fillStyle = banner.color;
  ctx.fillText(banner.text, 8 * scale + pad, 8 * scale + pad + fontPx - 2);
  ctx.restore();
}

function drawVelTrace(canvas, signals, frame, cfg) {
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
  const ymap = v => H - (v / yMax) * (H - 4) - 2;
  const xmap = f => (f / (N - 1)) * (W - 2) + 1;

  // Threshold lines
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.setLineDash([3, 3]);
  ctx.lineWidth = 1;
  let yh = ymap(cfg.highVelThreshold), yl = ymap(cfg.lowVelThreshold);
  ctx.beginPath(); ctx.moveTo(0, yh); ctx.lineTo(W, yh); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, yl); ctx.lineTo(W, yl); ctx.stroke();
  ctx.setLineDash([]);

  // Pick which is lead based on stance for color choice
  const leadIsL = cfg.stance === "orthodox";
  const colorL = leadIsL ? COLORS.lead : COLORS.rear;
  const colorR = leadIsL ? COLORS.rear : COLORS.lead;

  drawLine(ctx, signals.velL, stride, xmap, ymap, colorL);
  drawLine(ctx, signals.velR, stride, xmap, ymap, colorR);

  // Step plant ticks
  for (const ev of signals.steps) {
    const x = xmap(ev.plant);
    const color = (ev.ankle === J.L_ANKLE) ? colorL : colorR;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, H - 4);
    ctx.lineTo(x, H);
    ctx.stroke();
  }

  // Current frame line
  ctx.strokeStyle = COLORS.current;
  ctx.lineWidth = 1;
  const x = xmap(frame);
  ctx.beginPath();
  ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
}

function drawExtTrace(canvas, signals, frame, cfg) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const N = signals.extL.length;
  const stride = Math.max(1, Math.floor(N / W));
  let yMax = cfg.minWristExtension * 1.5;
  for (let f = 0; f < N; f += stride) {
    if (signals.extL[f] > yMax) yMax = signals.extL[f];
    if (signals.extR[f] > yMax) yMax = signals.extR[f];
  }
  const ymap = v => H - (v / yMax) * (H - 4) - 2;
  const xmap = f => (f / (N - 1)) * (W - 2) + 1;

  // Min-extension threshold line
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.setLineDash([3, 3]);
  ctx.lineWidth = 1;
  const yt = ymap(cfg.minWristExtension);
  ctx.beginPath(); ctx.moveTo(0, yt); ctx.lineTo(W, yt); ctx.stroke();
  ctx.setLineDash([]);

  drawLine(ctx, signals.extL, stride, xmap, ymap, COLORS.lExt);
  drawLine(ctx, signals.extR, stride, xmap, ymap, COLORS.rExt);

  // Peak ticks
  for (const peak of signals.extPeaks) {
    const x = xmap(peak.frame);
    const color = peak.side === "L" ? COLORS.lExt : COLORS.rExt;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, 4);
    ctx.stroke();
  }

  // Current frame
  ctx.strokeStyle = COLORS.current;
  ctx.lineWidth = 1;
  const x = xmap(frame);
  ctx.beginPath();
  ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
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

function velColor(v, cfg) {
  if (v > cfg.highVelThreshold) return "#7ec8ff";
  if (v < cfg.lowVelThreshold) return "#5fd97a";
  return "#f5b945";
}

function seekHack(state, f) {
  // Same approach as guard_drop.js — bounce the scrubber to force a redraw.
  const ev = new Event("input");
  const slider = document.getElementById("scrubber");
  if (!slider) return;
  slider.value = f;
  slider.dispatchEvent(ev);
}

function seekHackFromTable(f) {
  const ev = new Event("input");
  const slider = document.getElementById("scrubber");
  if (!slider) return;
  slider.value = f;
  slider.dispatchEvent(ev);
}
