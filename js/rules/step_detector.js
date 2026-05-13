// Step detector — gap-focused debugging companion to step+punch-sync.
//
// Same per-punch algorithm as step_punch_sync (gap = inter-ankle distance
// over torso height, smoothed; baseline = median across the clip; a step
// is the gap peak inside each punch's search window if it clears
// baseline + min_step_extension). This lens trims the panel down to just
// the gap signal + per-punch verdicts so you can iterate on the
// thresholds without the sync-side distractions.
//
// Requires a punch source loaded (state.labels or state.punches). With no
// punches we can't run per-punch detection — the panel says so and the
// gap track is shown unannotated.

import { J, torsoHeight } from "../skeleton.js";

const DEFAULTS = {
  gapSmoothSeconds: 0.083,
  searchWindowSec:  0.4,
  minStepExtension: 0.10,
  labelMarginMs:    120,
};

const COLORS = {
  gapLine:       "#a78bfa",
  baseline:      "rgba(255,255,255,0.45)",
  ankle:         "#ffffff",
  ankleStepping: "#ef4444",
  searchBand:    "rgba(255,255,255,0.07)",
  plantGood:     "#5fd97a",
  plantBad:      "#e85a5a",
  landMark:      "#ff8a5c",
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
      boneColor: "rgba(255,255,255,0.20)",
      boneWidth: 1.5,
      jointRadius: 3,
      highlightJoints: new Set([
        J.L_ANKLE, J.R_ANKLE, J.L_HIP, J.R_HIP,
      ]),
    };
  },

  mount(_host, state) {
    host = _host;
    host.innerHTML = template();
    signals = computeAll(state, cfg);
    lastPose = state.pose;

    wireSlider(state, "#sd-ext",    "minStepExtension",  v => v.toFixed(2));
    wireSlider(state, "#sd-smooth", "gapSmoothSeconds",  v => `${(v*1000).toFixed(0)} ms`);
    wireSlider(state, "#sd-search", "searchWindowSec",   v => `${(v*1000).toFixed(0)} ms`);

    const canvas = host.querySelector("#sd-gap-canvas");
    if (canvas) {
      canvas.style.cursor = "pointer";
      canvas.addEventListener("click", evt => {
        const rect = canvas.getBoundingClientRect();
        const frac = (evt.clientX - rect.left) / rect.width;
        const N = state.pose.n_frames;
        const target = Math.max(0, Math.min(N - 1, Math.round(frac * (N - 1))));
        seekHackSimple(target);
      });
    }

    renderPunchesTable();
  },

  update(state) {
    if (state.pose !== lastPose) {
      signals = computeAll(state, cfg);
      lastPose = state.pose;
      renderPunchesTable();
    }
    const f = state.frame;
    setText("sd-gap",       signals.gap[f]?.toFixed(3) ?? "—");
    setText("sd-baseline",  signals.baseline.toFixed(3));
    setText("sd-ext-above", (signals.gap[f] - signals.baseline).toFixed(3));
    setText("sd-source",    sourceText(signals));

    renderStatus(signals, f, cfg);

    host.querySelectorAll("tr[data-punch-idx]").forEach(tr => {
      const idx = parseInt(tr.getAttribute("data-punch-idx"), 10);
      const p = signals.punches[idx];
      const active = p && f >= p.search_start && f <= p.search_end;
      tr.classList.toggle("active", !!active);
    });

    drawGapTrace(host.querySelector("#sd-gap-canvas"), signals, f, cfg);
  },

  draw(ctx, state) {
    const f = state.frame;
    const p = state.pose;
    const s = state.renderScale || 1;

    // Draw the gap line between the two ankles, color-coded by whether we're
    // currently above baseline + min_step_extension (= a "step-like" gap).
    drawGapSegment(ctx, p, f, signals, cfg, s);

    // Annotate each ankle with its joint label and the live gap value.
    drawAnkleLabel(ctx, p, f, J.L_ANKLE, "L", signals, s);
    drawAnkleLabel(ctx, p, f, J.R_ANKLE, "R", signals, s);

    // If we're inside a punch's search window, show a top-right banner.
    const active = activePunchAt(signals, f, cfg);
    if (active) drawTopBanner(ctx, active, signals, cfg, s);
  },
};

// ── DOM ────────────────────────────────────────────────────────────────────

function template() {
  return `
    <h2>Step detector</h2>
    <p class="hint">For each punch the lens looks at <code>gap[f]</code> =
      inter-ankle distance / torso height, inside a window around the
      punch. If the gap peak inside that window exceeds the boxer's
      baseline by <code>min_step_extension</code>, the peak frame is
      called the <b>plant</b>. <span id="sd-source" class="muted small"></span></p>

    <h3>Now</h3>
    <div id="sd-step-status">—</div>

    <h3>Current frame</h3>
    <div class="metric-grid">
      <div class="metric"><div class="metric-label">gap</div><div class="metric-val" id="sd-gap">—</div></div>
      <div class="metric"><div class="metric-label">baseline</div><div class="metric-val" id="sd-baseline">—</div></div>
      <div class="metric"><div class="metric-label">ext above baseline</div><div class="metric-val" id="sd-ext-above">—</div></div>
    </div>

    <h3>gap (full clip)</h3>
    <p class="hint">Translucent bands = punch search windows. Dashed = baseline
      and baseline + min_step_extension. Green ticks = step-found plant
      frames, red = same but out of sync, dotted orange = punch LAND.
      Click to seek.</p>
    <canvas id="sd-gap-canvas" width="320" height="160"></canvas>

    <h3>Per-punch verdicts</h3>
    <div id="sd-punches-table"></div>

    <h3>Thresholds</h3>
    <label class="slider">
      <span>min_step_extension = <output id="sd-ext-out">${cfg.minStepExtension.toFixed(2)}</output> <span class="muted small">torso</span></span>
      <input type="range" id="sd-ext" min="0.02" max="0.40" step="0.01" value="${cfg.minStepExtension}">
      <span class="muted small">How far the gap must rise above baseline to count as a step.</span>
    </label>
    <label class="slider">
      <span>search_window = <output id="sd-search-out">${(cfg.searchWindowSec*1000).toFixed(0)} ms</output></span>
      <input type="range" id="sd-search" min="0.10" max="0.80" step="0.05" value="${cfg.searchWindowSec}">
      <span class="muted small">±this far around each punch window we hunt for the gap peak.</span>
    </label>
    <label class="slider">
      <span>smoothing = <output id="sd-smooth-out">${(cfg.gapSmoothSeconds*1000).toFixed(0)} ms</output></span>
      <input type="range" id="sd-smooth" min="0" max="0.250" step="0.01" value="${cfg.gapSmoothSeconds}">
      <span class="muted small">Moving-average window. Filters tracker jitter + idle sway.</span>
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
    renderPunchesTable();
    seekHack(state, state.frame);
  });
}

function renderStatus(signals, f, cfg) {
  const el = host.querySelector("#sd-step-status");
  if (!el) return;
  const active = activePunchAt(signals, f, cfg);
  if (!active) {
    el.innerHTML = `<div class="sd-banner sd-banner-idle">Not inside any punch's search window</div>`;
    return;
  }
  const verdict = !active.step_found
    ? `NO STEP (peak gap +${active.gap_above_baseline.toFixed(2)} < ${cfg.minStepExtension.toFixed(2)})`
    : `STEP at f${active.plant_frame} (+${active.gap_above_baseline.toFixed(2)} above baseline)`;
  const cls = !active.step_found ? "sd-banner-idle"
    : active.is_out_of_sync ? "sd-banner-r"
    : "sd-banner-l";
  el.innerHTML = `
    <div class="sd-banner ${cls}">
      ${verdict}
      <div class="sd-banner-sub">
        punch <b>${(active.punch_type || "?").replace(/_/g, " ")}</b> · ${active.hand}·${active.side}
        · LAND <b>f${active.land_frame ?? "?"}</b>
        ${active.gap_ms != null
          ? `· sync gap <b>${active.gap_ms >= 0 ? "+" : ""}${active.gap_ms.toFixed(0)} ms</b>`
          : ""}
      </div>
    </div>`;
}

function renderPunchesTable() {
  if (!signals) return;
  const rows = signals.punches.map((p, i) => {
    const cls = !p.step_found ? "unscored" : (p.is_out_of_sync ? "skipped" : "scored");
    const stepCell = p.step_found
      ? `<span class="role-lead">+${p.gap_above_baseline.toFixed(2)}</span>`
      : `<span class="muted">+${p.gap_above_baseline.toFixed(2)}</span>`;
    const gapCell = p.gap_ms != null
      ? `${p.gap_ms >= 0 ? "+" : ""}${p.gap_ms.toFixed(0)} ms`
      : "—";
    return `<tr data-seek="${p.land_frame ?? p.start_frame}" data-punch-idx="${i}">
      <td>${p.timestamp.toFixed(2)}s</td>
      <td>${(p.punch_type || "?").replace(/_/g, " ")}</td>
      <td>${stepCell}</td>
      <td>${gapCell}</td>
    </tr>`;
  }).join("");
  const found = signals.punches.filter(p => p.step_found).length;
  host.querySelector("#sd-punches-table").innerHTML = `
    <p class="hint" style="margin:0 0 6px">${found}/${signals.punches.length} punches have a detected step.</p>
    <table class="sps-tbl">
      <thead><tr><th>t</th><th>Type</th><th>Ext above</th><th>Sync gap</th></tr></thead>
      <tbody>${rows || `<tr><td colspan="4" class="muted">no punches loaded</td></tr>`}</tbody>
    </table>
  `;
  host.querySelectorAll("tr[data-seek]").forEach(tr => {
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => {
      seekHackSimple(parseInt(tr.getAttribute("data-seek"), 10));
    });
  });
}

// ── Compute (shared algorithm with step+punch-sync) ────────────────────────

function computeAll(state, cfg) {
  const pose = state.pose;
  const fps = pose.fps || 30;
  const N = pose.n_frames;

  const gapRaw = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const lx = pose.skeleton[(i * 17 + J.L_ANKLE) * 2];
    const ly = pose.skeleton[(i * 17 + J.L_ANKLE) * 2 + 1];
    const rx = pose.skeleton[(i * 17 + J.R_ANKLE) * 2];
    const ry = pose.skeleton[(i * 17 + J.R_ANKLE) * 2 + 1];
    const th = Math.max(1e-6, torsoHeight(pose, i));
    gapRaw[i] = Math.hypot(lx - rx, ly - ry) / th;
  }
  const smoothFrames = Math.max(1, Math.round(cfg.gapSmoothSeconds * fps));
  const gap = movingAvg(gapRaw, smoothFrames);
  const baseline = median(gap);

  // Wrist-extension tracks per anatomical side for LAND detection.
  const extL = wristExt(pose, J.L_WRIST, J.L_SHOULDER);
  const extR = wristExt(pose, J.R_WRIST, J.R_SHOULDER);

  let source = "none";
  let detections = null;
  if (state.labels?.detections?.length) {
    source = "labels"; detections = state.labels.detections;
  } else if (state.punches?.detections?.length) {
    source = "stgcn"; detections = state.punches.detections;
  }

  const searchFrames = Math.max(2, Math.round(cfg.searchWindowSec * fps));
  const punches = (detections || []).map((d, idx) =>
    analyzePunch(d, idx, gap, baseline, extL, extR, searchFrames, N, fps, cfg)
  );

  return { gap, baseline, punches, source, fps, smoothFrames, searchFrames };
}

function analyzePunch(d, idx, gap, baseline, extL, extR, searchFrames, N, fps, cfg) {
  // Stance auto-defaults to orthodox here; the user can override via the
  // step+punch-sync lens stance dropdown. Both lenses share state but not
  // cfg so we don't try to be clever about which side stepped — we report
  // the gap-peak frame regardless. The step+punch-sync lens is the place
  // for stance-aware scoring.
  const side = d.hand === "lead" ? "L" : d.hand === "rear" ? "R" : "?";
  const ext = side === "L" ? extL : extR;
  const sf = Math.max(0, d.start_frame);
  const ef = Math.min(N - 1, d.end_frame);

  // LAND = peak wrist→shoulder extension in the punch window.
  let landFrame = null, bestExt = -Infinity;
  for (let f = sf; f <= ef; f++) {
    if (ext[f] > bestExt) { bestExt = ext[f]; landFrame = f; }
  }

  const searchStart = Math.max(1, sf - searchFrames);
  const searchEnd   = Math.min(N - 1, ef + searchFrames);
  let peakF = searchStart, peakG = gap[searchStart] || 0;
  for (let f = searchStart; f <= searchEnd; f++) {
    if (gap[f] > peakG) { peakG = gap[f]; peakF = f; }
  }
  const gapAboveBaseline = peakG - baseline;
  const stepFound = gapAboveBaseline >= cfg.minStepExtension;
  const gap_ms = stepFound && landFrame != null
    ? (landFrame - peakF) * 1000 / fps
    : null;
  // Out-of-sync uses a fixed 100ms threshold here for visualization — the
  // step+punch-sync lens owns the configurable sync tolerance.
  const isOut = stepFound && gap_ms != null && Math.abs(gap_ms) > 100;
  return {
    idx,
    timestamp: d.timestamp,
    hand: d.hand,
    side,
    punch_type: d.punch_type,
    start_frame: sf,
    end_frame: ef,
    search_start: searchStart,
    search_end: searchEnd,
    land_frame: landFrame,
    plant_frame: stepFound ? peakF : null,
    peak_gap: peakG,
    gap_above_baseline: gapAboveBaseline,
    step_found: stepFound,
    gap_ms,
    is_out_of_sync: !!isOut,
  };
}

function wristExt(pose, wristIdx, shoulderIdx) {
  const N = pose.n_frames;
  const e = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const wc = pose.conf[i * 17 + wristIdx];
    if (wc < 0.05) { e[i] = 0; continue; }
    const wx = pose.skeleton[(i * 17 + wristIdx) * 2];
    const wy = pose.skeleton[(i * 17 + wristIdx) * 2 + 1];
    const sx = pose.skeleton[(i * 17 + shoulderIdx) * 2];
    const sy = pose.skeleton[(i * 17 + shoulderIdx) * 2 + 1];
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

function median(arr) {
  if (!arr.length) return 0;
  const sorted = Array.from(arr).sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return (sorted.length & 1) ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function activePunchAt(signals, frame, cfg) {
  if (!signals?.punches?.length) return null;
  const margin = Math.round(cfg.labelMarginMs * signals.fps / 1000);
  let best = null;
  let bestDist = Infinity;
  for (const p of signals.punches) {
    const lo = p.search_start - margin;
    const hi = p.search_end + margin;
    if (frame < lo || frame > hi) continue;
    const center = p.land_frame ?? Math.round((p.start_frame + p.end_frame) / 2);
    const d = Math.abs(frame - center);
    if (d < bestDist) { best = p; bestDist = d; }
  }
  return best;
}

function sourceText(signals) {
  if (signals.source === "labels") return `${signals.fps.toFixed(1)} fps · smoothing ≈ ${signals.smoothFrames}f · using GT labels`;
  if (signals.source === "stgcn")  return `${signals.fps.toFixed(1)} fps · smoothing ≈ ${signals.smoothFrames}f · using ST-GCN punches`;
  return `${signals.fps.toFixed(1)} fps · smoothing ≈ ${signals.smoothFrames}f · no punches loaded — load GT or ST-GCN to detect steps`;
}

// ── Drawing ────────────────────────────────────────────────────────────────

function drawGapSegment(ctx, pose, frame, signals, cfg, scale) {
  const lc = pose.conf[frame * 17 + J.L_ANKLE];
  const rc = pose.conf[frame * 17 + J.R_ANKLE];
  if (lc < 0.05 || rc < 0.05) return;
  const lx = pose.skeleton[(frame * 17 + J.L_ANKLE) * 2];
  const ly = pose.skeleton[(frame * 17 + J.L_ANKLE) * 2 + 1];
  const rx = pose.skeleton[(frame * 17 + J.R_ANKLE) * 2];
  const ry = pose.skeleton[(frame * 17 + J.R_ANKLE) * 2 + 1];
  const aboveBaseline = (signals.gap[frame] - signals.baseline) >= cfg.minStepExtension;
  ctx.save();
  ctx.strokeStyle = aboveBaseline ? COLORS.ankleStepping : COLORS.gapLine;
  ctx.lineWidth = (aboveBaseline ? 3 : 2) * scale;
  ctx.globalAlpha = 0.85;
  ctx.beginPath();
  ctx.moveTo(lx, ly);
  ctx.lineTo(rx, ry);
  ctx.stroke();
  ctx.restore();
}

function drawAnkleLabel(ctx, pose, frame, jointIdx, label, signals, scale) {
  const c = pose.conf[frame * 17 + jointIdx];
  if (c < 0.05) return;
  const x = pose.skeleton[(frame * 17 + jointIdx) * 2];
  const y = pose.skeleton[(frame * 17 + jointIdx) * 2 + 1];
  ctx.save();
  ctx.fillStyle = COLORS.ankle;
  ctx.beginPath();
  ctx.arc(x, y, 5 * scale, 0, Math.PI * 2);
  ctx.fill();
  const fontPx = Math.round(11 * scale);
  ctx.font = `bold ${fontPx}px ui-monospace, "SF Mono", monospace`;
  const txt = `${label} gap=${signals.gap[frame]?.toFixed(3) ?? "?"}`;
  const tw = ctx.measureText(txt).width;
  const tx = x + 10 * scale;
  const ty = y + 4 * scale;
  ctx.fillStyle = "rgba(0,0,0,0.6)";
  ctx.fillRect(tx - 2, ty - fontPx, tw + 6, fontPx + 4);
  ctx.fillStyle = COLORS.gapLine;
  ctx.fillText(txt, tx + 2, ty);
  ctx.restore();
}

function drawTopBanner(ctx, p, signals, cfg, scale) {
  const title = p.step_found
    ? `STEP DETECTED · ${p.hand}·${p.side}`
    : `NO STEP (peak +${p.gap_above_baseline.toFixed(2)})`;
  const color = !p.step_found ? "#f5b945"
              : (p.is_out_of_sync ? "#e85a5a" : "#5fd97a");
  const sub = p.step_found
    ? `plant f${p.plant_frame} · gap above baseline +${p.gap_above_baseline.toFixed(2)} · sync gap ${p.gap_ms != null ? (p.gap_ms >= 0 ? "+" : "") + p.gap_ms.toFixed(0) + " ms" : "—"}`
    : `gap inside window peaked at +${p.gap_above_baseline.toFixed(2)}, threshold ${cfg.minStepExtension.toFixed(2)}`;
  ctx.save();
  const padX = 12 * scale, padY = 10 * scale;
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

function drawGapTrace(canvas, signals, frame, cfg) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const N = signals.gap.length;
  const stride = Math.max(1, Math.floor(N / W));

  let yMin = signals.baseline - 0.05;
  let yMax = signals.baseline + cfg.minStepExtension * 2;
  for (let f = 0; f < N; f += stride) {
    if (signals.gap[f] < yMin) yMin = signals.gap[f];
    if (signals.gap[f] > yMax) yMax = signals.gap[f];
  }
  yMin = Math.max(0, yMin - 0.02);
  yMax = yMax + 0.02;
  const ymap = v => H - ((v - yMin) / (yMax - yMin)) * (H - 8) - 4;
  const xmap = f => (f / Math.max(1, N - 1)) * (W - 2) + 1;

  // Search-window bands.
  for (const p of signals.punches) {
    const x1 = xmap(p.search_start);
    const x2 = Math.max(x1 + 2, xmap(p.search_end));
    ctx.fillStyle = !p.step_found ? "rgba(245,185,69,0.10)"
      : (p.is_out_of_sync ? "rgba(232,90,90,0.20)" : "rgba(95,217,122,0.18)");
    ctx.fillRect(x1, 0, x2 - x1, H);
  }

  // Baseline + threshold lines.
  ctx.strokeStyle = COLORS.baseline;
  ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
  const yb = ymap(signals.baseline);
  ctx.beginPath(); ctx.moveTo(0, yb); ctx.lineTo(W, yb); ctx.stroke();
  const yt = ymap(signals.baseline + cfg.minStepExtension);
  ctx.beginPath(); ctx.moveTo(0, yt); ctx.lineTo(W, yt); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(255,255,255,0.65)";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(`baseline ${signals.baseline.toFixed(3)}`, 4, yb - 2);
  ctx.fillText(`+ min_step ${cfg.minStepExtension.toFixed(2)}`, 4, yt - 2);

  drawLine(ctx, signals.gap, stride, xmap, ymap, COLORS.gapLine);

  // Plant + LAND ticks per punch.
  for (const p of signals.punches) {
    if (p.step_found) {
      const x = xmap(p.plant_frame);
      ctx.strokeStyle = p.is_out_of_sync ? COLORS.plantBad : COLORS.plantGood;
      ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    if (p.land_frame != null) {
      const x = xmap(p.land_frame);
      ctx.strokeStyle = COLORS.landMark;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // Current frame.
  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.lineWidth = 1;
  const x = xmap(frame);
  ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
}

function drawLine(ctx, arr, stride, xmap, ymap, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  for (let f = 0; f < arr.length; f += stride) {
    const px = xmap(f), py = ymap(arr[f]);
    if (f === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();
}

// ── Tiny utilities ─────────────────────────────────────────────────────────

function setText(id, value) {
  const el = host.querySelector("#" + id);
  if (!el) return;
  el.innerHTML = value;
}

function seekHack(_state, f) {
  const ev = new Event("input");
  const slider = document.getElementById("scrubber");
  if (!slider) return;
  slider.value = f;
  slider.dispatchEvent(ev);
}

function seekHackSimple(f) {
  seekHack(null, f);
}
