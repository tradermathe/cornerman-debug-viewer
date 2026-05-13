// Hip rotation lens — sanity-check the proposed rule against the labeler's
// `rule_hip_rotation` verdicts directly on the source video.
//
// Per labeled punch, we compute the same metric the validation notebook
// uses (notebooks/hip_rotation_validation.ipynb in cornerman-backend):
//
//   gap[f]   = euclidean(L_hip[f], R_hip[f]) / torso_height[f]   (smoothed)
//   baseline = median(gap) across the whole round
//   per punch:
//     search        = [punch.start − search_window, punch.end + search_window]
//     peak_gap      = max(gap[search])
//     range_gap     = max(gap[search]) − min(gap[search])
//     total_var     = Σ |gap[f+1] − gap[f]|  in search
//     verdict       = peak_gap − baseline ≥ min_step_extension ? pass : fail
//
// The lens needs labels (state.labels) to be useful — without them the
// punches don't carry a coach verdict and there's nothing to compare to.
//
// What you'll see:
//   * Video overlay: hip line drawn purple, persistent labels at the
//     LAND frame of each punch ("label=fail · pred=pass ✗").
//   * Timeline (in the side panel): one colored band per labeled punch
//     across the gap track, color = the coach's label (green/red), with
//     a smaller inner tick = our prediction. Click to seek to that punch.
//   * Per-punch table with all three candidate signals + agreement.
//   * Slider for min_delta — drag to see false-positive/-negative trade
//     in real time.

import { J, torsoHeight } from "../skeleton.js";
import { activePunches, drawPunchHudStack } from "./overview.js";

const DEFAULTS = {
  gapSmoothSeconds: 0.083,
  searchWindowSec:  0.4,
  minDelta:         0.05,    // peak − baseline threshold for "rotation enough"
  // Punch types the rule actually applies to (jab + body shots excluded —
  // the labeler doesn't flag rotation on those).
  appliesTo: new Set([
    "cross_head", "cross_body",
    "lead_hook_head", "lead_uppercut_head",
    "rear_uppercut_head", "rear_hook_head",
  ]),
};

const COLORS = {
  hipLine:     "#a78bfa",
  hipLineWeak: "rgba(167,139,250,0.35)",
  labelPass:   "#5fd97a",
  labelFail:   "#e85a5a",
  predBand:    "rgba(255,255,255,0.18)",
  current:     "rgba(255,255,255,0.85)",
  baseline:    "rgba(255,255,255,0.45)",
  agree:       "#5fd97a",
  disagree:    "#e85a5a",
  unclear:     "#f5b945",
};

let host;
let cfg = { ...DEFAULTS };
let signals = null;
let lastPose = null;

export const HipRotationRule = {
  id: "hip_rotation",
  label: "Hip rotation (vs labels)",

  skeletonStyle() {
    return {
      boneColor: "rgba(255,255,255,0.22)",
      boneWidth: 1.5,
      jointRadius: 3,
      highlightJoints: new Set([J.L_HIP, J.R_HIP, J.L_SHOULDER, J.R_SHOULDER]),
    };
  },

  mount(_host, state) {
    host = _host;
    host.innerHTML = template();
    signals = computeAll(state, cfg);
    lastPose = state.pose;

    wireSlider(state, "#hr-min",    "minDelta",         v => v.toFixed(2));
    wireSlider(state, "#hr-smooth", "gapSmoothSeconds", v => `${(v*1000).toFixed(0)} ms`);
    wireSlider(state, "#hr-search", "searchWindowSec",  v => `${(v*1000).toFixed(0)} ms`);

    const canvas = host.querySelector("#hr-gap-canvas");
    if (canvas) {
      canvas.style.cursor = "pointer";
      canvas.addEventListener("click", evt => {
        const rect = canvas.getBoundingClientRect();
        const frac = (evt.clientX - rect.left) / rect.width;
        const N = state.pose.n_frames;
        seekHack(Math.max(0, Math.min(N - 1, Math.round(frac * (N - 1)))));
      });
    }

    renderHeader(state);
    renderTable();
  },

  update(state) {
    if (state.pose !== lastPose) {
      signals = computeAll(state, cfg);
      lastPose = state.pose;
      renderHeader(state);
      renderTable();
    }
    const f = state.frame;
    setText("hr-gap",  signals.gap[f]?.toFixed(3) ?? "—");
    setText("hr-baseline", signals.baseline.toFixed(3));
    setText("hr-noise", `±${signals.noiseFloor.toFixed(3)}`);

    // Active row.
    host.querySelectorAll("tr[data-punch-idx]").forEach(tr => {
      const idx = parseInt(tr.getAttribute("data-punch-idx"), 10);
      const p = signals.punches[idx];
      const active = p && f >= p.search_start && f <= p.search_end;
      tr.classList.toggle("active", !!active);
    });

    drawGapTrace(host.querySelector("#hr-gap-canvas"), signals, f, cfg);
  },

  draw(ctx, state) {
    const f = state.frame;
    const p = state.pose;
    const s = state.renderScale || 1;

    // Persistent hip line on every frame so you can SEE rotation as you scrub.
    drawHipLine(ctx, p, f, s);

    // Top-left labeler-style HUD: every punch whose [start, end] window
    // contains the current frame, stacked. Same helper Overview uses.
    const aps = activePunches(state, f);
    if (aps.length) drawPunchHudStack(ctx, aps, state, s);

    // At the exact LAND frame of any labeled punch we score, draw a verdict
    // label next to the boxer's head — separate from (and below) the HUD.
    for (const punch of signals.punches) {
      if (punch.land_frame === f) {
        drawLandLabel(ctx, p, f, punch, s);
      }
    }
  },
};

// ── DOM ────────────────────────────────────────────────────────────────────

function template() {
  return `
    <h2>Hip rotation · sanity-check on labels</h2>
    <p class="hint">Compares the proposed rule (gap-delta on 2D pose) against
      the labeler's <code>rule_hip_rotation</code> verdicts. Scrub through —
      the hip line is drawn at every frame, labeled punches get a
      <b>label=X · pred=Y</b> tag at their LAND frame so you can see
      where the algorithm agrees or disagrees with the coach.</p>

    <div id="hr-source-pill" class="hint" style="margin-bottom:8px"></div>

    <h3>Current frame</h3>
    <div class="metric-grid">
      <div class="metric"><div class="metric-label">gap (hips / torso)</div><div class="metric-val" id="hr-gap">—</div></div>
      <div class="metric"><div class="metric-label">baseline</div><div class="metric-val" id="hr-baseline">—</div></div>
      <div class="metric">
        <div class="metric-label">noise floor</div>
        <div class="metric-val" id="hr-noise">—</div>
        <div class="metric-sub muted">threshold should clear this</div>
      </div>
    </div>

    <h3>gap (full clip)</h3>
    <p class="hint">Translucent bands = labeled punches' search windows,
      colored by the coach's verdict (green=pass, red=fail). The inner
      tick within each band = our prediction at the current threshold.
      Click to seek.</p>
    <canvas id="hr-gap-canvas" width="320" height="160"></canvas>

    <h3>Per-punch verdicts</h3>
    <div id="hr-summary" class="hint"></div>
    <div id="hr-punch-table"></div>

    <h3>Thresholds</h3>
    <label class="slider">
      <span>min_delta = <output id="hr-min-out">${cfg.minDelta.toFixed(2)}</output> <span class="muted small">torso</span></span>
      <input type="range" id="hr-min" min="0.00" max="0.30" step="0.01" value="${cfg.minDelta}">
      <span class="muted small">Peak gap minus baseline below this = predict <i>fail</i>.</span>
    </label>
    <label class="slider">
      <span>smoothing = <output id="hr-smooth-out">${(cfg.gapSmoothSeconds*1000).toFixed(0)} ms</output></span>
      <input type="range" id="hr-smooth" min="0" max="0.250" step="0.01" value="${cfg.gapSmoothSeconds}">
      <span class="muted small">Moving-average on gap. Filters jitter.</span>
    </label>
    <label class="slider">
      <span>search_window = <output id="hr-search-out">${(cfg.searchWindowSec*1000).toFixed(0)} ms</output></span>
      <input type="range" id="hr-search" min="0.10" max="0.80" step="0.05" value="${cfg.searchWindowSec}">
      <span class="muted small">±this far around each punch window we hunt for the gap peak.</span>
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
    renderTable();
    seekHack(state.frame);
  });
}

function renderHeader(state) {
  const pill = host.querySelector("#hr-source-pill");
  if (state.labels?.detections?.length && hasVerdicts(state.labels.detections)) {
    const t = new Date(state.labels.fetched_at || Date.now()).toLocaleTimeString();
    const cached = state.labels.from_cache ? " (cached)" : "";
    pill.innerHTML =
      `<span class="role-lead">Ground truth labels</span> · live @ ${t}${cached} · ` +
      `${signals.punches.length} verdict-bearing labels`;
  } else if (state.labels?.detections?.length) {
    pill.innerHTML = `<span class="muted">Labels loaded but none carry a <code>rule_hip_rotation</code> verdict.</span>`;
  } else {
    pill.innerHTML = `<span class="muted">No labels loaded — this lens needs the live label fetch to be working.</span>`;
  }
}

function hasVerdicts(detections) {
  return detections.some(d => d.rule_hip_rotation === "pass" || d.rule_hip_rotation === "fail");
}

function renderTable() {
  if (!signals) return;
  const rows = signals.punches.map((p, i) => {
    const agreement = p.predicted === p.label;
    const aCls = agreement ? "scored" : "skipped";
    const aTxt = agreement ? "✓" : "✗";
    return `<tr class="${aCls}" data-seek="${p.land_frame ?? p.start_frame}" data-punch-idx="${i}">
      <td>${p.timestamp.toFixed(2)}s</td>
      <td>${(p.punch_type || "?").replace(/_/g, " ")}</td>
      <td><span class="role-${p.label === "fail" ? "rear" : "lead"}">${p.label}</span></td>
      <td><span class="role-${p.predicted === "fail" ? "rear" : "lead"}">${p.predicted}</span></td>
      <td>+${p.delta_search.toFixed(3)}</td>
      <td style="text-align:center;font-weight:bold;color:${agreement ? "#5fd97a" : "#e85a5a"}">${aTxt}</td>
    </tr>`;
  }).join("");
  const total = signals.punches.length;
  const agree = signals.punches.filter(p => p.predicted === p.label).length;
  const fpFn = signals.punches.reduce((acc, p) => {
    if (p.predicted !== p.label) {
      acc[p.predicted === "fail" ? "fp" : "fn"]++;
    }
    return acc;
  }, { fp: 0, fn: 0 });
  setHtml("hr-summary",
    `<b>${agree}/${total}</b> agreement (${total ? Math.round(100 * agree / total) : 0}%) · ` +
    `<span class="bad">${fpFn.fp}</span> false fails · ` +
    `<span class="bad">${fpFn.fn}</span> missed fails (false passes).`);
  setHtml("hr-punch-table", `
    <table class="sps-tbl">
      <thead><tr><th>t</th><th>Type</th><th>Label</th><th>Predicted</th><th>Δ gap</th><th>=?</th></tr></thead>
      <tbody>${rows || `<tr><td colspan="6" class="muted">no labeled in-scope punches in this round</td></tr>`}</tbody>
    </table>
  `);
  host.querySelectorAll("tr[data-seek]").forEach(tr => {
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => {
      seekHack(parseInt(tr.getAttribute("data-seek"), 10));
    });
  });
}

// ── Compute ────────────────────────────────────────────────────────────────

function computeAll(state, cfg) {
  const pose = state.pose;
  const fps = pose.fps || 30;
  const N = pose.n_frames;

  // gap[f] = |L_hip - R_hip| / torso_height, smoothed.
  const gapRaw = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const lx = pose.skeleton[(i * 17 + J.L_HIP) * 2];
    const ly = pose.skeleton[(i * 17 + J.L_HIP) * 2 + 1];
    const rx = pose.skeleton[(i * 17 + J.R_HIP) * 2];
    const ry = pose.skeleton[(i * 17 + J.R_HIP) * 2 + 1];
    const th = Math.max(1e-6, torsoHeight(pose, i));
    gapRaw[i] = Math.hypot(lx - rx, ly - ry) / th;
  }
  const smoothFrames = Math.max(1, Math.round(cfg.gapSmoothSeconds * fps));
  const gap = movingAvg(gapRaw, smoothFrames);
  const baseline = median(gap);
  const noiseFloor = mad(gap, baseline);

  // For LAND frame: pick the frame inside the punch window where the
  // *punching wrist* is most extended from its shoulder. We don't know
  // stance per-frame here so we just pick whichever wrist for the punch's
  // hand based on common orthodox convention (lead=L). Good enough for the
  // overlay's purposes — the visible label tells you what's what.
  const extL = wristExt(pose, J.L_WRIST, J.L_SHOULDER);
  const extR = wristExt(pose, J.R_WRIST, J.R_SHOULDER);

  // Filter detections to ones with a real rule_hip_rotation verdict +
  // in-scope punch types.
  const detections = (state.labels?.detections || []).filter(d => {
    const v = d.rule_hip_rotation;
    return (v === "pass" || v === "fail") && cfg.appliesTo.has(d.punch_type);
  });

  const searchFrames = Math.max(2, Math.round(cfg.searchWindowSec * fps));
  const punches = detections.map((d, idx) => {
    const sf = Math.max(0, d.start_frame);
    const ef = Math.min(N - 1, d.end_frame);
    const ss = Math.max(0, sf - searchFrames);
    const se = Math.min(N - 1, ef + searchFrames);
    let peak = gap[ss], peakAt = ss, trough = gap[ss], tv = 0;
    for (let f = ss; f <= se; f++) {
      const g = gap[f];
      if (g > peak)   { peak = g; peakAt = f; }
      if (g < trough)  trough = g;
      if (f > ss) tv += Math.abs(g - gap[f - 1]);
    }
    const delta = peak - baseline;
    const predicted = delta >= cfg.minDelta ? "pass" : "fail";

    // LAND = peak wrist extension in punch window. orthodox-lead → L, else R.
    const wristSide = (d.hand === "lead") ? "L" : "R";
    const ext = wristSide === "L" ? extL : extR;
    let landFrame = sf, bestE = -Infinity;
    for (let f = sf; f <= ef; f++) {
      if (ext[f] > bestE) { bestE = ext[f]; landFrame = f; }
    }

    return {
      idx,
      timestamp: d.timestamp,
      hand: d.hand,
      punch_type: d.punch_type,
      label: d.rule_hip_rotation,
      start_frame: sf,
      end_frame: ef,
      search_start: ss,
      search_end: se,
      land_frame: landFrame,
      peak_gap: peak,
      peak_frame: peakAt,
      trough_gap: trough,
      range_gap: peak - trough,
      total_var: tv,
      delta_search: delta,
      predicted,
    };
  });

  return { gap, baseline, noiseFloor, punches, fps };
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
    const th = Math.max(1e-6, torsoHeight(pose, i));
    e[i] = Math.hypot(wx - sx, wy - sy) / th;
  }
  return e;
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

function median(arr) {
  const s = Array.from(arr).sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
}

function mad(arr, center) {
  const devs = Array.from(arr, v => Math.abs(v - center)).sort((a, b) => a - b);
  if (!devs.length) return 0;
  const lo = Math.floor(devs.length * 0.2);
  const hi = Math.floor(devs.length * 0.8);
  const inner = devs.slice(lo, hi);
  return inner.length ? inner[Math.floor(inner.length / 2)] : 0;
}

// ── Drawing ────────────────────────────────────────────────────────────────

function drawHipLine(ctx, pose, frame, scale) {
  const lc = pose.conf[frame * 17 + J.L_HIP];
  const rc = pose.conf[frame * 17 + J.R_HIP];
  if (lc < 0.05 || rc < 0.05) return;
  const lx = pose.skeleton[(frame * 17 + J.L_HIP) * 2];
  const ly = pose.skeleton[(frame * 17 + J.L_HIP) * 2 + 1];
  const rx = pose.skeleton[(frame * 17 + J.R_HIP) * 2];
  const ry = pose.skeleton[(frame * 17 + J.R_HIP) * 2 + 1];
  ctx.save();
  ctx.strokeStyle = COLORS.hipLine;
  ctx.lineWidth = 3 * scale;
  ctx.globalAlpha = 0.85;
  ctx.beginPath();
  ctx.moveTo(lx, ly); ctx.lineTo(rx, ry);
  ctx.stroke();
  // Endpoint dots.
  ctx.fillStyle = COLORS.hipLine;
  ctx.beginPath(); ctx.arc(lx, ly, 5 * scale, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(rx, ry, 5 * scale, 0, Math.PI * 2); ctx.fill();
  // Midpoint label: current gap value.
  ctx.font = `bold ${Math.round(11 * scale)}px ui-monospace, monospace`;
  const mx = (lx + rx) / 2, my = (ly + ry) / 2 - 8 * scale;
  const txt = `gap=${signals?.gap?.[frame]?.toFixed(3) ?? "?"}`;
  const tw = ctx.measureText(txt).width;
  ctx.fillStyle = "rgba(0,0,0,0.7)";
  ctx.fillRect(mx - tw / 2 - 4, my - 14 * scale, tw + 8, 16 * scale);
  ctx.fillStyle = COLORS.hipLine;
  ctx.fillText(txt, mx - tw / 2, my - 2 * scale);
  ctx.restore();
}

function drawLandLabel(ctx, pose, frame, punch, scale) {
  // Position the label near the boxer's head — find the highest confident
  // joint to avoid drawing it off-screen on a partial detection.
  const i = pose.skeleton.length / 17 / 2;
  const nose = {
    x: pose.skeleton[(frame * 17 + J.NOSE) * 2],
    y: pose.skeleton[(frame * 17 + J.NOSE) * 2 + 1],
    c: pose.conf[frame * 17 + J.NOSE],
  };
  if (nose.c < 0.05) return;
  const x = nose.x;
  const y = Math.max(20 * scale, nose.y - 40 * scale);

  const agree = punch.predicted === punch.label;
  const titleColor = agree ? COLORS.agree : COLORS.disagree;
  const title = `label=${punch.label} · pred=${punch.predicted} ${agree ? "✓" : "✗"}`;
  const sub   = `Δ=${punch.delta_search >= 0 ? "+" : ""}${punch.delta_search.toFixed(3)}`;

  ctx.save();
  const titleSize = Math.round(13 * scale);
  const subSize = Math.round(10 * scale);
  ctx.font = `bold ${titleSize}px ui-monospace, monospace`;
  const tw = ctx.measureText(title).width;
  ctx.font = `${subSize}px ui-monospace, monospace`;
  const sw = ctx.measureText(sub).width;
  const padX = 6 * scale, padY = 5 * scale;
  const w = Math.max(tw, sw) + padX * 2;
  const h = titleSize + subSize + padY * 2 + 4 * scale;
  ctx.fillStyle = "rgba(0,0,0,0.82)";
  if (typeof ctx.roundRect === "function") {
    ctx.beginPath(); ctx.roundRect(x - w / 2, y - h, w, h, 6 * scale); ctx.fill();
  } else {
    ctx.fillRect(x - w / 2, y - h, w, h);
  }
  ctx.fillStyle = titleColor;
  ctx.fillRect(x - w / 2, y - h, 4 * scale, h);
  ctx.fillStyle = "#fff";
  ctx.font = `bold ${titleSize}px ui-monospace, monospace`;
  ctx.textBaseline = "top";
  ctx.fillText(title, x - w / 2 + padX, y - h + padY);
  ctx.fillStyle = "#aaa";
  ctx.font = `${subSize}px ui-monospace, monospace`;
  ctx.fillText(sub, x - w / 2 + padX, y - h + padY + titleSize + 2 * scale);
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
  let yMax = signals.baseline + Math.max(0.10, cfg.minDelta * 2);
  for (let f = 0; f < N; f += stride) {
    if (signals.gap[f] < yMin) yMin = signals.gap[f];
    if (signals.gap[f] > yMax) yMax = signals.gap[f];
  }
  yMin = Math.max(0, yMin - 0.02);
  yMax = yMax + 0.02;
  const ymap = v => H - ((v - yMin) / (yMax - yMin)) * (H - 8) - 4;
  const xmap = f => (f / Math.max(1, N - 1)) * (W - 2) + 1;

  // Label-colored bands.
  for (const p of signals.punches) {
    const x1 = xmap(p.search_start);
    const x2 = Math.max(x1 + 2, xmap(p.search_end));
    ctx.fillStyle = p.label === "fail" ? "rgba(232,90,90,0.18)" : "rgba(95,217,122,0.14)";
    ctx.fillRect(x1, 0, x2 - x1, H);
  }

  // Noise band.
  if (signals.noiseFloor > 0) {
    const yn1 = ymap(signals.baseline + signals.noiseFloor);
    const yn2 = ymap(signals.baseline - signals.noiseFloor);
    ctx.fillStyle = "rgba(255,255,255,0.07)";
    ctx.fillRect(0, yn1, W, yn2 - yn1);
  }

  // Baseline + threshold lines.
  ctx.strokeStyle = COLORS.baseline;
  ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
  const yb = ymap(signals.baseline);
  ctx.beginPath(); ctx.moveTo(0, yb); ctx.lineTo(W, yb); ctx.stroke();
  const yt = ymap(signals.baseline + cfg.minDelta);
  ctx.beginPath(); ctx.moveTo(0, yt); ctx.lineTo(W, yt); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(255,255,255,0.55)";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(`baseline ${signals.baseline.toFixed(3)} (±${signals.noiseFloor.toFixed(3)} noise)`, 4, yb - 2);
  ctx.fillText(`+ min_delta ${cfg.minDelta.toFixed(2)}`, 4, yt - 2);

  // Gap line.
  ctx.strokeStyle = COLORS.hipLine;
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  for (let f = 0; f < N; f += stride) {
    const px = xmap(f), py = ymap(signals.gap[f]);
    if (f === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // Per-punch peak marker, color = our prediction.
  for (const p of signals.punches) {
    const x = xmap(p.peak_frame);
    ctx.strokeStyle = p.predicted === "fail" ? COLORS.labelFail : COLORS.labelPass;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    // Agreement indicator: outline circle on top.
    const agree = p.predicted === p.label;
    ctx.strokeStyle = agree ? COLORS.agree : COLORS.disagree;
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.beginPath(); ctx.arc(x, 8, 4, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  }

  // Current frame line.
  ctx.strokeStyle = COLORS.current;
  ctx.lineWidth = 1;
  const x = xmap(frame);
  ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
}

// ── Tiny utilities ─────────────────────────────────────────────────────────

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
