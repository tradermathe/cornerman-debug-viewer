// Step-and-punch sync debug panel — v2 (per-punch, gap-based).
//
// Mental model: each punch is the unit. For every detected punch we ask
// "did a lead-foot step happen *in this punch's window*, and where?" The
// detector runs locally inside each punch's search window — there is no
// global "step events" list anymore. This removes the chicken-and-egg
// pairing problem the v1 algorithm had.
//
// Step detection per punch
//   gap[f]  = euclidean(L_ankle[f], R_ankle[f]) / torso_height[f]    (torso-normalized
//             inter-ankle distance, smoothed by a short moving average to
//             kill skeleton jitter and natural body sway)
//   baseline= median(gap) across the whole clip                      (rough
//             estimate of the boxer's idle stance width — the noise floor)
//   per punch: search = [punch.start - 0.4s, punch.end + 0.4s]
//              plant_frame = argmax(gap[f])    for f in search
//              step_found  = gap[plant_frame] - baseline >= min_step_extension
//
// One threshold owns the whole question: `min_step_extension`. If the gap
// peak inside the search window exceeds the baseline by more than that,
// we say "a step happened on this punch" and the plant frame is exactly
// the peak. Otherwise the punch is "no step".
//
// Punch source priority (unchanged):
//   1. state.labels   — ground truth from the labeler Sheet
//   2. state.punches  — ST-GCN export
//   3. (no fallback)  — we no longer hallucinate punches; without a real
//                       source we can't do per-punch step checks.

import { J, torsoHeight } from "../skeleton.js";
import { fetchLiveLabels } from "../sheet-labels.js";

const DEFAULTS = {
  stance: "orthodox",
  gapSmoothSeconds: 0.083,      // moving-average noise filter
  searchWindowSec:  0.4,        // ± around each punch window
  minStepExtension: 0.10,       // gap must exceed baseline by this many torso-heights
  syncToleranceMs:  100,        // |gap_ms| above this counts as out of sync
  minWristConfidence: 0.30,
  windowMarginMs:    80,        // overlay activates this far either side of the search window
};

const COLORS = {
  lead:    "#5fd97a",
  rear:    "#f5b945",
  arm:     "#ff8a5c",
  armDim:  "rgba(255,138,92,0.18)",
  bodyDim: "rgba(255,255,255,0.18)",
  current: "rgba(255,255,255,0.85)",
  oos:     "#e85a5a",
  good:    "#5fd97a",
  warn:    "#f5b945",
  gap:     "#a78bfa",            // purple for the gap track
  baseline:"rgba(255,255,255,0.45)",
};

function punchSide(hand, stance) {
  if (hand === "lead")  return stance === "orthodox" ? "L" : "R";
  if (hand === "rear")  return stance === "orthodox" ? "R" : "L";
  return "?";
}
function leadAnkleIdx(stance) { return stance === "orthodox" ? J.L_ANKLE : J.R_ANKLE; }
function rearAnkleIdx(stance) { return stance === "orthodox" ? J.R_ANKLE : J.L_ANKLE; }

let host;
let cfg = { ...DEFAULTS };
let signals = null;
let lastPose = null;

export const StepPunchSyncRule = {
  id: "step_punch_sync",
  label: "Step + punch sync",

  skeletonStyle() {
    return {
      boneColor: "rgba(255,255,255,0.18)",
      boneWidth: 1.5,
      jointRadius: 3,
      highlightJoints: new Set([
        J.L_ANKLE, J.R_ANKLE, J.L_WRIST, J.R_WRIST, J.L_SHOULDER, J.R_SHOULDER,
      ]),
    };
  },

  mount(_host, state) {
    host = _host;
    host.innerHTML = template();
    signals = computeAll(state, cfg);
    lastPose = state.pose;

    host.querySelector("#sps-stance").value = cfg.stance;
    host.querySelector("#sps-stance").addEventListener("change", e => {
      cfg.stance = e.target.value;
      signals = computeAll(state, cfg);
      renderTables(state);
      seekHack(state, state.frame);
    });

    const refreshBtn = host.querySelector("#sps-refresh");
    if (refreshBtn) {
      refreshBtn.addEventListener("click", async () => {
        const cacheBasename = state.pose?.source
          ? state.pose.source.replace(/\.npy$/i, "").replace(/_(yolo|vision)_r\d+$/i, "")
          : null;
        if (!cacheBasename) { refreshBtn.textContent = "no cache"; return; }
        refreshBtn.disabled = true;
        const prev = refreshBtn.textContent;
        refreshBtn.textContent = "fetching…";
        const live = await fetchLiveLabels({
          cacheBasename,
          cacheStartSec: state.pose.start_sec || 0,
          fps: state.pose.fps,
          nFrames: state.pose.n_frames,
          force: true,
        });
        refreshBtn.disabled = false;
        if (live.error) {
          state.labels = { error: live.error, cacheBasename, detections: [] };
          signals = computeAll(state, cfg);
          renderTables(state);
          seekHack(state, state.frame);
          refreshBtn.textContent = `failed: ${live.error}`;
          setTimeout(() => { refreshBtn.textContent = prev; }, 3000);
          return;
        }
        state.labels = live;
        signals = computeAll(state, cfg);
        renderTables(state);
        seekHack(state, state.frame);
        refreshBtn.textContent = `↻ ${live.detections.length} labels`;
        setTimeout(() => { refreshBtn.textContent = prev; }, 2500);
      });
    }

    wireSlider(state, "#sps-ext",    "minStepExtension",  v => v.toFixed(2));
    wireSlider(state, "#sps-sync",   "syncToleranceMs",   v => `${v.toFixed(0)} ms`);
    wireSlider(state, "#sps-search", "searchWindowSec",   v => `${(v*1000).toFixed(0)} ms`);
    wireSlider(state, "#sps-smooth", "gapSmoothSeconds",  v => `${(v*1000).toFixed(0)} ms`);

    renderTables(state);
  },

  update(state) {
    if (state.pose !== lastPose) {
      signals = computeAll(state, cfg);
      lastPose = state.pose;
      renderTables(state);
    }

    const f = state.frame;
    setText("sps-gap",  signals.gap[f]?.toFixed(3) ?? "—");
    setText("sps-baseline", signals.baseline.toFixed(3));
    setText("sps-ext-above", (signals.gap[f] - signals.baseline).toFixed(3));
    setText("sps-noise", `±${signals.noiseFloor.toFixed(3)}`);

    const ap = activePunchAt(signals, f, cfg);
    setText("sps-active", describeActive(ap, f, signals.fps, cfg));

    host.querySelectorAll("tr[data-punch-idx]").forEach(tr => {
      const idx = parseInt(tr.getAttribute("data-punch-idx"), 10);
      tr.classList.toggle("active", ap && ap.idx === idx);
    });

    drawGapTrace(host.querySelector("#sps-gap-canvas"), signals, f, cfg);
  },

  draw(ctx, state) {
    const f = state.frame;
    const p = state.pose;
    const s = state.renderScale || 1;
    const ap = activePunchAt(signals, f, cfg);

    const li = leadAnkleIdx(cfg.stance);
    const ri = rearAnkleIdx(cfg.stance);

    // Persistent markers: whenever the current frame IS the LAND or PLANT
    // frame of any punch, draw the matching ring + label — independent of
    // whether we're inside a search window. This way scrubbing through the
    // round always shows "yes, the algorithm called the plant here."
    for (const p2 of signals.punches) {
      if (p2.has_land && p2.land_frame === f) {
        const wristJ = p2.side === "L" ? J.L_WRIST : J.R_WRIST;
        emphasizeJoint(ctx, p, f, wristJ, COLORS.arm, "LAND", 14, s);
      }
      if (p2.step_found && p2.plant_frame === f) {
        const color = p2.is_out_of_sync ? COLORS.oos : COLORS.lead;
        emphasizeJoint(ctx, p, f, li, color, "PLANT", 16, s);
      }
    }

    if (ap) {
      drawSeg(ctx, p, f,
        ap.side === "L" ? J.L_SHOULDER : J.R_SHOULDER,
        ap.side === "L" ? J.L_WRIST    : J.R_WRIST,
        COLORS.arm, 3, 0.9, s);
      drawSeg(ctx, p, f,
        ap.side === "L" ? J.R_SHOULDER : J.L_SHOULDER,
        ap.side === "L" ? J.R_WRIST    : J.L_WRIST,
        COLORS.armDim, 1.5, 1, s);

      const wristJ = ap.side === "L" ? J.L_WRIST : J.R_WRIST;
      if (ap.has_land) {
        const offset = offsetText(ap.land_frame, f, signals.fps);
        emphasizeJoint(ctx, p, f, wristJ, COLORS.arm, `LAND ${offset}`, 14, s);
      } else {
        emphasizeJoint(ctx, p, f, wristJ, COLORS.warn, `LAND ?`, 12, s);
      }

      // PLANT marker — always on the lead ankle (current scope: lead foot only).
      if (ap.step_found) {
        const ankleJ = li;
        const offset = offsetText(ap.plant_frame, f, signals.fps);
        emphasizeJoint(ctx, p, f, ankleJ, COLORS.lead,
          `PLANT ${offset}`, 16, s);
      } else {
        // No step on this punch — draw dimmed ankles to make that explicit.
        smallDot(ctx, p, f, li, COLORS.lead, "lead", s);
        smallDot(ctx, p, f, ri, COLORS.rear, "rear", s);
      }

      drawBanner(ctx, bannerFor(ap, f, signals.fps, cfg), s);
    } else {
      smallDot(ctx, p, f, li, COLORS.lead, "lead", s);
      smallDot(ctx, p, f, ri, COLORS.rear, "rear", s);
      drawSeg(ctx, p, f, J.L_SHOULDER, J.L_WRIST, COLORS.bodyDim, 1.2, 1, s);
      drawSeg(ctx, p, f, J.R_SHOULDER, J.R_WRIST, COLORS.bodyDim, 1.2, 1, s);
    }
  },
};

// ── DOM ────────────────────────────────────────────────────────────────────

function template() {
  return `
    <h2>Step + punch sync</h2>
    <p class="hint">For each punch we look in a window around it and ask:
      did the inter-ankle <code>gap</code> peak high enough above the
      boxer's idle stance to count as a step? If yes, the peak frame is
      the <b>plant</b>; the gap to the punch's <b>LAND</b> tells us how
      tightly synced they are.</p>

    <div id="sps-source-pill" class="hint" style="margin-bottom:8px"></div>
    <button type="button" id="sps-refresh" class="muted small" style="margin-bottom:8px">Refresh from Sheet</button>

    <h3>Stance</h3>
    <select id="sps-stance">
      <option value="orthodox">Orthodox (lead = L)</option>
      <option value="southpaw">Southpaw (lead = R)</option>
    </select>

    <h3>Current frame</h3>
    <div class="metric-grid">
      <div class="metric"><div class="metric-label">gap</div><div class="metric-val" id="sps-gap">—</div></div>
      <div class="metric"><div class="metric-label">baseline</div><div class="metric-val" id="sps-baseline">—</div></div>
      <div class="metric"><div class="metric-label">ext above baseline</div><div class="metric-val" id="sps-ext-above">—</div></div>
      <div class="metric">
        <div class="metric-label">noise floor</div>
        <div class="metric-val" id="sps-noise">—</div>
        <div class="metric-sub muted">idle wobble · threshold should clear this</div>
      </div>
    </div>
    <p class="hint" style="margin-top:4px"><span id="sps-active" class="muted">—</span></p>

    <h3>gap (full clip)</h3>
    <p class="hint">Purple = inter-ankle distance / torso. Dashed = baseline +
      <code>min_step_extension</code> (the bar a punch's peak gap has to
      clear). Translucent bands = punch search windows; ticks = detected
      plant frames. Click to seek.</p>
    <canvas id="sps-gap-canvas" width="320" height="140"></canvas>

    <h3>Punches</h3>
    <div id="sps-punch-table"></div>

    <h3>Thresholds</h3>
    <label class="slider">
      <span>min_step_extension = <output id="sps-ext-out">${cfg.minStepExtension.toFixed(2)}</output> <span class="muted small">torso</span></span>
      <input type="range" id="sps-ext" min="0.02" max="0.40" step="0.01" value="${cfg.minStepExtension}">
      <span class="muted small">How far the gap must rise above the boxer's baseline stance width for it to count as a step. Higher = stricter.</span>
    </label>
    <label class="slider">
      <span>sync_tolerance = <output id="sps-sync-out">${cfg.syncToleranceMs.toFixed(0)} ms</output></span>
      <input type="range" id="sps-sync" min="20" max="300" step="10" value="${cfg.syncToleranceMs}">
      <span class="muted small">|LAND − plant| above this is "out of sync".</span>
    </label>
    <label class="slider">
      <span>search_window = <output id="sps-search-out">${(cfg.searchWindowSec*1000).toFixed(0)} ms</output></span>
      <input type="range" id="sps-search" min="0.10" max="0.80" step="0.05" value="${cfg.searchWindowSec}">
      <span class="muted small">±this far around each punch window we hunt for the gap peak.</span>
    </label>
    <label class="slider">
      <span>smoothing = <output id="sps-smooth-out">${(cfg.gapSmoothSeconds*1000).toFixed(0)} ms</output></span>
      <input type="range" id="sps-smooth" min="0" max="0.250" step="0.01" value="${cfg.gapSmoothSeconds}">
      <span class="muted small">Moving-average window on gap. Filters tracker jitter + natural sway.</span>
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
    renderTables(state);
    seekHack(state, state.frame);
  });
}

function renderTables(state) {
  if (!signals) return;
  const pill = host.querySelector("#sps-source-pill");
  const labelErr = state.labels?.error;
  if (signals.source === "labels") {
    const time = new Date(state.labels.fetched_at).toLocaleTimeString();
    const cached = state.labels.from_cache ? " (cached)" : "";
    const conf = state.labels.match_confidence || "?";
    pill.innerHTML =
      `<span class="role-lead">Ground truth</span> · ${signals.punches.length} labels · ` +
      `live @ ${time}${cached} · auto-matched (${conf}) → ` +
      `<code>${state.labels.source_video}</code>`;
  } else if (signals.source === "stgcn") {
    const errLine = labelErr
      ? `<br><span class="muted">Labels: <span class="bad">${labelErr}</span> — using ST-GCN.</span>`
      : "";
    pill.innerHTML =
      `<span class="role-rear">ST-GCN punches</span> · ${signals.punches.length} detected` + errLine;
  } else {
    const errLine = labelErr
      ? `<br><span class="muted">Labels: <span class="bad">${labelErr}</span>.</span>`
      : "";
    pill.innerHTML =
      `<span class="muted">No punches loaded — load GT labels or an ST-GCN punches.json to detect per-punch steps.</span>` + errLine;
  }

  const li = leadAnkleIdx(cfg.stance);
  const leadLabel = li === J.L_ANKLE ? "L" : "R";

  const punchRows = signals.punches.map(p => {
    let cls = !p.has_land ? "skipped"
            : !p.step_found ? "unscored"
            : "scored";
    let stepHtml;
    let gapHtml;
    if (!p.step_found) {
      stepHtml = `<span class="muted">no step</span>`;
      gapHtml = `<span class="muted">— (peak +${p.gap_above_baseline.toFixed(2)})</span>`;
    } else {
      stepHtml = `<span class="role-lead">step</span> · +${p.gap_above_baseline.toFixed(2)}`;
      if (p.gap_ms != null) {
        const gapCls = p.is_scored
          ? (Math.abs(p.gap_ms) <= cfg.syncToleranceMs ? "good" : "bad")
          : "muted";
        const sign = p.gap_ms >= 0 ? "+" : "";
        gapHtml = `<span class="${gapCls}">${sign}${p.gap_ms.toFixed(0)} ms</span>`;
      } else {
        gapHtml = `<span class="muted">no LAND</span>`;
      }
    }
    const seekTo = p.land_frame ?? p.plant_frame ?? p.start_frame ?? 0;
    const typeShort = (p.punch_type || "?").replace(/_/g, " ");
    return `
      <tr class="${cls}" data-seek="${seekTo}" data-punch-idx="${p.idx}">
        <td>${p.timestamp.toFixed(2)}s</td>
        <td>${typeShort}</td>
        <td>${p.hand}·${p.side}</td>
        <td>${stepHtml}</td>
        <td>${gapHtml}</td>
      </tr>`;
  }).join("");

  const scored = signals.punches.filter(p => p.is_scored).length;
  const oos = signals.punches.filter(p => p.is_out_of_sync).length;
  const nostep = signals.punches.filter(p => !p.step_found).length;

  host.querySelector("#sps-punch-table").innerHTML = `
    <p class="hint" style="margin:0 0 6px"><b>${oos}/${scored}</b>
      stepping punches out of sync · ${nostep} no step ·
      sync tol ±${cfg.syncToleranceMs.toFixed(0)} ms · lead ${leadLabel}</p>
    <table class="sps-tbl">
      <thead><tr><th>t</th><th>Type</th><th>Hand</th><th>Step</th><th>Gap</th></tr></thead>
      <tbody>${punchRows || `<tr><td colspan="5" class="muted">no punches</td></tr>`}</tbody>
    </table>
  `;

  host.querySelectorAll("tr[data-seek]").forEach(tr => {
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => {
      seekHackSimple(parseInt(tr.getAttribute("data-seek"), 10));
    });
  });
}

// ── Compute ────────────────────────────────────────────────────────────────

function computeAll(state, cfg) {
  const pose = state.pose;
  const fps = pose.fps || 30;
  const N = pose.n_frames;

  // Inter-ankle gap, torso-normalized, smoothed.
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

  // Baseline = median over the whole clip. Robust to brief excursions
  // (steps) — gives us the boxer's idle stance width.
  const baseline = median(gap);
  // Empirical noise floor: MAD of |gap − baseline| on the inner 60 % of
  // values, so step excursions don't inflate it. This is the threshold
  // bar that min_step_extension needs to clear.
  const noiseFloor = mad(gap, baseline);

  // Wrist-extension tracks (for LAND).
  const extL = wristExt(pose, J.L_WRIST, J.L_SHOULDER);
  const extR = wristExt(pose, J.R_WRIST, J.R_SHOULDER);

  // Source: labels > punches > none. No more heuristic wrist-peak fallback.
  let source = "none";
  let detections = null;
  if (state.labels?.detections?.length) {
    source = "labels";
    detections = state.labels.detections;
  } else if (state.punches?.detections?.length) {
    source = "stgcn";
    detections = state.punches.detections;
  }

  const searchFrames = Math.max(2, Math.round(cfg.searchWindowSec * fps));
  const li = leadAnkleIdx(cfg.stance);
  const punches = (detections || []).map((d, idx) =>
    analyzePunch(d, idx, gap, baseline, extL, extR, searchFrames, N, fps, cfg)
  );

  return {
    gap, baseline, noiseFloor, extL, extR, punches, source,
    fps, smoothFrames, searchFrames,
  };
}

// MAD of |arr - center| on the inner 60% (excludes step excursions).
function mad(arr, center) {
  const devs = Array.from(arr, v => Math.abs(v - center));
  devs.sort((a, b) => a - b);
  const lo = Math.floor(devs.length * 0.2);
  const hi = Math.floor(devs.length * 0.8);
  const inner = devs.slice(lo, hi);
  if (!inner.length) return 0;
  return inner[Math.floor(inner.length / 2)];
}

function analyzePunch(d, idx, gap, baseline, extL, extR, searchFrames, N, fps, cfg) {
  const side = punchSide(d.hand, cfg.stance);
  const wristJ = side === "L" ? J.L_WRIST : J.R_WRIST;
  const ext = side === "L" ? extL : extR;
  const sf = Math.max(0, d.start_frame);
  const ef = Math.min(N - 1, d.end_frame);
  const land = findLandFrame(ext, /* pose */ null, sf, ef, cfg.minWristConfidence);

  // Per-punch search window.
  const searchStart = Math.max(1, sf - searchFrames);
  const searchEnd   = Math.min(N - 1, ef + searchFrames);

  // Gap peak inside the search window.
  let peakF = searchStart;
  let peakG = gap[searchStart] || 0;
  for (let f = searchStart; f <= searchEnd; f++) {
    if (gap[f] > peakG) { peakG = gap[f]; peakF = f; }
  }
  const gapAboveBaseline = peakG - baseline;
  const stepFound = gapAboveBaseline >= cfg.minStepExtension;

  const gap_ms = (stepFound && land != null)
    ? (land - peakF) * 1000 / fps
    : null;
  const isOut = stepFound && land != null && Math.abs(gap_ms) > cfg.syncToleranceMs;
  const isScored = stepFound && land != null;

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
    land_frame: land,
    has_land: land != null,
    plant_frame: stepFound ? peakF : null,
    peak_gap: peakG,
    gap_above_baseline: gapAboveBaseline,
    step_found: stepFound,
    gap_frames: gap_ms != null ? (land - peakF) : null,
    gap_ms,
    is_out_of_sync: !!isOut,
    is_scored: !!isScored,
  };
}

function findLandFrame(ext, _pose, sf, ef, _minConf) {
  // Peak wrist→shoulder extension inside the punch window. Confidence
  // gating is omitted here since the punch source (labels or ST-GCN)
  // already curates which windows are valid punches.
  if (ef - sf < 1) return null;
  let bestF = sf, bestV = -Infinity;
  for (let f = sf; f <= ef; f++) {
    if (ext[f] > bestV) { bestV = ext[f]; bestF = f; }
  }
  return bestF;
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

// ── Active-punch resolution ────────────────────────────────────────────────

function activePunchAt(signals, frame, cfg) {
  if (!signals?.punches?.length) return null;
  const margin = Math.round(cfg.windowMarginMs * signals.fps / 1000);
  let best = null;
  let bestDist = Infinity;
  for (const p of signals.punches) {
    const plant = p.plant_frame ?? null;
    const lo = Math.min(p.search_start, p.land_frame ?? p.start_frame, plant ?? p.start_frame) - margin;
    const hi = Math.max(p.search_end,   p.land_frame ?? p.end_frame,   plant ?? p.end_frame) + margin;
    if (frame < lo || frame > hi) continue;
    const center = p.land_frame ?? Math.round((p.start_frame + p.end_frame) / 2);
    const d = Math.abs(frame - center);
    if (d < bestDist) { best = p; bestDist = d; }
  }
  return best;
}

function describeActive(ap, f, fps, cfg) {
  if (!ap) return "no active punch";
  const land = ap.has_land ? `LAND ${offsetText(ap.land_frame, f, fps)}` : "LAND ?";
  const plant = ap.step_found
    ? `PLANT ${offsetText(ap.plant_frame, f, fps)} · gap +${ap.gap_above_baseline.toFixed(2)}`
    : `no step (peak gap +${ap.gap_above_baseline.toFixed(2)} < ${cfg.minStepExtension.toFixed(2)})`;
  const gap = ap.gap_ms != null ? `gap ${ap.gap_ms >= 0 ? "+" : ""}${ap.gap_ms.toFixed(0)} ms` : "gap —";
  return `<b>${(ap.punch_type || "?").replace(/_/g, " ")}</b> · ${ap.hand}·${ap.side} · ${land} · ${plant} · ${gap}`;
}

function bannerFor(ap, _f, _fps, cfg) {
  if (!ap) return null;
  const gap = ap.gap_ms != null
    ? `${ap.gap_ms >= 0 ? "+" : ""}${ap.gap_ms.toFixed(0)} ms`
    : "—";
  const verdict = !ap.has_land ? "no LAND"
    : !ap.step_found ? `no step (peak +${ap.gap_above_baseline.toFixed(2)})`
    : (ap.is_out_of_sync ? "OUT OF SYNC" : "IN SYNC");
  const color = !ap.is_scored ? COLORS.warn
    : (ap.is_out_of_sync ? COLORS.oos : COLORS.good);
  const text = `${(ap.punch_type || "?").replace(/_/g, " ")} · ${ap.hand}·${ap.side} · gap ${gap} · ${verdict}`;
  return { text, color };
}

function offsetText(target, current, fps) {
  if (target == null) return "?";
  const delta = (target - current) * 1000 / fps;
  if (Math.abs(delta) < 1000 / fps / 2) return "now";
  if (delta > 0) return `in +${delta.toFixed(0)} ms`;
  return `${(-delta).toFixed(0)} ms ago`;
}

// ── Drawing ────────────────────────────────────────────────────────────────

function drawSeg(ctx, pose, frame, a, b, color, width, alpha, scale) {
  const ac = pose.conf[frame * 17 + a];
  const bc = pose.conf[frame * 17 + b];
  if (ac < 0.05 || bc < 0.05) return;
  const ax = pose.skeleton[(frame * 17 + a) * 2];
  const ay = pose.skeleton[(frame * 17 + a) * 2 + 1];
  const bx = pose.skeleton[(frame * 17 + b) * 2];
  const by = pose.skeleton[(frame * 17 + b) * 2 + 1];
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width * scale;
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(bx, by);
  ctx.stroke();
  ctx.restore();
}

function smallDot(ctx, pose, frame, jointIdx, color, label, scale) {
  const c = pose.conf[frame * 17 + jointIdx];
  if (c < 0.05) return;
  const x = pose.skeleton[(frame * 17 + jointIdx) * 2];
  const y = pose.skeleton[(frame * 17 + jointIdx) * 2 + 1];
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5 * scale;
  ctx.beginPath();
  ctx.arc(x, y, 7 * scale, 0, Math.PI * 2);
  ctx.stroke();
  const fontPx = Math.round(10 * scale);
  ctx.font = `${fontPx}px ui-monospace, monospace`;
  ctx.fillStyle = "rgba(0,0,0,0.6)";
  ctx.fillText(label, x + 10 * scale + 1, y + 4 * scale + 1);
  ctx.fillStyle = color;
  ctx.fillText(label, x + 10 * scale, y + 4 * scale);
  ctx.restore();
}

function emphasizeJoint(ctx, pose, frame, jointIdx, color, label, radius, scale) {
  const c = pose.conf[frame * 17 + jointIdx];
  if (c < 0.05) return;
  const x = pose.skeleton[(frame * 17 + jointIdx) * 2];
  const y = pose.skeleton[(frame * 17 + jointIdx) * 2 + 1];
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3 * scale;
  ctx.beginPath();
  ctx.arc(x, y, radius * scale, 0, Math.PI * 2);
  ctx.stroke();
  const fontPx = Math.round(12 * scale);
  ctx.font = `bold ${fontPx}px ui-monospace, "SF Mono", monospace`;
  const pad = 4 * scale;
  const tw = ctx.measureText(label).width;
  const bx = x + (radius + 4) * scale;
  const by = y - 4 * scale;
  ctx.fillStyle = "rgba(0,0,0,0.75)";
  ctx.fillRect(bx - 2, by - fontPx, tw + pad * 2, fontPx + 4);
  ctx.fillStyle = color;
  ctx.fillText(label, bx + pad - 2, by);
  ctx.restore();
}

function drawBanner(ctx, banner, scale) {
  if (!banner) return;
  ctx.save();
  const fontPx = Math.round(13 * scale);
  ctx.font = `bold ${fontPx}px ui-monospace, "SF Mono", monospace`;
  const pad = 6 * scale;
  const tw = ctx.measureText(banner.text).width;
  ctx.fillStyle = "rgba(0,0,0,0.7)";
  ctx.fillRect(8 * scale, 8 * scale, tw + pad * 2, fontPx + pad * 2);
  ctx.fillStyle = banner.color;
  ctx.fillText(banner.text, 8 * scale + pad, 8 * scale + pad + fontPx - 2);
  ctx.restore();
}

function drawGapTrace(canvas, signals, frame, cfg) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const N = signals.gap.length;
  const stride = Math.max(1, Math.floor(N / W));

  // Y range covers baseline ± a generous margin so we see steps.
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

  // Punch search windows as translucent bands.
  for (const p of signals.punches) {
    const x1 = xmap(p.search_start);
    const x2 = Math.max(x1 + 2, xmap(p.search_end));
    ctx.fillStyle = !p.is_scored
      ? "rgba(245,185,69,0.10)"
      : (p.is_out_of_sync ? "rgba(232,90,90,0.20)" : "rgba(95,217,122,0.18)");
    ctx.fillRect(x1, 0, x2 - x1, H);
  }

  // Noise-floor band around baseline.
  if (signals.noiseFloor > 0) {
    const yn1 = ymap(signals.baseline + signals.noiseFloor);
    const yn2 = ymap(signals.baseline - signals.noiseFloor);
    ctx.fillStyle = "rgba(255,255,255,0.07)";
    ctx.fillRect(0, yn1, W, yn2 - yn1);
  }

  // Baseline + min_step_extension threshold lines.
  ctx.strokeStyle = COLORS.baseline;
  ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
  const yb = ymap(signals.baseline);
  ctx.beginPath(); ctx.moveTo(0, yb); ctx.lineTo(W, yb); ctx.stroke();
  const yt = ymap(signals.baseline + cfg.minStepExtension);
  ctx.beginPath(); ctx.moveTo(0, yt); ctx.lineTo(W, yt); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(255,255,255,0.55)";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(`baseline ${signals.baseline.toFixed(3)} (±${signals.noiseFloor.toFixed(3)} noise)`, 4, yb - 2);
  ctx.fillText(`+ min_step ${cfg.minStepExtension.toFixed(2)}`, 4, yt - 2);

  drawLine(ctx, signals.gap, stride, xmap, ymap, COLORS.gap);

  // Plant + LAND markers per punch.
  for (const p of signals.punches) {
    if (p.step_found) {
      const x = xmap(p.plant_frame);
      ctx.strokeStyle = p.is_out_of_sync ? COLORS.oos : COLORS.good;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    if (p.has_land) {
      const x = xmap(p.land_frame);
      ctx.strokeStyle = COLORS.arm;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // Current frame line.
  ctx.strokeStyle = COLORS.current;
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
