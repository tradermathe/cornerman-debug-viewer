// Step-and-punch sync debug panel.
//
// Mental model: a PUNCH is the unit. For each punch we figure out:
//   - LAND   = the frame within the punch window where the punching wrist is
//              most extended from its shoulder (gated on wrist confidence).
//   - PLANT  = the nearest foot-plant frame within a search window around the
//              punch — picked from ankle-velocity step events.
//   - GAP    = land_frame - plant_frame, in ms. Sign tells you who was late.
//
// PUNCH SOURCES
// We accept punches from two places, picked at mount time:
//   * state.punches.detections (ST-GCN export from dump_punches.py) — preferred
//     when available; mirrors what the live app sees.
//   * Wrist-extension peaks above a threshold — fallback when no punches file
//     is loaded. Works OK for stepping straights, less well for hooks/uppercuts.
//
// OVERLAY
// The overlay is punch-window-centric: when the cursor sits anywhere from a
// bit before the punch start until a bit after the paired plant, we paint
// "punch context":
//   * Punching arm: bright shoulder→wrist line.
//   * Other arm + non-stepping ankle: dimmed.
//   * Wrist: ring + persistent label like "LAND in +33 ms" or "LAND −83 ms ago".
//   * Stepping ankle (if paired): ring + "PLANT in +50 ms" / "PLANT now".
//   * Top-left banner: punch type, hand, sync verdict, gap_ms.
//
// Outside any punch window the overlay falls back to small LEAD/rear ankle
// dots and the body skeleton, so you can still see stance.
//
// SIDE PANEL
// * Source pill: which punch source is loaded (and how many).
// * Stance selector — determines lead foot + maps hand → anatomical side.
// * Live current-frame metrics.
// * Ankle velocity sparkline (full clip) with thresholds + plant ticks.
// * Wrist→shoulder sparkline (full clip).
// * Punches table — one row per punch, scored / unscored / skipped, click to
//   seek to the LAND frame.
// * Step events table — all detected steps with click-to-seek.
// * Threshold sliders.

import { J, torsoHeight } from "../skeleton.js";
import { fetchLiveLabels } from "../sheet-labels.js";

const DEFAULTS = {
  stance: "orthodox",          // orthodox = lead is L; southpaw = lead is R
  highVelThreshold: 0.020,
  lowVelThreshold: 0.005,
  velSmoothSeconds: 0.083,
  searchWindowSec: 0.35,
  syncToleranceMs: 100,
  minWristConfidence: 0.30,    // average wrist conf gate for LAND detection
  minWristExtension: 0.5,      // only used in heuristic (no-punches-file) mode
  windowMarginMs: 80,          // extra context around the punch window in the overlay
};

const COLORS = {
  lead:    "#5fd97a",
  rear:    "#f5b945",
  arm:     "#ff8a5c",          // active punching arm
  armDim:  "rgba(255,138,92,0.18)",
  bodyDim: "rgba(255,255,255,0.18)",
  current: "rgba(255,255,255,0.85)",
  oos:     "#e85a5a",
  good:    "#5fd97a",
  warn:    "#f5b945",
};

// Anatomical resolution for the punching arm. Mirrors PUNCH_JOINTS in
// cornerman_rules/rules/step_punch_sync.py.
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
        J.L_ANKLE, J.R_ANKLE,
        J.L_WRIST, J.R_WRIST,
        J.L_SHOULDER, J.R_SHOULDER,
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

    // "Refresh from Sheet" — bypasses the in-session cache so newly added
    // rows in the labeler show up immediately.
    const refreshBtn = host.querySelector("#sps-refresh");
    if (refreshBtn) {
      refreshBtn.addEventListener("click", async () => {
        const basename = state.labels?.source_video
          ? null  // we have a confirmed match — reuse the cache_basename path below
          : null;
        // We always re-derive the basename from the pose source filename, so
        // refresh works even when the initial auto-match failed.
        const cacheBasename = state.pose?.source
          ? state.pose.source.replace(/\.npy$/i, "").replace(/_(yolo|vision)_r\d+$/i, "")
          : null;
        if (!cacheBasename) {
          refreshBtn.textContent = "no cache loaded";
          return;
        }
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

    wireSlider(state, "#sps-high",   "highVelThreshold",  v => v.toFixed(3));
    wireSlider(state, "#sps-low",    "lowVelThreshold",   v => v.toFixed(3));
    wireSlider(state, "#sps-sync",   "syncToleranceMs",   v => `${v.toFixed(0)} ms`);
    wireSlider(state, "#sps-search", "searchWindowSec",   v => `${(v*1000).toFixed(0)} ms`);
    wireSlider(state, "#sps-margin", "windowMarginMs",    v => `${v.toFixed(0)} ms`);
    wireSlider(state, "#sps-extmin", "minWristExtension", v => v.toFixed(2));

    renderTables(state);
  },

  update(state) {
    if (state.pose !== lastPose) {
      signals = computeAll(state, cfg);
      lastPose = state.pose;
      renderTables(state);
    }

    const f = state.frame;
    const li = leadAnkleIdx(cfg.stance);
    const vLead = li === J.L_ANKLE ? signals.velL[f] : signals.velR[f];
    const vRear = li === J.L_ANKLE ? signals.velR[f] : signals.velL[f];

    setText("sps-vel-lead", vLead.toFixed(4), velColor(vLead, cfg));
    setText("sps-vel-rear", vRear.toFixed(4), velColor(vRear, cfg));
    setText("sps-ext-l",    signals.extL[f].toFixed(2));
    setText("sps-ext-r",    signals.extR[f].toFixed(2));

    const ap = activePunchAt(signals, f, cfg);
    setText("sps-active", describeActive(ap, f, signals.fps, cfg));

    // Highlight the active row in the punches table.
    host.querySelectorAll("tr[data-punch-idx]").forEach(tr => {
      const idx = parseInt(tr.getAttribute("data-punch-idx"), 10);
      tr.classList.toggle("active", ap && ap.idx === idx);
    });

    drawVelTrace(host.querySelector("#sps-vel-canvas"), signals, f, cfg);
    drawExtTrace(host.querySelector("#sps-ext-canvas"), signals, f, cfg);
  },

  draw(ctx, state) {
    const f = state.frame;
    const p = state.pose;
    const s = state.renderScale || 1;
    const ap = activePunchAt(signals, f, cfg);

    const li = leadAnkleIdx(cfg.stance);
    const ri = rearAnkleIdx(cfg.stance);

    // When we're inside a punch, the lens takes over: only the punching arm
    // is bright; the stepping ankle has a persistent ring + offset label.
    if (ap) {
      drawSeg(ctx, p, f,
        ap.side === "L" ? J.L_SHOULDER : J.R_SHOULDER,
        ap.side === "L" ? J.L_WRIST    : J.R_WRIST,
        COLORS.arm, 3, 0.9, s);
      // Dim the other arm.
      drawSeg(ctx, p, f,
        ap.side === "L" ? J.R_SHOULDER : J.L_SHOULDER,
        ap.side === "L" ? J.R_WRIST    : J.L_WRIST,
        COLORS.armDim, 1.5, 1, s);

      // Persistent LAND marker on the punching wrist.
      const wristJ = ap.side === "L" ? J.L_WRIST : J.R_WRIST;
      if (ap.has_land) {
        const offset = offsetText(ap.land_frame, f, signals.fps);
        emphasizeJoint(ctx, p, f, wristJ, COLORS.arm, `LAND ${offset}`, 14, s);
      } else {
        emphasizeJoint(ctx, p, f, wristJ, COLORS.warn, `LAND ?`, 12, s);
      }

      // PLANT marker on the stepping ankle, color-coded by lead/rear.
      if (ap.paired_step) {
        const ankleJ = ap.paired_step.ankle;
        const role = (ankleJ === li) ? "lead" : "rear";
        const color = role === "lead" ? COLORS.lead : COLORS.rear;
        const offset = offsetText(ap.paired_step.plant, f, signals.fps);
        emphasizeJoint(ctx, p, f, ankleJ, color,
          `PLANT (${role}) ${offset}`, 16, s);
      } else {
        // No paired step — still show the lead ankle small so the user can
        // see "no step happened around this punch".
        smallDot(ctx, p, f, li, COLORS.lead, "lead", s);
        smallDot(ctx, p, f, ri, COLORS.rear, "rear", s);
      }

      drawBanner(ctx, bannerFor(ap, f, signals.fps, cfg), s);
    } else {
      // Idle mode: small LEAD/rear dots so stance is still visible, and faint
      // both-arm segments so the body has some color.
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
    <p class="hint">Coach cue: "step and punch land together". For each punch
      we mark <b>LAND</b> (wrist most extended) and the nearest foot-plant
      <b>PLANT</b>. <b>Only lead-foot steps</b> count toward the sync verdict.</p>

    <div id="sps-source-pill" class="hint" style="margin-bottom:8px"></div>
    <button type="button" id="sps-refresh" class="muted small" style="margin-bottom:8px">Refresh from Sheet</button>

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
    <p class="hint" style="margin-top:4px"><span id="sps-active" class="muted">—</span></p>

    <h3>Ankle velocity (full clip)</h3>
    <p class="hint">Green = lead ankle, amber = rear. Dashed lines = step
      thresholds. Vertical ticks = detected step plants. Bars at top = punches.</p>
    <canvas id="sps-vel-canvas" width="320" height="120"></canvas>

    <h3>Wrist → shoulder extension</h3>
    <p class="hint">For each detected punch, we pick the frame inside the
      window where the punching wrist is furthest from its shoulder = LAND.</p>
    <canvas id="sps-ext-canvas" width="320" height="80"></canvas>

    <h3>Punches</h3>
    <div id="sps-punch-table"></div>

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
      <span>overlay_margin = <output id="sps-margin-out">${cfg.windowMarginMs.toFixed(0)} ms</output></span>
      <input type="range" id="sps-margin" min="0" max="400" step="10" value="${cfg.windowMarginMs}">
    </label>
    <label class="slider" id="sps-extmin-wrap" style="display:none">
      <span>min_wrist_extension (fallback only) = <output id="sps-extmin-out">${cfg.minWristExtension.toFixed(2)}</output></span>
      <input type="range" id="sps-extmin" min="0.20" max="1.20" step="0.05" value="${cfg.minWristExtension}">
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

  // Source pill: tell the user where punches came from.
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
      `<span class="role-rear">ST-GCN punches</span> · ${signals.punches.length} detected · ` +
      `from <code>${state.punches?.source || "punches.json"}</code>` + errLine;
  } else {
    const errLine = labelErr
      ? `<br><span class="muted">Labels: <span class="bad">${labelErr}</span>.</span>`
      : "";
    pill.innerHTML =
      `<span class="role-rear">Heuristic punches</span> · ${signals.punches.length} wrist-peak candidates · ` +
      `drop a <code>*_punches.json</code> next to the cache for ST-GCN.` + errLine;
  }
  host.querySelector("#sps-extmin-wrap").style.display =
    signals.source === "heuristic" ? "" : "none";

  const li = leadAnkleIdx(cfg.stance);
  const leadLabel = li === J.L_ANKLE ? "L" : "R";

  // Punches table.
  const punchRows = signals.punches.map(p => {
    const role = p.paired_step ? (p.paired_step.ankle === li ? "lead" : "rear") : null;
    const cls = !p.has_land ? "skipped"
              : !p.paired_step ? "unscored"
              : !p.is_scored ? "unscored"
              : "scored";
    let gapHtml = "—";
    if (p.gap_ms != null) {
      const gapCls = p.is_scored
        ? (Math.abs(p.gap_ms) <= cfg.syncToleranceMs ? "good" : "bad")
        : "muted";
      const sign = p.gap_ms >= 0 ? "+" : "";
      gapHtml = `<span class="${gapCls}">${sign}${p.gap_ms.toFixed(0)} ms</span>`;
    } else if (!p.has_land) {
      gapHtml = `<span class="muted">no LAND</span>`;
    } else {
      gapHtml = `<span class="muted">no step</span>`;
    }
    const seekTo = p.land_frame ?? p.start_frame ?? 0;
    const typeShort = (p.punch_type || "?").replace(/_/g, " ");
    const pairedBadge = role
      ? `<span class="role-${role}">${role}</span>`
      : `<span class="muted">—</span>`;
    return `
      <tr class="${cls}" data-seek="${seekTo}" data-punch-idx="${p.idx}">
        <td>${p.timestamp.toFixed(2)}s</td>
        <td>${typeShort}</td>
        <td>${p.hand}·${p.side}</td>
        <td>${pairedBadge}</td>
        <td>${gapHtml}</td>
      </tr>`;
  }).join("");

  const scored = signals.punches.filter(p => p.is_scored).length;
  const oos = signals.punches.filter(p => p.is_out_of_sync).length;

  host.querySelector("#sps-punch-table").innerHTML = `
    <p class="hint" style="margin:0 0 6px"><b>${oos}/${scored}</b> lead-foot
      stepping punches out of sync · sync tol ±${cfg.syncToleranceMs.toFixed(0)} ms · lead ${leadLabel}</p>
    <table class="sps-tbl">
      <thead><tr><th>t</th><th>Type</th><th>Hand</th><th>Step</th><th>Gap</th></tr></thead>
      <tbody>${punchRows || `<tr><td colspan="5" class="muted">no punches</td></tr>`}</tbody>
    </table>
  `;

  // Step events table.
  const stepRows = signals.steps.map(ev => {
    const role = ev.ankle === li ? "lead" : "rear";
    return `<tr data-seek="${ev.plant}">
      <td>${(ev.plant / signals.fps).toFixed(2)}s</td>
      <td><span class="role-${role}">${ev.ankle === J.L_ANKLE ? "L" : "R"}·${role}</span></td>
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

  host.querySelectorAll("tr[data-seek]").forEach(tr => {
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => {
      const f = parseInt(tr.getAttribute("data-seek"), 10);
      seekHackSimple(f);
    });
  });
}

// ── Compute ────────────────────────────────────────────────────────────────

function computeAll(state, cfg) {
  const pose = state.pose;
  const fps = pose.fps || 30;
  const N = pose.n_frames;

  const velL = ankleVel(pose, J.L_ANKLE, cfg.velSmoothSeconds, fps);
  const velR = ankleVel(pose, J.R_ANKLE, cfg.velSmoothSeconds, fps);
  const extL = wristExt(pose, J.L_WRIST, J.L_SHOULDER);
  const extR = wristExt(pose, J.R_WRIST, J.R_SHOULDER);

  const stepsL = findSteps(velL, cfg.highVelThreshold, cfg.lowVelThreshold, J.L_ANKLE);
  const stepsR = findSteps(velR, cfg.highVelThreshold, cfg.lowVelThreshold, J.R_ANKLE);
  const steps = [...stepsL, ...stepsR].sort((a, b) => a.plant - b.plant);

  const searchFrames = Math.max(2, Math.round(cfg.searchWindowSec * fps));
  const leadIdx = leadAnkleIdx(cfg.stance);

  // Punch source priority:
  //   1. state.labels  — ground truth from the labeler Sheet (preferred)
  //   2. state.punches — ST-GCN export
  //   3. wrist-extension peaks — heuristic fallback
  let source = "heuristic";
  let sourceMeta = null;
  let punches = [];
  let detections = null;
  if (state.labels && Array.isArray(state.labels.detections) &&
      state.labels.detections.length > 0) {
    source = "labels";
    sourceMeta = state.labels;
    detections = state.labels.detections;
  } else if (state.punches && Array.isArray(state.punches.detections) &&
             state.punches.detections.length > 0) {
    source = "stgcn";
    sourceMeta = state.punches;
    detections = state.punches.detections;
  }

  if (detections) {
    punches = detections.map((d, idx) => {
      const side = punchSide(d.hand, cfg.stance);
      const wristJ = side === "L" ? J.L_WRIST : J.R_WRIST;
      const shoulderJ = side === "L" ? J.L_SHOULDER : J.R_SHOULDER;
      const ext = side === "L" ? extL : extR;
      const sf = Math.max(0, d.start_frame);
      const ef = Math.min(N, d.end_frame + 1);  // make end exclusive for slicing math
      const land = findLandFrame(pose, ext, sf, ef, wristJ, cfg.minWristConfidence);
      return assemblePunch({
        idx, timestamp: d.timestamp, hand: d.hand, side,
        punch_type: d.punch_type, category: d.category,
        start_frame: sf, end_frame: ef - 1,
        land_frame: land,
        land_ext: land !== null ? ext[land] : null,
        steps, searchFrames, leadIdx, fps,
      }, cfg);
    });
  } else {
    // Source 2: heuristic — local wrist-extension peaks above threshold.
    const peaksL = findPeaks(extL, cfg.minWristExtension, "L");
    const peaksR = findPeaks(extR, cfg.minWristExtension, "R");
    const allPeaks = [...peaksL, ...peaksR].sort((a, b) => a.frame - b.frame);
    punches = allPeaks.map((peak, idx) => {
      const ext = peak.side === "L" ? extL : extR;
      // Heuristic punch "window" = a small region around the peak so the
      // overlay still has a window to live in.
      const sf = Math.max(0, peak.frame - Math.round(0.15 * fps));
      const ef = Math.min(N - 1, peak.frame + Math.round(0.15 * fps));
      return assemblePunch({
        idx,
        timestamp: peak.frame / fps,
        hand: "?",  // unknown — heuristic doesn't know which is lead/rear
        side: peak.side,
        punch_type: "wrist_peak",
        category: null,
        start_frame: sf,
        end_frame: ef,
        land_frame: peak.frame,
        land_ext: peak.value,
        steps, searchFrames, leadIdx, fps,
      }, cfg);
    });
  }

  return { velL, velR, extL, extR, steps, punches, fps, source };
}

function assemblePunch(args, cfg) {
  const { steps, searchFrames, leadIdx, fps } = args;
  // Pair each punch with its nearest step plant whose plant falls within
  // ±searchFrames of LAND. Lead foot wins ties.
  let pairedStep = null;
  if (args.land_frame !== null) {
    let best = null;
    let bestDist = Infinity;
    for (const ev of steps) {
      const d = Math.abs(ev.plant - args.land_frame);
      if (d > searchFrames) continue;
      const tie = d === bestDist;
      if (d < bestDist || (tie && ev.ankle === leadIdx)) {
        best = ev; bestDist = d;
      }
    }
    pairedStep = best;
  }
  const gap_frames = (pairedStep && args.land_frame !== null)
    ? (args.land_frame - pairedStep.plant)
    : null;
  const gap_ms = gap_frames !== null ? gap_frames * 1000 / fps : null;
  const has_land = args.land_frame !== null;
  const is_lead_step = pairedStep != null && pairedStep.ankle === leadIdx;
  const is_scored = has_land && is_lead_step;
  const is_out_of_sync = is_scored && gap_ms !== null
    && Math.abs(gap_ms) > cfg.syncToleranceMs;
  return {
    idx: args.idx,
    timestamp: args.timestamp,
    hand: args.hand,
    side: args.side,
    punch_type: args.punch_type,
    category: args.category,
    start_frame: args.start_frame,
    end_frame: args.end_frame,
    land_frame: args.land_frame,
    land_ext: args.land_ext,
    has_land,
    paired_step: pairedStep,
    gap_frames,
    gap_ms,
    is_lead_step,
    is_scored,
    is_out_of_sync,
  };
}

function findLandFrame(pose, ext, sf, ef, wristIdx, minConf) {
  if (ef - sf < 1) return null;
  let confSum = 0;
  for (let f = sf; f < ef; f++) confSum += pose.conf[f * 17 + wristIdx];
  const avgConf = confSum / Math.max(1, ef - sf);
  if (avgConf < minConf) return null;
  let bestF = sf, bestV = -Infinity;
  for (let f = sf; f < ef; f++) {
    if (ext[f] > bestV) { bestV = ext[f]; bestF = f; }
  }
  return bestF;
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

function findPeaks(arr, minVal, side) {
  const peaks = [];
  const N = arr.length;
  for (let i = 2; i < N - 2; i++) {
    if (arr[i] < minVal) continue;
    if (arr[i] >= arr[i-1] && arr[i] >= arr[i-2] &&
        arr[i] >= arr[i+1] && arr[i] >= arr[i+2]) {
      if (peaks.length && i - peaks[peaks.length - 1].frame < 8) continue;
      peaks.push({ frame: i, value: arr[i], side });
    }
  }
  return peaks;
}

// ── Active-punch resolution ────────────────────────────────────────────────

function activePunchAt(signals, frame, cfg) {
  if (!signals?.punches?.length) return null;
  const margin = Math.round(cfg.windowMarginMs * signals.fps / 1000);
  let best = null;
  let bestDist = Infinity;
  for (const p of signals.punches) {
    const plant = p.paired_step?.plant ?? null;
    const lo = Math.min(p.start_frame, p.land_frame ?? p.start_frame, plant ?? p.start_frame) - margin;
    const hi = Math.max(p.end_frame,   p.land_frame ?? p.end_frame,   plant ?? p.end_frame) + margin;
    if (frame < lo || frame > hi) continue;
    // Multiple punches can overlap — pick the one whose LAND is closest.
    const center = p.land_frame ?? Math.round((p.start_frame + p.end_frame) / 2);
    const d = Math.abs(frame - center);
    if (d < bestDist) { best = p; bestDist = d; }
  }
  return best;
}

// ── Per-frame text helpers ─────────────────────────────────────────────────

function describeActive(ap, f, fps, cfg) {
  if (!ap) return "no active punch";
  const land = ap.has_land
    ? `LAND ${offsetText(ap.land_frame, f, fps)}`
    : "LAND ?";
  const plant = ap.paired_step
    ? `PLANT ${offsetText(ap.paired_step.plant, f, fps)} (${ap.paired_step.ankle === leadAnkleIdx(cfg.stance) ? "lead" : "rear"})`
    : "no paired step";
  const gap = ap.gap_ms != null
    ? `gap ${ap.gap_ms >= 0 ? "+" : ""}${ap.gap_ms.toFixed(0)} ms`
    : "gap —";
  return `<b>${(ap.punch_type || "?").replace(/_/g, " ")}</b> · ${ap.hand}·${ap.side} · ${land} · ${plant} · ${gap}`;
}

function bannerFor(ap, f, fps, cfg) {
  if (!ap) return null;
  const gap = ap.gap_ms != null
    ? `${ap.gap_ms >= 0 ? "+" : ""}${ap.gap_ms.toFixed(0)} ms`
    : "—";
  const verdict = !ap.has_land ? "no LAND"
    : !ap.paired_step ? "no step paired"
    : !ap.is_scored ? "rear-foot step (not scored)"
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
  ctx.font = `bold ${fontPx}px ui-monospace, monospace`;
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
  ctx.font = `bold ${fontPx}px ui-monospace, monospace`;
  const pad = 6 * scale;
  const tw = ctx.measureText(banner.text).width;
  ctx.fillStyle = "rgba(0,0,0,0.7)";
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
  const ymap = v => H - (v / yMax) * (H - 14) - 2;
  const xmap = f => (f / (N - 1)) * (W - 2) + 1;

  // Punch bars across the top (6 px).
  for (const p of signals.punches) {
    const x1 = xmap(p.start_frame), x2 = Math.max(x1 + 2, xmap(p.end_frame));
    ctx.fillStyle = !p.is_scored
      ? "rgba(245,185,69,0.5)"
      : (p.is_out_of_sync ? "rgba(232,90,90,0.8)" : "rgba(95,217,122,0.8)");
    ctx.fillRect(x1, 0, x2 - x1, 4);
  }

  // Threshold lines
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
  let yh = ymap(cfg.highVelThreshold), yl = ymap(cfg.lowVelThreshold);
  ctx.beginPath(); ctx.moveTo(0, yh); ctx.lineTo(W, yh); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, yl); ctx.lineTo(W, yl); ctx.stroke();
  ctx.setLineDash([]);

  // Velocity lines (lead/rear color)
  const leadIsL = cfg.stance === "orthodox";
  const colorL = leadIsL ? COLORS.lead : COLORS.rear;
  const colorR = leadIsL ? COLORS.rear : COLORS.lead;
  drawLine(ctx, signals.velL, stride, xmap, ymap, colorL);
  drawLine(ctx, signals.velR, stride, xmap, ymap, colorR);

  // Plant ticks at bottom
  for (const ev of signals.steps) {
    const x = xmap(ev.plant);
    const color = (ev.ankle === J.L_ANKLE) ? colorL : colorR;
    ctx.strokeStyle = color; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, H - 4); ctx.lineTo(x, H); ctx.stroke();
  }

  // Current frame line
  ctx.strokeStyle = COLORS.current;
  ctx.lineWidth = 1;
  const x = xmap(frame);
  ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
}

function drawExtTrace(canvas, signals, frame, cfg) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const N = signals.extL.length;
  const stride = Math.max(1, Math.floor(N / W));
  let yMax = 0.5;
  for (let f = 0; f < N; f += stride) {
    if (signals.extL[f] > yMax) yMax = signals.extL[f];
    if (signals.extR[f] > yMax) yMax = signals.extR[f];
  }
  const ymap = v => H - (v / yMax) * (H - 4) - 2;
  const xmap = f => (f / (N - 1)) * (W - 2) + 1;

  drawLine(ctx, signals.extL, stride, xmap, ymap, "#ff8a5c");
  drawLine(ctx, signals.extR, stride, xmap, ymap, "#ffd95c");

  // LAND ticks for each detected punch.
  for (const p of signals.punches) {
    if (!p.has_land) continue;
    const x = xmap(p.land_frame);
    ctx.strokeStyle = p.side === "L" ? "#ff8a5c" : "#ffd95c";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 4); ctx.stroke();
  }

  // Current frame
  ctx.strokeStyle = COLORS.current;
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

function velColor(v, cfg) {
  if (v > cfg.highVelThreshold) return "#7ec8ff";
  if (v < cfg.lowVelThreshold) return "#5fd97a";
  return "#f5b945";
}

function seekHack(state, f) {
  const ev = new Event("input");
  const slider = document.getElementById("scrubber");
  if (!slider) return;
  slider.value = f;
  slider.dispatchEvent(ev);
}

function seekHackSimple(f) {
  const ev = new Event("input");
  const slider = document.getElementById("scrubber");
  if (!slider) return;
  slider.value = f;
  slider.dispatchEvent(ev);
}
