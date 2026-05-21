// Punch direction review — walk through every labelled punch, see all four
// arrows side by side, decide whether the model is right.
//
// Per-frame raw arrow (amber)      — current frame's ankle line (the input)
// Median arrow      (orange/gold)  — median (dx, dy) across this punch window,
//                                    same number the analysis uses to fit
// Predicted facing  (cyan)         — sign · median_arrow + per-stance offset
//                                    (the model's prediction for this punch)
// GT label          (red dashed)   — the labelled direction from the
//                                    "Punch Directions" sheet
//
// The lens loops the current punch window the same way the orientation
// labeler does (auto-play, snap back when crossing end_sec). Next/Prev
// buttons (and N/P keys) step through labelled punches only — unlabelled
// punches in the same video are skipped.

import { J } from "../skeleton.js";
import { STANCE_FITS } from "./orientation_lens.js";
import { fetchPunchDirectionsAll } from "../sheet-labels.js";

const MIN_ANKLE_CONF = 0.30;
const COLOR_RAW    = "#ffd24a";  // amber
const COLOR_MEDIAN = "#ff9a32";  // bright orange
const COLOR_PRED   = "#3ad9e0";  // cyan
const COLOR_GT     = "#ff5d6c";  // red

let host = null;
let loopWindow = null;        // {start_frame, end_frame, uuid} or null
let activeIdx = -1;           // index into labelledPunches
let labelledPunches = [];     // [{detection, gtLabel, gtLabeler, medianArrow}]
let byUuidMap = null;         // Map<punch_uuid, {label, labeler, ts}> — all videos
let fetchInFlight = false;    // dedupe concurrent fetches
let lastDetectionsRef = null; // identity of the detections array we last joined
let lastStemForReset = null;  // detect video change to reset activeIdx
let fetchError = null;
let fetchInfo = "";
let videoEl = null;
let timeupdateHandler = null;
let keydownHandler = null;

function wrap180(deg) { return ((deg + 180) % 360 + 360) % 360 - 180; }

// ─── geometry helpers ─────────────────────────────────────────────────────

// Loose: no confidence gate, just requires the ankle xy to be finite
// numbers. Used by the visual draw — the live amber arrow should not
// flicker off whenever the pose detector loses brief confidence mid-punch
// (which happens during load-up / follow-through routinely). Returns
// minConf so the caller can style low-confidence frames differently.
function ankleVecLoose(pose, f, stance) {
  const lx = pose.skeleton[(f * 17 + J.L_ANKLE) * 2];
  const ly = pose.skeleton[(f * 17 + J.L_ANKLE) * 2 + 1];
  const rx = pose.skeleton[(f * 17 + J.R_ANKLE) * 2];
  const ry = pose.skeleton[(f * 17 + J.R_ANKLE) * 2 + 1];
  if (![lx, ly, rx, ry].every(Number.isFinite)) return null;
  // Arrow convention (2026-05-21 v2): orthodox arrow points R→L (toward
  // front foot); southpaw L→R.
  const orthodox = stance !== "southpaw";
  const dx = orthodox ? (lx - rx) : (rx - lx);
  const dy = orthodox ? (ly - ry) : (ry - ly);
  if (dx * dx + dy * dy < 1e-6) return null;
  const cL = pose.conf[f * 17 + J.L_ANKLE];
  const cR = pose.conf[f * 17 + J.R_ANKLE];
  return { lx, ly, rx, ry, dx, dy, minConf: Math.min(cL, cR) };
}

// Strict: gated by MIN_ANKLE_CONF. Used by computeMedianArrow — the
// median fit must stay on clean frames or it gets noisier.
function ankleVecAt(pose, f, stance) {
  const v = ankleVecLoose(pose, f, stance);
  if (!v || v.minConf < MIN_ANKLE_CONF) return null;
  return v;
}

function hipMidAt(pose, f) {
  const cLH = pose.conf[f * 17 + J.L_HIP];
  const cRH = pose.conf[f * 17 + J.R_HIP];
  if (cLH < 0.2 || cRH < 0.2) return null;
  const lhx = pose.skeleton[(f * 17 + J.L_HIP) * 2];
  const lhy = pose.skeleton[(f * 17 + J.L_HIP) * 2 + 1];
  const rhx = pose.skeleton[(f * 17 + J.R_HIP) * 2];
  const rhy = pose.skeleton[(f * 17 + J.R_HIP) * 2 + 1];
  if (![lhx, lhy, rhx, rhy].every(Number.isFinite)) return null;
  return { x: 0.5 * (lhx + rhx), y: 0.5 * (lhy + rhy) };
}

// Median across the punch window (same algorithm 07_punch_directions uses).
function computeMedianArrow(pose, det) {
  const dxs = [], dys = [];
  for (let f = det.start_frame; f <= det.end_frame; f++) {
    const v = ankleVecAt(pose, f, det.stance);
    if (!v) continue;
    dxs.push(v.dx); dys.push(v.dy);
  }
  if (dxs.length < 3) return null;
  dxs.sort((a, b) => a - b);
  dys.sort((a, b) => a - b);
  const mid = Math.floor(dxs.length / 2);
  const mdx = dxs.length % 2 ? dxs[mid] : 0.5 * (dxs[mid - 1] + dxs[mid]);
  const mdy = dys.length % 2 ? dys[mid] : 0.5 * (dys[mid - 1] + dys[mid]);
  if (mdx * mdx + mdy * mdy < 1e-6) return null;
  return { dx: mdx, dy: mdy, angle_deg: Math.atan2(mdy, mdx) * 180 / Math.PI,
           n_valid: dxs.length };
}

// ─── draw helpers ─────────────────────────────────────────────────────────

function drawArrow(ctx, x0, y0, angle_rad, length, color, { dashed = false, lineWidth = 3, headSize = 12 } = {}) {
  const x1 = x0 + length * Math.cos(angle_rad);
  const y1 = y0 + length * Math.sin(angle_rad);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lineWidth;
  if (dashed) ctx.setLineDash([6, 4]);
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  ctx.setLineDash([]);
  // arrowhead
  const a = 0.45;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - headSize * Math.cos(angle_rad - a), y1 - headSize * Math.sin(angle_rad - a));
  ctx.lineTo(x1 - headSize * Math.cos(angle_rad + a), y1 - headSize * Math.sin(angle_rad + a));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

// ─── data loading: punch direction labels (cached across videos) ─────────
//
// Punch Directions is fetched once per session (the sheet is all videos in
// one call). state.labels.detections, however, arrives asynchronously from
// tryLiveLabels — and might not be populated when the user first picks this
// lens. So we (a) ensure the fetch is in flight, (b) re-derive the joined
// list whenever the detections array reference changes, (c) reset cursor
// when the video stem changes.

async function ensureFetched(state) {
  if (byUuidMap || fetchInFlight) return;
  fetchInFlight = true;
  fetchError = null;
  fetchInfo = "fetching punch direction labels…";
  rebuildSidebar(state);
  try {
    const res = await fetchPunchDirectionsAll({});
    if (res.error) { fetchError = res.error; }
    else            { byUuidMap = res.byUuid; }
  } catch (e) {
    fetchError = e.message;
  } finally {
    fetchInFlight = false;
    rebuildLabelledPunches(state);
  }
}

function rebuildLabelledPunches(state) {
  if (!byUuidMap) { rebuildSidebar(state); return; }

  const stem = state.cacheBasename || "";
  const dets = state.labels?.detections;
  const stemChanged = stem !== lastStemForReset;
  // Skip the heavy re-join if neither the detections array nor the stem
  // changed — update() runs every frame; we don't want to recompute medians
  // on each tick.
  if (!stemChanged && dets === lastDetectionsRef && labelledPunches.length) {
    rebuildSidebar(state);
    return;
  }
  lastDetectionsRef = dets;
  lastStemForReset = stem;

  // Include every punch from Combined Data — labelled OR not. Direction
  // labels (GT) get attached when present so the red dashed arrow can
  // draw, but their absence doesn't block review. The amber raw / orange
  // median / cyan predicted arrows always draw — that's the whole point
  // when you're scrubbing through a video to sanity-check the model on
  // unlabelled footage.
  const next = [];
  let nWithGt = 0;
  for (const det of dets || []) {
    if (!det.punch_uuid) continue;
    const gt = byUuidMap.get(det.punch_uuid);
    const median = computeMedianArrow(state.pose, det);
    next.push({
      detection: det,
      gtLabel:   gt ? gt.label   : null,
      gtLabeler: gt ? gt.labeler : null,
      hasGt:     !!gt,
      median,
    });
    if (gt) nWithGt++;
  }
  next.sort((a, b) => a.detection.start_frame - b.detection.start_frame);

  const totalDets = dets?.length || 0;
  fetchInfo = `${next.length} punches · ${nWithGt} with GT direction label`
    + (totalDets > next.length
        ? ` (${totalDets - next.length} skipped — no punch_uuid)` : "");

  const hadNone = labelledPunches.length === 0;
  labelledPunches = next;

  // If the video changed OR we just got our first labelled punches, land on
  // the first one. Otherwise preserve the user's current cursor.
  if (stemChanged || (hadNone && next.length)) {
    if (next.length) seekToPunch(0, state);
    else             { loopWindow = null; activeIdx = -1; }
  } else if (activeIdx >= next.length) {
    activeIdx = next.length - 1;
  }
  rebuildSidebar(state);
}

// ─── punch navigation ─────────────────────────────────────────────────────

function seekToPunch(idx, state) {
  if (!labelledPunches.length) return;
  if (idx < 0) idx = 0;
  if (idx >= labelledPunches.length) idx = labelledPunches.length - 1;
  const det = labelledPunches[idx].detection;
  activeIdx = idx;
  loopWindow = { start_frame: det.start_frame, end_frame: det.end_frame,
                 uuid: det.punch_uuid };
  if (videoEl && state.fps) {
    videoEl.currentTime = (state.start_sec || 0) + det.start_frame / state.fps;
    if (videoEl.paused) {
      const p = videoEl.play();
      if (p && typeof p.catch === "function") p.catch(() => {});
    }
  }
  rebuildSidebar(state);
}

// Loop within the current punch window — snap back when frame goes past end.
// The handler guards on state.rule?.id so it goes dormant when the user picks
// a different lens (the viewer doesn't call unmount() so listeners would
// otherwise keep firing). Switching back to this lens re-activates it.
function installTimeupdateLoop(state) {
  if (!videoEl) return;
  if (timeupdateHandler) videoEl.removeEventListener("timeupdate", timeupdateHandler);
  timeupdateHandler = () => {
    if (state.rule?.id !== "punch_direction_review") return;
    if (!loopWindow || !state.fps) return;
    const endTime = (state.start_sec || 0) + (loopWindow.end_frame + 0.5) / state.fps;
    if (videoEl.currentTime > endTime) {
      const startTime = (state.start_sec || 0) + loopWindow.start_frame / state.fps;
      videoEl.currentTime = startTime;
    }
  };
  videoEl.addEventListener("timeupdate", timeupdateHandler);
}

function installKeyHandlers(state) {
  if (keydownHandler) document.removeEventListener("keydown", keydownHandler);
  keydownHandler = (e) => {
    if (state.rule?.id !== "punch_direction_review") return;
    const tag = e.target?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    if (e.key === "n" || e.key === "N") {
      e.preventDefault();
      seekToPunch(activeIdx + 1, state);
    } else if (e.key === "p" || e.key === "P") {
      e.preventDefault();
      seekToPunch(activeIdx - 1, state);
    }
  };
  document.addEventListener("keydown", keydownHandler);
}

// ─── sidebar ──────────────────────────────────────────────────────────────
// Two-pass design: buildSidebarSkeleton() runs ONCE per mount and creates
// the static structure (buttons, layout) so button click listeners persist.
// rebuildSidebar() runs every update() call and only mutates the dynamic
// text inside specific spans/divs — never replaces the buttons. Previous
// design re-set host.innerHTML on every frame, which destroyed the buttons
// faster than a click could complete → click events lost.

let latestState = null;  // module-level so static listeners can use latest

function buildSidebarSkeleton() {
  if (!host) return;
  host.innerHTML = `
    <h2>Punch direction review</h2>
    <p class="hint">
      <span style="color:${COLOR_RAW}">●</span> raw per-frame arrow ·
      <span style="color:${COLOR_MEDIAN}">●</span> median arrow over window ·
      <span style="color:${COLOR_PRED}">●</span> predicted facing ·
      <span style="color:${COLOR_GT}">●</span> GT label
    </p>
    <div class="ol-nav" style="display:flex; gap:8px; align-items:center; margin:10px 0 14px;">
      <button id="pdr-prev" class="orient-btn-action secondary" style="padding:6px 10px;">⏮ prev (P)</button>
      <button id="pdr-next" class="orient-btn-action secondary" style="padding:6px 10px;">next (N) ⏭</button>
      <span id="pdr-counter" style="margin-left:6px; color:#888; font-size:12px;"></span>
    </div>
    <div id="pdr-state" class="hint" style="line-height:1.55;"></div>
    <p class="hint" style="margin-top:14px; font-size:11px;">
      Punch loops automatically within [start, end]. Use N/P to step through
      punches. The lens iterates every punch_uuid in Combined Data for this
      video — GT arrow only draws when a Punch Directions row exists.
    </p>
  `;
  // Listeners attached once, persist for the life of the mount. They read
  // the freshest state from latestState (updated by rebuildSidebar / mount).
  host.querySelector("#pdr-prev")?.addEventListener("click",
    () => seekToPunch(activeIdx - 1, latestState));
  host.querySelector("#pdr-next")?.addEventListener("click",
    () => seekToPunch(activeIdx + 1, latestState));
}

function rebuildSidebar(state) {
  if (!host) return;
  latestState = state;
  // First call after mount: build the static skeleton.
  if (!host.querySelector("#pdr-state")) buildSidebarSkeleton();

  const current = labelledPunches[activeIdx] || null;
  const det = current?.detection;
  const med = current?.median;
  const fit = det ? STANCE_FITS[det.stance] : null;
  const predFromMedian = (med && fit)
    ? wrap180(fit.sign * med.angle_deg + fit.offset_deg)
    : null;
  const gt = current?.gtLabel ?? null;
  const err = (predFromMedian != null && gt != null)
    ? Math.abs(wrap180(predFromMedian - gt)) : null;

  const f = state.frame;
  // Loose so the sidebar's live-arrow readout doesn't blink off on
  // low-conf frames (matches the visual draw).
  const liveVec = det ? ankleVecLoose(state.pose, f, det.stance) : null;
  const liveAngle = liveVec
    ? Math.atan2(liveVec.dy, liveVec.dx) * 180 / Math.PI : null;

  const counter = host.querySelector("#pdr-counter");
  if (counter) {
    counter.textContent = labelledPunches.length
      ? `${activeIdx + 1} / ${labelledPunches.length}`
      : "no labelled punches";
  }

  const el = host.querySelector("#pdr-state");
  if (!el) return;
  const lines = [];

  // Diagnose: which side of the join is missing?
  // - byUuidMap null → Punch Directions sheet still fetching (or failed)
  // - state.labels null → Combined Data not loaded yet
  // - state.labels.error → Combined Data load failed (e.g., auto-match)
  // - state.labels.detections empty → no rows for this video in Combined Data
  // - labelledPunches empty BUT detections present → no uuid overlap
  if (fetchError) {
    lines.push(`<span class="bad">Punch Directions fetch failed: ${fetchError}</span>`);
  } else if (!byUuidMap) {
    lines.push(`<span class="muted">Fetching Punch Directions sheet…</span>`);
  } else if (!state.labels) {
    lines.push(`<span class="muted">Combined Data not loaded yet — waiting for tryLiveLabels…</span>`);
  } else if (state.labels.error) {
    lines.push(`<span class="bad">Combined Data error: ${state.labels.error}</span>`);
    lines.push(`<span class="muted">cacheBasename used for match: <code>${state.cacheBasename || "(none)"}</code></span>`);
  } else if (!state.labels.detections || state.labels.detections.length === 0) {
    lines.push(`<span class="bad">Combined Data has 0 detections for this video</span>`);
    lines.push(`<span class="muted">cacheBasename: <code>${state.cacheBasename || "(none)"}</code>, sheet matched: <code>${state.labels.source_video || "(none)"}</code></span>`);
  } else if (!labelledPunches.length) {
    lines.push(`<span class="bad">Combined Data has 0 detections with a punch_uuid for this video.</span>`);
    lines.push(`<span class="muted">Either this video has no punch labels yet, or the rows are missing punch_uuid values.</span>`);
  } else if (det) {
    lines.push(`<code>${det.punch_type}</code> · stance <code>${det.stance}</code>`);
    lines.push(`frames <code>${det.start_frame}-${det.end_frame}</code> (${(det.end_frame - det.start_frame + 1)} frames)`);
    if (liveAngle != null) {
      lines.push(`<span style="color:${COLOR_RAW}">live arrow:</span> <code>${liveAngle.toFixed(1)}°</code>`);
    }
    if (med) {
      lines.push(`<span style="color:${COLOR_MEDIAN}">median arrow:</span> <code>${med.angle_deg.toFixed(1)}°</code> (over ${med.n_valid} valid frames)`);
    } else {
      lines.push(`<span class="muted">median arrow: insufficient valid frames</span>`);
    }
    if (predFromMedian != null) {
      lines.push(`<span style="color:${COLOR_PRED}">predicted facing:</span> <code>${predFromMedian.toFixed(1)}°</code>`);
    }
    if (gt != null) {
      lines.push(`<span style="color:${COLOR_GT}">GT label:</span> <code>${gt.toFixed(1)}°</code>${current.gtLabeler ? ` by ${current.gtLabeler}` : ""}`);
      if (err != null) {
        lines.push(`error: <code>${err.toFixed(1)}°</code>`);
      }
    } else {
      lines.push(`<span class="muted">no GT direction label — prediction only</span>`);
    }
  }
  el.innerHTML = lines.join("<br>");
}

// ─── lens contract ────────────────────────────────────────────────────────

export const PunchDirectionReviewRule = {
  id: "punch_direction_review",
  label: "Punch direction review",

  mount(_host, state) {
    host = _host;
    videoEl = document.getElementById("video");
    rebuildSidebar(state);
    installTimeupdateLoop(state);
    installKeyHandlers(state);
    ensureFetched(state);
    rebuildLabelledPunches(state);
  },

  update(state) {
    ensureFetched(state);
    rebuildLabelledPunches(state);
    // If the user has scrubbed entirely AWAY from the active punch's window,
    // hop to whichever labelled punch the cursor is now inside. Inside (or
    // within a short tolerance of) the active window we stay locked.
    //
    // The tolerance handles a race: when playback crosses the end of the
    // active punch, the viewer's timeupdate listener updates state.frame
    // to `end+1` BEFORE my lens-installed timeupdate handler fires its
    // snap-back. update() runs in between — and without tolerance it sees
    // "frame outside active, frame inside overlapping punch 16" and
    // switches. Then the snap snaps to the WRONG punch's start. Flip-flop.
    //
    // 15-frame tolerance (~500ms at 30fps) absorbs that gap while staying
    // small enough that real manual scrubs to neighbouring punches still
    // auto-switch correctly.
    const ACTIVE_TOLERANCE_FRAMES = 15;
    if (labelledPunches.length) {
      const f = state.frame;
      const active = activeIdx >= 0 ? labelledPunches[activeIdx] : null;
      const nearActive = active &&
        f >= active.detection.start_frame - ACTIVE_TOLERANCE_FRAMES &&
        f <= active.detection.end_frame   + ACTIVE_TOLERANCE_FRAMES;
      if (!nearActive) {
        const inside = labelledPunches.findIndex(p =>
          f >= p.detection.start_frame && f <= p.detection.end_frame);
        if (inside !== -1 && inside !== activeIdx) {
          activeIdx = inside;
          const det = labelledPunches[inside].detection;
          loopWindow = { start_frame: det.start_frame, end_frame: det.end_frame,
                         uuid: det.punch_uuid };
          rebuildSidebar(state);
        }
      }
    }
  },

  draw(ctx, state) {
    if (!labelledPunches.length || activeIdx < 0) return;
    const current = labelledPunches[activeIdx];
    const det = current.detection;
    const med = current.median;
    const fit = STANCE_FITS[det.stance];
    const f = state.frame;
    const s = state.renderScale || 1;

    // (1) Live per-frame raw arrow. Use the LOOSE ankle vec (no conf gate)
    // so the amber arrow doesn't flicker off mid-punch — pose-detector
    // confidence routinely dips during load-up / follow-through. When conf
    // is below the strict threshold we still draw, just at half opacity
    // as a visual hint that this frame is uncertain.
    const live = ankleVecLoose(state.pose, f, det.stance);
    if (live) {
      ctx.save();
      ctx.globalAlpha = live.minConf >= MIN_ANKLE_CONF ? 1.0 : 0.5;
      ctx.strokeStyle = COLOR_RAW;
      ctx.fillStyle = COLOR_RAW;
      ctx.lineWidth = 2.5 * s;
      ctx.beginPath();
      // Orthodox: tail at R (back foot), head at L (front foot).
      // Southpaw: mirrored. Matches window_arrow / orientation_lens.
      const orthodox = det.stance !== "southpaw";
      const sx = orthodox ? live.rx : live.lx;
      const sy = orthodox ? live.ry : live.ly;
      const ex = orthodox ? live.lx : live.rx;
      const ey = orthodox ? live.ly : live.ry;
      ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();
      // small dots at each ankle for spatial context
      for (const [x, y] of [[live.lx, live.ly], [live.rx, live.ry]]) {
        ctx.beginPath(); ctx.arc(x, y, 3 * s, 0, Math.PI * 2); ctx.fill();
      }
      // arrowhead at the head end
      drawArrow(ctx, sx, sy, Math.atan2(ey - sy, ex - sx),
                Math.hypot(ex - sx, ey - sy), COLOR_RAW,
                { lineWidth: 0, headSize: 10 * s });
      ctx.restore();
    }

    // Ankle midpoint of CURRENT frame — anchor for the median arrow.
    // Use loose ankleVec here too so the orange median arrow stays anchored
    // even on low-conf frames; the median value itself is still computed
    // from the strict ankleVecAt above (in computeMedianArrow).
    let ankleMid = null;
    if (live) {
      ankleMid = { x: 0.5 * (live.lx + live.rx), y: 0.5 * (live.ly + live.ry) };
    }

    // (2) Median arrow — drawn from ankle midpoint, fixed length, dashed
    // so it visually layers with the live amber line without becoming
    // confusing when they overlap.
    if (med && ankleMid) {
      const len = 70 * s;
      drawArrow(ctx, ankleMid.x, ankleMid.y,
        Math.atan2(med.dy, med.dx), len, COLOR_MEDIAN,
        { dashed: true, lineWidth: 3 * s, headSize: 12 * s });
    }

    // Hip midpoint — anchor for predicted + GT arrows.
    const hip = hipMidAt(state.pose, f);
    if (!hip) return;

    // (3) Predicted facing from median (cyan). Constant across the punch.
    if (med && fit) {
      const predGt = wrap180(fit.sign * med.angle_deg + fit.offset_deg);
      const imgAngle = (90 - predGt) * Math.PI / 180;
      drawArrow(ctx, hip.x, hip.y, imgAngle, 80 * s, COLOR_PRED,
        { lineWidth: 3.5 * s, headSize: 14 * s });
    }

    // (4) GT label (red, dashed). Constant across the punch.
    const gt = current.gtLabel;
    if (gt != null) {
      const imgAngle = (90 - gt) * Math.PI / 180;
      drawArrow(ctx, hip.x, hip.y, imgAngle, 80 * s, COLOR_GT,
        { dashed: true, lineWidth: 3.5 * s, headSize: 14 * s });
    }
  },

  // Clean up listeners when lens unmounts (different rule picked).
  unmount() {
    if (videoEl && timeupdateHandler) {
      videoEl.removeEventListener("timeupdate", timeupdateHandler);
    }
    if (keydownHandler) {
      document.removeEventListener("keydown", keydownHandler);
    }
    timeupdateHandler = null;
    keydownHandler = null;
    loopWindow = null;
  },
};
