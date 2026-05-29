// Straights review — walk through every straight punch (jab/cross, head/body),
// see if it qualifies on orientation, and review the arm-extension data side
// by side with the canvas overlay. Mirrors punch_direction_review.js:
//   - one punch at a time, video loops within [start_frame, end_frame]
//   - N / P keys (or buttons) step to next / previous straight
//   - sidebar shows the data; canvas shows the arrows + arm + HUD
//
// Reuses arm_extension.js for the per-punch compute pipeline so the verdicts
// here match exactly what the Arm-extension lens shows. The orientation gate
// in arm_extension drives "qualifies" — when it's on and the fighter wasn't
// sideways, predicted = "skip" and this lens flags the punch as not
// qualifying for the extension verdict.

import { J } from "../skeleton.js";
import {
  computeArmExtension,
  ARM_EXT_DEFAULTS,
} from "./arm_extension.js";
import { STANCE_FITS } from "./orientation_lens.js";

const STRAIGHTS = new Set([
  "jab_head", "jab_body", "cross_head", "cross_body",
]);

const COLOR_PRED   = "#3ad9e0";  // cyan — predicted facing
const COLOR_ANKLE  = "#ffd24a";  // amber — ankle line
const COLOR_PASS   = "#5fd97a";
const COLOR_FAIL   = "#e85a5a";
const COLOR_SKIP   = "#7ec8ff";
const COLOR_UNCLEAR = "#f5b945";
const MIN_ANKLE_CONF = 0.30;

let host = null;
let videoEl = null;
let timeupdateHandler = null;
let keydownHandler = null;
let loopWindow = null;
let activeIdx = -1;
let punches = [];          // straights from arm_extension's compute pipeline
let lastDetectionsRef = null;
let lastStemForReset = null;
let lastPose = null;
let latestState = null;
let cfg = { ...ARM_EXT_DEFAULTS };

// ─── helpers ──────────────────────────────────────────────────────────────

function wrap180(deg) { return ((deg + 180) % 360 + 360) % 360 - 180; }

function colorFor(predicted) {
  if (predicted === "pass") return COLOR_PASS;
  if (predicted === "fail") return COLOR_FAIL;
  if (predicted === "skip") return COLOR_SKIP;
  return COLOR_UNCLEAR;
}

function ankleArrowAt(pose, f, stance) {
  const cL = pose.conf[f * 17 + J.L_ANKLE];
  const cR = pose.conf[f * 17 + J.R_ANKLE];
  if (cL < MIN_ANKLE_CONF || cR < MIN_ANKLE_CONF) return null;
  const lx = pose.skeleton[(f * 17 + J.L_ANKLE) * 2];
  const ly = pose.skeleton[(f * 17 + J.L_ANKLE) * 2 + 1];
  const rx = pose.skeleton[(f * 17 + J.R_ANKLE) * 2];
  const ry = pose.skeleton[(f * 17 + J.R_ANKLE) * 2 + 1];
  if (![lx, ly, rx, ry].every(Number.isFinite)) return null;
  const orthodox = stance !== "southpaw";
  return {
    lx, ly, rx, ry,
    fx: orthodox ? lx : rx, fy: orthodox ? ly : ry,
    bx: orthodox ? rx : lx, by: orthodox ? ry : ly,
  };
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

function drawArrowhead(ctx, x, y, angle, size) {
  const a = 0.45;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x - size * Math.cos(angle - a), y - size * Math.sin(angle - a));
  ctx.lineTo(x - size * Math.cos(angle + a), y - size * Math.sin(angle + a));
  ctx.closePath();
  ctx.fill();
}

// ─── data plumbing ────────────────────────────────────────────────────────

function rebuildPunches(state) {
  const stem = state.cacheBasename || "";
  const dets = state.labels?.detections;
  const stemChanged = stem !== lastStemForReset;
  const detsChanged = dets !== lastDetectionsRef;
  const poseChanged = state.pose !== lastPose;
  if (!stemChanged && !detsChanged && !poseChanged && punches.length) {
    rebuildSidebar(state);
    return;
  }
  lastStemForReset = stem;
  lastDetectionsRef = dets;
  lastPose = state.pose;

  // computeArmExtension already filters to cfg.appliesTo (= straights) and
  // attaches orientation_deg / orientation_sideways / predicted to each punch.
  const sig = computeArmExtension(state, cfg);
  const next = sig.punches
    .filter(p => STRAIGHTS.has(p.punch_type))
    .sort((a, b) => a.start_frame - b.start_frame);

  const hadNone = punches.length === 0;
  punches = next;
  if (stemChanged || (hadNone && next.length)) {
    if (next.length) seekToPunch(0, state);
    else { loopWindow = null; activeIdx = -1; }
  } else if (activeIdx >= next.length) {
    activeIdx = next.length - 1;
  }
  rebuildSidebar(state);
}

function seekToPunch(idx, state) {
  if (!punches.length) return;
  if (idx < 0) idx = 0;
  if (idx >= punches.length) idx = punches.length - 1;
  const p = punches[idx];
  activeIdx = idx;
  loopWindow = { start_frame: p.start_frame, end_frame: p.end_frame };
  if (videoEl && state.fps) {
    videoEl.currentTime = (state.start_sec || 0) + p.start_frame / state.fps;
    if (videoEl.paused) {
      const promise = videoEl.play();
      if (promise && typeof promise.catch === "function") promise.catch(() => {});
    }
  }
  rebuildSidebar(state);
}

function installTimeupdateLoop(state) {
  if (!videoEl) return;
  if (timeupdateHandler) videoEl.removeEventListener("timeupdate", timeupdateHandler);
  timeupdateHandler = () => {
    if (state.rule?.id !== "straights_review") return;
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
    if (state.rule?.id !== "straights_review") return;
    const tag = e.target?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    if (e.key === "n" || e.key === "N") { e.preventDefault(); seekToPunch(activeIdx + 1, state); }
    else if (e.key === "p" || e.key === "P") { e.preventDefault(); seekToPunch(activeIdx - 1, state); }
  };
  document.addEventListener("keydown", keydownHandler);
}

// ─── sidebar ──────────────────────────────────────────────────────────────

function buildSidebarSkeleton() {
  if (!host) return;
  host.innerHTML = `
    <h2>Straights review</h2>
    <p class="hint">
      Walk through every straight punch (jab/cross · head/body). Each punch
      first checks the <b>orientation gate</b> (fighter must be sideways,
      |angle| in <code>[${cfg.orientationMinAbsDeg}°, ${cfg.orientationMaxAbsDeg}°]</code>),
      then the arm-extension verdict — peak ratio + reach. Skip-status
      means we can't trust the geometry for this punch.
    </p>
    <div class="ol-nav" style="display:flex; gap:8px; align-items:center; margin:10px 0 14px;">
      <button id="sr-prev" class="orient-btn-action secondary" style="padding:6px 10px;">⏮ prev (P)</button>
      <button id="sr-next" class="orient-btn-action secondary" style="padding:6px 10px;">next (N) ⏭</button>
      <span id="sr-counter" style="margin-left:6px; color:#888; font-size:12px;"></span>
    </div>
    <div id="sr-state" class="hint" style="line-height:1.7;"></div>
    <p class="hint" style="margin-top:14px; font-size:11px;">
      Loops within the punch window. Settings (gate range, thresholds) come
      from the Arm-extension lens defaults; edit them there to tweak.
    </p>
  `;
  host.querySelector("#sr-prev")?.addEventListener("click",
    () => seekToPunch(activeIdx - 1, latestState));
  host.querySelector("#sr-next")?.addEventListener("click",
    () => seekToPunch(activeIdx + 1, latestState));
}

function pill(text, color) {
  return `<span style="display:inline-block;padding:1px 8px;border-radius:10px;
    font-size:12px;font-weight:600;letter-spacing:0.02em;
    background:${color}1f;color:${color};border:1px solid ${color}66">${text}</span>`;
}

function rebuildSidebar(state) {
  if (!host) return;
  latestState = state;
  if (!host.querySelector("#sr-state")) buildSidebarSkeleton();

  const counter = host.querySelector("#sr-counter");
  if (counter) {
    counter.textContent = punches.length
      ? `${activeIdx + 1} / ${punches.length}`
      : "no straights in this round";
  }

  const el = host.querySelector("#sr-state");
  if (!el) return;

  if (!punches.length) {
    el.innerHTML = state.labels?.detections
      ? `<span class="muted">No straights labeled in this round.</span>`
      : `<span class="muted">Waiting for label data…</span>`;
    return;
  }

  const p = punches[activeIdx];
  if (!p) { el.innerHTML = ""; return; }

  // 1) Orientation block — does it qualify?
  const oDeg = p.orientation_deg;
  const sideways = p.orientation_sideways;
  let orientLine;
  if (oDeg == null) {
    orientLine = `<span style="color:${COLOR_PRED}">orientation:</span> <code>—</code> · ${pill("UNKNOWN", COLOR_UNCLEAR)}`;
  } else {
    const a = Math.abs(oDeg);
    const status = sideways
      ? pill("SIDEWAYS ✓", COLOR_PASS)
      : pill("NOT SIDEWAYS ✗", COLOR_FAIL);
    orientLine = `<span style="color:${COLOR_PRED}">orientation:</span> `
      + `<code>${oDeg.toFixed(1)}°</code> · |a|=<code>${a.toFixed(1)}°</code> `
      + `· gate [${cfg.orientationMinAbsDeg}°, ${cfg.orientationMaxAbsDeg}°] · ${status}`;
  }

  // 2) Extension data
  const peakStr   = Number.isFinite(p.peak) ? p.peak.toFixed(3) : "—";
  const bendStr   = Number.isFinite(p.peak_bend_deg) ? `${p.peak_bend_deg.toFixed(1)}°` : "—";
  const reachStr  = Number.isFinite(p.peak_reach) ? p.peak_reach.toFixed(2) : "—";
  const travelStr = Number.isFinite(p.peak_travel) ? p.peak_travel.toFixed(2) : "—";

  const peakCol  = Number.isFinite(p.peak)
    ? (p.peak >= cfg.threshold ? COLOR_PASS : COLOR_FAIL) : "var(--muted, #888)";
  const reachCol = Number.isFinite(p.peak_reach)
    ? (p.peak_reach >= cfg.reachThreshold ? COLOR_PASS : COLOR_FAIL)
    : "var(--muted, #888)";

  // 3) Verdict
  const predCol = colorFor(p.predicted);
  const verdictLine = `<span style="color:${predCol}">predicted:</span> ${pill(p.predicted, predCol)}`
    + (p.label
        ? ` · <span class="muted">GT:</span> ${pill(p.label, colorFor(p.label))}`
          + (p.label === p.predicted
              ? ` <span style="color:${COLOR_PASS}">✓</span>`
              : ` <span style="color:${COLOR_FAIL}">✗</span>`)
        : ` · <span class="muted">no GT</span>`);

  const lines = [
    `<b>${p.punch_type}</b> · <code>${p.hand}</code> · stance <code>${p.stance}</code>`,
    `frames <code>${p.start_frame}-${p.end_frame}</code> (${p.end_frame - p.start_frame + 1} frames) · peak@<code>${p.land_frame}</code>`,
    "",
    `<b>1. Orientation gate</b>`,
    orientLine,
    "",
    `<b>2. Arm extension</b>`,
    `peak ratio <code style="color:${peakCol}">${peakStr}</code> `
      + `(threshold <code>${cfg.threshold.toFixed(2)}</code>) · bend <code>${bendStr}</code>`,
    `peak reach <code style="color:${reachCol}">${reachStr}</code> `
      + `(threshold <code>${cfg.reachThreshold.toFixed(2)}</code>) · travel <code>${travelStr}</code>`,
    `wrist src: <code>${p.peak_wr_source}</code> · sh <code>${Number.isFinite(p.peak_sh_conf) ? p.peak_sh_conf.toFixed(2) : "—"}</code> · el <code>${Number.isFinite(p.peak_el_conf) ? p.peak_el_conf.toFixed(2) : "—"}</code> · wr <code>${Number.isFinite(p.peak_wr_conf) ? p.peak_wr_conf.toFixed(2) : "—"}</code>`,
    "",
    `<b>3. Verdict</b>`,
    verdictLine,
  ];
  el.innerHTML = lines.join("<br>");
}

// ─── draw ─────────────────────────────────────────────────────────────────

function drawCanvas(ctx, state) {
  if (!punches.length || activeIdx < 0) return;
  const p = punches[activeIdx];
  const f = state.frame;
  const s = state.renderScale || 1;
  const pose = state.pose;

  // Ankle line — amber, for the input that drives orientation
  const ank = ankleArrowAt(pose, f, p.stance);
  if (ank) {
    ctx.save();
    ctx.strokeStyle = COLOR_ANKLE;
    ctx.fillStyle = COLOR_ANKLE;
    ctx.lineWidth = 2.5 * s;
    ctx.beginPath(); ctx.moveTo(ank.bx, ank.by); ctx.lineTo(ank.fx, ank.fy); ctx.stroke();
    drawArrowhead(ctx, ank.fx, ank.fy,
      Math.atan2(ank.fy - ank.by, ank.fx - ank.bx), 11 * s);
    for (const [x, y] of [[ank.lx, ank.ly], [ank.rx, ank.ry]]) {
      ctx.beginPath(); ctx.arc(x, y, 3 * s, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
  }

  // Predicted facing arrow (cyan) from hip midpoint — fixed across the
  // window (window-median orientation).
  const hip = hipMidAt(pose, f);
  if (hip && p.orientation_deg != null) {
    const imgAngle = (90 - p.orientation_deg) * Math.PI / 180;
    const len = 80 * s;
    const x1 = hip.x + len * Math.cos(imgAngle);
    const y1 = hip.y + len * Math.sin(imgAngle);
    ctx.save();
    ctx.strokeStyle = COLOR_PRED;
    ctx.fillStyle = COLOR_PRED;
    ctx.lineWidth = 3.5 * s;
    ctx.beginPath(); ctx.moveTo(hip.x, hip.y); ctx.lineTo(x1, y1); ctx.stroke();
    drawArrowhead(ctx, x1, y1, imgAngle, 14 * s);
    ctx.restore();
  }

  drawHud(ctx, p, s);
}

function drawHud(ctx, p, s) {
  const predCol = colorFor(p.predicted);
  const orientStatus = p.orientation_deg == null
    ? "orientation: —"
    : (p.orientation_sideways
        ? `orientation: ${p.orientation_deg.toFixed(0)}° sideways ✓`
        : `orientation: ${p.orientation_deg.toFixed(0)}° not sideways ✗`);
  const orientCol = p.orientation_deg == null ? COLOR_UNCLEAR
                   : (p.orientation_sideways ? COLOR_PASS : COLOR_FAIL);
  const peakTxt = Number.isFinite(p.peak)
    ? `peak r ${p.peak.toFixed(3)}` + (Number.isFinite(p.peak_bend_deg)
        ? `  ·  bend ${p.peak_bend_deg.toFixed(1)}°` : "")
    : "peak r —";
  const reachTxt = Number.isFinite(p.peak_reach)
    ? `reach ${p.peak_reach.toFixed(2)}` : "reach —";
  const titleTxt = `${p.hand} ${p.punch_type}  ·  ${p.stance}`;
  const predTxt  = `pred: ${p.predicted}`;
  const gtTxt    = p.label ? `GT: ${p.label}` : "GT: —";
  const agreeSym = p.label && p.predicted !== "unclear"
    ? (p.label === p.predicted ? "  ✓" : "  ✗") : "";

  const fontPx = 15 * s;
  const lineH  = 22 * s;
  const padX   = 14 * s;
  const padY   = 10 * s;
  const x0 = 24 * s, y0 = 24 * s;

  const lines = [titleTxt, orientStatus, peakTxt, reachTxt, predTxt, gtTxt + agreeSym];

  ctx.save();
  ctx.font = `bold ${fontPx}px ui-monospace, "SF Mono", Menlo, monospace`;
  const w = Math.max(...lines.map(t => ctx.measureText(t).width)) + 2 * padX;
  const h = padY * 2 + lineH * lines.length;

  ctx.fillStyle = "rgba(0,0,0,0.78)";
  const r = 10 * s;
  ctx.beginPath();
  ctx.moveTo(x0 + r, y0);
  ctx.arcTo(x0 + w, y0,     x0 + w, y0 + h, r);
  ctx.arcTo(x0 + w, y0 + h, x0,     y0 + h, r);
  ctx.arcTo(x0,     y0 + h, x0,     y0,     r);
  ctx.arcTo(x0,     y0,     x0 + w, y0,     r);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.lineWidth = 1 * s;
  ctx.stroke();

  let y = y0 + padY + fontPx;
  // 1) title
  ctx.fillStyle = "rgba(255,255,255,0.92)"; ctx.fillText(titleTxt, x0 + padX, y); y += lineH;
  // 2) orientation
  ctx.fillStyle = orientCol; ctx.fillText(orientStatus, x0 + padX, y); y += lineH;
  // 3) peak
  ctx.fillStyle = Number.isFinite(p.peak)
    ? (p.peak >= cfg.threshold ? COLOR_PASS : COLOR_FAIL)
    : "rgba(255,255,255,0.72)";
  ctx.fillText(peakTxt, x0 + padX, y); y += lineH;
  // 4) reach
  ctx.fillStyle = Number.isFinite(p.peak_reach)
    ? (p.peak_reach >= cfg.reachThreshold ? COLOR_PASS : COLOR_FAIL)
    : "rgba(255,255,255,0.72)";
  ctx.fillText(reachTxt, x0 + padX, y); y += lineH;
  // 5) pred
  ctx.fillStyle = predCol; ctx.fillText(predTxt, x0 + padX, y); y += lineH;
  // 6) GT
  ctx.fillStyle = p.label ? colorFor(p.label) : "rgba(255,255,255,0.55)";
  ctx.fillText(gtTxt, x0 + padX, y);
  if (agreeSym) {
    ctx.fillStyle = p.label === p.predicted ? COLOR_PASS : COLOR_FAIL;
    ctx.fillText(agreeSym, x0 + padX + ctx.measureText(gtTxt).width, y);
  }
  ctx.restore();
}

// ─── lens contract ────────────────────────────────────────────────────────

export const StraightsReviewRule = {
  id: "straights_review",
  label: "Straights review (orientation + extension)",

  requires(slot) {
    // Needs pose + glove like arm_extension — we delegate to the same compute.
    // v6 cache alone is sufficient (Vision + glove substitution baked in);
    // otherwise fall back to raw Vision/YOLO + glove sidecar.
    return !!slot?.vision_glove
      || (!!(slot?.vision || slot?.yolo) && !!slot?.glove);
  },

  mount(_host, state) {
    host = _host;
    videoEl = document.getElementById("video");
    cfg = { ...ARM_EXT_DEFAULTS };
    lastPose = null;
    lastDetectionsRef = null;
    lastStemForReset = null;
    rebuildPunches(state);
    installTimeupdateLoop(state);
    installKeyHandlers(state);
  },

  update(state) {
    rebuildPunches(state);
    // If the user scrubbed away from the active punch, hop to whichever
    // straight the cursor is now inside.
    const ACTIVE_TOLERANCE_FRAMES = 15;
    if (punches.length) {
      const f = state.frame;
      const active = activeIdx >= 0 ? punches[activeIdx] : null;
      const nearActive = active &&
        f >= active.start_frame - ACTIVE_TOLERANCE_FRAMES &&
        f <= active.end_frame   + ACTIVE_TOLERANCE_FRAMES;
      if (!nearActive) {
        const inside = punches.findIndex(p =>
          f >= p.start_frame && f <= p.end_frame);
        if (inside !== -1 && inside !== activeIdx) {
          activeIdx = inside;
          const p = punches[inside];
          loopWindow = { start_frame: p.start_frame, end_frame: p.end_frame };
          rebuildSidebar(state);
        }
      }
    }
  },

  draw(ctx, state) {
    drawCanvas(ctx, state);
  },

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
