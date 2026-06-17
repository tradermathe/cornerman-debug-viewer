// Angle-change lens — visualises the "pivot_rate" coaching rule on the
// deployed viewer.
//
// Mental model: between two punches, did the boxer's facing direction
// swing far enough to count as a meaningful angle change? We sample
// orientation at each punch (when feet are momentarily planted —
// outside punches the arrow shoots everywhere with footwork) and run a
// ratchet over the sequence:
//
//   anchor A := orientation at first valid punch
//   for each subsequent sample with orientation O_i:
//     d_i = shortest_signed_angle(O_i - A)         // in (-180°, 180°]
//     max_pos = max(max_pos, d_i)
//     min_neg = min(min_neg, d_i)
//     if max_pos - min_neg >= swing_threshold:
//        fire pivot, reset anchor := O_i, max_pos = min_neg = 0
//
// "Pivot" is technically the footwork move (one foot planted, body
// rotates around it); what we actually detect is ANY ≥swing° change in
// facing direction. The lens label says "Angle change" to keep the
// terminology honest.
//
// Orientation source
// ──────────────────
// The Python rule uses a trained LogReg classifier
// (cornerman_rules/models/orientation_v1.pkl) we don't have in the
// browser. Instead we reuse the ankle-arrow + per-stance fit from
// orientation_lens.js, which is the same model the Sheet labeller fits
// against. Per-punch sample = circular mean of per-frame facing
// estimates across the punch window, weighted by min(ankle conf).
//
// Caveat: because the orientation model differs from the Python rule's
// LogReg, per-punch angles here will not perfectly match what the rule
// engine would emit on the same round. The ratchet logic IS identical,
// so the structural read ("did it fire here, what swing was open") is
// directly translatable.

import { J } from "../skeleton.js";
import { STANCE_FITS } from "./orientation_lens.js";

const MIN_ANKLE_CONF = 0.30;

const DEFAULTS = {
  swingDeg: 100,      // matches rules_config.json pivot_swing_degrees
  minPunches: 8,      // matches min_punches gate
  sevMild: 12,        // seconds per pivot (>= mild)
  sevModerate: 20,
  sevSevere: 60,
  // 0–100 score: S-curve on sec-per-pivot with an explicit midpoint, so the
  // steep part sits where the coach wants it (not the geometric centre).
  // 0 at scoreStart (pivoting often enough), steepest (50) at scoreMid,
  // 100 at scoreSat (too stationary). One-sided.
  scoreStart: 12,
  scoreMid:   25,
  scoreSat:   60,
  scoreK:     10,
};

const COLORS = {
  anchor:  "#9ca3af",  // grey — anchor facing direction
  current: "#3ad9e0",  // cyan — current punch facing direction
  range:   "rgba(245,185,69,0.30)", // amber — open swing range
  fired:   "#5fd97a",  // green — a pivot fired on this punch
  warn:    "#f5b945",
  bad:     "#e85a5a",
};

function wrap180(d) { return ((d + 180) % 360 + 360) % 360 - 180; }
function signedDiff(a, b) {
  let d = ((a - b + 180) % 360 + 360) % 360 - 180;
  if (d === -180) d = 180;
  return d;
}

function pickSource(state) {
  if (state.labels?.detections?.length) {
    return { kind: "labels", detections: state.labels.detections };
  }
  if (state.punches?.detections?.length) {
    return { kind: "punches", detections: state.punches.detections };
  }
  return null;
}

// Per-punch facing direction: circular mean of per-frame ankle-arrow
// predictions across the punch window. Returns { angle_deg, used: bool,
// stance } where used=false means the window had no usable ankle data
// (drop from the ratchet but keep in the table for debug).
function punchSample(pose, det) {
  const stance = det.stance?.toLowerCase?.();
  const fit = stance ? STANCE_FITS[stance] : null;
  if (!fit) return { angle: null, used: false, stance: null };

  const sf = Math.max(0, det.start_frame ?? 0);
  const ef = Math.min(pose.n_frames - 1, det.end_frame ?? sf);
  let x = 0, y = 0, w = 0;
  let firstFrame = null, lastFrame = null;
  for (let f = sf; f <= ef; f++) {
    const cL = pose.conf[f * 17 + J.L_ANKLE];
    const cR = pose.conf[f * 17 + J.R_ANKLE];
    if (cL < MIN_ANKLE_CONF || cR < MIN_ANKLE_CONF) continue;
    const lx = pose.skeleton[(f * 17 + J.L_ANKLE) * 2];
    const ly = pose.skeleton[(f * 17 + J.L_ANKLE) * 2 + 1];
    const rx = pose.skeleton[(f * 17 + J.R_ANKLE) * 2];
    const ry = pose.skeleton[(f * 17 + J.R_ANKLE) * 2 + 1];
    if (![lx, ly, rx, ry].every(Number.isFinite)) continue;
    // Same convention as orientation_lens.js: orthodox arrow R→L,
    // southpaw L→R (arrowhead at FRONT foot).
    const orthodox = stance === "orthodox";
    const fx = orthodox ? lx : rx, fy = orthodox ? ly : ry;
    const bx = orthodox ? rx : lx, by = orthodox ? ry : ly;
    const dx = fx - bx, dy = fy - by;
    if (dx * dx + dy * dy < 1e-6) continue;
    const arrowAng = Math.atan2(dy, dx) * 180 / Math.PI;
    const gt = wrap180(fit.sign * arrowAng + fit.offset_deg);
    const ww = Math.min(cL, cR);
    const rad = gt * Math.PI / 180;
    x += ww * Math.cos(rad);
    y += ww * Math.sin(rad);
    w += ww;
    if (firstFrame == null) firstFrame = f;
    lastFrame = f;
  }
  if (w < 1e-6) return { angle: null, used: false, stance };
  const meanAng = Math.atan2(y, x) * 180 / Math.PI;
  return { angle: meanAng, used: true, stance, firstFrame, lastFrame };
}

// Shared score shape: normalised logistic on x∈[0,1].
function sigmoid01(x, k) {
  const L = (t) => 1 / (1 + Math.exp(-k * (t - 0.5)));
  return (L(x) - L(0)) / (L(1) - L(0));
}
// Map v through start/mid/sat so v=mid lands at the sigmoid's steep centre
// (x=0.5) — an asymmetric S that puts the cliff at `mid`, not the midpoint of
// [start, sat]. start→0, mid→0.5, sat→1.
function sigmoidAnchored(v, start, mid, sat, k) {
  if (!(v > start)) return 0;
  if (v >= sat) return 1;
  if (!(mid > start && mid < sat)) return sigmoid01((v - start) / (sat - start), k);
  const x = v <= mid
    ? 0.5 * (v - start) / (mid - start)
    : 0.5 + 0.5 * (v - mid) / (sat - mid);
  return sigmoid01(x, k);
}
// 0–100 from sec-per-pivot. 0 pivots → secPerPivot = roundSec, so it maps by
// the rate (a long no-pivot round scores higher than a short one).
function pivotScore(secPerPivot, cfg) {
  if (!Number.isFinite(secPerPivot)) return null;
  return Math.round(100 * sigmoidAnchored(secPerPivot, cfg.scoreStart, cfg.scoreMid, cfg.scoreSat, cfg.scoreK));
}

function compute(state, cfg) {
  const source = pickSource(state);
  const detections = (source?.detections || []).slice().sort(
    (a, b) => (a.start_frame ?? 0) - (b.start_frame ?? 0)
  );
  const fps = state.pose.fps || 30;
  const N = state.pose.n_frames;
  const roundSec = N / fps;

  if (detections.length < cfg.minPunches) {
    return {
      sourceKind: source?.kind || "none",
      detections,
      annotated: [],
      totalPivots: 0,
      secPerPivot: null,
      severity: "none",
      score: null,
      skipReason: detections.length === 0
        ? "no_punches"
        : `too_few_punches (need ≥${cfg.minPunches}, got ${detections.length})`,
      roundSec,
    };
  }

  const annotated = [];
  let anchor = null;
  let maxPos = 0, minNeg = 0;
  let pivotsSoFar = 0;
  let lastPivotTime = null;
  let punchesSince = 0;

  for (let i = 0; i < detections.length; i++) {
    const d = detections[i];
    const samp = punchSample(state.pose, d);
    const t = (d.timestamp != null)
      ? d.timestamp
      : (d.start_frame ?? 0) / fps;
    const ann = {
      idx: i,
      det: d,
      timestamp: t,
      angle: samp.angle,
      stance: samp.stance,
      used: samp.used,
      skip: !samp.used,
      anchor,
      dAngle: null,
      maxPos: maxPos,
      minNeg: minNeg,
      swing: maxPos - minNeg,
      fired: false,
      pivotsSoFar,
      punchesSince,
      secSinceLast: lastPivotTime == null ? null : (t - lastPivotTime),
      secToNext: null,   // filled in reverse pass
      degToPos: null,
      degToNeg: null,
    };
    if (samp.used) {
      if (anchor == null) {
        anchor = samp.angle;
        ann.anchor = anchor;
        ann.dAngle = 0;
      } else {
        const dd = signedDiff(samp.angle, anchor);
        if (dd > maxPos) maxPos = dd;
        if (dd < minNeg) minNeg = dd;
        ann.anchor = anchor;
        ann.dAngle = dd;
        ann.maxPos = maxPos;
        ann.minNeg = minNeg;
        ann.swing = maxPos - minNeg;
        if (maxPos - minNeg >= cfg.swingDeg) {
          ann.fired = true;
          pivotsSoFar += 1;
          lastPivotTime = t;
          anchor = samp.angle;
          maxPos = 0; minNeg = 0;
        }
      }
    }
    ann.pivotsSoFar = pivotsSoFar;
    // degToPos: how much further right (CCW positive) the boxer needs
    // to push their current d-from-anchor to fire. Mirror for left.
    if (ann.dAngle != null) {
      ann.degToPos = Math.max(0, ann.minNeg + cfg.swingDeg - ann.dAngle);
      ann.degToNeg = Math.max(0, ann.dAngle - (ann.maxPos - cfg.swingDeg));
    }
    if (ann.fired) punchesSince = 0; else punchesSince += 1;
    annotated.push(ann);
  }
  // Reverse pass: time until next pivot.
  let nextT = null;
  for (let i = annotated.length - 1; i >= 0; i--) {
    annotated[i].secToNext = nextT == null ? null : (nextT - annotated[i].timestamp);
    if (annotated[i].fired) nextT = annotated[i].timestamp;
  }

  const totalPivots = pivotsSoFar;
  const secPerPivot = totalPivots > 0 ? roundSec / totalPivots : roundSec;
  let severity = "none";
  if (totalPivots === 0) severity = "severe";
  else if (secPerPivot >= cfg.sevSevere) severity = "severe";
  else if (secPerPivot >= cfg.sevModerate) severity = "moderate";
  else if (secPerPivot >= cfg.sevMild) severity = "mild";

  return {
    sourceKind: source?.kind || "none",
    detections,
    annotated,
    totalPivots,
    secPerPivot,
    severity,
    score: pivotScore(secPerPivot, cfg),
    skipReason: null,
    roundSec,
  };
}

function activeIdx(annotated, frame) {
  if (!annotated.length) return null;
  let lo = 0, hi = annotated.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if ((annotated[mid].det.start_frame ?? 0) <= frame) {
      ans = mid; lo = mid + 1;
    } else hi = mid - 1;
  }
  return ans >= 0 ? ans : null;
}

function fmtDeg(v) {
  if (v == null) return "—";
  const sign = v > 0 ? "+" : "";
  return sign + v.toFixed(0) + "°";
}
function fmtSec(v) {
  if (v == null) return "—";
  return v.toFixed(1) + " s";
}

let host;
let cfg = { ...DEFAULTS };
let signals = null;
let lastPose = null;

export const AngleChangeRule = {
  id: "angle_change",
  label: "Angle change (pivot rate)",

  mount(_host, state) {
    host = _host;
    host.innerHTML = template();
    signals = compute(state, cfg);
    lastPose = state.pose;
    wireSliders(state);
    renderHeader(state);
    renderTable(state);
    renderLive(state);
  },

  update(state) {
    if (state.pose !== lastPose) {
      signals = compute(state, cfg);
      lastPose = state.pose;
      renderHeader(state);
      renderTable(state);
    }
    renderLive(state);
    // Active-row highlight.
    const ai = activeIdx(signals.annotated, state.frame);
    host.querySelectorAll("tr[data-ac-idx]").forEach(tr => {
      tr.classList.toggle("active",
        parseInt(tr.getAttribute("data-ac-idx"), 10) === ai);
    });
  },

  draw(ctx, state) {
    if (!signals?.annotated?.length) return;
    const ai = activeIdx(signals.annotated, state.frame);
    if (ai == null) return;
    const ann = signals.annotated[ai];
    if (!ann.used || ann.anchor == null) return;

    const s = state.renderScale || 1;
    // Anchor at the hip midpoint of the active frame (or middle of punch).
    const f = Math.max(0,
      Math.min(state.pose.n_frames - 1, state.frame)
    );
    const lhx = state.pose.skeleton[(f * 17 + J.L_HIP) * 2];
    const lhy = state.pose.skeleton[(f * 17 + J.L_HIP) * 2 + 1];
    const rhx = state.pose.skeleton[(f * 17 + J.R_HIP) * 2];
    const rhy = state.pose.skeleton[(f * 17 + J.R_HIP) * 2 + 1];
    if (![lhx, lhy, rhx, rhy].every(Number.isFinite)) return;
    const hx = 0.5 * (lhx + rhx), hy = 0.5 * (lhy + rhy);

    // Convert facing-angle (labeler convention, +x = right of boxer) to
    // image-space angle (y down). Matches orientation_lens.js.
    const toImg = gt => (90 - gt) * Math.PI / 180;
    const len = 80 * s;

    ctx.save();

    // Open swing range as a filled wedge between (anchor + minNeg) and
    // (anchor + maxPos). Shows the user "this is the territory the
    // boxer has visited since the last pivot."
    if (ann.swing > 0) {
      const a1 = toImg(ann.anchor + ann.minNeg);
      const a2 = toImg(ann.anchor + ann.maxPos);
      ctx.fillStyle = COLORS.range;
      ctx.beginPath();
      ctx.moveTo(hx, hy);
      // arc from a1 to a2 (image-y-down so we sweep accordingly).
      ctx.arc(hx, hy, len, Math.min(a1, a2), Math.max(a1, a2));
      ctx.closePath();
      ctx.fill();
    }

    // Anchor direction (grey, dashed).
    drawArrow(ctx, hx, hy, toImg(ann.anchor), len, COLORS.anchor, 2.5 * s, [5 * s, 4 * s]);

    // Current facing (cyan or green if this punch fired).
    const currentColor = ann.fired ? COLORS.fired : COLORS.current;
    drawArrow(ctx, hx, hy, toImg(ann.angle), len, currentColor, 3.5 * s, []);

    // Banner.
    const bannerLines = [
      `${ann.fired ? "★ FIRED" : "open"} · ${ann.swing.toFixed(0)}° / ${cfg.swingDeg}°`,
      `to fire: +${(ann.degToPos ?? 0).toFixed(0)}° R · −${(ann.degToNeg ?? 0).toFixed(0)}° L`,
    ];
    drawBanner(ctx, bannerLines, ann.fired ? COLORS.fired : COLORS.warn, s);

    ctx.restore();
  },
};

function drawArrow(ctx, x, y, angleRad, len, color, width, dash) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  if (dash.length) ctx.setLineDash(dash);
  const x1 = x + len * Math.cos(angleRad);
  const y1 = y + len * Math.sin(angleRad);
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x1, y1); ctx.stroke();
  ctx.setLineDash([]);
  // Arrowhead.
  const a = 0.45, size = Math.max(8, width * 3);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - size * Math.cos(angleRad - a), y1 - size * Math.sin(angleRad - a));
  ctx.lineTo(x1 - size * Math.cos(angleRad + a), y1 - size * Math.sin(angleRad + a));
  ctx.closePath(); ctx.fill();
  ctx.restore();
}

function drawBanner(ctx, lines, color, scale) {
  ctx.save();
  const fontPx = Math.round(13 * scale);
  ctx.font = `bold ${fontPx}px ui-monospace, "SF Mono", monospace`;
  const pad = 6 * scale;
  const lineGap = 2 * scale;
  const tws = lines.map(l => ctx.measureText(l).width);
  const tw = Math.max(...tws);
  const boxH = lines.length * fontPx + (lines.length - 1) * lineGap + pad * 2;
  ctx.fillStyle = "rgba(0,0,0,0.72)";
  ctx.fillRect(8 * scale, 8 * scale, tw + pad * 2, boxH);
  ctx.fillStyle = color;
  let y0 = 8 * scale + pad + fontPx - 2;
  for (const line of lines) {
    ctx.fillText(line, 8 * scale + pad, y0);
    y0 += fontPx + lineGap;
  }
  ctx.restore();
}

function template() {
  return `
    <h2>Angle change</h2>
    <p class="hint">
      Samples boxer facing direction at every punch (feet planted),
      then runs a ratchet: fires a pivot when the swing from the last
      anchor reaches <code>swing_degrees</code>. Reports <b>sec/pivot</b>
      — higher = more static. Coaching rule lives at
      <code>cornerman_rules/rules/pivot_rate.py</code>.
    </p>
    <p class="hint" style="font-size:11px; color:#f5b945">
      Note: this lens reuses the ankle-arrow facing model from the
      Orientation lens, not the LogReg classifier used by the Python
      rule. The ratchet logic is identical; per-punch sample angles
      may differ slightly from what the rule engine would emit.
    </p>

    <div id="ac-source-pill" class="hint" style="margin-bottom:8px"></div>

    <h3>Round verdict</h3>
    <div class="metric-grid">
      <div class="metric"><div class="metric-label">pivots</div><div class="metric-val" id="ac-pivots">—</div></div>
      <div class="metric"><div class="metric-label">sec / pivot</div><div class="metric-val" id="ac-spp">—</div></div>
      <div class="metric"><div class="metric-label">severity</div><div class="metric-val" id="ac-sev">—</div></div>
      <div class="metric"><div class="metric-label">score</div><div class="metric-val" id="ac-score">—</div></div>
      <div class="metric"><div class="metric-label">round</div><div class="metric-val" id="ac-round">—</div></div>
    </div>

    <h3>Current frame</h3>
    <div id="ac-live" class="hint" style="line-height:1.55; background:#0f3460; padding:8px 10px; border-radius:6px; margin-bottom:8px">
      <em>no punches yet</em>
    </div>

    <h3>Punches</h3>
    <div id="ac-punch-table"></div>

    <h3>Thresholds</h3>
    <label class="slider">
      <span>swing_degrees = <output id="ac-swing-out">${cfg.swingDeg}°</output></span>
      <input type="range" id="ac-swing" min="45" max="180" step="5" value="${cfg.swingDeg}">
      <span class="muted small">How far the boxer's facing must swing (peak-to-peak from anchor) to fire one pivot.</span>
    </label>
    <label class="slider">
      <span>min_punches = <output id="ac-minpunches-out">${cfg.minPunches}</output></span>
      <input type="range" id="ac-minpunches" min="1" max="20" step="1" value="${cfg.minPunches}">
      <span class="muted small">Below this many punches we skip the rule (defensive round).</span>
    </label>

    <h3>Mistake scoring</h3>
    <label class="slider">
      <span>starts being a mistake = <output id="ac-sstart-out">${cfg.scoreStart}s</output> / pivot</span>
      <input type="range" id="ac-sstart" min="4" max="40" step="1" value="${cfg.scoreStart}">
      <span class="muted small">sec/pivot at or below this = fine (score 0).</span>
    </label>
    <label class="slider">
      <span>steepest at = <output id="ac-smid-out">${cfg.scoreMid}s</output> / pivot</span>
      <input type="range" id="ac-smid" min="8" max="90" step="1" value="${cfg.scoreMid}">
      <span class="muted small">where the curve is steepest (score 50).</span>
    </label>
    <label class="slider">
      <span>full mistake = <output id="ac-ssat-out">${cfg.scoreSat}s</output> / pivot</span>
      <input type="range" id="ac-ssat" min="20" max="120" step="5" value="${cfg.scoreSat}">
      <span class="muted small">sec/pivot at or above this = 100 (too stationary).</span>
    </label>
    <label class="slider">
      <span>steepness = <output id="ac-sk-out">${cfg.scoreK.toFixed(1)}</output></span>
      <input type="range" id="ac-sk" min="2" max="20" step="0.5" value="${cfg.scoreK}">
      <span class="muted small">sharper mid cliff on the rate.</span>
    </label>
  `;
}

function wireSliders(state) {
  const sw = host.querySelector("#ac-swing");
  const swOut = host.querySelector("#ac-swing-out");
  sw.addEventListener("input", () => {
    cfg.swingDeg = parseInt(sw.value, 10);
    swOut.textContent = `${cfg.swingDeg}°`;
    signals = compute(state, cfg);
    renderHeader(state);
    renderTable(state);
    renderLive(state);
  });
  const mp = host.querySelector("#ac-minpunches");
  const mpOut = host.querySelector("#ac-minpunches-out");
  mp.addEventListener("input", () => {
    cfg.minPunches = parseInt(mp.value, 10);
    mpOut.textContent = `${cfg.minPunches}`;
    signals = compute(state, cfg);
    renderHeader(state);
    renderTable(state);
    renderLive(state);
  });
  // Score-curve sliders — only the sec/pivot → 0–100 mapping moves, so just
  // recompute (cheap) + re-render the header.
  const wireScore = (id, out, key, fmt) => {
    const s = host.querySelector(id), o = host.querySelector(out);
    s.addEventListener("input", () => {
      cfg[key] = parseFloat(s.value);
      o.textContent = fmt(cfg[key]);
      signals = compute(state, cfg);
      renderHeader(state);
    });
  };
  wireScore("#ac-sstart", "#ac-sstart-out", "scoreStart", v => `${v}s`);
  wireScore("#ac-smid", "#ac-smid-out", "scoreMid", v => `${v}s`);
  wireScore("#ac-ssat", "#ac-ssat-out", "scoreSat", v => `${v}s`);
  wireScore("#ac-sk", "#ac-sk-out", "scoreK", v => v.toFixed(1));
}

function renderHeader(state) {
  const pill = host.querySelector("#ac-source-pill");
  if (signals.sourceKind === "labels") {
    pill.innerHTML = `<span class="role-lead">Ground truth</span> · ${signals.detections.length} labels`;
  } else if (signals.sourceKind === "punches") {
    pill.innerHTML = `<span class="role-rear">ST-GCN punches</span> · ${signals.detections.length} detected`;
  } else {
    pill.innerHTML = `<span class="muted">No punches loaded — load a labels sheet or <code>*_punches.json</code> to score.</span>`;
  }
  host.querySelector("#ac-pivots").textContent = signals.totalPivots;
  host.querySelector("#ac-spp").textContent =
    signals.secPerPivot != null ? `${signals.secPerPivot.toFixed(1)} s` : "—";
  const sevEl = host.querySelector("#ac-sev");
  sevEl.textContent = signals.severity;
  sevEl.style.color = ({
    severe: COLORS.bad, moderate: "#f97316", mild: COLORS.warn, none: COLORS.fired,
  })[signals.severity] || "#aaa";
  const scEl = host.querySelector("#ac-score");
  scEl.textContent = signals.score == null ? "—" : signals.score;
  scEl.style.color = signals.score == null ? "#aaa"
    : signals.score >= 70 ? COLORS.bad
    : signals.score >= 40 ? "#f97316"
    : signals.score >= 15 ? COLORS.warn : COLORS.fired;
  host.querySelector("#ac-round").textContent = `${signals.roundSec.toFixed(1)} s`;
}

function renderTable(state) {
  const tbody = host.querySelector("#ac-punch-table");
  if (signals.skipReason) {
    tbody.innerHTML = `<p class="hint muted">Skipped: <code>${signals.skipReason}</code></p>`;
    return;
  }
  const rows = signals.annotated.map(a => {
    const cls = a.fired ? "scored" : (a.used ? "unscored" : "skipped");
    const toFireHtml = a.dAngle == null
      ? "—"
      : `<span class="good">+${(a.degToPos ?? 0).toFixed(0)}°</span>/<span class="warn">−${(a.degToNeg ?? 0).toFixed(0)}°</span>`;
    const angleTxt = a.angle == null ? "—" : `${a.angle.toFixed(0)}°`;
    const seek = a.det.start_frame ?? 0;
    return `
      <tr class="${cls}" data-seek="${seek}" data-ac-idx="${a.idx}">
        <td>${a.timestamp.toFixed(2)}s</td>
        <td>${(a.det.punch_type || "?").replace(/_/g, " ")}</td>
        <td>${a.det.hand || "?"}·${a.stance || "?"}</td>
        <td>${angleTxt}</td>
        <td>${fmtDeg(a.dAngle)}</td>
        <td>${a.swing.toFixed(0)}°</td>
        <td style="font-size:10px;">${toFireHtml}</td>
        <td>${a.fired ? "★" : ""}</td>
      </tr>`;
  }).join("");
  tbody.innerHTML = `
    <table class="sps-tbl">
      <thead><tr>
        <th>t</th><th>Type</th><th>Hand</th><th>Angle</th>
        <th>Δanchor</th><th>Swing</th><th>To fire (R/L)</th><th>Fired</th>
      </tr></thead>
      <tbody>${rows || `<tr><td colspan="8" class="muted">no punches</td></tr>`}</tbody>
    </table>`;
  tbody.querySelectorAll("tr[data-seek]").forEach(tr => {
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => {
      const f = parseInt(tr.getAttribute("data-seek"), 10);
      const slider = document.getElementById("scrubber");
      if (slider) { slider.value = f; slider.dispatchEvent(new Event("input")); }
    });
  });
}

function renderLive(state) {
  const el = host.querySelector("#ac-live");
  if (!el) return;
  if (!signals.annotated.length) {
    el.innerHTML = signals.skipReason
      ? `<em>skipped: ${signals.skipReason}</em>`
      : "<em>no punches yet</em>";
    return;
  }
  const ai = activeIdx(signals.annotated, state.frame);
  if (ai == null) {
    el.innerHTML = "<em>no punch before this frame</em>";
    return;
  }
  const a = signals.annotated[ai];
  const curAngle = a.angle == null ? "—" : `${a.angle.toFixed(0)}°`;
  const firedLine = a.fired
    ? `<div style="color:${COLORS.fired};font-weight:bold">★ This punch fired pivot ${a.pivotsSoFar}</div>`
    : "";
  el.innerHTML = `
    <div><b>Pivot ${a.pivotsSoFar} of ${signals.totalPivots}</b> · punch ${ai + 1}/${signals.annotated.length}</div>
    <div>Anchor ${fmtDeg(a.anchor)} · current ${curAngle} (Δ ${fmtDeg(a.dAngle)})</div>
    <div>Swing since anchor: <b>${fmtDeg(a.minNeg)} … ${fmtDeg(a.maxPos)}</b> = ${a.swing.toFixed(0)}° / ${cfg.swingDeg}°</div>
    <div>To fire: <span style="color:${COLORS.fired}">+${(a.degToPos ?? 0).toFixed(0)}°</span> right or <span style="color:${COLORS.warn}">−${(a.degToNeg ?? 0).toFixed(0)}°</span> left</div>
    <div>Punches since last pivot: ${a.punchesSince} · since: ${fmtSec(a.secSinceLast)} · till next: ${fmtSec(a.secToNext)}</div>
    ${firedLine}
  `;
}
