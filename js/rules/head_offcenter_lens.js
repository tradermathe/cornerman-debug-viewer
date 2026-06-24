// Head-off-the-center-line lens (straight punches) — BlazePose-33.
//
// The coaching rule: when you punch straight, your head should come OFF the
// center line so you're not sat there to be countered. This lens measures, for
// every straight punch, how far the head travels off the line — in torso heights.
//
// Why measure RELATIVE to the body, not in raw image x: if we just watched the
// head's screen-x, a fighter who steps or circles would look like they moved the
// head off line when the whole body just translated. So we anchor to the body
// center: head_offset = head_x − body_center_x. Whole-body movement cancels;
// only the head leaving the column registers.
//
// Why a VERTICAL reference, not the shoulder→hip spine axis: a slip IS a torso
// bend, so the spine tilts toward the slip. Referencing the tilted spine would
// cancel the very movement we want to catch. So body_center_x is a vertical line
// through the body center; a side-bend slip then correctly reads as the head
// leaving that line.
//
// Head point: x = midpoint of the left/right EXTENT of the visible head
// landmarks (nose, eyes, ears, mouth) — the center of the head's horizontal
// silhouette, which holds up under head rotation (frontal it's ear-to-ear, in
// profile ear-to-nose; both straddle the head center). These landmarks only
// exist on the full BlazePose-33 cache (state.blaze33), NOT the COCO-17 remap,
// so this lens reads blaze33 directly and gates joints on per-joint `visibility`.
//
// Center-line reference = the HIP center (hip_center_x). The hips stay planted
// through a slip, whereas the shoulder/hip midpoint drifts toward the head as you
// bend at the waist and so under-reads the slip — hips is the fairer benchmark.
// The mid line (midpoint of shoulder_center_x and hip_center_x) is kept only as a
// faint on-screen reference.
//
// AXIALITY GATE: this lens reads lateral (screen-x) head movement, which is only
// trustworthy when the boxer FACES the camera. A punch toward the camera (axial)
// means the boxer is square to it, so the head slip is in-plane; a side-on punch
// (low axiality) throws that slip into depth where we can't see it. So we join
// each punch to the trained axiality model by punch_uuid and KEEP only straights
// within ~45° of the camera axis (axiality >= cos45° ≈ 0.707), skipping side-on
// ones — the OPPOSITE direction to hit_height/arm_extension, which keep side-on
// punches because they read fist HEIGHT (foreshortens head-on, not lateral).
// Falls back to on-device per-punch axiality; missing axiality fails closed.
//
// Normalization: a body straight drops the level and FORESHORTENS the torso in
// the image, so we normalize by a STABLE per-round torso height (median over
// frames with all four torso joints visible), not the bent-over instantaneous one.
//
// Per straight punch [start,end]:
//   peak frame  = max |wrist − same-side shoulder| in the window (contact moment)
//   start frame = first frame in the window with a usable head + torso
//   off_center_at_peak = head_offset(peak) / torsoBaseline
//   head_travel        = (head_offset(peak) − head_offset(start)) / torsoBaseline
// Sign is image-space (+ = head right of center). The fault we flag is a SMALL
// |head_travel| — the head stayed on the line through the punch.

import { ensureAxialityModel, axialityForPunch } from "./axiality_model.js";
import { activeDetections, isStraightType } from "./_detections.js";
import { SCHEMAS } from "./_skeleton_schemas.js";
import { qualityOf, qualityColor, qualityBand } from "./_score.js";

// blaze33 channels: 0 x  1 y  2 z  3 xw  4 yw  5 zw  6 visibility  7 presence
const CH = 8, X = 0, Y = 1, VIS = 6, NJ = 33;

// BlazePose-33 joint indices.
const L_SHOULDER = 11, R_SHOULDER = 12;
const L_HIP = 23, R_HIP = 24;
// All head landmarks: nose(0), eye inner/main/outer (1-6), ears(7,8), mouth(9,10).
const HEAD_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const NAMES = SCHEMAS.blazepose33.names;   // joint index → landmark name (for labels)

const MIN_TORSO_PX = 5;
const DEFAULTS = {
  axialityMin:    Math.SQRT1_2,  // ≈0.7071 = cos45°; KEEP straights within 45° of the
                                 // camera axis (toward camera) — skip side-on ones
  scoreTarget:    0.25,          // head off-center (torso) for full marks; 0 = on the line
  scoreSteepness: 10,            // sigmoid k (shared shape with hit_height/arm_extension)
  minVis:         0.30,          // per-joint BlazePose visibility gate
};

const COLOR_CENTER = "#ffd24a";   // body-center vertical line (mid anchor)
const COLOR_HIP    = "#3ad9e0";   // hips-only vertical line
const COLOR_HEAD   = "#c08bff";   // head point / extremes

let host = null;
let cfg = { ...DEFAULTS };
let calib = null;            // { torsoBaseline, nValid, total }
let classified = [];         // per straight-punch results
let lastKey = null, lastBlaze = null, lastDets = null;

function b33(state) { return state.blaze33 || null; }
function cacheKey(state) { return `${state.cacheBasename || ""}__r${state.cacheRound ?? "?"}`; }

// --- blaze33 ↔ primary-timeline frame mapping -----------------------------
// Detections + state.frame live in the primary pose timeline; blaze33 is usually
// the same extraction (frame-aligned) but guard with the inspector's alignment
// check and fall back to time/PTS.
function blazeAligned(b, state) {
  return b.n_frames === state.n_frames
    && Math.abs((b.fps || 0) - (state.fps || 0)) < 0.01
    && Math.abs((b.start_sec || 0) - (state.start_sec || 0)) < 1e-3;
}
function blazeFrame(b, state, primaryFrame) {
  if (blazeAligned(b, state)) return Math.min(Math.max(primaryFrame, 0), b.n_frames - 1);
  const t = (state.start_sec || 0) + primaryFrame / (state.fps || 30);
  if (b.pts && b.pts.length) {
    let lo = 0, hi = b.pts.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (b.pts[m] < t) lo = m + 1; else hi = m; }
    if (lo > 0 && Math.abs(b.pts[lo - 1] - t) <= Math.abs(b.pts[lo] - t)) lo--;
    return lo;
  }
  const f = Math.round((t - (b.start_sec || 0)) * (b.fps || 30));
  return (f >= 0 && f < b.n_frames) ? f : null;
}

// --- geometry from blaze33 ------------------------------------------------
function jointAt(b, base, j, w, h) {
  const o = base + j * CH;
  if (!(b.data[o + VIS] >= cfg.minVis)) return null;
  const x = b.data[o + X] * w, y = b.data[o + Y] * h;
  return (Number.isFinite(x) && Number.isFinite(y)) ? { x, y } : null;
}

// Head x = midpoint of the horizontal EXTENT of the visible head landmarks
// ((leftmost + rightmost) / 2), not the centroid. This tracks the center of the
// head's silhouette through rotation: frontal the extremes are ear-to-ear, in
// profile they become ear-to-nose — both straddle the head roughly symmetrically,
// so the midpoint stays near the true head center even when the face has turned
// (and it isn't dragged sideways by the front-face cluster the way a centroid is).
// y is the mean of the visible points — only used to place the marker; the metric
// uses x. minX/maxX are returned so the overlay can show the span.
function headPoint(b, base, w, h) {
  let minX = Infinity, maxX = -Infinity, minJ = -1, maxJ = -1, minP = null, maxP = null, sy = 0, n = 0;
  for (const j of HEAD_JOINTS) {
    const p = jointAt(b, base, j, w, h);
    if (!p) continue;
    if (p.x < minX) { minX = p.x; minJ = j; minP = p; }
    if (p.x > maxX) { maxX = p.x; maxJ = j; maxP = p; }
    sy += p.y; n++;
  }
  return n ? { x: (minX + maxX) / 2, y: sy / n, n, minX, maxX, minJ, maxJ, minP, maxP } : null;
}

function torsoAt(b, base, w, h) {
  const ls = jointAt(b, base, L_SHOULDER, w, h);
  const rs = jointAt(b, base, R_SHOULDER, w, h);
  const lh = jointAt(b, base, L_HIP, w, h);
  const rh = jointAt(b, base, R_HIP, w, h);
  if (!ls || !rs || !lh || !rh) return null;
  const S = { x: (ls.x + rs.x) / 2, y: (ls.y + rs.y) / 2 };
  const P = { x: (lh.x + rh.x) / 2, y: (lh.y + rh.y) / 2 };
  const length = Math.hypot(S.x - P.x, S.y - P.y);
  if (length < MIN_TORSO_PX) return null;
  return { S, P, length, midX: (S.x + P.x) / 2, hipX: P.x };
}

// head_offset against both anchors at one blaze frame.
function offsetAt(b, base, w, h) {
  const head = headPoint(b, base, w, h);
  const torso = torsoAt(b, base, w, h);
  if (!head || !torso) return null;
  return { head, torso, offMid: head.x - torso.midX, offHip: head.x - torso.hipX };
}

// Stable per-round torso height: median |shoulder_mid − hip_mid| over frames with
// all four torso joints visible. Median (not instantaneous) so a deep body-shot
// crouch doesn't shrink the unit we divide by.
function calibrate(b, w, h) {
  const lens = [];
  for (let f = 0; f < b.n_frames; f++) {
    const t = torsoAt(b, f * NJ * CH, w, h);
    if (t) lens.push(t.length);
  }
  if (!lens.length) return { torsoBaseline: null, nValid: 0, total: b.n_frames };
  lens.sort((a, c) => a - c);
  return { torsoBaseline: lens[Math.floor(lens.length / 2)], nValid: lens.length, total: b.n_frames };
}

// --- punches --------------------------------------------------------------
function isBody(d) { return /_body$/.test(String(d.punch_type || "").toLowerCase()); }

function classifyPunch(b, state, d, idx, T, w, h) {
  const base = {
    idx, det: d, body: isBody(d),
    timestamp: d.timestamp, punch_type: d.punch_type || "?",
  };

  // Axiality gate — keep only punches toward the camera (boxer facing it, so the
  // lateral head read is in-plane); skip side-on straights > 45° off the axis.
  const ax = axialityForPunch(d.punch_uuid)?.predAxiality ?? d.axiality;
  if (ax == null || !Number.isFinite(ax)) return { ...base, axiality: null, skip: "no-ax" };
  if (ax < cfg.axialityMin) return { ...base, axiality: ax, skip: "sideways" };

  // Walk the whole punch window, recording head off-center (HIP anchor) per frame.
  // The score uses the FURTHEST the head gets off the line anywhere in the window
  // (its biggest excursion) — that frame is the "peak" we mark and seek to.
  const sP = Math.max(0, d.start_frame), eP = d.end_frame;
  const series = [];                 // { frame (primary), dist (signed, torso) }
  let peakPrimary = -1, peakDist = 0, peakAbs = -1, peakHead = null, startDist = null;
  for (let pf = sP; pf <= eP; pf++) {
    const bf = blazeFrame(b, state, pf);
    if (bf == null) continue;
    const o = offsetAt(b, bf * NJ * CH, w, h);
    if (!o) continue;
    const dist = o.offHip / T;
    series.push({ frame: pf, dist });
    if (startDist == null) startDist = dist;
    if (Math.abs(dist) > peakAbs) { peakAbs = Math.abs(dist); peakDist = dist; peakPrimary = pf; peakHead = o.head; }
  }
  if (peakPrimary < 0) return { ...base, axiality: ax, skip: "no head/torso" };

  const sc = scoreOffCenter(peakDist);   // scored on the furthest off-center in the window
  return {
    ...base, axiality: ax, skip: "",
    series, startDist,
    peakPrimary, peakDist,
    score: sc.quality, mistake: sc.mistake,
    headAtPeak: peakHead,
  };
}

function computePunches(state) {
  const b = b33(state);
  if (!b || !calib?.torsoBaseline) return [];
  const dets = (activeDetections(state) || []).filter(d => isStraightType(d.punch_type));
  return dets.map((d, i) => classifyPunch(b, state, d, i, calib.torsoBaseline, b.width, b.height));
}

function recompute(state) {
  const b = b33(state);
  if (!b) { calib = null; classified = []; return; }
  calib = calibrate(b, b.width, b.height);
  classified = computePunches(state);
}

function refreshIfNeeded(state) {
  const b = b33(state);
  const k = cacheKey(state);
  const dets = activeDetections(state);
  if (k !== lastKey || b !== lastBlaze || dets !== lastDets) {
    lastKey = k; lastBlaze = b; lastDets = dets;
    recompute(state);
  }
}

function seekTo(f) {
  const slider = document.getElementById("scrubber");
  if (!slider) return;
  slider.value = f;
  slider.dispatchEvent(new Event("input"));
}

function classifiedAtFrame(frame) {
  for (const c of classified) {
    if (frame >= c.det.start_frame && frame <= c.det.end_frame) return c;
  }
  return null;
}

// 0–100 quality from how far the head is off the line at the punch peak. 0 = head
// on the center line (worst); ramps via the house sigmoid (same shape as
// hit_height/arm_extension) to 100 at scoreTarget — flat-bad near the line, steep
// in the middle, flat-perfect past target. We keep the MISTAKE (0 = best) too,
// since _score.js's rollup + bands work in mistake space.
function sigmoid01(x, k) {
  const L = (t) => 1 / (1 + Math.exp(-k * (t - 0.5)));
  return (L(x) - L(0)) / (L(1) - L(0));
}
function scoreOffCenter(offTorso) {
  const x = Math.max(0, Math.min(1, Math.abs(offTorso) / cfg.scoreTarget));
  const quality = 100 * sigmoid01(x, cfg.scoreSteepness);
  return { quality, mistake: 100 - quality };
}
function fmtSigned(v, d = 2) { return v == null || !Number.isFinite(v) ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(d)}`; }
function setText(id, html) { const el = host?.querySelector("#" + id); if (el) el.innerHTML = html; }

const SKIP_LABEL = {
  "sideways": "side-on (skipped)",
  "no-ax":    "no axiality",
  "no head/torso": "no head/torso",
};

// --- per-punch table (right panel) ----------------------------------------
function renderTable(state) {
  const tableEl = host?.querySelector("#hoc-table");
  const sumEl = host?.querySelector("#hoc-summary");
  if (!tableEl || !sumEl) return;

  if (!calib?.torsoBaseline) {
    sumEl.innerHTML = `<span class="bad">No usable torso frames — can't normalize.</span>`;
    tableEl.innerHTML = "";
    return;
  }
  if (!classified.length) {
    sumEl.textContent = "No straight-punch labels loaded for this round (jab/cross, head or body).";
    tableEl.innerHTML = "";
    return;
  }

  const scored = classified.filter(c => !c.skip);
  const nSkip = classified.filter(c => c.skip === "sideways" || c.skip === "no-ax").length;

  // Round score = worst-decile mean of per-punch mistakes (surface the punches
  // where the head stayed on the line), flipped to quality + band like hit_height.
  let roundPart = "";
  if (scored.length) {
    const mis = scored.map(c => c.mistake).sort((a, b) => b - a);   // worst (highest mistake) first
    const k = Math.max(1, Math.round(mis.length / 10));
    const roundMistake = mis.slice(0, k).reduce((a, b) => a + b, 0) / k;
    const Q = qualityOf(roundMistake);
    roundPart = `round <b style="color:${Q.color}">${Q.q.toFixed(0)} ${Q.label.toUpperCase()}</b> (worst ${k} of ${scored.length}) · `;
  }
  sumEl.innerHTML = scored.length
    ? roundPart + `${scored.length} straights scored` + (nSkip ? ` · ${nSkip} side-on (skipped)` : "")
    : `No straights scored${nSkip ? ` — ${nSkip} skipped as side-on / no-axiality (load predictions_axiality_*.json)` : ""}.`;

  const rows = classified.map(c => {
    const t = Number.isFinite(c.timestamp) ? c.timestamp.toFixed(2) + "s" : "—";
    const seek = c.peakPrimary ?? c.det.start_frame;
    if (c.skip) {
      return `<tr data-frame="${seek}" style="cursor:pointer">
        <td>${c.idx + 1}</td><td>${t}</td><td>${c.punch_type}</td>
        <td colspan="2" class="muted">${SKIP_LABEL[c.skip] || c.skip}</td></tr>`;
    }
    const q = c.score, col = qualityColor(q);
    return `<tr data-frame="${seek}" style="cursor:pointer">
      <td>${c.idx + 1}</td><td>${t}</td><td>${c.punch_type}${c.body ? ' <span class="muted">b</span>' : ''}</td>
      <td class="num" style="color:${col};font-weight:700">${q.toFixed(0)}</td>
      <td class="num">${fmtSigned(c.peakDist)}</td></tr>`;
  }).join("");

  tableEl.innerHTML = `
    <table class="joint-table">
      <thead><tr><th>#</th><th>t</th><th>type</th><th>score</th><th>peak·dist</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

export const HeadOffCenterLensRule = {
  id: "head_offcenter",
  label: "Head off center line (straights)",

  // Needs the full 33-joint cache — mouth/eye landmarks aren't in the COCO-17 remap.
  requires(slot) { return !!(slot && slot.blazepose); },

  // Hide the base COCO-17 head joints (nose + eyes + ears) so the head shows ONLY
  // the two extreme landmarks this lens actually uses.
  skeletonStyle() { return { hideJoints: new Set([0, 1, 2, 3, 4]) }; },

  mount(_host, state) {
    host = _host;
    host.innerHTML = `
      <h2>Head off the center line</h2>
      <p class="hint">
        Per straight punch (jab/cross, head or body): how far the head moves off
        the body's center line, in torso heights. Head = midpoint of the left/right
        extent of the visible head landmarks on the <b>BlazePose-33</b> cache (holds
        up when the head turns); center line = a vertical through the shoulder/hip
        mid. Measured relative to the body so stepping or
        circling doesn't count. We measure against the <b>hip</b> center line —
        the hips stay planted through a slip, while the shoulder/hip mid drifts
        toward the head and under-reads it (the yellow mid line is just a faint
        reference). The lateral read only works
        front-on, so only straights <b>toward the camera</b> (within ~45° of the
        camera axis) are scored — side-on punches are gated out by axiality.
        Each straight is <b>scored 0–100</b> by how far the head gets off the line
        during the punch (0 = on the line, 100 at ≥ target), higher = better.
      </p>

      <h3>Live</h3>
      <div class="metric-grid">
        <div class="metric"><div class="metric-label">torso baseline</div><div class="metric-val" id="hoc-torso">—</div></div>
        <div class="metric"><div class="metric-label">distance · hip</div><div class="metric-val" id="hoc-offhip">—</div></div>
        <div class="metric"><div class="metric-label">mid (ref)</div><div class="metric-val" id="hoc-offmid">—</div></div>
      </div>
      <div id="hoc-extremes" class="hint" style="font-size:11px;margin:2px 0 6px">—</div>
      <div class="metric">
        <div class="metric-label">Active straight — score</div>
        <div class="metric-val" id="hoc-live-travel">—</div>
        <div class="metric-sub" id="hoc-live-sub"></div>
      </div>
      <div class="metric-label" style="margin-top:8px">Punch distance trace (torso vs frame)</div>
      <div id="hoc-graph" class="hint" style="margin-top:2px">—</div>

      <h3>Per-punch</h3>
      <div id="hoc-summary" class="hint"></div>
      <div id="hoc-table"></div>

      <h3>Tuning</h3>
      <label class="slider">
        <span>full score at off ≥ <output id="hoc-o1">${cfg.scoreTarget.toFixed(2)}</output> torso</span>
        <input type="range" id="hoc-s1" min="0.05" max="0.6" step="0.01" value="${cfg.scoreTarget}">
      </label>
      <label class="slider">
        <span>score steepness = <output id="hoc-o4">${cfg.scoreSteepness.toFixed(1)}</output></span>
        <input type="range" id="hoc-s4" min="2" max="20" step="0.5" value="${cfg.scoreSteepness}">
      </label>
      <label class="slider">
        <span>keep axiality ≥ <output id="hoc-o2">${cfg.axialityMin.toFixed(2)}</output> (toward camera)</span>
        <input type="range" id="hoc-s2" min="0" max="1" step="0.01" value="${cfg.axialityMin}">
      </label>
      <label class="slider">
        <span>min visibility = <output id="hoc-o3">${cfg.minVis.toFixed(2)}</output></span>
        <input type="range" id="hoc-s3" min="0" max="1" step="0.05" value="${cfg.minVis}">
      </label>
    `;

    const wire = (slider, out, key) => {
      const s = host.querySelector(slider), o = host.querySelector(out);
      s.addEventListener("input", () => {
        cfg[key] = parseFloat(s.value);
        o.textContent = cfg[key].toFixed(2);
        recompute(state);
        renderTable(state);
        window.__viewerRedraw?.();
      });
    };
    wire("#hoc-s1", "#hoc-o1", "scoreTarget");
    wire("#hoc-s4", "#hoc-o4", "scoreSteepness");
    wire("#hoc-s2", "#hoc-o2", "axialityMin");
    wire("#hoc-s3", "#hoc-o3", "minVis");

    // Click a row → seek to that punch's peak frame.
    host.addEventListener("click", e => {
      const tr = e.target.closest("tr[data-frame]");
      if (tr) { const f = parseInt(tr.dataset.frame, 10); if (Number.isFinite(f)) seekTo(f); }
    });

    // Axiality predictions load async; refresh the table + overlay when they land.
    ensureAxialityModel(state, () => { recompute(state); renderTable(state); window.__viewerRedraw?.(); });

    refreshIfNeeded(state);
    renderTable(state);
    renderLive(state);
  },

  update(state) {
    refreshIfNeeded(state);
    renderLive(state);
  },

  draw(ctx, state) { drawOverlay(ctx, state); },
};

function renderLive(state) {
    const b = b33(state);
    if (!b) {
      setText("hoc-torso", `<span class="bad">no blaze-33</span>`);
      setText("hoc-offmid", "—"); setText("hoc-offhip", "—");
      setText("hoc-live-travel", "—"); setText("hoc-live-sub", "");
      return;
    }
    const T = calib?.torsoBaseline;
    setText("hoc-torso", T ? `${T.toFixed(0)} px` : `<span class="bad">—</span>`);

    const f = blazeFrame(b, state, state.frame);
    const off = (T && f != null) ? offsetAt(b, f * NJ * CH, b.width, b.height) : null;
    setText("hoc-offmid", off ? `${fmtSigned(off.offMid / T)} T` : "—");
    setText("hoc-offhip", off ? `${fmtSigned(off.offHip / T)} T` : "—");
    setText("hoc-extremes", off
      ? `extremes: <code>${NAMES[off.head.minJ]}</code> ↔ <code>${NAMES[off.head.maxJ]}</code> · ${off.head.n}/11 head pts`
      : `<span class="muted">head not visible</span>`);

    const c = classifiedAtFrame(state.frame);
    if (!c) {
      setText("hoc-live-travel", "—");
      setText("hoc-live-sub", `<span class="muted">no straight at this frame</span>`);
    } else if (c.skip) {
      setText("hoc-live-travel", `<span class="muted">${SKIP_LABEL[c.skip] || c.skip}</span>`);
      setText("hoc-live-sub", `<code>${c.punch_type}</code>${c.axiality != null ? ` · axiality ${c.axiality.toFixed(2)}` : ""}`);
    } else {
      const col = qualityColor(c.score);
      setText("hoc-live-travel", `<span style="color:${col};font-weight:700">${c.score.toFixed(0)}</span> <span style="color:${col};font-size:12px;font-weight:700">${qualityBand(c.score).toUpperCase()}</span>`);
      setText("hoc-live-sub",
        `peak dist ${fmtSigned(c.peakDist)} T · ax ${c.axiality.toFixed(2)} · ` +
        (c.peakPrimary === state.frame ? `<span class="muted">at peak</span>` : `<a href="#" data-frame="${c.peakPrimary}">→ peak f${c.peakPrimary}</a>`));
    }
    setText("hoc-graph", graphSVG(c && !c.skip ? c : null, state.frame));
}

function drawOverlay(ctx, state) {
    const b = b33(state);
    if (!b || !calib?.torsoBaseline) return;
    const f = blazeFrame(b, state, state.frame);
    if (f == null) return;
    const w = b.width, h = b.height, s = state.renderScale || 1, H = ctx.canvas.height;
    const off = offsetAt(b, f * NJ * CH, w, h);
    if (!off) return;
    const { head, torso } = off;

    ctx.save();

    // Center line. We score against the HIP center (solid cyan) — the hips stay
    // planted through a slip; the shoulder/hip mid (dashed yellow) drifts toward
    // the head and under-reads it, so it's kept only as a faint reference.
    ctx.lineWidth = 1.5 * s;
    ctx.strokeStyle = COLOR_CENTER;
    ctx.globalAlpha = 0.7;
    ctx.setLineDash([6 * s, 5 * s]);
    ctx.beginPath(); ctx.moveTo(torso.midX, 0); ctx.lineTo(torso.midX, H); ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    ctx.lineWidth = 2 * s;
    ctx.strokeStyle = COLOR_HIP;
    ctx.beginPath(); ctx.moveTo(torso.hipX, 0); ctx.lineTo(torso.hipX, H); ctx.stroke();

    // Horizontal connector head → the hip center line (the offset we score).
    ctx.strokeStyle = COLOR_HEAD;
    ctx.globalAlpha = 0.6;
    ctx.lineWidth = 1.5 * s;
    ctx.beginPath(); ctx.moveTo(head.x, head.y); ctx.lineTo(torso.hipX, head.y); ctx.stroke();
    ctx.globalAlpha = 1;

    // The two extreme head landmarks (leftmost / rightmost) whose midpoint is the
    // head point — drawn at their real positions as open rings and labelled, so
    // it's clear exactly which landmarks are driving the read.
    if (head.minP && head.maxP) {
      ctx.strokeStyle = COLOR_HEAD;
      ctx.globalAlpha = 0.5; ctx.lineWidth = 1.5 * s;
      ctx.beginPath(); ctx.moveTo(head.minP.x, head.minP.y); ctx.lineTo(head.maxP.x, head.maxP.y); ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.font = `${Math.round(11 * s)}px ui-monospace, "SF Mono", monospace`;
      const tag = (p, name, toLeft) => {
        ctx.strokeStyle = COLOR_HEAD; ctx.lineWidth = 2 * s;
        ctx.beginPath(); ctx.arc(p.x, p.y, 5 * s, 0, Math.PI * 2); ctx.stroke();
        const tw = ctx.measureText(name).width;
        const lx = toLeft ? p.x - tw - 12 * s : p.x + 12 * s;
        const ly = p.y - 8 * s;
        ctx.fillStyle = "rgba(0,0,0,0.70)"; ctx.fillRect(lx - 3 * s, ly, tw + 6 * s, 15 * s);
        ctx.fillStyle = COLOR_HEAD; ctx.fillText(name, lx, ly + 11 * s);
      };
      tag(head.minP, NAMES[head.minJ], true);    // leftmost → label to the left
      tag(head.maxP, NAMES[head.maxJ], false);   // rightmost → label to the right
    }

    // Head midpoint — the point the metric actually uses (filled, white outline).
    ctx.fillStyle = COLOR_HEAD;
    ctx.beginPath(); ctx.arc(head.x, head.y, 6 * s, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5 * s; ctx.stroke();

    // Live offset label by the head.
    const T = calib.torsoBaseline;
    ctx.font = `bold ${Math.round(11 * s)}px ui-monospace, "SF Mono", monospace`;
    const txt = `${fmtSigned(off.offHip / T)} torso`;
    const tw = ctx.measureText(txt).width;
    const tx = head.x + 10 * s, ty = head.y - 18 * s;
    ctx.fillStyle = "rgba(0,0,0,0.70)";
    ctx.fillRect(tx, ty, tw + 8 * s, 16 * s);
    ctx.fillStyle = COLOR_HEAD;
    ctx.fillText(txt, tx + 4 * s, ty + 12 * s);

    // Active straight punch: ring the head at peak + top banner (scored only).
    const c = classifiedAtFrame(state.frame);
    if (c && !c.skip) {
      const col = qualityColor(c.score);
      if (c.peakPrimary === state.frame && c.headAtPeak) {
        ctx.strokeStyle = col; ctx.lineWidth = 2.5 * s;
        ctx.beginPath(); ctx.arc(c.headAtPeak.x, c.headAtPeak.y, 15 * s, 0, Math.PI * 2); ctx.stroke();
        ctx.globalAlpha = 0.30;
        ctx.beginPath(); ctx.arc(c.headAtPeak.x, c.headAtPeak.y, 22 * s, 0, Math.PI * 2); ctx.stroke();
        ctx.globalAlpha = 1;
      }
      banner(ctx, `${c.punch_type} → ${c.score.toFixed(0)} ${qualityBand(c.score).toUpperCase()}  ·  peak dist ${fmtSigned(c.peakDist)} torso`, col, s);
    } else if (c && (c.skip === "sideways" || c.skip === "no-ax")) {
      banner(ctx, `${c.punch_type} → side-on — head read skipped`, "#9aa0a6", s);
    }

    ctx.restore();
}

function banner(ctx, text, col, s) {
  const fontPx = Math.round(15 * s);
  ctx.font = `bold ${fontPx}px ui-monospace, "SF Mono", monospace`;
  const bw = ctx.measureText(text).width + 24 * s;
  const bx = (ctx.canvas.width - bw) / 2, by = 6 * s, bh = fontPx + 14 * s;
  ctx.fillStyle = "rgba(0,0,0,0.78)"; ctx.fillRect(bx, by, bw, bh);
  ctx.strokeStyle = col; ctx.lineWidth = 2 * s; ctx.strokeRect(bx + 0.5, by + 0.5, bw - 1, bh - 1);
  ctx.fillStyle = col; ctx.fillText(text, bx + 12 * s, by + fontPx + 4 * s);
}

// Mini line chart of head off-center distance (mid anchor) across the active
// punch window, with the score peak (furthest excursion) dot, the ±target lines,
// and a cursor at the current frame.
function graphSVG(c, currentFrame) {
  if (!c || !c.series || c.series.length < 2) {
    return `<span class="muted">scrub to a scored straight to see its distance trace</span>`;
  }
  const W = 260, Hh = 96, padX = 6, padT = 8, padB = 16;
  const pts = c.series;
  const f0 = pts[0].frame, f1 = pts[pts.length - 1].frame, fSpan = Math.max(1, f1 - f0);
  const maxAbs = Math.max(cfg.scoreTarget * 1.15, ...pts.map(p => Math.abs(p.dist)));
  const px = fr => padX + ((fr - f0) / fSpan) * (W - 2 * padX);
  const py = di => padT + (1 - (di + maxAbs) / (2 * maxAbs)) * (Hh - padT - padB);
  const col = qualityColor(c.score);
  const f = n => n.toFixed(1);
  const line = pts.map(p => `${f(px(p.frame))},${f(py(p.dist))}`).join(" ");
  const y0 = py(0), yT = py(cfg.scoreTarget), yTn = py(-cfg.scoreTarget);
  const pkx = px(c.peakPrimary), pky = py(c.peakDist);
  const cf = Math.max(f0, Math.min(f1, currentFrame)), cx = px(cf);
  const MONO = `font-size="9" font-family="ui-monospace,monospace"`;
  return `
    <svg viewBox="0 0 ${W} ${Hh}" width="100%" style="display:block">
      <line x1="${padX}" x2="${W - padX}" y1="${f(y0)}" y2="${f(y0)}" stroke="rgba(255,255,255,0.35)" stroke-width="1"/>
      <line x1="${padX}" x2="${W - padX}" y1="${f(yT)}" y2="${f(yT)}" stroke="${col}" stroke-width="0.75" stroke-dasharray="3 3" opacity="0.55"/>
      <line x1="${padX}" x2="${W - padX}" y1="${f(yTn)}" y2="${f(yTn)}" stroke="${col}" stroke-width="0.75" stroke-dasharray="3 3" opacity="0.55"/>
      <line x1="${f(cx)}" x2="${f(cx)}" y1="${padT}" y2="${Hh - padB}" stroke="#fff" stroke-width="1" opacity="0.5"/>
      <polyline points="${line}" fill="none" stroke="${COLOR_HEAD}" stroke-width="1.5"/>
      <circle cx="${f(pkx)}" cy="${f(pky)}" r="3.5" fill="${col}"/>
      <text x="${padX}" y="${Hh - 4}" fill="rgba(255,255,255,0.5)" ${MONO}>f${f0}</text>
      <text x="${W - padX}" y="${Hh - 4}" text-anchor="end" fill="rgba(255,255,255,0.5)" ${MONO}>f${f1}</text>
      <text x="${f(Math.min(W - 36, Math.max(24, pkx)))}" y="${f(Math.max(9, pky - 6))}" text-anchor="middle" fill="${col}" ${MONO}>peak ${fmtSigned(c.peakDist)}</text>
      <text x="${W - padX}" y="${f(yT - 2)}" text-anchor="end" fill="${col}" font-size="8" font-family="ui-monospace,monospace" opacity="0.7">target ${cfg.scoreTarget.toFixed(2)}</text>
    </svg>`;
}
