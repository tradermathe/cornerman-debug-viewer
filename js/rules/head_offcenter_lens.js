// Head-off-the-center-line lens (straight punches) — BlazePose-33.
//
// The coaching rule: when you punch straight, your head should come OFF the
// center line so you're not sat there to be countered. This lens measures,
// for every straight punch, how far the head travels off the line — in torso
// heights.
//
// Why measure RELATIVE to the body, not in raw image x: if we just watched the
// head's screen-x, a fighter who steps or circles would look like they're
// "moving the head off line" when the whole body just translated. So we anchor
// to the body center and report head_offset = head_x − body_center_x. Whole-
// body movement cancels; only the head leaving the column registers.
//
// Why a VERTICAL reference, not the shoulder→hip spine axis: a slip IS a torso
// bend, so the spine tilts toward the slip. Referencing the tilted spine would
// cancel the very movement we want to catch. So body_center_x is a vertical
// line through the body center; a side-bend slip then correctly reads the head
// leaving that line.
//
// Head point: centroid of the visible face landmarks the user picked —
// eyes (2,5), ears (7,8), mouth (9,10). These only exist on the full
// BlazePose-33 cache (state.blaze33), NOT the COCO-17 remap, so this lens
// reads blaze33 directly. Joints used per frame are gated by BlazePose's
// per-joint `visibility` channel.
//
// Two anchors, shown side by side, because which one to ship is an open
// question best settled on footage:
//   mid  = midpoint(shoulder_center_x, hip_center_x)   ← the user's pick
//   hip  = hip_center_x only                            ← more slip-sensitive
// When you bend to slip, the hips stay planted but the shoulders drift toward
// the head, so the midpoint anchor undercounts the slip vs hips-only. Watching
// both per punch tells you how much that matters on real video.
//
// Normalization: a body straight drops the level and FORESHORTENS the torso in
// the image, so the instantaneous torso length shrinks. Normalizing by that
// would inflate every body shot. So we normalize by a STABLE per-round torso
// height (median over frames where all four torso joints are visible), not the
// bent-over value.
//
// Per straight punch [start,end]:
//   peak frame  = max |wrist − same-side shoulder| in the window (contact moment)
//   start frame = first frame in the window with a usable head + torso
//   off_center_at_peak = head_offset(peak) / torsoBaseline
//   head_travel        = (head_offset(peak) − head_offset(start)) / torsoBaseline
// Sign is image-space (+ = head right of center). The fault we're flagging is a
// SMALL |head_travel| — the head stayed on the line through the punch.

// blaze33 channels: 0 x  1 y  2 z  3 xw  4 yw  5 zw  6 visibility  7 presence
const CH = 8, X = 0, Y = 1, VIS = 6, NJ = 33;

// BlazePose-33 joint indices.
const L_EYE = 2, R_EYE = 5, L_EAR = 7, R_EAR = 8, MOUTH_L = 9, MOUTH_R = 10;
const L_SHOULDER = 11, R_SHOULDER = 12, L_WRIST = 15, R_WRIST = 16;
const L_HIP = 23, R_HIP = 24;
const HEAD_JOINTS = [L_EYE, R_EYE, L_EAR, R_EAR, MOUTH_L, MOUTH_R];

const MIN_VIS = 0.30;
const MIN_TORSO_PX = 5;
// |head_travel| below this (in torso heights) = head stayed on the line = the
// fault. Provisional — calibrate on footage; that's what this lens is for.
const HEAD_TRAVEL_MIN = 0.10;

const COLOR_CENTER  = "#ffd24a";   // body-center vertical line (mid anchor)
const COLOR_HIP     = "#3ad9e0";   // hips-only vertical line
const COLOR_HEAD    = "#c08bff";   // head point (eyes/ears/mouth centroid)
const COLOR_GOOD    = "#5fd97a";   // moved off the line
const COLOR_FAULT   = "#e85a5a";   // stayed on the line

let host = null;
let calib = null;            // { torsoBaseline, nValid, total }
let classified = [];         // per-detection results
let lastKey = null, lastBlaze = null, lastDets = null;

function b33(state) { return state.blaze33 || null; }
function cacheKey(state) { return `${state.cacheBasename || ""}__r${state.cacheRound ?? "?"}`; }

// --- blaze33 ↔ primary-timeline frame mapping -----------------------------
// Detections + state.frame live in the primary pose timeline. blaze33 is
// usually the same extraction (frame-aligned), but guard with the same
// alignment check the inspector uses and fall back to time/PTS.
function blazeAligned(b, state) {
  return b.n_frames === state.n_frames
    && Math.abs((b.fps || 0) - (state.fps || 0)) < 0.01
    && Math.abs((b.start_sec || 0) - (state.start_sec || 0)) < 1e-3;
}
function blazeFrame(b, state, primaryFrame) {
  if (blazeAligned(b, state)) {
    return Math.min(Math.max(primaryFrame, 0), b.n_frames - 1);
  }
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
  if (!(b.data[o + VIS] >= MIN_VIS)) return null;
  const x = b.data[o + X] * w, y = b.data[o + Y] * h;
  return (Number.isFinite(x) && Number.isFinite(y)) ? { x, y } : null;
}

function headPoint(b, base, w, h) {
  let sx = 0, sy = 0, n = 0;
  for (const j of HEAD_JOINTS) {
    const p = jointAt(b, base, j, w, h);
    if (!p) continue;
    sx += p.x; sy += p.y; n++;
  }
  return n ? { x: sx / n, y: sy / n, n } : null;
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

// head_offset against both anchors at one frame.
function offsetAt(b, base, w, h) {
  const head = headPoint(b, base, w, h);
  const torso = torsoAt(b, base, w, h);
  if (!head || !torso) return null;
  return { head, torso, offMid: head.x - torso.midX, offHip: head.x - torso.hipX };
}

// Stable per-round torso height: median |shoulder_mid − hip_mid| over frames
// where all four torso joints clear MIN_VIS. Median (not instantaneous) so a
// deep body-shot crouch doesn't shrink the unit we divide by.
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
function isStraight(det) {
  return /^(jab|cross)/.test(String(det.punch_type || "").toLowerCase());
}
function isBody(det) { return /_body$/.test(String(det.punch_type || "").toLowerCase()); }

// (punch_type, stance) → "L"/"R". jab/lead_* = lead hand, cross/rear_* = rear;
// orthodox lead = L, southpaw lead = R.
function throwingHand(det) {
  const t = String(det.punch_type || "").toLowerCase();
  const stance = String(det.stance || "orthodox").toLowerCase();
  let isLead = null;
  if (t.startsWith("jab") || t.startsWith("lead_")) isLead = true;
  else if (t.startsWith("cross") || t.startsWith("rear_")) isLead = false;
  if (isLead === null) return null;
  if (stance === "southpaw") return isLead ? "R" : "L";
  return isLead ? "L" : "R";
}

// Frame of max wrist-from-shoulder distance in [sB,eB] (contact moment).
function peakFrame(b, w, h, sB, eB, side) {
  const wj = side === "L" ? L_WRIST : R_WRIST;
  const sj = side === "L" ? L_SHOULDER : R_SHOULDER;
  let bestF = -1, bestD = -1;
  for (let f = sB; f <= eB; f++) {
    const base = f * NJ * CH;
    const wp = jointAt(b, base, wj, w, h);
    const sp = jointAt(b, base, sj, w, h);
    if (!wp || !sp) continue;
    const d = Math.hypot(wp.x - sp.x, wp.y - sp.y);
    if (d > bestD) { bestD = d; bestF = f; }
  }
  return bestF >= 0 ? { frame: bestF, extension: bestD } : null;
}

function classifyPunch(b, state, det, T, w, h) {
  if (!isStraight(det)) return { det, straight: false };
  const hand = throwingHand(det);
  if (!hand) return { det, straight: true, error: "unknown hand" };
  const sB = blazeFrame(b, state, det.start_frame);
  const eB = blazeFrame(b, state, det.end_frame);
  if (sB == null || eB == null) return { det, straight: true, hand, error: "window not in blaze cache" };
  const peak = peakFrame(b, w, h, sB, eB, hand);
  if (!peak) return { det, straight: true, hand, error: "no confident peak frame" };

  // start offset = first usable frame from the window start onward.
  let startOff = null, startF = sB;
  for (let f = sB; f <= eB; f++) {
    const o = offsetAt(b, f * NJ * CH, w, h);
    if (o) { startOff = o; startF = f; break; }
  }
  const peakOff = offsetAt(b, peak.frame * NJ * CH, w, h);
  if (!startOff || !peakOff) return { det, straight: true, hand, peakFrame: peak.frame, error: "no head/torso at start or peak" };

  return {
    det, straight: true, hand,
    body: isBody(det),
    startFrame: startF,
    peakFrame: peak.frame,
    travelMid: (peakOff.offMid - startOff.offMid) / T,
    travelHip: (peakOff.offHip - startOff.offHip) / T,
    peakMid: peakOff.offMid / T,
    peakHip: peakOff.offHip / T,
    headAtPeak: peakOff.head,
    centerAtPeak: { mid: peakOff.torso.midX, hip: peakOff.torso.hipX },
  };
}

function refreshIfNeeded(state) {
  const b = b33(state);
  if (!b) { calib = null; classified = []; return; }
  const k = cacheKey(state);
  const dets = state.labels?.detections;
  let recalc = false;
  if (k !== lastKey || b !== lastBlaze) {
    lastKey = k; lastBlaze = b;
    calib = calibrate(b, b.width, b.height);
    recalc = true;
  }
  if (recalc || dets !== lastDets) {
    lastDets = dets;
    const T = calib?.torsoBaseline;
    classified = (Array.isArray(dets) && T)
      ? dets.map(d => classifyPunch(b, state, d, T, b.width, b.height))
      : [];
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

function travelColor(t) {
  return Math.abs(t) >= HEAD_TRAVEL_MIN ? COLOR_GOOD : COLOR_FAULT;
}
function fmtSigned(v, d = 2) { return v == null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(d)}`; }

// --- sidebar --------------------------------------------------------------
function renderSidebar(state) {
  const el = host?.querySelector("#hoc-state");
  if (!el) return;
  const b = b33(state);
  const lines = [];

  if (!b) {
    el.innerHTML = `<span class="bad">No BlazePose-33 cache loaded for this round — this lens needs the full 33-joint cache (eyes/ears/mouth).</span>`;
    return;
  }
  if (calib?.torsoBaseline) {
    lines.push(`<span class="muted">torso baseline (this round):</span> <code>${calib.torsoBaseline.toFixed(1)} px</code> <span class="muted">(median, n=${calib.nValid}/${calib.total})</span>`);
  } else {
    lines.push(`<span class="bad">no frames with all 4 torso joints visible — can't normalize.</span>`);
    el.innerHTML = lines.join("<br>");
    return;
  }
  lines.push(``);

  // Live readout at the current frame.
  const f = blazeFrame(b, state, state.frame);
  const off = f == null ? null : offsetAt(b, f * NJ * CH, b.width, b.height);
  if (!off) {
    lines.push(`<span class="muted">head or torso below visibility at this frame.</span>`);
  } else {
    const T = calib.torsoBaseline;
    lines.push(`<span style="color:${COLOR_HEAD}">head</span> off center (live): mid <code>${fmtSigned(off.offMid / T)}</code> · hip <code>${fmtSigned(off.offHip / T)}</code> torso <span class="muted">(${off.head.n}/6 face pts)</span>`);
  }
  lines.push(``);

  const c = classifiedAtFrame(state.frame);
  if (c && c.straight && !c.error) {
    const col = travelColor(c.travelMid);
    lines.push(`<span class="muted">active straight:</span> <code>${c.det.punch_type}</code>${c.body ? ' <span class="muted">(body)</span>' : ''} · ${c.hand}-hand`);
    lines.push(`<b>head_travel</b>: mid <span style="color:${col};font-weight:600">${fmtSigned(c.travelMid)}</span> · hip <code>${fmtSigned(c.travelHip)}</code> torso`);
    lines.push(`at-peak offset: mid <code>${fmtSigned(c.peakMid)}</code> · hip <code>${fmtSigned(c.peakHip)}</code> torso`);
    if (Math.abs(c.travelMid) < HEAD_TRAVEL_MIN) {
      lines.push(`<span style="color:${COLOR_FAULT}">⚠ head stayed on the line (|travel| &lt; ${HEAD_TRAVEL_MIN})</span>`);
    }
    lines.push(c.peakFrame === state.frame
      ? `<span class="muted">(at peak frame)</span>`
      : `<a href="#" data-jump="${c.peakFrame}" style="color:${COLOR_HIP}">→ jump to peak frame f${c.peakFrame}</a>`);
  } else if (c && c.straight && c.error) {
    lines.push(`<span class="muted">active straight:</span> <code>${c.det.punch_type}</code> — <span class="bad">${c.error}</span>`);
  } else {
    lines.push(`<span class="muted">no straight punch active at this frame.</span>`);
  }

  el.innerHTML = lines.join("<br>");
}

// --- per-punch summary table ----------------------------------------------
function mountSummary() {
  const slot = document.getElementById("stage-extras");
  if (!slot) return;
  slot.innerHTML = `
    <div id="hoc-summary-card" style="margin-top:12px;padding:10px 12px;background:var(--bg-card);border:1px solid var(--border);border-radius:8px">
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;gap:12px;flex-wrap:wrap">
        <span class="muted small">head_travel per straight punch (torso heights) — click a row to jump to its peak</span>
        <span id="hoc-summary-stats" class="muted small"></span>
      </div>
      <div id="hoc-summary-wrap" style="max-height:240px;overflow-y:auto"></div>
    </div>`;
  slot.addEventListener("click", e => {
    const row = e.target.closest("tr[data-peak]");
    if (row) { const f = parseInt(row.dataset.peak, 10); if (Number.isFinite(f)) seekTo(f); }
  });
}

function renderSummary(state) {
  const wrap = document.getElementById("hoc-summary-wrap");
  const stats = document.getElementById("hoc-summary-stats");
  if (!wrap) return;
  const straights = classified.filter(c => c.straight);
  if (!straights.length) {
    wrap.innerHTML = `<div class="muted small">no straight-punch labels loaded for this round (lens scopes to jab/cross, head or body).</div>`;
    if (stats) stats.textContent = "";
    return;
  }

  const f = state.frame;
  let nOk = 0, nFault = 0, nErr = 0;
  const rows = straights.map((c, i) => {
    const active = f >= c.det.start_frame && f <= c.det.end_frame;
    const isPeak = c.peakFrame === f;
    const bg = isPeak ? "background:rgba(58,217,224,0.18);" : (active ? "background:rgba(58,217,224,0.06);" : "");
    if (c.error) {
      nErr++;
      return `<tr data-peak="${c.det.start_frame}" title="${c.error}" style="cursor:pointer;${bg}">
        <td style="padding:2px 6px;color:var(--muted)">${i + 1}</td>
        <td style="padding:2px 6px"><code>${c.det.punch_type || "?"}</code></td>
        <td style="padding:2px 6px;color:var(--muted)">${c.det.start_frame}–${c.det.end_frame}</td>
        <td style="padding:2px 6px">${c.hand || "—"}</td>
        <td style="padding:2px 6px" colspan="3"><span class="bad">${c.error}</span></td>
      </tr>`;
    }
    const col = travelColor(c.travelMid);
    if (Math.abs(c.travelMid) >= HEAD_TRAVEL_MIN) nOk++; else nFault++;
    return `<tr data-peak="${c.peakFrame}" style="cursor:pointer;${bg}">
      <td style="padding:2px 6px;color:var(--muted)">${i + 1}</td>
      <td style="padding:2px 6px"><code>${c.det.punch_type}</code>${c.body ? ' <span class="muted">b</span>' : ''}</td>
      <td style="padding:2px 6px;color:var(--muted)">${c.det.start_frame}–${c.det.end_frame}</td>
      <td style="padding:2px 6px">${c.hand}</td>
      <td style="padding:2px 6px;color:${col};font-weight:600">${fmtSigned(c.travelMid)}</td>
      <td style="padding:2px 6px"><code>${fmtSigned(c.travelHip)}</code></td>
      <td style="padding:2px 6px"><code>${fmtSigned(c.peakMid)}</code></td>
    </tr>`;
  });

  wrap.innerHTML = `
    <table style="width:100%;border-collapse:collapse;font:11px ui-monospace,'SF Mono',monospace">
      <thead style="position:sticky;top:0;background:var(--bg-card);box-shadow:0 1px 0 var(--border)">
        <tr style="color:var(--muted);text-align:left">
          <th style="padding:4px 6px">#</th>
          <th style="padding:4px 6px">punch</th>
          <th style="padding:4px 6px">range</th>
          <th style="padding:4px 6px">hand</th>
          <th style="padding:4px 6px">travel·mid</th>
          <th style="padding:4px 6px">travel·hip</th>
          <th style="padding:4px 6px">peak·mid</th>
        </tr>
      </thead>
      <tbody>${rows.join("")}</tbody>
    </table>`;

  if (stats) {
    stats.innerHTML = `${straights.length} straights · ` +
      `<span style="color:${COLOR_GOOD}">moved ${nOk}</span> · ` +
      `<span style="color:${COLOR_FAULT}">on-line ${nFault}</span>` +
      (nErr ? ` · err ${nErr}` : "") +
      ` · |travel|≥${HEAD_TRAVEL_MIN} = off the line`;
  }
}

export const HeadOffCenterLensRule = {
  id: "head_offcenter",
  label: "Head off center line (straights)",

  // Needs the full 33-joint cache — mouth/eye landmarks aren't in the COCO-17 remap.
  requires(slot) { return !!(slot && slot.blazepose); },

  mount(_host, state) {
    host = _host;
    host.innerHTML = `
      <h2>Head off the center line</h2>
      <p class="hint">
        Per straight punch (jab/cross, head or body): how far the head moves off
        the body's center line, in torso heights. Head = centroid of the visible
        eyes/ears/mouth on the <b>BlazePose-33</b> cache; center line = a vertical
        through the shoulder/hip mid. Measured relative to the body so stepping or
        circling doesn't count — only the head leaving the column does.
      </p>
      <p class="hint">
        Two anchors shown together: <b>mid</b> = shoulder+hip midpoint (steadier),
        <b>hip</b> = hips only (more slip-sensitive). <code>peak·mid</code> is the
        head's absolute offset at contact; <code>travel</code> is how much it moved
        from the guard to that peak. Normalized by a stable per-round torso height
        so a body-shot crouch doesn't inflate it.
      </p>
      <p class="hint">
        Overlay: <span style="color:${COLOR_CENTER}">●</span> center line (mid) ·
        <span style="color:${COLOR_HIP}">●</span> hips-only line ·
        <span style="color:${COLOR_HEAD}">●</span> head. A straight reads
        <span style="color:${COLOR_GOOD}">green</span> when |travel| ≥ ${HEAD_TRAVEL_MIN}
        (off the line), <span style="color:${COLOR_FAULT}">red</span> when it stayed on it.
      </p>
      <div id="hoc-state" class="hint" style="line-height:1.55"></div>
      <p class="hint" style="margin-top:14px;font-size:11px">
        Flag threshold |travel| ≥ ${HEAD_TRAVEL_MIN} torso, MIN_VIS=${MIN_VIS}.
        Provisional — tune constants at the top of <code>head_offcenter_lens.js</code>.
      </p>`;
    host.addEventListener("click", e => {
      const a = e.target.closest("[data-jump]");
      if (!a) return;
      e.preventDefault();
      const f = parseInt(a.dataset.jump, 10);
      if (Number.isFinite(f)) seekTo(f);
    });
    mountSummary();
    refreshIfNeeded(state);
    renderSidebar(state);
    renderSummary(state);
  },

  update(state) {
    refreshIfNeeded(state);
    renderSidebar(state);
    renderSummary(state);
  },

  draw(ctx, state) {
    const b = b33(state);
    if (!b || !calib?.torsoBaseline) return;
    const f = blazeFrame(b, state, state.frame);
    if (f == null) return;
    const w = b.width, h = b.height, s = state.renderScale || 1, H = ctx.canvas.height;
    const off = offsetAt(b, f * NJ * CH, w, h);
    if (!off) return;
    const { head, torso } = off;

    ctx.save();

    // Vertical center line (mid anchor) + hips-only line.
    ctx.lineWidth = 2 * s;
    ctx.strokeStyle = COLOR_CENTER;
    ctx.beginPath(); ctx.moveTo(torso.midX, 0); ctx.lineTo(torso.midX, H); ctx.stroke();

    ctx.lineWidth = 1.5 * s;
    ctx.strokeStyle = COLOR_HIP;
    ctx.setLineDash([6 * s, 5 * s]);
    ctx.beginPath(); ctx.moveTo(torso.hipX, 0); ctx.lineTo(torso.hipX, H); ctx.stroke();
    ctx.setLineDash([]);

    // Horizontal connector head → center line (the offset we measure).
    ctx.strokeStyle = COLOR_HEAD;
    ctx.globalAlpha = 0.6;
    ctx.lineWidth = 1.5 * s;
    ctx.beginPath(); ctx.moveTo(head.x, head.y); ctx.lineTo(torso.midX, head.y); ctx.stroke();
    ctx.globalAlpha = 1;

    // Head point.
    ctx.fillStyle = COLOR_HEAD;
    ctx.beginPath(); ctx.arc(head.x, head.y, 6 * s, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,0.65)"; ctx.lineWidth = 1.5 * s; ctx.stroke();

    // Live offset label by the head.
    const T = calib.torsoBaseline;
    ctx.font = `bold ${Math.round(11 * s)}px ui-monospace, "SF Mono", monospace`;
    const txt = `${fmtSigned(off.offMid / T)} torso`;
    const tw = ctx.measureText(txt).width;
    const tx = head.x + 10 * s, ty = head.y - 18 * s;
    ctx.fillStyle = "rgba(0,0,0,0.70)";
    ctx.fillRect(tx, ty, tw + 8 * s, 16 * s);
    ctx.fillStyle = COLOR_HEAD;
    ctx.fillText(txt, tx + 4 * s, ty + 12 * s);

    // Active straight punch: ring the head at peak + top banner.
    const c = classifiedAtFrame(state.frame);
    if (c && c.straight && !c.error) {
      const col = travelColor(c.travelMid);
      if (c.peakFrame === state.frame && c.headAtPeak) {
        ctx.strokeStyle = col; ctx.lineWidth = 2.5 * s;
        ctx.beginPath(); ctx.arc(c.headAtPeak.x, c.headAtPeak.y, 15 * s, 0, Math.PI * 2); ctx.stroke();
        ctx.globalAlpha = 0.30;
        ctx.beginPath(); ctx.arc(c.headAtPeak.x, c.headAtPeak.y, 22 * s, 0, Math.PI * 2); ctx.stroke();
        ctx.globalAlpha = 1;
      }
      const fontPx = Math.round(15 * s);
      ctx.font = `bold ${fontPx}px ui-monospace, "SF Mono", monospace`;
      const text = `${c.det.punch_type} → head_travel ${fmtSigned(c.travelMid)} torso (mid) · ${fmtSigned(c.travelHip)} (hip)`;
      const bw = ctx.measureText(text).width + 24 * s;
      const bx = (ctx.canvas.width - bw) / 2, by = 6 * s, bh = fontPx + 14 * s;
      ctx.fillStyle = "rgba(0,0,0,0.78)"; ctx.fillRect(bx, by, bw, bh);
      ctx.strokeStyle = col; ctx.lineWidth = 2 * s; ctx.strokeRect(bx + 0.5, by + 0.5, bw - 1, bh - 1);
      ctx.fillStyle = col; ctx.fillText(text, bx + 12 * s, by + fontPx + 4 * s);
    }

    ctx.restore();
  },
};
