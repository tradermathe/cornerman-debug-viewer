// Body-shot detector lens — visualises the ankle-anchored "standing chest"
// reference for splitting head vs body shots geometrically, plus a per-punch
// summary that scores each detected punch against its GT label.
//
// Why ankle-anchored: the naïve "puncher's CURRENT chest" reference breaks
// when the fighter ducks to throw to the body — the chest moves down with
// the punch intent. Anchoring to the floor (ankle midpoint) instead lets us
// reconstruct where the chest WOULD be if the fighter were standing upright,
// regardless of current crouch depth.
//
// Why torso-normalized: a per-round pixel calibration breaks when the
// fighter moves toward/away from the camera — same body, different pixel
// size. So we calibrate a unitless RATIO (chest-above-floor / torso length)
// and multiply by the CURRENT torso length each frame. Torso length is the
// Euclidean shoulder-mid → hip-mid distance, which is invariant to leg bend
// and tracks camera distance proportionally.
//
// Calibration: standing_ratio = p95 of (ankle_mid_y − chest_y) / torso_now
// over every frame where all four torso joints + both ankles clear MIN_CONF.
// p95 is robust to outliers without needing a separate "upright-frame"
// detector. Typical adult anatomy gives ~1.7–2.0; if calibration lands
// outside that band something's wrong with the pose data.
//
// Per-frame signal at any wrist:
//   expected_chest_y = ankle_mid_y − standing_ratio × torso_now
//   signal = (wrist_y − expected_chest_y) / torso_now
//   > 0  → wrist below standing chest line → BODY territory
//   < 0  → wrist above standing chest line → HEAD territory
//
// Per-punch classification: each detected punch has a window [start, end].
// We pick the frame within that window where the throwing arm is most
// extended (|wrist − same-side shoulder| max) and read the signal there.
// Peak extension is the moment of intended contact for all six punch types:
// straights reach max forward, hooks reach max lateral, uppercuts reach
// max upward. Hand selection: punch_type prefix (jab/lead_* = lead;
// cross/rear_* = rear) combined with stance (orthodox lead = L, southpaw
// lead = R) maps to L/R wrist.
//
// GT integration: detections come from sheet-labels.js, which loads the
// Combined Data tab where labels carry `_head`/`_body` suffixes. The
// summary table scores every classified punch against that GT.
//
// Hip fallback (when ankles are off-screen): chest midpoint is HALF A TORSO
// above hips by construction. We set expected_chest_y = hip_y − (0.5 +
// boost) × torso_now. Caveat: hips drop with the body during a duck, so
// this fallback CAN'T detect ducked body shots — it'll classify them as
// head. The cyan line goes dashed in fallback mode so degraded behaviour
// is visible at a glance.

import { J } from "../skeleton.js";

const MIN_CONF = 0.30;
const HEAD_THRESHOLD = -0.15;     // signal ≤ this → head (in torso units)
const BODY_THRESHOLD =  0.15;     // signal ≥ this → body
const CALIB_PERCENTILE = 0.95;
const MIN_TORSO_PX = 5;
// Lift applied on top of the calibrated p95 ratio — pure visual nudge of the
// divider line. 1 torso ≈ 40cm in adult anatomy, so 0.05 ≈ 2cm.
const STANDING_RATIO_BOOST = 0.05;
// Hip fallback: chest midpoint sits half a torso above hips (by definition,
// chest = midpoint of shoulder-mid and hip-mid).
const HIP_TO_DIVIDER_RATIO = 0.5;

const COLOR_FLOOR    = "rgba(255,255,255,0.30)";
const COLOR_STANDING = "#3ad9e0";
const COLOR_CURRENT  = "#ffd24a";
const COLOR_HEAD     = "#5fd97a";
const COLOR_AMBIG    = "#f5b945";
const COLOR_BODY     = "#e85a5a";

let host;
let calibration = null;
let classifiedPunches = [];       // per-punch results, parallel to state.labels.detections
let lastCacheKey = null;
let lastPose = null;
let lastDets = null;

function cacheKey(state) {
  return `${state.cacheBasename || ""}__r${state.cacheRound ?? "?"}`;
}

function torsoAt(pose, f) {
  const base = f * 17;
  const cLS = pose.conf[base + J.L_SHOULDER];
  const cRS = pose.conf[base + J.R_SHOULDER];
  const cLH = pose.conf[base + J.L_HIP];
  const cRH = pose.conf[base + J.R_HIP];
  if (cLS < MIN_CONF || cRS < MIN_CONF || cLH < MIN_CONF || cRH < MIN_CONF) return null;
  const lsx = pose.skeleton[(base + J.L_SHOULDER) * 2];
  const lsy = pose.skeleton[(base + J.L_SHOULDER) * 2 + 1];
  const rsx = pose.skeleton[(base + J.R_SHOULDER) * 2];
  const rsy = pose.skeleton[(base + J.R_SHOULDER) * 2 + 1];
  const lhx = pose.skeleton[(base + J.L_HIP) * 2];
  const lhy = pose.skeleton[(base + J.L_HIP) * 2 + 1];
  const rhx = pose.skeleton[(base + J.R_HIP) * 2];
  const rhy = pose.skeleton[(base + J.R_HIP) * 2 + 1];
  if (![lsx,lsy,rsx,rsy,lhx,lhy,rhx,rhy].every(Number.isFinite)) return null;
  const sx = (lsx + rsx) / 2, sy = (lsy + rsy) / 2;
  const hx = (lhx + rhx) / 2, hy = (lhy + rhy) / 2;
  const dx = sx - hx, dy = sy - hy;
  const length = Math.sqrt(dx * dx + dy * dy);
  if (length < MIN_TORSO_PX) return null;
  return {
    shoulder: { x: sx, y: sy },
    hip:      { x: hx, y: hy },
    chest:    { x: (sx + hx) / 2, y: (sy + hy) / 2 },
    length,
  };
}

function ankleMidAt(pose, f) {
  const base = f * 17;
  const cL = pose.conf[base + J.L_ANKLE];
  const cR = pose.conf[base + J.R_ANKLE];
  if (cL < MIN_CONF || cR < MIN_CONF) return null;
  const lx = pose.skeleton[(base + J.L_ANKLE) * 2];
  const ly = pose.skeleton[(base + J.L_ANKLE) * 2 + 1];
  const rx = pose.skeleton[(base + J.R_ANKLE) * 2];
  const ry = pose.skeleton[(base + J.R_ANKLE) * 2 + 1];
  if (![lx, ly, rx, ry].every(Number.isFinite)) return null;
  return { x: (lx + rx) / 2, y: (ly + ry) / 2 };
}

function wristAt(pose, f, side) {
  const j = side === "L" ? J.L_WRIST : J.R_WRIST;
  const c = pose.conf[f * 17 + j];
  if (c < MIN_CONF) return null;
  const x = pose.skeleton[(f * 17 + j) * 2];
  const y = pose.skeleton[(f * 17 + j) * 2 + 1];
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return { x, y, conf: c };
}

function calibrate(pose) {
  const n = pose.conf.length / 17;
  const ratios = [];
  for (let f = 0; f < n; f++) {
    const torso = torsoAt(pose, f);
    const ankle = ankleMidAt(pose, f);
    if (!torso || !ankle) continue;
    const dist = ankle.y - torso.chest.y;
    if (dist <= 0) continue;
    ratios.push(dist / torso.length);
  }
  if (ratios.length === 0) {
    return { standingRatio: null, ratioMedian: null, nValidFrames: 0, totalFrames: n };
  }
  ratios.sort((a, b) => a - b);
  const idx = Math.min(ratios.length - 1, Math.floor(CALIB_PERCENTILE * ratios.length));
  const rawRatio = ratios[idx];
  return {
    standingRatio: rawRatio + STANDING_RATIO_BOOST,
    rawRatio,
    ratioMedian: ratios[Math.floor(ratios.length / 2)],
    nValidFrames: ratios.length,
    totalFrames: n,
  };
}

function classify(signal) {
  if (signal <= HEAD_THRESHOLD) return "head";
  if (signal >= BODY_THRESHOLD) return "body";
  return "ambig";
}

function colorFor(label) {
  if (label === "head") return COLOR_HEAD;
  if (label === "body") return COLOR_BODY;
  return COLOR_AMBIG;
}

function activePunch(state) {
  const dets = state.labels?.detections;
  if (!Array.isArray(dets) || dets.length === 0) return null;
  const f = state.frame;
  for (const d of dets) {
    if (f >= d.start_frame && f <= d.end_frame) return d;
  }
  return null;
}

function labelTarget(punchType) {
  if (typeof punchType !== "string") return null;
  if (punchType.endsWith("_head")) return "head";
  if (punchType.endsWith("_body")) return "body";
  return null;
}

function dividerAt(pose, f) {
  const torso = torsoAt(pose, f);
  if (!torso) return null;
  const ankle = ankleMidAt(pose, f);
  if (ankle && calibration?.standingRatio) {
    return {
      expected: ankle.y - calibration.standingRatio * torso.length,
      torso, ankle,
      anchor: "ankle",
      ratio: calibration.standingRatio,
    };
  }
  const effRatio = HIP_TO_DIVIDER_RATIO + STANDING_RATIO_BOOST;
  return {
    expected: torso.hip.y - effRatio * torso.length,
    torso, ankle: null,
    anchor: "hip",
    ratio: effRatio,
  };
}

// Returns the frame within [startF, endF] where the gloved hand is most
// extended (max Euclidean distance from same-side shoulder). That's the
// contact moment for all punch types: straights peak forward, hooks peak
// lateral, uppercuts peak upward. Returns null when no frame in the window
// has both joints confident enough.
function peakExtensionFrame(pose, startF, endF, side) {
  const wristJ    = side === "L" ? J.L_WRIST    : J.R_WRIST;
  const shoulderJ = side === "L" ? J.L_SHOULDER : J.R_SHOULDER;
  let bestF = -1, bestDist = -1;
  for (let f = startF; f <= endF; f++) {
    const wc = pose.conf[f * 17 + wristJ];
    const sc = pose.conf[f * 17 + shoulderJ];
    if (wc < MIN_CONF || sc < MIN_CONF) continue;
    const wx = pose.skeleton[(f * 17 + wristJ) * 2];
    const wy = pose.skeleton[(f * 17 + wristJ) * 2 + 1];
    const sx = pose.skeleton[(f * 17 + shoulderJ) * 2];
    const sy = pose.skeleton[(f * 17 + shoulderJ) * 2 + 1];
    if (![wx, wy, sx, sy].every(Number.isFinite)) continue;
    const d = Math.hypot(wx - sx, wy - sy);
    if (d > bestDist) { bestDist = d; bestF = f; }
  }
  return bestF >= 0 ? { frame: bestF, extension: bestDist } : null;
}

// Maps (punch_type, stance) → "L" or "R". jab/lead_* are thrown with the
// lead hand; cross/rear_* with the rear hand. Stance flips which side that
// is: orthodox lead = L, southpaw lead = R. Default to orthodox when stance
// missing.
function throwingHand(punch) {
  if (!punch?.punch_type) return null;
  const t = String(punch.punch_type).toLowerCase();
  const stance = String(punch.stance || "orthodox").toLowerCase();
  let isLead = null;
  if (t.startsWith("jab") || t.startsWith("lead_")) isLead = true;
  else if (t.startsWith("cross") || t.startsWith("rear_")) isLead = false;
  if (isLead === null) return null;
  if (stance === "southpaw") return isLead ? "R" : "L";
  return isLead ? "L" : "R";
}

// End-to-end per-punch result. Returns an object with whatever could be
// computed plus an `error` field when something blocked classification.
function classifyPunch(pose, punch) {
  const hand = throwingHand(punch);
  const gt = labelTarget(punch.punch_type);
  if (!hand) return { punch, hand: null, gt, error: "unknown hand" };
  const peak = peakExtensionFrame(pose, punch.start_frame, punch.end_frame, hand);
  if (!peak) return { punch, hand, gt, error: "no confident peak frame" };
  const info = dividerAt(pose, peak.frame);
  if (!info) return { punch, hand, gt, peakFrame: peak.frame, error: "no torso at peak" };
  const wristJ = hand === "L" ? J.L_WRIST : J.R_WRIST;
  const wx = pose.skeleton[(peak.frame * 17 + wristJ) * 2];
  const wy = pose.skeleton[(peak.frame * 17 + wristJ) * 2 + 1];
  if (!Number.isFinite(wx) || !Number.isFinite(wy)) {
    return { punch, hand, gt, peakFrame: peak.frame, error: "wrist NaN at peak" };
  }
  const sig = (wy - info.expected) / info.torso.length;
  const label = classify(sig);
  const match = gt && (label === "head" || label === "body") ? (label === gt) : null;
  return {
    punch, hand, gt,
    peakFrame: peak.frame,
    extension: peak.extension,
    wristAtPeak: { x: wx, y: wy },
    signal: sig,
    label,
    anchor: info.anchor,
    match,
  };
}

function classifyAllPunches(pose, dets) {
  if (!Array.isArray(dets)) return [];
  return dets.map(d => classifyPunch(pose, d));
}

function refreshIfNeeded(state) {
  const k = cacheKey(state);
  const dets = state.labels?.detections;
  let recalibrated = false;
  if (k !== lastCacheKey || state.pose !== lastPose) {
    lastCacheKey = k;
    lastPose = state.pose;
    calibration = calibrate(state.pose);
    recalibrated = true;
  }
  if (recalibrated || dets !== lastDets) {
    lastDets = dets;
    classifiedPunches = classifyAllPunches(state.pose, dets || []);
  }
}

// Push the viewer's scrubber to frame F via the existing input-event hook.
// Mirrors the trick punch_classifier.js uses so the seek goes through the
// canonical playback path rather than reaching into viewer-private state.
function seekTo(f) {
  const slider = document.getElementById("scrubber");
  if (!slider) return;
  slider.value = f;
  slider.dispatchEvent(new Event("input"));
}

function classifiedAtFrame(frame) {
  for (const c of classifiedPunches) {
    if (frame >= c.punch.start_frame && frame <= c.punch.end_frame) return c;
  }
  return null;
}

function renderSidebar(state) {
  if (!host) return;
  const el = host.querySelector("#bs-state");
  if (!el) return;

  const lines = [];

  if (calibration && calibration.standingRatio) {
    lines.push(`<span class="muted">calibration (this round):</span>`);
    lines.push(`standing_ratio: <code>${calibration.standingRatio.toFixed(2)}</code> torso-lengths &nbsp;(p${(CALIB_PERCENTILE*100).toFixed(0)}=<code>${calibration.rawRatio.toFixed(2)}</code> + boost <code>${STANDING_RATIO_BOOST.toFixed(2)}</code>, median <code>${calibration.ratioMedian.toFixed(2)}</code>, n=${calibration.nValidFrames}/${calibration.totalFrames})`);
  } else {
    lines.push(`<span class="muted">no ankle-anchored calibration — round had no frames with both ankles + 4 torso joints. Falling back to hip-anchored on every frame.</span>`);
  }
  lines.push(``);

  const f = state.frame;
  const info = dividerAt(state.pose, f);
  if (!info) {
    lines.push(`<span class="muted">torso below conf at this frame — can't compute expected_chest_y.</span>`);
    el.innerHTML = lines.join("<br>");
    return;
  }
  const { expected, torso, anchor, ratio } = info;

  if (anchor === "hip") {
    lines.push(`<span style="color:${COLOR_AMBIG}">⚠ hip fallback active</span> — ankles not visible. Divider sits at hip_y − ${ratio.toFixed(2)} × torso. <em>Can't detect ducks: ducked body shots will read as head.</em>`);
  }
  lines.push(`torso (now): <code>${torso.length.toFixed(1)} px</code> &nbsp;<span class="muted">(anchor: ${anchor})</span>`);
  lines.push(`<span style="color:${COLOR_STANDING}">standing chest line</span>: y = <code>${expected.toFixed(0)}</code>`);
  const drop = torso.chest.y - expected;
  const dropTorsos = drop / torso.length;
  const tag = Math.abs(dropTorsos) < 0.05 ? "upright" : (drop > 0 ? `ducked ${dropTorsos.toFixed(2)} torso` : `tall ${dropTorsos.toFixed(2)} torso`);
  lines.push(`<span style="color:${COLOR_CURRENT}">current chest</span>: y = <code>${torso.chest.y.toFixed(0)}</code> &nbsp;(${tag})`);
  lines.push(``);

  const punch = activePunch(state);
  const gt = punch ? labelTarget(punch.punch_type) : null;

  for (const side of ["L", "R"]) {
    const w = wristAt(state.pose, f, side);
    if (!w) {
      lines.push(`${side}-glove: <span class="muted">no detection</span>`);
      continue;
    }
    const sig = (w.y - expected) / torso.length;
    const label = classify(sig);
    const sigStr = `${sig >= 0 ? "+" : ""}${sig.toFixed(2)}`;
    let line = `${side}-glove (live): signal <code>${sigStr}</code> torso → <span style="color:${colorFor(label)}">${label.toUpperCase()}</span>`;
    lines.push(line);
  }

  lines.push(``);
  if (punch) {
    lines.push(`<span class="muted">active label:</span> <code>${punch.punch_type || "?"}</code> &nbsp;(stance <code>${punch.stance || "?"}</code>, frames ${punch.start_frame}–${punch.end_frame})`);
    const c = classifiedAtFrame(f);
    if (c) {
      if (c.label) {
        const col = colorFor(c.label);
        const sigStr = `${c.signal >= 0 ? "+" : ""}${c.signal.toFixed(2)}`;
        lines.push(`<span class="muted">lens verdict:</span> <span style="color:${col};font-weight:600">${c.label.toUpperCase()}</span> at peak <code>f${c.peakFrame}</code> (${c.hand}-glove, sig <code>${sigStr}</code>)`);
        if (gt) {
          lines.push(c.match
            ? ` &nbsp;<span style="color:${COLOR_HEAD}">✓ matches GT (${gt})</span>`
            : ` &nbsp;<span style="color:${COLOR_BODY}">✗ GT says ${gt}</span>`);
        }
        if (c.peakFrame !== f) {
          lines.push(`<a href="#" data-jump="${c.peakFrame}" style="color:${COLOR_STANDING}">→ jump to peak frame</a>`);
        } else {
          lines.push(`<span class="muted">(at peak frame)</span>`);
        }
      } else if (c.error) {
        lines.push(`<span class="muted">lens verdict:</span> <span class="bad">${c.error}</span>`);
      }
    }
  } else {
    lines.push(`<span class="muted">no labelled punch at this frame.</span>`);
  }

  el.innerHTML = lines.join("<br>");
}

function mountSummary() {
  const slot = document.getElementById("stage-extras");
  if (!slot) return;
  slot.innerHTML = `
    <div id="bs-summary-card" style="margin-top:12px;padding:10px 12px;background:var(--bg-card);border:1px solid var(--border);border-radius:8px">
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;gap:12px;flex-wrap:wrap">
        <span class="muted small">Per-punch verdicts — click row to jump to peak frame</span>
        <span id="bs-summary-stats" class="muted small"></span>
      </div>
      <div id="bs-summary-wrap" style="max-height:240px;overflow-y:auto"></div>
    </div>
  `;
  // Event delegation: click any row with data-peak to seek.
  slot.addEventListener("click", e => {
    const row = e.target.closest("tr[data-peak]");
    if (row) {
      const f = parseInt(row.dataset.peak, 10);
      if (Number.isFinite(f)) seekTo(f);
    }
  });
}

function renderSummary(state) {
  const wrap = document.getElementById("bs-summary-wrap");
  const stats = document.getElementById("bs-summary-stats");
  if (!wrap) return;
  if (!classifiedPunches.length) {
    wrap.innerHTML = `<div class="muted small">no punch detections loaded for this round.</div>`;
    if (stats) stats.textContent = "";
    return;
  }

  let nGt = 0, nMatch = 0;
  let nClassified = 0, nHead = 0, nBody = 0, nAmbig = 0, nErr = 0;
  const f = state.frame;

  const rows = classifiedPunches.map((c, i) => {
    const isActive = f >= c.punch.start_frame && f <= c.punch.end_frame;
    const isPeak   = c.peakFrame === f;
    const sigStr = c.signal != null ? `${c.signal >= 0 ? "+" : ""}${c.signal.toFixed(2)}` : "—";
    const verdictCol = c.label ? colorFor(c.label) : "var(--muted)";
    const verdictTxt = c.label ? c.label.toUpperCase() : (c.error ? "ERR" : "—");
    let matchCell = "<span class=\"muted\">—</span>";
    if (c.gt) {
      if (c.match === true) {
        matchCell = `<span style="color:${COLOR_HEAD}">✓ ${c.gt}</span>`;
      } else if (c.match === false) {
        matchCell = `<span style="color:${COLOR_BODY}">✗ want ${c.gt}</span>`;
      } else {
        matchCell = `<span class="muted">${c.gt} (ambig)</span>`;
      }
      nGt++;
      if (c.match === true) nMatch++;
    }
    if (c.label) {
      nClassified++;
      if (c.label === "head") nHead++;
      else if (c.label === "body") nBody++;
      else nAmbig++;
    } else if (c.error) {
      nErr++;
    }
    const bg = isPeak ? "background:rgba(58,217,224,0.18);" : (isActive ? "background:rgba(58,217,224,0.06);" : "");
    const seekTarget = c.peakFrame ?? c.punch.start_frame;
    const errTip = c.error ? ` title="${c.error}"` : "";
    return `
      <tr data-peak="${seekTarget}"${errTip} style="cursor:pointer;${bg}">
        <td style="padding:2px 6px;color:var(--muted)">${i + 1}</td>
        <td style="padding:2px 6px"><code>${c.punch.punch_type || "?"}</code></td>
        <td style="padding:2px 6px;color:var(--muted)">${c.punch.start_frame}–${c.punch.end_frame}</td>
        <td style="padding:2px 6px">${c.peakFrame ?? "—"}</td>
        <td style="padding:2px 6px">${c.hand || "—"}</td>
        <td style="padding:2px 6px"><code>${sigStr}</code></td>
        <td style="padding:2px 6px;color:${verdictCol};font-weight:600">${verdictTxt}</td>
        <td style="padding:2px 6px">${matchCell}</td>
      </tr>
    `;
  });

  wrap.innerHTML = `
    <table style="width:100%;border-collapse:collapse;font:11px ui-monospace,'SF Mono',monospace">
      <thead style="position:sticky;top:0;background:var(--bg-card);box-shadow:0 1px 0 var(--border)">
        <tr style="color:var(--muted);text-align:left">
          <th style="padding:4px 6px">#</th>
          <th style="padding:4px 6px">punch_type</th>
          <th style="padding:4px 6px">range</th>
          <th style="padding:4px 6px">peak</th>
          <th style="padding:4px 6px">hand</th>
          <th style="padding:4px 6px">signal</th>
          <th style="padding:4px 6px">verdict</th>
          <th style="padding:4px 6px">vs GT</th>
        </tr>
      </thead>
      <tbody>${rows.join("")}</tbody>
    </table>
  `;

  if (stats) {
    const acc = nGt > 0
      ? `<strong style="color:${nMatch/nGt>=0.8?COLOR_HEAD:nMatch/nGt>=0.6?COLOR_AMBIG:COLOR_BODY}">${(nMatch / nGt * 100).toFixed(0)}%</strong> (${nMatch}/${nGt})`
      : `<span class="muted">no _head/_body GT in labels</span>`;
    stats.innerHTML =
      `${classifiedPunches.length} punches · ` +
      `head ${nHead} · body ${nBody} · ambig ${nAmbig}` +
      (nErr ? ` · err ${nErr}` : "") +
      ` · GT acc: ${acc}`;
  }
}

export const BodyShotLensRule = {
  id: "bodyshot_lens",
  label: "Body-shot detector (ankle-anchored)",

  mount(_host, state) {
    host = _host;
    host.innerHTML = `
      <h2>Body-shot detector</h2>
      <p class="hint">
        Post-hoc geometric head-vs-body split for a punch classifier that
        only predicts punch <em>type</em>. Primary mode anchors to the floor
        via ankles (so a deep duck doesn't pull the chest reference down)
        and normalizes by current torso length (camera-distance invariant).
        Per-punch classification uses peak arm extension within each
        detection window. Falls back to hip-anchored when ankles aren't
        visible (can't detect ducks in fallback mode — line goes dashed).
      </p>
      <p class="hint">
        Overlays:
        <span style="color:${COLOR_STANDING}">●</span> standing chest line (solid = ankle, dashed = hip fallback) ·
        <span style="color:${COLOR_CURRENT}">●</span> current chest ·
        <span style="color:${COLOR_FLOOR}">●</span> anchor line.
        Gloves: <span style="color:${COLOR_HEAD}">●</span> head ·
        <span style="color:${COLOR_AMBIG}">●</span> ambiguous ·
        <span style="color:${COLOR_BODY}">●</span> body.
        Per-punch verdict + banner appear when a labelled punch is active.
      </p>
      <div id="bs-state" class="hint" style="line-height:1.55"></div>
      <p class="hint" style="margin-top:14px;font-size:11px">
        Thresholds τ_head=${HEAD_THRESHOLD}, τ_body=${BODY_THRESHOLD} (torso units).
        Calibration percentile p${(CALIB_PERCENTILE*100).toFixed(0)} + boost ${STANDING_RATIO_BOOST.toFixed(2)}.
        Tweak constants at top of <code>bodyshot_lens.js</code>.
      </p>
    `;
    // Wire jump-to-peak links in the sidebar via delegation.
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
    const W = ctx.canvas.width;
    const s = state.renderScale || 1;
    const f = state.frame;

    const info = dividerAt(state.pose, f);
    if (!info) return;
    const { expected, torso, ankle, anchor } = info;

    ctx.save();

    // Anchor reference line (dashed) at floor (ankle mode) or hip line (fallback).
    ctx.strokeStyle = COLOR_FLOOR;
    ctx.lineWidth = 1.5 * s;
    ctx.setLineDash([4 * s, 6 * s]);
    const anchorY = ankle ? ankle.y : torso.hip.y;
    ctx.beginPath(); ctx.moveTo(0, anchorY); ctx.lineTo(W, anchorY); ctx.stroke();
    ctx.setLineDash([]);

    // Standing chest line — solid in ankle mode, dashed in hip-fallback mode.
    ctx.strokeStyle = COLOR_STANDING;
    ctx.lineWidth = 2.5 * s;
    if (anchor === "hip") ctx.setLineDash([10 * s, 5 * s]);
    ctx.beginPath(); ctx.moveTo(0, expected); ctx.lineTo(W, expected); ctx.stroke();
    if (anchor === "hip") ctx.setLineDash([]);

    // Current chest dot + thin line.
    ctx.strokeStyle = COLOR_CURRENT;
    ctx.fillStyle = COLOR_CURRENT;
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 1.2 * s;
    ctx.beginPath(); ctx.moveTo(0, torso.chest.y); ctx.lineTo(W, torso.chest.y); ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.beginPath(); ctx.arc(torso.chest.x, torso.chest.y, 4 * s, 0, Math.PI * 2); ctx.fill();

    // Live glove markers — both hands, colored by current-frame signal.
    ctx.font = `bold ${Math.round(11 * s)}px ui-monospace, "SF Mono", monospace`;
    for (const side of ["L", "R"]) {
      const w = wristAt(state.pose, f, side);
      if (!w) continue;
      const sig = (w.y - expected) / torso.length;
      const label = classify(sig);
      const col = colorFor(label);

      ctx.strokeStyle = col;
      ctx.globalAlpha = 0.50;
      ctx.lineWidth = 1.5 * s;
      ctx.beginPath(); ctx.moveTo(w.x, w.y); ctx.lineTo(w.x, expected); ctx.stroke();
      ctx.globalAlpha = 1;

      ctx.fillStyle = col;
      ctx.beginPath(); ctx.arc(w.x, w.y, 9 * s, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = "rgba(0,0,0,0.65)";
      ctx.lineWidth = 1.5 * s; ctx.stroke();

      const txt = `${side} ${sig >= 0 ? "+" : ""}${sig.toFixed(2)} ${label}`;
      const tw = ctx.measureText(txt).width;
      const tx = w.x + 12 * s;
      const ty = w.y - 8 * s;
      ctx.fillStyle = "rgba(0,0,0,0.70)";
      ctx.fillRect(tx, ty, tw + 8 * s, 16 * s);
      ctx.fillStyle = col;
      ctx.fillText(txt, tx + 4 * s, ty + 12 * s);
    }

    // Per-punch overlays (banner + peak emphasis) — only when a labelled
    // punch is active.
    const c = classifiedAtFrame(f);
    if (c) {
      // Big ring around the throwing wrist when we're AT the peak frame, so
      // it's visible where the lens decided.
      if (c.peakFrame === f && c.label && c.wristAtPeak) {
        const col = colorFor(c.label);
        ctx.save();
        ctx.strokeStyle = col;
        ctx.lineWidth = 2.5 * s;
        ctx.beginPath();
        ctx.arc(c.wristAtPeak.x, c.wristAtPeak.y, 17 * s, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 0.30;
        ctx.beginPath();
        ctx.arc(c.wristAtPeak.x, c.wristAtPeak.y, 24 * s, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      // Top-center verdict banner.
      ctx.save();
      const fontPx = Math.round(15 * s);
      ctx.font = `bold ${fontPx}px ui-monospace, "SF Mono", monospace`;
      const verdict = c.label ? c.label.toUpperCase() : (c.error || "—");
      const sigPart = c.signal != null ? ` sig ${c.signal >= 0 ? "+" : ""}${c.signal.toFixed(2)}` : "";
      const peakPart = c.peakFrame != null ? ` @ peak f${c.peakFrame}` : "";
      const handPart = c.hand ? ` (${c.hand})` : "";
      let matchPart = "";
      if (c.gt) {
        if (c.match === true)  matchPart = `  ✓ ${c.gt}`;
        else if (c.match === false) matchPart = `  ✗ GT ${c.gt}`;
        else                   matchPart = `  GT ${c.gt}`;
      }
      const text = `${c.punch.punch_type || "punch"} → ${verdict}${peakPart}${handPart}${sigPart}${matchPart}`;
      const tw = ctx.measureText(text).width;
      const padX = 12 * s;
      const bannerH = (fontPx + 14) * s / s; // pixels; clarity over compression
      const bh = fontPx + 14 * s;
      const bw = tw + padX * 2;
      const bx = (W - bw) / 2;
      const by = 6 * s;
      ctx.fillStyle = "rgba(0,0,0,0.78)";
      ctx.fillRect(bx, by, bw, bh);
      ctx.strokeStyle = colorFor(c.label || "ambig");
      ctx.lineWidth = 2 * s;
      ctx.strokeRect(bx + 0.5, by + 0.5, bw - 1, bh - 1);
      ctx.fillStyle = colorFor(c.label || "ambig");
      ctx.fillText(text, bx + padX, by + fontPx + 4 * s);
      ctx.restore();
    }

    ctx.restore();
  },
};
