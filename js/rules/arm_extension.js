// Arm extension lens — "did the straight punch reach full extension?".
//
// Per-frame metric (both arms, every frame):
//
//   r[f] = |shoulder→wrist|  /  ( |shoulder→elbow| + |elbow→wrist| )
//
// Domain:
//   1.00  arm dead straight (elbow on the shoulder→wrist line)
//   0.95  ~25° bend at the elbow
//   0.71  90° bend
//   →0    arm fully folded
//
// Per labelled punch (straights only — jab/cross head/body):
//   peak       = max(r[f]) inside the punch window
//   predicted  = peak >= threshold ? "pass" : "fail"
//
// Wrist source: the glove detector when present and conf ≥ 0.20, else the
// pose model wrist (with its own conf gate). Anatomical side maps from
// (hand, stance) the same way guard_drop / step_punch_sync do.
//
// Compares predicted vs the labeler's rule_extension verdict when
// available — same agree/disagree pattern as the hip_rotation lens.

import { J } from "../skeleton.js";
import { gloveXY, gloveConf } from "../pose-loader.js";

const DEFAULTS = {
  threshold:     0.95,        // pass if peak ratio ≥ this
  minGloveConf:  0.20,
  minPoseConf:   0.20,
  // Only straights have a "should be extended" goal — hooks/uppercuts
  // are supposed to be bent at the elbow.
  appliesTo: new Set([
    "jab_head", "jab_body",
    "cross_head", "cross_body",
  ]),
};

const COLORS = {
  pass:        "#5fd97a",
  fail:        "#e85a5a",
  unclear:     "#f5b945",
  ratioGuide:  "rgba(255,255,255,0.45)",
  poseArm:     "rgba(255,200,80,0.85)",
  gloveArm:    "#d68bff",
  agree:       "#5fd97a",
  disagree:    "#e85a5a",
};

// (hand, stance) → anatomical side, mirroring guard_drop.GUARD_JOINTS logic.
const SIDE_FOR = {
  lead: { orthodox: "L", southpaw: "R" },
  rear: { orthodox: "R", southpaw: "L" },
};

const JOINTS_FOR_SIDE = {
  L: { shoulder: J.L_SHOULDER, elbow: J.L_ELBOW, wrist: J.L_WRIST, gloveSide: 0 },
  R: { shoulder: J.R_SHOULDER, elbow: J.R_ELBOW, wrist: J.R_WRIST, gloveSide: 1 },
};

let host;
let cfg = { ...DEFAULTS };
let signals = null;
let lastPose = null;

export const ArmExtensionRule = {
  id: "arm_extension",
  label: "Arm extension (straights)",

  requires(slot) {
    // Needs a pose cache AND a glove sidecar. Without the glove the 2D
    // wrist is too jittery for the ratio to mean anything — that's the
    // whole reason this lens exists.
    return !!(slot?.vision || slot?.yolo) && !!slot?.glove;
  },

  skeletonStyle() {
    return {
      boneColor: "rgba(255,255,255,0.22)",
      boneWidth: 1.5,
      jointRadius: 3,
      // Highlight the joints we use for the ratio, EXCEPT wrists — this lens
      // draws its own wrist markers (square = glove, ring = pose) so letting
      // the base renderer also paint an amber confidence dot at the same
      // spot creates visual noise. Same approach as the wrist_swap lens.
      hideJoints: new Set([J.L_WRIST, J.R_WRIST]),
      highlightJoints: new Set([
        J.L_SHOULDER, J.R_SHOULDER, J.L_ELBOW, J.R_ELBOW,
      ]),
    };
  },

  mount(_host, state) {
    host = _host;
    cfg = { ...DEFAULTS };
    signals = computeAll(state, cfg);
    lastPose = state.pose;

    host.innerHTML = renderTemplate(signals, cfg);
    renderPunchTable();
    renderAggregate();

    // Threshold slider
    const slider = host.querySelector("#ae-threshold");
    const out    = host.querySelector("#ae-threshold-out");
    if (slider) {
      slider.addEventListener("input", () => {
        cfg.threshold = Number(slider.value);
        out.textContent = cfg.threshold.toFixed(2);
        // Re-score punches against the new threshold — fast, no recompute of r[f]
        for (const p of signals.punches) {
          p.predicted = p.peak >= cfg.threshold ? "pass" : "fail";
        }
        renderPunchTable();
        renderAggregate();
        state.requestDraw?.();
      });
    }

    // Click a punch row to seek. Same scrubber-dispatch trick the other
    // lenses use — keeps the seek path single-sourced through the existing
    // input handler instead of inventing a new API.
    host.addEventListener("click", (ev) => {
      const tr = ev.target.closest("tr[data-frame]");
      if (!tr) return;
      const f = Number(tr.dataset.frame);
      const slider = document.getElementById("scrubber");
      if (!slider) return;
      slider.value = String(f);
      slider.dispatchEvent(new Event("input"));
    });
  },

  draw(ctx, state) {
    if (state.pose !== lastPose) {
      // Pose swapped (e.g. compare-mode toggle) — recompute.
      signals = computeAll(state, cfg);
      lastPose = state.pose;
    }
    const s = state.renderScale || 1;
    const f = state.frame;

    // Draw both arms with their current ratio overlay.
    drawArmRatio(ctx, state.pose, f, "L", signals.ratioL[f], cfg, s);
    drawArmRatio(ctx, state.pose, f, "R", signals.ratioR[f], cfg, s);
  },

  update(state) {
    if (state.pose !== lastPose) {
      signals = computeAll(state, cfg);
      lastPose = state.pose;
    }
    const f = state.frame;
    setMetric("ae-l-ratio", signals.ratioL[f], cfg);
    setMetric("ae-r-ratio", signals.ratioR[f], cfg);
    setText("ae-l-source", signals.sourceL[f] || "—");
    setText("ae-r-source", signals.sourceR[f] || "—");
  },
};

// ─── compute ───────────────────────────────────────────────────────────────

function computeAll(state, cfg) {
  const pose = state.pose;
  const N = pose.n_frames;
  const fps = pose.fps;
  // Source-video time offset — added to the label's cache-relative
  // timestamp so the per-punch table matches the playback timebar
  // (which uses absolute source time).
  const startSec = pose.start_sec || 0;

  const { ratio: ratioL, source: sourceL } = perFrameRatio(pose, "L", cfg);
  const { ratio: ratioR, source: sourceR } = perFrameRatio(pose, "R", cfg);

  // Labelled punches — filtered to straights, with optional rule_extension
  // verdict for agreement scoring.
  const detections = (state.labels?.detections || []).filter(d =>
    cfg.appliesTo.has(d.punch_type)
  );
  const punches = detections.map((d, idx) => {
    const sf = Math.max(0, d.start_frame);
    const ef = Math.min(N - 1, d.end_frame);

    const stance = (d.stance === "southpaw" || d.stance === "orthodox")
      ? d.stance : "orthodox";
    const side = SIDE_FOR[d.hand]?.[stance] || "L";
    const ratioArr = side === "L" ? ratioL : ratioR;
    const srcArr   = side === "L" ? sourceL : sourceR;

    let peak = -Infinity, peakFrame = sf, gloveFrames = 0, validFrames = 0;
    for (let f = sf; f <= ef; f++) {
      const r = ratioArr[f];
      if (!Number.isFinite(r)) continue;
      validFrames++;
      if (srcArr[f] === "glove") gloveFrames++;
      if (r > peak) { peak = r; peakFrame = f; }
    }
    const peakValid = Number.isFinite(peak);
    const predicted = peakValid
      ? (peak >= cfg.threshold ? "pass" : "fail")
      : "unclear";

    const label = d.rule_extension === "pass" || d.rule_extension === "fail"
      ? d.rule_extension : null;

    return {
      idx,
      timestamp: d.timestamp,
      t_abs: startSec + (Number.isFinite(d.timestamp) ? d.timestamp : 0),
      hand: d.hand,
      stance,
      side,
      punch_type: d.punch_type,
      start_frame: sf,
      end_frame: ef,
      land_frame: peakValid ? peakFrame : sf,
      peak: peakValid ? peak : NaN,
      glove_coverage: validFrames ? gloveFrames / validFrames : 0,
      predicted,
      label,
    };
  });

  return { ratioL, ratioR, sourceL, sourceR, punches, fps };
}

function perFrameRatio(pose, side, cfg) {
  const N = pose.n_frames;
  const joints = JOINTS_FOR_SIDE[side];
  const ratio  = new Float32Array(N);
  const source = new Array(N);
  for (let f = 0; f < N; f++) {
    const w = wristXY(pose, f, joints, cfg);
    if (!w) { ratio[f] = NaN; source[f] = null; continue; }

    const sc = pose.conf[f * 17 + joints.shoulder];
    const ec = pose.conf[f * 17 + joints.elbow];
    if (sc < cfg.minPoseConf || ec < cfg.minPoseConf) {
      ratio[f] = NaN; source[f] = null; continue;
    }
    const sx = pose.skeleton[(f * 17 + joints.shoulder) * 2];
    const sy = pose.skeleton[(f * 17 + joints.shoulder) * 2 + 1];
    const ex = pose.skeleton[(f * 17 + joints.elbow) * 2];
    const ey = pose.skeleton[(f * 17 + joints.elbow) * 2 + 1];

    const ue = Math.hypot(sx - ex, sy - ey);          // shoulder→elbow
    const fa = Math.hypot(ex - w.x, ey - w.y);        // elbow→wrist
    const sw = Math.hypot(sx - w.x, sy - w.y);        // shoulder→wrist
    const path = ue + fa;
    if (path < 1e-3) { ratio[f] = NaN; source[f] = null; continue; }
    // Bounded [0,1]. Clamp tiny float overshoots that can happen when the
    // wrist is collinear with shoulder–elbow.
    ratio[f] = Math.min(1, sw / path);
    source[f] = w.source;
  }
  return { ratio, source };
}

function wristXY(pose, frame, joints, cfg) {
  // Glove wrist first if the cache attached gloveWrists.
  const g = pose.gloveWrists;
  if (g) {
    const [gx, gy] = gloveXY(g, frame, joints.gloveSide);
    const gc       = gloveConf(g, frame, joints.gloveSide);
    if (gc >= cfg.minGloveConf && Number.isFinite(gx) && Number.isFinite(gy)) {
      return { x: gx, y: gy, source: "glove" };
    }
  }
  // Pose fallback.
  const px = pose.skeleton[(frame * 17 + joints.wrist) * 2];
  const py = pose.skeleton[(frame * 17 + joints.wrist) * 2 + 1];
  const pc = pose.conf[frame * 17 + joints.wrist];
  if (pc < cfg.minPoseConf || !Number.isFinite(px)) return null;
  return { x: px, y: py, source: "pose" };
}

// ─── render ────────────────────────────────────────────────────────────────

function renderTemplate(sig, cfg) {
  const hasLabels = sig.punches.some(p => p.label !== null);
  return `
    <h2>Arm extension (straights)</h2>
    <p class="hint">
      <code>r = |shoulder→wrist| / (|shoulder→elbow| + |elbow→wrist|)</code>.
      1.00 = dead straight, 0.95 ≈ 25° bend, 0.71 = 90° bend, 0 = folded.
    </p>

    <h3>Legend</h3>
    <ul class="hint" style="list-style:none;padding-left:0;margin:0 0 12px 0;line-height:1.7">
      <li><span style="display:inline-block;width:24px;height:3px;background:${COLORS.pass};vertical-align:middle"></span>
        &nbsp;arm bones (sh→el→wr) when ratio ≥ threshold</li>
      <li><span style="display:inline-block;width:24px;height:3px;background:${COLORS.fail};vertical-align:middle"></span>
        &nbsp;arm bones when ratio &lt; threshold</li>
      <li><span style="display:inline-block;width:24px;height:1px;background:${COLORS.ratioGuide};border-top:1px dashed ${COLORS.ratioGuide};vertical-align:middle"></span>
        &nbsp;dashed shoulder→wrist guide (the "if fully extended" line)</li>
      <li><span style="display:inline-block;width:12px;height:12px;background:rgba(0,0,0,0.55);border:2px solid ${COLORS.gloveArm};vertical-align:middle"></span>
        &nbsp;wrist from the <b>glove detector</b> (conf ≥ ${cfg.minGloveConf})</li>
      <li><span style="display:inline-block;width:14px;height:14px;border:2px solid ${COLORS.poseArm};border-radius:50%;vertical-align:middle"></span>
        &nbsp;wrist from the <b>pose model</b> (fallback when glove missing/low-conf)</li>
      <li><span style="color:${COLORS.pass};font-family:monospace">0.97</span>
        / <span style="color:${COLORS.fail};font-family:monospace">0.82</span>
        &nbsp;ratio readout next to each wrist, colored by pass/fail vs threshold</li>
    </ul>

    <h3>Live ratio</h3>
    <div class="metric-grid">
      <div class="metric">
        <div class="metric-label">L arm</div>
        <div class="metric-val" id="ae-l-ratio">—</div>
        <div class="metric-sub" id="ae-l-source">—</div>
      </div>
      <div class="metric">
        <div class="metric-label">R arm</div>
        <div class="metric-val" id="ae-r-ratio">—</div>
        <div class="metric-sub" id="ae-r-source">—</div>
      </div>
    </div>

    <h3>Pass threshold</h3>
    <div class="slider-row">
      <input type="range" id="ae-threshold" min="0.70" max="1.00" step="0.01" value="${cfg.threshold}" />
      <output id="ae-threshold-out">${cfg.threshold.toFixed(2)}</output>
      <span class="muted small">peak ratio per punch must reach this to pass</span>
    </div>

    <h3>Per-punch (straights only)</h3>
    <p class="hint">
      Peak = max ratio inside the punch window. Clickable rows seek to the
      peak frame. ${hasLabels
        ? "Verdict shows labeler vs predicted; ✓ when they agree."
        : "<span class='muted'>No <code>rule_extension</code> labels found — predicted only.</span>"}
    </p>
    <div id="ae-table-host"></div>

    <h3>Aggregate</h3>
    <div id="ae-aggregate" class="metric-grid"></div>
  `;
}

function renderPunchTable() {
  const tbody = signals.punches.length
    ? signals.punches.map(p => {
        const peakStr = Number.isFinite(p.peak) ? p.peak.toFixed(3) : "—";
        const predCol = p.predicted === "pass" ? COLORS.pass
                      : p.predicted === "fail" ? COLORS.fail
                      : COLORS.unclear;
        let verdict;
        if (p.label && Number.isFinite(p.peak)) {
          const agrees = p.label === p.predicted;
          const sym = agrees ? "✓" : "✗";
          const col = agrees ? COLORS.agree : COLORS.disagree;
          verdict = `<span style="color:${col}">label=${p.label} · pred=${p.predicted} ${sym}</span>`;
        } else {
          verdict = `<span style="color:${predCol}">pred=${p.predicted}</span>`;
        }
        const tsStr = Number.isFinite(p.t_abs) ? p.t_abs.toFixed(2) : "—";
        const cov = Number.isFinite(p.glove_coverage)
          ? `${Math.round(100 * p.glove_coverage)}% glove` : "—";
        return `<tr data-frame="${p.land_frame}" style="cursor:pointer">
          <td>${tsStr}s</td>
          <td>${p.punch_type}</td>
          <td>${p.hand}</td>
          <td>${p.side}</td>
          <td>${peakStr}</td>
          <td>${verdict}</td>
          <td class="muted small">${cov}</td>
        </tr>`;
      }).join("")
    : `<tr><td colspan="7" class="muted">no labeled straights in this round</td></tr>`;

  const tableHost = host.querySelector("#ae-table-host");
  if (tableHost) {
    tableHost.innerHTML = `
      <table class="rule-table">
        <thead><tr>
          <th>t</th><th>type</th><th>hand</th><th>side</th>
          <th>peak r</th><th>verdict</th><th>source</th>
        </tr></thead>
        <tbody>${tbody}</tbody>
      </table>`;
  }
}

function renderAggregate() {
  const host_ = host.querySelector("#ae-aggregate");
  if (!host_) return;
  const ps = signals.punches.filter(p => Number.isFinite(p.peak));
  if (!ps.length) {
    host_.innerHTML = `<div class="metric muted">no scorable punches</div>`;
    return;
  }
  const meanPeak = ps.reduce((s, p) => s + p.peak, 0) / ps.length;
  const passed = ps.filter(p => p.predicted === "pass").length;
  const labelled = ps.filter(p => p.label);
  const agree = labelled.filter(p => p.label === p.predicted).length;
  const agreePct = labelled.length
    ? `${Math.round(100 * agree / labelled.length)}%`
    : "—";
  host_.innerHTML = `
    <div class="metric">
      <div class="metric-label">scored</div>
      <div class="metric-val">${ps.length}</div>
      <div class="metric-sub">of ${signals.punches.length} labelled straights</div>
    </div>
    <div class="metric">
      <div class="metric-label">predicted pass</div>
      <div class="metric-val">${passed}</div>
      <div class="metric-sub">${Math.round(100 * passed / ps.length)}% reach ≥ ${cfg.threshold.toFixed(2)}</div>
    </div>
    <div class="metric">
      <div class="metric-label">mean peak r</div>
      <div class="metric-val">${meanPeak.toFixed(3)}</div>
    </div>
    <div class="metric">
      <div class="metric-label">label agreement</div>
      <div class="metric-val">${agreePct}</div>
      <div class="metric-sub">${agree} / ${labelled.length} match</div>
    </div>
  `;
}

// ─── draw ──────────────────────────────────────────────────────────────────

function drawArmRatio(ctx, pose, frame, side, ratio, cfg, scale) {
  const joints = JOINTS_FOR_SIDE[side];
  const sx = pose.skeleton[(frame * 17 + joints.shoulder) * 2];
  const sy = pose.skeleton[(frame * 17 + joints.shoulder) * 2 + 1];
  const ex = pose.skeleton[(frame * 17 + joints.elbow) * 2];
  const ey = pose.skeleton[(frame * 17 + joints.elbow) * 2 + 1];
  const w = wristXY(pose, frame, joints, cfg);
  if (!w) return;
  if (!Number.isFinite(sx) || !Number.isFinite(ex)) return;

  const color = !Number.isFinite(ratio) ? COLORS.unclear
              : ratio >= cfg.threshold ? COLORS.pass
              : COLORS.fail;

  ctx.save();
  ctx.lineWidth = 4 * scale;
  ctx.strokeStyle = color;
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.lineTo(ex, ey);
  ctx.lineTo(w.x, w.y);
  ctx.stroke();

  // Direct shoulder→wrist guide (dashed) — visualises the "if extended" line
  ctx.strokeStyle = COLORS.ratioGuide;
  ctx.lineWidth = 1.5 * scale;
  ctx.setLineDash([5 * scale, 4 * scale]);
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.lineTo(w.x, w.y);
  ctx.stroke();
  ctx.setLineDash([]);

  // Wrist marker — square if glove, ring if pose. Same shape vocabulary as
  // the wrist_swap lens.
  if (w.source === "glove") {
    const s_ = 8 * scale;
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(w.x - s_, w.y - s_, s_ * 2, s_ * 2);
    ctx.strokeStyle = COLORS.gloveArm;
    ctx.lineWidth = 3 * scale;
    ctx.strokeRect(w.x - s_, w.y - s_, s_ * 2, s_ * 2);
  } else {
    ctx.strokeStyle = COLORS.poseArm;
    ctx.lineWidth = 3 * scale;
    ctx.beginPath();
    ctx.arc(w.x, w.y, 9 * scale, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Ratio readout next to the wrist
  if (Number.isFinite(ratio)) {
    ctx.font = `${14 * scale}px ui-monospace, monospace`;
    ctx.fillStyle = "rgba(0,0,0,0.7)";
    ctx.fillRect(w.x + 12 * scale, w.y - 16 * scale, 56 * scale, 22 * scale);
    ctx.fillStyle = color;
    ctx.fillText(ratio.toFixed(2), w.x + 18 * scale, w.y + 1 * scale);
  }
  ctx.restore();
}

// ─── DOM helpers ───────────────────────────────────────────────────────────

function setText(id, value) {
  const el = host?.querySelector("#" + id);
  if (el) el.textContent = value;
}

function setMetric(id, ratio, cfg) {
  const el = host?.querySelector("#" + id);
  if (!el) return;
  if (!Number.isFinite(ratio)) {
    el.textContent = "—";
    el.style.color = "";
    return;
  }
  el.textContent = ratio.toFixed(3);
  el.style.color = ratio >= cfg.threshold ? COLORS.pass : COLORS.fail;
}

