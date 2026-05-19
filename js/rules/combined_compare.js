// Combined-pose lens — overlays raw Apple Vision wrists against the
// vision+glove "combined" pose cache (pose_cache_v5/ and successors) so we
// can validate the wrist substitution before pointing production at it.
//
// By construction the combined cache is the Vision skeleton with wrists 9
// and 10 replaced by the glove model on confident frames. All non-wrist
// joints are identical; the lens focuses on the wrists.
//
// Requires:
//   - slot.vision           raw apple_vision_pose_cache entry
//   - slot.vision_combined  pose_cache_v*/ entry (auto-tagged by the picker
//                           based on the folder name)
//
// Three modes:
//   "vision"   — render only the raw Vision wrists (baseline)
//   "combined" — render only the combined cache's wrists (production preview)
//   "both"     — render both with a faint line where they diverge

import { J } from "../skeleton.js";

const VISION_COL = "#5fd1ff";
const COMBINED_COL = "#34c759";
const DIVERGENCE_COL = "rgba(255,255,255,0.6)";

let host;
let mode = "both";
// Detected per-round frame lag of the combined cache vs raw vision. Recomputed
// in mount() because the combined meta (pose_cache_v5/) doesn't propagate the
// notebook's `frame_lag_vs_vision` value. lag = +1 → combined samples one
// physical frame AHEAD of vision (i.e. green appears one frame in vision's
// future), lag = -1 → combined samples one frame behind.
let detectedLag = 0;
let lagCompensate = false;

export const CombinedCompareRule = {
  id: "combined_compare",
  label: "Combined pose (Vision vs Vision+glove)",

  requires(slot) { return !!slot?.vision && !!slot?.vision_combined; },

  // Suppress wrists in the base skeleton renderer (and the forearm bones
  // touching them) — we'll redraw both wrist sources ourselves in the
  // active mode's colour.
  skeletonStyle() {
    return { hideJoints: new Set([J.L_WRIST, J.R_WRIST]) };
  },

  mount(_host, state) {
    host = _host;
    const v = state.pose, c = state.poseCombined;
    const ok = !!(v && c);
    const cmeta = c?.meta || {};
    const run = cmeta.wrist_run || cmeta.run || "unknown";
    const replaced = cmeta.wrists_replaced;
    // Detect frame lag client-side: cross-correlate L+R wrist-y between
    // vision and combined. Matches the notebook's check_alignment() so the
    // value displayed here is the same one §5 prints. We do this in the lens
    // because pose_cache_v5/'s meta doesn't carry `frame_lag_vs_vision`.
    detectedLag = ok ? detectFrameLag(v, c, 5) : 0;
    lagCompensate = false;     // start unchecked so the bug is visible
    const lagColor = detectedLag === 0 ? "var(--ok, #5fd97a)" : "var(--bad, #ff6b6b)";
    const lagSign = detectedLag > 0 ? "+" : "";
    host.innerHTML = `
      <h2>Combined pose — Vision vs Vision+glove</h2>
      ${ok ? "" : `<p class="hint" style="color:var(--bad)">
        Need BOTH a raw <code>apple_vision_pose_cache/</code> entry and a
        <code>pose_cache_v*/</code> entry for this round. Connect a Drive
        folder containing both, or pick the manual cache folder twice.
      </p>`}
      <p class="hint">
        <span style="color:${VISION_COL}">cyan</span> = raw Apple Vision wrist,
        <span style="color:${COMBINED_COL}">green</span> = combined cache wrist
        (what production reads).
        In <b>Both</b> mode, a faint white line connects the pair when they disagree.
      </p>
      <p class="hint">Combined run: <code>${run}</code></p>
      ${replaced ? `<p class="hint">Wrists replaced this round:
        L = <b>${replaced.L}</b> / ${replaced.T} frames,
        R = <b>${replaced.R}</b> / ${replaced.T} frames</p>` : ""}
      <p class="hint" style="border-left:3px solid ${lagColor};padding-left:8px;margin:10px 0">
        <b>Frame lag vs vision: <span style="color:${lagColor}">${lagSign}${detectedLag}</span></b>
        ${detectedLag === 0
          ? " — wrist samples align with vision."
          : ` — combined wrist samples ${detectedLag > 0 ? "<b>lead</b>" : "<b>trail</b>"}
            vision by ${Math.abs(detectedLag)} frame${Math.abs(detectedLag) === 1 ? "" : "s"}.
            The cache itself is misaligned; the fix is in <code>glove_wrist_cache_build.ipynb</code>
            §5 (ffmpeg re-encode timing) — not the lens.`}
      </p>
      <label class="hint" style="display:flex;align-items:center;gap:6px;margin:6px 0 12px;cursor:pointer">
        <input type="checkbox" id="cc-lag-comp"> Snap green to vision frame
        <span class="muted">(visual only — doesn't change what production reads)</span>
      </label>
      <div class="rule-pick" style="display:flex;gap:6px;flex-wrap:wrap;margin:8px 0 14px">
        <button data-mode="vision"   class="cc-mode">Vision wrists</button>
        <button data-mode="combined" class="cc-mode">Combined wrists</button>
        <button data-mode="both"     class="cc-mode">Both side-by-side</button>
      </div>
      <h3>Per-frame Δ (px)</h3>
      <p class="hint">Distance between raw Vision's wrist and the combined cache's
        wrist this frame. Conf shown as <code>V / C</code>.</p>
      <div class="metric-grid">
        <div class="metric"><div class="metric-label">L wrist</div><div class="metric-val" id="cc-l">—</div><div class="metric-sub" id="cc-l-conf">—</div></div>
        <div class="metric"><div class="metric-label">R wrist</div><div class="metric-val" id="cc-r">—</div><div class="metric-sub" id="cc-r-conf">—</div></div>
      </div>
      <h3>Coverage (this round)</h3>
      <p class="hint">% of frames where each source has a usable wrist (conf > 0.2),
        and how many frames the combined cache changes vs raw Vision (> 4 px).</p>
      <div class="metric-grid">
        <div class="metric"><div class="metric-label">L · Vision</div><div class="metric-val" id="cc-cov-vl">—</div></div>
        <div class="metric"><div class="metric-label">L · Combined</div><div class="metric-val" id="cc-cov-cl">—</div></div>
        <div class="metric"><div class="metric-label">R · Vision</div><div class="metric-val" id="cc-cov-vr">—</div></div>
        <div class="metric"><div class="metric-label">R · Combined</div><div class="metric-val" id="cc-cov-cr">—</div></div>
      </div>
      <p class="metric-sub" id="cc-diff">—</p>
    `;

    // Wire lag-compensate toggle. Off by default so the cache misalignment
    // is visible — the lens's job is to expose problems, not to mask them.
    const lagCheckbox = host.querySelector("#cc-lag-comp");
    if (lagCheckbox) {
      lagCheckbox.addEventListener("change", () => {
        lagCompensate = lagCheckbox.checked;
        if (state.requestDraw) state.requestDraw();
      });
    }

    // Wire mode buttons (same pattern as wrist_swap)
    for (const btn of host.querySelectorAll(".cc-mode")) {
      btn.style.cursor = "pointer";
      btn.style.padding = "5px 10px";
      btn.style.borderRadius = "4px";
      btn.style.border = "1px solid var(--muted, #555)";
      btn.style.background = "transparent";
      btn.style.color = "inherit";
      btn.addEventListener("click", () => {
        mode = btn.dataset.mode;
        for (const b of host.querySelectorAll(".cc-mode")) {
          b.style.background = (b.dataset.mode === mode) ? "var(--accent, #4da6ff)" : "transparent";
          b.style.color      = (b.dataset.mode === mode) ? "white" : "inherit";
        }
        if (state.requestDraw) state.requestDraw();
      });
      if (btn.dataset.mode === mode) {
        btn.style.background = "var(--accent, #4da6ff)";
        btn.style.color = "white";
      }
    }

    // Pre-compute coverage + divergence counts once per round load. The
    // divergence count uses lag-corrected indexing so a frame-shift bug in
    // the cache doesn't inflate it — we want "frames where the model
    // *actually* disagreed", not "frames where the comparison was misaligned".
    if (ok) {
      const N = Math.min(v.n_frames, c.n_frames);
      let vl=0, vr=0, cl=0, cr=0, dL=0, dR=0;
      const diff = (jointIdx, f, cf) => {
        const dx = v.skeleton[(f*17+jointIdx)*2]     - c.skeleton[(cf*17+jointIdx)*2];
        const dy = v.skeleton[(f*17+jointIdx)*2 + 1] - c.skeleton[(cf*17+jointIdx)*2 + 1];
        return Math.hypot(dx, dy);
      };
      for (let f = 0; f < N; f++) {
        const cvl = v.conf[f * 17 + J.L_WRIST];
        const cvr = v.conf[f * 17 + J.R_WRIST];
        const ccl = c.conf[f * 17 + J.L_WRIST];
        const ccr = c.conf[f * 17 + J.R_WRIST];
        if (cvl > 0.2) vl++;
        if (cvr > 0.2) vr++;
        if (ccl > 0.2) cl++;
        if (ccr > 0.2) cr++;
        const cf = f - detectedLag;
        if (cf < 0 || cf >= N) continue;
        const ccLshift = c.conf[cf * 17 + J.L_WRIST];
        const ccRshift = c.conf[cf * 17 + J.R_WRIST];
        if (cvl > 0.2 && ccLshift > 0.2 && diff(J.L_WRIST, f, cf) > 4) dL++;
        if (cvr > 0.2 && ccRshift > 0.2 && diff(J.R_WRIST, f, cf) > 4) dR++;
      }
      const pct = n => (100 * n / N).toFixed(1) + "%";
      setText("cc-cov-vl", pct(vl));
      setText("cc-cov-vr", pct(vr));
      setText("cc-cov-cl", pct(cl));
      setText("cc-cov-cr", pct(cr));
      const lagNote = detectedLag !== 0 ? " (lag-corrected)" : "";
      setText("cc-diff",
        `Combined differs from raw Vision${lagNote}: L ${dL} frames · R ${dR} frames`);
    }
  },

  draw(ctx, state) {
    const v = state.pose, c = state.poseCombined;
    if (!v || !c) return;
    const s = state.renderScale || 1;
    const f = state.frame;
    const cf = combinedFrame(state);

    const drawDot = (x, y, color) => {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 8 * s, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 2 * s;
      ctx.strokeStyle = "rgba(0,0,0,0.7)";
      ctx.stroke();
    };
    const drawForearm = (elbowJ, wristXY, color) => {
      const ec = v.conf[f * 17 + elbowJ];
      if (ec < 0.2) return;
      const ex = v.skeleton[(f * 17 + elbowJ) * 2];
      const ey = v.skeleton[(f * 17 + elbowJ) * 2 + 1];
      ctx.strokeStyle = color;
      ctx.lineWidth = 2 * s;
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(wristXY[0], wristXY[1]);
      ctx.stroke();
    };

    for (const [jointIdx, elbowJ] of [
      [J.L_WRIST, J.L_ELBOW],
      [J.R_WRIST, J.R_ELBOW],
    ]) {
      const vx = v.skeleton[(f * 17 + jointIdx) * 2];
      const vy = v.skeleton[(f * 17 + jointIdx) * 2 + 1];
      const vc = v.conf[f * 17 + jointIdx];
      const cx = cf == null ? NaN : c.skeleton[(cf * 17 + jointIdx) * 2];
      const cy = cf == null ? NaN : c.skeleton[(cf * 17 + jointIdx) * 2 + 1];
      const cc = cf == null ? 0   : c.conf[cf * 17 + jointIdx];
      const vOk = vc > 0.2;
      const cOk = cc > 0.2 && isFinite(cx);

      if (mode === "vision" || mode === "both") {
        if (vOk) {
          drawForearm(elbowJ, [vx, vy], "rgba(95,209,255,0.65)");
          drawDot(vx, vy, VISION_COL);
        }
      }
      if (mode === "combined" || mode === "both") {
        if (cOk) {
          drawForearm(elbowJ, [cx, cy], "rgba(52,199,89,0.65)");
          drawDot(cx, cy, COMBINED_COL);
        }
      }
      if (mode === "both" && vOk && cOk) {
        const d = Math.hypot(vx - cx, vy - cy);
        if (d > 4) {
          ctx.strokeStyle = DIVERGENCE_COL;
          ctx.lineWidth = 1.5 * s;
          ctx.setLineDash([4 * s, 3 * s]);
          ctx.beginPath();
          ctx.moveTo(vx, vy);
          ctx.lineTo(cx, cy);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    }
  },

  update(state) {
    const v = state.pose, c = state.poseCombined;
    if (!v || !c) {
      setText("cc-l", "—"); setText("cc-r", "—");
      setText("cc-l-conf", "—"); setText("cc-r-conf", "—");
      return;
    }
    const f = state.frame;
    const cf = combinedFrame(state);
    const upd = (id, j) => {
      const confId = `${id}-conf`;
      if (cf == null) { setText(id, "out of range"); setText(confId, "—"); return; }
      const vx = v.skeleton[(f  * 17 + j) * 2];
      const vy = v.skeleton[(f  * 17 + j) * 2 + 1];
      const vc = v.conf[f  * 17 + j];
      const cx = c.skeleton[(cf * 17 + j) * 2];
      const cy = c.skeleton[(cf * 17 + j) * 2 + 1];
      const cc = c.conf[cf * 17 + j];
      const vOk = vc > 0.05;
      const cOk = cc > 0.05 && isFinite(cx);
      setHTML(confId,
        `<span style="color:${VISION_COL}">V ${vOk ? vc.toFixed(2) : "—"}</span> · ` +
        `<span style="color:${COMBINED_COL}">C ${cOk ? cc.toFixed(2) : "—"}</span>`);
      if (!vOk || !cOk) { setText(id, "—"); return; }
      setText(id, `${Math.hypot(vx - cx, vy - cy).toFixed(0)} px`);
    };
    upd("cc-l", J.L_WRIST);
    upd("cc-r", J.R_WRIST);
  },
};

// Map the primary (raw vision) frame to the combined cache's frame by VIDEO
// TIME — the notebook keeps them aligned, but this matches what engine_compare
// does so we're robust against future drift.
//
// When lagCompensate is on we ALSO subtract the detected frame lag, so the
// green wrist drawn at lens frame t comes from the combined cache index that
// is physically simultaneous with vision[t]. This is a visualisation aid — it
// does NOT modify the cache file or what the production rules engine reads.
function combinedFrame(state) {
  const a = state.pose, b = state.poseCombined;
  const t = (a.start_sec || 0) + state.frame / a.fps;
  let cf = Math.round((t - (b.start_sec || 0)) * b.fps);
  if (lagCompensate) cf -= detectedLag;
  return (cf >= 0 && cf < b.n_frames) ? cf : null;
}

// Integer-lag cross-correlation on L+R wrist-y over the whole round. Same
// approach the notebook's check_alignment() uses (scipy.signal.correlate over
// ±maxLag). Sign convention: lag > 0 → combined (b) leads vision (a), so the
// green wrist appears in vision's "future" — exactly the "green is a frame
// ahead" symptom that motivated this code.
function detectFrameLag(a, b, maxLag) {
  const N = Math.min(a.n_frames, b.n_frames);
  const wristY = (p) => {
    const out = new Float32Array(N);
    for (let f = 0; f < N; f++) {
      const cL = p.conf[f * 17 + J.L_WRIST];
      const cR = p.conf[f * 17 + J.R_WRIST];
      const yL = p.skeleton[(f * 17 + J.L_WRIST) * 2 + 1];
      const yR = p.skeleton[(f * 17 + J.R_WRIST) * 2 + 1];
      let s = 0, n = 0;
      if (cL > 0.2) { s += yL; n++; }
      if (cR > 0.2) { s += yR; n++; }
      out[f] = n ? s / n : NaN;
    }
    return out;
  };
  const ya = wristY(a), yb = wristY(b);
  const demean = (arr) => {
    let s = 0, n = 0;
    for (const x of arr) if (!isNaN(x)) { s += x; n++; }
    const m = n ? s / n : 0;
    return arr.map(x => isNaN(x) ? 0 : x - m);
  };
  const da = demean(ya), db = demean(yb);
  let bestLag = 0, bestCorr = -Infinity;
  for (let lag = -maxLag; lag <= maxLag; lag++) {
    let c = 0;
    for (let t = 0; t < N; t++) {
      const tA = t + lag;
      if (tA < 0 || tA >= N) continue;
      c += da[tA] * db[t];
    }
    if (c > bestCorr) { bestCorr = c; bestLag = lag; }
  }
  return bestLag;
}

function setText(id, value) {
  const el = host?.querySelector("#" + id);
  if (el) el.textContent = value;
}

function setHTML(id, html) {
  const el = host?.querySelector("#" + id);
  if (el) el.innerHTML = html;
}
