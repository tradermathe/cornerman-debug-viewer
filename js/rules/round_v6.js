// Round v6 lens — shows exactly what the iOS app sees per round from the
// v6 cache (pose_cache_v6/<stem>_vision_glove_r<N>).
//
// Header surfaces the per-round glove-presence decision (yes/no, and where
// it came from — Sheet label vs MobileNetV3 classifier vs default), the
// engine that produced this skeleton (pure Vision when ungloved, Vision +
// glove_v6 when gloved), and the model versions baked into the meta.
//
// Skeleton: the v6 pose is drawn as the production skeleton. Wrists 9/10
// in the v6 cache are either Apple Vision's own wrists OR the v6 glove
// detector's wrists (whichever was confident at apply_glove_overrides
// time, gate ≥ 0.20). When the raw vision pose is ALSO loaded, the lens
// marks every wrist that differs from raw Vision in glove orange — those
// are the frames where the glove detector actually took over. Wrists that
// match raw vision render in pose cyan.
//
// Requires:
//   - slot.vision_glove   (the v6 combined cache)
//   Optional but useful: slot.vision (lets the lens flag which wrists
//   were overridden per frame by comparing to raw Vision).

import { J } from "../skeleton.js";

const POSE_COL  = "#5fd1ff";   // wrist same as raw vision (no override)
const GLOVE_COL = "#ffaa3c";   // wrist replaced by v6 glove detector
const OVERRIDE_PX = 1.5;       // sub-pixel diff = "same wrist"

let host;

// Pick whichever pose object is the v6 cache. The viewer attaches it to
// state.poseV6; if only the v6 cache was synced, it may also be state.pose
// itself (the loader falls back to vision_glove as primary).
function v6Pose(state) {
  if (state.poseV6) return state.poseV6;
  if (state.pose?.engine?.startsWith("apple_vision_2d+glove_v6")) return state.pose;
  if (state.pose?.meta?.engine?.includes("glove_v6")) return state.pose;
  return null;
}

// Raw vision is optional — used only for the per-frame override marker.
function visionPose(state) {
  if (state.pose?.engine === "apple_vision_2d") return state.pose;
  if (state.poseSecondary?.engine === "apple_vision_2d") return state.poseSecondary;
  return null;
}

export const RoundV6Rule = {
  id: "round_v6",
  label: "Round v6 (what iOS sees)",

  requires(slot) {
    return !!slot?.vision_glove;
  },

  // Hide raw vision's wrists in the base render — we redraw them ourselves
  // from the v6 cache, colour-coded by whether the glove detector took over.
  skeletonStyle() {
    return { hideJoints: new Set([J.L_WRIST, J.R_WRIST]) };
  },

  mount(_host, state) {
    host = _host;
    const v6 = v6Pose(state);
    const meta = v6?.meta || {};
    const ok = !!v6;
    const presence = meta.glove_presence || {};
    const replaced = !!meta.wrist_replaced_with_glove;
    const engine = meta.engine || v6?.engine || "?";
    const poseEng = meta.pose_engine || "apple_vision_2d";
    const gloveEng = meta.glove_engine || "—";
    const presenceSrc = presence.source || "?";
    const presenceVer = presence.presence_version || "—";
    const gloves = presence.gloves;
    const gateConf = meta.min_glove_conf ?? 0.20;
    const hasVision = !!visionPose(state);

    const gloveBadge = gloves
      ? `<span style="background:rgba(255,170,60,0.18);color:${GLOVE_COL};padding:2px 8px;border-radius:4px;font-weight:600">gloves</span>`
      : gloves === false
        ? `<span style="background:rgba(95,209,255,0.18);color:${POSE_COL};padding:2px 8px;border-radius:4px;font-weight:600">no gloves</span>`
        : `<span class="muted">unknown</span>`;
    const wristBadge = replaced
      ? `<span style="color:${GLOVE_COL}">v6 glove wrists baked in</span>`
      : `<span style="color:${POSE_COL}">pure Apple Vision wrists</span>`;

    host.innerHTML = `
      <h2>Round v6 — production skeleton</h2>
      ${ok ? "" : `<p class="hint" style="color:var(--bad)">
        No v6 cache for this round. Run <code>glove_cache_v6.ipynb</code>
        to produce <code>pose_cache_v6/&lt;stem&gt;_vision_glove_r{N}.{npy,_meta.json}</code>.
      </p>`}
      <p class="hint" style="margin:4px 0 10px">
        What the iOS app sees per round — the v6 combined cache rendered as-is.
        Wrists drawn in <span style="color:${POSE_COL}">cyan</span> match raw
        Apple Vision; wrists in <span style="color:${GLOVE_COL}">orange</span>
        were overridden by the v6 glove detector at apply-time
        (gate ≥ ${(+gateConf).toFixed(2)}).
        ${hasVision ? "" : `<br><span class="muted">Tip: include the raw <code>apple_vision_pose_cache/</code> in your Drive folder for per-frame override colour-coding — without it, every v6 wrist defaults to glove orange when this round is gloved.</span>`}
      </p>

      <h3>Round-level decision</h3>
      <div class="metric-grid" style="grid-template-columns:1fr 1fr">
        <div class="metric">
          <div class="metric-label">Gloves</div>
          <div class="metric-val">${gloveBadge}</div>
          <div class="metric-sub">source: <code>${presenceSrc}</code></div>
        </div>
        <div class="metric">
          <div class="metric-label">Wrists</div>
          <div class="metric-val" style="font-size:13px">${wristBadge}</div>
          <div class="metric-sub" id="rv6-replaced-count">—</div>
        </div>
      </div>

      <h3>Engine</h3>
      <div class="metric-grid" style="grid-template-columns:1fr">
        <div class="metric">
          <div class="metric-label">Combined engine</div>
          <div class="metric-val" style="font-size:13px"><code>${engine}</code></div>
        </div>
      </div>
      <div class="metric-grid" style="grid-template-columns:1fr 1fr;margin-top:8px">
        <div class="metric">
          <div class="metric-label">Pose</div>
          <div class="metric-val" style="font-size:12px"><code>${poseEng}</code></div>
        </div>
        <div class="metric">
          <div class="metric-label">Glove model</div>
          <div class="metric-val" style="font-size:12px"><code>${gloveEng}</code></div>
        </div>
      </div>
      ${presence.source === "classifier" ? `
      <div class="metric-grid" style="grid-template-columns:1fr;margin-top:8px">
        <div class="metric">
          <div class="metric-label">Presence classifier</div>
          <div class="metric-val" style="font-size:12px"><code>${presenceVer}</code></div>
          <div class="metric-sub">mean P(gloves) = ${presence.mean_prob_gloves != null ? Number(presence.mean_prob_gloves).toFixed(3) : "—"}</div>
        </div>
      </div>` : ""}

      <h3>This frame</h3>
      <div class="metric-grid">
        <div class="metric">
          <div class="metric-label">L wrist</div>
          <div class="metric-val" id="rv6-l-src">—</div>
          <div class="metric-sub" id="rv6-l-info">—</div>
        </div>
        <div class="metric">
          <div class="metric-label">R wrist</div>
          <div class="metric-val" id="rv6-r-src">—</div>
          <div class="metric-sub" id="rv6-r-info">—</div>
        </div>
      </div>
    `;

    // Pre-compute how many frames the v6 cache actually differs from raw
    // Vision (gate at the same confidence threshold the rules engine uses).
    // Without raw vision loaded, fall back to the meta's round-level flag.
    if (ok) {
      const v = visionPose(state);
      const N = v6.n_frames;
      let rL = 0, rR = 0, anyL = 0, anyR = 0;
      if (v && v.n_frames === N) {
        for (let f = 0; f < N; f++) {
          for (const [side, j] of [[0, J.L_WRIST], [1, J.R_WRIST]]) {
            const vc = v.conf[f * 17 + j];
            const v6c = v6.conf[f * 17 + j];
            if (v6c <= 0.05) continue;
            (side === 0 ? anyL++ : anyR++);
            if (vc <= 0.05) { side === 0 ? rL++ : rR++; continue; }
            const dx = v.skeleton[(f*17+j)*2]     - v6.skeleton[(f*17+j)*2];
            const dy = v.skeleton[(f*17+j)*2 + 1] - v6.skeleton[(f*17+j)*2 + 1];
            if (Math.hypot(dx, dy) > OVERRIDE_PX) {
              if (side === 0) rL++; else rR++;
            }
          }
        }
        setText("rv6-replaced-count",
          `L ${rL}/${anyL} · R ${rR}/${anyR} frames overridden`);
      } else if (replaced) {
        setText("rv6-replaced-count", "round-level override (pick raw vision for per-frame count)");
      } else {
        setText("rv6-replaced-count", "no override");
      }
    }
  },

  update(state) {
    const v6 = v6Pose(state);
    if (!v6) return;
    const v = visionPose(state);
    const f = state.frame;
    for (const [side, j, baseId] of [
      [0, J.L_WRIST, "rv6-l"],
      [1, J.R_WRIST, "rv6-r"],
    ]) {
      const v6x = v6.skeleton[(f * 17 + j) * 2];
      const v6y = v6.skeleton[(f * 17 + j) * 2 + 1];
      const v6c = v6.conf[f * 17 + j];
      const overridden = wristOverridden(v, v6, f, j);
      const src = v6c < 0.05
        ? `<span class="muted">no detection</span>`
        : overridden === true
          ? `<span style="color:${GLOVE_COL}">glove_v6</span>`
          : overridden === false
            ? `<span style="color:${POSE_COL}">vision</span>`
            : `<span class="muted">v6 only</span>`;
      setHTML(`${baseId}-src`, src);
      const info = v6c < 0.05
        ? "—"
        : `(${v6x.toFixed(0)}, ${v6y.toFixed(0)}) · conf ${v6c.toFixed(2)}`;
      setText(`${baseId}-info`, info);
    }
  },

  draw(ctx, state) {
    const v6 = v6Pose(state);
    if (!v6) return;
    const v = visionPose(state);
    const s = state.renderScale || 1;
    const f = state.frame;

    const drawDot = (x, y, color, r = 8) => {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, r * s, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 2 * s;
      ctx.strokeStyle = "rgba(0,0,0,0.7)";
      ctx.stroke();
    };
    const drawForearm = (elbowJ, wristXY, color) => {
      const ec = v6.conf[f * 17 + elbowJ];
      if (ec < 0.2) return;
      const ex = v6.skeleton[(f * 17 + elbowJ) * 2];
      const ey = v6.skeleton[(f * 17 + elbowJ) * 2 + 1];
      ctx.strokeStyle = color;
      ctx.lineWidth = 2 * s;
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(wristXY[0], wristXY[1]);
      ctx.stroke();
    };

    for (const [j, elbowJ] of [
      [J.L_WRIST, J.L_ELBOW],
      [J.R_WRIST, J.R_ELBOW],
    ]) {
      const x = v6.skeleton[(f * 17 + j) * 2];
      const y = v6.skeleton[(f * 17 + j) * 2 + 1];
      const c = v6.conf[f * 17 + j];
      if (c < 0.05) continue;
      const overridden = wristOverridden(v, v6, f, j);
      // When we can't compare to raw vision, fall back to the round-level
      // wrist_replaced flag for the colour — gloved rounds get orange,
      // ungloved rounds get cyan. It's a less precise signal but still
      // correct at the round level.
      const fallback = v6.meta?.wrist_replaced_with_glove ? GLOVE_COL : POSE_COL;
      const color = overridden === null ? fallback
                  : overridden ? GLOVE_COL : POSE_COL;
      drawForearm(elbowJ, [x, y], color);
      drawDot(x, y, color);
    }
  },
};

// Returns true if the v6 wrist differs from raw vision (= glove overrode),
// false if they match (= vision wrist), null if we can't compare (no raw
// vision loaded, or vision had no detection for this joint/frame).
function wristOverridden(visionP, v6, frame, joint) {
  if (!visionP || visionP.n_frames !== v6.n_frames) return null;
  const vc  = visionP.conf[frame * 17 + joint];
  const v6c = v6.conf[frame * 17 + joint];
  if (v6c < 0.05) return null;
  if (vc < 0.05) return true;     // vision had nothing, v6 has something → override
  const dx = visionP.skeleton[(frame*17+joint)*2]     - v6.skeleton[(frame*17+joint)*2];
  const dy = visionP.skeleton[(frame*17+joint)*2 + 1] - v6.skeleton[(frame*17+joint)*2 + 1];
  return Math.hypot(dx, dy) > OVERRIDE_PX;
}

function setText(id, value) {
  const el = host?.querySelector("#" + id);
  if (el) el.textContent = value;
}

function setHTML(id, html) {
  const el = host?.querySelector("#" + id);
  if (el) el.innerHTML = html;
}
