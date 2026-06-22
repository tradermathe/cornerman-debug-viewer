// Engine compare lens — overlay UP TO 5 skeleton engines on the same frame and
// toggle which ones are shown. Driven by `state.engines` (built in viewer.js
// start()): every COCO-17 engine loaded for the round — Vision / YOLO / RTMPose
// / MoveNet / YOLO11 / BlazePose / v6 / combined. No fixed "primary"; pick any
// subset to eyeball where the models disagree (esp. wrists/feet-adjacent joints).
//
// Each selected engine gets a stable colour from the palette (in selection
// order). Wrists are drawn extra-large. Engines are PTS-aligned to the on-screen
// video time, so caches with different fps/start still line up.
//
// (BlazePose here is the COCO-17 remap; its 33-joint feet cache rides the
// schema-aware skeleton_compare lens instead.)

import { J } from "../skeleton.js";

const MAX_SEL = 5;

// Distinct colours; [bone (translucent), joint/wrist (solid)].
const PALETTE = [
  { bone: "rgba(95,209,255,0.55)",  ink: "#5fd1ff" },  // cyan
  { bone: "rgba(255,138,60,0.55)",  ink: "#ff8a3c" },  // orange
  { bone: "rgba(95,217,122,0.55)",  ink: "#5fd97a" },  // green
  { bone: "rgba(224,64,251,0.55)",  ink: "#e040fb" },  // magenta
  { bone: "rgba(255,224,90,0.55)",  ink: "#ffe05a" },  // yellow
];

const HIGHLIGHT = new Set([J.NOSE, J.L_WRIST, J.R_WRIST, J.L_ELBOW, J.R_ELBOW]);
const DRAW_CONF_GATE = 0.3;

// COCO-17 skeleton tokens (what counts toward "≥2 engines to compare").
const COMPARE_TOKENS = ["vision", "yolo", "rtmpose", "movenet", "yolo11", "blazepose",
                        "vision_combined", "vision_glove"];

let host;
let selected = null;   // Set<engineKey>, persists across rounds

function engines(state) { return state.engines || []; }

// selected ∩ available, in available order; default to the first MAX_SEL.
function activeKeys(state) {
  const avail = engines(state).map(e => e.key);
  if (!selected) selected = new Set(avail.slice(0, MAX_SEL));
  const keys = avail.filter(k => selected.has(k));
  return keys.slice(0, MAX_SEL);
}

export const EngineCompareRule = {
  id: "engine_compare",
  label: "Engine compare (pick up to 5)",

  // Show whenever the round has ≥2 comparable skeleton engines.
  requires(slot) {
    if (!slot) return false;
    return COMPARE_TOKENS.filter(t => slot[t]).length >= 2;
  },

  // We draw every selected skeleton ourselves — suppress the base renderer.
  skeletonStyle() {
    return { boneColor: "rgba(0,0,0,0)", jointRadius: 0, minConf: Infinity, showImputed: false };
  },

  mount(_host, state) {
    host = _host;
    const avail = engines(state);
    if (!selected) selected = new Set(avail.map(e => e.key).slice(0, MAX_SEL));
    // prune stale keys from a previous round
    for (const k of [...selected]) if (!avail.some(e => e.key === k)) selected.delete(k);
    const active = new Set(activeKeys(state));
    const full = active.size >= MAX_SEL;

    const rows = avail.map(e => {
      const on = active.has(e.key);
      const idx = [...active].indexOf(e.key);
      const sw = on && idx >= 0 ? PALETTE[idx % PALETTE.length].ink : "#444";
      return `<label class="ec-row" style="display:flex;align-items:center;gap:8px;margin:3px 0;cursor:pointer">
        <input type="checkbox" data-key="${e.key}" ${on ? "checked" : ""} ${(!on && full) ? "disabled" : ""}>
        <span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:${sw};border:1px solid #0006"></span>
        <span style="color:${on ? "#eee" : "#888"}">${e.label}</span>
      </label>`;
    }).join("");

    host.innerHTML = `
      <h2>Engine compare</h2>
      <p class="hint">Overlay up to ${MAX_SEL} skeletons. Wrists drawn extra-large;
      each engine PTS-aligned to the video.</p>
      ${avail.length ? rows : `<p class="hint" style="color:var(--bad)">No skeleton engines loaded for this round.</p>`}
      <h3 style="margin-top:10px">Per-frame (raw px from the first selected)</h3>
      <div id="ec-frame" class="hint">—</div>
    `;
    host.querySelectorAll('input[type=checkbox]').forEach(cb => {
      cb.addEventListener("change", () => {
        const k = cb.dataset.key;
        if (cb.checked) {
          if (selected.size >= MAX_SEL) { cb.checked = false; return; }
          selected.add(k);
        } else {
          selected.delete(k);
        }
        this.mount(host, state);     // rebuild swatches + disabled state
        window.__viewerRedraw?.();
        this.update(state);
      });
    });
    this.update(state);
  },

  draw(ctx, state) {
    const s = state.renderScale || 1;
    const avail = engines(state);
    const active = activeKeys(state);
    active.forEach((key, i) => {
      const e = avail.find(x => x.key === key);
      if (!e) return;
      const fr = frameAt(e.pose, videoTime(state), state, e.pose === state.pose);
      if (fr == null) return;
      const c = PALETTE[i % PALETTE.length];
      drawEngineSkeleton(ctx, e.pose, fr, {
        boneColor: c.bone, jointColor: c.ink, wristColor: c.ink,
        boneWidth: 2 * s, jointRadius: 4 * s, wristRadius: 9 * s, strokeWidth: 2 * s,
      });
    });
  },

  update(state) {
    const el = host?.querySelector("#ec-frame");
    if (!el) return;
    const avail = engines(state);
    const active = activeKeys(state);
    if (!active.length) { el.innerHTML = "—"; return; }
    const t = videoTime(state);
    const line = (label, pose, ink) => {
      const fr = frameAt(pose, t, state, pose === state.pose);
      const conf = j => (fr == null ? "—" : (pose.conf[fr * 17 + j] ?? 0).toFixed(2));
      return `<span style="color:${ink}">${label}</span>: Lw ${conf(J.L_WRIST)} · Rw ${conf(J.R_WRIST)}`;
    };
    el.innerHTML = active.map((key, i) => {
      const e = avail.find(x => x.key === key);
      return e ? line(e.label, e.pose, PALETTE[i % PALETTE.length].ink) : "";
    }).filter(Boolean).join("<br>");
  },
};

// ── time alignment (per-engine PTS, falling back to fps model) ──────────────
function videoTime(state) {
  const v = (typeof document !== "undefined") && document.getElementById("video");
  return (v && isFinite(v.currentTime))
    ? v.currentTime
    : (state.pose?.start_sec || 0) + state.frame / (state.pose?.fps || 30);
}

function frameAt(pose, t, state, primary) {
  const bp = pose.pts;
  if (bp && bp.length && isFinite(t)) {
    let lo = 0, hi = bp.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (bp[m] < t) lo = m + 1; else hi = m; }
    let best = lo;
    if (lo > 0 && Math.abs(bp[lo - 1] - t) <= Math.abs(bp[lo] - t)) best = lo - 1;
    if (primary) return best;
    const tol = 0.75 / (pose.fps || 30);
    return (t < bp[0] - tol || t > bp[bp.length - 1] + tol) ? null : best;
  }
  if (primary) return state.frame;
  const sf = Math.round((t - (pose.start_sec || 0)) * pose.fps);
  return (sf >= 0 && sf < pose.n_frames) ? sf : null;
}

// Draw one engine's COCO-17 skeleton in a single colour (wrists enlarged).
function drawEngineSkeleton(ctx, pose, frame, style) {
  const EDGES = [
    [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 6], [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16],
    [0, 1], [0, 2], [1, 3], [2, 4],
  ];
  ctx.lineWidth = style.boneWidth;
  ctx.strokeStyle = style.boneColor;
  for (const [a, b] of EDGES) {
    const ca = pose.conf[frame * 17 + a];
    const cb = pose.conf[frame * 17 + b];
    if (ca < DRAW_CONF_GATE || cb < DRAW_CONF_GATE) continue;
    ctx.beginPath();
    ctx.moveTo(pose.skeleton[(frame * 17 + a) * 2], pose.skeleton[(frame * 17 + a) * 2 + 1]);
    ctx.lineTo(pose.skeleton[(frame * 17 + b) * 2], pose.skeleton[(frame * 17 + b) * 2 + 1]);
    ctx.stroke();
  }
  for (let j = 0; j < 17; j++) {
    const c = pose.conf[frame * 17 + j];
    if (c < DRAW_CONF_GATE) continue;
    const x = pose.skeleton[(frame * 17 + j) * 2];
    const y = pose.skeleton[(frame * 17 + j) * 2 + 1];
    const isWrist = j === J.L_WRIST || j === J.R_WRIST;
    const r = isWrist ? style.wristRadius : (HIGHLIGHT.has(j) ? style.jointRadius * 1.5 : style.jointRadius);
    ctx.fillStyle = isWrist ? style.wristColor : style.jointColor;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    if (isWrist) {
      ctx.lineWidth = style.strokeWidth ?? 2;
      ctx.strokeStyle = "rgba(0,0,0,0.65)";
      ctx.stroke();
    }
  }
}
