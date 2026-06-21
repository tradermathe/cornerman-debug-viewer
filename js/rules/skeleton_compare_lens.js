// Skeleton compare lens — overlay ANY two skeletons on the same video and
// pick which goes in each slot. Like engine_compare, but schema-aware (COCO-17,
// BlazePose-33, RTMPose Wholebody-133) and source-selectable.
//
// Built-in sources come from the loaded round (Vision / YOLO / v6, all COCO-17).
// Extra sources (BlazePose-33, RTMPose Wholebody-133) are loaded from .skel.json
// files via the file picker in the panel — generate them with
// cornerman-backend/pose_bakeoff/npz_to_skeljson.py for the round you're viewing.
// IMPORTANT: extract those from the UPRIGHT (display-oriented) video so the
// pixel coords match the browser's video + the Vision skeleton.
//
// Slot A = warm/orange, Slot B = cool/cyan. Wrists enlarged. Region toggles +
// confidence slider manage clutter on the dense 133-point skeleton.

import { SCHEMAS, GROUPS } from "./_skeleton_schemas.js";

const A_BONE = "rgba(255,138,60,0.65)",  A_JOINT = "#ff8a3c";
const B_BONE = "rgba(95,209,255,0.65)",  B_JOINT = "#5fd1ff";

function base64ToFloat32(b64) {
  const bin = atob(b64), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

function engineName(e) {
  return e === "yolo_pose" ? "YOLO"
    : e === "apple_vision_2d" ? "Vision"
    : e === "vision_combined" ? "Vision (comb)"
    : e === "blazepose_33" ? "BlazePose-33"
    : e === "rtmpose_wholebody_133" ? "RTMPose-133"
    : (e || "pose");
}
function schemaLabel(s) {
  return s === "blazepose33" ? "BlazePose-33"
    : s === "wholebody133" ? "RTMPose-133" : s;
}

let host, lastState = null;
let extras = [];              // [{ name, schema, pose }]
let slotA = null, slotB = null;   // selected source names
let THR = 0.2;
let enabled = new Set(GROUPS.filter(g => g[2]).map(g => g[0]));
let lastSrcSig = "";

// Built-in sources from the loaded round + any loaded extras.
function collectSources(state) {
  const seen = new Set(), out = [];
  const add = (p, suffix) => {
    if (!p || seen.has(p)) return;
    seen.add(p);
    out.push({ name: engineName(p.engine) + (suffix || ""), schema: p.schema || "coco17", pose: p });
  };
  if (state) { add(state.pose); add(state.poseSecondary); add(state.poseV6, " v6"); }
  for (const e of extras) out.push(e);
  return out;
}

function findSrc(srcs, name) { return srcs.find(s => s.name === name) || null; }

// Map the viewer's current video time to a source's own frame index.
function srcFrame(src, state) {
  const refFps = state.fps || (state.pose && state.pose.fps) || 30;
  const t = ((state.pose && state.pose.start_sec) || 0) + state.frame / refFps;
  const f = Math.round((t - ((src.pose.start_sec) || 0)) * (src.pose.fps || refFps));
  return (f >= 0 && f < src.pose.n_frames) ? f : null;
}

function drawSrc(ctx, src, frame, boneColor, jointColor, s) {
  const sch = SCHEMAS[src.schema] || SCHEMAS.coco17;
  const J = sch.n, tags = sch.point_tags, edges = sch.edges, names = sch.names;
  const sk = src.pose.skeleton, cf = src.pose.conf;
  if (frame * J * 2 + J * 2 > sk.length) return;
  const base = frame * J * 2, cbase = frame * J;

  ctx.lineWidth = 2 * s;
  ctx.strokeStyle = boneColor;
  for (const [a, b, tag] of edges) {
    if (!enabled.has(tag)) continue;
    const ca = cf[cbase + a], cb = cf[cbase + b];
    if (ca <= 0 || cb <= 0 || ca < THR || cb < THR) continue;
    ctx.beginPath();
    ctx.moveTo(sk[base + a * 2], sk[base + a * 2 + 1]);
    ctx.lineTo(sk[base + b * 2], sk[base + b * 2 + 1]);
    ctx.stroke();
  }
  for (let j = 0; j < J; j++) {
    if (!enabled.has(tags[j])) continue;
    const c = cf[cbase + j];
    if (c <= 0 || c < THR) continue;
    const x = sk[base + j * 2], y = sk[base + j * 2 + 1];
    const isWrist = names[j] === "left_wrist" || names[j] === "right_wrist";
    const dense = tags[j] === "face" || tags[j] === "hand_other";
    const r = (isWrist ? 7 : dense ? 2.5 : 4) * s;
    ctx.fillStyle = jointColor;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    if (isWrist) {
      ctx.lineWidth = 1.5 * s;
      ctx.strokeStyle = "rgba(255,255,255,0.9)";
      ctx.stroke();
    }
  }
}

function wristIdx(schema, side) {
  const names = (SCHEMAS[schema] || SCHEMAS.coco17).names;
  return names.indexOf(side === "L" ? "left_wrist" : "right_wrist");
}

// Repaint the canvas in place (no frame change). The scrubber 'input' trick
// only repaints when the frame actually changes, so use the viewer's exported
// redraw hook — the idiom every sibling overlay lens uses.
function forceRedraw() {
  window.__viewerRedraw?.();
}

function populateSelectors(srcs) {
  const optsHtml = srcs.map(s =>
    `<option value="${s.name}">${s.name} <span>(${schemaLabel(s.schema)})</span></option>`).join("");
  for (const [id, getCur, setCur] of [
    ["sc-a", () => slotA, v => slotA = v],
    ["sc-b", () => slotB, v => slotB = v],
  ]) {
    const sel = host.querySelector("#" + id);
    if (!sel) continue;
    sel.innerHTML = `<option value="">— none —</option>` + optsHtml;
    if (getCur() && srcs.some(s => s.name === getCur())) sel.value = getCur();
    else { sel.value = ""; setCur(null); }
  }
}

// Sensible defaults: BlazePose vs RTMPose if both present, else first two.
function autoSelect(srcs) {
  const byName = n => srcs.find(s => s.name === n);
  const blaze = srcs.find(s => s.schema === "blazepose33");
  const rtm = srcs.find(s => s.schema === "wholebody133");
  if (!slotA) slotA = (blaze || srcs[0])?.name || null;
  if (!slotB) slotB = (rtm || srcs.find(s => s.name !== slotA))?.name || null;
}

function syncSelectors(state) {
  const srcs = collectSources(state);
  const sig = srcs.map(s => s.name).join("|");
  if (sig !== lastSrcSig) {
    lastSrcSig = sig;
    autoSelect(srcs);
    populateSelectors(srcs);
  }
  return srcs;
}

async function onLoadFiles(files, state) {
  const status = host.querySelector("#sc-load-status");
  const round = (state && state.pose) || {};
  const msgs = [];
  let added = 0;
  for (const file of files) {
    try {
      const p = JSON.parse(await file.text());
      if (!SCHEMAS[p.schema]) throw new Error(`unknown schema "${p.schema}"`);
      const J = SCHEMAS[p.schema].n;   // schema is the single source of truth for stride
      const n = Number(p.n_frames);
      if (p.n_joints != null && Number(p.n_joints) !== J)
        throw new Error(`n_joints ${p.n_joints} != schema ${p.schema} (${J})`);
      const sk = base64ToFloat32(p.skeleton_b64), cf = base64ToFloat32(p.conf_b64);
      if (sk.length !== n * J * 2) throw new Error(`skeleton len ${sk.length} != ${n * J * 2}`);
      if (cf.length !== n * J) throw new Error(`conf len ${cf.length} != ${n * J}`);
      const pose = {
        skeleton: sk, conf: cf, n_frames: n, n_joints: J,
        fps: Number(p.fps) || (state.fps || 30), start_sec: 0,
        schema: p.schema, engine: p.engine || p.schema,
        width: Number(p.width) || 0, height: Number(p.height) || 0,
      };
      const name = engineName(pose.engine);
      extras = extras.filter(e => e.name !== name);   // replace if reloaded
      extras.push({ name, schema: p.schema, pose });
      added++;
      // Orientation guard: extras must come from the upright/display video. If
      // their dims don't match the loaded round, the overlay will be rotated.
      if (pose.width && round.width && (pose.width !== round.width || pose.height !== round.height))
        msgs.push(`<span class="bad">⚠ ${name}: ${pose.width}×${pose.height} ≠ round ${round.width}×${round.height} — extracted from the non-upright video? overlay may be rotated.</span>`);
    } catch (err) {
      msgs.push(`<span class="bad">${file.name}: ${err.message}</span>`);
    }
  }
  if (added) {
    lastSrcSig = "";   // force selector rebuild
    const srcs = syncSelectors(state);
    msgs.unshift(`Loaded ${added} skeleton${added === 1 ? "" : "s"}. Sources: ${srcs.map(s => s.name).join(", ")}`);
    forceRedraw();
  }
  status.innerHTML = msgs.join("<br>");
}

function renderFrameLine(state, srcs) {
  const el = host.querySelector("#sc-frame");
  if (!el) return;
  const a = findSrc(srcs, slotA), b = findSrc(srcs, slotB);
  if (!a || !b) { el.innerHTML = `<span class="muted">pick two sources to compare</span>`; return; }
  const fa = srcFrame(a, state), fb = srcFrame(b, state);
  // Stride from the SCHEMA, not pose.n_joints — built-in COCO-17 poses (from
  // loadPose) carry no n_joints field, which would make the index NaN.
  const Ja = (SCHEMAS[a.schema] || SCHEMAS.coco17).n;
  const Jb = (SCHEMAS[b.schema] || SCHEMAS.coco17).n;
  const wristDelta = side => {
    if (fa == null || fb == null) return "—";
    const ja = wristIdx(a.schema, side), jb = wristIdx(b.schema, side);
    if (ja < 0 || jb < 0) return "—";
    const ax = a.pose.skeleton[(fa * Ja + ja) * 2], ay = a.pose.skeleton[(fa * Ja + ja) * 2 + 1];
    const bx = b.pose.skeleton[(fb * Jb + jb) * 2], by = b.pose.skeleton[(fb * Jb + jb) * 2 + 1];
    const ca = a.pose.conf[fa * Ja + ja], cb = b.pose.conf[fb * Jb + jb];
    if (!(ca >= THR) || !(cb >= THR)) return "low conf";
    return `${Math.hypot(ax - bx, ay - by).toFixed(0)} px`;
  };
  el.innerHTML =
    `<strong>frame ${state.frame}:</strong> ` +
    `<span style="color:${A_JOINT}">A=${a.name}</span> vs <span style="color:${B_JOINT}">B=${b.name}</span> · ` +
    `L wrist Δ <b>${wristDelta("L")}</b> · R wrist Δ <b>${wristDelta("R")}</b>`;
}

export const SkeletonCompareRule = {
  id: "skeleton_compare",
  label: "Skeleton compare (select 2)",

  // We draw both skeletons ourselves; suppress the base renderer.
  skeletonStyle() {
    return { boneColor: "rgba(0,0,0,0)", jointRadius: 0, minConf: Infinity };
  },

  mount(_host, state) {
    host = _host;
    lastState = state;
    lastSrcSig = "";
    host.innerHTML = `
      <h2>Skeleton compare</h2>
      <p class="hint">Overlay any two skeletons on the video.
        <span style="color:${A_JOINT}">Slot A = orange</span>,
        <span style="color:${B_JOINT}">Slot B = cyan</span>. Wrists enlarged.</p>
      <div class="picker-row" style="gap:8px">
        <label><span>Slot A</span><select id="sc-a"></select></label>
        <label><span>Slot B</span><select id="sc-b"></select></label>
      </div>
      <label class="folder-pick" style="margin-top:8px">
        <span>Load extra skeletons <code>.skel.json</code> <span class="muted">(BlazePose / RTMPose)</span></span>
        <input type="file" id="sc-files" accept=".json,application/json" multiple>
      </label>
      <div id="sc-load-status" class="muted small" style="margin:4px 0 8px"></div>
      <div class="toggles" id="sc-toggles" style="display:flex;gap:12px;flex-wrap:wrap;font-size:12px;margin-bottom:6px">
        <strong style="color:#aaa">Show:</strong>
      </div>
      <label class="slider-row" style="display:block;font-size:13px;margin:4px 0">
        confidence ≥ <output id="sc-thr-out">${THR.toFixed(2)}</output>
        <input type="range" id="sc-thr" min="0" max="1" step="0.05" value="${THR}">
      </label>
      <h3>Current frame</h3>
      <div id="sc-frame" style="font-size:13px;line-height:1.6"></div>
      <p class="hint" style="margin-top:8px">Extras must be extracted from the upright/display video
        (portrait phone clips are rotated) or they won't line up. Don't cross-compare confidence
        between engines — different scales.</p>
    `;

    // Region toggles.
    const tg = host.querySelector("#sc-toggles");
    for (const [tag, label] of GROUPS) {
      const lab = document.createElement("label");
      lab.style.cssText = "display:flex;gap:4px;align-items:center;cursor:pointer";
      const cb = document.createElement("input");
      cb.type = "checkbox"; cb.checked = enabled.has(tag); cb.dataset.tag = tag;
      cb.addEventListener("change", () => {
        if (cb.checked) enabled.add(tag); else enabled.delete(tag);
        forceRedraw();
      });
      lab.appendChild(cb);
      lab.appendChild(document.createTextNode(label));
      tg.appendChild(lab);
    }

    host.querySelector("#sc-a").addEventListener("change", e => { slotA = e.target.value || null; forceRedraw(); });
    host.querySelector("#sc-b").addEventListener("change", e => { slotB = e.target.value || null; forceRedraw(); });
    host.querySelector("#sc-files").addEventListener("change", e => onLoadFiles([...e.target.files], lastState || state));
    const thr = host.querySelector("#sc-thr"), thrOut = host.querySelector("#sc-thr-out");
    thr.addEventListener("input", () => { THR = parseFloat(thr.value); thrOut.textContent = THR.toFixed(2); forceRedraw(); });

    syncSelectors(state);
    renderFrameLine(state, collectSources(state));
  },

  update(state) {
    if (!host) return;
    lastState = state;
    const srcs = syncSelectors(state);
    renderFrameLine(state, srcs);
  },

  draw(ctx, state) {
    lastState = state;
    const s = state.renderScale || 1;
    const srcs = collectSources(state);
    const a = findSrc(srcs, slotA), b = findSrc(srcs, slotB);
    // Draw B first so A (usually the one under scrutiny) sits on top.
    if (b) { const fb = srcFrame(b, state); if (fb != null) drawSrc(ctx, b, fb, B_BONE, B_JOINT, s); }
    if (a) { const fa = srcFrame(a, state); if (fa != null) drawSrc(ctx, a, fa, A_BONE, A_JOINT, s); }
  },
};
