// Demo dashboard entry. Purely additive — reuses ondevice-loader + drawSkeleton,
// touches nothing in viewer.js or the lenses.

import { drawSkeleton } from "../skeleton.js";
import { guardHeight, elbowFlare } from "./frameMetrics.js";
import { syntheticSession, loadLocalSession } from "./data.js";

const FEATURED_SESSION = "./demo-assets/session_1781788984153";
import { createState, detections, selectedPunch, throwingSide, roundSummary, visible, timecode } from "./state.js";
import { PER_PUNCH_RULES, SESSION_RULES } from "./rules.js";

const ARM = { L: { s: 5, e: 7, w: 9 }, R: { s: 6, e: 8, w: 10 } };
let S, els, timer = null;

async function boot() {
  try {
    let session;
    try { session = await loadLocalSession(FEATURED_SESSION); }
    catch (e) { console.warn("[demo] real session unavailable, using fixture:", e.message); session = syntheticSession(); }
    S = createState(session);
    cacheEls();
    buildStatic();
    attachTransport();
    const c = els.overlay; c.width = S.width; c.height = S.height;
    if (S.videoUrl) setupVideo();
    seek(0); renderAll();
  } catch (e) { console.error("[demo boot]", e && e.stack || e); }
}

function setupVideo() {
  els.film.src = S.videoUrl;
  els.film.addEventListener("loadedmetadata", () => {
    // Pose data covers the tail of the video; the head is pre-round footage.
    S.videoOffsetSec = Math.max(0, els.film.duration - S.nFrames / S.fps);
    seek(S.frame);   // position to round start as soon as the offset is known
  }, { once: true });
  els.film.addEventListener("loadeddata", () => seek(S.frame), { once: true });
  els.film.addEventListener("seeked", () => { if (!S.playing) drawFilm(); });
  els.film.addEventListener("ended", pause);
  // Some browsers don't paint a frame until the video plays. Draw the overlay
  // in lock-step with the first presented frame so it's aligned before any play.
  if (els.film.requestVideoFrameCallback) {
    els.film.requestVideoFrameCallback(() => { if (!S.playing) drawFilm(); });
  }
}

function cacheEls() {
  const $ = (id) => document.getElementById(id);
  els = {
    overlay: $("overlay"), film: $("film"), frameBadge: $("frameBadge"), punchCallout: $("punchCallout"), clipName: $("clipName"), timecode: $("timecode"),
    scrubber: $("scrubber"), scrubFill: $("scrubFill"), scrubHandle: $("scrubHandle"),
    timeline: $("timelineBody"), tlControls: $("tlControls"), tlSub: $("tlSub"),
    summary: $("summaryBody"), detail: $("detailBody"), frameStats: $("frameStats"),
  };
}

// ---- Film -------------------------------------------------------------------

function drawFilm() {
  const ctx = els.overlay.getContext("2d");
  ctx.clearRect(0, 0, S.width, S.height);
  drawSkeleton(ctx, S.pose, S.frame, {
    boneColor: "rgba(236,230,217,0.42)", boneWidth: 5, jointRadius: 7, minConf: 0.05, showImputed: false,
  });
  // Active limb(s) + top-left callout when the current frame is inside a punch
  // window. Multiple overlapping punches all show.
  const active = detections(S).filter((d) => S.frame >= d.start_frame && S.frame <= d.end_frame);
  active.forEach((d) => drawActiveLimb(ctx, d));
  els.punchCallout.innerHTML = active.map((d) =>
    `<span class="punch-chip ${d.hand}"><span class="dot"></span>${cap(d.hand)} ${prettyType(d.punch_type)}</span>`).join("");
  // Low-confidence amber rings.
  for (let j = 0; j < 17; j++) {
    const c = S.pose.conf[S.frame * 17 + j];
    if (c > 0.05 && c < 0.8) {
      const x = S.pose.skeleton[(S.frame * 17 + j) * 2], y = S.pose.skeleton[(S.frame * 17 + j) * 2 + 1];
      ctx.strokeStyle = "#F59E0B"; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(x, y, 13, 0, Math.PI * 2); ctx.stroke();
    }
  }
  els.frameBadge.textContent = `FRAME ${S.frame}`;
  els.timecode.innerHTML = `${timecode(S.frame, S.fps)}<span class="total"> / ${timecode(S.nFrames, S.fps)}</span>`;
}

function drawActiveLimb(ctx, punch) {
  const a = ARM[throwingSide(S, punch)];
  const pt = (j, f = S.frame) => [S.pose.skeleton[(f * 17 + j) * 2], S.pose.skeleton[(f * 17 + j) * 2 + 1]];
  // Motion tracer: wrist path from guard to now.
  ctx.strokeStyle = "rgba(184,92,61,0.22)"; ctx.lineWidth = 16; ctx.lineCap = "round";
  ctx.beginPath();
  for (let f = punch.start_frame; f <= S.frame; f++) { const [x, y] = pt(a.w, f); f === punch.start_frame ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
  ctx.stroke();
  // Rust bones + joints.
  ctx.strokeStyle = "#B85C3D"; ctx.lineWidth = 7; ctx.beginPath();
  const [sx, sy] = pt(a.s), [ex, ey] = pt(a.e), [wx, wy] = pt(a.w);
  ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.lineTo(wx, wy); ctx.stroke();
  for (const [x, y] of [[sx, sy], [ex, ey], [wx, wy]]) {
    ctx.fillStyle = "rgba(184,92,61,0.25)"; ctx.beginPath(); ctx.arc(x, y, 18, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#B85C3D"; ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2); ctx.fill();
  }
}

// ---- Playback / seeking -----------------------------------------------------

function redrawFrame() { updateScrub(); drawFilm(); renderFrameStats(); updatePlayhead(); }
function seek(f) {
  S.frame = Math.max(0, Math.min(S.nFrames - 1, f));
  if (S.videoUrl && els.film) els.film.currentTime = (S.frame + 0.5) / S.fps + S.videoOffsetSec;
  redrawFrame();
}
function updateScrub() { const pct = (S.frame / (S.nFrames - 1)) * 100; els.scrubFill.style.width = pct + "%"; els.scrubHandle.style.left = pct + "%"; }
function play() {
  if (S.playing) return;
  S.playing = true; setPlayGlyph();
  if (S.videoUrl && els.film) {
    els.film.play();
    const tick = (now, meta) => {
      if (!S.playing) return;
      S.frame = Math.max(0, Math.min(S.nFrames - 1, Math.round(((meta?.mediaTime ?? els.film.currentTime) - S.videoOffsetSec) * S.fps)));
      redrawFrame();
      if (els.film.requestVideoFrameCallback) els.film.requestVideoFrameCallback(tick);
      else timer = requestAnimationFrame(() => tick());
    };
    if (els.film.requestVideoFrameCallback) els.film.requestVideoFrameCallback(tick);
    else timer = requestAnimationFrame(() => tick());
  } else {
    timer = setInterval(() => { if (S.frame >= S.nFrames - 1) return pause(); seek(S.frame + 1); }, 1000 / S.fps);
  }
}
function pause() {
  S.playing = false; setPlayGlyph();
  if (S.videoUrl && els.film) els.film.pause();
  if (timer) { clearInterval(timer); cancelAnimationFrame(timer); timer = null; }
}
function setPlayGlyph() { const b = document.querySelector('[data-act="play"]'); if (b) b.textContent = S.playing ? "❚❚" : "▶"; }

function selectPunch(idx) {
  S.selIdx = idx; const p = selectedPunch(S); if (p) { pause(); seek(p.impact_frame ?? p.start_frame); }
  renderTimeline(); renderDetail();
}
function jumpPunch(dir) {
  const ds = detections(S).filter((d) => visible(S, d)).sort((a, b) => a.impact_frame - b.impact_frame);
  if (!ds.length) return;
  const cur = ds.findIndex((d) => d.idx === S.selIdx);
  const next = cur === -1 ? (dir > 0 ? 0 : ds.length - 1) : Math.max(0, Math.min(ds.length - 1, cur + dir));
  selectPunch(ds[next].idx);
}

// ---- Static shell + transport ----------------------------------------------

function buildStatic() {
  els.clipName.textContent = S.fixture ? "Sparring — Round 1 (sample data)" : "Sparring — Round 1";
}
function attachTransport() {
  document.querySelectorAll("[data-act]").forEach((b) => b.addEventListener("click", () => {
    const a = b.dataset.act;
    if (a === "play") S.playing ? pause() : play();
    else if (a === "prevF") { pause(); seek(S.frame - 1); }
    else if (a === "nextF") { pause(); seek(S.frame + 1); }
    else if (a === "prevP") jumpPunch(-1);
    else if (a === "nextP") jumpPunch(1);
  }));
  const seekFromEvent = (e) => { const r = els.scrubber.getBoundingClientRect(); pause(); seek(Math.round(((e.clientX - r.left) / r.width) * (S.nFrames - 1))); };
  els.scrubber.addEventListener("click", seekFromEvent);
  document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") { e.preventDefault(); pause(); seek(S.frame - 1); }
    else if (e.key === "ArrowRight") { e.preventDefault(); pause(); seek(S.frame + 1); }
  });
}

// ---- Renderers --------------------------------------------------------------

function renderAll() { renderTimeline(); renderSummary(); renderDetail(); renderFrameStats(); }

// Live per-frame form values. Stance width is the on-device continuous rule;
// guard height (per wrist) and elbow flare (per arm) are the same geometry the
// debug-viewer lenses use, run here from the pose. The actively-punching arm is
// excluded so a thrown punch doesn't read as a dropped guard / flared elbow.
function renderFrameStats() {
  const f = S.frame, rules = S.analysis?.rules || {};
  const row = (label, val, cls) =>
    `<div class="bar-row"><span class="lbl" style="width:auto">${label}</span>
       <span style="flex:1"></span><span class="pill ${cls}">${val}</span></div>`;
  const out = [`<div class="eyebrow">This frame</div>`];

  // Stance width — on-device per-frame rule
  const sw = rules.stance_width;
  if (sw && sw.validMask) {
    const valid = sw.validMask[f] === 1;
    const r = (sw.sepRatiosCorrected || sw.sepRatios || [])[f];
    const narrow = sw.violationMask && sw.violationMask[f] === 1;
    out.push(valid && r === r
      ? row("Stance width", `${narrow ? "Narrow" : "Good"} · ${r.toFixed(2)}`, narrow ? "warn" : "ok")
      : row("Stance width", "—", "neutral"));
  }

  // Guard height + elbow flare per arm (geometric). lead/rear → anatomical side.
  const leadSide = S.stance === "orthodox" ? "L" : "R";
  const sideOf = { lead: leadSide, rear: leadSide === "L" ? "R" : "L" };
  let active = null;  // anatomical side of the arm punching at this frame
  for (const d of detections(S)) {
    if (f >= d.start_frame - 3 && f <= d.end_frame + 3) { active = sideOf[d.hand] || null; break; }
  }

  for (const hand of ["lead", "rear"]) {
    const s = sideOf[hand];
    if (s === active) { out.push(row(`Guard · ${hand}`, "punching", "neutral")); continue; }
    const g = guardHeight(S.pose, f, ARM[s].w);
    out.push(row(`Guard · ${hand}`, g ? `${g.dropped ? "Dropped" : "Up"} · ${g.belowNose.toFixed(2)} / ${g.threshold.toFixed(2)}` : "—", g ? (g.dropped ? "warn" : "ok") : "neutral"));
  }
  for (const hand of ["lead", "rear"]) {
    const s = sideOf[hand];
    if (s === active) { out.push(row(`Elbow · ${hand}`, "punching", "neutral")); continue; }
    const e = elbowFlare(S.pose, f, ARM[s].s, ARM[s].e);
    const word = e ? (e.status === "flared" ? "Flared" : e.status === "out" ? "Out" : "Tucked") : "—";
    const cls = e ? (e.status === "flared" ? "bad" : e.status === "out" ? "warn" : "ok") : "neutral";
    out.push(row(`Elbow · ${hand}`, e ? `${word} · ${e.flare.toFixed(2)}` : "—", cls));
  }

  els.frameStats.innerHTML = out.join("");
}

function renderTimeline() {
  const ds = detections(S);
  els.tlSub.textContent = `${ds.length} detected`;
  els.tlControls.innerHTML = `
    <span class="chip ${S.filters.lead ? "on" : "off"}" data-lane="lead"><span class="lane-dot" style="background:var(--navy)"></span>Lead</span>
    <span class="chip ${S.filters.rear ? "on" : "off"}" data-lane="rear"><span class="lane-dot" style="background:var(--info)"></span>Rear</span>
    <span style="width:1px;height:18px;background:var(--border)"></span>
    ${["all", "jab", "cross", "hook", "uppercut"].map((t) => `<span class="chip ${S.filters.type === t ? "on" : ""}" data-type="${t}">${cap(t)}</span>`).join("")}`;
  els.tlControls.querySelectorAll("[data-lane]").forEach((c) => c.addEventListener("click", () => { S.filters[c.dataset.lane] = !S.filters[c.dataset.lane]; renderTimeline(); }));
  els.tlControls.querySelectorAll("[data-type]").forEach((c) => c.addEventListener("click", () => { S.filters.type = c.dataset.type; renderTimeline(); }));

  const markers = ds.map((d) => {
    const left = (d.impact_frame / S.nFrames) * 100;
    const lane = d.hand === "lead" ? 16 : 88;
    const color = d.hand === "lead" ? "#1B2A4A" : "#2563EB";
    const head = d.punch_type.includes("body") ? `background:${color}2e;border:1.5px solid ${color}` : `background:${color}`;
    const cls = `marker${d.idx === S.selIdx ? " sel" : ""}${visible(S, d) ? "" : " dim"}`;
    return `<div class="${cls}" data-idx="${d.idx}" title="Punch ${d.idx + 1} · ${cap(d.hand)} ${prettyType(d.punch_type)} · ${timecode(d.impact_frame, S.fps)}"
      style="left:calc(${left}% - 6px);top:${lane}px;${head}"></div>`;
  }).join("");
  const axis = [0, 30, 60, 90, 120, 150, 180].filter((s) => s <= S.nFrames / S.fps).map((s) =>
    `<span class="axis" style="left:${(s * S.fps / S.nFrames) * 100}%">${timecode(s * S.fps, S.fps)}</span>`).join("");

  els.timeline.innerHTML = `
    <div class="lane-label" style="top:30px"><span class="lane-dot" style="background:var(--navy)"></span>Lead</div>
    <div class="lane-label" style="top:102px"><span class="lane-dot" style="background:var(--info)"></span>Rear</div>
    <div class="lane lead"></div><div class="lane rear"></div>
    ${markers}<div class="playhead" id="playhead"></div>${axis}`;
  els.timeline.querySelectorAll(".marker").forEach((m) => m.addEventListener("click", () => selectPunch(+m.dataset.idx)));
  els.timeline.addEventListener("click", (e) => { if (e.target.classList.contains("marker")) return; const r = els.timeline.getBoundingClientRect(); pause(); seek(Math.round(((e.clientX - r.left) / r.width) * (S.nFrames - 1))); }, { once: false });
  updatePlayhead();
}
function updatePlayhead() { const ph = document.getElementById("playhead"); if (ph) ph.style.left = (S.frame / S.nFrames) * 100 + "%"; }

function renderSummary() {
  const r = roundSummary(S);
  const typeBar = (label, n) => `<div class="bar-row"><span class="lbl">${label}</span>
    <div class="bar-track"><div class="bar-fill" style="width:${(n / Math.max(1, r.total)) * 100}%"></div></div>
    <span class="val tnum">${n}</span></div>`;
  const formCards = SESSION_RULES.map((rule) => rule.evaluate(S.analysis)).filter(Boolean).map((c) =>
    `<div class="rule-card"><div class="meta"><div class="name">${c.title}</div><div class="sub">${c.cue || ""}</div></div>
     <span class="verdict ${c.verdict}">${c.headline}</span></div>`).join("");
  els.summary.innerHTML = `
    <div style="display:flex;align-items:flex-end;gap:14px;margin-bottom:6px">
      <div class="big-num tnum">${r.total}</div><div class="overline" style="padding-bottom:6px">Punches<br>thrown</div>
      <div style="flex:1"></div>
      <span class="count-pill lead tnum">${r.lead} lead</span><span class="count-pill rear tnum">${r.rear} rear</span>
    </div>
    <div class="eyebrow" style="margin-top:8px">Form this round</div>${formCards}
    <div class="eyebrow" style="margin-top:8px">Punches by type</div>
    ${typeBar("Jab", r.byType.jab)}${typeBar("Cross", r.byType.cross)}${typeBar("Hook", r.byType.hook)}${typeBar("Uppercut", r.byType.uppercut)}`;
}

function renderDetail() {
  const p = selectedPunch(S);
  if (!p) {
    els.detail.innerHTML = `<div class="empty"><div class="ring">◎</div>
      <div style="font-family:'Barlow Condensed';font-weight:600;font-size:16px;color:var(--ink-soft)">No punch selected</div>
      <div style="font-size:13px;max-width:240px">Click any marker on the timeline to inspect its mechanics and return-to-guard.</div></div>`;
    return;
  }
  const cards = PER_PUNCH_RULES.map((rule) => rule.evaluate(S.analysis, p)).map((c, i) => {
    const graph = PER_PUNCH_RULES[i].graph ? returnPathSVG(p) : "";
    return `<div class="rule-card"><div class="meta"><div class="name">${c.title}</div><div class="sub">${c.headline}${c.sub ? " · " + c.sub : ""}</div>${graph}</div>
      <span class="verdict ${c.verdict}">${verdictWord(c.verdict)}</span></div>`;
  }).join("");
  els.detail.innerHTML = `
    <div class="detail-head"><h3>Punch ${p.idx + 1}</h3>
      <span class="pill neutral">${prettyType(p.punch_type)}</span>
      <span class="pill ${p.hand === "lead" ? "neutral" : ""}" style="${p.hand === "rear" ? "color:var(--info);background:rgba(37,99,235,0.09)" : ""}">${cap(p.hand)} hand</span>
      <div style="flex:1"></div><button class="btn-primary" id="showFilm">▶ Show on film</button></div>
    <div class="eyebrow">Shot analysis</div>${cards}`;
  const sf = document.getElementById("showFilm"); if (sf) sf.addEventListener("click", () => seek(p.impact_frame ?? p.start_frame));
}

// Throwing-wrist trajectory across the punch window: outward solid rust, return dashed navy.
function returnPathSVG(p) {
  const a = ARM[throwingSide(S, p)];
  const pts = [];
  for (let f = p.start_frame; f <= p.end_frame; f++) pts.push([S.pose.skeleton[(f * 17 + a.w) * 2], S.pose.skeleton[(f * 17 + a.w) * 2 + 1]]);
  if (pts.length < 2) return "";
  const xs = pts.map((q) => q[0]), ys = pts.map((q) => q[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const W = 180, H = 46, pad = 6;
  const nx = (x) => pad + ((x - minX) / (maxX - minX || 1)) * (W - 2 * pad);
  const ny = (y) => pad + ((y - minY) / (maxY - minY || 1)) * (H - 2 * pad);
  const apex = pts.reduce((bi, _, i) => (Math.abs(p.start_frame + i - p.impact_frame) < Math.abs(p.start_frame + bi - p.impact_frame) ? i : bi), 0);
  const out = pts.slice(0, apex + 1).map((q, i) => `${i ? "L" : "M"}${nx(q[0]).toFixed(1)},${ny(q[1]).toFixed(1)}`).join(" ");
  const ret = pts.slice(apex).map((q, i) => `${i ? "L" : "M"}${nx(q[0]).toFixed(1)},${ny(q[1]).toFixed(1)}`).join(" ");
  return `<div style="background:#F9FAFB;border-radius:7px;padding:4px;margin-top:6px">
    <svg width="100%" viewBox="0 0 ${W} ${H}">
      <path d="${out}" fill="none" stroke="#B85C3D" stroke-width="2"/>
      <path d="${ret}" fill="none" stroke="#1B2A4A" stroke-width="2" stroke-dasharray="3 3"/>
      <circle cx="${nx(pts[0][0])}" cy="${ny(pts[0][1])}" r="3" fill="#1B2A4A"/>
      <circle cx="${nx(pts[apex][0])}" cy="${ny(pts[apex][1])}" r="3" fill="#B85C3D"/>
    </svg></div>`;
}

// ---- helpers ---------------------------------------------------------------
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);
const verdictWord = (v) => (v === "ok" ? "✓ Pass" : v === "bad" ? "✗ Fault" : v === "warn" ? "△ Check" : "— N/A");
function prettyType(t) {
  const base = t.replace(/^(lead|rear)_/, "").replace(/_(head|body)$/, "");
  return cap(base.replace("uppercut", "upper"));
}

boot();
