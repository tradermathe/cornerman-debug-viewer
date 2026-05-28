// Audio impact (mel-CNN) — per-round audio impact-classifier prob stream
// overlaid on the round timeline alongside GT punch spans, librosa-detected
// impact moments, and the discrete peaks the model would emit at threshold τ.
//
// Loads data.json from the sibling cornerman-audio-debug-viewer Pages site
// (CORS-allowed). Matches the round currently scoped in the main viewer by
// (cacheBasename, cacheRound) with a `_h264`-suffix-aware fuzzy fallback.
//
// Schema of data.json (one entry per held-out bagwork round):
//   {
//     "rounds": [
//       {
//         "stem": "Killer Heavy Bag Workout … so let's do it!",
//         "round_index": 1,
//         "fold": 0,
//         "duration_sec": 15.29,
//         "fps": 23.976,
//         "round_start_sec": 53.2,
//         "gt_lead":   [{start_sec, end_sec, mid_sec, class}, ...],
//         "gt_rear":   [{...}],
//         "impacts":   [<sec>, ...]             // librosa onset peak per GT span
//         "prob_t":    [<sec>, ...]             // CNN window centers (80 ms hop)
//         "prob_v":    [<float 0..1>, ...]      // CNN impact probability
//       },
//     ]
//   }
//
// Source / regeneration: cornerman-backend/audio_impact_phase2/build_viewer_data.py
// + deploy_to_repo.py. Phase-2 plan: PLAN.md in that folder.

const DATA_URL =
  "https://tradermathe.github.io/cornerman-audio-debug-viewer/data.json";

const TOL = 0.150;       // ±150 ms — matches Phase 1 / Step 5 eval
const PHASE1_BAR_PP = 16.5;

const COLORS = {
  prob:    "#8ab4f8",
  probFill:"rgba(138, 180, 248, 0.20)",
  leadGt:  "#f47174",
  rearGt:  "#6aa9f4",
  impact:  "#56d364",
  peak:    "#ffa657",
  thr:     "rgba(255, 166, 87, 0.55)",
  playhead:"#ff5555",
  grid:    "#333",
  text:    "#aaa",
};

let host;
let dump = null;            // parsed data.json
let dumpError = null;
let activeRound = null;     // round entry currently scoped
let activeStats = null;
let lastCacheKey = null;
let thr = 0.50;
let latestState = null;     // captured on every update() so click handlers
                            // can use the viewer's authoritative state.fps /
                            // state.start_sec (which can drift by a frame
                            // from data.json's per-round values, since the
                            // pose cache meta and predictions JSON were built
                            // by different pipelines).

export const AudioImpactLensRule = {
  id: "audio_impact",
  label: "Audio impact (mel-CNN)",

  mount(_host, state) {
    host = _host;
    host.innerHTML = template();
    mountStageTimeline();
    wireThreshold();
    lastCacheKey = cacheKey(state);
    ensureDumpLoaded().then(() => {
      refreshScope(state);
      paint(state);
    });
  },

  update(state) {
    latestState = state;
    const k = cacheKey(state);
    if (k !== lastCacheKey) {
      lastCacheKey = k;
      refreshScope(state);
    }
    paint(state);
  },

  draw() { /* no-op: rendering goes in #stage-extras, not on the video overlay */ },
};

function cacheKey(state) {
  if (!state.cacheBasename || state.cacheRound == null) return null;
  return `${state.cacheBasename}__r${state.cacheRound}`;
}

// ── DOM template ───────────────────────────────────────────────────────────

function template() {
  return `
    <h2>Audio impact (mel-CNN)</h2>
    <p class="hint">Per-time impact probability from the Phase-2 mel-CNN v3
      classifier (240 ms log-mel window, hard-negative training). The lens
      auto-scopes to the loaded video + round. Source data:
      <a href="${DATA_URL}" target="_blank" rel="noreferrer">data.json</a>.</p>

    <h3>Status</h3>
    <p id="ai-status" class="muted small" style="margin:0 0 10px">loading…</p>

    <h3>Stats <span class="muted small" style="font-weight:400">at τ = <span id="ai-tau">0.50</span></span></h3>
    <div class="stat-grid" style="display:grid;grid-template-columns:auto auto;gap:4px 12px;font-variant-numeric:tabular-nums;margin-bottom:10px">
      <div class="muted small">recall</div><div id="ai-recall">—</div>
      <div class="muted small">hits / GT</div><div id="ai-hits">—</div>
      <div class="muted small">predicted peaks</div><div id="ai-peaks">—</div>
      <div class="muted small">chance</div><div id="ai-chance">—</div>
      <div class="muted small">lift vs chance</div><div id="ai-lift">—</div>
      <div class="muted small">Phase-1 librosa bar</div><div>+${PHASE1_BAR_PP.toFixed(1)} pp</div>
    </div>

    <h3>Threshold τ</h3>
    <input type="range" id="ai-thr" min="0.01" max="0.99" step="0.01" value="0.50" style="width:100%">
    <p class="hint">Slider re-renders the timeline live. Lift is recall minus
      the union coverage of predicted-window-mass within ±${(TOL*1000)|0} ms
      of fired windows.</p>

    <h3>Legend</h3>
    <ul style="list-style:none;padding:0;margin:0 0 10px;font-size:12px;line-height:1.6">
      <li><span style="display:inline-block;width:10px;height:10px;background:${COLORS.prob};vertical-align:-1px;margin-right:6px;border-radius:2px"></span>prob stream</li>
      <li><span style="display:inline-block;width:10px;height:10px;background:${COLORS.leadGt};vertical-align:-1px;margin-right:6px;border-radius:2px"></span>lead-hand GT span</li>
      <li><span style="display:inline-block;width:10px;height:10px;background:${COLORS.rearGt};vertical-align:-1px;margin-right:6px;border-radius:2px"></span>rear-hand GT span</li>
      <li><span style="display:inline-block;width:10px;height:10px;background:${COLORS.impact};vertical-align:-1px;margin-right:6px;border-radius:2px"></span>impact moment (librosa)</li>
      <li><span style="display:inline-block;width:10px;height:10px;background:${COLORS.peak};vertical-align:-1px;margin-right:6px;border-radius:2px"></span>predicted peak ≥ τ</li>
    </ul>

    <details>
      <summary>What this lens shows</summary>
      <p class="hint">Training cards live in
        <code>audio_impact_phase2/SUMMARY_step4_5_melcnn.md</code>. The mel-CNN
        is held-out evaluated 5-fold; each round uses its own fold's model.
        Audio playback comes from the video the main viewer already loaded —
        seek with the scrubber, then watch the orange peaks against the green
        impact markers.</p>
      <p class="hint"><b>Timing.</b> The lens timeline is in round-relative
        seconds. The audio was extracted with <code>ffmpeg -ss round_start_sec
        -t duration_sec</code> from the source video, so prob_t = 0 ↔
        round_start_sec in the source. The playhead is computed from the
        viewer's <code>state.frame / state.fps</code> (the same values that
        drive the scrubber), so it stays locked to the video and audio you
        hear from the video element. Click-to-seek uses the viewer's fps too.</p>
    </details>
  `;
}

// ── Side-panel wiring ──────────────────────────────────────────────────────

function wireThreshold() {
  const slider = host.querySelector("#ai-thr");
  slider.addEventListener("input", () => {
    thr = parseFloat(slider.value);
    host.querySelector("#ai-tau").textContent = thr.toFixed(2);
    // Refresh stats + redraw using cached state via lastCacheKey.
    if (activeRound) {
      activeStats = computeStats(activeRound, thr);
      renderStats();
      // Redraw — fetch current state.frame via a stale snapshot is hard;
      // instead the next update() tick (≤ 60 Hz from video) will redraw.
      drawTimeline(null);
    }
  });
}

// ── Data load ──────────────────────────────────────────────────────────────

let dumpPromise = null;

async function ensureDumpLoaded() {
  if (dump || dumpError) return dump;
  if (!dumpPromise) {
    dumpPromise = fetch(DATA_URL, { cache: "force-cache" })
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(j => { dump = j; })
      .catch(e => {
        dumpError = e.message || String(e);
        const el = host?.querySelector("#ai-status");
        if (el) el.innerHTML = `<span style="color:#e85a5a">failed: ${dumpError}</span>`;
      });
  }
  return dumpPromise;
}

// ── Scope to current round ─────────────────────────────────────────────────

function refreshScope(state) {
  activeRound = null;
  activeStats = null;
  if (!dump || !state.cacheBasename || state.cacheRound == null) {
    paintEmpty();
    return;
  }
  const wantStem = String(state.cacheBasename).replace(/_h264$/, "");
  const wantRi = state.cacheRound;

  let idx = dump.rounds.findIndex(r =>
    r.round_index === wantRi && r.stem === state.cacheBasename);
  if (idx < 0) {
    idx = dump.rounds.findIndex(r =>
      r.round_index === wantRi && r.stem === wantStem);
  }
  if (idx < 0) {
    idx = dump.rounds.findIndex(r =>
      r.round_index === wantRi &&
      (wantStem.includes(r.stem) || r.stem.includes(wantStem)));
  }
  const el = host.querySelector("#ai-status");
  if (idx < 0) {
    activeRound = null;
    if (el) {
      el.innerHTML = `<span class="muted small">no audio data for ` +
        `<code>${state.cacheBasename}</code> · r${state.cacheRound}</span>`;
    }
    activeStats = null;
    renderStats();
    drawEmptyTimeline();
    return;
  }
  activeRound = dump.rounds[idx];
  if (el) {
    const note = (activeRound.stem === state.cacheBasename)
      ? `matched <code>${activeRound.stem}</code> · r${activeRound.round_index} (fold ${activeRound.fold})`
      : `fuzzy-matched (<code>${state.cacheBasename}</code> ↔ ` +
        `<code>${activeRound.stem}</code>) · r${activeRound.round_index} (fold ${activeRound.fold})`;
    el.innerHTML = note;
  }
  activeStats = computeStats(activeRound, thr);
  renderStats();
}

function paintEmpty() {
  const el = host?.querySelector("#ai-status");
  if (el) {
    if (dumpError) {
      el.innerHTML = `<span style="color:#e85a5a">failed: ${dumpError}</span>`;
    } else if (!dump) {
      el.textContent = "loading…";
    } else {
      el.innerHTML = `<span class="muted small">pick a video + round to scope</span>`;
    }
  }
  drawEmptyTimeline();
}

// ── Stats ──────────────────────────────────────────────────────────────────

function findPeaks(t_arr, p_arr, thr) {
  const peaks = [];
  for (let i = 1; i < p_arr.length - 1; i++) {
    if (p_arr[i] >= thr && p_arr[i] > p_arr[i - 1] && p_arr[i] >= p_arr[i + 1]) {
      peaks.push({ t: t_arr[i], p: p_arr[i] });
    }
  }
  return peaks;
}

function computeStats(round, thr) {
  const mids = [
    ...round.gt_lead.map(s => s.mid_sec),
    ...round.gt_rear.map(s => s.mid_sec),
  ];
  let hits = 0;
  for (const m of mids) {
    let best = 0;
    for (let i = 0; i < round.prob_t.length; i++) {
      if (Math.abs(round.prob_t[i] - m) <= TOL) {
        if (round.prob_v[i] > best) best = round.prob_v[i];
      }
    }
    if (best >= thr) hits++;
  }
  const peaks = findPeaks(round.prob_t, round.prob_v, thr);
  // chance = union ±TOL coverage of pred-windows / duration
  const intervals = [];
  for (let i = 0; i < round.prob_v.length; i++) {
    if (round.prob_v[i] >= thr) intervals.push([round.prob_t[i] - TOL, round.prob_t[i] + TOL]);
  }
  intervals.sort((a, b) => a[0] - b[0]);
  let cov = 0;
  if (intervals.length) {
    let [lo, hi] = intervals[0];
    for (let k = 1; k < intervals.length; k++) {
      if (intervals[k][0] <= hi) hi = Math.max(hi, intervals[k][1]);
      else { cov += hi - lo;[lo, hi] = intervals[k]; }
    }
    cov += hi - lo;
  }
  const chance = mids.length > 0 ? Math.min(cov / round.duration_sec, 1) : 0;
  const recall = mids.length > 0 ? hits / mids.length : 0;
  return { N: mids.length, hits, peaks, recall, chance, lift: recall - chance };
}

function renderStats() {
  const set = (id, txt, color) => {
    const el = host.querySelector("#" + id);
    if (!el) return;
    el.textContent = txt;
    if (color) el.style.color = color;
  };
  if (!activeStats) {
    set("ai-recall", "—"); set("ai-hits", "—"); set("ai-peaks", "—");
    set("ai-chance", "—"); set("ai-lift", "—");
    return;
  }
  const s = activeStats;
  set("ai-recall", (s.recall * 100).toFixed(0) + "%");
  set("ai-hits", `${s.hits} / ${s.N}`);
  set("ai-peaks", String(s.peaks.length));
  set("ai-chance", (s.chance * 100).toFixed(0) + "%");
  const lift = s.lift * 100;
  const liftCol = lift >= 5 ? "#56d364" : lift < 0 ? "#e85a5a" : null;
  set("ai-lift", (lift >= 0 ? "+" : "") + lift.toFixed(1) + " pp", liftCol);
}

// ── Stage-extras timeline ──────────────────────────────────────────────────

function mountStageTimeline() {
  const slot = document.getElementById("stage-extras");
  if (!slot) return;
  slot.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.id = "ai-timeline-wrap";
  wrap.style.cssText = "margin-top:12px;padding:10px 12px;background:var(--bg-card);border:1px solid var(--border);border-radius:8px";
  const label = document.createElement("div");
  label.className = "muted small";
  label.style.cssText = "margin-bottom:6px";
  label.textContent = "Audio-impact prob timeline (click to seek)";
  wrap.appendChild(label);
  const canvas = document.createElement("canvas");
  canvas.id = "ai-timeline";
  canvas.style.cssText = "display:block;width:100%;height:160px";
  canvas.width = 1200; canvas.height = 160;
  wrap.appendChild(canvas);
  slot.appendChild(wrap);

  canvas.addEventListener("click", e => {
    if (!activeRound) return;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const labelW = 56;
    const W = rect.width;
    const ratio = (cx - labelW) / Math.max(1, W - labelW - 4);
    const t = Math.max(0, Math.min(activeRound.duration_sec, ratio * activeRound.duration_sec));
    // Use the VIEWER's fps for the frame conversion, not the data.json's
    // per-round fps. They should match, but the viewer's value is what
    // the scrubber + video element actually round-trip on.
    const fps = (latestState && latestState.fps) || activeRound.fps;
    const f = Math.round(t * fps);
    seekHack(f);
  });
}

function drawEmptyTimeline() {
  const cv = document.getElementById("ai-timeline");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = "#666";
  ctx.font = "12px monospace";
  ctx.fillText("no data scoped to this round", 12, 20);
}

function paint(state) {
  drawTimeline(state);
}

function drawTimeline(state) {
  const cv = document.getElementById("ai-timeline");
  if (!cv || !activeRound) return;

  // Resize internal pixel buffer to CSS width (match punch_classifier pattern).
  const rect = cv.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const cssW = rect.width, cssH = 160;
  if (cv.width !== Math.round(cssW * dpr) || cv.height !== Math.round(cssH * dpr)) {
    cv.width = Math.round(cssW * dpr); cv.height = Math.round(cssH * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = cssW, H = cssH;
  ctx.clearRect(0, 0, W, H);

  const r = activeRound;
  const labelW = 56;
  const padR = 4;
  const plotW = W - labelW - padR;
  const dur = r.duration_sec;
  const t2x = t => labelW + (t / dur) * plotW;

  // Vertical layout
  const PROB_TOP = 8, PROB_H = H * 0.55;
  const GT_LEAD_Y = PROB_TOP + PROB_H + 8;
  const GT_REAR_Y = GT_LEAD_Y + 12;
  const IMP_Y = GT_REAR_Y + 14;
  const PEAK_Y = IMP_Y + 12;
  const AXIS_Y = H - 12;

  // Axis ticks
  ctx.strokeStyle = COLORS.grid; ctx.lineWidth = 1;
  ctx.fillStyle = COLORS.text;
  ctx.font = "10px monospace";
  const step = dur > 60 ? 5 : (dur > 20 ? 2 : 1);
  for (let t = 0; t <= dur; t += step) {
    const x = t2x(t);
    ctx.beginPath(); ctx.moveTo(x, PROB_TOP); ctx.lineTo(x, AXIS_Y); ctx.stroke();
    ctx.fillText(`${t}s`, x + 2, AXIS_Y + 10);
  }
  // 0.5 prob grid
  ctx.strokeStyle = "#444"; ctx.setLineDash([2, 3]);
  const y50 = PROB_TOP + 0.5 * PROB_H;
  ctx.beginPath(); ctx.moveTo(labelW, y50); ctx.lineTo(W - padR, y50); ctx.stroke();
  ctx.setLineDash([]);

  // Threshold line on prob plot
  ctx.strokeStyle = COLORS.thr; ctx.setLineDash([4, 3]);
  const thrY = PROB_TOP + (1 - thr) * PROB_H;
  ctx.beginPath(); ctx.moveTo(labelW, thrY); ctx.lineTo(W - padR, thrY); ctx.stroke();
  ctx.setLineDash([]);

  // Prob filled area
  ctx.fillStyle = COLORS.probFill;
  ctx.beginPath();
  ctx.moveTo(t2x(r.prob_t[0]), PROB_TOP + PROB_H);
  for (let i = 0; i < r.prob_t.length; i++) {
    ctx.lineTo(t2x(r.prob_t[i]), PROB_TOP + (1 - r.prob_v[i]) * PROB_H);
  }
  ctx.lineTo(t2x(r.prob_t[r.prob_t.length - 1]), PROB_TOP + PROB_H);
  ctx.closePath(); ctx.fill();

  ctx.strokeStyle = COLORS.prob; ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (let i = 0; i < r.prob_t.length; i++) {
    const x = t2x(r.prob_t[i]);
    const y = PROB_TOP + (1 - r.prob_v[i]) * PROB_H;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // GT spans
  ctx.fillStyle = COLORS.leadGt;
  for (const s of r.gt_lead) {
    ctx.fillRect(t2x(s.start_sec), GT_LEAD_Y, t2x(s.end_sec) - t2x(s.start_sec), 8);
  }
  ctx.fillStyle = COLORS.rearGt;
  for (const s of r.gt_rear) {
    ctx.fillRect(t2x(s.start_sec), GT_REAR_Y, t2x(s.end_sec) - t2x(s.start_sec), 8);
  }

  // Impact markers
  ctx.strokeStyle = COLORS.impact; ctx.lineWidth = 1.5;
  for (const t of r.impacts) {
    const x = t2x(t);
    ctx.beginPath(); ctx.moveTo(x, IMP_Y - 4); ctx.lineTo(x, IMP_Y + 4); ctx.stroke();
  }

  // Peaks
  const peaks = activeStats ? activeStats.peaks : findPeaks(r.prob_t, r.prob_v, thr);
  ctx.fillStyle = COLORS.peak;
  for (const p of peaks) {
    ctx.beginPath();
    ctx.arc(t2x(p.t), PEAK_Y, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  // Row labels
  ctx.fillStyle = COLORS.text; ctx.font = "10px monospace";
  ctx.fillText("prob", 6, PROB_TOP + PROB_H * 0.5 + 3);
  ctx.fillText("lead", 6, GT_LEAD_Y + 7);
  ctx.fillText("rear", 6, GT_REAR_Y + 7);
  ctx.fillText("impact", 6, IMP_Y + 3);
  ctx.fillText("peaks", 6, PEAK_Y + 3);

  // Playhead — derive time in round from state.frame / state.fps. This is
  // the viewer's authoritative round-relative time, so the playhead line
  // and the prob_t / GT / impact times stay consistent even when data.json's
  // per-round fps differs slightly from the pose cache's.
  if (state && state.fps > 0 && typeof state.frame === "number") {
    const tNow = state.frame / state.fps;
    if (tNow >= 0 && tNow <= dur) {
      const x = t2x(tNow);
      ctx.strokeStyle = COLORS.playhead; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x, PROB_TOP); ctx.lineTo(x, AXIS_Y); ctx.stroke();
    }
  }
}

// ── Seek shim ──────────────────────────────────────────────────────────────
// Same trick punch_classifier uses: dispatch a synthetic input on the
// viewer's scrubber instead of reaching into private state.
function seekHack(f) {
  const slider = document.getElementById("scrubber");
  if (!slider) return;
  slider.value = f;
  slider.dispatchEvent(new Event("input"));
}
