// Punch classifier (GT vs Pred) — visual replacement for the throwaway
// `eventacc_video_overlay.html` cell in
// `classifier_7class_kfold_eventacc_17joints_v6.ipynb`. The notebook's HTML
// fuzzy-matches dropped videos to its embedded prediction set by the first
// 20 chars of the filename, which silently picks the wrong round when two
// videos share a prefix — that's the "missynced with the wrong video" bug.
// Here we route the loaded video → its predictions by the exact
// (cacheBasename, round_index) pair the viewer already maintains.
//
// Input: ONE JSON file containing every round's per-frame GT + predictions.
// Auto-loaded from any file named `predictions_*.json` anywhere in the
// connected Drive folder (or in a manual cache-folder pick); manual file
// picker is kept as an override. The lens auto-scopes to the round matching
// state.cacheBasename + state.cacheRound. The round dropdown stays editable
// so you can manually pick a different round if the auto-match is wrong
// (e.g. cache stem includes `_h264` and the notebook used the raw mp4 name).
//
// If multiple `predictions_*.json` files exist (different training runs),
// the auto-load picker shows all of them sorted newest-first; the lens
// defaults to the most-recently-modified one.
//
// JSON schema (v1):
//   {
//     "schema_version": 1,
//     "model": "<checkpoint suffix>",
//     "exported_at": "<iso>",
//     "lead_class_names": ["idle", "jab_head", "jab_body", ...],   // 7 entries
//     "rear_class_names": ["idle", "cross_head", ...],             // 7 entries
//     "rounds": [
//       {
//         "video_stem": "Shadow boxing workout with an app",   // = Path(video_name).stem
//         "round_index": 0,
//         "n_frames": 1947,
//         "fps": 30.0,
//         "round_start_sec": 4.3,
//         "lead_pred":  [0,0,1,1,1,0,...],   // length n_frames, int 0..6
//         "rear_pred":  [...],
//         "lead_truth": [...],
//         "rear_truth": [...]
//       },
//       ...
//     ]
//   }
//
// Notebook export snippet (paste as a new cell after the `predictions...pt`
// save in cell 9; or run after loading the .pt via cell 16):
//
//   import json, numpy as np
//   from pathlib import Path
//   from datetime import datetime, timezone
//   rounds_out = []
//   for e in all_predictions:
//       stem = Path(e['video_name']).stem
//       ri = int(e['round_id'].rsplit('_r', 1)[1])
//       rounds_out.append({
//           'video_stem': stem, 'round_index': ri,
//           'n_frames': int(e['n_frames']), 'fps': float(e['fps']),
//           'round_start_sec': float(e['round_start_sec']),
//           'lead_pred':  np.asarray(e['lead_pred'],  dtype=np.int16).tolist(),
//           'rear_pred':  np.asarray(e['rear_pred'],  dtype=np.int16).tolist(),
//           'lead_truth': np.asarray(e['lead_truth'], dtype=np.int16).tolist(),
//           'rear_truth': np.asarray(e['rear_truth'], dtype=np.int16).tolist(),
//       })
//   out = {
//       'schema_version': 1, 'model': 'classifier_7class_eventacc_17j_vision_v1',
//       'exported_at': datetime.now(timezone.utc).isoformat(),
//       'lead_class_names': LEAD_CLASS_NAMES,
//       'rear_class_names': REAR_CLASS_NAMES,
//       'rounds': rounds_out,
//   }
//   path = '/content/drive/MyDrive/boxing_ai/models/predictions_eventacc_7class_latest.json'
//   Path(path).write_text(json.dumps(out))
//   print(f'Wrote {len(rounds_out)} rounds → {path}')

const COLORS = {
  // GT events — color by class family; muted hue when missed (red stripes).
  gtFamily: {
    straight: "#5fd97a",     // green
    hook:     "#3aa14d",
    uppercut: "#65d27a",
    unknown:  "#5fd97a",
  },
  gtMiss:        "#e85a5a",
  gtMissStripe:  "#7a2424",

  // Pred events — orange (straight family) / purple (hook+uppercut). False
  // alarms keep the family hue but get a hatched fill; mistypes are bright
  // magenta to call out wrong-class predictions specifically.
  predFamily: {
    straight: "#f5a23c",
    hook:     "#9b5cf6",
    uppercut: "#bf73ff",
    unknown:  "#f5a23c",
  },
  predMistype:   "#e040fb",
  predFAStripe:  "rgba(0,0,0,0.4)",

  playhead:      "rgba(255,255,255,0.85)",
  rowBg:         "#1d222b",
};

const CLASS_NAMES_FALLBACK = {
  lead: ["idle","jab_head","jab_body","lead_hook_head","lead_hook_body","lead_uppercut_head","lead_uppercut_body"],
  rear: ["idle","cross_head","cross_body","rear_hook_head","rear_hook_body","rear_uppercut_head","rear_uppercut_body"],
};

let host;
let dump = null;          // parsed JSON
let activeRound = null;   // round entry currently scoped
let signals = null;       // derived events + per-round stats
let lastPose = null;
let lastCacheKey = null;

export const PunchClassifierRule = {
  id: "punch_classifier",
  label: "Punch classifier (GT vs Pred)",

  skeletonStyle() {
    return { boneColor: "rgba(255,255,255,0.20)", boneWidth: 1.4, jointRadius: 2.5 };
  },

  mount(_host, state) {
    host = _host;
    host.innerHTML = template();
    lastPose = state.pose;
    lastCacheKey = cacheKey(state);
    wireFilePicker(state);
    wireAutoLoad(state);
    wireRoundPicker(state);
    refreshScope(state);
    // Kick off auto-load from state.predictionFiles (if any). Async; will
    // populate `dump` and refresh on completion.
    tryAutoLoad(state);
  },

  update(state) {
    // Re-scope when the loaded video / round changes.
    const k = cacheKey(state);
    if (k !== lastCacheKey || state.pose !== lastPose) {
      lastCacheKey = k;
      lastPose = state.pose;
      refreshScope(state);
    }
    if (!activeRound || !signals) return;
    drawTimeline(host.querySelector("#pc-timeline"), signals, state.frame);
    renderHud(state);
  },

  draw(ctx, state) {
    if (!activeRound || !signals) return;
    drawCanvasHud(ctx, state);
  },
};

// ── DOM template ───────────────────────────────────────────────────────────

function template() {
  return `
    <h2>Punch classifier — GT vs Pred</h2>
    <p class="hint">Per-frame ground truth and 7-class classifier output for
      this round. Drop the predictions JSON below; the lens auto-scopes to the
      loaded video + round.</p>

    <h3>Predictions JSON</h3>
    <div id="pc-auto-row" style="display:none;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:6px">
      <select id="pc-auto-select" style="flex:1;min-width:0;background:var(--bg-elev);color:var(--fg);border:1px solid var(--border);border-radius:6px;padding:6px 8px;font:inherit"></select>
      <span class="muted small">auto-loaded from Drive</span>
    </div>
    <p id="pc-auto-empty" class="hint" style="margin:0 0 6px">
      No <code>predictions_*.json</code> found in your Drive folder yet —
      run the notebook export snippet (see "Notebook export snippet" at the
      bottom of this panel) and the file will appear here automatically.
    </p>
    <details class="manual-fallback">
      <summary id="pc-manual-summary">Override with a manual file</summary>
      <label class="folder-pick">
        <input type="file" id="pc-file" accept="application/json,.json">
        <span class="muted small" id="pc-file-status">no file loaded</span>
      </label>
    </details>

    <h3>Round</h3>
    <select id="pc-round" disabled>
      <option value="">— load a predictions JSON —</option>
    </select>
    <p class="hint" id="pc-round-hint"></p>

    <h3>Round stats</h3>
    <div class="metric-grid">
      <div class="metric"><div class="metric-label">Recall (caught)</div><div class="metric-val" id="pc-recall">—</div><div class="metric-sub muted">hits / n_gt — mistypes count as caught</div></div>
      <div class="metric"><div class="metric-label">Type acc (when caught)</div><div class="metric-val" id="pc-type-acc">—</div><div class="metric-sub muted">correct / hits — class right given detection</div></div>
      <div class="metric"><div class="metric-label">Precision</div><div class="metric-val" id="pc-precision">—</div></div>
      <div class="metric"><div class="metric-label">EventAcc</div><div class="metric-val" id="pc-ea">—</div><div class="metric-sub muted">correct / (n_gt + fa) — penalizes mistype = miss</div></div>
      <div class="metric"><div class="metric-label">N punches (GT/Pred)</div><div class="metric-val" id="pc-counts">—</div></div>
    </div>
    <p class="hint" id="pc-stat-line"></p>

    <h3>Timeline</h3>
    <canvas id="pc-timeline" width="340" height="120"></canvas>
    <p class="hint">4 tracks: GT-lead, Pred-lead, GT-rear, Pred-rear. Green =
      GT hit · red striped = GT miss · orange/purple = Pred correct
      (straight / hook+uppercut family) · same hue hatched = false alarm ·
      magenta = mistype. Click the timeline to seek.</p>

    <details class="manual-fallback">
      <summary>Notebook export snippet</summary>
      <p class="hint">Paste into the classifier notebook after
        <code>all_predictions</code> is in memory (after cell 8 or after
        cell 16 reload) to (re)write the JSON next to the .pt file:</p>
      <pre class="hint" style="font-size:11px;white-space:pre-wrap;font-family:ui-monospace,monospace">import json, numpy as np
from pathlib import Path
from datetime import datetime, timezone
rounds_out = []
for e in all_predictions:
    stem = Path(e['video_name']).stem
    ri = int(e['round_id'].rsplit('_r', 1)[1])
    rounds_out.append({
        'video_stem': stem, 'round_index': ri,
        'n_frames': int(e['n_frames']), 'fps': float(e['fps']),
        'round_start_sec': float(e['round_start_sec']),
        'lead_pred':  np.asarray(e['lead_pred'],  dtype=np.int16).tolist(),
        'rear_pred':  np.asarray(e['rear_pred'],  dtype=np.int16).tolist(),
        'lead_truth': np.asarray(e['lead_truth'], dtype=np.int16).tolist(),
        'rear_truth': np.asarray(e['rear_truth'], dtype=np.int16).tolist(),
    })
out = {'schema_version': 1,
       'model': 'classifier_7class_eventacc_17j_vision_v1',
       'exported_at': datetime.now(timezone.utc).isoformat(),
       'lead_class_names': LEAD_CLASS_NAMES,
       'rear_class_names': REAR_CLASS_NAMES,
       'rounds': rounds_out}
path = '/content/drive/MyDrive/boxing_ai/models/predictions_eventacc_7class_latest.json'
Path(path).write_text(json.dumps(out))
print(f'Wrote {len(rounds_out)} rounds → {path}')</pre>
    </details>
  `;
}

// ── File picker + scope wiring ─────────────────────────────────────────────

function wireFilePicker(state) {
  const input = host.querySelector("#pc-file");
  input.addEventListener("change", async () => {
    const f = input.files?.[0];
    if (!f) return;
    setStatus(`reading ${f.name}…`);
    try {
      const text = await f.text();
      const parsed = JSON.parse(text);
      if (!parsed || !Array.isArray(parsed.rounds)) {
        throw new Error("JSON is missing a `rounds` array");
      }
      dump = parsed;
      setStatus(`${f.name} · ${dump.rounds.length} rounds · ` +
        `model ${dump.model || "?"} · exported ${dump.exported_at || "?"}`);
      populateRoundPicker();
      refreshScope(state);
    } catch (err) {
      dump = null;
      setStatus(`error: ${err.message}`);
      populateRoundPicker();
      refreshScope(state);
    }
  });
}

function setStatus(text) {
  const el = host.querySelector("#pc-file-status");
  if (el) el.textContent = text;
}

// Auto-load wiring: state.predictionFiles is a Map<filename, File | handle>
// populated by viewer.js from the Drive walker and the manual cache-folder
// picker. The lens shows a dropdown of every found file and defaults to the
// most-recently-modified one. The manual file picker stays as an override.
function wireAutoLoad(state) {
  const sel = host.querySelector("#pc-auto-select");
  if (!sel) return;
  sel.addEventListener("change", async () => {
    const name = sel.value;
    if (!name) return;
    await loadAutoFile(state, name);
  });
}

async function tryAutoLoad(state) {
  const files = state?.predictionFiles;
  const row = host.querySelector("#pc-auto-row");
  const sel = host.querySelector("#pc-auto-select");
  const empty = host.querySelector("#pc-auto-empty");
  const summary = host.querySelector("#pc-manual-summary");
  const setEmpty = () => {
    if (row)     row.style.display = "none";
    if (empty)   empty.style.display = "";
    if (summary) summary.textContent = "Load a predictions JSON";
  };
  if (!files || files.size === 0) { setEmpty(); return; }
  // Materialize to {name, file} so we can compare mtimes. drive.toFile()
  // works for both raw Files and FileSystemFileHandles.
  const items = [];
  for (const [name, value] of files) {
    let file;
    if (value instanceof File) file = value;
    else if (typeof value?.getFile === "function") {
      try { file = await value.getFile(); }
      catch { continue; }
    } else continue;
    items.push({ name, file });
  }
  if (!items.length) { setEmpty(); return; }
  // Newest first.
  items.sort((a, b) => (b.file.lastModified || 0) - (a.file.lastModified || 0));
  if (row)   row.style.display = "flex";
  if (empty) empty.style.display = "none";
  if (sel) {
    sel.innerHTML = "";
    for (const it of items) {
      const o = document.createElement("option");
      o.value = it.name;
      const t = it.file.lastModified
        ? new Date(it.file.lastModified).toLocaleString(undefined, { dateStyle: "short", timeStyle: "short" })
        : "?";
      o.textContent = `${it.name} · ${t}`;
      sel.appendChild(o);
    }
    sel.value = items[0].name;
  }
  if (summary) summary.textContent = "Override with a manual file";
  await loadAutoFile(state, items[0].name, items.map(it => [it.name, it.file]));
}

// Load (or reload) a specific auto-found file by name. The `materializedHint`
// is an optional pre-materialized [name, File] list from tryAutoLoad so we
// don't have to re-call getFile() if we already did once.
async function loadAutoFile(state, name, materializedHint) {
  const files = state?.predictionFiles;
  if (!files || !files.has(name)) return;
  let file = null;
  if (materializedHint) {
    const hit = materializedHint.find(([n]) => n === name);
    if (hit) file = hit[1];
  }
  if (!file) {
    const value = files.get(name);
    if (value instanceof File) file = value;
    else if (typeof value?.getFile === "function") {
      try { file = await value.getFile(); } catch { return; }
    }
  }
  if (!file) return;
  setStatus(`auto: reading ${file.name}…`);
  try {
    const text = await file.text();
    const parsed = JSON.parse(text);
    if (!parsed || !Array.isArray(parsed.rounds)) {
      throw new Error("JSON is missing a `rounds` array");
    }
    dump = parsed;
    setStatus(`auto · ${file.name} · ${dump.rounds.length} rounds · ` +
      `model ${dump.model || "?"} · exported ${dump.exported_at || "?"}`);
    populateRoundPicker();
    refreshScope(state);
  } catch (err) {
    dump = null;
    setStatus(`auto-load error: ${err.message}`);
    populateRoundPicker();
    refreshScope(state);
  }
}

function wireRoundPicker(state) {
  const sel = host.querySelector("#pc-round");
  sel.addEventListener("change", () => {
    if (!dump) return;
    const idx = parseInt(sel.value, 10);
    if (!Number.isFinite(idx)) return;
    activeRound = dump.rounds[idx] || null;
    signals = activeRound ? deriveSignals(activeRound, dump) : null;
    renderStats();
    renderHud(state);
    drawTimeline(host.querySelector("#pc-timeline"), signals, state.frame);
    // Force a viewer redraw so the canvas HUD updates immediately.
    if (typeof window !== "undefined" && window.__viewerRedraw) window.__viewerRedraw();
  });
}

function populateRoundPicker() {
  const sel = host.querySelector("#pc-round");
  sel.innerHTML = "";
  if (!dump || !dump.rounds.length) {
    sel.innerHTML = `<option value="">— load a predictions JSON —</option>`;
    sel.disabled = true;
    return;
  }
  // Group by video_stem; show each round with its stats so the user can spot
  // good/bad ones by EventAcc without leaving the picker.
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "— pick a round —";
  sel.appendChild(placeholder);
  dump.rounds.forEach((r, i) => {
    const o = document.createElement("option");
    o.value = String(i);
    const s = quickStats(r);
    const ea = s.eventAcc == null ? "—" : (s.eventAcc * 100).toFixed(0) + "%";
    o.textContent = `${r.video_stem} · r${r.round_index} · EA ${ea} (${s.nGt} GT / ${s.nPred} pred)`;
    sel.appendChild(o);
  });
  sel.disabled = false;
}

function cacheKey(state) {
  if (!state.cacheBasename || state.cacheRound == null) return null;
  return `${state.cacheBasename}__r${state.cacheRound}`;
}

function refreshScope(state) {
  if (!dump || !dump.rounds.length) {
    activeRound = null;
    signals = null;
    renderStats();
    return;
  }
  // Try exact match first, then `<stem>` substring either direction (cache
  // basename often has `_h264` appended that the notebook's stem doesn't).
  const wantStem = state.cacheBasename;
  const wantRi   = state.cacheRound;
  let idx = -1;
  if (wantStem != null && wantRi != null) {
    idx = dump.rounds.findIndex(r =>
      r.round_index === wantRi && r.video_stem === wantStem);
    if (idx < 0) {
      idx = dump.rounds.findIndex(r =>
        r.round_index === wantRi &&
        (wantStem.includes(r.video_stem) || r.video_stem.includes(wantStem)));
    }
  }
  const sel = host.querySelector("#pc-round");
  if (idx >= 0) {
    sel.value = String(idx);
    activeRound = dump.rounds[idx];
    signals = deriveSignals(activeRound, dump);
    const r = activeRound;
    const note = (r.video_stem === wantStem)
      ? `auto-matched to <code>${r.video_stem} · r${r.round_index}</code>`
      : `auto-matched (fuzzy: cache <code>${wantStem || "?"}</code> ↔ ` +
        `JSON <code>${r.video_stem}</code>) · r${r.round_index}`;
    host.querySelector("#pc-round-hint").innerHTML = note;
  } else {
    sel.value = "";
    activeRound = null;
    signals = null;
    host.querySelector("#pc-round-hint").innerHTML = wantStem
      ? `no round in this JSON matches cache <code>${wantStem}</code> r${wantRi} — ` +
        `pick one manually above.`
      : `load a video so the lens can auto-match a round.`;
  }
  renderStats();
}

// ── Signal derivation ──────────────────────────────────────────────────────

function deriveSignals(round, dumpRoot) {
  const fps = round.fps || 30;
  const leadClassNames = dumpRoot?.lead_class_names || CLASS_NAMES_FALLBACK.lead;
  const rearClassNames = dumpRoot?.rear_class_names || CLASS_NAMES_FALLBACK.rear;

  const leadGt   = runLength(round.lead_truth);
  const rearGt   = runLength(round.rear_truth);
  const leadPred = runLength(round.lead_pred);
  const rearPred = runLength(round.rear_pred);

  const leadTag = matchAndTag(leadGt, leadPred);
  const rearTag = matchAndTag(rearGt, rearPred);

  // Combine per-hand stats into a single per-round EventAcc.
  const totalGt   = leadTag.stats.nGt + rearTag.stats.nGt;
  const totalPred = leadTag.stats.nPred + rearTag.stats.nPred;
  const totalCorrect = leadTag.stats.correct + rearTag.stats.correct;
  const totalHits = leadTag.stats.hits + rearTag.stats.hits;
  const totalFA   = leadTag.stats.fa + rearTag.stats.fa;
  const denom = totalGt + totalFA;

  return {
    fps,
    n_frames: round.n_frames,
    leadClassNames,
    rearClassNames,
    lead: leadTag,
    rear: rearTag,
    stats: {
      eventAcc:  denom > 0  ? totalCorrect / denom    : null,
      recall:    totalGt > 0   ? totalHits / totalGt    : null,
      precision: totalPred > 0 ? totalHits / totalPred  : null,
      typeAcc:   totalHits > 0 ? totalCorrect / totalHits : null,
      nGt: totalGt, nPred: totalPred,
      correct: totalCorrect, missed: totalGt - totalHits,
      mistyped: totalHits - totalCorrect, fa: totalFA,
    },
  };
}

// Run-length encode a per-frame label array into typed events. Class 0 is
// idle; any run of class >= 1 is one event whose `type` is the (constant)
// class value across the run. Mirrors notebook's `extract_typed_events`.
function runLength(labels) {
  if (!labels?.length) return [];
  const out = [];
  let i = 0;
  const n = labels.length;
  while (i < n) {
    const c = labels[i];
    if (c > 0) {
      let j = i;
      while (j < n && labels[j] === c) j++;
      out.push({ start_frame: i, end_frame: j - 1, type: c });
      i = j;
    } else i++;
  }
  return out;
}

// One-to-one GT↔Pred matching by midpoint-inside-window, sorted by midpoint
// distance. Mirrors notebook's `match_and_tag_events` / `midpoint_in` with
// frame-space inputs (the notebook uses seconds; frames are equivalent given
// shared fps).
function matchAndTag(gt, pred) {
  const candidates = [];
  for (let gi = 0; gi < gt.length; gi++) {
    const g = gt[gi];
    const gMid = (g.start_frame + g.end_frame) / 2;
    for (let pi = 0; pi < pred.length; pi++) {
      const p = pred[pi];
      const pMid = (p.start_frame + p.end_frame) / 2;
      const midIn = (p.start_frame <= gMid && gMid <= p.end_frame) ||
                    (g.start_frame <= pMid && pMid <= g.end_frame);
      if (midIn) candidates.push({ d: Math.abs(gMid - pMid), gi, pi });
    }
  }
  candidates.sort((a, b) => a.d - b.d);
  const usedGt = new Set();
  const usedPred = new Set();
  const pairs = new Map();  // gi -> pi
  let correct = 0;
  for (const c of candidates) {
    if (usedGt.has(c.gi) || usedPred.has(c.pi)) continue;
    usedGt.add(c.gi);
    usedPred.add(c.pi);
    pairs.set(c.gi, c.pi);
    if (gt[c.gi].type === pred[c.pi].type) correct++;
  }
  const gtTagged = gt.map((g, gi) => ({
    ...g,
    status: usedGt.has(gi) ? "hit" : "miss",
    matched_pi: pairs.get(gi) ?? null,
  }));
  const predTagged = pred.map((p, pi) => {
    let status = "fa";
    let matched_gi = null;
    if (usedPred.has(pi)) {
      for (const [gi, ppi] of pairs.entries()) if (ppi === pi) matched_gi = gi;
      status = gt[matched_gi].type === p.type ? "correct" : "mistype";
    }
    return { ...p, status, matched_gi };
  });
  const hits = usedGt.size;
  const fa = pred.length - usedPred.size;
  return {
    gt: gtTagged,
    pred: predTagged,
    stats: {
      nGt: gt.length, nPred: pred.length,
      hits, correct, mistyped: hits - correct,
      missed: gt.length - hits, fa,
    },
  };
}

// Quick per-round stats for the dropdown — same shape as the full signals
// stats but without retaining the tagged event lists.
function quickStats(round) {
  const leadGt   = runLength(round.lead_truth);
  const rearGt   = runLength(round.rear_truth);
  const leadPred = runLength(round.lead_pred);
  const rearPred = runLength(round.rear_pred);
  const l = matchAndTag(leadGt, leadPred).stats;
  const r = matchAndTag(rearGt, rearPred).stats;
  const nGt   = l.nGt + r.nGt;
  const nPred = l.nPred + r.nPred;
  const correct = l.correct + r.correct;
  const denom = nGt + l.fa + r.fa;
  return {
    eventAcc: denom > 0 ? correct / denom : null,
    nGt, nPred,
  };
}

// ── Rendering ──────────────────────────────────────────────────────────────

function renderStats() {
  const set = (id, html) => {
    const el = host.querySelector("#" + id);
    if (el) el.innerHTML = html;
  };
  if (!signals) {
    set("pc-ea", "—"); set("pc-recall", "—"); set("pc-type-acc", "—");
    set("pc-precision", "—"); set("pc-counts", "—"); set("pc-stat-line", "");
    return;
  }
  const s = signals.stats;
  const pct = v => v == null ? "—" : (v * 100).toFixed(0) + "%";
  set("pc-ea", pct(s.eventAcc));
  set("pc-recall", pct(s.recall));
  set("pc-type-acc", pct(s.typeAcc));
  set("pc-precision", pct(s.precision));
  set("pc-counts", `${s.nGt} / ${s.nPred}`);
  set("pc-stat-line",
    `<span class="muted small">` +
    `correct ${s.correct} · missed ${s.missed} · false + ${s.fa} · mistyped ${s.mistyped}` +
    `</span>`);
}

function renderHud(_state) {
  // The HUD is drawn on the main video canvas (see drawCanvasHud) rather than
  // a side-panel element so it visually anchors next to the boxer. This
  // function is kept as a stub for future per-hand details we might want
  // in the side panel.
}

function drawCanvasHud(ctx, state) {
  const f = state.frame;
  const leadGt   = findEvent(signals.lead.gt, f);
  const leadPred = findEvent(signals.lead.pred, f);
  const rearGt   = findEvent(signals.rear.gt, f);
  const rearPred = findEvent(signals.rear.pred, f);
  const s = state.renderScale || 1;

  drawHudBox(ctx, "Lead", leadGt, leadPred,
    signals.leadClassNames, { x: 8 * s, y: 8 * s, align: "left" }, s);
  drawHudBox(ctx, "Rear", rearGt, rearPred,
    signals.rearClassNames, { x: ctx.canvas.width - 8 * s, y: 8 * s, align: "right" }, s);
}

function drawHudBox(ctx, label, gtEv, predEv, classNames, pos, scale) {
  const fontPx  = Math.round(13 * scale);
  const smallPx = Math.round(11 * scale);
  ctx.save();
  ctx.font = `bold ${fontPx}px ui-monospace, "SF Mono", monospace`;

  const gtName   = classNames[gtEv?.type ?? 0]   || "?";
  const predName = classNames[predEv?.type ?? 0] || "?";

  // 3 lines: header (hand), GT, Pred.
  const lines = [
    { text: label,                                       color: "#dddddd",      bg: "rgba(0,0,0,0.65)" },
    { text: `GT:   ${gtEv  ? gtName   : "idle"}`,        color: gtColor(gtEv),    bg: "rgba(0,0,0,0.65)" },
    { text: `Pred: ${predLine(predEv, gtEv, predName)}`, color: predColor(predEv, gtEv), bg: "rgba(0,0,0,0.65)" },
  ];
  const pad = 5 * scale;
  const lineH = fontPx + 4 * scale;
  let width = 0;
  for (const ln of lines) width = Math.max(width, ctx.measureText(ln.text).width);
  const boxW = width + pad * 2;
  const boxH = lineH * lines.length + pad * 2;
  const x = pos.align === "right" ? pos.x - boxW : pos.x;
  const y = pos.y;
  ctx.fillStyle = "rgba(0,0,0,0.65)";
  ctx.fillRect(x, y, boxW, boxH);
  // Inner text.
  let ty = y + pad + fontPx;
  for (const ln of lines) {
    ctx.fillStyle = ln.color;
    ctx.fillText(ln.text, x + pad, ty);
    ty += lineH;
  }
  ctx.restore();
}

function predLine(predEv, gtEv, predName) {
  if (!predEv) {
    // Mid-GT but pred silent → MISS callout.
    if (gtEv) return "MISS";
    return "idle";
  }
  if (predEv.status === "correct") return `${predName} ✓`;
  if (predEv.status === "mistype") return `${predName} (wrong)`;
  if (predEv.status === "fa")      return `${predName} (false+)`;
  return predName;
}

function gtColor(gtEv) {
  if (!gtEv)                  return "#888888";
  if (gtEv.status === "miss") return COLORS.gtMiss;
  return familyColor(gtEv.type, COLORS.gtFamily);
}
function predColor(predEv, gtEv) {
  if (!predEv && gtEv)                  return COLORS.gtMiss;
  if (!predEv)                          return "#888888";
  if (predEv.status === "mistype")      return COLORS.predMistype;
  return familyColor(predEv.type, COLORS.predFamily);
}

function findEvent(events, frame) {
  for (const e of events) {
    if (frame >= e.start_frame && frame <= e.end_frame) return e;
  }
  return null;
}

// ── Timeline ───────────────────────────────────────────────────────────────

let timelineWired = false;

function drawTimeline(canvas, sig, frame) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!sig) return;

  // 4 tracks stacked: GT-lead, Pred-lead, GT-rear, Pred-rear.
  const tracks = [
    { events: sig.lead.gt,   isGt: true,  label: "GT lead",   classNames: sig.leadClassNames },
    { events: sig.lead.pred, isGt: false, label: "Pred lead", classNames: sig.leadClassNames },
    { events: sig.rear.gt,   isGt: true,  label: "GT rear",   classNames: sig.rearClassNames },
    { events: sig.rear.pred, isGt: false, label: "Pred rear", classNames: sig.rearClassNames },
  ];
  const labelW = 56;
  const trackH = 16;
  const gap = 4;
  const trackTop = 4;
  const N = sig.n_frames;
  const xmap = f => labelW + (f / Math.max(1, N - 1)) * (W - labelW - 4);

  ctx.font = "10px ui-monospace, monospace";
  for (let i = 0; i < tracks.length; i++) {
    const t = tracks[i];
    const y = trackTop + i * (trackH + gap);
    // Row background.
    ctx.fillStyle = COLORS.rowBg;
    ctx.fillRect(labelW, y, W - labelW - 4, trackH);
    // Row label.
    ctx.fillStyle = "#8a93a3";
    ctx.fillText(t.label, 4, y + trackH - 4);
    // Events.
    for (const ev of t.events) {
      const x1 = xmap(ev.start_frame);
      const x2 = Math.max(x1 + 2, xmap(ev.end_frame));
      drawEventBar(ctx, x1, y, x2 - x1, trackH, ev, t.isGt);
    }
  }

  // Playhead across all tracks.
  const ph = xmap(frame);
  ctx.strokeStyle = COLORS.playhead;
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(ph, 0); ctx.lineTo(ph, H); ctx.stroke();

  // One-time wire click-to-seek.
  if (!timelineWired) {
    canvas.addEventListener("click", e => {
      if (!signals) return;
      const rect = canvas.getBoundingClientRect();
      const cx = (e.clientX - rect.left) * (canvas.width / rect.width);
      const ratio = (cx - labelW) / Math.max(1, W - labelW - 4);
      const f = Math.round(ratio * (signals.n_frames - 1));
      seekHack(Math.max(0, Math.min(signals.n_frames - 1, f)));
    });
    timelineWired = true;
  }
}

function drawEventBar(ctx, x, y, w, h, ev, isGt) {
  const color = isGt
    ? (ev.status === "miss" ? COLORS.gtMiss : familyColor(ev.type, COLORS.gtFamily))
    : (ev.status === "mistype" ? COLORS.predMistype : familyColor(ev.type, COLORS.predFamily));
  ctx.fillStyle = color;
  ctx.fillRect(x, y + 1, w, h - 2);
  // Hatched overlay for GT miss and Pred false alarm.
  const hatched = (isGt && ev.status === "miss") || (!isGt && ev.status === "fa");
  if (hatched) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y + 1, w, h - 2);
    ctx.clip();
    ctx.strokeStyle = isGt ? COLORS.gtMissStripe : COLORS.predFAStripe;
    ctx.lineWidth = 1;
    for (let k = -h; k < w + h; k += 4) {
      ctx.beginPath();
      ctx.moveTo(x + k, y + 1);
      ctx.lineTo(x + k + h, y + h - 1);
      ctx.stroke();
    }
    ctx.restore();
  }
  // Border for visibility on dark background.
  ctx.strokeStyle = "rgba(0,0,0,0.45)";
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, y + 1.5, Math.max(1, w - 1), h - 3);
}

function familyColor(typ, palette) {
  return palette[classFamily(typ)] || palette.unknown;
}

// Map class index → display family. Indices 1/2 (jab/cross variants) are
// "straight"; 3/4 are "hook"; 5/6 are "uppercut". 0 is idle (returns the
// "unknown" key, which is unused since idle events never become bars).
function classFamily(typ) {
  if (typ === 1 || typ === 2) return "straight";
  if (typ === 3 || typ === 4) return "hook";
  if (typ === 5 || typ === 6) return "uppercut";
  return "unknown";
}

// ── Seek shim ──────────────────────────────────────────────────────────────
// The viewer's scrubber owns the canonical "go to frame F" path, so the
// lens dispatches a synthetic input event on it instead of reaching into
// viewer-private state. Same trick step_punch_sync uses.
function seekHack(f) {
  const slider = document.getElementById("scrubber");
  if (!slider) return;
  slider.value = f;
  slider.dispatchEvent(new Event("input"));
}
