// Orientation labeler — keyboard-driven lens for labelling the boxer's
// facing direction on individual frames. Output feeds an ML model that
// every depth-sensitive rule can consume.
//
// Convention:
//   0°    boxer facing the camera (chest toward camera)
//  +90°   boxer's right side to camera (camera sees their left cheek)
//  -90°   boxer's left side to camera  (camera sees their right cheek)
//  ±180°  boxer's back to camera
//  ±45°   quarter-turn toward camera
//  ±135°  quarter-turn away from camera
//
// Keyboard (numpad mnemonic — the layout maps to angles spatially):
//   7=-135   8=180   9=+135
//   4=-90    5=0     6=+90
//   1=-45    2=skip  3=+45
//
// Labels are saved to localStorage under "ae_orientation_labels" as a
// JSON map. Use the "Export CSV" button to download for training.

const BIN_LABELS = {
  "5":   { angle: 0,    name: "facing camera" },
  "8":   { angle: 180,  name: "back to camera" },
  "4":   { angle: -90,  name: "left side to camera" },
  "6":   { angle: 90,   name: "right side to camera" },
  "1":   { angle: -45,  name: "facing camera, turned left" },
  "3":   { angle: 45,   name: "facing camera, turned right" },
  "7":   { angle: -135, name: "back to camera, turned left" },
  "9":   { angle: 135,  name: "back to camera, turned right" },
  "2":   { angle: null, name: "unclear / skip" },
};
const ANGLE_LIST = [0, 45, 90, 135, 180, -135, -90, -45];

const STORAGE_KEY = "ae_orientation_labels";

let host;
let state_ref = null;
let keyHandler = null;
let cfg = { labeler: "" };
let videoFingerprint = null;
let allLabels = loadLabels();

// ─── storage ───────────────────────────────────────────────────────────────

function loadLabels() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch { return {}; }
}
function saveLabels() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(allLabels)); }
  catch (e) { console.warn("orientation_labeler: localStorage save failed", e); }
}
function labelKey(video, round, frame) {
  return `${video}__r${round ?? 0}__f${frame}`;
}
function setLabel(video, round, frame, angle, labeler) {
  const k = labelKey(video, round, frame);
  if (angle === null) {
    // Skip — record it so we don't keep showing the same frame
    allLabels[k] = { angle: null, labeler, ts: Date.now() };
  } else {
    allLabels[k] = { angle, labeler, ts: Date.now() };
  }
  saveLabels();
}
function getLabel(video, round, frame) {
  return allLabels[labelKey(video, round, frame)] || null;
}
function countLabelsForVideo(video, round) {
  const prefix = `${video}__r${round ?? 0}__`;
  let scored = 0, skipped = 0;
  for (const k of Object.keys(allLabels)) {
    if (!k.startsWith(prefix)) continue;
    if (allLabels[k].angle === null) skipped++;
    else scored++;
  }
  return { scored, skipped, total: scored + skipped };
}

// ─── video / round identification ──────────────────────────────────────────

function videoIdFromState(state) {
  // Use the basename of whatever video is loaded; fall back to pose engine.
  // The labels need to outlive a session, so we pick a stable identifier.
  const file = state.videoFileName || "";
  return file.replace(/\.(mp4|mov|webm)$/i, "");
}
function roundFromState(state) {
  // Round number is encoded in the cache filename (eg "..._vision_r3.npy").
  // The viewer doesn't expose it directly; we can infer from pose.start_sec
  // being non-zero — but the safest stable id is the round_start match.
  // For labelling purposes, the (video, frame) pair is enough; round goes
  // into the CSV when we export.
  return null;  // export uses frame index only
}

// ─── lens API ──────────────────────────────────────────────────────────────

export const OrientationLabelerRule = {
  id: "orientation_labeler",
  label: "Orientation labeler (train data)",

  skeletonStyle() {
    return { boneColor: "rgba(255,255,255,0.40)", boneWidth: 1.5, jointRadius: 3 };
  },

  mount(_host, state) {
    host = _host;
    state_ref = state;
    videoFingerprint = videoIdFromState(state);
    cfg.labeler = localStorage.getItem("ae_orientation_labeler") || "";
    allLabels = loadLabels();

    host.innerHTML = renderTemplate();
    wireUI();
    attachKeyHandler();
    updatePanel();
  },

  update(state) {
    state_ref = state;
    updatePanel();
  },

  draw(ctx, state) {
    // Minimal canvas indicator — small label at top-center showing the
    // current frame's existing label (if any). Keeps it out of the way so
    // the labeler can see the body clearly.
    const s = state.renderScale || 1;
    const label = getLabel(videoFingerprint, roundFromState(state), state.frame);
    let txt;
    if (!label) {
      txt = `frame ${state.frame} · unlabeled`;
    } else if (label.angle === null) {
      txt = `frame ${state.frame} · skipped`;
    } else {
      txt = `frame ${state.frame} · ${label.angle > 0 ? "+" : ""}${label.angle}°`;
    }
    ctx.save();
    ctx.font = `${16 * s}px ui-monospace, monospace`;
    const tw = ctx.measureText(txt).width;
    const x = (ctx.canvas.width / 2) - (tw / 2) - 8 * s;
    const y = 16 * s;
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(x, y, tw + 16 * s, 24 * s);
    ctx.fillStyle = label?.angle === null ? "rgba(255,200,80,0.95)"
                    : (label ? "rgba(95,217,122,0.95)" : "rgba(255,255,255,0.75)");
    ctx.fillText(txt, x + 8 * s, y + 17 * s);
    ctx.restore();
  },
};

// ─── UI ────────────────────────────────────────────────────────────────────

function renderTemplate() {
  // Numpad-laid-out button grid: 9 cells, each shows the angle/skip and the key
  function btn(key) {
    const b = BIN_LABELS[key];
    const ang = b.angle === null ? "skip" : `${b.angle > 0 ? "+" : ""}${b.angle}°`;
    return `
      <button class="orient-btn" data-key="${key}" title="${b.name}">
        <div class="orient-key">[${key}]</div>
        <div class="orient-ang">${ang}</div>
      </button>`;
  }

  return `
    <style>
      .orient-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 6px;
        margin: 8px 0;
      }
      .orient-btn {
        background: #1f1f1f;
        color: #eee;
        border: 1px solid #444;
        border-radius: 6px;
        padding: 12px 6px;
        cursor: pointer;
        font-family: inherit;
        text-align: center;
        font-size: 12px;
        line-height: 1.4;
      }
      .orient-btn:hover { background: #2a2a2a; border-color: #666; }
      .orient-btn.is-selected {
        background: #2d4d2d; border-color: #5fd97a;
      }
      .orient-btn.is-skip.is-selected {
        background: #524422; border-color: #f5b945;
      }
      .orient-key { color: #888; font-size: 10px; letter-spacing: 0.05em; }
      .orient-ang { font-weight: 600; margin-top: 2px; }
      .ol-row { display: flex; gap: 8px; align-items: center; margin: 6px 0; }
      .ol-row label { flex: 0 0 80px; font-size: 12px; color: #aaa; }
      .ol-input {
        flex: 1; padding: 5px 8px; background: #1a1a1a; color: #eee;
        border: 1px solid #444; border-radius: 4px; font-family: inherit; font-size: 13px;
      }
      .ol-btn {
        padding: 6px 12px; background: #2563eb; color: white;
        border: none; border-radius: 4px; cursor: pointer; font-family: inherit;
        font-size: 13px;
      }
      .ol-btn.secondary { background: #444; }
      .ol-btn:hover { filter: brightness(1.15); }
      .ol-progress {
        height: 18px; background: #1a1a1a; border: 1px solid #444; border-radius: 4px;
        overflow: hidden; position: relative; font-size: 11px;
      }
      .ol-progress-bar { height: 100%; background: #5fd97a; transition: width 0.2s; }
      .ol-progress-text {
        position: absolute; inset: 0; display: flex; align-items: center;
        justify-content: center; color: white; mix-blend-mode: difference;
      }
    </style>

    <h2>Orientation labeler</h2>
    <p class="hint">
      Label individual frames with the boxer's facing direction. Use the
      numpad — its layout matches the angles spatially. After each
      keypress the lens auto-advances to a random unlabeled frame in this
      round. Labels persist in your browser (localStorage); export CSV
      when done.
    </p>

    <div class="ol-row">
      <label for="ol-labeler">labeler</label>
      <input type="text" id="ol-labeler" class="ol-input" placeholder="your name (saved per browser)" value="${cfg.labeler}" />
    </div>

    <h3>Current frame</h3>
    <p class="hint" id="ol-current">—</p>

    <div class="orient-grid" id="ol-buttons">
      ${btn("7")}${btn("8")}${btn("9")}
      ${btn("4")}${btn("5")}${btn("6")}
      ${btn("1")}${btn("2")}${btn("3")}
    </div>

    <div class="ol-row">
      <button class="ol-btn secondary" id="ol-next">Next random unlabeled (n)</button>
      <button class="ol-btn secondary" id="ol-clear">Clear this frame's label</button>
    </div>

    <h3>Progress (this video)</h3>
    <div class="ol-progress">
      <div class="ol-progress-bar" id="ol-bar" style="width:0%"></div>
      <div class="ol-progress-text" id="ol-progress-text">—</div>
    </div>
    <p class="hint muted small" id="ol-dist" style="margin:6px 0 0 0">—</p>

    <h3>Export</h3>
    <div class="ol-row">
      <button class="ol-btn" id="ol-export-video">Export CSV (this video)</button>
      <button class="ol-btn secondary" id="ol-export-all">Export CSV (all videos)</button>
    </div>
    <p class="hint muted small" style="margin:6px 0 0 0">
      CSV columns: <code>video, frame, label, labeler, ts</code>. Skipped
      frames have <code>label</code> empty.
    </p>
  `;
}

function wireUI() {
  // Click buttons
  for (const btn of host.querySelectorAll(".orient-btn")) {
    btn.addEventListener("click", () => {
      applyLabel(btn.dataset.key);
    });
  }
  host.querySelector("#ol-labeler").addEventListener("change", (e) => {
    cfg.labeler = e.target.value.trim();
    localStorage.setItem("ae_orientation_labeler", cfg.labeler);
  });
  host.querySelector("#ol-next").addEventListener("click", () => seekToRandomUnlabeled());
  host.querySelector("#ol-clear").addEventListener("click", () => clearCurrentLabel());
  host.querySelector("#ol-export-video").addEventListener("click", () => exportCSV("video"));
  host.querySelector("#ol-export-all").addEventListener("click", () => exportCSV("all"));
}

function attachKeyHandler() {
  // Detach any previous handler we owned (the lens can re-mount when video changes)
  if (keyHandler) document.removeEventListener("keydown", keyHandler, true);
  keyHandler = (e) => {
    // Self-clean if the lens is no longer mounted
    if (!document.body.contains(host)) {
      document.removeEventListener("keydown", keyHandler, true);
      keyHandler = null;
      return;
    }
    // Ignore when the user is typing in an input
    const tag = e.target?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA") return;

    const k = e.key;
    if (k === "n" || k === "N") {
      e.preventDefault(); e.stopPropagation();
      seekToRandomUnlabeled(); return;
    }
    if (BIN_LABELS[k]) {
      e.preventDefault(); e.stopPropagation();
      applyLabel(k);
    }
  };
  // Capture phase so we beat the viewer's own keydown handlers (which
  // listen for ArrowLeft/Right etc; numpad keys aren't bound there but
  // we want to be sure no future binding will steal "n" from us).
  document.addEventListener("keydown", keyHandler, true);
}

function applyLabel(key) {
  if (!state_ref) return;
  const bin = BIN_LABELS[key];
  if (!bin) return;
  setLabel(videoFingerprint, roundFromState(state_ref), state_ref.frame,
           bin.angle, cfg.labeler);
  updatePanel();
  window.__viewerRedraw?.();
  // Auto-advance to next unlabeled frame in this round
  seekToRandomUnlabeled();
}

function clearCurrentLabel() {
  if (!state_ref) return;
  const k = labelKey(videoFingerprint, roundFromState(state_ref), state_ref.frame);
  delete allLabels[k];
  saveLabels();
  updatePanel();
  window.__viewerRedraw?.();
}

function seekToRandomUnlabeled() {
  if (!state_ref) return;
  const N = state_ref.n_frames || 0;
  if (N <= 0) return;
  // Build the unlabeled set
  const unlabeled = [];
  for (let f = 0; f < N; f++) {
    const k = labelKey(videoFingerprint, roundFromState(state_ref), f);
    if (!allLabels[k]) unlabeled.push(f);
  }
  if (!unlabeled.length) {
    setText("ol-current", "Every frame in this round is already labeled or skipped.");
    return;
  }
  const f = unlabeled[Math.floor(Math.random() * unlabeled.length)];
  const slider = document.getElementById("scrubber");
  if (!slider) return;
  slider.value = String(f);
  slider.dispatchEvent(new Event("input"));
}

function updatePanel() {
  if (!host || !state_ref) return;
  const N = state_ref.n_frames || 0;
  const f = state_ref.frame;
  const label = getLabel(videoFingerprint, roundFromState(state_ref), f);
  let currTxt;
  if (!label) {
    currTxt = `frame ${f}/${N - 1} · unlabeled`;
  } else if (label.angle === null) {
    currTxt = `frame ${f}/${N - 1} · skipped (${label.labeler || "anon"})`;
  } else {
    const sign = label.angle > 0 ? "+" : "";
    currTxt = `frame ${f}/${N - 1} · ${sign}${label.angle}° (${label.labeler || "anon"})`;
  }
  setText("ol-current", currTxt);

  // Highlight which button matches the current frame's label
  for (const btn of host.querySelectorAll(".orient-btn")) {
    btn.classList.remove("is-selected");
    btn.classList.toggle("is-skip", btn.dataset.key === "2");
  }
  if (label) {
    const selKey = Object.keys(BIN_LABELS).find(k =>
      BIN_LABELS[k].angle === label.angle
    );
    if (selKey) {
      const b = host.querySelector(`.orient-btn[data-key="${selKey}"]`);
      if (b) b.classList.add("is-selected");
    }
  }

  // Progress bar
  const stats = countLabelsForVideo(videoFingerprint, roundFromState(state_ref));
  const pct = N > 0 ? (stats.total / N) * 100 : 0;
  const bar = host.querySelector("#ol-bar");
  if (bar) bar.style.width = `${pct.toFixed(1)}%`;
  setText("ol-progress-text",
    `${stats.total}/${N} labeled (${stats.scored} scored + ${stats.skipped} skipped)`);

  // Per-bin distribution
  const dist = labelDistribution(videoFingerprint, roundFromState(state_ref));
  setText("ol-dist", formatDistribution(dist));
}

function labelDistribution(video, round) {
  const prefix = `${video}__r${round ?? 0}__`;
  const counts = {};
  for (const ang of ANGLE_LIST) counts[ang] = 0;
  counts["skip"] = 0;
  for (const k of Object.keys(allLabels)) {
    if (!k.startsWith(prefix)) continue;
    const v = allLabels[k];
    if (v.angle === null) counts["skip"]++;
    else if (v.angle in counts) counts[v.angle]++;
  }
  return counts;
}

function formatDistribution(d) {
  const parts = ANGLE_LIST.map(a => {
    const sign = a > 0 ? "+" : "";
    return `${sign}${a}°: ${d[a]}`;
  });
  parts.push(`skip: ${d["skip"]}`);
  return parts.join(" · ");
}

function exportCSV(scope) {
  const rows = ["video,frame,label,labeler,ts"];
  for (const [key, v] of Object.entries(allLabels)) {
    const m = key.match(/^(.+?)__r(\d+|null)__f(\d+)$/);
    if (!m) continue;
    const [, vid, _r, frame] = m;
    if (scope === "video" && vid !== videoFingerprint) continue;
    const ang = v.angle === null ? "" : v.angle;
    const lab = (v.labeler || "").replace(/[",\n]/g, " ");
    const vidEsc = vid.includes(",") || vid.includes('"')
      ? `"${vid.replace(/"/g, '""')}"` : vid;
    const ts = new Date(v.ts || 0).toISOString();
    rows.push(`${vidEsc},${frame},${ang},${lab},${ts}`);
  }
  const blob = new Blob([rows.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = scope === "video"
    ? `orientation_labels_${videoFingerprint || "video"}.csv`
    : `orientation_labels_all.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function setText(id, value) {
  const el = host?.querySelector("#" + id);
  if (el) el.textContent = value;
}
