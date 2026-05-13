// Live ground-truth label fetcher.
//
// The labeler writes punches to a public Google Sheet ("Combined Data"
// tab). The gviz CSV endpoint is reachable without auth, so we can pull
// labels live in the browser — no `dump_labels.py` rerun, no sidecar JSON.
//
// Flow (option A — auto-match):
//   1. The viewer loads a pose cache. The cache filename gives us a
//      basename (e.g. `30 MIN SHADOWBOXING…_h264` from
//      `30 MIN SHADOWBOXING…_h264_vision_r0.npz`).
//   2. We fetch the Sheet CSV once per session (cached in memory), then
//      look for a unique `video_name` whose stem matches the cache
//      basename (case-insensitive, substring either direction).
//   3. If exactly one source matches we filter to its rows; if none, we
//      report it and the lens falls back to ST-GCN punches / heuristic.
//   4. Times are mapped to cache-relative frames using
//      `state.pose.start_sec` (read from the cache's `_meta.json`) and
//      `state.pose.fps`.
//
// Adding a new label in the labeler → click Refresh from Sheet in the
// step+punch-sync lens; the in-session cache is bypassed and the live
// rows are re-pulled.

const PUBLIC_SHEET_ID = "1CewEaweCBw9F-qSvNapiQMNj4wnidHqLA-I19whrly0";
const COMBINED_SHEET = "Combined Data";
const FORM_LABELS_SHEET = "Combined Form Labels";
// Form-rule fields we surface on each detection so per-rule lenses can
// score their predictions against the coach's verdict.
const FORM_LABEL_KEYS = [
  "rule_hand_extended", "rule_hand_low", "rule_hand_ushape",
  "rule_hip_rotation", "rule_rear_heel_lift", "rule_resting_hand",
  "rule_extension", "rule_punch_height",
];

const NON_PUNCH = new Set([
  "round_start", "round_end", "rest_start", "rest_end",
]);

// In-session cache of the parsed CSV. The Sheet's ~10k rows are ~1.5 MB
// over the wire — fetching once per session and reusing across cache
// picks keeps the viewer snappy. `force=true` bypasses for the Refresh
// button.
let cachedRows = null;
let cachedFetchedAt = 0;
let cachedFormByUuid = null;     // { punch_uuid -> {rule_*: 'pass'|'fail'|'unclear'|''} }
const CACHE_TTL_MS = 5 * 60 * 1000;

export function clearCache() {
  cachedRows = null;
  cachedFetchedAt = 0;
  cachedFormByUuid = null;
}

// ── Public API ──────────────────────────────────────────────────────────────

export function parseTimestamp(s) {
  if (s == null) return null;
  let t = String(s).trim().replace(/^["']|["']$/g, "").replace(",", ".");
  if (!t) return null;
  if (t.includes(":")) {
    const parts = t.split(":").map(Number);
    if (parts.some(n => Number.isNaN(n))) return null;
    if (parts.length === 2) return parts[0] * 60 + parts[1];
    if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
    return null;
  }
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}

export function handForLabel(label) {
  const l = String(label || "").trim().toLowerCase();
  if (l.startsWith("jab"))   return "lead";
  if (l.startsWith("cross")) return "rear";
  if (l.startsWith("lead_")) return "lead";
  if (l.startsWith("rear_")) return "rear";
  return null;
}

export function parseCsv(text) {
  const rows = [];
  let cur = [], field = "", inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }
        else { inQuotes = false; }
      } else { field += c; }
    } else {
      if (c === '"')      inQuotes = true;
      else if (c === ",") { cur.push(field); field = ""; }
      else if (c === "\n") { cur.push(field); rows.push(cur); cur = []; field = ""; }
      else if (c === "\r") { /* skip */ }
      else field += c;
    }
  }
  if (field.length || cur.length) { cur.push(field); rows.push(cur); }
  if (rows.length === 0) return [];
  const headers = rows.shift().map(h => h.trim());
  return rows
    .filter(r => r.some(v => v.length))
    .map(r => Object.fromEntries(headers.map((h, i) => [h, r[i] ?? ""])));
}

// Pull rows from both label sheets, cached. We fetch them in parallel
// because every cache load wants both anyway: Combined Data gives us the
// per-punch timing + video, Combined Form Labels gives us the per-rule
// pass/fail verdicts that lenses score themselves against.
export async function fetchRows({ force = false } = {}) {
  if (!force && cachedRows && Date.now() - cachedFetchedAt < CACHE_TTL_MS) {
    return {
      rows: cachedRows, formByUuid: cachedFormByUuid,
      fetchedAt: cachedFetchedAt, fromCache: true,
    };
  }
  const sheetUrl = (name) =>
    `https://docs.google.com/spreadsheets/d/${PUBLIC_SHEET_ID}` +
    `/gviz/tq?tqx=out:csv&sheet=${encodeURIComponent(name)}`;

  const [combinedResp, formResp] = await Promise.all([
    fetch(sheetUrl(COMBINED_SHEET),    { cache: "no-store" }),
    fetch(sheetUrl(FORM_LABELS_SHEET), { cache: "no-store" }),
  ]);
  if (!combinedResp.ok) throw new Error(`HTTP ${combinedResp.status} on ${COMBINED_SHEET}`);
  if (!formResp.ok)     throw new Error(`HTTP ${formResp.status} on ${FORM_LABELS_SHEET}`);

  cachedRows = parseCsv(await combinedResp.text());

  // Build the punch_uuid → form-verdicts map for fast join.
  const formRows = parseCsv(await formResp.text());
  cachedFormByUuid = new Map();
  for (const r of formRows) {
    const uuid = (r.punch_uuid || "").trim();
    if (!uuid) continue;
    const verdicts = {};
    for (const k of FORM_LABEL_KEYS) {
      const v = (r[k] || "").trim().toLowerCase();
      if (v) verdicts[k] = v;
    }
    if (Object.keys(verdicts).length) cachedFormByUuid.set(uuid, verdicts);
  }

  cachedFetchedAt = Date.now();
  return {
    rows: cachedRows, formByUuid: cachedFormByUuid,
    fetchedAt: cachedFetchedAt, fromCache: false,
  };
}

// Normalize a filename / basename for fuzzy matching: drop extension,
// lowercase, collapse non-alphanum to single spaces, trim.
function normalize(s) {
  return String(s)
    .toLowerCase()
    .replace(/\.[a-z0-9]+$/, "")     // drop one extension
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

// Find the source video in the Sheet that best matches a cache basename.
// Returns { name, confidence: 'exact' | 'substr' | 'tokens', n_rows } or null.
// Strategy:
//   1. Exact (normalized) match — strongest.
//   2. One direction substring (cache normalized contains video normalized,
//      or vice versa).
//   3. Tokens: every alphanum token of the shorter side appears as a token
//      in the longer side. Catches `name_r0` vs `name`, `name_h264` vs
//      `name`, etc.
// If multiple candidates tie, pick the one with the most label rows (i.e.
// the most specific source video).
export function findSourceByBasename(rows, cacheBasename) {
  if (!cacheBasename) return null;
  // Strip the cache-shape suffix `_<engine>_r<N>` so the basename we match
  // against is just the source name, e.g.
  //   `30 MIN SHADOWBOXING…_h264_vision_r0` → `30 MIN SHADOWBOXING…_h264`
  const cb = cacheBasename.replace(/_(yolo|vision)_r\d+$/i, "");
  const cbN = normalize(cb);
  if (!cbN) return null;

  // Tally counts per video for picking among ties.
  const counts = new Map();
  for (const r of rows) {
    const v = r.video_name;
    if (!v) continue;
    counts.set(v, (counts.get(v) || 0) + 1);
  }
  const names = [...counts.keys()];

  const cbTokens = new Set(cbN.split(" ").filter(Boolean));
  let exact = null;
  const substrHits = [];
  const tokenHits = [];

  for (const n of names) {
    const nN = normalize(n);
    if (!nN) continue;
    if (nN === cbN) { exact = n; break; }
    if (cbN.includes(nN) || nN.includes(cbN)) {
      substrHits.push({ name: n, score: Math.min(nN.length, cbN.length) });
      continue;
    }
    const nTokens = nN.split(" ").filter(Boolean);
    const shorter = nTokens.length < cbTokens.size ? nTokens : [...cbTokens];
    const longer  = nTokens.length < cbTokens.size ? new Set(cbTokens) : new Set(nTokens);
    if (shorter.length && shorter.every(t => longer.has(t))) {
      tokenHits.push({ name: n, score: shorter.length });
    }
  }

  const pickByCount = arr => {
    arr.sort((a, b) => (counts.get(b.name) || 0) - (counts.get(a.name) || 0));
    return arr[0]?.name || null;
  };

  if (exact)             return { name: exact,                 confidence: "exact",  n_rows: counts.get(exact) };
  if (substrHits.length) {
    const w = pickByCount(substrHits);
    return { name: w, confidence: "substr", n_rows: counts.get(w) };
  }
  if (tokenHits.length) {
    const w = pickByCount(tokenHits);
    return { name: w, confidence: "tokens", n_rows: counts.get(w) };
  }
  return null;
}

// Reshape filtered Sheet rows into the detection schema rule lenses expect.
// `formByUuid` (optional) is the map from fetchRows() — if present, every
// detection gets its rule_* verdict columns attached.
export function rowsToDetections(rows, { cacheStartSec = 0, fps, nFrames, formByUuid = null }) {
  const detections = [];
  for (const r of rows) {
    const label = String(r.label || "").trim().toLowerCase();
    if (!label || NON_PUNCH.has(label)) continue;
    const sStart = parseTimestamp(r.start_sec);
    const sEnd   = parseTimestamp(r.end_sec);
    if (sStart == null || sEnd == null || sEnd <= sStart) continue;
    const localStart = sStart - cacheStartSec;
    const localEnd   = sEnd   - cacheStartSec;
    if (localEnd <= 0 || localStart >= nFrames / fps) continue;

    const sf = Math.max(0, Math.round(localStart * fps));
    const ef = Math.min(nFrames - 1, Math.round(localEnd * fps));
    if (ef - sf < 1) continue;
    const punch_uuid = (r.punch_uuid || "").trim() || null;
    const det = {
      idx: detections.length,
      timestamp: (localStart + localEnd) / 2,
      start_time: localStart,
      end_time: localEnd,
      start_frame: sf,
      end_frame: ef,
      hand: handForLabel(label),
      punch_type: label,
      category: null,
      n_frames: ef - sf + 1,
      stance: String(r.stance || "").trim().toLowerCase() || null,
      punch_uuid,
      labeler: r.labeler || null,
      reviewed: r.reviewed || null,
    };
    if (formByUuid && punch_uuid && formByUuid.has(punch_uuid)) {
      const verdicts = formByUuid.get(punch_uuid);
      for (const [k, v] of Object.entries(verdicts)) det[k] = v;
    }
    detections.push(det);
  }
  detections.sort((a, b) => a.start_frame - b.start_frame);
  return detections;
}

// Top-level convenience: given a cache basename + cache offset + fps + frame
// count, fetch (cached) the Sheet, auto-match a source, and return a
// detections array. The lens code is one call away from live labels.
export async function fetchLiveLabels({
  cacheBasename, cacheStartSec = 0, fps, nFrames, force = false,
}) {
  let fetched;
  try {
    fetched = await fetchRows({ force });
  } catch (err) {
    return { error: err.message };
  }
  const match = findSourceByBasename(fetched.rows, cacheBasename);
  if (!match) {
    return {
      error: "no source-video auto-match in the Sheet for this cache",
      cacheBasename,
      fetched_at: fetched.fetchedAt,
      from_cache: fetched.fromCache,
    };
  }
  const videoRows = fetched.rows.filter(r => r.video_name === match.name);
  const detections = rowsToDetections(videoRows, {
    cacheStartSec, fps, nFrames, formByUuid: fetched.formByUuid,
  });
  return {
    source: "labels_sheet_live",
    schema_version: 1,
    source_video: match.name,
    match_confidence: match.confidence,
    n_rows_for_video: match.n_rows,
    cache_start_sec: cacheStartSec,
    fps,
    fetched_at: fetched.fetchedAt,
    from_cache: fetched.fromCache,
    total_punches: detections.length,
    detections,
  };
}
