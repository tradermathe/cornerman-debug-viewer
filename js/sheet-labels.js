// Live ground-truth label fetcher.
//
// The labeler writes punches to a public Google Sheet ("Combined Data"
// tab). Both the xlsx export and the gviz CSV endpoint are reachable
// without auth — we use the CSV endpoint here because it streams faster
// and parses in a few lines of JS, no third-party library needed.
//
// Flow:
//   1. The cache has a sibling `<stem>_labels.json` (produced by
//      `dump_labels.py` in cornerman-backend) which records which
//      `source_video` the cache was clipped from and the
//      `cache_start_sec` offset.
//   2. On every cache load, the viewer triggers fetchLiveLabels() with
//      those fields — we hit the live Sheet, filter to the source video,
//      apply the offset, and return a fresh detections array. New rows
//      added in the labeler since the last dump_labels.py run show up
//      immediately.
//   3. The `_labels.json` doubles as an offline cache: if the network
//      fails the viewer just uses the detections that were serialized
//      into it at dump time.

const PUBLIC_SHEET_ID = "1CewEaweCBw9F-qSvNapiQMNj4wnidHqLA-I19whrly0";
const COMBINED_SHEET = "Combined Data";

const NON_PUNCH = new Set([
  "round_start", "round_end", "rest_start", "rest_end",
]);

// Parse 'mm:ss.mmm' or plain seconds into a number of seconds. Returns null
// on garbage so the caller can skip the row.
export function parseTimestamp(s) {
  if (s == null) return null;
  let t = String(s).trim().replace(/^["']|["']$/g, "").replace(",", ".");
  if (!t) return null;
  if (t.includes(":")) {
    const parts = t.split(":");
    const nums = parts.map(p => Number(p));
    if (nums.some(n => Number.isNaN(n))) return null;
    if (nums.length === 2) return nums[0] * 60 + nums[1];
    if (nums.length === 3) return nums[0] * 3600 + nums[1] * 60 + nums[2];
    return null;
  }
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}

// jab is always lead, cross is always rear; hooks/uppercuts carry their hand.
export function handForLabel(label) {
  const l = String(label || "").trim().toLowerCase();
  if (l.startsWith("jab"))   return "lead";
  if (l.startsWith("cross")) return "rear";
  if (l.startsWith("lead_")) return "lead";
  if (l.startsWith("rear_")) return "rear";
  return null;
}

// Parse a CSV string into an array of row objects keyed by header.
// Quoted fields supported (the gviz CSV always quotes every cell so newlines
// inside a field don't break the row delimiter, but we still implement a
// real parser instead of split(',') just to be safe).
export function parseCsv(text) {
  const rows = [];
  let cur = [], field = "", inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }   // escaped ""
        else { inQuotes = false; }
      } else { field += c; }
    } else {
      if (c === '"') { inQuotes = true; }
      else if (c === ",") { cur.push(field); field = ""; }
      else if (c === "\n") { cur.push(field); rows.push(cur); cur = []; field = ""; }
      else if (c === "\r") { /* skip */ }
      else { field += c; }
    }
  }
  if (field.length || cur.length) { cur.push(field); rows.push(cur); }
  if (rows.length === 0) return [];
  const headers = rows.shift().map(h => h.trim());
  return rows
    .filter(r => r.some(v => v.length))
    .map(r => Object.fromEntries(headers.map((h, i) => [h, r[i] ?? ""])));
}

// Returns true if a sheet row's video_file/video_name matches the cache's
// recorded source video. Source can be a filename, a Drive URL, or a Drive
// file ID — we try all reasonable matches.
function matchesSource(row, sourceVideo) {
  if (!sourceVideo) return false;
  const hint = String(sourceVideo).trim().toLowerCase();
  const name = String(row.video_name || "").trim().toLowerCase();
  const file = String(row.video_file || "").trim().toLowerCase();
  if (!hint) return false;
  if (name === hint || file === hint) return true;
  // Drive ID embedded somewhere.
  const m = hint.match(/[a-z0-9_-]{20,}/i);
  if (m && file.includes(m[0])) return true;
  // Loose substring on filename — useful when the user passes a partial
  // filename like 'shadowbox' from the cache basename.
  if (name.includes(hint)) return true;
  return false;
}

// Re-shape sheet rows into the same detection schema the rule lenses expect.
// All times are converted to cache-local frame indices using `cacheStartSec`
// + `fps`. Out-of-range rows are dropped.
export function rowsToDetections(rows, { cacheStartSec = 0, fps, nFrames }) {
  const detections = [];
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
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
    detections.push({
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
      punch_uuid: r.punch_uuid || null,
      labeler: r.labeler || null,
      reviewed: r.reviewed || null,
    });
  }
  detections.sort((a, b) => a.start_frame - b.start_frame);
  return detections;
}

// Fetch the live "Combined Data" sheet as CSV and return all rows for a
// single source video, mapped to cache-local detection records.
export async function fetchLiveLabels({ sourceVideo, cacheStartSec, fps, nFrames }) {
  if (!sourceVideo) {
    return { source: "labels_sheet_live", detections: [], error: "no source_video" };
  }
  const url =
    `https://docs.google.com/spreadsheets/d/${PUBLIC_SHEET_ID}` +
    `/gviz/tq?tqx=out:csv&sheet=${encodeURIComponent(COMBINED_SHEET)}`;
  let csvText;
  try {
    const resp = await fetch(url, { cache: "no-store" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    csvText = await resp.text();
  } catch (err) {
    return { source: "labels_sheet_live", detections: [], error: err.message };
  }
  const rows = parseCsv(csvText).filter(r => matchesSource(r, sourceVideo));
  const detections = rowsToDetections(rows, { cacheStartSec, fps, nFrames });
  return {
    source: "labels_sheet_live",
    schema_version: 1,
    fps,
    fetched_at: new Date().toISOString(),
    total_rows_matched: rows.length,
    detections,
  };
}
