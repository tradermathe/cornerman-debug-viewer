// Loads punch detections from a sibling `<round>_<engine>_r<N>_punches.json`.
//
// File schema (produced by `dump_punches.py` in cornerman-backend):
//   {
//     "schema_version": 1,
//     "source": "stgcn_punch_detector" | "labels_xlsx" | ...,
//     "fps": 60.0,
//     "n_frames": 1947,
//     "detections": [
//       { "timestamp": 3.817, "start_time": 3.783, "end_time": 3.85,
//         "start_frame": 227, "end_frame": 231,
//         "hand": "rear", "punch_type": "rear_uppercut_head",
//         "category": "power", "n_frames": 4 },
//       ...
//     ]
//   }
//
// Returned shape (consumed by lenses, e.g. step_punch_sync):
//   {
//     source: "stgcn_punch_detector",
//     fps: 60.0,
//     detections: [ { ...; start_frame, end_frame already filled in } ],
//   }
//
// Caller is responsible for handing in the File object — viewer.js picks it
// up from the folder picker's per-engine slot.

export async function loadPunches(file) {
  if (!file) return null;
  let raw;
  try {
    raw = JSON.parse(await file.text());
  } catch (err) {
    throw new Error(`Could not parse ${file.name}: ${err.message}`);
  }
  if (!raw || !Array.isArray(raw.detections)) {
    throw new Error(`${file.name}: missing 'detections' array`);
  }
  const fps = Number(raw.fps) || 30;
  // Ensure every detection has frame indices (older exports might only have
  // seconds). We re-derive defensively rather than trust the file blindly so
  // the rule code always has a stable shape.
  const detections = raw.detections.map((d, i) => {
    const start = Number(d.start_time);
    const end = Number(d.end_time);
    return {
      idx: i,
      timestamp: Number(d.timestamp ?? (start + end) / 2),
      start_time: start,
      end_time: end,
      start_frame: Number.isFinite(d.start_frame)
        ? Number(d.start_frame)
        : Math.round(start * fps),
      end_frame: Number.isFinite(d.end_frame)
        ? Number(d.end_frame)
        : Math.round(end * fps),
      hand: d.hand || "?",                    // "lead" | "rear"
      punch_type: d.punch_type || "?",
      category: d.category || null,           // "jab" | "cross" | "power" | null
      n_frames: Number(d.n_frames) || 0,
    };
  });
  return {
    source: raw.source || "unknown",
    schema_version: Number(raw.schema_version) || 0,
    fps,
    detections,
  };
}
