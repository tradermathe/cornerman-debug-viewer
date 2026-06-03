// Shared loader for the temporal axiality model's per-punch predictions.
//
// train_axiality_temporal.py (cornerman-backend) writes
// `predictions_axiality_temporal.json` next to the Vision caches:
//
//   { "kind": "axiality_temporal", "model": "...", "exported_at": "...",
//     "levels": [0, 0.383, 0.707, 0.924, 1.0],
//     "names":  ["sideways","near_side","diagonal","near_axial","axial"],
//     "metrics": { ...cv_metrics... },
//     "punches": { "<punch_uuid>": { pred_axiality, pred_bucket,
//                                    gt_axiality, gt_bucket } } }
//
// The viewer auto-collects every `predictions_*.json` into
// state.predictionFiles; we pick the newest `predictions_axiality_*.json`,
// parse it once, and expose a synchronous lookup by punch_uuid. Predictions are
// out-of-fold (honest held-out), so the lens shows the same generalization the
// CV metrics report. The model only scored LABELED straights from the Sheet —
// `axialityForPunch` returns null for any punch the model never saw (real
// footage), so callers must fall back to the geometric lens for those.
//
// This lives in its own module (not inside forearm_axiality.js) so any rule can
// gate on the learned axiality bucket, not just the standalone lens. Usage:
//   import { ensureAxialityModel, axialityForPunch } from "./axiality_model.js";
//   ensureAxialityModel(state, onReady);           // idempotent; kicks async load
//   const p = axialityForPunch(det.punch_uuid);    // sync; {predAxiality,...}|null

export const AXIALITY_NAMES = ["sideways", "near_side", "diagonal", "near_axial", "axial"];
const NAME_RE = /^predictions_axiality_.*\.json$/i;

let preds = null;     // Map<punch_uuid, {predAxiality, predBucket, gtAxiality, gtBucket}>
let meta = null;      // { names, levels, metrics, model, file, n }
let lastErr = null;   // parse/read error message, or null
let filesRef = null;  // identity + size guard so we reload only on folder change
let filesSize = -1;
let token = 0;        // dedupe: a newer folder supersedes an in-flight load

async function materialize(value) {
  if (value instanceof File) return value;
  if (typeof value?.getFile === "function") {
    try { return await value.getFile(); } catch { return null; }
  }
  return null;
}

// Idempotent. Call freely from mount()/update(); the identity+size guard makes
// repeat calls free. `onReady` (optional) fires once when an async parse lands,
// so the caller can re-render. Resets cached state on every folder change.
export function ensureAxialityModel(state, onReady) {
  const files = state?.predictionFiles;
  const size = files ? files.size : 0;
  if (files === filesRef && size === filesSize) return;  // nothing changed
  filesRef = files; filesSize = size;
  preds = null; meta = null; lastErr = null;
  const myToken = ++token;
  if (!files || size === 0) return;

  (async () => {
    const cands = [];
    for (const [name, value] of files) {
      if (!NAME_RE.test(name)) continue;
      const file = await materialize(value);
      if (file) cands.push({ name, file });
    }
    if (!cands.length) return;                       // no axiality sidecar in folder
    cands.sort((a, b) => (b.file.lastModified || 0) - (a.file.lastModified || 0));
    let nextPreds = null, nextMeta = null, nextErr = null;
    try {
      const parsed = JSON.parse(await cands[0].file.text());
      const m = new Map();
      for (const [uuid, v] of Object.entries(parsed.punches || {})) {
        m.set(uuid, {
          predAxiality: v.pred_axiality, predBucket: v.pred_bucket,
          gtAxiality: v.gt_axiality, gtBucket: v.gt_bucket,
        });
      }
      nextPreds = m;
      nextMeta = {
        names: parsed.names || AXIALITY_NAMES,
        levels: parsed.levels || null,
        metrics: parsed.metrics || null,
        model: parsed.model || "?",
        file: cands[0].name, n: m.size,
      };
    } catch (e) {
      nextErr = e.message;
    }
    if (myToken !== token) return;                   // a newer folder won; drop this
    preds = nextPreds; meta = nextMeta; lastErr = nextErr;
    if (typeof onReady === "function") onReady();
  })();
}

// Sync lookup. null when no sidecar is loaded or the model never scored this
// punch (uuid absent) — callers gate accordingly.
export function axialityForPunch(uuid) {
  return (uuid && preds) ? (preds.get(uuid) || null) : null;
}

export function axialityModelMeta() { return meta; }
export function axialityModelError() { return lastErr; }

export function axialityBucketName(idx) {
  const names = (meta && meta.names) || AXIALITY_NAMES;
  return Number.isInteger(idx) && idx >= 0 && idx < names.length ? names[idx] : "—";
}
