// Demo rule descriptors — bridge from real `analysis.rules` to dashboard cards.
//
// Each descriptor is PURE and DOM-free: evaluate(analysis, punch|null) → a small
// presentation object. Per-punch rules read `rules[id].perPunch` (the loader's
// camelCase view of the sidecar's `per_punch` array) and join to the selected
// punch BY TIMESTAMP (rows are a subset — only scored punches appear).
//
// We surface VERDICT + REAL METRIC, never an invented 0–100 score. Field names
// match the real on-device sidecar (see docs/demo-data-contract.md):
//   arm_extension    → verdict, peak_bend, skip_reason "axial"
//   hit_height       → zone, height_frac, flag (no verdict field)
//   hand_return_path → verdict, return_sec, re_guarded
//   stance_width     → session-level (extras.mean_sep_ratio)
//   pivot_rate       → session-level (extras.secPerPivot / pivotCount)

const sevToVerdict = (sev) => (sev === "none" ? "ok" : sev === "severe" ? "bad" : "warn");
const pick = (o, ...keys) => { for (const k of keys) if (o && o[k] != null) return o[k]; return undefined; };

// Find this punch's per-punch row by timestamp (rows share the detection's ts).
function rowFor(analysis, id, punch) {
  const r = analysis?.rules?.[id];
  if (!r) return { state: "absent" };
  if (!Array.isArray(r.perPunch)) return { state: "no_perpunch", cue: r.coachCue };
  const row = r.perPunch.find((p) => Math.abs((p.timestamp ?? -99) - punch.timestamp) < 0.06);
  if (!row) return { state: "no_row", cue: r.coachCue };
  return { state: "ok", row, cue: r.coachCue };
}
const skipCard = (title, row, cue) => ({ title, verdict: "skip", headline: "Not scored",
  sub: row.skip_reason === "axial" ? "off-axis — gated out" : (row.skip_reason || "gated out"), cue });
const naCard = (title, state, cue) => ({ title, verdict: "skip", headline: "Not scored",
  sub: state === "no_perpunch" ? "round-level rule" : state === "absent" ? "not produced on-device" : "no result for this punch", cue });

export const PER_PUNCH_RULES = [
  {
    id: "arm_extension", title: "Max extension",
    evaluate(analysis, punch) {
      const { state, row, cue } = rowFor(analysis, "arm_extension", punch);
      if (state !== "ok") return naCard("Max extension", state, cue);
      if (row.verdict === "skip") return skipCard("Max extension", row, cue);
      return { title: "Max extension", verdict: row.verdict === "pass" ? "ok" : "bad",
        headline: row.verdict === "pass" ? "Full extension" : "Short of full reach",
        sub: row.peak_bend != null ? `peak bend ${(+row.peak_bend).toFixed(2)}` : "", cue };
    },
  },
  {
    id: "hit_height", title: "Hit height",
    evaluate(analysis, punch) {
      const { state, row, cue } = rowFor(analysis, "hit_height", punch);
      if (state !== "ok") return naCard("Hit height", state, cue);
      if (row.skip_reason) return skipCard("Hit height", row, cue);
      const ok = !row.flag;                                  // hit_height uses flag, not verdict
      return { title: "Hit height", verdict: ok ? "ok" : "bad",
        headline: row.zone ? `${cap(row.zone)} level` : "Off target",
        sub: row.height_frac != null ? `height ${(+row.height_frac).toFixed(2)}` : "", cue };
    },
  },
  {
    id: "hand_return_path", title: "Return path", graph: true,
    evaluate(analysis, punch) {
      const { state, row, cue } = rowFor(analysis, "hand_return_path", punch);
      if (state !== "ok") return naCard("Return path", state, cue);
      if (row.verdict === "skip") return skipCard("Return path", row, cue);
      return { title: "Return path", verdict: row.verdict === "pass" ? "ok" : "bad",
        headline: row.re_guarded ? "Re-guarded cleanly" : "Did not return to guard",
        sub: row.return_sec != null ? `re-guarded in ${Math.round(row.return_sec * 1000)} ms` : "", cue };
    },
  },
];

export const SESSION_RULES = [
  {
    id: "stance_width", title: "Stance width",
    evaluate(analysis) {
      const r = analysis?.rules?.stance_width; if (!r) return null;
      const sep = pick(r.extras, "mean_sep_ratio", "meanSepRatio");
      return { title: "Stance width", verdict: sevToVerdict(r.severity),
        headline: `${Math.round((1 - r.violationRatio) * 100)}% in range`,
        sub: sep != null ? `mean sep ${(+sep).toFixed(2)}` : "", cue: r.coachCue };
    },
  },
  {
    id: "pivot_rate", title: "Footwork / rotation",
    evaluate(analysis) {
      const r = analysis?.rules?.pivot_rate; if (!r) return null;
      const pivots = pick(r.extras, "pivotCount", "pivot_count");
      const spp = pick(r.extras, "secPerPivot", "sec_per_pivot");
      return { title: "Footwork / rotation", verdict: sevToVerdict(r.severity),
        headline: pivots === 0 ? "No pivots detected" : spp != null ? `${(+spp).toFixed(1)}s / pivot` : "—",
        sub: pivots != null ? `${pivots} pivots this round` : "", cue: r.coachCue };
    },
  },
];

const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);
