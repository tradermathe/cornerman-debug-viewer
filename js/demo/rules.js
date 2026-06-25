// Demo rule descriptors — the bridge from real `analysis.rules` to dashboard cards.
//
// Each descriptor is PURE and DOM-free: evaluate(analysis, punch|null) → a small
// presentation object. This is the "demo block" pattern: to feature a new rule,
// add a descriptor here. Per-punch rules read `rules[id].perPunch` (🚩#1 data-contract);
// if that array is absent on a real sidecar, they degrade to a clear "not scored" state
// instead of inventing a number.
//
// We deliberately surface VERDICT + REAL METRIC, never an invented 0–100 score.

const sevToVerdict = (sev) => (sev === "none" ? "ok" : sev === "severe" ? "bad" : "warn");

function perPunchRow(analysis, id, punch) {
  const r = analysis?.rules?.[id];
  if (!r) return { state: "absent" };
  if (!Array.isArray(r.perPunch)) return { state: "no_perpunch", cue: r.coachCue };
  const row = r.perPunch.find((p) => p.idx === punch.idx);
  if (!row) return { state: "no_row", cue: r.coachCue };
  return { state: "ok", row, cue: r.coachCue };
}

export const PER_PUNCH_RULES = [
  {
    id: "arm_extension", title: "Max extension",
    evaluate(analysis, punch) {
      const { state, row, cue } = perPunchRow(analysis, "arm_extension", punch);
      if (state !== "ok") return naCard("Max extension", state, cue);
      if (row.verdict === "skip")
        return { title: "Max extension", verdict: "skip", headline: "Not scored", sub: skipText(row), cue };
      return { title: "Max extension", verdict: row.verdict === "pass" ? "ok" : "bad",
        headline: row.verdict === "pass" ? "Full extension" : "Short of full reach",
        sub: `peak bend ${row.peak_bend?.toFixed?.(2) ?? row.peak_bend}` , cue };
    },
  },
  {
    id: "hit_height", title: "Hit height",
    evaluate(analysis, punch) {
      const { state, row, cue } = perPunchRow(analysis, "hit_height", punch);
      if (state !== "ok") return naCard("Hit height", state, cue);
      if (row.verdict === "skip")
        return { title: "Hit height", verdict: "skip", headline: "Not scored", sub: skipText(row), cue };
      return { title: "Hit height", verdict: row.verdict === "pass" ? "ok" : "bad",
        headline: `${cap(row.target)} level`, sub: `intended ${row.target} shot`, cue };
    },
  },
  {
    id: "hand_return_path", title: "Return path", graph: true,
    evaluate(analysis, punch) {
      const { state, row, cue } = perPunchRow(analysis, "hand_return_path", punch);
      if (state !== "ok") return naCard("Return path", state, cue);
      if (row.verdict === "skip")
        return { title: "Return path", verdict: "skip", headline: "Not scored", sub: skipText(row), cue };
      return { title: "Return path", verdict: row.verdict === "pass" ? "ok" : "bad",
        headline: row.verdict === "pass" ? "Re-guarded cleanly" : "Slow back to guard",
        sub: row.guard_ms != null ? `re-guarded in ${row.guard_ms} ms` : "", cue };
    },
  },
];

export const SESSION_RULES = [
  {
    id: "stance_width", title: "Stance width",
    evaluate(analysis) {
      const r = analysis?.rules?.stance_width; if (!r) return null;
      const pct = Math.round((1 - r.violationRatio) * 100);
      return { title: "Stance width", verdict: sevToVerdict(r.severity),
        headline: `${pct}% in range`,
        sub: r.extras?.mean_sep_ratio != null ? `mean sep ${r.extras.mean_sep_ratio.toFixed(2)}` : "",
        cue: r.coachCue };
    },
  },
  {
    id: "pivot_rate", title: "Rotation",
    evaluate(analysis) {
      const r = analysis?.rules?.pivot_rate; if (!r) return null;
      return { title: "Rotation", verdict: sevToVerdict(r.severity),
        headline: r.extras?.sec_per_pivot != null ? `${r.extras.sec_per_pivot.toFixed(1)}s / pivot` : "—",
        sub: r.extras?.pivot_count != null ? `${r.extras.pivot_count} pivots` : "", cue: r.coachCue };
    },
  },
];

function naCard(title, state, cue) {
  const sub = state === "no_perpunch" ? "rule is round-level (no per-punch row)"
    : state === "absent" ? "not produced on-device" : "no result for this punch";
  return { title, verdict: "skip", headline: "Not scored", sub, cue };
}
const skipText = (row) => row.skip_reason === "axial_gate" ? "off-axis — gated out"
  : row.skip_reason === "not_straight" ? "not a straight punch" : "gated out";
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);
