// Shared quality-score + status-band helper for the on-device rule lenses.
//
// The lenses compute a 0–100 MISTAKE magnitude internally (0 = perfect, 100 =
// worst) — that's what the sigmoids/rollups produce. For DISPLAY we flip it to
// a QUALITY score (100 = perfect) and map it to a status band. Bands here must
// match cornerman-backend/rules_config.json `scoring.bands` (the single source
// of truth that the Python scorer + corpus run use).

// QUALITY bands: 90–100 perfect · 75–90 good · 50–75 bad · 0–50 critical.
export const QUALITY_BANDS = [
  [90, 101, "perfect"],
  [75, 90, "good"],
  [50, 75, "bad"],
  [0, 50, "critical"],
];

const BAND_COLOR = {
  perfect: "#5fd97a",
  good: "#bcd95f",
  bad: "#e8b45a",
  critical: "#e85a5a",
};

// mistake (0 = perfect) → quality (100 = perfect), 1 dp.
export const toQuality = (mistake) =>
  mistake == null ? null : Math.round((100 - mistake) * 10) / 10;

export function qualityBand(q) {
  for (const [lo, hi, label] of QUALITY_BANDS) if (q >= lo && q < hi) return label;
  return "critical";
}

export const qualityColor = (q) => BAND_COLOR[qualityBand(q)];

// One call: mistake magnitude → { q, label, color }. Pass the rolled-up round
// mistake; get back everything a lens needs to render the round score + label.
export function qualityOf(mistake) {
  if (mistake == null) return { q: null, label: "—", color: "var(--text-muted,#888)" };
  const q = toQuality(mistake);
  const label = qualityBand(q);
  return { q, label, color: BAND_COLOR[label] };
}

// Small inline badge: "84 good" colored by band. mistake in, HTML out.
export function qualityBadge(mistake) {
  const { q, label, color } = qualityOf(mistake);
  if (q == null) return `<span class="muted">—</span>`;
  return `<b style="color:${color}">${q.toFixed(1)}</b> `
    + `<span style="color:${color};text-transform:uppercase;font-size:11px;font-weight:700">${label}</span>`;
}
