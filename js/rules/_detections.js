// Shared punch-source pick for the geometric lenses (hit_height, arm_extension,
// hand_return_path).
//
// These lenses were written for the labelled-data eval workflow, so they read
// punches from the hand-labelled GT (`state.labels.detections`). On an app-
// recorded on-device round there are no labels yet — the punches live in
// `state.punches.detections` (the ST-GCN classifier's predictions, the same
// ones the on-device rules actually run on). Fall back to those so the lens
// renders on unlabelled rounds too. Mirrors angle_change.js's source pick.
//
// Returns a stable array reference (or null) so callers can use identity
// (`dets !== lastDetections`) for change detection without re-rendering every
// frame.
export function activeDetections(state) {
  return state.labels?.detections?.length
    ? state.labels.detections
    : (state.punches?.detections || null);
}

// Straight (jab / cross) test that matches BOTH the labeler's suffixed types
// ("jab_head", "cross_body") and the on-device classifier's bare types
// ("jab", "cross"). Use instead of an exact `appliesTo.has(punch_type)` set,
// which only knows the suffixed form.
export function isStraightType(punchType) {
  return /(^|_)(jab|cross)(_|$)/i.test(punchType || "");
}
