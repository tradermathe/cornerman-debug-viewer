// Demo state + derived selectors. Plain object; app.js mutates `frame`/`selIdx`/etc
// and calls render(). Selectors are pure reads over the loaded session.

export function createState(session) {
  const { pose, analysis } = session;
  return {
    pose, analysis,
    fps: pose.fps, nFrames: pose.n_frames, width: pose.width, height: pose.height,
    stance: analysis?.ankleOrientation?.stance || "orthodox",
    fixture: !!session.fixture, videoUrl: session.videoUrl || null,
    // Seconds of video lead-in before pose data starts (pre-round framing/
    // countdown). Computed from video.duration once metadata loads; the
    // skeleton covers the TAIL of the video, so videoTime = poseTime + this.
    videoOffsetSec: 0,
    frame: 0, playing: false, selIdx: null,
    filters: { lead: true, rear: true, type: "all" },
  };
}

export const detections = (s) => s.analysis?.punches?.detections || [];
export const selectedPunch = (s) => (s.selIdx == null ? null : detections(s).find((d) => d.idx === s.selIdx) || null);

// COCO side (L/R) of the arm that throws this punch, from stance + lead/rear.
export function throwingSide(s, punch) {
  const leadSide = s.stance === "orthodox" ? "L" : "R";
  if (punch.hand === "lead") return leadSide;
  return leadSide === "L" ? "R" : "L";
}

export function roundSummary(s) {
  const ds = detections(s);
  const lead = ds.filter((d) => d.hand === "lead").length;
  const rear = ds.filter((d) => d.hand === "rear").length;
  const detailed = s.analysis?.punches?.breakdown_detailed || {};
  let head = 0, body = 0;
  for (const [k, v] of Object.entries(detailed)) (k.includes("body") ? (body += v) : (head += v));
  const tot = head + body || 1;
  const byType = { jab: 0, cross: 0, hook: 0, uppercut: 0 };
  for (const d of ds) {
    if (d.punch_type.includes("jab")) byType.jab++;
    else if (d.punch_type.includes("cross")) byType.cross++;
    else if (d.punch_type.includes("hook")) byType.hook++;
    else if (d.punch_type.includes("upper")) byType.uppercut++;
  }
  return {
    total: ds.length, lead, rear,
    headPct: Math.round((head / tot) * 100), bodyPct: Math.round((body / tot) * 100),
    byType,
    ppm: s.analysis?.punches?.punches_per_minute || 0,
  };
}

export function visible(s, punch) {
  if (punch.hand === "lead" && !s.filters.lead) return false;
  if (punch.hand === "rear" && !s.filters.rear) return false;
  const t = s.filters.type;
  if (t === "all") return true;
  if (t === "uppercut") return punch.punch_type.includes("upper");
  return punch.punch_type.includes(t);
}

export const timecode = (frame, fps) => {
  const sec = frame / fps;
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ff = Math.floor((sec - Math.floor(sec)) * 100);
  return `${m}:${String(s).padStart(2, "0")}.${String(ff).padStart(2, "0")}`;
};
