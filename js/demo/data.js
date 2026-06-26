// Demo data layer.
//
//   loadRealSession({ skeletonBlob, analysisBlob })  → uses the SAME parsers the
//     debug viewer uses (ondevice-loader.js), so dropping in a real session "just works".
//   syntheticSession()  → a schema-faithful fixture so the dashboard renders end-to-end
//     before a real session exists. Everything here mirrors the loader OUTPUT shape
//     (post-base64-decode), not the raw on-wire JSON.
//
// IMPORTANT: per-punch rule rows (peak_bend/verdict for arm_extension/hit_height/
// hand_return_path) are SYNTHESIZED here. On real sidecars they may be absent — see
// docs/demo-data-contract.md 🚩#1. The rule descriptors degrade gracefully when missing.

import { loadOnDeviceSkeleton, loadOnDeviceAnalysis } from "../ondevice-loader.js";

export async function loadRealSession({ skeletonBlob, analysisBlob }) {
  const pose = await loadOnDeviceSkeleton(skeletonBlob);
  const analysis = analysisBlob ? await loadOnDeviceAnalysis(analysisBlob) : null;
  return { pose, analysis };
}

// Load a session whose round files sit on the static server (downloaded from
// Firebase Storage into demo-assets/). Reuses the real parsers, so this is the
// exact shape a Firebase-loaded round would produce.
export async function loadLocalSession(base, round = 1) {
  const fetchBlob = async (suffix) => {
    const r = await fetch(`${base}/round_${round}${suffix}`);
    if (!r.ok) throw new Error(`fetch ${suffix} -> ${r.status}`);
    return r.blob();
  };
  const pose = await loadOnDeviceSkeleton(await fetchBlob("_skeleton.json"));
  const analysis = await loadOnDeviceAnalysis(await fetchBlob("_ondevice_analysis.json"));
  augmentDetections(analysis, pose.fps);
  return { pose, analysis, videoUrl: `${base}/round_${round}.mp4`, fixture: false };
}

// Real detections carry start/end time but no impact frame; the timeline +
// active-limb logic want one. Derive it from the punch timestamp.
function augmentDetections(analysis, fps) {
  for (const d of analysis?.punches?.detections || []) {
    if (d.impact_frame == null) d.impact_frame = Math.round((d.timestamp ?? d.start_time) * fps);
  }
}

// ---- Synthetic fixture ------------------------------------------------------

const J = { NOSE:0, L_EYE:1, R_EYE:2, L_EAR:3, R_EAR:4, L_SHOULDER:5, R_SHOULDER:6,
  L_ELBOW:7, R_ELBOW:8, L_WRIST:9, R_WRIST:10, L_HIP:11, R_HIP:12, L_KNEE:13, R_KNEE:14, L_ANKLE:15, R_ANKLE:16 };

// A deterministic pseudo-random so the fixture is stable across reloads.
function rng(seed) { let s = seed >>> 0; return () => (s = (s*1664525 + 1013904223) >>> 0) / 4294967296; }

const PUNCH_DEFS = [
  // frameAt is set below by spacing; type/hand drive the timeline + which arm extends.
  { hand:"lead", punch_type:"jab_head",        category:"jab",   axiality:0.74 },
  { hand:"rear", punch_type:"cross_head",      category:"cross", axiality:0.71 },
  { hand:"lead", punch_type:"lead_hook_head",  category:"power", axiality:0.28 },
  { hand:"lead", punch_type:"jab_body",        category:"jab",   axiality:0.66 },
  { hand:"rear", punch_type:"cross_body",      category:"cross", axiality:0.69 },
  { hand:"rear", punch_type:"rear_uppercut_head", category:"power", axiality:0.41 },
  { hand:"lead", punch_type:"jab_head",        category:"jab",   axiality:0.80 },
  { hand:"rear", punch_type:"cross_head",      category:"cross", axiality:0.52 },
  { hand:"lead", punch_type:"lead_hook_body",  category:"power", axiality:0.24 },
  { hand:"rear", punch_type:"cross_head",      category:"cross", axiality:0.73 },
  { hand:"lead", punch_type:"jab_head",        category:"jab",   axiality:0.77 },
  { hand:"rear", punch_type:"rear_uppercut_body", category:"power", axiality:0.39 },
];

export function syntheticSession() {
  const fps = 30, W = 1080, H = 1920;
  const stance = "orthodox";                 // lead = LEFT side joints
  const rand = rng(42);

  // Lay punches out across ~30s with a little jitter.
  const nFrames = 920;
  const punchWin = 16;                        // frames per punch (extend+return)
  const spacing = Math.floor((nFrames - 120) / PUNCH_DEFS.length);
  const detections = PUNCH_DEFS.map((d, i) => {
    const impact = 60 + i * spacing + Math.floor(rand() * 10);
    const start = impact - Math.floor(punchWin / 2);
    const end = impact + Math.floor(punchWin / 2);
    return {
      idx: i, hand: d.hand, punch_type: d.punch_type, category: d.category,
      axiality: d.axiality, n_frames: punchWin,
      start_frame: start, end_frame: end, impact_frame: impact,
      timestamp: +(impact / fps).toFixed(2),
      start_time: +(start / fps).toFixed(2), end_time: +(end / fps).toFixed(2),
    };
  });

  const pose = buildPose(nFrames, fps, W, H, detections, stance, rand);
  const analysis = buildAnalysis(nFrames, fps, detections, stance);
  return { pose, analysis, fixture: true };
}

// Build a standing COCO-17 figure with idle sway + per-punch arm extension.
function buildPose(n, fps, W, H, dets, stance, rand) {
  const skeleton = new Float32Array(n * 17 * 2);
  const conf = new Float32Array(n * 17);
  const cx = W * 0.5;
  // Vertical anchors (portrait frame).
  const yShoulder = H * 0.30, yElbow = H * 0.40, yWrist = H * 0.48,
        yHip = H * 0.55, yKnee = H * 0.72, yAnkle = H * 0.90, yHead = H * 0.22;
  const sx = W * 0.10;                         // half shoulder width
  const set = (f, j, x, y, c=0.97) => {
    skeleton[(f*17 + j)*2] = x; skeleton[(f*17 + j)*2 + 1] = y; conf[f*17 + j] = c;
  };
  const leadSide = stance === "orthodox" ? "L" : "R";

  for (let f = 0; f < n; f++) {
    const sway = Math.sin(f / 22) * W * 0.012;  // gentle idle bob
    const bob = Math.cos(f / 18) * H * 0.004;
    const X = cx + sway;
    set(f, J.NOSE, X, yHead + bob);
    set(f, J.L_EYE, X - 14, yHead - 8); set(f, J.R_EYE, X + 14, yHead - 8);
    set(f, J.L_EAR, X - 30, yHead - 2); set(f, J.R_EAR, X + 30, yHead - 2);
    set(f, J.L_SHOULDER, X - sx, yShoulder + bob); set(f, J.R_SHOULDER, X + sx, yShoulder + bob);
    set(f, J.L_HIP, X - sx*0.7, yHip); set(f, J.R_HIP, X + sx*0.7, yHip);
    set(f, J.L_KNEE, X - sx*0.7, yKnee); set(f, J.R_KNEE, X + sx*0.7, yKnee);
    set(f, J.L_ANKLE, X - sx*0.8, yAnkle); set(f, J.R_ANKLE, X + sx*0.8, yAnkle);
    // Guard position (hands up near chin) by default.
    set(f, J.L_ELBOW, X - sx*0.8, yElbow); set(f, J.R_ELBOW, X + sx*0.8, yElbow);
    set(f, J.L_WRIST, X - sx*0.45, yShoulder + H*0.04); set(f, J.R_WRIST, X + sx*0.45, yShoulder + H*0.04);
  }

  // Overlay each punch: extend the throwing arm toward the camera/centre at impact.
  for (const d of dets) {
    const side = d.hand === "lead" ? leadSide : (leadSide === "L" ? "R" : "L");
    const wristJ = side === "L" ? J.L_WRIST : J.R_WRIST;
    const elbowJ = side === "L" ? J.L_ELBOW : J.R_ELBOW;
    const shJ = side === "L" ? J.L_SHOULDER : J.R_SHOULDER;
    for (let f = d.start_frame; f <= d.end_frame; f++) {
      if (f < 0 || f >= n) continue;
      const t = 1 - Math.abs(f - d.impact_frame) / (d.n_frames / 2); // 0..1 extension
      const ext = Math.max(0, t);
      const shx = skeleton[(f*17 + shJ)*2], shy = skeleton[(f*17 + shJ)*2 + 1];
      // Extend forward + slightly up for head shots, down for body.
      const dir = side === "L" ? -1 : 1;
      const reach = ext * W * 0.20;
      const ty = d.punch_type.includes("body") ? yHip - H*0.02 : yShoulder - H*0.01;
      skeleton[(f*17 + wristJ)*2]     = shx + dir*reach*0.2 + reach*0.0;
      skeleton[(f*17 + wristJ)*2 + 1] = (skeleton[(f*17 + wristJ)*2 + 1])*(1-ext) + ty*ext;
      skeleton[(f*17 + elbowJ)*2]     = shx + dir*reach*0.12;
      skeleton[(f*17 + elbowJ)*2 + 1] = (shy + ty)/2;
      // Motion blur dips wrist/elbow confidence at peak extension.
      conf[f*17 + wristJ] = 0.97 - ext*0.22;
      conf[f*17 + elbowJ] = 0.97 - ext*0.12;
    }
  }

  return { skeleton, conf, fps, width: W, height: H, n_frames: n,
    engine: "apple_vision_2d", source: "synthetic", normalised: false,
    start_sec: 0, pre_buffer_sec: 0 };
}

function buildAnalysis(n, fps, dets, stance) {
  const validMask = new Uint8Array(n).fill(1);
  const violationMask = new Uint8Array(n);
  for (let f = 0; f < n; f++) if ((f % 140) < 18) violationMask[f] = 1;   // sporadic narrow stance

  // Per-punch synthetic verdicts. Gated rules only score straight, down-axis punches.
  const isStraight = (d) => /jab|cross/.test(d.punch_type);
  const gated = (d) => isStraight(d) && d.axiality >= 0.5;
  const armPerPunch = dets.map(d => {
    if (!gated(d)) return { idx: d.idx, skip_reason: isStraight(d) ? "axial_gate" : "not_straight", verdict: "skip" };
    const peak = 0.86 + (d.axiality - 0.5) * 0.3 - (d.idx === 7 ? 0.18 : 0);  // punch #7 is short
    return { idx: d.idx, peak_bend: +peak.toFixed(2), verdict: peak >= 0.92 ? "pass" : "fail" };
  });
  const hitPerPunch = dets.map(d => {
    if (!gated(d)) return { idx: d.idx, verdict: "skip", skip_reason: "axial_gate" };
    const target = d.punch_type.includes("body") ? "body" : "head";
    const ok = !(d.idx === 4);                                              // punch #4 a touch low
    return { idx: d.idx, target, verdict: ok ? "pass" : "fail" };
  });
  const returnPerPunch = dets.map(d => {
    if (!gated(d)) return { idx: d.idx, verdict: "skip", skip_reason: "axial_gate" };
    const guardMs = 180 + (d.idx % 4) * 60 + (d.idx === 1 ? 220 : 0);       // punch #1 slow to re-guard
    return { idx: d.idx, guard_ms: guardMs, verdict: guardMs <= 320 ? "pass" : "fail" };
  });

  const rule = (id, sev, vr, cue, extras, extra={}) => ({
    ruleId: id, version: "demo", severity: sev, violationRatio: vr, coachCue: cue,
    validFrames: n, violationFrames: Math.round(vr*n), clips: [], extras,
    validMask: null, violationMask: null, sepRatios: null, sepRatiosCorrected: null,
    axisRatioSmoothed: null, skipReason: null, perPunch: null,
    orientationAngles: null, orientationConfs: null, ...extra });

  return {
    n_frames: n, fps,
    ankleOrientation: { version:"v1", stance, validFrames: n,
      angles: Float32Array.from({length:n}, (_,f)=> 24 + Math.sin(f/30)*4),
      confidences: new Float32Array(n).fill(0.9) },
    rules: {
      stance_width: rule("stance_width", "mild", 0.13,
        "Keep your feet about shoulder-width — a touch narrow at times.",
        { mean_sep_ratio: 0.83, median_sep_ratio: 0.81 },
        { validMask, violationMask }),
      pivot_rate: rule("pivot_rate", "none", 0.04, "Good rotation through your shots.",
        { sec_per_pivot: 3.1, pivot_count: 9 },
        { perPunch: dets.filter(d=>d.category!=="jab").map(d => ({
            idx: d.idx, punch_type: d.punch_type, ratchet: false, verdict: "pass" })) }),
      arm_extension: rule("arm_extension", "mild", 0.18,
        "Snap your straight shots all the way out.",
        { punch_count: dets.length, scored_count: armPerPunch.filter(p=>p.verdict!=="skip").length,
          fail_count: armPerPunch.filter(p=>p.verdict==="fail").length },
        { perPunch: armPerPunch }),
      hit_height: rule("hit_height", "none", 0.08, "Good shot placement.",
        { scored_count: hitPerPunch.filter(p=>p.verdict!=="skip").length }, { perPunch: hitPerPunch }),
      hand_return_path: rule("hand_return_path", "mild", 0.16,
        "Bring the hand straight back to your chin.",
        { scored_count: returnPerPunch.filter(p=>p.verdict!=="skip").length }, { perPunch: returnPerPunch }),
    },
    punches: {
      source: "synthetic", fps, detections: dets,
      total_punches: dets.length,
      punches_per_minute: Math.round(dets.length / (n/fps) * 60),
      breakdown: countBy(dets, d => d.category),
      breakdown_detailed: countBy(dets, d => d.punch_type),
    },
  };
}

function countBy(arr, fn) { const o = {}; for (const x of arr) { const k = fn(x); o[k] = (o[k]||0)+1; } return o; }
