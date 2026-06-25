# Demo Dashboard — Data Contract

Bridge between the **design prototype** (`Cornerman AI Dashboard.dc.html` handoff) and the
**real on-device model output** parsed by `js/ondevice-loader.js`. For every field the design
renders, this says: is it **real**, **derivable**, or **must-cut/build** — so the demo only shows
numbers we can defend in front of an expert coach.

> **Guiding rule:** every number on screen for the featured session must trace to real output.
> The model speaks in **verdicts + metrics**, not invented 0–100 scores. Lead with the verdict
> ("Full extension") and back it with the real metric ("peak bend 0.97"). Cut anything we can't derive.

---

## What the loaders actually give us

`loadOnDeviceSkeleton(blob)` →
```
{ skeleton: Float32[n*17*2], conf: Float32[n*17], fps, width, height,
  n_frames, engine: "apple_vision_2d", normalised: false }
```
COCO-17 joints, **pixel** coords (top-left origin), per-joint confidence. Renders via `drawSkeleton(ctx, pose, frame, style)`.

`loadOnDeviceAnalysis(blob)` →
```
{ n_frames, fps,
  orientation:      { angles[], confidences[], validFrames },        // deprecated LogReg
  ankleOrientation: { stance, angles[], confidences[], validFrames } | null,  // trusted
  rules: {
    <ruleId>: { ruleId, version, severity, violationRatio, coachCue,
                validFrames, violationFrames, clips[], extras{},
                validMask:Uint8[], violationMask:Uint8[],
                sepRatios, sepRatiosCorrected, axisRatioSmoothed,   // stance_width
                skipReason, perPunch[]|null,                        // pivot_rate
                orientationAngles, orientationConfs }
  },
  punches: { detections[ { idx, timestamp, start_time, end_time,
                           start_frame, end_frame, hand, punch_type,
                           category, n_frames, axiality } ],
             total_punches, punches_per_minute, breakdown{}, breakdown_detailed{} } }
```

**On-device rules present** (the 5 Swift ports): `stance_width`, `arm_extension`, `pivot_rate`,
`hit_height`, `hand_return_path`. Everything else the design shows (elbow flare, guard height,
head-on-line, "angle thrown") is **not in the on-device sidecar**.

---

## 🚩 #1 open item — per-punch rule verdicts

The design's **Punch detail** panel shows a scored result *per punch* for Max extension / Hit height /
Return path. But the parsed sidecar only guarantees `perPunch` for **`pivot_rate`**; for
`arm_extension` / `hit_height` / `hand_return_path` the loader exposes **rule-level aggregates +
`extras`** (e.g. `straight_count`, `scored_count`, `fail_count`) — not a per-punch row with the
metric + verdict.

**Confirm against a real sidecar whether those rules write a `per_punch` array.** Two outcomes:

- **(A) They do** (or the iOS `RoundAnalyzer` `*_scores` arrays are in the sidecar) → extend
  `ondevice-loader.js` to surface them; the punch-detail panel works as designed.
- **(B) They don't** → either **(B1)** extend the iOS sidecar writer to emit per-punch verdicts
  (small, correct fix), or **(B2)** reframe the panel: show the **rule-level** verdict + the
  selected punch's window highlighted on the per-frame metric arrays (`validMask`/`violationMask`).

This is the single decision that most shapes the build. Until it's resolved, the punch-detail
panel is scaffolded against a fixture and flagged `TODO(per-punch)`.

---

## Panel-by-panel mapping

### Top bar
| Element | Source | Call |
|---|---|---|
| Round tabs (1/2/3) | one sidecar per round | **Real** — multi-round is post-MVP; ship Round 1 |
| Fighter name + initials | not in model | **Cosmetic** — set per featured session |
| Annotate / Export | n/a | **Defer** — depth-for-meeting, cut from first-touch video |

### Film panel — *fully grounded, this is the hero*
| Element | Source | Call |
|---|---|---|
| Video | the round `.mp4` | **Real** |
| Skeleton overlay | `drawSkeleton(ctx, pose, frame)` | **Real** — reuse renderer as-is |
| Active-limb highlight | throwing arm during a punch window (`hand` + `ankleOrientation.stance` → which side) | **Derivable** |
| Motion tracer | throwing wrist path across the punch window (`skeleton` keypoints) | **Derivable** |
| Per-joint confidence rings | `conf[joint] < 0.8` | **Real** |
| Frame badge / timecode | `frame`, `fps` | **Real** |

### Timeline — *fully grounded, strongest panel*
| Element | Source | Call |
|---|---|---|
| Markers positioned by frame | `detections[].start_frame / n_frames` | **Real** |
| Lead/Rear lanes | `detections[].hand` | **Real** |
| head=solid / body=outline | `punch_type` (`*_head` / `*_body`) | **Real** |
| Type filter (jab/cross/hook/upper) | `punch_type` prefix | **Real** |
| Playhead | `frame` | **Real** |

### Round summary
| Element | Source | Call |
|---|---|---|
| Total / lead / rear | `total_punches`, count by `hand` | **Real** |
| Body ratio (head/body %) | `breakdown_detailed` | **Real** |
| Punches by type | `breakdown` | **Real** |
| **➕ Form summary (recommended add)** | `stance_width.extras.mean_sep_ratio` + `severity`; `pivot_rate.extras.sec_per_pivot` | **Real & unused today** — for a coach this beats punch-counts |

### Punch detail — *centerpiece, gated on #1 above*
| Design field | Real source | Call |
|---|---|---|
| Max extension — **"% of reach"** | `arm_extension` per-punch `peak_bend` (relative arm length) | **Derivable** (relative) |
| Max extension — **"0.68 m"** | — pose is normalized 2D, **no metric calibration** | **CUT** |
| **0–100 scores** (97/85/79) | rules emit pass/fail/skip + metric, not 0–100 | **REFRAME → verdict + metric** |
| Hit height — "Head/Body level" | `punch_type` target + `hit_height` verdict | **Real** |
| **Angle thrown · "ideal 0–10°"** | no on-device source (closest: detection `axiality`, but that's camera-relative) | **CUT or build** |
| Return path — trajectory graph | throwing-wrist path over the window from `skeleton` | **Real & great** |
| Return path — "re-guarded in 266 ms" | derivable from frame count *iff* the rule exposes the return-complete frame | **Verify** |
| Mechanics @ impact (elbow/guard/head) | see per-frame mechanics below | **mostly CUT** |

### "This frame" mechanics
| Design field | Real source | Call |
|---|---|---|
| Feet angle | `ankleOrientation.angles[frame]` | **Real** |
| Tracking confidence bars | `conf[joint]` per frame | **Real** — keep, strong credibility |
| Elbow flare L/R | `elbow_tuck` lens exists but **not in on-device sidecar** | **CUT or port** |
| Guard lead/rear (High/Mid/Dropped) | `guard_height`/`guard_drop` lenses exist but **not on-device** | **CUT or port** |
| Head on line (Yes/No) | `head_offcenter` lens exists but **not on-device** | **CUT or build** |

---

## Net recommendation for the featured (first-touch) build

**Keep (all real):** film + skeleton overlay, timeline, round summary, tracking confidence,
feet angle, return-path trajectory graph, a new round-level **form summary** (stance + pivot).

**Reframe:** per-punch rule results as **verdict + real metric**, not 0–100 scores.

**Cut from the on-device demo:** meters, "angle thrown / ideal range", per-frame guard height,
elbow flare, head-on-line — unless we choose to port those rules on-device first.

**Resolve before wiring the centerpiece:** the per-punch verdict question (🚩 #1).

**Curate the hero session so:** (a) punch count visibly matches the film, (b) it contains a clear
PASS and a clear FAIL on the **coach's signature rule**, (c) the hero punch is a *straight, down-axis*
punch so the gated rules (`arm_extension`/`hit_height`/`return`) actually fire on it.
