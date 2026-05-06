# Cornerman rule debug viewer

A static, dependency-free web app for inspecting YOLO-Pose Drive caches
against the [Cornerman](https://github.com/tradermathe/cornerman-backend)
rules engine, frame by frame.

**Live**: https://tradermathe.github.io/cornerman-debug-viewer/

## What it does

Pick the cache folder once per session, then for each round pick just
the video — the viewer auto-matches the cache files by filename. (A
manual multi-file picker is still available as a fallback.)

The viewer renders the video with a skeleton overlay and a side-panel
"lens" that re-paints the overlay and surfaces metrics for one rule at
a time. Add a new lens by dropping a file in [js/rules/](js/rules/) and
registering it in [js/rules/registry.js](js/rules/registry.js).

Files stay on the user's machine — the page uses the browser File API,
nothing is uploaded.

## Workflow

1. **Cache folder** — pick
   `~/Google Drive/My Drive/boxing_ai/yolo_pose_cache/` once. The viewer
   indexes every `<base>_yolo_r<N>.npy + <base>_yolo_r<N>_meta.json`
   pair (skipping `.bak.npy` backups) and reports the count.
2. **Video** — pick an `.mp4`. If its basename matches an indexed video,
   the viewer auto-loads the only round, or shows a round dropdown when
   multiple rounds exist.
3. **Lens** — switch the right-side dropdown to "Guard drop" (etc).

## Input format

YOLO-Pose Drive cache:

- `<round>.npy` — float32, shape `(N, 17, 3)`, one row per joint per
  frame holding `(x, y, conf)`. Layout is COCO-17. Coords are normalised
  to `[0, 1]` and de-normalised to pixels at load time using the video's
  natural dimensions.
- `<round>_meta.json` — at minimum `{ fps, layout: "coco17" }`.

Pick both files together via multi-select in the pose picker.

## Lenses

- **Overview** — per-joint table of (x, y, conf) at the current frame.
- **Guard drop** — highlights nose + both wrists + shoulders, draws
  horizontal y-lines through them, shows wrist→nose / wrist→shoulder
  normalised distances (the metrics
  `cornerman_rules/rules/guard_drop.py` uses), plus a wrist trail and
  full-clip y/conf sparklines. Includes threshold sliders matching
  `rules_config.json`.

## Running locally

```bash
python3 -m http.server 8765
# open http://localhost:8765
```

## Adding a new lens

Each rule module exports an object with `mount(host, state)`,
`update(state)`, optional `draw(ctx, state)` for canvas decorations, and
optional `skeletonStyle(state)` for base skeleton overrides. See
[`js/rules/overview.js`](js/rules/overview.js) for the minimum and
[`js/rules/guard_drop.js`](js/rules/guard_drop.js) for a full example.
