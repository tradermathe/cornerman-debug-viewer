# Cornerman rule debug viewer

A static, dependency-free web app for inspecting Apple Vision pose data
against the [Cornerman](https://github.com/tradermathe/cornerman-backend)
rules engine, frame by frame.

**Live**: https://tradermathe.github.io/cornerman-debug-viewer/

## What it does

Pick a video (`.mp4`) and the matching pose JSON. The viewer renders the
video with a skeleton overlay and a side-panel "lens" that re-paints the
overlay and surfaces metrics for one rule at a time. Add a new lens by
dropping a file in [js/rules/](js/rules/) and registering it in
[js/rules/registry.js](js/rules/registry.js).

Files stay on the user's machine — the page uses the browser File API,
nothing is uploaded.

## Supported pose JSON formats

- **Parity / Mac-side** — straight-array `skeleton_frames` /
  `confidences`, written by the `apple_vision_pose` CLI.
- **Production iOS** — base64 float32 LE in `skeleton_b64` / `conf_b64`,
  written by the `cornerman-vision-pose` Expo module.

Joints are COCO-17 in pixel coords for both.

## Lenses

- **Overview** — per-joint table of (x, y, conf) at the current frame.
  Useful for spot-checking Vision's confidence behaviour.
- **Guard drop** — highlights nose + both wrists + shoulders, draws
  horizontal y-lines through them, shows wrist→nose / wrist→shoulder
  normalised distances (the metrics `cornerman_rules/rules/guard_drop.py`
  uses in the backend), plus a wrist trail and full-clip y/conf
  sparklines. Includes threshold sliders matching `rules_config.json`.

## Running locally

```bash
python3 -m http.server 8765
# open http://localhost:8765
```

## Adding a new lens

The rule-panel contract is small: each module exports an object with
`mount(host, state)`, `update(state)`, optional `draw(ctx, state)` for
canvas decorations, and optional `skeletonStyle(state)` for base
skeleton overrides. See [`js/rules/overview.js`](js/rules/overview.js)
for the minimum, and [`js/rules/guard_drop.js`](js/rules/guard_drop.js)
for a full-fat example.
