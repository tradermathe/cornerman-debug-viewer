#!/usr/bin/env bash
# Assemble dist/ = the partner demo ONLY (not the debug viewer), ready for
# `firebase deploy`. Pure copy, no build tooling needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DIST="$ROOT/dist"
SESSION="session_1781788984153"
A="demo-assets/$SESSION"

rm -rf "$DIST"
mkdir -p "$DIST/js/demo" "$DIST/$A"

cp "$ROOT/demo.html"               "$DIST/index.html"          # demo is the site root
cp "$ROOT"/js/demo/*               "$DIST/js/demo/"
cp "$ROOT/js/skeleton.js"          "$DIST/js/skeleton.js"      # reused renderer
cp "$ROOT/js/ondevice-loader.js"   "$DIST/js/ondevice-loader.js"

# Featured session: web-compressed video + skeleton + analysis sidecar.
cp "$ROOT/$A/round_1_skeleton.json"          "$DIST/$A/"
cp "$ROOT/$A/round_1_ondevice_analysis.json" "$DIST/$A/"
cp "$ROOT/$A/round_1_web.mp4"                "$DIST/$A/"

echo "Built $DIST ($(du -sh "$DIST" | cut -f1)) — demo only, no debug viewer."
