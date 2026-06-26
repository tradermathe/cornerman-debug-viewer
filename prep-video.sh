#!/usr/bin/env bash
# Prepare a session's round video for the demo: trim the pre-round lead-in
# (framing/countdown) so the video matches the pose timeline 1:1, force CFR 30,
# scale to 720x1280, and compress. Produces round_<N>_web.mp4 next to the source.
#
# Why: on-device pose extraction starts at round-start, but the recorded video
# includes ~several seconds of pre-roll. Without trimming, the skeleton overlay
# is offset from the footage (and needs a runtime seek to re-sync on load).
# Trimming makes frame 0 == round start, so the overlay is synced from first paint.
#
# Usage: ./prep-video.sh demo-assets/session_<id> [roundNumber]
set -euo pipefail
DIR="${1:?usage: prep-video.sh <session-dir> [round]}"
N="${2:-1}"
SRC="$DIR/round_${N}.mp4"
SKEL="$DIR/round_${N}_skeleton.json"
OUT="$DIR/round_${N}_web.mp4"
[ -f "$SRC" ]  || { echo "missing $SRC"; exit 1; }
[ -f "$SKEL" ] || { echo "missing $SKEL"; exit 1; }

read FPS NFRAMES < <(python3 -c "import json;d=json.load(open('$SKEL'));print(d['fps'], d['n_frames'])")
VFRAMES=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nk=1:nw=1 "$SRC")
PREROLL=$(( VFRAMES - NFRAMES ))
OFFSET=$(python3 -c "print(max(0,$PREROLL)/$FPS)")
echo "pose=${NFRAMES}f @ ${FPS}fps  video=${VFRAMES}f  pre-roll=${PREROLL}f (${OFFSET}s)"

ffmpeg -y -v error -ss "$OFFSET" -i "$SRC" -frames:v "$NFRAMES" -r "$FPS" \
  -vf scale=720:1280 -c:v libx264 -crf 28 -preset fast -movflags +faststart -an "$OUT"
echo "wrote $OUT ($(python3 -c "import os;print(f'{os.path.getsize(\"$OUT\")/1e6:.1f}MB')")) — frame 0 == round start"
