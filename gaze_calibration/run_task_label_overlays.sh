#!/usr/bin/env bash
# Build brightened screen+gaze overlay videos for BOTH task-label recordings.
# Run on your Mac (env with av/pandas/pillow, e.g. `conda activate pupil-cloud-local`):
#     bash "run_task_label_overlays.sh"
set -euo pipefail

DIR="/Users/xuebeichen/Desktop/task label"
SCRIPT="/Users/xuebeichen/Downloads/Github/pupil cloud local/screen_gaze_overlay.py"
WHITE=148
WIDTH=1920

# name | screen video | gaze csv | start_video_ns (Neon begin + offset) | extra args
run () {
  echo "===== $1 ====="
  python3 "$SCRIPT" \
    --screen_video   "$DIR/$2" \
    --gaze_csv       "$DIR/$3" \
    --start_video_ns "$4" \
    --out_video      "$DIR/$1_screen_bright.mp4" \
    --white $WHITE --out_width $WIDTH $5
}

run "motivation" "motivation&learning.mov" "gaze1.csv" 1786578506000000000 ""
run "awareness"  "awareness.mov"           "gaze2.csv" 1786578955000000000 "--duration_s 46"

echo "done -> $DIR/motivation_screen_bright.mp4 , $DIR/awareness_screen_bright.mp4"
