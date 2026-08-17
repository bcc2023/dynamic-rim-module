#!/usr/bin/env bash
# Dynamic RIM mapping for the "calibrated version" set: Beichen + Astrid,
# each with a calibration recording and an actual-task recording.
# Sync = --screen_start_wallclock (your noted screen-recording start times).
# Outputs (gaze CSV + 3-panel overlay video) land in a per-run out/ subfolder.
#
# Run in the pupil-cloud-local env:  bash "run_calibrated_dynamic_rim.sh"
set -euo pipefail

BE="/Users/xuebeichen/Desktop/official pilot/calibrated version/Beichen"
AS="/Users/xuebeichen/Desktop/official pilot/calibrated version/Astrid"
BENR="$BE/cursor_gaze_calibration_pilot_Bc_REFERENCE-IMAGE-MAPPER_Calibrated_bc_csv"
AENR="$AS/cursor_gaze_calibration_Astrid_REFERENCE-IMAGE-MAPPER_calibrated_bc_csv"

run () {   # out_dir  rim  raw  screen  wallclock
  local out="$1"
  mkdir -p "$out"
  echo "======== $out ========"
  # --out_csv_path / --out_video_path MUST be set, otherwise the tool pops a GUI
  # save dialog and dumps to your home folder with a random name.
  caffeinate -dimsu pl-dynamic-rim \
      --rim_folder_path   "$2" \
      --raw_folder_path   "$3" \
      --screen_video_path "$4" \
      --audio No_Audio \
      --screen_start_wallclock "$5" \
      --out_csv_path   "$out/mapped_gaze.csv" \
      --out_video_path "$out/mapped_overlay.mp4"
}

# ---- Beichen: calibration (short, has the 9-point + clock) ----
run "$BE/out_calibration" "$BENR" \
    "$BE/Timeseries Data + Scene Video-2/2026-08-14_11-07-26-b0d79dbd" \
    "$BE/calibration/Beichen Calibration.mov" \
    "2026-08-14 11:07:26.834"

# ---- Beichen: actual task (long ~11 min) ----
run "$BE/out_task" "$BENR" \
    "$BE/Timeseries Data + Scene Video-2/2026-08-14_17-06-30-e3d239d3" \
    "$BE/Beichen_task_screen.mov" \
    "2026-08-14 17:06:30.937"

# ---- Astrid: calibration ----
run "$AS/out_calibration" "$AENR" \
    "$AS/Timeseries Data + Scene Video-2/2026-08-14_11-42-39-463f6ba4" \
    "$AS/calibration/Calibration.mov" \
    "2026-08-14 11:42:39.551"

# ---- Astrid: actual task (long ~14 min) ----
run "$AS/out_task" "$AENR" \
    "$AS/Timeseries Data + Scene Video-2/2026-08-14_11-43-30-18ca845a" \
    "$AS/Astrid_task_screen.mov" \
    "2026-08-14 11:43:30.543"

echo "done."
