#!/usr/bin/env bash
# Generate ALL visualizations for every pilot session in one shot.
#
#   bash "run_all_visualizations.sh"
#
# For each session it produces, in that session's folder:
#   <NAME>_analysis.csv                     (aligned CSV + gaze-cursor distance)
#   <NAME>_trials.pdf                       (per-trial signed distance vs time)
#   <NAME>_aligned_difficulty.csv           (A/W + Fitts ID per segment)
#   <NAME>_aligned_constrained_difficulty.png
#   <NAME>_aligned_unconstrained_difficulty.png
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/analyze_aligned.py"
PILOT="/Users/xuebeichen/Desktop/official pilot"

# add/remove session names here
SESSIONS=(P152244 P161909)

for NAME in "${SESSIONS[@]}"; do
  P="$PILOT/$NAME"
  ALIGNED="$P/${NAME}_aligned.csv"
  JSON=$(ls "$P"/steering_experiment_*.json 2>/dev/null | head -1 || true)
  if [[ ! -f "$ALIGNED" ]]; then
    echo "!! skip $NAME: no ${NAME}_aligned.csv"; continue
  fi
  if [[ -z "${JSON:-}" ]]; then
    echo "!! skip $NAME: no steering_experiment_*.json"; continue
  fi
  echo "===== $NAME ====="
  python3 "$SCRIPT" \
    --aligned_csv "$ALIGNED" \
    --experiment_json "$JSON" \
    --out_csv "$P/${NAME}_analysis.csv" \
    --out_pdf "$P/${NAME}_trials.pdf"
done
echo "done."
