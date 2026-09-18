#!/usr/bin/env python3
"""
apply_vertical_recal.py — correct a residual VERTICAL gaze bias that appears only
in the task (not in the calibration dots), i.e. a small vertical shift of the
Neon between a separate calibration recording and the task recording.

The cursor is verified-correct (it sits in the tunnel / on the target because the
participant steered it there), and for these mostly-horizontal tunnels the gaze's
VERTICAL position should track the tunnel height = cursor height (the gaze *lead*
lives in the horizontal, which we leave untouched). So we fit, robustly,

        cursor_y_norm  ~  a * gaze_y_norm + b          (VERTICAL only)

from in-trial fixation samples (decile medians -> line, robust to corner-cutting),
and rewrite the gaze CSV's 'gaze position transf y [px]' as a*y + b*H. Horizontal
'gaze position transf x [px]' is NOT modified, so the gaze-lead signal is preserved.

USAGE
    python apply_vertical_recal.py \
        --aligned  Yanran/Yanran_task_aligned.csv \
        --gaze_csv Yanran/Yanran_task_gaze_calibrated.csv \
        --out      Yanran/Yanran_task_gaze_calibrated.csv \
        [--screen_h 2160] [--min_per_bin 200]
Then re-run `align` on --out to refresh the aligned CSV, and re-render the video.
"""
import argparse
import numpy as np
import pandas as pd

TY = "gaze position transf y [px]"


def istrial(s):
    return s.astype(str).isin(["True", "1", "1.0"])


def robust_line(gx, cy, nbin=10, min_per_bin=50):
    """Fit cy ~ a*gx + b through per-decile medians (robust to outliers)."""
    q = np.quantile(gx, np.linspace(0, 1, nbin + 1))
    xs, ys = [], []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (gx >= lo) & (gx <= hi)
        if m.sum() >= min_per_bin:
            xs.append(np.median(gx[m]))
            ys.append(np.median(cy[m]))
    xs, ys = np.array(xs), np.array(ys)
    a, b = np.polyfit(xs, ys, 1)
    return a, b, len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned", required=True, help="aligned CSV (source of gaze_y & cursor_y)")
    ap.add_argument("--gaze_csv", required=True, help="calibrated gaze CSV to correct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--screen_h", type=float, default=2160.0)
    ap.add_argument("--min_per_bin", type=int, default=200)
    a = ap.parse_args()

    al = pd.read_csv(a.aligned, usecols=["gaze_transf_y_norm", "cursor_screenvid_y_norm",
                                         "in_trial", "fixation_id"])
    m = istrial(al["in_trial"]) & al["fixation_id"].notna() \
        & al["cursor_screenvid_y_norm"].notna() & al["gaze_transf_y_norm"].notna()
    d = al[m]
    gy = d["gaze_transf_y_norm"].to_numpy()
    cy = d["cursor_screenvid_y_norm"].to_numpy()
    coef_a, coef_b, nb = robust_line(gy, cy, min_per_bin=a.min_per_bin)

    # before/after residual by third
    def bins(dy):
        out = []
        for lo, hi in [(0, .33), (.33, .66), (.66, 1)]:
            mm = (gy >= lo) & (gy < hi)
            out.append(np.median(dy[mm]) if mm.any() else np.nan)
        return out
    dy0 = gy - cy
    dy1 = (coef_a * gy + coef_b) - cy
    print(f"vertical recal fit (on {len(d)} in-trial fixation samples, {nb} decile knots):")
    print(f"   gaze_y_norm -> {coef_a:.4f} * gaze_y_norm + {coef_b:+.4f}")
    print(f"   dy(gaze-cursor) top/mid/bot  BEFORE: {['%+.4f'%v for v in bins(dy0)]}  median {np.median(dy0):+.4f}")
    print(f"   dy(gaze-cursor) top/mid/bot  AFTER : {['%+.4f'%v for v in bins(dy1)]}  median {np.median(dy1):+.4f}")

    # apply to the gaze CSV (vertical only): y_px -> a*y_px + b*H
    g = pd.read_csv(a.gaze_csv)
    g[TY] = coef_a * g[TY].astype(float) + coef_b * a.screen_h
    g.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(g)} rows; only '{TY}' changed)")


if __name__ == "__main__":
    main()
