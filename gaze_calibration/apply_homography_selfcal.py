#!/usr/bin/env python3
"""
apply_homography_selfcal.py — cursor-based homography self-calibration
(the pilot `self_calibrate` method, applied to the screen-video gaze).

Fits a projective HOMOGRAPHY mapping the mapped gaze onto the cursor path over
ALL in-trial samples (robust RANSAC):
        cursor_screenvid_px  ~  H · gaze_transf_px
and applies H to the calibrated gaze. A homography (8 params, global) absorbs the
RIM/screen projective distortion and any residual offset, so gaze that was grossly
mis-mapped (off by a large fraction of the screen) is pulled onto the tunnel.

Because it fits to the MOVING cursor it can slightly dampen the gaze lead (the
DV) — a global homography can't erase it (it survives as the residual), but use
this only for participants whose mapping is grossly off, where the accuracy gain
outweighs the small lead bias. Prints the gc improvement and the lead before/after.

USAGE
  python apply_homography_selfcal.py --aligned <P>_task_aligned.csv \
     --gaze_csv <P>_task_gaze_calibrated.csv --out <P>_task_gaze_calibrated.csv [--ransac_px 15]
Then re-align --out and re-run the analysis.
"""
import argparse, numpy as np, pandas as pd, cv2
TX="gaze position transf x [px]"; TY="gaze position transf y [px]"


def applyH(H, px):
    p = np.column_stack([px, np.ones(len(px))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned", required=True); ap.add_argument("--gaze_csv", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ransac_px", type=float, default=15.0)
    a = ap.parse_args()
    A = pd.read_csv(a.aligned)
    it = A["in_trial"].astype(str).isin(["True", "1", "1.0"])
    d = A[it].dropna(subset=["gaze_transf_x_px", "gaze_transf_y_px", "cursor_screenvid_x_px", "cursor_screenvid_y_px"])
    G = d[["gaze_transf_x_px", "gaze_transf_y_px"]].to_numpy(np.float32)
    C = d[["cursor_screenvid_x_px", "cursor_screenvid_y_px"]].to_numpy(np.float32)
    H, inl = cv2.findHomography(G, C, cv2.RANSAC, a.ransac_px)
    W = 3840.0
    def gc(px): p = applyH(H, px); return np.median(np.sqrt((p[:, 0] - d["cursor_screenvid_x_px"]) ** 2 + (p[:, 1] - d["cursor_screenvid_y_px"]) ** 2))
    old = np.median(np.sqrt((d["gaze_transf_x_px"] - d["cursor_screenvid_x_px"]) ** 2 + (d["gaze_transf_y_px"] - d["cursor_screenvid_y_px"]) ** 2))
    print(f"  homography on {len(d)} in-trial samples (inliers {int(inl.sum())}/{len(d)}); "
          f"gc {old:.0f}px ({old/W*100:.1f}%) -> {gc(G):.0f}px ({gc(G)/W*100:.1f}%)")
    g = pd.read_csv(a.gaze_csv)
    newp = applyH(H, g[[TX, TY]].to_numpy(float))
    g[TX], g[TY] = newp[:, 0], newp[:, 1]
    g.to_csv(a.out, index=False)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
