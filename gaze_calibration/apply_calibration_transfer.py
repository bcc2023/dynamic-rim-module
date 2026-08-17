#!/usr/bin/env python3
"""
Transfer the manual gaze-offset calibration (done on the CALIBRATION recording)
onto the TASK recording, in reference-image space, as a homography.

Why a homography: on the calibration recording we have the *same* gaze BEFORE and
AFTER the Cloud gaze-offset calibration, both mapped onto the reference image.
Each timestamp gives a pair (pre_ref -> post_ref). A constant scene-camera offset
seen through the flat screen is a projective map, so we fit a homography T from
those pairs. Cloud did NOT recalibrate the task recording, so we apply T to the
task's reference-image gaze, then re-project to the screen through the task's own
reference->screen homography (recovered from the task mapping's ref<->transf
columns). Output = corrected task gaze CSV (corrected reference + transf columns),
ready for compose_3panel.py / analyze_aligned.py.

USAGE
    python3 apply_calibration_transfer.py \
        --cal_pre  ".../Beichen_calibration_gaze.csv" \
        --cal_post ".../after .../gaze.csv" --cal_post_recording b0d79dbd \
        --task     ".../Beichen_task_gaze.csv" \
        --out      ".../Beichen_task_gaze_calibrated.csv"
"""
import argparse
import numpy as np
import pandas as pd

RX = "gaze position in reference image x [px]"
RY = "gaze position in reference image y [px]"
TX = "gaze position transf x [px]"
TY = "gaze position transf y [px]"
DET = "gaze detected in reference image"
TS = "timestamp [ns]"
REC = "recording id"


def _truthy(s):
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def fit_homography(src, dst):
    """Normalized-DLT homography src(N,2) -> dst(N,2)."""
    def norm(P):
        m = P.mean(0)
        s = np.sqrt(2) / (np.sqrt(((P - m) ** 2).sum(1)).mean() + 1e-12)
        Tn = np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])
        return (Tn @ np.c_[P, np.ones(len(P))].T).T[:, :2], Tn
    s, Ts = norm(src)
    d, Td = norm(dst)
    A = []
    for (x, y), (u, v) in zip(s, d):
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
    _, _, Vt = np.linalg.svd(np.array(A), full_matrices=False)
    H = np.linalg.inv(Td) @ Vt[-1].reshape(3, 3) @ Ts
    return H / H[2, 2]


def apply_H(H, P):
    Ph = (H @ np.c_[P, np.ones(len(P))].T).T
    return Ph[:, :2] / Ph[:, 2:3]


def robust_homography(src, dst, iters=2, keep=0.9):
    """Fit, drop the worst-residual pairs, refit — resists saccade/blink outliers."""
    idx = np.arange(len(src))
    H = fit_homography(src, dst)
    for _ in range(iters):
        r = np.hypot(*(apply_H(H, src) - dst).T)
        thr = np.quantile(r, keep)
        idx = np.where(r <= thr)[0]
        H = fit_homography(src[idx], dst[idx])
    return H, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal_pre", required=True, help="before-mapping output, calibration recording")
    ap.add_argument("--cal_post", required=True, help="after-calibration enrichment gaze.csv")
    ap.add_argument("--cal_post_recording", default=None, help="recording id prefix to filter cal_post")
    ap.add_argument("--task", required=True, help="before-mapping output, task recording")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # --- fit T: pre_ref -> post_ref on the calibration recording ---
    pre = pd.read_csv(args.cal_pre)
    post = pd.read_csv(args.cal_post)
    if args.cal_post_recording:
        post = post[post[REC].astype(str).str.startswith(args.cal_post_recording)]
    pre = pre[_truthy(pre[DET])][[TS, RX, RY]].dropna()
    post = post[_truthy(post[DET])][[TS, RX, RY]].dropna()
    m = pre.merge(post, on=TS, suffixes=("_pre", "_post"))
    src = m[[f"{RX}_pre", f"{RY}_pre"]].to_numpy()
    dst = m[[f"{RX}_post", f"{RY}_post"]].to_numpy()
    if len(m) < 8:
        raise SystemExit(f"too few matched calibration pairs: {len(m)}")
    raw_shift = np.hypot(*(dst - src).T).mean()
    T, keep = robust_homography(src, dst)
    res = np.hypot(*(apply_H(T, src[keep]) - dst[keep]).T)
    print(f"calibration pairs: {len(m)} (kept {len(keep)})")
    print(f"  raw pre→post shift: mean {raw_shift:.2f} px")
    print(f"  homography T residual: mean {res.mean():.2f} px, median {np.median(res):.2f} px")

    # --- recover task reference->screen homography from its ref<->transf ---
    task = pd.read_csv(args.task)
    det = _truthy(task[DET]).to_numpy()
    ref = task[[RX, RY]].to_numpy(float)
    trn = task[[TX, TY]].to_numpy(float)
    hv = det & np.isfinite(ref).all(1) & np.isfinite(trn).all(1)
    Href2scr, _ = robust_homography(ref[hv], trn[hv])
    sres = np.hypot(*(apply_H(Href2scr, ref[hv]) - trn[hv]).T)
    print(f"  ref→screen homography (recovered from task): residual mean {sres.mean():.2f} px, n {hv.sum()}")

    # --- apply: correct task reference gaze, then re-project to screen ---
    valid = det & np.isfinite(ref).all(1)
    ref_c = ref.copy()
    ref_c[valid] = apply_H(T, ref[valid])
    trn_c = trn.copy()
    trn_c[valid] = apply_H(Href2scr, ref_c[valid])

    out = task.copy()
    out[RX], out[RY] = ref_c[:, 0], ref_c[:, 1]
    out[TX], out[TY] = trn_c[:, 0], trn_c[:, 1]
    out.to_csv(args.out, index=False)
    moved = np.hypot(*(trn_c[valid] - trn[valid]).T)
    print(f"  task screen gaze shifted by: mean {np.nanmean(moved):.1f} px, "
          f"median {np.nanmedian(moved):.1f} px  ({valid.sum()} corrected samples)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
