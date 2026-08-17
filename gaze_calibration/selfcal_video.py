#!/usr/bin/env python3
"""
selfcal_video — produce a self-calibrated gaze CSV in the SCREEN-VIDEO frame,
so `compose3` can render the corrected gaze on the actual recording.

`self_calibrate` aligns gaze to the cursor in *task* coordinates. To draw it on
the screen video you need the gaze in screen-video pixels aligned to where the
cursor is *rendered*. This tool:
  1. maps the JSON cursor into screen-video px:  cursor_vid = scale*screenTraj + b
     (scale/bx/by come from detecting the on-screen start/target — see WORKFLOWS);
  2. time-aligns gaze (Neon ns) with the cursor (JSON Date.now ms; offset searched);
  3. fits a robust homography  C: gaze_transf_px -> cursor_vid_px  (the calibration
     in the video frame) and applies it to every gaze sample;
  4. back-projects the corrected screen gaze into reference-image px (via the
     recovered reference<->screen homography) so the Reference panel is corrected too.
Output: a gaze CSV with corrected `gaze position transf x/y [px]` and `gaze
position in reference image x/y [px]`, ready for `compose3`.

Note: the correction is in screen space; the Neon scene-cam panel keeps the raw
scene gaze (no scene-space reference to correct it).

USAGE
    python selfcal_video.py --gaze_csv <task>_gaze.csv \
        --experiment_json steering_*.json \
        --cursor_scale 2.0105 --cursor_bx 3857.0 --cursor_by 1706.7 \
        --out_csv <task>_gaze_selfcal_video.csv
"""
import argparse
import json

import numpy as np
import pandas as pd

TX = "gaze position transf x [px]"
TY = "gaze position transf y [px]"
RX = "gaze position in reference image x [px]"
RY = "gaze position in reference image y [px]"
DET = "gaze detected in reference image"
TS = "timestamp [ns]"


def _fitH(src, dst):
    def nrm(P):
        m = P.mean(0)
        s = np.sqrt(2) / (np.sqrt(((P - m) ** 2).sum(1)).mean() + 1e-12)
        T = np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])
        return (T @ np.c_[P, np.ones(len(P))].T).T[:, :2], T
    s, Ts = nrm(src)
    d, Td = nrm(dst)
    A = []
    for (x, y), (u, v) in zip(s, d):
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
    _, _, Vt = np.linalg.svd(np.array(A), full_matrices=False)
    H = np.linalg.inv(Td) @ Vt[-1].reshape(3, 3) @ Ts
    return H / H[2, 2]


def _appH(H, P):
    Ph = (H @ np.c_[P, np.ones(len(P))].T).T
    return Ph[:, :2] / Ph[:, 2:3]


def _robustH(src, dst, it=3, keep=0.8):
    idx = np.arange(len(src))
    H = _fitH(src, dst)
    for _ in range(it):
        r = np.hypot(*(_appH(H, src) - dst).T)
        idx = np.where(r <= np.quantile(r, keep))[0]
        if len(idx) < 12:
            break
        H = _fitH(src[idx], dst[idx])
    return H, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaze_csv", required=True)
    ap.add_argument("--experiment_json", required=True)
    ap.add_argument("--cursor_scale", type=float, required=True)
    ap.add_argument("--cursor_bx", type=float, required=True)
    ap.add_argument("--cursor_by", type=float, required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--tol_ms", type=float, default=50.0)
    args = ap.parse_args()

    g = pd.read_csv(args.gaze_csv)
    det = (g[DET].astype(str).str.lower().isin(["true", "1", "1.0"]).to_numpy()
           if DET in g.columns else np.ones(len(g), bool))
    gt = g[TS].to_numpy(float) / 1e9
    gtr = g[[TX, TY]].to_numpy(float)

    # cursor (JSON) -> screen-video px
    d = json.load(open(args.experiment_json))
    ct, cvid = [], []
    for tr in d.get("trialData", []):
        ts = tr.get("timestamps") or []
        st = tr.get("screenTrajectory") or []
        n = min(len(ts), len(st))
        if n < 2:
            continue
        ct.append(np.asarray(ts[:n], float) / 1e3)
        sc = np.array([[st[k]["x"], st[k]["y"]] for k in range(n)], float)
        cvid.append(args.cursor_scale * sc + np.array([args.cursor_bx, args.cursor_by]))
    ct = np.concatenate(ct); cvid = np.concatenate(cvid)
    o = np.argsort(ct); ct, cvid = ct[o], cvid[o]
    tol = args.tol_ms / 1000.0

    # --- fit C: gaze_transf -> cursor_vid, with clock-offset search ----------
    valid = det & np.isfinite(gtr).all(1)
    best = None
    for off in np.arange(-0.40, 0.401, 0.02):
        j = np.clip(np.searchsorted(ct, gt + off), 0, len(ct) - 1)
        m = valid & (np.abs(ct[j] - (gt + off)) < tol)
        if m.sum() < 100:
            continue
        C, idx = _robustH(gtr[m], cvid[j[m]])
        r = np.hypot(*(_appH(C, gtr[m][idx]) - cvid[j[m]][idx]).T)
        med = float(np.median(r))
        if best is None or med < best[0]:
            best = (med, float(off), C)
    med, off, C = best

    # before/after gaze->cursor distance (screen-video px)
    j = np.clip(np.searchsorted(ct, gt + off), 0, len(ct) - 1)
    m = valid & (np.abs(ct[j] - (gt + off)) < tol)
    before = np.hypot(*(gtr[m] - cvid[j[m]]).T)
    after = np.hypot(*(_appH(C, gtr[m]) - cvid[j[m]]).T)
    print(f"clock offset {off:+.2f}s, {m.sum()} matches")
    print(f"  gaze→cursor (screen px): before median {np.median(before):.0f}px "
          f"→ after median {np.median(after):.0f}px")

    # --- apply C to all gaze; back-project to reference px --------------------
    tr_c = gtr.copy()
    tr_c[valid] = _appH(C, gtr[valid])
    out = g.copy()
    out[TX], out[TY] = tr_c[:, 0], tr_c[:, 1]
    if RX in g.columns:
        rp = g[[RX, RY]].to_numpy(float)
        hv = valid & np.isfinite(rp).all(1) & np.isfinite(gtr).all(1)
        Hr2s, _ = _robustH(rp[hv], gtr[hv])       # reference -> screen (exact)
        Hs2r = np.linalg.inv(Hr2s)
        rc = rp.copy()
        rc[valid] = _appH(Hs2r, tr_c[valid])
        out[RX], out[RY] = rc[:, 0], rc[:, 1]

    out.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
