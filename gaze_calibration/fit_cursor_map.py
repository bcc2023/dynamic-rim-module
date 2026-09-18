#!/usr/bin/env python3
"""
fit_cursor_map — derive a recording's cursor→screen-video map (and clock offset)
automatically, so `align` can place the JSON cursor on the screen recording.

The task draws an ORANGE target disk at each trial's target; in the pre-trial
"waiting" frame it sits still at a known screen position (the cursor's
screenTrajectory end for that trial). We seek that frame for many trials, detect
the orange disk's video-pixel centroid, and least-squares fit a uniform map
    cursor_video_px = scale * screenTrajectory + (bx, by)
(3 params: scale, bx, by), with robust outlier trimming.

If a calibrated gaze CSV is given, we also estimate clock_offset_s (Neon − Mac)
by cross-correlating cursor-x(Mac time) with gaze-x(Neon time).

Output: a small JSON with cursor_scale / cursor_bx / cursor_by / screen_width /
screen_height / clock_offset_s — feed it to `align --cursor_map`.

USAGE
    python fit_cursor_map.py --screen_video <task>_screen.mov \
        --experiment_json steering_*.json \
        [--gaze_csv <task>_gaze_calibrated.csv] \
        --out_map <task>_cursor_map.json [--sample_every 3] [--creation_time ISO]
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

try:
    from scipy import ndimage
except Exception:
    ndimage = None

TS = "timestamp [ns]"
TX = "gaze position transf x [px]"
TY = "gaze position transf y [px]"


def ffprobe(video, show_entries, stream=False):
    sel = ["-select_streams", "v:0"] if stream else []
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", *sel, "-show_entries", show_entries,
         "-of", "default=nk=1:nw=1", video],
        capture_output=True, text=True).stdout.split()
    return out


def creation_epoch(video, override):
    if override:
        s = override
    else:
        s = (ffprobe(video, "format_tags=creation_time") or [None])[0]
        if s is None:
            s = (ffprobe(video, "stream_tags=creation_time", stream=True) or [None])[0]
    if not s:
        sys.exit("ERROR: no creation_time in video; pass --creation_time "
                 "'YYYY-MM-DDTHH:MM:SS.sssZ'")
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def orange_centroid(img):
    a = np.asarray(img).astype(int)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    m = (R > G + 20) & (G >= B - 2) & (R > 80) & (R < 215) & (B < 150)
    if m.sum() < 300:
        return None
    if ndimage is not None:
        lab, n = ndimage.label(m)
        if n == 0:
            return None
        sz = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        k = int(np.argmax(sz)) + 1
        if sz[k - 1] < 300:
            return None
        ys, xs = np.where(lab == k)
    else:
        ys, xs = np.where(m)
    return float(xs.mean()), float(ys.mean())


def _fit_uniform(S, V):
    A = np.zeros((2 * len(S), 3)); b = np.zeros(2 * len(S))
    A[0::2, 0] = S[:, 0]; A[0::2, 1] = 1; b[0::2] = V[:, 0]
    A[1::2, 0] = S[:, 1]; A[1::2, 2] = 1; b[1::2] = V[:, 1]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol


def fit_map(video, jf, cre, every, tmp):
    d = json.load(open(jf))["trialData"]
    trials = []
    for tr in d:
        ts = tr.get("timestamps") or []
        st = tr.get("screenTrajectory") or []
        if len(ts) >= 2 and len(st) >= 2:
            trials.append((ts[0] / 1000.0, st[-1]))
    trials.sort()
    S, V = [], []
    frame = os.path.join(tmp, "f.png")
    for tw, sT in trials[::every]:
        seek = tw - cre - 0.4
        if seek < 0:
            continue
        subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-ss", str(seek),
                        "-i", video, "-frames:v", "1", frame])
        if not os.path.exists(frame):
            continue
        o = orange_centroid(Image.open(frame).convert("RGB"))
        if o:
            S.append((sT["x"], sT["y"])); V.append(o)
    S = np.array(S, float); V = np.array(V, float)
    if len(S) < 4:
        sys.exit(f"ERROR: only {len(S)} orange detections; can't fit")
    sol = _fit_uniform(S, V); keep = np.ones(len(S), bool)
    for _ in range(3):
        pred = np.c_[sol[0] * S[:, 0] + sol[1], sol[0] * S[:, 1] + sol[2]]
        r = np.hypot(*(pred - V).T)
        keep = r <= np.quantile(r, 0.8)
        sol = _fit_uniform(S[keep], V[keep])
    pred = np.c_[sol[0] * S[:, 0] + sol[1], sol[0] * S[:, 1] + sol[2]]
    resid = float(np.median(np.hypot(*(pred - V).T)[keep]))
    return float(sol[0]), float(sol[1]), float(sol[2]), len(S), int(keep.sum()), resid


def estimate_offset(jf, gaze_csv, scale, bx, by):
    """clock_offset_s = Neon − Mac: pick the offset minimizing the median gaze↔
    cursor distance (cursor mapped to video px with the fitted map). Unlike raw
    position cross-correlation, this is not biased by the gaze lead — the lead is
    a roughly constant tangential floor at every offset, so the true time sync is
    where total distance is smallest. Returns (offset, median_dist_px)."""
    d = json.load(open(jf))["trialData"]
    mt, cxv, cyv = [], [], []
    for tr in d:
        ts = tr.get("timestamps") or []
        st = tr.get("screenTrajectory") or []
        n = min(len(ts), len(st))
        if n < 2:
            continue
        mt.append(np.asarray(ts[:n], float) / 1000.0)
        cxv.append(scale * np.array([st[k]["x"] for k in range(n)], float) + bx)
        cyv.append(scale * np.array([st[k]["y"] for k in range(n)], float) + by)
    mt = np.concatenate(mt); cxv = np.concatenate(cxv); cyv = np.concatenate(cyv)
    o = np.argsort(mt); mt, cxv, cyv = mt[o], cxv[o], cyv[o]
    g = pd.read_csv(gaze_csv)
    gt = g[TS].to_numpy(float) / 1e9
    gx = g[TX].to_numpy(float); gy = g[TY].to_numpy(float)
    ok = np.isfinite(gt) & np.isfinite(gx) & np.isfinite(gy)
    gt, gx, gy = gt[ok], gx[ok], gy[ok]
    best = (1e18, 0.0)
    for off in np.arange(-2.0, 2.001, 0.02):
        mac = gt - off
        m = (mac >= mt[0]) & (mac <= mt[-1])
        if m.sum() < 100:
            continue
        cx = np.interp(mac[m], mt, cxv); cy = np.interp(mac[m], mt, cyv)
        med = float(np.median(np.hypot(cx - gx[m], cy - gy[m])))
        if med < best[0]:
            best = (med, float(off))
    return best[1], best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen_video", required=True)
    ap.add_argument("--experiment_json", required=True)
    ap.add_argument("--gaze_csv", default=None,
                    help="calibrated gaze CSV; enables clock-offset estimate")
    ap.add_argument("--out_map", required=True)
    ap.add_argument("--sample_every", type=int, default=3)
    ap.add_argument("--creation_time", default=None)
    args = ap.parse_args()

    wh = ffprobe(args.screen_video, "stream=width,height", stream=True)
    W, H = (float(wh[0]), float(wh[1])) if len(wh) >= 2 else (3840.0, 2160.0)
    cre = creation_epoch(args.screen_video, args.creation_time)

    with tempfile.TemporaryDirectory() as tmp:
        scale, bx, by, n, kept, resid = fit_map(
            args.screen_video, args.experiment_json, cre, args.sample_every, tmp)
    print(f"cursor map: scale={scale:.4f} bx={bx:.1f} by={by:.1f}  "
          f"({kept}/{n} pts, median resid {resid:.1f}px)")

    offset, med = (0.0, None)
    if args.gaze_csv:
        offset, med = estimate_offset(args.experiment_json, args.gaze_csv, scale, bx, by)
        print(f"clock offset (Neon−Mac): {offset:+.2f}s  (min median gaze↔cursor {med:.0f}px)")

    out = {"cursor_scale": scale, "cursor_bx": bx, "cursor_by": by,
           "screen_width": W, "screen_height": H, "clock_offset_s": offset,
           "n_points": n, "kept": kept, "median_resid_px": resid,
           "offset_median_dist_px": med}
    json.dump(out, open(args.out_map, "w"), indent=2)
    print(f"wrote {args.out_map}")


if __name__ == "__main__":
    main()
