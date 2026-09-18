#!/usr/bin/env python3
"""
measure_lag.py — recover the true cursor clock offset from the screen video.

The steering-JSON cursor is logged on the Mac clock; the cursor map's
clock_offset_s (Neon - Mac) aligns it to Neon time. That per-session estimate
can be off by ~0.2-0.3 s, which silently biases the gaze-lead DV (the JSON
cursor lags the REAL on-screen cursor during motion; it hides at parked moments).

This cross-correlates the real cursor's motion in the screen recording (tracked
by frame differencing in the most-active 60 s window) against the JSON cursor.
    corrected clock_offset_s = original clock_offset_s + L        (L is negative)

REQUIRES a MILLISECOND screen_start_wallclock. With a wallclock only to the
second, L absorbs the wallclock error and is unreliable — validate the offset
against the crosshair with a frame sweep instead (see CALIBRATION_PROCESS.md).

USAGE
  python measure_lag.py "<person folder>" "YYYY-MM-DD HH:MM:SS.fff"
  (folder must hold steering_experiment_*.json and the TASK screen .mov = largest .mov)
"""
import numpy as np, subprocess, sys, json, glob, os, datetime, time


def load_json_cursor(folder):
    steer = glob.glob(f"{folder}/steering_experiment_*.json")[0]
    j = json.load(open(steer)); trials = j["trialData"]
    ts = []; xs = []
    for tr in trials:
        for t, s in zip(tr.get("timestamps", []), tr.get("screenTrajectory", [])):
            ts.append(t / 1000.0); xs.append(s["x"])
    ts = np.array(ts); xs = np.array(xs); o = np.argsort(ts)
    return ts[o], xs[o]


def measure(folder, sstart_wall, tz="America/Denver"):
    os.environ["TZ"] = tz; time.tzset()
    ts, xs = load_json_cursor(folder)
    sstart = datetime.datetime.strptime(sstart_wall, "%Y-%m-%d %H:%M:%S.%f").timestamp()
    r = ts - sstart
    rec = [x for x in glob.glob(f"{folder}/*.mov") if os.path.getsize(x) > 5e7]
    rec = sorted(rec, key=os.path.getsize)[-1]          # the task recording (largest)
    grid = np.arange(max(0, r.min()), r.max() - 60, 5.0)
    mot = [np.abs(np.diff(np.interp(np.arange(t, t + 60, 0.1), r, xs))).sum() for t in grid]
    W0 = float(grid[int(np.argmax(mot))])
    W, H, fps = 640, 360, 15
    cmd = ["ffmpeg", "-v", "error", "-ss", str(W0), "-t", "60", "-i", rec,
           "-vf", f"fps={fps},scale={W}:{H},format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    fr = np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, np.uint8)
    nf = len(fr) // (W * H); fr = fr[:nf * W * H].reshape(nf, H, W).astype(np.float32)
    d = np.abs(np.diff(fr, axis=0))[:, 90:270, :]; prof = d.sum(1)
    k = np.ones(9) / 9; profs = np.array([np.convolve(pp, k, "same") for pp in prof])
    tot = prof.sum(1); lo, hi = np.percentile(tot, [55, 97]); val = (tot > lo) & (tot < hi)
    cx = np.full(len(prof), np.nan); cx[val] = profs[val].argmax(1)
    p = W0 + np.arange(len(cx)) / fps; xj = np.interp(p, r, xs); m = np.isfinite(cx)
    cz = (cx - np.nanmean(cx)) / (np.nanstd(cx) + 1e-9); jz = (xj - np.mean(xj[m])) / (np.std(xj[m]) + 1e-9)
    lags = np.arange(-0.6, 0.6001, 0.02)
    res = np.array([(L, np.corrcoef(cz[m], np.interp(p - L, p, jz)[m])[0, 1]) for L in lags])
    kk = int(np.argmax(res[:, 1])); L, c = res[kk]
    return W0, nf, int(m.sum()), L, c


if __name__ == "__main__":
    folder, wall = sys.argv[1], sys.argv[2]
    W0, nf, nv, L, c = measure(folder, wall)
    print(f"{os.path.basename(folder):10s} window@{W0:.0f}s valid={nv} lag L={L:+.3f}s corr={c:+.2f}")
