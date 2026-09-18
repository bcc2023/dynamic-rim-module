#!/usr/bin/env python3
"""
full_selfcal_pipeline.py — one clean, consistent per-participant pass:
  9-point (adaptive) -> align -> parked-target self-cal (fit+apply on the SAME
  gaze, so no mismatched-base bug) -> re-align -> analysis (pilot 38-col format).
Reuses the existing cursor map (unchanged by gaze calibration). Writes the
<P>_task_aligned.csv, <P>_task_aligned_analysis.csv, and copies the analysis to
the collection as <pid>_task_aligned_analysis.csv.

USAGE
  python full_selfcal_pipeline.py --person Dewen --base "<batch folder>" \
     --pid p03 --raw_rel "<task raw folder rel path>" --collection "<task_aligned_all>"
"""
import argparse, sys, os, glob, subprocess, shutil, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from analyze_aligned import add_gaze_lead_column
TX = "gaze position transf x [px]"; TY = "gaze position transf y [px]"
from pilot_columns import PILOT_COLUMNS as PILOT   # the 38-col pilot schema, embedded


def des(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); e = np.ones_like(x)
    return np.column_stack([e, x, y, x * x, y * y, x * y])


def rfit(D, z, iters=4):
    c = np.linalg.lstsq(D, z, rcond=None)[0]
    for _ in range(iters):
        r = D @ c - z; s = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-9
        w = 1.0 / np.maximum(1.0, np.abs(r) / (3 * s))
        c = np.linalg.lstsq(D * w[:, None], z * w, rcond=None)[0]
    return c


def assign_id(ts, ev, col):
    s = ev["start timestamp [ns]"].to_numpy(); e = ev["end timestamp [ns]"].to_numpy(); i = ev[col].to_numpy()
    o = np.argsort(s); s, e, i = s[o], e[o], i[o]; j = np.searchsorted(s, ts, side="right") - 1
    out = np.full(len(ts), np.nan); ok = j >= 0; w = ok & (ts <= e[np.clip(j, 0, len(e) - 1)])
    out[w] = i[np.clip(j, 0, len(i) - 1)][w]; return out


def run(cmd): subprocess.run([str(c) for c in cmd], check=True, capture_output=True)


def gc(A):
    d = A[A["in_trial"].astype(str).isin(["True", "1", "1.0"])].dropna(subset=["cursor_screenvid_x_norm"])
    return np.sqrt((d["gaze_transf_x_norm"] - d["cursor_screenvid_x_norm"]) ** 2 + (d["gaze_transf_y_norm"] - d["cursor_screenvid_y_norm"]) ** 2).median()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--pid", required=True); ap.add_argument("--raw_rel", required=True); ap.add_argument("--collection", required=True)
    a = ap.parse_args(); P = a.person; d = os.path.join(a.base, P)
    cal = f"{d}/{P}_task_gaze_calibrated.csv"; af = glob.glob(f"{d}/[A-Za-z]*_task_aligned.csv")[0]
    cmap = glob.glob(f"{d}/{P}_task_cursor_map.json")[0]; steer = glob.glob(f"{d}/steering_experiment_*.json")[0]
    timing = f"{d}/calibration_9point_timing.csv"; task_gaze = f"{d}/{P}_task_gaze.csv"
    cal_gaze = (glob.glob(f"{d}/{P}_calibration_gaze.csv") or glob.glob(f"{d}/{P}_calib_gaze.csv"))[0]
    W = json.load(open(cmap)).get("screen_width", 3840.0); Hh = json.load(open(cmap)).get("screen_height", 2160.0)
    raw = os.path.join(a.base, a.raw_rel)
    gc0 = gc(pd.read_csv(af))
    # 1) fresh adaptive 9-point ; 2) align
    run(["python", f"{HERE}/apply_9point_calibration.py", "--cal_gaze", cal_gaze, "--timing", timing, "--task_gaze", task_gaze, "--out", cal, "--screen_w", int(W), "--screen_h", int(Hh)])
    run(["python", f"{HERE}/cursor_gaze_pipeline.py", "--gaze_csv", cal, "--experiment_json", steer, "--cursor_map", cmap, "--out_csv", af])
    # 3) parked self-cal (fit on THIS aligned, apply to THIS cal) -> re-align
    run(["python", f"{HERE}/apply_selfcal.py", "--aligned", af, "--gaze_csv", cal, "--out", cal])
    run(["python", f"{HERE}/cursor_gaze_pipeline.py", "--gaze_csv", cal, "--experiment_json", steer, "--cursor_map", cmap, "--out_csv", af])
    # 4) analysis (38-col pilot format)
    df = pd.read_csv(af); df, _ = add_gaze_lead_column(df, steer)
    gx, gy, cx, cy = df["gaze_transf_x_px"], df["gaze_transf_y_px"], df["cursor_screenvid_x_px"], df["cursor_screenvid_y_px"]
    dist = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2); v = df["in_trial"].astype(bool) & np.isfinite(dist)
    df["gaze_cursor_dist_px"] = np.where(v, dist, np.nan); df["gaze_cursor_signed_px"] = np.where(v, dist * np.sign(gx - cx), np.nan)
    df["time_to_catch_s"] = np.where(v & (df["cursor_speed"].abs() > 1e-3), df["gaze_lead_signed"] / df["cursor_speed"], np.nan)
    df["json_local_curvature"] = df["difficulty_curvature"]; df["json_local_width"] = df["difficulty_tunnelWidth"]
    ts = df["neon_timestamp_ns"].to_numpy(); fx = df["fixation_id"].isna().to_numpy()
    bl = assign_id(ts, pd.read_csv(f"{raw}/blinks.csv"), "blink id"); sa = assign_id(ts, pd.read_csv(f"{raw}/saccades.csv"), "saccade id")
    df["blink_id"] = np.where(fx, bl, np.nan); df["saccade_id"] = np.where(fx & np.isnan(df["blink_id"]), sa, np.nan)
    b2u = {c.split(" (")[0].strip().strip('"'): c for c in df.columns}
    out = pd.DataFrame({pc: df[b2u[pc.split(' (')[0].strip().strip('"')]] for pc in PILOT})
    outf = af.replace("_task_aligned.csv", "_task_aligned_analysis.csv"); out.to_csv(outf, index=False)
    tgt = f"{a.collection}/{a.pid}_task_aligned_analysis.csv"
    if os.path.abspath(tgt) != os.path.abspath(outf): shutil.copy(outf, tgt)
    print(f"{P:8s} gc {gc0:.3f} -> {gc(pd.read_csv(af)):.3f} | analysis {len(out.columns)}col pilot={list(out.columns)==PILOT} -> {a.pid}")


if __name__ == "__main__":
    main()
