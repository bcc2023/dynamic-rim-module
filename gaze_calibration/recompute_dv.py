#!/usr/bin/env python3
"""
recompute_dv.py — re-align the (already self-cal'd) task gaze against the cursor
using a CORRECTED clock offset, then rebuild the pilot-format analysis CSV.

Only the CURSOR timing changes (clock_offset_s); the calibrated gaze is untouched.
This fixes the gaze-lead DV, which was biased because the steering-JSON cursor was
time-shifted relative to the true on-screen cursor. Videos are NOT re-rendered
(they burn gaze + the real recorded cursor, never the JSON cursor).

USAGE
  python recompute_dv.py --person Dewen --base "<batch>" --pid p03 \
     --raw_rel "<task raw rel>" --collection "<task_aligned_all>" --clock_offset_s 0.10
"""
import argparse, sys, os, glob, subprocess, shutil
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from analyze_aligned import add_gaze_lead_column
from pilot_columns import PILOT_COLUMNS as PILOT   # the 38-col pilot schema, embedded


def assign_id(ts, ev, col):
    s = ev["start timestamp [ns]"].to_numpy(); e = ev["end timestamp [ns]"].to_numpy(); i = ev[col].to_numpy()
    o = np.argsort(s); s, e, i = s[o], e[o], i[o]; j = np.searchsorted(s, ts, side="right") - 1
    out = np.full(len(ts), np.nan); ok = j >= 0; w = ok & (ts <= e[np.clip(j, 0, len(e) - 1)])
    out[w] = i[np.clip(j, 0, len(i) - 1)][w]; return out


def run(cmd): subprocess.run([str(c) for c in cmd], check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--pid", required=True); ap.add_argument("--raw_rel", required=True)
    ap.add_argument("--collection", required=True); ap.add_argument("--clock_offset_s", type=float, required=True)
    ap.add_argument("--gaze_shift_s", type=float, default=0.0, help="advance gaze vs cursor (+ = gaze leads); match compose's --gaze_shift_s")
    a = ap.parse_args(); P = a.person; d = os.path.join(a.base, P)
    cal = f"{d}/{P}_task_gaze_calibrated.csv"; af = glob.glob(f"{d}/[A-Za-z]*_task_aligned.csv")[0]
    cmap = glob.glob(f"{d}/{P}_task_cursor_map.json")[0]; steer = glob.glob(f"{d}/steering_experiment_*.json")[0]
    raw = os.path.join(a.base, a.raw_rel)
    # re-align with the corrected cursor clock offset (gaze unchanged)
    run(["python", f"{HERE}/cursor_gaze_pipeline.py", "--gaze_csv", cal, "--experiment_json", steer,
         "--cursor_map", cmap, "--out_csv", af, "--clock_offset_s", a.clock_offset_s, "--gaze_shift_s", a.gaze_shift_s])
    # rebuild analysis (pilot 38-col)
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
    # report corrected directional lead
    dd = df[df["in_trial"].astype(bool)].dropna(subset=["cursor_screenvid_x_px","gaze_transf_x_px"]).sort_values("neon_timestamp_ns")
    cx2, cy2 = dd.cursor_screenvid_x_px.to_numpy(), dd.cursor_screenvid_y_px.to_numpy()
    gx2, gy2 = dd.gaze_transf_x_px.to_numpy(), dd.gaze_transf_y_px.to_numpy()
    vx, vy = np.gradient(cx2), np.gradient(cy2); sp = np.hypot(vx, vy); m = sp > np.percentile(sp, 40)
    lead = (gx2 - cx2) * vx / (sp + 1e-9) + (gy2 - cy2) * vy / (sp + 1e-9)
    print(f"{P:8s} pid {a.pid} offset {a.clock_offset_s:+.2f}  moving-median lead {np.median(lead[m]):+5.0f}px  {np.mean(lead[m]>0)*100:3.0f}% ahead  pilot={list(out.columns)==PILOT}")


if __name__ == "__main__":
    main()
