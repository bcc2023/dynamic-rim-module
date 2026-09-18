#!/usr/bin/env python3
"""
run_calibrated_pipeline.py — one command, per person, to produce the CALIBRATED
task outputs for the Umich Cursor set, AFTER pl-dynamic-rim has already made the
mapped gaze CSVs (<Person>_task_gaze.csv, <Person>_calibration_gaze.csv).

Per person it runs, in order:
  1. apply_9point_calibration.py  -> <Person>_task_gaze_calibrated.csv
  2. fit_cursor_map.py            -> <Person>_task_cursor_map.json
  3. cursor_gaze_pipeline.py      -> <Person>_task_aligned.csv   (32-col, == A/B/C)
  4. QC + conditional rig-shift correction (self-reporting, no eyeballing):
        - measures precision (calibration RMS, within-fixation jitter, on-target
          accuracy) and the residual VERTICAL gaze-vs-cursor bias;
        - if that bias exceeds --vfix_threshold, a small vertical shift of the
          Neon between the (separate) calibration and task recordings is assumed,
          and apply_vertical_recal.py corrects it (VERTICAL only; horizontal gaze,
          where the gaze-lead signal lives, is left untouched), then re-aligns.
  5. compose_3panel.py            -> <Person>_task_3panel_calibrated.mp4

Paths + the manual on-screen-clock wallclock come from rim_manifest.csv (the TASK
row). scene_start_ns = the raw folder's recording.begin; start_video_ns = the
wallclock in a PINNED UTC-6 tz (both Umich recordings reconcile only under UTC-6),
so the command is correct regardless of your Mac's timezone. Override with --tz.

USAGE (in the pupil-cloud-local conda env, from the "Umich Cursor" folder):
    caffeinate -dimsu python ".../run_calibrated_pipeline.py" --person Yanran
    #  --no-video        only the CSV (fast), skip the 20-min render
    #  --skip-existing   reuse an existing calibrated gaze CSV / cursor map
    #  --no_vfix         never apply the vertical correction (QC still reported)
    #  --force_vfix      always apply it, regardless of the measured bias
    #  --vfix_threshold  bias (fraction of screen height) that triggers it (def .02)
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, capture=False):
    print("\n$ " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd), flush=True)
    if capture:
        r = subprocess.run([str(c) for c in cmd], check=True, capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        if r.stderr:
            sys.stderr.write(r.stderr)
        return r.stdout
    subprocess.run([str(c) for c in cmd], check=True)
    return ""


def first(pattern, what):
    hits = sorted(glob.glob(pattern))
    if not hits:
        sys.exit(f"ERROR: no {what} matched {pattern!r}")
    return hits[0]


def wallclock_to_ns(wc, tz):
    import datetime
    os.environ["TZ"] = tz
    try:
        time.tzset()
    except Exception:
        pass
    for f in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f",
              "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return int(round(datetime.datetime.strptime(wc, f).timestamp() * 1e9))
        except ValueError:
            pass
    sys.exit(f"ERROR: could not parse wallclock {wc!r}")


def recording_begin_ns(raw_dir):
    ev = pd.read_csv(os.path.join(raw_dir, "events.csv"))
    row = ev[ev["name"] == "recording.begin"]
    if row.empty:
        sys.exit(f"ERROR: no recording.begin in {raw_dir}/events.csv")
    return int(row["timestamp [ns]"].iloc[0])


def align_and_merge(cal_out, steering, cmap, aligned, clock_offset=None):
    """Run align, then merge fixation_id (positional, exact) to get the 32-col CSV.
    clock_offset (Neon-Mac, s) overrides the cursor_map value so the cursor is
    time-aligned to the true on-screen cursor (see measure_lag.py / recompute_dv.py)."""
    cmd = ["python", os.path.join(HERE, "cursor_gaze_pipeline.py"),
           "--gaze_csv", cal_out, "--experiment_json", steering,
           "--cursor_map", cmap, "--out_csv", aligned]
    if clock_offset is not None:
        cmd += ["--clock_offset_s", str(clock_offset)]
    run(cmd)
    adf = pd.read_csv(aligned)
    if "fixation_id" not in adf.columns:
        gz = pd.read_csv(cal_out, usecols=["timestamp [ns]", "fixation id"]) \
            .sort_values("timestamp [ns]").reset_index(drop=True)
        pos = list(adf.columns).index("gaze_transf_y_norm") + 1
        if len(gz) == len(adf) and \
           (gz["timestamp [ns]"].astype("int64").to_numpy()
                == adf["neon_timestamp_ns"].astype("int64").to_numpy()).all():
            fid = gz["fixation id"].to_numpy()
        else:
            fmap = dict(zip(gz["timestamp [ns]"].astype("int64"), gz["fixation id"]))
            fid = adf["neon_timestamp_ns"].astype("int64").map(fmap)
        adf.insert(pos, "fixation_id", fid)
        adf.to_csv(aligned, index=False)


def qc_report(aligned, W, H, cal_rms, label):
    """Self-reported precision + the vertical bias that drives the rig-shift fix."""
    a = pd.read_csv(aligned, usecols=[
        "gaze_transf_x_px", "gaze_transf_y_px", "gaze_transf_y_norm",
        "cursor_screenvid_x_px", "cursor_screenvid_y_px", "cursor_screenvid_y_norm",
        "fixation_id", "in_trial", "cursor_speed"])
    it = a["in_trial"].astype(str).isin(["True", "1", "1.0"])
    # within-fixation jitter (precision floor: sensor + RIM)
    fx = a.dropna(subset=["fixation_id"])
    g = fx.groupby("fixation_id")
    jit = np.sqrt(g["gaze_transf_x_px"].std() ** 2 + g["gaze_transf_y_px"].std() ** 2)
    jit = jit[g.size() >= 10]
    jit_px = float(jit.median()) if len(jit) else float("nan")
    # vertical bias (gaze - cursor), in-trial, by screen third
    d = a[it].dropna(subset=["cursor_screenvid_y_norm"])
    dyn = (d["gaze_transf_y_norm"] - d["cursor_screenvid_y_norm"])
    vbias = float(dyn.median())
    thirds = []
    for lo, hi in [(0, .33), (.33, .66), (.66, 1)]:
        mm = d["gaze_transf_y_norm"].between(lo, hi)
        thirds.append(float(dyn[mm].median()) if mm.any() else float("nan"))
    # on-target accuracy (cursor parked + fixating)
    park = a[it & a["fixation_id"].notna() & (a["cursor_speed"] < 0.03)].dropna(subset=["cursor_screenvid_x_px"])
    acc = np.sqrt((park["gaze_transf_x_px"] - park["cursor_screenvid_x_px"]) ** 2
                  + (park["gaze_transf_y_px"] - park["cursor_screenvid_y_px"]) ** 2)
    acc_px = float(acc.median()) if len(park) else float("nan")

    print(f"\n---- {label} ----")
    if cal_rms is not None:
        print(f"   calibration accuracy at dots : {cal_rms:5.0f} px  ({cal_rms/W*100:.1f}% width)")
    print(f"   within-fixation jitter       : {jit_px:5.0f} px  ({jit_px/W*100:.1f}% width)   [precision floor]")
    print(f"   on-target accuracy           : {acc_px:5.0f} px  ({acc_px/W*100:.1f}% width)")
    print(f"   vertical bias gaze-cursor    : {vbias*H:+5.0f} px  ({vbias*100:+.1f}% height)   "
          f"top/mid/bot {thirds[0]*100:+.1f}/{thirds[1]*100:+.1f}/{thirds[2]*100:+.1f}%")
    return vbias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True)
    ap.add_argument("--base", default=".")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--rim_folder", default=None)
    ap.add_argument("--tz", default="America/Denver")
    ap.add_argument("--screen_w", type=float, default=3840.0)
    ap.add_argument("--screen_h", type=float, default=2160.0)
    ap.add_argument("--no-video", dest="no_video", action="store_true")
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    ap.add_argument("--clock_offset_s", type=float, default=None,
                    help="corrected Neon-Mac offset (s) for cursor timing; see measure_lag.py")
    ap.add_argument("--cursor_overlay", action="store_true",
                    help="overlay the (corrected) cursor as a cyan ring on the screen panel")
    ap.add_argument("--gaze_shift_s", type=float, default=0.0,
                    help="advance the gaze vs the screen (preview; + = gaze looks ahead; does not change the DV)")
    ap.add_argument("--start_s", type=float, default=None, help="render only a window (s from first gaze)")
    ap.add_argument("--duration_s", type=float, default=None, help="window length (s) for a quick preview")
    ap.add_argument("--no_vfix", action="store_true", help="never apply the vertical rig-shift correction")
    ap.add_argument("--force_vfix", action="store_true", help="always apply the vertical correction")
    ap.add_argument("--vfix_threshold", type=float, default=0.02,
                    help="|vertical bias| (fraction of screen height) that triggers the correction")
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()

    base = os.path.abspath(a.base)
    manifest = a.manifest or os.path.join(base, "rim_manifest.csv")
    m = pd.read_csv(manifest)
    row = m[(m["person"].str.lower() == a.person.lower()) & (m["kind"] == "task")]
    if row.empty:
        sys.exit(f"ERROR: no TASK row for person={a.person!r} in {manifest}")
    row = row.iloc[0]

    P = a.person
    pdir = os.path.join(base, P)
    raw_dir = os.path.join(base, str(row["raw_rel"]))
    screen_video = os.path.join(base, str(row["screen_rel"]))
    wc = str(row["screen_start_wallclock"]).strip()
    if not wc or wc.lower() == "nan":
        sys.exit(f"ERROR: blank screen_start_wallclock for {P} task in the manifest")

    steering = first(os.path.join(pdir, "steering_experiment_*.json"), "steering JSON")
    timing = os.path.join(pdir, "calibration_9point_timing.csv")
    cal_gaze = next((p for p in [os.path.join(pdir, f"{P}_calibration_gaze.csv"),
                                 os.path.join(pdir, f"{P}_calib_gaze.csv")] if os.path.exists(p)),
                    os.path.join(pdir, f"{P}_calibration_gaze.csv"))
    task_gaze = os.path.join(pdir, f"{P}_task_gaze.csv")
    cal_out = os.path.join(pdir, f"{P}_task_gaze_calibrated.csv")
    ninept = os.path.join(pdir, f"{P}_task_gaze_9pt.csv")   # 9-point-only backup (reversible)
    cmap = os.path.join(pdir, f"{P}_task_cursor_map.json")
    aligned = os.path.join(pdir, f"{P}_task_aligned.csv")
    video_out = os.path.join(pdir, f"{P}_task_3panel_calibrated.mp4")
    scene_video = first(os.path.join(raw_dir, "*.mp4"), "scene video (raw folder)")
    scene_gaze = os.path.join(raw_dir, "gaze.csv")
    rim = a.rim_folder or os.path.dirname(
        first(os.path.join(base, "*REFERENCE-IMAGE-MAPPER*", "reference_image.*"), "RIM reference_image"))
    ref_img = first(os.path.join(rim, "reference_image.*"), "reference_image")

    for f in (steering, timing, cal_gaze, task_gaze, screen_video, scene_video, scene_gaze):
        if not os.path.exists(f):
            sys.exit(f"ERROR: missing required input: {f}")

    scene_start_ns = recording_begin_ns(raw_dir)
    start_video_ns = wallclock_to_ns(wc, a.tz)
    implied = (start_video_ns - scene_start_ns) / 1e9
    print(f"\n==== {P} task ====")
    print(f"  scene_start_ns {scene_start_ns} | start_video_ns {start_video_ns} "
          f"(screen {abs(implied):.3f}s {'after' if implied>=0 else 'BEFORE'} Neon, "
          f"{'OK' if 0 <= implied < 10 else 'CHECK'})")

    # 1) 9-point calibration (capture its RMS for the QC line)
    cal_rms = None
    if a.skip_existing and os.path.exists(cal_out):
        print(f"\n[1/5] reuse existing {os.path.basename(cal_out)}")
    else:
        out = run(["python", os.path.join(HERE, "apply_9point_calibration.py"),
                   "--cal_gaze", cal_gaze, "--timing", timing, "--task_gaze", task_gaze,
                   "--out", cal_out, "--screen_w", int(a.screen_w), "--screen_h", int(a.screen_h)],
                  capture=True)
        mrms = re.search(r"after=([\d.]+)\s*px", out)
        cal_rms = float(mrms.group(1)) if mrms else None
        shutil.copy(cal_out, ninept)   # fresh 9-point-only backup

    # 2) cursor map (auto)
    if a.skip_existing and os.path.exists(cmap):
        print(f"\n[2/5] reuse existing {os.path.basename(cmap)}")
    else:
        run(["python", os.path.join(HERE, "fit_cursor_map.py"),
             "--screen_video", screen_video, "--experiment_json", steering,
             "--gaze_csv", cal_out, "--out_map", cmap])

    # 3) align -> 32-col task_aligned.csv
    print("\n[3/5] align + fixation merge")
    align_and_merge(cal_out, steering, cmap, aligned, a.clock_offset_s)

    # 4) QC + conditional vertical rig-shift correction
    print("\n[4/5] QC + rig-shift check")
    vbias = qc_report(aligned, a.screen_w, a.screen_h, cal_rms, "QC (after 9-point calibration)")
    do_vfix = a.force_vfix or (not a.no_vfix and abs(vbias) > a.vfix_threshold)
    if do_vfix:
        print(f"   -> vertical bias {vbias*100:+.1f}% exceeds ±{a.vfix_threshold*100:.0f}% "
              f"=> applying vertical rig-shift correction")
        if not os.path.exists(ninept):
            shutil.copy(cal_out, ninept)
        run(["python", os.path.join(HERE, "apply_vertical_recal.py"),
             "--aligned", aligned, "--gaze_csv", cal_out, "--out", cal_out,
             "--screen_h", int(a.screen_h)])
        align_and_merge(cal_out, steering, cmap, aligned, a.clock_offset_s)
        qc_report(aligned, a.screen_w, a.screen_h, cal_rms, "QC (after vertical correction)")
    else:
        why = "disabled (--no_vfix)" if a.no_vfix else f"within ±{a.vfix_threshold*100:.0f}%"
        print(f"   -> vertical bias {vbias*100:+.1f}% {why} => no correction needed")

    # 5) 3-panel video with the (corrected) calibrated gaze burned in
    if a.no_video:
        print("\n[5/5] --no-video: skipping the 3-panel render")
    else:
        print("\n[5/5] render 3-panel with calibrated gaze")
        comp = ["python", os.path.join(HERE, "compose_3panel.py"),
                "--scene_video", scene_video, "--scene_start_ns", scene_start_ns,
                "--reference_image", ref_img,
                "--screen_video", screen_video, "--start_video_ns", start_video_ns,
                "--gaze_csv", cal_out, "--scene_gaze_csv", scene_gaze,
                "--out_video", video_out, "--height", a.height]
        if a.cursor_overlay:
            comp += ["--cursor_csv", aligned]   # cyan ring = corrected cursor
        if a.gaze_shift_s:
            comp += ["--gaze_shift_s", a.gaze_shift_s]
        if a.start_s is not None:
            comp += ["--start_s", a.start_s]
        if a.duration_s is not None:
            comp += ["--duration_s", a.duration_s]
        run(comp)

    print(f"\nDONE {P}:  aligned {os.path.basename(aligned)}"
          f"{'' if a.no_video else '  |  video '+os.path.basename(video_out)}")


if __name__ == "__main__":
    main()
