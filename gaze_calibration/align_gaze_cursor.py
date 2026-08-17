#!/usr/bin/env python3
"""
Align cursor (steering-experiment JSON) with gaze (dynamic-rim CSV) on a single
timeline, and write one row per gaze sample with both positions.

WHY THIS IS NEEDED
------------------
The two streams live on two different clocks:
  * Gaze   -> Neon clock (absolute nanoseconds, in the gaze CSV)
  * Cursor -> your Mac's clock (unix milliseconds, in the experiment JSON)
Those clocks differ by a per-session offset (the Neon ran ~340 ms ahead in the
test-sync session). This script shifts the cursor onto the Neon timeline using
that offset, then interpolates the cursor to each gaze sample's time.

    neon_time = mac_time + clock_offset_s        # Neon minus Mac

Measure clock_offset_s per session (scene-camera method, or the Real-Time API).
It is NOT constant across recordings -- pass the right value each time.

COORDINATE SYSTEMS (they differ; all are emitted so you can pick)
-----------------------------------------------------------------
  gaze_transf_x/y_px : gaze in screen-video pixels (the perspective-mapped
                       screen coordinates from dynamic-rim). Native gaze space.
  gaze_ref_x/y_px    : gaze in reference-image pixels (raw RIM output).
  cursor_traj_x/y    : cursor normalized along the tunnel (0..1), from the
                       experiment's 'trajectory'. Resolution-independent.
  cursor_screen_x/y  : cursor in the experiment's raw 'screenTrajectory'
                       (macOS global points; may be negative on a secondary
                       display). NOT the same frame as gaze_transf.
  gaze_transf_x/y_norm : gaze_transf divided by screen size -> 0..1, for a
                       rough shared scale with cursor_traj.

  cursor_screenvid_x/y_px : cursor mapped into the SAME frame as gaze_transf
                       (screen-video pixels). Only emitted when you pass
                       --cursor_scale/--cursor_bx/--cursor_by. THIS is the
                       column to compare directly against gaze_transf_*_px.

DERIVING THE CURSOR CALIBRATION (once per display setup)
-------------------------------------------------------
screenTrajectory (macOS points) -> screen-video pixels is a uniform-scale
affine:   px = scale * os_point + b   (no rotation; square pixels).
Find it from TWO points whose position you know in both frames:
  * green start circle : os = cursor's screenTrajectory at a trial start;
                         px = its pixel centroid in a screen-video frame.
  * red target circle  : os = cursor's screenTrajectory at a completed-trial
                         end; px = its pixel centroid.
Then (uniform scale from the x-separated pair):
  scale = (px2_x - px1_x) / (os2_x - os1_x)
  bx    = px1_x - scale * os1_x
  by    = px1_y - scale * os1_y
Validate by overlaying the mapped cursor on a video frame -- it should sit on
the on-screen cursor. For the test-sync session this gave scale=2.002 (Retina
2x), bx=3833, by=1610. These stay valid as long as the display arrangement and
scaling don't change.

Cursor is interpolated only WITHIN each trial's time span; gaze samples that
fall between trials get NaN cursor (no fabricated data across gaps).

USAGE
-----
    python3 align_gaze_cursor.py \
        --experiment_json "steering_experiment_...json" \
        --gaze_csv        "gaze bug fixed.csv" \
        --clock_offset_s  0.340 \
        --output          "aligned_gaze_cursor.csv"

--gaze_csv accepts either a dynamic-rim output CSV (has transf columns) or a
plain RIM gaze.csv (reference columns only; transf columns come out empty).
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd


def load_cursor(path):
    """Return a long DataFrame of cursor samples across all trials.

    Backward compatible: newer logs add per-sample 'speeds', 'constrained?'
    and 'difficulty', plus richer per-trial 'condition' (transition_point,
    tunnelType). Absent fields come back as NaN/empty.
    """
    import json as _json
    d = json.load(open(path))
    rows = []
    for tr in d.get("trialData", []):
        ts = tr.get("timestamps", [])
        n = len(ts)
        traj = tr.get("trajectory") or [{}] * n
        strj = tr.get("screenTrajectory") or [{}] * n
        spd = tr.get("speeds") or [np.nan] * n
        con = tr.get("constrained?") or [""] * n
        dif = tr.get("difficulty") or [None] * n
        cond = tr.get("condition") or {}
        tp_ = cond.get("transition_point") or {}
        meta = {
            "trial_id": tr.get("trial_id"),
            "condition": cond.get("description", ""),
            "tunnelType": cond.get("tunnelType", ""),
            "transition_taskX": tp_.get("taskX", np.nan),
            "transition_screenX": tp_.get("screenX", np.nan),
            "condition_json": _json.dumps(cond, separators=(",", ":")),
        }
        for i, t in enumerate(ts):
            tp = traj[i] if i < len(traj) else {}
            sp = strj[i] if i < len(strj) else {}
            di = dif[i] if i < len(dif) else None
            di = di or {}
            row = {
                "mac_ms": t,
                "cursor_traj_x": tp.get("x", np.nan),
                "cursor_traj_y": tp.get("y", np.nan),
                "cursor_screen_x": sp.get("x", np.nan),
                "cursor_screen_y": sp.get("y", np.nan),
                "cursor_speed": spd[i] if i < len(spd) else np.nan,
                "constrained": con[i] if i < len(con) else "",
                "difficulty_tunnelWidth": di.get("tunnelWidth", np.nan),
                "difficulty_curvature": di.get("curvature", np.nan),
                "difficulty_curvatureChangeRate": di.get("curvatureChangeRate", np.nan),
            }
            row.update(meta)
            rows.append(row)
    c = pd.DataFrame(rows).sort_values("mac_ms").reset_index(drop=True)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_json", required=True)
    ap.add_argument("--gaze_csv", required=True)
    ap.add_argument(
        "--clock_offset_s",
        type=float,
        default=0.0,
        help="Neon minus Mac clock offset in seconds (measure per session; "
        "e.g. 0.340 for the test-sync recording). Default 0.",
    )
    ap.add_argument("--screen_width", type=float, default=3840)
    ap.add_argument("--screen_height", type=float, default=2160)
    # --- optional: put cursor in the SAME frame as gaze (screen-video pixels) ---
    # screenTrajectory (macOS points) -> screen-video pixels is a uniform-scale
    # affine:  px = scale * os_point + b   (scale = display's pixels-per-point,
    # e.g. 2.0 for Retina 2x). Derive scale/bx/by from two known correspondences
    # (an OS point whose screen-video pixel you know), e.g. the green start and
    # red target circles. For the test-sync session the validated values were
    # scale=2.002, bx=3833, by=1610 (see the README). Leave unset to skip.
    ap.add_argument("--cursor_scale", type=float, default=None,
                    help="pixels per OS point (e.g. 2.0 for Retina 2x)")
    ap.add_argument("--cursor_bx", type=float, default=None,
                    help="x offset: px_x = scale*screenTraj_x + bx")
    ap.add_argument("--cursor_by", type=float, default=None,
                    help="y offset: px_y = scale*screenTraj_y + by")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # --- gaze (Neon clock) ---
    g = pd.read_csv(args.gaze_csv)
    if "timestamp [ns]" not in g.columns:
        sys.exit("ERROR: gaze CSV has no 'timestamp [ns]' column.")
    g = g.sort_values("timestamp [ns]").reset_index(drop=True)
    gaze_neon_s = g["timestamp [ns]"].astype(np.float64).to_numpy() / 1e9

    out = pd.DataFrame()
    out["neon_timestamp_ns"] = g["timestamp [ns]"].astype(np.int64)
    out["neon_time_s"] = gaze_neon_s
    # equivalent Mac time (for cross-referencing the experiment log)
    out["mac_time_s"] = gaze_neon_s - args.clock_offset_s

    def col(name):
        return g[name].to_numpy() if name in g.columns else np.full(len(g), np.nan)

    out["gaze_ref_x_px"] = col("gaze position in reference image x [px]")
    out["gaze_ref_y_px"] = col("gaze position in reference image y [px]")
    out["gaze_transf_x_px"] = col("gaze position transf x [px]")
    out["gaze_transf_y_px"] = col("gaze position transf y [px]")
    out["gaze_transf_x_norm"] = out["gaze_transf_x_px"] / args.screen_width
    out["gaze_transf_y_norm"] = out["gaze_transf_y_px"] / args.screen_height

    # --- cursor (Mac clock -> Neon clock) ---
    c = load_cursor(args.experiment_json)
    c["neon_s"] = c["mac_ms"] / 1000.0 + args.clock_offset_s

    # interpolate cursor to each gaze time, but only WITHIN a trial
    for f in [
        "cursor_traj_x", "cursor_traj_y", "cursor_screen_x", "cursor_screen_y",
        "cursor_speed", "cursor_nearest_mac_ms", "cursor_interp_gap_ms",
        "difficulty_tunnelWidth", "difficulty_curvature",
        "difficulty_curvatureChangeRate", "transition_taskX", "transition_screenX",
    ]:
        out[f] = np.nan
    out["trial_id"] = np.nan
    out["condition"] = ""
    out["constrained"] = ""
    out["tunnelType"] = ""
    out["condition_json"] = ""
    out["in_trial"] = False

    for tid, grp in c.groupby("trial_id"):
        grp = grp.sort_values("neon_s")
        t = grp["neon_s"].to_numpy()
        if len(t) < 2:
            continue
        lo, hi = t[0], t[-1]
        m = (gaze_neon_s >= lo) & (gaze_neon_s <= hi)
        if not m.any():
            continue
        idx = np.where(m)[0]
        gt = gaze_neon_s[idx]
        # continuous -> interpolate
        for f in ["cursor_traj_x", "cursor_traj_y", "cursor_screen_x",
                  "cursor_screen_y", "cursor_speed"]:
            out.loc[idx, f] = np.interp(gt, t, grp[f].to_numpy())
        # nearest logged cursor sample: Mac timestamp + interpolation gap
        mac_ms = grp["mac_ms"].to_numpy()
        j = np.clip(np.searchsorted(t, gt), 1, len(t) - 1)
        pick = np.where((gt - t[j - 1]) <= (t[j] - gt), j - 1, j)
        out.loc[idx, "cursor_nearest_mac_ms"] = mac_ms[pick]
        out.loc[idx, "cursor_interp_gap_ms"] = (gt - t[pick]) * 1000.0
        # per-sample categorical/step labels -> use the NEAREST sample (never
        # interpolate constrained<->unconstrained or a stepped difficulty).
        for f in ["constrained", "difficulty_tunnelWidth", "difficulty_curvature",
                  "difficulty_curvatureChangeRate"]:
            out.loc[idx, f] = grp[f].to_numpy()[pick]
        # per-trial condition metadata (constant within trial)
        out.loc[idx, "trial_id"] = tid
        for f in ["condition", "tunnelType", "transition_taskX",
                  "transition_screenX", "condition_json"]:
            out.loc[idx, f] = grp[f].iloc[0]
        out.loc[idx, "in_trial"] = True

    # --- optional: cursor in screen-video pixels (same frame as gaze_transf) ---
    if None not in (args.cursor_scale, args.cursor_bx, args.cursor_by):
        out["cursor_screenvid_x_px"] = (
            args.cursor_scale * out["cursor_screen_x"] + args.cursor_bx
        )
        out["cursor_screenvid_y_px"] = (
            args.cursor_scale * out["cursor_screen_y"] + args.cursor_by
        )
        out["cursor_screenvid_x_norm"] = out["cursor_screenvid_x_px"] / args.screen_width
        out["cursor_screenvid_y_norm"] = out["cursor_screenvid_y_px"] / args.screen_height

    out.to_csv(args.output, index=False)

    n = len(out)
    n_in = int(out["in_trial"].sum())
    print(f"wrote {args.output}")
    print(f"  {n} gaze samples; {n_in} ({100*n_in/n:.0f}%) fall inside a trial "
          f"and got a cursor position")
    print(f"  clock offset applied: {args.clock_offset_s:+.3f} s (Neon - Mac)")
    if n_in:
        # quick sanity: correlation of gaze vs cursor horizontal, in-trial
        sub = out[out["in_trial"]]
        gx = sub["gaze_transf_x_norm"].to_numpy()
        cx = sub["cursor_traj_x"].to_numpy()
        ok = np.isfinite(gx) & np.isfinite(cx)
        if ok.sum() > 10 and np.std(gx[ok]) > 0 and np.std(cx[ok]) > 0:
            r = np.corrcoef(gx[ok], cx[ok])[0, 1]
            print(f"  sanity: gaze-x vs cursor-x correlation (in-trial) = {r:.3f}")


if __name__ == "__main__":
    main()
