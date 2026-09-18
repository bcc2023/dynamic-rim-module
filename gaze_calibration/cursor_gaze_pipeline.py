#!/usr/bin/env python3
"""
Post-process step to run AFTER pl-dynamic-rim.

Takes the dynamic-rim gaze CSV + the steering-experiment JSON and produces, in
one command:
  1. a same-timeframe CSV  -- gaze and cursor at each timestamp, both in
     screen-video pixels (plus raw/normalized columns and trial metadata);
  2. an overlay video      -- the logged cursor (blue, with a trail) and gaze
     (red) drawn on the screen recording, time-aligned, at true real time.

Two per-session calibrations are required and must be measured for each session
(they are NOT constant across recordings):
  --clock_offset_s : Neon minus Mac clock, in seconds (e.g. 0.340). Aligns the
                     cursor (Mac clock) to gaze (Neon clock) in TIME.
  --cursor_scale / --cursor_bx / --cursor_by : screenTrajectory (macOS points)
                     -> screen-video pixels, a uniform-scale affine
                     px = scale*os + b. Aligns the cursor to gaze in SPACE.
                     Derive with the two-landmark method (see README /
                     align_gaze_cursor.py). Test-sync values: 2.002 / 3833 / 1610.

The clock offset is also written on frame 0 by the on-screen clock method
(--screen_start_wallclock is the same value you passed to pl-dynamic-rim).

USAGE
-----
    python3 cursor_gaze_pipeline.py \
        --gaze_csv "gaze bug fixed.csv" \
        --experiment_json "steering_experiment_...json" \
        --screen_video "testsync_screen.mov" \
        --screen_start_wallclock "2026-08-10 17:18:46.697" \
        --clock_offset_s 0.340 \
        --cursor_scale 2.002 --cursor_bx 3833 --cursor_by 1610 \
        --out_csv   "aligned_gaze_cursor.csv" \
        --out_video "cursor_gaze_overlay.mp4" \
        --out_width 1000

Omit --out_video to produce only the CSV. Omit --out_width for full-res video.
"""
import argparse
import datetime
import json
import sys

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- helpers ---
def parse_wallclock(s):
    for f in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f",
              "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.datetime.strptime(s, f)
        except ValueError:
            pass
    sys.exit(f"ERROR: could not parse wallclock {s!r} (use 'YYYY-MM-DD HH:MM:SS.fff')")


def load_cursor_trials(path):
    """Per-trial dict with mac-second timestamps and per-sample arrays.

    Backward compatible: fields absent in older logs come back as NaN/empty.
    Newer logs add per-sample 'constrained?' and 'difficulty', per-sample
    'speeds', and richer 'condition' (transition_point, tunnelType, ...).
    """
    d = json.load(open(path))
    trials = []
    for tr in d.get("trialData", []):
        ts = np.array(tr.get("timestamps", []), float) / 1000.0
        n = len(ts)
        if n < 2:
            continue
        traj = tr.get("trajectory") or [{}] * n
        strj = tr.get("screenTrajectory") or [{}] * n
        cond = tr.get("condition") or {}
        # per-sample difficulty dicts (or None when unconstrained)
        dif = tr.get("difficulty") or [None] * n
        dif = (dif + [None] * n)[:n]
        con = tr.get("constrained?") or [""] * n
        con = (list(con) + [""] * n)[:n]
        spd = tr.get("speeds") or [np.nan] * n
        spd = (list(spd) + [np.nan] * n)[:n]
        tp = cond.get("transition_point") or {}
        trials.append({
            "mac_s": ts,
            "trial_id": tr.get("trial_id"),
            "condition": cond.get("description", ""),
            "traj_x": np.array([p.get("x", np.nan) for p in traj], float),
            "traj_y": np.array([p.get("y", np.nan) for p in traj], float),
            "scr_x": np.array([p.get("x", np.nan) for p in strj], float),
            "scr_y": np.array([p.get("y", np.nan) for p in strj], float),
            "speed": np.array(spd, float),
            # per-sample task labels (categorical -> nearest, not interpolated)
            "constrained": np.array(con, dtype=object),
            "diff_width": np.array([(x or {}).get("tunnelWidth", np.nan) for x in dif], float),
            "diff_curv": np.array([(x or {}).get("curvature", np.nan) for x in dif], float),
            "diff_ccr": np.array([(x or {}).get("curvatureChangeRate", np.nan) for x in dif], float),
            # per-trial condition metadata
            "tunnelType": cond.get("tunnelType", ""),
            "transition_taskX": tp.get("taskX", np.nan),
            "transition_screenX": tp.get("screenX", np.nan),
            "condition_json": json.dumps(cond, separators=(",", ":")),
        })
    return trials


# ------------------------------------------------------------ step 1: CSV ---
def build_aligned(args):
    g = pd.read_csv(args.gaze_csv)
    if "timestamp [ns]" not in g.columns:
        sys.exit("ERROR: gaze CSV has no 'timestamp [ns]' column.")
    g = g.sort_values("timestamp [ns]").reset_index(drop=True)
    gaze_neon_s = g["timestamp [ns]"].astype(np.float64).to_numpy() / 1e9

    out = pd.DataFrame()
    out["neon_timestamp_ns"] = g["timestamp [ns]"].astype(np.int64)
    out["neon_time_s"] = gaze_neon_s
    out["mac_time_s"] = gaze_neon_s - args.clock_offset_s
    # advance the gaze vs the cursor by gaze_shift_s (+ = gaze leads). Used ONLY
    # for cursor matching, so the DV mirrors the video's --gaze_shift_s; the row's
    # neon_timestamp_ns (and saccade/blink lookup) stays at the gaze's true time.
    gaze_match_s = gaze_neon_s - args.gaze_shift_s

    def col(n):
        return g[n].to_numpy() if n in g.columns else np.full(len(g), np.nan)

    out["gaze_ref_x_px"] = col("gaze position in reference image x [px]")
    out["gaze_ref_y_px"] = col("gaze position in reference image y [px]")
    out["gaze_transf_x_px"] = col("gaze position transf x [px]")
    out["gaze_transf_y_px"] = col("gaze position transf y [px]")
    out["gaze_transf_x_norm"] = out["gaze_transf_x_px"] / args.screen_width
    out["gaze_transf_y_norm"] = out["gaze_transf_y_px"] / args.screen_height
    # fixation id carried straight from the (timestamp-sorted) gaze CSV so the
    # aligned output always has the full 32-column pilot schema, even on a
    # standalone re-align (no separate merge step needed).
    out["fixation_id"] = g["fixation id"].to_numpy() if "fixation id" in g.columns else np.nan

    for f in ["cursor_traj_x", "cursor_traj_y", "cursor_screen_x", "cursor_screen_y",
              "cursor_speed", "cursor_nearest_mac_ms", "cursor_interp_gap_ms",
              "difficulty_tunnelWidth", "difficulty_curvature",
              "difficulty_curvatureChangeRate", "transition_taskX",
              "transition_screenX"]:
        out[f] = np.nan
    out["trial_id"] = np.nan
    out["condition"] = ""
    out["constrained"] = ""
    out["tunnelType"] = ""
    out["condition_json"] = ""
    out["in_trial"] = False

    for tr in load_cursor_trials(args.experiment_json):
        neon = tr["mac_s"] + args.clock_offset_s   # cursor sample times, on Neon clock
        lo, hi = neon[0], neon[-1]
        m = (gaze_match_s >= lo) & (gaze_match_s <= hi)
        if not m.any():
            continue
        idx = np.where(m)[0]
        gt = gaze_match_s[idx]
        # continuous quantities: linearly interpolated onto each gaze timestamp
        out.loc[idx, "cursor_traj_x"] = np.interp(gt, neon, tr["traj_x"])
        out.loc[idx, "cursor_traj_y"] = np.interp(gt, neon, tr["traj_y"])
        out.loc[idx, "cursor_screen_x"] = np.interp(gt, neon, tr["scr_x"])
        out.loc[idx, "cursor_screen_y"] = np.interp(gt, neon, tr["scr_y"])
        out.loc[idx, "cursor_speed"] = np.interp(gt, neon, tr["speed"])
        # nearest ACTUAL logged cursor sample: its Mac-clock timestamp + time gap
        j = np.clip(np.searchsorted(neon, gt), 1, len(neon) - 1)
        pick = np.where((gt - neon[j - 1]) <= (neon[j] - gt), j - 1, j)
        out.loc[idx, "cursor_nearest_mac_ms"] = np.round(tr["mac_s"][pick] * 1000.0)
        out.loc[idx, "cursor_interp_gap_ms"] = (gt - neon[pick]) * 1000.0
        # per-sample task labels: use the NEAREST sample (categorical / step
        # values -- interpolating "constrained"<->"unconstrained" or a tunnel
        # width that jumps at a transition would be meaningless).
        out.loc[idx, "constrained"] = tr["constrained"][pick]
        out.loc[idx, "difficulty_tunnelWidth"] = tr["diff_width"][pick]
        out.loc[idx, "difficulty_curvature"] = tr["diff_curv"][pick]
        out.loc[idx, "difficulty_curvatureChangeRate"] = tr["diff_ccr"][pick]
        # per-trial condition metadata (constant within the trial)
        out.loc[idx, "trial_id"] = tr["trial_id"]
        out.loc[idx, "condition"] = tr["condition"]
        out.loc[idx, "tunnelType"] = tr["tunnelType"]
        out.loc[idx, "transition_taskX"] = tr["transition_taskX"]
        out.loc[idx, "transition_screenX"] = tr["transition_screenX"]
        out.loc[idx, "condition_json"] = tr["condition_json"]
        out.loc[idx, "in_trial"] = True

    # cursor in the SAME frame as gaze_transf (screen-video pixels)
    out["cursor_screenvid_x_px"] = args.cursor_scale * out["cursor_screen_x"] + args.cursor_bx
    out["cursor_screenvid_y_px"] = args.cursor_scale * out["cursor_screen_y"] + args.cursor_by
    out["cursor_screenvid_x_norm"] = out["cursor_screenvid_x_px"] / args.screen_width
    out["cursor_screenvid_y_norm"] = out["cursor_screenvid_y_px"] / args.screen_height

    out.to_csv(args.out_csv, index=False)
    n, n_in = len(out), int(out["in_trial"].sum())
    print(f"[csv] wrote {args.out_csv}: {n} gaze samples, {n_in} ({100*n_in/n:.0f}%) with cursor")
    sub = out[out["in_trial"]]
    if len(sub) > 10:
        dx = (sub["gaze_transf_x_px"] - sub["cursor_screenvid_x_px"])
        dy = (sub["gaze_transf_y_px"] - sub["cursor_screenvid_y_px"])
        dist = np.sqrt(dx**2 + dy**2)
        ok = np.isfinite(dist)
        if ok.any():
            print(f"[csv] gaze-cursor distance (same frame): median {np.median(dist[ok]):.0f}px, "
                  f"mean dx {dx[ok].mean():+.0f}px (+=gaze ahead)")


# --------------------------------------------------------- step 2: video ---
def render_overlay(args):
    import av
    from PIL import ImageDraw

    start_unix = parse_wallclock(args.screen_start_wallclock).timestamp()

    # cursor -> screen-video px, per trial
    cursor = []
    for tr in load_cursor_trials(args.experiment_json):
        cursor.append((tr["mac_s"],
                       args.cursor_scale * tr["scr_x"] + args.cursor_bx,
                       args.cursor_scale * tr["scr_y"] + args.cursor_by))
    if not cursor:
        print("[video] no cursor trials; skipping video")
        return

    gaze = None
    g = pd.read_csv(args.gaze_csv).sort_values("timestamp [ns]")
    if "gaze position transf x [px]" in g.columns:
        gaze = (g["timestamp [ns]"].astype(np.float64).to_numpy() / 1e9,
                g["gaze position transf x [px]"].to_numpy(),
                g["gaze position transf y [px]"].to_numpy())

    probe = av.open(args.screen_video)
    vs = probe.streams.video[0]
    tb = float(vs.time_base); W0, H0 = vs.width, vs.height
    probe.close()
    sc = args.out_width / W0 if (args.out_width and args.out_width < W0) else 1.0
    OW = int(round(W0 * sc)); OW -= OW % 2
    OH = int(round(H0 * sc)); OH -= OH % 2

    cur_lo = min(t[0][0] for t in cursor)
    cur_hi = max(t[0][-1] for t in cursor)
    vrel_lo = max(0.0, cur_lo - start_unix)
    vrel_hi = cur_hi - start_unix
    step = 1.0 / args.fps
    grid = np.arange(vrel_lo, vrel_hi, step)
    print(f"[video] {W0}x{H0} -> {OW}x{OH}, {len(grid)} frames @ {args.fps}fps "
          f"({len(grid)/args.fps:.1f}s)")

    out = av.open(args.out_video, "w")
    ostream = out.add_stream("libx264", rate=args.fps)
    ostream.width, ostream.height = OW, OH
    ostream.pix_fmt = "yuv420p"

    def cursor_at(mac_t):
        for ts, px, py in cursor:
            if ts[0] <= mac_t <= ts[-1]:
                return np.interp(mac_t, ts, px), np.interp(mac_t, ts, py)
        return None

    def draw(gt_rel, base, trail):
        img = base.copy()
        dr = ImageDraw.Draw(img, "RGBA")
        mac_t = start_unix + gt_rel
        neon_t = mac_t + args.clock_offset_s
        if gaze is not None:
            gt, gx, gy = gaze
            if gt[0] <= neon_t <= gt[-1]:
                x = np.interp(neon_t, gt, gx) * sc
                y = np.interp(neon_t, gt, gy) * sc
                r = max(4, int(14 * sc))
                dr.ellipse([x-r, y-r, x+r, y+r], outline=(255, 0, 0, 255),
                           width=max(2, int(4 * sc)))
        c = cursor_at(mac_t)
        if c is not None:
            x, y = c[0] * sc, c[1] * sc
            trail.append((x, y)); del trail[:-args.trail]
            for k, (tx, ty) in enumerate(trail):
                a = int(60 + 180 * (k + 1) / len(trail))
                rr = max(2, int(8 * sc))
                dr.ellipse([tx-rr, ty-rr, tx+rr, ty+rr], fill=(0, 80, 255, a))
            r = max(4, int(16 * sc))
            dr.ellipse([x-r, y-r, x+r, y+r], outline=(0, 80, 255, 255),
                       width=max(2, int(5 * sc)))
        else:
            trail.clear()
        return img

    container = av.open(args.screen_video)
    vs = container.streams.video[0]
    gi, n_grid = 0, len(grid)
    current, trail, written = None, [], 0
    for pkt in container.demux(vs):
        if gi >= n_grid:
            break
        for fr in pkt.decode():
            if fr.pts is None:
                continue
            t = fr.pts * tb
            if t < vrel_lo - step:
                continue
            while gi < n_grid and grid[gi] < t:
                base = current
                if base is None:
                    im = fr.to_image().convert("RGB")
                    base = im.resize((OW, OH)) if sc != 1.0 else im
                for p in ostream.encode(av.VideoFrame.from_image(draw(grid[gi], base, trail))):
                    out.mux(p)
                written += 1; gi += 1
            im = fr.to_image().convert("RGB")
            current = im.resize((OW, OH)) if sc != 1.0 else im
            if t > vrel_hi + step:
                gi = n_grid
                break
    while gi < n_grid and current is not None:
        for p in ostream.encode(av.VideoFrame.from_image(draw(grid[gi], current, trail))):
            out.mux(p)
        written += 1; gi += 1
    for p in ostream.encode():
        out.mux(p)
    out.close()
    lg = "blue=cursor, red=gaze" if gaze is not None else "blue=cursor"
    print(f"[video] wrote {args.out_video}: {written} frames ({lg})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaze_csv", required=True, help="dynamic-rim output CSV")
    ap.add_argument("--experiment_json", required=True)
    ap.add_argument("--cursor_map", default=None,
                    help="JSON from fit-cursor-map with cursor_scale/bx/by, "
                    "screen_width/height, clock_offset_s. Fills any of the "
                    "individual options below that you don't pass explicitly.")
    ap.add_argument("--clock_offset_s", type=float, default=None,
                    help="Neon minus Mac, seconds (per session)")
    ap.add_argument("--gaze_shift_s", type=float, default=0.0,
                    help="advance the gaze vs the cursor (+ = gaze leads); mirrors compose's --gaze_shift_s so the DV matches the video")
    ap.add_argument("--cursor_scale", type=float, default=None)
    ap.add_argument("--cursor_bx", type=float, default=None)
    ap.add_argument("--cursor_by", type=float, default=None)
    ap.add_argument("--screen_width", type=float, default=None)
    ap.add_argument("--screen_height", type=float, default=None)
    ap.add_argument("--out_csv", required=True)
    # video (optional)
    ap.add_argument("--out_video", default=None)
    ap.add_argument("--screen_video", default=None)
    ap.add_argument("--screen_start_wallclock", default=None)
    ap.add_argument("--out_width", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--trail", type=int, default=1,
                    help="number of cursor points to draw (1 = just the dot, "
                    "no trail; higher = longer fading tail)")
    args = ap.parse_args()

    # --cursor_map fills any spatial/clock option not passed explicitly.
    if args.cursor_map:
        m = json.load(open(args.cursor_map))
        for k in ("cursor_scale", "cursor_bx", "cursor_by",
                  "screen_width", "screen_height", "clock_offset_s"):
            if getattr(args, k) is None and m.get(k) is not None:
                setattr(args, k, float(m[k]))
    # defaults / required checks
    if args.screen_width is None:
        args.screen_width = 3840
    if args.screen_height is None:
        args.screen_height = 2160
    if args.clock_offset_s is None:
        args.clock_offset_s = 0.0
    _need = [n for n in ("cursor_scale", "cursor_bx", "cursor_by")
             if getattr(args, n) is None]
    if _need:
        sys.exit("ERROR: need " + ", ".join("--" + n for n in _need)
                 + " (or pass --cursor_map)")

    build_aligned(args)

    if args.out_video:
        missing = [n for n in ("screen_video", "screen_start_wallclock")
                   if getattr(args, n) is None]
        if missing:
            sys.exit(f"ERROR: --out_video needs also: {', '.join('--'+m for m in missing)}")
        render_overlay(args)


if __name__ == "__main__":
    main()
