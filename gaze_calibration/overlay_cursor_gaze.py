#!/usr/bin/env python3
"""
Overlay the LOGGED cursor (blue) -- and optionally gaze (red) -- onto the
screen recording, time-aligned, and write a real-time video.

The cursor comes from the experiment's screenTrajectory (macOS points) mapped
into screen-video pixels with a uniform-scale affine (px = scale*os + b; derive
it with the two-landmark method, see align_gaze_cursor.py). Gaze comes from a
dynamic-rim CSV (transf screen pixels). Both are placed at the right moment
using the on-screen-clock start time + the per-session Neon-Mac clock offset.

Output plays at true real time (frames sampled on a 1/30 s grid), with a short
fading trail so movement is visible.

USAGE
-----
    python3 overlay_cursor_gaze.py \
        --screen_video "testsync_screen.mov" \
        --experiment_json "steering_experiment_...json" \
        --screen_start_wallclock "2026-08-10 17:18:46.697" \
        --clock_offset_s 0.340 \
        --cursor_scale 2.002 --cursor_bx 3833 --cursor_by 1610 \
        --gaze_csv "gaze bug fixed.csv" \
        --out_video "cursor_overlay.mp4" \
        --out_width 1280

--gaze_csv is optional (omit to draw only the blue cursor).
--out_width downsamples for speed; omit for full resolution.
"""
import argparse
import datetime
import json
import sys

import av
import numpy as np
from PIL import Image, ImageDraw


def wallclock_to_unix_s(s):
    for f in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f",
              "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.datetime.strptime(s, f).timestamp()
        except ValueError:
            pass
    sys.exit(f"could not parse wallclock {s!r}")


def load_cursor_px(path, scale, bx, by):
    """Per-trial arrays of (mac_time_s, px_x, px_y)."""
    d = json.load(open(path))
    trials = []
    for tr in d.get("trialData", []):
        ts = np.array(tr["timestamps"], float) / 1000.0
        st = tr.get("screenTrajectory") or []
        if len(st) != len(ts) or len(ts) < 2:
            continue
        ox = np.array([p["x"] for p in st], float)
        oy = np.array([p["y"] for p in st], float)
        trials.append((ts, scale * ox + bx, scale * oy + by))
    return trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen_video", required=True)
    ap.add_argument("--experiment_json", required=True)
    ap.add_argument("--screen_start_wallclock", required=True)
    ap.add_argument("--clock_offset_s", type=float, default=0.0,
                    help="Neon minus Mac, seconds (per session)")
    ap.add_argument("--cursor_scale", type=float, required=True)
    ap.add_argument("--cursor_bx", type=float, required=True)
    ap.add_argument("--cursor_by", type=float, required=True)
    ap.add_argument("--gaze_csv", default=None)
    ap.add_argument("--out_video", required=True)
    ap.add_argument("--out_width", type=int, default=0,
                    help="downscale output to this width (0 = full res)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--trail", type=int, default=12,
                    help="number of trailing points to draw")
    args = ap.parse_args()

    start_unix = wallclock_to_unix_s(args.screen_start_wallclock)  # Mac time of frame 0
    cursor = load_cursor_px(args.experiment_json, args.cursor_scale,
                            args.cursor_bx, args.cursor_by)

    # gaze (optional): Neon time -> transf px
    gaze = None
    if args.gaze_csv:
        import pandas as pd
        g = pd.read_csv(args.gaze_csv).sort_values("timestamp [ns]")
        gt = g["timestamp [ns]"].astype(np.float64).to_numpy() / 1e9  # Neon s
        gx = g["gaze position transf x [px]"].to_numpy()
        gy = g["gaze position transf y [px]"].to_numpy()
        gaze = (gt, gx, gy)

    # video geometry (cheap: read stream header only)
    probe = av.open(args.screen_video)
    vs = probe.streams.video[0]
    tb = float(vs.time_base)
    W0, H0 = vs.width, vs.height
    probe.close()

    if args.out_width and args.out_width < W0:
        sc = args.out_width / W0
    else:
        sc = 1.0
    OW = int(round(W0 * sc)); OW -= OW % 2
    OH = int(round(H0 * sc)); OH -= OH % 2

    # real-time grid over the experiment span (Mac -> video-relative)
    cur_lo = min(t[0][0] for t in cursor)
    cur_hi = max(t[0][-1] for t in cursor)
    vrel_lo = max(0.0, cur_lo - start_unix)
    vrel_hi = cur_hi - start_unix
    step = 1.0 / args.fps
    grid = np.arange(vrel_lo, vrel_hi, step)
    print(f"screen video {W0}x{H0}; rendering {len(grid)} frames at {args.fps}fps "
          f"-> {len(grid)/args.fps:.1f}s, out {OW}x{OH}")

    out = av.open(args.out_video, "w")
    ostream = out.add_stream("libx264", rate=args.fps)
    ostream.width, ostream.height = OW, OH
    ostream.pix_fmt = "yuv420p"

    def cursor_at(mac_t):
        for ts, px, py in cursor:
            if ts[0] <= mac_t <= ts[-1]:
                return np.interp(mac_t, ts, px), np.interp(mac_t, ts, py)
        return None

    def render_tick(gt_rel, base_img, trail):
        """Draw overlays for one grid tick onto a copy of base_img (already OW×OH)."""
        img = base_img.copy()
        dr = ImageDraw.Draw(img, "RGBA")
        mac_t = start_unix + gt_rel
        neon_t = mac_t + args.clock_offset_s
        if gaze is not None:
            gt, gx, gy = gaze
            if gt[0] <= neon_t <= gt[-1]:
                x = np.interp(neon_t, gt, gx) * sc
                y = np.interp(neon_t, gt, gy) * sc
                r = max(4, int(14 * sc))
                dr.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0, 255),
                           width=max(2, int(4 * sc)))
        c = cursor_at(mac_t)
        if c is not None:
            x, y = c[0] * sc, c[1] * sc
            trail.append((x, y))
            del trail[:-args.trail]
            for k, (tx, ty) in enumerate(trail):
                a = int(60 + 180 * (k + 1) / len(trail))
                rr = max(2, int(8 * sc))
                dr.ellipse([tx - rr, ty - rr, tx + rr, ty + rr], fill=(0, 80, 255, a))
            r = max(4, int(16 * sc))
            dr.ellipse([x - r, y - r, x + r, y + r], outline=(0, 80, 255, 255),
                       width=max(2, int(5 * sc)))
        else:
            trail.clear()
        return img

    # STREAM screen frames once; keep only the latest downscaled image in memory.
    # For each grid tick, use the most recent screen frame at/just before it.
    container = av.open(args.screen_video)
    vs = container.streams.video[0]
    gi = 0
    n_grid = len(grid)
    current = None  # latest downscaled PIL image
    trail = []
    written = 0
    for pkt in container.demux(vs):
        if gi >= n_grid:
            break
        for fr in pkt.decode():
            if fr.pts is None:
                continue
            t = fr.pts * tb
            if t < vrel_lo - step:
                continue  # skip frames well before the window (don't decode-to-image)
            # emit any grid ticks that this frame has now passed
            while gi < n_grid and grid[gi] < t:
                base = current if current is not None else (
                    fr.to_image().convert("RGB").resize((OW, OH)) if sc != 1.0
                    else fr.to_image().convert("RGB"))
                img = render_tick(grid[gi], base, trail)
                for p in ostream.encode(av.VideoFrame.from_image(img)):
                    out.mux(p)
                written += 1
                gi += 1
            # update current (downscaled)
            im = fr.to_image().convert("RGB")
            current = im.resize((OW, OH)) if sc != 1.0 else im
            if t > vrel_hi + step:
                gi = n_grid
                break
    # flush remaining ticks with last frame
    while gi < n_grid and current is not None:
        img = render_tick(grid[gi], current, trail)
        for p in ostream.encode(av.VideoFrame.from_image(img)):
            out.mux(p)
        written += 1
        gi += 1
    for p in ostream.encode():
        out.mux(p)
    out.close()
    print(f"  encoded {written} frames")
    print(f"wrote {args.out_video}")
    if gaze is not None:
        print("  blue = logged cursor, red = gaze")
    else:
        print("  blue = logged cursor")


if __name__ == "__main__":
    main()
