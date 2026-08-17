#!/usr/bin/env python3
"""
Render a cursor+gaze overlay DIRECTLY FROM an aligned CSV (the output of
cursor_gaze_pipeline.py / align_gaze_cursor.py), to verify that CSV.

Unlike cursor_gaze_pipeline.py (which re-derives cursor from the experiment
JSON), this reads the aligned CSV's own columns:
    cursor_screenvid_x/y_px  -> blue dot (the cursor, in screen-video pixels)
    gaze_transf_x/y_px       -> red dot  (gaze, in screen-video pixels)
and draws them on the screen recording, time-matched via mac_time_s. If the
blue dot sits on the real cursor and red on the gaze, the aligned CSV is good.

Screen frames are matched to CSV rows by wall-clock: a frame at video-relative
time t is at Mac time  (frame-0 wallclock) + t, matched to the nearest
mac_time_s in the CSV. No clock offset needed here -- the CSV already carries
mac_time_s.

USAGE
-----
    python3 overlay_from_aligned.py \
        --aligned_csv "P152244_aligned.csv" \
        --screen_video "P152244_screen.mov" \
        --screen_start_wallclock "2026-08-11 15:31:52.432" \
        --out_video "P152244_verify_overlay.mp4" \
        --out_width 1280

Optional --start_s / --duration_s render only a window (quick spot-check
instead of the whole recording).
"""
import argparse
import datetime
import sys

import av
import numpy as np
import pandas as pd
from PIL import ImageDraw


def parse_wallclock(s):
    for f in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f",
              "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.datetime.strptime(s, f).timestamp()
        except ValueError:
            pass
    sys.exit(f"could not parse wallclock {s!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_csv", required=True)
    ap.add_argument("--screen_video", required=True)
    ap.add_argument("--screen_start_wallclock", required=True)
    ap.add_argument("--out_video", required=True)
    ap.add_argument("--out_width", type=int, default=1280)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--trail", type=int, default=1)
    ap.add_argument("--start_s", type=float, default=None,
                    help="video-relative start time to render from")
    ap.add_argument("--duration_s", type=float, default=None,
                    help="seconds to render (default: to end)")
    args = ap.parse_args()

    start_unix = parse_wallclock(args.screen_start_wallclock)

    d = pd.read_csv(args.aligned_csv)
    need = ["mac_time_s", "cursor_screenvid_x_px", "cursor_screenvid_y_px",
            "gaze_transf_x_px", "gaze_transf_y_px"]
    miss = [c for c in need if c not in d.columns]
    if miss:
        sys.exit(f"aligned CSV missing columns: {miss}")
    d = d.sort_values("mac_time_s").reset_index(drop=True)
    mac = d["mac_time_s"].to_numpy()
    cvx = d["cursor_screenvid_x_px"].to_numpy()
    cvy = d["cursor_screenvid_y_px"].to_numpy()
    gtx = d["gaze_transf_x_px"].to_numpy()
    gty = d["gaze_transf_y_px"].to_numpy()

    probe = av.open(args.screen_video)
    vs = probe.streams.video[0]
    tb = float(vs.time_base); W0, H0 = vs.width, vs.height
    probe.close()
    sc = args.out_width / W0 if (args.out_width and args.out_width < W0) else 1.0
    OW = int(round(W0 * sc)); OW -= OW % 2
    OH = int(round(H0 * sc)); OH -= OH % 2

    # render window (video-relative)
    lo = mac[0] - start_unix
    hi = mac[-1] - start_unix
    if args.start_s is not None:
        lo = max(lo, args.start_s)
    if args.duration_s is not None:
        hi = min(hi, lo + args.duration_s)
    step = 1.0 / args.fps
    grid = np.arange(max(0.0, lo), hi, step)
    print(f"screen {W0}x{H0} -> {OW}x{OH}; {len(grid)} frames @ {args.fps}fps "
          f"({len(grid)/args.fps:.1f}s)")

    out = av.open(args.out_video, "w")
    ostream = out.add_stream("libx264", rate=args.fps)
    ostream.width, ostream.height = OW, OH
    ostream.pix_fmt = "yuv420p"

    def nearest(mac_t):
        i = int(np.searchsorted(mac, mac_t))
        i = min(max(i, 0), len(mac) - 1)
        if i > 0 and abs(mac[i - 1] - mac_t) < abs(mac[i] - mac_t):
            i -= 1
        # ignore matches too far (>0.1s) -> no data at this instant
        if abs(mac[i] - mac_t) > 0.1:
            return None
        return i

    def draw(gt_rel, base, trail):
        img = base.copy()
        dr = ImageDraw.Draw(img, "RGBA")
        i = nearest(start_unix + gt_rel)
        if i is not None:
            if np.isfinite(gtx[i]):
                x, y = gtx[i] * sc, gty[i] * sc
                r = max(4, int(14 * sc))
                dr.ellipse([x-r, y-r, x+r, y+r], outline=(255, 0, 0, 255),
                           width=max(2, int(4 * sc)))
            if np.isfinite(cvx[i]):
                x, y = cvx[i] * sc, cvy[i] * sc
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
    gi, n = 0, len(grid)
    current, trail, written = None, [], 0
    for pkt in container.demux(vs):
        if gi >= n:
            break
        for fr in pkt.decode():
            if fr.pts is None:
                continue
            t = fr.pts * tb
            if t < grid[0] - step:
                continue
            while gi < n and grid[gi] < t:
                base = current
                if base is None:
                    im = fr.to_image().convert("RGB")
                    base = im.resize((OW, OH)) if sc != 1.0 else im
                for p in ostream.encode(av.VideoFrame.from_image(draw(grid[gi], base, trail))):
                    out.mux(p)
                written += 1; gi += 1
            im = fr.to_image().convert("RGB")
            current = im.resize((OW, OH)) if sc != 1.0 else im
            if t > grid[-1] + step:
                gi = n
                break
    while gi < n and current is not None:
        for p in ostream.encode(av.VideoFrame.from_image(draw(grid[gi], current, trail))):
            out.mux(p)
        written += 1; gi += 1
    for p in ostream.encode():
        out.mux(p)
    out.close()
    print(f"wrote {args.out_video}: {written} frames (blue=cursor, red=gaze, from the aligned CSV)")


if __name__ == "__main__":
    main()
