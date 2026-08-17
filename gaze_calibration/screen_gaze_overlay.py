#!/usr/bin/env python3
"""
Render a screen recording with the RIM-mapped gaze overlaid, and brighten it.

This rebuilds the *screen panel* of what `pl-dynamic-rim` produces, directly from
the mapped-gaze CSV (its `gaze position transf x/y [px]` columns) plus the
original screen recording. Useful when the tool's own 3-panel video render fails
(e.g. produces a 0-second file) even though the gaze CSV is fine.

Sync: a screen frame at video-relative time t corresponds to wall time
    start_video_ns + t   (start_video_ns = Neon recording.begin + screen offset).
Each frame is matched to the nearest mapped-gaze sample within --tol_ms.

Brightness: a simple white-point stretch. --white 200 maps input value 200 -> 255
(everything scaled up and clipped), which lifts a dim/grey capture back to white.

USAGE
    python3 screen_gaze_overlay.py \
        --screen_video   "awareness.mov" \
        --gaze_csv       "gaze2.csv" \
        --start_video_ns 1786578955000000000 \
        --out_video      "awareness_screen_bright.mp4" \
        --white 200 --out_width 1920
"""
import argparse
import av
import numpy as np
import pandas as pd
from PIL import ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen_video", required=True)
    ap.add_argument("--gaze_csv", required=True)
    ap.add_argument("--start_video_ns", required=True,
                    help="wall time (ns) of screen frame 0 = Neon recording.begin + offset")
    ap.add_argument("--out_video", required=True)
    ap.add_argument("--out_width", type=int, default=1920)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--white", type=float, default=148.0,
                    help="input white point for the brightness stretch (lower = brighter). "
                         "These captures top out near ~140; ~148 restores white without "
                         "washing out light-grey detail (128 over-clips).")
    ap.add_argument("--tol_ms", type=float, default=40.0)
    ap.add_argument("--start_s", type=float, default=None)
    ap.add_argument("--duration_s", type=float, default=None)
    args = ap.parse_args()

    start = float(args.start_video_ns)
    g = pd.read_csv(args.gaze_csv)
    ts = g["timestamp [ns]"].to_numpy(float)
    gx = g["gaze position transf x [px]"].to_numpy(float)
    gy = g["gaze position transf y [px]"].to_numpy(float)
    o = np.argsort(ts)
    ts, gx, gy = ts[o], gx[o], gy[o]

    # brightness LUT (white-point stretch), applied to all channels
    lut = list(np.clip(np.arange(256) * (255.0 / args.white), 0, 255).astype("uint8")) * 3

    probe = av.open(args.screen_video)
    vs = probe.streams.video[0]
    tb = float(vs.time_base)
    W0, H0 = vs.width, vs.height
    probe.close()
    sc = args.out_width / W0 if (args.out_width and args.out_width < W0) else 1.0
    OW = int(round(W0 * sc)); OW -= OW % 2
    OH = int(round(H0 * sc)); OH -= OH % 2

    # render grid (video-relative seconds) covering the screen video
    lo = 0.0 if args.start_s is None else args.start_s
    hi = (float(vs.duration * tb) if vs.duration else None)
    if hi is None:
        hi = ts[-1] / 1e9 - start / 1e9 + 1.0
    if args.duration_s is not None:
        hi = lo + args.duration_s
    step = 1.0 / args.fps
    grid = np.arange(lo, hi, step)
    print(f"screen {W0}x{H0} -> {OW}x{OH}; {len(grid)} frames @ {args.fps}fps "
          f"({len(grid)/args.fps:.1f}s); white={args.white}")

    out = av.open(args.out_video, "w")
    ostream = out.add_stream("libx264", rate=args.fps)
    ostream.width, ostream.height = OW, OH
    ostream.pix_fmt = "yuv420p"
    ostream.options = {"crf": "22", "preset": "veryfast"}

    def nearest(wall_ns):
        i = int(np.searchsorted(ts, wall_ns))
        i = min(max(i, 0), len(ts) - 1)
        if i > 0 and abs(ts[i - 1] - wall_ns) < abs(ts[i] - wall_ns):
            i -= 1
        if abs(ts[i] - wall_ns) > args.tol_ms * 1e6:
            return None
        return i

    def draw(gt_rel, base):
        img = base.copy()
        i = nearest(start + gt_rel * 1e9)
        if i is not None and np.isfinite(gx[i]):
            x, y = gx[i] * sc, gy[i] * sc
            r = max(6, int(18 * sc))
            dr = ImageDraw.Draw(img, "RGBA")
            dr.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0, 255),
                       width=max(3, int(5 * sc)))
        return img

    container = av.open(args.screen_video)
    vs = container.streams.video[0]
    if args.start_s:                          # seek so chunks don't decode from 0
        try:
            container.seek(int(max(0.0, lo - 1.0) / tb), stream=vs, backward=True)
        except Exception as e:
            print(f"(seek failed, decoding from start: {e})")
    gi, n = 0, len(grid)
    current, written = None, 0
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
                    base = base.point(lut)
                for p in ostream.encode(av.VideoFrame.from_image(draw(grid[gi], base))):
                    out.mux(p)
                written += 1; gi += 1
            im = fr.to_image().convert("RGB")
            im = im.resize((OW, OH)) if sc != 1.0 else im
            current = im.point(lut)          # brighten once per real frame
            if t > grid[-1] + step:
                gi = n
                break
    while gi < n and current is not None:
        for p in ostream.encode(av.VideoFrame.from_image(draw(grid[gi], current))):
            out.mux(p)
        written += 1; gi += 1
    for p in ostream.encode():
        out.mux(p)
    out.close()
    print(f"wrote {args.out_video}: {written} frames (screen + red gaze circle, brightened)")


if __name__ == "__main__":
    main()
