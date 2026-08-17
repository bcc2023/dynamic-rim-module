#!/usr/bin/env python3
"""
pl-dynamic-rim style 3-panel composite, built by hand and STREAMING both videos
(so it scales to long, dynamic screen recordings):

    [ Neon scene cam + gaze ] [ Reference image + gaze ] [ Screen video + gaze (bright) ]

Gaze dots:
  * Reference & Screen panels  -> from --gaze_csv columns
        'gaze position in reference image x/y [px]'  (reference panel)
        'gaze position transf x/y [px]'              (screen panel)
    Pass a *calibrated* gaze CSV here to show corrected gaze.
  * Neon panel                 -> from --scene_gaze_csv raw scene-camera gaze
        'gaze x/y [px]'   (like pl-dynamic-rim's merged video). Omit to skip it.

Sync (Neon nanosecond clock):
  gaze/reference dots : gaze CSV 'timestamp [ns]'
  screen frame pts t  : wall = --start_video_ns + t
  scene  frame pts t  : wall = --scene_start_ns  + t

USAGE
    python3 compose_3panel.py \
        --scene_video ".../<scene>.mp4"        --scene_start_ns   <neon begin ns> \
        --reference_image ".../reference_image.jpeg" \
        --screen_video ".../screen.mov"        --start_video_ns   <neon begin + offset ns> \
        --gaze_csv ".../<...>_gaze[_calibrated].csv" \
        --scene_gaze_csv ".../<recording>/gaze.csv" \
        --out_video ".../out_3panel.mp4" --height 720 --white 148
    (optional --start_s / --duration_s render just a window for a quick check)
"""
import argparse
import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def label(img, text):
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, 8 * len(text) + 12, 22], fill=(0, 0, 0))
    dr.text((6, 5), text, fill=(255, 255, 255))


def circle(img, x, y, r, w=3):
    ImageDraw.Draw(img).ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=w)


class Streamer:
    """Time-indexed frame stream: .at(wall_ns) returns the frame whose wall time
    is <= wall_ns (monotone advance). Frames scaled to (panel_w, H)."""
    def __init__(self, path, start_ns, H, lut=None):
        self.cont = av.open(path)
        self.vs = self.cont.streams.video[0]
        self.tb = float(self.vs.time_base)
        self.W0, self.H0 = self.vs.width, self.vs.height
        self.pw = self.W0 * H // self.H0
        self.pw -= self.pw % 2
        self.H = H
        self.start = float(start_ns)
        self.lut = lut
        self._it = self._gen()
        self._cur = next(self._it, (None, None))
        self._nxt = next(self._it, None)
        self._cache_w = None
        self._cache_im = None

    def _gen(self):
        for fr in self.cont.decode(self.vs):
            if fr.pts is None:
                continue
            yield self.start + fr.pts * self.tb * 1e9, fr

    def at(self, w):
        while self._nxt is not None and self._nxt[0] <= w:
            self._cur = self._nxt
            self._nxt = next(self._it, None)
        cw, fr = self._cur
        if fr is None:
            return Image.new("RGB", (self.pw, self.H))
        if cw != self._cache_w:                      # convert once per real frame
            im = Image.fromarray(fr.to_ndarray(format="rgb24")).resize((self.pw, self.H))
            if self.lut is not None:
                im = im.point(self.lut)
            self._cache_w, self._cache_im = cw, im
        return self._cache_im.copy()


def load_gaze(path, cols):
    d = pd.read_csv(path)
    ts = d["timestamp [ns]"].to_numpy(float)
    o = np.argsort(ts)
    out = {"ts": ts[o]}
    for k, c in cols.items():
        out[k] = d[c].to_numpy(float)[o] if c in d.columns else None
    return out


def nearest(ts, w, tol_ms):
    i = int(np.searchsorted(ts, w))
    i = min(max(i, 0), len(ts) - 1)
    if i > 0 and abs(ts[i - 1] - w) < abs(ts[i] - w):
        i -= 1
    return i if abs(ts[i] - w) <= tol_ms * 1e6 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_video", required=True)
    ap.add_argument("--scene_start_ns", required=True)
    ap.add_argument("--reference_image", required=True)
    ap.add_argument("--screen_video", required=True)
    ap.add_argument("--start_video_ns", required=True)
    ap.add_argument("--gaze_csv", required=True)
    ap.add_argument("--scene_gaze_csv", default=None)
    ap.add_argument("--out_video", required=True)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--white", type=float, default=148.0)
    ap.add_argument("--tol_ms", type=float, default=50.0)
    ap.add_argument("--start_s", type=float, default=None, help="window start (s from first gaze sample)")
    ap.add_argument("--duration_s", type=float, default=None, help="window length (s)")
    args = ap.parse_args()

    H = args.height - args.height % 2
    lut = list(np.clip(np.arange(256) * (255.0 / args.white), 0, 255).astype("uint8")) * 3

    g = load_gaze(args.gaze_csv, {
        "rx": "gaze position in reference image x [px]",
        "ry": "gaze position in reference image y [px]",
        "tx": "gaze position transf x [px]",
        "ty": "gaze position transf y [px]"})
    sg = load_gaze(args.scene_gaze_csv, {"x": "gaze x [px]", "y": "gaze y [px]"}) \
        if args.scene_gaze_csv else None

    ref = Image.open(args.reference_image).convert("RGB")
    RW, RH = ref.size
    rpw = RW * H // RH; rpw -= rpw % 2
    ref_base = ref.resize((rpw, H))
    label(ref_base, "Reference Image")

    scene = Streamer(args.scene_video, args.scene_start_ns, H)
    screen = Streamer(args.screen_video, args.start_video_ns, H, lut=lut)
    NW, NH0 = scene.W0, scene.H0

    lo, hi = g["ts"].min(), g["ts"].max()
    if args.start_s is not None:
        lo = g["ts"].min() + args.start_s * 1e9
    if args.duration_s is not None:
        hi = lo + args.duration_s * 1e9
    grid = np.arange(lo, hi, 1e9 / args.fps)
    OW = scene.pw + rpw + screen.pw; OW -= OW % 2
    print(f"composite {OW}x{H}; {len(grid)} frames @ {args.fps}fps ({len(grid)/args.fps:.1f}s)"
          f"{' + Neon gaze' if sg else ''}")

    out = av.open(args.out_video, "w")
    os_ = out.add_stream("libx264", rate=args.fps)
    os_.width, os_.height = OW, H
    os_.pix_fmt = "yuv420p"
    os_.options = {"crf": "20", "preset": "veryfast"}

    written = 0
    for w in grid:
        neon = scene.at(w)
        if sg is not None:
            j = nearest(sg["ts"], w, args.tol_ms)
            if j is not None and np.isfinite(sg["x"][j]):
                circle(neon, sg["x"][j] * scene.pw / NW, sg["y"][j] * H / NH0, max(5, int(10 * H / 1200)))
        label(neon, "Neon")

        rimg = ref_base.copy()
        simg = screen.at(w)
        i = nearest(g["ts"], w, args.tol_ms)
        if i is not None:
            if g["rx"] is not None and np.isfinite(g["rx"][i]):
                circle(rimg, g["rx"][i] * rpw / RW, g["ry"][i] * H / RH, max(6, int(14 * H / RH)))
            if g["tx"] is not None and np.isfinite(g["tx"][i]):
                circle(simg, g["tx"][i] * screen.pw / screen.W0, g["ty"][i] * H / screen.H0,
                       max(6, int(16 * H / screen.H0)))
        label(simg, "Screen Video")

        canvas = Image.new("RGB", (OW, H))
        canvas.paste(neon, (0, 0))
        canvas.paste(rimg, (scene.pw, 0))
        canvas.paste(simg, (scene.pw + rpw, 0))
        for p in os_.encode(av.VideoFrame.from_image(canvas)):
            out.mux(p)
        written += 1
    for p in os_.encode():
        out.mux(p)
    out.close()
    print(f"wrote {args.out_video}: {written} frames ({OW}x{H})")


if __name__ == "__main__":
    main()
