#!/usr/bin/env python3
"""
Check how well a screen recording is aligned with its Neon recording,
BEFORE spending time running pl-dynamic-rim.

WHY THIS EXISTS
---------------
pl-dynamic-rim assumes your screen recording and your Neon recording started at
the exact same instant. It never checks. If they didn't, every gaze point is
displaced on the screen video by the difference -- silently, with no error.

Both files do carry real wall-clock time, so the offset can be measured:
  * Neon      -> absolute UTC nanoseconds in events.csv (recording.begin)
  * QuickTime -> 'creation_time' tag in the .mov/.mp4 container

PRECISION LIMIT -- IMPORTANT
----------------------------
QuickTime stores creation time as WHOLE SECONDS (seconds since 1904 in the mvhd
atom). So this measurement is only good to about +/- 0.5 s. It reliably catches
gross desync -- someone starting one recording seconds or minutes before the
other -- but it CANNOT verify sub-second alignment.

If your analysis needs better than ~1 s, measure it directly with a visual sync
marker: flash something unmistakable on screen at the very start of the session,
visible in BOTH the screen capture and the scene camera. Then align on that
frame. That gets you to frame accuracy (~30-40 ms).

Usage:
    python3 check_sync.py \\
        --raw_folder_path   "/path/to/Timeseries Data + Scene Video/<recording>" \\
        --screen_video_path "/path/to/screen_recording.mov"
"""
import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd

try:
    import av
except ImportError:
    sys.exit("ERROR: PyAV not installed. Activate your env: conda activate pupil-cloud-local")


def screen_start_ns(path):
    """Wall-clock start of the screen recording, from container metadata."""
    with av.open(path) as c:
        raw = c.metadata.get("creation_time")
        duration_s = float(c.duration) / 1_000_000 if c.duration else None
    if not raw:
        return None, duration_s, None
    iso = raw.replace("Z", "+00:00")
    dt = datetime.datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return int(dt.timestamp() * 1e9), duration_s, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_folder_path", required=True)
    ap.add_argument("--screen_video_path", required=True)
    args = ap.parse_args()

    events_path = os.path.join(args.raw_folder_path, "events.csv")
    if not os.path.exists(events_path):
        sys.exit(f"ERROR: no events.csv in {args.raw_folder_path}")

    events = pd.read_csv(events_path, dtype={"timestamp [ns]": np.uint64})

    begin = events.loc[events["name"] == "recording.begin", "timestamp [ns]"]
    if begin.empty:
        sys.exit("ERROR: events.csv has no recording.begin event")
    neon_begin = int(begin.values[0])

    end = events.loc[events["name"] == "recording.end", "timestamp [ns]"]
    neon_end = int(end.values[0]) if not end.empty else None

    scr_ns, scr_dur, scr_dt = screen_start_ns(args.screen_video_path)

    def fmt(ns):
        return datetime.datetime.fromtimestamp(
            ns / 1e9, datetime.timezone.utc
        ).isoformat()

    print("=" * 62)
    print("NEON")
    print(f"  recording.begin : {fmt(neon_begin)}")
    if neon_end:
        print(f"  recording.end   : {fmt(neon_end)}")
        print(f"  duration        : {(neon_end - neon_begin)/1e9:.3f} s")

    print("\nSCREEN RECORDING")
    if scr_ns is None:
        print("  creation_time   : ABSENT from container metadata")
        print(f"  duration        : {scr_dur:.3f} s" if scr_dur else "")
        print("\n" + "=" * 62)
        print("VERDICT: cannot measure. This file carries no wall-clock stamp,")
        print("so alignment can't be verified. Re-encoding or editing a video")
        print("often strips this tag -- use the ORIGINAL recording if you have it.")
        sys.exit(0)

    print(f"  creation_time   : {scr_dt.isoformat()}")
    print(f"  duration        : {scr_dur:.3f} s")

    offset = (scr_ns - neon_begin) / 1e9

    print("\n" + "=" * 62)
    print(f"START OFFSET : {offset:+.3f} s")
    print()
    print("  Defined as (screen start) - (Neon start).")
    if offset > 0:
        print(f"  POSITIVE -> the SCREEN recording started {abs(offset):.3f} s LATER.")
        print("              You hit record on the Neon first.")
    elif offset < 0:
        print(f"  NEGATIVE -> the SCREEN recording started {abs(offset):.3f} s EARLIER.")
        print("              You hit record on the screen capture first.")
    else:
        print("  ZERO -> both started in the same measured second.")
    print("  Accurate to about +/- 0.5 s (see resolution note below).")

    print()
    if abs(offset) <= 1.0:
        print("VERDICT: OK -- do NOT pass an offset.")
        print("  This is within the measurement resolution, so the starts were")
        print("  effectively simultaneous -- which is what pl-dynamic-rim already")
        print("  assumes. Correcting by less than the measurement error would add")
        print("  noise rather than remove it. Run normally.")
        print("  For sub-second certainty you still need a visual sync marker.")
    else:
        if abs(offset) <= 3.0:
            print("VERDICT: MARGINAL -- correcting is probably worthwhile.")
            print("  Uncorrected, gaze will be displaced by roughly this much:")
            print("  tolerable for coarse region-level analysis, too large for")
            print("  fixation-level timing.")
        else:
            print("VERDICT: PROBLEM -- correct this before trusting any output.")
            print("  Uncorrected, gaze will land far from where you actually")
            print("  looked. The video will still look plausible. The CSV will")
            print("  still be wrong.")
        print()
        print("  Add this flag to your pl-dynamic-rim command:")
        print()
        print(f"      --screen_offset_s {offset:.3f}")
        print()
        print("  (Requires patch_all.py to have been applied -- the flag does")
        print("   not exist in the stock package.)")
        print("  Residual error after correcting will still be up to ~0.5 s.")

    if neon_end:
        dur_gap = scr_dur - (neon_end - neon_begin) / 1e9
        print(f"\nDuration difference: {dur_gap:+.2f} s (screen minus Neon)")
        print("  Note this reflects when you STOPPED, not when you started, and")
        print("  is harmless on its own -- the run is truncated at recording.end.")

    print("\nAlways confirm by watching the output video: if the gaze dot")
    print("reaches things noticeably before or after you actually looked at")
    print("them, the sync is off regardless of what this script reports.")


if __name__ == "__main__":
    main()
