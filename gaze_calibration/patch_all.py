#!/usr/bin/env python3
"""
Consolidated bug-fix patcher for pupil-labs-dynamic-rim
(yurahwang97 fork, commit a017966).

WHAT THIS IS
------------
Four upstream bugs in the package's own source code. None of them are
specific to any particular recording, camera, or dataset -- they are code
defects that will affect EVERY recording you process. Three of the four fire
unconditionally on every single run.

Because they live in installed source under site-packages, they are ERASED by
any `pip install`, upgrade, or fresh conda env. Re-run this script after any
of those. It is idempotent: already-applied fixes are detected and skipped,
so running it twice is harmless.

Usage:
    python3 patch_all.py            # patch the active environment
    python3 patch_all.py --check    # report status, change nothing

--------------------------------------------------------------------------
THE FOUR FIXES
--------------------------------------------------------------------------

1. video/read.py -- negative decode timestamps  [DATA-DEPENDENT]
   read_video_ts() casts frame .dts to np.uint64. Videos with B-frame
   reordering (very common in macOS screen recordings) legitimately carry
   negative dts on the first frames, and newer numpy raises rather than
   silently wrapping:
       OverflowError: Python integer -4770 out of bounds for uint64
   dts is never returned by the function -- it feeds only a debug log
   comparison -- so widening to int64 is safe.
   Triggers on: any video with negative dts. Not all recordings, but you
   cannot predict which, and the fix is harmless when unneeded.

2. video/read.py -- audio streams have no average_rate  [ALWAYS]
   read_video_ts() unconditionally reads stream.average_rate, which exists
   only on video streams in PyAV. Any run using --audio Device_Mic (the
   DEFAULT) or Screen_Audio dies with:
       AttributeError: 'AudioCodecContext' object has no attribute 'average_rate'
   The caller discards this value for audio streams, so returning None is safe.
   Triggers on: every run that uses audio. Fixing this is what makes any
   audio option usable at all; without it you are stuck on --audio No_Audio.

3. dynamic_rim.py -- putText on a float64 canvas  [ALWAYS]
   save_videos() builds `bkg` with np.zeros(...) => float64, then draws
   labels on it with cv2.putText, which requires 8-bit:
       cv2.error: (-215:Assertion failed) img.depth() == CV_8U in putText
   The uint8 conversion happens one step too late. Moving the existing
   cv2.normalize call above the label block fixes it with no visual change
   (same NORM_MINMAX stretch, just computed earlier).
   Triggers on: every run with labels enabled (the default) on OpenCV 5.x,
   which tightened this assertion.

4. dynamic_rim.py -- unbound merge_audio_task  [ALWAYS, with No_Audio]
   progress_bar.stop_task(merge_audio_task) is called unconditionally, but
   merge_audio_task is only assigned inside `if merged_audio is not None
   and _recording:`. With --audio No_Audio it is never bound:
       UnboundLocalError: cannot access local variable 'merge_audio_task'
   Fires after out_container.close(), so the video survives but the
   --saveCSV export never runs.
   Triggers on: every run with --audio No_Audio.

--------------------------------------------------------------------------
NOT FIXED HERE (because they are not bugs)
--------------------------------------------------------------------------
"Resized reference image, gaze not plotted as xy contained a NaN value"
   Expected. Gaze estimation comes online a beat after recording.begin, so
   the earliest frames have no gaze sample to plot. Correct behavior.
   A "fix" would mean extrapolating gaze onto frames where none was
   recorded -- i.e. fabricating data. Left alone deliberately.
"""
import argparse
import importlib.util
import os
import sys

# ---------------------------------------------------------------- fixes ----

FIXES = [
    {
        "name": "1. read.py: negative dts -> int64",
        "file": "video/read.py",
        "old": """            np.array(pts, dtype=np.uint64),
            np.array(dts, dtype=np.uint64),
            np.array(ts, dtype=np.uint64),
""",
        "new": """            np.array(pts, dtype=np.uint64),
            np.array(dts, dtype=np.int64),
            np.array(ts, dtype=np.uint64),
""",
        "applied_marker": "np.array(dts, dtype=np.int64)",
    },
    {
        # NB: this is the same fix upstream pupil-labs already shipped
        # (commit after this fork's base). Matching their wording exactly
        # keeps divergence minimal. stream.rate is the audio sample rate.
        "name": "2. read.py: audio stream average_rate",
        "file": "video/read.py",
        "old": [
            # pristine fork source
            """        fps = stream.average_rate  # alt base_rate or guessed_rate
""",
            # earlier version of this patcher used None; migrate it to upstream's
            """        fps = None if audio else stream.average_rate  # alt base_rate or guessed_rate
""",
        ],
        "new": """        fps = (
            stream.average_rate if not audio else stream.rate
        )  # alt base_rate or guessed_rate
""",
        "applied_marker": "stream.average_rate if not audio else stream.rate",
    },
    {
        "name": "3. dynamic_rim.py: putText on float64 canvas",
        "file": "dynamic_rim.py",
        "old": """            if args.labels:
                # Add text to the frames
                y, w, h = 60, 360, 60
                label_device = ""
                with open(os.path.join(args.raw_folder_path, "info.json"), "r") as f:
                    import json

                    info = json.load(f)
                    # check if there is a frame_name field in the info.json file
                    label_device = "Neon" if "frame_name" in info else "Pupil Invisible"

                labels = [
                    {"text": label_device, "margin": 10},
                    {"text": "Reference Image", "margin": 10 + _etframe.width},
                    {
                        "text": "Screen Video",
                        "margin": 10 + (_etframe.width + refimg_finalwidth),
                    },
                ]
                for label in labels:
                    overlay = bkg[
                        y : y + h, (label["margin"] - 10) : (label["margin"] - 10 + w)
                    ]
                    whiterect = np.ones(overlay.shape) * 255
                    res = cv2.addWeighted(overlay, 0.5, whiterect, 0.5, 0)
                    bkg[
                        y : y + h, (label["margin"] - 10) : (label["margin"] - 10) + w
                    ] = res
                    bkg = cv2.putText(
                        bkg,
                        label["text"],
                        (label["margin"], 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 0),
                        2,
                        2,
                    )
            # Get the final frame
            out_ = bkg.copy()
            out_ = cv2.normalize(out_, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
""",
        "new": """            # Get the final frame (normalize to 8-bit *before* drawing labels:
            # cv2.putText requires CV_8U, but bkg is float64)
            out_ = cv2.normalize(bkg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            if args.labels:
                # Add text to the frames
                y, w, h = 60, 360, 60
                label_device = ""
                with open(os.path.join(args.raw_folder_path, "info.json"), "r") as f:
                    import json

                    info = json.load(f)
                    # check if there is a frame_name field in the info.json file
                    label_device = "Neon" if "frame_name" in info else "Pupil Invisible"

                labels = [
                    {"text": label_device, "margin": 10},
                    {"text": "Reference Image", "margin": 10 + _etframe.width},
                    {
                        "text": "Screen Video",
                        "margin": 10 + (_etframe.width + refimg_finalwidth),
                    },
                ]
                for label in labels:
                    overlay = out_[
                        y : y + h, (label["margin"] - 10) : (label["margin"] - 10 + w)
                    ]
                    whiterect = (np.ones(overlay.shape) * 255).astype(np.uint8)
                    res = cv2.addWeighted(overlay, 0.5, whiterect, 0.5, 0)
                    out_[
                        y : y + h, (label["margin"] - 10) : (label["margin"] - 10) + w
                    ] = res
                    out_ = cv2.putText(
                        out_,
                        label["text"],
                        (label["margin"], 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 0),
                        2,
                        2,
                    )
""",
        "applied_marker": "cv2.putText requires CV_8U",
    },
    {
        "name": "4. dynamic_rim.py: unbound merge_audio_task",
        "file": "dynamic_rim.py",
        "old": """        if _recording:
            out_container.close()
        progress_bar.stop_task(merge_audio_task)
""",
        "new": """        if _recording:
            out_container.close()
        if merged_audio is not None and _recording:
            progress_bar.stop_task(merge_audio_task)
""",
        "applied_marker": """        if merged_audio is not None and _recording:
            progress_bar.stop_task(merge_audio_task)""",
    },
    {
        "name": "5. parser.py: add --screen_offset_s flag  [FEATURE]",
        "file": "parser.py",
        "old": """    parser.set_defaults(visualise=False)
    return parser
""",
        "new": '''    parser.add_argument(
        "--screen_offset_s",
        default=0.0,
        type=float,
        help=(
            "Seconds between the Neon's recording.begin and the true start of "
            "the screen recording, i.e. (screen start) - (Neon start). "
            "POSITIVE if screen capture began AFTER the Neon recording; "
            "NEGATIVE if it began BEFORE. Default 0 assumes they started at the "
            "same instant. Measure it with check_sync.py."
        ),
    )
    parser.set_defaults(visualise=False)
    return parser
''',
        "applied_marker": '"--screen_offset_s"',
    },
    {
        "name": "6. dynamic_rim.py: apply --screen_offset_s  [FEATURE]",
        "file": "dynamic_rim.py",
        "old": """    start_video_ns = recording_begin.values[0]
    # Create some timestamps [ns] for the screen video to match, use the start_video_ns
    # and the ts of the screen video
    sc_timestamps_ns = sc_ts + start_video_ns
""",
        "new": """    start_video_ns = recording_begin.values[0]
    # --- screen/Neon start-offset correction -------------------------------
    # By default this code assumes the screen recording and the Neon recording
    # started at the same instant. --screen_offset_s lets you correct a measured
    # difference:  offset = (screen start) - (Neon start), in seconds.
    #   positive -> screen capture began AFTER  the Neon recording
    #   negative -> screen capture began BEFORE the Neon recording
    # Applied to start_video_ns so it also covers Screen_Audio below. Device_Mic
    # is anchored to the Neon's own clock and is deliberately left untouched.
    # int64 intermediate: start_video_ns is uint64 and a negative offset would
    # otherwise wrap around.
    _offset_s = getattr(args, "screen_offset_s", 0.0) or 0.0
    if _offset_s:
        start_video_ns = np.uint64(
            np.int64(start_video_ns) + np.int64(round(_offset_s * 1e9))
        )
        _when = "after" if _offset_s > 0 else "before"
        logging.info(
            "Applied screen start offset of %+.3f s "
            "(screen capture began %s the Neon recording)" % (_offset_s, _when)
        )
    # -----------------------------------------------------------------------
    # Create some timestamps [ns] for the screen video to match, use the start_video_ns
    # and the ts of the screen video
    sc_timestamps_ns = sc_ts + start_video_ns
""",
        "applied_marker": "screen/Neon start-offset correction",
    },
    {
        # The fork renamed upstream's 'start.video' to 'recording.begin', which
        # COLLIDES with the event the Neon emits automatically at recording
        # start. dynamic_rim takes .values[0] -- the first match -- so it would
        # pick the Neon's automatic event and silently discard the precisely
        # measured one this script writes. Restore upstream's name.
        "name": "7. recording.py: emit 'start.video' not 'recording.begin'",
        "file": "recording/recording.py",
        "old": """                await device.send_event(
                    "recording.begin",
                    event_timestamp_unix_ns=np.mean([aftereq, befreq]) - offset,
                )
""",
        "new": """                await device.send_event(
                    "start.video",
                    event_timestamp_unix_ns=np.mean([aftereq, befreq]) - offset,
                )
""",
        "applied_marker": '"start.video",',
    },
    {
        # Must come AFTER fix 6: fix 6 anchors on the line
        # `start_video_ns = recording_begin.values[0]`, which this fix
        # deliberately leaves untouched by reusing the recording_begin name.
        "name": "8. dynamic_rim.py: prefer 'start.video', fall back to 'recording.begin'",
        "file": "dynamic_rim.py",
        "old": """    recording_begin = events_df.loc[
        events_df["name"] == "recording.begin", "timestamp [ns]"
    ]
    if recording_begin.empty:
        raise ValueError("events.csv does not contain a recording.begin event")
""",
        "new": '''    # Alignment anchor for the screen recording. Prefer 'start.video', which
    # recording.py writes at the exact moment screen capture began (corrected
    # for the Neon's clock offset) -- that is true measured synchronisation.
    # Fall back to 'recording.begin', the Neon's own automatic event, which
    # aligns correctly ONLY if both recordings started at the same instant.
    _start_video = events_df.loc[events_df["name"] == "start.video", "timestamp [ns]"]
    recording_begin = events_df.loc[
        events_df["name"] == "recording.begin", "timestamp [ns]"
    ]
    if not _start_video.empty:
        logging.info(
            "Aligning on the 'start.video' event (measured sync)"
        )
        recording_begin = _start_video
    elif recording_begin.empty:
        raise ValueError(
            "events.csv contains neither a 'start.video' nor a "
            "'recording.begin' event"
        )
    else:
        logging.warning(
            "No 'start.video' event found -- falling back to 'recording.begin'. "
            "This ASSUMES the screen recording and the Neon recording started "
            "at the same instant. Verify with check_sync.py, or correct with "
            "--screen_offset_s."
        )
''',
        "applied_marker": "Prefer 'start.video'",
    },
    {
        # Must run AFTER patch 5 (both anchor on the set_defaults line, and 5
        # creates the --screen_offset_s arg this pairs with).
        "name": "9. parser.py: add --screen_start_wallclock flag  [FEATURE]",
        "file": "parser.py",
        "old": """    parser.set_defaults(visualise=False)
    return parser
""",
        "new": '''    parser.add_argument(
        "--screen_start_wallclock",
        default=None,
        help=(
            "Absolute wall-clock time shown by an on-screen millisecond clock "
            "at the start of the screen recording, e.g. '2026-08-10 "
            "17:18:46.697'. Interpreted in the COMPUTER'S LOCAL timezone (what "
            "the on-screen clock displays). Pins the screen recording's first "
            "frame to real time with millisecond precision, bypassing the "
            "whole-second creation_time limit. Takes precedence over "
            "--screen_offset_s."
        ),
    )
    parser.add_argument(
        "--screen_start_at_video_s",
        default=0.0,
        type=float,
        help=(
            "Video-relative time in seconds of the frame where you read the "
            "on-screen clock for --screen_start_wallclock. Default 0.0 = the "
            "very first frame. Use this if the clock is only legible a moment "
            "into the recording."
        ),
    )
    parser.set_defaults(visualise=False)
    return parser
''',
        "applied_marker": '"--screen_start_wallclock"',
    },
    {
        # Must run AFTER patch 6 (its "old" is patch 6's output block).
        "name": "10. dynamic_rim.py: apply --screen_start_wallclock  [FEATURE]",
        "file": "dynamic_rim.py",
        "old": """    start_video_ns = recording_begin.values[0]
    # --- screen/Neon start-offset correction -------------------------------
    # By default this code assumes the screen recording and the Neon recording
    # started at the same instant. --screen_offset_s lets you correct a measured
    # difference:  offset = (screen start) - (Neon start), in seconds.
    #   positive -> screen capture began AFTER  the Neon recording
    #   negative -> screen capture began BEFORE the Neon recording
    # Applied to start_video_ns so it also covers Screen_Audio below. Device_Mic
    # is anchored to the Neon's own clock and is deliberately left untouched.
    # int64 intermediate: start_video_ns is uint64 and a negative offset would
    # otherwise wrap around.
    _offset_s = getattr(args, "screen_offset_s", 0.0) or 0.0
    if _offset_s:
        start_video_ns = np.uint64(
            np.int64(start_video_ns) + np.int64(round(_offset_s * 1e9))
        )
        _when = "after" if _offset_s > 0 else "before"
        logging.info(
            "Applied screen start offset of %+.3f s "
            "(screen capture began %s the Neon recording)" % (_offset_s, _when)
        )
    # -----------------------------------------------------------------------
""",
        "new": '''    start_video_ns = recording_begin.values[0]
    # --- absolute screen start from an on-screen clock ---------------------
    # If you recorded a millisecond wall-clock on screen, read the time it
    # shows at a known video frame and pass it via --screen_start_wallclock.
    # That pins the screen recording's frame 0 to real time with millisecond
    # precision -- the video container's own creation_time is only whole-second.
    # Takes precedence over --screen_offset_s.
    _wallclock = getattr(args, "screen_start_wallclock", None)
    _offset_s = getattr(args, "screen_offset_s", 0.0) or 0.0
    if _wallclock:
        import datetime as _dt

        _parsed = None
        for _f in (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                _parsed = _dt.datetime.strptime(_wallclock, _f)
                break
            except ValueError:
                pass
        if _parsed is None:
            raise ValueError(
                "Could not parse --screen_start_wallclock=%r. "
                "Use 'YYYY-MM-DD HH:MM:SS.fff'." % (_wallclock,)
            )
        # A naive datetime's .timestamp() is interpreted in the computer's
        # LOCAL timezone -- exactly what an on-screen wall clock shows.
        _at = getattr(args, "screen_start_at_video_s", 0.0) or 0.0
        _wc_ns = int(round(_parsed.timestamp() * 1e9)) - int(round(_at * 1e9))
        _implied = (_wc_ns - int(start_video_ns)) / 1e9
        _dir = "after" if _implied > 0 else "before"
        logging.info(
            "On-screen clock: reading %s (local) at video +%.3fs pins screen "
            "frame 0 to real time. Screen recording began %.3f s %s the Neon "
            "recording.begin." % (_wallclock, _at, abs(_implied), _dir)
        )
        start_video_ns = np.uint64(_wc_ns)
        if _offset_s:
            logging.warning(
                "--screen_offset_s=%+.3f IGNORED: --screen_start_wallclock "
                "takes precedence." % _offset_s
            )
    # --- screen/Neon start-offset correction -------------------------------
    # --screen_offset_s corrects a measured difference when you did NOT use an
    # on-screen clock:  offset = (screen start) - (Neon start), in seconds.
    #   positive -> screen capture began AFTER  the Neon recording
    #   negative -> screen capture began BEFORE the Neon recording
    # Applied to start_video_ns so it also covers Screen_Audio below. Device_Mic
    # is anchored to the Neon's own clock and is deliberately left untouched.
    # int64 intermediate: start_video_ns is uint64 and a negative offset would
    # otherwise wrap around.
    elif _offset_s:
        start_video_ns = np.uint64(
            np.int64(start_video_ns) + np.int64(round(_offset_s * 1e9))
        )
        _when = "after" if _offset_s > 0 else "before"
        logging.info(
            "Applied screen start offset of %+.3f s "
            "(screen capture began %s the Neon recording)" % (_offset_s, _when)
        )
    # -----------------------------------------------------------------------
''',
        "applied_marker": "absolute screen start from an on-screen clock",
    },
    {
        # Real-time render. The screen recording is variable-frame-rate (a frame
        # is written only when the screen changes), but the output is muxed at a
        # constant 30 fps, one output frame per source frame. That makes playback
        # lurch -- slow-motion where source frames are dense (fast cursor
        # motion), fast-forward where they're sparse (static screen). Fix:
        # resample the merged table onto a regular 1/30 s REAL-TIME grid so each
        # output frame corresponds to a real instant. Static screen -> same frame
        # repeats (correctly frozen); motion -> true 30 fps. Does not touch the
        # CSV (saved from gaze_rim_df) or audio (merged_audio, muxed by real ts).
        "name": "11. dynamic_rim.py: real-time render (resample merged to 30fps grid)",
        "file": "dynamic_rim.py",
        "old": """    merged = merged[merged["timestamp [ns]"] <= end_video_ns]

    if audio in audioSources and audio != audioSources.No_Audio:
""",
        "new": """    merged = merged[merged["timestamp [ns]"] <= end_video_ns]

    # --- resample onto a real-time grid so the output plays at true speed ---
    # The screen recording is variable-frame-rate: a frame is written only when
    # the screen changes. Writing one output frame per source frame at a
    # constant 30 fps makes playback lurch (slow-motion where frames are dense,
    # fast-forward where sparse). Instead, sample the merged table on a regular
    # 1/30 s REAL-TIME grid: each output frame = the source frame nearest that
    # real instant. Static screen -> the same frame repeats (correctly frozen);
    # motion -> true 30 fps. Gaze/eye columns ride along by nearest match so
    # alignment is preserved. Untouched: the CSV (saved from gaze_rim_df) and
    # the audio path (merged_audio, muxed by its own real timestamps).
    if len(merged) > 1:
        _fps = 30
        _step = int(round(1e9 / _fps))
        _src = merged.sort_values("timestamp [ns]").reset_index(drop=True)
        _src["timestamp [ns]"] = _src["timestamp [ns]"].astype(np.int64)
        _t0 = int(_src["timestamp [ns]"].iloc[0])
        _t1 = int(_src["timestamp [ns]"].iloc[-1])
        _grid = np.arange(_t0, _t1 + 1, _step, dtype=np.int64)
        _grid_df = pd.DataFrame({"timestamp [ns]": _grid})
        merged = pd.merge_asof(
            _grid_df, _src, on="timestamp [ns]", direction="nearest"
        )
        logging.info(
            "Real-time render: %d variable-rate source frames -> %d frames at "
            "%d fps over %.1f s." % (len(_src), len(merged), _fps, (_t1 - _t0) / 1e9)
        )
    # -----------------------------------------------------------------------

    if audio in audioSources and audio != audioSources.No_Audio:
""",
        "applied_marker": "resample onto a real-time grid",
    },
    {
        # Save the CSV BEFORE the (long) video render, and allow skipping the
        # render. The gaze CSV's data is fully computed before save_videos, but
        # stock code writes it only AFTER -- so on a long recording you wait
        # through the whole render (or lose the CSV if it crashes) just to get
        # the CSV. Reorder: CSV first, then render; and add --no_video to skip
        # the render when you only need the CSV (e.g. for cursor_gaze_pipeline).
        "name": "12. dynamic_rim.py: save CSV before render + honor --no_video",
        "file": "dynamic_rim.py",
        "old": """    # Plot the videos
    logging.info("Plotting/Recording...")
    save_videos(args, merged, ref_img, _screen, merged_audio, audio)
    if args.saveCSV:
        logging.info("Saving CSV...")
        gaze_rim_df.to_csv(get_savedir(args.out_csv_path, "csv"), index=True)
""",
        "new": """    # Save the CSV FIRST: its data is ready now, so writing it before the
    # (potentially very long) video render makes it available immediately and
    # keeps it even if rendering is interrupted. --no_video skips the render.
    if args.saveCSV:
        logging.info("Saving CSV...")
        gaze_rim_df.to_csv(get_savedir(args.out_csv_path, "csv"), index=True)
    if getattr(args, "no_video", False):
        logging.info("Skipping video render (--no_video).")
    else:
        logging.info("Plotting/Recording...")
        save_videos(args, merged, ref_img, _screen, merged_audio, audio)
""",
        "applied_marker": "Save the CSV FIRST",
    },
    {
        "name": "13. parser.py: add --no_video flag  [FEATURE]",
        "file": "parser.py",
        "old": """    parser.set_defaults(visualise=False)
    return parser
""",
        "new": """    parser.add_argument(
        "--no_video",
        action="store_true",
        help=(
            "Skip the merged-video render and only write the CSV. Much faster "
            "on long recordings when you just need the gaze CSV (e.g. to feed "
            "cursor_gaze_pipeline.py)."
        ),
    )
    parser.set_defaults(visualise=False, no_video=False)
    return parser
""",
        "applied_marker": '"--no_video"',
    },
    {
        # The 3-panel merged canvas width = scene + reference + screen. Its width
        # (and sometimes height) can be ODD, depending on the reference image's
        # aspect ratio. H.264 with yuv420p requires EVEN width and height, so
        # libx264 refuses to open ("Generic error ... avcodec_open2 libx264").
        # Round the output stream size down to even here; crop frames to match
        # in patch 15.
        "name": "14. dynamic_rim.py: even output dimensions (libx264 needs even)",
        "file": "dynamic_rim.py",
        "old": """        out_video.height = mheight
        out_video.width = _etframe.width + _scframe.width + refimg_finalwidth
        out_video.pix_fmt = "yuv420p"
""",
        "new": """        # H.264 / yuv420p require EVEN width and height; the composited canvas
        # can be odd (reference-image aspect ratio), which makes libx264 refuse
        # to open. Round the stream size down to even; frames are cropped to
        # match where they are encoded (patch 15).
        out_video.height = mheight - (mheight % 2)
        _cw = _etframe.width + _scframe.width + refimg_finalwidth
        out_video.width = _cw - (_cw % 2)
        out_video.pix_fmt = "yuv420p"
""",
        "applied_marker": "H.264 / yuv420p require EVEN",
    },
    {
        # Must run AFTER patch 14 (relies on out_video.height/width being set to
        # the even values). Crop each composited frame to the stream's even size.
        "name": "15. dynamic_rim.py: crop frames to even stream size",
        "file": "dynamic_rim.py",
        "old": """                out_frame = av.VideoFrame.from_ndarray(out_, format="rgb24")
""",
        "new": """                out_ = np.ascontiguousarray(
                    out_[: out_video.height, : out_video.width]
                )
                out_frame = av.VideoFrame.from_ndarray(out_, format="rgb24")
""",
        "applied_marker": "out_[: out_video.height, : out_video.width]",
    },
]


def find_package(explicit=None):
    """Locate the package to patch.

    With --path: a source checkout, e.g. a git clone. Accepts either the
    repo root (we'll find src/pupil_labs/dynamic_content_on_rim) or the
    package directory itself.

    Without --path: the installed package in whatever env is active.
    """
    if explicit:
        p = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.isdir(p):
            sys.exit(f"ERROR: no such directory: {p}")
        # repo root?
        nested = os.path.join(p, "src", "pupil_labs", "dynamic_content_on_rim")
        if os.path.isdir(nested):
            return nested
        # already the package dir?
        if os.path.exists(os.path.join(p, "dynamic_rim.py")):
            return p
        sys.exit(
            f"ERROR: {p} doesn't look like the dynamic-rim source.\n"
            "Expected either a repo root containing "
            "src/pupil_labs/dynamic_content_on_rim/, or that directory itself."
        )

    spec = importlib.util.find_spec("pupil_labs.dynamic_content_on_rim")
    if spec is None or not spec.submodule_search_locations:
        sys.exit(
            "ERROR: pupil_labs.dynamic_content_on_rim not found.\n"
            "Activate the right conda env first:  conda activate pupil-cloud-local\n"
            "Or point at a source checkout with --path."
        )
    return list(spec.submodule_search_locations)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check", action="store_true", help="report status without modifying files"
    )
    ap.add_argument(
        "--path",
        default=None,
        help="patch a source checkout (git clone) instead of the installed package",
    )
    args = ap.parse_args()

    pkg = find_package(args.path)
    print(f"package: {pkg}\n")

    applied = skipped = failed = 0

    for fix in FIXES:
        target = os.path.join(pkg, fix["file"])
        if not os.path.exists(target):
            print(f"  [MISSING FILE] {fix['name']}  ({fix['file']})")
            failed += 1
            continue

        src = open(target).read()

        if fix["applied_marker"] in src:
            print(f"  [already ok]   {fix['name']}")
            skipped += 1
            continue

        # "old" may be a single string or several accepted variants
        # (e.g. pristine source, or output of an earlier version of this script)
        candidates = fix["old"] if isinstance(fix["old"], list) else [fix["old"]]
        match = next((c for c in candidates if src.count(c) == 1), None)

        if match is None:
            counts = ", ".join(str(src.count(c)) for c in candidates)
            print(
                f"  [!! FAILED]    {fix['name']}\n"
                f"                 no variant matched exactly once (counts: {counts}).\n"
                f"                 Upstream source may have changed. Not modifying."
            )
            failed += 1
            continue

        if args.check:
            print(f"  [would patch]  {fix['name']}")
            applied += 1
            continue

        open(target, "w").write(src.replace(match, fix["new"], 1))
        print(f"  [PATCHED]      {fix['name']}")
        applied += 1

    verb = "would apply" if args.check else "applied"
    print(f"\n{verb}: {applied}   already ok: {skipped}   failed: {failed}")

    if failed:
        sys.exit(1)

    # sanity check: files still parse
    if not args.check:
        import py_compile

        for f in ("video/read.py", "dynamic_rim.py"):
            try:
                py_compile.compile(os.path.join(pkg, f), doraise=True)
            except py_compile.PyCompileError as e:
                sys.exit(f"ERROR: {f} no longer compiles after patching:\n{e}")
        print("syntax check: all patched files compile OK")


if __name__ == "__main__":
    main()
