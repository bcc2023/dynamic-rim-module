#!/usr/bin/env python3
"""
gaze_calibration — unified CLI for a Pupil Labs Neon + Reference-Image-Mapper
"gaze on a screen" toolkit: map gaze onto a screen recording, check/transfer the
gaze calibration, render QA overlays, and analyze the result. General to any
on-screen task; the `analyze` command ships a steering-task example you can swap.

Each subcommand dispatches to a focused tool script in this folder, so the tools
stay independently runnable and testable while you also get one entry point:

    python cli.py <command> [options]
    python cli.py <command> --help     # options for one command

Commands
    align           Align the experiment-JSON cursor onto mapped gaze -> *_aligned.csv
                    (run AFTER pl-dynamic-rim).
    analyze         Distance, per-trial traces, and steering-difficulty plots
                    (Fitts ID for pointing; ID_W / ID_K + gaze lead for steering).
    calib-transfer  Transfer a manual gaze-offset calibration from the calibration
                    recording onto the task recording (homography in reference space).
    compose3        Build the Neon | Reference | Screen 3-panel video, gaze circle on
                    all three panels (streaming; scales to long recordings).
    overlay         Screen video + gaze circle, brightened (single panel).
    checksync       Report a screen<->Neon recording offset from container creation_time.

The scene->reference->screen *mapping itself* is done by the external tool
`pl-dynamic-rim` (installed separately from the yurahwang97 fork); see README.md /
WORKFLOWS.md. This CLI covers everything downstream of that.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# command -> (script filename, one-line help)
COMMANDS = {
    "align":          ("cursor_gaze_pipeline.py",
                       "cursor(JSON) + gaze -> aligned CSV (after pl-dynamic-rim)"),
    "analyze":        ("analyze_aligned.py",
                       "distance, per-trial traces, steering-difficulty plots"),
    "calib-transfer": ("apply_calibration_transfer.py",
                       "transfer calibration offset (calibration -> task), homography "
                       "[only valid when calibration & task are the same wearing]"),
    "selfcal":        ("self_calibrate.py",
                       "within-session gaze calibration from the task cursor "
                       "[use when calibration is a different wearing]"),
    "selfcal-video":  ("selfcal_video.py",
                       "self-cal in screen-video px (for compose3); needs cursor_scale/bx/by"),
    "compose3":       ("compose_3panel.py",
                       "Neon|Reference|Screen 3-panel video, gaze on all panels"),
    "overlay":        ("screen_gaze_overlay.py",
                       "screen video + gaze circle, brightened"),
    "checksync":      ("check_sync.py",
                       "screen<->Neon offset from creation_time (coarse)"),
}


def usage(stream=sys.stdout):
    print("usage: cli.py <command> [options]\n", file=stream)
    print("commands:", file=stream)
    for name, (_, desc) in COMMANDS.items():
        print(f"  {name:15s} {desc}", file=stream)
    print("\nrun 'cli.py <command> --help' for a command's options.",
          file=stream)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "help"):
        usage()
        return 0
    cmd = argv[0]
    if cmd not in COMMANDS:
        print(f"cli.py: unknown command {cmd!r}\n", file=sys.stderr)
        usage(sys.stderr)
        return 2
    script = os.path.join(HERE, COMMANDS[cmd][0])
    if not os.path.exists(script):
        print(f"cli.py: missing tool {script}", file=sys.stderr)
        return 1
    # Hand the remaining args to the tool's own argparse, then run it as __main__.
    sys.argv = [script] + argv[1:]
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
