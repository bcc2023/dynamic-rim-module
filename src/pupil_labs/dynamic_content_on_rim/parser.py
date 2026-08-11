import argparse
from enum import Enum


class audioSources(Enum):
    No_Audio = 0
    Device_Mic = 1  # Pupil Invisible microphone
    Screen_Audio = 2  # Screen Audio as defined by your recording method


def init_parser():
    parser = argparse.ArgumentParser(description="Pupil Labs - Dynamic RIM Module")
    parser.add_argument(
        "--screen_video_path", default=None, type=str, help="Path to the screen video"
    )
    parser.add_argument(
        "--raw_folder_path", default=None, type=str, help="Path to the raw folder"
    )
    parser.add_argument(
        "--rim_folder_path", default=None, type=str, help="Path to the RIM folder"
    )
    parser.add_argument(
        "--corners_screen",
        default=None,
        type=str,
        help="Path to the corners_screen file",
    )
    parser.add_argument(
        "--out_video_path",
        default=None,
        type=str,
        help="Path where to save the output video",
    )
    parser.add_argument(
        "--out_csv_path",
        default=None,
        type=str,
        help="Path where to save the output CSV",
    )
    parser.add_argument(
        "--audio",
        default="Device_Mic",
        choices=["No_Audio", "Device_Mic", "Screen_Audio"],
        type=str,
        help="Audio source, between device, computer, or none",
    )
    parser.add_argument(
        "--saveCSV", default=True, type=bool, help="Save the gaze data as a CSV file"
    )
    parser.add_argument(
        "--labels", default=True, type=bool, help="Show labels on the video"
    )
    parser.add_argument(
        "-p",
        "--visualise",
        action="store_true",
        help="Visualise the video as it creates",
    )
    parser.add_argument(
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
    parser.add_argument(
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
