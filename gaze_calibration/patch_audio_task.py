"""
One-off patch for pupil-labs-dynamic-rim (yurahwang97 fork, commit a017966).

Bug: at the end of save_videos(), progress_bar.stop_task(merge_audio_task) is
called unconditionally, but `merge_audio_task` is only ever assigned inside
    if merged_audio is not None and _recording:
When running with --audio No_Audio, merged_audio is None, so the variable is
never bound and the run dies at the very last step with:
    UnboundLocalError: cannot access local variable 'merge_audio_task'
    where it is not associated with a value

Note this fires AFTER out_container.close(), so the merged .mp4 is already
fully written and valid -- the crash only prevents the final log line and,
back in main(), the --saveCSV export.

Fix: guard the stop_task() call with the same condition that defines the task.

Usage:
    python3 patch_audio_task.py
"""
import sys

path = "/Users/xuebeichen/miniconda3/envs/pupil-cloud-local/lib/python3.11/site-packages/pupil_labs/dynamic_content_on_rim/dynamic_rim.py"

old = """        if _recording:
            out_container.close()
        progress_bar.stop_task(merge_audio_task)
        logging.info("Video saved to: " + args.out_video_path)
"""

new = """        if _recording:
            out_container.close()
        if merged_audio is not None and _recording:
            progress_bar.stop_task(merge_audio_task)
        logging.info("Video saved to: " + args.out_video_path)
"""

src = open(path).read()
count = src.count(old)
print("occurrences found:", count)
if count != 1:
    print(
        "ABORT: expected exactly 1 occurrence - file may already be patched, "
        "or differs from the expected commit. Not touching it."
    )
    sys.exit(1)

src = src.replace(old, new, 1)
open(path, "w").write(src)
print("patched OK")
