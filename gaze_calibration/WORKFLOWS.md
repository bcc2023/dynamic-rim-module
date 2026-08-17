# gaze_calibration — Workflows (read me first)

Operator's guide for the toolkit: how to map gaze onto a screen, check/transfer
the gaze calibration, align & analyze, and render the QA 3-panel. The recipes are
general; the paths, session table, and analysis below are a **worked example**
from one steering study (swap in your own). `README.md` is the overview; this
file is the step-by-step. Commands assume the tools live in
`dynamic-rim-module/gaze_calibration/` (invoke the CLI as
`python .../gaze_calibration/cli.py <command>`).

Environment: conda env `pupil-cloud-local`. The mapping tool is an **editable**
install of the fork `dynamic-rim-module/` (yurahwang97 fork), already patched by
`patch_all.py`. Run everything from that env.

---

## Script map

**Active pipeline**
| Script | Role |
|---|---|
| `patch_all.py` | Idempotent bug-fix patcher for `pl-dynamic-rim` (run once after (re)install). |
| `run_calibrated_dynamic_rim.sh` | Template: run `pl-dynamic-rim` on several recordings (edit paths per study). |
| `cursor_gaze_pipeline.py` | AFTER `pl-dynamic-rim`: align the experiment-JSON cursor onto the mapped gaze → `*_aligned.csv`. |
| `analyze_aligned.py` | Distance, per-trial traces, and steering difficulty plots from an `*_aligned.csv`. |
| `run_all_visualizations.sh` | Run `analyze_aligned.py` over several sessions. |
| `apply_calibration_transfer.py` | Transfer a manual gaze-offset calibration from the calibration recording onto the task recording (homography). |
| `compose_3panel.py` | Build the `Neon | Reference | Screen` 3-panel video (gaze on all three panels), streaming — scales to long videos. |
| `screen_gaze_overlay.py` | Lightweight single-panel: screen video + gaze circle, brightened. |
| `check_sync.py` | Report screen↔Neon offset from `creation_time` (±0.5 s coarse check). |

**Helper / superseded** (kept for reference, not the main path):
`overlay_from_aligned.py` (verify an aligned CSV by overlaying its own columns),
`align_gaze_cursor.py` + `overlay_cursor_gaze.py` (early standalone versions,
superseded by `cursor_gaze_pipeline.py`), `patch_audio_task.py` /
`patch_labels_dtype.py` (one-off patches folded into `patch_all.py`),
`run_task_label_overlays.sh` (task-label-specific overlay run).

---

## Workflow A — Dynamic RIM mapping (scene gaze → reference → screen)

Maps Pupil Cloud RIM gaze onto a **screen recording**. Three inputs per recording:

1. `--rim_folder_path` — the RIM enrichment export (`gaze.csv`, `sections.csv`, `reference_image.jpeg`). May bundle several recordings; the tool picks the right one by matching the raw folder's `info.json` `recording_id`.
2. `--raw_folder_path` — the recording's **`Timeseries Data + Scene Video/<recording>`** folder (needs the scene `.mp4` + `world_timestamps.csv` + `events.csv` + `info.json` + `gaze.csv`). NOT the plain `Timeseries Data` folder (no scene video).
3. `--screen_video_path` — the screen `.mov`.

**Always pass `--out_csv_path` and `--out_video_path`.** If omitted, the tool opens a GUI "save" dialog and dumps to `~` with a random UUID name (this is what caused the "which file is which" mess). Send outputs to a per-recording folder.

**Sync (pick one, best first):**
- `--screen_start_wallclock "YYYY-MM-DD HH:MM:SS.mmm"` — the wall-clock time of screen **frame 0**, in the Mac's **local** timezone (type what an on-screen ms clock shows, or the recorder's start time). Best precision.
- `--screen_offset_s <sec>` — `screen_start − Neon_begin` in seconds; timezone-proof. Use when you only have the `.mov` `creation_time` (±0.5 s, since QuickTime rounds to whole seconds).
- On-screen ms clock: embed `clock.html` (or the calibration page) so the recording captures a live clock; read frame 0 → `--screen_start_wallclock`.

Command:
```bash
caffeinate -dimsu pl-dynamic-rim \
  --rim_folder_path   "<enrichment>" \
  --raw_folder_path   "<...>/Timeseries Data + Scene Video/<recording>" \
  --screen_video_path "<screen>.mov" \
  --audio No_Audio \
  --screen_start_wallclock "YYYY-MM-DD HH:MM:SS.mmm" \
  --out_csv_path   "<out>/mapped_gaze.csv" \
  --out_video_path "<out>/mapped_overlay.mp4"
```
For multiple recordings, copy `run_calibrated_dynamic_rim.sh`, edit the `run …`
lines, `bash` it. Deriving the sync offset: `Neon_begin` is `section start time
[ns]` in the enrichment `sections.csv`; convert to local and compare with the
screen start.

Output = a gaze CSV with `gaze position in reference image x/y [px]` and `gaze
position transf x/y [px]` (screen pixels), plus the 3-panel overlay video.

---

## Workflow B — Cursor↔gaze alignment + steering analysis

For the **steering experiment** (browser task with a JSON log).

1. `cursor_gaze_pipeline.py` — time-align the experiment JSON's cursor onto the
   mapped gaze samples (Mac clock → Neon clock via the measured offset) and map
   cursor task-coords → screen-video px. Produces `*_aligned.csv` (cursor +
   gaze in screen-video px, `in_trial`, `constrained`, per-sample difficulty,
   `trial_id`, etc.).
2. `analyze_aligned.py --aligned_csv <..._aligned.csv> --experiment_json <steering_*.json>`
   produces:
   - **Step 1**: `gaze_cursor_dist_px` (Euclidean) + signed variant per in-trial sample.
   - **Step 2**: per-trial trace PDF — signed gaze−cursor pixel distance vs time, constrained/unconstrained shaded. (Unchanged; the paper note does not ask to change this.)
   - **Step 3**: difficulty aggregates:
     - **Unconstrained (pointing)** — Fitts ID `= log2(1 + D/W)`, `W = 2·targetRadius`, `D = condition.distance` (mixed trials: `distance − transition_point.taskX`). y = mean gaze-cursor distance.
     - **Constrained (steering)** — new "Curves Ahead" index: `ID_W = ∫ds/W` (width) and `ID_K = ∫|κ|ds` (total turning). `ID_K = max(curvature-field integral, turning of the RDP-simplified centerline)` so it's right for smooth sinusoids AND sharp corners. Two scatters: **width family** (straight, narrow→wide, wide→narrow) vs `ID_W`, **curvature family** (sinusoids, corners) vs `ID_K`. y = **gaze lead** = signed arc-length of gaze − cursor projected onto the tunnel centerline (x-monotone projection; gaze mapped to task coords via the affine `cursor_traj → cursor_screenvid_px` fit).
   Run over many sessions with `run_all_visualizations.sh`.

Key frames: JSON `trajectory` = recorded cursor path in **task coords**;
`cursor_traj_*` in the aligned CSV is already task coords; gaze is only in
screen px → mapped to task via the affine fit (≈1–2 px residual).

---

## Workflow C — Calibration transfer (before→after) + 3-panel

Neon's manual gaze-offset calibration in Pupil Cloud is applied **per recording**.
When only the *calibration* recording was corrected (the *task* recording's raw
gaze is byte-identical before/after), transfer the correction to the task:

1. `apply_calibration_transfer.py` — on the **calibration** recording, fit a
   **homography** `T: pre_ref → post_ref` (before- vs after-calibration gaze on
   the reference image, matched by timestamp). Apply `T` to the **task**
   recording's reference gaze, then re-project to the screen through the task's
   own reference→screen homography (recovered exactly from its `ref↔transf`
   columns). Output = corrected task gaze CSV.
   ```bash
   python3 apply_calibration_transfer.py \
     --cal_pre  "<...>/<P>_calibration_gaze.csv" \
     --cal_post "<after enrichment>/gaze.csv" --cal_post_recording <cal_rec_id> \
     --task     "<...>/<P>_task_gaze.csv" \
     --out      "<...>/<P>_task_gaze_calibrated.csv"
   ```
   Sanity: "raw pre→post shift" should be sizeable (Beichen ≈126 px in
   reference space) — if ~0 the calibration recording wasn't recalibrated.
2. `compose_3panel.py` — render the 3-panel with the corrected gaze. Pass
   `--scene_gaze_csv <raw recording gaze.csv>` to draw the red gaze circle on
   the **Neon (left)** panel too, like the tool's merged video.
   ```bash
   python3 compose_3panel.py \
     --scene_video "<...>/<scene>.mp4"        --scene_start_ns <Neon begin ns> \
     --reference_image "<enrichment>/reference_image.jpeg" \
     --screen_video "<...>/<P>_task_screen.mov" --start_video_ns <Neon begin + offset ns> \
     --gaze_csv "<...>/<P>_task_gaze_calibrated.csv" \
     --scene_gaze_csv "<...>/<recording>/gaze.csv" \
     --out_video "<...>/<P>_task_3panel_calibrated.mp4" --height 720 --white 148
   ```
   `scene_start_ns` = Neon `recording.begin` (ns). `start_video_ns` = that +
   `(screen_start − Neon_begin)` in ns = screen frame-0 wall time.
   **Spot-check first** with `--start_s 60 --duration_s 20` (a 20 s clip renders
   in <1 min) before the full pass. Full task renders are ~15–30 min (≈20k
   frames, two video decodes, Python compositing — no progress bar).

Caveat: `T` is measured from the calibration recording's geometry, so the
transfer assumes comparable screen/camera pose between calibration and task
(they can be hours apart). The 9-dot layout (`calibration_9point_sequential.html`)
is the ground-truth cross-check.

---

## Gotchas & conventions (these bite every time)

- **iCloud eviction** — reads fail with `Errno 35`/`Errno 60` ("Resource
  deadlock avoided" / "Operation timed out") when a file is a dataless iCloud
  placeholder. Fix: Finder → right-click → **Download Now**, or `brctl download
  "<path>"`. Durable fix: turn off iCloud "Optimize Mac Storage", or keep the
  data in a non-synced folder (`~/Downloads`, `~/pilot_data` — NOT the synced
  Desktop/Documents). In the sandbox these files simply can't be materialized;
  do those runs on the Mac.
- **Screen-recording filenames** contain a narrow no-break space `U+202F` before
  `AM`/`PM` (`…30 ⎵ PM.mov`), so a typed normal-space path "doesn't exist".
  Rename with a glob: `mv Screen\ Recording*PM.mov <P>_task_screen.mov`.
- **`pl-dynamic-rim` output** — pass `--out_csv_path` / `--out_video_path` or it
  opens a GUI save dialog and writes UUID files to `~`.
- **Sandbox is slow at 4K video** (~0.2–0.4× realtime); full renders belong on
  the Mac. Use `--start_s/--duration_s` (compose) or short segments to verify.
- **Even dimensions** — H.264/yuv420p need even width/height; the scripts round
  down. (Historic `libx264` failure was an odd 8437-px canvas.)
- **Filenames with spaces / `&`** — always quote paths in commands.

---

## Session parameters

RIM sync: `Neon_begin` from enrichment `sections.csv` (`section start time [ns]`);
`screen_start` measured/noted; `start_video_ns = Neon_begin + (screen_start −
Neon_begin)`.

| Session | recording id | Neon begin [ns] | screen start (local) | offset | notes |
|---|---|---|---|---|---|
| P152244 (steering) | 0f7c3ec2 | — | 2026-08-11 15:31:52.432 | clock_offset 0.117 | screen 3840×2248, cursor_by=1717 |
| P161909 (steering) | e593c5b6 | — | 2026-08-11 16:19:32.134 | clock_offset 0.003 | cursor_by=1610 |
| task-label motivation | 96fe3cf3 | 1786578505666000000 | 2026-08-12 17:48:26 | +0.334 s | start_video_ns 1786578506000000000 |
| task-label awareness | 110bacfc | 1786578954727000000 | 2026-08-12 17:55:55 | +0.273 s | start_video_ns 1786578955000000000; screen near-static |
| Beichen calibration | b0d79dbd | 1786727246514000000 | 2026-08-14 11:07:26.834 | +0.32 s | 9-point; recalibrated (~22 px scene offset) |
| Beichen task | e3d239d3 | 1786748790188000000 | 2026-08-14 17:06:30.937 | +0.749 s | start_video_ns 1786748790937000000; NOT recalibrated → use transfer |
| Astrid calibration | 463f6ba4 | 1786729359258000000 | 2026-08-14 11:42:39.551 | +0.29 s | 9-point |
| Astrid task | 18ca845a | 1786729410244000000 | 2026-08-14 11:43:30.543 | +0.299 s | start_video_ns 1786729410543000000 |

Calibration-transfer results: **Beichen** — pre→post reference shift ≈126 px, `T`
residual ≈10 px, task screen gaze moved ≈80 px. **Astrid** — pending (run on the
Mac once files are downloaded).

Folder convention: `official pilot/<before|after> calibration/<Person>/…` with
mapped outputs named `<Person>_<calibration|task>_{gaze.csv,overlay.mp4}` and
corrected task gaze `<Person>_task_gaze_calibrated.csv`.
