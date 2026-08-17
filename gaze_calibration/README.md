# gaze_calibration

A general toolkit for **gaze on a screen** with **Pupil Labs Neon** glasses and
**Pupil Cloud Reference Image Mapper (RIM)** enrichments: map gaze onto a screen
recording, **check and transfer the gaze calibration** (9-point validation +
before/after offset transfer), render QA overlay videos, and align/analyze the
result. It works for any on-screen task; the `analyze` command ships a
steering-task example you can replace with your own analysis.

> Full operator guide, per-session parameters, and gotchas live in
> **[WORKFLOWS.md](WORKFLOWS.md)** — read that to run any workflow end to end.

## Pipeline

```
Neon recording ─┐
                ├─(Pupil Cloud RIM)→ gaze on reference image
screen capture ─┘
      │
      ▼  pl-dynamic-rim (external fork)        gaze on the screen video
      ▼  cli.py align                + cursor  → *_aligned.csv
      ▼  cli.py analyze              distance / difficulty plots
      ▼  cli.py calib-transfer       apply calibration to the task
      ▼  cli.py compose3 / overlay   QA videos (gaze overlays)
```

## Install

```bash
conda create -n pupil-cloud-local python=3.11 -y
conda activate pupil-cloud-local

# analysis/overlay deps
pip install -r requirements.txt

# the scene->reference->screen mapping tool (separate, editable fork)
pip install "pupil-labs-dynamic-rim @ git+https://github.com/yurahwang97/dynamic-rim-module.git@a0179667"
python patch_all.py --path dynamic-rim-module   # apply local bug fixes once
```

## Usage

One entry point with subcommands (each also runnable as its own script):

```bash
python cli.py <command> [options]
python cli.py <command> --help
```

| command | does |
|---|---|
| `align` | Align the experiment-JSON cursor onto the mapped gaze → `*_aligned.csv` (run after `pl-dynamic-rim`). |
| `analyze` | Distance, per-trial traces, and steering-difficulty plots (Fitts `ID` for pointing; `ID_W` / `ID_K` + gaze lead for steering). |
| `calib-transfer` | Transfer a manual gaze-offset calibration from the calibration recording onto the task recording (homography in reference space). |
| `compose3` | `Neon | Reference | Screen` 3-panel video, gaze circle on all three panels (streaming). |
| `overlay` | Screen video + gaze circle, brightened (single panel). |
| `checksync` | Report a screen↔Neon offset from container `creation_time` (coarse ±0.5 s). |

The mapping step itself (`pl-dynamic-rim`) is the external fork above; everything
downstream is in this repo.

## Example

```bash
# 1) map gaze onto the screen recording (external tool) — see WORKFLOWS.md §A
pl-dynamic-rim --rim_folder_path ... --raw_folder_path ... --screen_video_path ... \
  --audio No_Audio --screen_start_wallclock "YYYY-MM-DD HH:MM:SS.mmm" \
  --out_csv_path mapped_gaze.csv --out_video_path mapped_overlay.mp4

# 2) align cursor + gaze, then analyze
python cli.py align   --gaze_csv mapped_gaze.csv --experiment_json steering_*.json ... --out_csv S_aligned.csv
python cli.py analyze --aligned_csv S_aligned.csv --experiment_json steering_*.json

# 3) transfer a calibration onto the task, then render the 3-panel QA video
python cli.py calib-transfer --cal_pre C_calibration_gaze.csv \
  --cal_post after/gaze.csv --cal_post_recording <rec_id> \
  --task C_task_gaze.csv --out C_task_gaze_calibrated.csv
python cli.py compose3 --scene_video ... --reference_image ... --screen_video ... \
  --gaze_csv C_task_gaze_calibrated.csv --scene_gaze_csv <recording>/gaze.csv \
  --scene_start_ns <ns> --start_video_ns <ns> --out_video C_task_3panel.mp4
```

See **[WORKFLOWS.md](WORKFLOWS.md)** for the sync-parameter derivation, the
per-session table, and the recurring gotchas (iCloud eviction, the `U+202F`
screen-recording filename gremlin, always passing `--out_*_path`, etc.).

## Repo layout

- `cli.py` — unified CLI dispatcher.
- `cursor_gaze_pipeline.py`, `analyze_aligned.py`, `apply_calibration_transfer.py`,
  `compose_3panel.py`, `screen_gaze_overlay.py`, `check_sync.py` — the tools.
- `patch_all.py` — bug-fix patcher for the external mapping fork.
- `run_*.sh` — batch-run templates for a study.
- `WORKFLOWS.md` — the operator guide + parameters.
