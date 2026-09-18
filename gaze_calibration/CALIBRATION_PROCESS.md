# Cursor‑steering gaze calibration — end‑to‑end process

Maps Neon + RIM gaze onto the screen recording, calibrates it, and produces the
task‑aligned analysis CSV (the gaze‑lead DV) plus a 3‑panel QC video.
Per‑participant folders sit under a batch folder with a `rim_manifest.csv`
(one `task` + one `calibration` row per person).

## Stages (per participant)

1. **9‑point calibration (spatial)** — `apply_9point_calibration.py` measures the
   gaze during each of the 9 known on‑screen dots of the calibration recording
   (ground truth) and fits gaze→screen, then applies it to the task gaze.
   **Model = HOMOGRAPHY by default** (reuses `fit_homography` from
   `apply_calibration_transfer.py`): the physically correct model for a constant
   scene‑camera offset seen through a flat screen. Chosen over the old poly2 because
   leave‑one‑out on 9 dots showed poly2 was overfitting (true held‑out error 3–4× its
   in‑sample RMS); the homography generalizes better (p06new 209 vs 239px, p08new
   103 vs 135px). Falls back to bilinear/affine only if the dot grid is too sparse.
   Per‑dot residual after this step (~100–200px) is gaze noise, not a model failure —
   the parked self‑cal (step 3, hundreds of samples) tightens it on the task.
2. **Align** — `cursor_gaze_pipeline.py` maps RIM gaze → screen px via the cursor
   map and joins the steering‑JSON cursor at each gaze timestamp → `<P>_task_aligned.csv`.
3. **Parked‑target self‑cal** — `apply_selfcal.py` refines the mapping on
   parked‑on‑target fixations only (lead‑safe: it does not touch the moving‑cursor lead).
   Steps 1–3 are wrapped by **`full_selfcal_pipeline.py`** (one clean pass; also writes the analysis CSV).
4. **Cursor clock‑offset correction** — see below. THE STEP THAT WAS MISSING.
5. **Analysis CSV** — pilot 38‑col format with `gaze_lead_signed`, `gaze_cursor_dist_px`,
   `time_to_catch_s`, `json_local_curvature/width`, `saccade_id`, `blink_id`.
   The 38‑column schema is embedded in `pilot_columns.py` (no dependency on where the
   pilot CSV lives — it moved once and silently broke the analysis step).
6. **3‑panel video** — `run_calibrated_pipeline.py` (Neon | Reference | Screen),
   gaze burned in; optional cyan cursor overlay.

## Step 4 — cursor clock‑offset correction (IMPORTANT for the DV)

The steering‑experiment JSON logs the cursor on the Mac clock. It is aligned to
the Neon clock by `clock_offset_s` (Neon − Mac), stored in each `<P>_task_cursor_map.json`.
The original per‑session estimate (an on‑screen‑clock method) was off by ~0.15–0.30 s,
so the JSON cursor lagged the REAL on‑screen cursor during motion. This silently
biased the gaze‑lead DV (it made gaze look *ahead* when it was *behind*). It does
NOT show at parked moments, so calibration accuracy / self‑cal looked fine.

**Measure** the true offset from the video (cross‑correlate the real cursor's motion
against the JSON cursor):

    python measure_lag.py "<batch>/<Person>" "<screen_start_wallclock>"
    # prints:  <Person> window@<t>s valid=<n> lag L=<L>s corr=<c>
    # corrected clock_offset_s = original_clock_offset_s + L      (L is negative)

**Apply** (re‑align with the corrected offset + rebuild the analysis CSV; gaze
untouched, no re‑render needed for the numbers):

    python recompute_dv.py --person <P> --base "<batch>" --pid <pXX> \
        --raw_rel "<task raw rel>" --collection "<task_aligned_all>" --clock_offset_s <corrected>

**CAVEAT — measure_lag needs a millisecond screen_start_wallclock.** It cross‑correlates
the JSON cursor against the video cursor in recording‑relative time, so any error in the
wallclock (e.g. one taken only to the second, from a .mov's creation_time) feeds straight
into L. For such recordings ALWAYS validate the offset directly against the real on‑screen
crosshair before trusting it: on a mid‑trial 4K frame, overlay the aligned cursor at several
candidate offsets (compute each as an interp lookup on an existing aligned CSV, shifting the
gaze‑time by the offset delta) and pick the one whose ring sits on the crosshair. In practice
the cursor map's own fitted clock_offset_s is often already correct; measure_lag can then
over‑correct. Example: Jiashu — measure_lag said L=−0.50 (→ −0.68) but the frame sweep showed
the correct offset is ≈ −0.20 (the cursor map's own −0.18), i.e. no correction was needed.

## Videos (for visual QC of gaze vs cursor)

Videos burn the gaze + the REAL recorded cursor, so they never used the JSON cursor —
the cursor fix does not change them. To also draw the corrected cursor as a cyan ring
(so gaze‑vs‑cursor is easy to see), re‑render with the corrected offset:

    caffeinate -dimsu python run_calibrated_pipeline.py --person <P> --base . \
        --skip-existing --no_vfix --clock_offset_s <corrected> --cursor_overlay

Add `--start_s <t> --duration_s <n>` for a quick preview clip.

## Measured values (Umich cursor batches, 2026‑08‑27)

| pid | person  | original clock_offset_s | measured lag L (s) | corrected clock_offset_s | corr |
|-----|---------|-------------------------|--------------------|--------------------------|------|
| p01 | Yanran  | +0.12 | −0.28 | −0.16 | 0.85 |
| p02 | Zihan   | +0.20 | −0.28 | −0.08 | 0.89 |
| p03 | dewen   | +0.40 | −0.30 | +0.10 | 0.89 |
| p04 | haoyang | +0.20 | −0.16 | +0.04 | 0.85 |
| p05 | Sean    | +0.20 | −0.18 | +0.02 | 0.89 |
| p06 | Yue     | −0.18 | −0.24 | −0.42 | 0.83 |
| p07 | Kai     | +0.12 | −0.22 | −0.10 | 0.86 |
| p08 | Beichen | +0.06 | −0.18 | −0.12 | 0.73 |
| p09 | Yura    | +0.12 | −0.22 | −0.10 | 0.89 |

## Step 4b — gaze‑vs‑screen shift (`--gaze_shift_s`)

Separate from the cursor timing: the gaze is placed against the screen from the raw
Mac wallclock and reads ~0.2 s LATE (behind the cursor) for everyone — see the open
issue below. `--gaze_shift_s <s>` (on `compose_3panel.py`, `run_calibrated_pipeline.py`,
`cursor_gaze_pipeline.py`, `recompute_dv.py`) advances the gaze vs the cursor by `s`
seconds so it leads. `+ = gaze leads`. It changes BOTH the video and the DV when passed
to recompute_dv (cursor‑matching only; the row's neon time / saccade lookup is untouched).
Validated for Jiashu at **+0.25 s** by sweeping the gaze on real frames until it sat ahead
on the tunnel. THIS SETS THE GAZE‑LEAD RESULT, so apply one agreed value consistently
across all participants and confirm it against an independent latency anchor before writeup.

## Open issue — residual gaze lag (~0.17 s)

After the cursor fix, gaze lags the cursor by ~0.17 s (±0.05) for ALL 9 participants
(implied `tau`, where directional lead ≈ −tau·speed). Because it is consistent in
sign and size across everyone, it is likely a systematic gaze‑vs‑screen timing offset
(a `start_video_ns` error, e.g. a constant screen‑capture latency) rather than real
reactive tracking — though ~170 ms is also a plausible visuomotor tracking delay.
Not yet resolved; needs an independent gaze↔screen time reference to separate the two.
The same pipeline produced the pilot numbers, so the pilot's "gaze leads" result
should be re‑checked against the real on‑screen cursor.
