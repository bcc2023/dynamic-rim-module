#!/usr/bin/env python3
"""
pilot_columns.py — the 38-column schema of the pilot task-aligned analysis CSV
(Participant C, C_task_aligned_analysis.csv). Every task_aligned_analysis.csv we
produce must have exactly these columns, in this order, computed the same way.
Embedded here so the pipeline no longer depends on where the pilot file lives.
"""
PILOT_COLUMNS = [
    'neon_timestamp_ns',
    'neon_time_s',
    'mac_time_s',
    'gaze_ref_x_px',
    'gaze_ref_y_px',
    'gaze_transf_x_px (calibrated)',
    'gaze_transf_y_px (calibrated)',
    'gaze_transf_x_norm (calibrated; = gaze_transf_x_px / screen_width)',
    'gaze_transf_y_norm (calibrated; = gaze_transf_y_px / screen_height)',
    'fixation_id',
    'saccade_id',
    'blink_id',
    'cursor_traj_x (JSON trajectory; task units)',
    'cursor_traj_y (JSON trajectory; task units)',
    'cursor_screen_x (JSON screenTrajectory; screen coords)',
    'cursor_screen_y (JSON screenTrajectory; screen coords)',
    'cursor_speed',
    'cursor_nearest_mac_ms',
    'cursor_interp_gap_ms (ms; = gaze time - nearest logged cursor-sample time)',
    'difficulty_curvatureChangeRate',
    'transition_taskX (task units; x where a 2-segment tunnel switches; blank if none)',
    'transition_screenX (screen coords; x where a 2-segment tunnel switches; blank if none)',
    'trial_id',
    'condition',
    'constrained',
    'tunnelType',
    'condition_json',
    'in_trial',
    'cursor_screenvid_x_px (screen coords mapped to screen-video pixels; = cursor_scale * cursor_screen_x + cursor_bx)',
    'cursor_screenvid_y_px (screen coords mapped to screen-video pixels; = cursor_scale * cursor_screen_y + cursor_by)',
    'cursor_screenvid_x_norm (= cursor_screenvid_x_px / screen_width)',
    'cursor_screenvid_y_norm (= cursor_screenvid_y_px / screen_height)',
    'gaze_cursor_dist_px (px; Euclidean distance between gaze_transf_*_px and cursor_screenvid_*_px)',
    'gaze_cursor_signed_px (px; = gaze_cursor_dist_px signed by lead direction, + = gaze ahead of cursor)',
    'gaze_lead_signed (task units; = s(gaze) - s(cursor) along centerline, + = gaze ahead)',
    'json_local_curvature (difficulty.curvature)',
    'json_local_width (difficulty.tunnelWidth)',
    'time_to_catch_s (= gaze_lead_signed / cursor_speed; seconds; time for the cursor to reach the current gaze position; NaN when cursor_speed ~0)',
]
