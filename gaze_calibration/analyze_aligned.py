#!/usr/bin/env python3
"""
Analysis / visualization of a p*_aligned.csv (output of cursor_gaze_pipeline.py).

Built up step by step. Run per session:
    python3 analyze_aligned.py --aligned_csv "P152244_aligned.csv" --out_csv "P152244_analysis.csv"

Coordinates used for gaze-cursor comparison are both in screen-video pixels:
    gaze:   gaze_transf_x_px,      gaze_transf_y_px
    cursor: cursor_screenvid_x_px, cursor_screenvid_y_px

------------------------------------------------------------------------------
STEP 1 — Euclidean distance between cursor and gaze, at each in-trial timestamp.
------------------------------------------------------------------------------
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tunnel_centerline import centerline_for_condition, project_arclen  # noqa: E402


# --- path length A (arc-length corrected), from the attached length.py --------
def calculate_path_length_from_curvature(points, curvatures):
    """Cumulative path length given 2D points and per-point curvatures.
    Each segment is treated as a circular arc of the averaged curvature."""
    points = np.asarray(points, float)
    curvatures = np.asarray(curvatures, float)
    chord_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    avg_curv = 0.5 * (curvatures[:-1] + curvatures[1:])
    seg = np.zeros_like(chord_lengths)
    for i in range(len(chord_lengths)):
        kappa, chord = avg_curv[i], chord_lengths[i]
        if abs(kappa) < 1e-6:
            seg[i] = chord
        else:
            seg[i] = (1.0 / kappa) * 2.0 * np.arcsin(np.clip(kappa * chord / 2.0, -1.0, 1.0))
    return np.concatenate(([0.0], np.cumsum(seg)))


def step1_gaze_cursor_distance(df):
    """Add gaze-cursor distance columns (screen-video pixels), in-trial only:
      gaze_cursor_dist_px    : Euclidean distance (unsigned magnitude)
      gaze_cursor_signed_px  : same magnitude, signed by lead direction --
                               POSITIVE when gaze is ahead of the cursor
                               (further right, larger x = task progress
                               direction), NEGATIVE when the cursor is ahead.
    """
    gx = df["gaze_transf_x_px"]
    gy = df["gaze_transf_y_px"]
    cx = df["cursor_screenvid_x_px"]
    cy = df["cursor_screenvid_y_px"]
    dist = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
    valid = df["in_trial"].astype(bool) & np.isfinite(dist)
    df["gaze_cursor_dist_px"] = np.where(valid, dist, np.nan)
    # sign: + gaze ahead (gaze_x > cursor_x), - cursor ahead
    lead = np.sign(gx - cx)
    df["gaze_cursor_signed_px"] = np.where(valid, dist * lead, np.nan)
    return df


def _presentations_in_order(df, experiment_json):
    """List of (index, trial_id, round, condition_desc, boolean-mask) — one per
    trial PRESENTATION, in the order they actually happened.

    Uses the experiment JSON as the source of truth: each trialData entry is a
    distinct presentation (a trial can be shown more than once, e.g. round 1 &
    round 2, sharing the same trial_id). Aligned rows are matched to a
    presentation by its Mac-clock time window (via the CSV's mac_time_s).
    Falls back to grouping by trial_id (ordered by first appearance) if no JSON.
    """
    out = []
    if experiment_json:
        d = json.load(open(experiment_json))
        mac = df["mac_time_s"].to_numpy(float)
        intrial = df["in_trial"].to_numpy(bool)
        # sequential trial number = order of first appearance of each trial_id
        # (trial_id is a shuffled condition id; this is the "Trial N/73" counter)
        ordinal = {}
        for tr in d.get("trialData", []):
            tid = tr.get("trial_id")
            if tid not in ordinal:
                ordinal[tid] = len(ordinal) + 1
        n_trials = len(ordinal)
        for i, tr in enumerate(d.get("trialData", [])):
            ts = tr.get("timestamps") or []
            if len(ts) < 2:
                continue
            lo, hi = ts[0] / 1000.0, ts[-1] / 1000.0
            mask = intrial & (mac >= lo) & (mac <= hi)
            if mask.sum() < 2:
                continue
            tid = tr.get("trial_id")
            out.append((i + 1, ordinal.get(tid), n_trials, tid, tr.get("round"),
                        (tr.get("condition") or {}).get("description", ""), mask))
    else:
        order = (df[df["in_trial"]].groupby("trial_id")["neon_time_s"].min()
                 .sort_values().index)
        n_trials = len(order)
        for j, tid in enumerate(order):
            mask = (df["in_trial"] & (df["trial_id"] == tid)).to_numpy(bool)
            desc = str(df.loc[mask, "condition"].iloc[0]) if mask.any() else ""
            out.append((j + 1, j + 1, n_trials, tid, None, desc, mask))
    return out


# --- 4-block per-trial segmentation (start dot / constrained / unconstrained /
#     target dot), from the experiment canvas design -----------------------------
_TPY = {"top": 0.05, "middle": 0.13, "bottom": 0.21}
_BLOCK_COL = {"start": "#1f77b4", "target": "#ff7f0e",
              "constrained": "#2ca02c", "unconstrained": "#d62728"}


def _cond_lookup(experiment_json):
    if not experiment_json:
        return {}
    d = json.load(open(experiment_json))
    return {(tr.get("trial_id"), tr.get("round")): (tr.get("condition") or {})
            for tr in d.get("trialData", [])}


def _dot_target_radius(c):
    """Target-dot radius from the trial condition (computeTargetRadius port)."""
    tt = (c or {}).get("tunnelType")
    if tt in ("wide_to_narrow", "narrow_to_wide"):
        return 0.5 * c.get("segment2Width", 0.02)
    if tt == "straight":
        return c.get("radiusRatio", 0.5) * c.get("tunnelWidth", 0.02)
    if tt in ("unconstrained_pointing", "constrained_to_unconstrained"):
        return c.get("targetRadius", 0.01)
    return 0.5 * (c or {}).get("tunnelWidth", 0.02)


def _dot_block_spans(g, cond):
    """Classify a trial's in-order samples into 4 contiguous blocks: on the START
    dot (r=0.008), constrained / unconstrained while traversing, on the TARGET dot
    (computeTargetRadius). Start/target centred on the design positions. Returns
    (spans list of (i0,i1,key), target_radius)."""
    cur = g[["cursor_traj_x", "cursor_traj_y"]].to_numpy(float)
    n = len(cur)
    con = (g["constrained"].astype(str).to_numpy() if "constrained" in g.columns
           else np.array(["constrained"] * n))
    start = cur[0]
    tt = (cond or {}).get("tunnelType")
    if tt in ("unconstrained_pointing", "constrained_to_unconstrained"):
        tgt = np.array([cond.get("distance", cur[-1, 0]),
                        _TPY.get(cond.get("targetPosition"), cur[-1, 1])])
    else:
        tgt = cur[-1]
    Rt = _dot_target_radius(cond or {})
    on_s = np.hypot(*(cur - start).T) <= 0.008
    on_t = np.hypot(*(cur - tgt).T) <= Rt
    k = 0
    while k < n and on_s[k]:
        k += 1
    m = n
    while m > 0 and on_t[m - 1]:
        m -= 1
    spans = []
    if k > 0:
        spans.append((0, k - 1, "start"))
    j = k
    while j < m:
        b = j
        while j < m and con[j] == con[b]:
            j += 1
        spans.append((b, j - 1, con[b]))
    if m < n:
        spans.append((m, n - 1, "target"))
    return spans, Rt


def _draw_dot_blocks(ax, t, spans, Rt):
    """Draw the 4 shaded blocks + on-block constrained/unconstrained labels;
    return legend handles (start / constrained / unconstrained / target)."""
    from matplotlib.patches import Patch
    n = len(t); bl = ax.get_xaxis_transform()
    for i0, i1, key in spans:
        if i1 < i0:
            continue
        x0 = t[i0] - (t[i0] - t[i0 - 1]) / 2 if i0 > 0 else t[0]
        x1 = t[i1] + (t[i1 + 1] - t[i1]) / 2 if i1 < n - 1 else t[-1]
        ax.axvspan(x0, x1, color=_BLOCK_COL.get(key, "#888"), alpha=0.20, lw=0)
        if key in ("constrained", "unconstrained"):
            ax.text((x0 + x1) / 2, 0.95, key, transform=bl, ha="center", va="top",
                    color=_BLOCK_COL[key], fontsize=9, fontweight="bold")
    # constrained/unconstrained are labelled on their blocks; the legend only
    # needs the two dot colours (their blocks are too narrow to label on-plot).
    return [Patch(facecolor=_BLOCK_COL[k2], alpha=0.35, label=lab) for k2, lab in
            [("start", "on start dot (r=0.008)"), ("target", f"on target dot (r={Rt:g})")]]


def step2_plot_per_trial(df, out_pdf, experiment_json=None):
    """One page per trial PRESENTATION (in chronological order): gaze-cursor
    distance vs time. Shade/label the constrained vs unconstrained segments; if
    a presentation has both, draw a vertical divider at the transition and label
    each side; if only one condition, label the whole plot with it.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    has_con = "constrained" in df.columns
    df = df.sort_values("neon_time_s").reset_index(drop=True)
    presentations = _presentations_in_order(df, experiment_json)
    condmap = _cond_lookup(experiment_json)
    n = 0
    with PdfPages(out_pdf) as pdf:
        for idx, ordn, n_trials, tid, rnd, desc, mask in presentations:
            g = df[mask].sort_values("neon_time_s")
            if len(g) < 2:
                continue
            t = g["neon_time_s"].to_numpy(float)
            t = t - t[0]                                   # seconds from trial start
            y = g["gaze_cursor_signed_px"].to_numpy(float)  # signed: + gaze ahead

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(t, y, ".", color="tab:blue", ms=3)   # points, no connecting line
            ax.axhline(0, color="gray", lw=0.8)           # gaze-ahead / cursor-ahead line
            ax.set_xlabel("time since trial start (s)")
            ax.set_ylabel("signed gaze-cursor distance (px)\n(+ gaze ahead  /  − cursor ahead)")
            ord_s = f"{int(ordn)}/{n_trials}" if (ordn is not None and ordn == ordn) else "?"
            rnd_s = f"  round {int(rnd)}" if (rnd is not None and rnd == rnd) else ""
            tid_s = int(tid) if tid == tid else "?"        # NaN-safe
            ax.set_title(f"Trial {ord_s}{rnd_s}   —   {desc}   (id {tid_s})", fontsize=9)
            ax.margins(x=0)

            spans, Rt = _dot_block_spans(g, condmap.get((tid, rnd), {}))
            handles = _draw_dot_blocks(ax, t, spans, Rt)
            ax.legend(handles=handles, fontsize=7.5, loc="upper right", framealpha=0.95)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            n += 1
    print(f"STEP 2 — wrote {out_pdf}: {n} trial presentations "
          f"(chronological order, one per page)")


# --- per-segment difficulty definitions ---------------------------------------
def _constrained_AW(tr):
    """Steering difficulty A/W for the CONSTRAINED segment of one presentation.
    A = path length (arc-length from curvature, length.py), W = tunnel width.
    Returns {'A','W','difficulty'} or None if no usable constrained segment."""
    traj = tr.get("trajectory") or []
    con = tr.get("constrained?") or []
    dif = tr.get("difficulty") or []
    n = min(len(traj), len(con)) if con else len(traj)
    if n < 2:
        return None
    pts = np.array([[traj[i]["x"], traj[i]["y"]] for i in range(n)], float)
    curv = np.array([(dif[i] or {}).get("curvature", 0.0) if i < len(dif) else 0.0
                     for i in range(n)], float)
    width = np.array([(dif[i] or {}).get("tunnelWidth", np.nan) if i < len(dif) else np.nan
                      for i in range(n)], float)
    lab = np.array(con[:n]) if con else np.array(["constrained"] * n)
    idx = np.where(lab == "constrained")[0]
    if len(idx) < 2:
        return None
    lo, hi = idx[0], idx[-1] + 1               # constrained portion (contiguous)
    A = calculate_path_length_from_curvature(pts[lo:hi], curv[lo:hi])[-1]
    seg_w = width[lo:hi]
    w = np.nanmedian(seg_w) if not np.all(np.isnan(seg_w)) else np.nan
    if not (w and np.isfinite(w) and w != 0):
        return None
    return {"A": A, "W": w, "difficulty": A / w}


def _unconstrained_ID(tr):
    """Fitts' law index of difficulty for the UNCONSTRAINED (free-pointing)
    segment of one presentation, Shannon formulation (MacKenzie):

        ID = log2(1 + D / W)

      W = 2 * targetRadius  (target diameter, the width along the movement axis)
      D = pointing amplitude:
          unconstrained_pointing        -> D = condition.distance
          constrained_to_unconstrained  -> D = distance - transition_point.taskX
                                           (only the free-pointing tail after the
                                            constraint ends; the constrained half
                                            is excluded)
    Returns {'D','W','difficulty'} or None if this presentation has no
    unconstrained segment / lacks the geometry."""
    con = tr.get("constrained?") or []
    if not any(x == "unconstrained" for x in con):
        return None
    c = tr.get("condition") or {}
    r = c.get("targetRadius")
    dist = c.get("distance")
    if r is None or dist is None:
        return None
    W = 2.0 * r
    tx = (c.get("transition_point") or {}).get("taskX")
    if c.get("tunnelType") == "constrained_to_unconstrained" and tx is not None:
        D = dist - tx                          # free-pointing amplitude only
    else:
        D = dist                               # fully unconstrained: whole move
    if not (W > 0 and D > 0):
        return None
    return {"D": D, "W": W, "difficulty": float(np.log2(1.0 + D / W))}


def _segment_difficulty(tr, segment):
    r = _constrained_AW(tr) if segment == "constrained" else _unconstrained_ID(tr)
    return r["difficulty"] if r else None


def step3_difficulty_table(experiment_json):
    """One row per presentation-segment with its difficulty. Constrained rows
    carry A/W; unconstrained rows carry the Fitts ID = log2(1+D/W)."""
    d = json.load(open(experiment_json))
    rows = []
    for tr in d.get("trialData", []):
        c = _constrained_AW(tr)
        if c:
            rows.append({"trial_id": tr.get("trial_id"), "round": tr.get("round"),
                         "segment": "constrained", "A": c["A"], "D": np.nan,
                         "W": c["W"], "difficulty": c["difficulty"]})
        u = _unconstrained_ID(tr)
        if u:
            rows.append({"trial_id": tr.get("trial_id"), "round": tr.get("round"),
                         "segment": "unconstrained", "A": np.nan, "D": u["D"],
                         "W": u["W"], "difficulty": u["difficulty"]})
    return pd.DataFrame(rows)


def _iter_presentation_segments(df, experiment_json, segment):
    """Yield (difficulty, mean_dist) for each trial PRESENTATION that contains a
    `segment` ('constrained'/'unconstrained') portion: difficulty from that
    presentation's condition, mean gaze-cursor distance over only that
    presentation's rows labelled `segment`. One pair per presentation (so the
    two rounds of a trial are two separate points)."""
    d = json.load(open(experiment_json))
    mac = df["mac_time_s"].to_numpy(float)
    intrial = df["in_trial"].to_numpy(bool)
    con_col = (df["constrained"].astype(str).to_numpy()
               if "constrained" in df.columns else None)
    dist = df["gaze_cursor_dist_px"].to_numpy(float)
    for tr in d.get("trialData", []):
        ts = tr.get("timestamps") or []
        if len(ts) < 2:
            continue
        lo, hi = ts[0] / 1000.0, ts[-1] / 1000.0
        mask = intrial & (mac >= lo) & (mac <= hi)
        if con_col is not None:
            mask = mask & (con_col == segment)
        if mask.sum() < 1:            # keep any presentation with >=1 mapped sample
            continue                  # (a single sample's value is its own mean)
        diff = _segment_difficulty(tr, segment)
        if diff is None or not np.isfinite(diff):
            continue
        md = np.nanmean(dist[mask])
        if not np.isfinite(md):
            continue
        yield diff, md


def step_plot_difficulty(df, experiment_json, segment, out_png):
    """Aggregate plot for one condition: x = difficulty, y = mean gaze-cursor
    distance over that presentation's timestamps in that condition. One point
    per presentation-segment (scatter only, no fit line). The difficulty formula
    is printed on the plot."""
    import matplotlib.pyplot as plt

    pairs = list(_iter_presentation_segments(df, experiment_json, segment))
    xs = np.array([p[0] for p in pairs], float)
    ys = np.array([p[1] for p in pairs], float)
    color = "#2ca02c" if segment == "constrained" else "#d62728"

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, s=28, color=color, alpha=0.7, edgecolor="k", linewidth=0.3)
    title = f"{segment.capitalize()} condition: gaze-cursor distance vs difficulty"

    if segment == "constrained":
        formula = r"$\mathrm{ID} = A\,/\,W$"
        xlabel = r"difficulty   $\mathrm{ID}=A/W$   (path length / tunnel width)"
        caption = (
            "Each point = one trial presentation's constrained segment: mean gaze-cursor distance vs difficulty.\n"
            "Difficulty  ID = A / W   (higher = harder).   A = steering path length (from JSON trajectory[] + difficulty[].curvature),\n"
            "W = tunnel width = median of JSON difficulty[].tunnelWidth."
        )
    else:
        formula = r"$\mathrm{ID} = \log_2\!\left(1 + D/W\right)$"
        xlabel = r"difficulty   $\mathrm{ID}=\log_2(1+D/W)$   (Fitts' law, Shannon form)"
        caption = (
            "Each point = one trial presentation's unconstrained (free-pointing) segment.\n"
            "Difficulty  ID = log2(1 + D/W)  (Shannon / MacKenzie; higher = harder).\n"
            "W = target width = 2 × JSON condition.targetRadius.   D = pointing amplitude = JSON condition.distance\n"
            "(constrained→unconstrained trials: D = condition.distance − condition.transition_point.taskX)."
        )
    # formula box, prominent, top-left inside the axes
    ax.text(0.03, 0.97, formula, transform=ax.transAxes, va="top", ha="left",
            fontsize=13, color=color,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color, alpha=0.9))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mean gaze-cursor distance (px)")
    ax.set_title(title, fontsize=10)
    n_lines = caption.count("\n") + 1
    bottom = 0.045 + 0.036 * n_lines          # room scales with caption length
    fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=7,
             style="italic", color="0.35")
    fig.tight_layout(rect=[0, bottom, 1, 1])
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"STEP — wrote {out_png}: {len(xs)} {segment} presentation-segments")


# ============================================================================
# NEW constrained difficulty (Curves Ahead, CHI 2025):
#   ID_W = ∫ ds/W(s)         steering-law part (width)
#   ID_K = ∫ |κ(s)| ds       curvature part (total absolute turning angle)
# plotted against GAZE LEAD = signed arc-length difference of gaze and cursor,
# both projected onto the tunnel centerline (+ = gaze ahead).
# Width family (straight, narrow→wide, wide→narrow) is plotted vs ID_W;
# curvature family (sinusoids, corners) vs ID_K (each family varies one part).
# ============================================================================
def _rdp(P, eps):
    """Douglas–Peucker simplification — strips cursor jitter, keeps real corners."""
    P = np.asarray(P, float)
    if len(P) < 3:
        return P
    a, b = P[0], P[-1]; ab = b - a; L = float(np.hypot(*ab))
    if L < 1e-12:
        dist = np.hypot(*(P - a).T)
    else:
        dist = np.abs((P[:, 0]-a[0])*ab[1] - (P[:, 1]-a[1])*ab[0]) / L
    i = int(np.argmax(dist))
    if dist[i] > eps:
        return np.vstack([_rdp(P[:i+1], eps)[:-1], _rdp(P[i:], eps)])
    return np.vstack([a, b])


def _turning(P):
    """Total absolute turning angle (rad) of a polyline."""
    P = np.asarray(P, float)
    if len(P) < 3:
        return 0.0
    dseg = np.diff(P, axis=0)
    th = np.arctan2(dseg[:, 1], dseg[:, 0])
    dth = np.diff(th); dth = (dth + np.pi) % (2*np.pi) - np.pi
    return float(np.abs(dth).sum())


def _constrained_centerline(tr):
    """Constrained portion of the trial centerline (recorded, task coords) with
    per-point width and curvature. Returns (P[n,2], W[n], K[n]) or None."""
    traj = tr.get("trajectory") or []
    con = tr.get("constrained?") or []
    dif = tr.get("difficulty") or []
    n = min(len(traj), len(con)) if con else len(traj)
    if n < 3:
        return None
    P = np.array([[traj[i]["x"], traj[i]["y"]] for i in range(n)], float)
    W = np.array([(dif[i] or {}).get("tunnelWidth", np.nan) if i < len(dif) else np.nan
                  for i in range(n)], float)
    K = np.array([(dif[i] or {}).get("curvature", 0.0) if i < len(dif) else 0.0
                  for i in range(n)], float)
    lab = np.array(con[:n]) if con else np.array(["constrained"] * n)
    idx = np.where(lab == "constrained")[0]
    if len(idx) < 3:
        return None
    return P[idx[0]:idx[-1]+1], W[idx[0]:idx[-1]+1], K[idx[0]:idx[-1]+1]


def _exact_centerline(tr, clip_constrained=False):
    """EXACT analytic tunnel centerline (task coords) for this trial, regenerated
    from its `condition` via the experiment's own path generators
    (tunnel_centerline.centerline_for_condition) — NOT the recorded cursor path.
    Returns (P[n,2], meta) or None. If clip_constrained, keep only the corridor
    (constrained) portion of a constrained_to_unconstrained trial."""
    c = tr.get("condition") or {}
    traj = tr.get("trajectory") or []
    tgt = (traj[-1]["x"], traj[-1]["y"]) if traj else None
    P, meta = centerline_for_condition(c, target_pos=tgt)
    if clip_constrained and c.get("tunnelType") == "constrained_to_unconstrained":
        P = P[P[:, 0] <= 0.23]
    if len(P) < 3:
        return None
    return P, meta


def _tunnel_ID(tr, curv_thresh=0.5):
    """ID_W = ∫ds/W and ID_K = ∫|κ|ds for the constrained segment, measured on the
    EXACT analytic centerline:
      ID_W = Σ ds / W(s)  (W from the condition's width model; piecewise for
             narrow↔wide),
      ID_K = total absolute turning angle of the exact centerline (= ∫|κ|ds),
             which is exact for smooth sinusoids AND sharp corners alike.
    family comes from the tunnel type (sinusoids/corners -> curvature; straight,
    narrow↔wide -> width)."""
    ec = _exact_centerline(tr, clip_constrained=True)
    if ec is None:
        return None
    P, meta = ec
    Wfn = meta.get("width_fn")
    if Wfn is None:
        return None
    ds = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    smid = 0.5 * (s[:-1] + s[1:])
    W = np.asarray(Wfn(smid), float)
    with np.errstate(divide="ignore", invalid="ignore"):
        IDW = float(np.nansum(ds / np.where(W > 0, W, np.nan)))
    IDK = _turning(P)
    family = meta.get("family")
    if family not in ("width", "curvature"):
        family = "curvature" if IDK > curv_thresh else "width"
    return {"ID_W": IDW, "ID_K": IDK, "family": family}


def _fit_px_to_task(df):
    """Affine (task x,y)->(screenvid px) from cursor pairs; return px->task fn."""
    m = (df["in_trial"].astype(bool) & np.isfinite(df["cursor_traj_x"])
         & np.isfinite(df["cursor_screenvid_x_px"]))
    T = df.loc[m, ["cursor_traj_x", "cursor_traj_y"]].to_numpy()
    S = df.loc[m, ["cursor_screenvid_x_px", "cursor_screenvid_y_px"]].to_numpy()
    A, *_ = np.linalg.lstsq(np.hstack([T, np.ones((len(T), 1))]), S, rcond=None)
    M, t = A[:2, :], A[2, :]
    Minv = np.linalg.inv(M)
    rms = float(np.sqrt((((np.hstack([T, np.ones((len(T), 1))]) @ A) - S) ** 2)
                        .sum(1)).mean())
    return (lambda px: (np.asarray(px, float) - t) @ Minv), rms


def _gaze_lead(mask, tr, ctx, gtx):
    """Mean signed gaze lead over `mask` rows: project cursor and gaze onto the
    constrained centerline by x-monotone arc length and take s_gaze - s_cursor
    (task units, + = gaze ahead). ctx/gtx are full-length cursor-task-x and
    gaze-task-x arrays."""
    cl = _constrained_centerline(tr)
    if cl is None:
        return np.nan, 0
    P = cl[0]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    xu, iu = np.unique(P[:, 0], return_index=True)     # monotone x -> arc length
    if len(xu) < 2:
        return np.nan, 0
    su = s[iu]
    h = np.interp(gtx[mask], xu, su) - np.interp(ctx[mask], xu, su)
    h = h[np.isfinite(h)]
    return (float(np.mean(h)), len(h)) if len(h) else (np.nan, 0)


def constrained_gazelead_table(df, experiment_json):
    """One row per constrained presentation: ID_W, ID_K, family, mean gaze lead."""
    d = json.load(open(experiment_json))
    px_to_task, rms = _fit_px_to_task(df)
    print(f"  affine gaze→task fit: {rms:.1f}px RMS")
    gtx = px_to_task(df[["gaze_transf_x_px", "gaze_transf_y_px"]].to_numpy())[:, 0]
    ctx = df["cursor_traj_x"].to_numpy(float)
    mac = df["mac_time_s"].to_numpy(float)
    intr = df["in_trial"].to_numpy(bool)
    con = (df["constrained"].astype(str).to_numpy()
           if "constrained" in df.columns else None)
    rows = []
    for tr in d.get("trialData", []):
        idw = _tunnel_ID(tr)
        if idw is None:
            continue
        ts = tr.get("timestamps") or []
        if len(ts) < 2:
            continue
        lo, hi = ts[0] / 1000.0, ts[-1] / 1000.0
        mask = intr & (mac >= lo) & (mac <= hi)
        if con is not None:
            mask = mask & (con == "constrained")
        if mask.sum() < 3:
            continue
        h, n = _gaze_lead(mask, tr, ctx, gtx)
        if not np.isfinite(h):
            continue
        rows.append({"trial_id": tr.get("trial_id"), "round": tr.get("round"),
                     "tunnelType": (tr.get("condition") or {}).get("tunnelType"),
                     "family": idw["family"], "ID_W": idw["ID_W"],
                     "ID_K": idw["ID_K"], "gaze_lead": h, "n": n})
    return pd.DataFrame(rows)


def step_plot_gazelead(tbl, family, out_png):
    """Constrained aggregate scatter: gaze lead vs ID_W (width family) or vs ID_K
    (curvature family). One point per presentation, scatter only."""
    import matplotlib.pyplot as plt

    xcol = "ID_W" if family == "width" else "ID_K"
    sub = tbl[tbl["family"] == family]
    xs = sub[xcol].to_numpy(float)
    ys = sub["gaze_lead"].to_numpy(float)
    color = "#2ca02c" if family == "width" else "#9467bd"

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, s=28, color=color, alpha=0.75, edgecolor="k", linewidth=0.3)
    ax.axhline(0, color="gray", lw=0.8)
    if family == "width":
        title = "Width family: gaze lead vs steering difficulty"
        formula = r"$\mathrm{ID_W}=\int ds/W$"
        xlabel = r"steering difficulty   $\mathrm{ID_W}=\int ds/W$   (∫ 1/width along the path)"
        caption = (
            "Width family (straight, narrow→wide, wide→narrow).  One point = one presentation's constrained segment.\n"
            "x = ID_W = ∫ds/W over the centerline.   y = mean gaze lead = signed arc-length of gaze − cursor on the\n"
            "centerline (task units; + = gaze ahead of cursor)."
        )
    else:
        title = "Curvature family: gaze lead vs turning difficulty"
        formula = r"$\mathrm{ID_K}=\int |\kappa|\,ds$"
        xlabel = r"turning difficulty   $\mathrm{ID_K}=\int|\kappa|ds$   (total turning angle, rad)"
        caption = (
            "Curvature family (sinusoids, corners).  One point = one presentation's constrained segment.\n"
            "x = ID_K = ∫|κ|ds = total absolute turning angle.   y = mean gaze lead = signed arc-length of gaze −\n"
            "cursor on the centerline (task units; + = gaze ahead of cursor)."
        )
    ax.text(0.03, 0.97, formula, transform=ax.transAxes, va="top", ha="left",
            fontsize=13, color=color,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color, alpha=0.9))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mean gaze lead  (task units, + = gaze ahead)")
    ax.set_title(title, fontsize=10)
    fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=7,
             style="italic", color="0.35")
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"wrote {out_png}: {len(xs)} {family}-family presentations")


# ============================================================================
# Extended outputs: distance AND gaze lead vs difficulty, per-trial time series,
# organized into 3 subfolders (Distance-based / Gaze-lead-based / Per-trial ...).
# ============================================================================
def _straight_s(a, b, pts):
    """Signed arc length of pts projected onto the line a->b (a = 0)."""
    d = b - a
    L = float(np.hypot(*d))
    if L < 1e-9:
        return np.zeros(len(pts))
    return (np.asarray(pts, float) - a) @ (d / L)


def _smooth_centerline(P, M=140, win=13):
    """Tunnel-centerline proxy: recorded path resampled uniformly in x and
    smoothed. The tunnels are narrow so the cursor hugs the centre, so this
    approximates the tunnel centerline (not the raw, jittery cursor path).
    Returns (xs, s) — x grid and cumulative arc length on the smoothed line."""
    x, y = P[:, 0], P[:, 1]
    o = np.argsort(x); x, y = x[o], y[o]
    xu, iu = np.unique(x, return_index=True); yu = y[iu]
    if len(xu) < 4:
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xu), np.diff(yu)))])
        return xu, s
    xs = np.linspace(xu[0], xu[-1], M)
    ys = np.interp(xs, xu, yu)
    if 3 <= win < M:
        k = np.ones(win) / win
        ys = np.convolve(np.pad(ys, win // 2, mode="edge"), k, "valid")[:M]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    return xs, s


def add_gaze_lead_column(df, experiment_json):
    """Add df['gaze_lead_signed'] (TASK UNITS, + = gaze ahead) per in-trial
    sample: project cursor and gaze onto that trial's centerline and take
    s_gaze - s_cursor. Constrained trials project onto the EXACT analytic tunnel
    centerline (regenerated from the condition); pointing uses the straight
    start→target line.

    Gaze lead is an arc length along the tunnel centerline, and the centerline is
    defined in the experiment's task coordinate space (where the tunnel geometry
    is authored), so it comes out in task units."""
    d = json.load(open(experiment_json))
    px_to_task, rms = _fit_px_to_task(df)
    gtask = px_to_task(df[["gaze_transf_x_px", "gaze_transf_y_px"]].to_numpy())
    ctask = df[["cursor_traj_x", "cursor_traj_y"]].to_numpy(float)
    mac = df["mac_time_s"].to_numpy(float)
    intr = df["in_trial"].to_numpy(bool)
    lead = np.full(len(df), np.nan)
    for tr in d.get("trialData", []):
        ts = tr.get("timestamps") or []
        traj = tr.get("trajectory") or []
        if len(ts) < 2 or len(traj) < 2:
            continue
        lo, hi = ts[0] / 1000.0, ts[-1] / 1000.0
        idx = np.where(intr & (mac >= lo) & (mac <= hi))[0]
        if len(idx) < 1:
            continue
        if (tr.get("condition") or {}).get("tunnelType") == "unconstrained_pointing":
            P = np.array([[p["x"], p["y"]] for p in traj], float)
            sc = _straight_s(P[0], P[-1], ctask[idx])
            sg = _straight_s(P[0], P[-1], gtask[idx])
        else:
            ec = _exact_centerline(tr)         # exact analytic tunnel centerline
            if ec is None:
                continue
            P = ec[0]
            sc = project_arclen(P, ctask[idx])
            sg = project_arclen(P, gtask[idx])
        lead[idx] = sg - sc
    df["gaze_lead_signed"] = lead   # task units (arc length in the task coord space)
    return df, rms


def agg_table(df, experiment_json, segment):
    """Per presentation of `segment`: mean_dist (px), gaze_lead (task units), and
    difficulty — constrained: ID_W/ID_K/family/AW; unconstrained: fitts."""
    d = json.load(open(experiment_json))
    mac = df["mac_time_s"].to_numpy(float)
    intr = df["in_trial"].to_numpy(bool)
    con = (df["constrained"].astype(str).to_numpy()
           if "constrained" in df.columns else None)
    dist = df["gaze_cursor_dist_px"].to_numpy(float)
    lead = (df["gaze_lead_signed"].to_numpy(float)
            if "gaze_lead_signed" in df.columns else np.full(len(df), np.nan))
    rows = []
    for tr in d.get("trialData", []):
        ts = tr.get("timestamps") or []
        if len(ts) < 2:
            continue
        lo, hi = ts[0] / 1000.0, ts[-1] / 1000.0
        mask = intr & (mac >= lo) & (mac <= hi)
        if con is not None:
            mask = mask & (con == segment)
        if mask.sum() < 3:
            continue
        row = {"trial_id": tr.get("trial_id"), "round": tr.get("round"),
               "tunnelType": (tr.get("condition") or {}).get("tunnelType"),
               "mean_dist": float(np.nanmean(dist[mask])),
               "gaze_lead": float(np.nanmean(lead[mask]))}
        if segment == "constrained":
            idw = _tunnel_ID(tr)
            if idw is None:
                continue
            aw = _constrained_AW(tr)
            row.update(family=idw["family"], ID_W=idw["ID_W"], ID_K=idw["ID_K"],
                       AW=(aw["difficulty"] if aw else np.nan))
        else:
            u = _unconstrained_ID(tr)
            if u is None:
                continue
            row["fitts"] = u["difficulty"]
        rows.append(row)
    return pd.DataFrame(rows)


def _scatter(sub, xcol, ycol, xlabel, ylabel, title, out_png, color,
             zeroline=False, seg=None, footnote=None):
    import matplotlib.pyplot as plt
    if xcol not in sub or ycol not in sub:
        print(f"  (skip {os.path.basename(out_png)}: no data)")
        return
    xs = sub[xcol].to_numpy(float)
    ys = sub[ycol].to_numpy(float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    fig, ax = plt.subplots(figsize=(7, 5.4) if footnote else (7, 5))
    ax.scatter(xs, ys, s=28, color=color, alpha=0.75, edgecolor="k", linewidth=0.3)
    if zeroline:
        ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # segment (constrained / unconstrained) written into the title, colour-coded
    if seg:
        segcol = {"constrained": "#2ca02c", "unconstrained": "#d62728"}.get(seg, "0.2")
        ax.set_title(f"[{seg.upper()}]  {title}", fontsize=10, color=segcol)
    else:
        ax.set_title(title, fontsize=10)
    if footnote:
        fig.text(0.5, 0.015, footnote, ha="center", va="bottom", fontsize=7,
                 style="italic", color="0.35")
        fig.tight_layout(rect=[0, 0.13, 1, 1])
    else:
        fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  wrote {os.path.basename(out_png)} ({len(xs)} pts)")


def step_plot_lead_time(df, out_pdf, experiment_json):
    """Per-trial presentation: signed gaze lead vs time (task units), with the
    constrained (green) vs unconstrained (red) segments shaded/labelled and a
    divider at the transition — same scheme as the per-trial distance plot."""
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    if "gaze_lead_signed" not in df.columns:
        return
    has_con = "constrained" in df.columns
    condmap = _cond_lookup(experiment_json)
    df = df.sort_values("neon_time_s").reset_index(drop=True)
    n = 0
    with PdfPages(out_pdf) as pdf:
        for idx, ordn, n_trials, tid, rnd, desc, mask in _presentations_in_order(df, experiment_json):
            g = df[mask].sort_values("neon_time_s")
            if len(g) < 2:
                continue
            t = g["neon_time_s"].to_numpy(float)
            t = t - t[0]
            y = g["gaze_lead_signed"].to_numpy(float)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(t, y, ".", color="#9467bd", ms=3)
            ax.axhline(0, color="gray", lw=0.8)
            ax.set_xlabel("time since trial start (s)")
            ax.set_ylabel("signed gaze lead along centerline (task units)\n(+ gaze ahead  /  − cursor ahead)")
            ord_s = f"{int(ordn)}/{n_trials}" if (ordn is not None and ordn == ordn) else "?"
            rnd_s = f"  round {int(rnd)}" if (rnd is not None and rnd == rnd) else ""
            tid_s = int(tid) if tid == tid else "?"
            ax.set_title(f"Trial {ord_s}{rnd_s}   —   {desc}   (id {tid_s})", fontsize=9)
            ax.margins(x=0)
            spans, Rt = _dot_block_spans(g, condmap.get((tid, rnd), {}))
            handles = _draw_dot_blocks(ax, t, spans, Rt)
            ax.legend(handles=handles, fontsize=7.5, loc="upper right", framealpha=0.95)
            ax.text(0.5, -0.14,
                    "Gaze lead = arc length along the tunnel centerline (the experiment's task coordinate space), so it is in task units.",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7,
                    style="italic", color="0.4")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            n += 1
    print(f"  wrote {os.path.basename(out_pdf)} ({n} pages)")


def step_speed_distance(df, out_dir, base, experiment_json):
    """Cursor speed vs the Euclidean gaze-cursor distance, split into
    constrained | unconstrained panels.  Writes one figure:
        <base>_speed_vs_euclidean.png
      x = Euclidean gaze-cursor distance (screen-video px)
      y = cursor speed (task units/s), the experiment's per-sample speed
      black line = median cursor speed per distance bin."""
    import matplotlib.pyplot as plt
    need = {"gaze_cursor_dist_px", "cursor_traj_x", "cursor_traj_y",
            "mac_time_s", "neon_time_s", "in_trial"}
    if not need.issubset(df.columns):
        print("  (skip speed/euclidean: missing columns)")
        return
    # remove any superseded variants from earlier layouts
    for _s in ("_speed_distance.png", "_gazelead_vs_speed.png", "_gazelead_vs_distance.png",
               "_euclidean_vs_speed.png", "_euclidean_vs_distance.png"):
        _p = os.path.join(out_dir, base + _s)
        if os.path.exists(_p):
            try:
                os.remove(_p)
            except Exception:
                pass
    has_con = "constrained" in df.columns
    SP, EU, SEG = [], [], []
    for idx, ordn, n_trials, tid, rnd, desc, mask in _presentations_in_order(df, experiment_json):
        g = df[mask].sort_values("neon_time_s")
        t = g["mac_time_s"].to_numpy(float)
        px = g["cursor_traj_x"].to_numpy(float); py = g["cursor_traj_y"].to_numpy(float)
        eu = g["gaze_cursor_dist_px"].to_numpy(float)
        con = (g["constrained"].astype(str).to_numpy() if has_con else np.array(["?"] * len(g)))
        ok = np.isfinite(t) & np.isfinite(px) & np.isfinite(py)
        t, px, py, eu, con = t[ok], px[ok], py[ok], eu[ok], con[ok]
        o = np.argsort(t); t, px, py, eu, con = t[o], px[o], py[o], eu[o], con[o]
        keep = np.concatenate([[True], np.diff(t) > 1e-4])
        t, px, py, eu, con = t[keep], px[keep], py[keep], eu[keep], con[keep]
        if len(t) < 5:
            continue
        speed = np.hypot(np.gradient(px, t), np.gradient(py, t))         # task units / s
        SP.append(speed); EU.append(eu); SEG.append(con)
    if not SP:
        print("  (skip speed/euclidean: no data)")
        return
    SP = np.concatenate(SP); EU = np.concatenate(EU); SEG = np.concatenate(SEG)
    cmap = {"constrained": "#2ca02c", "unconstrained": "#d62728"}
    xmax = np.nanpercentile(EU, 99); ymax = np.nanpercentile(SP, 99.5)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharex=True, sharey=True)
    for a, s in zip(axes, ["constrained", "unconstrained"]):
        col = cmap[s]; m = (SEG == s) & np.isfinite(EU) & np.isfinite(SP)
        x, y = EU[m], SP[m]
        a.scatter(x, y, s=6, color=col, alpha=0.28, linewidths=0, label="samples")
        if len(x) > 50:
            bins = np.linspace(0, xmax, 16); bidx = np.digitize(x, bins); bx, by = [], []
            for b in range(1, len(bins)):
                v = y[bidx == b]
                if len(v) >= 10:
                    bx.append((bins[b - 1] + bins[b]) / 2); by.append(np.median(v))
            a.plot(bx, by, "o-", color="black", ms=4, lw=1.6, label="median speed per distance bin")
        a.set_title(f"[{s.upper()}]   n={int(m.sum())} samples", fontsize=10, color=col)
        a.set_xlabel("Euclidean gaze-cursor distance (px)")
        a.set_ylabel("cursor speed (task units/s)")
        a.set_xlim(0, xmax); a.set_ylim(0, ymax * 1.05); a.legend(fontsize=8, loc="upper right")
    fig.suptitle("Cursor speed vs Euclidean gaze-cursor distance   (constrained | unconstrained)", fontsize=12)
    fig.text(0.5, 0.02,
             "x = Euclidean gaze-cursor distance = sqrt((gaze_x-cursor_x)^2 + (gaze_y-cursor_y)^2), screen-video px.   "
             "y = cursor speed = experiment's per-sample sqrt((dx/dt)^2 + (dy/dt)^2), task units/s.",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(os.path.join(out_dir, base + "_speed_vs_euclidean.png"), dpi=120)
    plt.close(fig)
    print(f"  wrote {base}_speed_vs_euclidean.png ({len(SP)} samples)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_csv", required=True)
    ap.add_argument("--experiment_json", required=True)
    ap.add_argument("--out_dir", default=None,
                    help="output root (default: folder of the aligned CSV)")
    args = ap.parse_args()

    df = pd.read_csv(args.aligned_csv)
    df = step1_gaze_cursor_distance(df)
    df, rms = add_gaze_lead_column(df, args.experiment_json)
    print(f"gaze→task affine fit: {rms:.1f}px RMS")

    root = args.out_dir or os.path.dirname(os.path.abspath(args.aligned_csv))
    base = os.path.splitext(os.path.basename(args.aligned_csv))[0]
    dist_dir = os.path.join(root, "Distance-based")
    lead_dir = os.path.join(root, "Gaze-lead-based")
    trial_dir = os.path.join(root, "Per-trial time series")
    speed_dir = os.path.join(root, "Speed-and-distance")
    for dd in (dist_dir, lead_dir, trial_dir, speed_dir):
        os.makedirs(dd, exist_ok=True)
    df.to_csv(os.path.join(root, base + "_analysis.csv"), index=False)

    C = agg_table(df, args.experiment_json, "constrained")
    U = agg_table(df, args.experiment_json, "unconstrained")
    Cw = C[C.get("family") == "width"] if len(C) else C
    Ck = C[C.get("family") == "curvature"] if len(C) else C
    C.to_csv(os.path.join(root, base + "_constrained_table.csv"), index=False)
    U.to_csv(os.path.join(root, base + "_unconstrained_table.csv"), index=False)

    FIT = r"Fitts ID $=\log_2(1+D/W)$"
    IDW = r"$\mathrm{ID_W}=\int ds/W$"
    IDK = r"$\mathrm{ID_K}=\int|\kappa|\,ds$"
    DIST = "mean gaze-cursor distance (px)"
    LEAD = "mean gaze lead along centerline (task units, + = gaze ahead)"

    print("Distance-based/:")
    _scatter(U, "fitts", "mean_dist", FIT, DIST, "distance vs Fitts ID",
             os.path.join(dist_dir, base + "_unconstrained_dist_vs_Fitts.png"), "#d62728",
             seg="unconstrained")
    _scatter(C, "AW", "mean_dist", "steering difficulty  A/W (old)", DIST,
             "distance vs A/W (old index)",
             os.path.join(dist_dir, base + "_constrained_dist_vs_AW.png"), "#2ca02c",
             seg="constrained")
    _scatter(Cw, "ID_W", "mean_dist", IDW, DIST, "width family: distance vs ID_W",
             os.path.join(dist_dir, base + "_width_dist_vs_IDW.png"), "#2ca02c",
             seg="constrained")
    _scatter(Ck, "ID_K", "mean_dist", IDK, DIST, "curvature family: distance vs ID_K",
             os.path.join(dist_dir, base + "_curvature_dist_vs_IDK.png"), "#9467bd",
             seg="constrained")

    LEAD_NOTE = (
        "Gaze lead: at each in-trial sample, project the cursor and the gaze onto the trial's centerline\n"
        "(exact analytic tunnel path; straight start→target for unconstrained) → arc-length positions s(cursor), s(gaze).\n"
        "Per-sample lead h = s(gaze) − s(cursor);  each point = one trial's MEAN h across its samples  (+ = gaze ahead).\n"
        "Gaze lead is an arc length along the tunnel centerline, defined in the experiment's task coordinate space\n"
        "(where the tunnel geometry is authored), so it comes out in task units."
    )
    print("Gaze-lead-based/:")
    _scatter(U, "fitts", "gaze_lead", FIT, LEAD, "gaze lead vs Fitts ID",
             os.path.join(lead_dir, base + "_unconstrained_lead_vs_Fitts.png"), "#d62728",
             zeroline=True, seg="unconstrained", footnote=LEAD_NOTE)
    _scatter(Cw, "ID_W", "gaze_lead", IDW, LEAD, "width family: gaze lead vs ID_W",
             os.path.join(lead_dir, base + "_width_lead_vs_IDW.png"), "#2ca02c",
             zeroline=True, seg="constrained", footnote=LEAD_NOTE)
    _scatter(Ck, "ID_K", "gaze_lead", IDK, LEAD, "curvature family: gaze lead vs ID_K",
             os.path.join(lead_dir, base + "_curvature_lead_vs_IDK.png"), "#9467bd",
             zeroline=True, seg="constrained", footnote=LEAD_NOTE)

    print("Per-trial time series/:")
    step2_plot_per_trial(df, os.path.join(trial_dir, base + "_signed_distance_vs_time.pdf"),
                         experiment_json=args.experiment_json)
    step_plot_lead_time(df, os.path.join(trial_dir, base + "_signed_gazelead_vs_time.pdf"),
                        args.experiment_json)

    print("Speed-and-distance/:")
    step_speed_distance(df, speed_dir, base, args.experiment_json)


if __name__ == "__main__":
    main()
