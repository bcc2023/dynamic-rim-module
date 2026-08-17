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

import numpy as np
import pandas as pd


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

            colors = {"constrained": "#2ca02c", "unconstrained": "#d62728"}
            if has_con:
                con = g["constrained"].astype(str).to_numpy()
                # contiguous segments of equal condition
                changes = np.where(con[:-1] != con[1:])[0]
                bounds = [0] + [c + 1 for c in changes] + [len(con)]
                # segment edges meet exactly at the transition midpoints, so the
                # colored regions are contiguous (no unshaded gap / "blank")
                trans = [(t[ci] + t[ci + 1]) / 2 for ci in changes]
                edges = [t[0]] + trans + [t[-1]]
                blend = ax.get_xaxis_transform()          # x in data, y in axes fraction
                for bi in range(len(bounds) - 1):
                    lo = bounds[bi]
                    lab = con[lo]
                    col = colors.get(lab, "#7f7f7f")
                    x0, x1 = edges[bi], edges[bi + 1]
                    ax.axvspan(x0, x1, color=col, alpha=0.08)
                    ax.text((x0 + x1) / 2, 0.96, lab, transform=blend,
                            ha="center", va="top",
                            color=col, fontsize=9, fontweight="bold")
                # vertical divider(s) at each transition
                for xt in trans:
                    ax.axvline(xt, color="black", ls="--", lw=1.2)
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


def _tunnel_ID(tr, rdp_eps=0.005, curv_thresh=0.5):
    """ID_W = ∫ds/W and ID_K = ∫|κ|ds for the constrained segment.
    ID_K = max(curvature-field integral, turning of the RDP-simplified path):
    the field is right for smooth sinusoids, RDP geometry for sharp corners
    (whose curvature field is 0), and both ~0 for straight/width tunnels.
    family = 'curvature' if ID_K exceeds curv_thresh else 'width'."""
    cl = _constrained_centerline(tr)
    if cl is None:
        return None
    P, W, K = cl
    ds = np.linalg.norm(np.diff(P, axis=0), axis=1)
    Wmid = 0.5 * (W[:-1] + W[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        IDW = float(np.nansum(ds / Wmid))
    IDK_field = float(np.nansum(0.5 * (np.abs(K[:-1]) + np.abs(K[1:])) * ds))
    IDK = max(IDK_field, _turning(_rdp(P, rdp_eps)))
    return {"ID_W": IDW, "ID_K": IDK,
            "family": "curvature" if IDK > curv_thresh else "width"}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_csv", required=True)
    ap.add_argument("--experiment_json", default=None,
                    help="steering JSON, needed for path length / difficulty (steps 3-4)")
    ap.add_argument("--out_csv", default=None,
                    help="where to write the augmented CSV (default: <input>_analysis.csv)")
    ap.add_argument("--out_pdf", default=None,
                    help="per-trial distance plots (default: <input>_trials.pdf)")
    args = ap.parse_args()

    df = pd.read_csv(args.aligned_csv)

    # --- STEP 1 -------------------------------------------------------------
    df = step1_gaze_cursor_distance(df)
    d = df["gaze_cursor_dist_px"].dropna()
    print("STEP 1 — gaze-cursor Euclidean distance (in-trial, px):")
    print(f"  n in-trial samples: {len(d)}")
    print(f"  mean {d.mean():.1f}, median {d.median():.1f}, "
          f"std {d.std():.1f}, min {d.min():.1f}, max {d.max():.1f}")

    out = args.out_csv or args.aligned_csv.rsplit(".", 1)[0] + "_analysis.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")

    # --- STEP 2 -------------------------------------------------------------
    out_pdf = args.out_pdf or args.aligned_csv.rsplit(".", 1)[0] + "_trials.pdf"
    step2_plot_per_trial(df, out_pdf, experiment_json=args.experiment_json)

    # --- STEP 3 : difficulty aggregate plots --------------------------------
    if args.experiment_json:
        base = args.aligned_csv.rsplit(".", 1)[0]

        # Unconstrained (pointing) — Fitts ID, unchanged
        diff_df = step3_difficulty_table(args.experiment_json)
        diff_df.to_csv(base + "_difficulty.csv", index=False)
        step_plot_difficulty(df, args.experiment_json, "unconstrained",
                             base + "_unconstrained_difficulty.png")

        # Constrained (steering) — NEW: gaze lead vs ID_W (width) / ID_K (curvature)
        print("STEP 3 — constrained gaze lead vs new difficulty index:")
        tbl = constrained_gazelead_table(df, args.experiment_json)
        tbl.to_csv(base + "_gazelead_ID.csv", index=False)
        nw = (tbl["family"] == "width").sum()
        nk = (tbl["family"] == "curvature").sum()
        print(f"  {len(tbl)} constrained presentations "
              f"({nw} width family, {nk} curvature family) → {base}_gazelead_ID.csv")
        step_plot_gazelead(tbl, "width", base + "_width_gazelead_IDW.png")
        step_plot_gazelead(tbl, "curvature", base + "_curvature_gazelead_IDK.png")
    else:
        print("(skipping step 3: pass --experiment_json to compute difficulty)")


if __name__ == "__main__":
    main()
