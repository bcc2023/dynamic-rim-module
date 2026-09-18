#!/usr/bin/env python3
"""
centerline_explainer — show REAL tunnels from a session and the centerline that
runs down the middle of each, plus how gaze lead is measured along it.

For each example the grey band is the actual tunnel corridor the participant saw
(walls at centerline ± tunnelWidth/2, exactly as the experiment draws it); the
dark line down the middle is the EXACT analytic centerline, regenerated from the
trial's `condition` via the experiment's own path generators (tunnel_centerline.py,
ported from steering-experiment/src/utils/tunnelGenerator.js). Faint dots are the
recorded cursor path, which hugs the centerline.

Panels: a wide sinusoid tunnel (smooth curve), a corner tunnel (sharp turns), and
an unconstrained-pointing trial (no corridor — open area, straight centerline).

Gaze lead h = s(gaze) − s(cursor): project gaze and cursor to their nearest point
on the centerline, take the signed arc-length difference (+ = gaze ahead).

USAGE
    python centerline_explainer.py --experiment_json steering_*.json \
        --out centerline_explainer.png [--name Beichen]
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tunnel_centerline import centerline_for_condition  # noqa: E402


def arclen(P):
    return np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def ribbon(P, W):
    """Corridor polygon of width W around centerline P (walls at ± W/2 along the
    local normal). Returns (polygon, upper_wall, lower_wall)."""
    d = np.gradient(P, axis=0)
    t = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    n = np.column_stack([-t[:, 1], t[:, 0]])
    up = P + n * (W / 2.0)
    lo = P - n * (W / 2.0)
    return np.vstack([up, lo[::-1]]), up, lo


def draw_axis_corridor(ax, P, W, col="0.83"):
    """Corner corridor with HARD 90° corners (matches the experiment): fill an
    axis-aligned rectangle per segment (horizontal -> walls at y ± W/2; vertical
    -> walls at x ± W/2) plus a W×W square at each corner. No diagonal edges."""
    hw = W / 2.0
    horiz = lambda a, b: abs(b[0] - a[0]) >= abs(b[1] - a[1])
    for a, b in zip(P[:-1], P[1:]):
        if horiz(a, b):
            x0, x1 = sorted([a[0], b[0]]); y = 0.5 * (a[1] + b[1])
            ax.fill([x0, x1, x1, x0], [y - hw, y - hw, y + hw, y + hw], color=col, lw=0, zorder=0)
        else:
            y0, y1 = sorted([a[1], b[1]]); x = 0.5 * (a[0] + b[0])
            ax.fill([x - hw, x + hw, x + hw, x - hw], [y0, y0, y1, y1], color=col, lw=0, zorder=0)
    o = np.array([1 if horiz(P[i], P[i + 1]) else 0 for i in range(len(P) - 1)])
    for i in np.where(np.diff(o) != 0)[0] + 1:      # square-fill each right-angle corner
        x, y = P[i]
        ax.fill([x - hw, x + hw, x + hw, x - hw], [y - hw, y - hw, y + hw, y + hw], color=col, lw=0, zorder=0)


def pick_widest(td, types):
    best = None
    for t in td:
        c = t.get("condition") or {}
        if c.get("tunnelType") in types:
            w = c.get("tunnelWidth") or 0
            if best is None or w > best[0]:
                best = (w, t)
    return best[1] if best else None


def pick_far(td):
    best = None
    for t in td:
        c = t.get("condition") or {}
        if c.get("tunnelType") == "unconstrained_pointing":
            dd = c.get("distance") or 0
            if best is None or dd > best[0]:
                best = (dd, t)
    return best[1] if best else None


def draw_tunnel(ax, tr, name, gaze_demo=False):
    """Draw one real constrained tunnel: corridor band + exact centerline."""
    cond = tr.get("condition") or {}
    Praw = np.array([[p["x"], p["y"]] for p in tr.get("trajectory", [])], float)
    P, meta = centerline_for_condition(cond)
    W = meta.get("width") or cond.get("tunnelWidth") or 0.02

    if cond.get("tunnelType") == "corner":
        draw_axis_corridor(ax, P, W)                       # hard 90° corners
        ax.plot([], [], "s", color="0.83", ms=11, label=f"tunnel corridor (W={W:g})")
    else:
        poly, up, lo = ribbon(P, W)
        ax.fill(poly[:, 0], poly[:, 1], color="0.83", zorder=0, label=f"tunnel corridor (W={W:g})")
        ax.plot(up[:, 0], up[:, 1], "-", color="0.45", lw=1)
        ax.plot(lo[:, 0], lo[:, 1], "-", color="0.45", lw=1)
    if len(Praw):
        ax.plot(Praw[:, 0], Praw[:, 1], ".", color="0.6", ms=2, zorder=1, label="recorded cursor")
    ax.plot(P[:, 0], P[:, 1], "-", color="crimson", lw=2.2, zorder=3, label="centerline (exact)")
    ax.plot(*P[0], "o", color="green", ms=10, zorder=4, label="start (s=0)")
    ax.plot(*P[-1], "*", color="red", ms=15, zorder=4, label="target (s=L)")

    if gaze_demo:
        s = arclen(P)
        at = lambda fr: min(int(np.searchsorted(s, fr * s[-1])), len(P) - 1)

        def normal(i):
            t = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]
            t = t / (np.linalg.norm(t) + 1e-9)
            return np.array([-t[1], t[0]])

        ic, ig = at(0.42), at(0.60)
        cfoot, gfoot = P[ic], P[ig]                 # projections (feet) on the centerline
        off = W * 0.7 + 0.015
        cur = cfoot + normal(ic) * off              # actual cursor: off-centre, one side
        gaze = gfoot - normal(ig) * off             # actual gaze: off-centre, other side
        # gaze-lead arc along the centerline between the two projected feet
        seg = (s >= s[ic]) & (s <= s[ig])
        ax.plot(P[seg, 0], P[seg, 1], "-", color="orange", lw=7, alpha=0.65, zorder=2)
        # cursor  ->  its projection (foot) on the centerline
        ax.annotate("", xy=cfoot, xytext=cur, zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=1.6, shrinkA=6, shrinkB=2))
        ax.plot(*cur, "o", color="tab:blue", ms=10, zorder=6)
        ax.plot(*cfoot, "X", color="tab:blue", ms=11, mec="white", mew=1.2, zorder=6)
        ax.annotate("cursor", cur, textcoords="offset points", xytext=(0, 11), color="tab:blue", fontsize=8.5, ha="center", fontweight="bold")
        ax.annotate("s(cursor)", cfoot, textcoords="offset points", xytext=(-10, 13), color="tab:blue", fontsize=8, ha="center")
        # gaze  ->  its projection (foot) on the centerline
        ax.annotate("", xy=gfoot, xytext=gaze, zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color="red", lw=1.6, shrinkA=8, shrinkB=2))
        ax.plot(*gaze, "o", mfc="none", mec="red", ms=13, mew=2.2, zorder=6)
        ax.plot(*gfoot, "X", color="red", ms=11, mec="white", mew=1.2, zorder=6)
        ax.annotate("gaze", gaze, textcoords="offset points", xytext=(0, 13), color="red", fontsize=8.5, ha="center", fontweight="bold")
        ax.annotate("s(gaze)", gfoot, textcoords="offset points", xytext=(12, -14), color="red", fontsize=8, ha="center")
        ax.annotate("h = gaze lead = s(gaze) − s(cursor)\n(both mapped onto the centerline)",
                    (P[at(0.51)][0], P[at(0.51)][1]),
                    textcoords="offset points", xytext=(0, -46), color="darkorange", fontsize=8.5, ha="center", fontweight="bold")

    tt = cond.get("tunnelType") or "sinusoidal"
    extra = ", hard 90° turns" if tt == "corner" else ""
    ax.set_title(f"{tt} tunnel\nconstraint: cursor must stay inside the corridor  (width W = {W:g}{extra})",
                 fontsize=9)
    ax.invert_yaxis(); ax.set_aspect("equal", "datalim")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    td = json.load(open(args.experiment_json))["trialData"]
    t_sin = pick_widest(td, {None, "gentle_sinusoidal", "sharp_sinusoidal"})
    t_cor = pick_widest(td, {"corner"})
    t_unc = pick_far(td)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.0))
    draw_tunnel(axes[0], t_sin, args.name, gaze_demo=True)
    draw_tunnel(axes[1], t_cor, args.name)

    # unconstrained: full open canvas, green start button + orange target (real radius)
    ax = axes[2]
    condu = t_unc.get("condition") or {}
    Pu = np.array([[p["x"], p["y"]] for p in t_unc["trajectory"]], float)
    a, b = Pu[0], Pu[-1]
    tr_rad = condu.get("targetRadius", 0.015)
    CW, CH = 0.46 * 1.065, 0.26                       # NORMALIZED canvas (canvasSize.js)
    ax.add_patch(plt.Rectangle((0, 0), CW, CH, facecolor="0.93", edgecolor="0.5", lw=1.2, zorder=0))
    ax.plot(Pu[:, 0], Pu[:, 1], ".", color="0.6", ms=2, zorder=1, label="recorded cursor")
    ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="crimson", lw=2.2, zorder=3, label="centerline (straight)")
    ax.add_patch(plt.Circle(a, 0.012, facecolor="tab:green", edgecolor="0.3", lw=1, zorder=4))
    ax.add_patch(plt.Circle(b, tr_rad, facecolor="#c8813f", edgecolor="0.3", lw=1, zorder=4))
    ax.plot([], [], "s", color="0.93", label="open canvas (no corridor)")
    ax.plot([], [], "o", color="tab:green", ms=9, label="start (s=0)")
    ax.plot([], [], "o", color="#c8813f", ms=9, label=f"target (s=L, r={tr_rad:g})")
    ax.set_xlim(-0.015, CW + 0.015); ax.set_ylim(CH, 0)     # y increases downward (screen)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"unconstrained pointing  (open canvas)\nconstraint: none along the path — just reach the target  (radius r = {tr_rad:g})",
                 fontsize=9)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

    fig.suptitle("Tunnels and their centerlines", fontsize=13)
    fig.text(0.5, 0.105,
             "Grey band = the corridor the cursor had to stay inside (walls at centerline ± W/2).   "
             "Red line = exact centerline (from the trial's condition, not the recorded cursor).",
             ha="center", va="bottom", fontsize=8, color="0.3")
    fig.text(0.5, 0.055,
             "Gaze lead — project the cursor and the gaze each onto the centerline (nearest point) to get "
             "arc-length positions s(cursor) and s(gaze) = distance along the centerline from the start.",
             ha="center", va="bottom", fontsize=8, color="#8a5a00", fontweight="bold")
    fig.text(0.5, 0.015,
             "Gaze lead  h = s(gaze) − s(cursor)   (arc-length along the centerline;  positive = gaze ahead of the cursor).",
             ha="center", va="bottom", fontsize=8, color="#8a5a00", fontweight="bold")
    fig.tight_layout(rect=[0, 0.15, 1, 0.93])
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
