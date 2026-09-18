#!/usr/bin/env python3
"""
tunnel_centerline — reconstruct the EXACT tunnel centerline for a steering trial
from its `condition`, by porting the experiment's own path generators
(steering-experiment/src/utils/tunnelGenerator.js + the dispatch in
steering_experiment_v2.jsx `setupTrial`).

This replaces the earlier "smooth the recorded cursor path" proxy: the centerline
here is the analytic path the tunnel was drawn from, so gaze-lead / arc-length /
curvature (ID_K) are measured against the true centerline, not the cursor.

Coordinates are the task/normalized frame used in the JSON `trajectory`
(x in [0, 0.46] m, y around yBase=0.13), i.e. directly overlayable on `trajectory`.

Public API
    centerline_for_condition(condition) -> (P, meta)
        P    : (N,2) float array, the centerline polyline in task coords.
        meta : dict {kind, width, width_fn(s)->W, family, curved(bool)}.
               width_fn returns the corridor width at arc-length s (constant for
               most types; piecewise for wide_to_narrow / narrow_to_wide).
    TARGET_POSITION_Y : the 3 target y-positions.

The dispatch mirrors setupTrial exactly:
    straight / gentle_sinusoidal        -> generate_tunnel_path(width, curvature|0)
    (no tunnelType, has curvature)      -> generate_tunnel_path(width, curvature)   # ids 1-5
    sharp_sinusoidal                    -> generate_tunnel_path(width, curvature)    # else branch
    corner                              -> generate_corner_path(width, .., numCorners|3, cornerOffset|.05, .002)
    sequential                          -> generate_sequential_path (straight->parabola if segmentType=='curvature')
    wide_to_narrow / narrow_to_wide     -> generate_sequential_path (flat centerline; width steps)
    constrained_to_unconstrained        -> flat corridor (x<mid) then straight line to the off-axis target
    unconstrained_pointing              -> straight line start->target
"""
import numpy as np

TUNNEL_STEP = 0.001            # experimentConstants.TUNNEL_STEP
TARGET_POSITION_Y = {"top": 0.05, "middle": 0.13, "bottom": 0.21}


# --------------------------------------------------------------------------- #
# ports of tunnelGenerator.js                                                 #
# --------------------------------------------------------------------------- #
def generate_tunnel_path(width, curvature, startX=0.0, endX=0.46,
                         yBase=0.13, wavelength=0.23, step=0.002):
    """Sine-wave centerline: y = yBase + curvature*sin(2*pi*x/wavelength).
    curvature=0 -> straight horizontal line."""
    xs = np.arange(startX, endX, step)
    ys = yBase + curvature * np.sin(2 * np.pi * xs / wavelength)
    return np.column_stack([xs, ys])


def generate_corner_path(width, startX=0.0, endX=0.46, yBase=0.13,
                         numCorners=3, cornerOffset=0.05, step=0.002):
    """90-degree corner path: equal-length horizontal runs joined by vertical
    jumps of +/- cornerOffset (alternating). Faithful port of generateCornerPath."""
    P = []
    horiz = (endX - startX) / (numCorners + 1)
    cx, cy = startX, yBase
    for k in range(numCorners):
        xEnd = cx + horiz
        x = cx
        while x < xEnd + step * 0.5:
            if x <= xEnd:
                P.append((x, cy))
            x += step
        if not P or abs(P[-1][0] - xEnd) > 1e-6:
            P.append((xEnd, cy))
        else:
            P[-1] = (xEnd, cy)
        cx = xEnd
        direction = 1.0 if k % 2 == 0 else -1.0
        yEnd = cy + direction * cornerOffset
        if direction > 0:
            y = cy + step
            while y < yEnd + step * 0.5:
                P.append((cx, y)); y += step
        else:
            y = cy - step
            while y > yEnd - step * 0.5:
                P.append((cx, y)); y -= step
        if not P or abs(P[-1][1] - yEnd) > step * 0.5:
            P.append((cx, yEnd))
        cy = yEnd
    finalXEnd = min(cx + horiz, endX)
    x = cx + step
    while x < finalXEnd + step * 0.5:
        if x <= finalXEnd:
            P.append((x, cy))
        x += step
    if not P or abs(P[-1][0] - endX) > 1e-6:
        P.append((endX, cy))
    else:
        P[-1] = (endX, cy)
    return np.asarray(P, float)


def generate_sequential_path(condition, startX=0.0, endX=0.46,
                             yBase=0.13, step=TUNNEL_STEP):
    """Two-segment path. Only segmentType=='curvature' bends (second half is a
    single quadratic peak y = yBase + amp*(1-(2u-1)^2)); every other sequential
    variant (wide_to_narrow / narrow_to_wide / plain) has a FLAT centerline and
    only the corridor *width* changes between halves."""
    seg = (endX - startX) / 2.0
    xs = np.arange(startX, endX, step)
    ys = np.full_like(xs, yBase)
    if condition.get("segmentType") == "curvature":
        amp = condition.get("segment2Curvature", 0.0)
        m = xs >= (startX + seg)
        u = (xs[m] - (startX + seg)) / seg          # 0..1 over second half
        ys[m] = yBase + amp * (1.0 - (2.0 * u - 1.0) ** 2)
    return np.column_stack([xs, ys])


# --------------------------------------------------------------------------- #
# dispatch (mirror of setupTrial) + width model                              #
# --------------------------------------------------------------------------- #
_CURVED = {"gentle_sinusoidal", "sharp_sinusoidal", "corner", "sequential", None}


def _const_width(w):
    return (lambda s: np.full_like(np.asarray(s, float), w))


def centerline_for_condition(condition, target_pos=None):
    """Return (P, meta) for a trial condition. target_pos overrides the target
    point (task coords) for pointing types; otherwise TARGET_POSITION_Y is used."""
    c = condition or {}
    tt = c.get("tunnelType")
    meta = {"kind": tt, "family": None, "curved": False,
            "width": None, "width_fn": None}

    if tt == "corner":
        P = generate_corner_path(c.get("tunnelWidth"), 0.0, 0.46, 0.13,
                                 c.get("numCorners", 3), c.get("cornerOffset", 0.05), 0.002)
        w = c.get("tunnelWidth")
        meta.update(family="curvature", curved=True, width=w, width_fn=_const_width(w))
        return P, meta

    if tt in ("straight", "gentle_sinusoidal") or (tt is None and "curvature" in c):
        curv = c.get("curvature", 0.0)
        P = generate_tunnel_path(c.get("tunnelWidth"), curv or 0.0)
        w = c.get("tunnelWidth")
        # straight (curv=0) -> width family; any sinusoid (gentle / id1-5) -> curvature
        meta.update(family="curvature" if curv else "width", curved=bool(curv),
                    width=w, width_fn=_const_width(w))
        return P, meta

    if tt == "sharp_sinusoidal":
        P = generate_tunnel_path(c.get("tunnelWidth"), c.get("curvature", 0.05))
        w = c.get("tunnelWidth")
        meta.update(family="curvature", curved=True, width=w, width_fn=_const_width(w))
        return P, meta

    if tt in ("wide_to_narrow", "narrow_to_wide", "sequential"):
        P = generate_sequential_path(c)
        s = _arclen(P)
        w1, w2 = c.get("segment1Width"), c.get("segment2Width")
        half = 0.23
        if w1 is not None and w2 is not None:
            xmid_s = float(np.interp(half, P[:, 0], s))
            meta["width_fn"] = lambda ss, a=w1, b=w2, m=xmid_s: np.where(np.asarray(ss, float) < m, a, b)
            meta["width"] = min(w1, w2)
        curved = c.get("segmentType") == "curvature"
        meta.update(family="curvature" if curved else "width", curved=curved)
        return P, meta

    if tt == "constrained_to_unconstrained":
        # first half: real corridor along y=yBase; second half: open, to off-axis target
        tp = target_pos or (c.get("distance", 0.46),
                            TARGET_POSITION_Y.get(c.get("targetPosition", "middle"), 0.13))
        mid = 0.23
        seg1 = generate_sequential_path(c)
        seg1 = seg1[seg1[:, 0] <= mid]
        n2 = 60
        seg2 = np.column_stack([np.linspace(mid, tp[0], n2),
                                np.linspace(0.13, tp[1], n2)])
        P = np.vstack([seg1, seg2[1:]])
        w1 = c.get("segment1Width")
        meta.update(family="width", curved=False, width=w1,
                    width_fn=_const_width(w1 if w1 else 0.02))
        return P, meta

    # unconstrained_pointing (and any fallback): straight start->target line
    tp = target_pos or (c.get("distance", 0.46),
                        TARGET_POSITION_Y.get(c.get("targetPosition", "middle"), 0.13))
    P = np.column_stack([np.linspace(0.0, tp[0], 80),
                         np.linspace(TARGET_POSITION_Y["middle"], tp[1], 80)])
    meta.update(family="pointing", curved=False, width=None, width_fn=None)
    return P, meta


def _arclen(P):
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])


def project_arclen(P, pts):
    """Project pts (M,2) onto polyline P by nearest point; return signed arc-length s."""
    s = _arclen(P)
    A = P[:-1]; B = P[1:]
    AB = B - A
    L2 = (AB ** 2).sum(1) + 1e-12
    out = np.empty(len(pts))
    for i, q in enumerate(pts):
        t = np.clip(((q - A) * AB).sum(1) / L2, 0, 1)
        proj = A + t[:, None] * AB
        d2 = ((proj - q) ** 2).sum(1)
        j = int(np.argmin(d2))
        out[i] = s[j] + t[j] * np.hypot(*AB[j])
    return out


if __name__ == "__main__":
    # smoke test: print centerline extent for a few synthetic conditions
    for c in [{"tunnelWidth": 0.02, "curvature": 0.025},
              {"tunnelType": "sharp_sinusoidal", "tunnelWidth": 0.02, "curvature": 0.05},
              {"tunnelType": "corner", "tunnelWidth": 0.02, "numCorners": 2, "cornerOffset": 0.1},
              {"tunnelType": "straight", "tunnelWidth": 0.02, "curvature": 0}]:
        P, m = centerline_for_condition(c)
        print(f"{str(c.get('tunnelType')):18s} N={len(P):4d}  x[{P[:,0].min():.3f},{P[:,0].max():.3f}] "
              f"y[{P[:,1].min():.3f},{P[:,1].max():.3f}]  curved={m['curved']} family={m['family']}")
