#!/usr/bin/env python3
"""
self_calibrate — within-session gaze calibration from the task cursor.

Use this when a Neon gaze-offset calibration can't be trusted for the task —
e.g. the calibration recording is a *different wearing* of the glasses than the
task (see `calib-transfer`'s caveat: if the two are hours apart the offset does
not transfer, and forcing it injects a fixed misalignment).

Idea: during the task the participant's gaze tracks the moving cursor, so the
cursor is a within-session reference. We fit a homography that maps the mapped
gaze (screen-video px, pl-dynamic-rim's `gaze position transf x/y [px]`) onto the
cursor's task coordinates (experiment JSON `trajectory`), time-aligned by wall
clock (gaze = Neon ns, JSON = Date.now ms; a small clock offset is searched).
The homography absorbs the RIM/screen geometry *and* the gaze offset, so the
result sits on the cursor/tunnel. Needs no screen video and no calibration
recording.

Outputs: the gaze in task coordinates (aligned to the cursor), a residual report,
and a per-trial PDF (cursor path + calibrated gaze) so you can see it land on the
path.

USAGE
    python self_calibrate.py \
        --gaze_csv <task>_gaze.csv \
        --experiment_json steering_experiment_*.json \
        --out_csv <task>_gaze_selfcal.csv \
        --out_pdf <task>_selfcal_trials.pdf
"""
import argparse
import json

import numpy as np
import pandas as pd

TX = "gaze position transf x [px]"
TY = "gaze position transf y [px]"
DET = "gaze detected in reference image"
TS = "timestamp [ns]"


# --- homography helpers (normalized DLT + robust trimming) --------------------
def _fitH(src, dst):
    def nrm(P):
        m = P.mean(0)
        s = np.sqrt(2) / (np.sqrt(((P - m) ** 2).sum(1)).mean() + 1e-12)
        T = np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])
        return (T @ np.c_[P, np.ones(len(P))].T).T[:, :2], T
    s, Ts = nrm(src)
    d, Td = nrm(dst)
    A = []
    for (x, y), (u, v) in zip(s, d):
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
    _, _, Vt = np.linalg.svd(np.array(A), full_matrices=False)
    H = np.linalg.inv(Td) @ Vt[-1].reshape(3, 3) @ Ts
    return H / H[2, 2]


def _appH(H, P):
    Ph = (H @ np.c_[P, np.ones(len(P))].T).T
    return Ph[:, :2] / Ph[:, 2:3]


def _robustH(src, dst, it=3, keep=0.8):
    idx = np.arange(len(src))
    H = _fitH(src, dst)
    for _ in range(it):
        r = np.hypot(*(_appH(H, src) - dst).T)
        idx = np.where(r <= np.quantile(r, keep))[0]
        if len(idx) < 12:
            break
        H = _fitH(src[idx], dst[idx])
    return H, idx


def load_gaze(path):
    g = pd.read_csv(path)
    det = (g[DET].astype(str).str.lower().isin(["true", "1", "1.0"])
           if DET in g.columns else np.ones(len(g), bool))
    m = det.to_numpy() & np.isfinite(g[TX]).to_numpy() & np.isfinite(g[TY]).to_numpy()
    return g[TS].to_numpy(float)[m] / 1e9, g.loc[m, [TX, TY]].to_numpy(float)


def load_cursor(json_path):
    d = json.load(open(json_path))
    trials, ct, cxy = [], [], []
    ordinal = {}
    for tr in d.get("trialData", []):
        tid = tr.get("trial_id")
        if tid not in ordinal:
            ordinal[tid] = len(ordinal) + 1
    for tr in d.get("trialData", []):
        ts = tr.get("timestamps") or []
        tj = tr.get("trajectory") or []
        n = min(len(ts), len(tj))
        if n < 2:
            continue
        t = np.asarray(ts[:n], float) / 1e3
        P = np.array([[tj[k]["x"], tj[k]["y"]] for k in range(n)], float)
        ct.append(t)
        cxy.append(P)
        trials.append(dict(trial_id=tr.get("trial_id"),
                           ordinal=ordinal.get(tr.get("trial_id")),
                           n_trials=len(ordinal), round=tr.get("round"),
                           type=(tr.get("condition") or {}).get("tunnelType"),
                           t=t, P=P))
    return np.concatenate(ct), np.concatenate(cxy), trials, d


def tunnel_width(d):
    ws = [(x or {}).get("tunnelWidth") for tr in d.get("trialData", [])
          for x in (tr.get("difficulty") or []) if x and x.get("tunnelWidth")]
    return float(np.median(ws)) if ws else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaze_csv", required=True, help="task mapped-gaze CSV (transf px)")
    ap.add_argument("--experiment_json", required=True)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_pdf", default=None)
    ap.add_argument("--tol_ms", type=float, default=50.0, help="gaze<->cursor match tolerance")
    args = ap.parse_args()

    gt, gxy = load_gaze(args.gaze_csv)
    ct, cxy, trials, d = load_cursor(args.experiment_json)
    o = np.argsort(ct); ct, cxy = ct[o], cxy[o]
    W = tunnel_width(d)
    tol = args.tol_ms / 1000.0

    # --- clock-offset search: pick the offset giving the tightest fit ---------
    best = None
    for off in np.arange(-0.40, 0.401, 0.02):
        j = np.clip(np.searchsorted(ct, gt + off), 0, len(ct) - 1)
        m = np.abs(ct[j] - (gt + off)) < tol
        if m.sum() < 50:
            continue
        H, idx = _robustH(gxy[m], cxy[j[m]])
        r = np.hypot(*(_appH(H, gxy[m][idx]) - cxy[j[m]][idx]).T)
        med = float(np.median(r))
        if best is None or med < best[0]:
            best = (med, float(off), H)
    if best is None:
        raise SystemExit("could not match gaze to cursor — check timestamps/columns")
    med, off, H = best

    # --- apply to every gaze sample -------------------------------------------
    g_task = _appH(H, gxy)
    j = np.clip(np.searchsorted(ct, gt + off), 0, len(ct) - 1)
    dtm = np.abs(ct[j] - (gt + off))
    matched = dtm < tol
    cur = cxy[j]
    dist = np.hypot(*(g_task - cur).T)
    dm = dist[matched]

    print(f"self-calibration: {matched.sum()} gaze↔cursor matches, clock offset {off:+.2f}s")
    print(f"  gaze→cursor distance (task units): median {np.median(dm):.4f}, "
          f"mean {np.mean(dm):.4f}, 90th pct {np.quantile(dm, 0.9):.4f}")
    if np.isfinite(W):
        print(f"  tunnel width {W:.3f}  →  median gaze-cursor = {np.median(dm)/W:.2f}× width "
              f"(gaze lands on the path)")

    out_csv = args.out_csv or args.gaze_csv.rsplit(".", 1)[0] + "_selfcal.csv"
    pd.DataFrame({
        TS: (gt * 1e9).astype("int64"),
        "gaze_task_x": g_task[:, 0], "gaze_task_y": g_task[:, 1],
        "cursor_task_x": np.where(matched, cur[:, 0], np.nan),
        "cursor_task_y": np.where(matched, cur[:, 1], np.nan),
        "gaze_cursor_dist": np.where(matched, dist, np.nan),
        "match_dt_s": dtm,
    }).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    # --- per-trial PDF: cursor path + calibrated gaze -------------------------
    out_pdf = args.out_pdf or args.gaze_csv.rsplit(".", 1)[0] + "_selfcal_trials.pdf"
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    gt_off = gt + off
    n = 0
    with PdfPages(out_pdf) as pdf:
        for tr in trials:
            t0, t1 = tr["t"][0], tr["t"][-1]
            sel = (gt_off >= t0) & (gt_off <= t1) & matched
            if sel.sum() < 3:
                continue
            fig, ax = plt.subplots(figsize=(8, 3.2))
            ax.plot(tr["P"][:, 0], tr["P"][:, 1], "-", color="0.6", lw=6,
                    alpha=0.5, label="cursor path (tunnel)")
            gg = g_task[sel]
            tt = gt_off[sel]
            sc = ax.scatter(gg[:, 0], gg[:, 1], c=tt - tt[0], cmap="viridis",
                            s=10, label="calibrated gaze")
            ax.plot(tr["P"][0, 0], tr["P"][0, 1], "o", color="green", ms=8)  # start
            ax.plot(tr["P"][-1, 0], tr["P"][-1, 1], "*", color="red", ms=14)  # target
            ax.set_aspect("equal", adjustable="datalim")
            ax.invert_yaxis()   # task y grows downward on screen
            ax.set_title(f"Trial {tr['ordinal']}/{tr['n_trials']}  round {tr['round']}  —  "
                         f"{tr['type']}", fontsize=9)
            ax.legend(loc="upper left", fontsize=7, framealpha=0.6)
            ax.set_xlabel("task x"); ax.set_ylabel("task y")
            cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
            cb.set_label("s into trial", fontsize=7)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)
            n += 1
    print(f"wrote {out_pdf}: {n} trial pages (cursor path + calibrated gaze)")


if __name__ == "__main__":
    main()
