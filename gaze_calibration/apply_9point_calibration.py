#!/usr/bin/env python3
"""
9-point on-screen calibration transfer.

The CALIBRATION recording shows 9 dots at known screen fractions
(calibration_9point_timing.csv). We measure the mapped gaze (transf px) during
each dot's stable window, pair it with the dot's true screen pixel
(x_frac*W, y_frac*H), and fit a per-axis polynomial correction. That correction
is applied to the TASK recording's transf gaze (same wearing => same offset),
giving a calibrated task gaze CSV for cursor_gaze_pipeline.py.

MODEL ORDER IS ADAPTIVE (this matters when dots are missing):
  * poly2   [1,x,y,x^2,y^2,xy] — needs >=8 dots AND >=3 distinct x AND >=3
            distinct y positions, otherwise the quadratic terms are
            unconstrained and the fit OVERFITS + extrapolates off-screen
            (e.g. a participant missing the whole right column of dots).
  * bilinear[1,x,y,xy]          — >=5 dots, >=2 distinct x & y.
  * affine  [1,x,y]             — fallback.
Override with --model {auto,poly2,bilinear,affine} (default auto).

USAGE
  python3 apply_9point_calibration.py \
    --cal_gaze ... --timing ... --task_gaze ... --out ... \
    [--screen_w 3840 --screen_h 2160 --win_start 0.5 --win_end 2.2 --min_samples 30 --model auto]
"""
import argparse, os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_calibration_transfer import fit_homography, apply_H   # normalized-DLT homography (existing code)
TX="gaze position transf x [px]"; TY="gaze position transf y [px]"; TS="timestamp [ns]"


def design(x, y, order):
    x=np.asarray(x,float); y=np.asarray(y,float); o=np.ones_like(x)
    if order=="affine":   return np.column_stack([o, x, y])
    if order=="bilinear": return np.column_stack([o, x, y, x*y])
    return np.column_stack([o, x, y, x*x, y*y, x*y])          # poly2


def pick_order(xf, yf, n, forced="auto"):
    if forced and forced != "auto":
        return forced
    nx=len(np.unique(np.round(np.asarray(xf,float),2)))
    ny=len(np.unique(np.round(np.asarray(yf,float),2)))
    # homography = the physically right model for a constant scene-camera offset seen
    # through a flat screen (projective). 8 DOF; a spread 3x3 grid constrains it well.
    if n>=6 and nx>=3 and ny>=3: return "homography"
    if n>=5 and nx>=2 and ny>=2: return "bilinear"
    return "affine"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cal_gaze",required=True); ap.add_argument("--timing",required=True)
    ap.add_argument("--task_gaze",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--screen_w",type=float,default=3840); ap.add_argument("--screen_h",type=float,default=2160)
    ap.add_argument("--win_start",type=float,default=0.5); ap.add_argument("--win_end",type=float,default=2.2)
    ap.add_argument("--min_samples",type=int,default=30)
    ap.add_argument("--model",default="auto",choices=["auto","homography","poly2","bilinear","affine"])
    a=ap.parse_args()
    g=pd.read_csv(a.cal_gaze); t=pd.read_csv(a.timing)
    meas=[]; targ=[]; xf=[]; yf=[]
    for _,r in t.iterrows():
        on=int(r["onset_epoch_ms"])*1_000_000
        m=(g[TS]>=on+int(a.win_start*1e9))&(g[TS]<=on+int(a.win_end*1e9))
        if int(m.sum())<a.min_samples:
            print(f"  dot {int(r['point_id'])} ({r['x_frac']:.2f},{r['y_frac']:.2f}): only {int(m.sum())} samples -> skipped")
            continue
        meas.append([g.loc[m,TX].median(), g.loc[m,TY].median()])
        targ.append([r["x_frac"]*a.screen_w, r["y_frac"]*a.screen_h])
        xf.append(r["x_frac"]); yf.append(r["y_frac"])
    meas=np.array(meas); targ=np.array(targ)
    order=pick_order(xf, yf, len(meas), a.model)
    nx=len(np.unique(np.round(xf,2))); ny=len(np.unique(np.round(yf,2)))
    print(f"using {len(meas)} dots ({nx} distinct x, {ny} distinct y) -> model={order}"
          + ("  [auto-reduced: too few/too-collinear dots for a homography]" if a.model=="auto" and order in ("bilinear","affine") else ""))
    tk=pd.read_csv(a.task_gaze)
    if order=="homography":
        H=fit_homography(meas,targ)                       # measured gaze px -> true dot px
        pred=apply_H(H,meas); dof=2*len(meas)-8
        P=tk[[TX,TY]].to_numpy(float); ok=np.isfinite(P).all(1)
        Pc=P.copy(); Pc[ok]=apply_H(H,P[ok]); tk[TX]=Pc[:,0]; tk[TY]=Pc[:,1]
    else:
        D=design(meas[:,0],meas[:,1],order)
        cx,_,_,_=np.linalg.lstsq(D,targ[:,0],rcond=None)
        cy,_,_,_=np.linalg.lstsq(D,targ[:,1],rcond=None)
        pred=np.column_stack([D@cx, D@cy]); dof=len(meas)-D.shape[1]
        Dt=design(tk[TX].values,tk[TY].values,order); tk[TX]=Dt@cx; tk[TY]=Dt@cy
    before=np.sqrt(np.mean(np.sum((meas-targ)**2,1)))
    after =np.sqrt(np.mean(np.sum((pred-targ)**2,1)))
    print(f"dot RMS error  before={before:.1f}px  after={after:.1f}px  (dof={dof}; "
          f"{'OK' if dof>=2 else 'WARN exactly/over-determined -> overfit risk'})")
    tk.to_csv(a.out,index=False)
    print(f"wrote {a.out}  ({len(tk)} rows)")


if __name__=="__main__": main()
