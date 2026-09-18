#!/usr/bin/env python3
"""
apply_selfcal.py — refine a calibrated task gaze using the task's own targets.

At PARKED-ON-TARGET fixations (cursor stationary + gaze in a fixation), the
participant is looking at the stationary red target, so gaze should equal the
target = cursor position. These moments occur at many target positions across
the screen, giving a dense reference set. We fit a per-axis 2D polynomial
    cursor_px ~ f(gaze_x, gaze_y)
on those pairs (robust, IRLS-trimmed) and apply it to ALL task gaze.

WHY PARKED ONLY: fitting over the moving cursor would pull gaze onto the cursor
and BIAS the gaze lead (the DV). Parked moments have no lead, so the fit captures
only the calibration/geometry error; the horizontal lead during motion is
preserved as the residual (gaze still sits ahead on the tunnel).

USAGE
  python apply_selfcal.py --aligned <P>_task_aligned.csv \
     --gaze_csv <P>_task_gaze_calibrated.csv --out <P>_task_gaze_calibrated.csv \
     [--order poly2|bilinear|affine  --speed_max 0.03  --screen_w 3840 --screen_h 2160]
Then re-align the --out gaze CSV.
"""
import argparse, numpy as np, pandas as pd
TX="gaze position transf x [px]"; TY="gaze position transf y [px]"


def des(x, y, order):
    x=np.asarray(x,float); y=np.asarray(y,float); e=np.ones_like(x)
    if order=="affine":   return np.column_stack([e,x,y])
    if order=="bilinear": return np.column_stack([e,x,y,x*y])
    return np.column_stack([e,x,y,x*x,y*y,x*y])


def robust_fit(D, z, iters=4):
    w=np.ones(len(z))
    c=np.linalg.lstsq(D,z,rcond=None)[0]
    for _ in range(iters):
        r=D@c-z; s=np.median(np.abs(r-np.median(r)))*1.4826+1e-9
        w=1.0/np.maximum(1.0, np.abs(r)/(3*s))      # down-weight >3 MAD
        c=np.linalg.lstsq(D*w[:,None], z*w, rcond=None)[0]
    return c


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--aligned",required=True); ap.add_argument("--gaze_csv",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--screen_w",type=float,default=3840); ap.add_argument("--screen_h",type=float,default=2160)
    ap.add_argument("--speed_max",type=float,default=0.03); ap.add_argument("--order",default="poly2",choices=["affine","bilinear","poly2"])
    a=ap.parse_args()
    A=pd.read_csv(a.aligned)
    it=A["in_trial"].astype(str).isin(["True","1","1.0"])
    park=A[it & A["fixation_id"].notna() & (A["cursor_speed"]<a.speed_max)].dropna(
        subset=["cursor_screenvid_x_px","cursor_screenvid_y_px","gaze_transf_x_px","gaze_transf_y_px"])
    gx=park["gaze_transf_x_px"].to_numpy(); gy=park["gaze_transf_y_px"].to_numpy()
    tx=park["cursor_screenvid_x_px"].to_numpy(); ty=park["cursor_screenvid_y_px"].to_numpy()
    order=a.order
    if len(park) < 200 and order=="poly2": order="bilinear"
    print(f"parked-on-target pairs: {len(park)}  target span x[{tx.min()/a.screen_w:.2f},{tx.max()/a.screen_w:.2f}] "
          f"y[{ty.min()/a.screen_h:.2f},{ty.max()/a.screen_h:.2f}]  model={order}")
    D=des(gx,gy,order); cxc=robust_fit(D,tx); cyc=robust_fit(D,ty)
    ax=D@cxc-tx; ay=D@cyc-ty; bx=gx-tx; by=gy-ty
    print(f"parked residual |dx|,|dy| median:  before ({np.median(np.abs(bx)):.0f},{np.median(np.abs(by)):.0f})px  "
          f"after ({np.median(np.abs(ax)):.0f},{np.median(np.abs(ay)):.0f})px")
    g=pd.read_csv(a.gaze_csv); Dg=des(g[TX].values,g[TY].values,order)
    g[TX]=Dg@cxc; g[TY]=Dg@cyc
    g.to_csv(a.out,index=False)
    print(f"wrote {a.out}  ({len(g)} rows; horizontal gaze-lead during motion preserved)")


if __name__=="__main__": main()
