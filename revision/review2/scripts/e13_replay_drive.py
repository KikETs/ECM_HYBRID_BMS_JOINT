"""E13 step 2: review1 followup replay_preceding_drive.py applied to out/E13/replay_rows.csv."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK as R2, REPO as REPO_ROOT, R1 as REV1
import bisect
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv, datetime as dt
from pathlib import Path
import re, sys
import numpy as np
import torch

ROOT=Path(R2)
BASE=Path(REV1); R=Path(REPO_ROOT); A=R/"analysis"
OUT=ROOT/"out"/"E13"
sys.path[:0]=[str(A),str(R/"nested/scripts"),str(R/"repro")]
import eval_sop_amps as ev
from ecm_pool import surfaces
from ekf_soc import EKF
from sop_trim_features import TrimFeatures
from temp_defects import defective
import hull_solver

CELLS=("BOOST","BOOST_NEGPULSE","BOOST_NEGPULSE_1S","BOOST_REST","CC","CC_CELL2")
def read(p):
    with p.open(newline="") as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stamp(file):
    m=re.search(r"(\d\d-\d\d-\d\d)_([\d.]+)",str(file))
    if not m:return None
    for fm in ("%m-%d-%y %H.%M","%m-%d-%y %H"):
        try:return dt.datetime.strptime(f"{m[1]} {m[2]}",fm)
        except ValueError:pass
    return None
def branch(cell):
    sd,sc=surfaces(cell)
    avec=lambda kind:A/"cache_t"/f"uypydj_{cell}_{kind}.npz"
    with np.load(avec("HPPC"),allow_pickle=True) as z:
        hppc_dates={int(s.partition("|")[0]):stamp(s) for s in z["files"]}
    with np.load(avec("Fifteen_Drive_Cycles"),allow_pickle=True) as z:
        d={k:z[k] for k in ("files","lens","valid","t","I","V","T")}
    with np.load(A/"results/soh_pred.npz") as z:
        cycles, health=z[f"{cell}_cycle"], z[f"{cell}_pred"]
    bad={n for c,n in defective("drive") if c==cell}
    edges=np.r_[0,np.cumsum(d["lens"])]
    indexed=[]
    for n,s in enumerate(d["files"]):
        date=stamp(s)
        k=int(s.partition("|")[0])
        if date is not None and k not in bad:
            last=int(edges[n+1]-1)
            ended=date+dt.timedelta(seconds=float(d["t"][last]-d["t"][edges[n]]))
            indexed.append((k,n,ended,date))
    idx={}
    def feature(index):
        if index in idx:return idx[index]
        n=index
        k=int(str(d["files"][n]).partition("|")[0])
        sol=cycles<k
        if not sol.any():
            idx[index]=None
            return None
        h=float(health[np.flatnonzero(sol)[np.argmax(cycles[sol])]])
        b,e=edges[n],edges[n+1]; t,I,V,T=(d[f][b:e] for f in ("t","I","V","T"))
        valid=d["valid"][b:e] & np.isfinite(t+I+V+T)
        if not valid.any():return None
        ts,cur,volt,tmp=(a[valid] for a in (t,I,V,T))
        rest=np.flatnonzero((abs(cur)<=.1)&(ts-ts[0]<=300))
        if len(rest)<2 or ts[rest[-1]]-ts[rest[0]]<5.:
            idx[index]=None
            return None
        init=rest[-1]
        socgrid=np.linspace(.02,1.,500)
        vv,inside=sd.ocv(socgrid,np.full(len(socgrid),h))
        if not inside.any():
            idx[index]=None
            return None
        v0=float(np.median(volt[rest]))
        z0=socgrid[inside][np.argmin(abs(vv[inside]-v0))]
        ekf=EKF(sd,sc,h,R_volt=float(np.interp(h,[.70,.90,1.],[.110,.035,.015])),
            gamma=0.,i_gate=1.,rest_hold_s=30.)
        ekf.x[0]=z0
        track=TrimFeatures(sd,sc)
        p=init;mark=ts[init]+600.; xs=[]
        for j in range(init+1,len(ts)):
            if ts[j]-ts[p]<.96:continue
            delta=float(ts[j]-ts[p]);p=j
            z,_=ekf.step(float(cur[j]),float(volt[j]),float(tmp[j]),delta)
            track.update(delta,float(cur[j]),float(volt[j]),float(tmp[j]),float(z),h)
            if ts[j]>=mark:
                xs.append(track.vector(z,h));mark=ts[j]+600.
        idx[index]=(np.stack(xs[-12:]),h) if len(xs)>=12 else None
        return idx[index]
    models={}
    for dname,sub in (("discharge","runs_trim_a8"),("charge","runs_trim_a8_chg")):
        models[dname]=torch.load(A/sub/f"model_A8_{cell}.pt",map_location="cpu",weights_only=False)["seeds"]
    def factors(dname,x):
        x=x.copy();x[:,1:]=0
        out=[]
        for state in models[dname]:
            W=next(v.numpy() for k,v in state["model"].items() if k.endswith("weight"))
            B=next(v.numpy() for k,v in state["model"].items() if k.endswith("bias"))
            out.append(np.exp(np.tanh(np.clip((x-state["mu"])/state["sd"],-4,4)@W.T+B)*[.470,.588]))
        return np.mean(out,0).max(0)
    out=[]; prev_fit={}
    for r in read(OUT/"replay_rows.csv"):
        if r["cell"]!=cell:continue
        c=int(r["cycle"]); date=hppc_dates.get(c)
        has = sorted((j for j in indexed if j[0]<c and date and j[2]<date),key=lambda x:(x[0],x[2]),reverse=True)
        prep={**r,"drive_cycle":"","drive_ended":"","hppc_started":date.isoformat() if date else "",
              "drive_soh":"","paired_status":"no_timestamp_confirmed_history","pred_A8_A":"","pred_A0_A":""}
        if has:
            k,n,ended,start=has[0]
            prep.update(drive_cycle=k,drive_ended=ended.isoformat(),paired_status="initial_rest_or_history_inadequate")
            f=feature(n)
            if f is not None:
                x,h=f; zd=float(r["SOC_est"]); hp=float(r["pred_soh"]); tau=float(r["tau_s"])
                dname=r["direction"]; ch=dname=="charge"; surf=sc if ch else sd; vp=float(r["v_pre_measured_before_group"])
                vlim=4.2 if ch else 2.5
                if (n,dname) not in prev_fit: prev_fit[n,dname]=factors(dname,x)
                kf,ks=prev_fit[n,dname]
                a8,_,_=hull_solver.solve(surf,zd,hp,vp,vlim,tau,kf,ks,ch,ev.r_eff)
                a0,_,_=hull_solver.solve(surf,zd,hp,vp,vlim,tau,1.,1.,ch,ev.r_eff)
                prep.update(drive_soh=h,paired_status="ok" if np.isfinite(a8+a0) else "solver_exclusion",
                            pred_A8_A=abs(a8),pred_A0_A=abs(a0))
        out.append(prep)
    print(cell,"rows",len(out),"history runs",len(idx),"scored",sum(r["paired_status"]=="ok" for r in out),flush=True)
    return out

if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows=[row for rr in pool.map(branch,CELLS) for row in rr]
    assert rows
    dest=OUT;dest.mkdir(parents=True,exist_ok=True)
    write(dest/"estimated_state_preceding_drive.csv",rows)
    factors={}
    for r in read(BASE/"statistics"/"strict_percell.csv"):
        if r["version"]=="strict_inner":
            factors[(r["cell"],r["direction"],r["tau_s"],r["model"])]=float(r["lambda"])
    out=[]
    for d in ("discharge","charge"):
        for t in ("2.0","10.0"):
            for c in CELLS:
                pts=[r for r in rows if r["direction"]==d and r["tau_s"]==t and r["cell"]==c and r["paired_status"]=="ok"]
                if not pts: continue
                for m,k in (("hyb","pred_A8_A"),("ecm","pred_A0_A")):
                    if (c,d,t,m) not in factors:continue
                    y=np.array([float(r["meas_A"]) for r in pts])
                    p=factors[c,d,t,m]*np.array([float(r[k]) for r in pts])
                    out.append(dict(direction=d,tau_s=t,cell=c,model=m,n=len(y),
                                    n_exceed=int(np.count_nonzero(p>y)),usable_pct=float(np.median(100*p/y))))
    if out:write(dest/"fixed_lambda_summary.csv",out)
