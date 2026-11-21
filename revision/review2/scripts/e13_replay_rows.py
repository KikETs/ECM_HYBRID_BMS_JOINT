"""E13 step 1: review1 coupled_state.py on every eligible HPPC cycle (writes out/E13/replay_rows.csv)."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK as R2, REPO as REPO_ROOT, R1 as REV1
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import csv
import os
import pickle
import sys

import numpy as np
import torch

ROOT = Path(R2)
REPO = Path(REPO_ROOT)
ANALYSIS = REPO / "analysis"
BUNDLE = REPO / "nested"
OUT = ROOT / "out" / "E13"
OUT.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(ANALYSIS), str(BUNDLE/"scripts"), str(REPO/"repro")]

from ecm_pool import surfaces
import eval_sop_amps as ev
from eval_a13 import pulse_index
from ekf_soc import EKF
import hull_solver
from sop_trim_features import TrimFeatures
from run_safety_strict import fit_lambda

CELLS = ("BOOST","BOOST_NEGPULSE","BOOST_NEGPULSE_1S","BOOST_REST","CC","CC_CELL2")
DIRS = ("discharge","charge")
def table(path):
    with open(path,newline="") as f:
        return list(csv.DictReader(f))

def replay(cell):
    sd, sc = surfaces(cell)
    with np.load(ANALYSIS / "results" / "soh_pred.npz") as asset:
        cycs, soh_preds = asset[f"{cell}_cycle"], asset[f"{cell}_pred"]
    index, arrays = pulse_index(cell,cache=str(ANALYSIS/"cache_t"))
    locations = {}
    for (cyc, grp, rank, direction),(base,a,_,end,_) in index.items():
        tag = cyc, grp
        if tag not in locations or base+a < locations[tag][1]:
            locations[tag] = base, base+a, end
    candidates = defaultdict(list)
    for direction, lab in zip(DIRS, (ev.LABEL, ev.LABEL_CHG)):
        for r in table(lab):
            key = int(r["cycle"]), int(r["soc_group"])
            if (r["cell"]==cell and float(r["extrap"])<=1.5 and
                abs(float(r["I_star_lin4_A"]))>.5 and key in locations and np.any(cycs<key[0])):
                candidates[key[0]].append((direction,r,*locations[key]))
    ordered = sorted(candidates)
    selected = ordered  # E13: every eligible HPPC cycle (review1 used 8 evenly spaced cycles per cell)
    models={}
    for d, sub in zip(DIRS, ("runs_trim_a8","runs_trim_a8_chg")):
        seeds=torch.load(ANALYSIS/sub/f"model_A8_{cell}.pt",map_location="cpu",weights_only=False)["seeds"]
        models[d]=[(s["model"],s["mu"],s["sd"]) for s in seeds]
    def residual(d, snapshots):
        X=np.array(snapshots)
        X[:,1:] = 0
        factors=[]
        for st,mu,sigma in models[d]:
            w = next(v.numpy() for k,v in st.items() if k.endswith("weight"))
            b = next(v.numpy() for k,v in st.items() if k.endswith("bias"))
            factors.append(np.exp(np.tanh(np.clip((X-mu)/sigma,-4,4)@w.T+b)*[.470,.588]))
        return np.mean(factors,0).max(axis=0)
    results = []
    for cyc in selected:
        observations = candidates[cyc]
        base = min(r[2] for r in observations)
        assert all(r[2]==base for r in observations)
        soh = float(soh_preds[np.flatnonzero(cycs<cyc)[np.argmax(cycs[cycs<cyc])]])
        first_event = min(idx[3] for idx in observations)
        last_event = max(idx[3] for idx in observations)
        t, I, V, T = (arrays[k] for k in ("t","I","V","T"))
        valid = arrays["valid"] & np.isfinite(I) & np.isfinite(V) & np.isfinite(T) & np.isfinite(t)
        prior = np.arange(base,first_event)
        rest = prior[valid[prior] & (t[prior]-t[base]<=300.) & (abs(I[prior])<.1)]
        if len(rest)<2:
            print(cell,cyc,"initial rest unavailable",flush=True)
            continue
        initial = int(rest[-1])
        if t[initial]-t[rest[0]]<5.:
            print(cell,cyc,"initial rest shorter than 5 s",flush=True)
            continue
        grid = np.linspace(.02,1.,500)
        volts, inside = sd.ocv(grid,np.full(len(grid),soh))
        if not inside.any():
            print(cell,cyc,"initial OCV outside SOH hull",flush=True)
            continue
        ocv = float(np.median(V[rest]))
        pool_soc = grid[inside]
        z0 = float(pool_soc[np.argmin(abs(volts[inside]-ocv))])
        noise = float(np.interp(soh,[.70,.90,1.],[.110,.035,.015]))
        ekf=EKF(sd,sc,soh,R_volt=noise,gamma=0.,i_gate=1.,rest_hold_s=30.)
        ekf.x[0]=z0
        features=TrimFeatures(sd,sc)
        snapshots=[]
        next_time=float(t[initial])+600.
        last=initial
        # Use a one-second causal sample grid and finish each estimate before
        # the first pulse of its SOC group; the evaluated pulse is excluded.
        wanted = sorted({r[3] for r in observations})
        states={}
        cursor=0
        for pos in range(initial+1,last_event+1):
            while cursor<len(wanted) and pos == wanted[cursor]:
                states[pos]=(float(ekf.x[0]),tuple(snapshots[-12:]),last)
                cursor+=1
            if pos==last_event or not valid[pos] or t[pos]-t[last]<.98:
                continue
            dt=float(t[pos]-t[last])
            z,_=ekf.step(float(I[pos]),float(V[pos]),float(T[pos]),dt)
            features.update(dt,float(I[pos]),float(V[pos]),float(T[pos]),float(z),soh)
            last=pos
            if t[pos]>=next_time:
                snapshots.append(features.vector(z,soh))
                next_time=float(t[pos])+600.
        for direction,r,_,start,_ in observations:
            if start not in states:
                continue
            z,snaps,prev = states[start]
            K = float(r["I_star_lin4_A"])
            tau=float(r["tau_s"])
            v_l=float(r["V_min_V"])
            v_pre=float(V[prev])
            key=f"{cell}|{cyc}|{tau:.1f}|{r['V_pre_V']}|{round(K,3)}"
            common={"cell":cell,"cycle":cyc,"direction":direction,"tau_s":tau,
                    "key":key,"pred_soh":soh,"SOC_est":z,"SOC_ref":r["SOC"],
                    "v_pre_measured_before_group":v_pre,"source_sample":prev,
                    "first_pulse_sample":start,"delta_to_pulse_s":float(t[start]-t[prev]),
                    "n_feature_blocks":len(snaps),"meas_A":abs(K)}
            if len(snaps)<12:
                results.append({**common,"status":"short_history","i0_A":"","i8_A":""})
                continue
            assert prev<start
            surface=sc if direction=="charge" else sd
            kf,ks=residual(direction,snaps)
            i0,n0,_=hull_solver.solve(surface,z,soh,v_pre,v_l,tau,1.,1.,direction=="charge",ev.r_eff)
            i8,n8,_=hull_solver.solve(surface,z,soh,v_pre,v_l,tau,kf,ks,direction=="charge",ev.r_eff)
            results.append({**common,"status":"ok" if np.isfinite(i0) and np.isfinite(i8) else "solver_excluded",
                            "i0_A":abs(i0),"i8_A":abs(i8)})
        print(cell,cyc,len(observations),"labels",len(snapshots),"snapshots",flush=True)
    return results

def write(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    with ProcessPoolExecutor(max_workers=6) as pool:
        data=[r for cell in pool.map(replay,CELLS) for r in cell]
    assert data
    write(OUT/"replay_rows.csv",data)
    with (Path(R2)/"baseline"/"arms"/"arm_rows_replace_level2_hull2.pkl").open("rb") as f:
        baseline=pickle.load(f)
    treated=[]
    for direction in DIRS:
        for tau in (2.,10.):
            for cell in CELLS:
                source=[r for r in data if r["cell"]==cell and r["status"]=="ok" and
                        r["direction"]==direction and r["tau_s"]==tau]
                def keep(r): return r["tau"]==tau and r["status"]=="ok" and r["meas"]>.5 and r["extrap"]<=1.5
                ref={r["key"]:r for r in baseline["outer","NONE",cell,direction,"eo","replace"] if keep(r)}
                common=[r for r in source if r["key"] in ref]
                inner=[r for held in CELLS if held!=cell for r in baseline["inner",cell,held,direction,"eo","replace"] if keep(r)]
                if not common or len(inner)<25:
                    continue
                for model,target in (("ecm","i0_A"),("hyb","i8_A")):
                    f=fit_lambda(np.array([r[model] for r in inner]),np.array([r["meas"] for r in inner]),0.)
                    measured=np.array([r["meas_A"] for r in common])
                    p0=f*np.array([ref[r["key"]][model] for r in common])
                    p1=f*np.array([r[target] for r in common])
                    treated.append(dict(direction=direction,tau_s=tau,cell=cell,model=model,n_common=len(common),
                                        lambda_fixed=f,baseline_exceed=int((p0>measured).sum()),
                                        replay_exceed=int((p1>measured).sum()),
                                        baseline_usable=float(np.median(p0/measured)*100),
                                        replay_usable=float(np.median(p1/measured)*100)))
    if treated:
        write(OUT/"matched_fixed_lambda.csv",treated)
    print("replay statuses", {name:sum(r["status"]==name for r in data) for name in ("ok","short_history","solver_excluded")})
    print("matched comparison rows",len(treated))

if __name__=="__main__":
    main()
