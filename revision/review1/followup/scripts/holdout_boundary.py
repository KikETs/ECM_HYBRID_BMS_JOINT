from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from pathlib import Path
import csv,re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
R=ROOT.parent/"repo"
sys.path.insert(0,str(R/"analysis"))
from uypydj_hppc_resistance import find_pulses,rank_pulses,defective_hppc
CELLS=("BOOST","BOOST_NEGPULSE","BOOST_NEGPULSE_1S","BOOST_REST","CC","CC_CELL2")
BAD=defective_hppc()

def task(cell):
    p=R/"analysis/cache_t"/f"uypydj_{cell}_HPPC.npz"
    with np.load(p,allow_pickle=True) as z:
        d={k:z[k] for k in ("t","I","V","SOC","T","SOH","files","lens","valid")}
    out=[]; summary=[]
    edges=np.r_[0,np.cumsum(d["lens"])]
    for f,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        name=str(d["files"][f]); cycle=int(name.partition("|")[0])
        if (cell,cycle) in BAD: continue
        t,I,V,S,T,H,F=(d[k][lo:hi] for k in ("t","I","V","SOC","T","SOH","valid"))
        blocks=defaultdict(list)
        for a,b,g,rank in rank_pulses(find_pulses(t,I),t,I,(S-1.)*3.):
            if a<1: continue
            dt=t[a:b]-t[a]
            for tau in (2.,10.):
                k=int(np.argmin(abs(dt-tau)))
                if abs(dt[k]-tau)>.15 or k<1 or not F[a:a+k+1].all(): continue
                ii=I[a:a+k+1]
                current=float(np.median(ii))
                if abs(current)<.5 or not np.isfinite(ii).all() or np.max(abs(ii-current))>max(.05,.02*abs(current)): continue
                direction="discharge" if current<0 else "charge"
                voltage=float(V[a+k])
                if not all(np.isfinite([voltage,V[a-1],H[a],S[a],T[a]])):continue
                rec=dict(cell=cell,cycle=cycle,group=g,tau_s=tau,direction=direction,rank=rank,file=name,
                         pulse_offset=int(a),actual_s=float(dt[k]),I_A=current,V_V=voltage,V_pre=float(V[a-1]),SOC=float(S[a]),SOH=float(H[a]),T_C=float(T[a]))
                blocks[(g,tau,direction)].append(rec)
        for (_,tau,direction),rows in blocks.items():
            cutoff=2.5 if direction=="discharge" else 4.2
            candidates=sorted(rows,key=lambda r:abs(r["V_V"]-cutoff))
            held=candidates[0]
            others=[r for r in rows if r is not held]
            if len(others)<3 or len({r["rank"] for r in others})<3: continue
            xx=np.array([r["I_A"] for r in others]); yy=np.array([r["V_V"] for r in others])
            if np.ptp(yy)<.020 or np.ptp(xx)<.5:continue
            b,a=np.polyfit(xx,yy,1)
            if b<=0.:continue
            hit=(cutoff-a)/b
            if hit*held["I_A"]<=0:continue
            out.append(dict(**held,fit_n=len(others),held_distance_mV=abs(held["V_V"]-cutoff)*1000,
                            predicted_boundary_A=hit,pred_minus_applied_magnitude_A=abs(hit)-abs(held["I_A"]),
                            held_voltage_prediction_error_mV=(a+b*held["I_A"]-held["V_V"])*1e3,
                            fit_r2=1-np.sum((yy-(a+b*xx))**2)/np.sum((yy-yy.mean())**2),
                            reserved_inside_training_current_span=int(min(abs(xx))<=abs(held["I_A"])<=max(abs(xx)))))
    return out

def write(p,r):
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows=[r for block in pool.map(task,CELLS) for r in block]
    dest=ROOT/"boundary"
    dest.mkdir(exist_ok=True)
    write(dest/"held_out_raw_pulses.csv",rows)
    summaries=[]
    for h in (2.,10.):
        for d in ("charge","discharge"):
            for band in (2.5,5.,10.,20.):
                a=[r for r in rows if r["tau_s"]==h and r["direction"]==d and r["held_distance_mV"]<=band+1e-7]
                e=np.array([r["pred_minus_applied_magnitude_A"] for r in a])
                cluster=[]
                for c in sorted({r["cell"] for r in a}):
                    cluster.append(np.array([r["pred_minus_applied_magnitude_A"] for r in a if r["cell"]==c]))
                rng=np.random.default_rng(12345)
                boot=[]
                if len(cluster)>1:
                    for chosen in rng.integers(0,len(cluster),(4000,len(cluster))):
                        boot.append(float(np.median(np.concatenate([cluster[j] for j in chosen]))))
                summaries.append(dict(tau_s=h,direction=d,band_mV=band,count=len(a),cells=len(cluster),
                    median_bias_A=float(np.median(e)) if len(e) else "",
                    bias_cell_boot_ci_low_A=np.quantile(boot,.025) if boot else "",
                    bias_cell_boot_ci_hi_A=np.quantile(boot,.975) if boot else "",
                    mae_A=np.mean(abs(e)) if len(e) else "",p95_abs_error_A=np.quantile(abs(e),.95) if len(e) else ""))
    write(dest/"cutoff_proximity_summary.csv",summaries)
    print("Retrospective raw-trace held-pulse tests",len(rows),flush=True)
    for a in summaries:
        if a["band_mV"]==10:print(a,flush=True)
