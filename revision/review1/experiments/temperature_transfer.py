from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/"repo"; DEST=ROOT/"external"
sys.path[:0]=[str(REPO/"analysis"),str(REPO/"repro")]
from ecm_surface import ECMSurface
from run_chen2026_baseline import simulate_power, OutOfHull, tcell_map, SOH_25C, chen_search
from stages import CELLS
def load(p):
    with open(p,newline="") as f: return list(csv.DictReader(f))
def save(p,rows):
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def run_cold(name):
    TC=tcell_map()
    out=[]
    source=load(REPO/"analysis"/"rpcwby_sop_test3.csv")
    for direction, col in (("discharge","SOP_disch"),("charge","SOP_char")):
        surf=ECMSurface(name,direction)
        soh=SOH_25C
        def ocv(s):
            v, _=surf.ocv(s,soh)
            m, _=surf.hyst_M(s,soh)
            # The existing simulator subtracts the offset. Pass the negative
            # offset for charge so voltage follows the charge branch.
            return float(v[0]), -m if direction=="charge" else m
        for r in source:
            if not r[col] or float(r["tau_s"])!=10.:
                continue
            soc,temp,target=float(r["SOC"]),int(r["temp_C"]),abs(float(r[col]))
            if target<=0:
                continue
            T=TC.get(temp,float(temp))
            pred=np.nan
            if direction=="discharge":
                pred,status=chen_search(surf,soc,soh,T,10.,ocv)
            else:
                a,b=0.,30.
                status="unscored"
                for _ in range(24):
                    P=(a+b)/2
                    try:
                        v,i=simulate_power(surf,soc,soh,P,T,10.,ocv,sign=1.,v_stop=.05)
                    except OutOfHull:
                        status="out-of-hull"
                        pred=np.nan
                        break
                    if np.isfinite(v) and v<=4.1 and i<=5.:
                        a,pred,status=P,P,"ok"
                    else:
                        b=P
                    if b-a<1e-3:
                        break
            out.append(dict(surface=name,direction=direction,temp_C=temp,soc=soc,meas_W=target,
                            pred_W=pred,status=status))
    return out
def cross(rows, keys, model_cols, source, name):
    intersections=defaultdict(list)
    for r in rows:
        values=[float(r[k]) if r[k] else np.nan for k in model_cols]
        intersections[tuple(r[k] for k in keys)].append((r,values))
    def iss(row,k):
        return np.isfinite(row[1][k]) and row[1][k]>0
    out=[]
    for keys_, rr in sorted(intersections.items()):
        temps=sorted(set(int(r["temp_C"]) for r,_ in rr))
        if 25 not in temps:
            continue
        for ref_t, test_t in [(25,t) for t in temps if t!=25]+[(t,25) for t in temps if t!=25]:
            for k,col in enumerate(model_cols):
                selected=[(r,v) for r,v in rr if iss((r,v),k)]
                train=[(float(r[source]),v[k]) for r,v in selected if int(r["temp_C"])==ref_t]
                test=[(float(r[source]),v[k]) for r,v in selected if int(r["temp_C"])==test_t]
                if len(train)<3 or not test:
                    continue
                y,p=np.array(test).T
                f=min(m/r for m,r in train)
                threshold=min(y/p)
                frozen=.6832 if keys_[-1]=="discharge" else .5860
                out.append(dict(case="|".join(keys_),model=col,fit_C=ref_t,test_C=test_t,
                    n_fit=len(train),n_test=len(test),fitted_lambda=f,
                    lambda_needed_on_test=threshold,frozen_exceed=int(np.sum(frozen*p>y)),
                    refit_exceed=int(np.sum(f*p>y+1e-8)),no_refit_max_W=max(0.,float(np.max(frozen*p-y))),
                    refit_max_W=max(0.,float(np.max(f*p-y))),refit_median_ratio=float(np.median(f*p/y))))
    save(DEST/name,out)

if __name__=="__main__":
    if sys.argv[1]=="cold":
        with ProcessPoolExecutor(max_workers=6) as workers:
            rows=[r for block in workers.map(run_cold,CELLS) for r in block]
        save(DEST/"test3_all_surface_both_directions.csv",rows)
        cross(rows,("surface","direction"),("pred_W",),"meas_W","test3_temperature_transfer.csv")
        print("Test3 10 s measured-input proportion by temperature/direction, counted per model",len(rows))
    else:
        selected=[dict(r,surface=c) for c in CELLS for r in load(DEST/f"{c}.csv")]
        cross(selected,("surface","fold","direction"),("SOP_A0_W","SOP_A8_W"),"SOP_meas_W","test2_temperature_transfer.csv")
        print("Test2 bidirectional condition holdout",len(selected),"rows across surfaces/folds")
