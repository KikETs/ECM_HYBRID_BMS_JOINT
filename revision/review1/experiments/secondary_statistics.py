from pathlib import Path
from collections import defaultdict
import csv
import itertools
import re
import numpy as np

RUN=Path(__file__).resolve().parents[1]
S=RUN/"statistics"
M=RUN/"baseline"/"methods"

def load(p):
    with p.open(newline="") as f: return list(csv.DictReader(f))
def save(p,rows):
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
assets={}
for p in sorted(M.glob("safety_strict_percell_*_est.csv")):
    name=p.name.split("_")[-2]
    if name!="a0":
        assets[name]={(r["direction"],r["tau_s"],r["cell"]):r for r in load(p)}
assert len(assets)==6
out=[]
for (dire,tau) in sorted(set((d,t) for d,t,c in assets["a8"])):
    for name, baseline in assets.items():
        if name=="a8":
            continue
        shared=sorted(k for k in assets["a8"].keys() & baseline.keys() if k[:2]==(dire,tau))
        r=[(assets["a8"][k],baseline[k]) for k in shared]
        # Equal cells for this secondary comparison: some methods score
        # different pulse rows and the saved cells do not prove shared keys.
        delta=np.array([float(a["usable_pct"])-float(b["usable_pct"]) for a,b in r])
        boot=delta[np.random.default_rng(12345).integers(0,len(r),(4000,len(r)))].mean(1)
        out.append(dict(direction=dire,tau_s=tau,method=name,n_cells=len(delta),
                        a8_n=sum(int(a["n"]) for a,b in r),method_n=sum(int(b["n"]) for a,b in r),
                        comparison_unit="same cells; matched pulse keys unverified",
                        a8_minus_method_pp=np.mean(delta),median_difference_pp=np.median(delta),
                        ci_low_pp=np.quantile(boot,.025),ci_high_pp=np.quantile(boot,.975),
                        a8_exceed=sum(int(a["exceed"]) for a,b in r),
                        method_exceed=sum(int(b["exceed"]) for a,b in r)))
save(S/"secondary_cell_paired.csv",out)

cold=load(RUN/"external"/"test3_all_surface_both_directions.csv")
tr=[]
for min_W in (0.,.1,.5,1.):
    for surface, d in sorted(set((r["surface"],r["direction"]) for r in cold)):
        kept=[r for r in cold if r["surface"]==surface and r["direction"]==d and float(r["meas_W"])>min_W and np.isfinite(float(r["pred_W"]))]
        warm=[r for r in kept if int(r["temp_C"])==25]
        if not warm:
            continue
        L=min(float(r["meas_W"])/float(r["pred_W"]) for r in warm)
        for temp in (-20,-10,0,10,40):
            test=[r for r in kept if int(r["temp_C"])==temp]
            if not test:
                continue
            fitted=np.array([L*float(r["pred_W"]) for r in test])
            y=np.array([float(r["meas_W"]) for r in test])
            tr.append(dict(surface=surface,direction=d,min_measured_W=min_W,fit_C=25,test_C=temp,
                           n_fit=len(warm),n_test=len(test),lambda_fitted=L,exceed=int((fitted>y+1e-8).sum()),
                           max_positive_W=max(0,float((fitted-y).max())),median_fraction=np.median(fitted/y)))
save(RUN/"external"/"test3_threshold_sensitivity.csv",tr)
print("Secondary six-method paired cell comparisons",len(out),"Test3 threshold sensitivity",len(tr))
