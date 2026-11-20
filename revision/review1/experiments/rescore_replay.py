from collections import defaultdict
import csv
from pathlib import Path
import pickle
import numpy as np

RUN=Path(__file__).resolve().parents[1]
def load(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

rows=load(RUN/"coupled_state"/"replay_rows.csv")
strict=load(RUN/"statistics"/"strict_percell.csv")
with (RUN/"baseline"/"arms"/"arm_rows_replace_level2_hull2.pkl").open("rb") as f:
    baseline=pickle.load(f)
cases=defaultdict(list)
for r in rows:
    if r["status"]=="ok":
        cases[r["direction"],r["tau_s"],r["cell"]].append(r)
out=[]
for lam in strict:
    if lam["version"]!="strict_inner": continue
    d,tau,c,m=(lam[k] for k in ("direction","tau_s","cell","model"))
    b={r["key"]:r for r in baseline["outer","NONE",c,d,"eo","replace"]
       if r["status"]=="ok" and r["tau"]==float(tau) and r["extrap"]<=1.5 and r["meas"]>.5}
    p=[r for r in cases[d,tau,c] if r["key"] in b]
    if not p: continue
    factor=float(lam["lambda"]); k="i8_A" if m=="hyb" else "i0_A"
    y=np.array([float(r["meas_A"]) for r in p])
    a=factor*np.array([float(b[r["key"]][m]) for r in p])
    q=factor*np.array([float(r[k]) for r in p])
    out.append(dict(direction=d,tau_s=tau,model=m,cell=c,lambda_strict=factor,n_common=len(y),
                    baseline_exceed=int((a>y).sum()),replay_exceed=int((q>y).sum()),
                    baseline_usable=float(np.median(a/y)*100),replay_usable=float(np.median(q/y)*100)))
assert len(out)==44
p=RUN/"coupled_state"/"matched_strict_lambda.csv"
with p.open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print("Replayed rows with original and strict lambda compared in",len(out),"cell/model/condition pairs")
for d,t in (("discharge","2.0"),("discharge","10.0"),("charge","2.0"),("charge","10.0")):
    rr=[r for r in out if (r["direction"],r["tau_s"],r["model"])==(d,t,"hyb")]
    print(d,t,sum(r["n_common"] for r in rr),sum(r["replay_exceed"] for r in rr))
