from collections import defaultdict
import csv
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/"repo"
sys.path.insert(0,str(REPO/"analysis"))
from data30t import load_meas_mat
from uypydj_hppc_resistance import find_pulses, rank_pulses

result=[]
for temp in (-20,-10,0,10,25,40):
    for p in sorted((REPO/"raw"/"Mendeley"/f"{temp}degC").glob("*HPPC*.mat")):
        r=load_meas_mat(p); t,V,I,Ah=(r[k] for k in ("t","V","I","Ah"))
        jumps=np.flatnonzero(abs(np.diff(Ah))>.5)
        end=int(jumps[0])+1 if len(jumps) else len(Ah)
        t,V,I,Ah=(a[:end] for a in (t,V,I,Ah))
        by=defaultdict(dict)
        for a,b,g,rank in rank_pulses(find_pulses(t,I),t,I,Ah):
            rel=t[a:b]-t[a]
            for tau in (2.,10.):
                k=int(np.argmin(abs(rel-tau)))
                if abs(rel[k]-tau)>.151 or a<1:
                    continue
                ip=float(np.median(I[a:a+k+1]))
                if abs(ip)<1 or np.max(abs(I[a:a+k+1]-ip))>max(.1,abs(ip)*.02):
                    continue
                di="discharge" if ip<0 else "charge"
                by[(g,tau,di)][rank]=(ip,float(V[a+k]),float(rel[k]),float(1.+Ah[a]/3))
        for (g,tau,d), pts in by.items():
            if len(pts)<3:
                continue
            cutoff=4.2 if d=="charge" else 2.5
            for rank,(ip,v,stamp,z) in pts.items():
                others=[p for k,p in pts.items() if k!=rank]
                i,v2,*_=np.array(others).T
                if np.ptp(i)<.2:
                    continue
                b,a=np.polyfit(i,v2,1)
                if b<=0:
                    continue
                fit=(cutoff-a)/b
                result.append(dict(file=p.name,temp_C=temp,direction=d,tau_s=tau,soc_group=g,soc=z,
                    omitted_rank=rank,actual_s=stamp,measured_A=ip,measured_V=v,
                    cutoff_distance_mV=abs(v-cutoff)*1e3,estimated_cutoff_A=fit,
                    endpoint_prediction_error_mV=abs(a+b*ip-v)*1e3,estimated_minus_applied_abs_A=abs(fit)-abs(ip)))
assert result
with (ROOT/"reference"/"mendeley_held_rank_endpoints.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(result[0])); w.writeheader(); w.writerows(result)
for d in ("discharge","charge"):
    for tau in (2.,10.):
        rr=[r for r in result if r["direction"]==d and r["tau_s"]==tau]
        near=[r for r in rr if r["cutoff_distance_mV"]<=10.0001]
        print(d,tau,len(rr),"full-duration stable omitted pulses;",len(near),"within 10 mV")
        if near:
            print("  actual-to-predicted cutoff current", [round(r["estimated_minus_applied_abs_A"],3) for r in near])
