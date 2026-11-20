from pathlib import Path
from collections import defaultdict
import csv,pickle
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; P=ROOT.parent; A=P/"repo"/"analysis"
def rows(p):
    with p.open(newline="") as f:return list(csv.DictReader(f))
def write(p,r):
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)

labels={}
for d,name in (("charge","sop_label_charge.csv"),("discharge","sop_label_measured.csv")):
    for r in rows(A/name):
        c,cycle,g,t=r["cell"],int(r["cycle"]),int(r["soc_group"]),float(r["tau_s"])
        key=f"{c}|{cycle}|{t:.1f}|{r['V_pre_V']}|{round(float(r['I_star_lin4_A']),3)}"
        labels[c,cycle,g,t,d]=(key,r)
with (P/"baseline"/"arms"/"arm_rows_replace_level2_hull2.pkl").open("rb") as f:
    result=pickle.load(f)
outer={}
for (level,_,c,d,arm,rule), rr in result.items():
    if level=="outer" and arm=="eo" and rule=="replace":
        for r in rr:
            if r["status"]=="ok" and r["meas"]>.5 and r["extrap"]<=1.5:
                outer[c,d,r["key"]]=r
lam={(r["cell"],r["direction"],float(r["tau_s"]),r["model"]):float(r["lambda"])
     for r in rows(P/"statistics"/"strict_percell.csv") if r["version"]=="strict_inner"}
out=[]
for r in rows(ROOT/"boundary"/"held_out_raw_pulses.csv"):
    if float(r["held_distance_mV"])>10.000001:continue
    cell,g,cycle,t,d=(r["cell"],int(r["group"]),int(r["cycle"]),float(r["tau_s"]),r["direction"])
    saved=labels.get((cell,cycle,g,t,d))
    if not saved:continue
    key,label=saved
    pred=outer.get((cell,d,key))
    if pred is None:continue
    same_mV=abs(float(r["V_pre"])-float(label["V_pre_V"]))*1e3
    dsoc=abs(float(r["SOC"])-float(label["SOC"]))
    observed=float(r["V_V"])
    volts_inside=observed<=4.2 if d=="charge" else observed>=2.5
    for mod in ("hyb","ecm"):
        lc=lam.get((cell,d,t,mod))
        if lc is None:continue
        Ia=lc*pred[mod]
        Iobs=abs(float(r["I_A"]))
        compatible=same_mV<=10.00001 and dsoc<=.00501
        out.append(dict(cell=cell,cycle=cycle,group=g,direction=d,tau_s=t,model=mod,
            actual_A=Iobs,measured_voltage=observed,baseline_HPPC_fit_A=pred["meas"],allow_A=Ia,
            allowed_minus_observed_A=Ia-Iobs,measured_endpoint_inside=int(volts_inside),
            model_allows_measured_violating_current=int(not volts_inside and Ia>=Iobs),
            model_at_or_below_observed_inside_current=int(volts_inside and Ia<=Iobs),
            comparable_preconditions=int(compatible),pre_voltage_delta_mV=same_mV,SOC_delta=dsoc))
assert out
write(ROOT/"boundary"/"direct_observable_a0_a8.csv",out)
for m in ("hyb","ecm"):
    rr=[r for r in out if r["model"]==m and r["comparable_preconditions"]]
    print(m,"comparable",len(rr),"cells",len({r["cell"] for r in rr}),"inside",sum(r["measured_endpoint_inside"] for r in rr),
          "allowed violating applied current",sum(r["model_allows_measured_violating_current"] for r in rr))
