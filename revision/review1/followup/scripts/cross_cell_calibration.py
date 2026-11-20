from collections import defaultdict
from pathlib import Path
import csv, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT.parent
E=ROOT/"external"
CELLS=("BOOST","BOOST_NEGPULSE","BOOST_NEGPULSE_1S","BOOST_REST","CC","CC_CELL2")

def read(p):
    with p.open(newline="") as f: return list(csv.DictReader(f))
def write(p,rows):
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def collapse(files,model):
    D=defaultdict(list)
    for surf,path in files:
        for r in read(path):
            if not r[model] or not r["SOP_A0_W"] or not r["SOP_A8_W"]:continue
            y,p=float(r["SOP_meas_W"]),float(r[model])
            if y>0 and p>0 and np.isfinite(y+p):
                D[surf,r["fold"],r["direction"],int(r["temp_C"])].append((y,p))
    return D
def score(rows,lambda_):
    y,p=np.array(rows).T
    excess=lambda_*p-y
    return int((excess>1e-8).sum()), max(0.,float(excess.max())), float(np.median(100*lambda_*p/y))

def main():
    external={}; results=[]; bysim=[]
    for label, files in (("Test1",[(s,E/f"Test1_{s}.csv") for s in CELLS]),("Test2",[(s,BASE/"external"/f"{s}.csv") for s in CELLS])):
        for m in ("SOP_A0_W","SOP_A8_W"):
            external[label,m]=collapse(files,m)
    for m in ("SOP_A0_W","SOP_A8_W"):
        for test,train in (("Test1","Test2"),("Test2","Test1")):
            for key, obs in sorted(external[test,m].items()):
                surface,fold,direction,T=key
                p0=.6832 if direction=="discharge" else .5860
                train_record=external[train,m]
                for Tmin in (0.,1.):
                    test_rows=[r for r in obs if r[0]>Tmin]
                    if not test_rows:continue
                    for rule in ("temperature_matched","25C_only","pooled_10_25"):
                        pool=[]
                        for train_t in ((T,) if rule=="temperature_matched" else ((25,) if rule=="25C_only" else (10,25))):
                            pool+=train_record.get((surface,fold,direction,train_t),[])
                        tr=[r for r in pool if r[0]>Tmin]
                        if not tr:continue
                        lam=min(y/p for y,p in tr)
                        fo,fmax,fu=score(test_rows,p0)
                        po,pmax,pu=score(test_rows,lam)
                        results.append(dict(model=m,test_cell=test,cal_cell=train,surface=surface,fold=fold,direction=direction,
                                            test_temp_C=T,policy=rule,min_measured_W=Tmin,n_cal=len(tr),n=len(test_rows),
                                            original_lambda=p0,new_lambda=lam,original_exceed=fo,calibrated_exceed=po,
                                            original_usable_pct=fu,calibrated_usable_pct=pu,
                                            original_max_excess_W=fmax,calibrated_max_excess_W=pmax))
    write(E/"cross_physical_cell.csv",results)

    tests=list(E.glob("Test3_all_prefix_*.csv"))
    if len(tests)!=6:
        raise ValueError(f"Missing independent discharge continuation: {len(tests)}/6")
    cold=[]; counts=defaultdict(set)
    for f in tests:
        for r in read(f):
            counts[(int(r["T_set_C"]),int(r["tau_s"]))].add((r["member"],r["raw_row"]))
            if not np.isfinite(float(r["rcsop_W"])):continue
            c,q,T=r["surface"],r["fold"],int(r["T_set_C"])
            train=[item for train_cell in ("Test1","Test2") for temp in (10,25)
                for item in external[train_cell,"SOP_A8_W"].get((c,q,"discharge",temp),[])]
            if not train:continue
            lam=min(y/p for y,p in train)
            measured=float(r["measured_W"]); pred=float(r["rcsop_W"])
            cold.append(dict(**r,lambda_other_two_cells=lam,
                    exceed_frozen=int(.6832*pred>measured+1e-8),
                    exceed_cross_cell=int(lam*pred>measured+1e-8)))
    print("Third-cell measured raw pulse inventory", {str(k):len(v) for k,v in counts.items()})
    if cold:write(E/"test3_two_other_cell_calibration.csv",cold)
    print("Test3",len(cold),"scored surface/fold/pulse combinations")
    primary=[r for r in results if r["policy"]=="temperature_matched" and r["min_measured_W"]==0. and r["model"]=="SOP_A8_W"]
    for key in sorted({(r["test_cell"],r["direction"],r["test_temp_C"]) for r in primary}):
        rs=[r for r in primary if (r["test_cell"],r["direction"],r["test_temp_C"])==key]
        print("Test1/Test2",key,"n",sum(r["n"] for r in rs),"exceed",
              sum(r["original_exceed"] for r in rs),"->",sum(r["calibrated_exceed"] for r in rs))
if __name__=="__main__":
    main()
