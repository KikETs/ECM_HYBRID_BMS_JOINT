import argparse
from collections import defaultdict
import csv
import itertools
from pathlib import Path
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np
from build_strict_inner import REPO, RUN, EXPERIMENT, CELLS, DIRECTIONS

B = REPO / "nested"
baseline_path = RUN / "baseline" / "arms" / "arm_rows_replace_level2_hull2.pkl"
out = RUN / "statistics"
out.mkdir(exist_ok=True)
sys.path[:0] = [str(REPO/"repro"), str(B/"scripts"/"nested")]
from run_safety_strict import fit_lambda

def read(path):
    with path.open(newline="") as f: return list(csv.DictReader(f))
def write(path, rows):
    if not rows:
        raise ValueError(f"Empty output: {path}")
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def boot(d,w):
    gen=np.random.default_rng(12345)
    idx=gen.integers(0,len(d),(4000,len(d)))
    return np.quantile(np.sum(d[idx]*w[idx],axis=1)/np.sum(w[idx],axis=1),[.025,.975])

def get_strict_inner():
    outdir=EXPERIMENT/"scored"
    outdir.mkdir(exist_ok=True)
    cached=outdir/"strict_results_inner.pkl"
    if cached.exists():
        with cached.open("rb") as f:
            return pickle.load(f)
    os.environ.update(RCSOP_REPO=str(REPO), RCSOP_BUNDLE=str(B),
                      INNERTRIM=str(EXPERIMENT/"predictions"), SOCTAG="_fixpool",
                      SOLVER="hull", OUTDIR=str(RUN/"baseline"/"arms"))
    from eval_arms_nested import rows_for
    tasks=[("inner",c,j,d,"eo","replace") for c in CELLS for j in CELLS if j!=c for d in DIRECTIONS]
    with mp.Pool(10) as pool:
        rows=dict(pool.starmap(rows_for,tasks))
    assert len(rows)==60
    with (outdir/"strict_results_inner.pkl").open("wb") as f:
        pickle.dump(rows,f)
    return rows

def main(mode):
    with baseline_path.open("rb") as f: pre=pickle.load(f)
    sources={"original":pre}
    if mode=="strict":
        sources["strict_inner"]=get_strict_inner()
    labels={}
    for direction,name in (("discharge","sop_label_measured.csv"),("charge","sop_label_charge.csv")):
        for r in read(REPO/"analysis"/name):
            key=f"{r['cell']}|{r['cycle']}|{float(r['tau_s']):.1f}|{r['V_pre_V']}|{round(float(r['I_star_lin4_A']),3)}"
            if r["I_star_lin2hi_A"]:
                val=float(r["I_star_lin2hi_A"])
                if val * (1 if direction=="charge" else -1) > .5:
                    labels[direction,key]=abs(val)
    by_cell, paired, uncertainty=[],[],[]
    for version, source in sources.items():
        for direction in DIRECTIONS:
            for tau in (2.,10.):
                def keep(r):
                    return r["tau"]==tau and r["status"]=="ok" and r["meas"]>.5 and r["extrap"]<=1.5
                both=[]
                for c in CELLS:
                    rr=[r for r in pre["outer","NONE",c,direction,"eo","replace"] if keep(r)]
                    irows=[r for j in CELLS if j!=c for r in source["inner",c,j,direction,"eo","replace"] if keep(r)]
                    if not rr or len(irows)<25:
                        continue
                    pair=[]
                    for model in ("ecm","hyb"):
                        limits=fit_lambda(np.array([r[model] for r in irows]),np.array([r["meas"] for r in irows]),0.)
                        y=np.array([r["meas"] for r in rr]); p=limits*np.array([r[model] for r in rr])
                        dd=p-y
                        record={"version":version,"direction":direction,"tau_s":tau,"cell":c,"model":model,
                                "lambda":limits,"n_inner":len(irows),"n_outer":len(rr),"n_exceed":int((dd>0).sum()),
                                "max_positive_A":max(0.,float(dd.max())),
                                "median_usable_pct":float(np.median(p/y)*100)}
                        by_cell.append(record); pair.append(record)
                        # Hold rows fixed and change only the observed reference fit.
                        for reference in ("all_rank","high_two"):
                            ri=[r for r in irows if (direction,r["key"]) in labels]
                            ro=[r for r in rr if (direction,r["key"]) in labels]
                            if len(ri)<25 or not ro:
                                continue
                            def y_(r): return r["meas"] if reference=="all_rank" else labels[direction,r["key"]]
                            l=fit_lambda(np.array([r[model] for r in ri]),np.array([y_(r) for r in ri]),0.)
                            yy=np.array([y_(r) for r in ro]); pp=l*np.array([r[model] for r in ro])
                            uncertainty.append(dict(version=version,direction=direction,tau_s=tau,cell=c,model=model,reference=reference,
                                                    n_inner=len(ri),n_outer=len(ro),lambda_c=l,
                                                    exceed=int((pp>yy).sum()),usable=float(np.median(pp/yy)*100)))
                    both.append(pair)
                if not both: continue
                n=np.array([m[0]["n_outer"] for m in both])
                a,b=(np.array([m[i]["median_usable_pct"] for m in both]) for i in (0,1))
                assert np.all(n>0)
                low,high=boot(b-a,n)
                paired.append(dict(version=version,direction=direction,tau_s=tau,n_cells=len(both),n_rows=n.sum(),
                                   a0_usable_pct=np.average(a,weights=n),a8_usable_pct=np.average(b,weights=n),
                                   paired_diff_pp=np.average(b-a,weights=n),pair_ci_lo_pp=low,pair_ci_hi_pp=high,
                                   a0_exceed=sum(x[0]["n_exceed"] for x in both),a8_exceed=sum(x[1]["n_exceed"] for x in both),
                                   cell_differences=";".join(f"{x[0]['cell']}:{k:.5f}" for x,k in zip(both,b-a))))
    write(out/f"{mode}_percell.csv",by_cell)
    write(out/f"{mode}_paired.csv",paired)
    write(out/"reference_fit_sensitivity_percell.csv",uncertainty)
    if mode=="strict":
        data={}
        for r in by_cell:
            data.setdefault((r["direction"],r["tau_s"],r["cell"],r["model"]),{})[r["version"]]=r
        diffs=[]
        for key, results in data.items():
            a,b=results["original"],results["strict_inner"]
            assert a["n_outer"]==b["n_outer"]
            diffs.append(dict(direction=key[0],tau_s=key[1],cell=key[2],model=key[3],
                              n_outer=a["n_outer"],original_lambda=a["lambda"],strict_lambda=b["lambda"],
                              lambda_change=b["lambda"]-a["lambda"],original_exceed=a["n_exceed"],
                              strict_exceed=b["n_exceed"],usable_change_pp=b["median_usable_pct"]-a["median_usable_pct"]))
        write(out/"strict_vs_original.csv",diffs)
    print(f"{mode}: {len(by_cell)} per-cell, {len(paired)} paired-condition summaries",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",choices=("baseline","strict"))
    main(ap.parse_args().mode)
