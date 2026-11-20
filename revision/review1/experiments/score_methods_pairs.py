import csv
import os
from pathlib import Path
import pickle
import sys
import numpy as np

def pairs(res, module, out):
    fields=[]
    for d in ("discharge","charge"):
        for tau in (2.,10.):
            tol=module.TOL[d]
            for m in ("a3","shrink","lstm","gru","ffrls"):
                effects=[]; weights=[]; n_exceed=[0,0]
                for c in module.CELLS:
                    tr=[[r for j in module.CELLS if j!=c for r in res[(k,"inner",c,j,d)] if module.keep(r,tau,"hyb")]
                        for k in ("a8",m)]
                    if any(len(r)<module.MIN_TRAIN for r in tr):
                        continue
                    L=[module.fit_lambda(np.array([r["hyb"] for r in rows]),np.array([r["meas"] for r in rows]),tol)
                       for rows in tr]
                    valid=[[r for r in res[k,"outer",c,c,d] if module.keep(r,tau,"hyb")] for k in ("a8",m)]
                    outer=[{r["key"]:r for r in rows} for rows in valid]
                    if any(len(rows)!=len(q) for rows,q in zip(valid,outer)):
                        raise ValueError((m,c,d,tau,"duplicate outer pulse keys"))
                    common=sorted(outer[0].keys()&outer[1].keys())
                    if not common:
                        continue
                    y=np.array([outer[0][r]["meas"] for r in common])
                    assert all(outer[0][r]["meas"]==outer[1][r]["meas"] for r in common)
                    p=[lam*np.array([pred[r]["hyb"] for r in common]) for lam,pred in zip(L,outer)]
                    effects.append(100*(np.median(p[0]/y)-np.median(p[1]/y)))
                    weights.append(len(common))
                    for n in range(2):
                        n_exceed[n]+=int(np.sum(p[n]-y>tol))
                delta=np.array(effects); w=np.array(weights)
                if not len(delta):
                    continue
                idx=np.random.default_rng(12345).integers(0,len(delta),(4000,len(delta)))
                boot=np.sum((delta*w)[idx],axis=1)/np.sum(w[idx],axis=1)
                ci=np.quantile(boot,[.025,.975])
                fields.append(dict(direction=d,tau_s=tau,method=m,n_cells=len(delta),n_common=sum(weights),
                                   paired_diff_pp=np.average(delta,weights=w),ci_lo_pp=ci[0],ci_hi_pp=ci[1],
                                   rc_sop_exceed=n_exceed[0],method_exceed=n_exceed[1]))
    assert len(fields)==20
    with open(out/"matched_pairs.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(fields[0])); w.writeheader(); w.writerows(fields)
    print(f"Saved {len(fields)} matched common-pulse comparator effects.",flush=True)

if __name__=="__main__":
    source=Path(os.environ["RCSOP_BUNDLE"])/"scripts/nested/score_methods_nested.py"
    p=source.read_text()
    before="""extrap=float(r['extrap']), tau=tau))"""
    after="""extrap=float(r['extrap']), tau=tau,
                        key=(cyc, r['soc_group'], r['V_pre_V'], r['I_star_lin4_A'])))"""
    calc="""        res = dict(pool.starmap(rows_for, tasks))"""
    inject="""        res = dict(pool.starmap(rows_for, tasks))
    pairs(res, sys.modules[__name__], Path(OUT))"""
    assert p.count(before)==p.count(calc)==1
    p=p.replace(before,after).replace(calc,inject)
    exec(compile(p,str(source),"exec"))
