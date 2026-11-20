from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import itertools
import os
import sys

import numpy as np
from build_strict_inner import REPO, RUN, EXPERIMENT, CELLS, DIRECTIONS
from train_strict_inner import EXP
sys.path.insert(0,str(REPO/"repro"))

def get_direction(pair,direction,seq,trim):
    sub=DIRECTIONS[direction]
    data=EXPERIMENT/"datasets"/"__".join(pair)/sub
    dirs=[(c,j,EXP/f"{sub}_no{c}") for c,j in (pair,pair[::-1])]
    source=trim.load_cells(str(data))
    assert set(source)==set(CELLS)-set(pair)
    training_seq=seq.load_cells(str(data))
    test_cases=[]
    for c,j,p in dirs:
        t=trim.load_cells(str(p))[j]
        t_seq=seq.load_cells(str(p))[j]
        test_cases.append((c,j,t,t_seq))
    return source, training_seq, test_cases

def mb(d, seq):
    ind=seq.to_sequences(d)
    return {"idx":ind,"Xs":np.asarray(d["X"],float)[ind],
            "Yl":np.asarray(d["Y"],float)[ind][:,0],
            "NOMl":np.asarray(d["NOM"],float)[ind][:,0],
            "Il":np.asarray(d["I"],float)[ind][:,0]}

def output(c,direction,method):
    default={"a3":"v2","shrink":"shrink","ffrls":"ffrls","lstm":"lstm","gru":"gru"}[method]
    suffix="_chg" if direction=="charge" else ""
    return EXPERIMENT/"basetrees"/f"no{c}"/"analysis"/f"runs_trim_{default}{suffix}"

def task(pair,direction):
    import torch
    import sop_trim as trim
    import sop_baseline_fill as shrink
    import run_sop_seq_baselines as seq
    from train_graph import train_graph
    torch.set_num_threads(1)
    source, training_seq, tests=get_direction(pair,direction,seq,trim)
    features=[mb(training_seq[c],seq) for c in sorted(training_seq)]
    via=[mb(item[-1],seq) for item in tests]
    def params(method):
        return [output(h[0],direction,method)/f"pred_A3_{h[1]}.npz" for h in tests]
    def save_old(method,kf,ks,recurrent):
        offset=0
        for case, ts in zip(tests,via):
            c,j,t,tsource=case
            n=len(ts["Xs"]) if recurrent else len(t["I"])
            sl=slice(offset,offset+n)
            dest=str(output(c,direction,method))
            if recurrent:
                seq.dump(dest,j,kf[sl],ks[sl],tsource,ts["idx"])
            else:
                shrink.dump(dest,j,kf[sl],ks[sl],t)
            offset=sl.stop
    if MODE=="cpu":
        if not all(x.exists() for x in params("shrink")):
            a,b=shrink.fit_alpha([source[k] for k in sorted(source)])
            r,s=map(np.concatenate,zip(*(shrink.ratios(v[2]) for v in tests)))
            save_old("shrink",np.exp(a*r),np.exp(b*s),False)
        if not all(x.exists() for x in params("ffrls")):
            best_ff,best=.80,np.inf
            for ff in (.80,.90,.95,.98,1.):
                loss=0.
                for t in features:
                    k=seq.ffrls(t["Xs"],ff)
                    kf=np.clip(1.+k[:,0],seq.KF_LO,seq.KF_HI)
                    ks=np.clip(1.+k[:,1],seq.KS_LO,seq.KS_HI)
                    y=shrink.pred_dv(kf,ks,{"I":t["Il"],"NOM":t["NOMl"]})
                    loss+=float(np.sum((y-t["Yl"])**2))
                if loss<best:
                    best_ff,best=ff,loss
            k=np.concatenate([seq.ffrls(t["Xs"],best_ff) for t in via])
            save_old("ffrls",1.+k[:,0],1.+k[:,1],True)
    else:
        keys=("X","Y","NOM","I")
        if not all(x.exists() for x in params("a3")):
            tr={k:np.concatenate([source[t][k] for t in sorted(source)]) for k in keys}
            te={k:np.concatenate([t[2][k] for t in tests]) for k in keys}
            all_seeds=[train_graph(trim.TrimLinear,tr,te,300,.003,seed,"cuda") for seed in (0,1,2)]
            save_old("a3",np.mean([m[1] for m in all_seeds],0),np.mean([m[2] for m in all_seeds],0),False)
        for tag,cls in (("lstm",seq.SeqLSTM),("gru",seq.SeqGRU)):
            if all(x.exists() for x in params(tag)):
                continue
            tr={k:np.concatenate([t[k] for t in features]) for k in ("Xs","Yl","NOMl","Il")}
            te={k:np.concatenate([t[k] for t in via]) for k in ("Xs","Yl","NOMl","Il")}
            def remap(d):
                return {"X":d["Xs"],"Y":d["Yl"],"NOM":d["NOMl"],"I":d["Il"]}
            all_seeds=[train_graph(cls,remap(tr),remap(te),300,.003,seed,"cuda") for seed in (0,1,2)]
            save_old(tag,np.mean([m[1] for m in all_seeds],0),np.mean([m[2] for m in all_seeds],0),True)
    print(MODE,pair,direction,"done",flush=True)

if __name__=="__main__":
    MODE=sys.argv[1]
    pool_size=6 if MODE=="cpu" else int(os.environ.get("SOP_SECONDARY_GPU_WORKERS","4"))
    with ProcessPoolExecutor(max_workers=pool_size) as pool:
        result=[pool.submit(task,pair,d) for pair in itertools.combinations(CELLS,2) for d in DIRECTIONS]
        for job in result:
            job.result()
    for method in (("shrink","ffrls") if MODE=="cpu" else ("a3","lstm","gru")):
        assert len([p for d in DIRECTIONS for c in CELLS for p in output(c,d,method).glob("pred_A3_*.npz")])==60
