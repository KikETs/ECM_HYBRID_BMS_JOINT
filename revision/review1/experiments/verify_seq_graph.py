import importlib, sys, time
import numpy as np
from train_strict_inner import EXPERIMENT, EXP, REPO
import sop_trim as trim
sys.path.insert(0,str(REPO/"repro"))
import run_sop_seq_baselines as seq
from strict_comparators import mb
from train_graph import train_graph

mode = sys.argv[1]
pair=("BOOST","BOOST_NEGPULSE")
ds=EXPERIMENT/"datasets"/"__".join(pair)/"trim_chg"
t=seq.load_cells(str(ds))
src=[mb(t[c],seq) for c in sorted(t)]
test=mb(seq.load_cells(str(EXP/"trim_chg_noBOOST"))["BOOST_NEGPULSE"],seq)
tr={k:np.concatenate([x[k] for x in src]) for k in ("Xs","Yl","NOMl","Il")}
cls=seq.SeqGRU if mode=="gru" else seq.SeqLSTM
begin=time.monotonic()
want=seq.train_seq(cls,tr,test,300,.003,0,"cuda")
old=time.monotonic()
def remap(d):
    return {"X":d["Xs"],"Y":d["Yl"],"NOM":d["NOMl"],"I":d["Il"]}
_,kf,ks,*_=train_graph(cls,remap(tr),remap(test),300,.003,0,"cuda")
elapsed=time.monotonic()-old
print(mode,"eager",old-begin,"graph",elapsed,flush=True)
for x,y in zip(want[:2],(kf,ks)):
    print("max difference", np.max(abs(x-y)),flush=True)
    np.testing.assert_allclose(x,y,atol=1e-5,rtol=2e-5)
print("verified",mode,"graph",flush=True)
