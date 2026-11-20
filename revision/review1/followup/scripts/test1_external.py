from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import csv, hashlib, os, subprocess, sys, zipfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
R=BASE/"repo"
OUT=ROOT/"external"
OUT.mkdir(exist_ok=True)
sys.path[:0]=[str(R/"analysis"),str(R/"repro"),str(BASE/"experiments")]
import rpcwby_us06_trim as extract
import run_external_a8 as eva
import strict_features as fast
from ecm_pool import surfaces
from stages import CELLS

def history_one(entry):
    cycle,path=entry
    with zipfile.ZipFile(extract.ZIPS[1]) as zf:
        with zf.open(path) as f:
            first=f.readline().decode("utf-8-sig").strip().split(",")
        tc=extract.cell_temp_col(first)
        with zf.open(path) as f:
            df=pd.read_csv(f,usecols=extract.BASE_COLS+tc,encoding="utf-8-sig",low_memory=False)
    t,I,V=(df[k].to_numpy(float) for k in extract.BASE_COLS)
    cand={k:df[k].to_numpy(float) for k in tc}
    tname=max(cand,key=lambda c:np.nanstd(np.diff(cand[c])))
    T=cand[tname]
    keep=np.isfinite(t+I+V+T) & (t>=np.nanmax(t)-extract.WINDOW_H*3600)
    t,I,V,T=(a[keep] for a in (t,I,V,T))
    soc,anchors=extract.soc_track(t,I,V)
    soh=extract.soh_of(cycle,"RPC_CC")
    sd,sc=surfaces("CC")
    snapshots=fast.calc(sd,sc,t,I,V,T,np.clip(soc,.02,1.),np.full(len(t),soh))
    drive=snapshots[snapshots[:,5]>=extract.DUTY_MIN]
    print("Test1",cycle,"earlier phase",Path(path).name,"blocks",len(drive),flush=True)
    return cycle,drive[-12:,:12],path

def raw_features():
    with zipfile.ZipFile(extract.ZIPS[1]) as z:
        inputs=extract.pairs(z)
    with ProcessPoolExecutor(max_workers=6) as p:
        items=list(p.map(history_one,sorted(inputs.items())))
    keep=[(c,x,path) for c,x,path in items if len(x)==12]
    assert len(keep)>=8
    np.savez(OUT/"test1_raw_features.npz", cycles=np.array([c for c,x,path in keep]),
             X=np.stack([x for c,x,path in keep]),source=np.array([path for c,x,path in keep]))

def score(surface):
    sys.argv=[eva.__file__,"--surface",surface,"--out",str(OUT/f"Test1_{surface}.csv")]
    src=Path(eva.__file__).read_text()
    edits=[
        ("if r['sheet'] == 'Test#2'","if r['sheet'] == 'Test#1'"),
        ("r.get('cell') == 'RPC_US06'", "r.get('cell') == 'RPC_CC'"),
        ("FEATS = os.path.join(ANALYSIS, 'cache', 'us06', 't2_feats_CC.npz')", f"FEATS = {str(OUT/'test1_raw_features.npz')!r}"),
    ]
    for before,after in edits:
        assert src.count(before)==1
        src=src.replace(before,after)
    src=src.replace("RPCWBY Test#2   ","RPCWBY Test#1   ")
    exec(compile(src,eva.__file__,"exec"),{"__name__":"__main__","__file__":eva.__file__})

if __name__=="__main__":
    if sys.argv[1]=="features":
        raw_features()
    else:
        score(sys.argv[1])
