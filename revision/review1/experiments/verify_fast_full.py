from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from build_strict_inner import REPO, CELLS, ds, surfaces
from train_strict_inner import EXP
from strict_features import build_features

BASE=REPO/"nested"/"assets"/"innerpool"
def compare(case):
    c,j=case
    p=BASE/f"no{c}"
    sd,sc=surfaces(j,outdir=str(p/"pool"),ecm_csv=str(p/"uypydj_ecm.csv"),ocv_csv=str(p/"uypydj_ocv.csv"))
    f=build_features(j,sd,sc,ds.CACHE_T)
    result={}
    for direction,sub in (("discharge","trim"),("charge","trim_chg")):
        labels,_=ds.build_labels(j,sc if direction=="charge" else sd,direction=direction)
        X,Y,NOM,I,m,XE=ds.to_arrays(ds.pair(labels,f))
        reference=EXP/f"{sub}_no{c}"/f"trim_{j}.npz"
        with np.load(reference) as a:
            for k,v in {"X":X,"Y":Y,"NOM":NOM,"I":I,"XE":XE}.items():
                np.testing.assert_allclose(v,a[k],rtol=1e-6,atol=2e-5)
            np.testing.assert_equal(m["cycle"],a["m_cycle"])
        result[direction]=len(X)
    print(case,result,flush=True)
    return result

if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=3) as p:
        r=list(p.map(compare,[("BOOST","CC"),("CC","BOOST_NEGPULSE_1S"),("BOOST_REST","CC_CELL2")]))
    assert all(min(q.values())>0 for q in r)
    print("Three complete cached cell folds validated in both current directions.")
