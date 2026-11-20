from pathlib import Path
import numpy as np
from build_strict_inner import REPO, CELLS
from soh_ridge import pick_alpha, _fit, _predict

source=REPO.parent.parent/"repo"/"analysis"/"cache"/"soh_charge.npz"
with np.load(source,allow_pickle=True) as z:
    x,y,cell,cycle=(z[k] for k in ("X","y","cell","cycle"))
for c in CELLS:
    p=REPO/"nested"/"assets"/"innersoh"/f"soh_pred_no{c}.npz"
    with np.load(p) as existing:
        assert f"{c}_pred" not in existing
        for j in CELLS:
            if j==c: continue
            training=(cell!=c)&(cell!=j)
            target=cell==j
            assert set(cell[training])==set(CELLS)-{c,j}
            a,_=pick_alpha(x[training],y[training],cell[training])
            w,b,mu,sd=_fit(x[training],y[training],a)
            pred=_predict(w,b,mu,sd,x[target]).astype(np.float32)
            np.testing.assert_array_equal(cycle[target],existing[f"{j}_cycle"])
            np.testing.assert_allclose(pred,existing[f"{j}_pred"],atol=1e-7,rtol=1e-7)
print("SOH four-cell fit and inner tuning verified for all 30 (c,j) splits.")
