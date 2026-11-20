import sys
import numpy as np
from build_strict_inner import REPO, ds, surfaces
import strict_features as fast

cell="CC"
sd,sc=surfaces(cell)
a=ds.build_features(cell,sd,sc,max_runs=2)
b=fast.build_features(cell,sd,sc,ds.CACHE_T,max_runs=2)
assert len(a)==len(b)>0
for x,y in zip(a,b):
    assert x["cycle"]==y["cycle"]
    for k in ("x","xe","exc"):
        np.testing.assert_allclose(x[k],y[k],atol=2e-5,rtol=1e-6)
print("validated", len(a), "snapshots against scalar source in two full drive traces")
