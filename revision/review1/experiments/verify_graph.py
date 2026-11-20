import time
import numpy as np
from train_strict_inner import EXPERIMENT, EXP
import sop_trim as trim
from train_graph import train_graph

pair=("BOOST","BOOST_NEGPULSE")
data=EXPERIMENT/"datasets"/"__".join(pair)/"trim_chg"
tc=trim.load_cells(str(data))
tr={k:np.concatenate([tc[t][k] for t in sorted(tc)]) for k in ("X","Y","NOM","I")}
te=trim.load_cells(str(EXP/"trim_chg_noBOOST"))["BOOST_NEGPULSE"]
a=time.monotonic()
expected=trim.train_fold(trim.TrimLinear,tr,te,300,.003,0,"cuda",ablate=list(range(1,12)))
b=time.monotonic()
got=train_graph(trim.TrimLinear,tr,te,300,.003,0,"cuda",ablate=list(range(1,12)))
c=time.monotonic()
print("eager seconds",b-a,"graph seconds",c-b)
for x,y in zip(expected[:3],got[:3]):
    print("max difference", np.max(abs(x-y)))
    np.testing.assert_allclose(x,y,atol=1e-5,rtol=2e-5)
print("verified graph",flush=True)
