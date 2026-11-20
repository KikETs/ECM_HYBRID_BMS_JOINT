"""E9-7: EPS = 1e-9 A^2 relative to e_ii, from feature column 2 = log10(e_ii + EPS) in the strict-inner trim datasets."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK as R2, REPO as REPO_ROOT, R1 as REV1
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import *
base = os.path.join(REV1, 'strict_inner', 'datasets')
rows = []
for kind in ('trim', 'trim_chg'):
    fs = sorted(glob.glob(os.path.join(base, '*', kind, '*.npz')))
    vals = []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        X = z['X'] if 'X' in z.files else None
        if X is None:
            continue
        X = X.reshape(-1, X.shape[-1])
        vals.append(X[:, 2])
    if not vals:
        print('no X in', kind, fs[:2]); continue
    v = np.concatenate(vals); v = v[np.isfinite(v)]
    e_ii = 10 ** v - 1e-9
    rows.append(dict(kind=kind, n_files=len(fs), n=len(v), log10_eii_min=float(v.min()), e_ii_min_A2=float(e_ii.min()),
                     e_ii_p001_A2=float(np.quantile(e_ii, .001)), frac_eii_lt_1e_6=float((e_ii < 1e-6).mean()),
                     eps_rel_max=float(1e-9 / max(e_ii.min(), 1e-30))))
write_csv(os.path.join(out_dir('E9'), 'E9_eps.csv'), rows)
print(rows)
