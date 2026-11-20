"""E9 (minor 2, 3, 7)
2. A8 is one-dimensional: inputs 1..11 are zeroed before fitting/inference, so k_f and k_s depend on dR_fast only.
   Check: stored model weights, and that k_f, k_s are monotone functions of each other within a fold.
3. cyclewise max: are the k_f and k_s maxima taken from the same observation?
7. EPS = 1e-9 A^2 in dR = e_ir / (e_ii + EPS): size relative to e_ii (exc = log10(e_ii + EPS) is stored per row).
"""
import os, sys, glob
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import *

OUT = out_dir('E9')
A = os.path.join(REPO, 'analysis')
rows_w, rows_agg, rows_eps = [], [], []
for run in ('runs_trim_a8', 'runs_trim_a8_chg'):
    for c in CELLS:
        mp = os.path.join(A, run, f'model_A8_{c}.pt')
        sd = torch.load(mp, map_location='cpu', weights_only=False)
        st = sd.get('state_dict', sd.get('model', sd)) if isinstance(sd, dict) else sd
        W = None
        if isinstance(st, dict):
            for k, v in st.items():
                if hasattr(v, 'shape') and k.endswith('weight'):
                    W = v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v)
                    if W.ndim == 3: W = W.mean(0)
        meta = {k: sd[k] for k in ('ablate',) if isinstance(sd, dict) and k in sd}
        z = np.load(os.path.join(A, run, f'pred_A8_{c}.npz'), allow_pickle=True)
        kf, ks, cy, exc = z['k_f'].astype(float), z['k_s'].astype(float), z['cycle'].astype(int), z['exc'].astype(float)
        # monotone relation k_s = g(k_f) within the fold: Spearman of log k
        from scipy.stats import spearmanr
        rho = spearmanr(np.log(kf), np.log(ks)).correlation
        rows_w.append(dict(run=run, cell=c, keys=','.join(list(sd.keys())[:8]) if isinstance(sd, dict) else type(sd).__name__,
                           ablate=str(meta.get('ablate')), W_shape=str(None if W is None else W.shape),
                           W_col0=str(None if W is None else np.round(W[:, 0], 4).tolist()),
                           W_other_absmax=None if W is None else float(np.abs(W[:, 1:]).max()),
                           spearman_logkf_logks=float(rho)))
        same, n, dk = 0, 0, []
        for k in np.unique(cy):
            m = np.flatnonzero(cy == k)
            i_f = m[np.argmax(kf[m])]; i_s = m[np.argmax(ks[m])]
            n += 1; same += int(i_f == i_s or (kf[i_s] == kf[i_f] and ks[i_f] == ks[i_s]))
            dk.append(ks[i_s] / ks[i_f] - 1)       # how much larger the separate ks max is than ks at the kf-argmax row
        rows_agg.append(dict(run=run, cell=c, cycles=n, same_row=same, same_pct=100 * same / n,
                             ks_sep_vs_joint_max_pct=100 * float(np.max(dk)), ks_sep_vs_joint_median_pct=100 * float(np.median(dk))))
        e_ii = 10 ** exc - 1e-9
        rows_eps.append(dict(run=run, cell=c, n=len(exc), e_ii_min_A2=float(e_ii.min()), e_ii_p01_A2=float(np.quantile(e_ii, .01)),
                             eps_over_eii_max=float(1e-9 / e_ii.min())))
write_csv(os.path.join(OUT, 'E9_a8_weights.csv'), rows_w)
write_csv(os.path.join(OUT, 'E9_cyclewise_max.csv'), rows_agg)
write_csv(os.path.join(OUT, 'E9_eps.csv'), rows_eps)
print('E9 done')
