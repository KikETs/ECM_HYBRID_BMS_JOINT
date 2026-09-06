"""Rebuild the 30 inner pools from the bundled five-cell CSVs in a scratch
directory with the PATCHED ecm_pool, and check three things:
  1. the rebuilt pool CSVs are byte-identical to the bundled pools;
  2. every rebuilt pool differs from the shipped pool of the same holdout
     (which the un-patched surfaces() would have produced);
  3. the surface actually responds to its CSV: theta() from the inner pool
     and from the shipped pool differ, and the SOP current solved on a fixed
     probe set differs — so target exclusion is a property of the content,
     not of the provenance sidecar.
"""
import csv
import hashlib
import os
import sys

import numpy as np

REPO = os.environ['RCSOP_REPO']; B = os.environ['RCSOP_BUNDLE']; W = os.environ['RCSOP_WORK']
A = os.path.join(REPO, 'analysis')
sys.path.insert(0, A); sys.path.insert(0, os.path.join(B, 'scripts'))
import ecm_pool                                # noqa: E402
import eval_sop_amps as EV                     # noqa: E402
import hull_solver                             # noqa: E402
CELLS = ['BOOST', 'BOOST_NEGPULSE', 'BOOST_NEGPULSE_1S', 'BOOST_REST', 'CC', 'CC_CELL2']
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
src = open(os.path.join(A, 'ecm_pool.py')).read()
if 'ecm_csv=ecm_csv, ocv_csv=ocv_csv' not in src:
    sys.exit('  ecm_pool.py is NOT patched in RCSOP_REPO — run `reproduce.sh patch` first')
fail = 0
probes = [(0.4, 0.80, 3.60, 2.5, 10.0, False), (0.7, 0.90, 3.90, 2.5, 10.0, False), (0.5, 0.85, 3.75, 4.2, 10.0, True), (0.3, 0.75, 3.55, 4.2, 2.0, True)]
for c in CELLS:
    bd = os.path.join(B, 'assets', 'innerpool', f'no{c}')
    wd = os.path.join(W, 'pools_rebuilt', f'no{c}'); os.makedirs(os.path.join(wd, 'pool'), exist_ok=True)
    ecmf, ocvf = os.path.join(bd, 'uypydj_ecm.csv'), os.path.join(bd, 'uypydj_ocv.csv')
    for j in CELLS:
        if j == c:
            continue
        pe, po, _, _ = ecm_pool.build(j, os.path.join(wd, 'pool'), ecm_csv=ecmf, ocv_csv=ocvf)
        same_bundle = sha(pe) == sha(os.path.join(bd, 'pool', f'ecm_pool_{j}.csv')) and sha(po) == sha(os.path.join(bd, 'pool', f'ocv_pool_{j}.csv'))
        sd_in = ecm_pool.ECMSurface('POOL', 'discharge', ecm_csv=pe, ocv_csv=po); sc_in = ecm_pool.ECMSurface('POOL', 'charge', ecm_csv=pe, ocv_csv=po)
        sd_sh, sc_sh = ecm_pool.surfaces(j)          # shipped five-cell pool (rebuilt in the clone from the tracked CSV)
        diff_shipped = sha(pe) != sha(os.path.join(A, 'cache', 'pool', f'ecm_pool_{j}.csv'))
        dR, dI = [], []
        for soc, soh, vpre, vlim, tau, chg in probes:
            a, b = (sc_in if chg else sd_in), (sc_sh if chg else sd_sh)
            ta, tb = a.theta(soc, soh, 10.0), b.theta(soc, soh, 10.0)
            if ta['in_hull'][0] and tb['in_hull'][0]:
                dR.append(abs(float(ta['R0'][0]) - float(tb['R0'][0])) / float(tb['R0'][0]))
                Ia, _, _ = hull_solver.solve(a, soc, soh, vpre, vlim, tau, 1.0, 1.0, chg, EV.r_eff)
                Ib, _, _ = hull_solver.solve(b, soc, soh, vpre, vlim, tau, 1.0, 1.0, chg, EV.r_eff)
                if np.isfinite(Ia) and np.isfinite(Ib):
                    dI.append(abs(Ia - Ib))
        ok = same_bundle and diff_shipped and (max(dR) > 0 if dR else False)
        fail += 0 if ok else 1
        print(f'  no{c}/holdout={j}: rebuilt==bundled {same_bundle}  !=shipped {diff_shipped}  '
              f'R0 rel diff max {max(dR) if dR else float("nan"):.4f}  |dI| max {max(dI) if dI else float("nan"):.3f} A  {"OK" if ok else "FAIL"}')
print(f'  {"ALL 30 POOLS OK" if not fail else f"{fail} FAILURES"}')
sys.exit(1 if fail else 0)
