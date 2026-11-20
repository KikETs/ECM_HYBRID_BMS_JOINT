"""E8 (R2-8, minor 5): the rated-capacity SOC axis.

The pipeline's SOC is the rated fraction measured from the empty (cut-off) point, SOC_rated = SOC_aged * s with
s = CAP / 3.0 Ah (analysis/data30t.py). A BMS that tracks SOC against its own (aged) capacity must convert before
calling the model. Two runtime cases on the primary arm (estimated SOH), lambda kept from the submitted calibration
and also recalibrated:
  aged_unconverted : SOC_in = min(SOC_rated / s, 1)      (aged-capacity SOC fed without conversion)
  converted_estSOH : SOC_in = SOC_rated * s / s_hat, s_hat = SOH_est * CAP0 / 3.0   (conversion with estimated SOH)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK as R2, REPO as REPO_ROOT, R1 as REV1
import os, sys, csv, pickle, multiprocessing as mp
from collections import defaultdict
import numpy as np

os.environ.update(RCSOP_REPO=REPO_ROOT, RCSOP_BUNDLE=os.path.join(REPO_ROOT, 'nested'), SOLVER='hull',
                  SOCTAG='_fixpool', OUTDIR=os.path.join(R2, 'baseline', 'arms'),
                  INNERPOOL=os.path.join(REPO_ROOT, 'nested', 'assets', 'innerpool'),
                  INNERTRIM=os.path.join(REPO_ROOT, 'nested', 'assets', 'innertrim'),
                  INNERSOH=os.path.join(REPO_ROOT, 'nested', 'assets', 'innersoh'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'nested', 'scripts', 'nested'))
import eval_arms_nested as EA      # noqa: E402
sys.path.insert(0, os.path.join(R2, 'scripts'))
from r2common import CELLS, fit_lambda, write_csv, cell_boot, load_arms, keep_factory   # noqa: E402

OUT = os.path.join(R2, 'out', 'E8'); os.makedirs(OUT, exist_ok=True)
CAP = {}
for r in csv.DictReader(open(os.path.join(EA.A, 'uypydj_hppc_resistance.csv'), encoding='utf-8')):
    if r['CAP_Ah']:
        CAP[r['protocol'], int(r['cycle'])] = float(r['CAP_Ah'])
CAP0 = {c: CAP[min((k for k in CAP if k[0] == c), key=lambda k: k[1])] for c in CELLS}


def rows_variant(level, outer, cell, direction, variant):
    charge = direction == 'charge'
    lab = EA.EV.LABEL_CHG if charge else EA.EV.LABEL
    rows = [r for r in csv.DictReader(open(lab, encoding='utf-8')) if r['cell'] == cell]
    if level == 'outer':
        sd, sc = EA.ecm_pool.surfaces(cell)
        tdir = os.path.join(EA.A, EA.OUTER_TRIM[direction])
        soh_map = dict(np.load(os.path.join(EA.A, 'results', 'soh_pred.npz')))
    else:
        pd_ = os.path.join(EA.INNERPOOL, f'no{outer}')
        sd, sc = EA.ecm_pool.surfaces(cell, outdir=os.path.join(pd_, 'pool'), ecm_csv=os.path.join(pd_, 'uypydj_ecm.csv'),
                                      ocv_csv=os.path.join(pd_, 'uypydj_ocv.csv'))
        tdir = os.path.join(EA.INNERTRIM, f'{EA.SUB[direction]}_no{outer}')
        soh_map = dict(np.load(os.path.join(EA.INNERSOH, f'soh_pred_no{outer}.npz')))
    surf = sc if charge else sd
    K = EA.EV.trim_k_agg(cell, tdir, EA.AGG)
    out = []
    caps = sorted((k[1], v) for k, v in CAP.items() if k[0] == cell)
    for r in rows:
        cyc = int(r['cycle']); tau = float(r['tau_s'])
        if r.get('I_star_lin4_A') in (None, '', 'nan') or cyc not in K:
            continue
        meas = float(r['I_star_lin4_A'])
        cc, pp = soh_map[f'{cell}_cycle'], soh_map[f'{cell}_pred']
        prior = cc <= cyc
        soh = float(pp[prior][int(np.argmax(cc[prior]))]) if prior.any() else float(pp[int(np.argmin(cc))])
        cap = min(caps, key=lambda x: abs(x[0] - cyc))[1]
        s = cap / 3.0; s_hat = soh * CAP0[cell] / 3.0
        soc_ref = float(r['SOC'])
        soc = min(soc_ref / s, 1.0) if variant == 'aged_unconverted' else soc_ref * s / s_hat
        soc = float(np.clip(soc, 0.02, 1.0))
        v_pre, v_min = float(r['V_pre_V']), float(r['V_min_V'])
        kf, ks = K[cyc]
        i0, _, _ = EA.hull_solver.solve(surf, soc, soh, v_pre, v_min, tau, 1.0, 1.0, charge, EA.EV.r_eff)
        i3, _, _ = EA.hull_solver.solve(surf, soc, soh, v_pre, v_min, tau, kf, ks, charge, EA.EV.r_eff)
        rec = dict(cell=cell, cycle=cyc, tau=tau, extrap=float(r['extrap']), meas=abs(meas), soc_ref=soc_ref, soc_in=soc, s=s,
                   key=f"{cell}|{cyc}|{tau:.1f}|{r['V_pre_V']}|{round(meas, 3)}", status='ok')
        if not (np.isfinite(i0) and np.isfinite(i3)):
            rec['status'] = 'hull'
        else:
            rec.update(hyb=abs(i3), ecm=abs(i0))
        out.append(rec)
    return (variant, level, outer, cell, direction), out


if __name__ == '__main__':
    cache = os.path.join(OUT, 'rows.pkl')
    if os.path.exists(cache):
        res = pickle.load(open(cache, 'rb'))
    else:
        tasks = [(lvl, ('NONE' if lvl == 'outer' else c), j, d, v) for v in ('aged_unconverted', 'converted_estSOH')
                 for d in ('discharge', 'charge') for c in CELLS for (lvl, j) in [('outer', c)] + [('inner', j) for j in CELLS if j != c]]
        with mp.Pool(16) as pool:
            res = dict(pool.starmap(rows_variant, tasks))
        pickle.dump(res, open(cache, 'wb'))
    base = load_arms()
    TOL = {'discharge': 0.0, 'charge': 0.5}
    summ = []
    for v in ('aged_unconverted', 'converted_estSOH'):
        for d in ('discharge', 'charge'):
            for tau in (10.0, 2.0):
                keep = keep_factory(tau)
                pcs_fixed, pcs_re, exf, exr, nrow, shift = [], [], 0, 0, 0, []
                for c in CELLS:
                    bi = [r for j in CELLS if j != c for r in base['inner', c, j, d, 'eo', 'replace'] if keep(r)]
                    if len(bi) < 25:
                        continue
                    lam0 = fit_lambda(np.array([r['hyb'] for r in bi]), np.array([r['meas'] for r in bi]), TOL[d])
                    vo = [r for r in res[(v, 'outer', 'NONE', c, d)] if keep(r)]
                    vi = [r for j in CELLS if j != c for r in res[(v, 'inner', c, j, d)] if keep(r)]
                    if not vo or len(vi) < 25:
                        continue
                    lam1 = fit_lambda(np.array([r['hyb'] for r in vi]), np.array([r['meas'] for r in vi]), TOL[d])
                    p = np.array([r['hyb'] for r in vo]); y = np.array([r['meas'] for r in vo])
                    exf += int((lam0 * p - y > TOL[d]).sum()); exr += int((lam1 * p - y > TOL[d]).sum()); nrow += len(p)
                    pcs_fixed.append((float(np.median(lam0 * p / y) * 100), len(p))); pcs_re.append((float(np.median(lam1 * p / y) * 100), len(p)))
                    shift += [r['soc_in'] - r['soc_ref'] for r in vo]
                if not pcs_fixed:
                    continue
                w = [x[1] for x in pcs_fixed]
                summ.append(dict(variant=v, direction=d, tau_s=tau, n_rows=nrow, soc_shift_median=float(np.median(shift)),
                                 soc_shift_p95=float(np.quantile(np.abs(shift), .95)),
                                 exceed_fixed_lambda=exf, usable_fixed_lambda=float(np.average([x[0] for x in pcs_fixed], weights=w)),
                                 exceed_recalibrated=exr, usable_recalibrated=float(np.average([x[0] for x in pcs_re], weights=w))))
    write_csv(os.path.join(OUT, 'E8_summary.csv'), summ)
    for s in summ:
        print(s)
