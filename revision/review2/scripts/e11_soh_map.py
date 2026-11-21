"""E11 (R2-8): conversion between the record SOH target and the capacity-anchor SOH axis.

The SOH estimator is trained on the record SOH target (soh_pred.npz `{cell}_y`), whereas the ECM surfaces and the
SOP labels are indexed by the capacity-anchor SOH (label column `SOH`). This script
  1. pairs both definitions at the same characterization cycles and reports the gap;
  2. fits a linear map anchor = a + b * record leaving out every cell involved in a fold (the outer cell, and for
     inner folds also the inner cell) and checks it on the left-out cell;
  3. re-solves the primary arm (estimated SOH, reference SOC) with the mapped SOH, and with the anchor SOH itself
     (oracle), and scores them with the submitted lambda and with lambda re-fitted on the re-solved inner rows.
The variant `identity` re-solves with the unmodified estimate and must reproduce the baseline rows.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK as R2, REPO as REPO_ROOT, R1 as REV1
import os, sys, csv, pickle, multiprocessing as mp
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

OUT = os.path.join(R2, 'out', 'E11'); os.makedirs(OUT, exist_ok=True)
VARIANTS = ('identity', 'mapped', 'anchor_oracle')


def anchor_by_cycle():
    """capacity-anchor SOH per (cell, cycle) from the discharge and charge label files"""
    out = {}
    for lab in (EA.EV.LABEL, EA.EV.LABEL_CHG):
        for r in csv.DictReader(open(lab, encoding='utf-8')):
            out.setdefault((r['cell'], int(r['cycle'])), set()).add(round(float(r['SOH']), 6))
    bad = {k: v for k, v in out.items() if len(v) > 1}
    assert not bad, list(bad.items())[:3]
    return {k: next(iter(v)) for k, v in out.items()}


ANCHOR = anchor_by_cycle()
SOHP = dict(np.load(os.path.join(EA.A, 'results', 'soh_pred.npz')))


MAXLAG = 10   # SOH records are on drive runs; each characterization is paired with the latest preceding run


def matched(c, key='_y'):
    """[(characterization cycle, record-or-estimate value at the latest preceding drive run, anchor)]"""
    cc, vv = SOHP[f'{c}_cycle'], SOHP[f'{c}{key}']
    out = []
    for (cell, cyc), an in sorted(ANCHOR.items()):
        if cell != c:
            continue
        prior = (cc <= cyc) & (cc >= cyc - MAXLAG)
        if prior.any():
            i = np.flatnonzero(prior)[int(np.argmax(cc[prior]))]
            if np.isfinite(vv[i]):
                out.append((cyc, float(vv[i]), an))
    return out


def pairs(cells):
    """(record target, anchor) at matched cycles"""
    m = [(v, an) for c in cells for _, v, an in matched(c)]
    return np.array([x for x, _ in m]), np.array([y for _, y in m])


def fit_map(exclude):
    x, y = pairs([c for c in CELLS if c not in exclude])
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def rows_variant(level, outer, cell, direction, variant):
    charge = direction == 'charge'
    lab = EA.EV.LABEL_CHG if charge else EA.EV.LABEL
    rows = [r for r in csv.DictReader(open(lab, encoding='utf-8')) if r['cell'] == cell]
    if level == 'outer':
        sd, sc = EA.ecm_pool.surfaces(cell)
        tdir = os.path.join(EA.A, EA.OUTER_TRIM[direction])
        soh_map = SOHP
        a, b = fit_map({cell})
    else:
        pd_ = os.path.join(EA.INNERPOOL, f'no{outer}')
        sd, sc = EA.ecm_pool.surfaces(cell, outdir=os.path.join(pd_, 'pool'), ecm_csv=os.path.join(pd_, 'uypydj_ecm.csv'),
                                      ocv_csv=os.path.join(pd_, 'uypydj_ocv.csv'))
        tdir = os.path.join(EA.INNERTRIM, f'{EA.SUB[direction]}_no{outer}')
        soh_map = dict(np.load(os.path.join(EA.INNERSOH, f'soh_pred_no{outer}.npz')))
        a, b = fit_map({cell, outer})
    surf = sc if charge else sd
    K = EA.EV.trim_k_agg(cell, tdir, EA.AGG)
    out = []
    for r in rows:
        cyc = int(r['cycle']); tau = float(r['tau_s'])
        if r.get('I_star_lin4_A') in (None, '', 'nan') or cyc not in K:
            continue
        meas = float(r['I_star_lin4_A'])
        cc, pp = soh_map[f'{cell}_cycle'], soh_map[f'{cell}_pred']
        prior = cc <= cyc
        soh_est = float(pp[prior][int(np.argmax(cc[prior]))]) if prior.any() else float(pp[int(np.argmin(cc))])
        soh_anchor = float(r['SOH'])
        soh = {'identity': soh_est, 'mapped': a + b * soh_est, 'anchor_oracle': soh_anchor}[variant]
        soc = float(r['SOC'])
        v_pre, v_min = float(r['V_pre_V']), float(r['V_min_V'])
        kf, ks = K[cyc]
        i0, _, _ = EA.hull_solver.solve(surf, soc, soh, v_pre, v_min, tau, 1.0, 1.0, charge, EA.EV.r_eff)
        i3, _, _ = EA.hull_solver.solve(surf, soc, soh, v_pre, v_min, tau, kf, ks, charge, EA.EV.r_eff)
        rec = dict(cell=cell, cycle=cyc, tau=tau, extrap=float(r['extrap']), meas=abs(meas), soh_est=soh_est, soh_in=soh,
                   soh_anchor=soh_anchor, key=f"{cell}|{cyc}|{tau:.1f}|{r['V_pre_V']}|{round(meas, 3)}", status='ok')
        if not (np.isfinite(i0) and np.isfinite(i3)):
            rec['status'] = 'hull'
        else:
            rec.update(hyb=abs(i3), ecm=abs(i0))
        out.append(rec)
    return (variant, level, outer, cell, direction), out


def gap_tables():
    rows, loco = [], []
    for c in CELLS:
        x, y = pairs([c])
        a, b = fit_map({c})
        rows.append(dict(cell=c, n_cycles=len(x), record_min=x.min(), record_max=x.max(), anchor_min=y.min(), anchor_max=y.max(),
                         gap_median=float(np.median(y - x)), gap_absmax=float(np.max(np.abs(y - x))),
                         map_a=a, map_b=b, mapped_resid_median=float(np.median(y - (a + b * x))),
                         mapped_resid_absmax=float(np.max(np.abs(y - (a + b * x))))))
        # estimate-side error before and after mapping (estimate at the characterization cycle itself)
        for cyc, p, an in matched(c, '_pred'):
            loco.append(dict(cell=c, cycle=int(cyc), anchor=an, est=p, est_mapped=a + b * p))
    x, y = pairs(CELLS); b, a = np.polyfit(x, y, 1)
    rows.append(dict(cell='ALL', n_cycles=len(x), record_min=x.min(), record_max=x.max(), anchor_min=y.min(), anchor_max=y.max(),
                     gap_median=float(np.median(y - x)), gap_absmax=float(np.max(np.abs(y - x))), map_a=float(a), map_b=float(b),
                     mapped_resid_median=float(np.median(y - (a + b * x))), mapped_resid_absmax=float(np.max(np.abs(y - (a + b * x))))))
    e0 = np.array([r['est'] - r['anchor'] for r in loco]); e1 = np.array([r['est_mapped'] - r['anchor'] for r in loco])
    rows.append(dict(cell='EST_vs_ANCHOR', n_cycles=len(e0), gap_median=float(np.median(e0)), gap_absmax=float(np.max(np.abs(e0))),
                     mapped_resid_median=float(np.median(e1)), mapped_resid_absmax=float(np.max(np.abs(e1)))))
    write_csv(os.path.join(OUT, 'E11_soh_gap.csv'), rows)
    write_csv(os.path.join(OUT, 'E11_soh_pairs.csv'), loco)
    return rows


if __name__ == '__main__':
    for r in gap_tables():
        print(r)
    cache = os.path.join(OUT, 'rows.pkl')
    if os.path.exists(cache):
        res = pickle.load(open(cache, 'rb'))
    else:
        tasks = [(lvl, ('NONE' if lvl == 'outer' else c), j, d, v) for v in VARIANTS
                 for d in ('discharge', 'charge') for c in CELLS for (lvl, j) in [('outer', c)] + [('inner', j) for j in CELLS if j != c]]
        with mp.Pool(16) as pool:
            res = dict(pool.starmap(rows_variant, tasks))
        pickle.dump(res, open(cache, 'wb'))
    base = load_arms()
    TOL = {'discharge': 0.0, 'charge': 0.5}
    # reproduction check for the identity variant
    mism = 0; nchk = 0
    for d in ('discharge', 'charge'):
        for c in CELLS:
            bo = {r['key']: r for r in base['outer', 'NONE', c, d, 'eo', 'replace'] if r['status'] == 'ok'}
            for r in res[('identity', 'outer', 'NONE', c, d)]:
                if r['status'] == 'ok' and r['key'] in bo:
                    nchk += 1; mism += int(abs(r['hyb'] - bo[r['key']]['hyb']) > 1e-6 or abs(r['ecm'] - bo[r['key']]['ecm']) > 1e-6)
    print('identity reproduction: checked', nchk, 'mismatch', mism)
    summ = []
    for v in VARIANTS:
        for d in ('discharge', 'charge'):
            for tau in (10.0, 2.0):
                keep = keep_factory(tau)
                for model in ('hyb', 'ecm'):
                    pcs_fixed, pcs_re, exf, exr, nrow, lam1s = [], [], 0, 0, 0, []
                    for c in CELLS:
                        bi = [r for j in CELLS if j != c for r in base['inner', c, j, d, 'eo', 'replace'] if keep(r)]
                        if len(bi) < 25:
                            continue
                        lam0 = fit_lambda(np.array([r[model] for r in bi]), np.array([r['meas'] for r in bi]), TOL[d])
                        vo = [r for r in res[(v, 'outer', 'NONE', c, d)] if keep(r)]
                        vi = [r for j in CELLS if j != c for r in res[(v, 'inner', c, j, d)] if keep(r)]
                        if not vo or len(vi) < 25:
                            continue
                        lam1 = fit_lambda(np.array([r[model] for r in vi]), np.array([r['meas'] for r in vi]), TOL[d]); lam1s.append(lam1)
                        p = np.array([r[model] for r in vo]); y = np.array([r['meas'] for r in vo])
                        exf += int((lam0 * p - y > TOL[d]).sum()); exr += int((lam1 * p - y > TOL[d]).sum()); nrow += len(p)
                        pcs_fixed.append((float(np.median(lam0 * p / y) * 100), len(p)))
                        pcs_re.append((float(np.median(lam1 * p / y) * 100), len(p)))
                    if not pcs_fixed:
                        continue
                    w = [x[1] for x in pcs_fixed]
                    lo, hi = cell_boot([x[0] for x in pcs_re], w)
                    summ.append(dict(variant=v, direction=d, tau_s=tau, model=model, n_rows=nrow,
                                     exceed_fixed_lambda=exf, usable_fixed_lambda=float(np.average([x[0] for x in pcs_fixed], weights=w)),
                                     exceed_recalibrated=exr, usable_recalibrated=float(np.average([x[0] for x in pcs_re], weights=w)),
                                     recal_boot_lo=float(lo), recal_boot_hi=float(hi), lam_re_min=min(lam1s), lam_re_max=max(lam1s)))
    write_csv(os.path.join(OUT, 'E11_summary.csv'), summ)
    for s in summ:
        print(s)
