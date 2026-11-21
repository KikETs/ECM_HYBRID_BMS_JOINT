"""E12 (R2-3/4): cycle-level performance of RC-SOP on the primary arm.

For every outer cell and condition, the LOCO-calibrated lambda_c (zero tolerance) is applied to the outer rows, the
rows are grouped by characterization cycle, and the per-cycle median of lambda_c*I_model/I_ref and the per-cycle
exceedance count are summarized. Same rows and lambda as Table 3 / Supplementary Table S6.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import load_arms, score_condition, write_csv, out_dir   # noqa: E402

OUT = out_dir('E12')


def cyc_of(r):
    return int(r['cycle']) if 'cycle' in r else int(str(r['key']).split('|')[1])


if __name__ == '__main__':
    arms = load_arms()
    table, cyc_rows = [], []
    for d in ('discharge', 'charge'):
        for tau in (10.0, 2.0):
            for model in ('hyb', 'ecm'):
                recs = score_condition(arms, d, tau, model=model, tol=0.0)
                allmed = []
                for rec in recs:
                    by = {}
                    for ratio, r in zip(rec['ratios'], rec['rows']):
                        by.setdefault(cyc_of(r), []).append(ratio)
                    med = np.array([np.median(v) * 100 for _, v in sorted(by.items())])
                    exc = np.array([int(np.sum(np.array(v) > 1.0 + 1e-12)) for _, v in sorted(by.items())])
                    for (cyc, v), m, e in zip(sorted(by.items()), med, exc):
                        cyc_rows.append(dict(direction=d, tau_s=tau, model=model, cell=rec['cell'], cycle=cyc, n_rows=len(v),
                                             usable_med_pct=float(m), exceed=int(e), lam=rec['lam']))
                    allmed += list(med)
                    table.append(dict(direction=d, tau_s=tau, model=model, cell=rec['cell'], lam=rec['lam'], n_rows=rec['n_outer'],
                                      n_cycles=len(med), cyc_med_median=float(np.median(med)), cyc_med_p10=float(np.percentile(med, 10)),
                                      cyc_med_p90=float(np.percentile(med, 90)), cyc_med_min=float(med.min()),
                                      cyc_med_max=float(med.max()), cycles_with_exceed=int((exc > 0).sum()),
                                      rows_exceed=int(exc.sum())))
                if allmed:
                    a = np.array(allmed)
                    table.append(dict(direction=d, tau_s=tau, model=model, cell='ALL', n_rows=sum(r['n_outer'] for r in recs),
                                      n_cycles=len(a), cyc_med_median=float(np.median(a)), cyc_med_p10=float(np.percentile(a, 10)),
                                      cyc_med_p90=float(np.percentile(a, 90)), cyc_med_min=float(a.min()), cyc_med_max=float(a.max()),
                                      cycles_with_exceed=int(sum(t['cycles_with_exceed'] for t in table[-len(recs):])),
                                      rows_exceed=int(sum(t['rows_exceed'] for t in table[-len(recs):]))))
    write_csv(os.path.join(OUT, 'E12_cycle_summary.csv'), table)
    write_csv(os.path.join(OUT, 'E12_cycle_rows.csv'), cyc_rows)
    for t in table:
        if t['model'] == 'hyb':
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()})
