"""E12 add-on (R2-7): the retained-current difference in amperes on the Table 3 rows (zero tolerance)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import load_arms, score_condition, write_csv, out_dir   # noqa: E402

if __name__ == '__main__':
    arms = load_arms(); out = []
    for d in ('discharge', 'charge'):
        for tau in (10.0, 2.0):
            h = {r['cell']: r for r in score_condition(arms, d, tau, model='hyb', tol=0.0)}
            e = {r['cell']: r for r in score_condition(arms, d, tau, model='ecm', tol=0.0)}
            ref, gain, ah, ae = [], [], [], []
            for c in h:
                y = np.array([r['meas'] for r in h[c]['rows']])
                ph = h[c]['lam'] * np.array([r['hyb'] for r in h[c]['rows']])
                pe = e[c]['lam'] * np.array([r['ecm'] for r in e[c]['rows']])
                assert [r['key'] for r in h[c]['rows']] == [r['key'] for r in e[c]['rows']]
                ref += list(y); gain += list(ph - pe); ah += list(ph); ae += list(pe)
            ref, gain = np.array(ref), np.array(gain)
            out.append(dict(direction=d, tau_s=tau, n_rows=len(ref), ref_median_A=float(np.median(ref)),
                            ref_p10_A=float(np.percentile(ref, 10)), ref_p90_A=float(np.percentile(ref, 90)),
                            rcsop_median_A=float(np.median(ah)), nominal_median_A=float(np.median(ae)),
                            gain_median_A=float(np.median(gain)), gain_p10_A=float(np.percentile(gain, 10)),
                            gain_p90_A=float(np.percentile(gain, 90)), rows_gain_positive=int((gain > 0).sum())))
    write_csv(os.path.join(out_dir('E12'), 'E12_amps.csv'), out)
    for r in out:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
