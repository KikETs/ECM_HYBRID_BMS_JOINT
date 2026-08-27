"""Pack level re-measured — section 28 redone on estimated SOH.

Section 28's conclusion (28.4) was "at the adopted lambda, exceedance is zero
for N = 1 to 192, and the risk only shows when the margin is reduced".  But
that lambda was set on true SOH (the point section 29 raised).  Estimated SOH
tightens lambda (discharge 0.679 -> 0.594), so the pack must be re-examined on
top of that.

The adopted evaluation configuration.  Section 31.1 pinned the settings down
by finding the A3 version that reproduces section 16's lambda (discharge
tolerance 0.0 A, charge tolerance 0.5 A); after 32.7 the adopted trim became
A8.  A8's lambda is discharge 0.683 / 0.470, charge 0.586 / 0.560.

Here lambda is re-set leaving out the holdout cell (separately for the oracle
and estimated versions), and that lambda is used to simulate N-cell packs and
report exceedance and usable current.
"""
import argparse
import csv

import numpy as np

from sop_safety import EXTRAP_MAX

NSIM = 4000
NS = [1, 12, 48, 96, 192]
SOH_EDGES = [0.0, 0.75, 0.85, 0.95, 2.0]
SOC_EDGES = [0.0, 0.35, 0.55, 0.75, 2.0]


def load(path):
    r = list(csv.DictReader(open(path, encoding='utf-8')))

    def g(k):
        return np.array([float(x[k]) if x[k] not in ('', 'nan') else np.nan
                         for x in r])
    return dict(meas=np.abs(g('I_meas_A')), hyb=np.abs(g('I_A3_A')),
                extrap=g('extrap'), soh=g('SOH'), soc=g('SOC'),
                tau=g('tau_s'), cell=np.array([x['cell'] for x in r]))


def keep(d, tau):
    return (np.isfinite(d['meas']) & np.isfinite(d['hyb'])
            & (d['meas'] > 0.5) & (d['extrap'] <= EXTRAP_MAX)
            & (np.round(d['tau'], 1) == tau))


def lam_loco(d, tau, tol):
    """Median of the lambdas set leaving out each cell (as in section 16)."""
    m = keep(d, tau)
    out = []
    for c in sorted(set(d['cell'][m])):
        tr = m & (d['cell'] != c)
        if tr.sum() < 25:
            continue
        lo, hi = 0.02, 1.6
        for _ in range(70):
            mid = (lo + hi) / 2
            if np.max(mid * d['hyb'][tr] - d['meas'][tr]) > tol:
                hi = mid
            else:
                lo = mid
        out.append(lo)
    return float(np.median(out)) if out else np.nan


def pack(d, tau, lam, rng, min_group=8):
    m = keep(d, tau)
    sh = np.digitize(d['soh'][m], SOH_EDGES)
    sc = np.digitize(d['soc'][m], SOC_EDGES)
    g = np.array([f'{a}|{b}' for a, b in zip(sh, sc)])
    meas, pred = d['meas'][m], d['hyb'][m]
    ks, cnt = np.unique(g, return_counts=True)
    ks = ks[cnt >= min_group]
    if len(ks) == 0:
        return None
    idx = {k: np.where(g == k)[0] for k in ks}
    w = np.array([len(idx[k]) for k in ks], float)
    w /= w.sum()
    out = {}
    for N in NS:
        gsel = rng.choice(len(ks), size=NSIM, p=w)
        pm = np.empty(NSIM)
        pp = np.empty(NSIM)
        for i, gi in enumerate(gsel):
            ii = idx[ks[gi]]
            pick = ii[rng.integers(0, len(ii), N)]
            pm[i] = meas[pick].min()
            pp[i] = pred[pick].min()
        o = lam * pp - pm
        out[N] = dict(exc=float(np.mean(o > 0) * 100),
                      worst=float(max(o.max(), 0.0)),
                      util=float(np.median(lam * pp / pm) * 100))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-dir', default='results/eval')
    ap.add_argument('--out', default='results/tables/pack.csv')
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    E = a.eval_dir
    j = lambda n: __import__('os').path.join(E, n + '.csv')
    SETS = [('discharge', 10.0, 0.0, j('a8_disc_oracle'), j('a8_disc_est')),
            ('discharge', 2.0, 0.0, j('a8_disc_oracle'), j('a8_disc_est')),
            ('charge', 10.0, 0.5, j('a8_char_oracle'), j('a8_char_est')),
            ('charge', 2.0, 0.5, j('a8_char_oracle'), j('a8_char_est'))]

    print(f"  {'direction':<11}{'tau':>5}{'SOH in':>11}{'lambda':>9}"
          f"{'cell n':>8}"
          + ''.join(f"{f'N={n} exc':>11}" for n in NS)
          + f"{'N=192 worst':>13}{'N=192 curr':>12}", flush=True)
    print('  ' + '-' * (44 + 11 * len(NS) + 25), flush=True)
    store = {}
    for nm, tau, tol, po, pe in SETS:
        for arm, path in (('oracle', po), ('estimated', pe)):
            d = load(path)
            lam = lam_loco(d, tau, tol)
            r = pack(d, tau, lam, rng)
            store[(nm, tau, arm)] = (lam, r)
            print(f"  {nm:<11}{tau:>5.0f}{arm:>11}{lam:>9.3f}"
                  f"{keep(d, tau).sum():>8}"
                  + ''.join(f"{r[n]['exc']:>10.1f}%" for n in NS)
                  + f"{r[NS[-1]]['worst']:>12.2f}A{r[NS[-1]]['util']:>11.1f}%",
                  flush=True)

    import csv as _csv, os as _os
    _os.makedirs(_os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = _csv.writer(f)
        w.writerow(['direction', 'tau_s', 'soh', 'lambda', 'N',
                    'exceed_pct', 'worst_A', 'usable_pct'])
        for (nm, tau, arm), (lam, r) in store.items():
            for N in NS:
                w.writerow([nm, tau, arm, f'{lam:.3f}', N,
                            f"{r[N]['exc']:.1f}", f"{r[N]['worst']:.2f}",
                            f"{r[N]['util']:.1f}"])
    print(f"\n  -> {a.out}", flush=True)

    print("\n  == using the oracle-SOH lambda directly on the estimated-SOH"
          " version (margin not re-set)", flush=True)
    print(f"  {'direction':<11}{'tau':>5}{'lambda used':>13}"
          + ''.join(f"{f'N={n} exc':>11}" for n in NS)
          + f"{'N=192 worst':>13}", flush=True)
    for nm, tau, tol, po, pe in SETS:
        lam_o = store[(nm, tau, 'oracle')][0]
        d = load(pe)
        r = pack(d, tau, lam_o, rng)
        print(f"  {nm:<11}{tau:>5.0f}{lam_o:>13.3f}"
              + ''.join(f"{r[n]['exc']:>10.1f}%" for n in NS)
              + f"{r[NS[-1]]['worst']:>12.2f}A", flush=True)


if __name__ == '__main__':
    main()
