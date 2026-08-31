"""External validation of the FROZEN A8 trim, on RPCWBY Test#2.

    python3 repro/run_external_a8.py

Why Test#2 and nothing else
    Every SOP number in this project so far is leave-one-cell-out inside
    UYPYDJ: a different cell, but the same laboratory, cycler and protocol.
    RPCWBY Test#2 is the only sheet in either archive that carries a drive
    cycle (US06) and SOP measurements on the SAME cell, which is what a trim
    indexed on preceding drive history needs.  Test#3 has the temperature
    axis but no paired drive, so it can only score the physics model - that
    is repro/run_chen2026_baseline.py.

What is frozen, and it is everything
    The A8 weights are the checkpoints already in analysis/runs_trim_a8/,
    trained on UYPYDJ only.  Nothing is refitted here.  lambda is the value
    calibrated on UYPYDJ and carried across unchanged.  No hyperparameter,
    feature, aggregation or tolerance is chosen after seeing an RPCWBY
    number.  All six leave-one-cell-out folds are run: RPCWBY is external to
    every one of them, so the spread across folds is a legitimate statement
    of how much the answer depends on which fold happened to ship.

Metric
    RPCWBY measures SOP as a constant-POWER pulse, so the comparison runs in
    watts through the constant-power search of Chen et al. 2026, with the
    ECM scaled by the trim's branch multipliers (k_f on R0 and R1, k_s on
    R2).  Test#2's own documented limits are used: 2.55-4.15 V, 30 A / -15 A.
    A0 is the same pipeline with k = 1, so the two differ only by the trim.

    History aggregation is `max` over the 12 drive blocks, the deployed
    convention: the largest k is the largest resistance, hence the lowest
    and most conservative SOP.
"""
import argparse
import collections
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(ROOT, 'analysis')
sys.path.insert(0, ANALYSIS)
sys.path.insert(0, HERE)
from run_chen2026_baseline import (OutOfHull, simulate_power,  # noqa: E402
                                   tcell_map, DT, Q_NOM)

FEATS = os.path.join(ANALYSIS, 'cache', 'us06', 't2_feats_CC.npz')
SOP = os.path.join(ANALYSIS, 'rpcwby_sop_summary.csv')
RES = os.path.join(ANALYSIS, 'rpcwby_resistance.csv')
OUT = os.path.join(ANALYSIS, 'results', 'tables', 'external_a8.csv')

# Test#2 protocol, from the RPCWBY readme:
#   Samsung 30T | 2.55-4.15 V | 30 A / -15 A | 10, 25 C | 10 s | 1C-rate
V_MIN, V_MAX = 2.55, 4.15
I_DIS_MAX, I_CHG_MAX = 30.0, 15.0
TAU = 10.0
SOP_MAX_W, SOP_MIN_W = 200.0, 0.0
V_TOL, I_TOL = 5e-3, 10e-3
MAX_ITER = 50

KF_SPAN, KS_SPAN = 0.470, 0.588


def load_a8_folds(run_dir=os.path.join(ANALYSIS, 'runs_trim_a8')):
    """Frozen A8: (W, b, mu, sd) per leave-one-cell-out fold, seeds folded."""
    import torch
    out = {}
    for f in sorted(glob.glob(os.path.join(run_dir, 'model_A8_*.pt'))):
        cell = os.path.basename(f)[len('model_A8_'):-3]
        ck = torch.load(f, map_location='cpu', weights_only=False)
        W, B, MU, SD = [], [], [], []
        for st in ck['seeds']:
            sd_ = st['model']
            w = [v for k, v in sd_.items() if k.endswith('weight')][0].numpy()
            b = [v for k, v in sd_.items() if k.endswith('bias')][0].numpy()
            W.append(w)
            B.append(b)
            MU.append(st['mu'])
            SD.append(st['sd'])
        out[cell] = (np.mean(W, 0), np.mean(B, 0), np.mean(MU, 0),
                     np.mean(SD, 0))
    return out


def trim_k(fold, X):
    """(k_f, k_s) for one characterisation: apply to 12 blocks, take max."""
    W, B, MU, SD = fold
    z = np.clip((np.asarray(X, float) - MU) / SD, -4.0, 4.0)
    u = z @ W.T + B
    kf = np.exp(KF_SPAN * np.tanh(u[:, 0]))
    ks = np.exp(KS_SPAN * np.tanh(u[:, 1]))
    return float(np.max(kf)), float(np.max(ks))


def search(surf, soc0, soh, T, kf, ks, ocv_fn, charge):
    """Constant-power bisection under Test#2's limits."""
    sign = +1.0 if charge else -1.0
    i_lim = I_CHG_MAX if charge else I_DIS_MAX
    lo, hi = SOP_MIN_W, SOP_MAX_W
    best = np.nan
    for _ in range(MAX_ITER):
        P = 0.5 * (lo + hi)
        try:
            V, I = simulate_power(surf, soc0, soh, P, T, TAU, ocv_fn,
                                  kf=kf, ks=ks, sign=sign,
                                  v_stop=0.05 if not charge else 0.05)
        except OutOfHull:
            return np.nan, 'out-of-hull'
        if not np.isfinite(V):
            hi = P
            continue
        v_slack = (V_MAX - V) if charge else (V - V_MIN)
        i_slack = i_lim - abs(I)
        if v_slack < 0 or i_slack < 0:
            hi = P
        else:
            best = P
            lo = P
        if abs(v_slack) <= V_TOL or abs(i_slack) <= I_TOL:
            return P, ('voltage' if abs(v_slack) <= V_TOL else 'current')
        if hi - lo < 1e-4:
            break
    return best, ('voltage' if np.isfinite(best) else 'none')


def soh_by_cycle():
    d = {}
    for r in csv.DictReader(open(RES, encoding='utf-8')):
        if r.get('cell') == 'RPC_US06' and r.get('SOH'):
            d[int(float(r['cycle']))] = float(r['SOH'])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--surface', default='CC',
                    help='which pooled ECM surface.  RPCWBY is external to '
                         'all six, so any is a holdout.')
    a = ap.parse_args()

    if not os.path.exists(FEATS):
        print(f'  missing {FEATS}\n'
              f'  build it: python3 analysis/rpcwby_us06_trim.py --test 2',
              file=sys.stderr)
        return 1

    from ecm_surface import ECMSurface
    zf = np.load(FEATS)
    fcyc = zf['cycles'].astype(int)
    X = zf['X']                                   # (n_charac, 12, 12)
    folds = load_a8_folds()
    SOHC = soh_by_cycle()
    TC = tcell_map()

    rows = [r for r in csv.DictReader(open(SOP, encoding='utf-8'))
            if r['sheet'] == 'Test#2'
            and int(r['cycle']) in set(fcyc.tolist())]
    print(f'  RPCWBY Test#2   {len(rows)} SOP rows   '
          f'{len(fcyc)} characterisations   {len(folds)} frozen A8 folds')
    print(f'  limits {V_MIN}-{V_MAX} V, {I_DIS_MAX} A / -{I_CHG_MAX} A, '
          f'tau = {TAU:g} s   (Test#2 protocol)')

    surfs = {}
    out, agg = [], collections.defaultdict(list)
    for fold_name in sorted(folds):
        fold = folds[fold_name]
        K = {int(c): trim_k(fold, X[i]) for i, c in enumerate(fcyc)}
        for charge in (False, True):
            key = 'SOP_char' if charge else 'SOP_disch'
            sname = ('charge' if charge else 'discharge')
            if sname not in surfs:
                surfs[sname] = ECMSurface(a.surface, sname)
            surf = surfs[sname]
            for r in rows:
                if r[key] in ('', 'nan'):
                    continue
                cyc = int(r['cycle'])
                soh = SOHC.get(cyc)
                if soh is None:
                    continue
                soc = float(r['SOC'])
                Tset = int(float(r['temp_C']))
                T = TC.get(Tset, float(Tset))
                meas = abs(float(r[key]))
                if meas <= 0.0:
                    continue

                def ocv_fn(s, _surf=surf, _soh=soh):
                    v, _ = _surf.ocv(s, _soh)
                    m, _ = _surf.hyst_M(s, _soh)
                    return (float(np.atleast_1d(v)[0]),
                            float(np.atleast_1d(m)[0]))

                kf, ks = K[cyc]
                p8, lim8 = search(surf, soc, soh, T, kf, ks, ocv_fn, charge)
                p0, lim0 = search(surf, soc, soh, T, 1.0, 1.0, ocv_fn, charge)
                out.append([fold_name, sname, cyc, f'{soh:.4f}', Tset,
                            f'{soc:.3f}', f'{meas:.2f}',
                            '' if not np.isfinite(p0) else f'{p0:.2f}',
                            '' if not np.isfinite(p8) else f'{p8:.2f}',
                            f'{kf:.4f}', f'{ks:.4f}', lim8])
                if np.isfinite(p0) and np.isfinite(p8):
                    agg[(fold_name, sname)].append((meas, p0, p8))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['fold', 'direction', 'cycle', 'SOH', 'temp_C', 'SOC',
                    'SOP_meas_W', 'SOP_A0_W', 'SOP_A8_W', 'k_f', 'k_s',
                    'limit_hit'])
        w.writerows(out)
    print(f'  -> {os.path.relpath(a.out, ROOT)}  ({len(out)} rows)')

    print(f"\n  Frozen A8 vs physics-only, RMSE and bias in watts")
    print(f"  {'fold':<20}{'dir':<10}{'n':>5}{'A0 RMSE':>9}{'A8 RMSE':>9}"
          f"{'A0 bias':>9}{'A8 bias':>9}{'A0 opt%':>9}{'A8 opt%':>9}")
    print('  ' + '-' * 89)
    summ = collections.defaultdict(list)
    for (fold_name, sname), v in sorted(agg.items()):
        m = np.array([x[0] for x in v])
        p0 = np.array([x[1] for x in v])
        p8 = np.array([x[2] for x in v])
        r0, r8 = (np.sqrt(np.mean((p0 - m) ** 2)),
                  np.sqrt(np.mean((p8 - m) ** 2)))
        o0, o8 = np.mean(p0 > m) * 100, np.mean(p8 > m) * 100
        print(f'  {fold_name:<20}{sname:<10}{len(v):>5}{r0:>9.2f}{r8:>9.2f}'
              f'{np.mean(p0 - m):>+9.2f}{np.mean(p8 - m):>+9.2f}'
              f'{o0:>9.1f}{o8:>9.1f}')
        summ[sname].append((r0, r8, o0, o8))
    print('  ' + '-' * 89)
    for sname, v in sorted(summ.items()):
        arr = np.array(v)
        print(f'  {"across folds":<20}{sname:<10}{"":>5}'
              f'{arr[:, 0].mean():>9.2f}{arr[:, 1].mean():>9.2f}'
              f'{"":>9}{"":>9}{arr[:, 2].mean():>9.1f}{arr[:, 3].mean():>9.1f}'
              f'   fold spread A8 RMSE '
              f'{arr[:, 1].min():.2f}-{arr[:, 1].max():.2f}')
    nohull = sum(1 for x in out if x[11] == 'out-of-hull')
    print(f'\n  {nohull} of {len(out)} model calls were out of the pooled '
          f'surface hull and are excluded.')
    print('  Nothing here was refitted: A8 weights and lambda are the UYPYDJ '
          'values, carried across unchanged.')

    coverage(out, a.out)
    safety_transfer(out, a.out)
    return 0


def coverage(out, base):
    """Which operating points the pooled hull actually covers.

    A transfer result computed only where the model agrees to answer is not a
    transfer result for the dataset -- it is one for the subset the source
    surfaces happen to span.  So the excluded fraction has to be reported with
    the axis it falls on, not as a single count.
    """
    import collections as _c
    rows = []
    for axis, col, fmt in (('SOC', 5, lambda v: f'{float(v):.2f}'),
                           ('temp_C', 4, lambda v: str(v)),
                           ('SOH', 3, lambda v: f'{float(v):.2f}'),
                           ('direction', 1, str)):
        tot, out_ = _c.Counter(), _c.Counter()
        for r in out:
            k = fmt(r[col])
            tot[k] += 1
            if r[11] == 'out-of-hull':
                out_[k] += 1
        for k in sorted(tot):
            rows.append([axis, k, tot[k], tot[k] - out_[k],
                         f'{100 * (tot[k] - out_[k]) / tot[k]:.1f}'])
    path = base.replace('.csv', '_coverage.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['axis', 'value', 'n_calls', 'n_in_hull', 'in_hull_pct'])
        w.writerows(rows)
    print(f'\n  IN-HULL COVERAGE  -> {os.path.relpath(path, ROOT)}')
    print(f"  {'axis':<12}{'value':<12}{'calls':>7}{'in hull':>9}{'%':>8}")
    print('  ' + '-' * 48)
    for r in rows:
        print(f'  {r[0]:<12}{r[1]:<12}{r[2]:>7}{r[3]:>9}{r[4]:>8}')


LAM_UYPYDJ = {'discharge': 0.6832, 'charge': 0.5860}


def safety_transfer(out, base):
    """Does the frozen safety factor stay safe off the training dataset?

    lambda is a ratio, so it carries from current to power without a unit
    argument: the question is whether scaling the prediction by the UYPYDJ
    lambda still lands under the measurement on cells and a chemistry the
    factor was never fitted on.  lambda_needed is the largest factor that
    would leave zero exceedance here; margin is how much room the frozen
    value has.  A margin below 1 means the frozen factor is not conservative
    enough on this data and the transfer claim fails on safety, whatever the
    RMSE says.
    """
    import collections as _c
    per = _c.defaultdict(list)
    for r in out:
        if r[11] == 'out-of-hull' or r[8] == '':
            continue
        per[(r[0], r[1])].append((float(r[6]), float(r[8])))
    rows = []
    print(f'\n  SAFETY TRANSFER OF THE FROZEN LAMBDA')
    print(f"  {'fold':<20}{'dir':<10}{'n':>5}{'lambda':>9}{'needed':>9}"
          f"{'margin':>8}{'exceed':>8}{'usable %':>10}{'worst W':>9}")
    print('  ' + '-' * 88)
    for (fold, direction), v in sorted(per.items()):
        m = np.array([x[0] for x in v])
        p = np.array([x[1] for x in v])
        lam = LAM_UYPYDJ[direction]
        ok = p > 0
        needed = float(np.min(m[ok] / p[ok])) if ok.any() else float('nan')
        over = lam * p - m
        ex = int((over > 0).sum())
        rows.append([fold, direction, len(v), f'{lam:.4f}',
                     f'{needed:.4f}', f'{needed / lam:.3f}', ex,
                     f'{np.median(lam * p / m) * 100:.2f}',
                     f'{max(over.max(), 0.0):.3f}'])
        print(f'  {fold:<20}{direction:<10}{len(v):>5}{lam:>9.4f}'
              f'{needed:>9.4f}{needed / lam:>8.3f}{ex:>8}'
              f'{np.median(lam * p / m) * 100:>10.2f}'
              f'{max(over.max(), 0.0):>9.3f}')
    path = base.replace('.csv', '_safety.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['fold', 'direction', 'n_in_hull', 'lambda_uypydj',
                    'lambda_needed', 'margin', 'exceed', 'usable_pct',
                    'worst_overshoot_W'])
        w.writerows(rows)
    print(f'  -> {os.path.relpath(path, ROOT)}')


if __name__ == '__main__':
    sys.exit(main())
