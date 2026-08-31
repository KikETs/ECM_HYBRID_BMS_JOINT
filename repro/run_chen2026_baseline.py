"""Chen et al. 2026 constant-power binary-search SOP, driven by this project's ECM.

    python3 repro/run_chen2026_baseline.py

Reference
    J. Chen, Q. Yao, P. Kollmeyer, C. Vidal, M. Naguib, S. Panchal, A. Emadi,
    "Battery state-of-power estimation: A machine learning battery model with
    numerical searching approach", J. Energy Storage 175 (2026) 123023.
    CC BY-NC 4.0.  Dataset: doi 10.5683/SP3/RPCWBY (CC BY 4.0) - the same
    RPCWBY archive this project already holds.

What is implemented, and it is their framework not their model
    Their contribution has two separable halves.  The battery model is an
    LSTM (about 1.05 M parameters, 4.1 MB) mapping a sequence of
    {SOC, T, power} to voltage.  The SOP estimator is a model-agnostic
    binary search on constant power, which the paper states explicitly can
    wrap "LSTM, ECM, or EM".  This file implements the search exactly as
    published - Fig. 1(b), eqs (3)-(4), tolerance and iteration cap from
    Table 3 - and drives it with this project's pooled 2RC ECM.

    Their search, per iteration:
        SOP_hat = (SOP_max + SOP_min) / 2
        for k in 1..L at dt = 1 s:
            V_hat_k = model(SOC_k, T, SOP_hat)
            I_hat_k = SOP_hat / V_hat_k                       (3)
            SOC_k+1 = SOC_k - I_hat_k * dt / (Q_nom * 3600)   (4)
        accept if the terminal V or I sits within tolerance of its limit,
        otherwise move the bracket.

One deviation, and it is forced
    Their LSTM consumes power directly, so V_hat follows from one forward
    pass.  An ECM is written in current, so V and I are coupled at constant
    power and eq (3) becomes a fixed point.  It is solved here by the same
    relaxation the paper uses between iterations, to 1e-6 V.  Nothing else
    differs.

What this can and cannot settle
    It CAN answer: does the published search framework, wrapped around this
    project's physics model, reach a different answer than this project's
    own constant-current inversion, on the authors' own external data, in
    the authors' own units.

    It CANNOT be read as a controlled head-to-head against their Table 7.
    Their Table 3 sets V_min = 3.2 V, chosen so that voltage-limited cases
    appear; the Test#3 measurements were taken at 2.55-4.1 V per the dataset
    readme, and Table 5's test split ("10 s SOP measurement with UDDS
    discharge profile") does not identify a single sheet in the archive.
    Their numbers are therefore quoted as published, under their limits, and
    are indicative only.  Running here under Test#3's documented limits is
    the defensible choice.
"""
import argparse
import collections
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(ROOT, 'analysis')
sys.path.insert(0, ANALYSIS)

LAB = os.path.join(ANALYSIS, 'rpcwby_sop_test3.csv')
TPULSE = os.path.join(ANALYSIS, 'rpcwby_temp_pulses.csv')
OUT = os.path.join(ANALYSIS, 'results', 'tables', 'chen2026_baseline.csv')

# Test#3 protocol, from the RPCWBY readme:
#   Samsung 30T | 2.55-4.1 V | 30 A / -5 A | -20..40 C | 2, 10, 30 s | 0.33C
V_MIN, V_MAX = 2.55, 4.1
I_DIS_MAX, I_CHG_MAX = 30.0, 5.0
Q_NOM = 3.0

# Chen Table 3
SOP_MAX_W, SOP_MIN_W = 200.0, 0.0
V_TOL, I_TOL = 5e-3, 10e-3
MAX_ITER = 50
DT = 1.0

# sec 22: the per-temperature capacities in the sheet are low-temperature
# capacity loss, not aging (the test order is not monotone in temperature),
# so one SOH is used for every block.
SOH_25C = 2.6 / 3.0

# Chen Table 7, as published.  Quoted, not reproduced - see the docstring.
CHEN_TABLE7 = {
    'LSTM': {-20: 1.13, -10: 2.10, 0: 1.89, 10: 1.88, 25: 2.33, 40: 1.53,
             'overall': 1.87},
    'LSTM without SOP measurement':
        {-20: 8.48, -10: 16.50, 0: 2.57, 10: 3.06, 25: 3.90, 40: 6.47,
         'overall': 8.02},
    'ECM (theirs)': {-20: 2.16, -10: 4.54, 0: 7.46, 10: 5.65, 25: 7.88,
                     40: 6.75, 'overall': 6.17},
}


def tcell_map():
    d = collections.defaultdict(list)
    for r in csv.DictReader(open(TPULSE, encoding='utf-8')):
        d[int(float(r['temp_set_C']))].append(float(r['T_cell_C']))
    return {k: float(np.median(v)) for k, v in d.items()}


class OutOfHull(Exception):
    """The pooled surface has no parameters at this (SOC, SOH, I, T).

    Kept distinct from a limit violation on purpose.  Folding the two
    together makes the bisection read "no model" as "too much power" and
    walk the bracket down to nothing, which showed up as a 10.8 W estimate
    against a 107.4 W measurement at SOC = 1.000 - the hull edge.
    """


def params(surf, soc, soh, cur, T, kf=1.0, ks=1.0):
    """2RC parameters, optionally scaled by the trim's branch multipliers.

    kf scales the fast branch (R0, R1) and ks the slow branch (R2), which is
    the same split eval_sop_test3.r_eff uses:
        kf * (R0 + R1 * (1 - exp(-tau/t1))) + ks * R2 * (1 - exp(-tau/t2))
    """
    th = surf.theta(soc, soh, cur, T)
    if not bool(np.atleast_1d(th['in_hull'])[0]):
        raise OutOfHull()
    R0, R1, t1, R2, t2 = (float(th[k][0])
                          for k in ('R0', 'R1', 'tau1', 'R2', 'tau2'))
    return kf * R0, kf * R1, t1, ks * R2, t2


def simulate_power(surf, soc0, soh, P, T, L, ocv_fn, kf=1.0, ks=1.0,
                   sign=-1.0, v_stop=0.05):
    """Run L seconds at constant discharge power P > 0.  Returns (V_L, I_L).

    The RC states are propagated, not recomputed from scratch each step.
    An earlier version evaluated R_eff(tau = 1 s) independently at every
    step, so ten steps produced the one-second voltage instead of the
    ten-second one; polarisation never accumulated and the search returned
    far too much power.

        v_j(k+1) = v_j(k) * a_j + I * R_j * (1 - a_j),  a_j = exp(-dt/tau_j)
        V        = OCV(SOC) - M + I * R0 + v1 + v2

    Chen eq (3), I = P / V, is a fixed point once the model is written in
    current, so it is relaxed within each step.
    """
    soc, v1, v2 = soc0, 0.0, 0.0
    V = I = np.nan
    for _ in range(int(L)):
        ocv, m_half = ocv_fn(soc)
        if not np.isfinite(ocv):
            raise OutOfHull()
        I = sign * P / max(ocv - m_half, 1e-3)
        R0 = R1 = t1 = R2 = t2 = np.nan
        for _ in range(60):                       # fixed point on (3)
            R0, R1, t1, R2, t2 = params(surf, soc, soh, I, T, kf, ks)
            a1, a2 = np.exp(-DT / t1), np.exp(-DT / t2)
            nv1 = v1 * a1 + I * R1 * (1 - a1)
            nv2 = v2 * a2 + I * R2 * (1 - a2)
            V = ocv - m_half + I * R0 + nv1 + nv2
            if not np.isfinite(V) or V <= v_stop:
                return np.nan, np.nan             # genuine voltage collapse
            I_new = sign * P / V
            if abs(I_new - I) < 1e-6:
                I = I_new
                break
            I = 0.5 * I + 0.5 * I_new
        a1, a2 = np.exp(-DT / t1), np.exp(-DT / t2)
        v1 = v1 * a1 + I * R1 * (1 - a1)
        v2 = v2 * a2 + I * R2 * (1 - a2)
        soc = soc + I * DT / 3600.0 / Q_NOM       # eq (4), I < 0 discharges
        if soc <= 0.0 or soc >= 1.5:
            return np.nan, np.nan
    return V, I


def chen_search(surf, soc0, soh, T, L, ocv_fn):
    """Chen Fig. 1(b): bisect on constant power until a limit is met."""
    lo, hi = SOP_MIN_W, SOP_MAX_W
    best = np.nan
    for _ in range(MAX_ITER):
        P = 0.5 * (lo + hi)
        try:
            V, I = simulate_power(surf, soc0, soh, P, T, L, ocv_fn)
        except OutOfHull:
            # The model cannot speak here at all.  Report no estimate rather
            # than pretend the limit was hit.
            return np.nan, 'out-of-hull'
        if not np.isfinite(V):
            hi = P                                  # voltage collapsed
            continue
        v_slack = V - V_MIN
        i_slack = I_DIS_MAX - abs(I)
        if v_slack < 0 or i_slack < 0:
            hi = P
        else:
            best = P
            lo = P
        if abs(v_slack) <= V_TOL or abs(i_slack) <= I_TOL:
            return P, ('voltage' if abs(v_slack) <= V_TOL else 'current')
        if hi - lo < 1e-4:
            break
    lim = 'voltage' if np.isfinite(best) else 'none'
    return best, lim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--holdout', default='CC',
                    help='which pooled ECM surface to use.  RPCWBY is an '
                         'external cell, so no UYPYDJ cell is "held out" '
                         'here; every surface is equally external.')
    ap.add_argument('--tau', type=float, default=10.0)
    ap.add_argument('--soh', type=float, default=SOH_25C)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()

    from ecm_surface import ECMSurface
    surf = ECMSurface(a.holdout, 'discharge')
    TC = tcell_map()

    rows = [r for r in csv.DictReader(open(LAB, encoding='utf-8'))
            if r['SOP_disch'] not in ('', 'nan')
            and abs(float(r['tau_s']) - a.tau) < 0.05]
    print(f'  Test#3, tau = {a.tau:g} s   {len(rows)} measured rows   '
          f'SOH {a.soh:.3f}   surface {a.holdout}')
    print(f'  limits: {V_MIN}-{V_MAX} V, {I_DIS_MAX} A / -{I_CHG_MAX} A '
          f'(Test#3 protocol, not Chen Table 3)')

    def ocv_fn(soc):
        v, _ = surf.ocv(soc, a.soh)
        m, _ = surf.hyst_M(soc, a.soh)
        return float(np.atleast_1d(v)[0]), float(np.atleast_1d(m)[0])

    out = []
    per_T = collections.defaultdict(list)
    for r in rows:
        soc = float(r['SOC'])
        Tset = int(float(r['temp_C']))
        T = TC.get(Tset, float(Tset))
        meas = abs(float(r['SOP_disch']))
        pred, lim = chen_search(surf, soc, a.soh, T, a.tau, ocv_fn)
        err = (pred - meas) if np.isfinite(pred) else np.nan
        out.append([Tset, f'{T:.1f}', f'{soc:.3f}', f'{meas:.2f}',
                    '' if not np.isfinite(pred) else f'{pred:.2f}',
                    '' if not np.isfinite(err) else f'{err:+.2f}', lim])
        if np.isfinite(err):
            per_T[Tset].append(err)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['temp_set_C', 'temp_cell_C', 'SOC', 'SOP_meas_W',
                    'SOP_pred_W', 'error_W', 'limit_hit'])
        w.writerows(out)
    print(f'  -> {os.path.relpath(a.out, ROOT)}  ({len(out)} rows)')

    temps = sorted(per_T)
    allerr = np.concatenate([per_T[t] for t in temps]) if temps else np.array([])
    print("\n  RMSE of SOP, in watts - the same metric as Chen Table 7")
    print(f"  {'method':<30}" + ''.join(f'{t:>8}' for t in temps)
          + f"{'overall':>10}{'n':>6}")
    print('  ' + '-' * (30 + 8 * len(temps) + 16))
    ours = ''.join(
        f'{np.sqrt(np.mean(np.square(per_T[t]))):>8.2f}' for t in temps)
    n_ok = int(sum(len(per_T[t]) for t in temps))
    print(f"  {'this ECM + Chen search':<30}{ours}"
          f"{np.sqrt(np.mean(allerr ** 2)):>10.2f}{n_ok:>6}")
    for name, d in CHEN_TABLE7.items():
        line = ''.join(f"{d.get(t, float('nan')):>8.2f}" for t in temps)
        print(f'  {name + " [as published]":<30}{line}'
              f'{d["overall"]:>10.2f}{"-":>6}')
    miss = len(rows) - n_ok
    if miss:
        print(f'\n  {miss} of {len(rows)} rows returned no estimate '
              f'(outside the pooled surface hull) and are excluded from the '
              f'RMSE - Chen report all rows, so our column is not directly '
              f'comparable where this is non-zero.')
    hits = collections.Counter(x[6] for x in out)
    print(f'  limit hit: {dict(hits)}')
    print('\n  Chen rows are QUOTED from the paper under their Table 3 '
          'limits (V_min 3.2 V).\n  They are not a controlled comparison - '
          'see the module docstring.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
