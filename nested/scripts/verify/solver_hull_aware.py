"""Reference-independent, hull-aware root search for V_pre + I R_eff(I) = V_lim.

Why: the previous bracketed check (brentq on [-400, -0.1] / [0.1, 400])
evaluates the residual at the domain ends.  in_hull depends on the rate rank
(theta() combines the hulls of the two ranks that bracket |I|), so a point
can be inside the hull at 15 A and outside at 400 A.  A 'hull' failure at
the bracket ends therefore says nothing about whether a valid root exists.

Here, per row and per model (A0 nominal, A3 = the RC-SOP multipliers):
  * scan |I| on a log grid over the domain, keep the sub-intervals where
    R_eff is finite and positive (the valid domain);
  * inside each valid sub-interval, bracket every sign change of the residual
    and solve it with brentq (no use of the reference current anywhere);
  * record the number of valid sub-intervals, the number of roots, the roots,
    the residual at the domain ends (was the old failure a bracket artefact?),
    and the shipped fixed-point answer started at the reference current, its
    termination reason, its iteration count, its voltage residual, whether it
    lies inside the valid domain, and its distance to the nearest root.

Everything else (trim, aggregation, surfaces, estimated SOH) is the shipped
primary arm: runs_trim_a8{,_chg}, --trim-agg max, theta interpolation,
results/soh_pred.npz.  Nothing in the repository is written.
"""
import csv
import multiprocessing as mp
import os
import sys

import numpy as np
from scipy.optimize import brentq

REPO = os.environ['RCSOP_REPO']
A = os.path.join(REPO, 'analysis')
HERE = os.environ.get('OUTDIR', os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, A)
import eval_sop_amps as EV                       # noqa: E402
from ecm_pool import surfaces                    # noqa: E402

AGG = 'max'
TRIM = {'discharge': 'runs_trim_a8', 'charge': 'runs_trim_a8_chg'}
NGRID = 120


def resid(surf, soc, soh, v_pre, v_lim, tau, kf, ks, I):
    R = EV.r_eff(surf, soc, soh, I, tau, kf, ks)
    if not np.isfinite(R) or R <= 0:
        return np.nan
    return v_pre + I * R - v_lim


def fixed_point(surf, soc, soh, v_pre, v_lim, tau, kf, ks, I0, charge, iters=12):
    I = I0
    step = np.nan
    for it in range(iters):
        R = EV.r_eff(surf, soc, soh, I, tau, kf, ks)
        if not np.isfinite(R) or R <= 0:
            return np.nan, it, 'badR', np.nan
        I_new = (v_lim - v_pre) / R
        if not np.isfinite(I_new):
            return np.nan, it, 'nonfinite', np.nan
        I_new = float(np.clip(I_new, 0.1, 400.0) if charge
                      else np.clip(I_new, -400.0, -0.1))
        step = abs(I_new - I)
        if step < 1e-3:
            return I_new, it + 1, 'conv', step
        I = 0.5 * I + 0.5 * I_new
    return I, iters, 'cap', step


def hull_aware(surf, soc, soh, v_pre, v_lim, tau, kf, ks, charge):
    g = np.logspace(np.log10(0.1), np.log10(400.0), NGRID)
    grid = g if charge else -g[::-1]
    f = np.array([resid(surf, soc, soh, v_pre, v_lim, tau, kf, ks, I)
                  for I in grid])
    ok = np.isfinite(f)
    # valid sub-intervals
    ivals, roots = [], []
    i = 0
    while i < NGRID:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < NGRID and ok[j + 1]:
            j += 1
        ivals.append((float(grid[i]), float(grid[j])))
        for k in range(i, j):
            if f[k] == 0.0:
                roots.append(float(grid[k]))
            elif np.sign(f[k]) != np.sign(f[k + 1]):
                fn = lambda I: resid(surf, soc, soh, v_pre, v_lim, tau, kf, ks, I)
                try:
                    roots.append(float(brentq(fn, grid[k], grid[k + 1],
                                              xtol=1e-9, rtol=1e-12, maxiter=200)))
                except Exception:
                    roots.append(float('nan'))
        i = j + 1
    return ivals, roots, (f[0], f[-1])


def run_direction(direction):
    charge = direction == 'charge'
    lab = EV.LABEL_CHG if charge else EV.LABEL
    rows = list(csv.DictReader(open(lab, encoding='utf-8')))
    cells = sorted({r['cell'] for r in rows})
    EV.SOH_EST = dict(np.load(os.path.join(A, 'results', 'soh_pred.npz')))
    out = []
    for c in cells:
        surf = surfaces(c)[1 if charge else 0]
        K = EV.trim_k_agg(c, os.path.join(A, TRIM[direction]), AGG)
        for r in rows:
            if r['cell'] != c:
                continue
            cyc = int(r['cycle'])
            if r.get('I_star_lin4_A') in (None, '', 'nan') or cyc not in K:
                continue
            soc = float(r['SOC'])
            soh = EV.soh_estimated(c, cyc, float(r['SOH']))
            v_pre, v_lim, tau = float(r['V_pre_V']), float(r['V_min_V']), float(r['tau_s'])
            meas = float(r['I_star_lin4_A'])
            kf, ks = K[cyc]
            rec = dict(direction=direction, cell=c, cycle=cyc, tau=tau,
                       extrap=float(r['extrap']), meas=meas, soc=soc, soh=soh,
                       V_pre=v_pre)
            for key, (a, b) in (('A0', (1.0, 1.0)), ('A3', (kf, ks))):
                I, it, st, step = fixed_point(surf, soc, soh, v_pre, v_lim, tau,
                                              a, b, meas, charge)
                ivals, roots, (f0, f1) = hull_aware(surf, soc, soh, v_pre, v_lim,
                                                    tau, a, b, charge)
                roots_ok = [x for x in roots if np.isfinite(x)]
                inside = (np.isfinite(I) and
                          any(lo <= I <= hi for lo, hi in ivals))
                near = (min(abs(I - x) for x in roots_ok)
                        if roots_ok and np.isfinite(I) else np.nan)
                rec.update({
                    f'{key}_fp': I, f'{key}_fp_it': it, f'{key}_fp_st': st,
                    f'{key}_fp_step': step,
                    f'{key}_fp_vres': (resid(surf, soc, soh, v_pre, v_lim, tau, a, b, I)
                                       if np.isfinite(I) else np.nan),
                    f'{key}_fp_inside': int(bool(inside)),
                    f'{key}_n_valid': len(ivals),
                    f'{key}_valid': ';'.join(f'{lo:.3g}..{hi:.3g}' for lo, hi in ivals),
                    f'{key}_n_roots': len(roots_ok),
                    f'{key}_roots': ';'.join(f'{x:.4f}' for x in roots_ok),
                    f'{key}_root': (roots_ok[0] if len(roots_ok) == 1 else np.nan),
                    f'{key}_fp_to_root': near,
                    f'{key}_f_lo_end': f0, f'{key}_f_hi_end': f1,
                })
            out.append(rec)
    return out


def main():
    with mp.Pool(2) as pool:
        res = pool.map(run_direction, ['discharge', 'charge'])
    rows = res[0] + res[1]
    with open(os.path.join(HERE, 'hull_aware_rows.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'  {len(rows)} rows -> hull_aware_rows.csv', flush=True)


if __name__ == '__main__':
    main()
