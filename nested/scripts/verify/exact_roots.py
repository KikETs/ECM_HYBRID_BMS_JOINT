"""Independent root enumeration for V_pre + I R_eff(I) = V_lim that follows
the interpolation structure of ECMSurface.theta instead of a global grid.

Structure (ecm_surface.py):
  * theta() interpolates the five RC parameters linearly in |I| between the
    rank currents rank_I[0..3] and CLAMPS outside that range; in_hull is the
    AND of the two bracketing ranks' (SOC, SOH) hulls, so it is piecewise
    constant in |I| with breakpoints exactly at rank_I.
  * eval_sop_amps.r_eff returns nan when in_hull is False or R <= 0.
Therefore the valid domain is a union of whole segments
  [0.1, rI0], [rI0, rI1], [rI1, rI2], [rI2, rI3], [rI3, 400]
(minus any segment where R_eff <= 0 somewhere), and inside a segment
f(I) = V_pre + I R_eff(I) - V_lim is smooth (R_eff is affine in |I| in
R0,R1,R2 and enters tau1,tau2 through exp, so it is not exactly quadratic).

Per segment: 400 uniform samples, every sign change bracketed and solved by
brentq, and every local minimum of |f| below 2 mV WITHOUT a sign change
reported as a candidate tangent root (not counted, flagged).  The hull flag
is read directly from theta() at the segment ends and at the samples, so a
'valid' verdict never rests on R being finite alone.  Rows where the two
segment-end hull flags disagree (should not happen) are flagged.

Compared row by row against 50_solver/hull_aware_rows.csv (the 120-point
log-grid search) for both models on the shipped outer assets, and against
the level-2 inner/outer arm rows for the primary arm.
"""
import csv
import multiprocessing as mp
import os
import sys

import numpy as np
from scipy.optimize import brentq

V = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ['RCSOP_REPO']
A = os.path.join(REPO, 'analysis')
FLW = os.environ['RCSOP_WORK']
sys.path.insert(0, A); sys.path.insert(0, FLW)
import eval_sop_amps as EV                       # noqa: E402
from ecm_pool import surfaces                    # noqa: E402
import hull_solver                               # noqa: E402

NSEG = 150
TANGENT_MV = 2e-3


def f_and_flags(surf, soc, soh, v_pre, v_lim, tau, kf, ks, I):
    th = surf.theta(soc, soh, I)
    inh = bool(th['in_hull'][0])
    R = EV.r_eff(surf, soc, soh, I, tau, kf, ks)
    ok = inh and np.isfinite(R) and R > 0
    return (v_pre + I * R - v_lim) if ok else np.nan, inh, (np.isfinite(R) and R > 0)


def enumerate_roots(surf, soc, soh, v_pre, v_lim, tau, kf, ks, charge):
    rI = list(surf.rank_I)
    edges = [0.1] + [float(x) for x in rI if 0.1 < x < 400.0] + [400.0]
    sgn = 1.0 if charge else -1.0
    segs, roots, tangents, notes = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        xs = np.linspace(lo + 1e-9, hi - 1e-9, NSEG)
        vals, hulls, rpos = [], [], []
        for x in xs:
            fv, inh, rp = f_and_flags(surf, soc, soh, v_pre, v_lim, tau, kf, ks, sgn * x)
            vals.append(fv); hulls.append(inh); rpos.append(rp)
        vals = np.array(vals); hulls = np.array(hulls); rpos = np.array(rpos)
        if hulls[0] != hulls[-1]:
            notes.append(f'hull_flag_changes_inside_segment[{lo:.3g},{hi:.3g}]')
        valid = hulls & rpos
        if not valid.any():
            segs.append((lo, hi, 'outside'))
            continue
        if not valid.all():
            notes.append(f'partial_validity[{lo:.3g},{hi:.3g}]:{int(valid.sum())}/{NSEG}')
        segs.append((lo, hi, 'valid' if valid.all() else 'partial'))
        fn = lambda I: f_and_flags(surf, soc, soh, v_pre, v_lim, tau, kf, ks, I)[0]
        for k in range(NSEG - 1):
            if not (valid[k] and valid[k + 1]):
                continue
            a, b = vals[k], vals[k + 1]
            if a == 0.0:
                roots.append(float(sgn * xs[k]))
            elif np.sign(a) != np.sign(b):
                try:
                    roots.append(float(brentq(fn, sgn * xs[k], sgn * xs[k + 1], xtol=1e-10, maxiter=200)))
                except Exception as e:
                    notes.append(f'brentq_fail[{xs[k]:.4g}]')
        av = np.abs(vals)
        for k in range(1, NSEG - 1):
            if valid[k - 1] and valid[k] and valid[k + 1] and av[k] < av[k - 1] and av[k] <= av[k + 1] \
                    and av[k] < TANGENT_MV and np.sign(vals[k - 1]) == np.sign(vals[k + 1]):
                tangents.append(float(sgn * xs[k]))
    return segs, roots, tangents, notes


def run_direction(task):
    direction, only = task
    charge = direction == 'charge'
    lab = EV.LABEL_CHG if charge else EV.LABEL
    rows = list(csv.DictReader(open(lab, encoding='utf-8')))
    EV.SOH_EST = dict(np.load(os.path.join(A, 'results', 'soh_pred.npz')))
    TRIM = {'discharge': 'runs_trim_a8', 'charge': 'runs_trim_a8_chg'}
    out = []
    for c in [only]:
        surf = surfaces(c)[1 if charge else 0]
        K = EV.trim_k_agg(c, os.path.join(A, TRIM[direction]), 'max')
        for r in rows:
            if r['cell'] != c:
                continue
            cyc = int(r['cycle'])
            if r.get('I_star_lin4_A') in (None, '', 'nan') or cyc not in K:
                continue
            soc = float(r['SOC']); soh = EV.soh_estimated(c, cyc, float(r['SOH']))
            v_pre, v_lim, tau = float(r['V_pre_V']), float(r['V_min_V']), float(r['tau_s'])
            meas = float(r['I_star_lin4_A']); kf, ks = K[cyc]
            rec = dict(direction=direction, cell=c, cycle=cyc, tau=tau, extrap=float(r['extrap']), meas=meas)
            for key, (a_, b_) in (('A0', (1.0, 1.0)), ('A3', (kf, ks))):
                segs, roots, tang, notes = enumerate_roots(surf, soc, soh, v_pre, v_lim, tau, a_, b_, charge)
                Ih, nh, sh = hull_solver.solve(surf, soc, soh, v_pre, v_lim, tau, a_, b_, charge, EV.r_eff)
                fp = EV.solve_I(surf, soc, soh, v_pre, v_lim, tau, a_, b_, I0=meas, charge=charge)
                roots_s = sorted(roots, key=abs)
                rec.update({f'{key}_segments': ';'.join(f'{lo:.3g}..{hi:.3g}:{st}' for lo, hi, st in segs),
                            f'{key}_n_roots_exact': len(roots_s), f'{key}_roots_exact': ';'.join(f'{x:.4f}' for x in roots_s),
                            f'{key}_tangent_candidates': ';'.join(f'{x:.3f}' for x in tang), f'{key}_notes': ' '.join(notes),
                            f'{key}_n_roots_grid': nh, f'{key}_grid_status': sh, f'{key}_grid_root': Ih,
                            f'{key}_exact_min_root': (roots_s[0] if roots_s else np.nan),
                            f'{key}_grid_vs_exact': ('agree' if (nh == len(roots_s) and (not roots_s or abs(Ih - roots_s[0]) < 1e-3))
                                                     else 'DIFFER'),
                            f'{key}_fp_meas_start': fp,
                            f'{key}_fp_nearest_root_dist': (min(abs(fp - x) for x in roots_s) if roots_s and np.isfinite(fp) else np.nan),
                            f'{key}_fp_is_min_root': (int(roots_s and np.isfinite(fp) and abs(fp - roots_s[0]) == min(abs(fp - x) for x in roots_s))
                                                      if roots_s and np.isfinite(fp) else '')})
            out.append(rec)
    return out


if __name__ == '__main__':
    CELLS = ['BOOST', 'BOOST_NEGPULSE', 'BOOST_NEGPULSE_1S', 'BOOST_REST', 'CC', 'CC_CELL2']
    with mp.Pool(int(os.environ.get('NPROC', '8'))) as pool:
        res = pool.map(run_direction, [(d, c) for d in ('discharge', 'charge') for c in CELLS])
    rows = [r for rr in res for r in rr]
    os.makedirs(os.path.join(V, 'solver'), exist_ok=True)
    with open(os.path.join(V, 'solver', 'exact_roots_outer.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(len(rows), 'rows -> exact_roots_outer.csv')
