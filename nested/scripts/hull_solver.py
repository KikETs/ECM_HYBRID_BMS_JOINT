"""Reference-independent, hull-aware current solver with explicit failure.
Version 2 (2026-09-06): segment-aware search, vectorised.

solve(surf, soc, soh, v_pre, v_lim, tau, kf, ks, charge, r_eff)
    -> (I, n_roots, status)

The equation f(I) = V_pre + I R_eff(I) - V_lim is searched on the |I| domain
[0.1, 400] A split at the surface's rank currents.  ECMSurface.theta
interpolates the five RC parameters linearly in |I| between the two ranks
that bracket |I| (clamped outside), and its in_hull flag is the AND of the
two bracketing ranks' (SOC, SOH) hulls, so validity is constant inside a
segment and can only change at a rank current.  The parameter fields of the
two bracketing ranks are therefore evaluated once per segment and the
NSEG samples inside the segment are formed by the same linear blend theta()
uses; the brentq refinement calls the caller's r_eff, and one sample per
segment is cross-checked against r_eff (assert) so the fast path cannot
drift from the shipped evaluator.

Inside every valid segment:
  * NSEG uniform samples including both ends (minus 1e-9 A);
  * every sign change is bracketed and solved by brentq on r_eff;
  * every local minimum of |f| below REFINE_MV without a bracketed root is
    re-sampled REFINE_N times finer; sign changes found there are solved,
    otherwise the point is a tangent candidate.
Version 1 (hull_solver_v1_grid120.py) used 120 log-spaced points over the
whole domain and missed roots within one grid spacing of a rank current and
close root pairs (rverify_20260906/solver/exact_roots_outer.csv).

Returned current: the root of smallest magnitude.  Status:
    'ok' one root | 'multi' several (smallest returned) | 'noroot' valid
    domain, no sign change | 'tangent' no sign change but a |f| dip below
    REFINE_MV (failure, nan) | 'hull' no valid segment.
The reference current is never used.  Temperature is the reference (25 C):
theta()'s temperature factor is identity there, as in eval_sop_amps.
"""
import numpy as np
from scipy.optimize import brentq

NSEG = 200
REFINE_MV = 5e-3
REFINE_N = 40
_EPS = 1e-9


def _seg_params(surf, soc, soh, lo_amp, hi_amp):
    """(Vlo, Vhi, in_hull, frac_fn) for the segment [lo_amp, hi_amp] of |I|."""
    rI = np.asarray(surf.rank_I, float)
    mid = 0.5 * (lo_amp + hi_amp)
    w = float(np.interp(mid, rI, np.arange(len(rI))))
    lo, hi = int(np.floor(w)), int(np.ceil(w))
    vlo, oklo = surf.ranks[lo](soc, soh)
    vhi, okhi = surf.ranks[hi](soc, soh)
    vlo = np.atleast_2d(vlo)[0]; vhi = np.atleast_2d(vhi)[0]
    inh = bool(np.atleast_1d(oklo)[0]) and bool(np.atleast_1d(okhi)[0])
    if lo == hi:
        frac = lambda x: np.zeros_like(x)
    else:
        frac = lambda x: np.clip((x - rI[lo]) / (rI[hi] - rI[lo]), 0.0, 1.0)
    return vlo, vhi, inh, frac


def _f_vec(vlo, vhi, frac, xs, sgn, v_pre, v_lim, tau, kf, ks):
    fr = frac(xs)[:, None]
    M = vlo[None, :] * (1 - fr) + vhi[None, :] * fr
    R0, R1, t1, R2, t2 = (M[:, k] for k in range(5))
    R = (kf * (R0 / 1000.0 + R1 / 1000.0 * (1 - np.exp(-tau / t1)))
         + ks * R2 / 1000.0 * (1 - np.exp(-tau / t2)))
    f = v_pre + (sgn * xs) * R - v_lim
    return np.where(R > 0, f, np.nan)


def solve(surf, soc, soh, v_pre, v_lim, tau, kf, ks, charge, r_eff):
    sgn = 1.0 if charge else -1.0
    rI = [float(x) for x in np.asarray(surf.rank_I)]
    edges = [0.1] + [x for x in sorted(rI) if 0.1 < x < 400.0] + [400.0]

    def f_exact(I):
        R = r_eff(surf, soc, soh, I, tau, kf, ks)
        if not np.isfinite(R) or R <= 0:
            return np.nan
        return v_pre + I * R - v_lim

    roots, tangent, any_valid = [], False, False
    for lo_amp, hi_amp in zip(edges[:-1], edges[1:]):
        vlo, vhi, inh, frac = _seg_params(surf, soc, soh, lo_amp, hi_amp)
        if not inh:
            continue
        xs = np.linspace(lo_amp + _EPS, hi_amp - _EPS, NSEG)
        F = _f_vec(vlo, vhi, frac, xs, sgn, v_pre, v_lim, tau, kf, ks)
        ok = np.isfinite(F)
        if not ok.any():
            continue
        any_valid = True
        # cross-check the fast path against the shipped evaluator at one sample
        k0 = int(np.flatnonzero(ok)[len(np.flatnonzero(ok)) // 2])
        fe = f_exact(sgn * xs[k0])
        if not (np.isfinite(fe) and abs(fe - F[k0]) < 1e-9):
            raise RuntimeError(f'hull_solver fast path disagrees with r_eff at I={sgn * xs[k0]:.4f}: {F[k0]} vs {fe}')
        found = []
        for k in range(NSEG - 1):
            if not (ok[k] and ok[k + 1]):
                continue
            if F[k] == 0.0:
                found.append(k); roots.append(float(sgn * xs[k]))
            elif np.sign(F[k]) != np.sign(F[k + 1]):
                found.append(k)
                try:
                    roots.append(float(brentq(f_exact, sgn * xs[k], sgn * xs[k + 1], xtol=1e-9, rtol=1e-12, maxiter=200)))
                except Exception:
                    pass
        if ok[-1] and F[-1] == 0.0:
            roots.append(float(sgn * xs[-1]))
        av = np.where(ok, np.abs(F), np.inf)
        for k in range(NSEG):
            if not ok[k] or av[k] >= REFINE_MV:
                continue
            left = av[k - 1] if k > 0 else np.inf
            right = av[k + 1] if k < NSEG - 1 else np.inf
            if not (av[k] <= left and av[k] <= right) or any(abs(j - k) <= 1 for j in found):
                continue
            a = xs[max(k - 1, 0)]; b = xs[min(k + 1, NSEG - 1)]
            ys = np.linspace(a, b, REFINE_N)
            G = _f_vec(vlo, vhi, frac, ys, sgn, v_pre, v_lim, tau, kf, ks)
            hit = False
            for m in range(REFINE_N - 1):
                if np.isfinite(G[m]) and np.isfinite(G[m + 1]) and np.sign(G[m]) != np.sign(G[m + 1]):
                    try:
                        roots.append(float(brentq(f_exact, sgn * ys[m], sgn * ys[m + 1], xtol=1e-9, rtol=1e-12, maxiter=200)))
                        hit = True
                    except Exception:
                        pass
            if not hit:
                tangent = True
    if not any_valid:
        return np.nan, 0, 'hull'
    if not roots:
        return np.nan, 0, ('tangent' if tangent else 'noroot')
    roots = sorted(set(round(x, 9) for x in roots), key=abs)
    return roots[0], len(roots), ('ok' if len(roots) == 1 else 'multi')
