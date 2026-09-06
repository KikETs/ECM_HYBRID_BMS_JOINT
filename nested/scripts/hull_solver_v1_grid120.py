"""Reference-independent, hull-aware current solver with explicit failure.

solve(surf, soc, soh, v_pre, v_lim, tau, kf, ks, charge) -> (I, n_roots, status)

  * scans |I| on a log grid over the shipped domain ([0.1, 400] A), keeps the
    grid points where R_eff is finite and positive (inside the surface hull),
  * brackets every sign change of f(I) = V_pre + I R_eff(I) - V_lim between
    adjacent valid grid points and solves each with brentq,
  * returns the root of smallest magnitude (the first current at which the
    voltage constraint binds), the number of roots found, and a status:
        'ok'        exactly one root
        'multi'     several roots; the smallest-|I| one is returned
        'noroot'    valid domain exists but f has no sign change in it
        'hull'      no valid grid point at all
  The reference current is never used.  Rows with status 'noroot' or 'hull'
  carry I = nan and must be treated as failures by the caller.
"""
import numpy as np
from scipy.optimize import brentq

NGRID = 120
_G = np.logspace(np.log10(0.1), np.log10(400.0), NGRID)


def solve(surf, soc, soh, v_pre, v_lim, tau, kf, ks, charge, r_eff):
    grid = _G if charge else -_G[::-1]

    def f(I):
        R = r_eff(surf, soc, soh, I, tau, kf, ks)
        if not np.isfinite(R) or R <= 0:
            return np.nan
        return v_pre + I * R - v_lim

    F = np.array([f(I) for I in grid])
    ok = np.isfinite(F)
    if not ok.any():
        return np.nan, 0, 'hull'
    roots = []
    for k in range(NGRID - 1):
        if not (ok[k] and ok[k + 1]):
            continue
        if F[k] == 0.0:
            roots.append(float(grid[k]))
        elif np.sign(F[k]) != np.sign(F[k + 1]):
            try:
                roots.append(float(brentq(f, grid[k], grid[k + 1],
                                          xtol=1e-9, rtol=1e-12, maxiter=200)))
            except Exception:
                pass
    if ok[-1] and F[-1] == 0.0:
        roots.append(float(grid[-1]))
    if not roots:
        return np.nan, 0, 'noroot'
    I = min(roots, key=abs)
    return I, len(roots), ('ok' if len(roots) == 1 else 'multi')
