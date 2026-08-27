"""Dual EKF: fast state (SOC, V1, V2) plus slow parameters (R0 multiplier, capacity).

WHY TWO FILTERS AND NOT ONE AUGMENTED STATE
    SOC moves in seconds, capacity in weeks. Putting them in one covariance makes
    the fast state's innovations drag the slow parameter around, and the usual
    cure - shrinking the parameter's process noise until it stops moving - just
    turns it off. Running them separately keeps each one's noise honest.

WHAT m0 ACTUALLY IS, STATED PLAINLY
    It multiplies the ohmic branch, so it looks like a resistance estimate. It is
    NOT one. The open-loop ECM error grows from 13.7 mV at SOH 1.00 to 108 mV at
    0.70, and a drive-cycle refit of the same three multipliers already showed
    what happens when they are free: R1's multiplier goes NEGATIVE, R0's climbs
    to 3.5x, and held-out error triples (ecm_refine.py). m0 here is a model-error
    absorber that happens to enter through R0. Reporting it as measured ohmic
    resistance would be wrong.

    It is still worth estimating: the question this file answers is whether
    absorbing that error restores SOC accuracy below SOH 0.80, where the plain
    EKF converges in hours instead of minutes.

SOH IS NOT OBSERVED THROUGH COULOMB COUNTING ON THIS AXIS
    The obvious estimator, Q = charge counted / SOC traversed, DOES NOT WORK
    here and the reason is structural, not numerical. This project's SOC axis is
    anchored at full and divided by the 3.0 Ah rating, so dSOC/dt = I/(3600*3.0)
    BY DEFINITION. The ratio therefore returns 3.0 Ah whatever the cell's real
    capacity is. Run against cells from SOH 1.000 to 0.696 it returned 2.83-2.88
    Ah every time - a constant wearing the costume of an estimate.

    On a rated axis the health shows up in two other places:
      - WHICH OCV CURVE FITS. OCV(SOC, SOH) moves ~150 mV over the life at fixed
        SOC (findings.md), so an assumed SOH that is wrong leaves a systematic
        voltage innovation.
      - THE REACHABLE SOC RANGE, since an aged cell hits its voltage floor at
        SOC 1 - CAP/3 rather than 0.

    So SOH is estimated by scanning candidate curves and taking the one whose
    innovation is smallest. That is slower than a recursive update but it is
    what the axis actually supports.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecm_surface import ECMSurface  # noqa: E402

Q_RATED_AH = 3.0


class DualEKF:
    def __init__(self, sd, sc, soh, R_volt, dt=1.0,
                 q_soc=1e-9, q_v=1e-7, q_m0=1e-8, m0_bounds=(0.5, 6.0),
                 gamma=20.0):
        self.sd, self.sc, self.soh, self.dt = sd, sc, soh, dt
        self.R = R_volt ** 2
        self.Q = np.diag([q_soc, q_v, q_v])
        self.P = np.diag([0.04, 1e-4, 1e-4])
        self.x = np.zeros(3)
        # The SOC range the OCV table actually covers for this cell, so the
        # estimate is never pushed somewhere the measurement cannot see it.
        pts = sd.ocv_field.near.points
        self.soc_span = (float(pts[:, 0].min()), float(pts[:, 0].max()))
        self.m0 = 1.0
        self.Pm = 0.25
        self.q_m0 = q_m0
        self.lo, self.hi = m0_bounds
        # Hysteresis is driven purely by current throughput, so it is carried as
        # a deterministic auxiliary rather than a filter state - there is nothing
        # for the covariance to learn about it. Its magnitude M(SOC, SOH) is
        # measured, not fitted, and only gamma is free.
        self.gamma = gamma
        self.h = 0.0

    def _th(self, soc, I, T):
        s = self.sc if I > 0 else self.sd
        th = s.theta(soc, self.soh, I, T)
        o, _ = s.ocv(soc, self.soh)
        return (float(th["R0"][0]), float(th["R1"][0]), float(th["tau1"][0]),
                float(th["R2"][0]), float(th["tau2"][0]), float(o[0]), s)

    def _docv(self, soc, s, h=0.01):
        """dOCV/dSOC, never allowed to reach zero.

        Outside the OCV table the surface returns its nearest neighbour, so a
        centred difference there gives EXACTLY ZERO - and a zero here removes
        SOC from the measurement Jacobian entirely, after which the filter can
        never correct it again. Seen in practice: an initial SOC of 1.026 (the
        true value plus a 0.20 test offset) pinned the estimate at the 1.05 clamp
        for 12,000 s while the cell really fell to 0.25.

        So the evaluation points are pulled inside the table's own SOC span, and
        the slope is floored at a small positive value.
        """
        lo, hi = self.soc_span
        c = min(max(soc, lo + h), hi - h)
        a, _ = s.ocv(c + h, self.soh)
        b, _ = s.ocv(c - h, self.soh)
        d = (float(a[0]) - float(b[0])) / (2 * h)
        return d if abs(d) > 1e-3 else 1e-3

    def step(self, I, V, T=25.0):
        dt = self.dt
        soc, v1, v2 = self.x
        R0, R1, t1, R2, t2, _, _ = self._th(soc, I, T)
        a1, a2 = np.exp(-dt / t1), np.exp(-dt / t2)

        soc_p = min(max(soc + I * dt / 3600.0 / Q_RATED_AH,
                        self.soc_span[0]), self.soc_span[1])
        xp = np.array([soc_p, a1 * v1 + R1 * (1 - a1) * I,
                       a2 * v2 + R2 * (1 - a2) * I])
        F = np.diag([1.0, a1, a2])
        P = F @ self.P @ F.T + self.Q

        R0, R1, t1, R2, t2, ocv, s = self._th(xp[0], I, T)
        vh = 0.0
        if self.gamma:
            M, _ = s.hyst_M(xp[0], self.soh)
            vh = M * self.h
        y = ocv + vh + I * R0 * self.m0 + xp[1] + xp[2]
        innov = V - y

        # -- parameter branch first, on the same innovation ------------------
        # dy/dm0 = I*R0, so m0 is only observable while current flows; with no
        # current the gain is zero and the estimate simply holds.
        # Pm == 0 is the OFF switch. Setting only the process noise to zero does
        # NOT freeze the estimate - the prior covariance is still there and the
        # update still fires, which is how the first "m0 off" comparison came out
        # identical to the "m0 on" one.
        if self.Pm > 0:
            Hm = I * R0
            self.Pm += self.q_m0
            Sm = Hm * self.Pm * Hm + self.R
            Km = self.Pm * Hm / Sm
            self.m0 = float(np.clip(self.m0 + Km * innov, self.lo, self.hi))
            self.Pm = (1 - Km * Hm) * self.Pm

        # -- state branch ----------------------------------------------------
        y = ocv + vh + I * R0 * self.m0 + xp[1] + xp[2]
        H = np.array([self._docv(xp[0], s), 1.0, 1.0])
        S = H @ P @ H + self.R
        K = P @ H / S
        self.x = xp + K * (V - y)
        self.x[0] = min(max(self.x[0], self.soc_span[0]), self.soc_span[1])
        self.P = (np.eye(3) - np.outer(K, H)) @ P
        if self.gamma:
            ah = np.exp(-abs(self.gamma * I * dt / 3600.0 / Q_RATED_AH))
            self.h = ah * self.h + (1 - ah) * np.sign(I)
        return self.x[0], y, self.m0


def soh_by_scan(sd, sc, I, V, T, soc0, rv, candidates, start=0, gamma=20.0):
    """SOH whose OCV curve leaves the smallest voltage innovation.

    Each candidate reruns the state filter with that SOH's own OCV curve and
    parameter surface; the winner is the curve the measurements actually support.
    """
    best = (np.inf, np.nan)
    scores = []
    for cand in candidates:
        f = DualEKF(sd, sc, cand, rv, gamma=gamma)
        f.x = np.array([soc0, 0.0, 0.0])
        e2 = n = 0.0
        for j in range(len(I)):
            _, y, _ = f.step(I[j], V[j], T[j])
            if j >= start:
                e2 += (y - V[j]) ** 2; n += 1
        r = float(np.sqrt(e2 / max(n, 1)))
        scores.append((cand, r))
        if r < best[0]:
            best = (r, cand)
    return best[1], scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="CC")
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(__file__), "cache"))
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--max-samples", type=int, default=20000)
    ap.add_argument("--soc-error", type=float, default=0.20)
    ap.add_argument("--no-m0", action="store_true", help="freeze m0 at 1 (plain EKF)")
    ap.add_argument("--gamma", type=float, default=20.0,
                    help="hysteresis rate constant (0 disables)")
    ap.add_argument("--scan-soh", action="store_true",
                    help="also estimate SOH by scanning OCV curves")
    args = ap.parse_args()

    sd = ECMSurface(args.cell, "discharge")
    sc = ECMSurface(args.cell, "charge")
    z = np.load(os.path.join(args.cache,
                             f"uypydj_{args.cell}_Fifteen_Drive_Cycles.npz"))
    lens = z["lens"]; off = np.concatenate([[0], np.cumsum(lens)])
    idx = np.linspace(0, len(lens) - 1, args.runs).astype(int)
    cap_fresh = None

    print(f"cell {args.cell}   m0 추정 {'끔' if args.no_m0 else '켬'}, "
          f"초기 SOC {args.soc_error:+.2f}\n")
    print(f"{'run':>4} {'SOH참':>7} {'수렴':>7} {'말단오차':>9} {'SOC RMSE':>9} "
          f"{'전압':>8} {'m0':>6} {'SOH추정':>9}")
    for k in idx:
        sl = slice(off[k], off[k] + lens[k])
        soc, V, I, SOH, T, CAP = (z[x][sl] for x in ("SOC", "V", "I", "SOH", "T", "CAP"))
        ok = np.isfinite(soc) & np.isfinite(V) & np.isfinite(I) & np.isfinite(T)
        if ok.sum() < 2000:
            continue
        soc, V, I, T = (x[ok][:args.max_samples] for x in (soc, V, I, T))
        soh = float(np.nanmedian(SOH)); cap = float(np.nanmedian(CAP))
        if cap_fresh is None:
            cap_fresh = cap / soh if soh > 0 else cap
        rv = float(np.interp(soh, [0.70, 0.90, 1.00], [0.110, 0.035, 0.015]))

        f = DualEKF(sd, sc, soh, rv, q_m0=1e-8, gamma=args.gamma)
        if args.no_m0:
            f.Pm = 0.0
        f.x = np.array([soc[0] + args.soc_error, 0.0, 0.0])
        est = np.empty(len(I)); pr = np.empty(len(I)); m0 = np.empty(len(I))
        for j in range(len(I)):
            est[j], pr[j], m0[j] = f.step(I[j], V[j], T[j])

        load = np.flatnonzero(np.abs(I) > 1.0)
        t0 = load[0] if len(load) else 0
        e = est - soc
        c = np.flatnonzero(np.abs(e) < 0.02); c = c[c >= t0]
        sohe = np.nan
        if args.scan_soh:
            step = max(1, len(I) // 6000)
            cands = np.round(np.arange(0.68, 1.005, 0.04), 3)
            sohe, _ = soh_by_scan(sd, sc, I[::step], V[::step], T[::step],
                                  soc[0] + args.soc_error, rv, cands,
                                  start=t0 // step, gamma=args.gamma)
        print(f"{k:>4} {soh:>7.3f} {(c[0]-t0) if len(c) else -1:>6}s "
              f"{e[-1]:>+9.4f} {np.sqrt(np.mean(e[t0:]**2)):>9.4f} "
              f"{np.sqrt(np.mean((pr[t0:]-V[t0:])**2))*1000:>7.1f}m {m0[-1]:>6.2f} "
              f"{sohe:>9.3f}")


if __name__ == "__main__":
    main()
