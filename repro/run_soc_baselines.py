"""SOC baselines on the de-circularised perturbation benchmark.

    python3 repro/run_soc_baselines.py

The paper reports one SOC filter with only coulomb counting beside it.  This
runs the estimators a reviewer will expect, all on the same 36 drive runs,
the same causal sensor inputs, the same ECM surfaces and the same seven
perturbation conditions as analysis/soc_perturb_bench.py.

    coulomb counting        the open-loop integral
    1RC-EKF                 the adopted filter with the slow branch removed
                            (k_s = 0, so R2 contributes nothing and v2 stays
                            at zero) - a genuine 1RC, same code path
    2RC-EKF                 the adopted configuration
    adaptive-R EKF          measurement noise driven by the recent residual
                            spread (r_var_k), already in ekf_soc.py
    dual / augmented EKF    current-sensor bias estimated as a fourth state
                            (q_ib), which is the dual-estimation variant
                            section 30.5-30.8 examined
    UKF                     unscented, three states, sharing the SAME model
                            functions as the EKF by subclassing it, so the
                            only difference is the filter algebra

Scoring follows the correction in soc_headline.csv: the headline is the mean
over the SIX disturbance conditions, not over all seven rows.  Both are
printed so the difference stays visible.

Per-cell calibration, stated plainly: build_soc_runs.py constructs
ECMSurface(cell, ...) for the cell being driven, so every filter here reads
its own cell's characterisation surface.  That is per-cell calibrated
deployment.  It is not leave-one-cell-out and must not be described as such.
"""
import argparse
import collections
import csv
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(ROOT, 'analysis')
sys.path.insert(0, ANALYSIS)

RUNS = os.path.join(ANALYSIS, 'results', 'soc_runs.pkl')
OUT = os.path.join(ANALYSIS, 'results', 'tables', 'soc_baselines.csv')

PERTURB = [
    ('no distortion', {}),
    ('initial SOC +10 %p', dict(dsoc=+0.10)),
    ('initial SOC -10 %p', dict(dsoc=-0.10)),
    ('current offset +0.10 A', dict(ibias=+0.10)),
    ('current offset -0.10 A', dict(ibias=-0.10)),
    ('current gain +1 %', dict(igain=+0.01)),
    ('current gain -1 %', dict(igain=-0.01)),
]

GATE = dict(i_gate=1.0, rest_hold_s=30.0)


def rvolt(soh):
    return float(np.interp(soh, [0.70, 0.90, 1.00], [0.110, 0.035, 0.015]))


def make_ukf():
    """A UKF that inherits the EKF's model so only the algebra differs."""
    from ekf_soc import EKF, Q_RATED_AH

    class UKF(EKF):
        # alpha 1e-3 collapses the sigma spread to ~0.1 % of a standard
        # deviation, which is standard for a near-linear model but leaves
        # the weights badly conditioned here.  0.3 keeps the points far
        # enough apart to see the OCV curvature, which is the only reason
        # to prefer a UKF over the EKF on this model.
        ALPHA, BETA, KAPPA = 0.3, 2.0, 0.0
        P_FLOOR = 1e-14

        @staticmethod
        def _sym(A):
            return 0.5 * (A + A.T)

        @classmethod
        def _msqrt(cls, A):
            w, V = np.linalg.eigh(cls._sym(A))
            w = np.clip(w, cls.P_FLOOR, None)
            return (V * np.sqrt(w)) @ V.T

        def step(self, I, V_meas, T=25.0, dt=None):
            dt = self.dt if dt is None else float(min(max(dt, 1e-3), 60.0))
            n = 3
            lam = self.ALPHA ** 2 * (n + self.KAPPA) - n
            x = self.x[:n].copy()
            # Cholesky is not safe here.  Q is tiny (1e-9, 1e-7) and the
            # Kalman update subtracts, so P drifts indefinite within a few
            # hundred steps and cholesky raises.  A symmetric
            # eigendecomposition with the eigenvalues floored gives a square
            # root that always exists and never silently changes the filter
            # when P is healthy.
            P = self._sym(self.P[:n, :n])
            S = self._msqrt((n + lam) * P)
            sig = np.vstack([x, x + S, x - S])
            wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
            wc = wm.copy()
            wm[0] = lam / (n + lam)
            wc[0] = wm[0] + (1 - self.ALPHA ** 2 + self.BETA)

            if self.gamma > 0:
                ah = np.exp(-abs(self.gamma * I * dt / 3600.0 / Q_RATED_AH))
                self.h_det = ah * self.h_det + (1 - ah) * np.sign(I)

            prop = np.empty_like(sig)
            for i, s_ in enumerate(sig):
                R0, R1, t1, R2, t2, _, _ = self._theta(s_[0], I, T)
                a1, a2 = np.exp(-dt / t1), np.exp(-dt / t2)
                soc = s_[0] + I * dt / 3600.0 / Q_RATED_AH
                soc = min(max(soc, self.soc_span[0]), self.soc_span[1])
                prop[i] = (soc,
                           a1 * s_[1] + R1 * (1 - a1) * I,
                           a2 * s_[2] + R2 * (1 - a2) * I)
            xp = wm @ prop
            d = prop - xp
            Pp = self._sym((d * wc[:, None]).T @ d) + self.Q[:n, :n]

            if self.i_gate is not None:
                self.rest_s = (self.rest_s + dt if abs(I) <= self.i_gate
                               else 0.0)
            gated = self.i_gate is not None and (
                abs(I) > self.i_gate or self.rest_s < self.rest_hold_s)

            R0, R1, t1, R2, t2, ocv, s_surf = self._theta(xp[0], I, T)
            M, _ = s_surf.hyst_M(xp[0], self.soh)
            M = float(np.atleast_1d(M)[0])
            if not np.isfinite(M):
                M = 0.0
            y_hat = ocv + M * self.h_det + I * R0 + xp[1] + xp[2]
            if gated:
                self.x[:n], self.P[:n, :n] = xp, Pp
                return self.x[0], y_hat

            ys = np.empty(2 * n + 1)
            for i, s_ in enumerate(prop):
                R0i, _, _, _, _, ocvi, si = self._theta(s_[0], I, T)
                Mi, _ = si.hyst_M(s_[0], self.soh)
                Mi = float(np.atleast_1d(Mi)[0])
                if not np.isfinite(Mi):
                    Mi = 0.0
                ys[i] = ocvi + Mi * self.h_det + I * R0i + s_[1] + s_[2]
            ybar = float(wm @ ys)
            dy = ys - ybar
            Pyy = float((wc * dy) @ dy) + self.R
            Pxy = (d * (wc * dy)[:, None]).sum(0)
            K = Pxy / Pyy
            self.x[:n] = xp + K * (V_meas - ybar)
            self.x[0] = min(max(self.x[0], self.soc_span[0]),
                            self.soc_span[1])
            Pn = self._sym(Pp - np.outer(K, K) * Pyy)
            w, V = np.linalg.eigh(Pn)
            self.P[:n, :n] = (V * np.clip(w, self.P_FLOOR, None)) @ V.T
            return self.x[0], ybar

    return UKF


def run_filter(kind, r, pkw):
    """One drive run under one perturbation.  Returns RMSE in %p."""
    from ekf_soc import run as ekf_run
    If = r['I'] * (1.0 + pkw.get('igain', 0.0)) + pkw.get('ibias', 0.0)
    s0 = float(np.clip(float(r['soc'][0]) + pkw.get('dsoc', 0.0), 0.02, 0.98))
    rv = rvolt(r['soh'])

    if kind == 'coulomb counting':
        est, _ = ekf_run(r['sd'], r['sc'], r['soh'], If, r['V'], r['T'],
                         s0, 1e4, gamma=0.0)
    elif kind == '2RC-EKF (adopted)':
        est, _ = ekf_run(r['sd'], r['sc'], r['soh'], If, r['V'], r['T'],
                         s0, rv, gamma=0.0, **GATE)
    elif kind == '1RC-EKF':
        est, _ = ekf_run(r['sd'], r['sc'], r['soh'], If, r['V'], r['T'],
                         s0, rv, gamma=0.0, k_s=0.0, **GATE)
    elif kind == 'adaptive-R EKF':
        est, _ = ekf_run(r['sd'], r['sc'], r['soh'], If, r['V'], r['T'],
                         s0, rv, gamma=0.0, r_var_k=20.0, ew_rate=0.003,
                         **GATE)
    elif kind == 'dual EKF (current bias)':
        est, _ = ekf_run(r['sd'], r['sc'], r['soh'], If, r['V'], r['T'],
                         s0, rv, gamma=0.0, q_ib=1e-10, p0_ib=1e-5, **GATE)
    elif kind == 'UKF':
        UKF = make_ukf()
        f = UKF(r['sd'], r['sc'], r['soh'], R_volt=rv, gamma=0.0, **GATE)
        f.x[0] = s0
        out = np.empty(len(If))
        for i in range(len(If)):
            out[i], _ = f.step(float(If[i]), float(r['V'][i]),
                               float(r['T'][i]))
        est = out
    else:
        raise ValueError(kind)
    e = est - r['soc']
    return float(np.sqrt(np.mean(e ** 2)))


_RUNS = None


def _one(job):
    kind, pi = job
    pname, pkw = PERTURB[pi]
    errs = [run_filter(kind, r, pkw) for r in _RUNS]
    return (kind, pname, float(np.mean(errs)) * 100,
            float(np.max(errs)) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=RUNS)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--only', default=None)
    a = ap.parse_args()
    if not os.path.exists(a.runs):
        print(f'  missing {a.runs} — run repro/build_soc_runs.py',
              file=sys.stderr)
        return 1
    R = pickle.load(open(a.runs, 'rb'))
    kinds = ['coulomb counting', '1RC-EKF', '2RC-EKF (adopted)',
             'adaptive-R EKF', 'dual EKF (current bias)', 'UKF']
    if a.only:
        kinds = [k for k in kinds if a.only in k]
    print(f'  {len(R)} drive runs, {len(PERTURB)} conditions, '
          f'{len(kinds)} estimators')

    # Serial this is ~18 min per (estimator, condition) and the UKF is
    # seven times that, so 42 combinations would run for half a day.  Same
    # pool layout as analysis/soc_perturb_bench.py.
    global _RUNS
    _RUNS = R
    jobs = [(k, pi) for k in kinds for pi in range(len(PERTURB))]
    with mp.Pool(min(14, len(jobs))) as pool:
        res = pool.map(_one, jobs)

    rows, table = [], collections.defaultdict(dict)
    for (kind, pname, mean_pct, worst_pct) in res:
        table[kind][pname] = mean_pct
        rows.append([kind, pname, f'{mean_pct:.3f}', f'{worst_pct:.3f}'])
        print(f'    {kind:<26}{pname:<26}{mean_pct:>8.3f} %p', flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['estimator', 'perturbation', 'rmse_mean_pct',
                    'rmse_worst_run_pct'])
        w.writerows(rows)
    print(f'\n  -> {os.path.relpath(a.out, ROOT)}  ({len(rows)} rows)')

    print(f"\n  {'estimator':<26}{'undist':>9}{'mean6':>9}{'mean7':>9}"
          f"{'worst6':>9}")
    print('  ' + '-' * 62)
    for kind in kinds:
        d = table[kind]
        dist = [v for k, v in d.items() if k != 'no distortion']
        print(f"  {kind:<26}{d['no distortion']:>9.3f}{np.mean(dist):>9.3f}"
              f"{np.mean(list(d.values())):>9.3f}{max(dist):>9.3f}")
    print('\n  mean6 is the headline: the mean over the six disturbances.')
    print('  Every filter reads its own cell\'s characterisation surface — '
          'per-cell calibrated deployment, not leave-one-cell-out.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
