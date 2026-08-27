"""The SOC benchmark with the circularity broken.

The old benchmark gave the filter exactly the current that made the label and
started it at the exact initial SOC.  The label is SOC = 1 + Ah/3.0 and the
filter prediction is soc + I dt/3600/3.0, so the two are the same equation.
Not using voltage therefore wins unconditionally (measured: pure current
integration 0.12 %p vs the best filter 0.84 %p).

The reason to use a Kalman filter is that those three things do not hold in a
real vehicle: the initial SOC is unknown, the current sensor has an offset,
and it has a gain error.  Here each of the three is injected in turn and the
measurement repeated.  The label is built on the true current; only the filter
is given the distorted current.

Both the overall RMSE and the last-quarter RMSE are reported.  The former also
mixes in convergence speed; the latter shows only the error left after
convergence.
"""
import os
import numpy as np
import pickle
import multiprocessing as mp

RUNS = None

CONFIGS = [
    ('pure current integration', dict(_open=True)),
    ('EKF no gate', dict()),
    ('EKF adopted (gate)', dict(i_gate=1.0, rest_hold_s=30.0)),
    ('EKF gate + spread k=20', dict(i_gate=1.0, rest_hold_s=30.0,
                                    r_var_k=20., ew_rate=0.003)),
    ('EKF gate + spread k=200', dict(i_gate=1.0, rest_hold_s=30.0,
                                     r_var_k=200., ew_rate=0.003)),
]

PERTURB = [
    ('no distortion', dict()),
    ('initial SOC +10 %p', dict(dsoc=+0.10)),
    ('initial SOC -10 %p', dict(dsoc=-0.10)),
    ('current offset +0.10 A', dict(ibias=+0.10)),
    ('current offset -0.10 A', dict(ibias=-0.10)),
    ('current gain +1 %', dict(igain=+0.01)),
    ('current gain -1 %', dict(igain=-0.01)),
]


def _one(job):
    from ekf_soc import run as ekf_run
    ci, pi = job
    cname, ckw = CONFIGS[ci]
    pname, pkw = PERTURB[pi]
    ckw = dict(ckw)
    open_loop = ckw.pop('_open', False)
    full, tail = [], []
    for r in RUNS:
        rv = 1e4 if open_loop else float(
            np.interp(r['soh'], [0.70, 0.90, 1.00], [0.110, 0.035, 0.015]))
        # Distort only the current the filter sees; the label keeps the one
        # built from the true current
        If = r['I'] * (1.0 + pkw.get('igain', 0.0)) + pkw.get('ibias', 0.0)
        s0 = float(r['soc'][0]) + pkw.get('dsoc', 0.0)
        s0 = min(max(s0, 0.02), 0.98)
        est, _ = ekf_run(r['sd'], r['sc'], r['soh'], If, r['V'], r['T'],
                         s0, rv, gamma=0.0, **ckw)
        e = est - r['soc']
        full.append(float(np.sqrt(np.mean(e ** 2))))
        tail.append(float(np.sqrt(np.mean(e[-len(e) // 4:] ** 2))))
    return ci, pi, np.array(full), np.array(tail)


def main():
    global RUNS
    RUNS = pickle.load(open(os.environ.get('SOC_RUNS',
                            'results/soc_runs.pkl'), 'rb'))
    soh = np.array([r['soh'] for r in RUNS])
    jobs = [(c, p) for c in range(len(CONFIGS)) for p in range(len(PERTURB))]
    with mp.Pool(14) as pool:
        res = pool.map(_one, jobs)
    F = {(c, p): f for c, p, f, _ in res}
    T = {(c, p): t for c, p, _, t in res}
    import os as _os, csv as _csv
    _os.makedirs('results/tables', exist_ok=True)
    with open('results/tables/soc_perturb.csv', 'w', newline='',
              encoding='utf-8') as _f:
        _w = _csv.writer(_f)
        _w.writerow(['config', 'perturbation', 'rmse_full_pct',
                     'rmse_tail_pct', 'worst_pct'])
        for _c, (_cn, _) in enumerate(CONFIGS):
            for _p, (_pn, _) in enumerate(PERTURB):
                _w.writerow([_cn, _pn, f'{F[(_c, _p)].mean()*100:.3f}',
                             f'{T[(_c, _p)].mean()*100:.3f}',
                             f'{F[(_c, _p)].max()*100:.3f}'])
    print('  -> results/tables/soc_perturb.csv', flush=True)
    np.savez('results/soc_perturb.npz', soh=soh,
             full=np.array([F[(c, p)] for c in range(len(CONFIGS))
                            for p in range(len(PERTURB))]),
             tail=np.array([T[(c, p)] for c in range(len(CONFIGS))
                            for p in range(len(PERTURB))]))

    for tag, D in (('overall RMSE (%p, mean over 36 runs)', F),
                   ('last-quarter RMSE (%p) — error left after convergence',
                    T)):
        print(f"\n  == {tag}", flush=True)
        print(f"  {'distortion':<24}" + ''.join(f"{n[:16]:>18}"
                                                 for n, _ in CONFIGS),
              flush=True)
        print('  ' + '-' * (24 + 18 * len(CONFIGS)), flush=True)
        for p, (pname, _) in enumerate(PERTURB):
            row = ''.join(f"{D[(c, p)].mean()*100:>18.2f}"
                          for c in range(len(CONFIGS)))
            print(f"  {pname:<24}{row}", flush=True)
        row = ''.join(f"{np.mean([D[(c, p)].mean() for p in range(1, len(PERTURB))])*100:>18.2f}"
                      for c in range(len(CONFIGS)))
        print(f"  {'mean of the 6 distortions':<24}{row}", flush=True)


if __name__ == '__main__':
    main()
