"""SOP with SOC coming from the filter, not from the file.

WHAT WAS ORACLE ABOUT THE 4.94 A
    eval_sop_amps.py reads SOC and SOH out of the data file. A BMS estimates
    them. §11's error propagation injected synthetic SOC offsets and found the
    answer degrades to ~14 A at a systematic 2 % - larger than the entire benefit
    of the resistance correction. That was an assumption about the estimator;
    this runs the estimator.

THE FILTER USES THE POOLED SURFACE, NOT THE CELL'S OWN
    ekf_soc.py's driver instantiates ECMSurface(cell), which is that cell's own
    fitted ECM - fine for asking "does the filter converge", not fine for an
    end-to-end number, because the held-out cell's resistances would enter
    through the filter. Here the filter is given the leave-one-cell-out pooled
    surface, the same one the hybrid arm corrects, so the whole chain is LOCO.

    That also means the EKF numbers here are NOT comparable to ecm_kf_plan.md's
    0.0261 - those were measured with per-cell surfaces and are optimistic.

TWO START CONDITIONS
    `warm`  the filter enters the characterisation already correct. Isolates the
            error the model keeps producing while running.
    `cold`  it enters 0.10 wrong. Adds the convergence cost, which is what a BMS
            pays after a battery swap or a long park with drift.

SOH IS STILL TAKEN FROM THE FILE
    So this measures the SOC path only. The SOH path costs more per unit
    (dI*/dSOH is about 3x dI*/dSOC) and would need the charge-curve CNN's own
    per-cell output threaded in; that is a separate step and is not done here.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecm_pool import surfaces  # noqa: E402
from ekf_soc import run as ekf_run  # noqa: E402
from eval_a13 import pulse_index  # noqa: E402
from eval_sop_amps import solve_I, trim_k  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL = os.path.join(HERE, "sop_amps_eval.csv")
CACHE = os.path.join(HERE, "cache_t")
OUT = os.path.join(HERE, "sop_end_to_end.csv")


def soc_at_pulses(cell, sd, sc, cold, cache=CACHE):
    """EKF SOC at every pulse start of every HPPC run. Keyed (cycle, soc_group)."""
    import re
    z = np.load(os.path.join(cache, f"uypydj_{cell}_HPPC.npz"), allow_pickle=True)
    lens, files = z["lens"], z["files"]
    off = np.concatenate([[0], np.cumsum(lens)])
    cols = {k: z[k] for k in ("t", "V", "I", "T", "SOC", "SOH", "valid")}
    idx, _ = pulse_index(cell, cache)
    bybase = collections.defaultdict(list)
    for (cyc, grp, rank, d), (base, a, ks, end, ip) in idx.items():
        if d == "discharge" and rank == "3":
            bybase[base].append((cyc, grp, a))
    out = {}
    for k in range(len(lens)):
        base = int(off[k]); n = int(lens[k])
        if base not in bybase:
            continue
        sl = slice(base, base + n)
        t, V, I, T, soc, SOH, ok = (cols[x][sl] for x in
                                    ("t", "V", "I", "T", "SOC", "SOH", "valid"))
        g = (ok.astype(bool) & np.isfinite(V) & np.isfinite(I) & np.isfinite(T)
             & np.isfinite(soc))
        if g.sum() < 1000:
            continue
        soh = float(np.nanmedian(SOH))
        rv = float(np.interp(soh, [0.70, 0.90, 1.00], [0.110, 0.035, 0.015]))
        gi = np.flatnonzero(g)
        est, _ = ekf_run(sd, sc, soh, I[g], V[g], T[g],
                         float(soc[g][0]) + cold, rv, gamma=20.0, t=t[g])
        # map each pulse's local index onto the compacted series
        pos = np.searchsorted(gi, [a for _, _, a in bybase[base]])
        for (cyc, grp, a), p in zip(bybase[base], pos):
            if p >= len(est):
                continue
            out[(cyc, grp)] = (float(est[p]), float(soc[g][min(p, len(est) - 1)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--cold", type=float, default=0.0,
                    help="initial SOC offset the filter enters with")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.label, encoding="utf-8")))
    lab = [r for r in rows if float(r["tau_s"]) == 10.0]
    cells = sorted({r["cell"] for r in lab})
    out, drop = [], collections.Counter()
    for cell in cells:
        sd, sc = surfaces(cell)
        K = trim_k(cell)
        est = soc_at_pulses(cell, sd, sc, args.cold)
        n = 0
        for r in lab:
            if r["cell"] != cell:
                continue
            key = (int(r["cycle"]), int(float(r.get("soc_group", -1))
                                        if "soc_group" in r else -1))
            # sop_amps_eval.csv has no soc_group; match on SOC instead
            cand = [(k, v) for k, v in est.items() if k[0] == int(r["cycle"])]
            if not cand:
                drop["EKF 추정 없음"] += 1
                continue
            true_soc = float(r["SOC"])
            k, (se, st) = min(cand, key=lambda kv: abs(kv[1][1] - true_soc))
            if abs(st - true_soc) > 0.08:
                drop["SOC 매칭 실패"] += 1
                continue
            cyc = int(r["cycle"])
            if cyc not in K:
                drop["k 없음"] += 1
                continue
            kf, ks = K[cyc]
            vpre = float(r["V_pre_V"]); soh = float(r["SOH"])
            meas = float(r["I_meas_A"])
            i_true = solve_I(sd, true_soc, soh, vpre, 2.5, 10.0, kf, ks, I0=meas)
            i_ekf = solve_I(sd, float(np.clip(se, 0.02, 1.0)), soh, vpre, 2.5,
                            10.0, kf, ks, I0=meas)
            if not (np.isfinite(i_true) and np.isfinite(i_ekf)):
                drop["hull 밖"] += 1
                continue
            out.append({"cell": cell, "cycle": cyc, "SOH": soh,
                        "SOC_true": round(true_soc, 4),
                        "SOC_ekf": round(se, 4),
                        "soc_err": round(se - true_soc, 4),
                        "extrap": r["extrap"], "I_meas_A": meas,
                        "I_oracle_A": round(i_true, 3),
                        "I_ekf_A": round(i_ekf, 3)})
            n += 1
        print(f"  {cell:<20} {n:>5}행  EKF 추정점 {len(est)}")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"\n  {args.out}   {len(out):,}행   제외 {dict(drop)}")

    se = np.array([r["soc_err"] for r in out])
    im = np.array([r["I_meas_A"] for r in out])
    io_ = np.array([r["I_oracle_A"] for r in out])
    ie = np.array([r["I_ekf_A"] for r in out])
    rm = lambda p, m=None: float(np.sqrt(np.mean((p - im)[m if m is not None
                                                          else slice(None)] ** 2)))
    print(f"\n  SOC 오차: 중앙 {np.median(se):+.4f}  |중앙| {np.median(np.abs(se)):.4f}  "
          f"90% {np.percentile(np.abs(se),90):.4f}")
    print(f"\n  {'':<22} {'RMSE':>9} {'5A+ 낙관':>10}")
    for nm, p in (("오라클 SOC", io_), ("EKF SOC", ie)):
        opt = float(np.mean(np.abs(p) - np.abs(im) > 5.0) * 100)
        print(f"  {nm:<22} {rm(p):>8.2f}A {opt:>9.1f}%")
    print(f"\n  {'SOC 구간':<14} {'n':>6} {'|SOC 오차|':>10} {'오라클':>9} {'EKF':>9}")
    st = np.array([r["SOC_true"] for r in out])
    for lo, hi in ((0.02, 0.3), (0.3, 0.4), (0.4, 0.6), (0.6, 1.01)):
        m = (st >= lo) & (st < hi)
        if m.sum() < 30:
            continue
        print(f"  {f'{lo:.2f}~{hi:.2f}':<14} {m.sum():>6,} "
              f"{np.median(np.abs(se[m])):>10.4f} {rm(io_,m):>8.2f}A {rm(ie,m):>8.2f}A")


if __name__ == "__main__":
    main()
