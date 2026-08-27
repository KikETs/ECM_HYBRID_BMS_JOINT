"""Leave-one-cell-out pooled ECM surface — the nominal the trim corrects.

WHY THIS FILE EXISTS AT ALL
    `ECMSurface(cell)` filters `uypydj_ecm.csv` by cell name. Instantiating it
    for the held-out cell imports that cell's own R1 trajectory - the very
    quantity that spans 5.58x across cells (findings.md section 4.3.1) and the
    quantity the trim is supposed to estimate. Every held-out number computed
    that way would be leaked. The nominal must be built from the training cells
    only.

RESPONSE-SPACE POOLING, NOT PARAMETER-SPACE
    Averaging fitted (R2, tau2) pairs across cells is wrong because the response
    is nonlinear in tau2: two cells with (10 mOhm, 4 s) and (10 mOhm, 16 s) have
    the same mean parameters as (10, 10) and (10, 10), but a visibly different
    10 s response. Pool what is measured - the response - and re-derive the
    parameters from it.

    tau1 makes this exactly solvable rather than a fit. Measured tau1 median is
    0.244 s (99.1 % below 1 s), so at BOTH horizons that matter

        e1(2 s)  = exp(-2/0.244)  = 2.7e-4
        e1(10 s) = exp(-10/0.244) = 1e-18

    i.e. the fast branch is fully developed at 2 s and R0 and R1 enter as one
    number R_fast = R0 + R1. With tau2 taken as the cells' median (it varies far
    less than the resistances), the two horizon responses give two equations in
    two unknowns:

        d2  = R_fast + R_slow * a,   a = 1 - exp(-2/tau2)
        d10 = R_fast + R_slow * b,   b = 1 - exp(-10/tau2)
     => R_slow = (d10 - d2) / (b - a),   R_fast = d2 - R_slow * a

    Exactly determined. No optimiser, no initial guess, nothing to converge.

THE BUILD GATE
    A fresh cell should need no correction, so k_f and k_s computed for a fresh
    HELD-OUT cell against this pooled surface must come out at 1.00. If they do
    not, the pooling itself is biased and every later number inherits that bias.
    `verify_unit_prior` asserts |k - 1| <= 0.03 at SOH >= 0.97 and is meant to be
    run before any training, not after a disappointing result.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecm_surface import ECMSurface  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ECM_CSV = os.path.join(HERE, "uypydj_ecm.csv")
OCV_CSV = os.path.join(HERE, "uypydj_ocv.csv")
POOL_DIR = os.path.join(HERE, "cache", "pool")

TAU_A, TAU_B = 2.0, 10.0          # the two horizons the SOP work uses

# `rate_rank` is UYPYDJ's own label for its four stepped rates. RPCWBY's SOP
# search applies a continuum from 3 to 30 A and has no such label, so pooling the
# two together needs a key both can speak: the current itself. ECMSurface already
# treats rank as a rung on a current ladder - it interpolates across ranks at |I|
# using each rank's median current - so rewriting the label changes the grouping
# and nothing downstream. Edges keep 30 A and 34 A in one bin; within
# Butler-Volmer curvature they are the same operating point.
I_EDGES = (2.0, 7.0, 16.0, 26.0, 40.0)


def cur_bin(i):
    return int(np.clip(np.searchsorted(I_EDGES, abs(float(i))) - 1,
                       0, len(I_EDGES) - 2))


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def response_pool(rows, cells, key_mode="rank"):
    """Re-derive (R_fast, R_slow) from the pooled 2 s / 10 s response.

    Grouped by (direction, rate_rank, SOC bin, SOH bin) so that every pooled
    node is an average over CELLS at a comparable operating point, never an
    average over operating points.
    """
    rk = (lambda r: r["rate_rank"]) if key_mode == "rank" else \
         (lambda r: str(cur_bin(r["I_A"])))
    key = lambda r: (r["direction"], rk(r),
                     round(float(r["SOC"]) / 0.05) * 0.05,
                     round(float(r["SOH"]) / 0.02) * 0.02)
    grp = collections.defaultdict(list)
    for r in rows:
        if r["cell"] not in cells:
            continue
        grp[key(r)].append(r)

    out = []
    for (direction, rank, soc, soh), g in grp.items():
        if len({x["cell"] for x in g}) < 2:
            continue                      # a "pool" of one cell is not a pool
        f = lambda k: np.array([float(x[k]) for x in g])
        r0, r1, r2 = f("R0_mOhm"), f("R1_mOhm"), f("R2_mOhm")
        t1, t2 = f("tau1_s"), f("tau2_s")
        # Per-cell response at the two horizons, then pool the RESPONSE.
        d2 = r0 + r1 * (1 - np.exp(-TAU_A / t1)) + r2 * (1 - np.exp(-TAU_A / t2))
        d10 = r0 + r1 * (1 - np.exp(-TAU_B / t1)) + r2 * (1 - np.exp(-TAU_B / t2))
        D2, D10 = float(np.median(d2)), float(np.median(d10))
        T2 = float(np.median(t2))
        a = 1 - np.exp(-TAU_A / T2)
        b = 1 - np.exp(-TAU_B / T2)
        if b - a < 1e-6:
            continue
        R_slow = (D10 - D2) / (b - a)
        R_fast = D2 - R_slow * a
        if not (np.isfinite(R_fast) and np.isfinite(R_slow)):
            continue
        # R_fast is R0+R1 and is not separable at these horizons; the split is
        # carried only so ECMSurface's schema stays valid. tau1 is the pooled
        # median, which never enters a horizon >= 2 s.
        out.append({
            "cell": "POOL", "cycle": 0,
            "SOH": round(soh, 5), "CAP_Ah": float(np.median(f("CAP_Ah"))),
            "SOC": round(soc, 4), "direction": direction, "rate_rank": rank,
            "I_A": round(float(np.median(f("I_A"))), 3),
            "V_pre_V": round(float(np.median(f("V_pre_V"))), 5),
            "rest_before_s": 99999.0, "n_points": int(np.median(f("n_points"))),
            "R0_mOhm": round(max(R_fast, 1e-3), 4),
            "R1_mOhm": 1e-3,
            "tau1_s": round(float(np.median(t1)), 4),
            "R2_mOhm": round(max(R_slow, 1e-3), 4),
            "tau2_s": round(T2, 4),
            "fit_rmse_mV": 0.0,
        })
    return out


def pool_ocv(rows, cells):
    """OCV pooled on the common (SOC, SOH) grid. Linear in the response, so a
    median is already response-space pooling."""
    key = lambda r: (round(float(r["SOC"]) / 0.01) * 0.01,
                     round(float(r["SOH"]) / 0.02) * 0.02)
    grp = collections.defaultdict(list)
    for r in rows:
        if r["cell"] not in cells:
            continue
        grp[key(r)].append(r)
    out = []
    for (soc, soh), g in grp.items():
        if len({x["cell"] for x in g}) < 2:
            continue
        f = lambda k: np.array([float(x[k]) for x in g])
        out.append({
            "cell": "POOL", "cycle": 0, "SOH": round(soh, 5),
            "SOC": round(soc, 4),
            "OCV_V": round(float(np.median(f("OCV_V"))), 5),
            "V_dis_V": round(float(np.median(f("V_dis_V"))), 5),
            "V_chg_V": round(float(np.median(f("V_chg_V"))), 5),
            "hyst_mV": round(float(np.median(f("hyst_mV"))), 2),
        })
    return out


def align_extra(ex, base):
    """Remove the lab-to-lab offset before pooling, fitted on TRAINING cells only.

    The two datasets agree to +0.5 % at 10 s but +2.5 % at 2 s (section 14.3),
    and the two-horizon reduction amplifies that difference into about 5 % on the
    slow branch - enough to fail the build gate on five of six folds. Pooling an
    uncorrected offset puts a cycler difference into a surface that is supposed
    to describe a cell.

    The correction is two scalars, one per horizon, taken as the median ratio
    over matched (SOC, current, SOH) bins. They are fitted against the fold's
    TRAINING cells, so the held-out cell never enters. Anything the two labs
    disagree about that is NOT a constant per horizon survives, deliberately -
    a richer correction fitted on six cells would start absorbing real
    cell-to-cell spread, which is the quantity being estimated.
    """
    # FRESH cells only. The two datasets age under different protocols - fifteen
    # minute fast charge against 1C CC discharge - so at equal SOH their
    # resistances may legitimately differ, and matching across all SOH reads that
    # as a lab offset. Fitting over the whole range gave x0.90, against the x0.976
    # the fresh-cell comparison in 14.3 implies; the difference was aging, not
    # cyclers. Only where both are fresh is the offset identifiable.
    SOH_FRESH = 0.95
    key = lambda r: (round(float(r["SOC"]) / 0.1) * 0.1, cur_bin(r["I_A"]))

    def dd(r):
        R0 = float(r["R0_mOhm"]); R1 = float(r["R1_mOhm"]); R2 = float(r["R2_mOhm"])
        t1 = float(r["tau1_s"]); t2 = float(r["tau2_s"])
        return (R0 + R1 * (1 - np.exp(-TAU_A / t1)) + R2 * (1 - np.exp(-TAU_A / t2)),
                R0 + R1 * (1 - np.exp(-TAU_B / t1)) + R2 * (1 - np.exp(-TAU_B / t2)))

    bb = collections.defaultdict(list)
    for r in base:
        if r["direction"] != "discharge" or float(r["SOH"]) < SOH_FRESH:
            continue
        bb[key(r)].append(dd(r))
    r2, r10 = [], []
    for r in ex:
        if r["direction"] != "discharge" or float(r["SOH"]) < SOH_FRESH:
            continue
        k = key(r)
        if k not in bb:
            continue
        d2, d10 = dd(r)
        b2 = np.median([x[0] for x in bb[k]]); b10 = np.median([x[1] for x in bb[k]])
        if d2 > 0 and d10 > 0:
            r2.append(b2 / d2); r10.append(b10 / d10)
    if len(r2) < 30:
        return ex
    a2, a10 = float(np.median(r2)), float(np.median(r10))
    out = []
    for r in ex:
        d2, d10 = dd(r)
        t2 = float(r["tau2_s"])
        a = 1 - np.exp(-TAU_A / t2); b = 1 - np.exp(-TAU_B / t2)
        n2, n10 = a2 * d2, a10 * d10
        rs = (n10 - n2) / (b - a); rf = n2 - rs * a
        if not (np.isfinite(rf) and np.isfinite(rs)) or rf <= 0 or rs <= 0:
            continue
        q = dict(r); q["R0_mOhm"] = round(rf, 4); q["R2_mOhm"] = round(rs, 4)
        out.append(q)
    print(f"    정렬: 2s x{a2:.4f}, 10s x{a10:.4f}  (짝지은 칸 {len(r2)}), "
          f"{len(out)}/{len(ex)}행 유지")
    return out


def build(holdout, outdir=POOL_DIR, ecm_csv=ECM_CSV, ocv_csv=OCV_CSV,
          key_mode="rank", extra_csv=None, align=False):
    e, o = _rows(ecm_csv), _rows(ocv_csv)
    cells = sorted({r["cell"] for r in e})
    if holdout not in cells:
        raise ValueError(f"unknown cell {holdout}; have {cells}")
    keep = [c for c in cells if c != holdout]
    if extra_csv and align:
        ex = _rows(extra_csv)
        ex = align_extra(ex, [r for r in e if r["cell"] in keep])
        e = e + ex
        keep = keep + sorted({r["cell"] for r in ex})
    elif extra_csv:
        # Extra cells never contain the holdout, so they are always kept. They
        # join the RESISTANCE pool only - the OCV pool stays UYPYDJ-only because
        # RPCWBY has no OCV characterisation in this extraction.
        ex = _rows(extra_csv)
        e = e + ex
        keep = keep + sorted({r["cell"] for r in ex})
    os.makedirs(outdir, exist_ok=True)
    pe = os.path.join(outdir, f"ecm_pool_{holdout}.csv")
    po = os.path.join(outdir, f"ocv_pool_{holdout}.csv")
    er, orr = response_pool(e, keep, key_mode), pool_ocv(o, keep)
    for path, rows_ in ((pe, er), (po, orr)):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_[0].keys()))
            w.writeheader(); w.writerows(rows_)
    return pe, po, len(er), len(orr)


def surfaces(holdout, outdir=POOL_DIR, key_mode="rank", extra_csv=None,
             align=False):
    pe = os.path.join(outdir, f"ecm_pool_{holdout}.csv")
    po = os.path.join(outdir, f"ocv_pool_{holdout}.csv")
    if not (os.path.exists(pe) and os.path.exists(po)):
        build(holdout, outdir, key_mode=key_mode, extra_csv=extra_csv,
              align=align)
    return (ECMSurface("POOL", "discharge", ecm_csv=pe, ocv_csv=po),
            ECMSurface("POOL", "charge", ecm_csv=pe, ocv_csv=po))


def verify_unit_prior(holdout, soh_min=0.97, tol=0.03, outdir=POOL_DIR,
                      key_mode="rank", extra_csv=None, align=False):
    """A fresh held-out cell must need no correction against the pool.

    BOTH SIDES ARE REDUCED THE SAME WAY. The first version of this gate compared
    the held-out cell's FITTED R2 against the pool's RE-DERIVED R_slow, which are
    different quantities - the first comes from a 2RC fit with that cell's own
    tau2, the second from a two-horizon solve at the pooled tau2. It reported a
    systematic k_s of 0.96-0.98 on five of six cells, which was a property of the
    comparison, not of the pooling. Here the held-out cell is passed through the
    identical two-horizon reduction, at the POOLED tau2, so the gate measures
    what it claims to.
    """
    sd, _ = surfaces(holdout, outdir, key_mode, extra_csv, align)
    own = [r for r in _rows(ECM_CSV)
           if r["cell"] == holdout and r["direction"] == "discharge"
           and float(r["SOH"]) >= soh_min and r["rate_rank"] in ("2", "3")]
    kf, ks = [], []
    for r in own:
        soc, soh, I = float(r["SOC"]), float(r["SOH"]), abs(float(r["I_A"]))
        if not 0.29 <= soc <= 0.95:
            continue
        th = sd.theta(soc, soh, -I)
        if not bool(th["in_hull"][0]):
            continue
        R0 = float(r["R0_mOhm"]); R1 = float(r["R1_mOhm"]); R2 = float(r["R2_mOhm"])
        t1 = float(r["tau1_s"]); t2 = float(r["tau2_s"])
        # The cell's own measured response at the two horizons ...
        d2 = R0 + R1 * (1 - np.exp(-TAU_A / t1)) + R2 * (1 - np.exp(-TAU_A / t2))
        d10 = R0 + R1 * (1 - np.exp(-TAU_B / t1)) + R2 * (1 - np.exp(-TAU_B / t2))
        # ... reduced at the POOLED tau2, exactly as the pool itself was.
        T2 = float(th["tau2"][0])
        a = 1 - np.exp(-TAU_A / T2)
        b = 1 - np.exp(-TAU_B / T2)
        if b - a < 1e-6:
            continue
        own_slow = (d10 - d2) / (b - a)
        own_fast = d2 - own_slow * a
        pf = (float(th["R0"][0]) + float(th["R1"][0])) * 1000
        ps = float(th["R2"][0]) * 1000
        if pf > 0 and ps > 0 and own_fast > 0 and own_slow > 0:
            kf.append(own_fast / pf); ks.append(own_slow / ps)
    if len(kf) < 5:
        return {"n": len(kf), "ok": False, "why": "표본 부족"}
    kf, ks = np.array(kf), np.array(ks)
    ok = abs(np.median(kf) - 1) <= tol and abs(np.median(ks) - 1) <= tol
    return {"n": len(kf), "k_f": float(np.median(kf)), "k_s": float(np.median(ks)),
            "k_f_iqr": float(np.percentile(kf, 75) - np.percentile(kf, 25)),
            "k_s_iqr": float(np.percentile(ks, 75) - np.percentile(ks, 25)),
            "ok": bool(ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=POOL_DIR)
    ap.add_argument("--key", default="rank", choices=["rank", "current"])
    ap.add_argument("--align", action="store_true",
                    help="remove the lab offset of the extra cells first")
    ap.add_argument("--extra-csv", default=None,
                    help="additional cells for the resistance pool")
    args = ap.parse_args()
    cells = sorted({r["cell"] for r in _rows(ECM_CSV)})
    print(f"{'홀드아웃':<20} {'ECM행':>7} {'OCV행':>7} {'n':>5} "
          f"{'k_f':>7} {'k_s':>7} {'게이트':>7}")
    for c in cells:
        pe, po, ne, no = build(c, args.outdir, key_mode=args.key,
                               extra_csv=args.extra_csv, align=args.align)
        v = verify_unit_prior(c, outdir=args.outdir, key_mode=args.key,
                              extra_csv=args.extra_csv, align=args.align)
        if "k_f" in v:
            print(f"{c:<20} {ne:>7,} {no:>7,} {v['n']:>5} "
                  f"{v['k_f']:>7.3f} {v['k_s']:>7.3f} "
                  f"{'통과' if v['ok'] else '실패':>7}")
        else:
            print(f"{c:<20} {ne:>7,} {no:>7,} {v['n']:>5} {v['why']:>23}")


if __name__ == "__main__":
    main()
