"""Invert the reference LSTM to amperes: bisect the current until V hits the floor.

WHY THIS NEEDS A SEARCH AND THE ECM DOES NOT
    The ECM arm has R_eff in closed form, so V_pre + I*R_eff = V_min is solved
    directly. The LSTM is a map from a window of (SOC, T, I) to a voltage; there
    is no expression to rearrange. The only way to ask it "which current reaches
    the floor" is to propose currents and watch the voltage - which is the
    binary search the reference paper's SOP method already prescribes, so this
    is the model's own deployment procedure rather than something imposed on it.

THE SEARCH IS SOUND BECAUSE THE SIGN WAS CHECKED FIRST
    A bisection needs monotonicity. a13_psweep measured dV/ds on every cell and
    variant: the sign is correct in 100 % of pulses for all twelve combinations
    once the input is current rather than power. Had that check failed, the
    search would have converged to nonsense and the resulting amperes would have
    looked like a result.

THE PROPOSED CURRENT MOVES THE SOC CHANNEL TOO
    Drawing s times the current depletes s times as fast, so scaling only the
    current channel would hand the model a window whose SOC slope contradicts
    its current. Both move together, from the pre-pulse SOC forward.

THE BASE PULSE IS THE HIGHEST MEASURED RATE
    Scaling starts from rank 3 (~24 A) rather than rank 0 (~3 A), so reaching a
    90 A answer is a factor of four rather than thirty. The extrapolation is the
    same one the label makes and is reported the same way.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_a13 import RUNS_SOH, load_model, pulse_index  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL = os.path.join(HERE, "sop_label_measured.csv")
OUT = os.path.join(HERE, "sop_amps_eval_ai.csv")
S_LO, S_HI = 0.15, 8.0


def build(cols, feats, ctxa, SOH, fk, jobs, W, L):
    """Base windows for a batch. Returns X0 (unscaled), C, A, k0 (pulse start)."""
    X0 = np.empty((len(jobs), W, 3), np.float32)
    C = np.empty((len(jobs), L, 3), np.float32)
    A = np.empty(len(jobs), np.float32)
    k0 = np.empty(len(jobs), np.int64)
    for q, (j, a) in enumerate(jobs):
        w0 = j - W + 1
        X0[q] = feats[w0:j + 1]
        C[q] = ctxa[w0 - L:w0]
        A[q] = SOH[j]
        k0[q] = max(a, w0) - w0
    return X0, C, A, k0


def apply_scale(X0, k0, Ifull, jobs, W, dem, s):
    """Scale the demand channel and advance SOC with the scaled current."""
    X = X0.copy()
    for q, (j, a) in enumerate(jobs):
        k = int(k0[q])
        if k > W - 1:
            continue
        X[q, k:, dem] *= s[q]
        soc0 = X0[q, k - 1, 0] if k >= 1 else X0[q, k, 0]
        X[q, k:, 0] = soc0 + (X0[q, k:, 0] - soc0) * s[q]
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--tag", default="i")
    ap.add_argument("--variants", default="M1,M2")
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--iters", type=int, default=22)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    lab = [r for r in csv.DictReader(open(args.label, encoding="utf-8"))
           if float(r["tau_s"]) == 10.0]
    bycell = collections.defaultdict(list)
    for r in lab:
        bycell[r["cell"]].append(r)

    rows, drop = [], collections.Counter()
    for cell in sorted(bycell):
        idx, cols = pulse_index(cell)
        valid = cols["valid"].astype(bool)
        jobs, meta = [], []
        for r in bycell[cell]:
            hit = idx.get((int(r["cycle"]), int(r["soc_group"]), "3", "discharge"))
            if hit is None:
                drop["rank3 펄스 없음"] += 1
                continue
            base, a, ks, end, ip = hit
            if 10.0 not in ks:
                drop["10s 지평 없음"] += 1
                continue
            j = base + a + ks[10.0]
            if j - 400 + 1 < base or j >= end or not valid[j - 400:j + 1].all():
                drop["이력 부족"] += 1
                continue
            jobs.append((j, base + a)); meta.append((r, ip))
        if not jobs:
            continue
        per_var = {}
        for var in args.variants.split(","):
            ck_path = os.path.join(RUNS_SOH, f"{args.tag}_{cell}_{var}.pt")
            if not os.path.exists(ck_path):
                continue
            model, ck = load_model(ck_path, dev)
            W = int(ck["window"]); L = int(ck["ctx_len"])
            fk = tuple(ck.get("feat_keys") or ("SOC", "T", "P"))
            dem = fk.index("P") if "P" in fk else fk.index("I")
            sc, scc = ck["sc"], ck["sc_ctx"]
            lo = sc["lo"].cpu().numpy(); rng = sc["rng"].cpu().numpy()
            vlo = float(sc["vlo"]); vrng = float(sc["vrng"])
            need_ctx = bool(model.use_z)
            clo = scc["lo"].cpu().numpy() if need_ctx else None
            crng = scc["rng"].cpu().numpy() if need_ctx else None
            feats = np.stack([cols[k] for k in fk], 1).astype(np.float32)
            ctxa = np.stack([cols[k] for k in ("V", "I", "T")], 1).astype(np.float32)
            SOH = cols["SOH"].astype(np.float32)
            Ifull = cols["I"].astype(np.float32)
            got = np.empty(len(jobs), np.float32)
            with torch.no_grad():
                for st in range(0, len(jobs), args.chunk):
                    blk = jobs[st:st + args.chunk]
                    X0, C, A, k0 = build(cols, feats, ctxa, SOH, fk, blk, W, L)
                    c = (torch.from_numpy((C - clo) / crng).to(dev)
                         if need_ctx else None)
                    aux = torch.from_numpy(A).to(dev)
                    lo_s = np.full(len(blk), S_LO, np.float32)
                    hi_s = np.full(len(blk), S_HI, np.float32)
                    for _ in range(args.iters):
                        mid = (lo_s + hi_s) / 2
                        X = apply_scale(X0, k0, Ifull, blk, W, dem, mid)
                        x = torch.from_numpy((X - lo) / rng).to(dev)
                        p = model(x, ctx=c,
                                  soh=aux if model.use_soh else None)
                        v = p.reshape(-1).cpu().numpy() * vrng + vlo
                        below = v < 2.5              # too much current
                        hi_s = np.where(below, mid, hi_s)
                        lo_s = np.where(below, lo_s, mid)
                    got[st:st + len(blk)] = (lo_s + hi_s) / 2
            per_var[var] = got
            del model
            if dev == "cuda":
                torch.cuda.empty_cache()
        for q, (r, ip) in enumerate(meta):
            d = {k: r[k] for k in ("cell", "cycle", "SOH", "SOC", "extrap")}
            d["I_meas_A"] = float(r["I_star_lin4_A"]); d["I_base_A"] = round(ip, 3)
            for var, g in per_var.items():
                s = float(g[q])
                sat = s <= S_LO * 1.02 or s >= S_HI * 0.98
                d[f"I_{var}_A"] = round(ip * s, 3)
                d[f"s_{var}"] = round(s, 4)
                d[f"sat_{var}"] = int(sat)
            rows.append(d)
        print(f"  {cell:<20} {len(jobs):>5}개 펄스  {list(per_var)}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n  {args.out}   {len(rows):,}행   제외 {dict(drop)}")

    ex = np.array([float(r["extrap"]) for r in rows])
    im = np.array([r["I_meas_A"] for r in rows])
    print(f"\n  {'변형':<6} {'포화':>7} {'RMSE':>9} {'편향':>9}   "
          f"{'<=1.0':>9} {'1.0~1.5':>9} {'1.5~2.5':>9} {'>2.5':>9}")
    for var in args.variants.split(","):
        k = f"I_{var}_A"
        if k not in rows[0]:
            continue
        p = np.array([r[k] for r in rows]); sat = np.array([r[f"sat_{var}"] for r in rows])
        e = p - im
        ok = sat == 0
        line = f"  {var:<6} {sat.mean()*100:>6.1f}% " \
               f"{np.sqrt(np.mean(e[ok]**2)):>8.2f}A {np.median(e[ok]):>+8.2f}A   "
        for lo_, hi_ in ((0, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 1e9)):
            m = ok & (ex > lo_) & (ex <= hi_)
            line += f"{np.sqrt(np.mean(e[m]**2)) if m.sum()>20 else float('nan'):>8.2f}A "
        print(line)
    print("  (포화 = 이분 탐색이 경계에 붙은 행, RMSE 에서 제외)")


if __name__ == "__main__":
    main()
