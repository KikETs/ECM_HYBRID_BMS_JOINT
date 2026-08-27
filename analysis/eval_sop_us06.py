"""외부 셀에서, 자기 주행으로 뽑은 특징으로, 측정 SOP 를 맞힌다.

이 프로젝트에서 하이브리드가 받는 가장 바깥의 시험이다. 트림은 UYPYDJ 여섯 셀에서
학습됐고, 특징은 RPCWBY Test#2 의 US06 주행에서 새로 계산되며, 라벨은 저자들이
직접 뽑은 SOP 다. 사이클러도 실험실도 셀도 다르다.

채점 대상 행
    18.6 절과 같다 - |SOP| <= 76.5 W 인 전압 제한 행만. 그 위는 30 A 상한이 답을
    정했으므로 전압 모델을 시험하지 못한다.

집계
    19.7 절의 결론을 그대로 쓴다. 12 블록은 특성화 직전 2 h 의 주행이고, max 는
    그 위의 러닝 max 다 - 차량에서 64 B 상태로 구현된다.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecm_pool import surfaces                    # noqa: E402
from sop_trim import TrimLinear, TrimMLP, decode  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SOP = os.path.join(HERE, "rpcwby_sop_summary.csv")
RES = os.path.join(HERE, "rpcwby_resistance.csv")
FEAT = os.path.join(HERE, "cache", "us06")

CELLS = {2: "RPC_US06", 1: "RPC_CC"}
SHEETS = {2: "Test#2", 1: "Test#1"}
V_FLOOR, I_CLAMP = 2.55, 30.0
BOUND = V_FLOOR * I_CLAMP
V_CEIL, I_CLAMP_CHG = 4.15, 15.0
# 충전은 P 만으로 두 제한이 갈리지 않는다. 전류 제한이면 V = P/15 가 충전 중
# 단자전압이어야 하는데, 45 W 아래에서는 그것이 3.0 V 미만이라 불가능하다.
BOUND_CHG = 45.0
TAU = 10.0


def soh_map(cell):
    rows = [r for r in csv.DictReader(open(RES, encoding="utf-8"))
            if r["cell"] == cell]
    a = np.array(sorted({(float(r["cycle"]), float(r["SOH"])) for r in rows}))
    return a[:, 0], a[:, 1]


def load_trim(holdout, runs):
    ck = torch.load(os.path.join(HERE, runs, f"model_A3_{holdout}.pt"),
                    map_location="cpu", weights_only=False)
    models = []
    for st in ck["seeds"]:
        cls = TrimLinear if st["cls"] == "TrimLinear" else TrimMLP
        m = cls(st["n_in"]); m.load_state_dict(st["model"]); m.eval()
        models.append((m, st["mu"], st["sd"]))
    return models


def k_of(models, X, agg):
    """X: (blocks, 12) -> (k_f, k_s), 시드 평균 후 블록 집계."""
    kf, ks = [], []
    for m, mu, sd in models:
        x = torch.from_numpy(((X - mu) / sd).clip(-4, 4).astype(np.float32))
        with torch.no_grad():
            a, b = decode(m(x))
        kf.append(a.numpy()); ks.append(b.numpy())
    kf = np.mean(kf, 0); ks = np.mean(ks, 0)     # (blocks,)
    f = {"last": lambda v: float(v[-1]), "max": lambda v: float(np.max(v)),
         "median": lambda v: float(np.median(v))}[agg]
    return f(kf), f(ks), kf, ks


def r_eff(surf, soc, soh, I, T, kf, ks):
    th = surf.theta(soc, soh, I, T)
    if not bool(np.atleast_1d(th["in_hull"])[0]):
        return np.nan
    R0 = float(th["R0"][0]); R1 = float(th["R1"][0]); R2 = float(th["R2"][0])
    t1 = float(th["tau1"][0]); t2 = float(th["tau2"][0])
    return (kf * (R0 + R1 * (1 - np.exp(-TAU / t1)))
            + ks * R2 * (1 - np.exp(-TAU / t2)))


def solve_I(surf, soc, soh, v_pre, T, kf, ks, iters=24, charge=False):
    I = 5.0 if charge else -15.0
    lim = V_CEIL if charge else V_FLOOR
    for _ in range(iters):
        R = r_eff(surf, soc, soh, I, T, kf, ks)
        if not np.isfinite(R) or R <= 0:
            return np.nan
        nxt = float(np.clip((lim - v_pre) / R, 0.05, 400.0) if charge
                    else np.clip((lim - v_pre) / R, -400.0, -0.1))
        if abs(nxt - I) < 1e-3:
            return nxt
        I = 0.5 * I + 0.5 * nxt
    return I


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["discharge", "charge"],
                    default="discharge")
    ap.add_argument("--test", type=int, default=2, choices=[1, 2],
                    help="2 = US06 drive excitation, 1 = constant current")
    ap.add_argument("--holdouts",
                    default="CC,CC_CELL2,BOOST,BOOST_REST,BOOST_NEGPULSE,"
                            "BOOST_NEGPULSE_1S")
    ap.add_argument("--runs", default="runs_trim")
    ap.add_argument("--runs-chg", default="runs_trim_chg")
    ap.add_argument("--agg", default="max", choices=["last", "max", "median"])
    ap.add_argument("--lam-t10", type=float, default=1.0,
                    help="19.9 절의 지평별 여유. 1.0 이면 여유 없이 날것.")
    ap.add_argument("--out", default=os.path.join(HERE, "sop_us06_eval.csv"))
    a = ap.parse_args()

    cell, sheet = CELLS[a.test], SHEETS[a.test]
    cyc_a, soh_a = soh_map(cell)
    chg = a.direction == "charge"
    col = "SOP_char" if chg else "SOP_disch"
    lim_v = V_CEIL if chg else V_FLOOR
    rows = [r for r in csv.DictReader(open(SOP, encoding="utf-8"))
            if r["sheet"] == sheet and r[col] not in ("", "nan")
            and 0.05 < abs(float(r[col])) <= (BOUND_CHG if chg else BOUND)]
    print(f"  {sheet} ({cell}) 전압 제한 행 {len(rows)}")

    out = []
    import collections
    drop = collections.Counter()
    for h in a.holdouts.split(","):
        fp = os.path.join(FEAT, f"t{a.test}_feats_{h}.npz")
        if not os.path.exists(fp):
            print(f"  특징 없음: {h} — 건너뜀")
            continue
        z = np.load(fp)
        F = {int(c): z["X"][i] for i, c in enumerate(z["cycles"])}
        models = load_trim(h, a.runs if not chg else a.runs_chg)
        sd = surfaces(h)[1 if chg else 0]
        for r in rows:
            cy = float(r["cycle"])
            if int(cy) not in F:
                drop["주행 특징 없음"] += 1
                continue
            kf, ks, kfa, ksa = k_of(models, F[int(cy)], a.agg)
            soc = float(r["SOC"]); T = float(r["temp_C"])
            soh = float(np.interp(cy, cyc_a, soh_a))
            v, _ = sd.ocv(soc, soh)
            M, _ = sd.hyst_M(soc, soh)
            v_pre = float(np.atleast_1d(v)[0]) + (M if chg else -M)
            i0 = solve_I(sd, soc, soh, v_pre, T, 1.0, 1.0, charge=chg)
            i3 = solve_I(sd, soc, soh, v_pre, T, kf, ks, charge=chg)
            if not (np.isfinite(i0) and np.isfinite(i3)):
                drop["hull 밖 / 미수렴"] += 1
                continue
            out.append(dict(holdout=h, cycle=cy, SOH=round(soh, 4), SOC=soc,
                            T_C=T,
                            I_meas_A=round((1 if chg else -1)
                                           * abs(float(r[col])) / lim_v, 3),
                            I_A0_A=round(i0, 3), I_A3_A=round(a.lam_t10 * i3, 3),
                            k_f=round(kf, 4), k_s=round(ks, 4),
                            k_f_span=round(float(kfa.max() - kfa.min()), 4)))
    if not out:
        sys.exit("  채점 가능한 행이 없다")
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    im = np.abs([r["I_meas_A"] for r in out])
    i0 = np.abs([r["I_A0_A"] for r in out]); i3 = np.abs([r["I_A3_A"] for r in out])
    T = np.array([r["T_C"] for r in out]); soh = np.array([r["SOH"] for r in out])
    kf = np.array([r["k_f"] for r in out])
    print(f"  제외 {dict(drop)}")
    print(f"  {a.out}  {len(out)}행   k_f {kf.min():.3f}~{kf.max():.3f} "
          f"(중앙 {np.median(kf):.3f})\n")
    print(f"  {'':<16}{'n':>5}{'초과율':>8}{'최악':>9}{'가용률':>8}{'RMSE':>8}")

    def line(tag, m, p):
        if m.sum() < 3:
            return
        o = p[m] - im[m]
        print(f"  {tag:<16}{int(m.sum()):>5}{np.mean(o > 0)*100:>7.1f}%"
              f"{max(o.max(), 0):>8.2f}A{np.median(p[m]/im[m]):>8.3f}"
              f"{np.sqrt(np.mean(o**2)):>7.2f}A")

    all_m = np.ones(len(im), bool)
    line("ECM", all_m, i0)
    line("하이브리드", all_m, i3)
    for t in (10.0, 25.0):
        line(f"  ECM {t:.0f}C", T == t, i0)
        line(f"  하이브 {t:.0f}C", T == t, i3)
    for lo, hi in ((0.90, 1.01), (0.80, 0.90)):
        line(f"  하이브 SOH{lo:.2f}", (soh >= lo) & (soh < hi), i3)


if __name__ == "__main__":
    main()
