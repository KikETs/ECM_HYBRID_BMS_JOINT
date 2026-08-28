"""Score the ECM and hybrid arms in AMPERES against the measured SOP label.

EVERY NUMBER THIS PROJECT HAS PRODUCED SO FAR IS IN MILLIVOLTS
    The arms are voltage models; SOP is the current at which the voltage reaches
    its floor. 44.8 mV of pulse error is not an answer to "how many amps", and
    the translation is not a constant - it is dV/dI, which is the resistance
    itself, so a model that is wrong about resistance is wrong twice.

THE INVERSION IS THE DEPLOYMENT QUERY
    V(tau) = V_pre + I * R_eff(I, SOC, SOH, tau) = V_min, solved for I. R_eff
    depends on I - the measured resistance falls about 0.7x from 2.6 A to 29.6 A
    - so the solve is a fixed point, not a division. Starting from the label and
    iterating converges in a few steps; starting from a constant would bias the
    answer toward whatever constant was chosen.

ALL ROWS ARE SCORED, STRATIFIED BY EXTRAPOLATION
    Restricting to the 8 % of rows where the label interpolates would select on
    the LABEL rather than on the model, and would keep only low SOC - the regime
    where SOP is smallest and easiest. The label's extrapolation error is instead
    measured directly (fit the low rates, predict a rate that was applied):
    1.17 A at 1.44x, 2.51 A at 2.87x, and consistently signed so that linear
    extrapolation UNDER-states I*. That bias is reported beside the results
    rather than removed, because removing it would mean trusting a correction
    fitted on the same four points.

THE MULTIPLIER IS ONE VALUE PER CHARACTERISATION
    k_f and k_s come from the preceding drive cycle, so every pulse of one
    characterisation shares them (verified: 0/50 and 0/35 characterisations show
    any spread after the per-pulse deduplication the evaluation uses). They can
    therefore be applied at SOC below 0.29, where the trim was never scored -
    flagged in the output as `below_train_soc`, because a cell-level correction
    being valid there is an assumption, not a measurement.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import glob
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecm_pool import surfaces  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL = os.path.join(HERE, "sop_label_measured.csv")
LABEL_CHG = os.path.join(HERE, "sop_label_charge.csv")
TRIM = os.path.join(HERE, "runs_trim")
OUT = os.path.join(HERE, "sop_amps_eval.csv")
SOC_TRAIN_MIN = 0.29
TAU_A, TAU_B = 2.0, 10.0
TAU2_REF = 8.0


def trim_k(cell, runs=None):
    """(cycle) -> (k_f, k_s), one value per characterisation.

    `runs` selects which trained trim to read. The charge direction needs its
    own: the discharge trim is fitted on discharge pulses only, and applying its
    multipliers to charge resistances has no basis - measured, it LOST to the
    uncorrected ECM on two of six cells.
    """
    z = np.load(_pred_path(runs, cell),
                allow_pickle=True)
    key = list(zip(z["cycle"].astype(int).tolist(),
                   np.round(z["SOC"].astype(float), 4).tolist(),
                   z["rank"].astype(str).tolist()))
    last = {}
    for i, k in enumerate(key):
        last[k] = i
    idx = np.array(sorted(last.values()))
    cy = z["cycle"].astype(int)[idx]
    out = {}
    for c in np.unique(cy):
        m = cy == c
        out[int(c)] = (float(np.median(z["k_f"][idx][m])),
                       float(np.median(z["k_s"][idx][m])))
    return out


def trim_spread(cell, runs=None):
    """(cycle) -> 12 블록에 걸친 k_f 의 상대 산포. 불확실성 대용치.

    한 특성화의 12 블록은 직전 2 h 주행의 서로 다른 이력 창이다. 물리 저항은
    하나이므로, 블록끼리 어긋나는 정도는 '이 상태에서 트림이 얼마나 확신하지
    못하는가' 다. 여유를 그 위에 얹을 수 있으면 확신하는 곳에서 출력을 덜 버린다.
    """
    z = np.load(_pred_path(runs, cell),
                allow_pickle=True)
    cy = z["cycle"].astype(int); kf = z["k_f"]
    out = {}
    for c in np.unique(cy):
        m = cy == c
        out[int(c)] = float((kf[m].max() - kf[m].min())
                            / max(np.median(kf[m]), 1e-6))
    return out


# 트림 run 디렉터리는 rung 이름을 파일명에 담는다 (pred_A3_*, pred_A8_*).
# 하드코딩하면 A8 을 평가하려고 파일을 복사하게 되므로 여기서 자동으로 찾는다.
TRIM_RUNG = None          # None 이면 디렉터리에 있는 것을 쓴다


def _pred_path(runs, cell):
    d = runs or TRIM
    if TRIM_RUNG:
        return os.path.join(d, f"pred_{TRIM_RUNG}_{cell}.npz")
    hit = sorted(glob.glob(os.path.join(d, f"pred_A*_{cell}.npz")))
    hit = [h for h in hit if not h.endswith(f"pred_A0_{cell}.npz")]
    if not hit:
        raise FileNotFoundError(
            f"{d} 에 pred_A*_{cell}.npz 가 없다 — 트림을 먼저 학습해야 한다")
    if len(hit) > 1:
        raise RuntimeError(f"{d} 에 rung 이 여럿이다 {hit} — --trim-rung 으로 "
                           f"고를 것")
    return hit[0]


def trim_k_agg(cell, runs=None, agg="last"):
    """(cycle) -> (k_f, k_s), 이력 스냅샷을 어떻게 모을지 고를 수 있게.

    WHY THERE IS ANYTHING TO AGGREGATE
        데이터셋은 펄스 하나를 12 개의 서로 다른 주행 이력 창과 짝지어 담는다
        (m_exc 가 다르다). 물리 저항은 하나지만 모델이 보는 이력은 12 가지이므로
        k 도 12 개 나온다. 배치되는 BMS 가 보게 될 것은 그중 하나 - 어느 것일지는
        모른다.

        trim_k 는 (cycle, SOC, rank) 마다 '마지막' 행만 남긴다. 그 행들은 모두
        특성화 끝 시점이라 k 가 한 값으로 수렴하고, 그래서 SOP 평가는 사이클당
        숫자 하나만 쓴다 - 조건성이 통째로 버려진다.

        안전을 원한다면 이력 불확실성 위에서 상위 분위수를 취하는 편이 맞다.
        저항을 크게 잡는 쪽이 SOP 를 낮게(보수적으로) 만든다.
    """
    z = np.load(_pred_path(runs, cell),
                allow_pickle=True)
    cy = z["cycle"].astype(int)
    kf_all = z["k_f"]; ks_all = z["k_s"]
    if agg == "last":
        return trim_k(cell, runs)
    fn = {"median": lambda v: float(np.median(v)),
          "q75": lambda v: float(np.percentile(v, 75)),
          "q90": lambda v: float(np.percentile(v, 90)),
          "max": lambda v: float(np.max(v))}[agg]
    out = {}
    for c in np.unique(cy):
        m = cy == c
        out[int(c)] = (fn(kf_all[m]), fn(ks_all[m]))
    return out


def trim_k_soc(cell, runs=None):
    """(cycle) -> (SOC 배열, k_f 배열, k_s 배열), SOC 를 살려서.

    WHY THE PER-CYCLE MEDIAN IS WRONG AND NOT MERELY COARSE
        trim_k 는 한 특성화 사이클의 모든 SOC 펄스에 걸쳐 k 의 중앙값을 낸다.
        그러면 트림이 배운 SOC 의존성이 통째로 사라진다. 그리고 분위수 손실로
        학습한 트림에서는 더 나쁘다 - 상위 분위수를 노린 k 들의 중앙값은 더 이상
        상위 분위수가 아니므로, 안전 여유가 집계 단계에서 깎여 나간다.
    """
    z = np.load(_pred_path(runs, cell),
                allow_pickle=True)
    key = list(zip(z["cycle"].astype(int).tolist(),
                   np.round(z["SOC"].astype(float), 4).tolist(),
                   z["rank"].astype(str).tolist()))
    last = {}
    for i, k in enumerate(key):
        last[k] = i
    idx = np.array(sorted(last.values()))
    cy = z["cycle"].astype(int)[idx]
    sc = z["SOC"].astype(float)[idx]
    kf = z["k_f"][idx]; ks = z["k_s"][idx]
    out = {}
    for c in np.unique(cy):
        m = cy == c
        socs = np.unique(np.round(sc[m], 3))
        a, b, d = [], [], []
        for s0 in socs:
            q = m & (np.round(sc, 3) == s0)
            a.append(s0); b.append(float(np.median(kf[q])))
            d.append(float(np.median(ks[q])))
        out[int(c)] = (np.array(a), np.array(b), np.array(d))
    return out


USE_DTAU = False          # main() 이 --interp 로 켠다


def r_eff(surf, soc, soh, I, tau, kf, ks):
    if USE_DTAU:
        # 등가저항을 rank 축에서 직접 보간한다 (ecm_surface.d_tau 참조).
        # 빠른/느린 가지를 나누는 두-지평 환원은 그대로 유지한다.
        D, ok = surf.d_tau(soc, soh, I, [TAU_A, TAU_B])
        if not bool(np.atleast_1d(ok)[0]):
            return np.nan
        d2, d10 = float(D[0, 0]), float(D[0, 1])
        a = 1 - np.exp(-TAU_A / TAU2_REF); b = 1 - np.exp(-TAU_B / TAU2_REF)
        R_slow = (d10 - d2) / (b - a)
        R_fast = d2 - R_slow * a
        return kf * R_fast + ks * R_slow * (1 - np.exp(-tau / TAU2_REF))
    th = surf.theta(soc, soh, I)
    if not bool(th["in_hull"][0]):
        return np.nan
    R0 = float(th["R0"][0]); R1 = float(th["R1"][0]); R2 = float(th["R2"][0])
    t1 = float(th["tau1"][0]); t2 = float(th["tau2"][0])
    return (kf * (R0 + R1 * (1 - np.exp(-tau / t1)))
            + ks * R2 * (1 - np.exp(-tau / t2)))


def solve_I(surf, soc, soh, v_pre, v_min, tau, kf, ks, I0=-25.0, iters=12,
            charge=False):
    """V_pre + I*R_eff(I) = V_limit, fixed point in I.

    Charge is the mirror: the current is positive and the limit is the ceiling,
    so only the sign of the clip changes. The surface itself is already
    direction-specific - ecm_pool builds a charge and a discharge one - so the
    resistances come from the right branch without any special casing here.
    """
    I = I0
    for _ in range(iters):
        R = r_eff(surf, soc, soh, I, tau, kf, ks)
        if not np.isfinite(R) or R <= 0:
            return np.nan
        I_new = (v_min - v_pre) / R
        if not np.isfinite(I_new):
            return np.nan
        I_new = float(np.clip(I_new, 0.1, 400.0) if charge
                      else np.clip(I_new, -400.0, -0.1))
        if abs(I_new - I) < 1e-3:
            return I_new
        I = 0.5 * I + 0.5 * I_new           # damped; R(I) is steep at low |I|
    return I


SOH_EST = None


SOC_EST = None


def soc_estimated(cell, cycle, soc_true):
    """SOC as the EKF would have it when this pulse happens.

    SOC_EST maps a cell to (cycles, soc_error).  The error is the adopted
    EKF's terminal SOC error on the drive run nearest at or before this
    characterisation, which is the state the filter carries into the pulse.
    Nearest-preceding, not interpolated: a filter does not average its
    future.
    """
    if SOC_EST is None:
        return soc_true
    key = f"{cell}_cycle"
    if key not in SOC_EST:
        return soc_true
    cc, ee = SOC_EST[key], SOC_EST[f"{cell}_err"]
    prior = cc <= cycle
    e = (float(ee[prior][int(np.argmax(cc[prior]))]) if prior.any()
         else float(ee[int(np.argmin(cc))]))
    return float(np.clip(soc_true + e, 0.02, 1.0))


def soh_estimated(cell, cycle, fallback):
    """직전 충전에서 추정한 SOH.  그 앞이 없으면 가장 이른 예측."""
    key = f"{cell}_cycle"
    if key not in SOH_EST:
        return fallback
    cc, pp = SOH_EST[key], SOH_EST[f"{cell}_pred"]
    prior = cc <= cycle
    if prior.any():
        return float(pp[prior][int(np.argmax(cc[prior]))])
    return float(pp[int(np.argmin(cc))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--interp", choices=["theta", "dtau"], default="theta",
                    help="theta = RC 파라미터를 보간한 뒤 지수(현행). "
                         "dtau = rank 마다 등가저항을 만들고 그것을 보간.")
    ap.add_argument("--trim-agg",
                    choices=["last", "median", "q75", "q90", "max"],
                    default="last",
                    help="한 특성화의 12개 이력 스냅샷을 어떻게 모을지. "
                         "상위 분위수는 저항을 크게 잡아 SOP 를 보수적으로 "
                         "만든다.")
    ap.add_argument("--label-fit", choices=["lin4", "lin2hi"],
                    default="lin4",
                    help="lin4 = least squares on all four HPPC rates "
                         "(the shipped label; its chord is pulled up by "
                         "the low rates, so it UNDERSTATES |I*| where R "
                         "falls with current). lin2hi = the two highest "
                         "rates, taken where SOP actually lives - "
                         "physically better, but it moves the label the "
                         "way that flatters the model, so a safety "
                         "margin should still be calibrated on lin4.")
    ap.add_argument("--trim-key", choices=["cycle", "cycle_soc"],
                    default="cycle",
                    help="cycle = one k per characterisation (the "
                         "shipped protocol); cycle_soc keeps the "
                         "trim's SOC dependence")
    ap.add_argument("--trim", default=None,
                    help="trim run directory; default is the shipped "
                         "symmetric one (runs_trim / runs_trim_chg)")
    ap.add_argument("--direction", default="discharge",
                    choices=["discharge", "charge"])
    ap.add_argument("--trim-rung", default=None,
                    help="트림 run 디렉터리 안의 rung 이름 (A3, A8, ...). "
                         "생략하면 디렉터리에서 자동으로 찾는다.")
    ap.add_argument("--only-cells", default=None,
                    help="Comma-separated cells to evaluate.  Needed when the "
                         "trim directory holds a subset - a nested inner "
                         "split has no model for the outer cell, and without "
                         "this the run dies looking for one.")
    ap.add_argument("--soc-est", default=None,
                    help="npz of per-cell (cycle, SOC error) from the SOC "
                         "EKF.  Feeds ESTIMATED SOC into the inversion "
                         "instead of the label's true SOC.")
    ap.add_argument("--soh-est", default=None,
                    help="SOH 팔의 예측 npz.  주면 라벨의 정답 SOH 대신 "
                         "직전 충전에서 추정한 SOH 를 반전에 넣는다. "
                         "실제 시스템은 SOH 를 추정하므로 이쪽이 "
                         "시스템 수준의 수치다.")
    args = ap.parse_args()
    charge = args.direction == "charge"
    if args.trim_rung:
        globals()["TRIM_RUNG"] = args.trim_rung
    if args.soh_est:
        globals()["SOH_EST"] = dict(np.load(args.soh_est))
    if args.soc_est:
        globals()["SOC_EST"] = dict(np.load(args.soc_est))
    globals()["USE_DTAU"] = args.interp == "dtau"
    if args.label is None:
        args.label = LABEL_CHG if charge else LABEL
    if args.out is None:
        args.out = os.path.join(HERE, f"sop_amps_eval_{args.direction}.csv") \
            if charge else OUT

    rows = [r for r in csv.DictReader(open(args.label, encoding="utf-8"))]
    if args.only_cells:
        want = {c.strip() for c in args.only_cells.split(",") if c.strip()}
        rows = [r for r in rows if r["cell"] in want]
        if not rows:
            raise SystemExit(f"  --only-cells {sorted(want)} matched no rows")
    cells = sorted({r["cell"] for r in rows})
    surf = {c: surfaces(c)[1 if charge else 0] for c in cells}
    trim_dir = (os.path.join(HERE, args.trim) if args.trim
                else (os.path.join(HERE, "runs_trim_chg") if charge else None))
    by_soc = args.trim_key == "cycle_soc"
    K = {c: (trim_k_soc(c, trim_dir) if by_soc
             else trim_k_agg(c, trim_dir, args.trim_agg))
         for c in cells}
    SPREAD = {c: trim_spread(c, trim_dir) for c in cells}

    out, drop = [], collections.Counter()
    for r in rows:
        c = r["cell"]; cyc = int(r["cycle"])
        soc = float(r["SOC"]); soh = float(r["SOH"])
        if SOH_EST is not None:
            soh = soh_estimated(c, int(r["cycle"]), soh)
        if SOC_EST is not None:
            soc = soc_estimated(c, cyc, soc)
        v_pre = float(r["V_pre_V"]); v_min = float(r["V_min_V"])
        tau = float(r["tau_s"])
        mkey = "I_star_lin2hi_A" if args.label_fit == "lin2hi" else "I_star_lin4_A"
        if r.get(mkey) in (None, "", "nan"):
            drop["라벨 없음"] += 1
            continue
        meas = float(r[mkey])
        if cyc not in K[c]:
            drop["k 없음"] += 1
            continue
        if by_soc:
            socs, kfa, ksa = K[c][cyc]
            j = int(np.argmin(np.abs(socs - soc)))
            kf, ks = float(kfa[j]), float(ksa[j])
        else:
            kf, ks = K[c][cyc]
        i0 = solve_I(surf[c], soc, soh, v_pre, v_min, tau, 1.0, 1.0, I0=meas,
                     charge=charge)
        i3 = solve_I(surf[c], soc, soh, v_pre, v_min, tau, kf, ks, I0=meas,
                     charge=charge)
        if not (np.isfinite(i0) and np.isfinite(i3)):
            drop["hull 밖 / 미수렴"] += 1
            continue
        out.append({**{k: r[k] for k in ("cell", "cycle", "SOH", "SOC", "tau_s",
                                         "V_pre_V", "I_max_meas_A", "extrap")},
                    "I_meas_A": round(meas, 3),
                    "I_A0_A": round(i0, 3), "I_A3_A": round(i3, 3),
                    "err_A0_A": round(i0 - meas, 3),
                    "err_A3_A": round(i3 - meas, 3),
                    "k_f": round(kf, 4), "k_s": round(ks, 4),
                    "below_train_soc": int(soc < SOC_TRAIN_MIN),
                    "k_spread": round(SPREAD[c].get(cyc, np.nan), 5)})
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"  {args.out}   {len(out):,}행   제외 {dict(drop)}")

    ex = np.array([float(r["extrap"]) for r in out])
    e0 = np.array([r["err_A0_A"] for r in out])
    e3 = np.array([r["err_A3_A"] for r in out])
    im = np.abs([r["I_meas_A"] for r in out])
    rm = lambda e, m: float(np.sqrt(np.mean(e[m] ** 2)))

    print(f"\n  === 외삽 밴드별 (전류는 음수: 편향 음수 = 크기 과대 = 낙관/위험) ===")
    print(f"  {'배수':<12} {'n':>6} {'|I*| 중앙':>9} "
          f"{'ECM RMSE':>10} {'편향':>8} | {'하이브리드':>10} {'편향':>8}")
    for lo, hi, nm in ((0, 1.0, "<=1.0 내삽"), (1.0, 1.5, "1.0~1.5"),
                       (1.5, 2.5, "1.5~2.5"), (2.5, 1e9, ">2.5")):
        m = (ex > lo) & (ex <= hi)
        if m.sum() < 20:
            continue
        print(f"  {nm:<12} {m.sum():>6,} {np.median(im[m]):>8.1f}A "
              f"{rm(e0,m):>9.2f}A {np.median(e0[m]):>+7.2f}A | "
              f"{rm(e3,m):>9.2f}A {np.median(e3[m]):>+7.2f}A")
    m = np.ones(len(ex), bool)
    print(f"  {'전체':<12} {m.sum():>6,} {np.median(im):>8.1f}A "
          f"{rm(e0,m):>9.2f}A {np.median(e0):>+7.2f}A | "
          f"{rm(e3,m):>9.2f}A {np.median(e3):>+7.2f}A")

    print(f"\n  === 셀별 (전체 행) ===")
    print(f"  {'셀':<20} {'n':>6} {'ECM RMSE':>10} {'하이브리드':>11} {'개선':>8}")
    cc = np.array([r["cell"] for r in out])
    for c in cells:
        m = cc == c
        a, b = rm(e0, m), rm(e3, m)
        print(f"  {c:<20} {m.sum():>6,} {a:>9.2f}A {b:>10.2f}A "
              f"{(1-b/a)*100:>+7.1f}%")
    a, b = rm(e0, np.ones(len(cc), bool)), rm(e3, np.ones(len(cc), bool))
    print(f"  {'전체':<20} {len(cc):>6,} {a:>9.2f}A {b:>10.2f}A {(1-b/a)*100:>+7.1f}%")

    bt = np.array([r["below_train_soc"] for r in out], bool)
    print(f"\n  학습 SOC 범위(>=0.29) 안 {int((~bt).sum()):,}행: "
          f"ECM {rm(e0,~bt):.2f}A  하이브리드 {rm(e3,~bt):.2f}A")
    print(f"  그 아래 {int(bt.sum()):,}행: "
          f"ECM {rm(e0,bt):.2f}A  하이브리드 {rm(e3,bt):.2f}A")
    print(f"\n  라벨 자체의 외삽 편향: 1.44배에서 1.17 A, 2.87배에서 2.51 A "
          f"(I* 를 작게 보는 방향)")


if __name__ == "__main__":
    main()
