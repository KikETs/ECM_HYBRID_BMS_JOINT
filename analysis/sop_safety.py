"""SOP 예측을 안전 쪽으로 보정하고, 그 대가를 정직하게 센다.

문제
    SOP 를 RMSE 로만 재면 낙관(전류를 실제보다 크게 봄)과 보수가 같은 값으로
    벌해진다. BMS 에서는 둘이 전혀 같지 않다 - 낙관은 저전압 보호를 뚫고,
    보수는 출력을 못 쓸 뿐이다. 신뢰 라벨에서 기존 하이브리드는 82.9 % 가
    낙관이었다.

보정
    |I_hat| <- lambda * |I_hat|. lambda 는 홀드아웃 셀을 빼고 나머지에서
    초과율이 목표가 되도록 이분법으로 잡는다(LOCO). 조건화 축은 선택할 수 있고,
    축이 늘수록 여유를 필요한 곳에만 둘 수 있지만 각 칸의 표본이 줄어 보정 자체가
    흔들린다. 그 균형을 표로 보여주는 것이 이 파일의 목적이다.

왜 tau 로 나누는 것이 가장 크게 듣는가
    필요한 저항 배수의 중앙이 tau=10 s 에서 1.076 인데 tau=2 s 에서는 1.323 이고
    90 %tile 이 1.843 이다. 전역 lambda 하나는 tau=2 s 의 실패를 tau=10 s 에도
    물린다.
"""
from __future__ import annotations
import argparse, csv
import numpy as np

EXTRAP_MAX = 1.5


def load(path):
    r = list(csv.DictReader(open(path, encoding="utf-8")))
    g = lambda k: np.array([float(x[k]) if x[k] not in ("", "nan") else np.nan
                            for x in r])
    return dict(meas=np.abs(g("I_meas_A")), hyb=np.abs(g("I_A3_A")),
                ecm=np.abs(g("I_A0_A")), extrap=g("extrap"), SOH=g("SOH"),
                tau=g("tau_s"), cell=np.array([x["cell"] for x in r]))


def lam_for(pred, meas, target):
    """훈련 셀에서 초과율이 목표가 되는 최대 스칼라. 이분법."""
    lo, hi = 0.02, 1.6
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.mean(mid * pred > meas) > target:
            hi = mid
        else:
            lo = mid
    return lo


def bins(d, axes):
    """조건화 축 -> 각 행의 칸 라벨."""
    lab = np.zeros(len(d["meas"]), dtype=object)
    lab[:] = ""
    if "tau" in axes:
        lab = lab + np.array([f"t{int(t)}|" for t in d["tau"]], dtype=object)
    if "soh" in axes:
        e = np.digitize(d["SOH"], [0.80, 0.90])
        lab = lab + np.array([f"s{i}|" for i in e], dtype=object)
    return lab


def evaluate(d, pred_key, target, axes, min_train=25):
    m = np.isfinite(d["meas"]) & np.isfinite(d[pred_key]) & (d["meas"] > 0.5) \
        & (d["extrap"] <= EXTRAP_MAX)
    lab = bins(d, axes)
    P, M, L, dropped = [], [], {}, 0
    for c in sorted(set(d["cell"][m])):
        for b in sorted(set(lab[m])):
            te = m & (d["cell"] == c) & (lab == b)
            tr = m & (d["cell"] != c) & (lab == b)
            if te.sum() == 0:
                continue
            if tr.sum() < min_train:          # 칸이 얇으면 전역으로 후퇴
                tr = m & (d["cell"] != c)
                dropped += int(te.sum())
            lam = lam_for(d[pred_key][tr], d["meas"][tr], target)
            L.setdefault(b, []).append(lam)
            P.append(lam * d[pred_key][te]); M.append(d["meas"][te])
    P = np.concatenate(P); M = np.concatenate(M)
    over = P - M
    return dict(n=len(P), exc=float(np.mean(over > 0)), worst=float(np.max(over)),
                util=float(np.median(P / M)), fallback=dropped,
                lam={k: float(np.median(v)) for k, v in L.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--targets", default="0.10,0.05,0.01")
    a = ap.parse_args()
    print(f"  {'파일':<26}{'예측':<8}{'축':<10}{'목표':>6}{'n':>6}"
          f"{'초과율':>8}{'최악':>9}{'가용률':>8}  lambda")
    for path in a.csvs:
        d = load(path)
        for key, nm in (("ecm", "ECM"), ("hyb", "하이브")):
            for axes in ((), ("tau",), ("tau", "soh")):
                an = "+".join(axes) or "전역"
                for t in [float(x) for x in a.targets.split(",")]:
                    r = evaluate(d, key, t, axes)
                    ls = " ".join(f"{k.rstrip('|')}={v:.3f}"
                                  for k, v in sorted(r["lam"].items()))
                    print(f"  {path[:25]:<26}{nm:<8}{an:<10}{t*100:>5.0f}%"
                          f"{r['n']:>6}{r['exc']*100:>7.1f}%{r['worst']:>8.2f}A"
                          f"{r['util']:>8.3f}  {ls}")
            print()


if __name__ == "__main__":
    main()
