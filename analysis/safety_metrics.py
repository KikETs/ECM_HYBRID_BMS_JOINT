"""SOP 예측의 안전 측도. RMSE 가 가리는 것을 드러낸다.

측도
    초과율    P(|I_hat| > |I_true|)      저전압 보호를 뚫을 확률
    최악초과  max(|I_hat| - |I_true|)    뚫었을 때 얼마나
    가용률    median(|I_hat| / |I_true|) 얼마나 쓸 수 있게 남기는가

자명하게 안전한 예측(I=0)은 초과율 0 이고 가용률 0 이다. 두 값을 같이 읽지
않으면 아무 의미가 없다.

왜 신뢰 라벨만 쓰는가
    라벨은 HPPC 4-rate 팬을 전압 바닥까지 늘여 만든다. extrap = |I*|/max|I_meas|
    가 2.5 를 넘는 행은 |I*| 중앙이 95.6 A - 3 Ah 셀에서 32C 다. 그런 라벨은
    부풀려져 있고, 그에 대고 재면 어떤 모델이든 보수적으로 보인다. extrap<=1.5
    로 한정하면 자체 불확실성 spread_A 가 중앙 1.64 A 이고 두 적합(lin4/lin2hi)
    의 차이가 -0.19 A 로 편향이 없다.
"""
from __future__ import annotations
import csv, sys
import numpy as np

EXTRAP_MAX = 1.5


def load(path, pred_col="I_A3_A"):
    r = list(csv.DictReader(open(path, encoding="utf-8")))
    g = lambda k: np.array([float(x[k]) if x[k] not in ("", "nan") else np.nan
                            for x in r])
    return dict(meas=np.abs(g("I_meas_A")), pred=np.abs(g(pred_col)),
                ecm=np.abs(g("I_A0_A")), extrap=g("extrap"), SOH=g("SOH"),
                SOC=g("SOC"), tau=g("tau_s"),
                cell=np.array([x["cell"] for x in r]))


def metrics(pred, meas):
    over = pred - meas
    return dict(n=len(pred), exc=float(np.mean(over > 0)),
                worst=float(np.max(over)) if len(over) else np.nan,
                p95=float(np.percentile(over, 95)),
                util=float(np.median(pred / meas)),
                rmse=float(np.sqrt(np.mean(over ** 2))))


def report(tag, d, pred_key="pred", trusted=True):
    m = np.isfinite(d["meas"]) & np.isfinite(d[pred_key]) & (d["meas"] > 0.5)
    if trusted:
        m &= d["extrap"] <= EXTRAP_MAX
    r = metrics(d[pred_key][m], d["meas"][m])
    print(f"  {tag:<22}{r['n']:>6}{r['exc']*100:>8.1f}%{r['worst']:>9.2f}A"
          f"{r['p95']:>9.2f}A{r['util']:>8.3f}{r['rmse']:>8.2f}A")
    return r


if __name__ == "__main__":
    print(f"  {'':<22}{'n':>6}{'초과율':>9}{'최악초과':>10}{'95%tile':>10}"
          f"{'가용률':>8}{'RMSE':>8}")
    for p in sys.argv[1:]:
        d = load(p)
        report(p.replace("sop_amps_eval", "").replace(".csv", "") or "기준", d)
