"""외부 셀 두 대에서 하이브리드를 채점하고, 여기(Test#2)와 저기(Test#1)의 차이를 본다.

두 셀은 노화 프로파일이 다르다.
    Test#2  US06 주행           - 트림이 학습된 여기(drive cycle)와 같은 종류
    Test#1  정전류(18 A) 방전    - 여기의 종류가 다르다

트림의 핵심 특징은 EW{I r} / EW{I I} 로, 잔차를 전류에 회귀한 기울기다. 전류가 한
값뿐이면 그 회귀는 점 하나를 지나는 직선이 되고, 30 A 로의 전이는 순수한 외삽이
된다. 그것이 문제가 되는지를 여기서 처음 잰다.
"""
from __future__ import annotations
import argparse, csv, subprocess, sys, os
import numpy as np
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))


def run(test, agg, lam, runs, out):
    subprocess.run([sys.executable, os.path.join(HERE, "eval_sop_us06.py"),
                    "--test", str(test), "--agg", agg, "--runs", runs,
                    "--lam-t10", str(lam), "--out", out],
                   check=True, capture_output=True, cwd=HERE)
    r = list(csv.DictReader(open(out, encoding="utf-8")))
    u = {}
    for x in r:
        u.setdefault((x["cycle"], x["SOC"], x["T_C"]), []).append(x)
    M = np.array([abs(float(v[0]["I_meas_A"])) for v in u.values()])
    P3 = np.array([np.median([abs(float(y["I_A3_A"])) for y in v])
                   for v in u.values()])
    P0 = np.array([np.median([abs(float(y["I_A0_A"])) for y in v])
                   for v in u.values()])
    kf = np.array([np.median([float(y["k_f"]) for y in v]) for v in u.values()])
    T = np.array([float(k[2]) for k in u])
    return M, P0, P3, kf, T


def stat(P, M):
    o = P - M
    k, n = int((o > 0).sum()), len(o)
    hi = beta.ppf(0.95, k + 1, n - k) if k < n else 1.0
    return (n, k / n, hi, max(o.max(), 0.0), float(np.median(P / M)),
            float(np.sqrt(np.mean(o ** 2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_trim_w")
    a = ap.parse_args()
    print(f"  트림 {a.runs}   여유는 19.9 절에서 UYPYDJ 로 보정된 값을 그대로 쓴다"
          f" (ECM 0.665, 하이브 0.840)\n")
    hdr = (f"  {'':<30}{'n':>4}{'초과율':>8}{'95%상한':>9}{'최악':>8}"
           f"{'가용률':>8}{'RMSE':>8}")
    for test, nm in ((2, "Test#2  US06 주행"), (1, "Test#1  정전류 18 A")):
        print(f"  === {nm} ===")
        print(hdr)
        M, P0, P3, kf, T = run(test, "max", 1.0, a.runs,
                               os.path.join(HERE, f"sop_ext_t{test}.csv"))
        for tag, P, lam in (("ECM, 여유 없음", P0, 1.0),
                            ("ECM + 0.665", P0, 0.665),
                            ("하이브리드, 여유 없음", P3, 1.0),
                            ("하이브리드 + 0.840", P3, 0.840)):
            n, e, hi, w, u, rm = stat(lam * P, M)
            print(f"  {tag:<30}{n:>4}{e*100:>7.1f}%{hi*100:>8.1f}%"
                  f"{w:>7.2f}A{u:>8.3f}{rm:>7.2f}A")
        print(f"  k_f {kf.min():.3f}~{kf.max():.3f} 중앙 {np.median(kf):.3f}   "
              f"온도 {sorted(set(T.tolist()))}\n")


if __name__ == "__main__":
    main()
