"""외부 두 셀에서의 SOP: 예측 대 측정.

왼쪽 두 판은 산점도 - 대각선 위(왼쪽 위)가 낙관이고 저전압 보호를 뚫는 쪽이다.
오른쪽은 가용률을 맞춘 뒤의 초과율로, 두 팔을 공정하게 세운 비교다.
"""
from __future__ import annotations
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "fig_sop_external.png")


def load(f):
    r = list(csv.DictReader(open(os.path.join(HERE, f), encoding="utf-8")))
    u = {}
    for x in r:
        u.setdefault((x["cycle"], x["SOC"], x["T_C"]), []).append(x)
    M = np.array([abs(float(v[0]["I_meas_A"])) for v in u.values()])
    P3 = np.array([np.median([abs(float(y["I_A3_A"])) for y in v]) for v in u.values()])
    P0 = np.array([np.median([abs(float(y["I_A0_A"])) for y in v]) for v in u.values()])
    return M, P0, P3


def main():
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
    sets = [("sop_ext_t2.csv", "Test#2  —  US06 drive", 0),
            ("sop_ext_t1.csv", "Test#1  —  constant current", 1)]
    for f, title, k in sets:
        M, P0, P3 = load(f)
        a = ax[k]
        lim = [0, max(M.max(), P0.max()) * 1.08]
        a.fill_between(lim, lim, [lim[1]] * 2, color="crimson", alpha=0.06)
        a.plot(lim, lim, "-", color="grey", lw=1.2)
        a.plot(M, P0, "o", ms=7, mfc="none", mec="#777777", mew=1.4,
               label="pooled ECM")
        a.plot(M, P3, "o", ms=7, color="#c1440e", alpha=0.85,
               label="hybrid (trim from this cell's own drive)")
        a.set_xlim(lim); a.set_ylim(lim)
        a.set_xlabel("measured  |I*|   [A]")
        a.set_ylabel("predicted  |I*|   [A]")
        a.set_title(f"({'ab'[k]})  {title}   n={len(M)}", fontsize=10.5)
        a.legend(fontsize=8.5, loc="upper left")
        a.grid(alpha=0.25)
        a.text(0.97, 0.06, "shaded = optimistic\n(allows more current\nthan the cell has)",
               transform=a.transAxes, ha="right", va="bottom", fontsize=8,
               color="crimson", alpha=0.85)

    # (c) 가용률을 맞춘 초과율
    a = ax[2]
    US = np.linspace(0.62, 1.0, 25)
    for f, title, col, ls in (("sop_ext_t2.csv", "Test#2", "#1f6f8b", "-"),
                              ("sop_ext_t1.csv", "Test#1", "#c1440e", "-")):
        M, P0, P3 = load(f)
        for P, nm, style in ((P0, "ECM", "--"), (P3, "hybrid", "-")):
            ys = []
            for U in US:
                lam = U / np.median(P / M)
                ys.append(np.mean(lam * P > M) * 100)
            a.plot(US, ys, style, color=col, lw=1.8,
                   label=f"{title}, {nm}")
    a.axhline(5, color="crimson", lw=1, alpha=0.5)
    a.set_xlabel("utilisation  median(|I_pred| / |I_true|)")
    a.set_ylabel("exceedance  [%]")
    a.set_title("(c)  at matched usable current, which arm is safer", fontsize=10.5)
    a.legend(fontsize=8.5); a.grid(alpha=0.25)
    fig.suptitle("Hybrid SOP on two EXTERNAL cells  —  different lab, different cycler, "
                 "features recomputed from each cell's own aging drive", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"  {OUT}")


if __name__ == "__main__":
    main()
