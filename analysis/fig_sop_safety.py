"""안전-가용 프론티어. SOP 를 RMSE 하나로 보면 보이지 않는 축을 그린다.

왼쪽  목표 초과율을 쓸어가며 얻는 (초과율, 가용률) 곡선. 오른쪽으로 갈수록
      위험하고 위로 갈수록 출력을 많이 쓴다. 곡선이 위에 있는 쪽이 이긴다.
오른쪽 같은 5 % 초과율에서 남는 최악 초과. 곡선 위치가 같아도 꼬리는 다르다.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sop_safety import load, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "fig_sop_safety.png")
TARGETS = [0.40, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01]

ARMS = [
    ("ev_runs_trim_cycle.csv", "ecm", (), "ECM, one margin", "#777777", "--"),
    ("ev_runs_trim_cycle.csv", "ecm", ("tau",), "ECM, margin per horizon",
     "#777777", "-"),
    ("ev_runs_trim_cycle.csv", "hyb", (), "hybrid as shipped, one margin",
     "#e8a33d", "--"),
    ("ev_runs_trim_cycle.csv", "hyb", ("tau",),
     "hybrid as shipped, per horizon", "#e8a33d", "-"),
    ("ev_sym_max.csv", "hyb", ("tau",),
     "+ conservative history aggregation", "#c1440e", "-"),
    ("ev_q90_max.csv", "hyb", ("tau",),
     "+ pinball q=0.9 as well", "#1f6f8b", "-"),
]


def main():
    fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.2))
    for path, key, axes, lab, col, ls in ARMS:
        d = load(os.path.join(HERE, path))
        xs, ys, ws = [], [], []
        for t in TARGETS:
            r = evaluate(d, key, t, axes)
            xs.append(r["exc"] * 100); ys.append(r["util"]); ws.append(r["worst"])
        o = np.argsort(xs)
        ax[0].plot(np.array(xs)[o], np.array(ys)[o], ls, color=col, marker="o",
                   ms=3.5, lw=1.8, label=lab)
        ax[1].plot(np.array(xs)[o], np.array(ws)[o], ls, color=col, marker="o",
                   ms=3.5, lw=1.8, label=lab)
    for a in ax[:2]:
        a.axvline(5, color="crimson", lw=1, alpha=0.5)
        a.set_xlabel("realised exceedance  [%]   P(|I_pred| > |I_true|)")
        a.grid(alpha=0.25)
    ax[2].grid(alpha=0.25)
    ax[0].set_ylabel("utilisation   median(|I_pred| / |I_true|)")
    ax[0].set_title("(a) what safety costs in usable current")
    ax[1].set_ylabel("worst overshoot  [A]")
    ax[1].set_title("(b) the tail does not follow the rate")
    ax[0].legend(fontsize=8, loc="lower right")

    # (c) 분위수 손잡이가 얼마나 감쇠되는가
    QS = [("q050", 0.50), ("q070", 0.70), ("q080", 0.80), ("q90", 0.90),
          ("q095", 0.95), ("q099", 0.99)]
    for agg, col, mk in (("last", "#e8a33d", "o"), ("max", "#c1440e", "s")):
        xs, ys = [], []
        for pre, q in QS:
            f = os.path.join(HERE, f"ev_{pre}_{agg}.csv")
            if not os.path.exists(f):
                continue
            d = load(f)
            m = np.isfinite(d["meas"]) & np.isfinite(d["hyb"]) \
                & (d["meas"] > 0.5) & (d["extrap"] <= 1.5)
            xs.append((1 - q) * 100)
            ys.append(float(np.mean(d["hyb"][m] > d["meas"][m])) * 100)
        ax[2].plot(xs, ys, "-", color=col, marker=mk, ms=5, lw=1.8,
                   label=f"history aggregation: {agg}")
    lim = [0, 55]
    ax[2].plot(lim, lim, ":", color="grey", lw=1.2, label="what the loss promises")
    ax[2].set_xlim(0, 52); ax[2].set_ylim(0, 90)
    ax[2].set_xlabel("nominal exceedance from the pinball loss,  1 - q   [%]")
    ax[2].set_ylabel("realised exceedance  [%]")
    ax[2].set_title("(c) the quantile knob is a knob, not a guarantee")
    ax[2].legend(fontsize=8, loc="upper left")
    fig.suptitle("SOP safety-utilisation frontier  —  651 trustworthy rows (extrap<=1.5), margin calibrated leave-one-cell-out",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"  {OUT}")


if __name__ == "__main__":
    main()
