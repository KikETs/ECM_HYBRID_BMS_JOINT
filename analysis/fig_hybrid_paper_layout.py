"""The hybrid arm in the reference paper's figure layout.

THE AXES TRANSLATE, THE PANELS DO NOT CHANGE
    The paper sweeps chamber temperature and splits by drive cycle. This arm is
    measured on aged cells at a single temperature, so the sweep axis is SOH and
    the split is by aging protocol - one cell each. Panels keep their meaning:
    (a) a trace with its error, (b) error against the sweep variable, (c) every
    condition individually.

(a) IS A REAL TIME TRACE, NOT A SCATTER OF PULSE POINTS
    The 2RC state is stepped through the measured current from the start of a
    rest, so the curve is what the model would have produced online. Two curves:
    the pooled ECM at unit multipliers, and the same ECM with the trim's k_f, k_s
    for that characterisation. Nothing else differs between them, which is the
    point - the gap IS the 26 parameters.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecm_pool import surfaces  # noqa: E402
from eval_a13 import pulse_index  # noqa: E402
from eval_sop_amps import trim_k  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = ["BOOST", "BOOST_NEGPULSE", "BOOST_NEGPULSE_1S", "BOOST_REST",
         "CC", "CC_CELL2"]
SHORT = {"BOOST": "BOOST", "BOOST_NEGPULSE": "BNP", "BOOST_NEGPULSE_1S": "BNP_1S",
         "BOOST_REST": "B_REST", "CC": "CC", "CC_CELL2": "CC_2"}
BANDS = [(0.95, 1.01), (0.90, 0.95), (0.85, 0.90), (0.80, 0.85),
         (0.75, 0.80), (0.66, 0.75)]
BL = ["1.00-0.95", "0.95-0.90", "0.90-0.85", "0.85-0.80", "0.80-0.75", "0.75-0.68"]
COL = ["#0d47a1", "#1565c0", "#1e88e5", "#42a5f5", "#90caf9", "#bbdefb"]


def load_pulses(cell):
    """Per-pulse predictions of every arm, keyed the same way."""
    out = {}
    for rung in ("A0", "A3", "A4"):
        z = np.load(os.path.join(HERE, "runs_trim", f"pred_{rung}_{cell}.npz"),
                    allow_pickle=True)
        key = list(zip(z["cycle"].astype(int).tolist(),
                       np.round(z["SOC"].astype(float), 4).tolist(),
                       z["rank"].astype(str).tolist()))
        last = {}
        for i, k in enumerate(key):
            last[k] = i
        p = z["pred"] if "pred" in z else z["base"]
        out[rung] = {k: (p[i][1], z["Y"][i][1], float(z["SOH"][i]))
                     for k, i in last.items()}
    f = os.path.join(HERE, "runs_soh", f"pred_a13_i_M2_full_{cell}.npz")
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        out["M2"] = {(int(a), round(float(b), 4), str(r)):
                     (z["pred"][i][1], z["Y"][i][1], np.nan)
                     for i, (a, b, r) in enumerate(zip(z["cycle"], z["SOC"],
                                                       z["rank"]))}
    return out


def rmse_mv(pairs):
    if not pairs:
        return np.nan
    e = np.array([p - y for p, y, _ in pairs])
    return float(np.sqrt(np.mean(e ** 2)) * 1000)


def simulate(cell, cycle, kf, ks, n_pulses=4):
    """Step the 2RC model through a measured pulse train from a rest."""
    sd, _ = surfaces(cell)
    idx, cols = pulse_index(cell)
    got = [(g, r, base, a, ks_, end, ip) for (c, g, r, d), (base, a, ks_, end, ip)
           in idx.items() if c == cycle and d == "discharge"]
    if not got:
        return None
    got.sort(key=lambda x: x[3])
    # Pick a group in the MIDDLE of the run: the first group can sit at the very
    # start with no rest before it, and clamping a negative start silently walks
    # into the previous run - which is what produced a 100,000 s wide panel the
    # first time.
    groups = sorted({x[0] for x in got})
    seq = None
    for grp in groups[len(groups) // 3:]:
        cand = sorted([x for x in got if x[0] == grp], key=lambda x: x[3])[:n_pulses]
        if len(cand) < 2:
            continue
        base = cand[0][2]; n = cand[0][5] - base
        a0 = cand[0][3] - 200; a1 = cand[-1][3] + 300
        if a0 < 0 or a1 >= n:
            continue
        tt = cols["t"][base + a0:base + a1]
        # One SOC group's four rates are spread over ~7,500 s because each pulse
        # is followed by a long rest. That IS the pulse train; the earlier 2,500 s
        # cap rejected every group and left the panel empty.
        if tt[-1] - tt[0] > 9000 or tt[-1] <= tt[0]:
            continue
        seq = cand; break
    if seq is None:
        return None
    base = seq[0][2]
    a0 = seq[0][3] - 200; a1 = seq[-1][3] + 300
    sl = slice(base + a0, base + a1)
    t, V, I, SOC, SOH = (cols[k][sl] for k in ("t", "V", "I", "SOC", "SOH"))
    soh = float(np.nanmedian(SOH))
    out = {}
    for name, (a, b) in (("ECM", (1.0, 1.0)), ("Hybrid", (kf, ks))):
        v1 = v2 = 0.0
        pred = np.empty(len(t))
        for j in range(len(t)):
            dt = float(t[j] - t[j - 1]) if j else 1.0
            dt = min(max(dt, 1e-3), 60.0)
            th = sd.theta(float(SOC[j]), soh, float(I[j]))
            if not bool(th["in_hull"][0]):
                pred[j] = np.nan; continue
            R0 = float(th["R0"][0]) * a; R1 = float(th["R1"][0]) * a
            R2 = float(th["R2"][0]) * b
            t1 = float(th["tau1"][0]); t2 = float(th["tau2"][0])
            e1, e2 = np.exp(-dt / t1), np.exp(-dt / t2)
            v1 = v1 * e1 + R1 * (1 - e1) * float(I[j])
            v2 = v2 * e2 + R2 * (1 - e2) * float(I[j])
            ocv, _ = sd.ocv(float(SOC[j]), soh)
            pred[j] = float(ocv[0]) + float(I[j]) * R0 + v1 + v2
        # anchor on the first rest sample so the trace shows the RESPONSE, not
        # the OCV table's absolute offset
        off = np.nanmedian(pred[:200] - V[:200])
        out[name] = pred - off
    return t - t[0], V, I, out, soh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-cell", default="CC")
    ap.add_argument("--out", default=os.path.join(HERE, "fig_hybrid_paper_layout.png"))
    args = ap.parse_args()

    D = {c: load_pulses(c) for c in CELLS}
    K = {c: trim_k(c) for c in CELLS}

    fig = plt.figure(figsize=(15.5, 9.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.45, 1.05],
                          hspace=.5, wspace=.22)

    # ---- (a) ----
    cyc = sorted({k[0] for k in D[args.trace_cell]["A3"]})
    pick = cyc[len(cyc) * 3 // 4]
    kf, ks = K[args.trace_cell][pick]
    sim = simulate(args.trace_cell, pick, kf, ks)
    ax = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    if sim:
        t, V, I, pr, soh = sim
        ax.plot(t, V, color="k", lw=1.7, label="Measurement")
        ax.plot(t, pr["ECM"], color="#f9a825", lw=1.2,
                label="Pooled ECM, no correction")
        ax.plot(t, pr["Hybrid"], color="#2e7d32", lw=1.2,
                label="Hybrid linear (26 parameters)")
        ax.set_ylabel("Voltage  [V]"); ax.legend(fontsize=8.5); ax.grid(alpha=.25)
        ax.set_title(f"(a)  {SHORT[args.trace_cell]} HPPC pulse train, cycle {pick}, "
                     f"SOH {soh:.3f}  —  held out", fontsize=10.5, loc="left")
        plt.setp(ax.get_xticklabels(), visible=False)
        for nm, col in (("ECM", "#f9a825"), ("Hybrid", "#2e7d32")):
            ax2.plot(t, (pr[nm] - V) * 1000, color=col, lw=0.9, label=nm)
        ax2.axhline(0, color="k", lw=1); ax2.set_ylim(-200, 200)
        ax2.set_xlabel("Time  [s]"); ax2.set_ylabel("Error  [mV]"); ax2.grid(alpha=.25)

    # ---- (b) ----
    ax = fig.add_subplot(gs[0:2, 1])
    SER = [("A0", "Pooled ECM, no correction", "#f9a825", "-o"),
           ("M2", "Full AI M2  (1.08 M params)", "#c62828", "-^"),
           ("A4", "Hybrid MLP  (514)", "#1565c0", "-s"),
           ("A3", "Hybrid linear  (26)", "#2e7d32", "-D")]
    xs = np.arange(len(BANDS))
    for k, name, col, mk in SER:
        ys = []
        for lo, hi in BANDS:
            pool = []
            for c in CELLS:
                if k not in D[c]:
                    continue
                soh_map = {q: s for q, (_, _, s) in D[c]["A3"].items()}
                for q, v in D[c][k].items():
                    s = soh_map.get(q, np.nan)
                    if np.isfinite(s) and lo <= s < hi:
                        pool.append(v)
            ys.append(rmse_mv(pool))
        ax.plot(xs, ys, mk, color=col, lw=2.0, ms=7, label=name)
        for x, y in zip(xs, ys):
            if np.isfinite(y):
                ax.annotate(f"{y:.0f}", (x, y), fontsize=7.5, ha="center",
                            xytext=(0, 6), textcoords="offset points", color=col)
    ax.set_xticks(xs); ax.set_xticklabels(BL, fontsize=8.5)
    ax.set_xlabel("SOH band"); ax.set_ylabel("Pulse dV RMSE  [mV]")
    ax.set_title("(b)  pooled over the six cells", fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5); ax.grid(alpha=.25)

    # ---- (c) ----
    ax = fig.add_subplot(gs[2, :])
    for i, c in enumerate(CELLS):
        soh_map = {q: s for q, (_, _, s) in D[c]["A3"].items()}
        for j, (lo, hi) in enumerate(BANDS):
            pool = [v for q, v in D[c]["A3"].items()
                    if lo <= soh_map.get(q, np.nan) < hi]
            v = rmse_mv(pool)
            if not np.isfinite(v):
                continue
            b = ax.bar(i * 7.5 + j, v, 0.8, color=COL[j],
                       label=BL[j] if i == 0 else None)
            ax.bar_label(b, fmt="%.0f", fontsize=7, padding=1)
        ax.annotate(SHORT[c], (i * 7.5 + 2.5, -0.13),
                    xycoords=("data", "axes fraction"), ha="center", fontsize=11)
    ax.set_xticks([i * 7.5 + j for i in range(len(CELLS)) for j in range(len(BANDS))])
    ax.set_xticklabels([b.split("-")[1] for _ in CELLS for b in BL], fontsize=7)
    ax.set_ylabel("Pulse dV RMSE  [mV]")
    ax.set_title("(c)  hybrid linear, every cell and SOH band  —  "
                 "leave-one-cell-out throughout", fontsize=10.5, loc="left")
    ax.legend(fontsize=8, ncol=6, loc="upper left", title="SOH band",
              title_fontsize=8)
    ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
    ax.annotate("SOH (lower edge of band)", (0.5, -0.235),
                xycoords="axes fraction", ha="center", fontsize=9.5)
    plt.savefig(args.out, dpi=145, bbox_inches="tight")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
