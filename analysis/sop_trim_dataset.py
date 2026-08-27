"""Pair measured HPPC pulse response (labels) with drive-cycle features.

THE LABEL IS A MEASURED VOLTAGE DROP, NOT A FITTED PARAMETER
    dV_meas(tau) = V(t0+tau) - V(t0-), read straight out of
    uypydj_hppc_resistance.csv. Not the fitted R, not sop_reference.csv.

    sop_reference.csv is excluded from the loss on principle: it is derived from
    the same ECM the trim is correcting, so training against it would fit the
    model to its own assumptions and score well for it.

    Absolute voltage is excluded for a measured reason: the pooled OCV surface's
    leave-one-out error reaches 45-72 mV at SOH 0.76-0.80, while the whole
    resistance correction is worth about 40 mV at 29 A. A loss on absolute V
    routes OCV error straight into the resistance multipliers. dV differences
    the OCV away - and because every kept pulse follows a long rest, the RC
    states are zero at t0 and the differencing is exact rather than approximate.

BOTH HORIZONS ARE REQUIRED, AND THAT IS WHAT MAKES k_s IDENTIFIABLE
    At tau = 2 s the slow branch has barely developed; at 10 s it carries most of
    the response. One horizon alone leaves (k_f, k_s) on a ridge.

THE PAIRING DIRECTION IS THE DEPLOYABLE ONE
    Features come from the drive-cycle runs IMMEDIATELY PRECEDING the HPPC, never
    from the HPPC itself. A vehicle never performs an HPPC; if the features came
    from the labelled pulses the model would be reading the answer.

    This is also where the design's largest risk lives, stated plainly: the
    features are measured in a regime where the cell never rests, the labels in
    one where it always has. Nothing in this dataset can prove that map
    transfers - ablation A10b (evaluate on loaded, high-current drive-cycle
    samples) is the closest available probe.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecm_pool import surfaces  # noqa: E402
from sop_trim_features import (N_FEATURES, NAMES_EXTRA,  # noqa: E402
                               TrimFeatures)

HERE = os.path.dirname(os.path.abspath(__file__))
RES_CSV = os.path.join(HERE, "uypydj_hppc_resistance.csv")
ECM_CSV = os.path.join(HERE, "uypydj_ecm.csv")
CACHE_T = os.path.join(HERE, "cache_t")
OUT_DIR = os.path.join(HERE, "cache", "trim")

TAU_A, TAU_B = 2.0, 10.0
BLOCK_S = 600.0            # one feature row per this much preceding drive cycle
MIN_REST_MULT = 5.0        # rest before the pulse, in units of tau2
SOC_MIN = 0.29
# Charge pulses above about SOC 0.9 run into the 4.2 V ceiling and the cycler
# truncates them, so their V_tau is the limit rather than the cell's response -
# 10.2 % of charge rows sit at exactly 4.2000 V. Those are not resistance
# measurements and would teach the trim that the cell is infinitely resistive.
V_CLAMP = 4.195
SOC_MAX_CHG = 0.92


def build_labels(cell, surf, res_csv=RES_CSV, ecm_csv=ECM_CSV,
                 direction="discharge"):
    """Measured dV at both horizons per pulse, plus the pooled nominal there.

    The rest-before-pulse gate needs `rest_before_s`, which lives in the ECM fit
    table rather than the resistance table, so the two are joined on
    (cycle, SOC, rate_rank). Without it the differencing argument - RC states are
    zero at t0, so dV is exact - would be asserted rather than enforced.
    """
    rest = {}
    for r in csv.DictReader(open(ecm_csv, encoding="utf-8")):
        if r["cell"] != cell or r["direction"] != direction:
            continue
        rest[(int(r["cycle"]), round(float(r["SOC"]), 3), r["rate_rank"])] = \
            float(r["rest_before_s"])

    rows = [r for r in csv.DictReader(open(res_csv, encoding="utf-8"))
            if r["protocol"] == cell and r["direction"] == direction
            and r["rate_rank"] in ("2", "3")]
    grp = collections.defaultdict(dict)
    for r in rows:
        k = (int(r["cycle"]), int(r["soc_group"]), r["rate_rank"])
        grp[k][float(r["tau_s"])] = r

    out, drop = [], collections.Counter()
    for (cyc, grp_i, rank), by_tau in grp.items():
        if TAU_A not in by_tau or TAU_B not in by_tau:
            drop["한 지평만"] += 1
            continue
        ra, rb = by_tau[TAU_A], by_tau[TAU_B]
        soc = float(ra["SOC"]); soh = float(ra["SOH"])
        I = float(ra["I_A"])
        if direction == "charge":
            if soc > SOC_MAX_CHG:
                drop["고 SOC"] += 1
                continue
            if (float(ra["V_tau_V"]) >= V_CLAMP
                    or float(rb["V_tau_V"]) >= V_CLAMP):
                drop["4.2V 클램프"] += 1
                continue
        elif soc < SOC_MIN:
            drop["저 SOC"] += 1
            continue
        rb_s = rest.get((cyc, round(soc, 3), rank))
        if rb_s is None:
            drop["휴지 정보 없음"] += 1
            continue
        if rb_s < MIN_REST_MULT * 8.0:
            drop["휴지 부족"] += 1
            continue
        th = surf.theta(soc, soh, I)
        if not bool(th["in_hull"][0]):
            drop["hull 밖"] += 1
            continue
        R0n = float(th["R0"][0]); R1n = float(th["R1"][0]); R2n = float(th["R2"][0])
        t1n = float(th["tau1"][0]); t2n = float(th["tau2"][0])
        out.append({
            "cell": cell, "cycle": cyc, "rank": rank, "SOC": soc, "SOH": soh,
            "I": I,
            "dV2": float(ra["V_tau_V"]) - float(ra["V_pre_V"]),
            "dV10": float(rb["V_tau_V"]) - float(rb["V_pre_V"]),
            # Nominal, frozen here so training never touches the interpolator.
            "nf2": R0n + R1n * (1 - np.exp(-TAU_A / t1n)),
            "ns2": R2n * (1 - np.exp(-TAU_A / t2n)),
            "nf10": R0n + R1n * (1 - np.exp(-TAU_B / t1n)),
            "ns10": R2n * (1 - np.exp(-TAU_B / t2n)),
        })
    return out, drop


def build_features(cell, surf_dis, surf_chg, cache=CACHE_T, max_runs=None):
    """One feature row per BLOCK_S of drive cycle, tagged with its cycle number."""
    z = np.load(os.path.join(cache, f"uypydj_{cell}_Fifteen_Drive_Cycles.npz"),
                allow_pickle=True)
    lens, files = z["lens"], z["files"]
    off = np.concatenate([[0], np.cumsum(lens)])
    out = []
    idx = range(len(lens)) if max_runs is None else \
        np.linspace(0, len(lens) - 1, max_runs).astype(int)
    from temp_defects import defective
    _BAD_DRIVE = {n for c, n in defective("drive") if c == cell}
    for k in idx:
        m = re.match(r"(\d+)\|", str(files[k]))
        if not m:
            continue
        cyc = int(m.group(1))
        if cyc in _BAD_DRIVE:
            # 온도 채널 결함 주행 런.  특징은 파일 전체에서 계산되므로
            # 한 구간만 깨져도 그 파일의 EW 통계가 통째로 오염된다.
            continue
        sl = slice(off[k], off[k] + lens[k])
        t, I, V, T, SOC, SOH, ok = (z[x][sl] for x in
                                    ("t", "I", "V", "T", "SOC", "SOH", "valid"))
        good = ok & np.isfinite(V) & np.isfinite(I) & np.isfinite(T) & np.isfinite(SOC)
        if good.sum() < 2000:
            continue
        t, I, V, T, SOC, SOH = (x[good] for x in (t, I, V, T, SOC, SOH))
        tf = TrimFeatures(surf_dis, surf_chg)
        next_mark = t[0] + BLOCK_S
        for j in range(1, len(t)):
            if tf.update(float(t[j] - t[j - 1]), float(I[j]), float(V[j]),
                         float(T[j]), float(SOC[j]), float(SOH[j])) is None:
                continue
            if t[j] >= next_mark:
                out.append({"cell": cell, "cycle": cyc,
                            "x": tf.vector(float(SOC[j]), float(SOH[j])),
                            "xe": tf.vector_extra(),
                            "exc": tf.excitation()})
                next_mark = t[j] + BLOCK_S
    return out


def pair(labels, feats, max_gap=200, blocks_per_label=12):
    """Each label takes the LAST `blocks_per_label` blocks of the closest
    preceding drive-cycle run.

    Not every block of that run. A 29.5 h run yields ~177 blocks, so pairing all
    of them repeats each label 177 times and silently weights the loss by how
    long the preceding drive cycle happened to be - a property of the test
    schedule, not of the cell. Taking a fixed count from the end keeps every
    label equally weighted AND keeps the features close in time to the pulse
    they are supposed to describe.

    12 blocks is 2 h at the 600 s block size, which is several EW window lengths
    and therefore several near-independent reads of the same statistic.
    """
    by_cyc = collections.defaultdict(list)
    for f in feats:
        by_cyc[f["cycle"]].append(f)
    fc = np.array(sorted(by_cyc))
    rows = []
    for L in labels:
        prev = fc[fc < L["cycle"]]
        if len(prev) == 0:
            continue
        c = int(prev[-1])
        if L["cycle"] - c > max_gap:
            continue
        blocks = by_cyc[c][-blocks_per_label:]
        if len(blocks) < blocks_per_label:
            continue                       # keep the weighting exactly uniform
        for f in blocks:
            rows.append({**L, "x": f["x"], "xe": f["xe"], "exc": f["exc"],
                         "feat_cycle": c})
    return rows


def to_arrays(rows):
    X = np.stack([r["x"] for r in rows]).astype(np.float32)
    XE = np.stack([r["xe"] for r in rows]).astype(np.float32)
    Y = np.stack([[r["dV2"], r["dV10"]] for r in rows]).astype(np.float32)
    NOM = np.stack([[r["nf2"], r["ns2"], r["nf10"], r["ns10"]]
                    for r in rows]).astype(np.float32)
    I = np.array([r["I"] for r in rows], np.float32)
    meta = {k: np.array([r[k] for r in rows]) for k in
            ("cell", "cycle", "SOC", "SOH", "rank", "exc", "feat_cycle")}
    return X, Y, NOM, I, meta, XE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE_T)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--blocks-per-label", type=int, default=12)
    ap.add_argument("--only", default=None)
    ap.add_argument("--direction", default="discharge",
                    choices=["discharge", "charge"])
    ap.add_argument("--pool-dir", default=None,
                    help="pooled surface directory; NOM comes from it, so a "
                         "different pool needs its own dataset")
    args = ap.parse_args()

    cells = ([args.only] if args.only else
             sorted({r["protocol"] for r in
                     csv.DictReader(open(RES_CSV, encoding="utf-8"))}))
    os.makedirs(args.out, exist_ok=True)
    print(f"{'셀':<20} {'라벨':>6} {'특징블록':>8} {'짝지음':>7} {'SOH 범위':>16}")
    for cell in cells:
        sd, sc = (surfaces(cell, args.pool_dir) if args.pool_dir
                  else surfaces(cell))          # POOLED, holdout removed
        labels, drop = build_labels(cell, sc if args.direction == "charge" else sd,
                                    direction=args.direction)
        feats = build_features(cell, sd, sc, args.cache, args.max_runs)
        rows = pair(labels, feats, blocks_per_label=args.blocks_per_label)
        if not rows:
            print(f"{cell:<20} {len(labels):>6} {len(feats):>8} {'0':>7}  "
                  f"제외 {dict(drop)}")
            continue
        X, Y, NOM, I, meta, XE = to_arrays(rows)
        np.savez(os.path.join(args.out, f"trim_{cell}.npz"),
                 X=X, Y=Y, NOM=NOM, I=I, XE=XE,
                 names_extra=np.array(NAMES_EXTRA),
                 **{f"m_{k}": v for k, v in meta.items()})
        s = meta["SOH"].astype(float)
        print(f"{cell:<20} {len(labels):>6} {len(feats):>8} {len(rows):>7}  "
              f"{s.min():.3f}~{s.max():.3f}")
        if drop:
            print(f"{'':<20} 라벨 제외: {dict(drop)}")


if __name__ == "__main__":
    main()
