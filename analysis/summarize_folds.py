"""Collect the leave-one-cell-out results of both arms into one table.

WHY BOTH ARMS ARE NOT PUT IN ONE COLUMN
    The full-AI arm is scored on drive-cycle voltage RMSE; the hybrid arm on
    measured HPPC pulse dV RMSE. They answer the same question but are not the
    same number, and putting them side by side without saying so would invite
    exactly the comparison the data does not support. Rung A13 exists to make
    them commensurate; until it runs, they stay in separate blocks.
"""
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def full_ai():
    rows = {}
    for f in sorted(glob.glob(os.path.join(HERE, "runs_soh", "summary_f_*.json"))):
        m = re.match(r"summary_f_(.+)_(M\d)\.json", os.path.basename(f))
        if not m:
            continue
        d = json.load(open(f))
        rows.setdefault(m.group(1), {})[m.group(2)] = d["best_val_rmse_mV"]
    return rows


def sizes():
    rows = {}
    for f in sorted(glob.glob(os.path.join(HERE, "runs_soh", "summary_s*.json"))):
        m = re.match(r"summary_s(\d+)_(.+)_(M\d)\.json", os.path.basename(f))
        if not m:
            continue
        d = json.load(open(f))
        rows.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = \
            (d["best_val_rmse_mV"], d["params"])
    return rows


def hybrid():
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, "runs_trim", "summary_*.json"))):
        r = os.path.basename(f)[8:-5]
        out[r] = {x["cell"]: x for x in json.load(open(f))}
    return out


def main():
    fa = full_ai()
    print("=== Full AI 팔 : 드라이브사이클 전압 RMSE, 은닉 256 ===")
    print(f"  {'홀드아웃':<22} {'M1 (SOH)':>10} {'M2 (문맥 z)':>12} {'개선':>8}")
    imp = []
    for c in sorted(fa):
        a, b = fa[c].get("M1"), fa[c].get("M2")
        if a and b:
            imp.append(1 - b / a)
            print(f"  {c:<22} {a:>9.2f}m {b:>11.2f}m {(1-b/a)*100:>+7.1f}%")
        else:
            print(f"  {c:<22} {a if a else '진행중':>10} {b if b else '진행중':>12}")
    if imp:
        print(f"  {'평균':<22} {'':>10} {'':>12} {np.mean(imp)*100:>+7.1f}%"
              f"   (n={len(imp)}/6)")

    sz = sizes()
    if sz:
        print("\n=== 크기 축 ===")
        print(f"  {'은닉':>5} {'홀드아웃':<22} {'M1':>9} {'M2':>9} {'개선':>8} "
              f"{'파라미터':>10}")
        for (h, c) in sorted(sz):
            d = sz[(h, c)]
            a = d.get("M1"); b = d.get("M2")
            if a and b:
                print(f"  {h:>5} {c:<22} {a[0]:>8.2f}m {b[0]:>8.2f}m "
                      f"{(1-b[0]/a[0])*100:>+7.1f}% {b[1]:>10,}")

    hy = hybrid()
    if hy and "A0" in hy:
        print("\n=== Hybrid 팔 : 측정 HPPC 펄스 dV RMSE ===")
        cells = sorted(hy["A0"])
        rungs = [r for r in ("A0", "A3", "A4", "A5") if r in hy]
        print(f"  {'홀드아웃':<22} " + "".join(f"{r:>10}" for r in rungs))
        for c in cells:
            line = f"  {c:<22} "
            for r in rungs:
                v = hy[r][c].get("model", hy[r][c].get("A0"))
                line += f"{v:>9.1f}m"
            print(line)
        base = np.mean([hy["A0"][c]["A0"] for c in cells])
        line = f"  {'평균 개선':<22} "
        for r in rungs:
            if r == "A0":
                line += f"{'기준':>10}"
            else:
                m = np.mean([hy[r][c]["model"] for c in cells])
                line += f"{(1-m/base)*100:>+9.1f}%"
        print(line)

    print("\n  주의: 두 팔은 지표가 다르다 (드라이브사이클 전압 vs 펄스 dV).")
    print("        같은 축의 비교는 rung A13에서만 성립한다.")


if __name__ == "__main__":
    main()
