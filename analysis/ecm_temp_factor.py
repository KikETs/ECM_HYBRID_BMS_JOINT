"""Temperature correction factor g(SOC, |I|, T) = R(T) / R(25 degC), measured.

WHY A TABLE AND NOT AN ARRHENIUS COEFFICIENT
    The obvious form is R(T) = R_25 * exp(Ea/Rg * (1/T - 1/298)). Fitting that to
    the Mendeley HPPC gives a good correlation (r = 0.95-0.99) but a coefficient
    that is not a constant:

        by current   3-6 A  24.6 kJ/mol | 12-18 A  21.2 | 24-36 A  17.0
        by range     -20..0 C  28.7     | 0..40 C  18.2  | 10..40 C  14.3

    Ea nearly doubles between the warm and cold halves of the range, so a single
    Arrhenius term is a compromise everywhere and wrong at the ends. Since the
    ratio is MEASURED at six temperatures, it is tabulated directly instead. No
    functional form is imposed inside the measured range.

    The Ea numbers are still written out, because they are the honest way to
    extrapolate beyond -20..40 degC if that is ever needed, and because they are
    what makes the cross-dataset check below comparable.

CROSS-DATASET AGREEMENT, WHICH IS WHAT LICENSES COMBINING THE SOURCES
    At 30-36 A over 10-25 degC, three independent measurements agree:

        Mendeley HPPC (this file)   R10/R25 = 1.353
        RPCWBY Test#1               R10/R25 = 1.335
        RPCWBY Test#2               R10/R25 = 1.306

    Different cells, different labs' protocols, 4 % apart. The absolute 25 degC
    resistance agrees too (13.45 vs 13.06 mOhm). An earlier apparent 25 %
    disagreement was an artifact of comparing an Ea fitted over 10-40 degC
    against one fitted over 10-25 degC.

WHAT IS CORRECTED AND WHAT IS NOT
    The factor multiplies R0, R1 and R2 together. It is NOT applied per
    component, because component-wise Ea is not identifiable at the currents
    that matter: at 18 A the fitted R1 falls to 1.75 mOhm and its apparent Ea
    comes out NEGATIVE (-1.4 kJ/mol), which is a fitting artifact, not a cell.

    Time constants are left alone. Measured tau1 across -20..40 degC moves
    0.22 -> 0.10 -> 0.14 s and tau2 6.7 -> 11.2 -> 9.2 s, i.e. non-monotonically
    and weakly; there is no defensible trend to model.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "mendeley_ecm.csv")
OUT = os.path.join(HERE, "ecm_temp_factor.csv")

T_REF = 25.0
RG = 8.314
SOC_EDGES = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.01])
I_EDGES = np.array([0.0, 8.0, 22.0, 100.0])       # 3-6 A | 12-18 A | 24-36 A


def reff10(r):
    """The 10 s effective resistance the factor is defined on."""
    return (float(r["R0_mOhm"])
            + float(r["R1_mOhm"]) * (1 - np.exp(-10 / float(r["tau1_s"])))
            + float(r["R2_mOhm"]) * (1 - np.exp(-10 / float(r["tau2_s"]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.src, encoding="utf-8"))
            if r["direction"] == "discharge"]
    soc = np.array([float(r["SOC"]) for r in rows])
    cur = np.array([abs(float(r["I_A"])) for r in rows])
    tset = np.array([int(r["temp_set_C"]) for r in rows])
    tcell = np.array([float(r["T_cell_C"]) for r in rows])
    R = np.array([reff10(r) for r in rows])

    out = []
    for si in range(len(SOC_EDGES) - 1):
        ms = (soc >= SOC_EDGES[si]) & (soc < SOC_EDGES[si + 1])
        for ii in range(len(I_EDGES) - 1):
            m = ms & (cur >= I_EDGES[ii]) & (cur < I_EDGES[ii + 1])
            if m.sum() < 3:
                continue
            ref = R[m & (tset == 25)]
            if len(ref) == 0:
                continue
            r25 = float(np.median(ref))
            # Ea over the whole measured range, for extrapolation only.
            ea = ""
            if len(np.unique(tset[m])) >= 3:
                x = 1.0 / (tcell[m] + 273.15)
                y = np.log(R[m])
                ea = round(float(np.polyfit(x, y, 1)[0] * RG / 1000.0), 2)
            for T in sorted(set(tset[m])):
                mm = m & (tset == T)
                out.append({
                    "SOC_lo": SOC_EDGES[si], "SOC_hi": SOC_EDGES[si + 1],
                    "I_lo": I_EDGES[ii], "I_hi": I_EDGES[ii + 1],
                    "temp_set_C": T,
                    "T_cell_C": round(float(np.median(tcell[mm])), 2),
                    "n": int(mm.sum()),
                    "R_mOhm": round(float(np.median(R[mm])), 4),
                    "g": round(float(np.median(R[mm]) / r25), 5),
                    "Ea_kJ_mol": ea,
                })

    if not out:
        sys.exit("생성된 행이 없습니다")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"{args.out}: {len(out)}행")

    print(f"\n  g = R(T)/R(25 °C)   (SOC 0.45~0.60)")
    print(f"  {'T[C]':>6} " + "".join(f"{f'{lo:.0f}~{hi:.0f}A':>10}"
                                      for lo, hi in zip(I_EDGES[:-1], I_EDGES[1:])))
    for T in (-20, -10, 0, 10, 25, 40):
        line = f"  {T:>6} "
        for lo, hi in zip(I_EDGES[:-1], I_EDGES[1:]):
            v = [r for r in out if r["temp_set_C"] == T and r["I_lo"] == lo
                 and abs(r["SOC_lo"] - 0.45) < 1e-9]
            line += f"{v[0]['g']:>10.3f}" if v else f"{'-':>10}"
        print(line)
    eas = [r["Ea_kJ_mol"] for r in out if r["Ea_kJ_mol"] != ""]
    if eas:
        print(f"\n  Ea (외삽용, 전 구간 적합): {min(eas):.1f} ~ {max(eas):.1f} kJ/mol")


if __name__ == "__main__":
    main()
