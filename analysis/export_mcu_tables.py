"""하이브리드 SOP 를 MCU 로 내보낸다 — 표와 트림 가중치를 C 헤더로.

WHY THE GRID STORES D(tau) AND NOT THE RC PARAMETERS
    Python 쪽 `ECMSurface.theta` 는 R0/R1/tau1/R2/tau2 를 rank 축에서 각각
    보간한다. sop_hybrid_spec.md 26.3 이 그것의 결함을 측정했다 — tau2 가 상한
    3000 s 에 붙어 발산한 적합들 사이를 보간하면 어느 rank 에도 없는 조합이 나오고,
    전류가 커질 때 저항이 **오르는** 비물리적 구간이 생긴다.

    여기서는 rank 마다 D(2 s), D(10 s) 를 **미리 계산해서** 저장하고 그것을
    보간한다. 발산이 각 rank 안에 갇히므로 그 결함이 구조적으로 불가능해진다.
    저장량도 5 개가 아니라 2 개라 절반 이하다.

격자 크기는 측정으로 골랐다
    원 표가 SOC 20 x SOH 17 점이므로 그보다 촘촘할 이유가 없다. Python 산점 보간을
    기준으로 잰 10 s 등가저항 상대오차:

        24x12  중앙 0.65 %  95 % 4.46 %   18 KB
        32x16  중앙 0.30 %  95 % 2.88 %   32 KB     <- 채택
        48x24  중앙 0.20 %  95 % 1.75 %   72 KB

    32x16 이 무릎이다. 실제 SOP 라벨 657 행에서 Python 대비 차이는 중앙 0.217 A,
    낙관율 70.6 -> 70.9 %, 최악 20.35 -> 19.53 A — 재학습 산포에 묻히는 수준이다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _heavy():
    """Import torch and the trim model classes on first use.

    Everything this module validates about its arguments -- the rung the
    directory actually holds, whether an all-cell fit exists -- is decidable
    from file names alone.  Keeping torch out of module scope lets those
    guards fire (and be tested) on a machine that has no torch and no raw
    data, which is what CI is.
    """
    global torch, surfaces, TrimLinear, TrimMLP, decode, KF_SPAN, KS_SPAN
    import torch                                          # noqa: F401
    from ecm_pool import surfaces                          # noqa: F401
    from sop_trim import (TrimLinear, TrimMLP, decode,     # noqa: F401
                          KF_SPAN, KS_SPAN)


def resolve_rung(runs, holdout, want=None):
    """Return (checkpoint path, rung) for a run directory, by file name only.

    Raises if the directory holds no such fit, holds several rungs, or holds
    a rung other than the one the caller asked for -- exporting a mismatch
    would put one rung's weights on the board while the stage graph claims
    another.
    """
    import glob as _g
    hit = sorted(_g.glob(os.path.join(HERE, runs, f"model_A*_{holdout}.pt")))
    if not hit:
        raise FileNotFoundError(
            f"{runs} holds no model_A*_{holdout}.pt")
    if len(hit) > 1:
        raise RuntimeError(f"{runs} holds more than one rung: {hit}")
    found = os.path.basename(hit[0]).split("_")[1]
    if want is not None and found != want:
        raise SystemExit(
            f"  --rung {want} was requested but {runs} holds {found}. "
            f"Exporting it would put {found} weights on the board while "
            f"the stage graph claims {want}.")
    return hit[0], found

HERE = os.path.dirname(os.path.abspath(__file__))
TAU_A, TAU_B, TAU2_REF = 2.0, 10.0, 8.0
SOC_LO, SOC_HI = 0.10, 1.00
SOH_LO, SOH_HI = 0.70, 1.00


def grid(surf, ns, nh):
    SOC = np.linspace(SOC_LO, SOC_HI, ns)
    SOH = np.linspace(SOH_LO, SOH_HI, nh)
    G = np.zeros((len(surf.ranks), ns, nh, 2), np.float32)   # mOhm
    for k, fld in enumerate(surf.ranks):
        for i, s0 in enumerate(SOC):
            for j, h0 in enumerate(SOH):
                v = np.atleast_2d(fld(s0, h0)[0])[0]
                d = lambda t: (v[0] + v[1] * (1 - np.exp(-t / v[2]))
                               + v[3] * (1 - np.exp(-t / v[4])))
                G[k, i, j] = [d(TAU_A), d(TAU_B)]
    return G


def ocv_grid(surf, ns, nh):
    SOC = np.linspace(SOC_LO, SOC_HI, ns)
    SOH = np.linspace(SOH_LO, SOH_HI, nh)
    O = np.zeros((ns, nh, 2), np.float32)                     # V, V
    for i, s0 in enumerate(SOC):
        for j, h0 in enumerate(SOH):
            v, _ = surf.ocv(s0, h0)
            M, _ = surf.hyst_M(s0, h0)
            O[i, j] = [float(np.atleast_1d(v)[0]), M]
    return O


def carr_i8(name, q, per=16):
    q = np.asarray(q, np.int8).ravel()
    s = [f"static const int8_t {name}[{q.size}] = {{"]
    for i in range(0, q.size, per):
        s.append("  " + ", ".join(str(int(x)) for x in q[i:i + per]) + ",")
    s.append("};")
    return "\n".join(s)


def quant_grid(G):
    """rank x 지평 별 대칭 int8.  스케일은 그 칸의 최대 절대값 / 127.

    조회표는 신경망 가중치가 아니라 매끄러운 함수다. 축을 잘게 나눌수록 스케일이
    좁아져 정확해지는데, rank x 지평(8 칸)이면 상대오차 중앙 0.23 % / 95 % 0.77 %
    로 **격자화 오차(0.30 % / 2.88 %)보다 작다** — 양자화가 격자화에 묻힌다.
    """
    Q = np.zeros(G.shape, np.int8)
    S = np.zeros((G.shape[0], 2), np.float32)
    for k in range(G.shape[0]):
        for c in range(2):
            sc = float(np.abs(G[k, ..., c]).max()) / 127.0
            S[k, c] = sc
            Q[k, ..., c] = np.clip(np.round(G[k, ..., c] / sc), -127, 127)
    return Q, S


def cfloat(x):
    """A C float literal that is always a *floating* constant.

    "%.7g" % 0.0 is "0", and "0f" is an integer constant with a float
    suffix, which is invalid C.  gcc and arm-none-eabi-gcc both reject it:
        error: invalid suffix "f" on integer constant
    Any exactly-zero weight therefore produced a header that would not
    compile.  A8 has 27 of them (the eleven masked feature columns of
    trim_mu), so the adopted configuration's header never built.
    """
    s = f"{float(x):.7g}"
    if not any(c in s for c in ".eE") and s.lstrip("-").isdigit():
        s += ".0"
    return s + "f"


def carr(name, a, per=8):
    flat = np.asarray(a, np.float32).ravel()
    s = [f"static const float {name}[{flat.size}] = {{"]
    for i in range(0, flat.size, per):
        s.append("  " + ", ".join(cfloat(x) for x in flat[i:i + per]) + ",")
    s.append("};")
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="CC",
                    help="Which leave-one-cell-out fold to export.  This is a "
                         "BENCHMARK artifact: the exported weights are the "
                         "model fitted without this cell, not a model fitted "
                         "on all six.  Sizes and timings are identical for "
                         "any fold, so it is fine for measuring cost, but a "
                         "header exported this way is not the deployment "
                         "model.  See --deployment.")
    ap.add_argument("--deployment", action="store_true",
                    help="Refuse to export unless the trim directory holds an "
                         "all-cell fit (model_A*_ALL.pt).  Use this for the "
                         "header that actually ships.")
    ap.add_argument("--rung", default=None,
                    help="Accepted and checked against the rung found in the "
                         "trim directory.  It does not select anything - the "
                         "rung comes from the directory - but the stage graph "
                         "passes it, and silently ignoring it would let the "
                         "graph claim A8 while exporting A3.")
    # Default to the ADOPTED configuration (A8).  These used to default to
    # runs_trim_v2 / runs_trim_chg_v2, which are A3 - the superseded
    # comparison group - so an operator running the exporter with no
    # arguments shipped the wrong model.
    ap.add_argument("--trim", default="runs_trim_a8")
    ap.add_argument("--trim-chg", default="runs_trim_a8_chg")
    ap.add_argument("--ns", type=int, default=32)
    ap.add_argument("--nh", type=int, default=16)
    ap.add_argument("--int8", action="store_true",
                    help="ECM 격자를 int8 로 저장한다 (4 배 축소).")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "mcu",
                                                  "sop_tables.h"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    if a.deployment:
        import glob as _g
        for d in (a.trim, a.trim_chg):
            if not _g.glob(os.path.join(HERE, d, "model_A*_ALL.pt")):
                raise SystemExit(
                    f"  --deployment: {d} has no all-cell fit "
                    f"(model_A*_ALL.pt).  Train one before exporting a "
                    f"header that ships; exporting a leave-one-out fold as "
                    f"the deployment model is not defensible.")
        a.holdout = "ALL"

    # Resolve the rung from file names before anything heavy loads: a wrong
    # --rung must not cost a grid build to discover.
    for d in (a.trim, a.trim_chg):
        resolve_rung(d, a.holdout, a.rung)

    _heavy()
    sd, sc = surfaces(a.holdout)
    Gd, Gc = grid(sd, a.ns, a.nh), grid(sc, a.ns, a.nh)
    Od = ocv_grid(sd, a.ns, a.nh)

    # 트림 가중치: 시드 평균을 하나로 접는다 (선형이므로 정확히 접힌다)
    def fold(runs):
        # The rung comes from the directory.  Hard-coding A3 would mean
        # renaming files to export the adopted configuration (A8) -- the same
        # failure eval_sop_amps had.
        _path, _ = resolve_rung(runs, a.holdout, a.rung)
        ck = torch.load(_path, map_location="cpu", weights_only=False)
        W, B, MU, SD = [], [], [], []
        for st in ck["seeds"]:
            cls = TrimLinear if st["cls"] == "TrimLinear" else TrimMLP
            m = cls(st["n_in"]); m.load_state_dict(st["model"])
            lin = [x for x in m.modules() if isinstance(x, torch.nn.Linear)]
            if len(lin) != 1:
                raise SystemExit("  only a linear trim can be folded (rung A3)")
            W.append(lin[0].weight.detach().numpy())
            B.append(lin[0].bias.detach().numpy())
            MU.append(st["mu"]); SD.append(st["sd"])
        return (np.mean(W, 0), np.mean(B, 0), np.mean(MU, 0), np.mean(SD, 0))

    Wd, Bd, MUd, SDd = fold(a.trim)
    Wc, Bc, MUc, SDc = fold(a.trim_chg)

    if a.int8:
        Qd, Sd = quant_grid(Gd); Qc, Sc = quant_grid(Gc)
        grid_block = ("#define SOP_GRID_INT8 1\n\n"
                      + carr_i8('sop_grid_dis_q', Qd) + "\n\n"
                      + carr('sop_grid_dis_s', Sd) + "\n\n"
                      + carr_i8('sop_grid_chg_q', Qc) + "\n\n"
                      + carr('sop_grid_chg_s', Sc))
        n_ecm = Qd.size + Qc.size + (Sd.size + Sc.size) * 4
    else:
        grid_block = ("#define SOP_GRID_INT8 0\n\n"
                      + carr('sop_grid_dis', Gd) + "\n\n"
                      + carr('sop_grid_chg', Gc))
        n_ecm = (Gd.size + Gc.size) * 4
    n_ocv = Od.size * 4
    n_trim = (Wd.size + Bd.size + MUd.size + SDd.size) * 2 * 4
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(f"""/* 자동 생성 — analysis/export_mcu_tables.py.  손으로 고치지 말 것.
 *
 * 하이브리드 SOP 의 MCU 자산.  격자는 D(2 s), D(10 s) 를 저장한다 — RC 파라미터가
 * 아니다.  이유는 sop_hybrid_spec.md 26.3 (rank 축에서 RC 를 보간하면 발산한 tau2
 * 적합들 사이에서 비물리적 값이 나온다).
 *
 * 격자 {a.ns} x {a.nh}, SOC {SOC_LO}~{SOC_HI}, SOH {SOH_LO}~{SOH_HI}
 * ECM {n_ecm/1024:.1f} KB + OCV {n_ocv/1024:.1f} KB + 트림 {n_trim/1024:.2f} KB
 */
#ifndef SOP_TABLES_H
#define SOP_TABLES_H

#include <stdint.h>

#define SOP_NS {a.ns}
#define SOP_NH {a.nh}
#define SOP_NRANK {len(sd.ranks)}
#define SOP_SOC_LO {SOC_LO}f
#define SOP_SOC_HI {SOC_HI}f
#define SOP_SOH_LO {SOH_LO}f
#define SOP_SOH_HI {SOH_HI}f
#define SOP_TAU_A {TAU_A}f
#define SOP_TAU_B {TAU_B}f
#define SOP_TAU2 {TAU2_REF}f
#define SOP_KF_SPAN {KF_SPAN}f
#define SOP_KS_SPAN {KS_SPAN}f
#define SOP_NFEAT {Wd.shape[1]}

{carr('sop_rank_i_dis', sd.rank_I)}

{carr('sop_rank_i_chg', sc.rank_I)}

/* [rank][soc][soh][{{D2, D10}}]  단위 mOhm */
{grid_block}

/* [soc][soh][{{OCV_V, hyst_half_V}}] */
{carr('sop_ocv', Od)}

/* 트림: 시드 평균을 접은 12 -> 2 선형 */
{carr('trim_w_dis', Wd)}
{carr('trim_b_dis', Bd)}
{carr('trim_mu_dis', MUd)}
{carr('trim_sd_dis', SDd)}
{carr('trim_w_chg', Wc)}
{carr('trim_b_chg', Bc)}
{carr('trim_mu_chg', MUc)}
{carr('trim_sd_chg', SDc)}

#endif /* SOP_TABLES_H */
""")
    print(f"  {a.out}")
    print(f"  ECM 격자 {n_ecm/1024:.1f} KB  OCV {n_ocv/1024:.1f} KB  "
          f"트림 {n_trim/1024:.2f} KB   합계 {(n_ecm+n_ocv+n_trim)/1024:.1f} KB")
    print(f"  트림 형상 W{Wd.shape} B{Bd.shape}")


if __name__ == "__main__":
    main()
