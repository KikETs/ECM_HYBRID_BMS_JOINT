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
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecm_pool import surfaces                      # noqa: E402
from sop_trim import TrimLinear, TrimMLP, decode, KF_SPAN, KS_SPAN  # noqa: E402

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


def carr(name, a, per=8):
    flat = np.asarray(a, np.float32).ravel()
    s = [f"static const float {name}[{flat.size}] = {{"]
    for i in range(0, flat.size, per):
        s.append("  " + ", ".join(f"{x:.7g}f" for x in flat[i:i + per]) + ",")
    s.append("};")
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="CC",
                    help="풀 변형. 배치용이라면 전 셀 풀이 맞지만, 이 벤치는 "
                         "연산 비용을 재는 것이므로 어느 것이든 크기가 같다")
    ap.add_argument("--trim", default="runs_trim_v2")
    ap.add_argument("--trim-chg", default="runs_trim_chg_v2")
    ap.add_argument("--ns", type=int, default=32)
    ap.add_argument("--nh", type=int, default=16)
    ap.add_argument("--int8", action="store_true",
                    help="ECM 격자를 int8 로 저장한다 (4 배 축소).")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "mcu",
                                                  "sop_tables.h"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    sd, sc = surfaces(a.holdout)
    Gd, Gc = grid(sd, a.ns, a.nh), grid(sc, a.ns, a.nh)
    Od = ocv_grid(sd, a.ns, a.nh)

    # 트림 가중치: 시드 평균을 하나로 접는다 (선형이므로 정확히 접힌다)
    def fold(runs):
        # rung 은 디렉터리에서 찾는다 — A3 를 하드코딩하면 채택 구성(A8)을
        # 내보내려고 파일 이름을 바꾸게 된다 (eval_sop_amps 와 같은 문제).
        import glob as _g
        _hit = [x for x in sorted(_g.glob(os.path.join(
            HERE, runs, f"model_A*_{a.holdout}.pt")))]
        if not _hit:
            raise FileNotFoundError(
                f"{runs} 에 model_A*_{a.holdout}.pt 가 없다")
        if len(_hit) > 1:
            raise RuntimeError(f"{runs} 에 rung 이 여럿이다: {_hit}")
        ck = torch.load(_hit[0],
                        map_location="cpu", weights_only=False)
        W, B, MU, SD = [], [], [], []
        for st in ck["seeds"]:
            cls = TrimLinear if st["cls"] == "TrimLinear" else TrimMLP
            m = cls(st["n_in"]); m.load_state_dict(st["model"])
            lin = [x for x in m.modules() if isinstance(x, torch.nn.Linear)]
            if len(lin) != 1:
                raise SystemExit("  선형 트림만 접을 수 있다 (rung A3)")
            W.append(lin[0].weight.detach().numpy())
            B.append(lin[0].bias.detach().numpy())
            MU.append(st["mu"]); SD.append(st["sd"])
        return (np.mean(W, 0), np.mean(B, 0), np.mean(MU, 0), np.mean(SD, 0))

    Wd, Bd, MUd, SDd = fold(a.trim)
    Wc, Bc, MUc, SDc = fold(a.trim_chg)

    if a.int8:
        Qd, Sd = quant_grid(Gd); Qc, Sc = quant_grid(Gc)
        grid_block = (f"#define SOP_GRID_INT8 1\n\n"
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
