"""SOH CNN 을 C 헤더로 내보낸다.

WHY HAND-WRITTEN C AND NOT ONNX / ST Edge AI
    모델이 Conv1d 2 개, Linear 2 개, ReLU, MaxPool, AdaptiveAvgPool 뿐이다.
    ST Edge AI 런타임을 끌어오면 그 자체로 수십 KB 가 붙고, 무엇이 얼마나 걸리는지
    런타임 안에 가려진다. 이 벤치의 목적이 "무엇이 지배하는가" 이므로 직접 쓴다.
    같은 이유로 SOP 도 직접 썼다.

시드 3 개를 접지 않는다
    선형이 아니므로 가중치 평균이 예측 평균과 다르다. 채택 성적(RMSE 0.0128)은
    3 시드 예측 평균이므로, MCU 도 3 개를 다 돌려 평균해야 같은 것을 잰다.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def carr_i8(name, q, per=16):
    q = np.asarray(q, np.int8).ravel()
    s = [f"static const int8_t {name}[{q.size}] = {{"]
    for i in range(0, q.size, per):
        s.append("  " + ", ".join(str(int(x)) for x in q[i:i + per]) + ",")
    s.append("};")
    return "\n".join(s)


def quant_chan(W):
    """출력 채널별 대칭 int8.  [시드][출력][...] 이므로 앞 두 축을 살린다.

    6-fold 로 재보면 채널별 int8 이 RMSE 0.0128 -> 0.0129 (+1.7e-4) 다.
    텐서 하나로 묶으면 +5.3e-4 이고, int4 로 내리면 0.0216 으로 무너진다.
    """
    A = np.asarray(W, np.float32)
    ax = tuple(range(2, A.ndim))
    s = np.abs(A).max(axis=ax, keepdims=True) / 127.0
    s = np.where(s == 0, 1e-12, s)
    return np.clip(np.round(A / s), -127, 127).astype(np.int8), s.reshape(A.shape[:2])


def carr(name, a, per=8):
    f = np.asarray(a, np.float32).ravel()
    s = [f"static const float {name}[{f.size}] = {{"]
    for i in range(0, f.size, per):
        s.append("  " + ", ".join(f"{x:.7g}f" for x in f[i:i + per]) + ",")
    s.append("};")
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs_soh_cnn",
                                                   "soh_CC.pt"))
    ap.add_argument("--int8", action="store_true",
                    help="Conv/Linear 가중치를 채널별 int8 로 (편향은 float 유지)")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "mcu",
                                                  "soh_tables.h"))
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    seeds = ck["seeds"]
    n_seed = len(seeds)
    k = seeds[0]
    shapes = {kk: tuple(v.shape) for kk, v in k.items()}
    n_in = ck["n_in"]

    def stack(key):
        return np.stack([s[key].numpy() for s in seeds])

    body = [f"#define SOH_INT8 {1 if a.int8 else 0}"]
    tot = 0
    for kk in k:
        nm = "soh_" + kk.replace(".", "_")
        A = stack(kk)
        if a.int8 and "bias" not in kk:
            Q, S = quant_chan(A)
            body.append(carr_i8(nm + "_q", Q)); body.append(carr(nm + "_s", S))
            tot += Q.size + S.size * 4
        else:
            body.append(carr(nm, A)); tot += A.size * 4
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(f"""/* 자동 생성 — analysis/export_soh_mcu.py.  손으로 고치지 말 것.
 *
 * SOH dQ/dV CNN.  홀드아웃 {ck['holdout']}, 시드 {n_seed} 개, 시드당
 * {ck['nparam']:,} 파라미터.  가중치는 [시드][...] 순서로 쌓았다 — 비선형이므로
 * 가중치를 평균하면 안 되고 시드마다 돌려 예측을 평균해야 한다.
 *
 * 가중치 {tot/1024:.1f} KB
 */
#ifndef SOH_TABLES_H
#define SOH_TABLES_H

#include <stdint.h>

#define SOH_NSEED {n_seed}
#define SOH_NIN {n_in}
#define SOH_CH {shapes['conv.0.weight'][0]}
#define SOH_K {shapes['conv.0.weight'][2]}
#define SOH_POOL 8
#define SOH_HID {shapes['head.2.weight'][0]}

{carr('soh_mu', ck['mu'])}

{carr('soh_sd', ck['sd'])}

{chr(10).join(body)}

#endif /* SOH_TABLES_H */
""")
    print(f"  {a.out}")
    print(f"  시드 {n_seed}, 시드당 {ck['nparam']:,} 파라미터, 총 {tot/1024:.1f} KB")
    for kk, v in shapes.items():
        print(f"   {kk:<20}{v}")


if __name__ == "__main__":
    main()
