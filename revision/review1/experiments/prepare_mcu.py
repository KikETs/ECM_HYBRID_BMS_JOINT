import re
from pathlib import Path
import sys

import numpy as np

RUN = Path(__file__).resolve().parents[1]
REPO = RUN / "repo"
MCU = REPO / "mcu"
sys.path.insert(0, str(REPO / "analysis"))
from ecm_pool import surfaces
from export_mcu_tables import grid

def replace_once(path, old, new):
    src = path.read_text()
    if src.count(old) != 1:
        raise ValueError((path, old, src.count(old)))
    path.write_text(src.replace(old, new))

sd, sc = surfaces("ALL")
def export_hull():
    soc, soh = np.linspace(.1, 1., 32), np.linspace(.7, 1., 16)
    S, H = np.meshgrid(soc, soh, indexing="ij")
    decls = ["/* Generated conservatively from ALL deployment ECM/OCV hulls. */"]
    for name, surface, definition in (("dis", sd, "sop_grid_dis"), ("chg", sc, "sop_grid_chg")):
        header = (MCU / "sop_tables.h").read_text()
        values = re.search(r"static const float " + definition + r"\[\d+\] = \{(.*?)\};", header, re.S)
        ref = np.array([float(t) for t in re.findall(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?(?=f[,\s])", values[1])])
        generated = grid(surface, 32, 16)
        np.testing.assert_allclose(generated.reshape(-1), ref, atol=0.0003, rtol=0)
        bits = np.zeros(S.shape, np.uint8)
        for n, field in enumerate(surface.ranks):
            bits |= field(S.ravel(), H.ravel())[1].reshape(S.shape).astype(np.uint8) << n
        bits |= sd.ocv_field(S.ravel(), H.ravel())[1].reshape(S.shape).astype(np.uint8) << 4
        lines = [", ".join(str(int(v)) for v in row) for row in bits]
        decls.append(f"static const uint8_t sop_hull_{name}[SOP_NS*SOP_NH] = {{\n  " + ",\n  ".join(lines) + "\n};")
    (MCU / "sop_review_hull.h").write_text("\n".join(decls) + "\n")

export_hull()
status_decl = """
typedef enum {
  SOP_REVIEW_OK = 0, SOP_REVIEW_INPUT = 1, SOP_REVIEW_HULL = 2,
  SOP_REVIEW_RESISTANCE = 3, SOP_REVIEW_ITERATION_CAP = 4, SOP_REVIEW_RANGE = 5
} sop_review_status_t;
sop_review_status_t sop_review_solve(sop_dir_t dir, float soc, float soh,
                     float v_pre, float v_limit, float tau_s, float kf, float ks,
                     float *answer, uint32_t *iterations);
"""
header = MCU / "sop_core.h"
src = MCU / "sop_core.c"
main = MCU / "fw_sop" / "Src" / "main.c"
replace_once(header, "\n#endif\n", status_decl + "\n#endif\n")
replace_once(src, '#include <math.h>\n', '#include <math.h>\n#include "sop_review_hull.h"\n')
solver = """
static int review_in_hull(sop_dir_t dir, float soc, float soh, float current)
{
  const uint8_t *mask = dir == SOP_CHARGE ? sop_hull_chg : sop_hull_dis;
  const float *ranks = dir == SOP_CHARGE ? sop_rank_i_chg : sop_rank_i_dis;
  int i0, i1, j0, j1, k0, k1;
  float fi, fj, fk;
  frac_index(soc, SOP_SOC_LO, SOP_SOC_HI, SOP_NS, &i0, &i1, &fi);
  frac_index(soh, SOP_SOH_LO, SOP_SOH_HI, SOP_NH, &j0, &j1, &fj);
  rank_index(ranks, fabsf(current), &k0, &k1, &fk);
  unsigned required = (1U << 4) | (1U << k0) | (1U << k1);
  unsigned valid = mask[i0*SOP_NH+j0] & mask[i0*SOP_NH+j1]
                 & mask[i1*SOP_NH+j0] & mask[i1*SOP_NH+j1];
  return (valid & required) == required;
}

sop_review_status_t sop_review_solve(sop_dir_t dir, float soc, float soh,
                     float v_pre, float v_limit, float tau_s, float kf, float ks,
                     float *answer, uint32_t *iterations)
{
  *answer = 0.0f;
  *iterations = 0U;
  if (!isfinite(soc) || !isfinite(soh) || !isfinite(v_pre) || !isfinite(v_limit)
      || !isfinite(kf) || !isfinite(ks) || !(kf > 0.0f && ks > 0.0f)
      || !(tau_s == 2.0f || tau_s == 10.0f)
      || !(dir == SOP_CHARGE || dir == SOP_DISCHARGE))
    return SOP_REVIEW_INPUT;
  if (soc < SOP_SOC_LO || soc > SOP_SOC_HI || soh < SOP_SOH_LO || soh > SOP_SOH_HI)
    return SOP_REVIEW_HULL;
  int charge = dir == SOP_CHARGE;
  float gap = v_limit - v_pre;
  if (!(charge ? gap > 0.0f : gap < 0.0f)) return SOP_REVIEW_RANGE;
  float trial = charge ? 5.0f : -12.0f;
  const float low = charge ? 0.05f : -400.0f;
  const float high = charge ? 400.0f : -0.1f;
  while (*iterations < SOP_MAX_ITER)
  {
    if (!review_in_hull(dir, soc, soh, trial)) return SOP_REVIEW_HULL;
    float resistance = sop_r_eff(dir, soc, soh, trial, tau_s, kf, ks);
    ++*iterations;
    if (!isfinite(resistance) || !(resistance > 0.0f)) return SOP_REVIEW_RESISTANCE;
    float next = gap / resistance;
    if (!isfinite(next) || next < low || next > high) return SOP_REVIEW_RANGE;
    if (fabsf(next - trial) < SOP_TOL)
    {
      if (!review_in_hull(dir, soc, soh, next)) return SOP_REVIEW_HULL;
      *answer = next;
      return SOP_REVIEW_OK;
    }
    trial = 0.5f * trial + 0.5f * next;
  }
  return SOP_REVIEW_ITERATION_CAP;
}
"""
replace_once(src, "/* ---- 표 조회 보조:", solver + "\n/* ---- 표 조회 보조:")
replace_once(main, '#include "sop_core.h"', '#include "sop_core.h"\n#include <math.h>')
replace_once(main, "#define SOP_CMD_PACK     0x71U", "#define SOP_CMD_REVIEW4  0x72U\n#define SOP_CMD_PACK     0x71U")
replace_once(main, "&& cmd != SOP_CMD_CYCLE && cmd != SOP_CMD_PACK", "&& cmd != SOP_CMD_CYCLE && cmd != SOP_CMD_PACK && cmd != SOP_CMD_REVIEW4")
replace_once(main, "|| command == SOP_CMD_CYCLE || command == SOP_CMD_PACK",
             "|| command == SOP_CMD_CYCLE || command == SOP_CMD_PACK || command == SOP_CMD_REVIEW4")
replace_once(main, "  primask = __get_PRIMASK();\n  __disable_irq();\n  __DSB(); __ISB();\n  DWT->CYCCNT", """
#ifdef SOP_BENCH_PACK
  if (cmd == SOP_CMD_REVIEW4 && (q.dir & 0x80U))
  {
    sop_feat_reset(&g_feat);
    sop_ekf_reset(&g_ekf, q.soc, q.soh);
  }
#endif
  primask = __get_PRIMASK();
  __disable_irq();
  __DSB(); __ISB();
  DWT->CYCCNT""")
replace_once(main, "    case SOP_CMD_PACK:\n", """
    case SOP_CMD_REVIEW4:
    {
      float limits[4] = {0.0f};
      float x12[SOP_NFEAT] = {0.0f};
      if (!isfinite(q.current_a) || !isfinite(q.v_pre) || !isfinite(q.soh)
          || q.soh < SOP_SOH_LO || q.soh > SOP_SOH_HI)
      {
        unsigned status = isfinite(q.soh) && (q.soh < SOP_SOH_LO || q.soh > SOP_SOH_HI)
                          ? SOP_REVIEW_HULL : SOP_REVIEW_INPUT;
        it = status | (status << 3U) | (status << 6U) | (status << 9U);
      }
      else
      {
        float soc = sop_ekf_step(&g_ekf, 1.0f, q.current_a, q.v_pre, q.soh,
                                fabsf(q.current_a) <= 1.0f);
        (void)sop_feat_update_a8(&g_feat, 1.0f, q.current_a, q.v_pre, soc, q.soh, x12);
        uint32_t max_iterations = 0U;
        for (unsigned d = 0; d < 2U; ++d)
        {
          float kfast, kslow;
          sop_trim((sop_dir_t)d, x12, &kfast, &kslow);
          for (unsigned h = 0; h < 2U; ++h)
          {
            unsigned index = 2U*d + h;
            float cutoff = d == SOP_CHARGE ? 4.2f : 2.5f;
            float horizon = h == 0U ? 2.0f : 10.0f;
            uint32_t iterations;
            unsigned status = sop_review_solve((sop_dir_t)d, soc, q.soh, q.v_pre,
                               cutoff, horizon, kfast, kslow, &limits[index], &iterations);
            it |= status << (3U * index);
            if (iterations > max_iterations) max_iterations = iterations;
          }
        }
        it |= max_iterations << 16U;
      }
      kf = limits[0]; ks = limits[1]; reff = limits[2]; ist = limits[3];
      break;
    }
    case SOP_CMD_PACK:
""")
