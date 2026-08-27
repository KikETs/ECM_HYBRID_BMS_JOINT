/* 하이브리드 SOP 코어 — MCU 와 호스트가 같은 코드를 쓴다.
 *
 * 호스트에서 Python 과 대조한 뒤 그대로 펌웨어에 올린다. 두 곳에서 다른 코드를
 * 돌리면 MCU 측정이 무엇을 잰 것인지 알 수 없다.
 */
#ifndef SOP_CORE_H
#define SOP_CORE_H

#include <stdint.h>

typedef enum { SOP_DISCHARGE = 0, SOP_CHARGE = 1 } sop_dir_t;

/* 등가저항 D(tau) [Ohm].  보정 배수 kf, ks 를 적용한다. */
float sop_r_eff(sop_dir_t dir, float soc, float soh, float current_a,
                float tau_s, float kf, float ks);

/* 12 개 특징 -> (kf, ks).  x 는 정규화 전 원값. */
void sop_trim(sop_dir_t dir, const float *x12, float *kf, float *ks);

/* V_pre + I * R_eff(I) = v_limit 를 I 에 대해 푼다.
 * v_pre 는 **측정 단자전압**이어야 한다 (sop_hybrid_spec.md 23.4). */
float sop_solve(sop_dir_t dir, float soc, float soh, float v_pre,
                float v_limit, float tau_s, float kf, float ks,
                uint32_t *iters_out);

#endif

/* ---- 샘플마다 도는 것 ------------------------------------------------- */

/* 트림 특징 상태.  12 개 EW 통계 + 명목 ECM 전파.  차량에서는 64 B NVM 으로
 * 키 사이클을 건널 수 있다(sop_trim_features.py 의 restore 와 같은 취지). */
typedef struct
{
  float v1n, v2n, h;                 /* 명목 RC 와 이력 */
  float i_slow;
  float e_ir, e_ii, e_sr, e_ss;
  float e_hi, e_rest, e_duty, e_i2, e_T;
  float age_s;
  float r_fast, r_slow;              /* 마지막 명목값 (특징 10, 11) */
} sop_feat_t;

void sop_feat_reset(sop_feat_t *f);
/* 한 샘플 전진.  반환은 명목 단자전압, x12 에 특징을 채운다. */
float sop_feat_update(sop_feat_t *f, float dt_s, float I, float V, float T,
                      float soc, float soh, float *x12);

/* 채택 구성(A8)은 dR_fast 하나만 쓴다 — 학습된 가중치의 12 열 중 11 열이
 * 정확히 0 이다 (sop_hybrid_spec.md 29.7).  그래서 EW 상태가 e_ir, e_ii
 * 둘로 줄고 나머지 여덟 갱신이 사라진다.  잔차를 만드는 명목 전파와 표
 * 조회는 그대로 남으므로, 절감이 얼마인지는 재봐야 안다.
 * x1 에 dR_fast 하나를 채운다. */
float sop_feat_update_a8(sop_feat_t *f, float dt_s, float I, float V,
                         float soc, float soh, float *x1);

/* ---- SOC EKF (2RC + 이력), float32 ------------------------------------ */
typedef struct
{
  float soc, v1, v2, h;
  float P[9];                        /* 3x3 대칭, 행 우선 */
  float q_soc, q_v, r_volt;
} sop_ekf_t;

void sop_ekf_reset(sop_ekf_t *e, float soc0, float soh);
/* 한 스텝.  반환은 갱신된 SOC. */
float sop_ekf_step(sop_ekf_t *e, float dt_s, float I, float V, float soh,
                   int do_update);

/* 정밀도와 구조를 가르기 위한 변형 넷.
 * 27.6 절이 "기존 378.6 us 대비 51 배" 를 쓰지 못한다고 적은 이유가 이것이다 —
 * 정밀도(FP64 소프트웨어 -> float32 하드 FPU)와 구조(완전 2RC -> 빠른 가지를
 * 즉시항으로 접음)가 함께 바뀌었고 몫을 가르지 않았다. 여기서 가른다. */
float sop_ekf_step_f64(sop_ekf_t *e, float dt_s, float I, float V, float soh,
                       int do_update);          /* float32 구조 + double 산술 */
float sop_ekf2_step(sop_ekf_t *e, float dt_s, float I, float V, float soh,
                    int do_update);             /* 완전 2RC + float32 */
float sop_ekf2_step_f64(sop_ekf_t *e, float dt_s, float I, float V, float soh,
                        int do_update);         /* 완전 2RC + double */
