/* SOH 완전 정수 경로 — 활성화까지 int8, 누산 int32, SIMD MAC.
 *
 * WHY THIS EXISTS SEPARATELY FROM soh_core.c
 *   27.9 가 잰 int8 은 **가중치만** int8 이고 활성화는 float 였다. 그래서 곱셈
 *   전에 float 변환이 붙어 오히려 8.5 % 느렸다. 진짜 질문은 "활성화까지 정수로
 *   하면 하드 FPU 를 이기는가" 이고, 그것은 이 파일이 답한다.
 *
 *   CMSIS-NN 전체를 끌어오지 않는다. 커널 하나를 직접 써야 무엇이 얼마나
 *   걸리는지 런타임 안에 가려지지 않는다(27.1 과 같은 이유). 대신 M33 의 DSP
 *   확장(__SMLAD, 사이클당 int16 MAC 2 회)을 직접 쓴다.
 *
 * 양자화 규약
 *   활성화는 층별 대칭 int8, 스케일은 보정 집합의 최대 절대값에서 온다.
 *   누산 int32 -> (acc * s_w * s_in / s_out) 로 재양자화. ReLU 는 0 클램프.
 */
#include "soh_simd.h"
#include "soh_tables.h"
#include "soh_qparam.h"
#include <stdint.h>
#include <string.h>

#if !defined(__ARM_FEATURE_DSP)
/* 호스트 빌드용 대체 — 같은 산술, SIMD 만 없다. */
static inline int32_t smlad16(int32_t a, int32_t b, int32_t acc)
{
  int16_t a0 = (int16_t)(a & 0xFFFF), a1 = (int16_t)(a >> 16);
  int16_t b0 = (int16_t)(b & 0xFFFF), b1 = (int16_t)(b >> 16);
  return acc + (int32_t)a0 * b0 + (int32_t)a1 * b1;
}
#else
#include "cmsis_gcc.h"
static inline int32_t smlad16(int32_t a, int32_t b, int32_t acc)
{
  return __SMLAD(a, b, acc);
}
#endif

#define L1  SOH_NIN
#define L2  (SOH_NIN / 2)
#define C2  (SOH_CH * 2)
#define PAD (SOH_K / 2)

static inline int8_t sat8(int32_t v)
{
  if (v > 127) { return 127; }
  if (v < -128) { return -128; }
  return (int8_t)v;
}

/* acc(int32) 를 다음 층 int8 로. mult 는 부동소수 요청 배율. */
static inline int8_t requant(int32_t acc, float mult, int relu)
{
  float v = (float)acc * mult;
  if (relu && v < 0.0f) { v = 0.0f; }
  int32_t r = (int32_t)(v < 0.0f ? v - 0.5f : v + 0.5f);
  return sat8(r);
}

float soh_infer_simd(const float *x64)
{
  static int8_t z[L1], a[SOH_CH][L1], b[SOH_CH][L2], c[C2][L2];
  static int8_t pooled[C2 * SOH_POOL], h[SOH_HID];
  float total = 0.0f;

  for (int i = 0; i < L1; i++)
  {
    float v = (x64[i] - soh_mu[i]) / soh_sd[i] / SOHQ_S_IN;
    int32_t r = (int32_t)(v < 0.0f ? v - 0.5f : v + 0.5f);
    z[i] = sat8(r);
  }

  for (int s = 0; s < SOH_NSEED; s++)
  {
    const int8_t *w1 = soh_conv_0_weight_q + (size_t)s * SOH_CH * SOH_K;
    const float  *s1 = soh_conv_0_weight_s + (size_t)s * SOH_CH;
    const float  *b1 = soh_conv_0_bias     + (size_t)s * SOH_CH;
    const int8_t *w2 = soh_conv_3_weight_q + (size_t)s * C2 * SOH_CH * SOH_K;
    const float  *s2 = soh_conv_3_weight_s + (size_t)s * C2;
    const float  *b2 = soh_conv_3_bias     + (size_t)s * C2;
    const int8_t *w3 = soh_head_2_weight_q + (size_t)s * SOH_HID * (C2 * SOH_POOL);
    const float  *s3 = soh_head_2_weight_s + (size_t)s * SOH_HID;
    const float  *b3 = soh_head_2_bias     + (size_t)s * SOH_HID;
    const int8_t *w4 = soh_head_4_weight_q + (size_t)s * SOH_HID;
    const float  *s4 = soh_head_4_weight_s + (size_t)s;
    const float  *b4 = soh_head_4_bias     + (size_t)s;

    for (int o = 0; o < SOH_CH; o++)
    {
      float m = s1[o] * SOHQ_S_IN / SOHQ_S_A1;
      float bo = b1[o] / SOHQ_S_A1;
      for (int t = 0; t < L1; t++)
      {
        int32_t acc = 0;
        for (int q = 0; q < SOH_K; q++)
        {
          int ti = t + q - PAD;
          if (ti >= 0 && ti < L1) { acc += (int32_t)w1[o * SOH_K + q] * z[ti]; }
        }
        float v = (float)acc * m + bo;
        if (v < 0.0f) { v = 0.0f; }
        a[o][t] = sat8((int32_t)(v + 0.5f));
      }
    }
    for (int o = 0; o < SOH_CH; o++)
    {
      for (int t = 0; t < L2; t++)
      {
        int8_t u = a[o][2 * t], v = a[o][2 * t + 1];
        b[o][t] = u > v ? u : v;
      }
    }
    /* conv2 가 전체 MAC 의 86 % 다(81,920 / 95,264).  여기에 SIMD 를 걸지 않으면
     * 정수화는 재양자화 비용만 더하고 끝난다 — 실제로 처음 판이 그래서 32 % 느렸다.
     * 입력 채널 축(SOH_CH=16)을 int16 쌍으로 묶어 SMLAD 로 돈다. */
    static int16_t bcol[SOH_CH];
    for (int o = 0; o < C2; o++)
    {
      float m = s2[o] * SOHQ_S_A1 / SOHQ_S_A2;
      float bo = b2[o] / SOHQ_S_A2;
      for (int t = 0; t < L2; t++)
      {
        int32_t acc = 0;
        for (int q = 0; q < SOH_K; q++)
        {
          int ti = t + q - PAD;
          if (ti < 0 || ti >= L2) { continue; }
          for (int i = 0; i < SOH_CH; i++) { bcol[i] = b[i][ti]; }
          const int8_t *wr = w2 + (o * SOH_CH) * SOH_K + q;
          for (int i = 0; i < SOH_CH; i += 2)
          {
            int32_t wp = ((int32_t)(int16_t)wr[(i + 1) * SOH_K] << 16)
                       | ((int32_t)(uint16_t)(int16_t)wr[i * SOH_K]);
            int32_t bp = ((int32_t)bcol[i + 1] << 16)
                       | ((int32_t)(uint16_t)bcol[i]);
            acc = smlad16(wp, bp, acc);
          }
        }
        float v = (float)acc * m + bo;
        if (v < 0.0f) { v = 0.0f; }
        c[o][t] = sat8((int32_t)(v + 0.5f));
      }
    }
    const int span = L2 / SOH_POOL;
    for (int o = 0; o < C2; o++)
    {
      for (int p = 0; p < SOH_POOL; p++)
      {
        int32_t acc = 0;
        for (int t = 0; t < span; t++) { acc += c[o][p * span + t]; }
        pooled[o * SOH_POOL + p] = sat8((acc + span / 2) / span);
      }
    }
    /* 밀집층: int8 x int8 을 int16 쌍으로 묶어 SMLAD.  여기가 연산의 대부분이다
     * (256x32 + 32 = 8,224 MAC 중 8,192). */
    const int NP = C2 * SOH_POOL;
    static int16_t p16[C2 * SOH_POOL];
    for (int i = 0; i < NP; i++) { p16[i] = pooled[i]; }
    for (int o = 0; o < SOH_HID; o++)
    {
      const int8_t *wr = w3 + o * NP;
      int32_t acc = 0;
      for (int i = 0; i < NP; i += 2)
      {
        int32_t wp = ((int32_t)(int16_t)wr[i + 1] << 16)
                   | ((int32_t)(uint16_t)(int16_t)wr[i]);
        int32_t pp = ((int32_t)p16[i + 1] << 16)
                   | ((int32_t)(uint16_t)p16[i]);
        acc = smlad16(wp, pp, acc);
      }
      float v = (float)acc * (s3[o] * SOHQ_S_A2 / SOHQ_S_H) + b3[o] / SOHQ_S_H;
      if (v < 0.0f) { v = 0.0f; }
      h[o] = sat8((int32_t)(v + 0.5f));
    }
    int32_t acc = 0;
    for (int o = 0; o < SOH_HID; o++) { acc += (int32_t)w4[o] * h[o]; }
    total += (float)acc * s4[0] * SOHQ_S_H + b4[0];
  }
  return total / (float)SOH_NSEED;
}
