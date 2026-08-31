#include "soh_core.h"
#include "soh_tables.h"
#include <math.h>
#include <stddef.h>

#if SOH_RIDGE
/* Ridge on the standardised dQ/dV curve.
 *
 *     soh = b + sum_i w_i * (x_i - mu_i) / sd_i
 *
 * 64 multiply-accumulates.  The CNN body below is kept so the comparison
 * build still exists and the saving can be measured rather than asserted;
 * which one compiles is decided by the header the exporter wrote.        */
float soh_infer(const float *x64)
{
  float acc = SOH_B;
  for (int i = 0; i < SOH_NIN; i++)
  {
    acc += soh_w[i] * ((x64[i] - soh_mu[i]) / soh_sd[i]);
  }
  return acc;
}

#else

/* Conv1d(1->CH, k=5, pad=2) -> ReLU -> MaxPool1d(2)
 * -> Conv1d(CH->2CH, k=5, pad=2) -> ReLU -> AdaptiveAvgPool1d(8)
 * -> Linear(2CH*8 -> HID) -> ReLU -> Linear(HID -> 1)                 */
#define L1  SOH_NIN            /* 64 */
#define L2  (SOH_NIN / 2)      /* 32 */
#define C2  (SOH_CH * 2)       /* 32 */
#define PAD (SOH_K / 2)

static float run_seed(const float *z, int s)
{
  static float a[SOH_CH][L1];
  static float b[SOH_CH][L2];
  static float c[C2][L2];
  static float pooled[C2 * SOH_POOL];
  static float h[SOH_HID];

#if SOH_INT8
  /* 채널별 대칭 int8.  누산은 정수로 하고 채널 끝에서 한 번만 스케일을 곱한다 —
   * 곱셈이 줄고, 스케일이 채널 안에서 상수이므로 정밀도 손실도 없다. */
  const int8_t *w1 = soh_conv_0_weight_q + (size_t)s * SOH_CH * 1 * SOH_K;
  const float  *s1 = soh_conv_0_weight_s + (size_t)s * SOH_CH;
  const float  *b1 = soh_conv_0_bias     + (size_t)s * SOH_CH;
  const int8_t *w2 = soh_conv_3_weight_q + (size_t)s * C2 * SOH_CH * SOH_K;
  const float  *s2 = soh_conv_3_weight_s + (size_t)s * C2;
  const float  *b2 = soh_conv_3_bias     + (size_t)s * C2;
  const int8_t *w3 = soh_head_2_weight_q + (size_t)s * SOH_HID * (C2 * SOH_POOL);
  const float  *s3 = soh_head_2_weight_s + (size_t)s * SOH_HID;
  const float  *b3 = soh_head_2_bias     + (size_t)s * SOH_HID;
  const int8_t *w4 = soh_head_4_weight_q + (size_t)s * 1 * SOH_HID;
  const float  *s4 = soh_head_4_weight_s + (size_t)s * 1;
  const float  *b4 = soh_head_4_bias     + (size_t)s * 1;
#else
  const float *w1 = soh_conv_0_weight + (size_t)s * SOH_CH * 1 * SOH_K;
  const float *b1 = soh_conv_0_bias   + (size_t)s * SOH_CH;
  const float *w2 = soh_conv_3_weight + (size_t)s * C2 * SOH_CH * SOH_K;
  const float *b2 = soh_conv_3_bias   + (size_t)s * C2;
  const float *w3 = soh_head_2_weight + (size_t)s * SOH_HID * (C2 * SOH_POOL);
  const float *b3 = soh_head_2_bias   + (size_t)s * SOH_HID;
  const float *w4 = soh_head_4_weight + (size_t)s * 1 * SOH_HID;
  const float *b4 = soh_head_4_bias   + (size_t)s * 1;
#endif
#if SOH_INT8
#define WQ(w, i) ((float)(w)[i])
#define SCALE(sc, o) ((sc)[o])
#else
#define WQ(w, i) ((w)[i])
#define SCALE(sc, o) (1.0f)
#endif

  for (int o = 0; o < SOH_CH; o++)
  {
    for (int t = 0; t < L1; t++)
    {
      float acc = 0.0f;
      for (int q = 0; q < SOH_K; q++)
      {
        int ti = t + q - PAD;
        if (ti >= 0 && ti < L1) { acc += WQ(w1, o * SOH_K + q) * z[ti]; }
      }
#if SOH_INT8
      acc = acc * SCALE(s1, o) + b1[o];
#else
      acc += b1[o];
#endif
      a[o][t] = acc > 0.0f ? acc : 0.0f;
    }
  }
  for (int o = 0; o < SOH_CH; o++)
  {
    for (int t = 0; t < L2; t++)
    {
      float u = a[o][2 * t], v = a[o][2 * t + 1];
      b[o][t] = u > v ? u : v;
    }
  }
  for (int o = 0; o < C2; o++)
  {
    for (int t = 0; t < L2; t++)
    {
      float acc = 0.0f;
      for (int i = 0; i < SOH_CH; i++)
      {
        for (int q = 0; q < SOH_K; q++)
        {
          int ti = t + q - PAD;
          if (ti >= 0 && ti < L2)
          {
            acc += WQ(w2, (o * SOH_CH + i) * SOH_K + q) * b[i][ti];
          }
        }
      }
#if SOH_INT8
      acc = acc * SCALE(s2, o) + b2[o];
#else
      acc += b2[o];
#endif
      c[o][t] = acc > 0.0f ? acc : 0.0f;
    }
  }
  /* AdaptiveAvgPool1d(8): L2=32 -> 8, 균등 4 칸 */
  const int span = L2 / SOH_POOL;
  for (int o = 0; o < C2; o++)
  {
    for (int p = 0; p < SOH_POOL; p++)
    {
      float acc = 0.0f;
      for (int t = 0; t < span; t++) { acc += c[o][p * span + t]; }
      pooled[o * SOH_POOL + p] = acc / (float)span;
    }
  }
  for (int o = 0; o < SOH_HID; o++)
  {
    float acc = 0.0f;
    for (int i = 0; i < C2 * SOH_POOL; i++)
    {
      acc += WQ(w3, o * (C2 * SOH_POOL) + i) * pooled[i];
    }
#if SOH_INT8
    acc = acc * SCALE(s3, o) + b3[o];
#else
    acc += b3[o];
#endif
    h[o] = acc > 0.0f ? acc : 0.0f;
  }
  float out = 0.0f;
  for (int o = 0; o < SOH_HID; o++) { out += WQ(w4, o) * h[o]; }
#if SOH_INT8
  out = out * SCALE(s4, 0) + b4[0];
#else
  out += b4[0];
#endif
  return out;
}

float soh_infer(const float *x64)
{
  static float z[SOH_NIN];
  for (int i = 0; i < SOH_NIN; i++) { z[i] = (x64[i] - soh_mu[i]) / soh_sd[i]; }
  float acc = 0.0f;
  for (int s = 0; s < SOH_NSEED; s++) { acc += run_seed(z, s); }
  return acc / (float)SOH_NSEED;
}

#endif /* SOH_RIDGE */
