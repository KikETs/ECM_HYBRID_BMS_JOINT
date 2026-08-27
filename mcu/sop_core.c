#include "sop_core.h"
#include "sop_tables.h"
#include <math.h>

#define SOP_MAX_ITER 24U
#define SOP_TOL      1e-3f

static void frac_index(float v, float lo, float hi, int n, int *i0, int *i1,
                       float *f)
{
  float g = (v - lo) / (hi - lo) * (float)(n - 1);
  if (g < 0.0f) { g = 0.0f; }
  if (g > (float)(n - 1)) { g = (float)(n - 1); }
  int k = (int)g;
  if (k > n - 2) { k = (n > 1) ? n - 2 : 0; }
  *i0 = k;
  *i1 = (n > 1) ? k + 1 : k;
  *f = g - (float)k;
}

/* rank 축 위치.  rank_i 는 오름차순 대표 전류. */
static void rank_index(const float *rank_i, float mag, int *k0, int *k1,
                       float *f)
{
  int n = SOP_NRANK;
  if (mag <= rank_i[0])      { *k0 = 0; *k1 = 0; *f = 0.0f; return; }
  if (mag >= rank_i[n - 1])  { *k0 = n - 1; *k1 = n - 1; *f = 0.0f; return; }
  int k = 0;
  while (k < n - 2 && mag > rank_i[k + 1]) { k++; }
  *k0 = k; *k1 = k + 1;
  *f = (mag - rank_i[k]) / (rank_i[k + 1] - rank_i[k]);
}

/* [rank][soc][soh][2] 에서 이중선형 + rank 보간.  결과 mOhm.
 *
 * int8 판은 rank x 지평 별 스케일을 쓴다.  이중선형은 스케일이 같은 칸 안에서만
 * 일어나므로 정수 영역에서 보간한 뒤 한 번만 스케일을 곱하면 된다 — 곱셈이
 * 줄고 정밀도 손실도 없다.  rank 사이만 스케일이 다르므로 그때만 실수로 올린다. */
#if SOP_GRID_INT8
static void grid_d(const int8_t *g, const float *sc, int k0, int k1, float fk,
                   int i0, int i1, float fi, int j0, int j1, float fj,
                   float *d2, float *d10)
{
  const int SH = SOP_NH, ST = 2;
  float o[2];
  for (int c = 0; c < 2; c++)
  {
    float q0 =
      (float)g[(((k0 * SOP_NS + i0) * SH) + j0) * ST + c] * (1.0f - fi) * (1.0f - fj)
    + (float)g[(((k0 * SOP_NS + i1) * SH) + j0) * ST + c] * fi * (1.0f - fj)
    + (float)g[(((k0 * SOP_NS + i0) * SH) + j1) * ST + c] * (1.0f - fi) * fj
    + (float)g[(((k0 * SOP_NS + i1) * SH) + j1) * ST + c] * fi * fj;
    float q1 =
      (float)g[(((k1 * SOP_NS + i0) * SH) + j0) * ST + c] * (1.0f - fi) * (1.0f - fj)
    + (float)g[(((k1 * SOP_NS + i1) * SH) + j0) * ST + c] * fi * (1.0f - fj)
    + (float)g[(((k1 * SOP_NS + i0) * SH) + j1) * ST + c] * (1.0f - fi) * fj
    + (float)g[(((k1 * SOP_NS + i1) * SH) + j1) * ST + c] * fi * fj;
    o[c] = q0 * sc[k0 * 2 + c] * (1.0f - fk) + q1 * sc[k1 * 2 + c] * fk;
  }
  *d2 = o[0]; *d10 = o[1];
}
#else
static void grid_d(const float *g, int k0, int k1, float fk,
                   int i0, int i1, float fi, int j0, int j1, float fj,
                   float *d2, float *d10)
{
  const int SH = SOP_NH, ST = 2;
  float o[2];
  for (int c = 0; c < 2; c++)
  {
    float v0 =
      g[(((k0 * SOP_NS + i0) * SH) + j0) * ST + c] * (1.0f - fi) * (1.0f - fj)
    + g[(((k0 * SOP_NS + i1) * SH) + j0) * ST + c] * fi * (1.0f - fj)
    + g[(((k0 * SOP_NS + i0) * SH) + j1) * ST + c] * (1.0f - fi) * fj
    + g[(((k0 * SOP_NS + i1) * SH) + j1) * ST + c] * fi * fj;
    float v1 =
      g[(((k1 * SOP_NS + i0) * SH) + j0) * ST + c] * (1.0f - fi) * (1.0f - fj)
    + g[(((k1 * SOP_NS + i1) * SH) + j0) * ST + c] * fi * (1.0f - fj)
    + g[(((k1 * SOP_NS + i0) * SH) + j1) * ST + c] * (1.0f - fi) * fj
    + g[(((k1 * SOP_NS + i1) * SH) + j1) * ST + c] * fi * fj;
    o[c] = v0 * (1.0f - fk) + v1 * fk;
  }
  *d2 = o[0]; *d10 = o[1];
}
#endif

float sop_r_eff(sop_dir_t dir, float soc, float soh, float current_a,
                float tau_s, float kf, float ks)
{
  const float *ri = (dir == SOP_CHARGE) ? sop_rank_i_chg : sop_rank_i_dis;
  int i0, i1, j0, j1, k0, k1; float fi, fj, fk;
  frac_index(soc, SOP_SOC_LO, SOP_SOC_HI, SOP_NS, &i0, &i1, &fi);
  frac_index(soh, SOP_SOH_LO, SOP_SOH_HI, SOP_NH, &j0, &j1, &fj);
  float mag = current_a < 0.0f ? -current_a : current_a;
  rank_index(ri, mag, &k0, &k1, &fk);

  float d2, d10;
#if SOP_GRID_INT8
  grid_d((dir == SOP_CHARGE) ? sop_grid_chg_q : sop_grid_dis_q,
         (dir == SOP_CHARGE) ? sop_grid_chg_s : sop_grid_dis_s,
         k0, k1, fk, i0, i1, fi, j0, j1, fj, &d2, &d10);
#else
  grid_d((dir == SOP_CHARGE) ? sop_grid_chg : sop_grid_dis,
         k0, k1, fk, i0, i1, fi, j0, j1, fj, &d2, &d10);
#endif

  /* 두-지평 환원.  tau2 는 고정 기준값을 쓴다 — 26.3 의 발산을 피한다. */
  const float a = 1.0f - expf(-SOP_TAU_A / SOP_TAU2);
  const float b = 1.0f - expf(-SOP_TAU_B / SOP_TAU2);
  float r_slow = (d10 - d2) / (b - a);
  float r_fast = d2 - r_slow * a;
  float r = kf * r_fast + ks * r_slow * (1.0f - expf(-tau_s / SOP_TAU2));
  return r * 1e-3f;                       /* mOhm -> Ohm */
}

void sop_trim(sop_dir_t dir, const float *x12, float *kf, float *ks)
{
  const float *W  = (dir == SOP_CHARGE) ? trim_w_chg  : trim_w_dis;
  const float *B  = (dir == SOP_CHARGE) ? trim_b_chg  : trim_b_dis;
  const float *MU = (dir == SOP_CHARGE) ? trim_mu_chg : trim_mu_dis;
  const float *SD = (dir == SOP_CHARGE) ? trim_sd_chg : trim_sd_dis;
  float u[2];
  for (int o = 0; o < 2; o++)
  {
    float acc = B[o];
    for (int i = 0; i < SOP_NFEAT; i++)
    {
      float z = (x12[i] - MU[i]) / SD[i];
      if (z < -4.0f) { z = -4.0f; }
      if (z >  4.0f) { z =  4.0f; }
      acc += W[o * SOP_NFEAT + i] * z;
    }
    u[o] = acc;
  }
  *kf = expf(SOP_KF_SPAN * tanhf(u[0]));
  *ks = expf(SOP_KS_SPAN * tanhf(u[1]));
}

float sop_solve(sop_dir_t dir, float soc, float soh, float v_pre,
                float v_limit, float tau_s, float kf, float ks,
                uint32_t *iters_out)
{
  int charge = (dir == SOP_CHARGE);
  float I = charge ? 5.0f : -12.0f;
  uint32_t n = 0U;
  for (; n < SOP_MAX_ITER; n++)
  {
    float R = sop_r_eff(dir, soc, soh, I, tau_s, kf, ks);
    if (!(R > 0.0f)) { if (iters_out) { *iters_out = n + 1U; } return NAN; }
    float nx = (v_limit - v_pre) / R;
    if (charge) { if (nx < 0.05f) { nx = 0.05f; } if (nx > 400.0f) { nx = 400.0f; } }
    else        { if (nx > -0.1f) { nx = -0.1f; } if (nx < -400.0f) { nx = -400.0f; } }
    float d = nx - I; if (d < 0.0f) { d = -d; }
    if (d < SOP_TOL) { I = nx; n++; break; }
    I = 0.5f * I + 0.5f * nx;
  }
  if (iters_out) { *iters_out = n; }
  return I;
}

/* ---- 표 조회 보조: OCV 와 이력 반폭 ------------------------------------ */
static void ocv_lookup(float soc, float soh, float *ocv, float *m_half)
{
  int i0, i1, j0, j1; float fi, fj;
  frac_index(soc, SOP_SOC_LO, SOP_SOC_HI, SOP_NS, &i0, &i1, &fi);
  frac_index(soh, SOP_SOH_LO, SOP_SOH_HI, SOP_NH, &j0, &j1, &fj);
  const int SH = SOP_NH, ST = 2;
  for (int c = 0; c < 2; c++)
  {
    float v = sop_ocv[((i0 * SH) + j0) * ST + c] * (1.0f - fi) * (1.0f - fj)
            + sop_ocv[((i1 * SH) + j0) * ST + c] * fi * (1.0f - fj)
            + sop_ocv[((i0 * SH) + j1) * ST + c] * (1.0f - fi) * fj
            + sop_ocv[((i1 * SH) + j1) * ST + c] * fi * fj;
    if (c == 0) { *ocv = v; } else { *m_half = v; }
  }
}

/* D2/D10 -> (R_fast, R_slow) [Ohm], 보정 없이 명목값. */
static void nominal_rf_rs(sop_dir_t dir, float soc, float soh, float I,
                          float *rf, float *rs)
{
  const float *ri = (dir == SOP_CHARGE) ? sop_rank_i_chg : sop_rank_i_dis;
  int i0, i1, j0, j1, k0, k1; float fi, fj, fk;
  frac_index(soc, SOP_SOC_LO, SOP_SOC_HI, SOP_NS, &i0, &i1, &fi);
  frac_index(soh, SOP_SOH_LO, SOP_SOH_HI, SOP_NH, &j0, &j1, &fj);
  float mag = I < 0.0f ? -I : I;
  rank_index(ri, mag, &k0, &k1, &fk);
  float d2, d10;
#if SOP_GRID_INT8
  grid_d((dir == SOP_CHARGE) ? sop_grid_chg_q : sop_grid_dis_q,
         (dir == SOP_CHARGE) ? sop_grid_chg_s : sop_grid_dis_s,
         k0, k1, fk, i0, i1, fi, j0, j1, fj, &d2, &d10);
#else
  grid_d((dir == SOP_CHARGE) ? sop_grid_chg : sop_grid_dis,
         k0, k1, fk, i0, i1, fi, j0, j1, fj, &d2, &d10);
#endif
  const float a = 1.0f - expf(-SOP_TAU_A / SOP_TAU2);
  const float b = 1.0f - expf(-SOP_TAU_B / SOP_TAU2);
  *rs = (d10 - d2) / (b - a) * 1e-3f;
  *rf = (d2 * 1e-3f) - (*rs) * a;
}

#define EW(s, x, dt, tau) ((s) + (1.0f - expf(-(dt) / (tau))) * ((x) - (s)))
#define TAU_EW    600.0f
#define TAU_I     8.0f
#define TAU_DUTY  300.0f
#define I_REST    0.5f
#define I_DUTY    5.0f
#define I_HI      10.0f
#define Q_RATED   3.0f
#define GAMMA     20.0f

void sop_feat_reset(sop_feat_t *f)
{
  for (unsigned i = 0; i < sizeof(*f) / sizeof(float); i++)
  {
    ((float *)f)[i] = 0.0f;
  }
  f->e_T = 25.0f;
}

/* A8 — dR_fast 하나.  sop_feat_update 에서 쓰이지 않는 것을 전부 뺐다.
 * 남는 것: 명목 전파(v2n, h), 표 조회 둘, EW 둘. */
float sop_feat_update_a8(sop_feat_t *f, float dt_s, float I, float V,
                         float soc, float soh, float *x1)
{
  if (!(dt_s > 0.0f)) { return 0.0f; }
  if (dt_s > 60.0f) { dt_s = 60.0f; }
  sop_dir_t dir = (I > 0.0f) ? SOP_CHARGE : SOP_DISCHARGE;

  float rf, rs, ocv, mh;
  nominal_rf_rs(dir, soc, soh, I, &rf, &rs);
  ocv_lookup(soc, soh, &ocv, &mh);

  float v_hat = ocv + mh * f->h + I * rf + f->v1n + f->v2n;
  float r = V - v_hat;

  f->e_ir = EW(f->e_ir, I * r, dt_s, TAU_EW);
  f->e_ii = EW(f->e_ii, I * I, dt_s, TAU_EW);
  f->age_s += dt_s;

  float a2 = expf(-dt_s / SOP_TAU2);
  f->v2n = f->v2n * a2 + rs * (1.0f - a2) * I;
  f->v1n = 0.0f;
  float ah = expf(-((GAMMA * (I < 0.0f ? -I : I) * dt_s) / 3600.0f) / Q_RATED);
  f->h = ah * f->h + (1.0f - ah) * ((I > 0.0f) ? 1.0f : -1.0f);

  x1[0] = f->e_ir / (f->e_ii + 1e-9f) * 1e3f;
  return v_hat;
}

float sop_feat_update(sop_feat_t *f, float dt_s, float I, float V, float T,
                      float soc, float soh, float *x12)
{
  if (!(dt_s > 0.0f)) { return 0.0f; }
  if (dt_s > 60.0f) { dt_s = 60.0f; }
  sop_dir_t dir = (I > 0.0f) ? SOP_CHARGE : SOP_DISCHARGE;

  float rf, rs, ocv, mh;
  nominal_rf_rs(dir, soc, soh, I, &rf, &rs);
  ocv_lookup(soc, soh, &ocv, &mh);

  /* 명목 단자전압.  R_fast 는 즉시항 취급(tau1 이 0.244 s 중앙이므로 2 s 창에서
   * 이미 완성) — Python 의 v_hat = OCV + M h + I R0 + v1 + v2 와 같은 자리. */
  float v_hat = ocv + mh * f->h + I * rf + f->v1n + f->v2n;
  float r = V - v_hat;

  f->i_slow = EW(f->i_slow, I, dt_s, TAU_I);
  f->e_ir = EW(f->e_ir, I * r, dt_s, TAU_EW);
  f->e_ii = EW(f->e_ii, I * I, dt_s, TAU_EW);
  f->e_sr = EW(f->e_sr, f->i_slow * r, dt_s, TAU_EW);
  f->e_ss = EW(f->e_ss, f->i_slow * f->i_slow, dt_s, TAU_EW);
  float over = (I < 0.0f ? -I : I) - I_HI;
  if (over < 0.0f) { over = 0.0f; }
  f->e_hi = EW(f->e_hi, over * over, dt_s, TAU_EW);
  f->e_rest = EW(f->e_rest, ((I < 0.0f ? -I : I) < I_REST) ? 1.0f : 0.0f,
                 dt_s, TAU_EW);
  f->e_duty = EW(f->e_duty, ((I < 0.0f ? -I : I) > I_DUTY) ? 1.0f : 0.0f,
                 dt_s, TAU_DUTY);
  f->e_i2 = EW(f->e_i2, I * I, dt_s, TAU_EW);
  f->e_T = EW(f->e_T, T, dt_s, TAU_EW);
  f->age_s += dt_s;

  /* 명목 상태 전진 (RC 는 tau2 기준 하나로 접는다) */
  float a2 = expf(-dt_s / SOP_TAU2);
  f->v2n = f->v2n * a2 + rs * (1.0f - a2) * I;
  f->v1n = 0.0f;                       /* 빠른 가지는 즉시 완성 */
  float ah = expf(-((GAMMA * (I < 0.0f ? -I : I) * dt_s) / 3600.0f) / Q_RATED);
  f->h = ah * f->h + (1.0f - ah) * ((I > 0.0f) ? 1.0f : -1.0f);
  f->r_fast = rf * 1e3f; f->r_slow = rs * 1e3f;

  const float EPS = 1e-9f;
  x12[0] = f->e_ir / (f->e_ii + EPS) * 1e3f;
  x12[1] = f->e_sr / (f->e_ss + EPS) * 1e3f;
  x12[2] = log10f(f->e_ii + EPS);
  x12[3] = f->e_hi;
  x12[4] = f->e_rest;
  x12[5] = f->e_duty;
  x12[6] = soc;
  x12[7] = soh;
  x12[8] = f->e_T;
  x12[9] = sqrtf(f->e_i2 > 0.0f ? f->e_i2 : 0.0f);
  x12[10] = f->r_fast;
  x12[11] = f->r_slow;
  return v_hat;
}

/* ---- EKF --------------------------------------------------------------- */
void sop_ekf_reset(sop_ekf_t *e, float soc0, float soh)
{
  (void)soh;
  e->soc = soc0; e->v1 = 0.0f; e->v2 = 0.0f; e->h = 0.0f;
  for (int i = 0; i < 9; i++) { e->P[i] = 0.0f; }
  e->P[0] = 1e-4f; e->P[4] = 1e-6f; e->P[8] = 1e-6f;
  e->q_soc = 1e-9f; e->q_v = 1e-8f; e->r_volt = 4e-6f;
}

float sop_ekf_step(sop_ekf_t *e, float dt_s, float I, float V, float soh,
                   int do_update)
{
  sop_dir_t dir = (I > 0.0f) ? SOP_CHARGE : SOP_DISCHARGE;
  float rf, rs, ocv, mh;
  nominal_rf_rs(dir, e->soc, soh, I, &rf, &rs);
  ocv_lookup(e->soc, soh, &ocv, &mh);

  /* 예측 */
  e->soc += I * dt_s / 3600.0f / Q_RATED;
  float a2 = expf(-dt_s / SOP_TAU2);
  e->v2 = e->v2 * a2 + rs * (1.0f - a2) * I;
  e->v1 = 0.0f;
  float ah = expf(-((GAMMA * (I < 0.0f ? -I : I) * dt_s) / 3600.0f) / Q_RATED);
  e->h = ah * e->h + (1.0f - ah) * ((I > 0.0f) ? 1.0f : -1.0f);

  float F2 = a2;
  e->P[0] += e->q_soc;
  e->P[8] = F2 * F2 * e->P[8] + e->q_v;

  if (!do_update) { return e->soc; }

  /* 갱신.  dOCV/dSOC 를 유한차분으로. */
  float o1, m1, o2, m2;
  ocv_lookup(e->soc + 5e-3f, soh, &o1, &m1);
  ocv_lookup(e->soc - 5e-3f, soh, &o2, &m2);
  float dodz = (o1 - o2) / 1e-2f;

  float y_hat = ocv + mh * e->h + I * rf + e->v1 + e->v2;
  float innov = V - y_hat;
  float H0 = dodz, H2 = 1.0f;
  float S = H0 * e->P[0] * H0 + H2 * e->P[8] * H2 + e->r_volt;
  float K0 = e->P[0] * H0 / S;
  float K2 = e->P[8] * H2 / S;
  e->soc += K0 * innov;
  e->v2  += K2 * innov;
  e->P[0] -= K0 * H0 * e->P[0];
  e->P[8] -= K2 * H2 * e->P[8];
  if (e->soc < 0.0f) { e->soc = 0.0f; }
  if (e->soc > 1.0f) { e->soc = 1.0f; }
  return e->soc;
}


/* ---- 정밀도 x 구조 분해용 변형 ------------------------------------------
 * 산술 타입만 바꾸고 흐름은 sop_ekf_step 과 똑같이 유지한다. 완전 2RC 판은
 * 빠른 가지도 상태로 전파한다(tau1 = 0.244 s 중앙, 고정). */
#define TAU1_REF 0.244f

#define EKF_BODY(T, EXPF, TWO_RC)                                             \
  sop_dir_t dir = (I > 0.0f) ? SOP_CHARGE : SOP_DISCHARGE;                    \
  float rf, rs, ocv, mh;                                                      \
  nominal_rf_rs(dir, e->soc, soh, I, &rf, &rs);                               \
  ocv_lookup(e->soc, soh, &ocv, &mh);                                         \
  T dt = (T)dt_s, Ii = (T)I;                                                  \
  e->soc = (float)((T)e->soc + Ii * dt / (T)3600.0 / (T)Q_RATED);             \
  T a2 = EXPF(-dt / (T)SOP_TAU2);                                             \
  e->v2 = (float)((T)e->v2 * a2 + (T)rs * ((T)1.0 - a2) * Ii);                \
  if (TWO_RC) {                                                               \
    T a1 = EXPF(-dt / (T)TAU1_REF);                                           \
    e->v1 = (float)((T)e->v1 * a1 + (T)rf * ((T)1.0 - a1) * Ii);              \
  } else { e->v1 = 0.0f; }                                                    \
  T ah = EXPF(-((T)GAMMA * (Ii < 0 ? -Ii : Ii) * dt / (T)3600.0) / (T)Q_RATED); \
  e->h = (float)(ah * (T)e->h + ((T)1.0 - ah) * (Ii > 0 ? (T)1.0 : (T)-1.0)); \
  e->P[0] += e->q_soc;                                                        \
  e->P[8] = (float)((T)a2 * (T)a2 * (T)e->P[8] + (T)e->q_v);                  \
  if (!do_update) { return e->soc; }                                          \
  float o1, m1, o2, m2;                                                       \
  ocv_lookup(e->soc + 5e-3f, soh, &o1, &m1);                                  \
  ocv_lookup(e->soc - 5e-3f, soh, &o2, &m2);                                  \
  T dodz = ((T)o1 - (T)o2) / (T)1e-2;                                         \
  T yh = (T)ocv + (T)mh * (T)e->h + Ii * (T)(TWO_RC ? 0.0f : rf)              \
       + (T)e->v1 + (T)e->v2;                                                 \
  T innov = (T)V - yh;                                                        \
  T S = dodz * (T)e->P[0] * dodz + (T)e->P[8] + (T)e->r_volt;                 \
  T K0 = (T)e->P[0] * dodz / S, K2 = (T)e->P[8] / S;                          \
  e->soc = (float)((T)e->soc + K0 * innov);                                   \
  e->v2  = (float)((T)e->v2 + K2 * innov);                                    \
  e->P[0] = (float)((T)e->P[0] - K0 * dodz * (T)e->P[0]);                     \
  e->P[8] = (float)((T)e->P[8] - K2 * (T)e->P[8]);                            \
  if (e->soc < 0.0f) { e->soc = 0.0f; }                                       \
  if (e->soc > 1.0f) { e->soc = 1.0f; }                                       \
  return e->soc;

float sop_ekf_step_f64(sop_ekf_t *e, float dt_s, float I, float V, float soh,
                       int do_update)
{ EKF_BODY(double, exp, 0) }

float sop_ekf2_step(sop_ekf_t *e, float dt_s, float I, float V, float soh,
                    int do_update)
{ EKF_BODY(float, expf, 1) }

float sop_ekf2_step_f64(sop_ekf_t *e, float dt_s, float I, float V, float soh,
                        int do_update)
{ EKF_BODY(double, exp, 1) }
