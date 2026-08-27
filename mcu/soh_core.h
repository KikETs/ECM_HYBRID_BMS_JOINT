#ifndef SOH_CORE_H
#define SOH_CORE_H
/* dQ/dV CNN.  x 는 정규화 전 64 점 곡선.  반환은 SOH (시드 평균). */
float soh_infer(const float *x64);
#endif
