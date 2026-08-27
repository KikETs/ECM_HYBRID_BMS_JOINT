/* 호스트 대조 하네스: stdin 으로 질의를 받아 stdout 으로 결과를 낸다.
 * 펌웨어와 **같은** sop_core.c 를 링크한다. */
#include <stdio.h>
#include "sop_core.h"
int main(void)
{
  int dir; float soc, soh, vpre, vlim, tau, x[12];
  while (scanf("%d %f %f %f %f %f", &dir, &soc, &soh, &vpre, &vlim, &tau) == 6)
  {
    for (int i = 0; i < 12; i++) { if (scanf("%f", &x[i]) != 1) { return 1; } }
    float kf, ks; uint32_t it;
    sop_trim((sop_dir_t)dir, x, &kf, &ks);
    float r = sop_r_eff((sop_dir_t)dir, soc, soh, 10.0f, tau, kf, ks);
    float I = sop_solve((sop_dir_t)dir, soc, soh, vpre, vlim, tau, kf, ks, &it);
    printf("%.9g %.9g %.9g %.9g %u\n", kf, ks, r, I, it);
  }
  return 0;
}
