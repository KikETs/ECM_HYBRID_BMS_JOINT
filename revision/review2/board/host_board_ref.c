/* Host reference for the deployment-image board check: for each input line
   "dir soc soh vpre vlim tau x0 ... x11" print "kf ks status current iters" using the same trim and flagged solver as
   SOP_CMD_SOLVE in fw_sop (built with -DSOP_A8_ONLY). */
#include <stdio.h>
#include "sop_core.h"
int main(void)
{
  int d; float soc, soh, vpre, vlim, tau, x[12];
  while (scanf("%d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f", &d, &soc, &soh, &vpre, &vlim, &tau,
               &x[0], &x[1], &x[2], &x[3], &x[4], &x[5], &x[6], &x[7], &x[8], &x[9], &x[10], &x[11]) == 18)
  {
    float kf, ks, a = -1.0f; uint32_t it = 0;
    sop_trim((sop_dir_t)d, x, &kf, &ks);
    sop_review_status_t s = sop_review_solve((sop_dir_t)d, soc, soh, vpre, vlim, tau, kf, ks, &a, &it);
    printf("%.9g %.9g %d %.9g %u\n", kf, ks, (int)s, a, (unsigned)it);
  }
  return 0;
}
