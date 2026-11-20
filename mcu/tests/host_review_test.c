/* Host check of the deployment solver change: where sop_review_solve reports OK it must return the same current as the
   legacy sop_solve; every flagged case must return 0 A. Grid over SOC, SOH, pre-pulse voltage, horizon and direction. */
#include <stdio.h>
#include <math.h>
#include "sop_core.h"
int main(void)
{
  long n = 0, ok = 0, flagged = 0, mismatch = 0, nonzero_flag = 0; double maxdiff = 0;
  long cnt[8] = {0};
  for (int d = 0; d < 2; ++d)
  for (float soc = 0.05f; soc <= 1.0f; soc += 0.05f)
  for (float soh = 0.72f; soh <= 1.0f; soh += 0.04f)
  for (float dv = 0.0f; dv <= 0.30f; dv += 0.05f)
  for (int k = 0; k < 2; ++k)
  for (float kf = 0.8f; kf <= 1.3f; kf += 0.25f)
  {
    sop_dir_t dir = d ? SOP_CHARGE : SOP_DISCHARGE;
    float vlim = d ? 4.2f : 2.5f;
    float vpre = d ? 3.5f + dv * 2.0f : 3.4f + dv * 2.5f;
    float tau = k ? 10.0f : 2.0f;
    uint32_t it0 = 0, it1 = 0; float a = -1.0f;
    float legacy = sop_solve(dir, soc, soh, vpre, vlim, tau, kf, kf, &it0);
    sop_review_status_t s = sop_review_solve(dir, soc, soh, vpre, vlim, tau, kf, kf, &a, &it1);
    ++n; cnt[s]++;
    if (s == SOP_REVIEW_OK) {
      ++ok; double df = fabs((double)a - (double)legacy);
      if (df > maxdiff) maxdiff = df;
      if (df > 1e-4) ++mismatch;
    } else {
      ++flagged; if (a != 0.0f) ++nonzero_flag;
    }
  }
  printf("cases %ld ok %ld flagged %ld mismatch(>1e-4 A) %ld max|diff| %.3g A flagged-nonzero %ld\n", n, ok, flagged, mismatch, maxdiff, nonzero_flag);
  printf("status counts: ok %ld input %ld hull %ld resistance %ld itercap %ld range %ld\n", cnt[0], cnt[1], cnt[2], cnt[3], cnt[4], cnt[5]);
  return (mismatch || nonzero_flag) ? 1 : 0;
}
