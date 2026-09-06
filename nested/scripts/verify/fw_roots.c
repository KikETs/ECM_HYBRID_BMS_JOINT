/* Host diagnostic linking the firmware's own sop_core.c and sop_tables.h.
 * stdin lines:  dir soc soh v_pre v_lim tau kf ks
 * stdout:       n_roots roots... | sop_solve_result iters | f(lo) f(hi)
 * Roots of f(I) = v_pre + I*R_eff(I) - v_lim are found on a 0.02 A scan of the
 * firmware's own domain ([0.05,400] charge, [-400,-0.1] discharge) with
 * bisection on every sign change.  No reference current is used. */
#include <stdio.h>
#include <math.h>
#include "sop_core.h"

static float fres(sop_dir_t dir, float soc, float soh, float vpre, float vlim,
                  float tau, float kf, float ks, float I)
{
  float R = sop_r_eff(dir, soc, soh, I, tau, kf, ks);
  if (!(R > 0.0f)) return NAN;
  return vpre + I * R - vlim;
}

int main(void)
{
  int dir; float soc, soh, vpre, vlim, tau, kf, ks;
  while (scanf("%d %f %f %f %f %f %f %f", &dir, &soc, &soh, &vpre, &vlim, &tau, &kf, &ks) == 8)
  {
    int charge = dir == 1;
    double lo = charge ? 0.05 : -400.0, hi = charge ? 400.0 : -0.1;
    double step = 0.02; int n = 0; double roots[16];
    double prev = fres((sop_dir_t)dir, soc, soh, vpre, vlim, tau, kf, ks, (float)lo), x = lo;
    for (double xx = lo + step; xx <= hi + 1e-9; xx += step)
    {
      double cur = fres((sop_dir_t)dir, soc, soh, vpre, vlim, tau, kf, ks, (float)xx);
      if (isfinite(prev) && isfinite(cur) && ((prev < 0) != (cur < 0)) && n < 16)
      {
        double a = x, b = xx, fa = prev;
        for (int k = 0; k < 60; k++)
        {
          double m = 0.5 * (a + b);
          double fm = fres((sop_dir_t)dir, soc, soh, vpre, vlim, tau, kf, ks, (float)m);
          if ((fm < 0) == (fa < 0)) { a = m; fa = fm; } else { b = m; }
        }
        roots[n++] = 0.5 * (a + b);
      }
      prev = cur; x = xx;
    }
    uint32_t it = 0;
    float I = sop_solve((sop_dir_t)dir, soc, soh, vpre, vlim, tau, kf, ks, &it);
    printf("%d", n);
    for (int k = 0; k < n; k++) printf(" %.5f", roots[k]);
    printf(" | %.5f %u | %.6f %.6f\n", I, it,
           fres((sop_dir_t)dir, soc, soh, vpre, vlim, tau, kf, ks, (float)lo),
           fres((sop_dir_t)dir, soc, soh, vpre, vlim, tau, kf, ks, (float)hi));
  }
  return 0;
}
