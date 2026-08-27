#include <stdio.h>
#include "soh_core.h"
int main(void)
{
  float x[64];
  for (;;)
  {
    for (int i = 0; i < 64; i++) { if (scanf("%f", &x[i]) != 1) { return 0; } }
    printf("%.9g\n", soh_infer(x));
  }
}
