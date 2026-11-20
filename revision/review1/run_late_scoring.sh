#!/usr/bin/env bash
set -euo pipefail
run="$(cd -- "$(dirname -- "$0")" && pwd)"
mode="$1"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
py="$HOME/anaconda3/envs/torch_env/bin/python"
trap 'echo "$?" > "$run/logs/late_'"$mode"'.exit"' EXIT
wait_job() {
  while ! test -f "$run/logs/$1.exit"; do sleep 20; done
  test "$(cat "$run/logs/$1.exit")" = 0
}
if test "$mode" = external; then
  wait_job external
  "$py" -u "$run/experiments/temperature_transfer.py" warm
else
  wait_job comparators_cpu
  wait_job comparators_gpu
  export RCSOP_REPO="$run/repo" RCSOP_BUNDLE="$run/repo/nested"
  export INNERTRIM="$run/strict_inner/predictions" BASETREES="$run/strict_inner/basetrees"
  export SOLVER=hull OUT="$run/strict_inner/methods" NPROC=10
  "$py" -u "$run/experiments/score_methods_pairs.py"
fi
