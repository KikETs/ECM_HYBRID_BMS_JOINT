#!/usr/bin/env bash
set -euo pipefail
run="$(cd -- "$(dirname -- "$0")" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
py="$HOME/anaconda3/envs/torch_env/bin/python"
mode="$1"
trap 'echo "$?" > "$run/logs/comparators_'"$mode"'.exit"' EXIT
if test "$mode" = gpu; then
  while ! test -f "$run/logs/strict_all.exit" && test "$(find "$run/strict_inner/predictions" -name 'pred_A8_*.npz' | wc -l)" -lt 60; do sleep 15; done
fi
"$py" -u "$run/experiments/strict_comparators.py" "$mode"
