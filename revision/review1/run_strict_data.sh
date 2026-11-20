#!/usr/bin/env bash
set -euo pipefail
run="$(cd -- "$(dirname -- "$0")" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
trap 'echo "$?" > "$run/logs/strict_data.exit"' EXIT
"$HOME/anaconda3/envs/torch_env/bin/python" -u "$run/experiments/build_strict_inner.py"
