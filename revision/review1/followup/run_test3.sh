#!/bin/bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
trap 'echo "$?" > "$root/logs/test3_hybrid.exit"' EXIT
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$HOME/anaconda3/envs/torch_env/bin/python" -u "$root/scripts/test3_cold_hybrid.py"
