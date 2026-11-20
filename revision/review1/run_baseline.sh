#!/usr/bin/env bash
set -euo pipefail
run="$(cd -- "$(dirname -- "$0")" && pwd)"
export RCSOP_REPO="$run/repo" RCSOP_BUNDLE="$run/repo/nested" RCSOP_WORK="$run/baseline" RCSOP_PYTHON="$HOME/anaconda3/envs/torch_env/bin/python"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
trap 'echo "$?" > "$run/logs/baseline.exit"' EXIT
bash "$RCSOP_BUNDLE/reproduce.sh" check-pools
bash "$RCSOP_BUNDLE/reproduce.sh" score
