#!/bin/bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
trap 'echo "$?" > "$root/logs/state.exit"' EXIT
"$HOME/anaconda3/envs/torch_env/bin/python" -u "$root/scripts/replay_preceding_drive.py"
