#!/bin/bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
py="$HOME/anaconda3/envs/torch_env/bin/python"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
trap 'echo "$?" > "$root/logs/finalize.exit"' EXIT
for step in test1 test3_hybrid state wcet; do
  until test -f "$root/logs/$step.exit"; do sleep 15; done
  test "$(cat "$root/logs/$step.exit")" = 0
done
"$py" -u "$root/scripts/cross_cell_calibration.py" | tee "$root/logs/cross_cell_calibration.log"
"$py" "$root/scripts/package_followup.py" "$HOME/.cache/rc_sop_review1_followup_20261116.tar.gz"
