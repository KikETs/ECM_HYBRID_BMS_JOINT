#!/usr/bin/env bash
set -euo pipefail
run="$(cd -- "$(dirname -- "$0")" && pwd)"
out="$HOME/.cache/rc_sop_review1_20261116_results.tar.gz"
trap 'echo "$?" > "$run/logs/package.exit"' EXIT
while ! test -f "$run/logs/late_methods.exit"; do sleep 15; done
test "$(cat "$run/logs/late_methods.exit")" = 0
"$HOME/anaconda3/envs/torch_env/bin/python" "$run/experiments/package_results.py" "$out"
