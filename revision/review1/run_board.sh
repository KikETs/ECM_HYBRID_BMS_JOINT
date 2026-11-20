#!/usr/bin/env bash
set -euo pipefail
run="$(cd -- "$(dirname -- "$0")" && pwd)"
prog="$HOME/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/STM32_Programmer_CLI"
conn=(port=SWD sn=0024003C3235511138363730)
flash="$run/mcu/original_flash.bin"
build="$run/repo/mcu/fw_sop/Build/review1_coupled/sop_bench.elf"
sha256sum --check "$run/mcu/original_flash.sha256"
test -z "$(fuser /dev/ttyACM0 2>/dev/null || true)"
restore() {
  result=$?
  trap - EXIT
  echo "$result" > "$run/logs/mcu_capture.exit"
  "$prog" -c "${conn[@]}" -w "$flash" 0x08000000 -v -rst > "$run/logs/mcu_restore.log" 2>&1 || exit 31
  "$prog" -c "${conn[@]}" mode=HotPlug -u 0x08000000 0x200000 "$run/mcu/restored_flash.bin" > "$run/logs/mcu_restore_check.log" 2>&1 || exit 32
  cmp "$flash" "$run/mcu/restored_flash.bin" || exit 33
  sha256sum "$run/mcu/restored_flash.bin" > "$run/mcu/restored_flash.sha256"
  echo "$result" > "$run/logs/mcu.exit"
  exit "$result"
}
trap restore EXIT
"$prog" -c "${conn[@]}" -w "$build" -v -rst > "$run/logs/mcu_flash.log" 2>&1
sleep 2
"$HOME/anaconda3/envs/torch_env/bin/python" "$run/experiments/measure_mcu.py"
