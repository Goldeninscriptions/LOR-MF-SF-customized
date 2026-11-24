#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
runner="$script_dir/run_derivative_greville_1d.sh"
root=${RESULT_DIR:-$script_dir/results/derivative_greville_1d}
read -r -a degrees <<< "${DEGREES_LIST:-2 3 4 6 8}"
read -r -a sizes <<< "${SIZES_LIST:-8 16 32 64 128}"
mkdir -p "$root/cases"
lock=${BATCH_LOCK_FILE:-$script_dir/.derivative_greville_explicit_batch.lock}
exec 9>"$lock"; flock -n 9 || { echo "Another explicit DMG batch is running" >&2; exit 4; }
summary="$root/summary.csv"; first=true
for p in "${degrees[@]}"; do
  for n in "${sizes[@]}"; do
    file="$root/cases/p${p}_n${n}.csv"
    echo "[1D] p=$p elements=$n"
    "$runner" "$p" "$n" "$file"
    if [[ "$first" == true ]]; then
      cp "$file" "$summary"; first=false
    else
      tail -n +2 "$file" >> "$summary"
    fi
  done
done
echo "[done] $summary"
