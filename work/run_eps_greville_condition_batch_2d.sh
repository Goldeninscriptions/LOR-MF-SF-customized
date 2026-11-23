#!/usr/bin/env bash
set -euo pipefail

# Sequential p=3..7, n=4..1024 sweep.  Each case runs P first and P^{-1}K
# second; no mpirun processes overlap.  Existing converged files are resumed.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
single_case="$script_dir/run_eps_greville_condition_2d.sh"
result_root=${RESULT_DIR:-$script_dir/results/eps_greville_condition_batch_2d}
console_root=${CONSOLE_DIR:-$script_dir/logs/eps_greville_condition_batch_2d}
summary_file=${SUMMARY_FILE:-$result_root/summary.tsv}
failure_file=${FAILURE_FILE:-$result_root/failures.tsv}
resume=${RESUME:-true}
continue_on_error=${CONTINUE_ON_ERROR:-false}
cooldown_seconds=${COOLDOWN_SECONDS:-0}

for value in "$resume" "$continue_on_error"; do
  [[ "$value" == true || "$value" == false ]] || {
    echo "RESUME and CONTINUE_ON_ERROR must be true or false" >&2; exit 2; }
done
[[ "$cooldown_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "COOLDOWN_SECONDS must be nonnegative" >&2; exit 2; }

read -r -a degrees <<< "${DEGREES_LIST:-3 4 5 6 7}"
read -r -a sizes <<< "${SIZES_LIST:-4 8 16 32 64 128 256 512 1024}"
read -r -a targets <<< "${TARGETS_LIST:-p pinv_k}"
for degree in "${degrees[@]}"; do
  [[ "$degree" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid degree: $degree" >&2; exit 2; }
done
for size in "${sizes[@]}"; do
  [[ "$size" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid size: $size" >&2; exit 2; }
done
for target in "${targets[@]}"; do
  [[ "$target" == p || "$target" == pinv_k ]] || { echo "Invalid target: $target" >&2; exit 2; }
done
mkdir -p "$result_root" "$console_root"
printf 'p\tn\ttarget\tmatrix_source\tpc_mode\tmpi_ranks\tfree_dofs\tlambda_min\tlambda_max\tcondition_number\teps_iterations_min\teps_iterations_max\ttime_spectral_eps_s\ttime_case_total_s\tresult_file\n' > "$summary_file"
printf 'p\tn\ttarget\texit_status\tconsole_file\tresult_file\n' > "$failure_file"

is_complete() {
  local file=$1 expected=$2
  [[ -f "$file" ]] && grep -qx 'eigensolver_status = converged' "$file" &&
    grep -qx "spectral_condition_target = $expected" "$file" &&
    grep -qx 'preconditioner_matrix_source = greville_mass_tensor' "$file"
}

append_summary() {
  local file=$1
  awk -F ' = ' -v path="$file" '
    $1=="p" {p=$2} $1=="n" {n=$2; gsub(/[()]/,"",n); sub(/,.*/,"",n)}
    $1=="spectral_condition_target" {target=$2}
    $1=="preconditioner_matrix_source" {source=$2}
    $1=="mpi_ranks" {ranks=$2} $1=="total_free_dofs" {dofs=$2}
    $1=="lambda_min" {lmin=$2} $1=="lambda_max" {lmax=$2}
    $1=="eps_condition_number" {kappa=$2}
    $1=="eps_iterations_min" {imin=$2} $1=="eps_iterations_max" {imax=$2}
    $1=="time_spectral_eps_s" {eps_time=$2} $1=="time_case_total_s" {total=$2}
    END {printf "%s\t%s\t%s\t%s\ttensor_transpose\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",p,n,target,source,ranks,dofs,lmin,lmax,kappa,imin,imax,eps_time,total,path}
  ' "$file" >> "$summary_file"
}

completed=0 failed=0
total=$(( ${#degrees[@]} * ${#sizes[@]} * ${#targets[@]} ))
for degree in "${degrees[@]}"; do
  for elements in "${sizes[@]}"; do
    for target in "${targets[@]}"; do
      case_dir="$result_root/p${degree}/n${elements}x${elements}"
      log_dir="$console_root/p${degree}/n${elements}x${elements}"
      result_file="$case_dir/${target}.log"
      console_file="$log_dir/${target}.console.log"
      mkdir -p "$case_dir" "$log_dir"
      if [[ "$resume" == true ]] && is_complete "$result_file" "$target"; then
        echo "[skip] p=$degree n=${elements}x${elements} target=$target"
        append_summary "$result_file"; completed=$((completed+1)); continue
      fi
      echo "[batch $((completed+failed+1))/$total] p=$degree n=${elements}x${elements} target=$target"
      set +e
      "$single_case" "$degree" "$elements" "$target" "$result_file" \
        2>&1 | tee "$console_file"
      status=${PIPESTATUS[0]}
      set -e
      if (( status == 0 )) && is_complete "$result_file" "$target"; then
        append_summary "$result_file"; completed=$((completed+1))
      else
        (( status != 0 )) || status=1
        failed=$((failed+1))
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$degree" "$elements" \
          "$target" "$status" "$console_file" "$result_file" >> "$failure_file"
        echo "[failed] p=$degree n=${elements}x${elements} target=$target status=$status" >&2
        [[ "$continue_on_error" == true ]] || exit "$status"
      fi
      [[ "$cooldown_seconds" == 0 ]] || sleep "$cooldown_seconds"
    done
  done
done
echo "[done] completed=$completed failed=$failed summary=$summary_file failures=$failure_file"
(( failed == 0 ))
