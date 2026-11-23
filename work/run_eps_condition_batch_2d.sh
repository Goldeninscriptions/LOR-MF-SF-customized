#!/usr/bin/env bash
set -euo pipefail

# Strictly serial sweep of p=3..7 and n=4..1024. For every (p,n), run EPS on
# K, then P, then P^{-1}K. No command is backgrounded; each mpirun must exit
# before the next target or discretization starts.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
single_case="$script_dir/run_eps_condition_2d.sh"
result_root=${RESULT_DIR:-$script_dir/results/eps_condition_batch_2d}
console_root=${CONSOLE_DIR:-$script_dir/logs/eps_condition_batch_2d}
summary_file=${SUMMARY_FILE:-$result_root/summary.tsv}
failure_file=${FAILURE_FILE:-$result_root/failures.tsv}
resume=${RESUME:-true}
continue_on_error=${CONTINUE_ON_ERROR:-false}
cooldown_seconds=${COOLDOWN_SECONDS:-0}

for value in "$resume" "$continue_on_error"; do
  if [[ "$value" != true && "$value" != false ]]; then
    echo "RESUME and CONTINUE_ON_ERROR must be true or false" >&2
    exit 2
  fi
done
if ! [[ "$cooldown_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "COOLDOWN_SECONDS must be nonnegative" >&2
  exit 2
fi
if [[ ! -x "$single_case" ]]; then
  echo "Missing executable helper: $single_case" >&2
  exit 3
fi

degrees=(3 4 5 6 7)
sizes=(4 8 16 32 64 128 256 512 1024)
targets=(k p pinv_k)

mkdir -p "$result_root" "$console_root"
printf 'p\tn\ttarget\tmpi_ranks\tfree_dofs\tlambda_min\tlambda_max\tcondition_number\teps_iterations_min\teps_iterations_max\ttime_spectral_eps_s\tresult_file\n' > "$summary_file"
printf 'p\tn\ttarget\texit_status\tconsole_file\tresult_file\n' > "$failure_file"

is_complete() {
  local file=$1
  local expected_target=$2
  [[ -f "$file" ]] &&
    grep -qx 'eigensolver_status = converged' "$file" &&
    grep -qx "spectral_condition_target = $expected_target" "$file"
}

append_summary() {
  local file=$1
  awk -F ' = ' -v path="$file" '
    $1=="p" {p=$2}
    $1=="n" {n=$2; gsub(/[()]/, "", n); sub(/,.*/, "", n)}
    $1=="spectral_condition_target" {target=$2}
    $1=="mpi_ranks" {ranks=$2}
    $1=="total_free_dofs" {dofs=$2}
    $1=="lambda_min" {lmin=$2}
    $1=="lambda_max" {lmax=$2}
    $1=="eps_condition_number" {kappa=$2}
    $1=="eps_iterations_min" {imin=$2}
    $1=="eps_iterations_max" {imax=$2}
    $1=="time_spectral_eps_s" {time=$2}
    END {printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", p,n,target,ranks,dofs,lmin,lmax,kappa,imin,imax,time,path}
  ' "$file" >> "$summary_file"
}

completed=0
failed=0
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
        echo "[skip] completed p=$degree n=${elements}x${elements} target=$target"
        append_summary "$result_file"
        completed=$((completed + 1))
        continue
      fi

      echo "[batch $((completed + failed + 1))/$total] p=$degree n=${elements}x${elements} target=$target"
      if "$single_case" "$degree" "$elements" "$target" "$result_file" \
          2>&1 | tee "$console_file"; then
        if ! is_complete "$result_file" "$target"; then
          echo "Run exited successfully but result is not converged: $result_file" >&2
          status=1
        else
          append_summary "$result_file"
          completed=$((completed + 1))
          status=0
        fi
      else
        status=$?
      fi

      if (( status != 0 )); then
        failed=$((failed + 1))
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$degree" "$elements" "$target" "$status" \
          "$console_file" "$result_file" >> "$failure_file"
        echo "[failed] p=$degree n=${elements}x${elements} target=$target status=$status" >&2
        if [[ "$continue_on_error" != true ]]; then
          exit "$status"
        fi
      fi

      # Normally zero: process exit already releases all PETSc/MPI memory.
      # A nonzero override can give a batch scheduler or system monitor time.
      if [[ "$cooldown_seconds" != 0 ]]; then
        sleep "$cooldown_seconds"
      fi
    done
  done
done

echo "[done] completed=$completed failed=$failed summary=$summary_file failures=$failure_file"
(( failed == 0 ))
