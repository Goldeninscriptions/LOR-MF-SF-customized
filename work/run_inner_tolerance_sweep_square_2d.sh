#!/usr/bin/env bash
set -euo pipefail

# Compare approximate LOR inverse costs at fixed geometry, degree, mesh, MPI
# size, and outer tolerance. The default case targets the observed 512x512
# bottleneck; environment variables can reduce it for preliminary checks.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
geometry_file="$script_dir/input/square_2d.txt"
degree=${SWEEP_P:-3}
element_tuple=${SWEEP_N:-512,512}
mpi_rank=${MPI_RANK:-default}
outer_rtol=${OUTER_RTOL:-1.0e-10}
outer_atol=${OUTER_ATOL:-1.0e-10}
inner_atol=${INNER_ATOL:-1.0e-50}
inner_pc=${INNER_PC:-gamg}
compute_l2_error=${COMPUTE_L2_ERROR:-true}
read -r -a inner_rtols <<< "${INNER_RTOLS:-1e-10 1e-6 1e-4 1e-3 1e-2}"

# Reject empty or inconsistent sweeps before creating partial experiment data.
if [[ ${#inner_rtols[@]} -eq 0 ]]; then
  echo "INNER_RTOLS must contain at least one tolerance" >&2
  exit 2
fi
if [[ "$compute_l2_error" != true && "$compute_l2_error" != false ]]; then
  echo "COMPUTE_L2_ERROR must be true or false" >&2
  exit 2
fi
if [[ "$inner_pc" != jacobi && "$inner_pc" != gamg ]]; then
  echo "INNER_PC must be jacobi or gamg" >&2
  exit 2
fi

# Give every tolerance its own immutable result and console file, then collect
# the solver quantities needed to identify the minimum-time configuration.
element_tag=${element_tuple//,/x}
sweep_name="inner_tolerance_square_2d_${inner_pc}_p${degree}_n${element_tag}"
result_dir="$script_dir/results/$sweep_name"
console_dir="$script_dir/logs/$sweep_name"
summary_file="$script_dir/${sweep_name}.tsv"
mkdir -p "$result_dir" "$console_dir"

printf 'inner_pc\tinner_rtol\tstatus\tmpi_ranks\telapsed_s\touter_iterations_K\touter_iterations_LOR\tpc_applications\tinner_iterations_total\tinner_iterations_average\tsetup_time_K_s\tsetup_time_LOR_s\touter_time_K_s\touter_time_LOR_s\tinner_time_s\trelative_residual_K\trelative_residual_LOR\tL2_error_K\tL2_error_LOR\tresult\tconsole\n' > "$summary_file"

# Read one exact key from the line-oriented experiment result format.
result_value()
{
  local key=$1
  local file=$2
  local value
  value=$(awk -F ' = ' -v wanted="$key" '$1 == wanted {print $2; exit}' "$file")
  printf '%s' "${value:-NA}"
}

failed_cases=0
for inner_rtol in "${inner_rtols[@]}"; do
  # Sanitize scientific notation only for filenames; numerical arguments are
  # passed unchanged to PETSc.
  tolerance_tag=${inner_rtol//+/p}
  tolerance_tag=${tolerance_tag//-/m}
  tolerance_tag=${tolerance_tag//./d}
  stem="p${degree}_n${element_tag}_${inner_pc}_inner_rtol_${tolerance_tag}"
  result_file="$result_dir/${stem}.log"
  console_file="$console_dir/${stem}.console.log"

  echo "[sweep] p=$degree n=($element_tuple) inner_pc=$inner_pc inner_rtol=$inner_rtol inner_atol=$inner_atol"
  if [[ ${BATCH_DRY_RUN:-0} == 1 ]]; then
    continue
  fi

  start_time=$(date +%s)
  if PRECONDITIONER_SIDE=left INNER_PC="$inner_pc" \
     COMPUTE_SOLUTION=true \
     COMPUTE_L2_ERROR="$compute_l2_error" \
     COMPUTE_CONDITION_NUMBER=false \
     "$script_dir/run_case.sh" \
       "$mpi_rank" "$geometry_file" k_refine "$degree" "$element_tuple" \
       greville gmres_shell "$result_file" \
       "$outer_rtol" "$outer_atol" "$inner_rtol" "$inner_atol" \
       2>&1 | tee "$console_file"; then
    status=success
  else
    status=failed
    failed_cases=$((failed_cases + 1))
  fi
  elapsed=$(( $(date +%s) - start_time ))

  if [[ "$status" == success ]]; then
    printf '%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$inner_pc" "$inner_rtol" "$status" \
      "$(result_value mpi_ranks "$result_file")" "$elapsed" \
      "$(result_value 'iterations(K)' "$result_file")" \
      "$(result_value 'iterations(K_LOR^{-1}K)' "$result_file")" \
      "$(result_value 'pc_applications(K_LOR^{-1}K)' "$result_file")" \
      "$(result_value 'inner_iterations_total(K_LOR^{-1}K)' "$result_file")" \
      "$(result_value 'inner_iterations_average(K_LOR^{-1}K)' "$result_file")" \
      "$(result_value 'setup_time_seconds(K)' "$result_file")" \
      "$(result_value 'setup_time_seconds(K_LOR^{-1}K)' "$result_file")" \
      "$(result_value 'outer_solve_time_seconds(K)' "$result_file")" \
      "$(result_value 'outer_solve_time_seconds(K_LOR^{-1}K)' "$result_file")" \
      "$(result_value 'inner_solve_time_seconds(K_LOR^{-1}K)' "$result_file")" \
      "$(result_value 'relative_residual(K)' "$result_file")" \
      "$(result_value 'relative_residual(K_LOR^{-1}K)' "$result_file")" \
      "$(result_value 'L2_error(K)' "$result_file")" \
      "$(result_value 'L2_error(K_LOR^{-1}K)' "$result_file")" \
      "$result_file" "$console_file" >> "$summary_file"
  else
    printf '%s\t%s\t%s\tNA\t%d\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t%s\t%s\n' \
      "$inner_pc" "$inner_rtol" "$status" "$elapsed" \
      "$result_file" "$console_file" \
      >> "$summary_file"
  fi
done

if [[ ${BATCH_DRY_RUN:-0} == 1 ]]; then
  echo "Dry run completed; no tolerance cases were launched."
  exit 0
fi

echo "Tolerance sweep completed: $failed_cases failed"
echo "Summary: $summary_file"
if [[ $failed_cases -ne 0 ]]; then
  exit 1
fi
