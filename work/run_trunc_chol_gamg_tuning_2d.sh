#!/usr/bin/env bash
set -uo pipefail

# Four-stage manual tuning study for the 2D truncated-Cholesky preconditioner:
#   rtol      TUNING_P,TUNING_N: scan inner CG rtol at fixed GAMG settings.
#   threshold same case: scan GAMG graph threshold at a selected rtol.
#   smooth    same case: scan V-cycle level smoother steps.
#   transfer  apply the selected triple to p=4..8 and several mesh sizes.
#
# This script is safe by default: BATCH_DRY_RUN=1 only writes a manifest.
# Run the stages in order, inspect each summary, and explicitly carry the
# chosen setting to the next stage:
#
#   RUN_ID=gamg-study TUNING_STAGE=rtol BATCH_DRY_RUN=1 ./work/run_trunc_chol_gamg_tuning_2d.sh
#   RUN_ID=gamg-study TUNING_STAGE=rtol BATCH_DRY_RUN=0 ./work/run_trunc_chol_gamg_tuning_2d.sh
#   RUN_ID=gamg-study TUNING_STAGE=threshold SELECTED_INNER_RTOL=1e-6 BATCH_DRY_RUN=0 ./work/run_trunc_chol_gamg_tuning_2d.sh
#   RUN_ID=gamg-study TUNING_STAGE=smooth SELECTED_INNER_RTOL=1e-6 SELECTED_GAMG_THRESHOLD=0.05 BATCH_DRY_RUN=0 ./work/run_trunc_chol_gamg_tuning_2d.sh
#   RUN_ID=gamg-study TUNING_STAGE=transfer SELECTED_INNER_RTOL=1e-6 SELECTED_GAMG_THRESHOLD=0.05 SELECTED_GAMG_SMOOTH_STEPS=2 BATCH_DRY_RUN=0 ./work/run_trunc_chol_gamg_tuning_2d.sh
#
# Defaults and override grids:
#   INNER_RTOLS="1e-10 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2"
#   GAMG_THRESHOLDS="0.0 0.01 0.02 0.05 0.1"
#   GAMG_SMOOTH_STEPS="1 2 3 4"
#   TRANSFER_P_VALUES="4 5 6 7 8"
#   TRANSFER_N_VALUES="32 64 128 256"
#   TUNING_P=8 TUNING_N=256
# Fixed experiment policy:
#   outer GMRES: TUNING_LOR_PC=left_gmres_cg, rtol=1e-6, atol=1e-9
#   inner CG:    GAMG, atol=1e-6; rtol is stage-dependent
#   P:           truncated_cholesky_tensor, k=auto, boundary protection=true
#
# PETSc 3.24 options used below are scoped to the nested KSP by the inner_
# prefix: -inner_pc_gamg_threshold and -inner_mg_levels_ksp_max_it. The latter
# controls level smoother iterations per application; it is not GAMG's
# prolongator smoothing option (-pc_gamg_agg_nsmooths).

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
geometry_file="$script_dir/input/square_2d.txt"
stage=${TUNING_STAGE:-rtol}
dry_run=${BATCH_DRY_RUN:-1}
resume=${RESUME:-1}
timeout_sec=${TIMEOUT_SEC:-0}
fail_fast=${FAIL_FAST:-0}
mpi_rank=${MPI_RANK:-default}
maximum_mpi_ranks=${MAX_MPI_RANKS:-}
memory_limit=${TRUNC_CHOL_MAX_MEMORY_BYTES:-68719476736}
boundary_protection=${TRUNC_CHOL_BOUNDARY_PROTECTION:-true}
trunc_chol_k=${TRUNC_CHOL_K:-auto}
tuning_degree=${TUNING_P:-8}
tuning_size=${TUNING_N:-256}
lor_pc=${TUNING_LOR_PC:-left_gmres_cg}

outer_rtol=${OUTER_RTOL:-1.0e-6}
outer_atol=${OUTER_ATOL:-1.0e-9}
outer_dtol=${OUTER_DTOL:-1.0e4}
outer_max_it=${OUTER_MAX_IT:-10000}
inner_atol=${INNER_ATOL:-1.0e-6}
inner_dtol=${INNER_DTOL:-1.0e4}
inner_max_it=${INNER_MAX_IT:-10000}

read -r -a inner_rtols <<<"${INNER_RTOLS:-1e-10 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2}"
read -r -a gamg_thresholds <<<"${GAMG_THRESHOLDS:-0.0 0.01 0.02 0.05 0.1}"
read -r -a gamg_smooth_steps <<<"${GAMG_SMOOTH_STEPS:-1 2 3 4}"
read -r -a transfer_degrees <<<"${TRANSFER_P_VALUES:-4 5 6 7 8}"
read -r -a transfer_sizes <<<"${TRANSFER_N_VALUES:-32 64 128 256}"

selected_inner_rtol=${SELECTED_INNER_RTOL:-1.0e-6}
selected_threshold=${SELECTED_GAMG_THRESHOLD:-0.05}
selected_smooth_steps=${SELECTED_GAMG_SMOOTH_STEPS:-1}
base_threshold=${BASE_GAMG_THRESHOLD:-0.0}
base_smooth_steps=${BASE_GAMG_SMOOTH_STEPS:-1}

case "$stage" in rtol|threshold|smooth|transfer) ;; *)
  echo "TUNING_STAGE must be rtol, threshold, smooth, or transfer" >&2
  exit 2
esac
for boolean in "$dry_run" "$resume" "$fail_fast"; do
  [[ "$boolean" == 0 || "$boolean" == 1 ]] || {
    echo "BATCH_DRY_RUN, RESUME, and FAIL_FAST must be 0 or 1" >&2; exit 2;
  }
done
[[ "$boundary_protection" == true || "$boundary_protection" == false ]] || {
  echo "TRUNC_CHOL_BOUNDARY_PROTECTION must be true or false" >&2; exit 2;
}
[[ "$timeout_sec" =~ ^[0-9]+$ ]] || {
  echo "TIMEOUT_SEC must be a nonnegative integer" >&2; exit 2;
}
if (( timeout_sec > 0 )) && ! command -v timeout >/dev/null 2>&1; then
  echo "TIMEOUT_SEC requires the 'timeout' command" >&2
  exit 2
fi
[[ "$mpi_rank" == default || "$mpi_rank" =~ ^[1-9][0-9]*$ ]] || {
  echo "MPI_RANK must be default or a positive integer" >&2; exit 2;
}
[[ "$trunc_chol_k" == auto || "$trunc_chol_k" =~ ^[0-9]+$ ]] || {
  echo "TRUNC_CHOL_K must be auto or a nonnegative integer" >&2; exit 2;
}
[[ "$tuning_degree" =~ ^[4-8]$ ]] || {
  echo "TUNING_P must be in 4..8" >&2; exit 2;
}
[[ "$tuning_size" =~ ^[1-9][0-9]*$ ]] || {
  echo "TUNING_N must be a positive integer" >&2; exit 2;
}
case "$lor_pc" in left_gmres_cg|right_gmres_cg|right_fgmres_cg) ;; *)
  echo "TUNING_LOR_PC must be left_gmres_cg, right_gmres_cg, or right_fgmres_cg" >&2
  exit 2
esac
[[ "$selected_smooth_steps" =~ ^[1-9][0-9]*$ ]] || {
  echo "SELECTED_GAMG_SMOOTH_STEPS must be positive" >&2; exit 2;
}
for steps in "${gamg_smooth_steps[@]}"; do
  [[ "$steps" =~ ^[1-9][0-9]*$ ]] || {
    echo "GAMG_SMOOTH_STEPS entries must be positive integers" >&2; exit 2;
  }
done
for degree in "${transfer_degrees[@]}"; do
  [[ "$degree" =~ ^[4-8]$ ]] || {
    echo "TRANSFER_P_VALUES entries must be in 4..8" >&2; exit 2;
  }
done
for size in "${transfer_sizes[@]}"; do
  [[ "$size" =~ ^[1-9][0-9]*$ ]] || {
    echo "TRANSFER_N_VALUES entries must be positive integers" >&2; exit 2;
  }
done

short_commit=$(git -C "$project_dir" rev-parse --short HEAD 2>/dev/null || printf unknown)
run_id=${RUN_ID:-"gamg-tuning-$(date +%Y%m%d-%H%M%S)-$short_commit"}
run_root=${RUN_DIR:-"$script_dir/results/trunc_chol_gamg_tuning_2d/$run_id"}
case_dir="$run_root/cases/$stage"
log_dir="$run_root/logs/$stage"
status_dir="$run_root/status/$stage"
manifest="$run_root/manifest_${stage}.tsv"
summary="$run_root/summary_${stage}.tsv"
mkdir -p "$case_dir" "$log_dir" "$status_dir"

hash_text() {
  if command -v sha256sum >/dev/null 2>&1; then
    printf '%s' "$1" | sha256sum | awk '{print substr($1,1,12)}'
  else
    printf '%s' "$1" | shasum -a 256 | awk '{print substr($1,1,12)}'
  fi
}

result_value() {
  awk -F ' = ' -v key="$2" '$1 == key {print $2; exit}' "$1" 2>/dev/null
}

value_or_na() {
  local value
  value=$(result_value "$1" "$2")
  printf '%s' "${value:-NA}"
}

append_summary_row() {
  local case_id=$1 status=$2 return_code=$3 degree=$4 tuple=$5 rtol=$6
  local threshold=$7 smooth=$8 result_file=$9 console_file=${10} reason=${11}
  local source_result=$result_file
  [[ -s "$source_result" ]] || source_result=/dev/null
  local mpi_actual setup_pc setup_total inner_total inner_average
  local inner_maximum outer_iterations pc_applications solve_time inner_time
  local residual l2_error
  mpi_actual=$(value_or_na "$source_result" mpi_ranks)
  setup_pc=$(value_or_na "$source_result" time_pc_ksp_setup_s)
  setup_total=$(value_or_na "$source_result" "setup_time_seconds($lor_pc)")
  inner_total=$(value_or_na "$source_result" inner_iterations_total)
  inner_average=$(value_or_na "$source_result" "inner_iterations_average($lor_pc)")
  inner_maximum=$(value_or_na "$source_result" "inner_iterations_maximum($lor_pc)")
  outer_iterations=$(value_or_na "$source_result" outer_iterations)
  pc_applications=$(value_or_na "$source_result" pc_applications)
  solve_time=$(value_or_na "$source_result" time_outer_solve_s)
  inner_time=$(value_or_na "$source_result" time_pc_apply_total_s)
  residual=$(value_or_na "$source_result" relative_true_residual)
  l2_error=$(value_or_na "$source_result" l2_error)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_id" "$stage" "$lor_pc" "$status" "$return_code" "$degree" "$tuple" \
    "$rtol" "$inner_atol" "$threshold" "$smooth" "$mpi_actual" \
    "$setup_pc" "$setup_total" "$inner_total" "$inner_average" \
    "$inner_maximum" "$outer_iterations" "$pc_applications" "$solve_time" \
    "$inner_time" "$residual" "$l2_error" "$result_file" "$console_file" \
    "$reason" >> "$summary"
}

printf 'case_id\tstage\touter_mode\tp\tn\tinner_rtol\tinner_atol\tgamg_threshold\tgamg_smooth_steps\tmpi_ranks_requested\tconfig_hash\tcommand\n' > "$manifest.tmp"
printf 'case_id\tstage\touter_mode\tstatus\treturn_code\tp\tn\tinner_rtol\tinner_atol\tgamg_threshold\tgamg_smooth_steps\tmpi_ranks\tT_setup_pc_s\tT_setup_total_s\tN_inner\tN_inner_average\tN_inner_maximum\tN_outer\tpc_applications\tT_solve_s\tT_inner_total_s\trelative_true_residual\tL2_error\tresult_file\tconsole_file\treason\n' > "$summary.tmp"

declare -a records=()
add_case() {
  local degree=$1 size=$2 rtol=$3 threshold=$4 smooth=$5
  local tuple="$size,$size"
  local effective_k=$trunc_chol_k
  [[ "$effective_k" == auto ]] && effective_k=$(( (degree + 1) / 2 ))
  (( effective_k < degree - 1 )) || {
    echo "Invalid k=$effective_k for p=$degree; require k < p-1" >&2; exit 2;
  }
  local config="commit=$short_commit;stage=$stage;p=$degree;n=$tuple;k=$trunc_chol_k;bp=$boundary_protection;outer=$outer_rtol,$outer_atol,$outer_dtol,$outer_max_it;inner=$rtol,$inner_atol,$inner_dtol,$inner_max_it;threshold=$threshold;smooth=$smooth;ranks=$mpi_rank;max_ranks=$maximum_mpi_ranks;memory=$memory_limit"
  [[ "$lor_pc" == left_gmres_cg ]] || config+=";lor_pc=$lor_pc"
  local config_hash
  config_hash=$(hash_text "$config")
  local rtol_tag=${rtol//./d}; rtol_tag=${rtol_tag//-/m}; rtol_tag=${rtol_tag//+/p}
  local threshold_tag=${threshold//./d}; threshold_tag=${threshold_tag//-/m}; threshold_tag=${threshold_tag//+/p}
  local case_id="${stage}_p${degree}_n${size}_rtol${rtol_tag}_thr${threshold_tag}_smooth${smooth}_${config_hash}"
  [[ "$lor_pc" == left_gmres_cg ]] || \
    case_id="${stage}_${lor_pc}_p${degree}_n${size}_rtol${rtol_tag}_thr${threshold_tag}_smooth${smooth}_${config_hash}"
  local result_file="$case_dir/$case_id.result.log"
  local console_file="$log_dir/$case_id.console.log"
  local petsc_options="-inner_pc_gamg_threshold $threshold -inner_mg_levels_ksp_max_it $smooth"
  local command_text="PETSC_OPTIONS='$petsc_options' INNER_PC=gamg PRECONDITIONER_MATRIX_SOURCE=truncated_cholesky_tensor TRUNC_CHOL_K=$trunc_chol_k TRUNC_CHOL_BOUNDARY_PROTECTION=$boundary_protection COMPUTE_UNPRECONDITIONED_SOLUTION=false OUTER_RTOL=$outer_rtol OUTER_ATOL=$outer_atol INNER_RTOL=$rtol INNER_ATOL=$inner_atol ./work/run_case.sh $mpi_rank $geometry_file k_refine $degree $tuple greville $lor_pc $result_file"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_id" "$stage" "$lor_pc" "$degree" "$tuple" "$rtol" "$inner_atol" \
    "$threshold" "$smooth" "$mpi_rank" "$config_hash" "$command_text" \
    >> "$manifest.tmp"
  records+=("$case_id|$degree|$tuple|$rtol|$threshold|$smooth|$config_hash|$result_file|$console_file|$petsc_options")
}

case "$stage" in
  rtol)
    for rtol in "${inner_rtols[@]}"; do
      add_case "$tuning_degree" "$tuning_size" "$rtol" \
        "$base_threshold" "$base_smooth_steps"
    done
    ;;
  threshold)
    for threshold in "${gamg_thresholds[@]}"; do
      add_case "$tuning_degree" "$tuning_size" "$selected_inner_rtol" \
        "$threshold" "$base_smooth_steps"
    done
    ;;
  smooth)
    for smooth in "${gamg_smooth_steps[@]}"; do
      add_case "$tuning_degree" "$tuning_size" "$selected_inner_rtol" \
        "$selected_threshold" "$smooth"
    done
    ;;
  transfer)
    for degree in "${transfer_degrees[@]}"; do
      for size in "${transfer_sizes[@]}"; do
        add_case "$degree" "$size" "$selected_inner_rtol" \
          "$selected_threshold" "$selected_smooth_steps"
      done
    done
    ;;
esac
mv "$manifest.tmp" "$manifest"
mv "$summary.tmp" "$summary"

echo "Planned ${#records[@]} stage '$stage' cases in $run_root"
echo "Manifest: $manifest"
echo "Summary:  $summary"
if [[ "$dry_run" == 1 ]]; then
  echo "Dry run only. Re-run with the same RUN_ID and BATCH_DRY_RUN=0."
  exit 0
fi

failed=0
for record in "${records[@]}"; do
  IFS='|' read -r case_id degree tuple rtol threshold smooth config_hash \
    result_file console_file petsc_options <<<"$record"
  success_marker="$status_dir/$case_id.success"
  if [[ "$resume" == 1 && -s "$result_file" && -f "$success_marker" &&
        "$(<"$success_marker")" == "$config_hash" ]]; then
    echo "[resume] $case_id"
    append_summary_row "$case_id" resumed 0 "$degree" "$tuple" "$rtol" \
      "$threshold" "$smooth" "$result_file" "$console_file" \
      existing_success_with_matching_hash
    continue
  fi

  result_tmp="$case_dir/.$case_id.result.tmp.$$"
  console_tmp="$log_dir/.$case_id.console.tmp.$$"
  command=(env
    PETSC_OPTIONS="$petsc_options"
    INNER_PC=gamg
    PRECONDITIONER_MATRIX_SOURCE=truncated_cholesky_tensor
    TRUNC_CHOL_K="$trunc_chol_k"
    TRUNC_CHOL_BOUNDARY_PROTECTION="$boundary_protection"
    TRUNC_CHOL_MAX_MEMORY_BYTES="$memory_limit"
    COMPUTE_TRUNCATION_CERTIFICATE=true
    COMPUTE_SOLUTION=true COMPUTE_L2_ERROR=true
    COMPUTE_UNPRECONDITIONED_SOLUTION=false
    COMPUTE_CONDITION_NUMBER=false
    COMPUTE_SPECTRAL_CONDITION_NUMBER=false
    VERIFY_TENSOR_IDENTITY=false
    OUTER_RTOL="$outer_rtol" OUTER_ATOL="$outer_atol"
    OUTER_DTOL="$outer_dtol" OUTER_MAX_IT="$outer_max_it"
    INNER_RTOL="$rtol" INNER_ATOL="$inner_atol"
    INNER_DTOL="$inner_dtol" INNER_MAX_IT="$inner_max_it")
  [[ -n "$maximum_mpi_ranks" ]] && command+=(MAX_MPI_RANKS="$maximum_mpi_ranks")
  command+=("$script_dir/run_case.sh" "$mpi_rank" "$geometry_file"
    k_refine "$degree" "$tuple" greville "$lor_pc" "$result_tmp")

  printf '[command]' > "$console_tmp"
  printf ' %q' "${command[@]}" >> "$console_tmp"
  printf '\n' >> "$console_tmp"
  if (( timeout_sec > 0 )); then
    timeout "$timeout_sec" "${command[@]}" >> "$console_tmp" 2>&1
  else
    "${command[@]}" >> "$console_tmp" 2>&1
  fi
  return_code=$?

  status=failed
  reason="driver_returned_$return_code"
  if [[ $return_code -eq 124 ]]; then
    status=timeout; reason=timeout
  elif grep -q 'preflight conservative estimate.*exceeds configured limit' "$console_tmp"; then
    status=skipped_memory; reason=preconditioner_memory_preflight
  elif [[ $return_code -eq 0 && -s "$result_tmp" ]]; then
    status=success; reason=completed
  fi
  mv "$console_tmp" "$console_file"
  if [[ "$status" == success ]]; then
    mv "$result_tmp" "$result_file"
    printf '%s' "$config_hash" > "$success_marker"
  elif [[ -e "$result_tmp" ]]; then
    mv "$result_tmp" "$case_dir/$case_id.partial.log"
  fi

  append_summary_row "$case_id" "$status" "$return_code" "$degree" \
    "$tuple" "$rtol" "$threshold" "$smooth" "$result_file" \
    "$console_file" "$reason"
  solve_time=$(value_or_na "$result_file" time_outer_solve_s)
  inner_total=$(value_or_na "$result_file" inner_iterations_total)
  echo "[$status] $case_id T_solve=$solve_time N_inner=$inner_total"
  if [[ "$status" == failed || "$status" == timeout ]]; then
    failed=$((failed + 1))
    [[ "$fail_fast" == 1 ]] && break
  fi
done

echo "Stage '$stage' finished with $failed failed/timeout cases."
echo "Compare T_solve_s first; N_inner alone is not the optimization target."
echo "Summary: $summary"
(( failed == 0 ))
