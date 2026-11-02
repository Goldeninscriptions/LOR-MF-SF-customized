#!/usr/bin/env bash
set -euo pipefail

# Shared execution engine for geometry-specific batch scripts. This file is
# sourced after the caller defines its geometry, cases, outputs, and options.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
planner="$project_dir/build/case_planner"

# Fail early when this implementation file is accidentally executed directly
# or a geometry-specific entry point omits part of its public configuration.
required_scalars=(
  batch_name geometry_file refinement result_dir console_dir summary_file
  lor_points mpi_rank maximum_mpi_ranks lor_pc compute_solution
  compute_l2_error compute_condition_number svd_tol svd_max_it svd_ncv
)
for required_name in "${required_scalars[@]}"; do
  if ! declare -p "$required_name" >/dev/null 2>&1; then
    echo "Missing batch configuration: $required_name" >&2
    exit 2
  fi
done
if ! declare -p degrees >/dev/null 2>&1 ||
   ! declare -p element_cases >/dev/null 2>&1; then
  echo "Missing batch configuration: degrees or element_cases" >&2
  exit 2
fi
if [[ ${#degrees[@]} -eq 0 || ${#element_cases[@]} -eq 0 ]]; then
  echo "degrees and element_cases must not be empty" >&2
  exit 2
fi

# Supply stable fallbacks only for parameters absent from the selected block.
# An uncommented scheme block in the caller always takes precedence.
inner_pc=${inner_pc:-${INNER_PC:-jacobi}}
inner_rtol=${inner_rtol:-${INNER_RTOL:-1.0e-10}}
inner_atol=${inner_atol:-${INNER_ATOL:-1.0e-10}}
inner_max_it=${inner_max_it:-${INNER_MAX_IT:-10000}}
fixed_jacobi_steps=${fixed_jacobi_steps:-${FIXED_JACOBI_STEPS:-4}}
fixed_jacobi_omega=${fixed_jacobi_omega:-${FIXED_JACOBI_OMEGA:-0.6666666666666667}}

mkdir -p "$result_dir" "$console_dir"

total=$(( ${#degrees[@]} * ${#element_cases[@]} ))
case_number=0
failed_cases=0
batch_start=$(date +%s)
planned_ranks=()

# Validate the common rank selection before planning any case.
if [[ "$mpi_rank" != "default" ]] && ! [[ "$mpi_rank" =~ ^[1-9][0-9]*$ ]]; then
  echo "mpi_rank must be 'default' or a positive integer" >&2
  exit 2
fi
if [[ "$mpi_rank" == "default" && ! -x "$planner" ]]; then
  echo "Missing executable: $planner" >&2
  echo "Build it with: cd $project_dir/build && cmake ../examples && make -j" >&2
  exit 3
fi

# Reject inconsistent feature selections before creating any result files.
for feature_value in \
  "$compute_solution" "$compute_l2_error" "$compute_condition_number"; do
  if [[ "$feature_value" != "true" && "$feature_value" != "false" ]]; then
    echo "compute_solution, compute_l2_error, and compute_condition_number must be true or false" >&2
    exit 2
  fi
done
if [[ "$compute_l2_error" == "true" && "$compute_solution" != "true" ]]; then
  echo "compute_l2_error=true requires compute_solution=true" >&2
  exit 2
fi
# Validate the complete scheme and describe only the settings it actually uses.
case "$lor_pc" in
  left_cg_lu)
    preconditioner_side="left"
    outer_rtol=${outer_rtol:-${OUTER_RTOL:-1.0e-12}}
    outer_atol=${outer_atol:-${OUTER_ATOL:-1.0e-14}}
    outer_dtol=${outer_dtol:-${OUTER_DTOL:-1.0e8}}
    outer_max_it=${outer_max_it:-${OUTER_MAX_IT:-10000}}
    scheme_settings="outer_rtol=$outer_rtol,outer_atol=$outer_atol,outer_dtol=$outer_dtol,outer_max_it=$outer_max_it,exact_lor_lu"
    setting_tag="lu"
    ;;
  right_gmres_lu)
    preconditioner_side="right"
    outer_rtol=${outer_rtol:-${OUTER_RTOL:-1.0e-10}}
    outer_atol=${outer_atol:-${OUTER_ATOL:-1.0e-10}}
    outer_dtol=${outer_dtol:-${OUTER_DTOL:-1.0e4}}
    outer_max_it=${outer_max_it:-${OUTER_MAX_IT:-10000}}
    scheme_settings="outer_rtol=$outer_rtol,outer_atol=$outer_atol,outer_dtol=$outer_dtol,outer_max_it=$outer_max_it,exact_lor_lu"
    setting_tag="lu"
    ;;
  left_gmres_cg|right_gmres_cg|right_fgmres_cg)
    [[ "$lor_pc" == left_* ]] && preconditioner_side="left" || preconditioner_side="right"
    outer_rtol=${outer_rtol:-${OUTER_RTOL:-1.0e-10}}
    outer_atol=${outer_atol:-${OUTER_ATOL:-1.0e-10}}
    outer_dtol=${outer_dtol:-${OUTER_DTOL:-1.0e4}}
    outer_max_it=${outer_max_it:-${OUTER_MAX_IT:-10000}}
    if [[ "$inner_pc" != "jacobi" && "$inner_pc" != "gamg" ]]; then
      echo "inner_pc must be jacobi or gamg for a *_cg lor_pc" >&2
      exit 2
    fi
    scheme_settings="outer_rtol=$outer_rtol,outer_atol=$outer_atol,outer_dtol=$outer_dtol,outer_max_it=$outer_max_it,inner_pc=$inner_pc,inner_rtol=$inner_rtol,inner_atol=$inner_atol,inner_max_it=$inner_max_it"
    setting_tag="inner_${inner_pc}"
    ;;
  left_gmres_fixed_jacobi)
    preconditioner_side="left"
    outer_rtol=${outer_rtol:-${OUTER_RTOL:-1.0e-10}}
    outer_atol=${outer_atol:-${OUTER_ATOL:-1.0e-10}}
    outer_dtol=${outer_dtol:-${OUTER_DTOL:-1.0e4}}
    outer_max_it=${outer_max_it:-${OUTER_MAX_IT:-10000}}
    scheme_settings="outer_rtol=$outer_rtol,outer_atol=$outer_atol,outer_dtol=$outer_dtol,outer_max_it=$outer_max_it,fixed_steps=$fixed_jacobi_steps,fixed_omega=$fixed_jacobi_omega"
    setting_tag="fixed_m${fixed_jacobi_steps}"
    ;;
  *)
    echo "unsupported lor_pc: $lor_pc" >&2
    exit 2
    ;;
esac

# Reuse the production refinement and ID path in the serial planner, retaining
# one exact free-DOF-based MPI recommendation for every case in traversal order.
for degree in "${degrees[@]}"; do
  for element_tuple in "${element_cases[@]}"; do
    if [[ "$mpi_rank" == "default" ]]; then
      planner_command=(
        "$planner"
        -geometry "$geometry_file"
        -refinement "$refinement"
        -p "$degree"
        -n "$element_tuple"
        --recommended-ranks-only
      )
      if [[ -n "$maximum_mpi_ranks" ]]; then
        planner_command+=(--max-ranks "$maximum_mpi_ranks")
      fi
      planned_ranks+=("$("${planner_command[@]}")")
    else
      planned_ranks+=("$mpi_rank")
    fi
  done
done

# Print the complete deterministic run order before starting a potentially
# long batch, and initialize the machine-readable tab-separated summary.
echo "Planned $total cases for $batch_name from geometry: $geometry_file"
plan_index=0
for degree in "${degrees[@]}"; do
  for element_tuple in "${element_cases[@]}"; do
    mpi_ranks=${planned_ranks[$plan_index]}
    plan_index=$((plan_index + 1))
    echo "  mpi_ranks=$mpi_ranks refinement=$refinement p=$degree n=($element_tuple) point_set=$lor_points mode=$lor_pc side=$preconditioner_side settings=[$scheme_settings] solve=$compute_solution L2=$compute_l2_error condition=$compute_condition_number svd_tol=$svd_tol svd_max_it=$svd_max_it svd_ncv=$svd_ncv"
  done
done

# BATCH_DRY_RUN=1 prints and validates the case plan without launching solves.
if [[ ${BATCH_DRY_RUN:-0} == 1 ]]; then
  echo "Dry run completed; no cases were launched."
  exit 0
fi

printf 'case\tstatus\tmpi_ranks\trefinement\tp\tn\tpoint_set\tlor_pc\tpreconditioner_side\tscheme_settings\tcompute_solution\tcompute_l2_error\tcompute_condition_number\tsvd_tol\tsvd_max_it\tsvd_ncv\telapsed_s\tresult\tconsole\n' > "$summary_file"

plan_index=0
for degree in "${degrees[@]}"; do
  for element_tuple in "${element_cases[@]}"; do
    mpi_ranks=${planned_ranks[$plan_index]}
    plan_index=$((plan_index + 1))
    case_number=$((case_number + 1))
    element_tag=${element_tuple//,/x}
    stem="mpi${mpi_ranks}_${refinement}_p${degree}_n${element_tag}_${lor_points}_${lor_pc}_${preconditioner_side}_${setting_tag}_solve${compute_solution}_l2${compute_l2_error}_cond${compute_condition_number}"
    result_file="$result_dir/${stem}.log"
    console_file="$console_dir/${stem}.console.log"
    case_start=$(date +%s)

    echo "[$case_number/$total] starting mpi_ranks=$mpi_ranks refinement=$refinement p=$degree n=($element_tuple) point_set=$lor_points mode=$lor_pc side=$preconditioner_side settings=[$scheme_settings] solve=$compute_solution L2=$compute_l2_error condition=$compute_condition_number"
    scheme_environment=()
    case "$lor_pc" in
      left_gmres_cg|right_gmres_cg|right_fgmres_cg)
        scheme_environment+=(
          "INNER_PC=$inner_pc"
          "INNER_RTOL=$inner_rtol"
          "INNER_ATOL=$inner_atol"
          "INNER_MAX_IT=$inner_max_it"
        )
        ;;
      left_gmres_fixed_jacobi)
        scheme_environment+=(
          "FIXED_JACOBI_STEPS=$fixed_jacobi_steps"
          "FIXED_JACOBI_OMEGA=$fixed_jacobi_omega"
        )
        ;;
    esac
    if env "${scheme_environment[@]}" \
      OUTER_RTOL="$outer_rtol" OUTER_ATOL="$outer_atol" \
      OUTER_DTOL="$outer_dtol" OUTER_MAX_IT="$outer_max_it" \
      COMPUTE_SOLUTION="$compute_solution" COMPUTE_L2_ERROR="$compute_l2_error" \
      COMPUTE_CONDITION_NUMBER="$compute_condition_number" \
      SVD_TOL="$svd_tol" SVD_MAX_IT="$svd_max_it" SVD_NCV="$svd_ncv" \
      "$script_dir/run_case.sh" "$mpi_ranks" "$geometry_file" "$refinement" "$degree" \
      "$element_tuple" "$lor_points" "$lor_pc" "$result_file" \
      2>&1 | tee "$console_file"; then
      status="success"
    else
      status="failed"
      failed_cases=$((failed_cases + 1))
    fi

    case_end=$(date +%s)
    elapsed=$((case_end-case_start))
    printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\n' \
      "$case_number" "$status" "$mpi_ranks" "$refinement" "$degree" "$element_tuple" \
      "$lor_points" "$lor_pc" "$preconditioner_side" "$scheme_settings" "$compute_solution" "$compute_l2_error" \
      "$compute_condition_number" "$svd_tol" "$svd_max_it" "$svd_ncv" \
      "$elapsed" "$result_file" "$console_file" \
      >> "$summary_file"
    echo "[$case_number/$total] $status in $elapsed s; result=$result_file"
  done
done

batch_end=$(date +%s)
echo "Batch completed: $total cases, $failed_cases failed, $((batch_end-batch_start)) s"
echo "Summary: $summary_file"

# Return a failing batch status after every planned case has been attempted.
if [[ $failed_cases -ne 0 ]]; then
  exit 1
fi
