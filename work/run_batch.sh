#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
planner="$project_dir/build/case_planner"

# -------------------------- Batch configuration ---------------------------
# Two-dimensional example. For 3D, select a 3D geometry and use tuples such
# as "4,4,4" and "16,16,16".
geometry_file="$project_dir/work/input/square_2d.txt"

# Pure p-refinement template for square_2d.txt. Polynomial degree changes,
# while n must remain equal to the initial n=(1,1).
# refinement="p_refine"
# degrees=(3 4 5 6 7)
# element_cases=("1,1")

# Pure h-refinement template for square_2d.txt. Element counts change, while
# p must remain equal to the initial p=1.
# refinement="h_refine"
# degrees=(1)
# element_cases=("4,4" "16,16" "128,128")

# Active k-refinement batch: elevate the common degree first and then insert
# knots independently to reach each directional element-count tuple.
refinement="k_refine"
degrees=(3 4 5 6 7)
element_cases=("4,4" "8,8" "16,16" "32,32" "64,64" "128,128" "256,256" "512,512" "1024,1024")

# Result and real-time console logs are separated by p, n, and solver mode.
result_dir="$script_dir/results"
console_dir="$script_dir/logs"
summary_file="$script_dir/batch_summary.log"

# Active LOR point-set and solver presets.
lor_points="greville"
# lor_points="demko"

# Select one explicit rank count for every case, or let the planner choose a
# separate count from each case's exact free DOFs. Bash assignments have no
# spaces around '='.
mpi_rank=${MPI_RANK:-default}
# mpi_rank=8

# In default mode the planner detects macOS/Linux logical CPUs. This optional
# environment value may impose a smaller resource or scheduler limit.
maximum_mpi_ranks=${MAX_MPI_RANKS:-}

lor_pc="gmres_shell"
# lor_pc="cg_lu"
preconditioner_side=${PRECONDITIONER_SIDE:-left}
# preconditioner_side="right"

# SLEPc controls for the potentially difficult smallest singular values.
svd_tol="1.0e-8"
svd_max_it="100000"
svd_ncv="100"
# -------------------------------------------------------------------------

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
echo "Planned $total cases from geometry: $geometry_file"
plan_index=0
for degree in "${degrees[@]}"; do
  for element_tuple in "${element_cases[@]}"; do
    mpi_ranks=${planned_ranks[$plan_index]}
    plan_index=$((plan_index + 1))
    echo "  mpi_ranks=$mpi_ranks refinement=$refinement p=$degree n=($element_tuple) point_set=$lor_points mode=$lor_pc side=$preconditioner_side svd_tol=$svd_tol svd_max_it=$svd_max_it svd_ncv=$svd_ncv"
  done
done

# BATCH_DRY_RUN=1 prints and validates the case plan without launching solves.
if [[ ${BATCH_DRY_RUN:-0} == 1 ]]; then
  echo "Dry run completed; no cases were launched."
  exit 0
fi

printf 'case\tstatus\tmpi_ranks\trefinement\tp\tn\tpoint_set\tlor_pc\tpreconditioner_side\tsvd_tol\tsvd_max_it\tsvd_ncv\telapsed_s\tresult\tconsole\n' > "$summary_file"

plan_index=0
for degree in "${degrees[@]}"; do
  for element_tuple in "${element_cases[@]}"; do
    mpi_ranks=${planned_ranks[$plan_index]}
    plan_index=$((plan_index + 1))
    case_number=$((case_number + 1))
    element_tag=${element_tuple//,/x}
    stem="mpi${mpi_ranks}_${refinement}_p${degree}_n${element_tag}_${lor_points}_${lor_pc}_${preconditioner_side}"
    result_file="$result_dir/${stem}.log"
    console_file="$console_dir/${stem}.console.log"
    case_start=$(date +%s)

    echo "[$case_number/$total] starting mpi_ranks=$mpi_ranks refinement=$refinement p=$degree n=($element_tuple) point_set=$lor_points mode=$lor_pc side=$preconditioner_side"
    if PRECONDITIONER_SIDE="$preconditioner_side" \
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
    printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\n' \
      "$case_number" "$status" "$mpi_ranks" "$refinement" "$degree" "$element_tuple" \
      "$lor_points" "$lor_pc" "$preconditioner_side" "$svd_tol" "$svd_max_it" "$svd_ncv" \
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
