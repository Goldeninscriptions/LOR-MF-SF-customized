#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)

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
element_cases=("4,4" "16,16" "128,128")

# Result and real-time console logs are separated by p, n, and solver mode.
result_dir="$script_dir/results"
console_dir="$script_dir/logs"
summary_file="$script_dir/batch_summary.log"

# Active LOR point-set and solver presets.
lor_points="greville"
# lor_points="demko"

lor_pc="cg_lu"
# lor_pc="gmres_shell"
# -------------------------------------------------------------------------

mkdir -p "$result_dir" "$console_dir"

total=$(( ${#degrees[@]} * ${#element_cases[@]} ))
case_number=0
failed_cases=0
batch_start=$(date +%s)

# Print the complete deterministic run order before starting a potentially
# long batch, and initialize the machine-readable tab-separated summary.
echo "Planned $total cases from geometry: $geometry_file"
for degree in "${degrees[@]}"; do
  for element_tuple in "${element_cases[@]}"; do
    echo "  refinement=$refinement p=$degree n=($element_tuple) point_set=$lor_points mode=$lor_pc"
  done
done

# BATCH_DRY_RUN=1 prints and validates the case plan without launching solves.
if [[ ${BATCH_DRY_RUN:-0} == 1 ]]; then
  echo "Dry run completed; no cases were launched."
  exit 0
fi

printf 'case\tstatus\trefinement\tp\tn\tpoint_set\tlor_pc\telapsed_s\tresult\tconsole\n' > "$summary_file"

for degree in "${degrees[@]}"; do
  for element_tuple in "${element_cases[@]}"; do
    case_number=$((case_number + 1))
    element_tag=${element_tuple//,/x}
    stem="${refinement}_p${degree}_n${element_tag}_${lor_points}_${lor_pc}"
    result_file="$result_dir/${stem}.log"
    console_file="$console_dir/${stem}.console.log"
    case_start=$(date +%s)

    echo "[$case_number/$total] starting refinement=$refinement p=$degree n=($element_tuple) point_set=$lor_points mode=$lor_pc"
    if "$script_dir/run_case.sh" "$geometry_file" "$refinement" "$degree" \
      "$element_tuple" "$lor_points" "$lor_pc" "$result_file" \
      2>&1 | tee "$console_file"; then
      status="success"
    else
      status="failed"
      failed_cases=$((failed_cases + 1))
    fi

    case_end=$(date +%s)
    elapsed=$((case_end-case_start))
    printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\n' \
      "$case_number" "$status" "$refinement" "$degree" "$element_tuple" \
      "$lor_points" "$lor_pc" "$elapsed" "$result_file" "$console_file" \
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
