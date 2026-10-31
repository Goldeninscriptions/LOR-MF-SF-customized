#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_case.sh GEOMETRY_FILE REFINEMENT P N_TUPLE LOR_POINTS LOR_PC OUTPUT_FILE \
#       [OUTER_RTOL OUTER_ATOL INNER_RTOL INNER_ATOL]
# Example:
#   ./run_case.sh input/square_2d.txt k_refine 5 16,16 greville cg_lu \
#       results/p5_n16x16_cg_lu.log 1e-10 1e-12 1e-10 1e-10
# Alternative GMRES-shell example (uncomment to use as a command):
# ./run_case.sh input/square_2d.txt k_refine 5 16,16 demko gmres_shell \
#     results/p5_n16x16_gmres_shell.log 1e-10 1e-10 1e-10 1e-10

if [[ $# -ne 7 && $# -ne 11 ]]; then
  echo "Usage: $0 GEOMETRY_FILE REFINEMENT P N_TUPLE LOR_POINTS LOR_PC OUTPUT_FILE [OUTER_RTOL OUTER_ATOL INNER_RTOL INNER_ATOL]" >&2
  exit 2
fi

geometry_file=$1
refinement=$2
degree=$3
element_tuple=$4
lor_points=$5
lor_pc=$6
output_file=$7

# Reject unknown refinement policies before starting the executable.
case "$refinement" in
  p_refine|h_refine|k_refine) ;;
  *)
    echo "REFINEMENT must be p_refine, h_refine, or k_refine" >&2
    exit 2
    ;;
esac

# Accept only the two point-set strategies implemented by LORPointSet.
case "$lor_points" in
  greville|demko) ;;
  *)
    echo "LOR_POINTS must be greville or demko" >&2
    exit 2
    ;;
esac

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
driver="$project_dir/build/assembled_driver"

if [[ ! -x "$driver" ]]; then
  echo "Missing executable: $driver" >&2
  echo "Build it with: cd $project_dir/build && cmake ../examples && make -j" >&2
  exit 3
fi

mkdir -p "$(dirname "$output_file")"

# Mode-specific default residual settings.
if [[ "$lor_pc" == "cg_lu" ]]; then
  outer_rtol="1.0e-10"
  outer_atol="1.0e-12"
  outer_dtol="1.0e8"
elif [[ "$lor_pc" == "gmres_shell" ]]; then
  outer_rtol="1.0e-10"
  outer_atol="1.0e-10"
  outer_dtol="1.0e4"
else
  echo "LOR_PC must be cg_lu or gmres_shell" >&2
  exit 2
fi
outer_max_it="10000"
inner_rtol="1.0e-10"
inner_atol="1.0e-10"
inner_dtol="1.0e4"
inner_max_it="10000"

# SLEPc defaults may be overridden through environment variables without
# lengthening the positional single-case interface.
svd_tol=${SVD_TOL:-1.0e-8}
svd_max_it=${SVD_MAX_IT:-100000}
svd_ncv=${SVD_NCV:-100}

# Explicit command-line residuals override the mode defaults.
if [[ $# -eq 11 ]]; then
  outer_rtol=$8
  outer_atol=$9
  inner_rtol=${10}
  inner_atol=${11}
fi

common_options=(
  -geometry "$geometry_file"
  -refinement "$refinement"
  -p "$degree"
  -n "$element_tuple"
  -lor_points "$lor_points"
  -lor_pc "$lor_pc"
  -outer_rtol "$outer_rtol"
  -outer_atol "$outer_atol"
  -outer_dtol "$outer_dtol"
  -outer_max_it "$outer_max_it"
  -svd_tol "$svd_tol"
  -svd_max_it "$svd_max_it"
  -svd_ncv "$svd_ncv"
  -output "$output_file"
)

if [[ "$lor_pc" == "gmres_shell" ]]; then
  common_options+=(
    -inner_rtol "$inner_rtol"
    -inner_atol "$inner_atol"
    -inner_dtol "$inner_dtol"
    -inner_max_it "$inner_max_it"
  )
fi

echo "[run] refinement=$refinement requested_p=$degree requested_n=($element_tuple) point_set=$lor_points lor_pc=$lor_pc svd_tol=$svd_tol svd_max_it=$svd_max_it svd_ncv=$svd_ncv"
"$driver" "${common_options[@]}"
