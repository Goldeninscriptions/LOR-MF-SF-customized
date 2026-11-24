#!/usr/bin/env bash
set -euo pipefail
if (( $# < 4 )); then
  echo "Usage: $0 P EX EY RESULT_DIR [assembled_driver options...]" >&2
  exit 2
fi
p=$1; ex=$2; ey=$3; result_dir=$4; shift 4
[[ "$p" =~ ^([2-9]|[1-9][0-9]+)$ ]] || { echo "P must be >=2" >&2; exit 2; }
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
driver=${DRIVER:-$project_dir/build/assembled_driver}
launcher=${MPIEXEC:-mpirun}; ranks=${MPI_RANKS:-1}
geometry=${GEOMETRY:-$script_dir/input/square_2d.txt}
mkdir -p "$result_dir"
common=(
  -geometry "$geometry" -refinement k_refine -p "$p" -n "$ex,$ey"
  -lor_points greville -lor_pc left_cg_lu -preconditioner_side left
  -compute_solution true -compute_unpreconditioned_solution true
  -compute_l2_error true -compute_condition_number false
  -compute_spectral_condition_number "${COMPUTE_SPECTRUM:-true}"
  -spectral_condition_target pinv_k
  -outer_rtol "${OUTER_RTOL:-1.0e-10}" -outer_atol "${OUTER_ATOL:-1.0e-12}"
  -eps_tol "${EPS_TOL:-1.0e-10}" -eps_max_it "${EPS_MAX_IT:-4000}"
  -eps_ncv "${EPS_NCV:-40}" -eigmin_eps_stability_stop false
)
# Run serially so K, P, factors, and EPS workspaces from one method are fully
# released before the next preconditioner source starts.
"$launcher" -n "$ranks" "$driver" "${common[@]}" \
  -preconditioner_matrix_source lor_q1 \
  -output "$result_dir/lor_q1.log" "$@"
"$launcher" -n "$ranks" "$driver" "${common[@]}" \
  -preconditioner_matrix_source derivative_greville_mass \
  -output "$result_dir/derivative_greville_mass.log" "$@"
echo "[done] $result_dir"
