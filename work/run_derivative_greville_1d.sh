#!/usr/bin/env bash
set -euo pipefail
if (( $# < 3 )); then
  echo "Usage: $0 P ELEMENTS OUTPUT.csv [PETSc/SLEPc options...]" >&2
  exit 2
fi
p=$1; elements=$2; output=$3; shift 3
[[ "$p" =~ ^([2-9]|[1-9][0-9]+)$ ]] || { echo "P must be >=2" >&2; exit 2; }
[[ "$elements" =~ ^[1-9][0-9]*$ ]] || { echo "ELEMENTS must be positive" >&2; exit 2; }
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
driver=${DRIVER:-$project_dir/build/derivative_greville_stiffness_1d}
[[ -x "$driver" ]] || { echo "Missing executable: $driver" >&2; exit 3; }
mkdir -p "$(dirname "$output")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
"$driver" -p "$p" -n "$elements" -output "$output" \
  -compute_spectrum "${COMPUTE_SPECTRUM:-true}" \
  -compute_solve "${COMPUTE_SOLVE:-true}" \
  -eps_tol "${EPS_TOL:-1.0e-10}" -eps_max_it "${EPS_MAX_IT:-2000}" \
  -eps_ncv "${EPS_NCV:-40}" -eigmin_eps_stability_stop false "$@"
