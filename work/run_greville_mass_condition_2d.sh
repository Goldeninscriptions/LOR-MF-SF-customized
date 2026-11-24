#!/usr/bin/env bash
set -euo pipefail

if (( $# < 4 )); then
  echo "Usage: $0 preconditioner|preconditioned P N RESULT_FILE [PETSc options...]" >&2
  exit 2
fi
target=$1; degree=$2; elements=$3; result_file=$4; shift 4
case "$target" in preconditioner|preconditioned) ;; *) exit 2;; esac
[[ "$degree" =~ ^[1-9][0-9]*$ && "$elements" =~ ^[1-9][0-9]*$ ]] || exit 2
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
driver=${DRIVER:-$project_dir/build/greville_mass_condition_driver}
mpi_launcher=${MPIEXEC:-mpirun}
[[ -x "$driver" ]] || { echo "Missing executable: $driver" >&2; exit 3; }
direction_dofs=$((elements + degree)); global_dofs=$((direction_dofs * direction_dofs))
if [[ ${MPI_RANKS:-default} == default && "$target" == preconditioner ]]; then
  ranks=1
elif [[ ${MPI_RANKS:-default} == default ]]; then
  needed=$(((global_dofs + 4999) / 5000)); ranks=1
  for candidate in 1 2 4 8 16 28 56; do
    (( candidate <= ${MAX_MPI_RANKS:-56} && candidate <= direction_dofs )) || continue
    ranks=$candidate; (( candidate >= needed )) && break
  done
else
  ranks=$MPI_RANKS
fi
mkdir -p "$(dirname "$result_file")"
options=(
  -spectral_target "$target" -p "$degree" -n "$elements"
  -eps_tol "${EPS_TOL:-1.0e-8}" -eps_max_it "${EPS_MAX_IT:-10000}"
  -eps_ncv "${EPS_NCV:-40}" -output "$result_file"
)
if [[ "$target" == preconditioned ]]; then
  options+=(
    -eigmin_eps_stability_stop true
    -eigmin_eps_stability_window "${EIGMIN_STABILITY_WINDOW:-100}"
    -eigmin_eps_stability_tol "${EIGMIN_STABILITY_TOL:-5.0e-6}"
    -eigmin_eps_stability_residual "${EIGMIN_STABILITY_RESIDUAL:-1.0e-2}"
    -eigmin_eps_stability_min_it "${EIGMIN_STABILITY_MIN_IT:-300}"
  )
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
"$mpi_launcher" -n "$ranks" "$driver" "${options[@]}" "$@"
