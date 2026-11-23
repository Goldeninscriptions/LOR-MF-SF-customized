#!/usr/bin/env bash
set -euo pipefail

# Usage: run_greville_tensor_condition_2d.sh OP TARGET P N RESULT [PETSc options...]
if (( $# < 5 )); then
  echo "Usage: $0 mass|stiffness preconditioner|preconditioned P N RESULT_FILE [PETSc options...]" >&2
  exit 2
fi

operator_type=$1
spectral_target=$2
degree=$3
elements=$4
result_file=$5
shift 5
case "$operator_type" in mass|stiffness) ;; *) echo "operator must be mass or stiffness" >&2; exit 2;; esac
case "$spectral_target" in preconditioner|preconditioned) ;; *) echo "target must be preconditioner or preconditioned" >&2; exit 2;; esac
[[ "$degree" =~ ^[1-9][0-9]*$ ]] || { echo "P must be positive" >&2; exit 2; }
[[ "$elements" =~ ^[1-9][0-9]*$ ]] || { echo "N must be positive" >&2; exit 2; }

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
driver=${DRIVER:-$project_dir/build/greville_tensor_condition_driver}
mpi_launcher=${MPIEXEC:-mpirun}
[[ -x "$driver" ]] || { echo "Missing executable: $driver" >&2; exit 3; }
newer_input=$(find "$project_dir/include" "$project_dir/src" "$project_dir/examples" \
  -type f \( -name '*.hpp' -o -name '*.cpp' \) -newer "$driver" -print -quit)
if [[ -n "$newer_input" || "$project_dir/CMakeLists.txt" -nt "$driver" || \
      "$project_dir/examples/CMakeLists.txt" -nt "$driver" ]]; then
  echo "Stale executable: $driver; rebuild before running" >&2
  exit 3
fi

mpi_rank_setting=${MPI_RANKS:-default}
max_ranks=${MAX_MPI_RANKS:-56}
target_dofs_per_rank=${TARGET_DOFS_PER_RANK:-5000}
[[ "$max_ranks" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_MPI_RANKS must be positive" >&2; exit 2; }
[[ "$target_dofs_per_rank" =~ ^[1-9][0-9]*$ ]] || { echo "TARGET_DOFS_PER_RANK must be positive" >&2; exit 2; }
if [[ "$operator_type" == mass ]]; then
  direction_dofs=$((elements + degree))
else
  direction_dofs=$((elements + degree - 2))
fi
(( direction_dofs >= 2 )) || { echo "Case has fewer than two directional DOFs" >&2; exit 2; }
global_dofs=$((direction_dofs * direction_dofs))
if [[ "$mpi_rank_setting" == default && "$operator_type" == mass &&
      "$spectral_target" == preconditioner ]]; then
  mpi_ranks=1
elif [[ "$mpi_rank_setting" == default ]]; then
  needed=$(((global_dofs + target_dofs_per_rank - 1) / target_dofs_per_rank))
  mpi_ranks=1
  for candidate in 1 2 4 8 16 28 56; do
    if (( candidate <= max_ranks && candidate <= direction_dofs )); then
      mpi_ranks=$candidate
      (( candidate >= needed )) && break
    fi
  done
elif [[ "$mpi_rank_setting" =~ ^[1-9][0-9]*$ ]]; then
  mpi_ranks=$mpi_rank_setting
  (( mpi_ranks <= direction_dofs )) || {
    echo "MPI_RANKS=$mpi_ranks exceeds directional DOFs=$direction_dofs" >&2; exit 2; }
else
  echo "MPI_RANKS must be default or a positive integer" >&2
  exit 2
fi

eps_tol=${EPS_TOL:-1.0e-8}
eps_ncv=${EPS_NCV:-40}
eps_max_it=${EPS_MAX_IT:-10000}
stability_window=${EIGMIN_STABILITY_WINDOW:-100}
stability_tol=${EIGMIN_STABILITY_TOL:-5.0e-6}
stability_residual=${EIGMIN_STABILITY_RESIDUAL:-1.0e-2}
stability_min_it=${EIGMIN_STABILITY_MIN_IT:-300}

options=(
  -operator_type "$operator_type" -spectral_target "$spectral_target"
  -p "$degree" -n "$elements"
  -length_x "${LENGTH_X:-2.0}" -length_y "${LENGTH_Y:-1.0}"
  -eps_tol "$eps_tol" -eps_max_it "$eps_max_it" -eps_ncv "$eps_ncv"
  -output "$result_file"
)
if [[ "$operator_type" != mass || "$spectral_target" != preconditioner ]]; then
  options+=(
    -eigmax_st_type shift
    -eigmin_eps_stability_stop true
    -eigmin_eps_stability_window "$stability_window"
    -eigmin_eps_stability_tol "$stability_tol"
    -eigmin_eps_stability_residual "$stability_residual"
    -eigmin_eps_stability_min_it "$stability_min_it"
  )
  if [[ "$operator_type" == stiffness && "$spectral_target" == preconditioner ]]; then
    options+=(
      -eigmin_st_type sinvert -eigmin_eps_target 0
      -eigmin_eps_target_magnitude
    )
  else
    options+=( -eigmin_st_type shift -eigmin_eps_smallest_real )
  fi
fi
if [[ "$operator_type" == stiffness ]]; then
  options+=(
    -greville_stiffness_inner_ksp_type cg
    -greville_stiffness_inner_pc_type "${STIFFNESS_INNER_PC:-gamg}"
    -greville_stiffness_inner_ksp_rtol "${STIFFNESS_INNER_RTOL:-1.0e-12}"
    -greville_stiffness_inner_ksp_atol "${STIFFNESS_INNER_ATOL:-1.0e-14}"
    -greville_stiffness_inner_ksp_max_it "${STIFFNESS_INNER_MAX_IT:-5000}"
  )
fi

mkdir -p "$(dirname "$result_file")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
if [[ -z ${UCX_TLS:-} ]]; then
  export UCX_MM_FIFO_SIZE=${UCX_MM_FIFO_SIZE:-4096}
  export UCX_MM_FIFO_MAX_POLL=${UCX_MM_FIFO_MAX_POLL:-64}
  ucx_description="native,mm_fifo=$UCX_MM_FIFO_SIZE,mm_max_poll=$UCX_MM_FIFO_MAX_POLL"
else
  ucx_description="tls=$UCX_TLS"
fi
echo "[run] operator=$operator_type target=$spectral_target p=$degree n=${elements}x${elements} dofs=$global_dofs ranks=$mpi_ranks ucx=$ucx_description output=$result_file"
"$mpi_launcher" -n "$mpi_ranks" "$driver" "${options[@]}" "$@"
