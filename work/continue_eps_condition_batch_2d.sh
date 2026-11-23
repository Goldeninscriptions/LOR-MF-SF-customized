#!/usr/bin/env bash
set -euo pipefail

# Resume the 2D EPS sweep after the verified p=5, n=1024 P^{-1}K retry.
# Completed result files are retained and folded back into summary.tsv. Only
# missing/unconverged targets are launched.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Always use per-case automatic planning with the original 56-rank cap, and
# leave UCX transport selection untouched by default. Failed cases are logged
# while the sweep continues, so the native 56-rank configuration is audited.
export MPI_RANKS=default
export MAX_MPI_RANKS=${CONTINUE_MAX_MPI_RANKS:-56}
export RESUME=true
export COOLDOWN_SECONDS=${COOLDOWN_SECONDS:-1}
export CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-true}

if [[ -n ${CONTINUE_UCX_TLS:-} ]]; then
  export UCX_TLS=$CONTINUE_UCX_TLS
  ucx_description=$UCX_TLS
else
  unset UCX_TLS
  ucx_description=native
fi

echo "[continue] resume=true max_mpi_ranks=$MAX_MPI_RANKS UCX_TLS=$ucx_description continue_on_error=$CONTINUE_ON_ERROR cooldown=${COOLDOWN_SECONDS}s"
exec "$script_dir/run_eps_condition_batch_2d.sh"
