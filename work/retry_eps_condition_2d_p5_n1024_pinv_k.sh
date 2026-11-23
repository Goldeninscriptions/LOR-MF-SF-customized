#!/usr/bin/env bash
set -euo pipefail

# Targeted retry for the p=5, 1024x1024 P^{-1}K case.
#
# The 56-rank batch attempt failed inside UCX's mm shared-memory transport
# while PETSc GAMG was communicating a restriction operation. This retry:
#   1. uses 28 ranks to reduce endpoints and collective pressure;
#   2. forces UCX to tcp+self, bypassing the failing mm transport;
#   3. leaves EPS, ST, CG, GAMG and the Ritz-stability policy unchanged.
#
# The formal result path is the same one expected by the batch resume logic.
# The failed 56-rank console log is not overwritten.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
result_file=${OUTPUT_FILE:-$script_dir/results/eps_condition_batch_2d/p5/n1024x1024/pinv_k.log}
console_file=${CONSOLE_FILE:-$script_dir/logs/eps_condition_batch_2d/p5/n1024x1024/pinv_k.retry_r28_tcp.console.log}
retry_ranks=${MPI_RANKS:-28}
ucx_transports=${UCX_TLS:-tcp,self}

if ! [[ "$retry_ranks" =~ ^[1-9][0-9]*$ ]]; then
  echo "MPI_RANKS must be a positive integer" >&2
  exit 2
fi

mkdir -p "$(dirname "$result_file")" "$(dirname "$console_file")"

echo "[targeted retry] p=5 n=1024x1024 target=pinv_k ranks=$retry_ranks UCX_TLS=$ucx_transports"
echo "[targeted retry] result=$result_file"
echo "[targeted retry] console=$console_file"

MPI_RANKS=$retry_ranks \
UCX_TLS=$ucx_transports \
  "$script_dir/run_eps_condition_2d.sh" \
    5 1024 pinv_k "$result_file" \
    -eigmax_eps_monitor \
    -eigmin_eps_monitor \
    2>&1 | tee "$console_file"

if ! grep -qx 'eigensolver_status = converged' "$result_file"; then
  echo "Retry exited without a converged EPS result: $result_file" >&2
  exit 1
fi
if ! grep -qx 'spectral_condition_target = pinv_k' "$result_file"; then
  echo "Retry result has the wrong spectral target: $result_file" >&2
  exit 1
fi

echo "[targeted retry completed] $result_file"
