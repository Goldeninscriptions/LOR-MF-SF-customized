#!/usr/bin/env bash
set -uo pipefail

# Dedicated manual entry point for the expensive p=8, n=1024x1024 inner-CG
# rtol comparison. It reuses the resumable GAMG tuning engine and is a dry run
# unless the user explicitly sets BATCH_DRY_RUN=0.
#
# Preview:
#   BATCH_DRY_RUN=1 ./work/run_trunc_chol_gamg_rtol_p8_n1024_2d.sh
# Run/resume:
#   BATCH_DRY_RUN=0 ./work/run_trunc_chol_gamg_rtol_p8_n1024_2d.sh
#
# Defaults may be overridden, for example:
#   INNER_RTOLS="1e-8 1e-6 1e-5 1e-4" MPI_RANK=32 MAX_MPI_RANKS=32 \
#     BATCH_DRY_RUN=0 ./work/run_trunc_chol_gamg_rtol_p8_n1024_2d.sh

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export TUNING_STAGE=rtol
export TUNING_P=8
export TUNING_N=1024
export RUN_ID="${RUN_ID:-p8-n1024-gamg-rtol}"
export INNER_RTOLS="${INNER_RTOLS:-1e-10 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2}"

exec "$script_dir/run_trunc_chol_gamg_tuning_2d.sh" "$@"
