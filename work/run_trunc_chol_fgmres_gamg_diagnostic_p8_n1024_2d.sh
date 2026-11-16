#!/usr/bin/env bash
set -uo pipefail

# Diagnostic p=8, n=1024x1024 sweep requested after the left-GMRES study.
# This is a separate RUN_ID and does not overwrite existing results.
# Defaults:
#   outer: right FGMRES, rtol=1e-8, atol=1e-10
#   inner: CG+GAMG, atol=1e-12
#   inner rtol scan: 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3
#   truncated Cholesky: k=4, boundary protection=true
#   MPI ranks: 107 (the preceding p=8,n=1024 run); user-overridable
# Safe default is BATCH_DRY_RUN=1.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export TUNING_STAGE=rtol
export TUNING_P=8
export TUNING_N=1024
export TUNING_LOR_PC=right_fgmres_cg
export RUN_ID="${RUN_ID:-p8-n1024-right-fgmres-cg-gamg-diagnostic}"
export MPI_RANK="${MPI_RANK:-107}"
export OUTER_RTOL="${OUTER_RTOL:-1.0e-8}"
export OUTER_ATOL="${OUTER_ATOL:-1.0e-10}"
export INNER_ATOL="${INNER_ATOL:-1.0e-12}"
export INNER_RTOLS="${INNER_RTOLS:-1e-8 1e-7 1e-6 1e-5 1e-4 1e-3}"
export TRUNC_CHOL_K="${TRUNC_CHOL_K:-4}"
export TRUNC_CHOL_BOUNDARY_PROTECTION="${TRUNC_CHOL_BOUNDARY_PROTECTION:-true}"

exec "$script_dir/run_trunc_chol_gamg_tuning_2d.sh" "$@"
