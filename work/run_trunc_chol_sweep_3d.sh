#!/usr/bin/env bash
set -uo pipefail

# 3D entry point for the shared sequential sweep engine. Safe default is a
# dry run. See run_trunc_chol_sweep_2d.sh for all environment variables.
#
# Manual launch sequence:
#   RUN_ID=trial3d BATCH_DRY_RUN=1 ./work/run_trunc_chol_sweep_3d.sh
#   RUN_ID=trial3d BATCH_DRY_RUN=0 ./work/run_trunc_chol_sweep_3d.sh

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TRUNC_CHOL_SWEEP_DIMENSION=3
exec "$script_dir/run_trunc_chol_sweep_2d.sh" "$@"
