#!/usr/bin/env bash
set -euo pipefail

driver=$1
mpi_launcher=$2
project_dir=$3
test_dir=$(mktemp -d)
trap 'rm -rf "$test_dir"' EXIT

common=(
  -refinement k_refine -lor_points greville -lor_pc left_cg_lu
  -preconditioner_side left
  -preconditioner_matrix_source truncated_cholesky_tensor
  -compute_solution true -compute_l2_error true
)

run_case() {
  local ranks=$1 geometry=$2 degree=$3 elements=$4 bandwidth=$5 output=$6
  shift 6
  "$mpi_launcher" -n "$ranks" "$driver" \
    -geometry "$geometry" -p "$degree" -n "$elements" \
    -trunc_chol_k "$bandwidth" -output "$output" \
    "${common[@]}" "$@"
}

value() {
  awk -F ' = ' -v key="$2" '$1 == key {print $2}' "$1"
}

square="$project_dir/work/input/square_2d.txt"
square_scaled="$project_dir/work/input/square_2d_scaled_small.txt"
cube="$project_dir/work/input/cube_3d.txt"

# With this small p=4 case the protected boundary layers retain the complete
# factors, so independent dense algebra says P=K and both condition numbers
# must equal one.  This checks EPS and SVD definitions as well as convergence.
run_case 1 "$square" 4 4,4 auto "$test_dir/2d-r1.log" \
  -verify_tensor_identity true -compute_condition_number true \
  -compute_spectral_condition_number true
grep -q '^tensor_identity_verification_status = passed$' "$test_dir/2d-r1.log"
grep -q '^eigensolver_status = converged$' "$test_dir/2d-r1.log"
grep -q '^svd_status = converged$' "$test_dir/2d-r1.log"
awk -v x="$(value "$test_dir/2d-r1.log" kappa_lambda)" \
  'BEGIN {exit !(x > 0.99999999 && x < 1.00000001)}'
awk -v x="$(value "$test_dir/2d-r1.log" kappa_sigma)" \
  'BEGIN {exit !(x > 0.99999999 && x < 1.00000001)}'

# A genuinely truncated small matrix is independently reconstructed from
# PETSc's sparse ASCII views. NumPy's dense Hermitian eigensolver and dense
# SVD must agree with EPS/SVD, so the two condition-number definitions cannot
# accidentally be interchanged.
run_case 1 "$square" 4 16,16 2 "$test_dir/dense-reference.log" \
  -compute_condition_number true -compute_spectral_condition_number true \
  -dense_reference_K_view "ascii:$test_dir/K.txt:ascii_matlab" \
  -dense_reference_P_view "ascii:$test_dir/P.txt:ascii_matlab"
python3 - "$test_dir/K.txt" "$test_dir/P.txt" \
  "$(value "$test_dir/dense-reference.log" kappa_lambda)" \
  "$(value "$test_dir/dense-reference.log" kappa_sigma)" <<'PY'
import sys
import re
import numpy as np

def read_petsc(path):
    entries = []
    largest = -1
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            match = re.match(r"^(\d+)\s+(\d+)\s+(.+)$", line.strip())
            if match:
                row, column = int(match.group(1)) - 1, int(match.group(2)) - 1
                value = float(match.group(3))
                entries.append((row, column, value))
                largest = max(largest, row, column)
    matrix = np.zeros((largest + 1, largest + 1))
    for row, column, value in entries:
        matrix[row, column] = value
    return matrix

K = read_petsc(sys.argv[1])
P = read_petsc(sys.argv[2])
L = np.linalg.cholesky(P)
left = np.linalg.solve(L, K)
generalized = np.linalg.solve(L, left.T).T
eigenvalues = np.linalg.eigvalsh(generalized)
kappa_lambda = eigenvalues[-1] / eigenvalues[0]
singular = np.linalg.svd(np.linalg.solve(P, K), compute_uv=False)
kappa_sigma = singular[0] / singular[-1]
reported_lambda, reported_sigma = map(float, sys.argv[3:5])
if not np.isclose(kappa_lambda, reported_lambda, rtol=2e-7):
    raise SystemExit(f"dense/EPS mismatch: {kappa_lambda} vs {reported_lambda}")
if not np.isclose(kappa_sigma, reported_sigma, rtol=2e-7):
    raise SystemExit(f"dense/SVD mismatch: {kappa_sigma} vs {reported_sigma}")
PY

# Exercise actual 3D tensor assembly and distributed ownership.
run_case 2 "$cube" 3 3,3,3 1 "$test_dir/3d-r2.log" \
  -verify_tensor_identity true -compute_spectral_condition_number true
grep -q '^tensor_identity_verification_status = passed$' "$test_dir/3d-r2.log"
grep -q '^preconditioner_symmetric = true$' "$test_dir/3d-r2.log"
awk -v x="$(value "$test_dir/3d-r2.log" lambda_min)" \
  'BEGIN {exit !(x > 0)}'
awk -v x="$(value "$test_dir/3d-r2.log" 'relative_residual(left_cg_lu)')" \
  'BEGIN {exit !(x < 1e-9)}'
test -n "$(value "$test_dir/3d-r2.log" 'L2_error(left_cg_lu)')"

# One/two/four-rank structure and physical solution diagnostics must agree.
run_case 2 "$square" 4 4,4 2 "$test_dir/2d-r2.log" \
  -verify_tensor_identity true
run_case 4 "$square" 4 4,4 2 "$test_dir/2d-r4.log" \
  -verify_tensor_identity true
[[ "$(value "$test_dir/2d-r1.log" nnz_P)" == \
   "$(value "$test_dir/2d-r2.log" nnz_P)" ]]
[[ "$(value "$test_dir/2d-r1.log" nnz_P)" == \
   "$(value "$test_dir/2d-r4.log" nnz_P)" ]]
for parallel in "$test_dir/2d-r2.log" "$test_dir/2d-r4.log"; do
  awk -v a="$(value "$test_dir/2d-r1.log" 'L2_error(left_cg_lu)')" \
      -v b="$(value "$parallel" 'L2_error(left_cg_lu)')" \
      'BEGIN {d=a-b; if(d<0)d=-d; exit !(d < 1e-12)}'
  awk -v x="$(value "$parallel" 'relative_residual(left_cg_lu)')" \
      'BEGIN {exit !(x < 1e-9)}'
done

# Uniform physical rescaling may change values, not the sparse pattern.
run_case 1 "$square_scaled" 4 4,4 2 "$test_dir/2d-scaled.log" \
  -verify_tensor_identity true
grep -q '^tensor_identity_verification_status = passed$' "$test_dir/2d-scaled.log"
[[ "$(value "$test_dir/2d-r1.log" nnz_P)" == \
   "$(value "$test_dir/2d-scaled.log" nnz_P)" ]]

# The conservative cap must fail during preflight, before tensor-row maps and
# PETSc P allocation.  The diagnostic is part of the regression contract.
if run_case 1 "$square" 4 4,4 2 "$test_dir/low-memory.log" \
    -trunc_chol_max_memory_bytes 1 >"$test_dir/low-memory.console" 2>&1; then
  echo "Low-memory preflight unexpectedly succeeded" >&2
  exit 1
fi
grep -q 'no tensor rows or PETSc matrix were allocated' \
  "$test_dir/low-memory.console"

# Disabled optional work is explicitly marked and emits no fabricated beta or
# tensor-error value.
run_case 1 "$square" 4 4,4 2 "$test_dir/disabled.log" \
  -compute_truncation_certificate false -verify_tensor_identity false
grep -q '^trunc_chol_certificate_status = not_computed$' "$test_dir/disabled.log"
grep -q '^tensor_identity_verification_status = not_computed$' "$test_dir/disabled.log"
! grep -q '^trunc_chol_beta_direction_' "$test_dir/disabled.log"
! grep -q '^tensor_identity_relative_error =' "$test_dir/disabled.log"

# Nonconverged SLEPc runs remain nonconverged records, never successful
# condition estimates.
run_case 1 "$square" 8 8,8 4 "$test_dir/nonconverged.log" \
  -compute_condition_number true -compute_spectral_condition_number true \
  -svd_max_it 1 -eps_max_it 1
grep -q '^svd_status_K = not_converged$' "$test_dir/nonconverged.log"
! grep -q '^condition(K) =' "$test_dir/nonconverged.log"

# Functional-only regressions for all legacy matrix/point/geometry paths.
run_legacy() {
  local geometry=$1 points=$2 output=$3
  "$mpi_launcher" -n 1 "$driver" -geometry "$geometry" \
    -refinement k_refine -p 3 -n 4,4 -lor_points "$points" \
    -lor_pc left_cg_lu -preconditioner_side left \
    -preconditioner_matrix_source lor_q1 \
    -compute_solution true -compute_l2_error true -output "$output"
  awk -v x="$(value "$output" 'relative_residual(left_cg_lu)')" \
    'BEGIN {exit !(x < 1e-9)}'
  test -n "$(value "$output" 'L2_error(left_cg_lu)')"
}
run_legacy "$square" greville "$test_dir/legacy-greville.log"
run_legacy "$square" demko "$test_dir/legacy-demko.log"
run_legacy "$project_dir/work/input/curved_quadrilateral_2d.txt" greville \
  "$test_dir/legacy-curved.log"
