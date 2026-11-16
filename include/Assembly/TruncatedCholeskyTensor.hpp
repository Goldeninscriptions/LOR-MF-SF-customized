#ifndef LOR_ASSEMBLY_TRUNCATED_CHOLESKY_TENSOR_HPP
#define LOR_ASSEMBLY_TRUNCATED_CHOLESKY_TENSOR_HPP

#include "Assembly/BandedMassCholesky1D.hpp"
#include "Assembly/GlobalAssembly.hpp"

#include <array>

namespace lor
{

template <int Dim>
struct TruncatedCholeskyTensorResult
{
    GlobalMatrixAssemblyResult preconditioner;
    std::array<TruncatedDirectionResult1D, Dim> directions;
    std::array<double, Dim> physical_lengths{};
    std::array<double, Dim> gamma{};
    double tensor_identity_relative_error = 0.0;
    bool symmetric = false;
    double minimum_diagonal = 0.0;
    double theoretical_kappa_lambda_bound = 0.0;
    bool certificate_bound_available = false;
    double time_1d_mass_assembly_seconds = 0.0;
    double time_1d_full_cholesky_seconds = 0.0;
    double time_1d_factor_truncation_seconds = 0.0;
    double time_truncation_certificate_seconds = 0.0;
    double time_1d_reduced_operators_seconds = 0.0;
    double time_tensor_assembly_seconds = 0.0;
};

// Validate an axis-aligned affine geometry, build both one-dimensional mass
// families in every direction, verify the untruncated tensor identity against
// the assembled K, and assemble P_k directly into owned PETSc AIJ rows.
template <int Dim>
PetscErrorCode assemble_truncated_cholesky_tensor(
    const BSplinePatch<Dim> &patch, const ID &ID_map,
    Mat high_order_stiffness, int truncation_half_bandwidth,
    bool compute_certificate,
    TruncatedCholeskyTensorResult<Dim> *result);

extern template PetscErrorCode assemble_truncated_cholesky_tensor<2>(
    const BSplinePatch<2> &, const ID &, Mat, int, bool,
    TruncatedCholeskyTensorResult<2> *);
extern template PetscErrorCode assemble_truncated_cholesky_tensor<3>(
    const BSplinePatch<3> &, const ID &, Mat, int, bool,
    TruncatedCholeskyTensorResult<3> *);

} // namespace lor

#endif
