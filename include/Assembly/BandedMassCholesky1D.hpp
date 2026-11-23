#ifndef LOR_ASSEMBLY_BANDED_MASS_CHOLESKY_1D_HPP
#define LOR_ASSEMBLY_BANDED_MASS_CHOLESKY_1D_HPP

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace lor
{

// Compact lower-band storage for a symmetric matrix.  No N-by-N dense
// representation is used anywhere in the truncated-Cholesky path.
class SymmetricBandMatrix1D
{
public:
    SymmetricBandMatrix1D() = default;
    SymmetricBandMatrix1D(int size, int half_bandwidth);

    int size() const noexcept { return size_; }
    int half_bandwidth() const noexcept { return half_bandwidth_; }
    double value(int row, int column) const;
    double &lower(int row, int column);
    std::size_t stored_nonzeros(double tolerance = 0.0) const;

private:
    int size_ = 0;
    int half_bandwidth_ = 0;
    std::vector<double> entries_;
};

// Full row adjacency for a symmetric sparse one-dimensional operator.  Rows
// contain sorted (column,value) pairs and are directly consumable by the
// multidimensional tensor stencil.
struct SparseMatrix1D
{
    std::vector<std::vector<std::pair<int, double>>> rows;

    int size() const noexcept { return static_cast<int>(rows.size()); }
    double value(int row, int column) const;
    std::size_t nonzeros() const noexcept;
};

struct TruncationCertificate1D
{
    bool computed = false;
    double maximum_row_sum = 0.0;
    double maximum_column_sum = 0.0;
    double beta = 0.0;
    bool beta_below_one = false;
};

struct TruncatedMassResult1D
{
    int degree = 0;
    int size = 0;
    int original_half_bandwidth = 0;
    int truncation_half_bandwidth = 0;
    int complete_factor_half_bandwidth = 0;
    int truncated_factor_half_bandwidth = 0;
    bool boundary_protection = true;
    bool boundary_dofs_removed = false;
    SparseMatrix1D exact_mass;
    SparseMatrix1D approximate_mass;
    std::size_t complete_factor_nonzeros = 0;
    std::size_t truncated_factor_nonzeros = 0;
    std::size_t dropped_factor_entries = 0;
    double full_reconstruction_relative_error = 0.0;
    TruncationCertificate1D certificate;
    double mass_assembly_seconds = 0.0;
    double full_cholesky_seconds = 0.0;
    double truncation_seconds = 0.0;
    double certificate_seconds = 0.0;
};

struct TruncatedDirectionResult1D
{
    int original_degree = 0;
    int derivative_degree = 0;
    int num_original_basis = 0;
    TruncatedMassResult1D value_mass;
    TruncatedMassResult1D derivative_mass;
    SparseMatrix1D exact_stiffness;
    SparseMatrix1D approximate_stiffness;
    double derivative_identity_relative_error = 0.0;
    double reduced_operator_seconds = 0.0;
};

// Assemble the exact one-dimensional B-spline consistent mass matrix.  This
// is also the common reference operator used by alternative mass
// preconditioners, so they share the same basis evaluation and quadrature as
// the truncated-Cholesky path.
SparseMatrix1D assemble_bspline_mass_1d(int degree,
    const std::vector<double> &knots, bool remove_endpoints = false);

// Assemble the exact one-dimensional B-spline stiffness matrix after
// removing the two endpoint coefficients (homogeneous Dirichlet data).
SparseMatrix1D assemble_bspline_stiffness_1d(int degree,
    const std::vector<double> &knots);

// Assemble one p-degree value mass (with endpoint coefficients removed) and
// one complete (p-1)-degree derivative-space mass on the endpoint-trimmed
// knot vector, then factor, truncate, certify, and form D^T M D.
TruncatedDirectionResult1D build_truncated_direction_1d(
    int original_degree, const std::vector<double> &original_knots,
    int truncation_half_bandwidth, bool compute_certificate,
    bool boundary_protection = true);

// Small diagnostics used by the standalone verification executable.
double relative_frobenius_difference(
    const SparseMatrix1D &first, const SparseMatrix1D &second);
bool is_symmetric_positive_definite(const SparseMatrix1D &matrix,
    double symmetry_tolerance = 1.0e-12);

} // namespace lor

#endif
