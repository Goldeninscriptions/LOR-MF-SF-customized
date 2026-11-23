#ifndef LOR_ASSEMBLY_GREVILLE_MASS_PRECONDITIONER_HPP
#define LOR_ASSEMBLY_GREVILLE_MASS_PRECONDITIONER_HPP

#include "Assembly/BandedMassCholesky1D.hpp"

#include <vector>

namespace lor
{

// Separable factors for the Greville transformed mass preconditioner
//
//   P = T^T M_g T,  T_ji = N_i^p(g_j).
//
// All coefficient degrees of freedom, including both endpoints, are retained.
// In several dimensions each field is combined direction by direction with a
// Kronecker product; no multidimensional dense transformation is needed.
struct GrevilleMassPreconditioner1D
{
    int degree = 0;
    int elements = 0;
    std::vector<double> knots;
    std::vector<double> greville_points;
    SparseMatrix1D spline_mass;
    SparseMatrix1D greville_mass;
    SparseMatrix1D collocation;
    SparseMatrix1D transformed_mass;
    // Homogeneous-Dirichlet stiffness counterparts.  The full builder leaves
    // these empty because the unrestricted stiffness has a constant null
    // space; restrict_greville_mass_preconditioner_endpoints fills them on
    // the common interior coefficient/node space.
    SparseMatrix1D spline_stiffness;
    SparseMatrix1D greville_stiffness;
    SparseMatrix1D transformed_stiffness;
};

// Construct an open uniform knot vector on [0,1] with maximum interior
// continuity and the requested number of positive knot spans.
std::vector<double> open_uniform_knots_1d(int degree, int elements);

// Build M_s, M_g, T, and T^T M_g T from one arbitrary open knot vector.
GrevilleMassPreconditioner1D build_greville_mass_preconditioner_1d(
    int degree, const std::vector<double> &knots);

// Remove the two endpoint coefficient/node rows and columns.  For homogeneous
// full-boundary Dirichlet problems the endpoint rows of the collocation
// columns retained in the free space vanish, so the returned factors satisfy
// P_II = T_II^T M_g,II T_II exactly.
GrevilleMassPreconditioner1D restrict_greville_mass_preconditioner_endpoints(
    const GrevilleMassPreconditioner1D &full);

// Return slow_factor (x) fast_factor using direction-0-fastest ordering.  The
// historical SparseMatrix1D row container is reused here as a general sparse
// row matrix; only its storage operations, not a one-dimensional assumption,
// participate in this function.
SparseMatrix1D sparse_kronecker_product(
    const SparseMatrix1D &slow_factor,
    const SparseMatrix1D &fast_factor);

} // namespace lor

#endif
