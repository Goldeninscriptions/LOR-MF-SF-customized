#ifndef LOR_SOLVER_ERROR_ANALYSIS_HPP
#define LOR_SOLVER_ERROR_ANALYSIS_HPP

#include "Element/BSplineBasis1D.hpp"
#include "Element/GaussQuadrature.hpp"
#include "Mesh/BSplinePatch.hpp"
#include "Mesh/ID.hpp"
#include "Mesh/IEN.hpp"

#include <petscvec.h>

#include <array>
#include <functional>
#include <vector>

namespace lor
{

template <int Dim>
using ScalarExactSolution =
    std::function<double(const std::array<double, Dim> &)>;

// Integrate sqrt(int_Omega (u_h-u_exact)^2 dx) using the same Element data
// contract as Assembly. Dirichlet coefficients omitted by ID are reconstructed
// as zero before evaluating u_h at quadrature points.
template <int Dim>
PetscErrorCode compute_l2_error(const BSplinePatch<Dim> &patch,
    const BSplineIEN<Dim> &IEN, const ID &ID_map,
    const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables,
    Vec solution, const ScalarExactSolution<Dim> &exact,
    PetscReal *l2_error);

extern template PetscErrorCode compute_l2_error<2>(
    const BSplinePatch<2> &, const BSplineIEN<2> &, const ID &,
    const QuadratureRule1D &, const std::vector<BSplineBasis1D> &,
    Vec, const ScalarExactSolution<2> &, PetscReal *);
extern template PetscErrorCode compute_l2_error<3>(
    const BSplinePatch<3> &, const BSplineIEN<3> &, const ID &,
    const QuadratureRule1D &, const std::vector<BSplineBasis1D> &,
    Vec, const ScalarExactSolution<3> &, PetscReal *);

} // namespace lor

#endif
