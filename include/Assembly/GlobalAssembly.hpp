#ifndef LOR_ASSEMBLY_GLOBAL_ASSEMBLY_HPP
#define LOR_ASSEMBLY_GLOBAL_ASSEMBLY_HPP

#include "Assembly/LocalAssembly.hpp"
#include "Element/BSplineBasis1D.hpp"
#include "Element/GaussQuadrature.hpp"
#include "Mesh/BSplinePatch.hpp"
#include "Mesh/ID.hpp"
#include "Mesh/IEN.hpp"
#include "Mesh/LORMesh.hpp"

#include <petscmat.h>
#include <petscvec.h>

namespace lor
{

struct AssemblyStatistics
{
    // Extremal geometry-Jacobian determinants over all assembled elements.
    double min_det_jac = 0.0;
    double max_det_jac = 0.0;
};

// PETSc objects produced by one complete scalar global assembly. The caller
// owns these handles and releases them with destroy_global_system().
struct GlobalAssemblyResult
{
    GlobalAssemblyResult() = default;
    GlobalAssemblyResult(const GlobalAssemblyResult &) = delete;
    GlobalAssemblyResult &operator=(const GlobalAssemblyResult &) = delete;

    // Reduced stiffness matrix after homogeneous Dirichlet elimination.
    Mat stiffness = nullptr;

    // Reduced consistent mass matrix using the same ID numbering.
    Mat mass = nullptr;

    // Reduced load vector using ID as its PETSc index.
    Vec load = nullptr;

    // Geometry diagnostics accumulated during element traversal.
    AssemblyStatistics statistics;
};

// Destroy every PETSc object owned by result and reset its handles to null.
PetscErrorCode destroy_global_system(GlobalAssemblyResult *result);

// Assemble the high-order B-spline stiffness, mass, and load in one element
// traversal. Geometry/basis data remain owned by Patch and Element.
template <int Dim>
PetscErrorCode assemble_global_system(
    const BSplinePatch<Dim> &patch,
    const BSplineIEN<Dim> &IEN,
    const ID &ID_map,
    const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables,
    const ScalarSource<Dim> &source,
    GlobalAssemblyResult *result);

// Assemble the LOR Quad4/Hex8 stiffness, mass, and load with the same ID used
// by the coefficient-aligned high-order space.
template <int Dim>
PetscErrorCode assemble_global_system(
    const LORMesh<Dim> &mesh,
    const LORIEN<Dim> &IEN,
    const ID &ID_map,
    const QuadratureRule1D &quad,
    const ScalarSource<Dim> &source,
    GlobalAssemblyResult *result);

extern template PetscErrorCode assemble_global_system<2>(
    const BSplinePatch<2> &, const BSplineIEN<2> &, const ID &,
    const QuadratureRule1D &, const std::vector<BSplineBasis1D> &,
    const ScalarSource<2> &, GlobalAssemblyResult *);
extern template PetscErrorCode assemble_global_system<3>(
    const BSplinePatch<3> &, const BSplineIEN<3> &, const ID &,
    const QuadratureRule1D &, const std::vector<BSplineBasis1D> &,
    const ScalarSource<3> &, GlobalAssemblyResult *);
extern template PetscErrorCode assemble_global_system<2>(
    const LORMesh<2> &, const LORIEN<2> &, const ID &,
    const QuadratureRule1D &, const ScalarSource<2> &,
    GlobalAssemblyResult *);
extern template PetscErrorCode assemble_global_system<3>(
    const LORMesh<3> &, const LORIEN<3> &, const ID &,
    const QuadratureRule1D &, const ScalarSource<3> &,
    GlobalAssemblyResult *);

} // namespace lor

#endif
