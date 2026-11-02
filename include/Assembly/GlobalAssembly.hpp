#ifndef LOR_ASSEMBLY_GLOBAL_ASSEMBLY_HPP
#define LOR_ASSEMBLY_GLOBAL_ASSEMBLY_HPP

#include "Assembly/LocalAssembly.hpp"
#include "Assembly/PreconditionerSide.hpp"
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
    // Extremal geometry-Jacobian determinants over the quadrature points
    // visited by this one independently requested global assembly.
    double min_det_jac = 0.0;
    double max_det_jac = 0.0;
};

// One independently assembled reduced PETSc matrix. The variable holding this
// object determines whether matrix is stiffness or mass; the caller releases
// it with destroy_global_matrix().
struct GlobalMatrixAssemblyResult
{
    GlobalMatrixAssemblyResult() = default;
    GlobalMatrixAssemblyResult(const GlobalMatrixAssemblyResult &) = delete;
    GlobalMatrixAssemblyResult &operator=(
        const GlobalMatrixAssemblyResult &) = delete;

    // Reduced symmetric matrix after constrained rows and columns are omitted.
    Mat matrix = nullptr;

    // Geometry diagnostics from this matrix's element traversal.
    AssemblyStatistics statistics;

    // Structural entries predicted from tensor support before allocation.
    PetscInt predicted_nonzeros = 0;

    // PETSc storage diagnostics queried after final assembly. Counts are
    // global sums; memory_bytes is PETSc's reported matrix storage estimate.
    PetscLogDouble allocated_nonzeros = 0.0;
    PetscLogDouble used_nonzeros = 0.0;
    PetscLogDouble allocation_calls = 0.0;
    PetscLogDouble memory_bytes = 0.0;

    // Portable lower-bound estimate for values, column indices, and the two
    // MPIAIJ row-pointer arrays; PETSc allocator/object overhead is excluded.
    PetscLogDouble estimated_minimum_bytes = 0.0;
};

// One independently assembled reduced PETSc vector. The caller owns vector
// and releases it with destroy_global_vector().
struct GlobalVectorAssemblyResult
{
    GlobalVectorAssemblyResult() = default;
    GlobalVectorAssemblyResult(const GlobalVectorAssemblyResult &) = delete;
    GlobalVectorAssemblyResult &operator=(
        const GlobalVectorAssemblyResult &) = delete;

    // Reduced load vector indexed by unconstrained ID values.
    Vec vector = nullptr;

    // Geometry diagnostics from this vector's element traversal.
    AssemblyStatistics statistics;
};

// Extreme singular values and ordinary 2-norm condition number of one
// globally assembled distributed operator.
struct ConditionNumberResult
{
    PetscReal sigma_min = 0.0;
    PetscReal sigma_max = 0.0;
    PetscReal condition_number = 0.0;
};

// Exact inverse used by the optional preconditioned spectral analysis. The
// LOR choice factors K_LOR; stiffness_jacobi applies diag(K)^{-1} exactly.
enum class ConditioningPreconditioner
{
    exact_lor,
    stiffness_jacobi
};

// Destroy one independently owned PETSc result and reset its handle to null.
PetscErrorCode destroy_global_matrix(GlobalMatrixAssemblyResult *result);
PetscErrorCode destroy_global_vector(GlobalVectorAssemblyResult *result);

// Independently analyze cond_2(K) and the selected exact preconditioned
// operator, using either K_LOR^{-1} or diag(K)^{-1}. This expensive SLEPc
// operation is never implied by assembly; orchestration calls it explicitly.
PetscErrorCode compute_stiffness_condition_numbers(Mat stiffness,
    Mat lor_stiffness, PreconditionerSide side,
    ConditioningPreconditioner preconditioner, PetscReal svd_tolerance,
    PetscInt max_iterations, PetscInt subspace_dimension,
    ConditionNumberResult *original,
    ConditionNumberResult *preconditioned);

// Assemble only the high-order B-spline stiffness matrix.
template <int Dim>
PetscErrorCode assemble_global_stiffness(
    const BSplinePatch<Dim> &patch, const BSplineIEN<Dim> &IEN,
    const ID &ID_map, const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables,
    GlobalMatrixAssemblyResult *result);

// Assemble only the high-order B-spline consistent mass matrix.
template <int Dim>
PetscErrorCode assemble_global_mass(
    const BSplinePatch<Dim> &patch, const BSplineIEN<Dim> &IEN,
    const ID &ID_map, const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables,
    GlobalMatrixAssemblyResult *result);

// Assemble only the high-order B-spline load vector.
template <int Dim>
PetscErrorCode assemble_global_load(
    const BSplinePatch<Dim> &patch, const BSplineIEN<Dim> &IEN,
    const ID &ID_map, const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables,
    const ScalarSource<Dim> &source,
    GlobalVectorAssemblyResult *result);

// Assemble only the LOR Quad4/Hex8 stiffness matrix.
template <int Dim>
PetscErrorCode assemble_global_stiffness(
    const LORMesh<Dim> &mesh, const LORIEN<Dim> &IEN,
    const ID &ID_map, const QuadratureRule1D &quad,
    GlobalMatrixAssemblyResult *result);

// Assemble only the LOR Quad4/Hex8 consistent mass matrix.
template <int Dim>
PetscErrorCode assemble_global_mass(
    const LORMesh<Dim> &mesh, const LORIEN<Dim> &IEN,
    const ID &ID_map, const QuadratureRule1D &quad,
    GlobalMatrixAssemblyResult *result);

// Assemble only the LOR Quad4/Hex8 load vector.
template <int Dim>
PetscErrorCode assemble_global_load(
    const LORMesh<Dim> &mesh, const LORIEN<Dim> &IEN,
    const ID &ID_map, const QuadratureRule1D &quad,
    const ScalarSource<Dim> &source,
    GlobalVectorAssemblyResult *result);

// Assemble high-order stiffness and load in one element/Gauss traversal.
template <int Dim>
PetscErrorCode assemble_global_stiffness_load(
    const BSplinePatch<Dim> &patch, const BSplineIEN<Dim> &IEN,
    const ID &ID_map, const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables,
    const ScalarSource<Dim> &source,
    GlobalMatrixAssemblyResult *stiffness,
    GlobalVectorAssemblyResult *load);

// Assemble the reduced high-order stiffness and source load for prescribed
// B-spline Dirichlet coefficients g. Free rows receive the exact elimination
// correction -K_fc*g_c; constrained entries of the full coefficient vector
// are never inserted into the reduced PETSc system.
template <int Dim>
PetscErrorCode assemble_global_stiffness_load_with_dirichlet(
    const BSplinePatch<Dim> &patch, const BSplineIEN<Dim> &IEN,
    const ID &ID_map, const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables,
    const ScalarSource<Dim> &source,
    const std::vector<double> &dirichlet_coefficients,
    GlobalMatrixAssemblyResult *stiffness,
    GlobalVectorAssemblyResult *load);

// Assemble high-order mass and load in one element/Gauss traversal.
template <int Dim>
PetscErrorCode assemble_global_mass_load(
    const BSplinePatch<Dim> &patch, const BSplineIEN<Dim> &IEN,
    const ID &ID_map, const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables,
    const ScalarSource<Dim> &source, GlobalMatrixAssemblyResult *mass,
    GlobalVectorAssemblyResult *load);

// Assemble every high-order term in one element/Gauss traversal.
template <int Dim>
PetscErrorCode assemble_global_stiffness_mass_load(
    const BSplinePatch<Dim> &patch, const BSplineIEN<Dim> &IEN,
    const ID &ID_map, const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables,
    const ScalarSource<Dim> &source,
    GlobalMatrixAssemblyResult *stiffness,
    GlobalMatrixAssemblyResult *mass, GlobalVectorAssemblyResult *load);

// Assemble LOR stiffness and load in one cell/Gauss traversal.
template <int Dim>
PetscErrorCode assemble_global_stiffness_load(
    const LORMesh<Dim> &mesh, const LORIEN<Dim> &IEN,
    const ID &ID_map, const QuadratureRule1D &quad,
    const ScalarSource<Dim> &source,
    GlobalMatrixAssemblyResult *stiffness,
    GlobalVectorAssemblyResult *load);

// Assemble LOR mass and load in one cell/Gauss traversal.
template <int Dim>
PetscErrorCode assemble_global_mass_load(
    const LORMesh<Dim> &mesh, const LORIEN<Dim> &IEN,
    const ID &ID_map, const QuadratureRule1D &quad,
    const ScalarSource<Dim> &source, GlobalMatrixAssemblyResult *mass,
    GlobalVectorAssemblyResult *load);

// Assemble every LOR term in one cell/Gauss traversal.
template <int Dim>
PetscErrorCode assemble_global_stiffness_mass_load(
    const LORMesh<Dim> &mesh, const LORIEN<Dim> &IEN,
    const ID &ID_map, const QuadratureRule1D &quad,
    const ScalarSource<Dim> &source,
    GlobalMatrixAssemblyResult *stiffness,
    GlobalMatrixAssemblyResult *mass, GlobalVectorAssemblyResult *load);

} // namespace lor

#endif
