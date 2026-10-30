#include "Assembly/GlobalAssembly.hpp"

#include "Element/BSplineElement.hpp"
#include "Element/FEMHex8.hpp"
#include "Element/FEMQ1Basis.hpp"
#include "Element/FEMQuad4.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <type_traits>
#include <vector>

namespace lor
{
namespace
{

PetscErrorCode create_global_system(const PetscInt matrix_size,
    const PetscInt nonzeros_per_row, GlobalAssemblyResult &result)
{
    PetscFunctionBeginUser;

    // Create stiffness and mass with identical connectivity-based sparsity.
    PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, matrix_size, matrix_size,
        nonzeros_per_row, nullptr, &result.stiffness));
    PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, matrix_size, matrix_size,
        nonzeros_per_row, nullptr, &result.mass));
    PetscCall(MatSetOption(result.stiffness, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatSetOption(result.mass, MAT_SYMMETRIC, PETSC_TRUE));

    // Create and explicitly zero the load vector before ADD_VALUES insertion.
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, matrix_size, &result.load));
    PetscCall(VecSet(result.load, 0.0));
    PetscCall(VecSetOption(
        result.load, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE));

    // Initialize geometry statistics for element-wise min/max updates.
    result.statistics.min_det_jac = std::numeric_limits<double>::max();
    result.statistics.max_det_jac = 0.0;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode add_local_system(Mat stiffness, Mat mass, Vec load,
    const IEN &IEN_map, const ID &ID_map, const std::size_t element,
    const LocalAssemblyResult &local)
{
    PetscFunctionBeginUser;
    const int num_basis = IEN_map.num_local_basis();

    // Compose IEN with ID once, matching the eID array used by the reference
    // Parallel-Computing and MixPERIGEE global assemblers.
    std::vector<PetscInt> ID_values(num_basis);
    for (int local_basis = 0; local_basis < num_basis; ++local_basis)
        ID_values[local_basis] =
            ID_map.get_ID(IEN_map.get_IEN(element, local_basis));

    // Scatter complete Kloc/Mloc/Floc blocks as in the reference assemblers.
    // PETSc ignores negative matrix indices; the vector was configured with
    // VEC_IGNORE_NEGATIVE_INDICES, so ID=-1 applies homogeneous Dirichlet
    // elimination without per-entry branching.
    PetscCall(MatSetValues(stiffness, num_basis, ID_values.data(),
        num_basis, ID_values.data(), local.stiffness.data(), ADD_VALUES));
    PetscCall(MatSetValues(mass, num_basis, ID_values.data(),
        num_basis, ID_values.data(), local.mass.data(), ADD_VALUES));
    PetscCall(VecSetValues(load, num_basis, ID_values.data(),
        local.load.data(), ADD_VALUES));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode finalize_global_system(GlobalAssemblyResult &result)
{
    PetscFunctionBeginUser;

    // Finalize both sparse matrices and the load vector after all elements.
    PetscCall(MatAssemblyBegin(result.stiffness, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(result.stiffness, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyBegin(result.mass, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(result.mass, MAT_FINAL_ASSEMBLY));
    PetscCall(VecAssemblyBegin(result.load));
    PetscCall(VecAssemblyEnd(result.load));
    PetscFunctionReturn(PETSC_SUCCESS);
}

void update_statistics(const LocalAssemblyResult &local,
    AssemblyStatistics &statistics)
{
    // Accumulate geometry diagnostics independently of the equation terms.
    statistics.min_det_jac =
        std::min(statistics.min_det_jac, local.min_det_jac);
    statistics.max_det_jac =
        std::max(statistics.max_det_jac, local.max_det_jac);
}

template <int Dim>
void gather_control_points(const BSplinePatch<Dim> &patch,
    const BSplineIEN<Dim> &IEN_map, const std::size_t element,
    std::vector<typename BSplinePatch<Dim>::Point> &points)
{
    // Refill caller-owned storage so the element loop performs no repeated
    // control-point-vector allocation.
    for (int local = 0; local < IEN_map.num_local_basis(); ++local)
        points[local] = patch.control_point(
            IEN_map.get_IEN(element, local));
}

template <int Dim>
std::array<std::array<double, Dim>, 1 << Dim> gather_lor_nodes(
    const LORMesh<Dim> &mesh, const LORIEN<Dim> &IEN_map,
    const std::size_t cell)
{
    // Compose IEN with the LOR physical-node array outside Element.
    std::array<std::array<double, Dim>, 1 << Dim> points{};
    for (int local = 0; local < IEN_map.num_local_basis(); ++local)
        points[local] = mesh.phy_node(IEN_map.get_IEN(cell, local));
    return points;
}

} // namespace

PetscErrorCode destroy_global_system(GlobalAssemblyResult *result)
{
    PetscFunctionBeginUser;
    if (result == nullptr)
        PetscFunctionReturn(PETSC_SUCCESS);

    // PETSc destroy functions safely accept null handles and reset them.
    PetscCall(MatDestroy(&result->stiffness));
    PetscCall(MatDestroy(&result->mass));
    PetscCall(VecDestroy(&result->load));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template <int Dim>
PetscErrorCode assemble_global_system(const BSplinePatch<Dim> &patch,
    const BSplineIEN<Dim> &IEN_map, const ID &ID_map,
    const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables,
    const ScalarSource<Dim> &source, GlobalAssemblyResult *result)
{
    PetscFunctionBeginUser;

    // Validate all global objects before allocating PETSc storage.
    PetscCheck(result != nullptr, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
        "The global assembly result pointer must not be null");
    PetscCheck(result->stiffness == nullptr && result->mass == nullptr &&
        result->load == nullptr, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
        "The global assembly result already owns PETSc objects");
    PetscCheck(static_cast<bool>(source), PETSC_COMM_SELF,
        PETSC_ERR_ARG_NULL, "The scalar source function must not be empty");
    PetscCheck(static_cast<int>(basis_tables.size()) == Dim, PETSC_COMM_SELF,
        PETSC_ERR_ARG_SIZ,
        "One B-spline basis table is required per direction");
    PetscCheck(ID_map.num_global_indices() ==
        static_cast<int>(patch.num_basis()), PETSC_COMM_SELF,
        PETSC_ERR_ARG_SIZ, "ID does not match the B-spline patch");
    PetscCheck(IEN_map.num_elements() == patch.num_elements(),
        PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
        "B-spline IEN does not match the patch");

    // Estimate tensor-product row connectivity and allocate the reduced
    // stiffness, mass, and load objects.
    const PetscInt matrix_size = ID_map.num_free_dofs();
    PetscCheck(matrix_size > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
        "All coefficients are constrained; refine the geometry to create free dofs");
    PetscInt nonzeros_per_row = 1;
    for (int direction = 0; direction < Dim; ++direction)
        nonzeros_per_row *= 2 * patch.degree(direction) + 1;
    nonzeros_per_row = std::min(matrix_size, nonzeros_per_row);
    PetscCall(create_global_system(
        matrix_size, nonzeros_per_row, *result));

    // Follow the legacy gather -> Element -> LocalAssembly -> scatter
    // structure while reusing the caller-provided quadrature/basis tables.
    std::vector<typename BSplinePatch<Dim>::Point> control_points(
        IEN_map.num_local_basis());
    LocalAssemblyResult local;
    for (std::size_t element_index = 0;
         element_index < patch.num_elements(); ++element_index)
    {
        gather_control_points(
            patch, IEN_map, element_index, control_points);
        const BSplineElement<Dim> element(patch, element_index,
            control_points, quad, basis_tables);
        assemble_local_system<Dim>(element, source, local);
        update_statistics(local, result->statistics);
        PetscCall(add_local_system(result->stiffness, result->mass,
            result->load, IEN_map, ID_map, element_index, local));
    }

    // Finalize all three global objects together.
    PetscCall(finalize_global_system(*result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template <int Dim>
PetscErrorCode assemble_global_system(const LORMesh<Dim> &mesh,
    const LORIEN<Dim> &IEN_map, const ID &ID_map,
    const QuadratureRule1D &quad, const ScalarSource<Dim> &source,
    GlobalAssemblyResult *result)
{
    PetscFunctionBeginUser;

    // Validate the coefficient-aligned LOR topology before PETSc allocation.
    PetscCheck(result != nullptr, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
        "The global assembly result pointer must not be null");
    PetscCheck(result->stiffness == nullptr && result->mass == nullptr &&
        result->load == nullptr, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
        "The global assembly result already owns PETSc objects");
    PetscCheck(static_cast<bool>(source), PETSC_COMM_SELF,
        PETSC_ERR_ARG_NULL, "The scalar source function must not be empty");
    PetscCheck(ID_map.num_global_indices() ==
        static_cast<int>(mesh.num_nodes()), PETSC_COMM_SELF,
        PETSC_ERR_ARG_SIZ, "ID does not match the LOR mesh");
    PetscCheck(IEN_map.num_elements() == mesh.num_cells(),
        PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
        "LOR IEN does not match the mesh");

    // Q1 connectivity contains at most 3^Dim entries per interior row.
    const PetscInt matrix_size = ID_map.num_free_dofs();
    PetscCheck(matrix_size > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
        "All coefficients are constrained; refine the geometry to create free dofs");
    PetscInt nonzeros_per_row = 1;
    for (int direction = 0; direction < Dim; ++direction)
        nonzeros_per_row *= 3;
    nonzeros_per_row = std::min(matrix_size, nonzeros_per_row);
    PetscCall(create_global_system(
        matrix_size, nonzeros_per_row, *result));

    // Build the Q1 reference table once, then follow the same
    // gather -> Element -> LocalAssembly -> scatter structure as IGA.
    const FEMQ1Basis<Dim> ref_basis(quad);
    using FEMElement = std::conditional_t<Dim == 2, FEMQuad4, FEMHex8>;
    LocalAssemblyResult local;
    for (std::size_t cell_index = 0;
         cell_index < mesh.num_cells(); ++cell_index)
    {
        const auto phy_nodes =
            gather_lor_nodes(mesh, IEN_map, cell_index);
        const FEMElement element(phy_nodes, ref_basis);
        assemble_local_system(element, source, local);
        update_statistics(local, result->statistics);
        PetscCall(add_local_system(result->stiffness, result->mass,
            result->load, IEN_map, ID_map, cell_index, local));
    }

    // Finalize all three LOR objects together.
    PetscCall(finalize_global_system(*result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template PetscErrorCode assemble_global_system<2>(
    const BSplinePatch<2> &, const BSplineIEN<2> &, const ID &,
    const QuadratureRule1D &, const std::vector<BSplineBasis1D> &,
    const ScalarSource<2> &, GlobalAssemblyResult *);
template PetscErrorCode assemble_global_system<3>(
    const BSplinePatch<3> &, const BSplineIEN<3> &, const ID &,
    const QuadratureRule1D &, const std::vector<BSplineBasis1D> &,
    const ScalarSource<3> &, GlobalAssemblyResult *);
template PetscErrorCode assemble_global_system<2>(
    const LORMesh<2> &, const LORIEN<2> &, const ID &,
    const QuadratureRule1D &, const ScalarSource<2> &,
    GlobalAssemblyResult *);
template PetscErrorCode assemble_global_system<3>(
    const LORMesh<3> &, const LORIEN<3> &, const ID &,
    const QuadratureRule1D &, const ScalarSource<3> &,
    GlobalAssemblyResult *);

} // namespace lor
