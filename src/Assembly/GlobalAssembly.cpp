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

enum class MatrixQuantity
{
    stiffness,
    mass
};

// Initialize determinant extrema for an independent matrix or vector
// traversal before its first element contribution.
void initialize_statistics(AssemblyStatistics &statistics)
{
    statistics.min_det_jac = std::numeric_limits<double>::max();
    statistics.max_det_jac = 0.0;
}

// Create one reduced symmetric PETSc matrix with connectivity-based uniform
// row preallocation.
PetscErrorCode create_matrix(const MPI_Comm comm, const PetscInt local_size,
    const PetscInt global_size, const PetscInt nonzeros_per_row,
    GlobalMatrixAssemblyResult &result)
{
    PetscFunctionBeginUser;
    const PetscInt diagonal_nonzeros =
        std::min(nonzeros_per_row, local_size);
    const PetscInt off_diagonal_nonzeros = std::min(nonzeros_per_row,
        std::max<PetscInt>(0, global_size - local_size));
    PetscCall(MatCreateAIJ(comm, local_size, local_size,
        global_size, global_size, diagonal_nonzeros, nullptr,
        off_diagonal_nonzeros, nullptr, &result.matrix));
    PetscCall(MatSetOption(result.matrix, MAT_SYMMETRIC, PETSC_TRUE));
    initialize_statistics(result.statistics);
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Create one reduced load vector and configure ID=-1 entries to be ignored
// during homogeneous Dirichlet scatter.
PetscErrorCode create_vector(const MPI_Comm comm, const PetscInt local_size,
    const PetscInt global_size, GlobalVectorAssemblyResult &result)
{
    PetscFunctionBeginUser;
    PetscCall(VecCreateMPI(comm, local_size, global_size, &result.vector));
    PetscCall(VecSet(result.vector, 0.0));
    PetscCall(VecSetOption(
        result.vector, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE));
    initialize_statistics(result.statistics);
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Verify that PETSc's contiguous ownership matches the slab-derived ID range.
// This catches a mismatched local-size ordering before any shared-node entry is
// inserted into the distributed matrix or vector.
PetscErrorCode validate_matrix_ownership(Mat matrix, const IEN &IEN_map)
{
    PetscFunctionBeginUser;
    if (IEN_map.owned_ID_begin() >= 0)
    {
        PetscInt begin = 0;
        PetscInt end = 0;
        PetscCall(MatGetOwnershipRange(matrix, &begin, &end));
        PetscCheck(begin == IEN_map.owned_ID_begin() &&
            end == IEN_map.owned_ID_end(), IEN_map.communicator(),
            PETSC_ERR_ARG_INCOMP,
            "PETSc matrix rows do not match MPI slab ID ownership");
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode validate_vector_ownership(Vec vector, const IEN &IEN_map)
{
    PetscFunctionBeginUser;
    if (IEN_map.owned_ID_begin() >= 0)
    {
        PetscInt begin = 0;
        PetscInt end = 0;
        PetscCall(VecGetOwnershipRange(vector, &begin, &end));
        PetscCheck(begin == IEN_map.owned_ID_begin() &&
            end == IEN_map.owned_ID_end(), IEN_map.communicator(),
            PETSC_ERR_ARG_INCOMP,
            "PETSc vector rows do not match MPI slab ID ownership");
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Compose IEN with ID for one element/cell. The caller reuses ID_values across
// the complete traversal to avoid one allocation per local contribution.
void build_element_IDs(const IEN &IEN_map, const ID &ID_map,
    const std::size_t element, std::vector<PetscInt> &ID_values)
{
    for (int local = 0; local < IEN_map.num_local_basis(); ++local)
        ID_values[local] =
            ID_map.get_ID(IEN_map.get_IEN(element, local));
}

// Scatter a local matrix using an ID row already built for the current
// element. Fused assembly calls this for multiple outputs without repeating
// the IEN-to-ID composition.
PetscErrorCode add_local_matrix_with_IDs(Mat matrix,
    const LocalMatrixAssemblyResult &local,
    const std::vector<PetscInt> &ID_values)
{
    PetscFunctionBeginUser;
    const PetscInt num_basis = static_cast<PetscInt>(ID_values.size());
    PetscCall(MatSetValues(matrix, num_basis, ID_values.data(),
        num_basis, ID_values.data(), local.entries.data(), ADD_VALUES));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Scatter a local vector using the same prebuilt ID row as fused matrices.
PetscErrorCode add_local_vector_with_IDs(Vec vector,
    const LocalVectorAssemblyResult &local,
    const std::vector<PetscInt> &ID_values)
{
    PetscFunctionBeginUser;
    PetscCall(VecSetValues(vector, static_cast<PetscInt>(ID_values.size()),
        ID_values.data(), local.entries.data(), ADD_VALUES));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Scatter one complete row-major local matrix into its independently selected
// global stiffness or mass matrix.
PetscErrorCode add_local_matrix(Mat matrix, const IEN &IEN_map,
    const ID &ID_map, const std::size_t element,
    const LocalMatrixAssemblyResult &local,
    std::vector<PetscInt> &ID_values)
{
    PetscFunctionBeginUser;
    build_element_IDs(IEN_map, ID_map, element, ID_values);
    PetscCall(add_local_matrix_with_IDs(matrix, local, ID_values));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Scatter one local source vector into the independently selected global load
// vector; configured negative indices apply zero Dirichlet elimination.
PetscErrorCode add_local_vector(Vec vector, const IEN &IEN_map,
    const ID &ID_map, const std::size_t element,
    const LocalVectorAssemblyResult &local,
    std::vector<PetscInt> &ID_values)
{
    PetscFunctionBeginUser;
    build_element_IDs(IEN_map, ID_map, element, ID_values);
    PetscCall(add_local_vector_with_IDs(vector, local, ID_values));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Complete PETSc communication/state transitions for one matrix or vector.
PetscErrorCode finalize_matrix(Mat matrix)
{
    PetscFunctionBeginUser;
    PetscCall(MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode finalize_vector(Vec vector)
{
    PetscFunctionBeginUser;
    PetscCall(VecAssemblyBegin(vector));
    PetscCall(VecAssemblyEnd(vector));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Accumulate geometry diagnostics from either local result type.
template <class LocalResult>
void update_statistics(const LocalResult &local,
    AssemblyStatistics &statistics)
{
    statistics.min_det_jac =
        std::min(statistics.min_det_jac, local.min_det_jac);
    statistics.max_det_jac =
        std::max(statistics.max_det_jac, local.max_det_jac);
}

// Combine element-local geometry extrema over every MPI slab so all ranks
// expose identical diagnostics independent of the chosen process count.
PetscErrorCode reduce_statistics(
    const MPI_Comm comm, AssemblyStatistics &statistics)
{
    PetscFunctionBeginUser;
    double global_min = 0.0;
    double global_max = 0.0;
    PetscCallMPI(MPI_Allreduce(&statistics.min_det_jac, &global_min,
        1, MPI_DOUBLE, MPI_MIN, comm));
    PetscCallMPI(MPI_Allreduce(&statistics.max_det_jac, &global_max,
        1, MPI_DOUBLE, MPI_MAX, comm));
    statistics.min_det_jac = global_min;
    statistics.max_det_jac = global_max;
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Gather the high-order control points associated with one BSplineIEN row.
template <int Dim>
void gather_control_points(const BSplinePatch<Dim> &patch,
    const BSplineIEN<Dim> &IEN_map, const std::size_t element,
    std::vector<typename BSplinePatch<Dim>::Point> &points)
{
    for (int local = 0; local < IEN_map.num_local_basis(); ++local)
        points[local] = patch.control_point(
            IEN_map.get_IEN(element, local));
}

// Gather the conventional Quad4/Hex8 physical nodes associated with one
// LORIEN row.
template <int Dim>
std::array<std::array<double, Dim>, 1 << Dim> gather_lor_nodes(
    const LORMesh<Dim> &mesh, const LORIEN<Dim> &IEN_map,
    const std::size_t cell)
{
    std::array<std::array<double, Dim>, 1 << Dim> points{};
    for (int local = 0; local < IEN_map.num_local_basis(); ++local)
        points[local] = mesh.phy_node(IEN_map.get_IEN(cell, local));
    return points;
}

// Validate topology and basis-table objects shared by all three high-order
// assembly entry points.
template <int Dim>
PetscErrorCode validate_high_order_inputs(
    const BSplinePatch<Dim> &patch, const BSplineIEN<Dim> &IEN_map,
    const ID &ID_map,
    const std::vector<BSplineBasis1D> &basis_tables)
{
    PetscFunctionBeginUser;
    PetscCheck(static_cast<int>(basis_tables.size()) == Dim, PETSC_COMM_SELF,
        PETSC_ERR_ARG_SIZ,
        "One B-spline basis table is required per direction");
    PetscCheck(ID_map.num_global_indices() ==
        static_cast<int>(patch.num_basis()), PETSC_COMM_SELF,
        PETSC_ERR_ARG_SIZ, "ID does not match the B-spline patch");
    PetscCheck(IEN_map.num_elements() == patch.num_elements(),
        PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
        "B-spline IEN does not match the patch");
    PetscCheck(ID_map.num_free_dofs() > 0, PETSC_COMM_SELF,
        PETSC_ERR_ARG_SIZ,
        "All coefficients are constrained; refine the geometry to create free dofs");
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Validate topology objects shared by all three LOR assembly entry points.
template <int Dim>
PetscErrorCode validate_lor_inputs(const LORMesh<Dim> &mesh,
    const LORIEN<Dim> &IEN_map, const ID &ID_map)
{
    PetscFunctionBeginUser;
    PetscCheck(ID_map.num_global_indices() ==
        static_cast<int>(mesh.num_nodes()), PETSC_COMM_SELF,
        PETSC_ERR_ARG_SIZ, "ID does not match the LOR mesh");
    PetscCheck(IEN_map.num_elements() == mesh.num_cells(),
        PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
        "LOR IEN does not match the mesh");
    PetscCheck(ID_map.num_free_dofs() > 0, PETSC_COMM_SELF,
        PETSC_ERR_ARG_SIZ,
        "All coefficients are constrained; refine the geometry to create free dofs");
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Estimate the maximum high-order row connectivity used by both stiffness and
// mass preallocation.
template <int Dim>
PetscInt high_order_nonzeros_per_row(const BSplinePatch<Dim> &patch,
    const PetscInt matrix_size)
{
    PetscInt count = 1;
    for (int direction = 0; direction < Dim; ++direction)
        count *= 2 * patch.degree(direction) + 1;
    return std::min(matrix_size, count);
}

// Assemble exactly one selected high-order matrix. The independent traversal
// never allocates or computes the other matrix or the load vector.
template <int Dim>
PetscErrorCode assemble_high_order_matrix(
    const MatrixQuantity quantity, const BSplinePatch<Dim> &patch,
    const BSplineIEN<Dim> &IEN_map, const ID &ID_map,
    const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables,
    GlobalMatrixAssemblyResult *result)
{
    PetscFunctionBeginUser;

    // Validate result ownership and the common high-order topology.
    PetscCheck(result != nullptr, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
        "The global matrix result pointer must not be null");
    PetscCheck(result->matrix == nullptr, PETSC_COMM_SELF,
        PETSC_ERR_ARG_WRONGSTATE,
        "The global matrix result already owns a PETSc matrix");
    PetscCall(validate_high_order_inputs(
        patch, IEN_map, ID_map, basis_tables));

    // Allocate only the requested sparse matrix.
    const PetscInt matrix_size = ID_map.num_free_dofs();
    const PetscInt local_size = IEN_map.num_owned_dofs() < 0 ?
        matrix_size : IEN_map.num_owned_dofs();
    PetscCall(create_matrix(IEN_map.communicator(), local_size, matrix_size,
        high_order_nonzeros_per_row(patch, matrix_size), *result));
    PetscCall(validate_matrix_ownership(result->matrix, IEN_map));

    // Follow gather -> Element -> selected LocalAssembly -> matrix scatter.
    std::vector<typename BSplinePatch<Dim>::Point> control_points(
        IEN_map.num_local_basis());
    std::vector<PetscInt> ID_values(IEN_map.num_local_basis());
    LocalMatrixAssemblyResult local;
    for (std::size_t element_index = IEN_map.local_element_begin();
         element_index < IEN_map.local_element_end(); ++element_index)
    {
        gather_control_points(
            patch, IEN_map, element_index, control_points);
        const BSplineElement<Dim> element(patch, element_index,
            control_points, quad, basis_tables);
        if (quantity == MatrixQuantity::stiffness)
            assemble_local_stiffness<Dim>(element, local);
        else
            assemble_local_mass<Dim>(element, local);
        update_statistics(local, result->statistics);
        PetscCall(add_local_matrix(result->matrix, IEN_map, ID_map,
            element_index, local, ID_values));
    }

    // Finalize this matrix and reduce geometry diagnostics over all slabs.
    PetscCall(finalize_matrix(result->matrix));
    PetscCall(reduce_statistics(IEN_map.communicator(), result->statistics));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Assemble only the high-order source vector in its own element traversal.
template <int Dim>
PetscErrorCode assemble_high_order_load(
    const BSplinePatch<Dim> &patch, const BSplineIEN<Dim> &IEN_map,
    const ID &ID_map, const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables,
    const ScalarSource<Dim> &source, GlobalVectorAssemblyResult *result)
{
    PetscFunctionBeginUser;

    // Validate vector ownership, source availability, and common topology.
    PetscCheck(result != nullptr, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
        "The global vector result pointer must not be null");
    PetscCheck(result->vector == nullptr, PETSC_COMM_SELF,
        PETSC_ERR_ARG_WRONGSTATE,
        "The global vector result already owns a PETSc vector");
    PetscCheck(static_cast<bool>(source), PETSC_COMM_SELF,
        PETSC_ERR_ARG_NULL, "The scalar source function must not be empty");
    PetscCall(validate_high_order_inputs(
        patch, IEN_map, ID_map, basis_tables));

    // Allocate only the requested reduced load vector.
    const PetscInt global_size = ID_map.num_free_dofs();
    const PetscInt local_size = IEN_map.num_owned_dofs() < 0 ?
        global_size : IEN_map.num_owned_dofs();
    PetscCall(create_vector(IEN_map.communicator(), local_size,
        global_size, *result));
    PetscCall(validate_vector_ownership(result->vector, IEN_map));

    // Follow gather -> Element -> load-only LocalAssembly -> vector scatter.
    std::vector<typename BSplinePatch<Dim>::Point> control_points(
        IEN_map.num_local_basis());
    std::vector<PetscInt> ID_values(IEN_map.num_local_basis());
    LocalVectorAssemblyResult local;
    for (std::size_t element_index = IEN_map.local_element_begin();
         element_index < IEN_map.local_element_end(); ++element_index)
    {
        gather_control_points(
            patch, IEN_map, element_index, control_points);
        const BSplineElement<Dim> element(patch, element_index,
            control_points, quad, basis_tables);
        assemble_local_load<Dim>(element, source, local);
        update_statistics(local, result->statistics);
        PetscCall(add_local_vector(result->vector, IEN_map, ID_map,
            element_index, local, ID_values));
    }

    // Finalize the load independently of both matrices.
    PetscCall(finalize_vector(result->vector));
    PetscCall(reduce_statistics(IEN_map.communicator(), result->statistics));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Assemble exactly one selected LOR matrix with a shared reference Q1 table.
template <int Dim>
PetscErrorCode assemble_lor_matrix(const MatrixQuantity quantity,
    const LORMesh<Dim> &mesh, const LORIEN<Dim> &IEN_map,
    const ID &ID_map, const QuadratureRule1D &quad,
    GlobalMatrixAssemblyResult *result)
{
    PetscFunctionBeginUser;

    // Validate result ownership and coefficient-aligned LOR topology.
    PetscCheck(result != nullptr, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
        "The global matrix result pointer must not be null");
    PetscCheck(result->matrix == nullptr, PETSC_COMM_SELF,
        PETSC_ERR_ARG_WRONGSTATE,
        "The global matrix result already owns a PETSc matrix");
    PetscCall(validate_lor_inputs(mesh, IEN_map, ID_map));

    // Allocate only the selected Q1 matrix with at most 3^Dim neighbors per
    // interior tensor-grid row.
    const PetscInt matrix_size = ID_map.num_free_dofs();
    const PetscInt local_size = IEN_map.num_owned_dofs() < 0 ?
        matrix_size : IEN_map.num_owned_dofs();
    PetscInt nonzeros_per_row = 1;
    for (int direction = 0; direction < Dim; ++direction)
        nonzeros_per_row *= 3;
    PetscCall(create_matrix(IEN_map.communicator(), local_size, matrix_size,
        std::min(matrix_size, nonzeros_per_row), *result));
    PetscCall(validate_matrix_ownership(result->matrix, IEN_map));

    // Reuse one reference Q1 table for gather -> Element -> selected local
    // matrix integration -> scatter over all LOR cells.
    const FEMQ1Basis<Dim> ref_basis(quad);
    using FEMElement = std::conditional_t<Dim == 2, FEMQuad4, FEMHex8>;
    std::vector<PetscInt> ID_values(IEN_map.num_local_basis());
    LocalMatrixAssemblyResult local;
    for (std::size_t cell_index = IEN_map.local_element_begin();
         cell_index < IEN_map.local_element_end(); ++cell_index)
    {
        const auto phy_nodes =
            gather_lor_nodes(mesh, IEN_map, cell_index);
        const FEMElement element(phy_nodes, ref_basis);
        if (quantity == MatrixQuantity::stiffness)
            assemble_local_stiffness(element, local);
        else
            assemble_local_mass(element, local);
        update_statistics(local, result->statistics);
        PetscCall(add_local_matrix(result->matrix, IEN_map, ID_map,
            cell_index, local, ID_values));
    }

    // Finalize this LOR matrix without creating a load vector.
    PetscCall(finalize_matrix(result->matrix));
    PetscCall(reduce_statistics(IEN_map.communicator(), result->statistics));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Assemble only the LOR load vector in an independent Q1 traversal.
template <int Dim>
PetscErrorCode assemble_lor_load(const LORMesh<Dim> &mesh,
    const LORIEN<Dim> &IEN_map, const ID &ID_map,
    const QuadratureRule1D &quad, const ScalarSource<Dim> &source,
    GlobalVectorAssemblyResult *result)
{
    PetscFunctionBeginUser;

    // Validate vector ownership, source availability, and LOR topology.
    PetscCheck(result != nullptr, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
        "The global vector result pointer must not be null");
    PetscCheck(result->vector == nullptr, PETSC_COMM_SELF,
        PETSC_ERR_ARG_WRONGSTATE,
        "The global vector result already owns a PETSc vector");
    PetscCheck(static_cast<bool>(source), PETSC_COMM_SELF,
        PETSC_ERR_ARG_NULL, "The scalar source function must not be empty");
    PetscCall(validate_lor_inputs(mesh, IEN_map, ID_map));

    // Allocate only the requested reduced LOR load vector.
    const PetscInt global_size = ID_map.num_free_dofs();
    const PetscInt local_size = IEN_map.num_owned_dofs() < 0 ?
        global_size : IEN_map.num_owned_dofs();
    PetscCall(create_vector(IEN_map.communicator(), local_size,
        global_size, *result));
    PetscCall(validate_vector_ownership(result->vector, IEN_map));

    // Reuse one Q1 reference table for load-only integration over all cells.
    const FEMQ1Basis<Dim> ref_basis(quad);
    using FEMElement = std::conditional_t<Dim == 2, FEMQuad4, FEMHex8>;
    std::vector<PetscInt> ID_values(IEN_map.num_local_basis());
    LocalVectorAssemblyResult local;
    for (std::size_t cell_index = IEN_map.local_element_begin();
         cell_index < IEN_map.local_element_end(); ++cell_index)
    {
        const auto phy_nodes =
            gather_lor_nodes(mesh, IEN_map, cell_index);
        const FEMElement element(phy_nodes, ref_basis);
        assemble_local_load(element, source, local);
        update_statistics(local, result->statistics);
        PetscCall(add_local_vector(result->vector, IEN_map, ID_map,
            cell_index, local, ID_values));
    }

    // Finalize this LOR vector without creating either matrix.
    PetscCall(finalize_vector(result->vector));
    PetscCall(reduce_statistics(IEN_map.communicator(), result->statistics));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Assemble a high-order matrix/load combination in one element traversal.
// DoStiffness and DoMass select one or both matrices at compile time.
template <int Dim, bool DoStiffness, bool DoMass>
PetscErrorCode assemble_high_order_with_load(
    const BSplinePatch<Dim> &patch, const BSplineIEN<Dim> &IEN_map,
    const ID &ID_map, const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables,
    const ScalarSource<Dim> &source,
    GlobalMatrixAssemblyResult *stiffness,
    GlobalMatrixAssemblyResult *mass, GlobalVectorAssemblyResult *load)
{
    PetscFunctionBeginUser;

    // Validate every requested output before creating any PETSc object.
    static_assert(DoStiffness || DoMass,
        "A fused assembly must request at least one matrix");
    PetscCheck(load != nullptr, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
        "The fused load result pointer must not be null");
    PetscCheck(load->vector == nullptr, PETSC_COMM_SELF,
        PETSC_ERR_ARG_WRONGSTATE,
        "The fused load result already owns a PETSc vector");
    PetscCheck(static_cast<bool>(source), PETSC_COMM_SELF,
        PETSC_ERR_ARG_NULL, "The scalar source function must not be empty");
    if constexpr (DoStiffness)
    {
        PetscCheck(stiffness != nullptr, PETSC_COMM_SELF,
            PETSC_ERR_ARG_NULL,
            "The fused stiffness result pointer must not be null");
        PetscCheck(stiffness->matrix == nullptr, PETSC_COMM_SELF,
            PETSC_ERR_ARG_WRONGSTATE,
            "The fused stiffness result already owns a PETSc matrix");
    }
    if constexpr (DoMass)
    {
        PetscCheck(mass != nullptr, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
            "The fused mass result pointer must not be null");
        PetscCheck(mass->matrix == nullptr, PETSC_COMM_SELF,
            PETSC_ERR_ARG_WRONGSTATE,
            "The fused mass result already owns a PETSc matrix");
    }
    if constexpr (DoStiffness && DoMass)
        PetscCheck(stiffness != mass, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
            "Stiffness and mass require distinct global matrix results");
    PetscCall(validate_high_order_inputs(
        patch, IEN_map, ID_map, basis_tables));

    // Allocate exactly the requested global objects with the same high-order
    // connectivity estimate for either matrix.
    const PetscInt matrix_size = ID_map.num_free_dofs();
    const PetscInt local_size = IEN_map.num_owned_dofs() < 0 ?
        matrix_size : IEN_map.num_owned_dofs();
    const PetscInt nonzeros_per_row =
        high_order_nonzeros_per_row(patch, matrix_size);
    if constexpr (DoStiffness)
    {
        PetscCall(create_matrix(IEN_map.communicator(), local_size,
            matrix_size, nonzeros_per_row, *stiffness));
        PetscCall(validate_matrix_ownership(stiffness->matrix, IEN_map));
    }
    if constexpr (DoMass)
    {
        PetscCall(create_matrix(IEN_map.communicator(), local_size,
            matrix_size, nonzeros_per_row, *mass));
        PetscCall(validate_matrix_ownership(mass->matrix, IEN_map));
    }
    PetscCall(create_vector(IEN_map.communicator(), local_size,
        matrix_size, *load));
    PetscCall(validate_vector_ownership(load->vector, IEN_map));

    // Construct each physical element once, integrate all requested terms in
    // one Gauss loop, and compose its IEN/ID row once for every scatter.
    std::vector<typename BSplinePatch<Dim>::Point> control_points(
        IEN_map.num_local_basis());
    std::vector<PetscInt> ID_values(IEN_map.num_local_basis());
    LocalMatrixAssemblyResult local_stiffness;
    LocalMatrixAssemblyResult local_mass;
    LocalVectorAssemblyResult local_load;
    for (std::size_t element_index = IEN_map.local_element_begin();
         element_index < IEN_map.local_element_end(); ++element_index)
    {
        gather_control_points(
            patch, IEN_map, element_index, control_points);
        const BSplineElement<Dim> element(patch, element_index,
            control_points, quad, basis_tables);
        if constexpr (DoStiffness && DoMass)
            assemble_local_stiffness_mass_load<Dim>(element, source,
                local_stiffness, local_mass, local_load);
        else if constexpr (DoStiffness)
            assemble_local_stiffness_load<Dim>(
                element, source, local_stiffness, local_load);
        else
            assemble_local_mass_load<Dim>(
                element, source, local_mass, local_load);

        // All fused local results visit identical geometry points, but each
        // global result retains its own diagnostics for independent ownership.
        build_element_IDs(IEN_map, ID_map, element_index, ID_values);
        if constexpr (DoStiffness)
        {
            update_statistics(local_stiffness, stiffness->statistics);
            PetscCall(add_local_matrix_with_IDs(
                stiffness->matrix, local_stiffness, ID_values));
        }
        if constexpr (DoMass)
        {
            update_statistics(local_mass, mass->statistics);
            PetscCall(add_local_matrix_with_IDs(
                mass->matrix, local_mass, ID_values));
        }
        update_statistics(local_load, load->statistics);
        PetscCall(add_local_vector_with_IDs(
            load->vector, local_load, ID_values));
    }

    // Finalize every requested PETSc object after the shared traversal.
    if constexpr (DoStiffness)
    {
        PetscCall(finalize_matrix(stiffness->matrix));
        PetscCall(reduce_statistics(
            IEN_map.communicator(), stiffness->statistics));
    }
    if constexpr (DoMass)
    {
        PetscCall(finalize_matrix(mass->matrix));
        PetscCall(reduce_statistics(
            IEN_map.communicator(), mass->statistics));
    }
    PetscCall(finalize_vector(load->vector));
    PetscCall(reduce_statistics(IEN_map.communicator(), load->statistics));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Assemble an LOR matrix/load combination while reusing one Q1 basis table,
// one physical element, and one ID row per cell.
template <int Dim, bool DoStiffness, bool DoMass>
PetscErrorCode assemble_lor_with_load(
    const LORMesh<Dim> &mesh, const LORIEN<Dim> &IEN_map,
    const ID &ID_map, const QuadratureRule1D &quad,
    const ScalarSource<Dim> &source,
    GlobalMatrixAssemblyResult *stiffness,
    GlobalMatrixAssemblyResult *mass, GlobalVectorAssemblyResult *load)
{
    PetscFunctionBeginUser;

    // Validate every requested result and the common coefficient-aligned
    // topology before allocating global storage.
    static_assert(DoStiffness || DoMass,
        "A fused assembly must request at least one matrix");
    PetscCheck(load != nullptr, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
        "The fused load result pointer must not be null");
    PetscCheck(load->vector == nullptr, PETSC_COMM_SELF,
        PETSC_ERR_ARG_WRONGSTATE,
        "The fused load result already owns a PETSc vector");
    PetscCheck(static_cast<bool>(source), PETSC_COMM_SELF,
        PETSC_ERR_ARG_NULL, "The scalar source function must not be empty");
    if constexpr (DoStiffness)
    {
        PetscCheck(stiffness != nullptr, PETSC_COMM_SELF,
            PETSC_ERR_ARG_NULL,
            "The fused stiffness result pointer must not be null");
        PetscCheck(stiffness->matrix == nullptr, PETSC_COMM_SELF,
            PETSC_ERR_ARG_WRONGSTATE,
            "The fused stiffness result already owns a PETSc matrix");
    }
    if constexpr (DoMass)
    {
        PetscCheck(mass != nullptr, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
            "The fused mass result pointer must not be null");
        PetscCheck(mass->matrix == nullptr, PETSC_COMM_SELF,
            PETSC_ERR_ARG_WRONGSTATE,
            "The fused mass result already owns a PETSc matrix");
    }
    if constexpr (DoStiffness && DoMass)
        PetscCheck(stiffness != mass, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
            "Stiffness and mass require distinct global matrix results");
    PetscCall(validate_lor_inputs(mesh, IEN_map, ID_map));

    // Allocate exactly the selected Q1 matrices and load vector.
    const PetscInt matrix_size = ID_map.num_free_dofs();
    const PetscInt local_size = IEN_map.num_owned_dofs() < 0 ?
        matrix_size : IEN_map.num_owned_dofs();
    PetscInt nonzeros_per_row = 1;
    for (int direction = 0; direction < Dim; ++direction)
        nonzeros_per_row *= 3;
    nonzeros_per_row = std::min(matrix_size, nonzeros_per_row);
    if constexpr (DoStiffness)
    {
        PetscCall(create_matrix(IEN_map.communicator(), local_size,
            matrix_size, nonzeros_per_row, *stiffness));
        PetscCall(validate_matrix_ownership(stiffness->matrix, IEN_map));
    }
    if constexpr (DoMass)
    {
        PetscCall(create_matrix(IEN_map.communicator(), local_size,
            matrix_size, nonzeros_per_row, *mass));
        PetscCall(validate_matrix_ownership(mass->matrix, IEN_map));
    }
    PetscCall(create_vector(IEN_map.communicator(), local_size,
        matrix_size, *load));
    PetscCall(validate_vector_ownership(load->vector, IEN_map));

    // Reuse the reference table and fuse all selected cell integrations.
    const FEMQ1Basis<Dim> ref_basis(quad);
    using FEMElement = std::conditional_t<Dim == 2, FEMQuad4, FEMHex8>;
    std::vector<PetscInt> ID_values(IEN_map.num_local_basis());
    LocalMatrixAssemblyResult local_stiffness;
    LocalMatrixAssemblyResult local_mass;
    LocalVectorAssemblyResult local_load;
    for (std::size_t cell_index = IEN_map.local_element_begin();
         cell_index < IEN_map.local_element_end(); ++cell_index)
    {
        const auto phy_nodes =
            gather_lor_nodes(mesh, IEN_map, cell_index);
        const FEMElement element(phy_nodes, ref_basis);
        if constexpr (DoStiffness && DoMass)
            assemble_local_stiffness_mass_load(element, source,
                local_stiffness, local_mass, local_load);
        else if constexpr (DoStiffness)
            assemble_local_stiffness_load(
                element, source, local_stiffness, local_load);
        else
            assemble_local_mass_load(
                element, source, local_mass, local_load);

        // Scatter all cell outputs through the one coefficient-aligned ID row.
        build_element_IDs(IEN_map, ID_map, cell_index, ID_values);
        if constexpr (DoStiffness)
        {
            update_statistics(local_stiffness, stiffness->statistics);
            PetscCall(add_local_matrix_with_IDs(
                stiffness->matrix, local_stiffness, ID_values));
        }
        if constexpr (DoMass)
        {
            update_statistics(local_mass, mass->statistics);
            PetscCall(add_local_matrix_with_IDs(
                mass->matrix, local_mass, ID_values));
        }
        update_statistics(local_load, load->statistics);
        PetscCall(add_local_vector_with_IDs(
            load->vector, local_load, ID_values));
    }

    // Finalize all selected matrices and the shared-traversal load vector.
    if constexpr (DoStiffness)
    {
        PetscCall(finalize_matrix(stiffness->matrix));
        PetscCall(reduce_statistics(
            IEN_map.communicator(), stiffness->statistics));
    }
    if constexpr (DoMass)
    {
        PetscCall(finalize_matrix(mass->matrix));
        PetscCall(reduce_statistics(
            IEN_map.communicator(), mass->statistics));
    }
    PetscCall(finalize_vector(load->vector));
    PetscCall(reduce_statistics(IEN_map.communicator(), load->statistics));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

PetscErrorCode destroy_global_matrix(GlobalMatrixAssemblyResult *result)
{
    PetscFunctionBeginUser;
    if (result != nullptr)
        PetscCall(MatDestroy(&result->matrix));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode destroy_global_vector(GlobalVectorAssemblyResult *result)
{
    PetscFunctionBeginUser;
    if (result != nullptr)
        PetscCall(VecDestroy(&result->vector));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template <int Dim>
PetscErrorCode assemble_global_stiffness(
    const BSplinePatch<Dim> &patch, const BSplineIEN<Dim> &IEN_map,
    const ID &ID_map, const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables,
    GlobalMatrixAssemblyResult *result)
{
    return assemble_high_order_matrix(MatrixQuantity::stiffness,
        patch, IEN_map, ID_map, quad, basis_tables, result);
}

template <int Dim>
PetscErrorCode assemble_global_mass(
    const BSplinePatch<Dim> &patch, const BSplineIEN<Dim> &IEN_map,
    const ID &ID_map, const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables,
    GlobalMatrixAssemblyResult *result)
{
    return assemble_high_order_matrix(MatrixQuantity::mass,
        patch, IEN_map, ID_map, quad, basis_tables, result);
}

template <int Dim>
PetscErrorCode assemble_global_load(
    const BSplinePatch<Dim> &patch, const BSplineIEN<Dim> &IEN_map,
    const ID &ID_map, const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables,
    const ScalarSource<Dim> &source,
    GlobalVectorAssemblyResult *result)
{
    return assemble_high_order_load<Dim>(patch, IEN_map, ID_map,
        quad, basis_tables, source, result);
}

template <int Dim>
PetscErrorCode assemble_global_stiffness(
    const LORMesh<Dim> &mesh, const LORIEN<Dim> &IEN_map,
    const ID &ID_map, const QuadratureRule1D &quad,
    GlobalMatrixAssemblyResult *result)
{
    return assemble_lor_matrix(MatrixQuantity::stiffness,
        mesh, IEN_map, ID_map, quad, result);
}

template <int Dim>
PetscErrorCode assemble_global_mass(
    const LORMesh<Dim> &mesh, const LORIEN<Dim> &IEN_map,
    const ID &ID_map, const QuadratureRule1D &quad,
    GlobalMatrixAssemblyResult *result)
{
    return assemble_lor_matrix(MatrixQuantity::mass,
        mesh, IEN_map, ID_map, quad, result);
}

template <int Dim>
PetscErrorCode assemble_global_load(
    const LORMesh<Dim> &mesh, const LORIEN<Dim> &IEN_map,
    const ID &ID_map, const QuadratureRule1D &quad,
    const ScalarSource<Dim> &source,
    GlobalVectorAssemblyResult *result)
{
    return assemble_lor_load<Dim>(mesh, IEN_map, ID_map,
        quad, source, result);
}

template <int Dim>
PetscErrorCode assemble_global_stiffness_load(
    const BSplinePatch<Dim> &patch, const BSplineIEN<Dim> &IEN_map,
    const ID &ID_map, const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables,
    const ScalarSource<Dim> &source,
    GlobalMatrixAssemblyResult *stiffness,
    GlobalVectorAssemblyResult *load)
{
    // Select the high-order stiffness-plus-load fused traversal.
    return assemble_high_order_with_load<Dim, true, false>(patch, IEN_map,
        ID_map, quad, basis_tables, source, stiffness, nullptr, load);
}

template <int Dim>
PetscErrorCode assemble_global_mass_load(
    const BSplinePatch<Dim> &patch, const BSplineIEN<Dim> &IEN_map,
    const ID &ID_map, const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables,
    const ScalarSource<Dim> &source, GlobalMatrixAssemblyResult *mass,
    GlobalVectorAssemblyResult *load)
{
    // Select the high-order mass-plus-load fused traversal.
    return assemble_high_order_with_load<Dim, false, true>(patch, IEN_map,
        ID_map, quad, basis_tables, source, nullptr, mass, load);
}

template <int Dim>
PetscErrorCode assemble_global_stiffness_mass_load(
    const BSplinePatch<Dim> &patch, const BSplineIEN<Dim> &IEN_map,
    const ID &ID_map, const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables,
    const ScalarSource<Dim> &source,
    GlobalMatrixAssemblyResult *stiffness,
    GlobalMatrixAssemblyResult *mass, GlobalVectorAssemblyResult *load)
{
    // Select the high-order stiffness-plus-mass-plus-load fused traversal.
    return assemble_high_order_with_load<Dim, true, true>(patch, IEN_map,
        ID_map, quad, basis_tables, source, stiffness, mass, load);
}

template <int Dim>
PetscErrorCode assemble_global_stiffness_load(
    const LORMesh<Dim> &mesh, const LORIEN<Dim> &IEN_map,
    const ID &ID_map, const QuadratureRule1D &quad,
    const ScalarSource<Dim> &source,
    GlobalMatrixAssemblyResult *stiffness,
    GlobalVectorAssemblyResult *load)
{
    // Select the LOR stiffness-plus-load fused traversal.
    return assemble_lor_with_load<Dim, true, false>(mesh, IEN_map, ID_map,
        quad, source, stiffness, nullptr, load);
}

template <int Dim>
PetscErrorCode assemble_global_mass_load(
    const LORMesh<Dim> &mesh, const LORIEN<Dim> &IEN_map,
    const ID &ID_map, const QuadratureRule1D &quad,
    const ScalarSource<Dim> &source, GlobalMatrixAssemblyResult *mass,
    GlobalVectorAssemblyResult *load)
{
    // Select the LOR mass-plus-load fused traversal.
    return assemble_lor_with_load<Dim, false, true>(mesh, IEN_map, ID_map,
        quad, source, nullptr, mass, load);
}

template <int Dim>
PetscErrorCode assemble_global_stiffness_mass_load(
    const LORMesh<Dim> &mesh, const LORIEN<Dim> &IEN_map,
    const ID &ID_map, const QuadratureRule1D &quad,
    const ScalarSource<Dim> &source,
    GlobalMatrixAssemblyResult *stiffness,
    GlobalMatrixAssemblyResult *mass, GlobalVectorAssemblyResult *load)
{
    // Select the LOR stiffness-plus-mass-plus-load fused traversal.
    return assemble_lor_with_load<Dim, true, true>(mesh, IEN_map, ID_map,
        quad, source, stiffness, mass, load);
}

// Emit the 2D and 3D high-order/LOR variants for all single and fused public
// entry points whose implementations remain in this source file.
template PetscErrorCode assemble_global_stiffness<2>(
    const BSplinePatch<2> &, const BSplineIEN<2> &, const ID &,
    const QuadratureRule1D &, const std::vector<BSplineBasis1D> &,
    GlobalMatrixAssemblyResult *);
template PetscErrorCode assemble_global_stiffness<3>(
    const BSplinePatch<3> &, const BSplineIEN<3> &, const ID &,
    const QuadratureRule1D &, const std::vector<BSplineBasis1D> &,
    GlobalMatrixAssemblyResult *);
template PetscErrorCode assemble_global_mass<2>(
    const BSplinePatch<2> &, const BSplineIEN<2> &, const ID &,
    const QuadratureRule1D &, const std::vector<BSplineBasis1D> &,
    GlobalMatrixAssemblyResult *);
template PetscErrorCode assemble_global_mass<3>(
    const BSplinePatch<3> &, const BSplineIEN<3> &, const ID &,
    const QuadratureRule1D &, const std::vector<BSplineBasis1D> &,
    GlobalMatrixAssemblyResult *);
template PetscErrorCode assemble_global_load<2>(
    const BSplinePatch<2> &, const BSplineIEN<2> &, const ID &,
    const QuadratureRule1D &, const std::vector<BSplineBasis1D> &,
    const ScalarSource<2> &, GlobalVectorAssemblyResult *);
template PetscErrorCode assemble_global_load<3>(
    const BSplinePatch<3> &, const BSplineIEN<3> &, const ID &,
    const QuadratureRule1D &, const std::vector<BSplineBasis1D> &,
    const ScalarSource<3> &, GlobalVectorAssemblyResult *);
template PetscErrorCode assemble_global_stiffness<2>(
    const LORMesh<2> &, const LORIEN<2> &, const ID &,
    const QuadratureRule1D &, GlobalMatrixAssemblyResult *);
template PetscErrorCode assemble_global_stiffness<3>(
    const LORMesh<3> &, const LORIEN<3> &, const ID &,
    const QuadratureRule1D &, GlobalMatrixAssemblyResult *);
template PetscErrorCode assemble_global_mass<2>(
    const LORMesh<2> &, const LORIEN<2> &, const ID &,
    const QuadratureRule1D &, GlobalMatrixAssemblyResult *);
template PetscErrorCode assemble_global_mass<3>(
    const LORMesh<3> &, const LORIEN<3> &, const ID &,
    const QuadratureRule1D &, GlobalMatrixAssemblyResult *);
template PetscErrorCode assemble_global_load<2>(
    const LORMesh<2> &, const LORIEN<2> &, const ID &,
    const QuadratureRule1D &, const ScalarSource<2> &,
    GlobalVectorAssemblyResult *);
template PetscErrorCode assemble_global_load<3>(
    const LORMesh<3> &, const LORIEN<3> &, const ID &,
    const QuadratureRule1D &, const ScalarSource<3> &,
    GlobalVectorAssemblyResult *);
template PetscErrorCode assemble_global_stiffness_load<2>(
    const BSplinePatch<2> &, const BSplineIEN<2> &, const ID &,
    const QuadratureRule1D &, const std::vector<BSplineBasis1D> &,
    const ScalarSource<2> &, GlobalMatrixAssemblyResult *,
    GlobalVectorAssemblyResult *);
template PetscErrorCode assemble_global_stiffness_load<3>(
    const BSplinePatch<3> &, const BSplineIEN<3> &, const ID &,
    const QuadratureRule1D &, const std::vector<BSplineBasis1D> &,
    const ScalarSource<3> &, GlobalMatrixAssemblyResult *,
    GlobalVectorAssemblyResult *);
template PetscErrorCode assemble_global_mass_load<2>(
    const BSplinePatch<2> &, const BSplineIEN<2> &, const ID &,
    const QuadratureRule1D &, const std::vector<BSplineBasis1D> &,
    const ScalarSource<2> &, GlobalMatrixAssemblyResult *,
    GlobalVectorAssemblyResult *);
template PetscErrorCode assemble_global_mass_load<3>(
    const BSplinePatch<3> &, const BSplineIEN<3> &, const ID &,
    const QuadratureRule1D &, const std::vector<BSplineBasis1D> &,
    const ScalarSource<3> &, GlobalMatrixAssemblyResult *,
    GlobalVectorAssemblyResult *);
template PetscErrorCode assemble_global_stiffness_mass_load<2>(
    const BSplinePatch<2> &, const BSplineIEN<2> &, const ID &,
    const QuadratureRule1D &, const std::vector<BSplineBasis1D> &,
    const ScalarSource<2> &, GlobalMatrixAssemblyResult *,
    GlobalMatrixAssemblyResult *, GlobalVectorAssemblyResult *);
template PetscErrorCode assemble_global_stiffness_mass_load<3>(
    const BSplinePatch<3> &, const BSplineIEN<3> &, const ID &,
    const QuadratureRule1D &, const std::vector<BSplineBasis1D> &,
    const ScalarSource<3> &, GlobalMatrixAssemblyResult *,
    GlobalMatrixAssemblyResult *, GlobalVectorAssemblyResult *);
template PetscErrorCode assemble_global_stiffness_load<2>(
    const LORMesh<2> &, const LORIEN<2> &, const ID &,
    const QuadratureRule1D &, const ScalarSource<2> &,
    GlobalMatrixAssemblyResult *, GlobalVectorAssemblyResult *);
template PetscErrorCode assemble_global_stiffness_load<3>(
    const LORMesh<3> &, const LORIEN<3> &, const ID &,
    const QuadratureRule1D &, const ScalarSource<3> &,
    GlobalMatrixAssemblyResult *, GlobalVectorAssemblyResult *);
template PetscErrorCode assemble_global_mass_load<2>(
    const LORMesh<2> &, const LORIEN<2> &, const ID &,
    const QuadratureRule1D &, const ScalarSource<2> &,
    GlobalMatrixAssemblyResult *, GlobalVectorAssemblyResult *);
template PetscErrorCode assemble_global_mass_load<3>(
    const LORMesh<3> &, const LORIEN<3> &, const ID &,
    const QuadratureRule1D &, const ScalarSource<3> &,
    GlobalMatrixAssemblyResult *, GlobalVectorAssemblyResult *);
template PetscErrorCode assemble_global_stiffness_mass_load<2>(
    const LORMesh<2> &, const LORIEN<2> &, const ID &,
    const QuadratureRule1D &, const ScalarSource<2> &,
    GlobalMatrixAssemblyResult *, GlobalMatrixAssemblyResult *,
    GlobalVectorAssemblyResult *);
template PetscErrorCode assemble_global_stiffness_mass_load<3>(
    const LORMesh<3> &, const LORIEN<3> &, const ID &,
    const QuadratureRule1D &, const ScalarSource<3> &,
    GlobalMatrixAssemblyResult *, GlobalMatrixAssemblyResult *,
    GlobalVectorAssemblyResult *);

} // namespace lor
