#include "Assembly/GlobalAssembly.hpp"

#include "Assembly/LexicographicOrdering.hpp"
#include "Element/BSplineElement.hpp"
#include "Element/FEMHex8.hpp"
#include "Element/FEMQ1Basis.hpp"
#include "Element/FEMQuad4.hpp"

#include <petscksp.h>
#include <slepcsvd.h>

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

// Exact per-owned-row diagonal/off-diagonal capacities and their global sum.
// The arrays are passed directly to PETSc AIJ preallocation.
struct MatrixPreallocation
{
    std::vector<PetscInt> diagonal;
    std::vector<PetscInt> off_diagonal;
    PetscInt local_nonzeros = 0;
};

// Initialize determinant extrema for an independent matrix or vector
// traversal before its first element contribution.
void initialize_statistics(AssemblyStatistics &statistics)
{
    statistics.min_det_jac = std::numeric_limits<double>::max();
    statistics.max_det_jac = 0.0;
}

// Create one reduced symmetric PETSc matrix from exact row-wise tensor
// connectivity, avoiding a provisional trial matrix and unused MPI slots.
PetscErrorCode create_matrix(const MPI_Comm comm, const PetscInt local_size,
    const PetscInt global_size, const MatrixPreallocation &preallocation,
    GlobalMatrixAssemblyResult &result)
{
    PetscFunctionBeginUser;
    PetscCheck(static_cast<PetscInt>(preallocation.diagonal.size()) ==
        local_size, comm, PETSC_ERR_ARG_SIZ,
        "Diagonal preallocation length must match local rows");
    PetscCheck(static_cast<PetscInt>(preallocation.off_diagonal.size()) ==
        local_size, comm, PETSC_ERR_ARG_SIZ,
        "Off-diagonal preallocation length must match local rows");
    const PetscInt *diagonal = preallocation.diagonal.empty() ? nullptr :
        preallocation.diagonal.data();
    const PetscInt *off_diagonal =
        preallocation.off_diagonal.empty() ? nullptr :
        preallocation.off_diagonal.data();
    // Report the portable storage lower bound before PETSc attempts the
    // allocation, so an external batch log survives a subsequent OOM kill.
    PetscCallMPI(MPI_Allreduce(&preallocation.local_nonzeros,
        &result.predicted_nonzeros, 1, MPIU_INT, MPI_SUM, comm));
    int communicator_size = 1;
    PetscCallMPI(MPI_Comm_size(comm, &communicator_size));
    result.estimated_minimum_bytes =
        static_cast<PetscLogDouble>(result.predicted_nonzeros) *
            (sizeof(PetscScalar) + sizeof(PetscInt)) +
        static_cast<PetscLogDouble>(
            2 * (global_size + communicator_size)) * sizeof(PetscInt);
    PetscCall(PetscPrintf(comm,
        "[matrix allocation] rows=%" PetscInt_FMT
        ", predicted_nnz=%" PetscInt_FMT
        ", estimated_minimum=%.3f MiB\n",
        global_size, result.predicted_nonzeros,
        result.estimated_minimum_bytes / (1024.0 * 1024.0)));

    // Allocate exactly the predicted row structure and turn any missed entry
    // into an immediate correctness error rather than a hidden reallocation.
    PetscCall(MatCreateAIJ(comm, local_size, local_size,
        global_size, global_size, 0, diagonal,
        0, off_diagonal, &result.matrix));
    PetscCall(MatSetOption(result.matrix, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatSetOption(
        result.matrix, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
    initialize_statistics(result.statistics);
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Determine whether two directional B-spline basis functions occur together
// on at least one positive-measure active knot span.
template <int Dim>
bool share_active_span(const BSplinePatch<Dim> &patch, const int direction,
    const int first_basis, const int second_basis)
{
    const int degree = patch.degree(direction);
    const int lower_span = std::max(first_basis, second_basis);
    const int upper_span = std::min(first_basis, second_basis) + degree;
    const auto &spans = patch.active_spans(direction);
    const auto candidate = std::lower_bound(
        spans.begin(), spans.end(), lower_span);
    return candidate != spans.end() && *candidate <= upper_span;
}

// Build exact one-dimensional adjacency lists after removing the two
// all-Dirichlet boundary layers. Tensor products of these lists define the
// structural rows of a high-order B-spline matrix.
template <int Dim>
std::array<std::vector<std::vector<int>>, Dim> high_order_adjacency(
    const BSplinePatch<Dim> &patch)
{
    std::array<std::vector<std::vector<int>>, Dim> adjacency;
    for (int direction = 0; direction < Dim; ++direction)
    {
        const int count = patch.num_basis(direction);
        const int degree = patch.degree(direction);
        adjacency[direction].resize(count);
        for (int row = 1; row < count - 1; ++row)
            for (int column = std::max(1, row - degree);
                 column <= std::min(count - 2, row + degree); ++column)
                if (share_active_span(
                        patch, direction, row, column))
                    adjacency[direction][row].push_back(column);
    }
    return adjacency;
}

// Build the exact three-point one-dimensional adjacency of the Q1 LOR grid
// after removing all-boundary constrained nodes.
template <int Dim>
std::array<std::vector<std::vector<int>>, Dim> lor_adjacency(
    const LORMesh<Dim> &mesh)
{
    const auto shape = mesh.node_shape();
    std::array<std::vector<std::vector<int>>, Dim> adjacency;
    for (int direction = 0; direction < Dim; ++direction)
    {
        adjacency[direction].resize(shape[direction]);
        for (int row = 1; row < shape[direction] - 1; ++row)
            for (int column = std::max(1, row - 1);
                 column <= std::min(shape[direction] - 2, row + 1); ++column)
                adjacency[direction][row].push_back(column);
    }
    return adjacency;
}

// Convert separable directional adjacency into exact PETSc diagonal and
// off-diagonal counts. Slow-direction slab ownership makes the split a small
// one-dimensional count multiplied by the inner-direction connectivity.
template <int Dim>
MatrixPreallocation tensor_preallocation(
    const std::array<int, Dim> &shape,
    const std::array<std::vector<std::vector<int>>, Dim> &adjacency,
    const ID &ID_map, const IEN &IEN_map)
{
    const PetscInt owned_begin = IEN_map.owned_ID_begin() < 0 ? 0 :
        IEN_map.owned_ID_begin();
    const PetscInt owned_end = IEN_map.owned_ID_end() < 0 ?
        ID_map.num_free_dofs() : IEN_map.owned_ID_end();
    const PetscInt local_size = owned_end - owned_begin;
    MatrixPreallocation result;
    result.diagonal.resize(local_size);
    result.off_diagonal.resize(local_size);
    const LexicographicOrdering<Dim> ordering(shape);

    // Count each owned tensor row without materializing its Cartesian-product
    // column list. Only the slow-direction neighbors determine MPI ownership.
    for (PetscInt local_row = 0; local_row < local_size; ++local_row)
    {
        const PetscInt row_ID = owned_begin + local_row;
        const auto row = ordering.multi_index(ID_map.global_index(row_ID));
        PetscInt inner_count = 1;
        for (int direction = 0; direction < Dim - 1; ++direction)
            inner_count *= static_cast<PetscInt>(
                adjacency[direction][row[direction]].size());
        PetscInt diagonal_slow_count = 0;
        for (const int slow_column : adjacency[Dim - 1][row[Dim - 1]])
        {
            auto column = row;
            column[Dim - 1] = slow_column;
            const int column_ID = ID_map.get_ID(
                ordering.global_index(column));
            if (column_ID >= owned_begin && column_ID < owned_end)
                ++diagonal_slow_count;
        }
        const PetscInt total = inner_count * static_cast<PetscInt>(
            adjacency[Dim - 1][row[Dim - 1]].size());
        result.diagonal[local_row] = inner_count * diagonal_slow_count;
        result.off_diagonal[local_row] =
            total - result.diagonal[local_row];
        result.local_nonzeros += total;
    }
    return result;
}

// Create one reduced load vector and configure ID=-1 constrained entries to
// be ignored during scatter.
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

// Reusable packing buffers for the subset of one local contribution whose
// rows belong to this rank. Columns may remain off-diagonal because PETSc
// stores them locally in the owned row without an MPI stash.
struct OwnedRowScratch
{
    std::vector<PetscInt> rows;
    std::vector<PetscScalar> matrix_entries;
    std::vector<PetscScalar> vector_entries;
};

bool is_owned_row(const PetscInt ID_value, const IEN &IEN_map)
{
    if (ID_value < 0)
        return false;
    return IEN_map.owned_ID_begin() < 0 ||
        (ID_value >= IEN_map.owned_ID_begin() &&
         ID_value < IEN_map.owned_ID_end());
}

// Mark the local test-function rows that survive Dirichlet elimination and
// belong to this rank's PETSc ownership interval.
void build_owned_row_mask(const std::vector<PetscInt> &ID_values,
    const IEN &IEN_map, LocalRowMask &mask)
{
    mask.resize(ID_values.size());
    for (std::size_t row = 0; row < ID_values.size(); ++row)
        mask[row] = is_owned_row(ID_values[row], IEN_map) ? 1 : 0;
}

// Preserve the symmetric half-matrix kernel whenever this rank owns every
// unconstrained row touched by the element. Only interface-halo elements need
// the more expensive rectangular owned-row integration.
bool all_free_rows_are_owned(const std::vector<PetscInt> &ID_values,
    const IEN &IEN_map)
{
    return std::all_of(ID_values.begin(), ID_values.end(),
        [&IEN_map](const PetscInt ID_value)
        {
            return ID_value < 0 || is_owned_row(ID_value, IEN_map);
        });
}

// Scatter a local matrix using an ID row already built for the current
// element. Fused assembly calls this for multiple outputs without repeating
// the IEN-to-ID composition.
PetscErrorCode add_local_matrix_with_IDs(Mat matrix,
    const LocalMatrixAssemblyResult &local,
    const std::vector<PetscInt> &ID_values, const IEN &IEN_map,
    OwnedRowScratch &scratch)
{
    PetscFunctionBeginUser;
    const PetscInt num_basis = static_cast<PetscInt>(ID_values.size());
    scratch.rows.clear();
    scratch.matrix_entries.clear();
    scratch.matrix_entries.reserve(local.entries.size());
    for (PetscInt row = 0; row < num_basis; ++row)
        if (is_owned_row(ID_values[row], IEN_map))
        {
            scratch.rows.push_back(ID_values[row]);
            const auto begin = local.entries.begin() + row * num_basis;
            scratch.matrix_entries.insert(scratch.matrix_entries.end(),
                begin, begin + num_basis);
        }
    if (!scratch.rows.empty())
        PetscCall(MatSetValues(matrix,
            static_cast<PetscInt>(scratch.rows.size()), scratch.rows.data(),
            num_basis, ID_values.data(), scratch.matrix_entries.data(),
            ADD_VALUES));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Scatter a local vector using the same prebuilt ID row as fused matrices.
PetscErrorCode add_local_vector_with_IDs(Vec vector,
    const LocalVectorAssemblyResult &local,
    const std::vector<PetscInt> &ID_values, const IEN &IEN_map,
    OwnedRowScratch &scratch)
{
    PetscFunctionBeginUser;
    scratch.rows.clear();
    scratch.vector_entries.clear();
    for (std::size_t row = 0; row < ID_values.size(); ++row)
        if (is_owned_row(ID_values[row], IEN_map))
        {
            scratch.rows.push_back(ID_values[row]);
            scratch.vector_entries.push_back(local.entries[row]);
        }
    if (!scratch.rows.empty())
        PetscCall(VecSetValues(vector,
            static_cast<PetscInt>(scratch.rows.size()), scratch.rows.data(),
            scratch.vector_entries.data(), ADD_VALUES));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Scatter one complete row-major local matrix into its independently selected
// global stiffness or mass matrix.
PetscErrorCode add_local_matrix(Mat matrix, const IEN &IEN_map,
    const ID &ID_map, const std::size_t element,
    const LocalMatrixAssemblyResult &local,
    std::vector<PetscInt> &ID_values, OwnedRowScratch &scratch)
{
    PetscFunctionBeginUser;
    build_element_IDs(IEN_map, ID_map, element, ID_values);
    PetscCall(add_local_matrix_with_IDs(
        matrix, local, ID_values, IEN_map, scratch));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Scatter one local source vector into the independently selected global load
// vector; configured negative indices omit constrained equations.
PetscErrorCode add_local_vector(Vec vector, const IEN &IEN_map,
    const ID &ID_map, const std::size_t element,
    const LocalVectorAssemblyResult &local,
    std::vector<PetscInt> &ID_values, OwnedRowScratch &scratch)
{
    PetscFunctionBeginUser;
    build_element_IDs(IEN_map, ID_map, element, ID_values);
    PetscCall(add_local_vector_with_IDs(
        vector, local, ID_values, IEN_map, scratch));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Complete PETSc matrix assembly and retain global storage diagnostics for
// resource planning and cross-machine batch logs.
PetscErrorCode finalize_matrix(GlobalMatrixAssemblyResult &result)
{
    PetscFunctionBeginUser;
    PetscCall(MatAssemblyBegin(result.matrix, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(result.matrix, MAT_FINAL_ASSEMBLY));
    MatInfo info{};
    PetscCall(MatGetInfo(result.matrix, MAT_GLOBAL_SUM, &info));
    result.allocated_nonzeros = info.nz_allocated;
    result.used_nonzeros = info.nz_used;
    result.allocation_calls = info.mallocs;
    result.memory_bytes = info.memory;
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

    // Allocate only the requested sparse matrix from exact tensor support.
    const PetscInt matrix_size = ID_map.num_free_dofs();
    const PetscInt local_size = IEN_map.num_owned_dofs() < 0 ?
        matrix_size : IEN_map.num_owned_dofs();
    const MatrixPreallocation preallocation = tensor_preallocation<Dim>(
        patch.basis_shape(), high_order_adjacency(patch), ID_map, IEN_map);
    PetscCall(create_matrix(IEN_map.communicator(), local_size, matrix_size,
        preallocation, *result));
    PetscCall(validate_matrix_ownership(result->matrix, IEN_map));

    // Follow gather -> Element -> selected LocalAssembly -> matrix scatter.
    std::vector<typename BSplinePatch<Dim>::Point> control_points(
        IEN_map.num_local_basis());
    std::vector<PetscInt> ID_values(IEN_map.num_local_basis());
    LocalRowMask active_rows;
    OwnedRowScratch row_scratch;
    LocalMatrixAssemblyResult local;
    BSplineElement<Dim> element;
    for (std::size_t element_index = IEN_map.assembly_element_begin();
         element_index < IEN_map.assembly_element_end(); ++element_index)
    {
        gather_control_points(
            patch, IEN_map, element_index, control_points);
        build_element_IDs(IEN_map, ID_map, element_index, ID_values);
        build_owned_row_mask(ID_values, IEN_map, active_rows);
        element.reinitialize(patch, element_index,
            control_points, quad, basis_tables);
        if (quantity == MatrixQuantity::stiffness)
        {
            if (all_free_rows_are_owned(ID_values, IEN_map))
                assemble_local_stiffness<Dim>(element, local);
            else
                assemble_local_stiffness_rows<Dim>(
                    element, active_rows, local);
        }
        else
            assemble_local_mass<Dim>(element, local);
        update_statistics(local, result->statistics);
        PetscCall(add_local_matrix_with_IDs(result->matrix, local,
            ID_values, IEN_map, row_scratch));
    }

    // Finalize this matrix and reduce geometry diagnostics over all slabs.
    PetscCall(finalize_matrix(*result));
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
    LocalRowMask active_rows;
    OwnedRowScratch row_scratch;
    LocalVectorAssemblyResult local;
    BSplineElement<Dim> element;
    for (std::size_t element_index = IEN_map.assembly_element_begin();
         element_index < IEN_map.assembly_element_end(); ++element_index)
    {
        gather_control_points(
            patch, IEN_map, element_index, control_points);
        build_element_IDs(IEN_map, ID_map, element_index, ID_values);
        build_owned_row_mask(ID_values, IEN_map, active_rows);
        element.reinitialize(patch, element_index,
            control_points, quad, basis_tables);
        if (all_free_rows_are_owned(ID_values, IEN_map))
            assemble_local_load<Dim>(element, source, local);
        else
            assemble_local_load_rows<Dim>(
                element, source, active_rows, local);
        update_statistics(local, result->statistics);
        PetscCall(add_local_vector_with_IDs(result->vector, local,
            ID_values, IEN_map, row_scratch));
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

    // Allocate only the selected Q1 matrix from its exact tensor adjacency.
    const PetscInt matrix_size = ID_map.num_free_dofs();
    const PetscInt local_size = IEN_map.num_owned_dofs() < 0 ?
        matrix_size : IEN_map.num_owned_dofs();
    const MatrixPreallocation preallocation = tensor_preallocation<Dim>(
        mesh.node_shape(), lor_adjacency(mesh), ID_map, IEN_map);
    PetscCall(create_matrix(IEN_map.communicator(), local_size, matrix_size,
        preallocation, *result));
    PetscCall(validate_matrix_ownership(result->matrix, IEN_map));

    // Reuse one reference Q1 table for gather -> Element -> selected local
    // matrix integration -> scatter over all LOR cells.
    const FEMQ1Basis<Dim> ref_basis(quad);
    using FEMElement = std::conditional_t<Dim == 2, FEMQuad4, FEMHex8>;
    std::vector<PetscInt> ID_values(IEN_map.num_local_basis());
    OwnedRowScratch row_scratch;
    LocalMatrixAssemblyResult local;
    for (std::size_t cell_index = IEN_map.assembly_element_begin();
         cell_index < IEN_map.assembly_element_end(); ++cell_index)
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
            cell_index, local, ID_values, row_scratch));
    }

    // Finalize this LOR matrix without creating a load vector.
    PetscCall(finalize_matrix(*result));
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
    OwnedRowScratch row_scratch;
    LocalVectorAssemblyResult local;
    for (std::size_t cell_index = IEN_map.assembly_element_begin();
         cell_index < IEN_map.assembly_element_end(); ++cell_index)
    {
        const auto phy_nodes =
            gather_lor_nodes(mesh, IEN_map, cell_index);
        const FEMElement element(phy_nodes, ref_basis);
        assemble_local_load(element, source, local);
        update_statistics(local, result->statistics);
        PetscCall(add_local_vector(result->vector, IEN_map, ID_map,
            cell_index, local, ID_values, row_scratch));
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
    GlobalMatrixAssemblyResult *mass, GlobalVectorAssemblyResult *load,
    const std::vector<double> *dirichlet_coefficients = nullptr)
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
    if (dirichlet_coefficients != nullptr)
    {
        PetscCheck(DoStiffness, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
            "Dirichlet elimination requires a stiffness matrix");
        PetscCheck(dirichlet_coefficients->size() == patch.num_basis(),
            PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
            "Dirichlet coefficient vector must match the Patch basis count");
    }
    PetscCall(validate_high_order_inputs(
        patch, IEN_map, ID_map, basis_tables));

    // Allocate exactly the requested global objects with one shared exact
    // high-order tensor-connectivity description.
    const PetscInt matrix_size = ID_map.num_free_dofs();
    const PetscInt local_size = IEN_map.num_owned_dofs() < 0 ?
        matrix_size : IEN_map.num_owned_dofs();
    const MatrixPreallocation preallocation = tensor_preallocation<Dim>(
        patch.basis_shape(), high_order_adjacency(patch), ID_map, IEN_map);
    if constexpr (DoStiffness)
    {
        PetscCall(create_matrix(IEN_map.communicator(), local_size,
            matrix_size, preallocation, *stiffness));
        PetscCall(validate_matrix_ownership(stiffness->matrix, IEN_map));
    }
    if constexpr (DoMass)
    {
        PetscCall(create_matrix(IEN_map.communicator(), local_size,
            matrix_size, preallocation, *mass));
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
    LocalRowMask active_rows;
    OwnedRowScratch row_scratch;
    LocalMatrixAssemblyResult local_stiffness;
    LocalMatrixAssemblyResult local_mass;
    LocalVectorAssemblyResult local_load;
    BSplineElement<Dim> element;
    for (std::size_t element_index = IEN_map.assembly_element_begin();
         element_index < IEN_map.assembly_element_end(); ++element_index)
    {
        gather_control_points(
            patch, IEN_map, element_index, control_points);
        build_element_IDs(IEN_map, ID_map, element_index, ID_values);
        build_owned_row_mask(ID_values, IEN_map, active_rows);
        element.reinitialize(patch, element_index,
            control_points, quad, basis_tables);
        if constexpr (DoStiffness && DoMass)
            assemble_local_stiffness_mass_load<Dim>(element, source,
                local_stiffness, local_mass, local_load);
        else if constexpr (DoStiffness)
        {
            if (all_free_rows_are_owned(ID_values, IEN_map))
                assemble_local_stiffness_load<Dim>(
                    element, source, local_stiffness, local_load);
            else
                assemble_local_stiffness_load_rows<Dim>(element, source,
                    active_rows, local_stiffness, local_load);
        }
        else
            assemble_local_mass_load<Dim>(
                element, source, local_mass, local_load);

        // Compose the element equation row once. When prescribed coefficients
        // are present, eliminate constrained columns from every free load row
        // before the reduced matrix and vector are scattered.
        if (dirichlet_coefficients != nullptr)
            for (int row = 0; row < IEN_map.num_local_basis(); ++row)
            {
                if (active_rows[row] == 0)
                    continue;
                for (int column = 0;
                     column < IEN_map.num_local_basis(); ++column)
                    if (ID_values[column] < 0)
                    {
                        const int constrained_global_index =
                            IEN_map.get_IEN(element_index, column);
                        local_load.entries[row] -=
                            local_stiffness.entries[
                                row * IEN_map.num_local_basis() + column] *
                            (*dirichlet_coefficients)[
                                constrained_global_index];
                    }
            }

        // All fused local results visit identical geometry points, but each
        // global result retains its own diagnostics for independent ownership.
        if constexpr (DoStiffness)
        {
            update_statistics(local_stiffness, stiffness->statistics);
            PetscCall(add_local_matrix_with_IDs(
                stiffness->matrix, local_stiffness, ID_values,
                IEN_map, row_scratch));
        }
        if constexpr (DoMass)
        {
            update_statistics(local_mass, mass->statistics);
            PetscCall(add_local_matrix_with_IDs(
                mass->matrix, local_mass, ID_values,
                IEN_map, row_scratch));
        }
        update_statistics(local_load, load->statistics);
        PetscCall(add_local_vector_with_IDs(
            load->vector, local_load, ID_values, IEN_map, row_scratch));
    }

    // Finalize every requested PETSc object after the shared traversal.
    if constexpr (DoStiffness)
    {
        PetscCall(finalize_matrix(*stiffness));
        PetscCall(reduce_statistics(
            IEN_map.communicator(), stiffness->statistics));
    }
    if constexpr (DoMass)
    {
        PetscCall(finalize_matrix(*mass));
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
    const MatrixPreallocation preallocation = tensor_preallocation<Dim>(
        mesh.node_shape(), lor_adjacency(mesh), ID_map, IEN_map);
    if constexpr (DoStiffness)
    {
        PetscCall(create_matrix(IEN_map.communicator(), local_size,
            matrix_size, preallocation, *stiffness));
        PetscCall(validate_matrix_ownership(stiffness->matrix, IEN_map));
    }
    if constexpr (DoMass)
    {
        PetscCall(create_matrix(IEN_map.communicator(), local_size,
            matrix_size, preallocation, *mass));
        PetscCall(validate_matrix_ownership(mass->matrix, IEN_map));
    }
    PetscCall(create_vector(IEN_map.communicator(), local_size,
        matrix_size, *load));
    PetscCall(validate_vector_ownership(load->vector, IEN_map));

    // Reuse the reference table and fuse all selected cell integrations.
    const FEMQ1Basis<Dim> ref_basis(quad);
    using FEMElement = std::conditional_t<Dim == 2, FEMQuad4, FEMHex8>;
    std::vector<PetscInt> ID_values(IEN_map.num_local_basis());
    OwnedRowScratch row_scratch;
    LocalMatrixAssemblyResult local_stiffness;
    LocalMatrixAssemblyResult local_mass;
    LocalVectorAssemblyResult local_load;
    for (std::size_t cell_index = IEN_map.assembly_element_begin();
         cell_index < IEN_map.assembly_element_end(); ++cell_index)
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
                stiffness->matrix, local_stiffness, ID_values,
                IEN_map, row_scratch));
        }
        if constexpr (DoMass)
        {
            update_statistics(local_mass, mass->statistics);
            PetscCall(add_local_matrix_with_IDs(
                mass->matrix, local_mass, ID_values,
                IEN_map, row_scratch));
        }
        update_statistics(local_load, load->statistics);
        PetscCall(add_local_vector_with_IDs(
            load->vector, local_load, ID_values, IEN_map, row_scratch));
    }

    // Finalize all selected matrices and the shared-traversal load vector.
    if constexpr (DoStiffness)
    {
        PetscCall(finalize_matrix(*stiffness));
        PetscCall(reduce_statistics(
            IEN_map.communicator(), stiffness->statistics));
    }
    if constexpr (DoMass)
    {
        PetscCall(finalize_matrix(*mass));
        PetscCall(reduce_statistics(
            IEN_map.communicator(), mass->statistics));
    }
    PetscCall(finalize_vector(load->vector));
    PetscCall(reduce_statistics(IEN_map.communicator(), load->statistics));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// State for the exact left-LOR-preconditioned shell used only by the optional
// SLEPc condition-number analysis.
struct LeftPreconditionedContext
{
    Mat stiffness = nullptr;
    KSP lor_solver = nullptr;
    Vec work = nullptr;
};

PetscErrorCode apply_left_preconditioned(Mat matrix, Vec x, Vec y)
{
    PetscFunctionBeginUser;
    LeftPreconditionedContext *context = nullptr;
    PetscCall(MatShellGetContext(matrix, &context));
    PetscCall(MatMult(context->stiffness, x, context->work));
    PetscCall(KSPSolve(context->lor_solver, context->work, y));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode apply_left_preconditioned_transpose(Mat matrix, Vec x, Vec y)
{
    PetscFunctionBeginUser;
    LeftPreconditionedContext *context = nullptr;
    PetscCall(MatShellGetContext(matrix, &context));
    PetscCall(KSPSolve(context->lor_solver, x, context->work));
    PetscCall(MatMult(context->stiffness, context->work, y));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Compute one requested extreme singular value with an independently prefixed
// two-sided TRLanczos solve on the matrix communicator.
PetscErrorCode extreme_singular_value(Mat matrix, const SVDWhich which,
    const char *prefix, const PetscReal tolerance,
    const PetscInt max_iterations,
    const PetscInt requested_subspace_dimension, PetscReal &sigma)
{
    PetscFunctionBeginUser;
    const MPI_Comm comm = PetscObjectComm(
        reinterpret_cast<PetscObject>(matrix));

    // Clamp the Krylov space to the operator dimension while retaining one
    // expansion vector beyond the requested singular triplet.
    PetscInt rows = 0;
    PetscInt columns = 0;
    PetscCall(MatGetSize(matrix, &rows, &columns));
    const PetscInt operator_dimension = std::min(rows, columns);
    const PetscInt subspace_dimension =
        std::min(requested_subspace_dimension, operator_dimension);
    PetscCheck(subspace_dimension >= 2, comm, PETSC_ERR_ARG_SIZ,
        "SVD operator must have dimension at least two");

    // Configure one SVD object; prefixed PETSc options remain able to override
    // defaults for focused spectral experiments.
    SVD svd = nullptr;
    PetscCall(SVDCreate(comm, &svd));
    PetscCall(SVDSetOptionsPrefix(svd, prefix));
    PetscCall(SVDSetOperators(svd, matrix, nullptr));
    PetscCall(SVDSetProblemType(svd, SVD_STANDARD));
    PetscCall(SVDSetType(svd, SVDTRLANCZOS));
    PetscCall(SVDTRLanczosSetOneSide(svd, PETSC_FALSE));
    if (which == SVD_SMALLEST)
        PetscCall(SVDSetDimensions(svd, 1, subspace_dimension,
            subspace_dimension - 1));
    else
        PetscCall(SVDSetDimensions(svd, 1,
            PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(SVDSetTolerances(svd, tolerance, max_iterations));
    PetscCall(SVDSetWhichSingularTriplets(svd, which));
    PetscCall(SVDSetFromOptions(svd));
    PetscCall(SVDSolve(svd));

    // Reject a partial spectral result instead of emitting a misleading
    // condition number from an unconverged triplet.
    PetscInt converged = 0;
    SVDConvergedReason reason = SVD_CONVERGED_ITERATING;
    PetscCall(SVDGetConverged(svd, &converged));
    PetscCall(SVDGetConvergedReason(svd, &reason));
    PetscCheck(converged > 0 && reason > 0, comm,
        PETSC_ERR_NOT_CONVERGED,
        "SLEPc singular-value solve with prefix '%s' did not converge",
        prefix);
    PetscCall(SVDGetSingularTriplet(svd, 0, &sigma, nullptr, nullptr));
    PetscCall(SVDDestroy(&svd));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Combine two independently converged extreme singular values into cond_2.
PetscErrorCode condition_number(Mat matrix, const char *largest_prefix,
    const char *smallest_prefix, const PetscReal tolerance,
    const PetscInt max_iterations, const PetscInt subspace_dimension,
    ConditionNumberResult &result)
{
    PetscFunctionBeginUser;
    const MPI_Comm comm = PetscObjectComm(
        reinterpret_cast<PetscObject>(matrix));
    PetscCall(extreme_singular_value(matrix, SVD_LARGEST, largest_prefix,
        tolerance, max_iterations, subspace_dimension, result.sigma_max));
    PetscCall(extreme_singular_value(matrix, SVD_SMALLEST, smallest_prefix,
        tolerance, max_iterations, subspace_dimension, result.sigma_min));
    PetscCheck(result.sigma_min > 0.0, comm, PETSC_ERR_PLIB,
        "SLEPc reported a nonpositive minimum singular value");
    result.condition_number = result.sigma_max / result.sigma_min;
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

PetscErrorCode compute_stiffness_condition_numbers(Mat stiffness,
    Mat lor_stiffness, const PetscReal svd_tolerance,
    const PetscInt max_iterations, const PetscInt subspace_dimension,
    ConditionNumberResult *original, ConditionNumberResult *preconditioned)
{
    PetscFunctionBeginUser;
    const MPI_Comm comm = PetscObjectComm(
        reinterpret_cast<PetscObject>(stiffness));

    // Validate optional-analysis arguments independently of every assembly and
    // linear-solve entry point.
    PetscCheck(original != nullptr && preconditioned != nullptr,
        comm, PETSC_ERR_ARG_NULL,
        "Condition-number results must not be null");
    PetscCheck(svd_tolerance > 0.0, comm, PETSC_ERR_ARG_OUTOFRANGE,
        "SVD tolerance must be positive");
    PetscCheck(max_iterations > 0, comm, PETSC_ERR_ARG_OUTOFRANGE,
        "SVD maximum iteration count must be positive");
    PetscCheck(subspace_dimension >= 2, comm, PETSC_ERR_ARG_OUTOFRANGE,
        "SVD subspace dimension must be at least two");

    PetscInt high_rows = 0;
    PetscInt high_columns = 0;
    PetscInt lor_rows = 0;
    PetscInt lor_columns = 0;
    PetscCall(MatGetSize(stiffness, &high_rows, &high_columns));
    PetscCall(MatGetSize(lor_stiffness, &lor_rows, &lor_columns));
    PetscCheck(high_rows == high_columns && lor_rows == lor_columns &&
        high_rows == lor_rows, comm, PETSC_ERR_ARG_SIZ,
        "K and K_LOR must be square matrices of the same size");

    // Analyze the assembled high-order stiffness directly.
    PetscCall(condition_number(stiffness, "cond_k_max_", "cond_k_min_",
        svd_tolerance, max_iterations, subspace_dimension, *original));

    // Factor K_LOR once and reuse exact inverse applications throughout the
    // preconditioned shell's two singular-value solves.
    LeftPreconditionedContext context;
    context.stiffness = stiffness;
    PetscCall(KSPCreate(comm, &context.lor_solver));
    PetscCall(KSPSetOperators(context.lor_solver,
        lor_stiffness, lor_stiffness));
    PetscCall(KSPSetType(context.lor_solver, KSPPREONLY));
    PC pc = nullptr;
    PetscCall(KSPGetPC(context.lor_solver, &pc));
    PetscCall(PCSetType(pc, PCLU));
    PetscMPIInt comm_size = 1;
    PetscCallMPI(MPI_Comm_size(comm, &comm_size));
    if (comm_size > 1)
    {
#if defined(PETSC_HAVE_MUMPS)
        PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
#else
        SETERRQ(comm, PETSC_ERR_SUP,
            "Parallel cond_2(K_LOR^{-1}K) requires PETSc with MUMPS");
#endif
    }
    PetscCall(KSPSetErrorIfNotConverged(context.lor_solver, PETSC_TRUE));
    PetscCall(KSPSetUp(context.lor_solver));
    PetscCall(MatCreateVecs(stiffness, &context.work, nullptr));

    // Expose M=P^{-1}K and M^T=KP^{-1} so SLEPc computes the true 2-norm.
    Mat preconditioned_operator = nullptr;
    PetscInt high_local_rows = 0;
    PetscInt high_local_columns = 0;
    PetscCall(MatGetLocalSize(stiffness,
        &high_local_rows, &high_local_columns));
    PetscCall(MatCreateShell(comm, high_local_rows, high_local_columns,
        high_rows, high_columns, &context, &preconditioned_operator));
    PetscCall(MatShellSetOperation(preconditioned_operator, MATOP_MULT,
        reinterpret_cast<void (*)(void)>(apply_left_preconditioned)));
    PetscCall(MatShellSetOperation(preconditioned_operator,
        MATOP_MULT_TRANSPOSE,
        reinterpret_cast<void (*)(void)>(apply_left_preconditioned_transpose)));
    PetscCall(condition_number(preconditioned_operator,
        "cond_lor_k_max_", "cond_lor_k_min_", svd_tolerance,
        max_iterations, subspace_dimension, *preconditioned));

    // Release every object owned solely by this optional analysis call.
    PetscCall(MatDestroy(&preconditioned_operator));
    PetscCall(VecDestroy(&context.work));
    PetscCall(KSPDestroy(&context.lor_solver));
    PetscFunctionReturn(PETSC_SUCCESS);
}

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
PetscErrorCode assemble_global_stiffness_load_with_dirichlet(
    const BSplinePatch<Dim> &patch, const BSplineIEN<Dim> &IEN_map,
    const ID &ID_map, const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables,
    const ScalarSource<Dim> &source,
    const std::vector<double> &dirichlet_coefficients,
    GlobalMatrixAssemblyResult *stiffness,
    GlobalVectorAssemblyResult *load)
{
    // Reuse the fused stiffness/source traversal and activate its local
    // -K_fc*g_c correction before reduced PETSc insertion.
    return assemble_high_order_with_load<Dim, true, false>(patch, IEN_map,
        ID_map, quad, basis_tables, source, stiffness, nullptr, load,
        &dirichlet_coefficients);
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
template PetscErrorCode assemble_global_stiffness_load_with_dirichlet<2>(
    const BSplinePatch<2> &, const BSplineIEN<2> &, const ID &,
    const QuadratureRule1D &, const std::vector<BSplineBasis1D> &,
    const ScalarSource<2> &, const std::vector<double> &,
    GlobalMatrixAssemblyResult *, GlobalVectorAssemblyResult *);
template PetscErrorCode assemble_global_stiffness_load_with_dirichlet<3>(
    const BSplinePatch<3> &, const BSplineIEN<3> &, const ID &,
    const QuadratureRule1D &, const std::vector<BSplineBasis1D> &,
    const ScalarSource<3> &, const std::vector<double> &,
    GlobalMatrixAssemblyResult *, GlobalVectorAssemblyResult *);
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
