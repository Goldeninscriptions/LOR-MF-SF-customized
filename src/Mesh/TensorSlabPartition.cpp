#include "Mesh/TensorSlabPartition.hpp"

#include "Mesh/BSplinePatch.hpp"
#include "Mesh/ID.hpp"
#include "Mesh/LORMesh.hpp"

#include <limits>
#include <stdexcept>

namespace lor
{
namespace
{

// Return the start of rank's balanced half-open block in [0,total). This
// floor-product formula distributes nondivisible remainders without gaps or
// overlap and is identical on every rank.
std::size_t balanced_start(const std::size_t total,
    const int rank, const int size)
{
    return total * static_cast<std::size_t>(rank) /
        static_cast<std::size_t>(size);
}

// Multiply tensor extents preceding the slowest direction. One such product
// is the number of flat direction-0-fastest objects in a complete slab layer.
template <int Dim>
std::size_t objects_per_slowest_layer(const std::array<int, Dim> &shape)
{
    std::size_t count = 1;
    for (int direction = 0; direction < Dim - 1; ++direction)
        count *= static_cast<std::size_t>(shape[direction]);
    return count;
}

// Convert a size_t known to represent PETSc algebraic rows into the project's
// int-based ID range while rejecting an overflow before narrowing.
int checked_ID_count(const std::size_t count)
{
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("MPI slab ID count exceeds int range");
    return static_cast<int>(count);
}

} // namespace

template <int Dim>
TensorSlabPartition<Dim>::TensorSlabPartition(
    const BSplinePatch<Dim> &patch, const LORMesh<Dim> &lor_mesh,
    const ID &ID_map, const MPI_Comm comm)
    : comm_(comm), basis_shape_(patch.basis_shape())
{
    // Query the already initialized PETSc communicator and reject invalid
    // communicators before deriving any distributed ownership.
    if (comm_ == MPI_COMM_NULL)
        throw std::invalid_argument("MPI slab partition requires a communicator");
    if (MPI_Comm_rank(comm_, &rank_) != MPI_SUCCESS ||
        MPI_Comm_size(comm_, &size_) != MPI_SUCCESS)
        throw std::runtime_error("Cannot query MPI slab communicator");
    if (size_ < 1 || rank_ < 0 || rank_ >= size_)
        throw std::runtime_error("Invalid MPI rank or communicator size");

    // Record independent high-order and LOR tensor shapes. Their slowest
    // directional counts generally differ even though they share basis IDs.
    const Shape lor_node_shape = lor_mesh.node_shape();
    if (lor_node_shape != basis_shape_)
        throw std::invalid_argument(
            "LOR nodes must be coefficient-aligned with the B-spline basis");
    for (int direction = 0; direction < Dim; ++direction)
    {
        high_element_shape_[direction] = patch.num_elements(direction);
        lor_cell_shape_[direction] = lor_node_shape[direction] - 1;
        if (basis_shape_[direction] <= 2)
            throw std::invalid_argument(
                "Full Dirichlet MPI partition requires interior basis functions");
    }

    // Identify the slow-direction interior layers. Balanced floor division
    // remains valid even when a very large MPI count gives a rank an empty
    // interval; no object is dropped or duplicated in that case.
    const int slowest = Dim - 1;
    const int free_slowest_layers = basis_shape_[slowest] - 2;

    // Verify that ID represents exactly the full-boundary Dirichlet tensor
    // space assumed by contiguous slab ownership.
    std::size_t free_dofs_per_layer = 1;
    for (int direction = 0; direction < slowest; ++direction)
        free_dofs_per_layer *=
            static_cast<std::size_t>(basis_shape_[direction] - 2);
    const std::size_t expected_free_dofs = free_dofs_per_layer *
        static_cast<std::size_t>(free_slowest_layers);
    if (ID_map.num_global_indices() != static_cast<int>(patch.num_basis()) ||
        ID_map.num_free_dofs() != checked_ID_count(expected_free_dofs))
        throw std::invalid_argument(
            "ID is incompatible with full-boundary tensor slab ownership");

    // Partition free basis layers; because the slowest direction varies last,
    // these layer ranges map to contiguous ID intervals without renumbering.
    const std::size_t owned_layer_begin = balanced_start(
        free_slowest_layers, rank_, size_);
    const std::size_t owned_layer_end = balanced_start(
        free_slowest_layers, rank_ + 1, size_);
    owned_ID_begin_ = checked_ID_count(
        owned_layer_begin * free_dofs_per_layer);
    owned_ID_end_ = checked_ID_count(
        owned_layer_end * free_dofs_per_layer);

    // Partition high-order element layers independently and convert the layer
    // interval into one contiguous flat element interval.
    const std::size_t high_layer_begin = balanced_start(
        high_element_shape_[slowest], rank_, size_);
    const std::size_t high_layer_end = balanced_start(
        high_element_shape_[slowest], rank_ + 1, size_);
    const std::size_t high_per_layer =
        objects_per_slowest_layer<Dim>(high_element_shape_);
    high_element_begin_ = high_layer_begin * high_per_layer;
    high_element_end_ = high_layer_end * high_per_layer;

    // Partition the larger coefficient-aligned LOR cell grid with the same
    // ranks but its own slow-direction count, preserving complete coverage.
    const std::size_t lor_layer_begin = balanced_start(
        lor_cell_shape_[slowest], rank_, size_);
    const std::size_t lor_layer_end = balanced_start(
        lor_cell_shape_[slowest], rank_ + 1, size_);
    const std::size_t lor_per_layer =
        objects_per_slowest_layer<Dim>(lor_cell_shape_);
    lor_cell_begin_ = lor_layer_begin * lor_per_layer;
    lor_cell_end_ = lor_layer_end * lor_per_layer;
}

// Emit both paper dimensions from this source-owned template implementation.
template class TensorSlabPartition<2>;
template class TensorSlabPartition<3>;

} // namespace lor
