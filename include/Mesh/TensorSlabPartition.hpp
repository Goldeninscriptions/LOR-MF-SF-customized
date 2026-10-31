#ifndef LOR_MESH_TENSOR_SLAB_PARTITION_HPP
#define LOR_MESH_TENSOR_SLAB_PARTITION_HPP

#include <mpi.h>

#include <array>
#include <cstddef>

namespace lor
{

class ID;

template <int Dim>
class BSplinePatch;

template <int Dim>
class LORMesh;

// MPI work/ownership partition for a direction-0-fastest tensor grid. The
// slowest tensor direction (Dim-1) is divided into contiguous slabs so every
// rank owns a contiguous ID range without changing lexicographic global_index
// numbering. High-order elements and LOR cells are partitioned independently
// because their directional counts differ, but each object is assigned to
// exactly one rank and both matrices share the same distributed vector layout.
template <int Dim>
class TensorSlabPartition
{
    static_assert(Dim == 2 || Dim == 3,
        "TensorSlabPartition supports only 2D and 3D");

public:
    using Shape = std::array<int, Dim>;

    // Build a balanced slab partition on comm. All-boundary homogeneous
    // Dirichlet numbering is validated against ID because contiguous ownership
    // relies on each directional boundary layer being removed.
    TensorSlabPartition(const BSplinePatch<Dim> &patch,
        const LORMesh<Dim> &lor_mesh, const ID &ID_map,
        MPI_Comm comm = MPI_COMM_WORLD);

    // MPI communicator, rank, and number of participating ranks.
    MPI_Comm communicator() const noexcept { return comm_; }
    int rank() const noexcept { return rank_; }
    int size() const noexcept { return size_; }

    // Half-open contiguous algebraic ID interval owned by this rank. Its
    // length is supplied as the local PETSc Mat/Vec size.
    int owned_ID_begin() const noexcept { return owned_ID_begin_; }
    int owned_ID_end() const noexcept { return owned_ID_end_; }
    int num_owned_dofs() const noexcept
    {
        return owned_ID_end_ - owned_ID_begin_;
    }

    // Half-open global flat-index intervals for this rank's high-order
    // elements and LOR cells. Direction 0 varies fastest, so a slow-direction
    // slab is one contiguous interval and needs no explicit index list.
    std::size_t high_element_begin() const noexcept
    {
        return high_element_begin_;
    }
    std::size_t high_element_end() const noexcept
    {
        return high_element_end_;
    }
    std::size_t lor_cell_begin() const noexcept { return lor_cell_begin_; }
    std::size_t lor_cell_end() const noexcept { return lor_cell_end_; }

    // Directional shapes retained for diagnostics and consistency checks.
    const Shape &basis_shape() const noexcept { return basis_shape_; }
    const Shape &high_element_shape() const noexcept
    {
        return high_element_shape_;
    }
    const Shape &lor_cell_shape() const noexcept { return lor_cell_shape_; }

private:
    // Communicator is borrowed from PETSc and is never freed by this object.
    MPI_Comm comm_ = MPI_COMM_NULL;
    int rank_ = 0;
    int size_ = 1;

    // Complete tensor shapes use direction-0-fastest ordering everywhere.
    Shape basis_shape_{};
    Shape high_element_shape_{};
    Shape lor_cell_shape_{};

    // Contiguous ID ownership and work ranges computed by balanced integer
    // division; remainders are distributed without dropping any object.
    int owned_ID_begin_ = 0;
    int owned_ID_end_ = 0;
    std::size_t high_element_begin_ = 0;
    std::size_t high_element_end_ = 0;
    std::size_t lor_cell_begin_ = 0;
    std::size_t lor_cell_end_ = 0;
};

extern template class TensorSlabPartition<2>;
extern template class TensorSlabPartition<3>;

} // namespace lor

#endif
