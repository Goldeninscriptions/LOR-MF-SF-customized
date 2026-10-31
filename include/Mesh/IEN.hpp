#ifndef LOR_MESH_IEN_HPP
#define LOR_MESH_IEN_HPP

#include "Assembly/LexicographicOrdering.hpp"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <vector>

namespace lor
{

template <int Dim>
class BSplinePatch;

template <int Dim>
class LORMesh;

template <int Dim>
class TensorSlabPartition;

// Read-only element-to-global-basis map shared by assembled operators. Tensor
// subclasses evaluate IEN(e,a) on demand instead of storing the complete
// num_elements*num_local_basis table on every MPI rank.
class IEN
{
public:
    virtual ~IEN() = default;

    // Return the global basis/node index associated with local basis a of
    // element e, matching MixPERIGEE's get_IEN(e, a) convention.
    virtual int get_IEN(
        std::size_t element, int local_basis) const = 0;

    // Return the number of elements and local basis functions represented by
    // this immutable IEN table.
    std::size_t num_elements() const noexcept { return num_elements_; }
    int num_local_basis() const noexcept { return num_local_basis_; }

    // Half-open flat element/cell interval assigned to this MPI rank. Serial
    // constructors expose the complete interval.
    std::size_t local_element_begin() const noexcept
    {
        return local_element_begin_;
    }
    std::size_t local_element_end() const noexcept
    {
        return local_element_end_;
    }

    // Local contiguous ID ownership and borrowed communicator used by PETSc.
    // Negative endpoints are reserved for legacy serial construction.
    int owned_ID_begin() const noexcept { return owned_ID_begin_; }
    int owned_ID_end() const noexcept { return owned_ID_end_; }
    int num_owned_dofs() const noexcept
    {
        return owned_ID_begin_ < 0 ? -1 :
            owned_ID_end_ - owned_ID_begin_;
    }
    MPI_Comm communicator() const noexcept { return comm_; }

protected:
    IEN(std::size_t num_elements, int num_local_basis,
        std::size_t local_element_begin, std::size_t local_element_end,
        int owned_ID_begin, int owned_ID_end, MPI_Comm comm);

private:
    std::size_t num_elements_ = 0;
    int num_local_basis_ = 0;
    std::size_t local_element_begin_ = 0;
    std::size_t local_element_end_ = 0;
    int owned_ID_begin_ = -1;
    int owned_ID_end_ = -1;
    MPI_Comm comm_ = MPI_COMM_SELF;
};

// IEN table for active tensor-product B-spline knot-span elements.
template <int Dim>
class BSplineIEN final : public IEN
{
public:
    explicit BSplineIEN(const BSplinePatch<Dim> &patch);
    BSplineIEN(const BSplinePatch<Dim> &patch,
        const TensorSlabPartition<Dim> &partition);

    // Compute the active tensor basis global_index for local basis a of
    // global element e without accessing a stored global IEN table.
    int get_IEN(std::size_t element, int local_basis) const override;

private:
    std::array<int, Dim> degrees_{};
    std::array<std::vector<int>, Dim> active_spans_;

    // Cached immutable orderings eliminate allocation/reconstruction from the
    // high-frequency on-demand IEN(e,a) path.
    LexicographicOrdering<Dim> basis_ordering_;
    LexicographicOrdering<Dim> element_ordering_;
    LexicographicOrdering<Dim> local_ordering_;
};

// IEN table for tensor-product Q1 cells of the LOR mesh.
template <int Dim>
class LORIEN final : public IEN
{
public:
    explicit LORIEN(const LORMesh<Dim> &mesh);
    LORIEN(const LORMesh<Dim> &mesh,
        const TensorSlabPartition<Dim> &partition);

    // Compute the conventional Quad4/Hex8 corner global_index for global
    // LOR cell e and local corner a directly from tensor shapes.
    int get_IEN(std::size_t element, int local_basis) const override;

private:
    // Cached node/cell/corner orderings give constant-storage Q1 connectivity
    // without rebuilding ordering metadata for every cell corner.
    LexicographicOrdering<Dim> node_ordering_;
    LexicographicOrdering<Dim> cell_ordering_;
    LexicographicOrdering<Dim> corner_ordering_;
};

extern template class BSplineIEN<2>;
extern template class BSplineIEN<3>;
extern template class LORIEN<2>;
extern template class LORIEN<3>;

} // namespace lor

#endif
