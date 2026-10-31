#include "Mesh/IEN.hpp"

#include "Assembly/LexicographicOrdering.hpp"
#include "Mesh/BSplinePatch.hpp"
#include "Mesh/LORMesh.hpp"
#include "Mesh/TensorSlabPartition.hpp"

#include <array>
#include <stdexcept>

namespace lor
{
namespace
{

// Compute the number of local tensor-product B-spline basis functions.
template <int Dim>
int bspline_num_local_basis(const BSplinePatch<Dim> &patch)
{
    int count = 1;
    for (int direction = 0; direction < Dim; ++direction)
        count *= patch.degree(direction) + 1;
    return count;
}

// Copy the small directional topology required for on-demand B-spline IEN.
template <int Dim>
std::array<std::vector<int>, Dim> copy_active_spans(
    const BSplinePatch<Dim> &patch)
{
    std::array<std::vector<int>, Dim> spans;
    for (int direction = 0; direction < Dim; ++direction)
        spans[direction] = patch.active_spans(direction);
    return spans;
}

// Return the high-order tensor element shape without storing a full table.
template <int Dim>
std::array<int, Dim> high_element_shape(const BSplinePatch<Dim> &patch)
{
    std::array<int, Dim> shape{};
    for (int direction = 0; direction < Dim; ++direction)
        shape[direction] = patch.num_elements(direction);
    return shape;
}

// Copy the directional degrees before constructing the cached local ordering.
template <int Dim>
std::array<int, Dim> patch_degrees(const BSplinePatch<Dim> &patch)
{
    std::array<int, Dim> degrees{};
    for (int direction = 0; direction < Dim; ++direction)
        degrees[direction] = patch.degree(direction);
    return degrees;
}

// Return the local B-spline tensor shape from the retained directional degree.
template <int Dim>
std::array<int, Dim> local_bspline_shape(
    const std::array<int, Dim> &degrees)
{
    std::array<int, Dim> shape{};
    for (int direction = 0; direction < Dim; ++direction)
        shape[direction] = degrees[direction] + 1;
    return shape;
}

// Convert a LOR node tensor shape into its Q1 cell tensor shape.
template <int Dim>
std::array<int, Dim> lor_cell_shape(const LORMesh<Dim> &mesh)
{
    std::array<int, Dim> shape = mesh.node_shape();
    for (int direction = 0; direction < Dim; ++direction)
        --shape[direction];
    return shape;
}

// Return the binary tensor shape shared by Quad4 and Hex8 corners.
template <int Dim>
std::array<int, Dim> q1_corner_shape()
{
    std::array<int, Dim> shape{};
    shape.fill(2);
    return shape;
}

} // namespace

IEN::IEN(const std::size_t num_elements, const int num_local_basis,
    const std::size_t local_element_begin,
    const std::size_t local_element_end, const int owned_ID_begin,
    const int owned_ID_end, const MPI_Comm comm)
    : num_elements_(num_elements), num_local_basis_(num_local_basis),
      local_element_begin_(local_element_begin),
      local_element_end_(local_element_end),
      owned_ID_begin_(owned_ID_begin), owned_ID_end_(owned_ID_end), comm_(comm)
{
    // Validate the global topology and the nonoverlapping local work interval.
    if (num_local_basis_ <= 0)
        throw std::invalid_argument("IEN requires local basis functions");
    if (local_element_begin_ > local_element_end_ ||
        local_element_end_ > num_elements_)
        throw std::invalid_argument("Invalid local IEN element interval");
    if ((owned_ID_begin_ < 0) != (owned_ID_end_ < 0) ||
        (owned_ID_begin_ >= 0 && owned_ID_begin_ > owned_ID_end_))
        throw std::invalid_argument("Invalid local IEN ownership interval");
    if (comm_ == MPI_COMM_NULL)
        throw std::invalid_argument("IEN requires a valid MPI communicator");
}

template <int Dim>
BSplineIEN<Dim>::BSplineIEN(const BSplinePatch<Dim> &patch)
    : IEN(patch.num_elements(), bspline_num_local_basis(patch),
          0, patch.num_elements(), -1, -1, MPI_COMM_SELF),
      degrees_(patch_degrees(patch)), active_spans_(copy_active_spans(patch)),
      basis_ordering_(patch.basis_shape()),
      element_ordering_(high_element_shape(patch)),
      local_ordering_(local_bspline_shape<Dim>(degrees_))
{
}

template <int Dim>
BSplineIEN<Dim>::BSplineIEN(const BSplinePatch<Dim> &patch,
    const TensorSlabPartition<Dim> &partition)
    : IEN(patch.num_elements(), bspline_num_local_basis(patch),
          partition.high_element_begin(), partition.high_element_end(),
          partition.owned_ID_begin(), partition.owned_ID_end(),
          partition.communicator()),
      degrees_(patch_degrees(patch)), active_spans_(copy_active_spans(patch)),
      basis_ordering_(patch.basis_shape()),
      element_ordering_(high_element_shape(patch)),
      local_ordering_(local_bspline_shape<Dim>(degrees_))
{
    // Check that the partition was built for this exact high-order topology.
    if (partition.basis_shape() != basis_ordering_.shape() ||
        partition.high_element_shape() != element_ordering_.shape())
        throw std::invalid_argument("MPI partition does not match BSplineIEN");
}

template <int Dim>
int BSplineIEN<Dim>::get_IEN(
    const std::size_t element, const int local_basis) const
{
    // Validate logical indices before decoding their tensor components.
    if (element >= num_elements())
        throw std::out_of_range("Invalid element index in BSplineIEN");
    if (local_basis < 0 || local_basis >= num_local_basis())
        throw std::out_of_range("Invalid local basis index in BSplineIEN");

    // Decode the global element and local basis with direction 0 fastest.
    const auto element_multi_index =
        element_ordering_.multi_index(static_cast<int>(element));
    const auto local_multi_index = local_ordering_.multi_index(local_basis);

    // Convert each active knot span into the corresponding basis multi-index.
    std::array<int, Dim> basis_multi_index{};
    for (int direction = 0; direction < Dim; ++direction)
    {
        const int span =
            active_spans_[direction][element_multi_index[direction]];
        basis_multi_index[direction] = span - degrees_[direction] +
            local_multi_index[direction];
    }
    return basis_ordering_.global_index(basis_multi_index);
}

template <int Dim>
LORIEN<Dim>::LORIEN(const LORMesh<Dim> &mesh)
    : IEN(mesh.num_cells(), 1 << Dim, 0, mesh.num_cells(), -1, -1,
          MPI_COMM_SELF),
      node_ordering_(mesh.node_shape()),
      cell_ordering_(lor_cell_shape(mesh)),
      corner_ordering_(q1_corner_shape<Dim>())
{
}

template <int Dim>
LORIEN<Dim>::LORIEN(const LORMesh<Dim> &mesh,
    const TensorSlabPartition<Dim> &partition)
    : IEN(mesh.num_cells(), 1 << Dim,
          partition.lor_cell_begin(), partition.lor_cell_end(),
          partition.owned_ID_begin(), partition.owned_ID_end(),
          partition.communicator()),
      node_ordering_(mesh.node_shape()),
      cell_ordering_(lor_cell_shape(mesh)),
      corner_ordering_(q1_corner_shape<Dim>())
{
    // Check that the partition's LOR topology matches this mesh exactly.
    if (partition.basis_shape() != node_ordering_.shape() ||
        partition.lor_cell_shape() != cell_ordering_.shape())
        throw std::invalid_argument("MPI partition does not match LORIEN");
}

template <int Dim>
int LORIEN<Dim>::get_IEN(
    const std::size_t element, const int local_basis) const
{
    // Validate the global cell and conventional local corner indices.
    if (element >= num_elements())
        throw std::out_of_range("Invalid cell index in LORIEN");
    if (local_basis < 0 || local_basis >= num_local_basis())
        throw std::out_of_range("Invalid local corner index in LORIEN");

    // Decode the cell and map conventional Quad4/Hex8 corner order to the
    // direction-0-fastest binary tensor corner.
    constexpr std::array<int, 8> conventional_to_tensor{
        0, 1, 3, 2, 4, 5, 7, 6};
    const auto cell_multi_index =
        cell_ordering_.multi_index(static_cast<int>(element));
    const auto corner_multi_index = corner_ordering_.multi_index(
        conventional_to_tensor[local_basis]);

    // Add the binary corner offset and return the LOR node global_index.
    std::array<int, Dim> node_multi_index{};
    for (int direction = 0; direction < Dim; ++direction)
        node_multi_index[direction] = cell_multi_index[direction] +
            corner_multi_index[direction];
    return node_ordering_.global_index(node_multi_index);
}

// Emit all supported dimensions for both on-demand tensor IEN types.
template class BSplineIEN<2>;
template class BSplineIEN<3>;
template class LORIEN<2>;
template class LORIEN<3>;

} // namespace lor
