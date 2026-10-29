#include "Mesh/IEN.hpp"

#include "Mesh/BSplinePatch.hpp"
#include "Mesh/LORMesh.hpp"

#include <array>
#include <stdexcept>
#include <utility>

namespace lor
{
namespace
{

// Recover a tensor multi-index from a direction-0-fastest flat index.
template <int Dim>
std::array<int, Dim> unflatten(
    int flat, const std::array<int, Dim> &shape)
{
    std::array<int, Dim> index{};
    for (int direction = 0; direction < Dim; ++direction)
    {
        index[direction] = flat % shape[direction];
        flat /= shape[direction];
    }
    return index;
}

// Build the complete element-major IEN table for a B-spline patch.
template <int Dim>
std::vector<int> build_bspline_IEN(const BSplinePatch<Dim> &patch)
{
    std::array<int, Dim> local_shape{};
    int num_local_basis = 1;
    for (int direction = 0; direction < Dim; ++direction)
    {
        local_shape[direction] = patch.degree(direction) + 1;
        num_local_basis *= local_shape[direction];
    }

    std::vector<int> values(
        patch.num_elements() * static_cast<std::size_t>(num_local_basis));
    for (std::size_t element = 0; element < patch.num_elements(); ++element)
    {
        const auto spans = patch.element_spans(element);
        for (int local = 0; local < num_local_basis; ++local)
        {
            const auto local_index = unflatten<Dim>(local, local_shape);
            typename BSplinePatch<Dim>::Index global_index{};
            for (int direction = 0; direction < Dim; ++direction)
                global_index[direction] = spans[direction] -
                    patch.degree(direction) + local_index[direction];
            values[element * static_cast<std::size_t>(num_local_basis) +
                static_cast<std::size_t>(local)] =
                    patch.flatten_basis_index(global_index);
        }
    }
    return values;
}

// Build the complete element-major IEN table for the Q1 LOR mesh.
template <int Dim>
std::vector<int> build_lor_IEN(const LORMesh<Dim> &mesh)
{
    const auto node_shape = mesh.node_shape();
    typename LORMesh<Dim>::Index cell_shape{};
    for (int direction = 0; direction < Dim; ++direction)
        cell_shape[direction] = node_shape[direction] - 1;

    constexpr int num_local_basis = 1 << Dim;
    constexpr std::array<int, 8> conventional_to_tensor{
        0, 1, 3, 2, 4, 5, 7, 6};
    std::array<int, Dim> corner_shape{};
    corner_shape.fill(2);
    std::vector<int> values(
        mesh.num_cells() * static_cast<std::size_t>(num_local_basis));

    for (std::size_t cell = 0; cell < mesh.num_cells(); ++cell)
    {
        const auto cell_index =
            unflatten<Dim>(static_cast<int>(cell), cell_shape);
        for (int local = 0; local < num_local_basis; ++local)
        {
            // Convert the conventional Quad4/Hex8 local node order to the
            // direction-0-fastest binary corner index used by the tensor grid.
            // Quad4: 0,1,3,2. Hex8: 0,1,3,2,4,5,7,6.
            const int tensor_corner = conventional_to_tensor[local];
            const auto corner =
                unflatten<Dim>(tensor_corner, corner_shape);
            typename LORMesh<Dim>::Index node_index{};
            for (int direction = 0; direction < Dim; ++direction)
                node_index[direction] =
                    cell_index[direction] + corner[direction];

            int global_node = node_index[0];
            int stride = node_shape[0];
            for (int direction = 1; direction < Dim; ++direction)
            {
                global_node += node_index[direction] * stride;
                stride *= node_shape[direction];
            }
            values[cell * static_cast<std::size_t>(num_local_basis) +
                static_cast<std::size_t>(local)] = global_node;
        }
    }
    return values;
}

// Compute the number of local tensor-product B-spline basis functions.
template <int Dim>
int bspline_num_local_basis(const BSplinePatch<Dim> &patch)
{
    int count = 1;
    for (int direction = 0; direction < Dim; ++direction)
        count *= patch.degree(direction) + 1;
    return count;
}

} // namespace

IEN::IEN(const std::size_t num_elements, const int num_local_basis,
    std::vector<int> values)
    : num_elements_(num_elements), num_local_basis_(num_local_basis),
      values_(std::move(values))
{
    // Reject malformed tables at their construction boundary.
    if (num_local_basis_ <= 0)
        throw std::invalid_argument("IEN requires local basis functions");
    if (values_.size() !=
        num_elements_ * static_cast<std::size_t>(num_local_basis_))
        throw std::invalid_argument("IEN table size is inconsistent");
}

int IEN::get_IEN(
    const std::size_t element, const int local_basis) const
{
    // Validate both logical indices before accessing the flat table.
    if (element >= num_elements_)
        throw std::out_of_range("Invalid element index in IEN");
    if (local_basis < 0 || local_basis >= num_local_basis_)
        throw std::out_of_range("Invalid local basis index in IEN");
    return values_.at(element * static_cast<std::size_t>(num_local_basis_) +
        static_cast<std::size_t>(local_basis));
}

template <int Dim>
BSplineIEN<Dim>::BSplineIEN(const BSplinePatch<Dim> &patch)
    : IEN(patch.num_elements(), bspline_num_local_basis(patch),
          build_bspline_IEN(patch))
{
}

template <int Dim>
LORIEN<Dim>::LORIEN(const LORMesh<Dim> &mesh)
    : IEN(mesh.num_cells(), 1 << Dim, build_lor_IEN(mesh))
{
}

template class BSplineIEN<2>;
template class BSplineIEN<3>;
template class LORIEN<2>;
template class LORIEN<3>;

} // namespace lor
