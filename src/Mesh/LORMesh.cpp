#include "Mesh/LORMesh.hpp"

#include <stdexcept>

namespace lor
{

template <int Dim>
LORMesh<Dim>::LORMesh(const BSplinePatch<Dim> &spline_patch,
    const LORPointSet<Dim> &point_set)
    : para_nodes_(point_set.values()), node_shape_(point_set.shape())
{
    // Evaluate the complete tensor point set in one batch. This reuses every
    // directional B-spline evaluation across the other directions.
    phy_nodes_ =
        spline_patch.evaluate_geometry_grid(para_nodes_);
}

template <int Dim>
const std::vector<double> &LORMesh<Dim>::para_nodes(
    const int direction) const
{
    // Validate the direction before exposing the copied parameter sequence.
    if (direction < 0 || direction >= Dim)
        throw std::out_of_range("Invalid LOR mesh direction");
    return para_nodes_[direction];
}

template <int Dim>
std::size_t LORMesh<Dim>::num_cells() const noexcept
{
    std::size_t count = 1;
    for (int direction = 0; direction < Dim; ++direction)
        count *= static_cast<std::size_t>(node_shape_[direction] - 1);
    return count;
}

template <int Dim>
const typename LORMesh<Dim>::Point &LORMesh<Dim>::phy_node(
    const std::size_t node) const
{
    return phy_nodes_.at(node);
}

template class LORMesh<2>;
template class LORMesh<3>;

} // namespace lor

