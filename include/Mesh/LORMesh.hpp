#ifndef LOR_MESH_LOR_MESH_HPP
#define LOR_MESH_LOR_MESH_HPP

#include "Mesh/BSplinePatch.hpp"
#include "Mesh/LORPointSet.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace lor
{

template <int Dim>
class LORMesh
{
    static_assert(Dim == 2 || Dim == 3, "LORMesh supports only 2D and 3D");

public:
    using Point = std::array<double, Dim>;
    using Index = std::array<int, Dim>;
    using ParaPoints = typename LORPointSet<Dim>::Values;

    // Construct the tensor grid from a validated parameter point set, then
    // evaluate the spline geometry at all LOR nodes.
    LORMesh(const BSplinePatch<Dim> &spline_patch,
        const LORPointSet<Dim> &point_set);

    // Return the immutable parameter nodes in one direction.
    const std::vector<double> &para_nodes(int direction) const;
    Index node_shape() const noexcept { return node_shape_; }
    std::size_t num_nodes() const noexcept { return phy_nodes_.size(); }
    std::size_t num_cells() const noexcept;
    const Point &phy_node(std::size_t node) const;

private:
    ParaPoints para_nodes_;
    Index node_shape_{};
    std::vector<Point> phy_nodes_;
};

extern template class LORMesh<2>;
extern template class LORMesh<3>;

} // namespace lor

#endif

