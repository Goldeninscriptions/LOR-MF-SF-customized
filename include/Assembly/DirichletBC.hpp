#ifndef LOR_ASSEMBLY_DIRICHLET_BC_HPP
#define LOR_ASSEMBLY_DIRICHLET_BC_HPP

#include "Assembly/LexicographicOrdering.hpp"

#include <array>
#include <vector>

namespace lor
{

// Convert tensor-product boundary multi_index sets into immutable global_index
// sets used by ID and global assembly. The current paper prototype
// applies homogeneous Dirichlet conditions on every patch boundary.
template <int Dim>
class DirichletBC
{
    static_assert(Dim == 2 || Dim == 3, "DirichletBC supports only 2D and 3D");

public:
    using Index = std::array<int, Dim>;
    using BoundaryMultiIndices = std::vector<Index>;
    using BoundariesMultiIndices =
        std::array<BoundaryMultiIndices, 2 * Dim>;
    using BoundaryGlobalIndices = std::vector<int>;
    using BoundariesGlobalIndices =
        std::array<BoundaryGlobalIndices, 2 * Dim>;

    // Use an externally constructed ordering to validate all four 2D edges
    // or six 3D faces and convert their multi_index values to global_index.
    DirichletBC(const BoundariesMultiIndices &boundaries_multi_indices,
        const LexicographicOrdering<Dim> &ordering);

    // Return the complete number of global_index values covered by the
    // external ordering used during construction.
    int num_global_indices() const noexcept { return num_global_indices_; }

    // Return global_index values on one independently identified edge/face.
    // Boundary IDs follow BSplinePatch: 2*d is lower and 2*d+1 is upper.
    const BoundaryGlobalIndices &boundary_global_indices(
        int boundary_id) const;

    // Return the sorted, duplicate-free global_index values constrained by
    // the full homogeneous Dirichlet condition.
    const std::vector<int> &constrained_global_indices() const noexcept
    {
        return constrained_global_indices_;
    }

    // Return true if a global_index is constrained.
    bool is_constrained(int global_index) const;

private:
    int num_global_indices_ = 0;
    BoundariesGlobalIndices boundaries_global_indices_;
    std::vector<int> constrained_global_indices_;
};

extern template class DirichletBC<2>;
extern template class DirichletBC<3>;

} // namespace lor

#endif
