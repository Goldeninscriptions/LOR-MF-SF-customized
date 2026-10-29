#include "Assembly/DirichletBC.hpp"

#include <algorithm>
#include <stdexcept>

namespace lor
{

template <int Dim>
DirichletBC<Dim>::DirichletBC(
    const BoundariesMultiIndices &boundaries_multi_indices,
    const LexicographicOrdering<Dim> &ordering)
    : num_global_indices_(ordering.size())
{
    // Allocate construction-only markers once. constrained_mask[g] is true
    // exactly when global_index g appears on at least one boundary, so it
    // forms the duplicate-free union of edges/faces. boundary_visit_id[g]
    // stores the last boundary ID that contained g; equality with the current
    // ID detects a duplicate inside one boundary while still allowing corners
    // or edges to belong to several different boundaries.
    std::vector<bool> constrained_mask(
        static_cast<std::size_t>(ordering.size()), false);
    std::vector<int> boundary_visit_id(
        static_cast<std::size_t>(ordering.size()), -1);

    // Validate and convert every independent edge/face while retaining its
    // boundary identity.
    for (int boundary_id = 0; boundary_id < 2 * Dim; ++boundary_id)
    {
        const auto &boundary_multi_indices =
            boundaries_multi_indices[boundary_id];
        const int direction = boundary_id / 2;
        const int fixed_index = boundary_id % 2 == 0 ?
            0 : ordering.shape()[direction] - 1;
        int expected_size = 1;
        for (int other_direction = 0;
             other_direction < Dim; ++other_direction)
            if (other_direction != direction)
                expected_size *= ordering.shape()[other_direction];
        if (static_cast<int>(boundary_multi_indices.size()) != expected_size)
            throw std::invalid_argument(
                "Dirichlet boundary multi-index set has an invalid size");

        BoundaryGlobalIndices &boundary_global_indices =
            boundaries_global_indices_[boundary_id];
        boundary_global_indices.reserve(boundary_multi_indices.size());
        for (const Index &multi_index : boundary_multi_indices)
        {
            if (multi_index[direction] != fixed_index)
                throw std::invalid_argument(
                    "Dirichlet boundary has an invalid fixed multi-index");
            const int global_index = ordering.global_index(multi_index);
            if (boundary_visit_id[global_index] == boundary_id)
                throw std::invalid_argument(
                    "Dirichlet boundary contains a duplicate multi_index");
            boundary_visit_id[global_index] = boundary_id;
            boundary_global_indices.push_back(global_index);
            constrained_mask[global_index] = true;
        }
    }

    // Build the sorted full-boundary union in global lexicographic order.
    constrained_global_indices_.reserve(
        static_cast<std::size_t>(num_global_indices_));
    for (std::size_t global_index = 0;
         global_index < constrained_mask.size(); ++global_index)
        if (constrained_mask[global_index])
            constrained_global_indices_.push_back(
                static_cast<int>(global_index));
}

template <int Dim>
const typename DirichletBC<Dim>::BoundaryGlobalIndices &
DirichletBC<Dim>::boundary_global_indices(const int boundary_id) const
{
    // Preserve the same lower/upper directional boundary convention as Patch.
    if (boundary_id < 0 || boundary_id >= 2 * Dim)
        throw std::out_of_range("Invalid Dirichlet boundary ID");
    return boundaries_global_indices_[boundary_id];
}

template <int Dim>
bool DirichletBC<Dim>::is_constrained(const int global_index) const
{
    // Query the canonical sorted union without retaining a duplicate mask.
    if (global_index < 0 || global_index >= num_global_indices_)
        throw std::out_of_range("Invalid global index in DirichletBC");
    return std::binary_search(constrained_global_indices_.begin(),
        constrained_global_indices_.end(), global_index);
}

template class DirichletBC<2>;
template class DirichletBC<3>;

} // namespace lor
