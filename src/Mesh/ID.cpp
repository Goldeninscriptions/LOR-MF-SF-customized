#include "Mesh/ID.hpp"

#include <stdexcept>

namespace lor
{

ID::ID(const int num_global_indices,
    const std::vector<int> &constrained_global_indices)
{
    // Validate the complete global-index range before allocating ID storage.
    if (num_global_indices <= 0)
        throw std::invalid_argument(
            "ID requires a positive number of global indices");

    // Mark the constrained global indices supplied by the BC module.
    std::vector<bool> constrained(num_global_indices, false);
    for (const int global_index : constrained_global_indices)
    {
        if (global_index < 0 || global_index >= num_global_indices)
            throw std::out_of_range(
                "Constrained global_index is outside the ID range");
        constrained[global_index] = true;
    }

    // Number unconstrained IDs in increasing global-index order and retain
    // the inverse ID-to-global_index map.
    values_.assign(num_global_indices, -1);
    ID_to_global_index_.reserve(num_global_indices);
    for (int global_index = 0;
         global_index < num_global_indices; ++global_index)
    {
        if (constrained[global_index])
            continue;
        values_[global_index] = num_free_dofs_++;
        ID_to_global_index_.push_back(global_index);
    }
}

int ID::get_ID(const int global_index) const
{
    // Return the stored ID value, including -1 on a Dirichlet boundary.
    if (global_index < 0 ||
        global_index >= static_cast<int>(values_.size()))
        throw std::out_of_range("Invalid global_index in ID");
    return values_[global_index];
}

bool ID::is_dirichlet(const int global_index) const
{
    // Query the stored ID table without introducing any tensor-index logic.
    return get_ID(global_index) < 0;
}

int ID::global_index(const int ID_value) const
{
    // Apply the inverse ID map only for unconstrained algebraic unknowns.
    if (ID_value < 0 || ID_value >= num_free_dofs_)
        throw std::out_of_range("Invalid unconstrained ID value");
    return ID_to_global_index_[ID_value];
}

} // namespace lor
