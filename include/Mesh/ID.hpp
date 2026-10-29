#ifndef LOR_MESH_ID_HPP
#define LOR_MESH_ID_HPP

#include <vector>

namespace lor
{

// Immutable global_index-to-ID map. DirichletBC supplies constrained
// global_index values; their ID entries are -1 and every remaining entry is
// numbered consecutively for global matrix and vector assembly.
class ID
{
public:
    ID(int num_global_indices,
        const std::vector<int> &constrained_global_indices);

    // Return ID[global_index]. A Dirichlet degree of freedom returns -1,
    // matching the ID convention used by MixPERIGEE.
    int get_ID(int global_index) const;

    // Return whether global_index was supplied as a Dirichlet constraint.
    bool is_dirichlet(int global_index) const;

    // Return the global_index associated with one unconstrained ID value.
    int global_index(int ID_value) const;

    // Return the complete number of global indices and unconstrained IDs.
    int num_global_indices() const noexcept
    {
        return static_cast<int>(values_.size());
    }
    int num_free_dofs() const noexcept { return num_free_dofs_; }

private:
    int num_free_dofs_ = 0;
    std::vector<int> values_;
    std::vector<int> ID_to_global_index_;
};

} // namespace lor

#endif
