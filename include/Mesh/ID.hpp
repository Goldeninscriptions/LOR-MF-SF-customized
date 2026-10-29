#ifndef LOR_MESH_ID_HPP
#define LOR_MESH_ID_HPP

#include <array>
#include <vector>

namespace lor
{

// Immutable global-basis-to-equation map. For the current all-Dirichlet
// problem, boundary entries are -1 and interior entries follow the paper's
// direction-0-fastest lexicographic ordering.
template <int Dim>
class ID
{
    static_assert(Dim == 2 || Dim == 3, "ID supports only 2D and 3D");

public:
    using Index = std::array<int, Dim>;

    explicit ID(const Index &global_shape);

    // Return ID[A], the algebraic equation index of global basis A. A
    // Dirichlet basis returns -1, matching MixPERIGEE's ID convention.
    int get_ID(int global_basis) const;

    // Return the equation index associated with a global tensor index.
    int get_ID(const Index &global_tensor_index) const;

    // Return the full global and unconstrained algebraic sizes.
    int num_global_basis() const noexcept
    {
        return static_cast<int>(values_.size());
    }
    int num_free_dofs() const noexcept { return num_free_dofs_; }

    // Return true when a global tensor index lies on a Dirichlet boundary.
    bool is_dirichlet(const Index &global_tensor_index) const;

    // Apply the inverse lexicographic map from an equation index to the full
    // tensor index and flat global basis index.
    Index global_tensor_index(int equation) const;
    int global_basis(const Index &global_tensor_index) const;

private:
    Index global_shape_{};
    Index free_shape_{};
    Index free_strides_{};
    int num_free_dofs_ = 0;
    std::vector<int> values_;

    Index unflatten_global(int global_basis) const;
    void validate_global_tensor_index(const Index &index) const;
};

extern template class ID<2>;
extern template class ID<3>;

} // namespace lor

#endif
