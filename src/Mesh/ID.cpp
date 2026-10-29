#include "Mesh/ID.hpp"

#include <stdexcept>

namespace lor
{

template <int Dim>
ID<Dim>::ID(const Index &global_shape)
    : global_shape_(global_shape)
{
    // Validate the tensor shape and build the manuscript's free-index strides.
    int num_global_basis = 1;
    num_free_dofs_ = 1;
    for (int direction = 0; direction < Dim; ++direction)
    {
        if (global_shape_[direction] < 3)
            throw std::invalid_argument(
                "Each direction requires at least one interior basis index");
        free_shape_[direction] = global_shape_[direction] - 2;
        free_strides_[direction] = num_free_dofs_;
        num_global_basis *= global_shape_[direction];
        num_free_dofs_ *= free_shape_[direction];
    }

    // Construct ID once so every operator uses the same constraint and
    // lexicographic equation map without recomputing it.
    values_.resize(num_global_basis);
    for (int global = 0; global < num_global_basis; ++global)
    {
        const Index tensor = unflatten_global(global);
        if (is_dirichlet(tensor))
        {
            values_[global] = -1;
            continue;
        }

        int equation = 0;
        for (int direction = 0; direction < Dim; ++direction)
            equation += (tensor[direction] - 1) * free_strides_[direction];
        values_[global] = equation;
    }
}

template <int Dim>
void ID<Dim>::validate_global_tensor_index(const Index &index) const
{
    // Check all tensor components before applying a flattening formula.
    for (int direction = 0; direction < Dim; ++direction)
        if (index[direction] < 0 ||
            index[direction] >= global_shape_[direction])
            throw std::out_of_range("Invalid global tensor index in ID");
}

template <int Dim>
bool ID<Dim>::is_dirichlet(const Index &index) const
{
    // Any first or last directional layer belongs to the full Dirichlet
    // boundary used by the initial paper prototype.
    validate_global_tensor_index(index);
    for (int direction = 0; direction < Dim; ++direction)
        if (index[direction] == 0 ||
            index[direction] == global_shape_[direction] - 1)
            return true;
    return false;
}

template <int Dim>
typename ID<Dim>::Index ID<Dim>::unflatten_global(int global_basis_value) const
{
    // Invert the direction-0-fastest global basis numbering.
    if (global_basis_value < 0 ||
        global_basis_value >= static_cast<int>(values_.size()))
        throw std::out_of_range("Invalid global basis index in ID");
    Index tensor{};
    for (int direction = 0; direction < Dim; ++direction)
    {
        tensor[direction] =
            global_basis_value % global_shape_[direction];
        global_basis_value /= global_shape_[direction];
    }
    return tensor;
}

template <int Dim>
int ID<Dim>::get_ID(const int global_basis_value) const
{
    // Return the precomputed equation index, including -1 on the boundary.
    if (global_basis_value < 0 ||
        global_basis_value >= static_cast<int>(values_.size()))
        throw std::out_of_range("Invalid global basis index in ID");
    return values_[global_basis_value];
}

template <int Dim>
int ID<Dim>::get_ID(const Index &global_tensor_index_value) const
{
    // Compose tensor flattening with the stored ID array.
    return get_ID(global_basis(global_tensor_index_value));
}

template <int Dim>
typename ID<Dim>::Index ID<Dim>::global_tensor_index(
    const int equation) const
{
    // Apply the inverse paper ordering to recover the interior tensor index.
    if (equation < 0 || equation >= num_free_dofs_)
        throw std::out_of_range("Invalid algebraic equation index in ID");
    Index tensor{};
    for (int direction = 0; direction < Dim; ++direction)
        tensor[direction] = 1 +
            (equation / free_strides_[direction]) % free_shape_[direction];
    return tensor;
}

template <int Dim>
int ID<Dim>::global_basis(const Index &global_tensor_index_value) const
{
    // Flatten a full tensor index with direction 0 varying fastest.
    validate_global_tensor_index(global_tensor_index_value);
    int flat = global_tensor_index_value[0];
    int stride = global_shape_[0];
    for (int direction = 1; direction < Dim; ++direction)
    {
        flat += global_tensor_index_value[direction] * stride;
        stride *= global_shape_[direction];
    }
    return flat;
}

template class ID<2>;
template class ID<3>;

} // namespace lor
