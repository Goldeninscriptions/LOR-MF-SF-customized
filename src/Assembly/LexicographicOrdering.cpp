#include "Assembly/LexicographicOrdering.hpp"

#include <limits>
#include <stdexcept>

namespace lor
{

template <int Dim>
LexicographicOrdering<Dim>::LexicographicOrdering(const Index &shape)
    : shape_(shape)
{
    // Build direction-0-fastest strides while validating every tensor extent.
    int stride = 1;
    for (int direction = 0; direction < Dim; ++direction)
    {
        if (shape_[direction] <= 0)
            throw std::invalid_argument(
                "Lexicographic tensor extents must be positive");
        strides_[direction] = stride;
        if (stride > std::numeric_limits<int>::max() / shape_[direction])
            throw std::overflow_error(
                "Lexicographic tensor size exceeds integer range");
        stride *= shape_[direction];
    }

    // Store the validated total size for inverse-map range checking.
    size_ = stride;
}

template <int Dim>
int LexicographicOrdering<Dim>::global_index(const Index &multi_index) const
{
    // Validate each tensor component and accumulate its stride contribution.
    int global_index_value = 0;
    for (int direction = 0; direction < Dim; ++direction)
    {
        if (multi_index[direction] < 0 ||
            multi_index[direction] >= shape_[direction])
            throw std::out_of_range(
                "Tensor index is outside the lexicographic shape");
        global_index_value +=
            multi_index[direction] * strides_[direction];
    }
    return global_index_value;
}

template <int Dim>
typename LexicographicOrdering<Dim>::Index
LexicographicOrdering<Dim>::multi_index(int global_index_value) const
{
    // Validate the global_index before recovering every tensor component.
    if (global_index_value < 0 || global_index_value >= size_)
        throw std::out_of_range(
            "Global index is outside the lexicographic shape");
    Index multi_index_value{};
    for (int direction = Dim - 1; direction >= 0; --direction)
    {
        multi_index_value[direction] =
            global_index_value / strides_[direction];
        global_index_value %= strides_[direction];
    }
    return multi_index_value;
}

template class LexicographicOrdering<2>;
template class LexicographicOrdering<3>;

} // namespace lor
