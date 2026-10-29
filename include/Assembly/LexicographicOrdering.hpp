#ifndef LOR_ASSEMBLY_LEXICOGRAPHIC_ORDERING_HPP
#define LOR_ASSEMBLY_LEXICOGRAPHIC_ORDERING_HPP

#include <array>

namespace lor
{

// Own the single tensor-index ordering convention used throughout the code:
// parameter direction 0 varies fastest, followed by directions 1 and 2.
template <int Dim>
class LexicographicOrdering
{
    static_assert(Dim == 2 || Dim == 3,
        "LexicographicOrdering supports only 2D and 3D");

public:
    using Index = std::array<int, Dim>;

    // Construct an immutable ordering for a positive tensor-product shape.
    explicit LexicographicOrdering(const Index &shape);

    // Map a tensor multi_index to the corresponding global_index.
    int global_index(const Index &multi_index) const;

    // Map a global_index back to the corresponding tensor multi_index.
    Index multi_index(int global_index) const;

    // Return the directional shape and total number of tensor entries.
    const Index &shape() const noexcept { return shape_; }
    int size() const noexcept { return size_; }

private:
    Index shape_{};
    Index strides_{};
    int size_ = 0;
};

extern template class LexicographicOrdering<2>;
extern template class LexicographicOrdering<3>;

} // namespace lor

#endif
