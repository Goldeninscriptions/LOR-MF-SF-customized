#ifndef LOR_MESH_K_REFINEMENT_HPP
#define LOR_MESH_K_REFINEMENT_HPP

#include "Mesh/BSplinePatch.hpp"

namespace lor
{

// Classical IGA k-refinement: exact degree elevation is performed first,
// preserving old-knot continuity, and compatible simple knots are inserted
// afterwards. New breakpoints therefore receive the maximum continuity of the
// elevated degree.
template <int Dim>
class KRefinement
{
public:
    using Index = typename BSplinePatch<Dim>::Index;

    // Store the common final degree and independent final element counts.
    KRefinement(int target_degree, const Index &target_elements);

    // Return PRefinement(target_degree) followed by
    // HRefinement(target_elements), leaving the input Patch unchanged.
    BSplinePatch<Dim> apply(const BSplinePatch<Dim> &patch) const;

    // Return the two final-space targets used by the composed algorithm.
    int target_degree() const noexcept { return target_degree_; }
    const Index &target_elements() const noexcept { return target_elements_; }

private:
    // Common elevated degree for all parameter directions.
    int target_degree_ = 0;

    // Final nonzero-span counts after post-elevation knot insertion.
    Index target_elements_{};
};

extern template class KRefinement<2>;
extern template class KRefinement<3>;

} // namespace lor

#endif
