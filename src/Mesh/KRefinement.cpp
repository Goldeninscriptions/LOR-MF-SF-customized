#include "Mesh/KRefinement.hpp"

#include "Mesh/HRefinement.hpp"
#include "Mesh/PRefinement.hpp"

#include <stdexcept>

namespace lor
{

template <int Dim>
KRefinement<Dim>::KRefinement(const int target_degree,
    const Index &target_elements)
    : target_degree_(target_degree), target_elements_(target_elements)
{
    // Validate both parts of the composed policy at construction time.
    if (target_degree_ < 1)
        throw std::invalid_argument("K-refinement target degree must be positive");
    for (const int count : target_elements_)
        if (count < 1)
            throw std::invalid_argument(
                "K-refinement element counts must be positive");
}

template <int Dim>
BSplinePatch<Dim> KRefinement<Dim>::apply(
    const BSplinePatch<Dim> &patch) const
{
    // Elevate first so existing breakpoints preserve their original
    // continuity while all subsequently inserted breakpoints receive the
    // maximum continuity associated with target_degree_.
    const PRefinement<Dim> p_refinement(target_degree_);
    BSplinePatch<Dim> refined = p_refinement.apply(patch);

    // Insert the compatible target grid into the already elevated space.
    const HRefinement<Dim> h_refinement(target_elements_);
    return h_refinement.apply(refined);
}

template class KRefinement<2>;
template class KRefinement<3>;

} // namespace lor
