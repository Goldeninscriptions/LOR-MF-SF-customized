#ifndef LOR_MESH_P_REFINEMENT_HPP
#define LOR_MESH_P_REFINEMENT_HPP

#include "Mesh/BSplinePatch.hpp"

namespace lor
{

// Exact tensor-product B-spline degree elevation. Every parameter direction
// is elevated to the same target degree required by the paper. Interior knot
// multiplicities grow with the degree increment, preserving the original
// continuity, element partition, spline space nesting, and physical geometry.
template <int Dim>
class PRefinement
{
public:
    // Store the common final polynomial degree. It must be no smaller than
    // the common degree of the input Patch.
    explicit PRefinement(int target_degree);

    // Return an independently owned degree-elevated Patch. The input remains
    // unchanged and can be reused to construct other refinement cases.
    BSplinePatch<Dim> apply(const BSplinePatch<Dim> &patch) const;

    // Return the common degree requested for every parameter direction.
    int target_degree() const noexcept { return target_degree_; }

private:
    // Common final degree used to increase each distinct knot multiplicity by
    // target_degree-current_degree during the exact basis transformation.
    int target_degree_ = 0;
};

extern template class PRefinement<2>;
extern template class PRefinement<3>;

} // namespace lor

#endif
