#ifndef LOR_MESH_H_REFINEMENT_HPP
#define LOR_MESH_H_REFINEMENT_HPP

#include "Mesh/BSplinePatch.hpp"

namespace lor
{

// Exact tensor-product h-refinement by Boehm knot insertion. Polynomial
// degree and existing knot multiplicities remain unchanged; only new simple
// knots are inserted, so geometry is preserved and the old space is nested in
// the returned space.
template <int Dim>
class HRefinement
{
public:
    using Index = typename BSplinePatch<Dim>::Index;

    // Store the desired final number of nonzero knot spans independently in
    // every direction. The target grid must contain all existing breakpoints.
    explicit HRefinement(const Index &target_elements);

    // Return an independently owned knot-inserted Patch. Incompatible target
    // grids are rejected instead of silently rebuilding a non-nested space.
    BSplinePatch<Dim> apply(const BSplinePatch<Dim> &patch) const;

    // Return the requested final tensor element counts n_i.
    const Index &target_elements() const noexcept { return target_elements_; }

private:
    // Final nonzero-span counts. Directional entries may differ even though
    // the polynomial degree is common across directions.
    Index target_elements_{};
};

extern template class HRefinement<2>;
extern template class HRefinement<3>;

} // namespace lor

#endif
