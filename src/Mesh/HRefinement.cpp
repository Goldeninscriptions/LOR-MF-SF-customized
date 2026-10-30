#include "Mesh/HRefinement.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace lor
{
namespace
{

// Return whether a knot vector already contains a value within a domain-scaled
// tolerance. Multiplicity is irrelevant because h-refinement adds only new
// breakpoints and never increases existing multiplicities.
bool contains_knot(const std::vector<double> &knots, const double value,
    const double tolerance)
{
    return std::any_of(knots.begin(), knots.end(),
        [value, tolerance](const double knot) {
            return std::abs(knot - value) <= tolerance;
        });
}

} // namespace

template <int Dim>
HRefinement<Dim>::HRefinement(const Index &target_elements)
    : target_elements_(target_elements)
{
    // Validate all independent directional targets when the refinement policy
    // is created, before it is applied to a specific geometry.
    for (const int count : target_elements_)
        if (count < 1)
            throw std::invalid_argument(
                "H-refinement element counts must be positive");
}

template <int Dim>
BSplinePatch<Dim> HRefinement<Dim>::apply(
    const BSplinePatch<Dim> &patch) const
{
    // Copy the input because each Boehm insertion updates knot and control
    // arrays together while the caller's base geometry must remain reusable.
    BSplinePatch<Dim> refined = patch;

    // Refine every direction independently while retaining its own knot range
    // and target element count.
    for (int direction = 0; direction < Dim; ++direction)
    {
        const int current_elements = patch.num_elements(direction);
        const int target_elements = target_elements_[direction];
        if (target_elements < current_elements)
            throw std::invalid_argument(
                "H-refinement cannot reduce the number of elements");

        // Define the requested uniform final grid over the original parameter
        // domain and use a scale-aware comparison tolerance.
        const int degree = patch.degree(direction);
        const auto &old_knots = patch.knots(direction);
        const double lower = old_knots[degree];
        const double upper = old_knots[patch.num_basis(direction)];
        const double tolerance =
            1.0e-12 * std::max(1.0, std::abs(upper - lower));
        std::vector<double> target_breakpoints;
        target_breakpoints.reserve(target_elements + 1);
        for (int index = 0; index <= target_elements; ++index)
            target_breakpoints.push_back(lower + (upper - lower) *
                static_cast<double>(index) / target_elements);

        // Require every existing non-endpoint breakpoint to occur in the
        // target grid. This is the condition that makes the new spline space
        // a true knot-insertion refinement of the old one.
        for (std::size_t knot_index = 0; knot_index < old_knots.size();)
        {
            const double old_breakpoint = old_knots[knot_index];
            while (knot_index < old_knots.size() &&
                   std::abs(old_knots[knot_index] - old_breakpoint) <= tolerance)
                ++knot_index;
            if (old_breakpoint <= lower + tolerance ||
                old_breakpoint >= upper - tolerance)
                continue;
            if (!contains_knot(
                    target_breakpoints, old_breakpoint, tolerance))
                throw std::invalid_argument(
                    "H-refinement target grid does not contain every "
                    "existing breakpoint");
        }

        // Insert each missing interior target breakpoint exactly once. The
        // Patch's Boehm update increases basis/control-point counts and
        // preserves the physical geometry after every insertion.
        for (int index = 1; index < target_elements; ++index)
        {
            const double knot = target_breakpoints[index];
            if (!contains_knot(refined.knots(direction), knot, tolerance))
                refined.insert_knot(direction, knot);
        }

        // Verify the requested final count rather than silently accepting a
        // nonuniform or non-nested result.
        if (refined.num_elements(direction) != target_elements)
            throw std::runtime_error(
                "H-refinement did not produce the requested element count");
    }
    return refined;
}

template class HRefinement<2>;
template class HRefinement<3>;

} // namespace lor
