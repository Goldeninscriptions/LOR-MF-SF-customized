#include "Mesh/Refinement.hpp"

#include "Mesh/HRefinement.hpp"
#include "Mesh/KRefinement.hpp"
#include "Mesh/PRefinement.hpp"

#include <stdexcept>

namespace lor
{

template <int Dim>
BSplinePatch<Dim> apply_refinement(const BSplinePatch<Dim> &geometry,
    const RefinementRequest<Dim> &request)
{
    // Enforce the paper's common-degree assumption before dispatching to a
    // concrete refinement algorithm.
    const int initial_degree = geometry.degree(0);
    for (int direction = 1; direction < Dim; ++direction)
        if (geometry.degree(direction) != initial_degree)
            throw std::invalid_argument(
                "The paper configuration requires one common degree in all directions");

    // Pure p-refinement changes only the common degree and must preserve the
    // complete original element partition.
    if (request.mode == RefinementMode::p_refine)
    {
        for (int direction = 0; direction < Dim; ++direction)
            if (request.target_elements[direction] !=
                geometry.num_elements(direction))
                throw std::invalid_argument(
                    "p_refine requires target elements to equal the geometry counts");
        return PRefinement<Dim>(request.target_degree).apply(geometry);
    }

    // Pure h-refinement inserts knots while retaining the geometry's common
    // polynomial degree.
    if (request.mode == RefinementMode::h_refine)
    {
        if (request.target_degree != initial_degree)
            throw std::invalid_argument(
                "h_refine requires target degree to equal the geometry degree");
        return HRefinement<Dim>(request.target_elements).apply(geometry);
    }

    // Classical k-refinement elevates degree first and inserts new knots into
    // the elevated space second.
    return KRefinement<Dim>(request.target_degree,
        request.target_elements).apply(geometry);
}

// Emit the two supported paper dimensions from this implementation unit.
template BSplinePatch<2> apply_refinement<2>(
    const BSplinePatch<2> &, const RefinementRequest<2> &);
template BSplinePatch<3> apply_refinement<3>(
    const BSplinePatch<3> &, const RefinementRequest<3> &);

} // namespace lor
