#include "Mesh/PRefinement.hpp"

#include "Assembly/LexicographicOrdering.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace lor
{
namespace
{

// Evaluate a binomial coefficient in long-double arithmetic. Algorithm A5.9
// uses only ratios and products of these coefficients to elevate each Bezier
// segment, so this avoids the integer overflow of factorial-based formulas.
long double binomial_coefficient(const int n, int k)
{
    if (k < 0 || k > n)
        return 0.0L;
    k = std::min(k, n - k);
    long double value = 1.0L;
    for (int factor = 1; factor <= k; ++factor)
        value *= static_cast<long double>(n - k + factor) / factor;
    return value;
}

// Form an affine combination without relying on Point arithmetic operators.
// Both degree elevation and the built-in knot-removal stage of Algorithm A5.9
// repeatedly apply this same componentwise operation.
template <int Dim>
typename BSplinePatch<Dim>::Point affine_combine(
    const double alpha,
    const typename BSplinePatch<Dim>::Point &left,
    const typename BSplinePatch<Dim>::Point &right)
{
    typename BSplinePatch<Dim>::Point value{};
    for (int component = 0; component < Dim; ++component)
        value[component] = alpha * left[component] +
            (1.0 - alpha) * right[component];
    return value;
}

// Return the knot vector and control polygon produced by the standard B-spline
// curve degree-elevation algorithm (The NURBS Book, Algorithm A5.9). The
// algorithm elevates Bezier segments and removes the temporary decomposition
// knots, thereby increasing each old multiplicity by exactly degree_increment
// and preserving the original continuity and curve.
template <int Dim>
struct ElevatedCurve
{
    std::vector<double> knots;
    std::vector<typename BSplinePatch<Dim>::Point> control_points;
};

template <int Dim>
ElevatedCurve<Dim> elevate_curve(
    const std::vector<double> &old_knots,
    const int old_degree,
    const std::vector<typename BSplinePatch<Dim>::Point> &old_points,
    const int degree_increment)
{
    using Point = typename BSplinePatch<Dim>::Point;

    // Validate the one-dimensional curve definition before the index-heavy
    // standard algorithm accesses its endpoint and active-span entries.
    const int old_count = static_cast<int>(old_points.size());
    if (degree_increment < 1)
        throw std::invalid_argument(
            "Curve degree increment must be positive");
    if (static_cast<int>(old_knots.size()) !=
        old_count + old_degree + 1)
        throw std::invalid_argument(
            "Curve knot/control-point counts are inconsistent");

    // Precompute the Bezier degree-elevation coefficients
    // C(p,j) C(t,i-j) / C(p+t,i) used by every nonzero knot span.
    const int elevated_degree = old_degree + degree_increment;
    const int half_elevated_degree = elevated_degree / 2;
    std::vector<std::vector<double>> bezier_coefficients(
        elevated_degree + 1,
        std::vector<double>(old_degree + 1, 0.0));
    bezier_coefficients[0][0] = 1.0;
    bezier_coefficients[elevated_degree][old_degree] = 1.0;
    for (int elevated = 1;
         elevated <= half_elevated_degree; ++elevated)
    {
        const long double inverse = 1.0L /
            binomial_coefficient(elevated_degree, elevated);
        const int first = std::max(0, elevated - degree_increment);
        const int last = std::min(old_degree, elevated);
        for (int old = first; old <= last; ++old)
            bezier_coefficients[elevated][old] =
                static_cast<double>(inverse *
                    binomial_coefficient(old_degree, old) *
                    binomial_coefficient(
                        degree_increment, elevated - old));
    }
    for (int elevated = half_elevated_degree + 1;
         elevated < elevated_degree; ++elevated)
    {
        const int first = std::max(0, elevated - degree_increment);
        const int last = std::min(old_degree, elevated);
        for (int old = first; old <= last; ++old)
            bezier_coefficients[elevated][old] =
                bezier_coefficients[elevated_degree - elevated]
                    [old_degree - old];
    }

    // Initialize the elevated curve at the left endpoint and load the first
    // old-degree control polygon segment into the Bezier work array.
    ElevatedCurve<Dim> result;
    const int last_old_knot = static_cast<int>(old_knots.size()) - 1;
    const double first_knot = old_knots.front();
    result.control_points.push_back(old_points.front());
    result.knots.assign(elevated_degree + 1, first_knot);
    std::vector<Point> bezier_points(old_degree + 1);
    std::copy_n(old_points.begin(), old_degree + 1,
        bezier_points.begin());

    // These work arrays hold knot-insertion coefficients, the elevated
    // Bezier segment, and the overlap passed to the next old knot span.
    std::vector<double> alphas(std::max(0, old_degree - 1), 0.0);
    std::vector<Point> elevated_bezier(elevated_degree + 1);
    std::vector<Point> next_bezier(std::max(0, old_degree - 1));
    int output_knot_index = elevated_degree + 1;
    int output_point_index = 1;
    int previous_continuity = -1;
    int left_knot_index = old_degree;
    int right_knot_index = old_degree + 1;
    double left_knot = first_knot;

    // Process one distinct right knot at a time. Temporary knot insertion
    // converts the current B-spline span to Bezier form before elevation.
    while (right_knot_index < last_old_knot)
    {
        const int first_equal_knot = right_knot_index;
        while (right_knot_index < last_old_knot &&
               old_knots[right_knot_index] ==
                   old_knots[right_knot_index + 1])
            ++right_knot_index;
        const int multiplicity =
            right_knot_index - first_equal_knot + 1;
        const double right_knot = old_knots[right_knot_index];
        const int old_previous_continuity = previous_continuity;
        previous_continuity = old_degree - multiplicity;

        // Determine which elevated Bezier points are new after accounting for
        // continuity shared with the previous and next curve segments.
        const int first_elevated = old_previous_continuity > 0 ?
            (old_previous_continuity + 2) / 2 : 1;
        const int last_elevated = previous_continuity > 0 ?
            elevated_degree - (previous_continuity + 1) / 2 :
            elevated_degree;

        // Insert the right knot until the old-degree segment is Bezier and
        // retain the overlap control points required by the next segment.
        if (previous_continuity > 0)
        {
            const double numerator = right_knot - left_knot;
            for (int old = old_degree; old > multiplicity; --old)
                alphas[old - multiplicity - 1] = numerator /
                    (old_knots[left_knot_index + old] - left_knot);
            for (int insertion = 1;
                 insertion <= previous_continuity; ++insertion)
            {
                const int saved = previous_continuity - insertion;
                const int first_changed = multiplicity + insertion;
                for (int old = old_degree;
                     old >= first_changed; --old)
                    bezier_points[old] = affine_combine<Dim>(
                        alphas[old - first_changed],
                        bezier_points[old], bezier_points[old - 1]);
                next_bezier[saved] = bezier_points[old_degree];
            }
        }

        // Elevate the complete Bezier segment with the exact binomial
        // coefficient transformation.
        for (int elevated = first_elevated;
             elevated <= elevated_degree; ++elevated)
        {
            elevated_bezier[elevated].fill(0.0);
            const int first_old =
                std::max(0, elevated - degree_increment);
            const int last_old = std::min(old_degree, elevated);
            for (int old = first_old; old <= last_old; ++old)
                for (int component = 0; component < Dim; ++component)
                    elevated_bezier[elevated][component] +=
                        bezier_coefficients[elevated][old] *
                        bezier_points[old][component];
        }

        // Remove the temporary decomposition knots shared with the preceding
        // elevated segment. This restores the original continuity rather than
        // leaving a merely C0 chain of elevated Bezier pieces.
        if (old_previous_continuity > 1)
        {
            int first = output_knot_index - 2;
            int last = output_knot_index;
            const double denominator = right_knot - left_knot;
            const double beta =
                (right_knot - result.knots[output_knot_index - 1]) /
                denominator;
            for (int removal = 1;
                 removal < old_previous_continuity; ++removal)
            {
                int left = first;
                int right = last;
                int elevated_index =
                    right - output_knot_index + 1;
                while (right - left > removal)
                {
                    if (left < output_point_index)
                    {
                        const double alpha =
                            (right_knot - result.knots[left]) /
                            (left_knot - result.knots[left]);
                        result.control_points[left] =
                            affine_combine<Dim>(alpha,
                                result.control_points[left],
                                result.control_points[left - 1]);
                    }
                    if (right >= first_elevated)
                    {
                        double gamma = beta;
                        if (right - removal <=
                            output_knot_index - elevated_degree +
                                old_previous_continuity)
                            gamma =
                                (right_knot -
                                    result.knots[right - removal]) /
                                denominator;
                        elevated_bezier[elevated_index] =
                            affine_combine<Dim>(gamma,
                                elevated_bezier[elevated_index],
                                elevated_bezier[elevated_index + 1]);
                    }
                    ++left;
                    --right;
                    --elevated_index;
                }
                --first;
                ++last;
            }
        }

        // Append the left-knot multiplicity and the non-overlapping elevated
        // control points belonging to the completed segment.
        if (left_knot_index != old_degree)
            for (int repeat = 0;
                 repeat < elevated_degree - old_previous_continuity;
                 ++repeat)
            {
                result.knots.push_back(left_knot);
                ++output_knot_index;
            }
        for (int elevated = first_elevated;
             elevated <= last_elevated; ++elevated)
        {
            result.control_points.push_back(elevated_bezier[elevated]);
            ++output_point_index;
        }

        // Load the next old-degree segment or close the elevated curve with
        // the required clamped endpoint multiplicity.
        if (right_knot_index < last_old_knot)
        {
            for (int old = 0; old < previous_continuity; ++old)
                bezier_points[old] = next_bezier[old];
            for (int old = previous_continuity;
                 old <= old_degree; ++old)
                bezier_points[old] = old_points[
                    right_knot_index - old_degree + old];
            left_knot_index = right_knot_index;
            ++right_knot_index;
            left_knot = right_knot;
        }
        else
            result.knots.insert(result.knots.end(),
                elevated_degree + 1, right_knot);
    }

    // Check the defining B-spline size identity before returning the standard
    // algorithm's new knot vector and control polygon.
    if (static_cast<int>(result.knots.size()) !=
        static_cast<int>(result.control_points.size()) +
            elevated_degree + 1)
        throw std::runtime_error(
            "Degree elevation produced inconsistent knot/control-point counts");
    return result;
}

// Elevate every tensor fiber in one parameter direction and place the new
// control points back into direction-0-fastest global storage.
template <int Dim>
BSplinePatch<Dim> elevate_direction(const BSplinePatch<Dim> &patch,
    const int direction, const int degree_increment)
{
    using Index = typename BSplinePatch<Dim>::Index;
    using Point = typename BSplinePatch<Dim>::Point;

    // Build the old fiber traversal; setting the active direction extent to
    // one enumerates every independent curve exactly once.
    const Index old_shape = patch.basis_shape();
    const LexicographicOrdering<Dim> old_ordering(old_shape);
    Index fiber_shape = old_shape;
    fiber_shape[direction] = 1;
    const LexicographicOrdering<Dim> fiber_ordering(fiber_shape);

    // Elevate the first fiber to determine the common new knot vector and
    // directional control-point count shared by all tensor fibers.
    auto gather_fiber = [&](const Index &fixed_index) {
        std::vector<Point> line(old_shape[direction]);
        Index index = fixed_index;
        for (int entry = 0; entry < old_shape[direction]; ++entry)
        {
            index[direction] = entry;
            line[entry] =
                patch.control_point(old_ordering.global_index(index));
        }
        return line;
    };
    const Index first_fixed = fiber_ordering.multi_index(0);
    ElevatedCurve<Dim> first_curve = elevate_curve<Dim>(
        patch.knots(direction), patch.degree(direction),
        gather_fiber(first_fixed), degree_increment);

    // Allocate the elevated tensor grid and scatter the first transformed
    // curve before processing the remaining independent fibers.
    Index new_shape = old_shape;
    new_shape[direction] =
        static_cast<int>(first_curve.control_points.size());
    const LexicographicOrdering<Dim> new_ordering(new_shape);
    std::vector<Point> new_points(new_ordering.size());
    auto scatter_fiber = [&](const Index &fixed_index,
                             const std::vector<Point> &line) {
        Index index = fixed_index;
        for (int entry = 0; entry < new_shape[direction]; ++entry)
        {
            index[direction] = entry;
            new_points[new_ordering.global_index(index)] = line[entry];
        }
    };
    scatter_fiber(first_fixed, first_curve.control_points);

    // Apply the identical standard curve elevation to every remaining fiber.
    for (int fiber = 1; fiber < fiber_ordering.size(); ++fiber)
    {
        const Index fixed_index = fiber_ordering.multi_index(fiber);
        const ElevatedCurve<Dim> curve = elevate_curve<Dim>(
            patch.knots(direction), patch.degree(direction),
            gather_fiber(fixed_index), degree_increment);
        if (curve.knots != first_curve.knots ||
            curve.control_points.size() !=
                first_curve.control_points.size())
            throw std::runtime_error(
                "Tensor fibers produced inconsistent elevated spaces");
        scatter_fiber(fixed_index, curve.control_points);
    }

    // Preserve untouched directions and replace only the elevated direction's
    // degree, knot vector, and tensor control-point extent.
    Index new_degrees{};
    std::array<std::vector<double>, Dim> new_knots;
    for (int current_direction = 0;
         current_direction < Dim; ++current_direction)
    {
        new_degrees[current_direction] = patch.degree(current_direction);
        new_knots[current_direction] = patch.knots(current_direction);
    }
    new_degrees[direction] += degree_increment;
    new_knots[direction] = std::move(first_curve.knots);
    return BSplinePatch<Dim>(new_degrees, new_knots, new_points);
}

} // namespace

template <int Dim>
PRefinement<Dim>::PRefinement(const int target_degree)
    : target_degree_(target_degree)
{
    // Reject invalid targets at configuration time rather than after a tensor
    // collection of curve elevations has already been performed.
    if (target_degree_ < 1)
        throw std::invalid_argument(
            "P-refinement target degree must be positive");
}

template <int Dim>
BSplinePatch<Dim> PRefinement<Dim>::apply(
    const BSplinePatch<Dim> &patch) const
{
    // Enforce the paper's one-common-degree convention and reject degree
    // reduction, which is not a nested p-refinement operation.
    const int old_degree = patch.degree(0);
    for (int direction = 1; direction < Dim; ++direction)
        if (patch.degree(direction) != old_degree)
            throw std::invalid_argument(
                "P-refinement requires one common input degree");
    if (target_degree_ < old_degree)
        throw std::invalid_argument(
            "P-refinement cannot reduce the polynomial degree");
    if (target_degree_ == old_degree)
        return patch;

    // Elevate each tensor direction independently with Algorithm A5.9. The
    // common increment preserves each direction's distinct knot vector while
    // producing the common target degree required by the paper.
    const int degree_increment = target_degree_ - old_degree;
    BSplinePatch<Dim> refined = patch;
    for (int direction = 0; direction < Dim; ++direction)
        refined = elevate_direction<Dim>(
            refined, direction, degree_increment);
    return refined;
}

template class PRefinement<2>;
template class PRefinement<3>;

} // namespace lor
