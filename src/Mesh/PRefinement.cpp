#include "Mesh/PRefinement.hpp"

#include "Assembly/LexicographicOrdering.hpp"
#include "Element/BSplineBasis1D.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace lor
{
namespace
{

// Build the degree-elevated knot vector. Increasing every distinct knot's
// multiplicity by delta preserves C^(p-m) continuity while changing p to
// p+delta; endpoint multiplicities consequently remain clamped.
std::vector<double> elevated_knots(const std::vector<double> &old_knots,
    const int degree_increment)
{
    std::vector<double> new_knots;
    for (std::size_t first = 0; first < old_knots.size();)
    {
        std::size_t end = first + 1;
        while (end < old_knots.size() && old_knots[end] == old_knots[first])
            ++end;
        const int old_multiplicity = static_cast<int>(end - first);
        new_knots.insert(new_knots.end(),
            old_multiplicity + degree_increment, old_knots[first]);
        first = end;
    }
    return new_knots;
}

// Compute one Greville abscissa per target basis function. These points form
// a nonsingular collocation grid for the open B-spline spaces used here.
std::vector<double> greville_points(const int degree,
    const std::vector<double> &knots)
{
    const int count = static_cast<int>(knots.size()) - degree - 1;
    std::vector<double> points(count, 0.0);
    for (int basis = 0; basis < count; ++basis)
        for (int offset = 1; offset <= degree; ++offset)
            points[basis] += knots[basis + offset] / degree;
    return points;
}

// Factor one directional collocation matrix in row-major storage using
// partial-pivoted LU. The factor is reused for every tensor line and physical
// coordinate component in that direction.
void factorize(std::vector<double> &matrix, std::vector<int> &pivots,
    const int size)
{
    pivots.resize(size);
    for (int column = 0; column < size; ++column)
    {
        // Select and apply the largest available pivot in this column.
        int pivot = column;
        for (int row = column + 1; row < size; ++row)
            if (std::abs(matrix[row * size + column]) >
                std::abs(matrix[pivot * size + column]))
                pivot = row;
        if (std::abs(matrix[pivot * size + column]) < 1.0e-14)
            throw std::runtime_error(
                "Degree-elevation collocation matrix is singular");
        pivots[column] = pivot;
        if (pivot != column)
            for (int entry = 0; entry < size; ++entry)
                std::swap(matrix[column * size + entry],
                    matrix[pivot * size + entry]);

        // Store the lower multipliers and update the remaining upper block.
        for (int row = column + 1; row < size; ++row)
        {
            matrix[row * size + column] /=
                matrix[column * size + column];
            for (int entry = column + 1; entry < size; ++entry)
                matrix[row * size + entry] -=
                    matrix[row * size + column] *
                    matrix[column * size + entry];
        }
    }
}

// Solve one collocation right-hand side using the stored row permutations and
// triangular factors. Values enter as geometry samples and leave as control
// coefficients for one tensor line.
void solve_factorized(const std::vector<double> &matrix,
    const std::vector<int> &pivots, std::vector<double> &values,
    const int size)
{
    // Apply pivot permutations and forward substitution through L.
    for (int column = 0; column < size; ++column)
    {
        if (pivots[column] != column)
            std::swap(values[column], values[pivots[column]]);
        for (int row = column + 1; row < size; ++row)
            values[row] -= matrix[row * size + column] * values[column];
    }

    // Apply backward substitution through U.
    for (int row = size - 1; row >= 0; --row)
    {
        for (int column = row + 1; column < size; ++column)
            values[row] -= matrix[row * size + column] * values[column];
        values[row] /= matrix[row * size + row];
    }
}

// Apply one inverse directional collocation matrix to every tensor fiber.
// Direction 0 remains the fastest-varying global tensor index.
template <int Dim>
void transform_direction(
    std::vector<typename BSplinePatch<Dim>::Point> &coefficients,
    const std::array<int, Dim> &shape, const int direction,
    const std::vector<double> &factor, const std::vector<int> &pivots)
{
    const LexicographicOrdering<Dim> ordering(shape);
    std::array<int, Dim> fiber_shape = shape;
    const int fiber_size = fiber_shape[direction];
    fiber_shape[direction] = 1;
    const LexicographicOrdering<Dim> fiber_ordering(fiber_shape);
    std::vector<double> values(fiber_size);

    // Solve each physical-coordinate component independently on every fiber.
    for (int fiber = 0; fiber < fiber_ordering.size(); ++fiber)
    {
        std::array<int, Dim> index = fiber_ordering.multi_index(fiber);
        for (int component = 0; component < Dim; ++component)
        {
            for (int entry = 0; entry < fiber_size; ++entry)
            {
                index[direction] = entry;
                values[entry] =
                    coefficients[ordering.global_index(index)][component];
            }
            solve_factorized(factor, pivots, values, fiber_size);
            for (int entry = 0; entry < fiber_size; ++entry)
            {
                index[direction] = entry;
                coefficients[ordering.global_index(index)][component] =
                    values[entry];
            }
        }
    }
}

} // namespace

template <int Dim>
PRefinement<Dim>::PRefinement(const int target_degree)
    : target_degree_(target_degree)
{
    // Reject invalid targets at configuration time rather than during a
    // potentially expensive tensor geometry transformation.
    if (target_degree_ < 1)
        throw std::invalid_argument("P-refinement target degree must be positive");
}

template <int Dim>
BSplinePatch<Dim> PRefinement<Dim>::apply(
    const BSplinePatch<Dim> &patch) const
{
    // Enforce the paper's one-common-degree convention on the input and target.
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

    // Increase every directional knot multiplicity by the same degree delta.
    const int degree_increment = target_degree_ - old_degree;
    typename BSplinePatch<Dim>::Index target_degrees{};
    target_degrees.fill(target_degree_);
    std::array<std::vector<double>, Dim> target_knots;
    std::array<std::vector<double>, Dim> collocation_points;
    std::array<int, Dim> target_shape{};
    for (int direction = 0; direction < Dim; ++direction)
    {
        target_knots[direction] = elevated_knots(
            patch.knots(direction), degree_increment);
        collocation_points[direction] = greville_points(
            target_degree_, target_knots[direction]);
        target_shape[direction] =
            static_cast<int>(collocation_points[direction].size());
    }

    // Sample the exactly representable old geometry on the target tensor grid.
    std::vector<typename BSplinePatch<Dim>::Point> target_control_points =
        patch.evaluate_geometry_grid(collocation_points);

    // Invert the tensor collocation operator one direction at a time to obtain
    // the unique degree-elevated control coefficients.
    for (int direction = 0; direction < Dim; ++direction)
    {
        const int count = target_shape[direction];
        const BSplineBasis1D basis(
            target_degree_, target_knots[direction]);
        std::vector<double> factor(count * count, 0.0);
        for (int row = 0; row < count; ++row)
        {
            const auto evaluation =
                basis.evaluate_N(collocation_points[direction][row]);
            for (int local = 0;
                 local < static_cast<int>(evaluation.N.size()); ++local)
                factor[row * count + evaluation.first_basis + local] =
                    evaluation.N[local];
        }
        std::vector<int> pivots;
        factorize(factor, pivots, count);
        transform_direction<Dim>(target_control_points, target_shape,
            direction, factor, pivots);
    }

    // Construct a new immutable-topology Patch with the elevated basis data.
    return BSplinePatch<Dim>(
        target_degrees, target_knots, target_control_points);
}

template class PRefinement<2>;
template class PRefinement<3>;

} // namespace lor
