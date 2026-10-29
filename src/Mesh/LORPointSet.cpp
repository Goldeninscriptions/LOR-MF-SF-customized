#include "Mesh/LORPointSet.hpp"

#include "Element/BSplineBasis1D.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace lor
{
namespace
{

// Compute one directional Greville sequence from a degree and knot vector.
std::vector<double> build_greville_1d(
    const int degree, const std::vector<double> &knots)
{
    if (degree < 1)
        throw std::invalid_argument(
            "Greville and Demko points require degree at least one");

    // Average the p knots following each B-spline coefficient index.
    const int count = static_cast<int>(knots.size()) - degree - 1;
    std::vector<double> points(count, 0.0);
    for (int basis = 0; basis < count; ++basis)
        for (int offset = 1; offset <= degree; ++offset)
            points[basis] += knots[basis + offset] / degree;
    return points;
}

// Generate one Greville sequence independently in every parameter direction.
template <int Dim>
typename LORPointSet<Dim>::Values build_greville_points(
    const BSplinePatch<Dim> &patch)
{
    typename LORPointSet<Dim>::Values values;

    // Directions may share a degree but always use their own knot vectors.
    for (int direction = 0; direction < Dim; ++direction)
        values[direction] = build_greville_1d(
            patch.degree(direction), patch.knots(direction));
    return values;
}

// Evaluate a spline represented in the same B-spline basis as the patch.
double evaluate_spline(const BSplineBasis1D &basis,
    const std::vector<double> &coefficients, const double xi)
{
    const BSplineBasis1D::PointEvaluation evaluation = basis.evaluate_N(xi);
    double value = 0.0;
    for (std::size_t local = 0; local < evaluation.N.size(); ++local)
    {
        const int global = evaluation.first_basis +
            static_cast<int>(local);
        value += coefficients.at(static_cast<std::size_t>(global)) *
            evaluation.N[local];
    }
    return value;
}

// Solve a dense collocation system by partial-pivoted Gaussian elimination.
std::vector<double> solve_dense(std::vector<double> matrix,
    std::vector<double> rhs, const double pivot_tolerance)
{
    const int size = static_cast<int>(rhs.size());

    // Factor and eliminate one column at a time with row pivoting.
    for (int column = 0; column < size; ++column)
    {
        int pivot = column;
        double pivot_size = std::abs(
            matrix[column * size + column]);
        for (int row = column + 1; row < size; ++row)
        {
            const double candidate = std::abs(
                matrix[row * size + column]);
            if (candidate > pivot_size)
            {
                pivot = row;
                pivot_size = candidate;
            }
        }
        if (!(pivot_size > pivot_tolerance))
            throw std::runtime_error(
                "Demko collocation matrix is singular or ill-conditioned");

        if (pivot != column)
        {
            for (int entry = column; entry < size; ++entry)
                std::swap(matrix[column * size + entry],
                    matrix[pivot * size + entry]);
            std::swap(rhs[column], rhs[pivot]);
        }

        for (int row = column + 1; row < size; ++row)
        {
            const double factor = matrix[row * size + column] /
                matrix[column * size + column];
            matrix[row * size + column] = 0.0;
            for (int entry = column + 1; entry < size; ++entry)
                matrix[row * size + entry] -= factor *
                    matrix[column * size + entry];
            rhs[row] -= factor * rhs[column];
        }
    }

    // Recover the coefficient vector by backward substitution.
    std::vector<double> solution(size, 0.0);
    for (int row = size - 1; row >= 0; --row)
    {
        double value = rhs[row];
        for (int column = row + 1; column < size; ++column)
            value -= matrix[row * size + column] * solution[column];
        solution[row] = value / matrix[row * size + row];
    }
    return solution;
}

// Interpolate alternating +1/-1 data at the current Remez points.
std::vector<double> alternating_interpolant(
    const BSplineBasis1D &basis, const std::vector<double> &points,
    const double pivot_tolerance)
{
    const int size = static_cast<int>(points.size());
    std::vector<double> matrix(size * size, 0.0);
    std::vector<double> rhs(size, 0.0);

    // Assemble only the locally supported B-spline entries in each row.
    for (int row = 0; row < size; ++row)
    {
        const auto evaluation = basis.evaluate_N(points[row]);
        for (std::size_t local = 0; local < evaluation.N.size(); ++local)
        {
            const int column = evaluation.first_basis +
                static_cast<int>(local);
            matrix[row * size + column] = evaluation.N[local];
        }
        rhs[row] = row % 2 == 0 ? 1.0 : -1.0;
    }
    return solve_dense(
        std::move(matrix), std::move(rhs), pivot_tolerance);
}

// Maximize sign*s(x) on an interval while treating every internal knot as an
// explicit candidate. Dense span sampling brackets all practical Q1-paper
// degrees; golden-section refinement then removes sampling-position error.
std::pair<double, double> maximize_signed_spline(
    const BSplineBasis1D &basis, const std::vector<double> &coefficients,
    const std::vector<double> &knots, const int degree,
    const double left, const double right, const double sign,
    const DemkoOptions &options)
{
    std::vector<double> samples{left, right};
    const int samples_per_span = std::max(
        options.samples_per_span, 4 * degree + 1);

    // Sample each nonzero knot span separately so knot-side extrema cannot be
    // skipped by a grid crossing a reduced-continuity location.
    const int num_basis = static_cast<int>(coefficients.size());
    for (int span = degree; span < num_basis; ++span)
    {
        const double span_left = std::max(left, knots[span]);
        const double span_right = std::min(right, knots[span + 1]);
        if (!(span_right > span_left))
            continue;
        for (int sample = 0; sample <= samples_per_span; ++sample)
            samples.push_back(span_left + (span_right - span_left) *
                static_cast<double>(sample) / samples_per_span);
    }
    std::sort(samples.begin(), samples.end());
    samples.erase(std::unique(samples.begin(), samples.end(),
        [](const double first, const double second)
        {
            return std::abs(first - second) <= 1.0e-14 *
                std::max({1.0, std::abs(first), std::abs(second)});
        }), samples.end());

    // Evaluate the signed objective and retain the best explicit candidate.
    std::vector<double> values(samples.size(), 0.0);
    std::size_t best = 0;
    for (std::size_t sample = 0; sample < samples.size(); ++sample)
    {
        values[sample] = sign * evaluate_spline(
            basis, coefficients, samples[sample]);
        if (values[sample] > values[best])
            best = sample;
    }
    double best_point = samples[best];
    double best_value = values[best];

    // Refine every sampled local maximum, not merely the largest grid value.
    constexpr double inverse_phi = 0.6180339887498948482;
    for (std::size_t sample = 1; sample + 1 < samples.size(); ++sample)
    {
        if (values[sample] < values[sample - 1] ||
            values[sample] < values[sample + 1])
            continue;

        double a = samples[sample - 1];
        double b = samples[sample + 1];
        double c = b - inverse_phi * (b - a);
        double d = a + inverse_phi * (b - a);
        double fc = sign * evaluate_spline(basis, coefficients, c);
        double fd = sign * evaluate_spline(basis, coefficients, d);
        for (int iteration = 0;
             iteration < options.extremum_iterations; ++iteration)
        {
            if (fc < fd)
            {
                a = c;
                c = d;
                fc = fd;
                d = a + inverse_phi * (b - a);
                fd = sign * evaluate_spline(basis, coefficients, d);
            }
            else
            {
                b = d;
                d = c;
                fd = fc;
                c = b - inverse_phi * (b - a);
                fc = sign * evaluate_spline(basis, coefficients, c);
            }
        }
        const double point = 0.5 * (a + b);
        const double value = sign * evaluate_spline(
            basis, coefficients, point);
        if (value > best_value)
        {
            best_point = point;
            best_value = value;
        }
    }
    return {best_point, best_value};
}

// Compute the one-dimensional Chebyshev--Demko abscissae by Remez iteration.
std::vector<double> build_demko_1d(const int degree,
    const std::vector<double> &knots, const DemkoOptions &options)
{
    if (options.max_iterations < 1 || options.samples_per_span < 2 ||
        options.extremum_iterations < 1 ||
        !(options.equioscillation_tolerance > 0.0) ||
        !(options.point_tolerance > 0.0) ||
        !(options.pivot_tolerance > 0.0))
        throw std::invalid_argument("Invalid Demko iteration options");

    // Greville points provide a Schoenberg--Whitney-compatible initial set.
    std::vector<double> points = build_greville_1d(degree, knots);
    const int count = static_cast<int>(points.size());
    const double lower = knots[degree];
    const double upper = knots[count];
    const double scale = std::max(1.0, upper - lower);
    const BSplineBasis1D basis(degree, knots);
    double final_residual = std::numeric_limits<double>::infinity();
    double final_step = std::numeric_limits<double>::infinity();

    // Alternate interpolation and extremum extraction until equioscillation.
    for (int iteration = 0; iteration < options.max_iterations; ++iteration)
    {
        const std::vector<double> coefficients = alternating_interpolant(
            basis, points, options.pivot_tolerance);
        std::vector<double> next = points;
        next.front() = lower;
        next.back() = upper;

        // Each old point defines an ordered Remez cell bounded by neighboring
        // midpoints. Maximize the required alternating sign inside that cell.
        for (int point = 1; point < count - 1; ++point)
        {
            const double left = 0.5 * (points[point - 1] + points[point]);
            const double right = 0.5 * (points[point] + points[point + 1]);
            const double sign = point % 2 == 0 ? 1.0 : -1.0;
            next[point] = maximize_signed_spline(basis, coefficients,
                knots, degree, left, right, sign, options).first;
        }

        // Measure movement and equioscillation on the newly extracted extrema.
        double minimum_amplitude = std::numeric_limits<double>::max();
        double maximum_amplitude = 0.0;
        final_step = 0.0;
        for (int point = 0; point < count; ++point)
        {
            if (point > 0 && !(next[point] > next[point - 1]))
                throw std::runtime_error(
                    "Demko Remez update lost strict point ordering");
            const double value = evaluate_spline(
                basis, coefficients, next[point]);
            const double sign = point % 2 == 0 ? 1.0 : -1.0;
            if (!(sign * value > 0.0))
                throw std::runtime_error(
                    "Demko extrema lost the alternating sign pattern");
            const double amplitude = std::abs(value);
            minimum_amplitude = std::min(minimum_amplitude, amplitude);
            maximum_amplitude = std::max(maximum_amplitude, amplitude);
            final_step = std::max(final_step,
                std::abs(next[point] - points[point]));
        }
        final_residual = (maximum_amplitude - minimum_amplitude) /
            maximum_amplitude;

        // Independently scan the full domain to reject a missed larger peak.
        const double positive_max = maximize_signed_spline(basis,
            coefficients, knots, degree, lower, upper, 1.0, options).second;
        const double negative_max = maximize_signed_spline(basis,
            coefficients, knots, degree, lower, upper, -1.0, options).second;
        const double global_max = std::max(positive_max, negative_max);
        const bool global_check = global_max <= maximum_amplitude *
            (1.0 + 10.0 * options.equioscillation_tolerance);

        points = std::move(next);
        const bool point_stable =
            final_step <= options.point_tolerance * scale;
        const bool residual_strong = final_residual <=
            0.1 * options.equioscillation_tolerance;
        if (final_residual <= options.equioscillation_tolerance &&
            global_check && (point_stable || residual_strong))
            return points;
    }

    // Report both relevant failure diagnostics instead of returning an
    // unverified approximation as if it were a Demko point set.
    std::ostringstream message;
    message << "Demko Remez iteration did not converge: residual="
            << final_residual << ", point_step=" << final_step;
    throw std::runtime_error(message.str());
}

// Apply the univariate Demko construction independently in every direction.
template <int Dim>
typename LORPointSet<Dim>::Values build_demko_points(
    const BSplinePatch<Dim> &patch, const DemkoOptions &options)
{
    typename LORPointSet<Dim>::Values values;

    // Run a separate univariate Remez construction for each directional knot
    // vector, even when all directions use the same polynomial degree.
    for (int direction = 0; direction < Dim; ++direction)
        values[direction] = build_demko_1d(patch.degree(direction),
            patch.knots(direction), options);
    return values;
}

} // namespace

template <int Dim>
LORPointSet<Dim>::LORPointSet(
    const BSplinePatch<Dim> &patch, Values values)
    : values_(std::move(values))
{
    // Validate every directional sequence against the patch parameter domain.
    for (int direction = 0; direction < Dim; ++direction)
    {
        if (values_[direction].size() < 2)
            throw std::invalid_argument(
                "Each LOR direction requires at least two points");

        const int degree = patch.degree(direction);
        const double lower = patch.knots(direction)[degree];
        const double upper =
            patch.knots(direction)[patch.num_basis(direction)];
        if (values_[direction].front() < lower ||
            values_[direction].back() > upper)
            throw std::invalid_argument(
                "LOR points must lie inside the patch parameter domain");

        // Strict ordering prevents zero-width or overlapping LOR cells.
        for (std::size_t point = 1;
             point < values_[direction].size(); ++point)
            if (!(values_[direction][point] >
                  values_[direction][point - 1]))
                throw std::invalid_argument(
                    "LOR points must be strictly increasing");

        shape_[direction] =
            static_cast<int>(values_[direction].size());
    }
}

template <int Dim>
const std::vector<double> &LORPointSet<Dim>::points(
    const int direction) const
{
    // Reject an invalid direction before exposing the immutable sequence.
    if (direction < 0 || direction >= Dim)
        throw std::out_of_range("Invalid LOR point-set direction");
    return values_[direction];
}

template <int Dim>
GrevillePointSet<Dim>::GrevillePointSet(const BSplinePatch<Dim> &patch)
    : LORPointSet<Dim>(patch, build_greville_points(patch))
{
}

template <int Dim>
DemkoPointSet<Dim>::DemkoPointSet(const BSplinePatch<Dim> &patch,
    const DemkoOptions &options)
    : LORPointSet<Dim>(patch, build_demko_points(patch, options))
{
}

template class LORPointSet<2>;
template class LORPointSet<3>;
template class GrevillePointSet<2>;
template class GrevillePointSet<3>;
template class DemkoPointSet<2>;
template class DemkoPointSet<3>;

} // namespace lor
