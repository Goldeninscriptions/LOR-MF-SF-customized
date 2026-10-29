#include "Element/BSplineBasis1D.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace lor
{

// Store an independently owned spline definition for arbitrary-point
// evaluation. Owning the knot vector prevents dangling-reference lifetimes.
BSplineBasis1D::BSplineBasis1D(const int degree,
    const std::vector<double> &knots)
    : degree_(degree), knots_(knots)
{
    validate_definition(degree_, knots_);
}

// Build the optional Gauss value and first-derivative tables as part of
// construction, so callers cannot observe partially initialized tables.
BSplineBasis1D::BSplineBasis1D(const int degree,
    const std::vector<double> &knots,
    const std::vector<double> &gauss_points)
    : BSplineBasis1D(degree, knots)
{
    build_gauss_table(gauss_points);
}

// Evaluate the p + 1 active values at one arbitrary coordinate while also
// returning the global indexing information needed to identify them.
BSplineBasis1D::PointEvaluation BSplineBasis1D::evaluate_N(
    const double xi) const
{
    // Locate xi and request only the zeroth-derivative row from the general
    // private recurrence, avoiding derivative work during geometry sampling.
    PointEvaluation result;
    result.span = find_span(degree_, knots_, xi);
    result.first_basis = result.span - degree_;
    DerivativeValues values = evaluate_derivatives(
        degree_, knots_, xi, result.span, 0);

    // Move the computed row into the owning result without copying its
    // entries. Derivative rows remain an internal implementation path.
    result.N = std::move(values[0]);
    return result;
}

// Validate the invariant spline definition once before constructing tables.
void BSplineBasis1D::validate_definition(const int degree,
    const std::vector<double> &knots)
{
    if (degree < 0)
        throw std::invalid_argument("B-spline degree must be nonnegative");
    if (static_cast<int>(knots.size()) < 2 * degree + 2)
        throw std::invalid_argument("Knot vector is too short for the degree");
    if (!std::is_sorted(knots.begin(), knots.end()))
        throw std::invalid_argument("Knot vector must be nondecreasing");
}

// Derive the number of basis functions from m + 1 = n + p + 2.
int BSplineBasis1D::num_basis(const int degree,
    const std::vector<double> &knots) noexcept
{
    return static_cast<int>(knots.size()) - degree - 1;
}

// Locate the nonzero knot span containing the requested parameter.
int BSplineBasis1D::find_span(const int degree,
    const std::vector<double> &knots, const double parameter)
{
    // Establish the valid parametric domain and treat the right endpoint with
    // a scale-aware tolerance.
    const int n = num_basis(degree, knots) - 1;
    const double lower = knots[degree];
    const double upper = knots[n + 1];
    const double tolerance = 1.0e-13 * std::max(1.0, std::abs(upper - lower));
    if (parameter < lower - tolerance || parameter > upper + tolerance)
        throw std::out_of_range("Parameter lies outside the B-spline domain");
    if (parameter >= upper - tolerance)
        return n;

    // Apply the binary-search procedure from The NURBS Book, Algorithm A2.1.
    int low = degree;
    int high = n + 1;
    int middle = (low + high) / 2;
    while (parameter < knots[middle] || parameter >= knots[middle + 1])
    {
        if (parameter < knots[middle])
            high = middle;
        else
            low = middle;
        middle = (low + high) / 2;
    }
    return middle;
}

// Evaluate all active basis functions and their derivatives through the
// requested order using The NURBS Book, Algorithm A2.3.
BSplineBasis1D::DerivativeValues BSplineBasis1D::evaluate_derivatives(
    const int degree, const std::vector<double> &knots,
    const double parameter, const int span,
    const int maximum_derivative)
{
    // Validate the caller-supplied span and derivative order before indexing
    // the knot vector or allocating work arrays.
    const int p = degree;
    if (span < p || span >= num_basis(degree, knots))
        throw std::out_of_range("Invalid B-spline span");
    if (maximum_derivative < 0)
        throw std::invalid_argument("Maximum derivative must be nonnegative");

    // Allocate the triangular ndu table and the left/right knot-distance
    // workspaces used by the Cox-de Boor recurrence.
    std::vector<double> ndu((p + 1) * (p + 1), 0.0);
    std::vector<double> left(p + 1, 0.0);
    std::vector<double> right(p + 1, 0.0);
    const auto at = [p, &ndu](const int row, const int column) -> double &
    {
        return ndu[row * (p + 1) + column];
    };

    // Fill the triangular table. Its last column contains the p-degree basis
    // values, while its lower triangle stores denominators reused by the
    // derivative recurrence.
    at(0, 0) = 1.0;
    for (int j = 1; j <= p; ++j)
    {
        left[j] = parameter - knots[span + 1 - j];
        right[j] = knots[span + j] - parameter;
        double saved = 0.0;
        for (int r = 0; r < j; ++r)
        {
            at(j, r) = right[r + 1] + left[j - r];
            const double temporary = at(r, j - 1) / at(j, r);
            at(r, j) = saved + right[r + 1] * temporary;
            saved = left[j - r] * temporary;
        }
        at(j, j) = saved;
    }

    // Allocate the public result shape [derivative_order][local_basis] and
    // copy the zeroth derivatives (the basis values) from ndu.
    DerivativeValues derivatives(maximum_derivative + 1,
        std::vector<double>(p + 1, 0.0));
    for (int j = 0; j <= p; ++j)
        derivatives[0][j] = at(j, p);

    // Build the unscaled derivative coefficients for every active basis. Only
    // orders up to p need computation; all higher rows remain exactly zero.
    std::vector<double> a(2 * (p + 1), 0.0);
    for (int r = 0; r <= p; ++r)
    {
        std::fill(a.begin(), a.end(), 0.0);
        int s1 = 0;
        int s2 = 1;
        a[0] = 1.0;
        const int computed_derivatives = std::min(maximum_derivative, p);
        for (int order = 1; order <= computed_derivatives; ++order)
        {
            double value = 0.0;
            const int rk = r - order;
            const int pk = p - order;

            // Add the left-edge term when the derivative stencil does not
            // extend below the first active basis function.
            if (r >= order)
            {
                a[s2 * (p + 1)] =
                    a[s1 * (p + 1)] / at(pk + 1, rk);
                value = a[s2 * (p + 1)] * at(rk, pk);
            }

            // Accumulate the interior terms of the alternating difference
            // stencil for the current derivative order.
            const int first = (rk >= -1) ? 1 : -rk;
            const int last = (r - 1 <= pk) ? order - 1 : p - r;
            for (int j = first; j <= last; ++j)
            {
                a[s2 * (p + 1) + j] =
                    (a[s1 * (p + 1) + j] -
                        a[s1 * (p + 1) + j - 1]) /
                    at(pk + 1, rk + j);
                value += a[s2 * (p + 1) + j] * at(rk + j, pk);
            }

            // Add the right-edge term when it remains inside the active
            // degree-p basis range.
            if (r <= pk)
            {
                a[s2 * (p + 1) + order] =
                    -a[s1 * (p + 1) + order - 1] / at(pk + 1, r);
                value += a[s2 * (p + 1) + order] * at(r, pk);
            }

            // Save the unscaled derivative and swap the two rolling rows for
            // the next derivative order.
            derivatives[order][r] = value;
            std::swap(s1, s2);
        }
    }

    // Multiply each derivative row by p!/(p-k)!, completing Algorithm A2.3.
    int factor = p;
    for (int order = 1;
         order <= std::min(maximum_derivative, p); ++order)
    {
        for (int local_basis = 0; local_basis <= p; ++local_basis)
            derivatives[order][local_basis] *= factor;
        factor *= p - order;
    }
    return derivatives;
}

// Precompute zeroth- and first-derivative tables for all nonempty knot spans
// at a caller-selected set of reference points.
void BSplineBasis1D::build_gauss_table(
    const std::vector<double> &gauss_points)
{
    // Validate that the supplied points form a usable reference-element rule.
    if (gauss_points.empty())
        throw std::invalid_argument("Basis table requires at least one point");
    for (const double point : gauss_points)
        if (point < -1.0 || point > 1.0)
            throw std::out_of_range(
                "Basis table reference points must lie in [-1, 1]");

    // Record the quadrature extent. The first table dimension uses the actual
    // knot-span index, so callers need no private span-to-element map.
    const int number_of_table_points =
        static_cast<int>(gauss_points.size());

    // Allocate the two internally owned arrays in
    // [knot_span][point][local_basis] order, with local_basis as the contiguous
    // dimension. Unused spans remain zero.
    const std::size_t table_size =
        static_cast<std::size_t>(num_basis(degree_, knots_)) *
        number_of_table_points * (degree_ + 1);
    N_Gauss_.assign(table_size, 0.0);
    N_der1_Gauss_.assign(table_size, 0.0);

    // Map each reference point to its physical knot interval, evaluate the
    // recurrence once, and copy value/first-derivative rows into the tables.
    for (int span = degree_;
         span < num_basis(degree_, knots_); ++span)
    {
        // Ignore repeated-knot intervals because they have zero measure and
        // are not mesh elements; their allocated entries remain zero.
        if (!(knots_[span + 1] > knots_[span]))
            continue;

        // Cache the knot interval used to map all reference points for this
        // active span.
        const double left = knots_[span];
        const double right = knots_[span + 1];
        for (int point = 0; point < number_of_table_points; ++point)
        {
            // Map the point from [-1,1] into the current knot interval and
            // evaluate only the two derivative rows stored by this table.
            const double parameter = 0.5 *
                ((right - left) * gauss_points[point] + right + left);
            const DerivativeValues values =
                evaluate_derivatives(degree_, knots_, parameter, span, 1);
            for (int local_basis = 0;
                 local_basis <= degree_; ++local_basis)
            {
                // Copy one local basis entry into the common flattened layout
                // shared by N_Gauss and N_der1_Gauss.
                const std::size_t offset =
                    table_offset(degree_, number_of_table_points,
                        span, point, local_basis);
                N_Gauss_[offset] = values[0][local_basis];
                N_der1_Gauss_[offset] = values[1][local_basis];
            }
        }
    }
}

// Convert the internal knot-span and tensor-table indices into the common
// flattened array offset.
std::size_t BSplineBasis1D::table_offset(
    const int degree, const int number_of_table_points,
    const int span, const int point, const int local_basis) noexcept
{
    // Flatten [knot_span][quadrature_point][local_basis], with the local
    // basis index varying fastest for contiguous operator-kernel access.
    return (static_cast<std::size_t>(span) *
        number_of_table_points + point) * (degree + 1) + local_basis;
}

} // namespace lor

