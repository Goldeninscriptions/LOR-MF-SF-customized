#include "Mesh/BSplinePatch.hpp"

#include "Assembly/LexicographicOrdering.hpp"
#include "Element/BSplineBasis1D.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace lor
{
namespace
{

template <int Dim>
std::size_t product(const std::array<int, Dim> &shape)
{
    std::size_t value = 1;
    for (const int extent : shape)
        value *= static_cast<std::size_t>(extent);
    return value;
}

} // namespace

template <int Dim>
BSplinePatch<Dim>::BSplinePatch(const Index &degrees,
    const std::array<std::vector<double>, Dim> &knots,
    const std::vector<Point> &control_points)
    : degrees_(degrees), knots_(knots), control_points_(control_points)
{
    rebuild_topology();
    validate();
}

template <int Dim>
int BSplinePatch<Dim>::degree(const int direction) const
{
    if (direction < 0 || direction >= Dim)
        throw std::out_of_range("Invalid parametric direction");
    return degrees_[direction];
}

template <int Dim>
const std::vector<double> &BSplinePatch<Dim>::knots(const int direction) const
{
    if (direction < 0 || direction >= Dim)
        throw std::out_of_range("Invalid parametric direction");
    return knots_[direction];
}

template <int Dim>
int BSplinePatch<Dim>::num_basis(const int direction) const
{
    if (direction < 0 || direction >= Dim)
        throw std::out_of_range("Invalid parametric direction");
    return basis_shape_[direction];
}

template <int Dim>
int BSplinePatch<Dim>::num_elements(const int direction) const
{
    if (direction < 0 || direction >= Dim)
        throw std::out_of_range("Invalid parametric direction");
    return static_cast<int>(active_spans_[direction].size());
}

template <int Dim>
std::size_t BSplinePatch<Dim>::num_elements() const noexcept
{
    std::size_t count = 1;
    for (const auto &spans : active_spans_)
        count *= spans.size();
    return count;
}

template <int Dim>
const std::vector<int> &BSplinePatch<Dim>::active_spans(
    const int direction) const
{
    if (direction < 0 || direction >= Dim)
        throw std::out_of_range("Invalid parametric direction");
    return active_spans_[direction];
}

template <int Dim>
typename BSplinePatch<Dim>::Index BSplinePatch<Dim>::element_spans(
    std::size_t element) const
{
    if (element >= num_elements())
        throw std::out_of_range("Invalid element index");
    Index spans{};
    for (int direction = 0; direction < Dim; ++direction)
    {
        const std::size_t count = active_spans_[direction].size();
        const std::size_t local = element % count;
        element /= count;
        spans[direction] = active_spans_[direction][local];
    }
    return spans;
}

template <int Dim>
const typename BSplinePatch<Dim>::Boundary &BSplinePatch<Dim>::boundary(
    const int boundary_id) const
{
    // Boundary IDs enumerate lower/upper sides of each parameter direction.
    if (boundary_id < 0 || boundary_id >= 2 * Dim)
        throw std::out_of_range("Invalid B-spline patch boundary ID");
    return boundaries_[boundary_id];
}

template <int Dim>
const typename BSplinePatch<Dim>::Point &BSplinePatch<Dim>::control_point(
    const std::size_t index) const
{
    return control_points_.at(index);
}

template <int Dim>
typename BSplinePatch<Dim>::Point BSplinePatch<Dim>::evaluate_geometry(
    const std::array<double, Dim> &para) const
{
    // Represent a single arbitrary parameter point as a one-point tensor grid
    // so both single and batch evaluation share the same implementation.
    std::array<std::vector<double>, Dim> para_coors;
    for (int dir = 0; dir < Dim; ++dir)
        para_coors[dir].push_back(para[dir]);
    return evaluate_geometry_grid(para_coors).front();
}

template <int Dim>
std::vector<typename BSplinePatch<Dim>::Point>
BSplinePatch<Dim>::evaluate_geometry_grid(
    const std::array<std::vector<double>, Dim> &para_coors) const
{
    // Validate the tensor grid and record its directional shape.
    Index pt_shape{};
    std::size_t num_pts = 1;
    for (int dir = 0; dir < Dim; ++dir)
    {
        if (para_coors[dir].empty())
            throw std::invalid_argument(
                "Geometry evaluation grid must be nonempty in every direction");
        pt_shape[dir] = static_cast<int>(para_coors[dir].size());
        num_pts *= para_coors[dir].size();
    }

    // Evaluate each distinct one-dimensional parameter coordinate exactly
    // once, independently of how many tensor-product points reuse it.
    std::array<std::vector<int>, Dim> first_bases;
    std::array<std::vector<std::vector<double>>, Dim> basis_values;
    for (int dir = 0; dir < Dim; ++dir)
    {
        // Construct one evaluator for this direction and reuse it for every
        // distinct coordinate in the tensor grid. No Gauss table is needed.
        const BSplineBasis1D basis(
            degrees_[dir], knots_[dir]);
        first_bases[dir].resize(pt_shape[dir]);
        basis_values[dir].resize(pt_shape[dir]);
        for (int pt = 0; pt < pt_shape[dir]; ++pt)
        {
            const double para = para_coors[dir][pt];
            BSplineBasis1D::PointEvaluation eval = basis.evaluate_N(para);
            first_bases[dir][pt] = eval.first_basis;
            basis_values[dir][pt] = std::move(eval.N);
        }
    }

    // Record the local tensor-product basis shape shared by every grid point.
    Index local_shape{};
    for (int direction = 0; direction < Dim; ++direction)
        local_shape[direction] = degrees_[direction] + 1;
    const LexicographicOrdering<Dim> point_ordering(pt_shape);
    const LexicographicOrdering<Dim> local_ordering(local_shape);
    const LexicographicOrdering<Dim> basis_ordering(basis_shape_);

    // Form each physical point by combining the cached directional basis
    // values with the corresponding tensor-product control points.
    std::vector<Point> phy_pts(num_pts);
    for (int point_global_index = 0;
         point_global_index < point_ordering.size(); ++point_global_index)
    {
        const Index point_multi_index =
            point_ordering.multi_index(point_global_index);
        for (int local_global_index = 0;
             local_global_index < local_ordering.size(); ++local_global_index)
        {
            const Index local_multi_index =
                local_ordering.multi_index(local_global_index);
            Index basis_multi_index{};
            double value = 1.0;
            for (int dir = 0; dir < Dim; ++dir)
            {
                const int point = point_multi_index[dir];
                basis_multi_index[dir] = first_bases[dir][point] +
                    local_multi_index[dir];
                value *= basis_values[dir][point][local_multi_index[dir]];
            }
            const Point &control =
                control_points_[basis_ordering.global_index(basis_multi_index)];
            for (int component = 0; component < Dim; ++component)
                phy_pts[point_global_index][component] +=
                    value * control[component];
        }
    }
    return phy_pts;
}

template <int Dim>
void BSplinePatch<Dim>::insert_knot(const int direction, const double knot)
{
    if (direction < 0 || direction >= Dim)
        throw std::out_of_range("Invalid insertion direction");

    const int p = degrees_[direction];
    const auto &old_knots = knots_[direction];
    const double lower = old_knots[p];
    const double upper = old_knots[basis_shape_[direction]];
    const double tolerance = 1.0e-12 * std::max(1.0, std::abs(upper - lower));
    if (knot <= lower + tolerance || knot >= upper - tolerance)
        throw std::invalid_argument("Only interior knots can be inserted");

    int multiplicity = 0;
    for (const double value : old_knots)
        if (std::abs(value - knot) <= tolerance)
            ++multiplicity;
    if (multiplicity >= p)
        throw std::invalid_argument("Interior knot multiplicity would exceed degree");

    int span = p;
    const int old_count = basis_shape_[direction];
    while (span + 1 < static_cast<int>(old_knots.size()) &&
           knot >= old_knots[span + 1] - tolerance)
        ++span;
    span = std::min(span, old_count - 1);

    Index new_shape = basis_shape_;
    ++new_shape[direction];
    std::vector<Point> new_points(product<Dim>(new_shape));
    const LexicographicOrdering<Dim> old_ordering(basis_shape_);
    const LexicographicOrdering<Dim> new_ordering(new_shape);

    Index line_shape = basis_shape_;
    line_shape[direction] = 1;
    const LexicographicOrdering<Dim> line_ordering(line_shape);
    for (int line_global_index = 0;
         line_global_index < line_ordering.size(); ++line_global_index)
    {
        Index base = line_ordering.multi_index(line_global_index);

        std::vector<Point> old_line(old_count);
        std::vector<Point> new_line(old_count + 1);
        for (int i = 0; i < old_count; ++i)
        {
            base[direction] = i;
            old_line[i] =
                control_points_[old_ordering.global_index(base)];
        }

        for (int i = 0; i <= span - p; ++i)
            new_line[i] = old_line[i];
        for (int i = span - multiplicity; i < old_count; ++i)
            new_line[i + 1] = old_line[i];
        for (int i = span - p + 1; i <= span - multiplicity; ++i)
        {
            const double denominator = old_knots[i + p] - old_knots[i];
            if (std::abs(denominator) < std::numeric_limits<double>::epsilon())
                throw std::runtime_error("Degenerate knot insertion interval");
            const double alpha = (knot - old_knots[i]) / denominator;
            for (int component = 0; component < Dim; ++component)
                new_line[i][component] = alpha * old_line[i][component] +
                    (1.0 - alpha) * old_line[i - 1][component];
        }

        for (int i = 0; i < old_count + 1; ++i)
        {
            base[direction] = i;
            new_points[new_ordering.global_index(base)] = new_line[i];
        }
    }

    knots_[direction].insert(knots_[direction].begin() + span + 1, knot);
    basis_shape_ = new_shape;
    control_points_ = std::move(new_points);
    rebuild_topology();
    validate();
}

template <int Dim>
void BSplinePatch<Dim>::uniform_refine(const Index &elements_per_direction)
{
    for (int direction = 0; direction < Dim; ++direction)
    {
        if (elements_per_direction[direction] < 1)
            throw std::invalid_argument("Element counts must be positive");
        const double lower = knots_[direction][degrees_[direction]];
        const double upper = knots_[direction][basis_shape_[direction]];
        for (int i = 1; i < elements_per_direction[direction]; ++i)
        {
            const double knot = lower + (upper - lower) *
                static_cast<double>(i) / elements_per_direction[direction];
            bool exists = false;
            for (const double value : knots_[direction])
                exists = exists || std::abs(value - knot) < 1.0e-12;
            if (!exists)
                insert_knot(direction, knot);
        }
    }
}

template <int Dim>
void BSplinePatch<Dim>::validate() const
{
    for (int direction = 0; direction < Dim; ++direction)
    {
        if (degrees_[direction] < 1)
            throw std::invalid_argument("B-spline degrees must be positive");
        if (!std::is_sorted(knots_[direction].begin(), knots_[direction].end()))
            throw std::invalid_argument("Knot vectors must be nondecreasing");
        if (basis_shape_[direction] <= degrees_[direction])
            throw std::invalid_argument("Too few basis functions for degree");
        if (active_spans_[direction].empty())
            throw std::invalid_argument("Patch has no active knot spans");
    }
    if (control_points_.size() != product<Dim>(basis_shape_))
        throw std::invalid_argument("Control-point count does not match knot vectors");
}

template <int Dim>
void BSplinePatch<Dim>::rebuild_topology()
{
    // Rebuild the active nonzero knot-span lists in every direction.
    for (int direction = 0; direction < Dim; ++direction)
    {
        basis_shape_[direction] = static_cast<int>(knots_[direction].size()) -
            degrees_[direction] - 1;
        active_spans_[direction].clear();
        for (int span = degrees_[direction]; span < basis_shape_[direction]; ++span)
            if (knots_[direction][span + 1] > knots_[direction][span])
                active_spans_[direction].push_back(span);
    }

    // Boundary tensor indices depend only on the updated basis shape.
    rebuild_boundaries();
}

template <int Dim>
void BSplinePatch<Dim>::rebuild_boundaries()
{
    // Clear every independently stored edge/face before rebuilding it.
    for (Boundary &boundary_indices : boundaries_)
        boundary_indices.clear();

    // Traverse global basis indices in direction-0-fastest lexicographic
    // order and append each tensor index to every boundary containing it.
    const LexicographicOrdering<Dim> ordering(basis_shape_);
    for (int global_index = 0; global_index < ordering.size(); ++global_index)
    {
        const Index multi_index_value = ordering.multi_index(global_index);
        for (int direction = 0; direction < Dim; ++direction)
        {
            if (multi_index_value[direction] == 0)
                boundaries_[2 * direction].push_back(multi_index_value);
            if (multi_index_value[direction] ==
                basis_shape_[direction] - 1)
                boundaries_[2 * direction + 1].push_back(multi_index_value);
        }
    }
}

template class BSplinePatch<2>;
template class BSplinePatch<3>;

} // namespace lor

