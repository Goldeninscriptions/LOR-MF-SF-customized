#ifndef LOR_SOLVER_MANUFACTURED_SOLUTION_HPP
#define LOR_SOLVER_MANUFACTURED_SOLUTION_HPP

#include "Assembly/LexicographicOrdering.hpp"
#include "Mesh/BSplinePatch.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lor
{

// Physical-coordinate manufactured solution with a common affine Dirichlet
// trace and a geometry-compatible zero-trace bubble. Axis-aligned rectangles
// and boxes use a tensor sine bubble; the recorded curved quadrilateral uses
// the product of its four exact polynomial boundary factors.
template <int Dim>
class ManufacturedSolution
{
    static_assert(Dim == 2 || Dim == 3,
        "ManufacturedSolution supports only 2D and 3D");

public:
    using Point = std::array<double, Dim>;

    // Classify the refined Patch without relying on its input filename. Any
    // supported 3D geometry must remain an axis-aligned box; the one supported
    // non-axis-aligned 2D geometry is the recorded bicubic benchmark Patch.
    explicit ManufacturedSolution(const BSplinePatch<Dim> &patch)
    {
        lower_.fill(std::numeric_limits<double>::max());
        upper_.fill(std::numeric_limits<double>::lowest());
        for (std::size_t index = 0; index < patch.num_basis(); ++index)
            for (int direction = 0; direction < Dim; ++direction)
            {
                const double value = patch.control_point(index)[direction];
                lower_[direction] = std::min(lower_[direction], value);
                upper_[direction] = std::max(upper_[direction], value);
            }
        for (int direction = 0; direction < Dim; ++direction)
        {
            length_[direction] = upper_[direction] - lower_[direction];
            if (!(length_[direction] > 0.0))
                throw std::invalid_argument(
                    "Manufactured-solution bounding box is degenerate");
        }

        axis_aligned_ = has_axis_aligned_boundaries(patch);
        if constexpr (Dim == 3)
            if (!axis_aligned_)
                throw std::invalid_argument(
                    "The current 3D manufactured solution requires an "
                    "axis-aligned box");
        if constexpr (Dim == 2)
            if (!axis_aligned_)
                validate_curved_benchmark(patch);
    }

    // Evaluate u=1+sum_d x_d+bubble(x). In 2D the sum contains x and y only,
    // equivalent to embedding the physical point as (x,y,0).
    double exact(const Point &physical_coordinate) const noexcept
    {
        return affine_part(physical_coordinate) +
            (axis_aligned_ ? sine_bubble(physical_coordinate) :
                curved_bubble_scale_ * curved_bubble(physical_coordinate));
    }

    // Evaluate f=-Delta(u). The affine lift is harmonic. The sine bubble is a
    // Laplace eigenfunction, while the curved polynomial bubble uses the full
    // product-rule Laplacian of its four implicit boundary factors.
    double source(const Point &physical_coordinate) const noexcept
    {
        if (axis_aligned_)
        {
            double eigenvalue = 0.0;
            for (int direction = 0; direction < Dim; ++direction)
            {
                const double frequency = pi_ / length_[direction];
                eigenvalue += frequency * frequency;
            }
            return eigenvalue * sine_bubble(physical_coordinate);
        }
        return -curved_bubble_scale_ *
            curved_bubble_laplacian(physical_coordinate);
    }

    // Return the exact coefficient of the affine boundary trace. Boundary
    // control points of a curved B-spline are not generally points on the
    // physical boundary, so evaluating the complete bubble there is invalid.
    double boundary_coefficient(const Point &control_point) const noexcept
    {
        return affine_part(control_point);
    }

    bool uses_axis_aligned_bubble() const noexcept { return axis_aligned_; }

private:
    struct CurvedFactor
    {
        double value = 0.0;
        std::array<double, 2> gradient{};
        double laplacian = 0.0;
    };

    double affine_part(const Point &physical_coordinate) const noexcept
    {
        double value = 1.0;
        for (int direction = 0; direction < Dim; ++direction)
            value += physical_coordinate[direction];
        return value;
    }

    // Detect whether each parameter boundary lies on its corresponding
    // physical bounding-box face. This remains true after p/h/k refinement.
    bool has_axis_aligned_boundaries(const BSplinePatch<Dim> &patch) const
    {
        const LexicographicOrdering<Dim> ordering(patch.basis_shape());
        for (int direction = 0; direction < Dim; ++direction)
            for (int side = 0; side < 2; ++side)
            {
                const double expected = side == 0 ?
                    lower_[direction] : upper_[direction];
                const double tolerance =
                    1.0e-10 * std::max(1.0, length_[direction]);
                for (const auto &multi_index :
                     patch.boundary(2 * direction + side))
                {
                    const double actual = patch.control_point(
                        ordering.global_index(multi_index))[direction];
                    if (std::abs(actual - expected) > tolerance)
                        return false;
                }
            }
        return true;
    }

    // Guard the benchmark-specific implicit factors against accidental use on
    // another curved Patch. Open-knot corner control points interpolate the
    // four physical corners and remain invariant under p/h/k refinement.
    void validate_curved_benchmark(const BSplinePatch<Dim> &patch) const
    {
        if constexpr (Dim == 2)
        {
            const auto shape = patch.basis_shape();
            const LexicographicOrdering<Dim> ordering(shape);
            const std::array<typename LexicographicOrdering<Dim>::Index, 4>
                corner_indices{{
                    {0, 0}, {shape[0] - 1, 0},
                    {0, shape[1] - 1},
                    {shape[0] - 1, shape[1] - 1}
                }};
            const std::array<Point, 4> expected_corners{{
                {0.0, 0.0}, {4.0, 0.0}, {0.0, 3.0}, {4.0, 3.0}
            }};
            for (std::size_t corner = 0;
                 corner < corner_indices.size(); ++corner)
            {
                const Point &actual = patch.control_point(
                    ordering.global_index(corner_indices[corner]));
                for (int direction = 0; direction < Dim; ++direction)
                    if (std::abs(actual[direction] -
                            expected_corners[corner][direction]) > 1.0e-10)
                        throw std::invalid_argument(
                            "The curved manufactured bubble is defined only "
                            "for curved_quadrilateral_2d");
            }
        }
    }

    double sine_bubble(const Point &physical_coordinate) const noexcept
    {
        double value = 1.0;
        for (int direction = 0; direction < Dim; ++direction)
            value *= std::sin(pi_ *
                (physical_coordinate[direction] - lower_[direction]) /
                length_[direction]);
        return value;
    }

    // Cubic offset b(t) defining all four boundaries of the recorded curved
    // quadrilateral, together with derivatives in its normalized argument.
    static double curved_offset(const double t) noexcept
    {
        return 0.6 * t - 0.3 * t * t - 0.3 * t * t * t;
    }
    static double curved_offset_derivative(const double t) noexcept
    {
        return 0.6 - 0.6 * t - 0.9 * t * t;
    }
    static double curved_offset_second_derivative(const double t) noexcept
    {
        return -0.6 - 1.8 * t;
    }

    // Return bottom, top, left, and right implicit factors. Their product is
    // zero exactly on the four polynomial boundary curves.
    std::array<CurvedFactor, 4> curved_factors(
        const Point &physical_coordinate) const noexcept
    {
        const double x = physical_coordinate[0];
        const double y = physical_coordinate[1];
        const double tx = x / 4.0;
        const double ty = y / 3.0;
        const double bx = curved_offset(tx);
        const double by = curved_offset(ty);
        const double bx_derivative = curved_offset_derivative(tx) / 4.0;
        const double by_derivative = curved_offset_derivative(ty) / 3.0;
        const double bx_second =
            curved_offset_second_derivative(tx) / 16.0;
        const double by_second =
            curved_offset_second_derivative(ty) / 9.0;

        return {{
            {y - bx, {-bx_derivative, 1.0}, -bx_second},
            {3.0 - bx - y, {-bx_derivative, -1.0}, -bx_second},
            {x + by, {1.0, by_derivative}, by_second},
            {4.0 + by - x, {-1.0, by_derivative}, by_second}
        }};
    }

    double curved_bubble(const Point &physical_coordinate) const noexcept
    {
        const auto factors = curved_factors(physical_coordinate);
        double value = 1.0;
        for (const CurvedFactor &factor : factors)
            value *= factor.value;
        return value;
    }

    double curved_bubble_laplacian(
        const Point &physical_coordinate) const noexcept
    {
        const auto factors = curved_factors(physical_coordinate);
        double laplacian = 0.0;

        // Sum Delta(phi_i) times every undifferentiated remaining factor.
        for (std::size_t differentiated = 0;
             differentiated < factors.size(); ++differentiated)
        {
            double term = factors[differentiated].laplacian;
            for (std::size_t other = 0; other < factors.size(); ++other)
                if (other != differentiated)
                    term *= factors[other].value;
            laplacian += term;
        }

        // Add twice every pairwise gradient contraction, multiplied by the two
        // factors left undifferentiated by the product rule.
        for (std::size_t first = 0; first < factors.size(); ++first)
            for (std::size_t second = first + 1;
                 second < factors.size(); ++second)
            {
                double gradient_dot = 0.0;
                for (int direction = 0; direction < 2; ++direction)
                    gradient_dot += factors[first].gradient[direction] *
                        factors[second].gradient[direction];
                double term = 2.0 * gradient_dot;
                for (std::size_t other = 0;
                     other < factors.size(); ++other)
                    if (other != first && other != second)
                        term *= factors[other].value;
                laplacian += term;
            }
        return laplacian;
    }

    static constexpr double pi_ =
        3.141592653589793238462643383279502884;
    static constexpr double curved_bubble_scale_ = 0.1;

    // Physical bounding-box endpoints define the axis-aligned sine bubble.
    // For the curved benchmark they are retained only for classification.
    Point lower_{};
    Point upper_{};
    Point length_{};
    bool axis_aligned_ = false;
};

} // namespace lor

#endif
