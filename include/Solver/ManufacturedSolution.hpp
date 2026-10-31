#ifndef LOR_SOLVER_MANUFACTURED_SOLUTION_HPP
#define LOR_SOLVER_MANUFACTURED_SOLUTION_HPP

#include "Assembly/LexicographicOrdering.hpp"
#include "Mesh/BSplinePatch.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lor
{

// Manufactured solution for an axis-aligned rectangle or box:
// u(x)=product_d sin(pi*(x_d-a_d)/(b_d-a_d)),
// f(x)=-Delta u=[sum_d (pi/(b_d-a_d))^2] u(x).
template <int Dim>
class ManufacturedSolution
{
public:
    using Point = std::array<double, Dim>;

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

        // Reject a geometry whose parameter boundaries are not the physical
        // rectangle/box faces on which this analytic function has zero trace.
        validate_patch(patch);
    }

    // Verify that every parameter-boundary control point has the corresponding
    // constant physical coordinate of this solution's bounding box. Throws if
    // the current manufactured solution is not applicable to the Patch.
    void validate_patch(const BSplinePatch<Dim> &patch) const
    {
        const LexicographicOrdering<Dim> ordering(patch.basis_shape());
        for (int direction = 0; direction < Dim; ++direction)
            for (int side = 0; side < 2; ++side)
            {
                const double expected = side == 0 ?
                    lower_[direction] : upper_[direction];
                const double scale = std::max(1.0, length_[direction]);
                for (const auto &multi_index :
                    patch.boundary(2 * direction + side))
                {
                    const double actual = patch.control_point(
                        ordering.global_index(multi_index))[direction];
                    if (std::abs(actual - expected) > 1.0e-10 * scale)
                        throw std::invalid_argument(
                            "The manufactured exact solution requires an "
                            "axis-aligned rectangle or box geometry");
                }
            }
    }

    double exact(const Point &coor) const noexcept
    {
        constexpr double pi = 3.141592653589793238462643383279502884;
        double value = 1.0;
        for (int direction = 0; direction < Dim; ++direction)
            value *= std::sin(pi * (coor[direction] - lower_[direction]) /
                length_[direction]);
        return value;
    }

    double source(const Point &coor) const noexcept
    {
        constexpr double pi = 3.141592653589793238462643383279502884;
        double laplace_factor = 0.0;
        for (int direction = 0; direction < Dim; ++direction)
            laplace_factor +=
                (pi / length_[direction]) * (pi / length_[direction]);
        return laplace_factor * exact(coor);
    }

    const Point &lower() const noexcept { return lower_; }
    const Point &upper() const noexcept { return upper_; }

private:
    // Physical bounding-box endpoints and side lengths, indexed by direction.
    // They are immutable after construction and define both u_exact and f.
    Point lower_{};
    Point upper_{};
    Point length_{};
};

} // namespace lor

#endif
