#ifndef LOR_SOLVER_MANUFACTURED_SOLUTION_HPP
#define LOR_SOLVER_MANUFACTURED_SOLUTION_HPP

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
