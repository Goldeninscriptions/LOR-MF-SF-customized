#include "Element/FEMQ1Basis.hpp"

#include <stdexcept>

namespace lor
{

template <int Dim>
FEMQ1Basis<Dim>::FEMQ1Basis(const QuadratureRule1D &quad)
{
    // Validate the one-dimensional rule before forming its tensor product.
    if (quad.points.empty() || quad.points.size() != quad.weights.size())
        throw std::invalid_argument(
            "FEMQ1Basis requires a non-empty quadrature rule");
    for (const double point : quad.points)
        if (point < -1.0 || point > 1.0)
            throw std::out_of_range(
                "FEM Q1 quadrature points must lie in [-1,1]");

    // Allocate one record per tensor Gauss point.
    const std::size_t num_quad_1d = quad.points.size();
    std::size_t num_quad_pts = 1;
    for (int direction = 0; direction < Dim; ++direction)
        num_quad_pts *= num_quad_1d;
    quad_data_.resize(num_quad_pts);

    // Map conventional Quad4/Hex8 local order to binary tensor-corner order.
    constexpr std::array<int, 8> conventional_to_tensor{
        0, 1, 3, 2, 4, 5, 7, 6};

    // Precompute coordinates, weights, basis values, and reference gradients.
    for (std::size_t q = 0; q < num_quad_pts; ++q)
    {
        QuadData &data = quad_data_[q];
        std::size_t remainder = q;
        data.quad_weight = 1.0;
        for (int direction = 0; direction < Dim; ++direction)
        {
            const std::size_t point = remainder % num_quad_1d;
            remainder /= num_quad_1d;
            data.ref_coor[direction] = quad.points[point];
            data.quad_weight *= quad.weights[point];
        }

        // Evaluate tensor Q1 functions in conventional Quad4/Hex8 order.
        for (int local = 0; local < num_basis; ++local)
        {
            const int corner = conventional_to_tensor[local];
            data.N[local] = 1.0;
            for (int direction = 0; direction < Dim; ++direction)
            {
                const double sign =
                    ((corner >> direction) & 1) == 0 ? -1.0 : 1.0;
                data.N[local] *=
                    0.5 * (1.0 + sign * data.ref_coor[direction]);
            }

            // Differentiate one tensor factor and retain the others.
            for (int derivative = 0; derivative < Dim; ++derivative)
            {
                const double derivative_sign =
                    ((corner >> derivative) & 1) == 0 ? -1.0 : 1.0;
                double value = 0.5 * derivative_sign;
                for (int direction = 0; direction < Dim; ++direction)
                {
                    if (direction == derivative)
                        continue;
                    const double sign =
                        ((corner >> direction) & 1) == 0 ? -1.0 : 1.0;
                    value *=
                        0.5 * (1.0 + sign * data.ref_coor[direction]);
                }
                data.ref_grads[local][derivative] = value;
            }
        }
    }
}

// The following accessors expose cached reference data without mutation.
template <int Dim>
const typename FEMQ1Basis<Dim>::Point &FEMQ1Basis<Dim>::ref_coor(
    const std::size_t q) const
{
    return quad_data_.at(q).ref_coor;
}

template <int Dim>
double FEMQ1Basis<Dim>::quad_weight(const std::size_t q) const
{
    return quad_data_.at(q).quad_weight;
}

template <int Dim>
const std::array<double, FEMQ1Basis<Dim>::num_basis> &FEMQ1Basis<Dim>::N(
    const std::size_t q) const
{
    return quad_data_.at(q).N;
}

template <int Dim>
double FEMQ1Basis<Dim>::N(
    const std::size_t q, const std::size_t local) const
{
    return quad_data_.at(q).N.at(local);
}

template <int Dim>
const std::array<typename FEMQ1Basis<Dim>::Point,
    FEMQ1Basis<Dim>::num_basis> &
FEMQ1Basis<Dim>::N_der1_ref(const std::size_t q) const
{
    return quad_data_.at(q).ref_grads;
}

template <int Dim>
const typename FEMQ1Basis<Dim>::Point &FEMQ1Basis<Dim>::N_der1_ref(
    const std::size_t q, const std::size_t local) const
{
    return quad_data_.at(q).ref_grads.at(local);
}

template class FEMQ1Basis<2>;
template class FEMQ1Basis<3>;

} // namespace lor
