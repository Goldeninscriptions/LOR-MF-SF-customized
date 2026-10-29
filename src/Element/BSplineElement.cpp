#include "Element/BSplineElement.hpp"

#include <array>
#include <stdexcept>

namespace lor
{
namespace
{

// Compute the product of tensor extents.
template <int Dim>
int product(const std::array<int, Dim> &shape)
{
    int value = 1;
    for (const int extent : shape)
        value *= extent;
    return value;
}

// Recover a zero-based tensor multi-index from a flat lexicographic index,
// using direction 0 as the fastest-varying direction.
template <int Dim>
std::array<int, Dim> tensor_index(
    int flat, const std::array<int, Dim> &shape)
{
    std::array<int, Dim> index{};
    for (int direction = 0; direction < Dim; ++direction)
    {
        index[direction] = flat % shape[direction];
        flat /= shape[direction];
    }
    return index;
}

// Invert a 2D or 3D geometry Jacobian and return its determinant.
template <int Dim>
double invert(const std::array<std::array<double, Dim>, Dim> &matrix,
    std::array<std::array<double, Dim>, Dim> &inverse)
{
    // Apply the explicit 2-by-2 inverse formula.
    if constexpr (Dim == 2)
    {
        const double det = matrix[0][0] * matrix[1][1] -
            matrix[0][1] * matrix[1][0];
        inverse[0][0] = matrix[1][1] / det;
        inverse[0][1] = -matrix[0][1] / det;
        inverse[1][0] = -matrix[1][0] / det;
        inverse[1][1] = matrix[0][0] / det;
        return det;
    }

    // Apply the explicit 3-by-3 cofactor inverse formula.
    const double det =
        matrix[0][0] * (matrix[1][1] * matrix[2][2] -
                         matrix[1][2] * matrix[2][1]) -
        matrix[0][1] * (matrix[1][0] * matrix[2][2] -
                         matrix[1][2] * matrix[2][0]) +
        matrix[0][2] * (matrix[1][0] * matrix[2][1] -
                         matrix[1][1] * matrix[2][0]);
    inverse[0][0] = (matrix[1][1] * matrix[2][2] -
                      matrix[1][2] * matrix[2][1]) / det;
    inverse[0][1] = (matrix[0][2] * matrix[2][1] -
                      matrix[0][1] * matrix[2][2]) / det;
    inverse[0][2] = (matrix[0][1] * matrix[1][2] -
                      matrix[0][2] * matrix[1][1]) / det;
    inverse[1][0] = (matrix[1][2] * matrix[2][0] -
                      matrix[1][0] * matrix[2][2]) / det;
    inverse[1][1] = (matrix[0][0] * matrix[2][2] -
                      matrix[0][2] * matrix[2][0]) / det;
    inverse[1][2] = (matrix[0][2] * matrix[1][0] -
                      matrix[0][0] * matrix[1][2]) / det;
    inverse[2][0] = (matrix[1][0] * matrix[2][1] -
                      matrix[1][1] * matrix[2][0]) / det;
    inverse[2][1] = (matrix[0][1] * matrix[2][0] -
                      matrix[0][0] * matrix[2][1]) / det;
    inverse[2][2] = (matrix[0][0] * matrix[1][1] -
                      matrix[0][1] * matrix[1][0]) / det;
    return det;
}

} // namespace

template <int Dim>
BSplineElement<Dim>::BSplineElement(const BSplinePatch<Dim> &patch,
    const std::size_t element,
    const std::vector<typename BSplinePatch<Dim>::Point> &control_points,
    const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables)
{
    // Validate the element, one-dimensional rule, and directional tables
    // before allocating any tensor-product data.
    if (element >= patch.num_elements())
        throw std::out_of_range("Invalid B-spline element index");
    if (quad.points.empty() || quad.weights.size() != quad.points.size())
        throw std::invalid_argument("Invalid one-dimensional quadrature rule");
    if (basis_tables.size() != Dim)
        throw std::invalid_argument(
            "One B-spline basis table is required per direction");
    // Record the local tensor shape used by both basis and geometry
    // evaluation. Element receives only gathered coordinates and has no
    // knowledge of global topology or equation numbering.
    std::array<int, Dim> local_shape{};
    for (int direction = 0; direction < Dim; ++direction)
        local_shape[direction] = patch.degree(direction) + 1;
    this->num_local_basis_ = product<Dim>(local_shape);
    if (control_points.size() !=
        static_cast<std::size_t>(this->num_local_basis_))
        throw std::invalid_argument(
            "B-spline element has an inconsistent control-point count");

    // Build every Gauss-point record once for later equation-specific use by
    // Assembly, matrix-free kernels, or diagnostic output.
    build_quad_data(patch, element, control_points, quad, basis_tables,
        local_shape);
}

template <int Dim>
void BSplineElement<Dim>::build_quad_data(
    const BSplinePatch<Dim> &patch, const std::size_t element,
    const std::vector<typename BSplinePatch<Dim>::Point> &control_points,
    const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables,
    const std::array<int, Dim> &local_shape)
{
    // Cache element spans and tensor sizes. Both basis and quadrature flat
    // indices use direction 0 as the fastest-varying direction.
    const std::array<int, Dim> spans = patch.element_spans(element);
    const int num_local_basis = this->num_local_basis_;
    const int num_quad_1d = static_cast<int>(quad.points.size());
    std::array<int, Dim> quad_shape{};
    quad_shape.fill(num_quad_1d);
    const int num_quad_pts = product<Dim>(quad_shape);
    this->quad_data_.resize(num_quad_pts);

    // Precompute local tensor indices because their ordering is identical at
    // every Gauss point of this element.
    std::vector<std::array<int, Dim>> local_indices(num_local_basis);
    for (int local = 0; local < num_local_basis; ++local)
        local_indices[local] = tensor_index<Dim>(local, local_shape);

    // Precompute the affine reference-to-parameter map of this knot-span
    // element. These values do not depend on the Gauss-point index.
    std::array<double, Dim> para_midpoint{};
    std::array<double, Dim> para_half_length{};
    double ref_to_para_measure = 1.0;
    for (int direction = 0; direction < Dim; ++direction)
    {
        const double left = patch.knots(direction)[spans[direction]];
        const double right =
            patch.knots(direction)[spans[direction] + 1];
        para_midpoint[direction] = 0.5 * (right + left);
        para_half_length[direction] = 0.5 * (right - left);
        ref_to_para_measure *= para_half_length[direction];
    }

    // Build one complete equation-independent record at each tensor-product
    // Gauss point.
    for (int flat_quad = 0; flat_quad < num_quad_pts; ++flat_quad)
    {
        QuadData &data = this->quad_data_[flat_quad];
        data.quad_index = tensor_index<Dim>(flat_quad, quad_shape);
        data.basis_values.assign(num_local_basis, 0.0);
        data.para_grads.resize(num_local_basis);
        data.phy_grads.resize(num_local_basis);
        double tensor_quad_weight = 1.0;

        // Map reference Gauss coordinates to this knot-span element and
        // accumulate the tensor-product Gauss weight.
        for (int direction = 0; direction < Dim; ++direction)
        {
            const int q = data.quad_index[direction];
            const double ref = quad.points[q];
            data.ref_coor[direction] = ref;
            data.para_coor[direction] = para_midpoint[direction] +
                para_half_length[direction] * ref;
            tensor_quad_weight *= quad.weights[q];
        }

        // Form tensor-product basis values and parameter derivatives directly
        // from the precomputed one-dimensional Gauss tables.
        for (int local = 0; local < num_local_basis; ++local)
        {
            // Compute each directional table offset once and reuse it for the
            // value and all first-derivative tensor products.
            std::array<std::size_t, Dim> offsets{};
            for (int direction = 0; direction < Dim; ++direction)
                offsets[direction] =
                    (static_cast<std::size_t>(spans[direction]) *
                         num_quad_1d +
                     data.quad_index[direction]) *
                        local_shape[direction] +
                    local_indices[local][direction];

            double value = 1.0;
            for (int direction = 0; direction < Dim; ++direction)
                value *= basis_tables[direction].N_Gauss()[
                    offsets[direction]];
            data.basis_values[local] = value;

            // Differentiate one tensor factor at a time to obtain each
            // component of the parameter-space gradient.
            for (int derivative_direction = 0;
                 derivative_direction < Dim; ++derivative_direction)
            {
                double derivative = 1.0;
                for (int direction = 0; direction < Dim; ++direction)
                    derivative *= (direction == derivative_direction)
                        ? basis_tables[direction].N_der1_Gauss()[
                              offsets[direction]]
                        : basis_tables[direction].N_Gauss()[
                              offsets[direction]];
                data.para_grads[local][derivative_direction] =
                    derivative;
            }
        }

        // Combine basis data with local control points to evaluate the
        // physical coordinate and geometry Jacobian F = dx/dxi.
        for (int local = 0; local < num_local_basis; ++local)
        {
            const typename BSplinePatch<Dim>::Point &control =
                control_points[local];
            for (int phy = 0; phy < Dim; ++phy)
            {
                data.phy_coor[phy] +=
                    data.basis_values[local] * control[phy];
                for (int para = 0; para < Dim; ++para)
                    data.jac[phy][para] +=
                        control[phy] * data.para_grads[local][para];
            }
        }

        // Invert the geometry map and reject invalid elements. Keep det(F)
        // separate from the parameter-domain integration weight.
        data.det_jac = invert<Dim>(data.jac, data.inv_jac);
        if (!(data.det_jac > 1.0e-12))
            throw std::runtime_error(
                "Spline geometry has a non-positive or singular Jacobian");
        data.quad_weight = tensor_quad_weight * ref_to_para_measure;

        // Transform every local gradient with F^{-T}; these physical
        // derivatives are equation-independent and reusable by Assembly.
        for (int local = 0; local < num_local_basis; ++local)
            for (int phy = 0; phy < Dim; ++phy)
                for (int para = 0; para < Dim; ++para)
                    data.phy_grads[local][phy] +=
                        data.inv_jac[para][phy] *
                        data.para_grads[local][para];
    }
}

template class BSplineElement<2>;
template class BSplineElement<3>;

} // namespace lor
