#ifndef LOR_ELEMENT_BSPLINE_ELEMENT_HPP
#define LOR_ELEMENT_BSPLINE_ELEMENT_HPP

#include "Element/BSplineBasis1D.hpp"
#include "Element/GaussQuadrature.hpp"
#include "Element/IGAElement.hpp"
#include "Mesh/BSplinePatch.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace lor
{

template <int Dim>
class BSplineElement final : public IGAElement<Dim>
{
public:
    // Construct all equation-independent tensor-product data for one active
    // knot-span element. No local or global discrete matrix is formed here.
    BSplineElement(const BSplinePatch<Dim> &patch,
        std::size_t element,
        const std::vector<typename BSplinePatch<Dim>::Point> &control_points,
        const QuadratureRule1D &quad,
        const std::vector<BSplineBasis1D> &basis_tables);

private:
    using QuadData = typename IGAElement<Dim>::QuadData;

    void build_quad_data(const BSplinePatch<Dim> &patch,
        std::size_t element,
        const std::vector<typename BSplinePatch<Dim>::Point> &control_points,
        const QuadratureRule1D &quad,
        const std::vector<BSplineBasis1D> &basis_tables,
        const std::array<int, Dim> &local_shape);
};

extern template class BSplineElement<2>;
extern template class BSplineElement<3>;

} // namespace lor

#endif

