#ifndef LOR_ELEMENT_FEM_Q1_BASIS_HPP
#define LOR_ELEMENT_FEM_Q1_BASIS_HPP

#include "Element/GaussQuadrature.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace lor
{

// Read-only reference-element table shared by all Q1 cells in one assembly.
// Direction 0 varies fastest for both local tensor corners and quadrature.
template <int Dim>
class FEMQ1Basis
{
    static_assert(Dim == 2 || Dim == 3, "FEMQ1Basis supports only 2D and 3D");

public:
    static constexpr int num_basis = 1 << Dim;
    using Point = std::array<double, Dim>;

    // Construct the reference basis table once from a one-dimensional rule.
    explicit FEMQ1Basis(const QuadratureRule1D &quad);

    // Return the number of tensor Gauss points stored in this table.
    int num_quad_pts() const noexcept
    {
        return static_cast<int>(quad_data_.size());
    }

    // Return the reference coordinate and tensor Gauss weight at point q.
    const Point &ref_coor(std::size_t q) const;
    double quad_weight(std::size_t q) const;

    // Return all Q1 values, or one local value, at reference Gauss point q.
    const std::array<double, num_basis> &N(std::size_t q) const;
    double N(std::size_t q, std::size_t local) const;

    // Return derivatives with respect to reference coordinates. These are not
    // physical gradients; FEMQuad4/FEMHex8 apply the geometry transformation.
    const std::array<Point, num_basis> &N_der1_ref(std::size_t q) const;
    const Point &N_der1_ref(std::size_t q, std::size_t local) const;

private:
    struct QuadData
    {
        Point ref_coor{};
        std::array<double, num_basis> N{};
        std::array<Point, num_basis> ref_grads{};
        double quad_weight = 0.0;
    };

    std::vector<QuadData> quad_data_;
};

extern template class FEMQ1Basis<2>;
extern template class FEMQ1Basis<3>;

} // namespace lor

#endif
