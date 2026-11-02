#ifndef LOR_ELEMENT_IGA_ELEMENT_HPP
#define LOR_ELEMENT_IGA_ELEMENT_HPP

#include <array>
#include <cstddef>
#include <vector>

namespace lor
{

// Common read-only interface and storage contract for 2D/3D isogeometric
// elements. Derived classes define the basis family and fill the tensor-product
// data; users and Assembly consume it through the same dimension-independent
// accessors.
template <int Dim>
class IGAElement
{
    static_assert(Dim == 2 || Dim == 3,
        "IGAElement supports only 2D and 3D");

public:
    using Point = std::array<double, Dim>;
    using Matrix = std::array<std::array<double, Dim>, Dim>;

    virtual ~IGAElement() = default;

    // Return the number of local basis functions and tensor Gauss points.
    int num_local_basis() const noexcept
    {
        return num_local_basis_;
    }
    int num_quad_pts() const noexcept
    {
        return static_cast<int>(quad_data_.size());
    }

    // Return the tensor Gauss multi_index corresponding to quadrature
    // global_index q. Direction 0 varies fastest.
    const std::array<int, Dim> &quad_multi_index(const std::size_t q) const
    {
        return quad_data_.at(q).quad_multi_index;
    }

    // Return one local basis value N_local at Gauss point q.
    double N(const std::size_t q, const std::size_t local) const
    {
        return basis_values_[q * num_local_basis_ + local];
    }

    // Return the physical gradient of one local basis function.
    const Point &N_der1(
        const std::size_t q, const std::size_t local) const
    {
        return phy_grads_[q * num_local_basis_ + local];
    }

    // Return reference, spline-parameter, and physical coordinates of Gauss
    // point q, respectively.
    const Point &ref_coor(const std::size_t q) const
    {
        return quad_data_.at(q).ref_coor;
    }
    const Point &para_coor(const std::size_t q) const
    {
        return quad_data_.at(q).para_coor;
    }
    const Point &phy_coor(const std::size_t q) const
    {
        return quad_data_.at(q).phy_coor;
    }

    // Return F = dx/dxi and F^{-1} at Gauss point q.
    const Matrix &Jac(const std::size_t q) const
    {
        return quad_data_.at(q).jac;
    }
    const Matrix &inv_Jac(const std::size_t q) const
    {
        return quad_data_.at(q).inv_jac;
    }

    // Return det(dx/dxi). This geometric determinant is deliberately separate
    // from quad_weight(q).
    double det_Jac(const std::size_t q) const
    {
        return quad_data_.at(q).det_jac;
    }

    // Return the parameter-domain quadrature weight
    // product(w_qd) * det(dxi/dxi_hat). It never contains det(dx/dxi); a
    // physical-domain integral must multiply this value by det_Jac(q).
    double quad_weight(const std::size_t q) const
    {
        return quad_data_.at(q).quad_weight;
    }

protected:
    IGAElement() = default;

    // Internal equation-independent record for one tensor-product Gauss point.
    // Derived IGA elements fill these fields; users and Assembly access them
    // only through the public functions above.
    struct QuadData
    {
        std::array<int, Dim> quad_multi_index{};
        Point ref_coor{};
        Point para_coor{};
        Point phy_coor{};

        Matrix jac{};
        Matrix inv_jac{};
        double det_jac = 0.0;

        double quad_weight = 0.0;
    };

    int num_local_basis_ = 0;

    // Contiguous q-major tables. Entry q*num_local_basis_+a stores basis a at
    // Gauss point q; vector capacity is retained across element rebuilds.
    std::vector<double> basis_values_;
    std::vector<Point> phy_grads_;

    // One-q parameter-gradient workspace retained across element rebuilds.
    // Parameter gradients are consumed while forming geometry and physical
    // gradients, so storing them for every Gauss point is unnecessary.
    std::vector<Point> para_gradient_workspace_;

    // Per-Gauss geometry records are also retained and overwritten in place.
    std::vector<QuadData> quad_data_;
};

} // namespace lor

#endif
