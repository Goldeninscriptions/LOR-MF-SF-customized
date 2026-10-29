#include "Element/FEMHex8.hpp"

#include <stdexcept>

namespace lor
{

FEMHex8::FEMHex8(
    const std::array<Point, num_basis> &phy_nodes,
    const FEMQ1Basis<3> &ref_basis)
    : ref_basis_(ref_basis)
{
    // Allocate only geometry-dependent data; reference data remain shared.
    quad_data_.resize(static_cast<std::size_t>(ref_basis_.num_quad_pts()));

    // Precompute the geometry, Jacobian, and physical gradients for this cell.
    for (std::size_t q = 0; q < quad_data_.size(); ++q)
    {
        QuadData &data = quad_data_[q];
        const auto &basis_values = ref_basis_.N(q);
        const auto &ref_grads = ref_basis_.N_der1_ref(q);

        // Map the Gauss point and form J = dx/dref from the physical nodes.
        for (int local = 0; local < num_basis; ++local)
            for (int phy = 0; phy < 3; ++phy)
            {
                data.phy_coor[phy] +=
                    basis_values[local] * phy_nodes[local][phy];
                for (int ref = 0; ref < 3; ++ref)
                    data.jac[phy][ref] +=
                        phy_nodes[local][phy] * ref_grads[local][ref];
            }

        // Compute det(J) and the explicit cofactor inverse.
        data.det_jac =
            data.jac[0][0] * (data.jac[1][1] * data.jac[2][2] -
                              data.jac[1][2] * data.jac[2][1]) -
            data.jac[0][1] * (data.jac[1][0] * data.jac[2][2] -
                              data.jac[1][2] * data.jac[2][0]) +
            data.jac[0][2] * (data.jac[1][0] * data.jac[2][1] -
                              data.jac[1][1] * data.jac[2][0]);
        if (!(data.det_jac > 1.0e-12))
            throw std::runtime_error(
                "Hex8 has a non-positive or singular Jacobian");
        const double inv_det = 1.0 / data.det_jac;
        data.inv_jac[0][0] = (data.jac[1][1] * data.jac[2][2] -
                              data.jac[1][2] * data.jac[2][1]) * inv_det;
        data.inv_jac[0][1] = (data.jac[0][2] * data.jac[2][1] -
                              data.jac[0][1] * data.jac[2][2]) * inv_det;
        data.inv_jac[0][2] = (data.jac[0][1] * data.jac[1][2] -
                              data.jac[0][2] * data.jac[1][1]) * inv_det;
        data.inv_jac[1][0] = (data.jac[1][2] * data.jac[2][0] -
                              data.jac[1][0] * data.jac[2][2]) * inv_det;
        data.inv_jac[1][1] = (data.jac[0][0] * data.jac[2][2] -
                              data.jac[0][2] * data.jac[2][0]) * inv_det;
        data.inv_jac[1][2] = (data.jac[0][2] * data.jac[1][0] -
                              data.jac[0][0] * data.jac[1][2]) * inv_det;
        data.inv_jac[2][0] = (data.jac[1][0] * data.jac[2][1] -
                              data.jac[1][1] * data.jac[2][0]) * inv_det;
        data.inv_jac[2][1] = (data.jac[0][1] * data.jac[2][0] -
                              data.jac[0][0] * data.jac[2][1]) * inv_det;
        data.inv_jac[2][2] = (data.jac[0][0] * data.jac[1][1] -
                              data.jac[0][1] * data.jac[1][0]) * inv_det;

        // Transform every reference gradient with J^{-T}.
        for (int local = 0; local < num_basis; ++local)
            for (int phy = 0; phy < 3; ++phy)
                for (int ref = 0; ref < 3; ++ref)
                    data.phy_grads[local][phy] +=
                        data.inv_jac[ref][phy] * ref_grads[local][ref];
    }
}

// The following accessors expose cached values without permitting mutation.
const std::array<double, FEMHex8::num_basis> &FEMHex8::N(
    const std::size_t q) const
{
    return ref_basis_.N(q);
}

double FEMHex8::N(const std::size_t q, const std::size_t local) const
{
    return ref_basis_.N(q, local);
}

const std::array<FEMHex8::Point, FEMHex8::num_basis> &
FEMHex8::N_der1(const std::size_t q) const
{
    return quad_data_.at(q).phy_grads;
}

const FEMHex8::Point &FEMHex8::N_der1(
    const std::size_t q, const std::size_t local) const
{
    return quad_data_.at(q).phy_grads.at(local);
}

const FEMHex8::Point &FEMHex8::ref_coor(const std::size_t q) const
{
    return ref_basis_.ref_coor(q);
}

const FEMHex8::Point &FEMHex8::phy_coor(const std::size_t q) const
{
    return quad_data_.at(q).phy_coor;
}

const FEMHex8::Matrix &FEMHex8::Jac(const std::size_t q) const
{
    return quad_data_.at(q).jac;
}

const FEMHex8::Matrix &FEMHex8::inv_Jac(const std::size_t q) const
{
    return quad_data_.at(q).inv_jac;
}

double FEMHex8::det_Jac(const std::size_t q) const
{
    return quad_data_.at(q).det_jac;
}

double FEMHex8::quad_weight(const std::size_t q) const
{
    return ref_basis_.quad_weight(q);
}

} // namespace lor
