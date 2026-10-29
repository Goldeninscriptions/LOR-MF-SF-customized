#include "Element/FEMQuad4.hpp"

#include <stdexcept>

namespace lor
{

FEMQuad4::FEMQuad4(
    const std::array<Point, num_basis> &phy_nodes,
    const FEMQ1Basis<2> &ref_basis)
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
            for (int phy = 0; phy < 2; ++phy)
            {
                data.phy_coor[phy] +=
                    basis_values[local] * phy_nodes[local][phy];
                for (int ref = 0; ref < 2; ++ref)
                    data.jac[phy][ref] +=
                        phy_nodes[local][phy] * ref_grads[local][ref];
            }

        // Invert the Jacobian and reject an invalid physical cell.
        data.det_jac = data.jac[0][0] * data.jac[1][1] -
                       data.jac[0][1] * data.jac[1][0];
        if (!(data.det_jac > 1.0e-12))
            throw std::runtime_error(
                "Quad4 has a non-positive or singular Jacobian");
        const double inv_det = 1.0 / data.det_jac;
        data.inv_jac = {{
            {{ data.jac[1][1] * inv_det, -data.jac[0][1] * inv_det}},
            {{-data.jac[1][0] * inv_det,  data.jac[0][0] * inv_det}}}};

        // Transform every reference gradient with J^{-T}.
        for (int local = 0; local < num_basis; ++local)
            for (int phy = 0; phy < 2; ++phy)
                for (int ref = 0; ref < 2; ++ref)
                    data.phy_grads[local][phy] +=
                        data.inv_jac[ref][phy] * ref_grads[local][ref];
    }
}

// The following accessors expose cached values without permitting mutation.
const std::array<double, FEMQuad4::num_basis> &FEMQuad4::N(
    const std::size_t q) const
{
    return ref_basis_.N(q);
}

double FEMQuad4::N(const std::size_t q, const std::size_t local) const
{
    return ref_basis_.N(q, local);
}

const std::array<FEMQuad4::Point, FEMQuad4::num_basis> &
FEMQuad4::N_der1(const std::size_t q) const
{
    return quad_data_.at(q).phy_grads;
}

const FEMQuad4::Point &FEMQuad4::N_der1(
    const std::size_t q, const std::size_t local) const
{
    return quad_data_.at(q).phy_grads.at(local);
}

const FEMQuad4::Point &FEMQuad4::ref_coor(const std::size_t q) const
{
    return ref_basis_.ref_coor(q);
}

const FEMQuad4::Point &FEMQuad4::phy_coor(const std::size_t q) const
{
    return quad_data_.at(q).phy_coor;
}

const FEMQuad4::Matrix &FEMQuad4::Jac(const std::size_t q) const
{
    return quad_data_.at(q).jac;
}

const FEMQuad4::Matrix &FEMQuad4::inv_Jac(const std::size_t q) const
{
    return quad_data_.at(q).inv_jac;
}

double FEMQuad4::det_Jac(const std::size_t q) const
{
    return quad_data_.at(q).det_jac;
}

double FEMQuad4::quad_weight(const std::size_t q) const
{
    return ref_basis_.quad_weight(q);
}

} // namespace lor
