#ifndef LOR_ELEMENT_FEM_HEX8_HPP
#define LOR_ELEMENT_FEM_HEX8_HPP

#include "Element/FEMQ1Basis.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace lor
{

// Eight-node trilinear hexahedron on the reference cube [-1,1]^3. Nodes
// 0--3 form the lower face counterclockwise and 4--7 the corresponding upper
// face, matching the conventional PERIGEE Hex8 order.
class FEMHex8
{
public:
    static constexpr int num_basis = 8;
    using Point = std::array<double, 3>;
    using Matrix = std::array<std::array<double, 3>, 3>;

    // Construct the geometry-dependent Gauss data for one physical cell. The
    // reference Q1 table is shared by all cells and must outlive this element.
    FEMHex8(const std::array<Point, num_basis> &phy_nodes,
        const FEMQ1Basis<3> &ref_basis);
    FEMHex8(const std::array<Point, num_basis> &,
        FEMQ1Basis<3> &&) = delete;

    // Return the number of local basis functions and tensor Gauss points.
    int num_local_basis() const noexcept { return num_basis; }
    int num_quad_pts() const noexcept
    {
        return ref_basis_.num_quad_pts();
    }

    // Return all basis values, or one local value, at Gauss point q.
    const std::array<double, num_basis> &N(std::size_t q) const;
    double N(std::size_t q, std::size_t local) const;

    // Return all physical gradients, or one local gradient, at Gauss point q.
    const std::array<Point, num_basis> &N_der1(std::size_t q) const;
    const Point &N_der1(std::size_t q, std::size_t local) const;

    // Return reference and physical coordinates at Gauss point q.
    const Point &ref_coor(std::size_t q) const;
    const Point &phy_coor(std::size_t q) const;

    // Return dx/dref, its inverse, and its scalar determinant at Gauss point q.
    const Matrix &Jac(std::size_t q) const;
    const Matrix &inv_Jac(std::size_t q) const;
    double det_Jac(std::size_t q) const;

    // Return the tensor Gauss weight on [-1,1]^3. It excludes det_Jac(q).
    double quad_weight(std::size_t q) const;

private:
    struct QuadData
    {
        Point phy_coor{};
        std::array<Point, num_basis> phy_grads{};
        Matrix jac{};
        Matrix inv_jac{};
        double det_jac = 0.0;
    };

    const FEMQ1Basis<3> &ref_basis_;
    std::vector<QuadData> quad_data_;
};

} // namespace lor

#endif
