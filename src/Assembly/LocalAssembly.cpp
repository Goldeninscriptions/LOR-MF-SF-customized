#include "Assembly/LocalAssembly.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace lor
{
namespace
{

// Assemble all scalar element terms from the common read-only interface
// shared by high-order IGA and low-order Quad4/Hex8 elements.
template <int Dim, class Element>
void assemble_precomputed_system(
    const Element &element,
    const ScalarSource<Dim> &source,
    LocalAssemblyResult &result)
{
    // Reject an empty source before entering the quadrature loop.
    if (!source)
        throw std::invalid_argument(
            "Local assembly requires a scalar source function");

    // Allocate and zero Kloc, Mloc, and Floc once for this element, following
    // the LocalAssembly structure used in Parallel-Computing and MixPERIGEE.
    // assign() retains existing capacity when result is reused across elements.
    const int num_basis = element.num_local_basis();
    result.stiffness.assign(num_basis * num_basis, 0.0);
    result.mass.assign(num_basis * num_basis, 0.0);
    result.load.assign(num_basis, 0.0);
    result.min_det_jac = std::numeric_limits<double>::max();
    result.max_det_jac = 0.0;

    // Traverse quadrature once so geometry, basis values, and physical
    // gradients are shared by stiffness, mass, and load integration.
    for (int q = 0; q < element.num_quad_pts(); ++q)
    {
        const double det_jac = element.det_Jac(q);
        const double measure = element.quad_weight(q) * det_jac;
        const auto &basis_values = element.N(q);
        const auto &grads = element.N_der1(q);
        const double source_value = source(element.phy_coor(q));
        result.min_det_jac = std::min(result.min_det_jac, det_jac);
        result.max_det_jac = std::max(result.max_det_jac, det_jac);

        // Accumulate Floc and the symmetric Kloc/Mloc entries. Compute one
        // triangle and mirror it into the complete row-major arrays required
        // by PETSc and future matrix-free reference checks.
        for (int row = 0; row < num_basis; ++row)
        {
            result.load[row] +=
                basis_values[row] * source_value * measure;
            for (int column = row; column < num_basis; ++column)
            {
                double grad_dot = 0.0;
                for (int direction = 0; direction < Dim; ++direction)
                    grad_dot += grads[row][direction] *
                        grads[column][direction];
                const int entry = row * num_basis + column;
                const double stiffness_value = grad_dot * measure;
                const double mass_value = basis_values[row] *
                    basis_values[column] * measure;
                result.stiffness[entry] += stiffness_value;
                result.mass[entry] += mass_value;
                if (column != row)
                {
                    const int transpose = column * num_basis + row;
                    result.stiffness[transpose] += stiffness_value;
                    result.mass[transpose] += mass_value;
                }
            }
        }
    }
}

} // namespace

template <int Dim>
void assemble_local_system(
    const IGAElement<Dim> &element,
    const ScalarSource<Dim> &source,
    LocalAssemblyResult &result)
{
    // Dispatch the IGA interface to the common element integration kernel.
    assemble_precomputed_system<Dim>(element, source, result);
}

void assemble_local_system(
    const FEMQuad4 &element,
    const ScalarSource<2> &source,
    LocalAssemblyResult &result)
{
    // Dispatch Quad4 through the same scalar integration kernel.
    assemble_precomputed_system<2>(element, source, result);
}

void assemble_local_system(
    const FEMHex8 &element,
    const ScalarSource<3> &source,
    LocalAssemblyResult &result)
{
    // Dispatch Hex8 through the same scalar integration kernel.
    assemble_precomputed_system<3>(element, source, result);
}

template void assemble_local_system<2>(
    const IGAElement<2> &, const ScalarSource<2> &,
    LocalAssemblyResult &);
template void assemble_local_system<3>(
    const IGAElement<3> &, const ScalarSource<3> &,
    LocalAssemblyResult &);

} // namespace lor
