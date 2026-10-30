#ifndef LOR_ASSEMBLY_LOCAL_ASSEMBLY_HPP
#define LOR_ASSEMBLY_LOCAL_ASSEMBLY_HPP

#include "Element/FEMHex8.hpp"
#include "Element/FEMQuad4.hpp"
#include "Element/IGAElement.hpp"

#include <array>
#include <functional>
#include <vector>

namespace lor
{

template <int Dim>
using ScalarSource = std::function<double(const std::array<double, Dim> &)>;

// Dense equation-specific data produced for one scalar Laplace element. The
// three containers use the same local basis ordering exposed by Element.
struct LocalAssemblyResult
{
    // Row-major local stiffness matrix: integral grad(N_A).grad(N_B) dx.
    std::vector<double> stiffness;

    // Row-major local consistent mass matrix: integral N_A N_B dx.
    std::vector<double> mass;

    // Local load vector: integral N_A f(x) dx.
    std::vector<double> load;

    // Minimum and maximum geometry-Jacobian determinants on this element.
    double min_det_jac = 0.0;
    double max_det_jac = 0.0;
};

// Assemble stiffness, mass, and load together in one quadrature traversal.
// Reusing result across elements reuses its vector allocations.
template <int Dim>
void assemble_local_system(
    const IGAElement<Dim> &element,
    const ScalarSource<Dim> &source,
    LocalAssemblyResult &result);

void assemble_local_system(
    const FEMQuad4 &element,
    const ScalarSource<2> &source,
    LocalAssemblyResult &result);

void assemble_local_system(
    const FEMHex8 &element,
    const ScalarSource<3> &source,
    LocalAssemblyResult &result);

extern template void assemble_local_system<2>(
    const IGAElement<2> &, const ScalarSource<2> &,
    LocalAssemblyResult &);
extern template void assemble_local_system<3>(
    const IGAElement<3> &, const ScalarSource<3> &,
    LocalAssemblyResult &);

} // namespace lor

#endif
