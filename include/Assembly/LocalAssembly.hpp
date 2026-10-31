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

// Dense square matrix produced for one element. entries uses row-major local
// basis ordering, and the determinant extrema describe the geometry points
// visited by this particular stiffness or mass integration.
struct LocalMatrixAssemblyResult
{
    // Row-major matrix entries with num_local_basis^2 values.
    std::vector<double> entries;

    // Minimum and maximum geometry-Jacobian determinants on this element.
    double min_det_jac = 0.0;
    double max_det_jac = 0.0;
};

// Dense load vector produced for one element. entries follows the same local
// basis ordering as Element and IEN; determinant extrema cover this load
// integration's quadrature traversal.
struct LocalVectorAssemblyResult
{
    // Vector entries with one value per local basis function.
    std::vector<double> entries;

    // Minimum and maximum geometry-Jacobian determinants on this element.
    double min_det_jac = 0.0;
    double max_det_jac = 0.0;
};

// Integrate only integral grad(N_A).grad(N_B) dx. Reusing result across
// elements retains the dense row-major allocation.
template <int Dim>
void assemble_local_stiffness(const IGAElement<Dim> &element,
    LocalMatrixAssemblyResult &result);
void assemble_local_stiffness(const FEMQuad4 &element,
    LocalMatrixAssemblyResult &result);
void assemble_local_stiffness(const FEMHex8 &element,
    LocalMatrixAssemblyResult &result);

// Integrate only the consistent mass entries integral N_A N_B dx.
template <int Dim>
void assemble_local_mass(const IGAElement<Dim> &element,
    LocalMatrixAssemblyResult &result);
void assemble_local_mass(const FEMQuad4 &element,
    LocalMatrixAssemblyResult &result);
void assemble_local_mass(const FEMHex8 &element,
    LocalMatrixAssemblyResult &result);

// Integrate only the load entries integral N_A f(x) dx.
template <int Dim>
void assemble_local_load(const IGAElement<Dim> &element,
    const ScalarSource<Dim> &source, LocalVectorAssemblyResult &result);
void assemble_local_load(const FEMQuad4 &element,
    const ScalarSource<2> &source, LocalVectorAssemblyResult &result);
void assemble_local_load(const FEMHex8 &element,
    const ScalarSource<3> &source, LocalVectorAssemblyResult &result);

// Integrate stiffness and load together. Geometry measure, basis access, and
// quadrature traversal are shared while the two outputs remain independently
// owned by the caller.
template <int Dim>
void assemble_local_stiffness_load(const IGAElement<Dim> &element,
    const ScalarSource<Dim> &source,
    LocalMatrixAssemblyResult &stiffness,
    LocalVectorAssemblyResult &load);
void assemble_local_stiffness_load(const FEMQuad4 &element,
    const ScalarSource<2> &source, LocalMatrixAssemblyResult &stiffness,
    LocalVectorAssemblyResult &load);
void assemble_local_stiffness_load(const FEMHex8 &element,
    const ScalarSource<3> &source, LocalMatrixAssemblyResult &stiffness,
    LocalVectorAssemblyResult &load);

// Integrate mass and load together in one quadrature traversal.
template <int Dim>
void assemble_local_mass_load(const IGAElement<Dim> &element,
    const ScalarSource<Dim> &source, LocalMatrixAssemblyResult &mass,
    LocalVectorAssemblyResult &load);
void assemble_local_mass_load(const FEMQuad4 &element,
    const ScalarSource<2> &source, LocalMatrixAssemblyResult &mass,
    LocalVectorAssemblyResult &load);
void assemble_local_mass_load(const FEMHex8 &element,
    const ScalarSource<3> &source, LocalMatrixAssemblyResult &mass,
    LocalVectorAssemblyResult &load);

// Integrate stiffness, mass, and load together. This is the fused local path
// when all assembled Laplace terms are required by a caller.
template <int Dim>
void assemble_local_stiffness_mass_load(const IGAElement<Dim> &element,
    const ScalarSource<Dim> &source,
    LocalMatrixAssemblyResult &stiffness,
    LocalMatrixAssemblyResult &mass, LocalVectorAssemblyResult &load);
void assemble_local_stiffness_mass_load(const FEMQuad4 &element,
    const ScalarSource<2> &source, LocalMatrixAssemblyResult &stiffness,
    LocalMatrixAssemblyResult &mass, LocalVectorAssemblyResult &load);
void assemble_local_stiffness_mass_load(const FEMHex8 &element,
    const ScalarSource<3> &source, LocalMatrixAssemblyResult &stiffness,
    LocalMatrixAssemblyResult &mass, LocalVectorAssemblyResult &load);

extern template void assemble_local_stiffness<2>(
    const IGAElement<2> &, LocalMatrixAssemblyResult &);
extern template void assemble_local_stiffness<3>(
    const IGAElement<3> &, LocalMatrixAssemblyResult &);
extern template void assemble_local_mass<2>(
    const IGAElement<2> &, LocalMatrixAssemblyResult &);
extern template void assemble_local_mass<3>(
    const IGAElement<3> &, LocalMatrixAssemblyResult &);
extern template void assemble_local_load<2>(
    const IGAElement<2> &, const ScalarSource<2> &,
    LocalVectorAssemblyResult &);
extern template void assemble_local_load<3>(
    const IGAElement<3> &, const ScalarSource<3> &,
    LocalVectorAssemblyResult &);
extern template void assemble_local_stiffness_load<2>(
    const IGAElement<2> &, const ScalarSource<2> &,
    LocalMatrixAssemblyResult &, LocalVectorAssemblyResult &);
extern template void assemble_local_stiffness_load<3>(
    const IGAElement<3> &, const ScalarSource<3> &,
    LocalMatrixAssemblyResult &, LocalVectorAssemblyResult &);
extern template void assemble_local_mass_load<2>(
    const IGAElement<2> &, const ScalarSource<2> &,
    LocalMatrixAssemblyResult &, LocalVectorAssemblyResult &);
extern template void assemble_local_mass_load<3>(
    const IGAElement<3> &, const ScalarSource<3> &,
    LocalMatrixAssemblyResult &, LocalVectorAssemblyResult &);
extern template void assemble_local_stiffness_mass_load<2>(
    const IGAElement<2> &, const ScalarSource<2> &,
    LocalMatrixAssemblyResult &, LocalMatrixAssemblyResult &,
    LocalVectorAssemblyResult &);
extern template void assemble_local_stiffness_mass_load<3>(
    const IGAElement<3> &, const ScalarSource<3> &,
    LocalMatrixAssemblyResult &, LocalMatrixAssemblyResult &,
    LocalVectorAssemblyResult &);

} // namespace lor

#endif
