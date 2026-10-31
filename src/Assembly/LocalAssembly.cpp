#include "Assembly/LocalAssembly.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace lor
{
namespace
{

// Reset matrix storage and geometry diagnostics for an independently invoked
// stiffness or mass integration.
template <class Element>
void initialize_matrix_result(const Element &element,
    LocalMatrixAssemblyResult &result)
{
    const int num_basis = element.num_local_basis();
    result.entries.assign(num_basis * num_basis, 0.0);
    result.min_det_jac = std::numeric_limits<double>::max();
    result.max_det_jac = 0.0;
}

// Reset vector storage and geometry diagnostics for an independently invoked
// load integration.
template <class Element>
void initialize_vector_result(const Element &element,
    LocalVectorAssemblyResult &result)
{
    result.entries.assign(element.num_local_basis(), 0.0);
    result.min_det_jac = std::numeric_limits<double>::max();
    result.max_det_jac = 0.0;
}

// Update determinant diagnostics from the quadrature point currently used by
// one independent local integration.
template <class Result>
void update_geometry_statistics(const double det_jac, Result &result)
{
    result.min_det_jac = std::min(result.min_det_jac, det_jac);
    result.max_det_jac = std::max(result.max_det_jac, det_jac);
}

// Integrate the symmetric Laplace bilinear form through the common read-only
// interface shared by high-order IGA and low-order Q1 elements.
template <int Dim, class Element>
void assemble_precomputed_stiffness(const Element &element,
    LocalMatrixAssemblyResult &result)
{
    // Allocate only the requested local matrix; no mass or load storage is
    // touched by this call.
    initialize_matrix_result(element, result);
    const int num_basis = element.num_local_basis();

    // Accumulate one triangle of Kloc and mirror it into the complete
    // row-major array expected by PETSc scatter operations.
    for (int q = 0; q < element.num_quad_pts(); ++q)
    {
        const double det_jac = element.det_Jac(q);
        const double measure = element.quad_weight(q) * det_jac;
        const auto &grads = element.N_der1(q);
        update_geometry_statistics(det_jac, result);
        for (int row = 0; row < num_basis; ++row)
            for (int column = row; column < num_basis; ++column)
            {
                double grad_dot = 0.0;
                for (int direction = 0; direction < Dim; ++direction)
                    grad_dot += grads[row][direction] *
                        grads[column][direction];
                const double value = grad_dot * measure;
                result.entries[row * num_basis + column] += value;
                if (column != row)
                    result.entries[column * num_basis + row] += value;
            }
    }
}

// Integrate the symmetric consistent mass bilinear form independently of the
// stiffness and source terms.
template <class Element>
void assemble_precomputed_mass(const Element &element,
    LocalMatrixAssemblyResult &result)
{
    // Allocate only Mloc so callers that request mass do not implicitly pay
    // for gradients, stiffness entries, or source evaluation.
    initialize_matrix_result(element, result);
    const int num_basis = element.num_local_basis();

    // Accumulate one triangle of Mloc and mirror the off-diagonal entries.
    for (int q = 0; q < element.num_quad_pts(); ++q)
    {
        const double det_jac = element.det_Jac(q);
        const double measure = element.quad_weight(q) * det_jac;
        const auto &basis_values = element.N(q);
        update_geometry_statistics(det_jac, result);
        for (int row = 0; row < num_basis; ++row)
            for (int column = row; column < num_basis; ++column)
            {
                const double value = basis_values[row] *
                    basis_values[column] * measure;
                result.entries[row * num_basis + column] += value;
                if (column != row)
                    result.entries[column * num_basis + row] += value;
            }
    }
}

// Integrate the source functional independently of both local matrices.
template <int Dim, class Element>
void assemble_precomputed_load(const Element &element,
    const ScalarSource<Dim> &source, LocalVectorAssemblyResult &result)
{
    // Reject an empty source before allocating or entering quadrature.
    if (!source)
        throw std::invalid_argument(
            "Local load assembly requires a scalar source function");
    initialize_vector_result(element, result);
    const int num_basis = element.num_local_basis();

    // Evaluate f once per quadrature point and accumulate Floc in local basis
    // order without constructing either dense local matrix.
    for (int q = 0; q < element.num_quad_pts(); ++q)
    {
        const double det_jac = element.det_Jac(q);
        const double measure = element.quad_weight(q) * det_jac;
        const auto &basis_values = element.N(q);
        const double source_value = source(element.phy_coor(q));
        update_geometry_statistics(det_jac, result);
        for (int row = 0; row < num_basis; ++row)
            result.entries[row] +=
                basis_values[row] * source_value * measure;
    }
}

// Fuse any supported combination containing load into one Gauss traversal.
// Compile-time switches remove work and storage for quantities not requested
// by the public combination entry point.
template <int Dim, bool DoStiffness, bool DoMass, class Element>
void assemble_precomputed_with_load(const Element &element,
    const ScalarSource<Dim> &source,
    LocalMatrixAssemblyResult *stiffness,
    LocalMatrixAssemblyResult *mass, LocalVectorAssemblyResult &load)
{
    // A fused load path always needs a valid source and load storage; matrix
    // pointers are required only when their compile-time switch is enabled.
    if (!source)
        throw std::invalid_argument(
            "Local fused assembly requires a scalar source function");
    if constexpr (DoStiffness && DoMass)
    {
        if (stiffness == mass)
            throw std::invalid_argument(
                "Stiffness and mass require distinct local results");
    }
    if constexpr (DoStiffness)
    {
        if (stiffness == nullptr)
            throw std::invalid_argument(
                "Local fused assembly requires a stiffness result");
        initialize_matrix_result(element, *stiffness);
    }
    if constexpr (DoMass)
    {
        if (mass == nullptr)
            throw std::invalid_argument(
                "Local fused assembly requires a mass result");
        initialize_matrix_result(element, *mass);
    }
    initialize_vector_result(element, load);
    const int num_basis = element.num_local_basis();

    // Reuse determinant, physical measure, basis values, and source value for
    // every requested term at the current quadrature point.
    for (int q = 0; q < element.num_quad_pts(); ++q)
    {
        const double det_jac = element.det_Jac(q);
        const double measure = element.quad_weight(q) * det_jac;
        const auto &basis_values = element.N(q);
        const double source_value = source(element.phy_coor(q));
        update_geometry_statistics(det_jac, load);
        if constexpr (DoStiffness)
            update_geometry_statistics(det_jac, *stiffness);
        if constexpr (DoMass)
            update_geometry_statistics(det_jac, *mass);

        // Accumulate the load while the source and basis values are hot.
        for (int row = 0; row < num_basis; ++row)
            load.entries[row] +=
                basis_values[row] * source_value * measure;

        // Accumulate one triangle of each requested symmetric matrix and
        // mirror its off-diagonal contribution into row-major storage.
        if constexpr (DoStiffness)
        {
            const auto &grads = element.N_der1(q);
            for (int row = 0; row < num_basis; ++row)
                for (int column = row; column < num_basis; ++column)
                {
                    double grad_dot = 0.0;
                    for (int direction = 0; direction < Dim; ++direction)
                        grad_dot += grads[row][direction] *
                            grads[column][direction];
                    const double stiffness_value = grad_dot * measure;
                    stiffness->entries[row * num_basis + column] +=
                        stiffness_value;
                    if (column != row)
                        stiffness->entries[column * num_basis + row] +=
                            stiffness_value;
                    if constexpr (DoMass)
                    {
                        const double mass_value = basis_values[row] *
                            basis_values[column] * measure;
                        mass->entries[row * num_basis + column] += mass_value;
                        if (column != row)
                            mass->entries[column * num_basis + row] +=
                                mass_value;
                    }
                }
        }
        else if constexpr (DoMass)
        {
            // Mass-plus-load does not access physical basis gradients.
            for (int row = 0; row < num_basis; ++row)
                for (int column = row; column < num_basis; ++column)
                {
                    const double mass_value = basis_values[row] *
                        basis_values[column] * measure;
                    mass->entries[row * num_basis + column] += mass_value;
                    if (column != row)
                        mass->entries[column * num_basis + row] += mass_value;
                }
        }
    }
}

} // namespace

template <int Dim>
void assemble_local_stiffness(const IGAElement<Dim> &element,
    LocalMatrixAssemblyResult &result)
{
    // Dispatch the IGA interface to the stiffness-only integration kernel.
    assemble_precomputed_stiffness<Dim>(element, result);
}

void assemble_local_stiffness(const FEMQuad4 &element,
    LocalMatrixAssemblyResult &result)
{
    // Dispatch Quad4 through the same stiffness-only kernel.
    assemble_precomputed_stiffness<2>(element, result);
}

void assemble_local_stiffness(const FEMHex8 &element,
    LocalMatrixAssemblyResult &result)
{
    // Dispatch Hex8 through the same stiffness-only kernel.
    assemble_precomputed_stiffness<3>(element, result);
}

template <int Dim>
void assemble_local_mass(const IGAElement<Dim> &element,
    LocalMatrixAssemblyResult &result)
{
    // Dispatch the IGA interface to the mass-only integration kernel.
    assemble_precomputed_mass(element, result);
}

void assemble_local_mass(const FEMQuad4 &element,
    LocalMatrixAssemblyResult &result)
{
    // Dispatch Quad4 through the same mass-only kernel.
    assemble_precomputed_mass(element, result);
}

void assemble_local_mass(const FEMHex8 &element,
    LocalMatrixAssemblyResult &result)
{
    // Dispatch Hex8 through the same mass-only kernel.
    assemble_precomputed_mass(element, result);
}

template <int Dim>
void assemble_local_load(const IGAElement<Dim> &element,
    const ScalarSource<Dim> &source, LocalVectorAssemblyResult &result)
{
    // Dispatch the IGA interface to the load-only integration kernel.
    assemble_precomputed_load<Dim>(element, source, result);
}

void assemble_local_load(const FEMQuad4 &element,
    const ScalarSource<2> &source, LocalVectorAssemblyResult &result)
{
    // Dispatch Quad4 through the same load-only kernel.
    assemble_precomputed_load<2>(element, source, result);
}

void assemble_local_load(const FEMHex8 &element,
    const ScalarSource<3> &source, LocalVectorAssemblyResult &result)
{
    // Dispatch Hex8 through the same load-only kernel.
    assemble_precomputed_load<3>(element, source, result);
}

template <int Dim>
void assemble_local_stiffness_load(const IGAElement<Dim> &element,
    const ScalarSource<Dim> &source,
    LocalMatrixAssemblyResult &stiffness,
    LocalVectorAssemblyResult &load)
{
    // Fuse IGA stiffness and load integration over the same Gauss points.
    assemble_precomputed_with_load<Dim, true, false>(
        element, source, &stiffness, nullptr, load);
}

void assemble_local_stiffness_load(const FEMQuad4 &element,
    const ScalarSource<2> &source, LocalMatrixAssemblyResult &stiffness,
    LocalVectorAssemblyResult &load)
{
    // Fuse Quad4 stiffness and load integration.
    assemble_precomputed_with_load<2, true, false>(
        element, source, &stiffness, nullptr, load);
}

void assemble_local_stiffness_load(const FEMHex8 &element,
    const ScalarSource<3> &source, LocalMatrixAssemblyResult &stiffness,
    LocalVectorAssemblyResult &load)
{
    // Fuse Hex8 stiffness and load integration.
    assemble_precomputed_with_load<3, true, false>(
        element, source, &stiffness, nullptr, load);
}

template <int Dim>
void assemble_local_mass_load(const IGAElement<Dim> &element,
    const ScalarSource<Dim> &source, LocalMatrixAssemblyResult &mass,
    LocalVectorAssemblyResult &load)
{
    // Fuse IGA mass and load integration over the same Gauss points.
    assemble_precomputed_with_load<Dim, false, true>(
        element, source, nullptr, &mass, load);
}

void assemble_local_mass_load(const FEMQuad4 &element,
    const ScalarSource<2> &source, LocalMatrixAssemblyResult &mass,
    LocalVectorAssemblyResult &load)
{
    // Fuse Quad4 mass and load integration.
    assemble_precomputed_with_load<2, false, true>(
        element, source, nullptr, &mass, load);
}

void assemble_local_mass_load(const FEMHex8 &element,
    const ScalarSource<3> &source, LocalMatrixAssemblyResult &mass,
    LocalVectorAssemblyResult &load)
{
    // Fuse Hex8 mass and load integration.
    assemble_precomputed_with_load<3, false, true>(
        element, source, nullptr, &mass, load);
}

template <int Dim>
void assemble_local_stiffness_mass_load(const IGAElement<Dim> &element,
    const ScalarSource<Dim> &source,
    LocalMatrixAssemblyResult &stiffness,
    LocalMatrixAssemblyResult &mass, LocalVectorAssemblyResult &load)
{
    // Fuse every IGA term while retaining three independently owned outputs.
    assemble_precomputed_with_load<Dim, true, true>(
        element, source, &stiffness, &mass, load);
}

void assemble_local_stiffness_mass_load(const FEMQuad4 &element,
    const ScalarSource<2> &source, LocalMatrixAssemblyResult &stiffness,
    LocalMatrixAssemblyResult &mass, LocalVectorAssemblyResult &load)
{
    // Fuse every Quad4 term in one reference-element traversal.
    assemble_precomputed_with_load<2, true, true>(
        element, source, &stiffness, &mass, load);
}

void assemble_local_stiffness_mass_load(const FEMHex8 &element,
    const ScalarSource<3> &source, LocalMatrixAssemblyResult &stiffness,
    LocalMatrixAssemblyResult &mass, LocalVectorAssemblyResult &load)
{
    // Fuse every Hex8 term in one reference-element traversal.
    assemble_precomputed_with_load<3, true, true>(
        element, source, &stiffness, &mass, load);
}

// Emit the two supported IGA dimensions for every template entry point whose
// implementation is intentionally kept in this source file.
template void assemble_local_stiffness<2>(
    const IGAElement<2> &, LocalMatrixAssemblyResult &);
template void assemble_local_stiffness<3>(
    const IGAElement<3> &, LocalMatrixAssemblyResult &);
template void assemble_local_mass<2>(
    const IGAElement<2> &, LocalMatrixAssemblyResult &);
template void assemble_local_mass<3>(
    const IGAElement<3> &, LocalMatrixAssemblyResult &);
template void assemble_local_load<2>(
    const IGAElement<2> &, const ScalarSource<2> &,
    LocalVectorAssemblyResult &);
template void assemble_local_load<3>(
    const IGAElement<3> &, const ScalarSource<3> &,
    LocalVectorAssemblyResult &);
template void assemble_local_stiffness_load<2>(
    const IGAElement<2> &, const ScalarSource<2> &,
    LocalMatrixAssemblyResult &, LocalVectorAssemblyResult &);
template void assemble_local_stiffness_load<3>(
    const IGAElement<3> &, const ScalarSource<3> &,
    LocalMatrixAssemblyResult &, LocalVectorAssemblyResult &);
template void assemble_local_mass_load<2>(
    const IGAElement<2> &, const ScalarSource<2> &,
    LocalMatrixAssemblyResult &, LocalVectorAssemblyResult &);
template void assemble_local_mass_load<3>(
    const IGAElement<3> &, const ScalarSource<3> &,
    LocalMatrixAssemblyResult &, LocalVectorAssemblyResult &);
template void assemble_local_stiffness_mass_load<2>(
    const IGAElement<2> &, const ScalarSource<2> &,
    LocalMatrixAssemblyResult &, LocalMatrixAssemblyResult &,
    LocalVectorAssemblyResult &);
template void assemble_local_stiffness_mass_load<3>(
    const IGAElement<3> &, const ScalarSource<3> &,
    LocalMatrixAssemblyResult &, LocalMatrixAssemblyResult &,
    LocalVectorAssemblyResult &);

} // namespace lor
