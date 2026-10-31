#ifndef LOR_MESH_REFINEMENT_HPP
#define LOR_MESH_REFINEMENT_HPP

#include "Mesh/BSplinePatch.hpp"

#include <array>

namespace lor
{

// Select one mathematically distinct refinement policy without coupling Mesh
// to command-line strings or IO records.
enum class RefinementMode
{
    p_refine,
    h_refine,
    k_refine
};

// Immutable refinement request for a tensor-product Patch. All directions use
// target_degree, while target_elements retains an independent count per
// direction as required by the paper.
template <int Dim>
struct RefinementRequest
{
    RefinementMode mode = RefinementMode::k_refine;
    int target_degree = 0;
    std::array<int, Dim> target_elements{};
};

// Validate the selected pure/mixed policy and return the refined analysis
// Patch without modifying the minimal input geometry.
template <int Dim>
BSplinePatch<Dim> apply_refinement(const BSplinePatch<Dim> &geometry,
    const RefinementRequest<Dim> &request);

extern template BSplinePatch<2> apply_refinement<2>(
    const BSplinePatch<2> &, const RefinementRequest<2> &);
extern template BSplinePatch<3> apply_refinement<3>(
    const BSplinePatch<3> &, const RefinementRequest<3> &);

} // namespace lor

#endif
