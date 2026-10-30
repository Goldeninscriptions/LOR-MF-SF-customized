#ifndef LOR_IO_PATCH_INPUT_HPP
#define LOR_IO_PATCH_INPUT_HPP

#include "IO/GeometryInput.hpp"
#include "Mesh/BSplinePatch.hpp"

namespace lor
{

// Convert the minimal knot-vector/control-point geometry description into an
// unrefined Patch. Analysis-space changes belong exclusively to the three
// Mesh refinement policies and are deliberately absent from this IO adapter.
template <int Dim>
BSplinePatch<Dim> make_geometry_patch(const GeometryInput &input);

extern template BSplinePatch<2> make_geometry_patch<2>(
    const GeometryInput &);
extern template BSplinePatch<3> make_geometry_patch<3>(
    const GeometryInput &);

} // namespace lor

#endif
