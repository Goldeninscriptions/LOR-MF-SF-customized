#include "IO/PatchInput.hpp"

#include <array>
#include <stdexcept>
#include <vector>

namespace lor
{

template <int Dim>
BSplinePatch<Dim> make_geometry_patch(const GeometryInput &input)
{
    // Reject a mismatched compile-time dimension before copying any geometry
    // arrays into fixed-size containers.
    if (input.dimension != Dim)
        throw std::invalid_argument(
            "Geometry dimension does not match requested Patch dimension");

    // Copy the independently inferred directional degrees and knot vectors.
    typename BSplinePatch<Dim>::Index degrees{};
    std::array<std::vector<double>, Dim> knots;
    for (int direction = 0; direction < Dim; ++direction)
    {
        degrees[direction] = input.degrees[direction];
        knots[direction] = input.knot_vectors[direction];
    }

    // Convert dynamically sized input coordinates into the Patch's fixed-size
    // Point type without changing direction-0-fastest tensor ordering.
    std::vector<typename BSplinePatch<Dim>::Point> control_points;
    control_points.reserve(input.control_points.size());
    for (const auto &input_point : input.control_points)
    {
        typename BSplinePatch<Dim>::Point point{};
        for (int component = 0; component < Dim; ++component)
            point[component] = input_point[component];
        control_points.push_back(point);
    }

    // Construct the minimal geometry Patch; no refinement occurs in IO.
    return BSplinePatch<Dim>(degrees, knots, control_points);
}

template BSplinePatch<2> make_geometry_patch<2>(const GeometryInput &);
template BSplinePatch<3> make_geometry_patch<3>(const GeometryInput &);

} // namespace lor
