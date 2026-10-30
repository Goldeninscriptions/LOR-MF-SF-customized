#ifndef LOR_IO_GEOMETRY_INPUT_HPP
#define LOR_IO_GEOMETRY_INPUT_HPP

#include <string>
#include <vector>

namespace lor
{

struct GeometryInput
{
    int dimension = 0;
    std::vector<int> degrees;
    std::vector<std::vector<double>> knot_vectors;
    std::vector<std::vector<double>> control_points;
};

GeometryInput read_geometry_input(const std::string &filename);

} // namespace lor

#endif

