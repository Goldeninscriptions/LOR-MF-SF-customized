#include "IO/GeometryInput.hpp"

#include "Assembly/LexicographicOrdering.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace lor
{
namespace
{

std::string remove_comments(const std::string &text)
{
    std::istringstream input(text);
    std::ostringstream output;
    std::string line;
    while (std::getline(input, line))
    {
        const std::size_t comment = line.find('#');
        output << line.substr(0, comment) << '\n';
    }
    return output.str();
}

std::vector<double> parse_numbers(const std::string &text)
{
    static const std::regex number_pattern(
        R"([-+]?(?:(?:[0-9]+(?:\.[0-9]*)?)|(?:\.[0-9]+))(?:[eE][-+]?[0-9]+)?)");
    std::vector<double> values;
    for (std::sregex_iterator match(text.begin(), text.end(), number_pattern), end;
         match != end; ++match)
    {
        const double value = std::stod(match->str());
        if (!std::isfinite(value))
            throw std::runtime_error("Input contains a non-finite number");
        values.push_back(value);
    }
    return values;
}

std::string braced_value(const std::string &text, const std::string &key)
{
    const std::size_t key_position = text.find(key);
    if (key_position == std::string::npos)
        throw std::runtime_error("Missing input field: " + key);
    const std::size_t equals = text.find('=', key_position + key.size());
    const std::size_t opening = text.find('{', equals);
    if (equals == std::string::npos || opening == std::string::npos)
        throw std::runtime_error("Expected '= {...}' after " + key);

    int depth = 0;
    for (std::size_t position = opening; position < text.size(); ++position)
    {
        if (text[position] == '{')
            ++depth;
        else if (text[position] == '}')
        {
            --depth;
            if (depth == 0)
                return text.substr(opening + 1, position - opening - 1);
        }
    }
    throw std::runtime_error("Unclosed brace in field: " + key);
}

int infer_open_degree(const std::vector<double> &knots, const int direction)
{
    if (knots.size() < 4)
        throw std::runtime_error("Knot vector " + std::to_string(direction) +
            " is too short");
    if (!std::is_sorted(knots.begin(), knots.end()))
        throw std::runtime_error("Knot vector " + std::to_string(direction) +
            " must be nondecreasing");

    const double scale = std::max(1.0, std::abs(knots.back() - knots.front()));
    const double tolerance = 1.0e-12 * scale;
    int leading_multiplicity = 1;
    while (leading_multiplicity < static_cast<int>(knots.size()) &&
           std::abs(knots[leading_multiplicity] - knots.front()) <= tolerance)
        ++leading_multiplicity;
    int trailing_multiplicity = 1;
    while (trailing_multiplicity < static_cast<int>(knots.size()) &&
           std::abs(knots[knots.size() - 1 - trailing_multiplicity] -
                    knots.back()) <= tolerance)
        ++trailing_multiplicity;

    if (leading_multiplicity != trailing_multiplicity)
        throw std::runtime_error("Knot vector " + std::to_string(direction) +
            " must have matching endpoint multiplicities");
    const int degree = leading_multiplicity - 1;
    if (degree < 1)
        throw std::runtime_error("Could not infer a positive degree from knot vector " +
            std::to_string(direction));
    if (!(knots[degree] < knots[knots.size() - degree - 1]))
        throw std::runtime_error("Knot vector " + std::to_string(direction) +
            " has an empty parametric domain");
    return degree;
}

} // namespace

GeometryInput read_geometry_input(const std::string &filename)
{
    std::ifstream stream(filename);
    if (!stream)
        throw std::runtime_error("Could not open geometry input file: " + filename);
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    const std::string text = remove_comments(buffer.str());

    static const std::regex dimension_pattern(R"(dimension\s*=\s*([23]))");
    std::smatch dimension_match;
    if (!std::regex_search(text, dimension_match, dimension_pattern))
        throw std::runtime_error("Missing or invalid 'dimension = 2|3' field");

    GeometryInput input;
    input.dimension = std::stoi(dimension_match[1].str());
    input.knot_vectors.resize(input.dimension);
    input.degrees.resize(input.dimension);
    std::vector<int> basis_counts(input.dimension, 0);
    std::size_t expected_points = 1;
    for (int direction = 0; direction < input.dimension; ++direction)
    {
        const std::string key = "knot_vector_" + std::to_string(direction + 1);
        input.knot_vectors[direction] = parse_numbers(braced_value(text, key));
        input.degrees[direction] = infer_open_degree(
            input.knot_vectors[direction], direction + 1);
        const int basis_count = static_cast<int>(input.knot_vectors[direction].size()) -
            input.degrees[direction] - 1;
        if (basis_count <= input.degrees[direction])
            throw std::runtime_error("Knot vector " + std::to_string(direction + 1) +
                " defines too few basis functions");
        basis_counts[direction] = basis_count;
        expected_points *= static_cast<std::size_t>(basis_count);
    }

    // Construct the dimension-specific ordering once and reuse it for every
    // explicitly indexed control point in this input file.
    std::optional<LexicographicOrdering<2>> ordering_2d;
    std::optional<LexicographicOrdering<3>> ordering_3d;
    if (input.dimension == 2)
        ordering_2d.emplace(
            LexicographicOrdering<2>::Index{basis_counts[0], basis_counts[1]});
    else
        ordering_3d.emplace(LexicographicOrdering<3>::Index{
            basis_counts[0], basis_counts[1], basis_counts[2]});

    const std::string points_text = braced_value(text, "control_points");
    static const std::regex entry_pattern(
        R"(\(([^()]*)\)\s*:\s*\(([^()]*)\))");
    input.control_points.resize(expected_points);
    std::vector<bool> assigned(expected_points, false);
    std::size_t assigned_count = 0;
    for (std::sregex_iterator match(points_text.begin(), points_text.end(), entry_pattern), end;
         match != end; ++match)
    {
        const std::vector<double> index_values = parse_numbers((*match)[1].str());
        std::vector<double> point = parse_numbers((*match)[2].str());
        if (static_cast<int>(index_values.size()) != input.dimension)
            throw std::runtime_error("Every control-point index must have exactly " +
                std::to_string(input.dimension) + " entries");
        if (static_cast<int>(point.size()) != input.dimension)
            throw std::runtime_error("Every control point must have exactly " +
                std::to_string(input.dimension) + " coordinates");

        std::vector<int> multi_index(input.dimension);
        for (int direction = 0; direction < input.dimension; ++direction)
        {
            const double rounded = std::round(index_values[direction]);
            if (std::abs(index_values[direction] - rounded) > 1.0e-12)
                throw std::runtime_error("Control-point indices must be integers");
            const int index = static_cast<int>(rounded);
            if (index < 0 || index >= basis_counts[direction])
                throw std::runtime_error("Control-point index in direction " +
                    std::to_string(direction + 1) + " is out of range");
            multi_index[direction] = index;
        }
        const int global_index = input.dimension == 2 ?
            ordering_2d->global_index({multi_index[0], multi_index[1]}) :
            ordering_3d->global_index(
                {multi_index[0], multi_index[1], multi_index[2]});
        if (assigned[global_index])
            throw std::runtime_error("Duplicate control-point tensor index");
        input.control_points[global_index] = std::move(point);
        assigned[global_index] = true;
        ++assigned_count;
    }
    if (assigned_count != expected_points)
        throw std::runtime_error("Indexed control-point count is " +
            std::to_string(assigned_count) + ", but knot vectors require " +
            std::to_string(expected_points));
    return input;
}

} // namespace lor

