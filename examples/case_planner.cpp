#include "Assembly/DirichletBC.hpp"
#include "Assembly/LexicographicOrdering.hpp"
#include "IO/GeometryInput.hpp"
#include "IO/HardwareInfo.hpp"
#include "IO/PatchInput.hpp"
#include "Mesh/ID.hpp"
#include "Mesh/Refinement.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct PlannerOptions
{
    std::string geometry_file;
    std::string refinement;
    int degree = 0;
    std::vector<int> elements;
    int maximum_ranks = 0;
    bool ranks_only = false;
};

// Parse one positive integer and reject trailing characters.
int parse_positive_integer(const std::string &text, const char *name)
{
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != text.size() || value < 1)
        throw std::invalid_argument(std::string(name) +
            " must be a positive integer");
    return value;
}

// Decode the same comma-separated directional element tuple accepted by the
// assembled driver.
std::vector<int> parse_elements(const std::string &text)
{
    std::vector<int> values;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ','))
    {
        if (token.empty())
            throw std::invalid_argument("Empty entry in -n element tuple");
        values.push_back(parse_positive_integer(token, "-n entry"));
    }
    if (values.empty())
        throw std::invalid_argument("-n requires an element tuple");
    return values;
}

// Read planner arguments without initializing PETSc, SLEPc, or MPI.
PlannerOptions parse_options(const int argc, char **argv)
{
    PlannerOptions options;
    for (int argument = 1; argument < argc; ++argument)
    {
        const std::string name = argv[argument];
        if (name == "--recommended-ranks-only")
        {
            options.ranks_only = true;
            continue;
        }
        if (argument + 1 >= argc)
            throw std::invalid_argument("Missing value after " + name);
        const std::string value = argv[++argument];
        if (name == "-geometry")
            options.geometry_file = value;
        else if (name == "-refinement")
            options.refinement = value;
        else if (name == "-p")
            options.degree = parse_positive_integer(value, "-p");
        else if (name == "-n")
            options.elements = parse_elements(value);
        else if (name == "--max-ranks")
            options.maximum_ranks =
                parse_positive_integer(value, "--max-ranks");
        else
            throw std::invalid_argument("Unknown planner option: " + name);
    }

    // Require the same four discretization inputs used by a real case.
    if (options.geometry_file.empty() || options.refinement.empty() ||
        options.degree < 1 || options.elements.empty())
        throw std::invalid_argument(
            "Required options: -geometry FILE -refinement MODE -p DEGREE "
            "-n n1,n2[,n3]");
    if (options.refinement != "p_refine" &&
        options.refinement != "h_refine" &&
        options.refinement != "k_refine")
        throw std::invalid_argument(
            "-refinement must be p_refine, h_refine, or k_refine");
    return options;
}

// Translate dynamic command-line values into the fixed-dimensional Mesh
// request used by the production refinement implementation.
template <int Dim>
lor::RefinementRequest<Dim> make_request(const PlannerOptions &options)
{
    if (options.elements.size() != Dim)
        throw std::invalid_argument(
            "The -n tuple length must match the geometry dimension");
    lor::RefinementRequest<Dim> request;
    request.target_degree = options.degree;
    std::copy(options.elements.begin(), options.elements.end(),
        request.target_elements.begin());
    if (options.refinement == "p_refine")
        request.mode = lor::RefinementMode::p_refine;
    else if (options.refinement == "h_refine")
        request.mode = lor::RefinementMode::h_refine;
    else
        request.mode = lor::RefinementMode::k_refine;
    return request;
}

// Apply the real refinement and boundary-numbering path, then report the
// exact free system size and the rank policy derived from it.
template <int Dim>
int plan_case(const lor::GeometryInput &input,
    const PlannerOptions &options,
    const lor::HardwareConcurrency &hardware)
{
    const lor::BSplinePatch<Dim> geometry =
        lor::make_geometry_patch<Dim>(input);
    const lor::BSplinePatch<Dim> patch = lor::apply_refinement(
        geometry, make_request<Dim>(options));

    // Construct the same all-boundary Dirichlet ID map as the solver so the
    // planner counts algebraic unknowns after elimination, not control points.
    const lor::LexicographicOrdering<Dim> ordering(patch.basis_shape());
    const lor::DirichletBC<Dim> boundary_conditions(
        patch.boundaries(), ordering);
    const lor::ID ID_map(boundary_conditions.num_global_indices(),
        boundary_conditions.constrained_global_indices());

    // A command-line cap can model a scheduler allocation; otherwise the
    // detected logical-CPU capacity is the hardware limit m.
    const int maximum_ranks = options.maximum_ranks > 0 ?
        std::min(options.maximum_ranks, hardware.logical_processors) :
        hardware.logical_processors;
    const int recommended = lor::recommend_mpi_ranks(
        static_cast<std::size_t>(ID_map.num_free_dofs()), maximum_ranks);
    if (options.ranks_only)
    {
        std::cout << recommended << '\n';
        return 0;
    }

    // Emit stable key-value records that remain readable by people and shell
    // scripts without requiring a separate serialization dependency.
    std::cout << "dimension = " << Dim << '\n'
              << "refinement = " << options.refinement << '\n'
              << "degree = " << patch.degree(0) << '\n'
              << "elements = (";
    for (int direction = 0; direction < Dim; ++direction)
        std::cout << (direction == 0 ? "" : ",")
                  << patch.num_elements(direction);
    std::cout << ")\nbasis_shape = (";
    for (int direction = 0; direction < Dim; ++direction)
        std::cout << (direction == 0 ? "" : ",")
                  << patch.num_basis(direction);
    std::cout << ")\ntotal_basis = " << patch.num_basis() << '\n'
              << "free_dofs = " << ID_map.num_free_dofs() << '\n'
              << "detected_logical_processors = "
              << hardware.logical_processors << '\n'
              << "hardware_source = " << hardware.source << '\n'
              << "maximum_mpi_ranks = " << maximum_ranks << '\n'
              << "target_dofs_per_rank = 10000\n"
              << "recommended_mpi_ranks = " << recommended << '\n';
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        // Read the minimal geometry once and dispatch the two supported fixed
        // dimensions without initializing any parallel runtime.
        const PlannerOptions options = parse_options(argc, argv);
        const lor::GeometryInput input =
            lor::read_geometry_input(options.geometry_file);
        const lor::HardwareConcurrency hardware =
            lor::detect_hardware_concurrency();
        if (input.dimension == 2)
            return plan_case<2>(input, options, hardware);
        if (input.dimension == 3)
            return plan_case<3>(input, options, hardware);
        throw std::invalid_argument("Only dimensions 2 and 3 are supported");
    }
    catch (const std::exception &exception)
    {
        std::cerr << "Case planner error: " << exception.what() << '\n';
        return 1;
    }
}
