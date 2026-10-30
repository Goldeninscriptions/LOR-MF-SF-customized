#include "IO/ExperimentIO.hpp"

#include <fstream>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace lor
{
namespace
{

std::vector<int> parse_elements(const std::string &text)
{
    // Split the comma-separated tensor shape while retaining strict control
    // over empty, partial, negative, and zero-valued entries.
    std::vector<int> values;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ','))
    {
        if (token.empty())
            throw std::invalid_argument("Empty entry in -n element tuple");
        std::size_t consumed = 0;
        const int value = std::stoi(token, &consumed);
        if (consumed != token.size() || value < 1)
            throw std::invalid_argument(
                "Every -n element count must be a positive integer");
        values.push_back(value);
    }
    return values;
}

} // namespace

ExperimentOptions read_experiment_options(const int dimension)
{
    // Allocate fixed PETSc option buffers and track the three required inputs.
    ExperimentOptions options;
    char geometry[PETSC_MAX_PATH_LEN] = "";
    char output[PETSC_MAX_PATH_LEN] = "results.log";
    char elements[256] = "";
    char refinement[32] = "k_refine";
    char mode[32] = "cg_lu";
    char point_set[32] = "greville";
    PetscBool geometry_set = PETSC_FALSE;
    PetscBool degree_set = PETSC_FALSE;
    PetscBool elements_set = PETSC_FALSE;

    // Read refinement, discretization, LOR, and output selections.
    PetscOptionsGetString(nullptr, nullptr, "-geometry", geometry,
        sizeof(geometry), &geometry_set);
    PetscOptionsGetString(nullptr, nullptr, "-output", output,
        sizeof(output), nullptr);
    PetscOptionsGetInt(nullptr, nullptr, "-p", &options.degree, &degree_set);
    PetscOptionsGetString(nullptr, nullptr, "-n", elements,
        sizeof(elements), &elements_set);
    PetscOptionsGetString(nullptr, nullptr, "-refinement", refinement,
        sizeof(refinement), nullptr);
    PetscOptionsGetString(nullptr, nullptr, "-lor_pc", mode,
        sizeof(mode), nullptr);
    PetscOptionsGetString(nullptr, nullptr, "-lor_points", point_set,
        sizeof(point_set), nullptr);

    // Transfer validated primitive strings into the typed option record.
    if (!geometry_set || !degree_set || !elements_set)
        throw std::invalid_argument(
            "Required options: -geometry FILE -p DEGREE -n n1,n2[,n3]");
    options.geometry_file = geometry;
    options.output_file = output;
    options.refinement_mode = refinement;
    options.elements = parse_elements(elements);
    if (static_cast<int>(options.elements.size()) != dimension)
        throw std::invalid_argument(
            "The -n tuple length must match the geometry dimension");
    if (options.degree < 1)
        throw std::invalid_argument("-p must be positive");
    if (options.refinement_mode != "p_refine" &&
        options.refinement_mode != "h_refine" &&
        options.refinement_mode != "k_refine")
        throw std::invalid_argument(
            "-refinement must be p_refine, h_refine, or k_refine");
    options.lor_pc = parse_lor_pc_mode(mode);
    options.lor_points = point_set;
    if (options.lor_points != "greville" && options.lor_points != "demko")
        throw std::invalid_argument("-lor_points must be greville or demko");

    // Apply any optional outer/inner KSP and SVD stopping-control overrides.
    PetscOptionsGetReal(nullptr, nullptr, "-outer_rtol",
        &options.tolerances.outer_rtol, nullptr);
    PetscOptionsGetReal(nullptr, nullptr, "-outer_atol",
        &options.tolerances.outer_atol, nullptr);
    PetscOptionsGetReal(nullptr, nullptr, "-outer_dtol",
        &options.tolerances.outer_dtol, nullptr);
    PetscOptionsGetInt(nullptr, nullptr, "-outer_max_it",
        &options.tolerances.outer_max_it, nullptr);
    PetscOptionsGetReal(nullptr, nullptr, "-inner_rtol",
        &options.tolerances.inner_rtol, nullptr);
    PetscOptionsGetReal(nullptr, nullptr, "-inner_atol",
        &options.tolerances.inner_atol, nullptr);
    PetscOptionsGetReal(nullptr, nullptr, "-inner_dtol",
        &options.tolerances.inner_dtol, nullptr);
    PetscOptionsGetInt(nullptr, nullptr, "-inner_max_it",
        &options.tolerances.inner_max_it, nullptr);
    PetscOptionsGetReal(nullptr, nullptr, "-svd_tol",
        &options.svd_tolerance, nullptr);
    return options;
}

std::string read_geometry_filename()
{
    // Read geometry first so the driver can dispatch its 2D/3D template before
    // it parses the dimension-dependent element-count tuple.
    char geometry[PETSC_MAX_PATH_LEN] = "";
    PetscBool geometry_set = PETSC_FALSE;
    PetscOptionsGetString(nullptr, nullptr, "-geometry", geometry,
        sizeof(geometry), &geometry_set);
    if (!geometry_set)
        throw std::invalid_argument("Required option: -geometry FILE");
    return geometry;
}

void write_experiment_results(const std::string &filename,
    const ExperimentResults &results)
{
    // Create an explicitly requested parent directory for direct driver runs;
    // a bare filename continues to write in the current working directory.
    const std::filesystem::path path(filename);
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream output(filename);
    if (!output)
        throw std::runtime_error("Cannot open result file: " + filename);
    // Write refinement metadata from the realized Patch before numerical
    // diagnostics so every result file is self-describing.
    output << "refinement = " << results.refinement_mode << '\n'
           << "p = " << results.actual_degree << '\n'
           << "n = (";
    for (std::size_t direction = 0;
         direction < results.actual_elements.size(); ++direction)
        output << (direction == 0 ? "" : ",")
               << results.actual_elements[direction];
    output << ")\n";

    // Write the two condition/iteration/error triplets in scientific format.
    output << std::scientific << std::setprecision(16)
           << "cond_2(K) = " << results.condition_K << '\n'
           << "iterations(K) = " << results.iterations_K << '\n'
           << "L2_error(K) = " << results.l2_error_K << '\n'
           << "cond_2(K_LOR^{-1}K) = " << results.condition_lor_K << '\n'
           << "iterations(K_LOR^{-1}K) = "
           << results.iterations_lor_K << '\n'
           << "L2_error(K_LOR^{-1}K) = "
           << results.l2_error_lor_K << '\n';
}

} // namespace lor
