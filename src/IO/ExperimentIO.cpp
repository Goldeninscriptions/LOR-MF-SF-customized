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
    char mode[32] = "gmres_shell";
    char side[16] = "left";
    char inner_pc[16] = "jacobi";
    char point_set[32] = "greville";
    PetscBool geometry_set = PETSC_FALSE;
    PetscBool degree_set = PETSC_FALSE;
    PetscBool elements_set = PETSC_FALSE;
    PetscBool compute_solution = PETSC_TRUE;
    PetscBool compute_l2_error = PETSC_TRUE;
    PetscBool compute_condition_number = PETSC_FALSE;

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
    PetscOptionsGetString(nullptr, nullptr, "-preconditioner_side", side,
        sizeof(side), nullptr);
    PetscOptionsGetString(nullptr, nullptr, "-inner_pc", inner_pc,
        sizeof(inner_pc), nullptr);
    PetscOptionsGetString(nullptr, nullptr, "-lor_points", point_set,
        sizeof(point_set), nullptr);
    PetscOptionsGetBool(nullptr, nullptr, "-compute_solution",
        &compute_solution, nullptr);
    PetscOptionsGetBool(nullptr, nullptr, "-compute_l2_error",
        &compute_l2_error, nullptr);
    PetscOptionsGetBool(nullptr, nullptr, "-compute_condition_number",
        &compute_condition_number, nullptr);

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
    options.preconditioner_side = parse_preconditioner_side(side);
    options.inner_pc = parse_inner_pc_mode(inner_pc);
    options.lor_points = point_set;
    if (options.lor_points != "greville" && options.lor_points != "demko")
        throw std::invalid_argument("-lor_points must be greville or demko");
    options.compute_solution = compute_solution == PETSC_TRUE;
    options.compute_l2_error = compute_l2_error == PETSC_TRUE;
    options.compute_condition_number =
        compute_condition_number == PETSC_TRUE;
    if (options.compute_l2_error && !options.compute_solution)
        throw std::invalid_argument(
            "-compute_l2_error true requires -compute_solution true");

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

    // Read generous but user-adjustable SLEPc iteration and search-subspace
    // limits for the difficult smallest-singular-value calculations.
    PetscOptionsGetInt(nullptr, nullptr, "-svd_max_it",
        &options.svd_max_iterations, nullptr);
    PetscOptionsGetInt(nullptr, nullptr, "-svd_ncv",
        &options.svd_subspace_dimension, nullptr);
    if (options.compute_condition_number)
    {
        if (!(options.svd_tolerance > 0.0))
            throw std::invalid_argument("-svd_tol must be positive");
        if (options.svd_max_iterations < 1)
            throw std::invalid_argument("-svd_max_it must be positive");
        if (options.svd_subspace_dimension < 2)
            throw std::invalid_argument("-svd_ncv must be at least two");
    }
    return options;
}

template <int Dim>
RefinementRequest<Dim> make_refinement_request(
    const ExperimentOptions &options)
{
    // Validate the dynamic tuple before copying it into the Mesh-owned fixed
    // tensor shape.
    if (static_cast<int>(options.elements.size()) != Dim)
        throw std::invalid_argument(
            "Refinement element tuple does not match template dimension");
    RefinementRequest<Dim> request;
    request.target_degree = options.degree;
    for (int direction = 0; direction < Dim; ++direction)
        request.target_elements[direction] = options.elements[direction];

    // Translate the already validated text option at the IO/Mesh boundary.
    if (options.refinement_mode == "p_refine")
        request.mode = RefinementMode::p_refine;
    else if (options.refinement_mode == "h_refine")
        request.mode = RefinementMode::h_refine;
    else
        request.mode = RefinementMode::k_refine;
    return request;
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
           << "mpi_ranks = " << results.mpi_ranks << '\n'
           << "p = " << results.actual_degree << '\n'
           << "n = (";
    for (std::size_t direction = 0;
         direction < results.actual_elements.size(); ++direction)
        output << (direction == 0 ? "" : ",")
               << results.actual_elements[direction];
    output << ")\n";

    // Preserve the feature selection in every result file, then omit all
    // numerical fields belonging to computations that did not run.
    output << std::boolalpha
           << "lor_pc = " << lor_pc_mode_name(results.lor_pc) << '\n'
           << "preconditioner_side = "
           << preconditioner_side_name(results.preconditioner_side) << '\n'
           << "inner_pc = " << inner_pc_mode_name(results.inner_pc) << '\n'
           << "compute_solution = " << results.computed_solution << '\n'
           << "compute_l2_error = " << results.computed_l2_error << '\n'
           << "compute_condition_number = "
           << results.computed_condition_number << '\n'
           << std::scientific << std::setprecision(16);

    // Preserve the actual stopping policy beside every computed solution so
    // performance and error changes can be attributed to the correct sweep.
    if (results.computed_solution)
    {
        output << "outer_rtol = " << results.tolerances.outer_rtol << '\n'
               << "outer_atol = " << results.tolerances.outer_atol << '\n'
               << "outer_dtol = " << results.tolerances.outer_dtol << '\n'
               << "outer_max_it = "
               << results.tolerances.outer_max_it << '\n';
        if (results.lor_pc == LORPCMode::gmres_shell)
            output << "inner_rtol = "
                   << results.tolerances.inner_rtol << '\n'
                   << "inner_atol = "
                   << results.tolerances.inner_atol << '\n'
                   << "inner_dtol = "
                   << results.tolerances.inner_dtol << '\n'
                   << "inner_max_it = "
                   << results.tolerances.inner_max_it << '\n';
    }

    // Record predicted tensor structure and PETSc's post-assembly storage so
    // remote batches can detect reallocations and memory regressions.
    output << "predicted_nnz(K) = " << results.predicted_nonzeros_K << '\n'
           << "used_nnz(K) = " << results.used_nonzeros_K << '\n'
           << "allocated_nnz(K) = " << results.allocated_nonzeros_K << '\n'
           << "allocation_calls(K) = " << results.allocation_calls_K << '\n'
           << "memory_bytes(K) = " << results.memory_bytes_K << '\n'
           << "estimated_minimum_bytes(K) = "
           << results.estimated_minimum_bytes_K << '\n'
           << "predicted_nnz(K_LOR) = "
           << results.predicted_nonzeros_lor << '\n'
           << "used_nnz(K_LOR) = " << results.used_nonzeros_lor << '\n'
           << "allocated_nnz(K_LOR) = "
           << results.allocated_nonzeros_lor << '\n'
           << "allocation_calls(K_LOR) = "
           << results.allocation_calls_lor << '\n'
           << "memory_bytes(K_LOR) = " << results.memory_bytes_lor << '\n';
    output << "estimated_minimum_bytes(K_LOR) = "
           << results.estimated_minimum_bytes_lor << '\n';
    // Select one algebraically accurate operator label for every optional
    // diagnostic emitted below.
    const bool left =
        results.preconditioner_side == PreconditionerSide::left;
    const char *preconditioned_operator = left ?
        "K_LOR^{-1}K" : "KK_LOR^{-1}";
    if (results.computed_condition_number)
        output << "cond_2(K) = " << results.condition_K << '\n'
               << "cond_2(" << preconditioned_operator << ") = "
               << results.condition_lor_K << '\n';
    if (results.computed_solution)
    {
        output << "iterations(K) = " << results.iterations_K << '\n'
               << "iterations(" << preconditioned_operator << ") = "
               << results.iterations_lor_K << '\n'
               << "setup_time_seconds(K) = "
               << results.setup_time_K << '\n'
               << "outer_solve_time_seconds(K) = "
               << results.outer_solve_time_K << '\n'
               << "setup_time_seconds(" << preconditioned_operator << ") = "
               << results.setup_time_lor_K << '\n'
               << "outer_solve_time_seconds(" << preconditioned_operator
               << ") = " << results.outer_solve_time_lor_K << '\n'
               << "residual(K) = " << results.residual_K << '\n'
               << "relative_residual(K) = "
               << results.relative_residual_K << '\n'
               << "residual(" << preconditioned_operator << ") = "
               << results.residual_lor_K << '\n'
               << "relative_residual(" << preconditioned_operator << ") = "
               << results.relative_residual_lor_K << '\n';
        if (results.lor_pc == LORPCMode::gmres_shell)
        {
            // Report both cumulative nested work and the per-application
            // averages needed to interpret an outer-iteration reduction.
            const double applications =
                static_cast<double>(results.pc_applications_lor_K);
            const double average_iterations = applications > 0.0 ?
                static_cast<double>(results.inner_iterations_lor_K) /
                    applications : 0.0;
            const double average_time = applications > 0.0 ?
                results.inner_solve_time_lor_K / applications : 0.0;
            output << "pc_applications(" << preconditioned_operator << ") = "
                   << results.pc_applications_lor_K << '\n'
                   << "inner_iterations_total(" << preconditioned_operator
                   << ") = " << results.inner_iterations_lor_K << '\n'
                   << "inner_iterations_average(" << preconditioned_operator
                   << ") = " << average_iterations << '\n'
                   << "inner_solve_time_seconds(" << preconditioned_operator
                   << ") = " << results.inner_solve_time_lor_K << '\n'
                   << "inner_solve_time_average_seconds("
                   << preconditioned_operator << ") = "
                   << average_time << '\n';
        }
    }
    if (results.computed_l2_error)
        output << "L2_error(K) = " << results.l2_error_K << '\n'
               << "L2_error(" << preconditioned_operator << ") = "
               << results.l2_error_lor_K << '\n';
}

// Emit fixed-dimensional option adapters used by the one runtime driver.
template RefinementRequest<2> make_refinement_request<2>(
    const ExperimentOptions &);
template RefinementRequest<3> make_refinement_request<3>(
    const ExperimentOptions &);

} // namespace lor
