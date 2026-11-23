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

PreconditionerMatrixSource parse_preconditioner_matrix_source(
    const std::string &name)
{
    if (name == "lor_q1")
        return PreconditionerMatrixSource::lor_q1;
    if (name == "truncated_cholesky_tensor")
        return PreconditionerMatrixSource::truncated_cholesky_tensor;
    if (name == "greville_mass_tensor")
        return PreconditionerMatrixSource::greville_mass_tensor;
    throw std::invalid_argument(
        "-preconditioner_matrix_source must be lor_q1, "
        "truncated_cholesky_tensor, or greville_mass_tensor");
}

const char *preconditioner_matrix_source_name(
    const PreconditionerMatrixSource source) noexcept
{
    switch (source)
    {
    case PreconditionerMatrixSource::lor_q1:
        return "lor_q1";
    case PreconditionerMatrixSource::truncated_cholesky_tensor:
        return "truncated_cholesky_tensor";
    case PreconditionerMatrixSource::greville_mass_tensor:
        return "greville_mass_tensor";
    }
    return "unknown";
}

SpectralConditionTarget parse_spectral_condition_target(
    const std::string &name)
{
    if (name == "k")
        return SpectralConditionTarget::stiffness;
    if (name == "p")
        return SpectralConditionTarget::preconditioner;
    if (name == "pinv_k")
        return SpectralConditionTarget::preconditioned;
    throw std::invalid_argument(
        "-spectral_condition_target must be k, p, or pinv_k");
}

const char *spectral_condition_target_name(
    const SpectralConditionTarget target) noexcept
{
    switch (target)
    {
    case SpectralConditionTarget::stiffness:
        return "k";
    case SpectralConditionTarget::preconditioner:
        return "p";
    case SpectralConditionTarget::preconditioned:
        return "pinv_k";
    }
    return "unknown";
}

ExperimentOptions read_experiment_options(const int dimension)
{
    // Allocate fixed PETSc option buffers and track the three required inputs.
    ExperimentOptions options;
    char geometry[PETSC_MAX_PATH_LEN] = "";
    char output[PETSC_MAX_PATH_LEN] = "results.log";
    char elements[256] = "";
    char refinement[32] = "k_refine";
    char mode[32] = "left_gmres_fixed_jacobi";
    char side[16] = "left";
    char inner_pc[16] = "jacobi";
    char point_set[32] = "greville";
    char matrix_source[64] = "lor_q1";
    char spectral_target[32] = "pinv_k";
    char truncation_bandwidth[32] = "auto";
    PetscBool geometry_set = PETSC_FALSE;
    PetscBool degree_set = PETSC_FALSE;
    PetscBool elements_set = PETSC_FALSE;
    PetscBool side_set = PETSC_FALSE;
    PetscBool inner_pc_set = PETSC_FALSE;
    PetscBool inner_rtol_set = PETSC_FALSE;
    PetscBool inner_atol_set = PETSC_FALSE;
    PetscBool inner_dtol_set = PETSC_FALSE;
    PetscBool inner_max_it_set = PETSC_FALSE;
    PetscBool fixed_steps_set = PETSC_FALSE;
    PetscBool fixed_omega_set = PETSC_FALSE;
    PetscBool compute_solution = PETSC_TRUE;
    PetscBool compute_unpreconditioned_solution = PETSC_TRUE;
    PetscBool compute_l2_error = PETSC_TRUE;
    PetscBool compute_condition_number = PETSC_FALSE;
    PetscBool compute_spectral_condition_number = PETSC_FALSE;
    PetscBool compute_truncation_certificate = PETSC_TRUE;
    PetscBool verify_tensor_identity = PETSC_FALSE;
    PetscBool trunc_chol_boundary_protection = PETSC_TRUE;

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
        sizeof(side), &side_set);
    PetscOptionsGetString(nullptr, nullptr, "-inner_pc", inner_pc,
        sizeof(inner_pc), &inner_pc_set);
    PetscOptionsGetString(nullptr, nullptr, "-lor_points", point_set,
        sizeof(point_set), nullptr);
    PetscOptionsGetString(nullptr, nullptr, "-preconditioner_matrix_source",
        matrix_source, sizeof(matrix_source), nullptr);
    PetscOptionsGetString(nullptr, nullptr, "-spectral_condition_target",
        spectral_target, sizeof(spectral_target), nullptr);
    PetscBool truncation_bandwidth_set = PETSC_FALSE;
    PetscOptionsGetString(nullptr, nullptr, "-trunc_chol_k",
        truncation_bandwidth, sizeof(truncation_bandwidth),
        &truncation_bandwidth_set);
    PetscOptionsGetBool(nullptr, nullptr, "-compute_solution",
        &compute_solution, nullptr);
    PetscOptionsGetBool(nullptr, nullptr,
        "-compute_unpreconditioned_solution",
        &compute_unpreconditioned_solution, nullptr);
    PetscOptionsGetBool(nullptr, nullptr, "-compute_l2_error",
        &compute_l2_error, nullptr);
    PetscOptionsGetBool(nullptr, nullptr, "-compute_condition_number",
        &compute_condition_number, nullptr);
    PetscOptionsGetBool(nullptr, nullptr,
        "-compute_spectral_condition_number",
        &compute_spectral_condition_number, nullptr);
    PetscOptionsGetBool(nullptr, nullptr,
        "-compute_truncation_certificate",
        &compute_truncation_certificate, nullptr);
    PetscOptionsGetBool(nullptr, nullptr, "-verify_tensor_identity",
        &verify_tensor_identity, nullptr);
    PetscOptionsGetBool(nullptr, nullptr,
        "-trunc_chol_boundary_protection",
        &trunc_chol_boundary_protection, nullptr);
    PetscOptionsGetReal(nullptr, nullptr, "-trunc_chol_max_memory_bytes",
        &options.trunc_chol_max_memory_bytes, nullptr);

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
    options.preconditioner_matrix_source =
        parse_preconditioner_matrix_source(matrix_source);
    options.spectral_condition_target =
        parse_spectral_condition_target(spectral_target);
    options.trunc_chol_k_was_auto =
        !truncation_bandwidth_set ||
        std::string(truncation_bandwidth) == "auto";
    if (!options.trunc_chol_k_was_auto)
    {
        const std::string text = truncation_bandwidth;
        std::size_t consumed = 0;
        options.trunc_chol_k = std::stoi(text, &consumed);
        if (consumed != text.size() || options.trunc_chol_k < 0)
            throw std::invalid_argument(
                "-trunc_chol_k must be auto or a nonnegative integer");
    }
    const bool uses_inner_cg =
        options.lor_pc == LORPCMode::left_gmres_cg ||
        options.lor_pc == LORPCMode::right_gmres_cg ||
        options.lor_pc == LORPCMode::right_fgmres_cg;
    const bool uses_fixed_jacobi =
        options.lor_pc == LORPCMode::left_gmres_fixed_jacobi;
    options.preconditioner_side = lor_pc_mode_side(options.lor_pc);
    if (side_set && parse_preconditioner_side(side) !=
        options.preconditioner_side)
        throw std::invalid_argument(
            "-preconditioner_side conflicts with the direction encoded by "
            "-lor_pc");
    if (uses_inner_cg)
        options.inner_pc = parse_inner_pc_mode(inner_pc);
    options.lor_points = point_set;
    if (options.lor_points != "greville" && options.lor_points != "demko")
        throw std::invalid_argument("-lor_points must be greville or demko");
    options.compute_solution = compute_solution == PETSC_TRUE;
    options.compute_unpreconditioned_solution =
        compute_unpreconditioned_solution == PETSC_TRUE;
    options.compute_l2_error = compute_l2_error == PETSC_TRUE;
    options.compute_condition_number =
        compute_condition_number == PETSC_TRUE;
    options.compute_spectral_condition_number =
        compute_spectral_condition_number == PETSC_TRUE;
    options.compute_truncation_certificate =
        compute_truncation_certificate == PETSC_TRUE;
    options.verify_tensor_identity = verify_tensor_identity == PETSC_TRUE;
    options.trunc_chol_boundary_protection =
        trunc_chol_boundary_protection == PETSC_TRUE;
    if (!(options.trunc_chol_max_memory_bytes > 0.0) ||
        !std::isfinite(options.trunc_chol_max_memory_bytes))
        throw std::invalid_argument(
            "-trunc_chol_max_memory_bytes must be finite and positive");
    if (options.compute_l2_error && !options.compute_solution)
        throw std::invalid_argument(
            "-compute_l2_error true requires -compute_solution true");
    const bool uses_k_gamg =
        options.lor_pc == LORPCMode::left_gmres_k_gamg ||
        options.lor_pc == LORPCMode::right_gmres_k_gamg;
    if (uses_k_gamg && options.compute_condition_number)
        throw std::invalid_argument(
            "K-GAMG modes do not support -compute_condition_number true; "
            "use physical residual and L2 error diagnostics");
    const bool uses_direct_k = !lor_pc_mode_uses_lor_matrix(options.lor_pc);
    if (options.preconditioner_matrix_source ==
            PreconditionerMatrixSource::truncated_cholesky_tensor &&
        uses_direct_k)
        throw std::invalid_argument(
            "-preconditioner_matrix_source truncated_cholesky_tensor conflicts "
            "with a direct K-Jacobi/K-GAMG lor_pc mode");
    if (options.preconditioner_matrix_source ==
            PreconditionerMatrixSource::greville_mass_tensor &&
        uses_direct_k)
        throw std::invalid_argument(
            "-preconditioner_matrix_source greville_mass_tensor conflicts "
            "with a direct K-Jacobi/K-GAMG lor_pc mode");
    if (options.preconditioner_matrix_source ==
            PreconditionerMatrixSource::greville_mass_tensor &&
        dimension != 2)
        throw std::invalid_argument(
            "greville_mass_tensor currently supports only 2D");
    if (options.preconditioner_matrix_source ==
            PreconditionerMatrixSource::greville_mass_tensor &&
        (options.compute_solution || options.compute_condition_number))
        throw std::invalid_argument(
            "greville_mass_tensor is currently an EPS-only matrix-free "
            "source; disable solution and SVD condition computations");
    if (uses_direct_k && options.compute_spectral_condition_number &&
        options.spectral_condition_target !=
            SpectralConditionTarget::stiffness)
        throw std::invalid_argument(
            "spectral targets p and pinv_k require an assembled "
            "preconditioner matrix and are unavailable for direct K modes");
    if (options.preconditioner_matrix_source ==
        PreconditionerMatrixSource::truncated_cholesky_tensor)
    {
        if (options.degree <= 1)
            throw std::invalid_argument(
                "truncated_cholesky_tensor requires p>=2; p=1 has no "
                "truncatable derivative mass bandwidth");
        const int effective_k = options.trunc_chol_k_was_auto ?
            (options.degree + 1) / 2 : options.trunc_chol_k;
        if (effective_k >= options.degree - 1)
            throw std::invalid_argument(
                "The requested/default trunc_chol_k is not strictly smaller "
                "than derivative mass half-bandwidth p-1; for p=2 or p=3 "
                "specify a smaller explicit -trunc_chol_k");
    }

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
        &options.tolerances.inner_rtol, &inner_rtol_set);
    PetscOptionsGetReal(nullptr, nullptr, "-inner_atol",
        &options.tolerances.inner_atol, &inner_atol_set);
    PetscOptionsGetReal(nullptr, nullptr, "-inner_dtol",
        &options.tolerances.inner_dtol, &inner_dtol_set);
    PetscOptionsGetInt(nullptr, nullptr, "-inner_max_it",
        &options.tolerances.inner_max_it, &inner_max_it_set);
    PetscOptionsGetInt(nullptr, nullptr, "-fixed_jacobi_steps",
        &options.fixed_jacobi.steps, &fixed_steps_set);
    PetscOptionsGetReal(nullptr, nullptr, "-fixed_jacobi_omega",
        &options.fixed_jacobi.omega, &fixed_omega_set);
    PetscOptionsGetReal(nullptr, nullptr, "-svd_tol",
        &options.svd_tolerance, nullptr);
    PetscOptionsGetReal(nullptr, nullptr, "-eps_tol",
        &options.eps_tolerance, nullptr);

    // Reject parameters that do not belong to the selected complete scheme;
    // this prevents result metadata from implying inactive nested algorithms.
    const bool any_inner_option = inner_pc_set || inner_rtol_set ||
        inner_atol_set || inner_dtol_set || inner_max_it_set;
    const bool any_fixed_option = fixed_steps_set || fixed_omega_set;
    if (!uses_inner_cg && any_inner_option)
        throw std::invalid_argument(
            "inner CG options are valid only for a *_cg lor_pc mode");
    if (!uses_fixed_jacobi && any_fixed_option)
        throw std::invalid_argument(
            "fixed Jacobi options are valid only for "
            "left_gmres_fixed_jacobi");

    // Scheme B must remain one fixed polynomial for every right-hand side;
    // reject nonpositive work or an unstable endpoint relaxation parameter.
    if (uses_fixed_jacobi)
    {
        if (options.fixed_jacobi.steps < 1)
            throw std::invalid_argument(
                "-fixed_jacobi_steps must be positive");
        if (!(options.fixed_jacobi.omega > 0.0 &&
              options.fixed_jacobi.omega < 1.0))
            throw std::invalid_argument(
                "-fixed_jacobi_omega must lie between zero and one");
    }

    // Read generous but user-adjustable SLEPc iteration and search-subspace
    // limits for the difficult smallest-singular-value calculations.
    PetscOptionsGetInt(nullptr, nullptr, "-svd_max_it",
        &options.svd_max_iterations, nullptr);
    PetscOptionsGetInt(nullptr, nullptr, "-svd_ncv",
        &options.svd_subspace_dimension, nullptr);
    PetscOptionsGetInt(nullptr, nullptr, "-eps_max_it",
        &options.eps_max_iterations, nullptr);
    PetscOptionsGetInt(nullptr, nullptr, "-eps_ncv",
        &options.eps_subspace_dimension, nullptr);
    if (options.compute_condition_number)
    {
        if (!(options.svd_tolerance > 0.0))
            throw std::invalid_argument("-svd_tol must be positive");
        if (options.svd_max_iterations < 1)
            throw std::invalid_argument("-svd_max_it must be positive");
        if (options.svd_subspace_dimension < 2)
            throw std::invalid_argument("-svd_ncv must be at least two");
    }
    if (options.compute_spectral_condition_number)
    {
        if (!(options.eps_tolerance > 0.0))
            throw std::invalid_argument("-eps_tol must be positive");
        if (options.eps_max_iterations < 1)
            throw std::invalid_argument("-eps_max_it must be positive");
        if (options.eps_subspace_dimension < 2)
            throw std::invalid_argument("-eps_ncv must be at least two");
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
           << "total_free_dofs = " << results.total_free_dofs << '\n'
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
           << "lor_points = " << results.lor_points << '\n'
           << "preconditioner_matrix_source = "
           << preconditioner_matrix_source_name(
                results.preconditioner_matrix_source) << '\n'
           << "preconditioner_side = "
           << preconditioner_side_name(results.preconditioner_side) << '\n';
    const bool uses_inner_cg =
        results.lor_pc == LORPCMode::left_gmres_cg ||
        results.lor_pc == LORPCMode::right_gmres_cg ||
        results.lor_pc == LORPCMode::right_fgmres_cg;
    if (uses_inner_cg)
        output << "inner_pc = " << inner_pc_mode_name(results.inner_pc)
               << '\n';
    output << "compute_solution = " << results.computed_solution << '\n'
           << "compute_unpreconditioned_solution = "
           << results.computed_unpreconditioned_solution << '\n'
           << "unpreconditioned_solution_status = "
           << (results.computed_unpreconditioned_solution ?
               "computed" : "not_computed") << '\n'
           << "compute_l2_error = " << results.computed_l2_error << '\n'
           << "compute_condition_number = "
           << results.computed_condition_number << '\n'
           << "compute_spectral_condition_number = "
           << results.computed_spectral_condition_number << '\n'
           << "spectral_condition_target = "
           << spectral_condition_target_name(
                results.spectral_condition_target) << '\n'
           << std::scientific << std::setprecision(16);

    if (results.preconditioner_matrix_source ==
        PreconditionerMatrixSource::truncated_cholesky_tensor)
    {
        output << "trunc_chol_k_requested = "
               << (results.trunc_chol_k_was_auto ? "auto" :
                   std::to_string(results.trunc_chol_k_effective)) << '\n'
               << "trunc_chol_k_effective = "
               << results.trunc_chol_k_effective << '\n'
               << "trunc_chol_original_degree_p = "
               << results.trunc_chol_original_degree_p << '\n'
               << "trunc_chol_mass_degree_q = "
               << results.trunc_chol_mass_degree_q << '\n'
               << "trunc_chol_boundary_protection = "
               << results.trunc_chol_boundary_protection << '\n'
               << "trunc_chol_boundary_preservation = "
               << (results.trunc_chol_boundary_protection ?
                   "conservative_w_layers" : "strict_k_band") << '\n'
               << "trunc_chol_dropped_entries = "
               << results.trunc_chol_dropped_entries << '\n'
               << "trunc_chol_nnz_complete_factors = "
               << results.trunc_chol_complete_factor_nonzeros << '\n'
               << "trunc_chol_nnz_truncated_factors = "
               << results.trunc_chol_truncated_factor_nonzeros << '\n'
               << "trunc_chol_certificate_computed = "
               << results.trunc_chol_certificate_computed << '\n';
        for (std::size_t direction = 0;
             direction < results.trunc_chol_value_original_half_bandwidth.size();
             ++direction)
            output << "trunc_chol_direction_" << direction
                   << "_value_original_half_bandwidth = "
                   << results.trunc_chol_value_original_half_bandwidth[direction]
                   << '\n'
                   << "trunc_chol_direction_" << direction
                   << "_value_complete_factor_half_bandwidth = "
                   << results.trunc_chol_value_complete_factor_half_bandwidth[direction]
                   << '\n'
                   << "trunc_chol_direction_" << direction
                   << "_value_truncated_factor_half_bandwidth = "
                   << results.trunc_chol_value_truncated_factor_half_bandwidth[direction]
                   << '\n'
                   << "trunc_chol_direction_" << direction
                   << "_value_complete_factor_nnz = "
                   << results.trunc_chol_value_complete_factor_nonzeros[direction]
                   << '\n'
                   << "trunc_chol_direction_" << direction
                   << "_value_truncated_factor_nnz = "
                   << results.trunc_chol_value_truncated_factor_nonzeros[direction]
                   << '\n'
                   << "trunc_chol_direction_" << direction
                   << "_value_dropped_factor_entries = "
                   << results.trunc_chol_value_dropped_factor_entries[direction]
                   << '\n'
                   << "trunc_chol_direction_" << direction
                   << "_derivative_original_half_bandwidth = "
                   << results.trunc_chol_derivative_original_half_bandwidth[direction]
                   << '\n'
                   << "trunc_chol_direction_" << direction
                   << "_derivative_complete_factor_half_bandwidth = "
                   << results.trunc_chol_derivative_complete_factor_half_bandwidth[direction]
                   << '\n'
                   << "trunc_chol_direction_" << direction
                   << "_derivative_truncated_factor_half_bandwidth = "
                   << results.trunc_chol_derivative_truncated_factor_half_bandwidth[direction]
                   << '\n'
                   << "trunc_chol_direction_" << direction
                   << "_derivative_complete_factor_nnz = "
                   << results.trunc_chol_derivative_complete_factor_nonzeros[direction]
                   << '\n'
                   << "trunc_chol_direction_" << direction
                   << "_derivative_truncated_factor_nnz = "
                   << results.trunc_chol_derivative_truncated_factor_nonzeros[direction]
                   << '\n'
                   << "trunc_chol_direction_" << direction
                   << "_derivative_dropped_factor_entries = "
                   << results.trunc_chol_derivative_dropped_factor_entries[direction]
                   << '\n';
        if (results.trunc_chol_certificate_computed)
            for (std::size_t direction = 0;
                 direction < results.trunc_chol_beta_value_mass.size();
                 ++direction)
                output << "trunc_chol_beta_direction_" << direction
                   << "_value_mass = "
                   << results.trunc_chol_beta_value_mass[direction] << '\n'
                   << "trunc_chol_beta_direction_" << direction
                   << "_derivative_mass = "
                   << results.trunc_chol_beta_derivative_mass[direction]
                   << '\n'
                   << "trunc_chol_beta_direction_" << direction
                   << "_below_one = "
                   << results.trunc_chol_beta_below_one[direction] << '\n'
                   << "trunc_chol_certificate_direction_" << direction
                   << "_value_maximum_row_sum = "
                   << results.trunc_chol_certificate_value_maximum_row_sum[direction]
                   << '\n'
                   << "trunc_chol_certificate_direction_" << direction
                   << "_value_maximum_column_sum = "
                   << results.trunc_chol_certificate_value_maximum_column_sum[direction]
                   << '\n'
                   << "trunc_chol_certificate_direction_" << direction
                   << "_derivative_maximum_row_sum = "
                   << results.trunc_chol_certificate_derivative_maximum_row_sum[direction]
                   << '\n'
                   << "trunc_chol_certificate_direction_" << direction
                   << "_derivative_maximum_column_sum = "
                   << results.trunc_chol_certificate_derivative_maximum_column_sum[direction]
                   << '\n';
        output << "trunc_chol_certificate_status = "
               << (!results.trunc_chol_certificate_computed ?
                   "not_computed" :
                   (results.trunc_chol_certificate_bound_available ?
                    "computed_beta_below_one" :
                    "computed_bound_unavailable")) << '\n';
        if (results.trunc_chol_certificate_bound_available)
            output << "trunc_chol_theoretical_kappa_lambda_bound = "
                   << results.trunc_chol_theoretical_kappa_lambda_bound
                   << '\n';
        output << "tensor_identity_verification_status = "
               << (!results.tensor_identity_verification_computed ?
                   "not_computed" :
                   (results.tensor_identity_verification_passed ?
                    "passed" : "failed")) << '\n';
        if (results.tensor_identity_verification_computed)
            output << "tensor_identity_relative_error = "
                   << results.tensor_identity_relative_error << '\n';
        output << "derivative_identity_max_relative_error = "
               << results.derivative_identity_max_relative_error << '\n'
               << "preconditioner_symmetric = "
               << results.preconditioner_symmetric << '\n'
               << "preconditioner_minimum_diagonal = "
               << results.preconditioner_minimum_diagonal << '\n'
               << "trunc_chol_conservative_memory_estimate_bytes = "
               << results.trunc_chol_conservative_memory_estimate_bytes << '\n'
               << "trunc_chol_memory_limit_bytes = "
               << results.trunc_chol_memory_limit_bytes << '\n';
    }

    // Preserve the actual stopping policy beside every computed solution so
    // performance and error changes can be attributed to the correct sweep.
    if (results.computed_solution)
    {
        output << "outer_rtol = " << results.tolerances.outer_rtol << '\n'
               << "outer_atol = " << results.tolerances.outer_atol << '\n'
               << "outer_dtol = " << results.tolerances.outer_dtol << '\n'
               << "outer_max_it = "
               << results.tolerances.outer_max_it << '\n';
        if (results.lor_pc == LORPCMode::left_gmres_cg ||
            results.lor_pc == LORPCMode::right_gmres_cg ||
            results.lor_pc == LORPCMode::right_fgmres_cg)
            output << "inner_rtol = "
                   << results.tolerances.inner_rtol << '\n'
                   << "inner_atol = "
                   << results.tolerances.inner_atol << '\n'
                   << "inner_dtol = "
                   << results.tolerances.inner_dtol << '\n'
                   << "inner_max_it = "
                   << results.tolerances.inner_max_it << '\n';
        else if (results.lor_pc == LORPCMode::left_gmres_fixed_jacobi)
            output << "fixed_jacobi_steps = "
                   << results.fixed_jacobi.steps << '\n'
                   << "fixed_jacobi_omega = "
                   << results.fixed_jacobi.omega << '\n';
    }

    // Record predicted tensor structure and PETSc's post-assembly storage so
    // remote batches can detect reallocations and memory regressions.
    output << "assembled(K) = " << results.assembled_high_matrix << '\n'
           << "predicted_nnz(K) = " << results.predicted_nonzeros_K << '\n'
           << "nnz_K = " << results.used_nonzeros_K << '\n'
           << "used_nnz(K) = " << results.used_nonzeros_K << '\n'
           << "allocated_nnz(K) = " << results.allocated_nonzeros_K << '\n'
           << "allocation_calls(K) = " << results.allocation_calls_K << '\n'
           << "memory_bytes(K) = " << results.memory_bytes_K << '\n'
           << "estimated_minimum_bytes(K) = "
           << results.estimated_minimum_bytes_K << '\n'
           << "assembled(K_LOR) = " << results.assembled_lor_matrix << '\n';
    if (results.assembled_lor_matrix)
        output << "predicted_nnz(K_LOR) = "
               << results.predicted_nonzeros_lor << '\n'
               << "used_nnz(K_LOR) = "
               << results.used_nonzeros_lor << '\n'
               << "allocated_nnz(K_LOR) = "
               << results.allocated_nonzeros_lor << '\n'
               << "allocation_calls(K_LOR) = "
               << results.allocation_calls_lor << '\n'
               << "memory_bytes(K_LOR) = "
               << results.memory_bytes_lor << '\n'
               << "estimated_minimum_bytes(K_LOR) = "
               << results.estimated_minimum_bytes_lor << '\n'
               << "nnz_P = " << results.used_nonzeros_lor << '\n'
               << "memory_bytes_P = " << results.memory_bytes_lor << '\n';
    output << "memory_bytes_K = " << results.memory_bytes_K << '\n'
           << "time_original_matrix_assembly_s = "
           << results.time_original_matrix_assembly_s << '\n'
           << "time_preconditioner_matrix_construction_s = "
           << results.time_preconditioner_matrix_construction_s << '\n'
           << "time_1d_mass_assembly_s = "
           << results.time_1d_mass_assembly_s << '\n'
           << "time_1d_full_cholesky_s = "
           << results.time_1d_full_cholesky_s << '\n'
           << "time_1d_factor_truncation_s = "
           << results.time_1d_factor_truncation_s << '\n'
           << "time_truncation_certificate_s = "
           << results.time_truncation_certificate_s << '\n'
           << "time_1d_reduced_operators_s = "
           << results.time_1d_reduced_operators_s << '\n'
           << "time_tensor_pc_assembly_s = "
           << results.time_tensor_pc_assembly_s << '\n'
           << "time_tensor_memory_preflight_s = "
           << results.time_tensor_memory_preflight_s << '\n'
           << "time_tensor_identity_verification_s = "
           << results.time_tensor_identity_verification_s << '\n'
           << "time_spectral_eps_s = "
           << results.time_spectral_eps_s << '\n'
           << "time_singular_svd_s = "
           << results.time_singular_svd_s << '\n'
           << "time_l2_error_s = " << results.time_l2_error_s << '\n'
           << "time_case_total_s = " << results.time_case_total_s << '\n';
    // Keep the ideal exact-LOR spectral operator separate from the actual
    // algorithm label used by iterative solve and error diagnostics.
    const bool left =
        results.preconditioner_side == PreconditionerSide::left;
    const char *ideal_operator =
        results.lor_pc == LORPCMode::left_gmres_k_jacobi ?
        "D_K^{-1}K" :
        (results.preconditioner_matrix_source ==
            PreconditionerMatrixSource::truncated_cholesky_tensor ?
            (left ? "P_k^{-1}K" : "KP_k^{-1}") :
            (left ? "K_LOR^{-1}K" : "KK_LOR^{-1}"));
    const char *solve_label = lor_pc_mode_name(results.lor_pc);
    const char *outer_ksp_type = results.lor_pc == LORPCMode::left_cg_lu ?
        "cg" :
        ((results.lor_pc == LORPCMode::right_fgmres_cg ||
          results.lor_pc == LORPCMode::right_fgmres_amg ||
          results.lor_pc == LORPCMode::right_fgmres_jacobi) ?
            "fgmres" : "gmres");
    const char *inverse_application =
        (results.lor_pc == LORPCMode::left_cg_lu ||
         results.lor_pc == LORPCMode::right_gmres_lu) ? "lu" :
        ((results.lor_pc == LORPCMode::right_fgmres_amg ||
          results.lor_pc == LORPCMode::left_gmres_amg) ? "amg" :
         (results.lor_pc == LORPCMode::right_fgmres_jacobi ? "jacobi" :
         (uses_inner_cg ?
            (results.inner_pc == InnerPCMode::jacobi ?
                "cg_jacobi" : "cg_gamg") :
            (results.lor_pc == LORPCMode::left_gmres_fixed_jacobi ?
                "fixed_jacobi" : "direct_K_pc"))));
    output << "outer_ksp_type = " << outer_ksp_type << '\n'
           << "inverse_application_mode = " << inverse_application << '\n';
    if (!results.computed_condition_number)
        output << "svd_status = not_computed\n"
               << "svd_status_K = not_computed\n";
    if (results.computed_condition_number)
    {
        output << "svd_status = "
               << (results.singular_condition_converged ?
                   "converged" : "not_converged") << '\n'
               << "svd_status_K = "
               << (results.original_singular_condition_converged ?
                   "converged" : "not_converged") << '\n'
               << "svd_iterations_min = "
               << results.svd_iterations_min << '\n'
               << "svd_iterations_max = "
               << results.svd_iterations_max << '\n'
               << "svd_reason_min = " << results.svd_reason_min << '\n'
               << "svd_reason_max = " << results.svd_reason_max << '\n';
        if (results.original_singular_condition_converged)
            output << "cond_2(K) = " << results.condition_K << '\n';
        if (results.singular_condition_converged)
            output << "cond_2(" << ideal_operator << ") = "
                   << results.condition_lor_K << '\n'
                   << "kappa_sigma = " << results.condition_lor_K << '\n'
                   << "sigma_min = " << results.sigma_min << '\n'
                   << "sigma_max = " << results.sigma_max << '\n';
    }
    if (!results.computed_spectral_condition_number)
        output << "eigensolver_status = not_computed\n";
    if (results.computed_spectral_condition_number)
    {
        output << "eigensolver_status = "
               << (results.spectral_condition_converged ?
                   "converged" : "not_converged") << '\n'
               << "eps_iterations_min = "
               << results.eps_iterations_min << '\n'
               << "eps_iterations_max = "
               << results.eps_iterations_max << '\n'
               << "eps_reason_min = " << results.eps_reason_min << '\n'
               << "eps_reason_max = " << results.eps_reason_max << '\n'
               << "time_eps_lambda_min_s = "
               << results.time_eps_lambda_min_s << '\n'
               << "time_eps_lambda_max_s = "
               << results.time_eps_lambda_max_s << '\n';
        if (results.spectral_condition_converged)
            output << "eps_condition_number = "
                   << results.kappa_lambda << '\n'
                   << "kappa_lambda = " << results.kappa_lambda << '\n'
                   << "lambda_min = " << results.lambda_min << '\n'
                   << "lambda_max = " << results.lambda_max << '\n'
                   << "eps_relative_error_min = "
                   << results.lambda_relative_error_min << '\n'
                   << "eps_relative_error_max = "
                   << results.lambda_relative_error_max << '\n'
                   << "eps_min_convergence_criterion = "
                   << (results.lambda_min_converged_by_stability ?
                       "ritz_stability" : "residual") << '\n'
                   << "eps_min_stability_window = "
                   << results.lambda_min_stability_window << '\n'
                   << "eps_min_stability_relative_span = "
                   << results.lambda_min_stability_relative_span << '\n'
                   << "eps_min_stability_residual_limit = "
                   << results.lambda_min_stability_residual_limit << '\n';
    }
    if (results.computed_solution)
    {
        if (results.computed_unpreconditioned_solution)
            output << "iterations(K) = " << results.iterations_K << '\n'
                   << "setup_time_seconds(K) = "
                   << results.setup_time_K << '\n'
                   << "outer_solve_time_seconds(K) = "
                   << results.outer_solve_time_K << '\n'
                   << "residual(K) = " << results.residual_K << '\n'
                   << "relative_residual(K) = "
                   << results.relative_residual_K << '\n'
                   << "ksp_reason(K) = "
                   << static_cast<int>(results.reason_K) << '\n';
        output << "iterations(" << solve_label << ") = "
               << results.iterations_lor_K << '\n'
               << "setup_time_seconds(" << solve_label << ") = "
               << results.setup_time_lor_K << '\n'
               << "preconditioner_setup_time_seconds(" << solve_label
               << ") = " << results.preconditioner_setup_time_lor_K << '\n'
               << "outer_solve_time_seconds(" << solve_label
               << ") = " << results.outer_solve_time_lor_K << '\n'
               << "residual(" << solve_label << ") = "
               << results.residual_lor_K << '\n'
               << "relative_residual(" << solve_label << ") = "
               << results.relative_residual_lor_K << '\n'
               << "outer_iterations = " << results.iterations_lor_K << '\n'
               << "pc_applications = "
               << results.pc_applications_lor_K << '\n'
               << "relative_true_residual = "
               << results.relative_residual_lor_K << '\n'
               << "ksp_reason = "
               << static_cast<int>(results.reason_lor_K) << '\n'
               << "time_pc_ksp_setup_s = "
               << results.preconditioner_setup_time_lor_K << '\n'
               << "time_outer_solve_s = "
               << results.outer_solve_time_lor_K << '\n'
               << "time_pc_apply_total_s = "
               << results.inner_solve_time_lor_K << '\n';
        if (results.lor_pc == LORPCMode::left_gmres_cg ||
            results.lor_pc == LORPCMode::right_gmres_cg ||
            results.lor_pc == LORPCMode::right_fgmres_cg)
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
            output << "inner_iterations_total = "
                   << results.inner_iterations_lor_K << '\n'
                   << "pc_applications(" << solve_label << ") = "
                   << results.pc_applications_lor_K << '\n'
                   << "inner_iterations_total(" << solve_label
                   << ") = " << results.inner_iterations_lor_K << '\n'
                   << "inner_iterations_average(" << solve_label
                   << ") = " << average_iterations << '\n'
                   << "inner_iterations_maximum(" << solve_label
                   << ") = " << results.maximum_inner_iterations_lor_K << '\n'
                   << "inner_failed_applications(" << solve_label
                   << ") = " << results.failed_inner_applications_lor_K << '\n'
                   << "inner_last_reason(" << solve_label
                   << ") = " << static_cast<int>(
                        results.last_inner_reason_lor_K) << '\n'
                   << "inner_solve_time_seconds(" << solve_label
                   << ") = " << results.inner_solve_time_lor_K << '\n'
                   << "inner_solve_time_average_seconds("
                   << solve_label << ") = "
                   << average_time << '\n';
            if (results.inner_pc == InnerPCMode::gamg)
                output << "amg_setup_time_seconds = "
                       << results.preconditioner_setup_time_lor_K << '\n'
                       << "amg_vcycle_applications_status = not_recorded\n";
        }
        else if (results.lor_pc == LORPCMode::right_fgmres_amg ||
                 results.lor_pc == LORPCMode::left_gmres_amg)
        {
            const double applications = static_cast<double>(
                results.pc_applications_lor_K);
            const double average_time = applications > 0.0 ?
                results.inner_solve_time_lor_K / applications : 0.0;
            output << "inner_iterations_status = not_applicable\n"
                   << "inner_cg_iterations_status = not_applicable\n"
                   << "amg_vcycle_applications = "
                   << results.pc_applications_lor_K << '\n'
                   << "amg_setup_time_seconds = "
                   << results.preconditioner_setup_time_lor_K << '\n'
                   << "amg_apply_time_seconds = "
                   << results.inner_solve_time_lor_K << '\n'
                   << "amg_apply_time_average_seconds = "
                   << average_time << '\n';
        }
        else if (results.lor_pc == LORPCMode::right_fgmres_jacobi)
        {
            const double applications = static_cast<double>(
                results.pc_applications_lor_K);
            const double average_time = applications > 0.0 ?
                results.inner_solve_time_lor_K / applications : 0.0;
            output << "inner_iterations_status = not_applicable\n"
                   << "inner_cg_iterations_status = not_applicable\n"
                   << "jacobi_applications = "
                   << results.pc_applications_lor_K << '\n'
                   << "jacobi_setup_time_seconds = "
                   << results.preconditioner_setup_time_lor_K << '\n'
                   << "jacobi_apply_time_seconds = "
                   << results.inner_solve_time_lor_K << '\n'
                   << "jacobi_apply_time_average_seconds = "
                   << average_time << '\n';
        }
        else if (results.lor_pc == LORPCMode::left_cg_lu ||
                 results.lor_pc == LORPCMode::right_gmres_lu)
            output << "inner_iterations_status = not_applicable\n"
                   << "inner_cg_iterations_status = not_applicable\n"
                   << "lu_triangular_solve_applications = "
                   << results.pc_applications_lor_K << '\n'
                   << "lu_factor_setup_time_seconds = "
                   << results.preconditioner_setup_time_lor_K << '\n'
                   << "lu_apply_time_seconds = "
                   << results.inner_solve_time_lor_K << '\n';
        else if (results.lor_pc == LORPCMode::left_gmres_fixed_jacobi)
        {
            // Scheme B reports deterministic polynomial work separately from
            // Krylov iterations while reusing the common timing counters.
            output << "pc_applications(" << solve_label << ") = "
                   << results.pc_applications_lor_K << '\n'
                   << "fixed_jacobi_steps_total(" << solve_label
                   << ") = " << results.inner_iterations_lor_K << '\n'
                   << "fixed_jacobi_time_seconds(" << solve_label
                   << ") = " << results.inner_solve_time_lor_K << '\n';
        }
    }
    if (results.computed_l2_error)
    {
        if (results.computed_unpreconditioned_solution)
            output << "L2_error(K) = " << results.l2_error_K << '\n';
        else
            output << "L2_error(K)_status = not_computed\n";
        output << "L2_error(" << solve_label << ") = "
               << results.l2_error_lor_K << '\n'
               << "l2_error = " << results.l2_error_lor_K << '\n';
    }
}

// Emit fixed-dimensional option adapters used by the one runtime driver.
template RefinementRequest<2> make_refinement_request<2>(
    const ExperimentOptions &);
template RefinementRequest<3> make_refinement_request<3>(
    const ExperimentOptions &);

} // namespace lor
