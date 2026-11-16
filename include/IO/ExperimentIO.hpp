#ifndef LOR_IO_EXPERIMENT_IO_HPP
#define LOR_IO_EXPERIMENT_IO_HPP

#include "Solver/LinearSolver.hpp"
#include "Mesh/Refinement.hpp"

#include <petscsys.h>

#include <string>
#include <vector>

namespace lor
{

enum class PreconditionerMatrixSource
{
    lor_q1,
    truncated_cholesky_tensor
};

PreconditionerMatrixSource parse_preconditioner_matrix_source(
    const std::string &name);
const char *preconditioner_matrix_source_name(
    PreconditionerMatrixSource source) noexcept;

struct ExperimentOptions
{
    // Geometry description read by GeometryInput and result-file target.
    std::string geometry_file;
    std::string output_file;

    // Refinement policy selected from p_refine, h_refine, or k_refine.
    std::string refinement_mode = "k_refine";

    // Requested common final degree and final element counts. The active
    // refinement policy validates which entries are allowed to change.
    int degree = 0;
    std::vector<int> elements;

    // LOR point strategy, inverse-application hierarchy, algebraic side, and
    // complete preconditioned solver configuration and optional nested PC.
    // inner_pc is meaningful only for the three modes ending in _cg; the
    // default mode instead uses a fixed weighted-Jacobi polynomial inverse.
    std::string lor_points = "greville";
    PreconditionerMatrixSource preconditioner_matrix_source =
        PreconditionerMatrixSource::lor_q1;
    int trunc_chol_k = -1; // -1 means auto = ceil(p/2).
    bool trunc_chol_k_was_auto = true;
    bool compute_truncation_certificate = true;
    bool trunc_chol_boundary_protection = true;
    bool verify_tensor_identity = false;
    PetscLogDouble trunc_chol_max_memory_bytes =
        64.0 * 1024.0 * 1024.0 * 1024.0;
    LORPCMode lor_pc = LORPCMode::left_gmres_fixed_jacobi;
    PreconditionerSide preconditioner_side = PreconditionerSide::left;
    InnerPCMode inner_pc = InnerPCMode::jacobi;

    // Independent experiment features. Error analysis consumes solution
    // vectors and therefore requires compute_solution; condition numbers use
    // only the two assembled stiffness matrices.
    bool compute_solution = true;
    bool compute_unpreconditioned_solution = true;
    bool compute_l2_error = true;
    bool compute_condition_number = false;
    bool compute_spectral_condition_number = false;

    // KSP stopping controls and independent SLEPc singular-value controls.
    // Inner tolerances apply only to *_cg, and fixed_jacobi applies only to
    // left_gmres_fixed_jacobi.
    SolverTolerances tolerances;
    FixedJacobiParameters fixed_jacobi;
    PetscReal svd_tolerance = 1.0e-8;
    PetscInt svd_max_iterations = 100000;
    PetscInt svd_subspace_dimension = 100;
    PetscReal eps_tolerance = 1.0e-8;
    PetscInt eps_max_iterations = 100000;
    PetscInt eps_subspace_dimension = 100;
};

// Self-describing experiment record used by both direct and batch workflows.
// The two errors compare solutions of the same K u=F system; they may differ
// only because the solver modes terminate at different tolerances.
struct ExperimentResults
{
    // Actual launch and case metadata queried at runtime, never copied blindly
    // from requested command-line targets.
    std::string refinement_mode;
    int mpi_ranks = 1;
    int actual_degree = 0;
    std::vector<int> actual_elements;
    PetscInt total_free_dofs = 0;

    // Record which optional computations actually ran. Disabled quantities
    // are omitted from the result file rather than written as artificial zero.
    bool computed_solution = false;
    bool computed_unpreconditioned_solution = false;
    bool computed_l2_error = false;
    bool computed_condition_number = false;
    bool computed_spectral_condition_number = false;

    // Preconditioner application and side used consistently by the solve and
    // labels. inner_pc is emitted as metadata only for a *_cg configuration.
    LORPCMode lor_pc = LORPCMode::left_gmres_fixed_jacobi;
    PreconditionerSide preconditioner_side = PreconditionerSide::left;
    InnerPCMode inner_pc = InnerPCMode::jacobi;
    PreconditionerMatrixSource preconditioner_matrix_source =
        PreconditionerMatrixSource::lor_q1;

    bool trunc_chol_k_was_auto = true;
    int trunc_chol_k_effective = -1;
    int trunc_chol_original_degree_p = 0;
    int trunc_chol_mass_degree_q = 0;
    bool trunc_chol_boundary_protection = true;
    std::vector<int> trunc_chol_value_original_half_bandwidth;
    std::vector<int> trunc_chol_value_complete_factor_half_bandwidth;
    std::vector<int> trunc_chol_value_truncated_factor_half_bandwidth;
    std::vector<int> trunc_chol_derivative_original_half_bandwidth;
    std::vector<int> trunc_chol_derivative_complete_factor_half_bandwidth;
    std::vector<int> trunc_chol_derivative_truncated_factor_half_bandwidth;
    std::vector<std::size_t> trunc_chol_value_complete_factor_nonzeros;
    std::vector<std::size_t> trunc_chol_value_truncated_factor_nonzeros;
    std::vector<std::size_t> trunc_chol_value_dropped_factor_entries;
    std::vector<std::size_t> trunc_chol_derivative_complete_factor_nonzeros;
    std::vector<std::size_t> trunc_chol_derivative_truncated_factor_nonzeros;
    std::vector<std::size_t> trunc_chol_derivative_dropped_factor_entries;
    std::vector<double> trunc_chol_beta_value_mass;
    std::vector<double> trunc_chol_beta_derivative_mass;
    std::vector<double> trunc_chol_certificate_value_maximum_row_sum;
    std::vector<double> trunc_chol_certificate_value_maximum_column_sum;
    std::vector<double> trunc_chol_certificate_derivative_maximum_row_sum;
    std::vector<double> trunc_chol_certificate_derivative_maximum_column_sum;
    std::vector<bool> trunc_chol_beta_below_one;
    std::size_t trunc_chol_dropped_entries = 0;
    std::size_t trunc_chol_complete_factor_nonzeros = 0;
    std::size_t trunc_chol_truncated_factor_nonzeros = 0;
    bool trunc_chol_certificate_computed = false;
    bool trunc_chol_certificate_bound_available = false;
    PetscReal trunc_chol_theoretical_kappa_lambda_bound = 0.0;
    PetscReal tensor_identity_relative_error = 0.0;
    bool tensor_identity_verification_computed = false;
    bool tensor_identity_verification_passed = false;
    PetscLogDouble trunc_chol_conservative_memory_estimate_bytes = 0.0;
    PetscLogDouble trunc_chol_memory_limit_bytes = 0.0;
    PetscReal derivative_identity_max_relative_error = 0.0;
    bool preconditioner_symmetric = false;
    PetscReal preconditioner_minimum_diagonal = 0.0;

    // Actual outer and inner KSP stopping controls used by this run. Keeping
    // them in the result record makes tolerance-sweep files self-describing.
    SolverTolerances tolerances;
    FixedJacobiParameters fixed_jacobi;

    // Global PETSc storage diagnostics for the assembled high-order and
    // optional LOR stiffness matrices. K-only reference modes skip K_LOR.
    bool assembled_lor_matrix = false;
    PetscInt predicted_nonzeros_K = 0;
    PetscLogDouble allocated_nonzeros_K = 0.0;
    PetscLogDouble used_nonzeros_K = 0.0;
    PetscLogDouble allocation_calls_K = 0.0;
    PetscLogDouble memory_bytes_K = 0.0;
    PetscLogDouble estimated_minimum_bytes_K = 0.0;
    PetscInt predicted_nonzeros_lor = 0;
    PetscLogDouble allocated_nonzeros_lor = 0.0;
    PetscLogDouble used_nonzeros_lor = 0.0;
    PetscLogDouble allocation_calls_lor = 0.0;
    PetscLogDouble memory_bytes_lor = 0.0;
    PetscLogDouble estimated_minimum_bytes_lor = 0.0;

    // Unpreconditioned high-order stiffness diagnostics, including physical
    // absolute and right-hand-side-scaled residual norms.
    PetscReal condition_K = 0.0;
    PetscInt iterations_K = 0;
    PetscReal residual_K = 0.0;
    PetscReal relative_residual_K = 0.0;
    PetscReal l2_error_K = 0.0;
    PetscLogDouble setup_time_K = 0.0;
    PetscLogDouble outer_solve_time_K = 0.0;

    // Selected-side preconditioned diagnostics for the same high-order
    // equation. The preconditioner may use K_LOR or be formed directly on K.
    PetscReal condition_lor_K = 0.0;
    PetscInt iterations_lor_K = 0;
    PetscReal residual_lor_K = 0.0;
    PetscReal relative_residual_lor_K = 0.0;
    PetscReal l2_error_lor_K = 0.0;
    PetscLogDouble setup_time_lor_K = 0.0;
    PetscLogDouble outer_solve_time_lor_K = 0.0;

    // Nested preconditioner work. The work counter means CG iterations for
    // *_cg schemes and deterministic Jacobi steps for the fixed-linear scheme;
    // counts and time are maximum-cost-rank totals.
    PetscInt pc_applications_lor_K = 0;
    PetscInt inner_iterations_lor_K = 0;
    PetscInt maximum_inner_iterations_lor_K = 0;
    PetscInt failed_inner_applications_lor_K = 0;
    KSPConvergedReason last_inner_reason_lor_K = KSP_CONVERGED_ITERATING;
    PetscLogDouble inner_solve_time_lor_K = 0.0;
    PetscLogDouble preconditioner_setup_time_lor_K = 0.0;
    KSPConvergedReason reason_K = KSP_CONVERGED_ITERATING;
    KSPConvergedReason reason_lor_K = KSP_CONVERGED_ITERATING;

    // New source-independent and truncated-source phase timings. MPI phases
    // are rank-maximum wall times.
    PetscLogDouble time_original_matrix_assembly_s = 0.0;
    PetscLogDouble time_1d_mass_assembly_s = 0.0;
    PetscLogDouble time_1d_full_cholesky_s = 0.0;
    PetscLogDouble time_1d_factor_truncation_s = 0.0;
    PetscLogDouble time_truncation_certificate_s = 0.0;
    PetscLogDouble time_1d_reduced_operators_s = 0.0;
    PetscLogDouble time_tensor_pc_assembly_s = 0.0;
    PetscLogDouble time_tensor_memory_preflight_s = 0.0;
    PetscLogDouble time_tensor_identity_verification_s = 0.0;
    PetscLogDouble time_spectral_eps_s = 0.0;
    PetscLogDouble time_singular_svd_s = 0.0;
    PetscLogDouble time_l2_error_s = 0.0;
    PetscLogDouble time_case_total_s = 0.0;

    bool spectral_condition_converged = false;
    PetscReal kappa_lambda = 0.0;
    PetscReal lambda_min = 0.0;
    PetscReal lambda_max = 0.0;
    PetscReal lambda_relative_error_min = 0.0;
    PetscReal lambda_relative_error_max = 0.0;
    PetscInt eps_iterations_min = 0;
    PetscInt eps_iterations_max = 0;
    int eps_reason_min = 0;
    int eps_reason_max = 0;

    bool singular_condition_converged = false;
    bool original_singular_condition_converged = false;
    PetscReal sigma_min = 0.0;
    PetscReal sigma_max = 0.0;
    PetscInt svd_iterations_min = 0;
    PetscInt svd_iterations_max = 0;
    int svd_reason_min = 0;
    int svd_reason_max = 0;
};

// Read the experiment command line after PETSc/SLEPc initialization.
std::string read_geometry_filename();
ExperimentOptions read_experiment_options(int dimension);

// Convert dynamic command-line p/n/mode fields into the fixed-dimensional,
// IO-independent Mesh request consumed by apply_refinement.
template <int Dim>
RefinementRequest<Dim> make_refinement_request(
    const ExperimentOptions &options);

extern template RefinementRequest<2> make_refinement_request<2>(
    const ExperimentOptions &);
extern template RefinementRequest<3> make_refinement_request<3>(
    const ExperimentOptions &);

// Write actual refined-space metadata and only the enabled numerical results,
// with one key-value item on every line.
void write_experiment_results(const std::string &filename,
    const ExperimentResults &results);

} // namespace lor

#endif
