#ifndef LOR_IO_EXPERIMENT_IO_HPP
#define LOR_IO_EXPERIMENT_IO_HPP

#include "Solver/LinearSolver.hpp"
#include "Mesh/Refinement.hpp"

#include <petscsys.h>

#include <string>
#include <vector>

namespace lor
{

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
    LORPCMode lor_pc = LORPCMode::left_gmres_fixed_jacobi;
    PreconditionerSide preconditioner_side = PreconditionerSide::left;
    InnerPCMode inner_pc = InnerPCMode::jacobi;

    // Independent experiment features. Error analysis consumes solution
    // vectors and therefore requires compute_solution; condition numbers use
    // only the two assembled stiffness matrices.
    bool compute_solution = true;
    bool compute_l2_error = true;
    bool compute_condition_number = false;

    // KSP stopping controls and independent SLEPc singular-value controls.
    // Inner tolerances apply only to *_cg, and fixed_jacobi applies only to
    // left_gmres_fixed_jacobi.
    SolverTolerances tolerances;
    FixedJacobiParameters fixed_jacobi;
    PetscReal svd_tolerance = 1.0e-8;
    PetscInt svd_max_iterations = 100000;
    PetscInt svd_subspace_dimension = 100;
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

    // Record which optional computations actually ran. Disabled quantities
    // are omitted from the result file rather than written as artificial zero.
    bool computed_solution = false;
    bool computed_l2_error = false;
    bool computed_condition_number = false;

    // Preconditioner application and side used consistently by the solve and
    // labels. inner_pc is emitted as metadata only for a *_cg configuration.
    LORPCMode lor_pc = LORPCMode::left_gmres_fixed_jacobi;
    PreconditionerSide preconditioner_side = PreconditionerSide::left;
    InnerPCMode inner_pc = InnerPCMode::jacobi;

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
    PetscLogDouble inner_solve_time_lor_K = 0.0;
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
