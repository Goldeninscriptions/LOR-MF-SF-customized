#ifndef LOR_SOLVER_LINEAR_SOLVER_HPP
#define LOR_SOLVER_LINEAR_SOLVER_HPP

#include "Assembly/PreconditionerSide.hpp"

#include <petscksp.h>

#include <string>

namespace lor
{

// Complete solver/preconditioner configurations exposed to users. Each stable
// name records the preconditioning side, outer Krylov method, source matrix,
// and inverse approximation, so incompatible combinations cannot be requested.
enum class LORPCMode
{
    left_cg_lu,
    right_gmres_lu,
    left_gmres_cg,
    right_gmres_cg,
    right_fgmres_cg,
    left_gmres_k_jacobi,
    left_gmres_k_gamg,
    right_gmres_k_gamg,
    left_gmres_fixed_jacobi
};

// Algebraic preconditioner used inside each approximate K_LOR CG solve.
// Jacobi is the legacy baseline; GAMG builds a reusable multigrid hierarchy.
enum class InnerPCMode
{
    jacobi,
    gamg
};

// Relative, absolute, divergence, and iteration limits for the two nested KSP
// levels. Inner values apply only to the three schemes ending in _cg.
struct SolverTolerances
{
    // Stopping thresholds and iteration limit for the outer K or PC(K) solve.
    // Defaults reproduce the effective Parallel-Computing GMRES settings.
    PetscReal outer_rtol = 1.0e-10;
    PetscReal outer_atol = 1.0e-10;
    PetscReal outer_dtol = 1.0e4;
    PetscInt outer_max_it = 10000;

    // Each inner K_LOR CG application uses the same legacy thresholds by
    // default; LU and fixed-Jacobi schemes ignore these values.
    PetscReal inner_rtol = 1.0e-10;
    PetscReal inner_atol = 1.0e-10;
    PetscReal inner_dtol = 1.0e4;
    PetscInt inner_max_it = 10000;
};

// Fixed coefficients defining scheme B's linear weighted-Jacobi polynomial.
// Starting from zero, every preconditioner application executes exactly steps
// updates with the same dimensionless relaxation factor omega.
struct FixedJacobiParameters
{
    PetscInt steps = 4;
    PetscReal omega = 2.0 / 3.0;
};

struct LinearSolveResult
{
    // Owned solution vector. Release it with destroy_linear_solve_result().
    Vec solution = nullptr;

    // Number of iterations taken by the outer CG or GMRES method.
    PetscInt iterations = 0;

    // PCSHELL application count and accumulated inner-CG iterations or fixed
    // Jacobi steps. Both remain zero for every non-shell preconditioner.
    PetscInt pc_applications = 0;
    PetscInt inner_iterations = 0;

    // Maximum-rank wall times in seconds. setup_time_seconds measures explicit
    // KSP/PC setup; outer_solve_time_seconds measures the complete outer
    // KSPSolve and therefore includes every nested preconditioner application.
    PetscLogDouble setup_time_seconds = 0.0;
    PetscLogDouble outer_solve_time_seconds = 0.0;

    // Maximum-rank sum of time spent inside all inner CG solves or fixed
    // Jacobi applications. It is nested within outer_solve_time_seconds, not
    // additional elapsed time, and remains zero for non-shell preconditioners.
    PetscLogDouble inner_solve_time_seconds = 0.0;

    // Common physical residual ||F-Ku||_2 recomputed after convergence.
    PetscReal residual_norm = 0.0;

    // Scale-independent physical residual ||F-Ku||_2/||F||_2. For a zero
    // right-hand side this stores the absolute residual instead.
    PetscReal relative_residual_norm = 0.0;

    // PETSc termination code for the outer KSP.
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
};

LORPCMode parse_lor_pc_mode(const std::string &name);
const char *lor_pc_mode_name(LORPCMode mode) noexcept;
PreconditionerSide lor_pc_mode_side(LORPCMode mode) noexcept;

// Report whether the selected configuration needs an assembled K_LOR. The
// direct K-Jacobi and K-GAMG reference modes operate only on stiffness.
bool lor_pc_mode_uses_lor_matrix(LORPCMode mode) noexcept;

// Parse and print the stable inner-preconditioner spellings shared by command
// line options, result metadata, and batch scripts.
InnerPCMode parse_inner_pc_mode(const std::string &name);
const char *inner_pc_mode_name(InnerPCMode mode) noexcept;

// Parse and print the stable left/right spellings shared by PETSc options,
// batch metadata, console summaries, and result files.
PreconditionerSide parse_preconditioner_side(const std::string &name);
const char *preconditioner_side_name(PreconditionerSide side) noexcept;

// Solve K u = F without a preconditioner, using the outer method associated
// with mode so the baseline and preconditioned runs share stopping settings.
PetscErrorCode solve_unpreconditioned(Mat stiffness, Vec load,
    LORPCMode mode, const SolverTolerances &tolerances,
    LinearSolveResult *result);

// Solve K u = F with the complete configuration selected by mode. The mode
// fixes the side and outer method; inner_pc_mode affects only schemes ending
// in _cg, fixed_jacobi affects only left_gmres_fixed_jacobi, and the K-only
// reference modes accept a null lor_stiffness.
PetscErrorCode solve_with_lor(Mat stiffness, Mat lor_stiffness, Vec load,
    LORPCMode mode, InnerPCMode inner_pc_mode,
    const SolverTolerances &tolerances,
    const FixedJacobiParameters &fixed_jacobi,
    LinearSolveResult *result);

PetscErrorCode destroy_linear_solve_result(LinearSolveResult *result);

} // namespace lor

#endif
