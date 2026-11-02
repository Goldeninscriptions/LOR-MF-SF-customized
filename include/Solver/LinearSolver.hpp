#ifndef LOR_SOLVER_LINEAR_SOLVER_HPP
#define LOR_SOLVER_LINEAR_SOLVER_HPP

#include "Assembly/PreconditionerSide.hpp"

#include <petscksp.h>

#include <string>

namespace lor
{

enum class LORPCMode
{
    // Exact LU application of the SPD LOR matrix; the outer method is CG for
    // left preconditioning and GMRES for the optional right configuration.
    cg_lu,

    // Outer restarted GMRES with a selectable left/right PCSHELL and inner
    // CG/Jacobi LOR solves.
    gmres_shell
};

// Relative, absolute, divergence, and iteration limits for the two nested KSP
// levels. Inner values are ignored by cg_lu because its LOR solve is direct.
struct SolverTolerances
{
    // Stopping thresholds and iteration limit for the outer K or PC(K) solve.
    PetscReal outer_rtol = 1.0e-12;
    PetscReal outer_atol = 1.0e-14;
    PetscReal outer_dtol = 1.0e8;
    PetscInt outer_max_it = 10000;

    // Stopping thresholds and limit for each inner K_LOR CG application.
    PetscReal inner_rtol = 1.0e-12;
    PetscReal inner_atol = 1.0e-14;
    PetscReal inner_dtol = 1.0e4;
    PetscInt inner_max_it = 10000;
};

struct LinearSolveResult
{
    // Owned solution vector. Release it with destroy_linear_solve_result().
    Vec solution = nullptr;

    // Number of iterations taken by the outer CG or GMRES method.
    PetscInt iterations = 0;

    // PCSHELL application count and accumulated inner-CG iteration count.
    // Both remain zero in an unpreconditioned or cg_lu solve.
    PetscInt pc_applications = 0;
    PetscInt inner_iterations = 0;

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

// Parse and print the stable left/right spellings shared by PETSc options,
// batch metadata, console summaries, and result files.
PreconditionerSide parse_preconditioner_side(const std::string &name);
const char *preconditioner_side_name(PreconditionerSide side) noexcept;

// Solve K u = F without a preconditioner, using the outer method associated
// with mode so the baseline and preconditioned runs share stopping settings.
PetscErrorCode solve_unpreconditioned(Mat stiffness, Vec load,
    LORPCMode mode, const SolverTolerances &tolerances,
    LinearSolveResult *result);

// Solve K u = F with either LU(K_LOR) or the legacy-style
// GMRES+PCSHELL+inner-CG/Jacobi LOR application. The independently selected
// side determines whether PETSc realizes K_LOR^{-1}K or K K_LOR^{-1}.
PetscErrorCode solve_with_lor(Mat stiffness, Mat lor_stiffness, Vec load,
    LORPCMode mode, PreconditionerSide side,
    const SolverTolerances &tolerances,
    LinearSolveResult *result);

PetscErrorCode destroy_linear_solve_result(LinearSolveResult *result);

} // namespace lor

#endif
