#include "Solver/LinearSolver.hpp"

#include <stdexcept>

namespace lor
{
namespace
{

// State owned by the GMRES PCSHELL. Each application approximately solves the
// LOR system with an inner Jacobi-preconditioned CG iteration.
struct GMRESShellContext
{
    KSP inner_ksp = nullptr;
    PetscInt applications = 0;
    PetscInt total_inner_iterations = 0;
};

PetscErrorCode apply_gmres_shell(PC pc, Vec residual, Vec correction)
{
    PetscFunctionBeginUser;
    GMRESShellContext *context = nullptr;
    PetscCall(PCShellGetContext(pc, &context));
    PetscCall(KSPSolve(context->inner_ksp, residual, correction));
    PetscInt iterations = 0;
    PetscCall(KSPGetIterationNumber(context->inner_ksp, &iterations));
    ++context->applications;
    context->total_inner_iterations += iterations;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode destroy_gmres_shell(PC pc)
{
    PetscFunctionBeginUser;
    GMRESShellContext *context = nullptr;
    PetscCall(PCShellGetContext(pc, &context));
    if (context != nullptr)
    {
        PetscCall(KSPDestroy(&context->inner_ksp));
        PetscCall(PetscFree(context));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode solve(Mat stiffness, Mat lor_stiffness, Vec load,
    const LORPCMode mode, const SolverTolerances &tolerances,
    LinearSolveResult *result)
{
    PetscFunctionBeginUser;
    const MPI_Comm comm = PetscObjectComm(
        reinterpret_cast<PetscObject>(stiffness));
    PetscCheck(result != nullptr, comm, PETSC_ERR_ARG_NULL,
        "LinearSolveResult must not be null");
    PetscCheck(result->solution == nullptr, comm,
        PETSC_ERR_ARG_WRONGSTATE, "LinearSolveResult already owns a solution");

    // Configure the outer method and common stopping tolerances.
    const bool use_shell = mode == LORPCMode::gmres_shell;
    KSP ksp = nullptr;
    PetscCall(KSPCreate(comm, &ksp));
    PetscCall(KSPSetOperators(ksp, stiffness,
        lor_stiffness != nullptr && !use_shell ? lor_stiffness : stiffness));
    PetscCall(KSPSetType(ksp, use_shell ? KSPGMRES : KSPCG));
    if (use_shell)
        PetscCall(KSPGMRESSetRestart(ksp, 200));
    PetscCall(KSPSetNormType(ksp, use_shell ?
        KSP_NORM_PRECONDITIONED : KSP_NORM_UNPRECONDITIONED));
    PetscCall(KSPSetTolerances(ksp, tolerances.outer_rtol,
        tolerances.outer_atol, tolerances.outer_dtol,
        tolerances.outer_max_it));
    PetscCall(KSPSetErrorIfNotConverged(ksp, PETSC_TRUE));

    // Attach no PC, an exact LOR LU, or the nested shell preconditioner.
    PC pc = nullptr;
    PetscCall(KSPGetPC(ksp, &pc));
    GMRESShellContext *shell_context = nullptr;
    if (lor_stiffness == nullptr)
        PetscCall(PCSetType(pc, PCNONE));
    else if (!use_shell)
        PetscCall(PCSetType(pc, PCLU));
    else
    {
        PetscCall(PCSetType(pc, PCSHELL));
        PetscCall(PetscNew(&shell_context));
        PetscCall(KSPCreate(comm, &shell_context->inner_ksp));
        PetscCall(KSPSetOperators(shell_context->inner_ksp,
            lor_stiffness, lor_stiffness));
        PetscCall(KSPSetType(shell_context->inner_ksp, KSPCG));
        PetscCall(KSPSetInitialGuessNonzero(
            shell_context->inner_ksp, PETSC_FALSE));
        PetscCall(KSPSetTolerances(shell_context->inner_ksp,
            tolerances.inner_rtol, tolerances.inner_atol,
            tolerances.inner_dtol, tolerances.inner_max_it));
        PC inner_pc = nullptr;
        PetscCall(KSPGetPC(shell_context->inner_ksp, &inner_pc));
        PetscCall(PCSetType(inner_pc, PCJACOBI));
        PetscCall(PCShellSetContext(pc, shell_context));
        PetscCall(PCShellSetApply(pc, apply_gmres_shell));
        PetscCall(PCShellSetDestroy(pc, destroy_gmres_shell));
    }

    // Solve from zero and retain the solution for subsequent error analysis.
    PetscCall(VecDuplicate(load, &result->solution));
    PetscCall(VecSet(result->solution, 0.0));
    PetscCall(KSPSolve(ksp, load, result->solution));
    PetscCall(KSPGetIterationNumber(ksp, &result->iterations));
    PetscCall(KSPGetConvergedReason(ksp, &result->reason));
    if (shell_context != nullptr)
    {
        result->pc_applications = shell_context->applications;
        result->inner_iterations = shell_context->total_inner_iterations;
    }

    // Recompute the same physical residual norm for every solver mode.
    Vec residual = nullptr;
    PetscCall(VecDuplicate(load, &residual));
    PetscCall(MatMult(stiffness, result->solution, residual));
    PetscCall(VecAYPX(residual, -1.0, load));
    PetscCall(VecNorm(residual, NORM_2, &result->residual_norm));
    PetscCall(VecDestroy(&residual));
    PetscCall(KSPDestroy(&ksp));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

LORPCMode parse_lor_pc_mode(const std::string &name)
{
    if (name == "cg_lu")
        return LORPCMode::cg_lu;
    if (name == "gmres_shell")
        return LORPCMode::gmres_shell;
    throw std::invalid_argument("lor_pc must be cg_lu or gmres_shell");
}

const char *lor_pc_mode_name(const LORPCMode mode) noexcept
{
    return mode == LORPCMode::cg_lu ? "cg_lu" : "gmres_shell";
}

PetscErrorCode solve_unpreconditioned(Mat stiffness, Vec load,
    const LORPCMode mode, const SolverTolerances &tolerances,
    LinearSolveResult *result)
{
    return solve(stiffness, nullptr, load, mode, tolerances, result);
}

PetscErrorCode solve_with_lor(Mat stiffness, Mat lor_stiffness, Vec load,
    const LORPCMode mode, const SolverTolerances &tolerances,
    LinearSolveResult *result)
{
    return solve(stiffness, lor_stiffness, load, mode, tolerances, result);
}

PetscErrorCode destroy_linear_solve_result(LinearSolveResult *result)
{
    PetscFunctionBeginUser;
    if (result != nullptr)
        PetscCall(VecDestroy(&result->solution));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace lor
