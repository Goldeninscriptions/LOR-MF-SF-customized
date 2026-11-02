#include "Solver/LinearSolver.hpp"

#include <stdexcept>

namespace lor
{
namespace
{

enum class ShellApplyKind
{
    inner_cg,
    fixed_jacobi
};

// State owned by the outer PCSHELL. It applies either a variable inner-CG
// solve or scheme B's fixed linear weighted-Jacobi polynomial.
struct LORShellContext
{
    ShellApplyKind kind = ShellApplyKind::inner_cg;
    Mat lor_stiffness = nullptr; // Non-owning; outlives the outer KSP.
    KSP inner_ksp = nullptr;
    Vec diagonal_inverse = nullptr;
    Vec work = nullptr;
    PetscInt fixed_jacobi_steps = 0;
    PetscReal fixed_jacobi_omega = 0.0;
    PetscInt applications = 0;
    PetscInt total_inner_iterations = 0;
    PetscLogDouble total_inner_solve_time_seconds = 0.0;
};

PetscErrorCode initialize_fixed_jacobi(LORShellContext *context)
{
    PetscFunctionBeginUser;
    PetscCheck(context != nullptr, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
        "LOR shell context must not be null");

    // Cache D^{-1} once. Positivity follows from the assembled SPD LOR
    // stiffness matrix and prevents an undefined pointwise reciprocal.
    PetscCall(MatCreateVecs(context->lor_stiffness,
        &context->diagonal_inverse, nullptr));
    PetscCall(MatGetDiagonal(context->lor_stiffness,
        context->diagonal_inverse));
    PetscReal minimum_diagonal = 0.0;
    PetscCall(VecMin(context->diagonal_inverse, nullptr,
        &minimum_diagonal));
    PetscCheck(minimum_diagonal > 0.0,
        PetscObjectComm(reinterpret_cast<PetscObject>(context->lor_stiffness)),
        PETSC_ERR_ARG_WRONG, "LOR diagonal must be strictly positive");
    PetscCall(VecReciprocal(context->diagonal_inverse));
    PetscCall(VecDuplicate(context->diagonal_inverse, &context->work));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode apply_lor_shell(PC pc, Vec residual, Vec correction)
{
    PetscFunctionBeginUser;
    LORShellContext *context = nullptr;
    PetscCall(PCShellGetContext(pc, &context));

    // Time only the repeated approximate-inverse application; inner KSP or
    // diagonal setup has already been charged to the setup phase.
    PetscLogDouble start = 0.0;
    PetscLogDouble end = 0.0;
    PetscCall(PetscTime(&start));
    PetscInt iterations = 0;
    if (context->kind == ShellApplyKind::inner_cg)
    {
        PetscCall(KSPSolve(context->inner_ksp, residual, correction));
        PetscCall(KSPGetIterationNumber(context->inner_ksp, &iterations));
    }
    else
    {
        // Apply the same degree-(m-1) linear polynomial B_m to every input:
        // z <- z + omega D^{-1}(r-Pz), starting from z=0 for exactly m steps.
        PetscCall(VecSet(correction, 0.0));
        for (PetscInt step = 0;
             step < context->fixed_jacobi_steps; ++step)
        {
            PetscCall(MatMult(context->lor_stiffness, correction,
                context->work));
            PetscCall(VecAYPX(context->work, -1.0, residual));
            PetscCall(VecPointwiseMult(context->work,
                context->diagonal_inverse, context->work));
            PetscCall(VecAXPY(correction, context->fixed_jacobi_omega,
                context->work));
        }
        iterations = context->fixed_jacobi_steps;
    }
    PetscCall(PetscTime(&end));

    // Accumulate comparable application, work-step, and nested-time totals.
    ++context->applications;
    context->total_inner_iterations += iterations;
    context->total_inner_solve_time_seconds += end - start;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode destroy_lor_shell(PC pc)
{
    PetscFunctionBeginUser;
    LORShellContext *context = nullptr;
    PetscCall(PCShellGetContext(pc, &context));
    if (context != nullptr)
    {
        PetscCall(KSPDestroy(&context->inner_ksp));
        PetscCall(VecDestroy(&context->diagonal_inverse));
        PetscCall(VecDestroy(&context->work));
        PetscCall(PetscFree(context));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode solve(Mat stiffness, Mat lor_stiffness, Vec load,
    const LORPCMode mode,
    const bool apply_preconditioner,
    const InnerPCMode inner_pc_mode,
    const SolverTolerances &tolerances,
    const FixedJacobiParameters &fixed_jacobi,
    LinearSolveResult *result)
{
    PetscFunctionBeginUser;
    const MPI_Comm comm = PetscObjectComm(
        reinterpret_cast<PetscObject>(stiffness));
    PetscCheck(result != nullptr, comm, PETSC_ERR_ARG_NULL,
        "LinearSolveResult must not be null");
    PetscCheck(result->solution == nullptr, comm,
        PETSC_ERR_ARG_WRONGSTATE, "LinearSolveResult already owns a solution");

    const bool has_lor = lor_stiffness != nullptr;
    const bool use_exact_lu = mode == LORPCMode::left_cg_lu ||
        mode == LORPCMode::right_gmres_lu;
    const bool use_inner_cg = mode == LORPCMode::left_gmres_cg ||
        mode == LORPCMode::right_gmres_cg ||
        mode == LORPCMode::right_fgmres_cg;
    const bool use_fixed_jacobi =
        mode == LORPCMode::left_gmres_fixed_jacobi;
    const bool use_k_jacobi =
        mode == LORPCMode::left_gmres_k_jacobi;
    const bool use_k_gamg =
        mode == LORPCMode::left_gmres_k_gamg ||
        mode == LORPCMode::right_gmres_k_gamg;
    const bool use_shell = use_inner_cg || use_fixed_jacobi;
    const bool use_fgmres = apply_preconditioner &&
        mode == LORPCMode::right_fgmres_cg;
    const bool use_gmres = mode != LORPCMode::left_cg_lu;
    const PreconditionerSide side = lor_pc_mode_side(mode);
    PetscCheck(fixed_jacobi.steps > 0, comm, PETSC_ERR_ARG_OUTOFRANGE,
        "fixed_jacobi_steps must be positive");
    PetscCheck(fixed_jacobi.omega > 0.0 && fixed_jacobi.omega < 1.0,
        comm, PETSC_ERR_ARG_OUTOFRANGE,
        "fixed_jacobi_omega must lie strictly between zero and one");
    PetscCheck(!apply_preconditioner || !lor_pc_mode_uses_lor_matrix(mode) ||
        has_lor, comm, PETSC_ERR_ARG_NULL,
        "Selected LOR preconditioner requires an assembled K_LOR matrix");

    // Use standard GMRES for the fixed linear map and FGMRES only for A1's
    // variable inner solve. The unpreconditioned baseline remains GMRES.
    KSP ksp = nullptr;
    PetscCall(KSPCreate(comm, &ksp));
    PetscCall(KSPSetOperators(ksp, stiffness,
        apply_preconditioner && use_exact_lu ? lor_stiffness : stiffness));
    PetscCall(KSPSetType(ksp, use_fgmres ? KSPFGMRES :
        (use_gmres ? KSPGMRES : KSPCG)));
    if (use_gmres)
        PetscCall(KSPGMRESSetRestart(ksp, 200));

    // Match the paper and Parallel-Computing default with explicit PC_LEFT;
    // retain PC_RIGHT as an independently selectable research path. PETSc
    // GMRES monitors the compatible preconditioned or physical residual norm.
    if (apply_preconditioner)
        PetscCall(KSPSetPCSide(ksp,
            side == PreconditionerSide::left ? PC_LEFT : PC_RIGHT));
    PetscCall(KSPSetNormType(ksp,
        apply_preconditioner && side == PreconditionerSide::left ?
            KSP_NORM_PRECONDITIONED : KSP_NORM_UNPRECONDITIONED));
    PetscCall(KSPSetTolerances(ksp, tolerances.outer_rtol,
        tolerances.outer_atol, tolerances.outer_dtol,
        tolerances.outer_max_it));
    PetscCall(KSPSetErrorIfNotConverged(ksp, PETSC_TRUE));

    // Attach no PC, a direct algebraic PC on K, exact LOR LU, or a LOR shell.
    PC pc = nullptr;
    PetscCall(KSPGetPC(ksp, &pc));
    LORShellContext *shell_context = nullptr;
    if (!apply_preconditioner)
        PetscCall(PCSetType(pc, PCNONE));
    else if (use_k_jacobi)
        PetscCall(PCSetType(pc, PCJACOBI));
    else if (use_k_gamg)
        PetscCall(PCSetType(pc, PCGAMG));
    else if (!use_shell)
        PetscCall(PCSetType(pc, PCLU));
    else
    {
        PetscCall(PCSetType(pc, PCSHELL));
        PetscCall(PetscNew(&shell_context));
        shell_context->lor_stiffness = lor_stiffness;
        if (use_inner_cg)
        {
            // Configure A1/legacy variable inner CG and its reusable algebraic
            // Jacobi or GAMG preconditioner.
            shell_context->kind = ShellApplyKind::inner_cg;
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
            PetscCall(PCSetType(inner_pc,
                inner_pc_mode == InnerPCMode::jacobi ? PCJACOBI : PCGAMG));
        }
        else
        {
            // Store fixed scheme-B coefficients; D^{-1} is created inside the
            // explicitly timed setup phase below.
            shell_context->kind = ShellApplyKind::fixed_jacobi;
            shell_context->fixed_jacobi_steps = fixed_jacobi.steps;
            shell_context->fixed_jacobi_omega = fixed_jacobi.omega;
        }
        PetscCall(PCShellSetContext(pc, shell_context));
        PetscCall(PCShellSetApply(pc, apply_lor_shell));
        PetscCall(PCShellSetDestroy(pc, destroy_lor_shell));
    }

    // Set up all nested solvers explicitly and record the maximum-rank setup
    // wall time separately from the iterative solve phase.
    PetscCallMPI(MPI_Barrier(comm));
    PetscLogDouble setup_start = 0.0;
    PetscLogDouble setup_end = 0.0;
    PetscCall(PetscTime(&setup_start));
    if (shell_context != nullptr && shell_context->inner_ksp != nullptr)
        PetscCall(KSPSetUp(shell_context->inner_ksp));
    else if (shell_context != nullptr)
        PetscCall(initialize_fixed_jacobi(shell_context));
    PetscCall(KSPSetUp(ksp));
    PetscCall(PetscTime(&setup_end));
    const PetscLogDouble local_setup_time = setup_end - setup_start;
    PetscCallMPI(MPI_Allreduce(&local_setup_time,
        &result->setup_time_seconds, 1, MPI_DOUBLE, MPI_MAX, comm));

    // Solve from zero and retain the solution for subsequent error analysis.
    // A preconditioned outer time includes every nested inner CG application.
    PetscCall(VecDuplicate(load, &result->solution));
    PetscCall(VecSet(result->solution, 0.0));
    PetscCallMPI(MPI_Barrier(comm));
    PetscLogDouble solve_start = 0.0;
    PetscLogDouble solve_end = 0.0;
    PetscCall(PetscTime(&solve_start));
    PetscCall(KSPSolve(ksp, load, result->solution));
    PetscCall(PetscTime(&solve_end));
    const PetscLogDouble local_outer_time = solve_end - solve_start;
    PetscCallMPI(MPI_Allreduce(&local_outer_time,
        &result->outer_solve_time_seconds, 1, MPI_DOUBLE, MPI_MAX, comm));

    // Collect rank-consistent iteration counts and maximum-rank nested cost.
    PetscCall(KSPGetIterationNumber(ksp, &result->iterations));
    PetscCall(KSPGetConvergedReason(ksp, &result->reason));
    if (shell_context != nullptr)
    {
        PetscCallMPI(MPI_Allreduce(&shell_context->applications,
            &result->pc_applications, 1, MPIU_INT, MPI_MAX, comm));
        PetscCallMPI(MPI_Allreduce(&shell_context->total_inner_iterations,
            &result->inner_iterations, 1, MPIU_INT, MPI_MAX, comm));
        PetscCallMPI(MPI_Allreduce(
            &shell_context->total_inner_solve_time_seconds,
            &result->inner_solve_time_seconds, 1,
            MPI_DOUBLE, MPI_MAX, comm));
    }

    // Recompute the same absolute and right-hand-side-scaled physical
    // residuals for every mode rather than trusting a projected KSP estimate.
    Vec residual = nullptr;
    PetscCall(VecDuplicate(load, &residual));
    PetscCall(MatMult(stiffness, result->solution, residual));
    PetscCall(VecAYPX(residual, -1.0, load));
    PetscCall(VecNorm(residual, NORM_2, &result->residual_norm));
    PetscReal load_norm = 0.0;
    PetscCall(VecNorm(load, NORM_2, &load_norm));
    result->relative_residual_norm = load_norm > 0.0 ?
        result->residual_norm / load_norm : result->residual_norm;
    PetscCall(VecDestroy(&residual));
    PetscCall(KSPDestroy(&ksp));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

// Translate one complete user-facing solver configuration into a typed mode.
LORPCMode parse_lor_pc_mode(const std::string &name)
{
    if (name == "left_cg_lu")
        return LORPCMode::left_cg_lu;
    if (name == "right_gmres_lu")
        return LORPCMode::right_gmres_lu;
    if (name == "left_gmres_cg")
        return LORPCMode::left_gmres_cg;
    if (name == "right_gmres_cg")
        return LORPCMode::right_gmres_cg;
    if (name == "right_fgmres_cg")
        return LORPCMode::right_fgmres_cg;
    if (name == "left_gmres_k_jacobi")
        return LORPCMode::left_gmres_k_jacobi;
    if (name == "left_gmres_k_gamg")
        return LORPCMode::left_gmres_k_gamg;
    if (name == "right_gmres_k_gamg")
        return LORPCMode::right_gmres_k_gamg;
    if (name == "left_gmres_fixed_jacobi")
        return LORPCMode::left_gmres_fixed_jacobi;
    throw std::invalid_argument(
        "lor_pc must be left_cg_lu, right_gmres_lu, left_gmres_cg, "
        "right_gmres_cg, right_fgmres_cg, left_gmres_k_jacobi, "
        "left_gmres_k_gamg, right_gmres_k_gamg, or "
        "left_gmres_fixed_jacobi");
}

// Preserve a stable complete configuration name in logs and result files.
const char *lor_pc_mode_name(const LORPCMode mode) noexcept
{
    switch (mode)
    {
        case LORPCMode::left_cg_lu:
            return "left_cg_lu";
        case LORPCMode::right_gmres_lu:
            return "right_gmres_lu";
        case LORPCMode::left_gmres_cg:
            return "left_gmres_cg";
        case LORPCMode::right_gmres_cg:
            return "right_gmres_cg";
        case LORPCMode::right_fgmres_cg:
            return "right_fgmres_cg";
        case LORPCMode::left_gmres_k_jacobi:
            return "left_gmres_k_jacobi";
        case LORPCMode::left_gmres_k_gamg:
            return "left_gmres_k_gamg";
        case LORPCMode::right_gmres_k_gamg:
            return "right_gmres_k_gamg";
        case LORPCMode::left_gmres_fixed_jacobi:
            return "left_gmres_fixed_jacobi";
    }
    return "unknown";
}

// Derive the algebraic side from the configuration rather than accepting a
// second option that could contradict its name.
PreconditionerSide lor_pc_mode_side(const LORPCMode mode) noexcept
{
    return mode == LORPCMode::right_gmres_lu ||
        mode == LORPCMode::right_gmres_cg ||
        mode == LORPCMode::right_fgmres_cg ||
        mode == LORPCMode::right_gmres_k_gamg ?
        PreconditionerSide::right : PreconditionerSide::left;
}

// Separate K-only reference schemes from configurations requiring K_LOR.
bool lor_pc_mode_uses_lor_matrix(const LORPCMode mode) noexcept
{
    return mode != LORPCMode::left_gmres_k_jacobi &&
        mode != LORPCMode::left_gmres_k_gamg &&
        mode != LORPCMode::right_gmres_k_gamg;
}

// Translate the user-facing low-order solver choice into a typed mode.
InnerPCMode parse_inner_pc_mode(const std::string &name)
{
    if (name == "jacobi")
        return InnerPCMode::jacobi;
    if (name == "gamg")
        return InnerPCMode::gamg;
    throw std::invalid_argument("inner_pc must be jacobi or gamg");
}

// Provide one stable inner-PC spelling for logs and result files.
const char *inner_pc_mode_name(const InnerPCMode mode) noexcept
{
    return mode == InnerPCMode::jacobi ? "jacobi" : "gamg";
}

// Translate the user-facing algebraic direction at the IO/Solver boundary.
PreconditionerSide parse_preconditioner_side(const std::string &name)
{
    if (name == "left")
        return PreconditionerSide::left;
    if (name == "right")
        return PreconditionerSide::right;
    throw std::invalid_argument(
        "preconditioner_side must be left or right");
}

// Provide one stable text spelling for logs, result files, and batch metadata.
const char *preconditioner_side_name(
    const PreconditionerSide side) noexcept
{
    return side == PreconditionerSide::left ? "left" : "right";
}

PetscErrorCode solve_unpreconditioned(Mat stiffness, Vec load,
    const LORPCMode mode, const SolverTolerances &tolerances,
    LinearSolveResult *result)
{
    const FixedJacobiParameters unused_fixed_jacobi;
    return solve(stiffness, nullptr, load, mode, false,
        InnerPCMode::jacobi, tolerances,
        unused_fixed_jacobi, result);
}

PetscErrorCode solve_with_lor(Mat stiffness, Mat lor_stiffness, Vec load,
    const LORPCMode mode, const InnerPCMode inner_pc_mode,
    const SolverTolerances &tolerances,
    const FixedJacobiParameters &fixed_jacobi,
    LinearSolveResult *result)
{
    return solve(stiffness, lor_stiffness, load, mode, true, inner_pc_mode,
        tolerances, fixed_jacobi, result);
}

PetscErrorCode destroy_linear_solve_result(LinearSolveResult *result)
{
    PetscFunctionBeginUser;
    if (result != nullptr)
        PetscCall(VecDestroy(&result->solution));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace lor
