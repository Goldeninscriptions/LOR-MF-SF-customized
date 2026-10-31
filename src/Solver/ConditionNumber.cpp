#include "Solver/ConditionNumber.hpp"

#include <petscksp.h>
#include <slepcsvd.h>

#include <algorithm>

namespace lor
{
namespace
{

struct LeftPreconditionedContext
{
    Mat stiffness = nullptr;
    KSP lor_solver = nullptr;
    Vec work = nullptr;
};

PetscErrorCode apply_left_preconditioned(Mat matrix, Vec x, Vec y)
{
    PetscFunctionBeginUser;
    LeftPreconditionedContext *context = nullptr;
    PetscCall(MatShellGetContext(matrix, &context));
    PetscCall(MatMult(context->stiffness, x, context->work));
    PetscCall(KSPSolve(context->lor_solver, context->work, y));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode apply_left_preconditioned_transpose(Mat matrix, Vec x, Vec y)
{
    PetscFunctionBeginUser;
    LeftPreconditionedContext *context = nullptr;
    PetscCall(MatShellGetContext(matrix, &context));
    PetscCall(KSPSolve(context->lor_solver, x, context->work));
    PetscCall(MatMult(context->stiffness, context->work, y));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode extreme_singular_value(Mat matrix, const SVDWhich which,
    const char *prefix, const PetscReal tolerance,
    const PetscInt max_iterations,
    const PetscInt requested_subspace_dimension, PetscReal &sigma)
{
    PetscFunctionBeginUser;

    // Every SVD object must live on the same communicator as its distributed
    // operator; using PETSC_COMM_SELF here would give each rank an unrelated
    // serial spectral problem.
    const MPI_Comm comm = PetscObjectComm(
        reinterpret_cast<PetscObject>(matrix));

    // Limit the requested Krylov subspace to the operator dimension while
    // retaining at least one expansion vector beyond the requested triplet.
    PetscInt rows = 0;
    PetscInt columns = 0;
    PetscCall(MatGetSize(matrix, &rows, &columns));
    const PetscInt operator_dimension = std::min(rows, columns);
    const PetscInt subspace_dimension =
        std::min(requested_subspace_dimension, operator_dimension);
    PetscCheck(subspace_dimension >= 2, comm,
        PETSC_ERR_ARG_SIZ,
        "SVD operator must have dimension at least two");

    // Configure a two-sided TRLanczos solve with a substantially larger
    // iteration allowance for ill-conditioned operators. The enlarged search
    // space is reserved for the difficult smallest-singular-value solve.
    SVD svd = nullptr;
    PetscCall(SVDCreate(comm, &svd));
    PetscCall(SVDSetOptionsPrefix(svd, prefix));
    PetscCall(SVDSetOperators(svd, matrix, nullptr));
    PetscCall(SVDSetProblemType(svd, SVD_STANDARD));
    PetscCall(SVDSetType(svd, SVDTRLANCZOS));
    PetscCall(SVDTRLanczosSetOneSide(svd, PETSC_FALSE));
    if (which == SVD_SMALLEST)
        PetscCall(SVDSetDimensions(svd, 1, subspace_dimension,
            subspace_dimension - 1));
    else
        PetscCall(SVDSetDimensions(svd, 1,
            PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(SVDSetTolerances(svd, tolerance, max_iterations));
    PetscCall(SVDSetWhichSingularTriplets(svd, which));
    PetscCall(SVDSetFromOptions(svd));
    PetscCall(SVDSolve(svd));

    PetscInt converged = 0;
    SVDConvergedReason reason = SVD_CONVERGED_ITERATING;
    PetscCall(SVDGetConverged(svd, &converged));
    PetscCall(SVDGetConvergedReason(svd, &reason));
    PetscCheck(converged > 0 && reason > 0, comm,
        PETSC_ERR_NOT_CONVERGED,
        "SLEPc singular-value solve with prefix '%s' did not converge",
        prefix);
    PetscCall(SVDGetSingularTriplet(svd, 0, &sigma, nullptr, nullptr));
    PetscCall(SVDDestroy(&svd));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode condition_number(Mat matrix, const char *largest_prefix,
    const char *smallest_prefix, const PetscReal tolerance,
    const PetscInt max_iterations,
    const PetscInt subspace_dimension,
    ConditionNumberResult &result)
{
    PetscFunctionBeginUser;
    const MPI_Comm comm = PetscObjectComm(
        reinterpret_cast<PetscObject>(matrix));
    PetscCall(extreme_singular_value(matrix, SVD_LARGEST, largest_prefix,
        tolerance, max_iterations, subspace_dimension, result.sigma_max));
    PetscCall(extreme_singular_value(matrix, SVD_SMALLEST, smallest_prefix,
        tolerance, max_iterations, subspace_dimension, result.sigma_min));
    PetscCheck(result.sigma_min > 0.0, comm, PETSC_ERR_PLIB,
        "SLEPc reported a nonpositive minimum singular value");
    result.condition_number = result.sigma_max / result.sigma_min;
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

PetscErrorCode compute_stiffness_condition_numbers(Mat stiffness,
    Mat lor_stiffness, const PetscReal svd_tolerance,
    const PetscInt max_iterations,
    const PetscInt subspace_dimension,
    ConditionNumberResult *original, ConditionNumberResult *preconditioned)
{
    PetscFunctionBeginUser;
    const MPI_Comm comm = PetscObjectComm(
        reinterpret_cast<PetscObject>(stiffness));
    PetscCheck(original != nullptr && preconditioned != nullptr,
        comm, PETSC_ERR_ARG_NULL,
        "Condition-number results must not be null");
    PetscCheck(svd_tolerance > 0.0, comm, PETSC_ERR_ARG_OUTOFRANGE,
        "SVD tolerance must be positive");
    PetscCheck(max_iterations > 0, comm,
        PETSC_ERR_ARG_OUTOFRANGE,
        "SVD maximum iteration count must be positive");
    PetscCheck(subspace_dimension >= 2, comm,
        PETSC_ERR_ARG_OUTOFRANGE,
        "SVD subspace dimension must be at least two");

    PetscInt high_rows = 0;
    PetscInt high_columns = 0;
    PetscInt lor_rows = 0;
    PetscInt lor_columns = 0;
    PetscCall(MatGetSize(stiffness, &high_rows, &high_columns));
    PetscCall(MatGetSize(lor_stiffness, &lor_rows, &lor_columns));
    PetscCheck(high_rows == high_columns && lor_rows == lor_columns &&
        high_rows == lor_rows, comm, PETSC_ERR_ARG_SIZ,
        "K and K_LOR must be square matrices of the same size");

    // Analyze K directly.
    PetscCall(condition_number(stiffness, "cond_k_max_", "cond_k_min_",
        svd_tolerance, max_iterations, subspace_dimension, *original));

    // Create an exact reusable LOR solve solely for the spectral MatShell.
    LeftPreconditionedContext context;
    context.stiffness = stiffness;
    PetscCall(KSPCreate(comm, &context.lor_solver));
    PetscCall(KSPSetOperators(context.lor_solver,
        lor_stiffness, lor_stiffness));
    PetscCall(KSPSetType(context.lor_solver, KSPPREONLY));
    PC pc = nullptr;
    PetscCall(KSPGetPC(context.lor_solver, &pc));
    PetscCall(PCSetType(pc, PCLU));

    // The spectral experiment requires the action of the exact inverse of
    // K_LOR. For more than one rank, MUMPS supplies the distributed sparse LU
    // factorization; a PETSc build without MUMPS cannot perform this exact
    // parallel condition-number experiment.
    PetscMPIInt comm_size = 1;
    PetscCallMPI(MPI_Comm_size(comm, &comm_size));
    if (comm_size > 1)
    {
#if defined(PETSC_HAVE_MUMPS)
        PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
#else
        SETERRQ(comm, PETSC_ERR_SUP,
            "Parallel cond_2(K_LOR^{-1}K) requires PETSc with MUMPS");
#endif
    }
    PetscCall(KSPSetErrorIfNotConverged(context.lor_solver, PETSC_TRUE));
    PetscCall(KSPSetUp(context.lor_solver));
    PetscCall(MatCreateVecs(stiffness, &context.work, nullptr));

    // Supply M=P^{-1}K and M^T=KP^{-1} to obtain the true 2-norm.
    Mat preconditioned_operator = nullptr;
    PetscInt high_local_rows = 0;
    PetscInt high_local_columns = 0;
    PetscCall(MatGetLocalSize(stiffness,
        &high_local_rows, &high_local_columns));
    PetscCall(MatCreateShell(comm, high_local_rows, high_local_columns,
        high_rows, high_columns, &context, &preconditioned_operator));
    PetscCall(MatShellSetOperation(preconditioned_operator, MATOP_MULT,
        reinterpret_cast<void (*)(void)>(apply_left_preconditioned)));
    PetscCall(MatShellSetOperation(preconditioned_operator,
        MATOP_MULT_TRANSPOSE,
        reinterpret_cast<void (*)(void)>(apply_left_preconditioned_transpose)));
    PetscCall(condition_number(preconditioned_operator,
        "cond_lor_k_max_", "cond_lor_k_min_", svd_tolerance,
        max_iterations, subspace_dimension,
        *preconditioned));

    PetscCall(MatDestroy(&preconditioned_operator));
    PetscCall(VecDestroy(&context.work));
    PetscCall(KSPDestroy(&context.lor_solver));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace lor
