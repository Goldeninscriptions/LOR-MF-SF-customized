#include "Solver/GeneralizedConditionNumber.hpp"

#include <slepceps.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace lor
{
namespace
{

enum class Extreme
{
    maximum,
    minimum
};

struct ExtremeResult
{
    PetscReal eigenvalue = 0.0;
    PetscReal relative_error = 0.0;
    PetscInt iterations = 0;
    int convergence_reason = 0;
    PetscLogDouble solve_time = 0.0;
    bool converged_by_stability = false;
    PetscReal stability_relative_span = 0.0;
    PetscReal stability_residual_limit = 0.0;
    PetscInt stability_window = 0;
};

struct StabilityConvergenceContext
{
    EPSConv base_test = EPS_CONV_REL;
    PetscReal eps_tolerance = 0.0;
    PetscReal relative_span_tolerance = 5.0e-6;
    PetscReal relative_residual_limit = 1.0e-2;
    PetscInt window = 100;
    PetscInt minimum_iterations = 300;
    PetscInt last_iteration = -1;
    std::vector<PetscReal> values;
    bool accepted = false;
    PetscReal accepted_relative_span = 0.0;
};

PetscErrorCode evaluate_base_convergence(EPS eps, const EPSConv test,
    const PetscScalar eigr, const PetscScalar eigi, const PetscReal residual,
    PetscReal *error)
{
    PetscFunctionBeginUser;
    switch (test)
    {
    case EPS_CONV_ABS:
        PetscCall(EPSConvergedAbsolute(
            eps, eigr, eigi, residual, error, nullptr));
        break;
    case EPS_CONV_NORM:
        PetscCall(EPSConvergedNorm(
            eps, eigr, eigi, residual, error, nullptr));
        break;
    case EPS_CONV_REL:
    default:
        PetscCall(EPSConvergedRelative(
            eps, eigr, eigi, residual, error, nullptr));
        break;
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode stability_convergence_test(EPS eps, const PetscScalar eigr,
    const PetscScalar eigi, const PetscReal residual, PetscReal *error,
    void *raw_context)
{
    PetscFunctionBeginUser;
    auto *context = static_cast<StabilityConvergenceContext *>(raw_context);
    PetscCheck(context != nullptr, PetscObjectComm(
        reinterpret_cast<PetscObject>(eps)), PETSC_ERR_ARG_NULL,
        "Ritz-value stability context must not be null");

    // Preserve the convergence test selected before installing this composite
    // criterion. The stability path can only make that test less conservative
    // for the first (wanted) Ritz value in a new outer iteration.
    PetscCall(evaluate_base_convergence(eps, context->base_test,
        eigr, eigi, residual, error));

    PetscInt iteration = 0;
    PetscCall(EPSGetIterationNumber(eps, &iteration));
    if (iteration == context->last_iteration || context->accepted)
        PetscFunctionReturn(PETSC_SUCCESS);
    context->last_iteration = iteration;

    const PetscReal value = PetscRealPart(eigr);
    const PetscReal imaginary = PetscAbsReal(PetscImaginaryPart(eigr)) +
        PetscAbsScalar(eigi);
    PetscReal relative_residual = std::numeric_limits<PetscReal>::infinity();
    PetscCall(EPSConvergedRelative(eps, eigr, eigi, residual,
        &relative_residual, nullptr));
    if (!std::isfinite(value) || value <= 0.0 ||
        !std::isfinite(imaginary) || imaginary != 0.0 ||
        !std::isfinite(relative_residual))
        PetscFunctionReturn(PETSC_SUCCESS);

    context->values.push_back(value);
    if (static_cast<PetscInt>(context->values.size()) > context->window)
        context->values.erase(context->values.begin());
    if (iteration < context->minimum_iterations ||
        static_cast<PetscInt>(context->values.size()) < context->window ||
        relative_residual > context->relative_residual_limit ||
        *error <= context->eps_tolerance)
        PetscFunctionReturn(PETSC_SUCCESS);

    const auto bounds = std::minmax_element(
        context->values.begin(), context->values.end());
    const PetscReal scale = std::max(PetscAbsReal(value), PETSC_SMALL);
    const PetscReal relative_span = (*bounds.second - *bounds.first) / scale;
    if (relative_span > context->relative_span_tolerance)
        PetscFunctionReturn(PETSC_SUCCESS);

    context->accepted = true;
    context->accepted_relative_span = relative_span;
    // EPS considers a Ritz pair converged when this estimate is below its
    // tolerance. The actual relative residual is recomputed and reported after
    // EPSSolve, so this synthetic value is never exposed as the final error.
    *error = 0.5 * context->eps_tolerance;
    PetscCall(PetscPrintf(PetscObjectComm(reinterpret_cast<PetscObject>(eps)),
        "[eigmin] accepted stable Ritz value at iteration %" PetscInt_FMT
        ": lambda=%.12e, window=%" PetscInt_FMT
        ", relative_span=%.6e, relative_error_estimate=%.6e\n",
        iteration, static_cast<double>(value), context->window,
        static_cast<double>(relative_span),
        static_cast<double>(relative_residual)));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode validate_compatible_matrices(Mat A, Mat M)
{
    PetscFunctionBeginUser;
    PetscCheck(A != nullptr, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
        "Generalized EPS matrix A must not be null");
    const MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(A));
    PetscBool symmetry_known_A = PETSC_FALSE;
    PetscBool symmetric_A = PETSC_FALSE;
    PetscCall(MatIsSymmetricKnown(A, &symmetry_known_A, &symmetric_A));
    PetscCheck(!symmetry_known_A || symmetric_A, comm, PETSC_ERR_ARG_WRONG,
        "EPS matrix A is marked nonsymmetric");
    if (M == nullptr)
    {
        if (!symmetry_known_A)
            PetscCall(PetscInfo(A,
                "EPS_HEP requires symmetric A; the matrix has no PETSc "
                "symmetry declaration\n"));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    int communicator_relation = MPI_UNEQUAL;
    PetscCallMPI(MPI_Comm_compare(comm,
        PetscObjectComm(reinterpret_cast<PetscObject>(M)),
        &communicator_relation));
    PetscCheck(communicator_relation == MPI_IDENT ||
        communicator_relation == MPI_CONGRUENT, comm, PETSC_ERR_ARG_NOTSAMECOMM,
        "A and M must use identical or congruent MPI communicators");

    PetscInt rows_A = 0;
    PetscInt columns_A = 0;
    PetscInt rows_M = 0;
    PetscInt columns_M = 0;
    PetscInt local_rows_A = 0;
    PetscInt local_columns_A = 0;
    PetscInt local_rows_M = 0;
    PetscInt local_columns_M = 0;
    PetscCall(MatGetSize(A, &rows_A, &columns_A));
    PetscCall(MatGetSize(M, &rows_M, &columns_M));
    PetscCall(MatGetLocalSize(A, &local_rows_A, &local_columns_A));
    PetscCall(MatGetLocalSize(M, &local_rows_M, &local_columns_M));
    PetscCheck(rows_A == columns_A && rows_M == columns_M &&
        rows_A == rows_M, comm, PETSC_ERR_ARG_SIZ,
        "A and M must have the same square global dimensions");
    PetscCheck(local_rows_A == local_rows_M &&
        local_columns_A == local_columns_M, comm, PETSC_ERR_ARG_SIZ,
        "A and M must have compatible local row and column layouts");

    PetscInt row_begin_A = 0;
    PetscInt row_end_A = 0;
    PetscInt row_begin_M = 0;
    PetscInt row_end_M = 0;
    PetscInt column_begin_A = 0;
    PetscInt column_end_A = 0;
    PetscInt column_begin_M = 0;
    PetscInt column_end_M = 0;
    PetscCall(MatGetOwnershipRange(A, &row_begin_A, &row_end_A));
    PetscCall(MatGetOwnershipRange(M, &row_begin_M, &row_end_M));
    PetscCall(MatGetOwnershipRangeColumn(A,
        &column_begin_A, &column_end_A));
    PetscCall(MatGetOwnershipRangeColumn(M,
        &column_begin_M, &column_end_M));
    PetscCheck(row_begin_A == row_begin_M && row_end_A == row_end_M &&
        column_begin_A == column_begin_M && column_end_A == column_end_M,
        comm, PETSC_ERR_ARG_SIZ,
        "A and M must use the same distributed ownership ranges");

    PetscBool symmetry_known_M = PETSC_FALSE;
    PetscBool symmetric_M = PETSC_FALSE;
    PetscCall(MatIsSymmetricKnown(M, &symmetry_known_M, &symmetric_M));
    PetscCheck(!symmetry_known_M || symmetric_M, comm, PETSC_ERR_ARG_WRONG,
        "M is marked nonsymmetric and is invalid for EPS_GHEP");
    if (!symmetry_known_A || !symmetry_known_M)
        PetscCall(PetscInfo(A,
            "EPS_GHEP requires symmetric positive-definite A and M; "
            "at least one matrix has no PETSc symmetry declaration\n"));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode solve_extreme(Mat A, Mat M, const Extreme extreme,
    const PetscReal tolerance, const PetscInt max_iterations,
    const PetscInt requested_subspace_dimension, ExtremeResult *result,
    const GeneralizedConditionKSPConfigurator configure_ksp,
    void *configure_ksp_context)
{
    PetscFunctionBeginUser;
    const MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(A));
    PetscCheck(result != nullptr, comm, PETSC_ERR_ARG_NULL,
        "Extreme generalized-eigenvalue result must not be null");
    PetscInt size = 0;
    PetscCall(MatGetSize(A, &size, nullptr));
    const PetscInt ncv = std::min(size, requested_subspace_dimension);
    PetscCheck(ncv >= 2, comm, PETSC_ERR_ARG_SIZ,
        "Generalized EPS operator must have dimension at least two");

    EPS eps = nullptr;
    PetscCall(EPSCreate(comm, &eps));
    PetscCall(EPSSetOptionsPrefix(eps,
        extreme == Extreme::maximum ? "eigmax_" : "eigmin_"));
    PetscCall(EPSSetOperators(eps, A, M));
    PetscCall(EPSSetProblemType(eps, M == nullptr ? EPS_HEP : EPS_GHEP));
    PetscCall(EPSSetType(eps, EPSKRYLOVSCHUR));
    PetscCall(EPSSetDimensions(eps, 1, ncv, PETSC_DEFAULT));
    PetscCall(EPSSetTolerances(eps, tolerance, max_iterations));

    PetscCall(EPSSetWhichEigenpairs(eps,
        extreme == Extreme::maximum ?
            EPS_LARGEST_REAL : EPS_SMALLEST_REAL));
    // Keep both extremes on the same generalized operator M^{-1}A. In this
    // B-spline/LOR application M is the sparse low-order matrix, whereas A is
    // much denser; zero-shift shift-and-invert would instead require repeated
    // solves with A and is therefore intentionally not the default.
    ST st = nullptr;
    PetscCall(EPSGetST(eps, &st));
    PetscCall(STSetType(st, STSHIFT));
    PetscCall(STSetShift(st, 0.0));

    // This call follows all programmatic defaults so every EPS/ST/KSP/PC
    // choice above can be independently overridden through the prefix.
    PetscCall(EPSSetFromOptions(eps));
    if (M != nullptr && configure_ksp != nullptr)
    {
        KSP ksp = nullptr;
        PetscCall(STGetKSP(st, &ksp));
        PetscCall(configure_ksp(ksp, configure_ksp_context));
    }

    StabilityConvergenceContext stability;
    PetscBool stability_enabled =
        extreme == Extreme::minimum ? PETSC_TRUE : PETSC_FALSE;
    const char *prefix = extreme == Extreme::maximum ? "eigmax_" : "eigmin_";
    PetscCall(PetscOptionsGetBool(nullptr, prefix,
        "-eps_stability_stop", &stability_enabled, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, prefix,
        "-eps_stability_window", &stability.window, nullptr));
    PetscCall(PetscOptionsGetReal(nullptr, prefix,
        "-eps_stability_tol", &stability.relative_span_tolerance, nullptr));
    PetscCall(PetscOptionsGetReal(nullptr, prefix,
        "-eps_stability_residual", &stability.relative_residual_limit,
        nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, prefix,
        "-eps_stability_min_it", &stability.minimum_iterations, nullptr));
    if (stability_enabled)
    {
        PetscCheck(extreme == Extreme::minimum, comm, PETSC_ERR_ARG_WRONG,
            "Ritz-value stability convergence is supported only for eigmin");
        PetscCheck(stability.window >= 2, comm, PETSC_ERR_ARG_OUTOFRANGE,
            "eigmin stability window must be at least two iterations");
        PetscCheck(stability.minimum_iterations >= stability.window, comm,
            PETSC_ERR_ARG_OUTOFRANGE,
            "eigmin stability minimum iteration must be at least the window");
        PetscCheck(stability.relative_span_tolerance > 0.0 &&
            std::isfinite(stability.relative_span_tolerance), comm,
            PETSC_ERR_ARG_OUTOFRANGE,
            "eigmin stability relative-span tolerance must be positive");
        PetscCheck(stability.relative_residual_limit > 0.0 &&
            std::isfinite(stability.relative_residual_limit), comm,
            PETSC_ERR_ARG_OUTOFRANGE,
            "eigmin stability residual limit must be positive");
        PetscInt configured_max_iterations = 0;
        PetscCall(EPSGetTolerances(
            eps, &stability.eps_tolerance, &configured_max_iterations));
        PetscCall(EPSGetConvergenceTest(eps, &stability.base_test));
        PetscCheck(stability.base_test != EPS_CONV_USER, comm,
            PETSC_ERR_SUP,
            "eigmin Ritz stability cannot wrap another user convergence test");
        stability.values.reserve(static_cast<std::size_t>(stability.window));
        PetscCall(EPSSetConvergenceTestFunction(eps,
            stability_convergence_test, &stability, nullptr));
    }
    PetscLogDouble local_start = 0.0;
    PetscLogDouble local_end = 0.0;
    PetscCall(PetscTime(&local_start));
    PetscCall(EPSSolve(eps));
    PetscCall(PetscTime(&local_end));
    const PetscLogDouble local_time = local_end - local_start;
    PetscCallMPI(MPI_Allreduce(&local_time, &result->solve_time, 1,
        MPI_DOUBLE, MPI_MAX, comm));

    PetscInt converged = 0;
    EPSConvergedReason reason = EPS_CONVERGED_ITERATING;
    PetscCall(EPSGetConverged(eps, &converged));
    PetscCall(EPSGetConvergedReason(eps, &reason));
    PetscCall(EPSGetIterationNumber(eps, &result->iterations));
    result->convergence_reason = static_cast<int>(reason);
    result->converged_by_stability = stability.accepted;
    result->stability_relative_span = stability.accepted_relative_span;
    result->stability_residual_limit = stability_enabled ?
        stability.relative_residual_limit : 0.0;
    result->stability_window = stability_enabled ? stability.window : 0;

    const char *name = extreme == Extreme::maximum ? "largest" : "smallest";
    if (reason <= 0 || converged < 1)
    {
        PetscCall(EPSDestroy(&eps));
        SETERRQ(comm, PETSC_ERR_NOT_CONVERGED,
            "%s generalized eigenvalue did not converge "
            "(reason=%d, converged eigenpairs=%" PetscInt_FMT ")",
            name, static_cast<int>(reason), converged);
    }

    PetscScalar eigenvalue = 0.0;
    PetscCall(EPSGetEigenpair(eps, 0, &eigenvalue, nullptr,
        nullptr, nullptr));
    result->eigenvalue = PetscRealPart(eigenvalue);
    PetscCall(EPSComputeError(eps, 0, EPS_ERROR_RELATIVE,
        &result->relative_error));
    PetscCall(EPSDestroy(&eps));

    const PetscReal imaginary_part = PetscImaginaryPart(eigenvalue);
    const PetscReal imaginary_scale = std::max<PetscReal>(
        1.0, PetscAbsReal(result->eigenvalue));
    PetscCheck(std::isfinite(result->eigenvalue) &&
        std::isfinite(imaginary_part) &&
        PetscAbsReal(imaginary_part) <= 10.0 * tolerance * imaginary_scale,
        comm, PETSC_ERR_FP,
        "The %s generalized eigenvalue is not finite and real", name);
    PetscCheck(result->eigenvalue > 0.0, comm, PETSC_ERR_ARG_WRONG,
        "The %s generalized eigenvalue is not strictly positive; "
        "the supplied matrices may violate the SPD assumptions", name);
    PetscCheck(std::isfinite(result->relative_error) &&
        result->relative_error >= 0.0, comm, PETSC_ERR_FP,
        "The %s generalized eigenpair has an invalid relative error", name);
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

PetscErrorCode compute_generalized_condition_number(Mat A, Mat M,
    const PetscReal eps_tolerance, const PetscInt max_iterations,
    const PetscInt subspace_dimension, GeneralizedConditionResult *result,
    const GeneralizedConditionKSPConfigurator configure_ksp,
    void *configure_ksp_context)
{
    PetscFunctionBeginUser;
    PetscCheck(A != nullptr, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
        "Generalized EPS matrix A must not be null");
    const MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(A));
    PetscCheck(result != nullptr, comm, PETSC_ERR_ARG_NULL,
        "Generalized condition-number result must not be null");
    PetscCheck(eps_tolerance > 0.0 && std::isfinite(eps_tolerance),
        comm, PETSC_ERR_ARG_OUTOFRANGE,
        "EPS tolerance must be finite and positive");
    PetscCheck(max_iterations > 0, comm, PETSC_ERR_ARG_OUTOFRANGE,
        "EPS maximum iterations must be positive");
    PetscCheck(subspace_dimension >= 2, comm, PETSC_ERR_ARG_OUTOFRANGE,
        "EPS subspace dimension must be at least two");
    PetscCall(validate_compatible_matrices(A, M));
    *result = GeneralizedConditionResult{};

    PetscCallMPI(MPI_Barrier(comm));
    PetscLogDouble local_total_start = 0.0;
    PetscLogDouble local_total_end = 0.0;
    PetscCall(PetscTime(&local_total_start));
    ExtremeResult maximum;
    ExtremeResult minimum;
    PetscCall(solve_extreme(A, M, Extreme::maximum, eps_tolerance,
        max_iterations, subspace_dimension, &maximum,
        configure_ksp, configure_ksp_context));
    PetscCall(solve_extreme(A, M, Extreme::minimum, eps_tolerance,
        max_iterations, subspace_dimension, &minimum,
        configure_ksp, configure_ksp_context));
    PetscCall(PetscTime(&local_total_end));
    const PetscLogDouble local_total_time =
        local_total_end - local_total_start;
    PetscCallMPI(MPI_Allreduce(&local_total_time, &result->total_time, 1,
        MPI_DOUBLE, MPI_MAX, comm));

    result->lambda_min = minimum.eigenvalue;
    result->lambda_max = maximum.eigenvalue;
    result->min_eigenpair_error = minimum.relative_error;
    result->max_eigenpair_error = maximum.relative_error;
    result->min_converged_by_stability = minimum.converged_by_stability;
    result->min_stability_relative_span = minimum.stability_relative_span;
    result->min_stability_residual_limit = minimum.stability_residual_limit;
    result->min_stability_window = minimum.stability_window;
    result->min_eps_iterations = minimum.iterations;
    result->max_eps_iterations = maximum.iterations;
    result->min_convergence_reason = minimum.convergence_reason;
    result->max_convergence_reason = maximum.convergence_reason;
    result->min_solve_time = minimum.solve_time;
    result->max_solve_time = maximum.solve_time;
    result->condition_number = result->lambda_max / result->lambda_min;
    PetscCheck(std::isfinite(result->condition_number) &&
        result->condition_number > 0.0, comm, PETSC_ERR_FP,
        "Generalized spectral condition number is invalid");
    result->converged = true;
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace lor
