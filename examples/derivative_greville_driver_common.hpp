#ifndef LOR_EXAMPLES_DERIVATIVE_GREVILLE_DRIVER_COMMON_HPP
#define LOR_EXAMPLES_DERIVATIVE_GREVILLE_DRIVER_COMMON_HPP

#include "Solver/GeneralizedConditionNumber.hpp"

#include <petscksp.h>

#include <cmath>

namespace derivative_greville_example
{

struct AnalysisResult
{
    lor::GeneralizedConditionResult spectrum;
    PetscInt outer_iterations = 0;
    PetscReal true_relative_residual = 0.0;
    PetscLogDouble solve_seconds = 0.0;
    KSPConvergedReason solve_reason = KSP_CONVERGED_ITERATING;
    bool spectrum_computed = false;
    bool solve_computed = false;
};

inline PetscErrorCode configure_cholesky(KSP ksp, void *)
{
    PetscFunctionBeginUser;
    PetscCall(KSPSetType(ksp, KSPPREONLY));
    PetscCall(KSPSetErrorIfNotConverged(ksp, PETSC_TRUE));
    PC pc = nullptr;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCCHOLESKY));
    PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode run_analysis(Mat stiffness, Mat preconditioner,
    const bool compute_spectrum, const bool compute_solve,
    const PetscReal eps_tolerance, const PetscInt eps_max_iterations,
    const PetscInt eps_subspace_dimension, AnalysisResult *result)
{
    PetscFunctionBeginUser;
    const MPI_Comm comm = PetscObjectComm(
        reinterpret_cast<PetscObject>(stiffness));
    PetscCheck(result != nullptr, comm, PETSC_ERR_ARG_NULL,
        "Analysis result must not be null");
    int ranks = 1;
    PetscCallMPI(MPI_Comm_size(comm, &ranks));
    PetscCheck(ranks == 1 || (!compute_spectrum && !compute_solve), comm,
        PETSC_ERR_SUP,
        "Standalone direct-factor analysis currently requires one MPI rank");
    if (compute_spectrum)
    {
        PetscCall(lor::compute_generalized_condition_number(
            stiffness, preconditioner, eps_tolerance,
            eps_max_iterations, eps_subspace_dimension,
            &result->spectrum, configure_cholesky, nullptr));
        result->spectrum_computed = true;
    }
    if (compute_solve)
    {
        Vec solution = nullptr;
        Vec rhs = nullptr;
        Vec residual = nullptr;
        PetscCall(MatCreateVecs(stiffness, &solution, &rhs));
        PetscCall(VecDuplicate(rhs, &residual));
        PetscCall(VecSet(rhs, 1.0));
        PetscCall(VecSet(solution, 0.0));
        KSP ksp = nullptr;
        PetscCall(KSPCreate(comm, &ksp));
        PetscCall(KSPSetOptionsPrefix(ksp, "dmg_"));
        PetscCall(KSPSetOperators(ksp, stiffness, preconditioner));
        PetscCall(KSPSetType(ksp, KSPCG));
        PetscCall(KSPSetTolerances(ksp, 1.0e-10, 1.0e-14,
            PETSC_DEFAULT, 10000));
        PetscCall(KSPSetErrorIfNotConverged(ksp, PETSC_TRUE));
        PC pc = nullptr;
        PetscCall(KSPGetPC(ksp, &pc));
        PetscCall(PCSetType(pc, PCCHOLESKY));
        PetscCall(KSPSetFromOptions(ksp));
        PetscLogDouble start = 0.0;
        PetscLogDouble end = 0.0;
        PetscCall(PetscTime(&start));
        PetscCall(KSPSolve(ksp, rhs, solution));
        PetscCall(PetscTime(&end));
        result->solve_seconds = end - start;
        PetscCall(KSPGetIterationNumber(ksp, &result->outer_iterations));
        PetscCall(KSPGetConvergedReason(ksp, &result->solve_reason));
        PetscCall(MatMult(stiffness, solution, residual));
        PetscCall(VecAYPX(residual, -1.0, rhs));
        PetscReal residual_norm = 0.0;
        PetscReal rhs_norm = 0.0;
        PetscCall(VecNorm(residual, NORM_2, &residual_norm));
        PetscCall(VecNorm(rhs, NORM_2, &rhs_norm));
        result->true_relative_residual = residual_norm / rhs_norm;
        PetscCheck(std::isfinite(result->true_relative_residual), comm,
            PETSC_ERR_FP, "True relative residual is not finite");
        result->solve_computed = true;
        PetscCall(KSPDestroy(&ksp));
        PetscCall(VecDestroy(&residual));
        PetscCall(VecDestroy(&rhs));
        PetscCall(VecDestroy(&solution));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace derivative_greville_example

#endif
