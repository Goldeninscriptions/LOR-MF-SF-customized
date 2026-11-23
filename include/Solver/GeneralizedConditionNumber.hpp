#ifndef LOR_SOLVER_GENERALIZED_CONDITION_NUMBER_HPP
#define LOR_SOLVER_GENERALIZED_CONDITION_NUMBER_HPP

#include <petscksp.h>
#include <petscsys.h>

namespace lor
{

using GeneralizedConditionKSPConfigurator =
    PetscErrorCode (*)(KSP ksp, void *context);

// Results for the generalized Hermitian-definite problem
//
//     A x = lambda M x,  A=A^T>0, M=M^T>0.
//
// The caller retains ownership of A and M. All times are communicator-maximum
// wall times so that they represent the parallel critical path.
struct GeneralizedConditionResult
{
    PetscReal lambda_min = 0.0;
    PetscReal lambda_max = 0.0;
    PetscReal condition_number = 0.0;
    PetscReal min_eigenpair_error = 0.0;
    PetscReal max_eigenpair_error = 0.0;
    // The smallest eigenpair may satisfy a guarded Ritz-value stability
    // criterion before its conservative relative residual reaches eps_tol.
    bool min_converged_by_stability = false;
    PetscReal min_stability_relative_span = 0.0;
    PetscReal min_stability_residual_limit = 0.0;
    PetscInt min_stability_window = 0;
    PetscInt min_eps_iterations = 0;
    PetscInt max_eps_iterations = 0;
    int min_convergence_reason = 0;
    int max_convergence_reason = 0;
    PetscLogDouble min_solve_time = 0.0;
    PetscLogDouble max_solve_time = 0.0;
    PetscLogDouble total_time = 0.0;
    bool converged = false;
};

// Compute lambda_max and lambda_min with two EPS solves. When M is non-null,
// solve the generalized Hermitian-definite problem on M^{-1}A. When M is
// null, solve the standard Hermitian problem for A directly; no identity
// matrix is allocated. Runtime options use the prefixes eigmax_ and eigmin_,
// respectively. In SLEPc 3.24 this gives, for
// example, -eigmin_eps_type, -eigmin_st_type, -eigmin_st_ksp_type, and
// -eigmin_st_pc_type. Supplying
//
//   -eigmin_st_type sinvert -eigmin_eps_target 0
//   -eigmin_eps_target_magnitude
//
// explicitly selects zero-shift shift-and-invert if it is ever desired.
// The smallest solve also supports a guarded long-window stability test:
//
//   -eigmin_eps_stability_stop <bool>
//   -eigmin_eps_stability_window <iterations>
//   -eigmin_eps_stability_tol <relative span>
//   -eigmin_eps_stability_residual <relative residual ceiling>
//   -eigmin_eps_stability_min_it <iterations>
//
// Standard residual convergence always remains active and takes precedence.
PetscErrorCode compute_generalized_condition_number(Mat A, Mat M,
    PetscReal eps_tolerance, PetscInt max_iterations,
    PetscInt subspace_dimension, GeneralizedConditionResult *result,
    GeneralizedConditionKSPConfigurator configure_ksp = nullptr,
    void *configure_ksp_context = nullptr);

} // namespace lor

#endif
