#ifndef LOR_SOLVER_DISTRIBUTED_GREVILLE_MASS_PC_HPP
#define LOR_SOLVER_DISTRIBUTED_GREVILLE_MASS_PC_HPP

#include "Assembly/GrevilleMassPreconditioner.hpp"
#include "Solver/GrevilleMassPC.hpp"
#include "Solver/GeneralizedConditionNumber.hpp"

#include <petscksp.h>

#include <vector>

namespace lor
{

enum class DistributedGrevillePCMode
{
    tensor_transpose,
    distributed_band
};

const char *distributed_greville_pc_mode_name(
    DistributedGrevillePCMode mode) noexcept;

struct DistributedGrevillePCApplyRecord
{
    PetscInt application = 0;
    PetscLogDouble local_direction_seconds = 0.0;
    PetscLogDouble distributed_direction_seconds = 0.0;
    PetscLogDouble communication_seconds = 0.0;
    PetscLogDouble total_seconds = 0.0;
    PetscLogDouble communicated_bytes = 0.0;
};

struct DistributedGrevillePCStatistics
{
    PetscLogDouble setup_seconds = 0.0;
    PetscLogDouble solve_seconds = 0.0;
    PetscLogDouble operator_apply_seconds = 0.0;
    PetscLogDouble pc_apply_seconds = 0.0;
    PetscLogDouble pc_communication_seconds = 0.0;
    PetscLogDouble communicated_bytes = 0.0;
    PetscLogDouble factor_memory_bytes = 0.0;
    PetscLogDouble workspace_memory_bytes = 0.0;
    PetscLogDouble process_memory_before_setup_bytes = 0.0;
    PetscLogDouble process_memory_after_setup_bytes = 0.0;
    PetscInt operator_applications = 0;
    PetscInt pc_applications = 0;
    PetscInt inner_krylov_iterations = 0;
    std::vector<GrevilleMassPCIterationRecord> iterations;
    std::vector<DistributedGrevillePCApplyRecord> applications;
};

struct DistributedGrevillePCSolveResult
{
    Vec solution = nullptr;
    PetscInt outer_iterations = 0;
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    PetscReal residual_norm = 0.0;
    PetscReal relative_residual_norm = 0.0;
    DistributedGrevillePCStatistics statistics;
};

struct DistributedGrevillePCConfiguration2D
{
    const GrevilleMassPreconditioner1D *x_direction = nullptr;
    const GrevilleMassPreconditioner1D *y_direction = nullptr;
    DistributedGrevillePCMode mode =
        DistributedGrevillePCMode::tensor_transpose;
    DistributedGrevillePCStatistics *statistics = nullptr;
    double stiffness_x_weight = 1.0;
    double stiffness_y_weight = 1.0;
};

// Create M_y (x) M_x as a distributed MatShell. Rows use contiguous y slabs
// and complete x lines, matching both parallel preconditioner algorithms.
PetscErrorCode create_distributed_tensor_mass_2d(MPI_Comm comm,
    const GrevilleMassPreconditioner1D &x_direction,
    const GrevilleMassPreconditioner1D &y_direction, Mat *matrix);

// Create P_y (x) P_x, P_d=T_d^T M_g,d T_d, without assembling the
// multidimensional sparse Kronecker product.
PetscErrorCode create_distributed_tensor_greville_preconditioner_2d(
    MPI_Comm comm, const GrevilleMassPreconditioner1D &x_direction,
    const GrevilleMassPreconditioner1D &y_direction, Mat *matrix);

// Create the homogeneous-Dirichlet tensor-sum stiffness operators
//
//   K_s,y (x) M_s,x + M_s,y (x) K_s,x,
//   P_K,y (x) P_M,x + P_M,y (x) P_K,x,
//
// as distributed MatShell objects without assembling either 2D matrix.
PetscErrorCode create_distributed_tensor_spline_stiffness_2d(
    MPI_Comm comm, const GrevilleMassPreconditioner1D &x_direction,
    const GrevilleMassPreconditioner1D &y_direction, Mat *matrix,
    double x_derivative_weight = 1.0,
    double y_derivative_weight = 1.0);
PetscErrorCode create_distributed_tensor_greville_stiffness_2d(
    MPI_Comm comm, const GrevilleMassPreconditioner1D &x_direction,
    const GrevilleMassPreconditioner1D &y_direction, Mat *matrix,
    double x_derivative_weight = 1.0,
    double y_derivative_weight = 1.0);

// Configure an existing KSP with the exact tensor inverse.  This callback has
// the signature accepted by compute_generalized_condition_number and is used
// by EPS/ST for the generalized problem Kx=lambda Px.
PetscErrorCode configure_distributed_greville_pc_2d(
    KSP ksp, void *raw_configuration);

// Configure P_K^{-1}=T^{-1} K_g^{-1} T^{-T}. K_g is assembled as a sparse
// Q1 tensor-sum and solved by a nested, options-configurable CG+GAMG KSP.
// Nested options use the prefix -greville_stiffness_inner_.
PetscErrorCode configure_distributed_greville_stiffness_pc_2d(
    KSP ksp, void *raw_configuration);

// Exploit eig(P_y x P_x)=eig(P_y)eig(P_x), while still obtaining each
// directional extreme with EPS.  The small 1D minimum uses shift-and-invert
// with a direct sparse solve, avoiding thousands of 2D smallest-real steps.
PetscErrorCode compute_tensor_greville_preconditioner_condition_2d(
    MPI_Comm comm, const GrevilleMassPreconditioner1D &x_direction,
    const GrevilleMassPreconditioner1D &y_direction,
    PetscReal eps_tolerance, PetscInt max_iterations,
    PetscInt subspace_dimension, GeneralizedConditionResult *result);

// Solve the distributed tensor mass system using one of the two exact
// applications of (T_y (x) T_x)^T (M_gy (x) M_gx) (T_y (x) T_x).
PetscErrorCode solve_mass_with_distributed_greville_pc_2d(Mat spline_mass,
    Vec rhs, const GrevilleMassPreconditioner1D &x_direction,
    const GrevilleMassPreconditioner1D &y_direction,
    DistributedGrevillePCMode mode,
    const GrevilleMassPCSolveOptions &options,
    DistributedGrevillePCSolveResult *result);

PetscErrorCode destroy_distributed_greville_pc_result(
    DistributedGrevillePCSolveResult *result);

} // namespace lor

#endif
