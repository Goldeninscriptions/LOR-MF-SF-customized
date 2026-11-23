#ifndef LOR_SOLVER_GREVILLE_MASS_PC_HPP
#define LOR_SOLVER_GREVILLE_MASS_PC_HPP

#include "Assembly/GrevilleMassPreconditioner.hpp"

#include <petscksp.h>

#include <array>
#include <vector>

namespace lor
{

struct GrevilleMassPCApplyRecord
{
    PetscInt application = 0;
    PetscInt transpose_T_line_solves = 0;
    PetscInt greville_mass_line_solves = 0;
    PetscInt T_line_solves = 0;
    PetscLogDouble copy_seconds = 0.0;
    PetscLogDouble transpose_T_seconds = 0.0;
    PetscLogDouble greville_mass_seconds = 0.0;
    PetscLogDouble T_seconds = 0.0;
    PetscLogDouble total_seconds = 0.0;
};

struct GrevilleMassPCIterationRecord
{
    PetscInt iteration = 0;
    PetscReal residual_norm = 0.0;
    PetscLogDouble elapsed_seconds = 0.0;
};

struct GrevilleMassPCStatistics
{
    PetscLogDouble setup_seconds = 0.0;
    PetscLogDouble outer_solve_seconds = 0.0;
    PetscLogDouble total_apply_seconds = 0.0;
    PetscLogDouble total_copy_seconds = 0.0;
    PetscLogDouble total_transpose_T_seconds = 0.0;
    PetscLogDouble total_greville_mass_seconds = 0.0;
    PetscLogDouble total_T_seconds = 0.0;
    PetscLogDouble source_factor_storage_bytes = 0.0;
    PetscLogDouble factor_memory_bytes = 0.0;
    PetscLogDouble workspace_memory_bytes = 0.0;
    PetscLogDouble operator_reported_memory_bytes = 0.0;
    PetscLogDouble operator_minimum_memory_bytes = 0.0;
    PetscLogDouble estimated_total_memory_bytes = 0.0;
    PetscLogDouble process_memory_before_setup_bytes = 0.0;
    PetscLogDouble process_memory_after_setup_bytes = 0.0;
    PetscInt pc_applications = 0;
    // T and M_g are factored direct band solves. This remains zero by design;
    // line-solve work is reported per phase in applications instead.
    PetscInt inner_krylov_iterations = 0;
    std::vector<GrevilleMassPCApplyRecord> applications;
    std::vector<GrevilleMassPCIterationRecord> iterations;
};

struct GrevilleMassPCSolveOptions
{
    PetscReal relative_tolerance = 1.0e-12;
    PetscReal absolute_tolerance = 1.0e-14;
    PetscReal divergence_tolerance = 1.0e6;
    PetscInt maximum_iterations = 1000;
    bool print_iteration_monitor = false;
};

struct GrevilleMassPCSolveResult
{
    Vec solution = nullptr;
    PetscInt outer_iterations = 0;
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    PetscReal residual_norm = 0.0;
    PetscReal relative_residual_norm = 0.0;
    GrevilleMassPCStatistics statistics;
};

// Solve M_s x=b by CG with the exact tensor application
//
//   (T_0 x ... x T_d)^{-1}
//   (M_{g,0} x ... x M_{g,d})^{-1}
//   (T_0 x ... x T_d)^{-T}.
//
// The current implementation intentionally supports one MPI rank. A future
// distributed implementation needs tensor transposes/pencil communication;
// silently gathering the full vector on every rank is deliberately avoided.
template <int Dim>
PetscErrorCode solve_mass_with_greville_tensor_pc(Mat spline_mass, Vec rhs,
    const std::array<GrevilleMassPreconditioner1D, Dim> &directions,
    const GrevilleMassPCSolveOptions &options,
    GrevilleMassPCSolveResult *result);

PetscErrorCode destroy_greville_mass_pc_result(
    GrevilleMassPCSolveResult *result);

extern template PetscErrorCode solve_mass_with_greville_tensor_pc<1>(
    Mat, Vec, const std::array<GrevilleMassPreconditioner1D, 1> &,
    const GrevilleMassPCSolveOptions &, GrevilleMassPCSolveResult *);
extern template PetscErrorCode solve_mass_with_greville_tensor_pc<2>(
    Mat, Vec, const std::array<GrevilleMassPreconditioner1D, 2> &,
    const GrevilleMassPCSolveOptions &, GrevilleMassPCSolveResult *);
extern template PetscErrorCode solve_mass_with_greville_tensor_pc<3>(
    Mat, Vec, const std::array<GrevilleMassPreconditioner1D, 3> &,
    const GrevilleMassPCSolveOptions &, GrevilleMassPCSolveResult *);

} // namespace lor

#endif
