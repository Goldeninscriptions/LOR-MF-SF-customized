#ifndef LOR_SOLVER_CONDITION_NUMBER_HPP
#define LOR_SOLVER_CONDITION_NUMBER_HPP

#include <petscmat.h>

namespace lor
{

struct ConditionNumberResult
{
    // Smallest and largest singular values of the analyzed operator.
    PetscReal sigma_min = 0.0;
    PetscReal sigma_max = 0.0;

    // Ordinary 2-norm condition number sigma_max/sigma_min.
    PetscReal condition_number = 0.0;
};

// Compute cond_2(K) and cond_2(K_LOR^{-1}K) independently of the linear KSP.
PetscErrorCode compute_stiffness_condition_numbers(Mat stiffness,
    Mat lor_stiffness, PetscReal svd_tolerance,
    ConditionNumberResult *original,
    ConditionNumberResult *preconditioned);

} // namespace lor

#endif
