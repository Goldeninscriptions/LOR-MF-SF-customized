#ifndef LOR_SOLVER_ASSEMBLED_LAPLACE_EXPERIMENT_HPP
#define LOR_SOLVER_ASSEMBLED_LAPLACE_EXPERIMENT_HPP

#include "IO/ExperimentIO.hpp"
#include "IO/GeometryInput.hpp"

#include <petscsys.h>

#include <mpi.h>

namespace lor
{

// Orchestrate one complete assembled Laplace experiment after IO has supplied
// a minimal geometry and typed options. Refinement runs only on root; all PETSc
// assembly, KSP, SVD, and error-analysis stages are collective on comm.
template <int Dim>
PetscErrorCode run_assembled_laplace_experiment(
    const GeometryInput &geometry_input,
    const ExperimentOptions &options, int root, MPI_Comm comm);

extern template PetscErrorCode run_assembled_laplace_experiment<2>(
    const GeometryInput &, const ExperimentOptions &, int, MPI_Comm);
extern template PetscErrorCode run_assembled_laplace_experiment<3>(
    const GeometryInput &, const ExperimentOptions &, int, MPI_Comm);

} // namespace lor

#endif
