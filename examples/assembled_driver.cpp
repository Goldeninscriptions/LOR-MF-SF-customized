/*
 * Manufactured solution used by this driver:
 *
 *   u_exact(x) = product_d sin(pi (x_d-a_d)/(b_d-a_d)),
 *   f(x) = -Delta u_exact
 *        = [sum_d (pi/(b_d-a_d))^2] u_exact(x).
 *
 * ManufacturedSolution implements and validates this formula; the executable
 * remains responsible only for runtime initialization and dimension dispatch.
 */

#include "IO/DistributedGeometryInput.hpp"
#include "IO/ExperimentIO.hpp"
#include "Solver/AssembledLaplaceExperiment.hpp"

#include <slepcsys.h>

#include <exception>
#include <stdexcept>
#include <string>

int main(int argc, char **argv)
{
    // Initialize the one PETSc/SLEPc/MPI runtime used by every project module.
    SlepcInitialize(&argc, &argv, nullptr, nullptr);
    PetscErrorCode error = PETSC_SUCCESS;

    // Keep the executable as a thin runtime dispatcher: IO reads once on rank
    // zero, while the Solver experiment owns refinement, assembly, and solves.
    try
    {
        constexpr int root = 0;
        const std::string geometry_file = lor::read_geometry_filename();
        const lor::GeometryInput geometry =
            lor::read_distributed_geometry_input(
                geometry_file, root, PETSC_COMM_WORLD);
        const lor::ExperimentOptions options =
            lor::read_experiment_options(geometry.dimension);

        // Dispatch the fixed-dimensional implementation after collective IO
        // has made the geometry dimension identical on every rank.
        if (geometry.dimension == 2)
            error = lor::run_assembled_laplace_experiment<2>(
                geometry, options, root, PETSC_COMM_WORLD);
        else if (geometry.dimension == 3)
            error = lor::run_assembled_laplace_experiment<3>(
                geometry, options, root, PETSC_COMM_WORLD);
        else
            throw std::runtime_error("Only dimensions 2 and 3 are supported");
    }
    catch (const std::exception &exception)
    {
        // Root-only printing avoids repeated diagnostics after collectively
        // propagated IO or refinement exceptions.
        PetscPrintf(PETSC_COMM_WORLD, "Experiment error: %s\n",
            exception.what());
        error = PETSC_ERR_LIB;
    }

    // Finalize collective libraries exactly once after all module cleanup.
    SlepcFinalize();
    return static_cast<int>(error);
}
