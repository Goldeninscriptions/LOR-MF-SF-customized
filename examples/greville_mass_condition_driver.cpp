#include "Assembly/GrevilleMassPreconditioner.hpp"
#include "Solver/DistributedGrevilleMassPC.hpp"
#include "Solver/GeneralizedConditionNumber.hpp"

#include <slepceps.h>

#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

namespace
{

PetscErrorCode run()
{
    PetscFunctionBeginUser;
    const MPI_Comm comm = PETSC_COMM_WORLD;
    PetscInt degree = 0;
    PetscInt elements = 0;
    PetscBool degree_set = PETSC_FALSE;
    PetscBool elements_set = PETSC_FALSE;
    char target_buffer[32] = "preconditioned";
    char output_buffer[PETSC_MAX_PATH_LEN] = "greville_mass_condition.log";
    PetscReal eps_tolerance = 1.0e-8;
    PetscInt eps_max_iterations = 10000;
    PetscInt eps_subspace_dimension = 40;
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-p",
        &degree, &degree_set));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-n",
        &elements, &elements_set));
    PetscCall(PetscOptionsGetString(nullptr, nullptr, "-spectral_target",
        target_buffer, sizeof(target_buffer), nullptr));
    PetscCall(PetscOptionsGetString(nullptr, nullptr, "-output",
        output_buffer, sizeof(output_buffer), nullptr));
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-eps_tol",
        &eps_tolerance, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-eps_max_it",
        &eps_max_iterations, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-eps_ncv",
        &eps_subspace_dimension, nullptr));
    PetscCheck(degree_set && degree >= 1, comm, PETSC_ERR_ARG_WRONG,
        "-p must specify a positive spline degree");
    PetscCheck(elements_set && elements >= 1, comm, PETSC_ERR_ARG_WRONG,
        "-n must specify a positive element count");
    const std::string spectral_target(target_buffer);
    PetscCheck(spectral_target == "preconditioner" ||
        spectral_target == "preconditioned", comm, PETSC_ERR_ARG_WRONG,
        "-spectral_target must be preconditioner or preconditioned");

    lor::GrevilleMassPreconditioner1D x_direction;
    lor::GrevilleMassPreconditioner1D y_direction;
    try
    {
        x_direction = lor::build_greville_mass_preconditioner_1d(
            degree, lor::open_uniform_knots_1d(degree, elements));
        y_direction = x_direction;
    }
    catch (const std::exception &error)
    {
        SETERRQ(comm, PETSC_ERR_ARG_WRONG, "%s", error.what());
    }

    Mat high_operator = nullptr;
    Mat transformed_preconditioner = nullptr;
    PetscCall(lor::create_distributed_tensor_mass_2d(
        comm, x_direction, y_direction, &high_operator));
    PetscCall(lor::create_distributed_tensor_greville_preconditioner_2d(
        comm, x_direction, y_direction, &transformed_preconditioner));
    lor::DistributedGrevillePCConfiguration2D configuration;
    configuration.x_direction = &x_direction;
    configuration.y_direction = &y_direction;
    configuration.mode = lor::DistributedGrevillePCMode::tensor_transpose;

    PetscInt global_dofs = 0;
    PetscCall(MatGetSize(transformed_preconditioner, &global_dofs, nullptr));
    lor::GeneralizedConditionResult result;
    if (spectral_target == "preconditioner")
        PetscCall(lor::compute_tensor_greville_preconditioner_condition_2d(
            comm, x_direction, y_direction, eps_tolerance,
            eps_max_iterations, eps_subspace_dimension, &result));
    else
        PetscCall(lor::compute_generalized_condition_number(
            high_operator, transformed_preconditioner,
            eps_tolerance, eps_max_iterations, eps_subspace_dimension,
            &result, lor::configure_distributed_greville_pc_2d,
            &configuration));

    int rank = 0;
    int ranks = 1;
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCallMPI(MPI_Comm_size(comm, &ranks));
    if (rank == 0)
    {
        std::ofstream output(output_buffer);
        PetscCheck(output.good(), comm, PETSC_ERR_FILE_OPEN,
            "Cannot open result file %s", output_buffer);
        output << std::setprecision(17)
               << "operator_type = mass\n"
               << "spectral_target = " << spectral_target << '\n'
               << "p = " << degree << '\n'
               << "n = " << elements << ',' << elements << '\n'
               << "mpi_ranks = " << ranks << '\n'
               << "total_dofs = " << global_dofs << '\n'
               << "lambda_min = " << result.lambda_min << '\n'
               << "lambda_max = " << result.lambda_max << '\n'
               << "condition_number = " << result.condition_number << '\n'
               << "eps_iterations_min = " << result.min_eps_iterations << '\n'
               << "eps_iterations_max = " << result.max_eps_iterations << '\n'
               << "min_eigenpair_error = "
               << result.min_eigenpair_error << '\n'
               << "max_eigenpair_error = "
               << result.max_eigenpair_error << '\n'
               << "min_converged_by_stability = "
               << (result.min_converged_by_stability ? "true" : "false")
               << '\n'
               << "time_eps_s = " << result.total_time << '\n'
               << "eigensolver_status = "
               << (result.converged ? "converged" : "not_converged")
               << '\n';
    }
    PetscCall(MatDestroy(&transformed_preconditioner));
    PetscCall(MatDestroy(&high_operator));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

int main(int argc, char **argv)
{
    SlepcInitialize(&argc, &argv, nullptr, nullptr);
    const PetscErrorCode error = run();
    SlepcFinalize();
    return error == PETSC_SUCCESS ? 0 : 1;
}
