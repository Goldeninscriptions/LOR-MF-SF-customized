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
    MPI_Comm comm = PETSC_COMM_WORLD;
    PetscInt degree = 0;
    PetscInt elements = 0;
    PetscBool degree_set = PETSC_FALSE;
    PetscBool elements_set = PETSC_FALSE;
    char operator_buffer[32] = "mass";
    char target_buffer[32] = "preconditioned";
    char output_buffer[PETSC_MAX_PATH_LEN] = "greville_tensor_condition.log";
    PetscReal eps_tolerance = 1.0e-8;
    PetscInt eps_max_iterations = 10000;
    PetscInt eps_subspace_dimension = 40;
    PetscReal length_x = 1.0;
    PetscReal length_y = 1.0;
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-p",
        &degree, &degree_set));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-n",
        &elements, &elements_set));
    PetscCall(PetscOptionsGetString(nullptr, nullptr, "-operator_type",
        operator_buffer, sizeof(operator_buffer), nullptr));
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
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-length_x",
        &length_x, nullptr));
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-length_y",
        &length_y, nullptr));
    PetscCheck(degree_set && degree >= 1, comm, PETSC_ERR_ARG_WRONG,
        "-p must specify a positive spline degree");
    PetscCheck(elements_set && elements >= 1, comm, PETSC_ERR_ARG_WRONG,
        "-n must specify a positive element count");
    const std::string operator_type(operator_buffer);
    const std::string spectral_target(target_buffer);
    PetscCheck(operator_type == "mass" || operator_type == "stiffness",
        comm, PETSC_ERR_ARG_WRONG,
        "-operator_type must be mass or stiffness");
    PetscCheck(spectral_target == "preconditioner" ||
        spectral_target == "preconditioned", comm, PETSC_ERR_ARG_WRONG,
        "-spectral_target must be preconditioner or preconditioned");
    PetscCheck(length_x > 0.0 && length_y > 0.0, comm,
        PETSC_ERR_ARG_OUTOFRANGE,
        "-length_x and -length_y must be positive");

    lor::GrevilleMassPreconditioner1D x_direction;
    lor::GrevilleMassPreconditioner1D y_direction;
    try
    {
        const auto full = lor::build_greville_mass_preconditioner_1d(
            degree, lor::open_uniform_knots_1d(degree, elements));
        if (operator_type == "mass")
        {
            x_direction = full;
            y_direction = full;
        }
        else
        {
            x_direction =
                lor::restrict_greville_mass_preconditioner_endpoints(full);
            y_direction = x_direction;
        }
    }
    catch (const std::exception &error)
    {
        SETERRQ(comm, PETSC_ERR_ARG_WRONG, "%s", error.what());
    }

    Mat high_operator = nullptr;
    Mat transformed_preconditioner = nullptr;
    lor::DistributedGrevillePCConfiguration2D configuration;
    configuration.x_direction = &x_direction;
    configuration.y_direction = &y_direction;
    configuration.mode = lor::DistributedGrevillePCMode::tensor_transpose;
    configuration.stiffness_x_weight = length_y / length_x;
    configuration.stiffness_y_weight = length_x / length_y;
    lor::GeneralizedConditionKSPConfigurator configure_pc = nullptr;
    if (operator_type == "mass")
    {
        PetscCall(lor::create_distributed_tensor_mass_2d(
            comm, x_direction, y_direction, &high_operator));
        PetscCall(lor::create_distributed_tensor_greville_preconditioner_2d(
            comm, x_direction, y_direction, &transformed_preconditioner));
        configure_pc = lor::configure_distributed_greville_pc_2d;
    }
    else
    {
        PetscCall(lor::create_distributed_tensor_spline_stiffness_2d(
            comm, x_direction, y_direction, &high_operator,
            configuration.stiffness_x_weight,
            configuration.stiffness_y_weight));
        PetscCall(lor::create_distributed_tensor_greville_stiffness_2d(
            comm, x_direction, y_direction, &transformed_preconditioner,
            configuration.stiffness_x_weight,
            configuration.stiffness_y_weight));
        configure_pc =
            lor::configure_distributed_greville_stiffness_pc_2d;
    }

    PetscInt global_dofs = 0;
    PetscCall(MatGetSize(transformed_preconditioner, &global_dofs, nullptr));
    lor::GeneralizedConditionResult result;
    if (spectral_target == "preconditioner" && operator_type == "mass")
        PetscCall(lor::compute_tensor_greville_preconditioner_condition_2d(
            comm, x_direction, y_direction, eps_tolerance,
            eps_max_iterations, eps_subspace_dimension, &result));
    else if (spectral_target == "preconditioner")
        PetscCall(lor::compute_generalized_condition_number(
            transformed_preconditioner, nullptr,
            eps_tolerance, eps_max_iterations, eps_subspace_dimension,
            &result, configure_pc, &configuration));
    else
        PetscCall(lor::compute_generalized_condition_number(
            high_operator, transformed_preconditioner,
            eps_tolerance, eps_max_iterations, eps_subspace_dimension,
            &result, configure_pc, &configuration));

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
               << "operator_type = " << operator_type << '\n'
               << "spectral_target = " << spectral_target << '\n'
               << "p = " << degree << '\n'
               << "n = " << elements << ',' << elements << '\n'
               << "length_x = " << length_x << '\n'
               << "length_y = " << length_y << '\n'
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
        PetscCheck(output.good(), comm, PETSC_ERR_FILE_WRITE,
            "Failed while writing result file %s", output_buffer);
    }
    PetscCall(PetscPrintf(comm,
        "Completed Greville tensor %s target=%s p=%" PetscInt_FMT
        " n=%" PetscInt_FMT "x%" PetscInt_FMT
        " dofs=%" PetscInt_FMT " kappa=%.12e time=%.6e s\n",
        operator_type.c_str(), spectral_target.c_str(), degree,
        elements, elements, global_dofs,
        static_cast<double>(result.condition_number),
        static_cast<double>(result.total_time)));
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
