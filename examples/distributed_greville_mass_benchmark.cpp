#include "Assembly/GrevilleMassPreconditioner.hpp"
#include "Solver/DistributedGrevilleMassPC.hpp"

#include <slepcsys.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

void check(const PetscErrorCode error, const char *operation)
{
    if (error != PETSC_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed");
}

lor::DistributedGrevillePCMode parse_mode(const std::string &name)
{
    if (name == "tensor_transpose")
        return lor::DistributedGrevillePCMode::tensor_transpose;
    if (name == "distributed_band")
        return lor::DistributedGrevillePCMode::distributed_band;
    throw std::invalid_argument(
        "-mode must be tensor_transpose or distributed_band");
}

} // namespace

int main(int argc, char **argv)
{
    check(SlepcInitialize(&argc, &argv, nullptr, nullptr),
        "SlepcInitialize");
    int exit_code = 0;
    try
    {
        PetscInt degree = 3;
        PetscInt elements = 1024;
        PetscReal tolerance = 1.0e-8;
        PetscInt maximum_iterations = 200;
        PetscBool detailed = PETSC_FALSE;
        char mode_buffer[64] = "tensor_transpose";
        check(PetscOptionsGetInt(nullptr, nullptr, "-p", &degree, nullptr),
            "PetscOptionsGetInt(-p)");
        check(PetscOptionsGetInt(nullptr, nullptr, "-n", &elements, nullptr),
            "PetscOptionsGetInt(-n)");
        check(PetscOptionsGetReal(nullptr, nullptr, "-rtol", &tolerance,
            nullptr), "PetscOptionsGetReal(-rtol)");
        check(PetscOptionsGetInt(nullptr, nullptr, "-max_it",
            &maximum_iterations, nullptr), "PetscOptionsGetInt(-max_it)");
        check(PetscOptionsGetBool(nullptr, nullptr, "-detailed", &detailed,
            nullptr), "PetscOptionsGetBool(-detailed)");
        check(PetscOptionsGetString(nullptr, nullptr, "-mode", mode_buffer,
            sizeof(mode_buffer), nullptr), "PetscOptionsGetString(-mode)");
        if (degree <= 0 || elements <= 0)
            throw std::invalid_argument("p and n must be positive");
        const auto mode = parse_mode(mode_buffer);

        auto x = lor::build_greville_mass_preconditioner_1d(
            degree, lor::open_uniform_knots_1d(degree, elements));
        auto y = lor::build_greville_mass_preconditioner_1d(
            degree, lor::open_uniform_knots_1d(degree, elements));
        Mat mass = nullptr;
        check(lor::create_distributed_tensor_mass_2d(
            PETSC_COMM_WORLD, x, y, &mass),
            "create_distributed_tensor_mass_2d");
        Vec exact = nullptr;
        Vec rhs = nullptr;
        check(MatCreateVecs(mass, &exact, &rhs), "MatCreateVecs");
        PetscInt begin = 0;
        PetscInt end = 0;
        check(VecGetOwnershipRange(exact, &begin, &end),
            "VecGetOwnershipRange");
        PetscScalar *entries = nullptr;
        check(VecGetArray(exact, &entries), "VecGetArray(exact)");
        for (PetscInt global = begin; global < end; ++global)
            entries[global - begin] =
                std::sin(1.0e-3 * (global + 1)) +
                0.2 * std::cos(7.0e-4 * (global + 3));
        check(VecRestoreArray(exact, &entries), "VecRestoreArray(exact)");
        check(MatMult(mass, exact, rhs), "MatMult(rhs)");

        lor::GrevilleMassPCSolveOptions options;
        options.relative_tolerance = tolerance;
        options.absolute_tolerance = 1.0e-14;
        options.maximum_iterations = maximum_iterations;
        lor::DistributedGrevillePCSolveResult result;
        check(lor::solve_mass_with_distributed_greville_pc_2d(
            mass, rhs, x, y, mode, options, &result),
            "solve_mass_with_distributed_greville_pc_2d");
        check(VecAXPY(result.solution, -1.0, exact), "VecAXPY(error)");
        PetscReal error_norm = 0.0;
        PetscReal exact_norm = 0.0;
        check(VecNorm(result.solution, NORM_2, &error_norm),
            "VecNorm(error)");
        check(VecNorm(exact, NORM_2, &exact_norm), "VecNorm(exact)");
        const PetscReal relative_solution_error = error_norm / exact_norm;
        PetscMPIInt ranks = 0;
        check(MPI_Comm_size(PETSC_COMM_WORLD, &ranks), "MPI_Comm_size");
        const auto &statistics = result.statistics;
        check(PetscPrintf(PETSC_COMM_WORLD,
            "RESULT mode=%s ranks=%d p=%" PetscInt_FMT
            " n=%" PetscInt_FMT " dofs=%" PetscInt_FMT
            " iterations=%" PetscInt_FMT " reason=%d"
            " relative_residual=%.12e relative_solution_error=%.12e"
            " setup_s=%.9e solve_s=%.9e operator_s=%.9e"
            " pc_s=%.9e pc_comm_s=%.9e communicated_bytes=%.0f"
            " factor_bytes_per_rank=%.0f workspace_bytes_per_rank=%.0f"
            " process_before_bytes=%.0f process_after_bytes=%.0f\n",
            lor::distributed_greville_pc_mode_name(mode), ranks,
            degree, elements,
            static_cast<PetscInt>(x.spline_mass.size()) *
                y.spline_mass.size(),
            result.outer_iterations, static_cast<int>(result.reason),
            static_cast<double>(result.relative_residual_norm),
            static_cast<double>(relative_solution_error),
            statistics.setup_seconds, statistics.solve_seconds,
            statistics.operator_apply_seconds,
            statistics.pc_apply_seconds,
            statistics.pc_communication_seconds,
            statistics.communicated_bytes,
            statistics.factor_memory_bytes,
            statistics.workspace_memory_bytes,
            statistics.process_memory_before_setup_bytes,
            statistics.process_memory_after_setup_bytes), "PetscPrintf");
        if (detailed)
        {
            for (const auto &iteration : statistics.iterations)
                check(PetscPrintf(PETSC_COMM_WORLD,
                    "ITERATION mode=%s index=%" PetscInt_FMT
                    " residual=%.12e elapsed_s=%.9e\n",
                    lor::distributed_greville_pc_mode_name(mode),
                    iteration.iteration,
                    static_cast<double>(iteration.residual_norm),
                    iteration.elapsed_seconds), "PetscPrintf(iteration)");
            for (const auto &application : statistics.applications)
                check(PetscPrintf(PETSC_COMM_WORLD,
                    "APPLICATION mode=%s index=%" PetscInt_FMT
                    " local_s=%.9e distributed_s=%.9e comm_s=%.9e"
                    " total_s=%.9e bytes=%.0f\n",
                    lor::distributed_greville_pc_mode_name(mode),
                    application.application,
                    application.local_direction_seconds,
                    application.distributed_direction_seconds,
                    application.communication_seconds,
                    application.total_seconds,
                    application.communicated_bytes),
                    "PetscPrintf(application)");
        }

        if (result.reason <= 0 || result.relative_residual_norm >
            20.0 * tolerance || relative_solution_error > 2.0e-5)
            throw std::runtime_error(
                "Distributed Greville solve failed stability checks");
        check(lor::destroy_distributed_greville_pc_result(&result),
            "destroy_distributed_greville_pc_result");
        check(VecDestroy(&exact), "VecDestroy(exact)");
        check(VecDestroy(&rhs), "VecDestroy(rhs)");
        check(MatDestroy(&mass), "MatDestroy(mass)");
    }
    catch (const std::exception &error)
    {
        PetscPrintf(PETSC_COMM_WORLD,
            "distributed_greville_mass_benchmark: %s\n", error.what());
        exit_code = 1;
    }
    check(SlepcFinalize(), "SlepcFinalize");
    return exit_code;
}
