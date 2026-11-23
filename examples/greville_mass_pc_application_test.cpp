#include "Assembly/GrevilleMassPreconditioner.hpp"
#include "Solver/GrevilleMassPC.hpp"

#include <slepcsys.h>

#include <array>
#include <cmath>
#include <iomanip>
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

Mat petsc_matrix(const lor::SparseMatrix1D &matrix)
{
    Mat result = nullptr;
    const PetscInt size = matrix.size();
    std::vector<PetscInt> nonzeros(size);
    for (PetscInt row = 0; row < size; ++row)
        nonzeros[row] = static_cast<PetscInt>(matrix.rows[row].size());
    check(MatCreateSeqAIJ(PETSC_COMM_SELF, size, size, 0,
        nonzeros.data(), &result), "MatCreateSeqAIJ");
    check(MatSetOption(result, MAT_SYMMETRIC, PETSC_TRUE),
        "MatSetOption(MAT_SYMMETRIC)");
    check(MatSetOption(result, MAT_SPD, PETSC_TRUE),
        "MatSetOption(MAT_SPD)");
    for (PetscInt row = 0; row < size; ++row)
        for (const auto &entry : matrix.rows[row])
            check(MatSetValue(result, row, entry.first, entry.second,
                INSERT_VALUES), "MatSetValue");
    check(MatAssemblyBegin(result, MAT_FINAL_ASSEMBLY), "MatAssemblyBegin");
    check(MatAssemblyEnd(result, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd");
    return result;
}

struct SolveSummary
{
    PetscInt dofs = 0;
    PetscInt iterations = 0;
    PetscInt applications = 0;
    double relative_solution_error = 0.0;
    lor::GrevilleMassPCStatistics statistics;
};

template <int Dim>
SolveSummary solve_known_problem(const lor::SparseMatrix1D &mass,
    const std::array<lor::GrevilleMassPreconditioner1D, Dim> &directions)
{
    Mat matrix = petsc_matrix(mass);
    Vec exact = nullptr;
    Vec rhs = nullptr;
    check(MatCreateVecs(matrix, &exact, &rhs), "MatCreateVecs");
    PetscScalar *entries = nullptr;
    check(VecGetArray(exact, &entries), "VecGetArray(exact)");
    for (int index = 0; index < mass.size(); ++index)
        entries[index] = std::sin(0.37 * (index + 1)) +
            0.25 * std::cos(0.11 * (index + 2));
    check(VecRestoreArray(exact, &entries), "VecRestoreArray(exact)");
    check(MatMult(matrix, exact, rhs), "MatMult(rhs)");

    lor::GrevilleMassPCSolveOptions options;
    options.relative_tolerance = 1.0e-11;
    options.absolute_tolerance = 1.0e-14;
    options.maximum_iterations = 200;
    lor::GrevilleMassPCSolveResult result;
    check(lor::solve_mass_with_greville_tensor_pc<Dim>(
        matrix, rhs, directions, options, &result),
        "solve_mass_with_greville_tensor_pc");

    check(VecAXPY(result.solution, -1.0, exact), "VecAXPY(error)");
    PetscReal error_norm = 0.0;
    PetscReal exact_norm = 0.0;
    check(VecNorm(result.solution, NORM_2, &error_norm),
        "VecNorm(error)");
    check(VecNorm(exact, NORM_2, &exact_norm), "VecNorm(exact)");
    if (result.reason <= 0 || result.relative_residual_norm > 2.0e-11 ||
        error_norm / exact_norm > 2.0e-8)
        throw std::runtime_error(
            "Greville mass PC solve did not reach the requested accuracy");

    SolveSummary summary;
    summary.dofs = mass.size();
    summary.iterations = result.outer_iterations;
    summary.applications = result.statistics.pc_applications;
    summary.relative_solution_error = error_norm / exact_norm;
    summary.statistics = std::move(result.statistics);
    check(lor::destroy_greville_mass_pc_result(&result),
        "destroy_greville_mass_pc_result");
    check(VecDestroy(&exact), "VecDestroy(exact)");
    check(VecDestroy(&rhs), "VecDestroy(rhs)");
    check(MatDestroy(&matrix), "MatDestroy(matrix)");
    return summary;
}

void print_summary(const std::string &case_name, const SolveSummary &summary)
{
    const auto &stats = summary.statistics;
    std::cout << "summary\t" << case_name << '\t' << summary.dofs << '\t'
              << summary.iterations << '\t' << summary.applications << '\t'
              << std::setprecision(12) << summary.relative_solution_error
              << '\t' << stats.setup_seconds
              << '\t' << stats.outer_solve_seconds
              << '\t' << stats.factor_memory_bytes
              << '\t' << stats.workspace_memory_bytes
              << '\t' << stats.operator_reported_memory_bytes
              << '\t' << stats.operator_minimum_memory_bytes
              << '\t' << stats.source_factor_storage_bytes
              << '\t' << stats.estimated_total_memory_bytes
              << '\t' << stats.total_apply_seconds
              << '\t' << stats.total_transpose_T_seconds
              << '\t' << stats.total_greville_mass_seconds
              << '\t' << stats.total_T_seconds
              << '\t' << stats.process_memory_before_setup_bytes
              << '\t' << stats.process_memory_after_setup_bytes << '\n';
    for (const auto &iteration : stats.iterations)
        std::cout << "iteration\t" << case_name << '\t'
                  << iteration.iteration << '\t'
                  << iteration.residual_norm << '\t'
                  << iteration.elapsed_seconds << '\n';
    for (const auto &application : stats.applications)
        std::cout << "application\t" << case_name << '\t'
                  << application.application << '\t'
                  << application.transpose_T_line_solves << '\t'
                  << application.greville_mass_line_solves << '\t'
                  << application.T_line_solves << '\t'
                  << application.copy_seconds << '\t'
                  << application.transpose_T_seconds << '\t'
                  << application.greville_mass_seconds << '\t'
                  << application.T_seconds << '\t'
                  << application.total_seconds << '\n';
}

void run_one_dimensional_cases()
{
    for (int degree = 2; degree <= 10; ++degree)
    {
        auto direction = lor::build_greville_mass_preconditioner_1d(
            degree, lor::open_uniform_knots_1d(degree, 64));
        std::array<lor::GrevilleMassPreconditioner1D, 1> directions = {
            std::move(direction)};
        const auto summary = solve_known_problem<1>(
            directions[0].spline_mass, directions);
        print_summary("1D_p" + std::to_string(degree) + "_n64", summary);
    }
}

void run_two_dimensional_case()
{
    auto x = lor::build_greville_mass_preconditioner_1d(
        3, lor::open_uniform_knots_1d(3, 16));
    auto y = lor::build_greville_mass_preconditioner_1d(
        3, lor::open_uniform_knots_1d(3, 16));
    const auto mass = lor::sparse_kronecker_product(
        y.spline_mass, x.spline_mass);
    std::array<lor::GrevilleMassPreconditioner1D, 2> directions = {
        std::move(x), std::move(y)};
    const auto summary = solve_known_problem<2>(mass, directions);
    print_summary("2D_p3_n16x16", summary);
}

} // namespace

int main(int argc, char **argv)
{
    check(SlepcInitialize(&argc, &argv, nullptr, nullptr),
        "SlepcInitialize");
    int exit_code = 0;
    try
    {
        std::cout << "summary_columns\tcase\tdofs\touter_iterations"
                  << "\tpc_applications\trelative_solution_error"
                  << "\tsetup_s\tsolve_s\tfactor_bytes\tworkspace_bytes"
                  << "\toperator_reported_bytes\toperator_minimum_bytes"
                  << "\tsource_factor_bytes\testimated_total_bytes"
                  << "\tpc_apply_total_s\tT_transpose_total_s"
                  << "\tMg_total_s\tT_total_s"
                  << "\tprocess_before_bytes\tprocess_after_bytes\n";
        std::cout << "iteration_columns\tcase\titeration\tresidual_norm"
                  << "\telapsed_s\n";
        std::cout << "application_columns\tcase\tapplication"
                  << "\tT_transpose_line_solves\tMg_line_solves"
                  << "\tT_line_solves\tcopy_s\tT_transpose_s"
                  << "\tMg_s\tT_s\ttotal_s\n";
        run_one_dimensional_cases();
        run_two_dimensional_case();
    }
    catch (const std::exception &error)
    {
        std::cerr << "greville_mass_pc_application_test: "
                  << error.what() << '\n';
        exit_code = 1;
    }
    check(SlepcFinalize(), "SlepcFinalize");
    return exit_code;
}
