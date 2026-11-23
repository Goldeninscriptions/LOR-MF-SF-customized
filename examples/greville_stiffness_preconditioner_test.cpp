#include "Assembly/GrevilleMassPreconditioner.hpp"

#include <slepceps.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct SpectralResult
{
    double minimum = 0.0;
    double maximum = 0.0;
    double condition = 0.0;
};

void check(const PetscErrorCode error, const char *operation)
{
    if (error != PETSC_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed");
}

Mat petsc_matrix(const lor::SparseMatrix1D &matrix)
{
    Mat result = nullptr;
    const PetscInt size = matrix.size();
    std::vector<PetscInt> row_nonzeros(size, 0);
    for (PetscInt row = 0; row < size; ++row)
        row_nonzeros[row] =
            static_cast<PetscInt>(matrix.rows[row].size());
    check(MatCreateSeqAIJ(PETSC_COMM_SELF, size, size, 0,
        row_nonzeros.data(), &result), "MatCreateSeqAIJ");
    check(MatSetOption(result, MAT_SYMMETRIC, PETSC_TRUE),
        "MatSetOption(MAT_SYMMETRIC)");
    check(MatSetOption(result, MAT_SPD, PETSC_TRUE),
        "MatSetOption(MAT_SPD)");
    for (PetscInt row = 0; row < size; ++row)
        for (const auto &entry : matrix.rows[row])
            check(MatSetValue(result, row, entry.first, entry.second,
                INSERT_VALUES), "MatSetValue");
    check(MatAssemblyBegin(result, MAT_FINAL_ASSEMBLY),
        "MatAssemblyBegin");
    check(MatAssemblyEnd(result, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd");
    return result;
}

SpectralResult spectral_condition(const lor::SparseMatrix1D &operator_matrix,
    const lor::SparseMatrix1D *metric)
{
    Mat A = petsc_matrix(operator_matrix);
    Mat B = metric == nullptr ? nullptr : petsc_matrix(*metric);
    EPS eps = nullptr;
    check(EPSCreate(PETSC_COMM_SELF, &eps), "EPSCreate");
    check(EPSSetOperators(eps, A, B), "EPSSetOperators");
    check(EPSSetProblemType(eps, metric == nullptr ? EPS_HEP : EPS_GHEP),
        "EPSSetProblemType");
    check(EPSSetType(eps, EPSLAPACK), "EPSSetType(EPSLAPACK)");
    check(EPSSetWhichEigenpairs(eps, EPS_ALL), "EPSSetWhichEigenpairs");
    check(EPSSolve(eps), "EPSSolve");

    PetscInt converged = 0;
    check(EPSGetConverged(eps, &converged), "EPSGetConverged");
    if (converged != operator_matrix.size())
        throw std::runtime_error("LAPACK did not return the complete spectrum");
    SpectralResult result;
    result.minimum = std::numeric_limits<double>::max();
    result.maximum = -std::numeric_limits<double>::max();
    for (PetscInt index = 0; index < converged; ++index)
    {
        PetscScalar real = 0.0;
        PetscScalar imaginary = 0.0;
        check(EPSGetEigenvalue(eps, index, &real, &imaginary),
            "EPSGetEigenvalue");
        const double value = PetscRealPart(real);
        result.minimum = std::min(result.minimum, value);
        result.maximum = std::max(result.maximum, value);
    }
    if (!(result.minimum > 0.0))
        throw std::runtime_error("Expected a positive definite spectrum");
    result.condition = result.maximum / result.minimum;
    check(EPSDestroy(&eps), "EPSDestroy");
    check(MatDestroy(&A), "MatDestroy(A)");
    if (B != nullptr)
        check(MatDestroy(&B), "MatDestroy(B)");
    return result;
}

void run_sweep()
{
    constexpr int elements = 64;
    std::cout << "dimension\tp\telements\tdofs\tkappa_Ks"
              << "\tkappa_Kg_inv_Ks\tkappa_TtKgT_inv_Ks"
              << "\tlambda_min_TtKgT\tlambda_max_TtKgT\n";
    for (int degree = 2; degree <= 10; ++degree)
    {
        const auto full = lor::build_greville_mass_preconditioner_1d(
            degree, lor::open_uniform_knots_1d(degree, elements));
        const auto factors =
            lor::restrict_greville_mass_preconditioner_endpoints(full);
        const auto original = spectral_condition(
            factors.spline_stiffness, nullptr);
        const auto direct = spectral_condition(
            factors.spline_stiffness, &factors.greville_stiffness);
        const auto transformed = spectral_condition(
            factors.spline_stiffness, &factors.transformed_stiffness);
        std::cout << "1D\t" << degree << '\t' << elements << '\t'
                  << factors.spline_stiffness.size() << '\t'
                  << std::setprecision(12) << original.condition << '\t'
                  << direct.condition << '\t'
                  << transformed.condition << '\t'
                  << transformed.minimum << '\t'
                  << transformed.maximum << '\n';
    }
}

void run_fixed_degree_mesh_sweep()
{
    constexpr int degree = 7;
    std::cout << "dimension\tp\telements\tmass_dofs\tstiffness_dofs"
              << "\tkappa_Ms\tkappa_TtMgT_inv_Ms"
              << "\tlambda_min_TtMgT\tlambda_max_TtMgT"
              << "\tkappa_Ks\tkappa_TtKgT_inv_Ks"
              << "\tlambda_min_TtKgT\tlambda_max_TtKgT\n";
    for (int elements = 4; elements <= 1024; elements *= 2)
    {
        const auto full = lor::build_greville_mass_preconditioner_1d(
            degree, lor::open_uniform_knots_1d(degree, elements));
        const auto restricted =
            lor::restrict_greville_mass_preconditioner_endpoints(full);
        const auto original_mass = spectral_condition(
            full.spline_mass, nullptr);
        const auto transformed_mass = spectral_condition(
            full.spline_mass, &full.transformed_mass);
        const auto original_stiffness = spectral_condition(
            restricted.spline_stiffness, nullptr);
        const auto transformed_stiffness = spectral_condition(
            restricted.spline_stiffness,
            &restricted.transformed_stiffness);
        std::cout << "1D\t" << degree << '\t' << elements << '\t'
                  << full.spline_mass.size() << '\t'
                  << restricted.spline_stiffness.size() << '\t'
                  << std::setprecision(12)
                  << original_mass.condition << '\t'
                  << transformed_mass.condition << '\t'
                  << transformed_mass.minimum << '\t'
                  << transformed_mass.maximum << '\t'
                  << original_stiffness.condition << '\t'
                  << transformed_stiffness.condition << '\t'
                  << transformed_stiffness.minimum << '\t'
                  << transformed_stiffness.maximum << '\n';
    }
}

} // namespace

int main(int argc, char **argv)
{
    check(SlepcInitialize(&argc, &argv, nullptr, nullptr),
        "SlepcInitialize");
    int exit_code = 0;
    try
    {
        PetscBool mesh_sweep = PETSC_FALSE;
        check(PetscOptionsGetBool(nullptr, nullptr, "-mesh_sweep",
            &mesh_sweep, nullptr), "PetscOptionsGetBool(mesh_sweep)");
        if (mesh_sweep)
            run_fixed_degree_mesh_sweep();
        else
            run_sweep();
    }
    catch (const std::exception &error)
    {
        std::cerr << "greville_stiffness_preconditioner_test: "
                  << error.what() << '\n';
        exit_code = 1;
    }
    SlepcFinalize();
    return exit_code;
}
