#include "Assembly/GrevilleMassPreconditioner.hpp"

#include <slepceps.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

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
        {
            const PetscInt column = entry.first;
            const PetscScalar value = entry.second;
            check(MatSetValue(result, row, column, value, INSERT_VALUES),
                "MatSetValue");
        }
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
    check(EPSSetWhichEigenpairs(eps, EPS_ALL),
        "EPSSetWhichEigenpairs(EPS_ALL)");
    check(EPSSetFromOptions(eps), "EPSSetFromOptions");
    check(EPSSolve(eps), "EPSSolve");

    PetscInt converged = 0;
    check(EPSGetConverged(eps, &converged), "EPSGetConverged");
    if (converged != operator_matrix.size())
        throw std::runtime_error(
            "Dense LAPACK eigensolve did not return the complete spectrum");

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

bool agrees_with_two_decimal_reference(
    const double value, const double reference)
{
    return std::abs(value - reference) <= 0.011;
}

void validate_conversation_reference(const int degree,
    const double original, const double direct, const double transformed)
{
    // The shared conversation reports p=2,3,4,6,8,10 and rounds to two
    // decimal places. The missing odd degrees are still emitted for the
    // requested complete p=2,...,10 sweep.
    const std::map<int, std::array<double, 3>> reference = {
        {2, {10.57, 3.25, 2.40}},
        {3, {26.82, 9.02, 2.98}},
        {4, {69.13, 24.52, 3.56}},
        {6, {480.43, 179.52, 4.46}},
        {8, {3468.33, 1326.37, 5.47}},
        {10, {25647.03, 9929.76, 6.77}},
    };
    const auto found = reference.find(degree);
    if (found == reference.end())
        return;
    if (!agrees_with_two_decimal_reference(original, found->second[0]) ||
        !agrees_with_two_decimal_reference(direct, found->second[1]) ||
        !agrees_with_two_decimal_reference(transformed, found->second[2]))
        throw std::runtime_error(
            "Computed spectrum differs from the shared conversation table");
}

void run_one_dimensional_sweep()
{
    std::cout << "dimension\tp\telements\tdofs\tkappa_Ms"
              << "\tkappa_Mg_inv_Ms\tkappa_TtMgT_inv_Ms\n";
    for (int degree = 2; degree <= 10; ++degree)
    {
        const auto factors = lor::build_greville_mass_preconditioner_1d(
            degree, lor::open_uniform_knots_1d(degree, 64));
        const auto original = spectral_condition(factors.spline_mass, nullptr);
        const auto direct = spectral_condition(
            factors.spline_mass, &factors.greville_mass);
        const auto transformed = spectral_condition(
            factors.spline_mass, &factors.transformed_mass);
        validate_conversation_reference(degree, original.condition,
            direct.condition, transformed.condition);
        std::cout << "1D\t" << degree << "\t64\t"
                  << factors.spline_mass.size() << '\t'
                  << std::setprecision(12) << original.condition << '\t'
                  << direct.condition << '\t'
                  << transformed.condition << '\n';
    }
}

void assert_relative_close(const double actual, const double expected,
    const double tolerance, const char *description)
{
    if (std::abs(actual - expected) >
        tolerance * std::max(1.0, std::abs(expected)))
        throw std::runtime_error(std::string(description) +
            " violates the tensor-product spectral identity");
}

void run_two_dimensional_tensor_test()
{
    constexpr int degree = 3;
    constexpr int elements = 16;
    const auto factors = lor::build_greville_mass_preconditioner_1d(
        degree, lor::open_uniform_knots_1d(degree, elements));

    // Build the actual two-dimensional sparse operators, rather than merely
    // squaring a one-dimensional result. Direction 0 varies fastest.
    const auto spline_2d = lor::sparse_kronecker_product(
        factors.spline_mass, factors.spline_mass);
    const auto direct_2d = lor::sparse_kronecker_product(
        factors.greville_mass, factors.greville_mass);
    const auto transformed_2d = lor::sparse_kronecker_product(
        factors.transformed_mass, factors.transformed_mass);

    const auto direct_1d = spectral_condition(
        factors.spline_mass, &factors.greville_mass);
    const auto transformed_1d = spectral_condition(
        factors.spline_mass, &factors.transformed_mass);
    const auto direct = spectral_condition(spline_2d, &direct_2d);
    const auto transformed = spectral_condition(spline_2d, &transformed_2d);
    assert_relative_close(direct.condition,
        direct_1d.condition * direct_1d.condition, 2.0e-11,
        "Direct Greville 2D condition number");
    assert_relative_close(transformed.condition,
        transformed_1d.condition * transformed_1d.condition, 2.0e-11,
        "Transformed Greville 2D condition number");

    std::cout << "2D\t" << degree << '\t' << elements << "x" << elements
              << '\t' << spline_2d.size() << "\tn/a\t"
              << std::setprecision(12) << direct.condition << '\t'
              << transformed.condition << '\n';
    std::cout << "tensor_check\tdirect_1d_squared\t"
              << direct_1d.condition * direct_1d.condition
              << "\ttransformed_1d_squared\t"
              << transformed_1d.condition * transformed_1d.condition << '\n';
}

} // namespace

int main(int argc, char **argv)
{
    check(SlepcInitialize(&argc, &argv, nullptr, nullptr),
        "SlepcInitialize");
    int exit_code = 0;
    try
    {
        run_one_dimensional_sweep();
        run_two_dimensional_tensor_test();
    }
    catch (const std::exception &error)
    {
        std::cerr << "greville_mass_preconditioner_test: "
                  << error.what() << '\n';
        exit_code = 1;
    }
    check(SlepcFinalize(), "SlepcFinalize");
    return exit_code;
}
