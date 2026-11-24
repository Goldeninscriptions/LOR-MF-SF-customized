#include "Assembly/DerivativeGrevilleStiffness.hpp"
#include "Assembly/GrevilleMassPreconditioner.hpp"
#include "Element/BSplineBasis1D.hpp"
#include "Solver/GeneralizedConditionNumber.hpp"

#include <petscsys.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

PetscErrorCode configure_validation_cholesky(KSP ksp, void *)
{
    PetscFunctionBeginUser;
    PetscCall(KSPSetType(ksp, KSPPREONLY));
    PC pc = nullptr;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCCHOLESKY));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode check_dimensions(Mat matrix, const PetscInt rows,
    const PetscInt columns, const char *name)
{
    PetscFunctionBeginUser;
    PetscInt actual_rows = 0;
    PetscInt actual_columns = 0;
    PetscCall(MatGetSize(matrix, &actual_rows, &actual_columns));
    PetscCheck(actual_rows == rows && actual_columns == columns,
        PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
        "%s has size %" PetscInt_FMT "x%" PetscInt_FMT
        ", expected %" PetscInt_FMT "x%" PetscInt_FMT,
        name, actual_rows, actual_columns, rows, columns);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode check_row_nonzeros(Mat matrix, const PetscInt exact,
    const PetscInt maximum, const char *name)
{
    PetscFunctionBeginUser;
    PetscInt rows = 0;
    PetscCall(MatGetSize(matrix, &rows, nullptr));
    for (PetscInt row = 0; row < rows; ++row)
    {
        PetscInt count = 0;
        const PetscInt *columns = nullptr;
        const PetscScalar *values = nullptr;
        PetscCall(MatGetRow(matrix, row, &count, &columns, &values));
        const bool valid = (exact < 0 || count == exact) &&
            (maximum < 0 || count <= maximum);
        PetscCall(MatRestoreRow(matrix, row, &count, &columns, &values));
        PetscCheck(valid, PETSC_COMM_SELF, PETSC_ERR_PLIB,
            "%s row %" PetscInt_FMT " has an invalid support size",
            name, row);
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode relative_frobenius(Mat first, Mat second, PetscReal *relative)
{
    PetscFunctionBeginUser;
    Mat difference = nullptr;
    PetscCall(MatDuplicate(first, MAT_COPY_VALUES, &difference));
    PetscCall(MatAXPY(difference, -1.0, second, DIFFERENT_NONZERO_PATTERN));
    PetscReal numerator = 0.0;
    PetscReal denominator = 0.0;
    PetscCall(MatNorm(difference, NORM_FROBENIUS, &numerator));
    PetscCall(MatNorm(first, NORM_FROBENIUS, &denominator));
    PetscCall(MatDestroy(&difference));
    PetscCheck(denominator > 0.0, PETSC_COMM_SELF, PETSC_ERR_FP,
        "Reference matrix has zero Frobenius norm");
    *relative = numerator / denominator;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode check_symmetric_spd(Mat matrix, const char *name)
{
    PetscFunctionBeginUser;
    PetscBool symmetric = PETSC_FALSE;
    PetscCall(MatIsSymmetric(matrix, 1.0e-11, &symmetric));
    PetscCheck(symmetric, PETSC_COMM_SELF, PETSC_ERR_PLIB,
        "%s is not symmetric", name);
    PetscInt n = 0;
    PetscCall(MatGetSize(matrix, &n, nullptr));
    std::vector<PetscInt> indices(n);
    for (PetscInt i = 0; i < n; ++i)
        indices[i] = i;
    std::vector<PetscScalar> values(
        static_cast<std::size_t>(n) * n, 0.0);
    PetscCall(MatGetValues(matrix, n, indices.data(), n, indices.data(),
        values.data()));
    for (PetscInt row = 0; row < n; ++row)
        for (PetscInt column = 0; column <= row; ++column)
        {
            double pivot = PetscRealPart(values[
                static_cast<std::size_t>(row) * n + column]);
            for (PetscInt inner = 0; inner < column; ++inner)
                pivot -= PetscRealPart(values[
                    static_cast<std::size_t>(row) * n + inner]) *
                    PetscRealPart(values[
                    static_cast<std::size_t>(column) * n + inner]);
            if (row == column)
            {
                PetscCheck(pivot > 1.0e-14, PETSC_COMM_SELF, PETSC_ERR_PLIB,
                    "%s failed dense Cholesky at row %" PetscInt_FMT,
                    name, row);
                values[static_cast<std::size_t>(row) * n + column] =
                    std::sqrt(pivot);
            }
            else
                values[static_cast<std::size_t>(row) * n + column] =
                    pivot / PetscRealPart(values[
                    static_cast<std::size_t>(column) * n + column]);
        }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode deterministic_vector(Mat matrix, Vec *vector)
{
    PetscFunctionBeginUser;
    PetscCall(MatCreateVecs(matrix, vector, nullptr));
    PetscInt n = 0;
    PetscCall(VecGetSize(*vector, &n));
    for (PetscInt i = 0; i < n; ++i)
    {
        const PetscScalar value =
            std::sin(0.37 * (i + 1)) + 0.25 * std::cos(0.19 * (i + 2));
        PetscCall(VecSetValue(*vector, i, value, INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(*vector));
    PetscCall(VecAssemblyEnd(*vector));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode sparse_to_sequential_mat(
    const lor::SparseMatrix1D &source, Mat *matrix)
{
    PetscFunctionBeginUser;
    PetscInt maximum = 1;
    for (const auto &row : source.rows)
        maximum = std::max<PetscInt>(maximum, row.size());
    PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, source.size(), source.size(),
        maximum, nullptr, matrix));
    for (int row = 0; row < source.size(); ++row)
        for (const auto &entry : source.rows[row])
            PetscCall(MatSetValue(*matrix, row, entry.first,
                entry.second, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(*matrix, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*matrix, MAT_FINAL_ASSEMBLY));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Test-only negative control for the removed T_p^T K_g T_p construction.
// It is deliberately local to validation and is not exposed as a solver or
// assembly option.
PetscErrorCode assemble_removed_old_stiffness_reference(
    const int degree, const std::vector<double> &knots, Mat *old_matrix)
{
    PetscFunctionBeginUser;
    const auto full = lor::build_greville_mass_preconditioner_1d(
        degree, knots);
    const auto restricted =
        lor::restrict_greville_mass_preconditioner_endpoints(full);
    Mat collocation = nullptr;
    PetscCall(sparse_to_sequential_mat(restricted.collocation, &collocation));
    const int full_size = full.greville_points.size();
    const int free_size = full_size - 2;
    std::vector<std::map<int, double>> rows(free_size);
    for (int cell = 0; cell + 1 < full_size; ++cell)
    {
        const double h = full.greville_points[cell + 1] -
            full.greville_points[cell];
        const int vertices[2] = {cell - 1, cell};
        const double values[4] = {1.0 / h, -1.0 / h,
                                  -1.0 / h, 1.0 / h};
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b)
                if (vertices[a] >= 0 && vertices[a] < free_size &&
                    vertices[b] >= 0 && vertices[b] < free_size)
                    rows[vertices[a]][vertices[b]] += values[2 * a + b];
    }
    Mat low_stiffness = nullptr;
    PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, free_size, free_size,
        3, nullptr, &low_stiffness));
    for (int row = 0; row < free_size; ++row)
        for (const auto &entry : rows[row])
            PetscCall(MatSetValue(low_stiffness, row, entry.first,
                entry.second, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(low_stiffness, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(low_stiffness, MAT_FINAL_ASSEMBLY));
    PetscCall(MatPtAP(low_stiffness, collocation, MAT_INITIAL_MATRIX,
        PETSC_DEFAULT, old_matrix));
    PetscCall(MatDestroy(&low_stiffness));
    PetscCall(MatDestroy(&collocation));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode check_energy(Mat preconditioner, Mat bx, Mat mass_x,
    Mat by, Mat mass_y, const PetscReal tolerance)
{
    PetscFunctionBeginUser;
    Vec coefficient = nullptr;
    Vec pc = nullptr;
    PetscCall(deterministic_vector(preconditioner, &coefficient));
    PetscCall(VecDuplicate(coefficient, &pc));
    PetscCall(MatMult(preconditioner, coefficient, pc));
    PetscScalar assembled_energy = 0.0;
    PetscCall(VecDot(coefficient, pc, &assembled_energy));
    PetscReal component_energy = 0.0;
    for (const auto &component :
         {std::pair<Mat, Mat>{bx, mass_x}, {by, mass_y}})
    {
        if (component.first == nullptr)
            continue;
        Vec sampled = nullptr;
        Vec weighted = nullptr;
        PetscCall(MatCreateVecs(component.first, nullptr, &sampled));
        PetscCall(VecDuplicate(sampled, &weighted));
        PetscCall(MatMult(component.first, coefficient, sampled));
        PetscCall(MatMult(component.second, sampled, weighted));
        PetscScalar value = 0.0;
        PetscCall(VecDot(sampled, weighted, &value));
        component_energy += PetscRealPart(value);
        PetscCall(VecDestroy(&weighted));
        PetscCall(VecDestroy(&sampled));
    }
    const PetscReal scale = std::max<PetscReal>(
        1.0, std::abs(PetscRealPart(assembled_energy)));
    PetscCheck(std::abs(PetscRealPart(assembled_energy) - component_energy) <=
        tolerance * scale, PETSC_COMM_SELF, PETSC_ERR_PLIB,
        "Component Q1 energy does not match assembled preconditioner");
    PetscCall(VecDestroy(&pc));
    PetscCall(VecDestroy(&coefficient));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode check_independent_p1_energy(
    const lor::DerivativeGreville1DResult &matrices)
{
    PetscFunctionBeginUser;
    Vec coefficient = nullptr;
    Vec pc = nullptr;
    Vec sampled = nullptr;
    PetscCall(deterministic_vector(matrices.preconditioner, &coefficient));
    PetscCall(VecDuplicate(coefficient, &pc));
    PetscCall(MatCreateVecs(
        matrices.sampled_derivative, nullptr, &sampled));
    PetscCall(MatMult(matrices.preconditioner, coefficient, pc));
    PetscCall(MatMult(matrices.sampled_derivative, coefficient, sampled));
    PetscScalar assembled = 0.0;
    PetscCall(VecDot(coefficient, pc, &assembled));
    const PetscScalar *values = nullptr;
    PetscCall(VecGetArrayRead(sampled, &values));
    double integrated = 0.0;
    const auto &points = matrices.derivative_greville_points;
    for (int cell = 0; cell + 1 < static_cast<int>(points.size()); ++cell)
    {
        const double h = points[cell + 1] - points[cell];
        const double a = PetscRealPart(values[cell]);
        const double b = PetscRealPart(values[cell + 1]);
        integrated += h * (2.0 * a * a + 2.0 * b * b + 2.0 * a * b) / 6.0;
    }
    PetscCall(VecRestoreArrayRead(sampled, &values));
    PetscCheck(std::abs(PetscRealPart(assembled) - integrated) <=
        3.0e-12 * std::max(1.0, std::abs(integrated)), PETSC_COMM_SELF,
        PETSC_ERR_PLIB, "Independent P1 derivative energy check failed");
    PetscCall(VecDestroy(&sampled));
    PetscCall(VecDestroy(&pc));
    PetscCall(VecDestroy(&coefficient));
    PetscFunctionReturn(PETSC_SUCCESS);
}

double q1_energy(const PetscScalar *values,
    const std::vector<double> &points_x,
    const std::vector<double> &points_y, const double scale)
{
    constexpr double pattern[16] = {
        4,2,1,2, 2,4,2,1, 1,2,4,2, 2,1,2,4};
    const int nx = points_x.size();
    const int ny = points_y.size();
    double energy = 0.0;
    for (int j = 0; j + 1 < ny; ++j)
        for (int i = 0; i + 1 < nx; ++i)
        {
            const int ids[4] = {
                lor::derivative_greville_id_2d(i, j, nx),
                lor::derivative_greville_id_2d(i + 1, j, nx),
                lor::derivative_greville_id_2d(i + 1, j + 1, nx),
                lor::derivative_greville_id_2d(i, j + 1, nx)};
            const double factor = scale *
                (points_x[i + 1] - points_x[i]) *
                (points_y[j + 1] - points_y[j]) / 36.0;
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b)
                    energy += factor * pattern[4 * a + b] *
                        PetscRealPart(values[ids[a]]) *
                        PetscRealPart(values[ids[b]]);
        }
    return energy;
}

PetscErrorCode check_independent_q1_energy(
    const lor::DerivativeGreville2DResult &matrices)
{
    PetscFunctionBeginUser;
    Vec coefficient = nullptr;
    Vec pc = nullptr;
    Vec sampled_x = nullptr;
    Vec sampled_y = nullptr;
    PetscCall(deterministic_vector(matrices.preconditioner, &coefficient));
    PetscCall(VecDuplicate(coefficient, &pc));
    PetscCall(MatCreateVecs(
        matrices.sampled_derivative_x, nullptr, &sampled_x));
    PetscCall(MatCreateVecs(
        matrices.sampled_derivative_y, nullptr, &sampled_y));
    PetscCall(MatMult(matrices.preconditioner, coefficient, pc));
    PetscCall(MatMult(matrices.sampled_derivative_x, coefficient, sampled_x));
    PetscCall(MatMult(matrices.sampled_derivative_y, coefficient, sampled_y));
    PetscScalar assembled = 0.0;
    PetscCall(VecDot(coefficient, pc, &assembled));
    const PetscScalar *x = nullptr;
    const PetscScalar *y = nullptr;
    PetscCall(VecGetArrayRead(sampled_x, &x));
    PetscCall(VecGetArrayRead(sampled_y, &y));
    const double integrated = q1_energy(x,
        matrices.derivative_greville_x, matrices.original_greville_y,
        matrices.x_derivative_weight) + q1_energy(y,
        matrices.original_greville_x, matrices.derivative_greville_y,
        matrices.y_derivative_weight);
    PetscCall(VecRestoreArrayRead(sampled_y, &y));
    PetscCall(VecRestoreArrayRead(sampled_x, &x));
    PetscCheck(std::abs(PetscRealPart(assembled) - integrated) <=
        8.0e-11 * std::max(1.0, std::abs(integrated)), PETSC_COMM_SELF,
        PETSC_ERR_PLIB, "Independent two-grid Q1 energy check failed");
    PetscCall(VecDestroy(&sampled_y));
    PetscCall(VecDestroy(&sampled_x));
    PetscCall(VecDestroy(&pc));
    PetscCall(VecDestroy(&coefficient));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode validate_1d_derivative(const int degree, const int elements)
{
    PetscFunctionBeginUser;
    const auto knots = lor::open_uniform_knots_1d(degree, elements);
    lor::DerivativeGreville1DResult matrices;
    lor::DerivativeGrevilleAssemblyOptions options;
    PetscCall(lor::assemble_derivative_greville_stiffness_1d(
        PETSC_COMM_SELF, degree, knots, options, &matrices));
    Vec coefficients = nullptr;
    Vec derivatives = nullptr;
    PetscCall(MatCreateVecs(matrices.derivative, &coefficients, &derivatives));
    const int n = matrices.num_full_dofs;
    for (int i = 0; i < n; ++i)
        PetscCall(VecSetValue(coefficients, i,
            std::sin(0.31 * (i + 1)), INSERT_VALUES));
    PetscCall(VecAssemblyBegin(coefficients));
    PetscCall(VecAssemblyEnd(coefficients));
    PetscCall(MatMult(matrices.derivative, coefficients, derivatives));
    const PetscScalar *c = nullptr;
    const PetscScalar *d = nullptr;
    PetscCall(VecGetArrayRead(coefficients, &c));
    PetscCall(VecGetArrayRead(derivatives, &d));
    const lor::BSplineBasis1D original(degree, knots);
    const lor::BSplineBasis1D reduced(
        degree - 1, matrices.derivative_knots);
    for (const double point : {0.137, 0.456, 0.821})
    {
        const auto high = original.evaluate_N_der1(point);
        double direct = 0.0;
        for (int local = 0; local <= degree; ++local)
            direct += PetscRealPart(c[high.first_basis + local]) *
                high.dN[local];
        const auto low = reduced.evaluate_N(point);
        double represented = 0.0;
        for (int local = 0; local < degree; ++local)
            represented += PetscRealPart(d[low.first_basis + local]) *
                low.N[local];
        PetscCheck(std::abs(direct - represented) <=
            5.0e-12 * std::max(1.0, std::abs(direct)), PETSC_COMM_SELF,
            PETSC_ERR_PLIB, "Exact 1D derivative representation failed");
    }
    PetscCall(VecRestoreArrayRead(derivatives, &d));
    PetscCall(VecRestoreArrayRead(coefficients, &c));
    for (int a = 0; a + 1 < n; ++a)
    {
        const double delta =
            (knots[a + degree + 1] - knots[a + 1]) / degree;
        const double greville_delta = matrices.original_greville_points[a + 1] -
            matrices.original_greville_points[a];
        PetscCheck(std::abs(delta - greville_delta) <= 1.0e-14,
            PETSC_COMM_SELF, PETSC_ERR_PLIB,
            "Derivative scale and original Greville spacing differ");
    }
    if (degree >= 3)
    {
        Mat old_reference = nullptr;
        PetscCall(assemble_removed_old_stiffness_reference(
            degree, knots, &old_reference));
        PetscReal old_difference = 0.0;
        PetscCall(relative_frobenius(
            matrices.preconditioner, old_reference, &old_difference));
        PetscCheck(old_difference > 1.0e-8, PETSC_COMM_SELF,
            PETSC_ERR_PLIB,
            "New derivative-space P unexpectedly aliases old T_p^T K_g T_p");
        PetscCall(MatDestroy(&old_reference));
    }
    PetscCall(check_independent_p1_energy(matrices));
    PetscCall(VecDestroy(&derivatives));
    PetscCall(VecDestroy(&coefficients));
    PetscCall(lor::destroy_derivative_greville_result(&matrices));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode validate_1d_p2()
{
    PetscFunctionBeginUser;
    constexpr int degree = 2;
    constexpr int elements = 8;
    lor::DerivativeGreville1DResult matrices;
    lor::DerivativeGrevilleAssemblyOptions options;
    PetscCall(lor::assemble_derivative_greville_stiffness_1d(
        PETSC_COMM_SELF, degree,
        lor::open_uniform_knots_1d(degree, elements), options, &matrices));
    const int n = elements + degree;
    PetscCall(check_dimensions(matrices.derivative, n - 1, n, "D_p"));
    PetscCall(check_dimensions(matrices.collocation,
        n - 1, n - 1, "T_{p-1}"));
    PetscCall(check_dimensions(matrices.greville_mass,
        n - 1, n - 1, "M_g"));
    PetscCall(check_dimensions(matrices.preconditioner,
        n - 2, n - 2, "P_free"));
    PetscCall(check_row_nonzeros(matrices.derivative, 2, 2, "D_p"));
    PetscReal difference = 0.0;
    PetscCall(relative_frobenius(
        matrices.stiffness, matrices.preconditioner, &difference));
    PetscCheck(difference <= 2.0e-12, PETSC_COMM_SELF, PETSC_ERR_PLIB,
        "1D p=2 requires P=K; relative difference %.6e",
        static_cast<double>(difference));
    Mat exact_identity = nullptr;
    PetscCall(MatPtAP(matrices.exact_derivative_mass,
        matrices.derivative_free, MAT_INITIAL_MATRIX,
        PETSC_DEFAULT, &exact_identity));
    PetscReal derivative_identity = 0.0;
    PetscCall(relative_frobenius(
        matrices.stiffness, exact_identity, &derivative_identity));
    PetscCheck(derivative_identity <= 2.0e-12, PETSC_COMM_SELF,
        PETSC_ERR_PLIB, "K=D^T M D failed in 1D");
    PetscCall(check_energy(matrices.preconditioner,
        matrices.sampled_derivative, matrices.greville_mass,
        nullptr, nullptr, 2.0e-12));
    PetscCall(check_independent_p1_energy(matrices));
    PetscCall(check_symmetric_spd(matrices.preconditioner, "1D P"));
    PetscCall(check_symmetric_spd(matrices.stiffness, "1D K"));
    lor::GeneralizedConditionResult spectrum;
    PetscCall(lor::compute_generalized_condition_number(
        matrices.stiffness, matrices.preconditioner,
        1.0e-10, 1000, std::min(20, n - 2), &spectrum,
        configure_validation_cholesky, nullptr));
    PetscCheck(std::abs(spectrum.condition_number - 1.0) <= 1.0e-8,
        PETSC_COMM_SELF, PETSC_ERR_PLIB,
        "1D p=2 generalized condition number is not one");
    PetscCall(MatDestroy(&exact_identity));
    PetscCall(lor::destroy_derivative_greville_result(&matrices));
    PetscCall(validate_1d_derivative(4, 9));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode validate_2d_derivatives(
    const lor::DerivativeGreville2DResult &matrices)
{
    PetscFunctionBeginUser;
    const int p = matrices.degree;
    const int nx = matrices.num_basis_x;
    const int ny = matrices.num_basis_y;
    Vec coefficients = nullptr;
    Vec dx = nullptr;
    Vec dy = nullptr;
    PetscCall(MatCreateVecs(matrices.derivative_x, &coefficients, &dx));
    PetscCall(MatCreateVecs(matrices.derivative_y, nullptr, &dy));
    for (int full = 0; full < nx * ny; ++full)
        PetscCall(VecSetValue(coefficients, full,
            std::sin(0.17 * (full + 1)) + 0.1 * std::cos(0.29 * full),
            INSERT_VALUES));
    PetscCall(VecAssemblyBegin(coefficients));
    PetscCall(VecAssemblyEnd(coefficients));
    PetscCall(MatMult(matrices.derivative_x, coefficients, dx));
    PetscCall(MatMult(matrices.derivative_y, coefficients, dy));
    const PetscScalar *c = nullptr;
    const PetscScalar *cx = nullptr;
    const PetscScalar *cy = nullptr;
    PetscCall(VecGetArrayRead(coefficients, &c));
    PetscCall(VecGetArrayRead(dx, &cx));
    PetscCall(VecGetArrayRead(dy, &cy));
    const lor::BSplineBasis1D ox(p, matrices.knots_x);
    const lor::BSplineBasis1D oy(p, matrices.knots_y);
    const lor::BSplineBasis1D rx(p - 1, matrices.derivative_knots_x);
    const lor::BSplineBasis1D ry(p - 1, matrices.derivative_knots_y);
    const double x = 0.371;
    const double y = 0.613;
    const auto vx = ox.evaluate_N_der1(x);
    const auto vy = oy.evaluate_N_der1(y);
    const auto lx = rx.evaluate_N(x);
    const auto ly = ry.evaluate_N(y);
    double direct_x = 0.0;
    double direct_y = 0.0;
    for (int jy = 0; jy <= p; ++jy)
        for (int ix = 0; ix <= p; ++ix)
        {
            const int full = lor::derivative_greville_id_2d(
                vx.first_basis + ix, vy.first_basis + jy, nx);
            direct_x += PetscRealPart(c[full]) * vx.dN[ix] * vy.N[jy];
            direct_y += PetscRealPart(c[full]) * vx.N[ix] * vy.dN[jy];
        }
    double represented_x = 0.0;
    for (int jy = 0; jy <= p; ++jy)
        for (int ix = 0; ix < p; ++ix)
        {
            const int id = lor::derivative_greville_id_2d(
                lx.first_basis + ix, vy.first_basis + jy, nx - 1);
            represented_x += PetscRealPart(cx[id]) * lx.N[ix] * vy.N[jy];
        }
    double represented_y = 0.0;
    for (int jy = 0; jy < p; ++jy)
        for (int ix = 0; ix <= p; ++ix)
        {
            const int id = lor::derivative_greville_id_2d(
                vx.first_basis + ix, ly.first_basis + jy, nx);
            represented_y += PetscRealPart(cy[id]) * vx.N[ix] * ly.N[jy];
        }
    PetscCheck(std::abs(direct_x - represented_x) <=
        1.0e-11 * std::max(1.0, std::abs(direct_x)), PETSC_COMM_SELF,
        PETSC_ERR_PLIB, "Exact 2D x derivative representation failed");
    PetscCheck(std::abs(direct_y - represented_y) <=
        1.0e-11 * std::max(1.0, std::abs(direct_y)), PETSC_COMM_SELF,
        PETSC_ERR_PLIB, "Exact 2D y derivative representation failed");
    PetscCall(VecRestoreArrayRead(dy, &cy));
    PetscCall(VecRestoreArrayRead(dx, &cx));
    PetscCall(VecRestoreArrayRead(coefficients, &c));
    PetscCall(VecDestroy(&dy));
    PetscCall(VecDestroy(&dx));
    PetscCall(VecDestroy(&coefficients));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode validate_2d_rectangular()
{
    PetscFunctionBeginUser;
    constexpr int p = 3;
    constexpr int ex = 4;
    constexpr int ey = 7;
    lor::DerivativeGreville2DResult matrices;
    lor::DerivativeGrevilleAssemblyOptions options;
    PetscCall(lor::assemble_derivative_greville_stiffness_2d(
        PETSC_COMM_SELF, p,
        lor::open_uniform_knots_1d(p, ex),
        lor::open_uniform_knots_1d(p, ey), 2.0, 1.0,
        options, &matrices));
    const int nx = ex + p;
    const int ny = ey + p;
    PetscCheck(matrices.x_derivative_weight == 0.5 &&
        matrices.y_derivative_weight == 2.0, PETSC_COMM_SELF,
        PETSC_ERR_PLIB, "Physical rectangle weights are incorrect");
    PetscCheck(lor::derivative_greville_id_2d(1, 1, nx) == nx + 1,
        PETSC_COMM_SELF, PETSC_ERR_PLIB, "x-fast numbering is incorrect");
    PetscCall(check_dimensions(matrices.derivative_x,
        (nx - 1) * ny, nx * ny, "D_x"));
    PetscCall(check_dimensions(matrices.derivative_y,
        nx * (ny - 1), nx * ny, "D_y"));
    PetscCall(check_dimensions(matrices.collocation_x,
        (nx - 1) * ny, (nx - 1) * ny, "T_x"));
    PetscCall(check_dimensions(matrices.collocation_y,
        nx * (ny - 1), nx * (ny - 1), "T_y"));
    PetscCall(check_dimensions(matrices.preconditioner,
        (nx - 2) * (ny - 2), (nx - 2) * (ny - 2), "P_2D"));
    PetscCall(check_row_nonzeros(matrices.derivative_x, 2, 2, "D_x"));
    PetscCall(check_row_nonzeros(matrices.derivative_y, 2, 2, "D_y"));
    PetscCall(check_row_nonzeros(matrices.collocation_x, -1,
        p * (p + 1), "T_x"));
    PetscCall(check_row_nonzeros(matrices.collocation_y, -1,
        p * (p + 1), "T_y"));
    PetscCall(validate_2d_derivatives(matrices));
    PetscCall(check_energy(matrices.preconditioner,
        matrices.sampled_derivative_x, matrices.greville_mass_x,
        matrices.sampled_derivative_y, matrices.greville_mass_y, 5.0e-11));
    PetscCall(check_independent_q1_energy(matrices));
    Mat exact_x = nullptr;
    Mat exact_y = nullptr;
    Mat exact_sum = nullptr;
    PetscCall(MatPtAP(matrices.exact_derivative_mass_x,
        matrices.derivative_x_free, MAT_INITIAL_MATRIX,
        PETSC_DEFAULT, &exact_x));
    PetscCall(MatPtAP(matrices.exact_derivative_mass_y,
        matrices.derivative_y_free, MAT_INITIAL_MATRIX,
        PETSC_DEFAULT, &exact_y));
    PetscCall(MatDuplicate(exact_x, MAT_COPY_VALUES, &exact_sum));
    PetscCall(MatAXPY(exact_sum, 1.0, exact_y, DIFFERENT_NONZERO_PATTERN));
    PetscReal identity_error = 0.0;
    PetscCall(relative_frobenius(
        matrices.stiffness, exact_sum, &identity_error));
    PetscCheck(identity_error <= 2.0e-11, PETSC_COMM_SELF, PETSC_ERR_PLIB,
        "2D K=D_x^T W_x D_x+D_y^T W_y D_y failed");
    PetscCall(check_symmetric_spd(matrices.preconditioner, "2D P"));
    PetscCall(check_symmetric_spd(matrices.stiffness, "2D K"));
    PetscReal surrogate_difference = 0.0;
    PetscCall(relative_frobenius(
        matrices.stiffness, matrices.preconditioner, &surrogate_difference));
    PetscCheck(surrogate_difference > 1.0e-8, PETSC_COMM_SELF,
        PETSC_ERR_PLIB, "p=3 surrogate unexpectedly aliases exact stiffness");
    PetscCall(MatDestroy(&exact_sum));
    PetscCall(MatDestroy(&exact_y));
    PetscCall(MatDestroy(&exact_x));
    PetscCall(lor::destroy_derivative_greville_result(&matrices));

    // p=2 in 2D is checked only for dimensions, energy, symmetry and SPD;
    // unlike the one-dimensional case, equality P=K is intentionally absent.
    lor::DerivativeGreville2DResult quadratic;
    PetscCall(lor::assemble_derivative_greville_stiffness_2d(
        PETSC_COMM_SELF, 2,
        lor::open_uniform_knots_1d(2, 4),
        lor::open_uniform_knots_1d(2, 5), 2.0, 1.0,
        options, &quadratic));
    PetscCall(check_symmetric_spd(quadratic.preconditioner, "2D p=2 P"));
    PetscCall(check_energy(quadratic.preconditioner,
        quadratic.sampled_derivative_x, quadratic.greville_mass_x,
        quadratic.sampled_derivative_y, quadratic.greville_mass_y, 5.0e-11));
    PetscCall(check_independent_q1_energy(quadratic));
    PetscCall(lor::destroy_derivative_greville_result(&quadratic));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode run()
{
    PetscFunctionBeginUser;
    char case_name[64] = "all";
    PetscCall(PetscOptionsGetString(nullptr, nullptr, "-validation_case",
        case_name, sizeof(case_name), nullptr));
    const std::string selected(case_name);
    PetscCheck(selected == "all" || selected == "1d_p2" ||
        selected == "2d_rectangular", PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
        "-validation_case must be all, 1d_p2, or 2d_rectangular");
    if (selected == "all" || selected == "1d_p2")
        PetscCall(validate_1d_p2());
    if (selected == "all" || selected == "2d_rectangular")
        PetscCall(validate_2d_rectangular());
    PetscCall(PetscPrintf(PETSC_COMM_SELF,
        "derivative_greville_validation: %s passed\n", case_name));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

int main(int argc, char **argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    const PetscErrorCode error = run();
    PetscFinalize();
    return error == PETSC_SUCCESS ? 0 : 1;
}
