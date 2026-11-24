#include "Assembly/DerivativeGrevilleStiffness.hpp"

#include "Assembly/BandedMassCholesky1D.hpp"
#include "Element/BSplineBasis1D.hpp"
#include "Element/GaussQuadrature.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace lor
{
namespace
{

int basis_count(const int degree, const std::vector<double> &knots)
{
    const int count = static_cast<int>(knots.size()) - degree - 1;
    if (degree < 0 || count <= degree)
        throw std::invalid_argument("Invalid B-spline degree/knot-vector pair");
    if (!std::is_sorted(knots.begin(), knots.end()))
        throw std::invalid_argument("Knot vector must be nondecreasing");
    return count;
}

int positive_spans(const int degree, const std::vector<double> &knots)
{
    const int count = basis_count(degree, knots);
    int elements = 0;
    for (int span = degree; span < count; ++span)
        if (knots[span + 1] > knots[span])
            ++elements;
    return elements;
}

PetscErrorCode create_aij(const MPI_Comm comm, const PetscInt rows,
    const PetscInt columns, const PetscInt row_nonzero_estimate, Mat *matrix)
{
    PetscFunctionBeginUser;
    PetscCheck(matrix != nullptr, comm, PETSC_ERR_ARG_NULL,
        "Output matrix pointer must not be null");
    const PetscInt estimate = std::max<PetscInt>(1,
        std::min(columns, row_nonzero_estimate));
    PetscCall(MatCreateAIJ(comm, PETSC_DECIDE, PETSC_DECIDE,
        rows, columns, estimate, nullptr, estimate, nullptr, matrix));
    PetscCall(MatSetOption(*matrix, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode finish_matrix(Mat matrix, const bool symmetric,
    const bool positive_definite)
{
    PetscFunctionBeginUser;
    PetscCall(MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY));
    if (symmetric)
        PetscCall(MatSetOption(matrix, MAT_SYMMETRIC, PETSC_TRUE));
    if (positive_definite)
        PetscCall(MatSetOption(matrix, MAT_SPD, PETSC_TRUE));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscInt matrix_nonzeros(Mat matrix)
{
    if (matrix == nullptr)
        return 0;
    MatInfo info{};
    if (MatGetInfo(matrix, MAT_GLOBAL_SUM, &info) != PETSC_SUCCESS)
        return -1;
    return static_cast<PetscInt>(std::llround(info.nz_used));
}

PetscErrorCode maximum_elapsed(const MPI_Comm comm,
    const PetscLogDouble begin, PetscLogDouble *elapsed)
{
    PetscFunctionBeginUser;
    PetscLogDouble end = 0.0;
    PetscCall(PetscTime(&end));
    const PetscLogDouble local = end - begin;
    PetscCallMPI(MPI_Allreduce(&local, elapsed, 1,
        MPI_DOUBLE, MPI_MAX, comm));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode sparse_1d_to_petsc(const MPI_Comm comm,
    const SparseMatrix1D &source, Mat *matrix)
{
    PetscFunctionBeginUser;
    PetscInt maximum_row = 1;
    for (const auto &row : source.rows)
        maximum_row = std::max<PetscInt>(maximum_row, row.size());
    PetscCall(create_aij(comm, source.size(), source.size(),
        maximum_row, matrix));
    PetscInt begin = 0;
    PetscInt end = 0;
    PetscCall(MatGetOwnershipRange(*matrix, &begin, &end));
    for (PetscInt row = begin; row < end; ++row)
        for (const auto &entry : source.rows[row])
            PetscCall(MatSetValue(*matrix, row, entry.first,
                entry.second, INSERT_VALUES));
    PetscCall(finish_matrix(*matrix, true, true));
    PetscFunctionReturn(PETSC_SUCCESS);
}

std::vector<int> free_map_1d(const int size)
{
    if (size <= 2)
        throw std::invalid_argument("Dirichlet restriction leaves no 1D DOF");
    std::vector<int> map(size, -1);
    for (int i = 1; i + 1 < size; ++i)
        map[i] = i - 1;
    return map;
}

std::vector<int> free_map_2d(const int nx, const int ny)
{
    if (nx <= 2 || ny <= 2)
        throw std::invalid_argument("Dirichlet restriction leaves no 2D DOF");
    std::vector<int> map(static_cast<std::size_t>(nx) * ny, -1);
    int free = 0;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            if (i > 0 && i + 1 < nx && j > 0 && j + 1 < ny)
                map[derivative_greville_id_2d(i, j, nx)] = free++;
    return map;
}

PetscErrorCode assemble_restriction(const MPI_Comm comm,
    const std::vector<int> &free_map, const int free_size, Mat *restriction)
{
    PetscFunctionBeginUser;
    PetscCall(create_aij(comm, free_map.size(), free_size, 1, restriction));
    PetscInt begin = 0;
    PetscInt end = 0;
    PetscCall(MatGetOwnershipRange(*restriction, &begin, &end));
    for (PetscInt row = begin; row < end; ++row)
        if (free_map[row] >= 0)
            PetscCall(MatSetValue(*restriction, row, free_map[row],
                1.0, INSERT_VALUES));
    PetscCall(finish_matrix(*restriction, false, false));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode assemble_derivative_1d(const MPI_Comm comm,
    const int degree, const std::vector<double> &knots, Mat *derivative)
{
    PetscFunctionBeginUser;
    const int n = basis_count(degree, knots);
    PetscCall(create_aij(comm, n - 1, n, 2, derivative));
    PetscInt begin = 0;
    PetscInt end = 0;
    PetscCall(MatGetOwnershipRange(*derivative, &begin, &end));
    for (PetscInt row = begin; row < end; ++row)
    {
        const double denominator = knots[row + degree + 1] - knots[row + 1];
        PetscCheck(denominator > 0.0, comm, PETSC_ERR_ARG_WRONG,
            "Derivative difference has a nonpositive knot span");
        const PetscInt columns[2] = {row, row + 1};
        const PetscScalar scale = static_cast<double>(degree) / denominator;
        const PetscScalar values[2] = {-scale, scale};
        PetscCall(MatSetValues(*derivative, 1, &row, 2, columns,
            values, INSERT_VALUES));
    }
    PetscCall(finish_matrix(*derivative, false, false));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode assemble_collocation_1d(const MPI_Comm comm,
    const int degree, const std::vector<double> &knots,
    const std::vector<double> &points, Mat *collocation)
{
    PetscFunctionBeginUser;
    const int n = basis_count(degree, knots);
    PetscCheck(static_cast<int>(points.size()) == n, comm,
        PETSC_ERR_ARG_SIZ, "Collocation point and basis counts differ");
    PetscCall(create_aij(comm, n, n, degree + 1, collocation));
    const BSplineBasis1D basis(degree, knots);
    PetscInt begin = 0;
    PetscInt end = 0;
    PetscCall(MatGetOwnershipRange(*collocation, &begin, &end));
    for (PetscInt row = begin; row < end; ++row)
    {
        const auto value = basis.evaluate_N(points[row]);
        for (int local = 0; local <= degree; ++local)
            if (value.N[local] != 0.0)
                PetscCall(MatSetValue(*collocation, row,
                    value.first_basis + local, value.N[local],
                    INSERT_VALUES));
    }
    PetscCall(finish_matrix(*collocation, false, false));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode assemble_p1_mass(const MPI_Comm comm,
    const std::vector<double> &points, const double scale, Mat *mass)
{
    PetscFunctionBeginUser;
    PetscCheck(points.size() >= 2, comm, PETSC_ERR_ARG_SIZ,
        "P1 mass requires at least two grid points");
    const PetscInt n = points.size();
    PetscCall(create_aij(comm, n, n, 3, mass));
    int rank = 0;
    int ranks = 1;
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCallMPI(MPI_Comm_size(comm, &ranks));
    for (PetscInt cell = rank; cell + 1 < n; cell += ranks)
    {
        const double h = points[cell + 1] - points[cell];
        PetscCheck(h > 0.0, comm, PETSC_ERR_ARG_WRONG,
            "P1 mass grid is not strictly increasing");
        const PetscInt indices[2] = {cell, cell + 1};
        const PetscScalar factor = scale * h / 6.0;
        const PetscScalar values[4] = {
            2.0 * factor, factor, factor, 2.0 * factor};
        PetscCall(MatSetValues(*mass, 2, indices, 2, indices,
            values, ADD_VALUES));
    }
    PetscCall(finish_matrix(*mass, true, true));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode assemble_derivative_x_2d(const MPI_Comm comm,
    const int degree, const std::vector<double> &knots_x,
    const int ny, Mat *derivative)
{
    PetscFunctionBeginUser;
    const int nx = basis_count(degree, knots_x);
    const PetscInt rows = static_cast<PetscInt>(nx - 1) * ny;
    const PetscInt columns = static_cast<PetscInt>(nx) * ny;
    PetscCall(create_aij(comm, rows, columns, 2, derivative));
    PetscInt begin = 0;
    PetscInt end = 0;
    PetscCall(MatGetOwnershipRange(*derivative, &begin, &end));
    for (PetscInt row = begin; row < end; ++row)
    {
        const int a = row % (nx - 1);
        const int j = row / (nx - 1);
        const double denominator = knots_x[a + degree + 1] - knots_x[a + 1];
        PetscCheck(denominator > 0.0, comm, PETSC_ERR_ARG_WRONG,
            "D_x has a nonpositive knot span");
        const PetscScalar scale = static_cast<double>(degree) / denominator;
        const PetscInt columns_row[2] = {
            derivative_greville_id_2d(a, j, nx),
            derivative_greville_id_2d(a + 1, j, nx)};
        const PetscScalar values[2] = {-scale, scale};
        PetscCall(MatSetValues(*derivative, 1, &row, 2, columns_row,
            values, INSERT_VALUES));
    }
    PetscCall(finish_matrix(*derivative, false, false));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode assemble_derivative_y_2d(const MPI_Comm comm,
    const int degree, const std::vector<double> &knots_y,
    const int nx, Mat *derivative)
{
    PetscFunctionBeginUser;
    const int ny = basis_count(degree, knots_y);
    const PetscInt rows = static_cast<PetscInt>(nx) * (ny - 1);
    const PetscInt columns = static_cast<PetscInt>(nx) * ny;
    PetscCall(create_aij(comm, rows, columns, 2, derivative));
    PetscInt begin = 0;
    PetscInt end = 0;
    PetscCall(MatGetOwnershipRange(*derivative, &begin, &end));
    for (PetscInt row = begin; row < end; ++row)
    {
        const int i = row % nx;
        const int b = row / nx;
        const double denominator = knots_y[b + degree + 1] - knots_y[b + 1];
        PetscCheck(denominator > 0.0, comm, PETSC_ERR_ARG_WRONG,
            "D_y has a nonpositive knot span");
        const PetscScalar scale = static_cast<double>(degree) / denominator;
        const PetscInt columns_row[2] = {
            derivative_greville_id_2d(i, b, nx),
            derivative_greville_id_2d(i, b + 1, nx)};
        const PetscScalar values[2] = {-scale, scale};
        PetscCall(MatSetValues(*derivative, 1, &row, 2, columns_row,
            values, INSERT_VALUES));
    }
    PetscCall(finish_matrix(*derivative, false, false));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode assemble_mixed_collocation(const MPI_Comm comm,
    const int degree_x, const std::vector<double> &knots_x,
    const std::vector<double> &points_x,
    const int degree_y, const std::vector<double> &knots_y,
    const std::vector<double> &points_y, Mat *collocation)
{
    PetscFunctionBeginUser;
    const int nx = basis_count(degree_x, knots_x);
    const int ny = basis_count(degree_y, knots_y);
    PetscCheck(static_cast<int>(points_x.size()) == nx &&
        static_cast<int>(points_y.size()) == ny, comm, PETSC_ERR_ARG_SIZ,
        "Mixed collocation grid and basis dimensions differ");
    const PetscInt size = static_cast<PetscInt>(nx) * ny;
    PetscCall(create_aij(comm, size, size,
        (degree_x + 1) * (degree_y + 1), collocation));
    const BSplineBasis1D basis_x(degree_x, knots_x);
    const BSplineBasis1D basis_y(degree_y, knots_y);
    PetscInt begin = 0;
    PetscInt end = 0;
    PetscCall(MatGetOwnershipRange(*collocation, &begin, &end));
    for (PetscInt row = begin; row < end; ++row)
    {
        const int i = row % nx;
        const int j = row / nx;
        const auto x = basis_x.evaluate_N(points_x[i]);
        const auto y = basis_y.evaluate_N(points_y[j]);
        for (int ly = 0; ly <= degree_y; ++ly)
            for (int lx = 0; lx <= degree_x; ++lx)
            {
                const double value = x.N[lx] * y.N[ly];
                if (value == 0.0)
                    continue;
                const PetscInt column = derivative_greville_id_2d(
                    x.first_basis + lx, y.first_basis + ly, nx);
                PetscCall(MatSetValue(*collocation, row, column,
                    value, INSERT_VALUES));
            }
    }
    PetscCall(finish_matrix(*collocation, false, false));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode assemble_q1_mass_2d(const MPI_Comm comm,
    const std::vector<double> &points_x,
    const std::vector<double> &points_y,
    const double scale, Mat *mass)
{
    PetscFunctionBeginUser;
    PetscCheck(points_x.size() >= 2 && points_y.size() >= 2,
        comm, PETSC_ERR_ARG_SIZ, "Q1 mass requires a nonempty cell grid");
    const int nx = points_x.size();
    const int ny = points_y.size();
    PetscCall(create_aij(comm, static_cast<PetscInt>(nx) * ny,
        static_cast<PetscInt>(nx) * ny, 9, mass));
    int rank = 0;
    int ranks = 1;
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCallMPI(MPI_Comm_size(comm, &ranks));
    int cell_number = 0;
    constexpr double pattern[16] = {
        4,2,1,2, 2,4,2,1, 1,2,4,2, 2,1,2,4};
    for (int j = 0; j + 1 < ny; ++j)
        for (int i = 0; i + 1 < nx; ++i, ++cell_number)
        {
            if (cell_number % ranks != rank)
                continue;
            const double hx = points_x[i + 1] - points_x[i];
            const double hy = points_y[j + 1] - points_y[j];
            PetscCheck(hx > 0.0 && hy > 0.0, comm, PETSC_ERR_ARG_WRONG,
                "Q1 Greville grid is not strictly increasing");
            const PetscInt ids[4] = {
                derivative_greville_id_2d(i, j, nx),
                derivative_greville_id_2d(i + 1, j, nx),
                derivative_greville_id_2d(i + 1, j + 1, nx),
                derivative_greville_id_2d(i, j + 1, nx)};
            PetscScalar values[16];
            const double factor = scale * hx * hy / 36.0;
            for (int entry = 0; entry < 16; ++entry)
                values[entry] = factor * pattern[entry];
            PetscCall(MatSetValues(*mass, 4, ids, 4, ids,
                values, ADD_VALUES));
        }
    PetscCall(finish_matrix(*mass, true, true));
    PetscFunctionReturn(PETSC_SUCCESS);
}

std::size_t table_index(const int span, const int point,
    const int local, const int points, const int degree)
{
    return (static_cast<std::size_t>(span) * points + point) *
        (degree + 1) + local;
}

PetscErrorCode assemble_mixed_bspline_mass_2d(const MPI_Comm comm,
    const int degree_x, const std::vector<double> &knots_x,
    const int degree_y, const std::vector<double> &knots_y,
    const double scale, Mat *mass)
{
    PetscFunctionBeginUser;
    const int nx = basis_count(degree_x, knots_x);
    const int ny = basis_count(degree_y, knots_y);
    const int order = std::max(degree_x, degree_y) + 1;
    const QuadratureRule1D quadrature = gauss_legendre_rule(order);
    const BSplineBasis1D basis_x(degree_x, knots_x, quadrature.points);
    const BSplineBasis1D basis_y(degree_y, knots_y, quadrature.points);
    const auto &Nx = basis_x.N_Gauss();
    const auto &Ny = basis_y.N_Gauss();
    const PetscInt size = static_cast<PetscInt>(nx) * ny;
    const int support = (degree_x + 1) * (degree_y + 1);
    PetscCall(create_aij(comm, size, size,
        (2 * degree_x + 1) * (2 * degree_y + 1), mass));
    int rank = 0;
    int ranks = 1;
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCallMPI(MPI_Comm_size(comm, &ranks));
    int element = 0;
    for (int sy = degree_y; sy < ny; ++sy)
    {
        const double bottom = knots_y[sy];
        const double top = knots_y[sy + 1];
        if (!(top > bottom))
            continue;
        for (int sx = degree_x; sx < nx; ++sx)
        {
            const double left = knots_x[sx];
            const double right = knots_x[sx + 1];
            if (!(right > left))
                continue;
            const int owner = element++ % ranks;
            if (owner != rank)
                continue;
            std::vector<PetscInt> ids(support);
            for (int ly = 0; ly <= degree_y; ++ly)
                for (int lx = 0; lx <= degree_x; ++lx)
                    ids[lx + (degree_x + 1) * ly] =
                        derivative_greville_id_2d(
                            sx - degree_x + lx,
                            sy - degree_y + ly, nx);
            std::vector<PetscScalar> local(
                static_cast<std::size_t>(support) * support, 0.0);
            for (int qy = 0; qy < order; ++qy)
                for (int qx = 0; qx < order; ++qx)
                {
                    const double weight = scale * 0.25 *
                        (right - left) * (top - bottom) *
                        quadrature.weights[qx] * quadrature.weights[qy];
                    for (int ay = 0; ay <= degree_y; ++ay)
                        for (int ax = 0; ax <= degree_x; ++ax)
                        {
                            const int a = ax + (degree_x + 1) * ay;
                            const double va =
                                Nx[table_index(sx, qx, ax, order, degree_x)] *
                                Ny[table_index(sy, qy, ay, order, degree_y)];
                            for (int by = 0; by <= degree_y; ++by)
                                for (int bx = 0; bx <= degree_x; ++bx)
                                {
                                    const int b = bx + (degree_x + 1) * by;
                                    const double vb =
                                        Nx[table_index(sx, qx, bx, order, degree_x)] *
                                        Ny[table_index(sy, qy, by, order, degree_y)];
                                    local[static_cast<std::size_t>(a) * support + b] +=
                                        weight * va * vb;
                                }
                        }
                }
            PetscCall(MatSetValues(*mass, support, ids.data(), support,
                ids.data(), local.data(), ADD_VALUES));
        }
    }
    PetscCall(finish_matrix(*mass, true, true));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode assemble_high_stiffness_2d(const MPI_Comm comm,
    const int degree, const std::vector<double> &knots_x,
    const std::vector<double> &knots_y,
    const std::vector<int> &free_map,
    const double weight_x, const double weight_y, Mat *stiffness)
{
    PetscFunctionBeginUser;
    const int nx = basis_count(degree, knots_x);
    const int ny = basis_count(degree, knots_y);
    const int free_size = (nx - 2) * (ny - 2);
    const int order = degree + 1;
    const QuadratureRule1D quadrature = gauss_legendre_rule(order);
    const BSplineBasis1D basis_x(degree, knots_x, quadrature.points);
    const BSplineBasis1D basis_y(degree, knots_y, quadrature.points);
    const auto &Nx = basis_x.N_Gauss();
    const auto &Dx = basis_x.N_der1_Gauss();
    const auto &Ny = basis_y.N_Gauss();
    const auto &Dy = basis_y.N_der1_Gauss();
    PetscCall(create_aij(comm, free_size, free_size,
        (2 * degree + 1) * (2 * degree + 1), stiffness));
    int rank = 0;
    int ranks = 1;
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCallMPI(MPI_Comm_size(comm, &ranks));
    int element = 0;
    for (int sy = degree; sy < ny; ++sy)
    {
        const double bottom = knots_y[sy];
        const double top = knots_y[sy + 1];
        if (!(top > bottom))
            continue;
        for (int sx = degree; sx < nx; ++sx)
        {
            const double left = knots_x[sx];
            const double right = knots_x[sx + 1];
            if (!(right > left))
                continue;
            const int owner = element++ % ranks;
            if (owner != rank)
                continue;
            std::vector<PetscInt> ids;
            std::vector<int> locals;
            for (int ly = 0; ly <= degree; ++ly)
                for (int lx = 0; lx <= degree; ++lx)
                {
                    const int full = derivative_greville_id_2d(
                        sx - degree + lx, sy - degree + ly, nx);
                    if (free_map[full] >= 0)
                    {
                        ids.push_back(free_map[full]);
                        locals.push_back(lx + (degree + 1) * ly);
                    }
                }
            const int retained = ids.size();
            std::vector<PetscScalar> local(
                static_cast<std::size_t>(retained) * retained, 0.0);
            for (int qy = 0; qy < order; ++qy)
                for (int qx = 0; qx < order; ++qx)
                {
                    const double measure = 0.25 *
                        (right - left) * (top - bottom) *
                        quadrature.weights[qx] * quadrature.weights[qy];
                    for (int ar = 0; ar < retained; ++ar)
                    {
                        const int ax = locals[ar] % (degree + 1);
                        const int ay = locals[ar] / (degree + 1);
                        const std::size_t aix = table_index(
                            sx, qx, ax, order, degree);
                        const std::size_t aiy = table_index(
                            sy, qy, ay, order, degree);
                        for (int br = 0; br < retained; ++br)
                        {
                            const int bx = locals[br] % (degree + 1);
                            const int by = locals[br] / (degree + 1);
                            const std::size_t bix = table_index(
                                sx, qx, bx, order, degree);
                            const std::size_t biy = table_index(
                                sy, qy, by, order, degree);
                            local[static_cast<std::size_t>(ar) * retained + br] +=
                                measure * (
                                    weight_x * Dx[aix] * Dx[bix] *
                                        Ny[aiy] * Ny[biy] +
                                    weight_y * Nx[aix] * Nx[bix] *
                                        Dy[aiy] * Dy[biy]);
                        }
                    }
                }
            if (retained > 0)
                PetscCall(MatSetValues(*stiffness, retained, ids.data(),
                    retained, ids.data(), local.data(), ADD_VALUES));
        }
    }
    PetscCall(finish_matrix(*stiffness, true, true));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode form_ptap(Mat weight, Mat map, Mat *result)
{
    PetscFunctionBeginUser;
    PetscCall(MatPtAP(weight, map, MAT_INITIAL_MATRIX,
        PETSC_DEFAULT, result));
    PetscCall(MatSetOption(*result, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatSetOption(*result, MAT_SPD, PETSC_TRUE));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode destroy_1d_intermediates(DerivativeGreville1DResult &result)
{
    PetscFunctionBeginUser;
    PetscCall(MatDestroy(&result.derivative));
    PetscCall(MatDestroy(&result.restriction));
    PetscCall(MatDestroy(&result.derivative_free));
    PetscCall(MatDestroy(&result.collocation));
    PetscCall(MatDestroy(&result.greville_mass));
    PetscCall(MatDestroy(&result.exact_derivative_mass));
    PetscCall(MatDestroy(&result.sampled_derivative));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode destroy_2d_intermediates(DerivativeGreville2DResult &result)
{
    PetscFunctionBeginUser;
    PetscCall(MatDestroy(&result.derivative_x));
    PetscCall(MatDestroy(&result.derivative_y));
    PetscCall(MatDestroy(&result.restriction));
    PetscCall(MatDestroy(&result.derivative_x_free));
    PetscCall(MatDestroy(&result.derivative_y_free));
    PetscCall(MatDestroy(&result.collocation_x));
    PetscCall(MatDestroy(&result.collocation_y));
    PetscCall(MatDestroy(&result.greville_mass_x));
    PetscCall(MatDestroy(&result.greville_mass_y));
    PetscCall(MatDestroy(&result.exact_derivative_mass_x));
    PetscCall(MatDestroy(&result.exact_derivative_mass_y));
    PetscCall(MatDestroy(&result.sampled_derivative_x));
    PetscCall(MatDestroy(&result.sampled_derivative_y));
    PetscCall(MatDestroy(&result.preconditioner_x));
    PetscCall(MatDestroy(&result.preconditioner_y));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

std::vector<double> derivative_space_knots(const int degree,
    const std::vector<double> &original_knots)
{
    if (degree < 2)
        throw std::invalid_argument(
            "derivative_greville_mass requires original degree p>=2");
    static_cast<void>(basis_count(degree, original_knots));
    if (original_knots.size() < 3)
        throw std::invalid_argument("Original knot vector cannot be trimmed");
    std::vector<double> result(
        original_knots.begin() + 1, original_knots.end() - 1);
    const int original_count = basis_count(degree, original_knots);
    if (basis_count(degree - 1, result) != original_count - 1)
        throw std::logic_error("Derivative knot trimming produced wrong size");
    return result;
}

std::vector<double> strict_greville_points(const int degree,
    const std::vector<double> &knots)
{
    if (degree < 1)
        throw std::invalid_argument("Greville degree must be at least one");
    const int n = basis_count(degree, knots);
    std::vector<double> result(n, 0.0);
    for (int basis = 0; basis < n; ++basis)
        for (int offset = 1; offset <= degree; ++offset)
            result[basis] += knots[basis + offset] / degree;
    for (int i = 0; i + 1 < n; ++i)
        if (!(result[i + 1] > result[i]))
            throw std::invalid_argument(
                "Greville points must be strictly increasing");
    return result;
}

int derivative_greville_id_2d(const int i, const int j, const int extent_x)
{
    if (extent_x <= 0 || i < 0 || i >= extent_x || j < 0)
        throw std::out_of_range("Invalid x-fast derivative-space index");
    return i + extent_x * j;
}

PetscErrorCode assemble_derivative_greville_stiffness_1d(
    const MPI_Comm comm, const int degree, const std::vector<double> &knots,
    const DerivativeGrevilleAssemblyOptions &options,
    DerivativeGreville1DResult *result)
{
    PetscFunctionBeginUser;
    PetscCheck(result != nullptr, comm, PETSC_ERR_ARG_NULL,
        "Derivative Greville 1D result must not be null");
    PetscCheck(result->preconditioner == nullptr && result->stiffness == nullptr,
        comm, PETSC_ERR_ARG_WRONGSTATE,
        "Derivative Greville 1D result already owns matrices");
    try
    {
        result->degree = degree;
        result->knots = knots;
        result->num_full_dofs = basis_count(degree, knots);
        result->num_free_dofs = result->num_full_dofs - 2;
        result->num_derivative_dofs = result->num_full_dofs - 1;
        result->elements = positive_spans(degree, knots);
        result->derivative_knots = derivative_space_knots(degree, knots);
        result->original_greville_points = strict_greville_points(degree, knots);
        result->derivative_greville_points = strict_greville_points(
            degree - 1, result->derivative_knots);
    }
    catch (const std::exception &error)
    {
        SETERRQ(comm, PETSC_ERR_ARG_WRONG, "%s", error.what());
    }

    if (options.assemble_reference_stiffness)
    {
        PetscLogDouble start = 0.0;
        PetscCallMPI(MPI_Barrier(comm));
        PetscCall(PetscTime(&start));
        try
        {
            const SparseMatrix1D stiffness = assemble_bspline_stiffness_1d(
                degree, knots);
            PetscCall(sparse_1d_to_petsc(comm, stiffness, &result->stiffness));
        }
        catch (const std::exception &error)
        {
            SETERRQ(comm, PETSC_ERR_ARG_WRONG, "%s", error.what());
        }
        PetscCall(maximum_elapsed(comm, start,
            &result->assembly_stiffness_seconds));
        result->nnz_stiffness = matrix_nonzeros(result->stiffness);
    }

    PetscLogDouble start = 0.0;
    PetscCallMPI(MPI_Barrier(comm));
    PetscCall(PetscTime(&start));
    PetscCall(assemble_derivative_1d(
        comm, degree, knots, &result->derivative));
    const auto free_map = free_map_1d(result->num_full_dofs);
    PetscCall(assemble_restriction(comm, free_map,
        result->num_free_dofs, &result->restriction));
    PetscCall(MatMatMult(result->derivative, result->restriction,
        MAT_INITIAL_MATRIX, PETSC_DEFAULT, &result->derivative_free));
    PetscCall(assemble_collocation_1d(comm, degree - 1,
        result->derivative_knots, result->derivative_greville_points,
        &result->collocation));
    PetscCall(assemble_p1_mass(comm, result->derivative_greville_points,
        1.0, &result->greville_mass));
    if (options.assemble_exact_derivative_masses)
        try
        {
            const SparseMatrix1D exact = assemble_bspline_mass_1d(
                degree - 1, result->derivative_knots, false);
            PetscCall(sparse_1d_to_petsc(
                comm, exact, &result->exact_derivative_mass));
        }
        catch (const std::exception &error)
        {
            SETERRQ(comm, PETSC_ERR_ARG_WRONG, "%s", error.what());
        }
    PetscCall(MatMatMult(result->collocation, result->derivative_free,
        MAT_INITIAL_MATRIX, PETSC_DEFAULT, &result->sampled_derivative));
    PetscCall(form_ptap(result->greville_mass,
        result->sampled_derivative, &result->preconditioner));
    PetscCall(maximum_elapsed(comm, start,
        &result->assembly_preconditioner_seconds));
    result->nnz_derivative = matrix_nonzeros(result->derivative);
    result->nnz_collocation = matrix_nonzeros(result->collocation);
    result->nnz_preconditioner = matrix_nonzeros(result->preconditioner);
    if (!options.retain_intermediates)
        PetscCall(destroy_1d_intermediates(*result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode assemble_derivative_greville_stiffness_2d(
    const MPI_Comm comm, const int degree,
    const std::vector<double> &knots_x,
    const std::vector<double> &knots_y,
    const double length_x, const double length_y,
    const DerivativeGrevilleAssemblyOptions &options,
    DerivativeGreville2DResult *result)
{
    PetscFunctionBeginUser;
    PetscCheck(result != nullptr, comm, PETSC_ERR_ARG_NULL,
        "Derivative Greville 2D result must not be null");
    PetscCheck(result->preconditioner == nullptr && result->stiffness == nullptr,
        comm, PETSC_ERR_ARG_WRONGSTATE,
        "Derivative Greville 2D result already owns matrices");
    PetscCheck(length_x > 0.0 && length_y > 0.0 &&
        std::isfinite(length_x) && std::isfinite(length_y), comm,
        PETSC_ERR_ARG_OUTOFRANGE, "Physical rectangle lengths must be positive");
    try
    {
        result->degree = degree;
        result->knots_x = knots_x;
        result->knots_y = knots_y;
        result->num_basis_x = basis_count(degree, knots_x);
        result->num_basis_y = basis_count(degree, knots_y);
        result->elements_x = positive_spans(degree, knots_x);
        result->elements_y = positive_spans(degree, knots_y);
        result->num_full_dofs = result->num_basis_x * result->num_basis_y;
        result->num_free_dofs =
            (result->num_basis_x - 2) * (result->num_basis_y - 2);
        result->num_x_derivative_dofs =
            (result->num_basis_x - 1) * result->num_basis_y;
        result->num_y_derivative_dofs =
            result->num_basis_x * (result->num_basis_y - 1);
        result->length_x = length_x;
        result->length_y = length_y;
        result->x_derivative_weight = length_y / length_x;
        result->y_derivative_weight = length_x / length_y;
        result->derivative_knots_x = derivative_space_knots(degree, knots_x);
        result->derivative_knots_y = derivative_space_knots(degree, knots_y);
        result->original_greville_x = strict_greville_points(degree, knots_x);
        result->original_greville_y = strict_greville_points(degree, knots_y);
        result->derivative_greville_x = strict_greville_points(
            degree - 1, result->derivative_knots_x);
        result->derivative_greville_y = strict_greville_points(
            degree - 1, result->derivative_knots_y);
    }
    catch (const std::exception &error)
    {
        SETERRQ(comm, PETSC_ERR_ARG_WRONG, "%s", error.what());
    }
    const auto free_map = free_map_2d(
        result->num_basis_x, result->num_basis_y);

    if (options.assemble_reference_stiffness)
    {
        PetscLogDouble start = 0.0;
        PetscCallMPI(MPI_Barrier(comm));
        PetscCall(PetscTime(&start));
        PetscCall(assemble_high_stiffness_2d(comm, degree,
            knots_x, knots_y, free_map,
            result->x_derivative_weight, result->y_derivative_weight,
            &result->stiffness));
        PetscCall(maximum_elapsed(comm, start,
            &result->assembly_stiffness_seconds));
        result->nnz_stiffness = matrix_nonzeros(result->stiffness);
    }

    PetscLogDouble start = 0.0;
    PetscCallMPI(MPI_Barrier(comm));
    PetscCall(PetscTime(&start));
    PetscCall(assemble_derivative_x_2d(comm, degree, knots_x,
        result->num_basis_y, &result->derivative_x));
    PetscCall(assemble_derivative_y_2d(comm, degree, knots_y,
        result->num_basis_x, &result->derivative_y));
    PetscCall(assemble_restriction(comm, free_map,
        result->num_free_dofs, &result->restriction));
    PetscCall(MatMatMult(result->derivative_x, result->restriction,
        MAT_INITIAL_MATRIX, PETSC_DEFAULT, &result->derivative_x_free));
    PetscCall(MatMatMult(result->derivative_y, result->restriction,
        MAT_INITIAL_MATRIX, PETSC_DEFAULT, &result->derivative_y_free));

    PetscCall(assemble_mixed_collocation(comm,
        degree - 1, result->derivative_knots_x,
        result->derivative_greville_x,
        degree, knots_y, result->original_greville_y,
        &result->collocation_x));
    PetscCall(assemble_mixed_collocation(comm,
        degree, knots_x, result->original_greville_x,
        degree - 1, result->derivative_knots_y,
        result->derivative_greville_y,
        &result->collocation_y));

    PetscCall(assemble_q1_mass_2d(comm,
        result->derivative_greville_x, result->original_greville_y,
        result->x_derivative_weight, &result->greville_mass_x));
    PetscCall(assemble_q1_mass_2d(comm,
        result->original_greville_x, result->derivative_greville_y,
        result->y_derivative_weight, &result->greville_mass_y));
    if (options.assemble_exact_derivative_masses)
    {
        PetscCall(assemble_mixed_bspline_mass_2d(comm,
            degree - 1, result->derivative_knots_x,
            degree, knots_y, result->x_derivative_weight,
            &result->exact_derivative_mass_x));
        PetscCall(assemble_mixed_bspline_mass_2d(comm,
            degree, knots_x, degree - 1, result->derivative_knots_y,
            result->y_derivative_weight,
            &result->exact_derivative_mass_y));
    }

    PetscCall(MatMatMult(result->collocation_x, result->derivative_x_free,
        MAT_INITIAL_MATRIX, PETSC_DEFAULT, &result->sampled_derivative_x));
    PetscCall(MatMatMult(result->collocation_y, result->derivative_y_free,
        MAT_INITIAL_MATRIX, PETSC_DEFAULT, &result->sampled_derivative_y));
    PetscCall(form_ptap(result->greville_mass_x,
        result->sampled_derivative_x, &result->preconditioner_x));
    PetscCall(form_ptap(result->greville_mass_y,
        result->sampled_derivative_y, &result->preconditioner_y));
    PetscCall(MatDuplicate(result->preconditioner_x,
        MAT_COPY_VALUES, &result->preconditioner));
    PetscCall(MatAXPY(result->preconditioner, 1.0,
        result->preconditioner_y, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatSetOption(result->preconditioner,
        MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatSetOption(result->preconditioner, MAT_SPD, PETSC_TRUE));
    PetscCall(maximum_elapsed(comm, start,
        &result->assembly_preconditioner_seconds));

    result->nnz_derivative_x = matrix_nonzeros(result->derivative_x);
    result->nnz_derivative_y = matrix_nonzeros(result->derivative_y);
    result->nnz_collocation_x = matrix_nonzeros(result->collocation_x);
    result->nnz_collocation_y = matrix_nonzeros(result->collocation_y);
    result->nnz_preconditioner = matrix_nonzeros(result->preconditioner);
    if (!options.retain_intermediates)
        PetscCall(destroy_2d_intermediates(*result));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode destroy_derivative_greville_result(
    DerivativeGreville1DResult *result)
{
    PetscFunctionBeginUser;
    if (result == nullptr)
        PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(destroy_1d_intermediates(*result));
    PetscCall(MatDestroy(&result->preconditioner));
    PetscCall(MatDestroy(&result->stiffness));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode destroy_derivative_greville_result(
    DerivativeGreville2DResult *result)
{
    PetscFunctionBeginUser;
    if (result == nullptr)
        PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(destroy_2d_intermediates(*result));
    PetscCall(MatDestroy(&result->preconditioner));
    PetscCall(MatDestroy(&result->stiffness));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace lor
