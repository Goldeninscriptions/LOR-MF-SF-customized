#include "Solver/GrevilleMassPC.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace lor
{
namespace
{

struct BandLU
{
    int size = 0;
    int lower_bandwidth = 0;
    int upper_bandwidth = 0;
    std::vector<double> entries;

    int width() const noexcept
    {
        return lower_bandwidth + upper_bandwidth + 1;
    }

    double value(const int row, const int column) const noexcept
    {
        if (column < row - lower_bandwidth ||
            column > row + upper_bandwidth)
            return 0.0;
        return entries[static_cast<std::size_t>(row) * width() +
            column - row + lower_bandwidth];
    }

    double &at(const int row, const int column)
    {
        return entries[static_cast<std::size_t>(row) * width() +
            column - row + lower_bandwidth];
    }
};

struct TridiagonalCholesky
{
    std::vector<double> diagonal;
    std::vector<double> subdiagonal;
};

struct DirectionApplication
{
    BandLU collocation;
    TridiagonalCholesky greville_mass;
};

template <int Dim>
struct GrevillePCContext
{
    std::array<int, Dim> shape{};
    std::array<DirectionApplication, Dim> directions;
    std::vector<double> values;
    std::vector<double> line;
    GrevilleMassPCStatistics *statistics = nullptr;
};

BandLU factor_collocation(const SparseMatrix1D &matrix)
{
    BandLU result;
    result.size = matrix.size();
    for (int row = 0; row < matrix.size(); ++row)
        for (const auto &entry : matrix.rows[row])
        {
            result.lower_bandwidth = std::max(
                result.lower_bandwidth, row - entry.first);
            result.upper_bandwidth = std::max(
                result.upper_bandwidth, entry.first - row);
        }
    result.entries.assign(static_cast<std::size_t>(result.size) *
        result.width(), 0.0);
    for (int row = 0; row < matrix.size(); ++row)
        for (const auto &entry : matrix.rows[row])
            result.at(row, entry.first) = entry.second;

    // Greville B-spline collocation matrices are strictly totally positive.
    // Their natural-order band LU therefore needs no pivoting and preserves
    // the original bandwidth.
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon();
    for (int pivot = 0; pivot < result.size; ++pivot)
    {
        double row_scale = 0.0;
        for (int column = std::max(0,
                 pivot - result.lower_bandwidth);
             column <= std::min(result.size - 1,
                 pivot + result.upper_bandwidth); ++column)
            row_scale = std::max(row_scale,
                std::abs(result.value(pivot, column)));
        if (!(std::abs(result.value(pivot, pivot)) >
              tolerance * std::max(1.0, row_scale)))
            throw std::runtime_error(
                "Greville collocation band LU encountered a zero pivot");
        for (int row = pivot + 1;
             row <= std::min(result.size - 1,
                 pivot + result.lower_bandwidth); ++row)
        {
            result.at(row, pivot) /= result.value(pivot, pivot);
            const double multiplier = result.value(row, pivot);
            for (int column = pivot + 1;
                 column <= std::min(result.size - 1,
                     pivot + result.upper_bandwidth); ++column)
                if (column >= row - result.lower_bandwidth &&
                    column <= row + result.upper_bandwidth)
                    result.at(row, column) -= multiplier *
                        result.value(pivot, column);
        }
    }
    return result;
}

TridiagonalCholesky factor_greville_mass(const SparseMatrix1D &matrix)
{
    TridiagonalCholesky result;
    result.diagonal.resize(matrix.size());
    result.subdiagonal.resize(std::max(0, matrix.size() - 1));
    for (int row = 0; row < matrix.size(); ++row)
        for (const auto &entry : matrix.rows[row])
            if (std::abs(row - entry.first) > 1 && entry.second != 0.0)
                throw std::invalid_argument(
                    "Greville P1 mass matrix must be tridiagonal");
    result.diagonal[0] = std::sqrt(matrix.value(0, 0));
    for (int row = 1; row < matrix.size(); ++row)
    {
        result.subdiagonal[row - 1] =
            matrix.value(row, row - 1) / result.diagonal[row - 1];
        const double pivot = matrix.value(row, row) -
            result.subdiagonal[row - 1] * result.subdiagonal[row - 1];
        if (!(pivot > 0.0))
            throw std::runtime_error(
                "Greville P1 mass Cholesky encountered a nonpositive pivot");
        result.diagonal[row] = std::sqrt(pivot);
    }
    return result;
}

void solve_collocation(const BandLU &factor, std::vector<double> &line)
{
    for (int row = 0; row < factor.size; ++row)
        for (int column = std::max(0,
                 row - factor.lower_bandwidth); column < row; ++column)
            line[row] -= factor.value(row, column) * line[column];
    for (int row = factor.size - 1; row >= 0; --row)
    {
        for (int column = row + 1;
             column <= std::min(factor.size - 1,
                 row + factor.upper_bandwidth); ++column)
            line[row] -= factor.value(row, column) * line[column];
        line[row] /= factor.value(row, row);
    }
}

void solve_collocation_transpose(
    const BandLU &factor, std::vector<double> &line)
{
    // U^T y=b.
    for (int row = 0; row < factor.size; ++row)
    {
        for (int column = std::max(0,
                 row - factor.upper_bandwidth); column < row; ++column)
            line[row] -= factor.value(column, row) * line[column];
        line[row] /= factor.value(row, row);
    }
    // L^T x=y, with unit diagonal.
    for (int row = factor.size - 1; row >= 0; --row)
        for (int column = row + 1;
             column <= std::min(factor.size - 1,
                 row + factor.lower_bandwidth); ++column)
            line[row] -= factor.value(column, row) * line[column];
}

void solve_greville_mass(
    const TridiagonalCholesky &factor, std::vector<double> &line)
{
    line[0] /= factor.diagonal[0];
    for (int row = 1; row < static_cast<int>(line.size()); ++row)
        line[row] = (line[row] -
            factor.subdiagonal[row - 1] * line[row - 1]) /
            factor.diagonal[row];
    line.back() /= factor.diagonal.back();
    for (int row = static_cast<int>(line.size()) - 2; row >= 0; --row)
        line[row] = (line[row] -
            factor.subdiagonal[row] * line[row + 1]) /
            factor.diagonal[row];
}

enum class SolveKind
{
    collocation,
    collocation_transpose,
    greville_mass
};

template <int Dim>
PetscInt apply_direction(GrevillePCContext<Dim> &context,
    const int direction, const SolveKind kind)
{
    const int line_size = context.shape[direction];
    int stride = 1;
    for (int previous = 0; previous < direction; ++previous)
        stride *= context.shape[previous];
    const int block = stride * line_size;
    const int outer_blocks = static_cast<int>(context.values.size()) / block;
    context.line.resize(line_size);
    PetscInt solves = 0;
    for (int outer = 0; outer < outer_blocks; ++outer)
        for (int inner = 0; inner < stride; ++inner)
        {
            const int base = outer * block + inner;
            for (int index = 0; index < line_size; ++index)
                context.line[index] = context.values[base + index * stride];
            if (kind == SolveKind::collocation)
                solve_collocation(
                    context.directions[direction].collocation, context.line);
            else if (kind == SolveKind::collocation_transpose)
                solve_collocation_transpose(
                    context.directions[direction].collocation, context.line);
            else
                solve_greville_mass(
                    context.directions[direction].greville_mass, context.line);
            for (int index = 0; index < line_size; ++index)
                context.values[base + index * stride] = context.line[index];
            ++solves;
        }
    return solves;
}

template <int Dim>
PetscErrorCode apply_pc(PC pc, Vec input, Vec output)
{
    PetscFunctionBeginUser;
    GrevillePCContext<Dim> *context = nullptr;
    PetscCall(PCShellGetContext(pc, &context));
    GrevilleMassPCApplyRecord record;
    record.application = context->statistics->pc_applications + 1;
    PetscLogDouble total_start = 0.0;
    PetscLogDouble stage_start = 0.0;
    PetscLogDouble stage_end = 0.0;
    PetscCall(PetscTime(&total_start));

    PetscCall(PetscTime(&stage_start));
    const PetscScalar *source = nullptr;
    PetscScalar *destination = nullptr;
    PetscCall(VecGetArrayRead(input, &source));
    for (std::size_t index = 0; index < context->values.size(); ++index)
        context->values[index] = PetscRealPart(source[index]);
    PetscCall(VecRestoreArrayRead(input, &source));
    PetscCall(PetscTime(&stage_end));
    record.copy_seconds += stage_end - stage_start;

    PetscCall(PetscTime(&stage_start));
    for (int direction = 0; direction < Dim; ++direction)
        record.transpose_T_line_solves += apply_direction(
            *context, direction, SolveKind::collocation_transpose);
    PetscCall(PetscTime(&stage_end));
    record.transpose_T_seconds = stage_end - stage_start;

    PetscCall(PetscTime(&stage_start));
    for (int direction = 0; direction < Dim; ++direction)
        record.greville_mass_line_solves += apply_direction(
            *context, direction, SolveKind::greville_mass);
    PetscCall(PetscTime(&stage_end));
    record.greville_mass_seconds = stage_end - stage_start;

    PetscCall(PetscTime(&stage_start));
    for (int direction = 0; direction < Dim; ++direction)
        record.T_line_solves += apply_direction(
            *context, direction, SolveKind::collocation);
    PetscCall(PetscTime(&stage_end));
    record.T_seconds = stage_end - stage_start;

    PetscCall(PetscTime(&stage_start));
    PetscCall(VecGetArray(output, &destination));
    for (std::size_t index = 0; index < context->values.size(); ++index)
        destination[index] = context->values[index];
    PetscCall(VecRestoreArray(output, &destination));
    PetscCall(PetscTime(&stage_end));
    record.copy_seconds += stage_end - stage_start;
    record.total_seconds = stage_end - total_start;
    ++context->statistics->pc_applications;
    context->statistics->total_apply_seconds += record.total_seconds;
    context->statistics->total_copy_seconds += record.copy_seconds;
    context->statistics->total_transpose_T_seconds +=
        record.transpose_T_seconds;
    context->statistics->total_greville_mass_seconds +=
        record.greville_mass_seconds;
    context->statistics->total_T_seconds += record.T_seconds;
    context->statistics->applications.push_back(record);
    PetscFunctionReturn(PETSC_SUCCESS);
}

template <int Dim>
PetscErrorCode destroy_pc(PC pc)
{
    PetscFunctionBeginUser;
    GrevillePCContext<Dim> *context = nullptr;
    PetscCall(PCShellGetContext(pc, &context));
    delete context;
    PetscFunctionReturn(PETSC_SUCCESS);
}

struct MonitorContext
{
    GrevilleMassPCStatistics *statistics = nullptr;
    PetscLogDouble start = 0.0;
    bool print = false;
};

PetscErrorCode monitor_iteration(KSP ksp, const PetscInt iteration,
    const PetscReal residual, void *raw_context)
{
    PetscFunctionBeginUser;
    auto *context = static_cast<MonitorContext *>(raw_context);
    PetscLogDouble now = 0.0;
    PetscCall(PetscTime(&now));
    context->statistics->iterations.push_back(
        {iteration, residual, now - context->start});
    if (context->print)
        PetscCall(PetscPrintf(PetscObjectComm(
            reinterpret_cast<PetscObject>(ksp)),
            "[greville-mass-cg] iteration=%" PetscInt_FMT
            " residual=%.12e elapsed=%.6e s\n",
            iteration, static_cast<double>(residual), now - context->start));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template <int Dim>
PetscLogDouble factor_memory(const GrevillePCContext<Dim> &context)
{
    PetscLogDouble bytes = 0.0;
    for (const auto &direction : context.directions)
        bytes += static_cast<PetscLogDouble>(
            direction.collocation.entries.capacity() +
            direction.greville_mass.diagonal.capacity() +
            direction.greville_mass.subdiagonal.capacity()) * sizeof(double);
    return bytes;
}

PetscLogDouble sparse_storage(const SparseMatrix1D &matrix)
{
    PetscLogDouble bytes = static_cast<PetscLogDouble>(
        matrix.rows.capacity()) * sizeof(matrix.rows[0]);
    for (const auto &row : matrix.rows)
        bytes += static_cast<PetscLogDouble>(row.capacity()) *
            sizeof(row[0]);
    return bytes;
}

PetscLogDouble source_storage(
    const GrevilleMassPreconditioner1D &direction)
{
    return static_cast<PetscLogDouble>(direction.knots.capacity() +
        direction.greville_points.capacity()) * sizeof(double) +
        sparse_storage(direction.spline_mass) +
        sparse_storage(direction.greville_mass) +
        sparse_storage(direction.collocation) +
        sparse_storage(direction.transformed_mass);
}

} // namespace

template <int Dim>
PetscErrorCode solve_mass_with_greville_tensor_pc(Mat spline_mass, Vec rhs,
    const std::array<GrevilleMassPreconditioner1D, Dim> &directions,
    const GrevilleMassPCSolveOptions &options,
    GrevilleMassPCSolveResult *result)
{
    PetscFunctionBeginUser;
    static_assert(Dim >= 1 && Dim <= 3,
        "Greville tensor PC supports one, two, or three dimensions");
    const MPI_Comm comm = PetscObjectComm(
        reinterpret_cast<PetscObject>(spline_mass));
    PetscCheck(result != nullptr, comm, PETSC_ERR_ARG_NULL,
        "Greville mass solve result must not be null");
    PetscCheck(result->solution == nullptr, comm, PETSC_ERR_ARG_WRONGSTATE,
        "Greville mass solve result already owns a solution");
    int comm_size = 0;
    PetscCallMPI(MPI_Comm_size(comm, &comm_size));
    PetscCheck(comm_size == 1, comm, PETSC_ERR_SUP,
        "Greville tensor PC currently requires one MPI rank; distributed tensor transpose is not implemented");
    PetscCheck(options.relative_tolerance > 0.0 &&
        options.absolute_tolerance >= 0.0 &&
        options.divergence_tolerance > 1.0 &&
        options.maximum_iterations > 0, comm, PETSC_ERR_ARG_OUTOFRANGE,
        "Invalid Greville mass CG tolerances");

    PetscInt rows = 0;
    PetscInt columns = 0;
    PetscCall(MatGetSize(spline_mass, &rows, &columns));
    PetscCheck(rows == columns, comm, PETSC_ERR_ARG_SIZ,
        "Spline mass matrix must be square");
    PetscInt vector_size = 0;
    PetscCall(VecGetSize(rhs, &vector_size));
    PetscCheck(vector_size == rows, comm, PETSC_ERR_ARG_SIZ,
        "Right-hand side size must match the spline mass matrix");

    auto *context = new GrevillePCContext<Dim>();
    context->statistics = &result->statistics;
    PetscInt tensor_size = 1;
    PetscLogDouble setup_start = 0.0;
    PetscLogDouble setup_end = 0.0;
    PetscCall(PetscMemoryGetCurrentUsage(
        &result->statistics.process_memory_before_setup_bytes));
    PetscCall(PetscTime(&setup_start));
    try
    {
        for (int direction = 0; direction < Dim; ++direction)
        {
            context->shape[direction] =
                directions[direction].spline_mass.size();
            if (context->shape[direction] <= 0 ||
                tensor_size > std::numeric_limits<PetscInt>::max() /
                    context->shape[direction])
                throw std::overflow_error(
                    "Greville tensor shape exceeds PetscInt");
            tensor_size *= context->shape[direction];
            context->directions[direction].collocation =
                factor_collocation(directions[direction].collocation);
            context->directions[direction].greville_mass =
                factor_greville_mass(directions[direction].greville_mass);
        }
        if (tensor_size != rows)
            throw std::invalid_argument(
                "Directional Greville factors do not match matrix size");
        context->values.resize(tensor_size);
        context->line.resize(*std::max_element(
            context->shape.begin(), context->shape.end()));
    }
    catch (const std::exception &error)
    {
        delete context;
        SETERRQ(comm, PETSC_ERR_ARG_WRONG, "%s", error.what());
    }
    PetscCall(PetscTime(&setup_end));
    PetscCall(PetscMemoryGetCurrentUsage(
        &result->statistics.process_memory_after_setup_bytes));
    result->statistics.setup_seconds = setup_end - setup_start;
    result->statistics.factor_memory_bytes = factor_memory(*context);
    for (const auto &direction : directions)
        result->statistics.source_factor_storage_bytes +=
            source_storage(direction);
    result->statistics.workspace_memory_bytes =
        static_cast<PetscLogDouble>(context->values.capacity() +
            context->line.capacity()) * sizeof(double);
    MatInfo info{};
    PetscCall(MatGetInfo(spline_mass, MAT_GLOBAL_SUM, &info));
    result->statistics.operator_reported_memory_bytes = info.memory;
    result->statistics.operator_minimum_memory_bytes =
        info.nz_used * (sizeof(PetscScalar) + sizeof(PetscInt)) +
        static_cast<PetscLogDouble>(rows + 1) * sizeof(PetscInt);
    result->statistics.estimated_total_memory_bytes =
        result->statistics.source_factor_storage_bytes +
        result->statistics.factor_memory_bytes +
        result->statistics.workspace_memory_bytes +
        std::max(result->statistics.operator_reported_memory_bytes,
            result->statistics.operator_minimum_memory_bytes) +
        static_cast<PetscLogDouble>(3) * rows * sizeof(PetscScalar);

    KSP ksp = nullptr;
    PetscCall(KSPCreate(comm, &ksp));
    PetscCall(KSPSetOperators(ksp, spline_mass, spline_mass));
    PetscCall(KSPSetType(ksp, KSPCG));
    PetscCall(KSPSetNormType(ksp, KSP_NORM_PRECONDITIONED));
    PetscCall(KSPSetTolerances(ksp, options.relative_tolerance,
        options.absolute_tolerance, options.divergence_tolerance,
        options.maximum_iterations));
    PetscCall(KSPSetErrorIfNotConverged(ksp, PETSC_TRUE));
    PC pc = nullptr;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCSHELL));
    PetscCall(PCShellSetContext(pc, context));
    PetscCall(PCShellSetApply(pc, apply_pc<Dim>));
    PetscCall(PCShellSetDestroy(pc, destroy_pc<Dim>));
    PetscCall(PCShellSetName(pc, "Greville tensor T^{-1}M_g^{-1}T^{-T}"));

    PetscCall(VecDuplicate(rhs, &result->solution));
    PetscCall(VecSet(result->solution, 0.0));
    MonitorContext monitor{&result->statistics, 0.0,
        options.print_iteration_monitor};
    PetscCall(PetscTime(&monitor.start));
    PetscCall(KSPMonitorSet(ksp, monitor_iteration, &monitor, nullptr));
    PetscCall(KSPSolve(ksp, rhs, result->solution));
    PetscLogDouble solve_end = 0.0;
    PetscCall(PetscTime(&solve_end));
    result->statistics.outer_solve_seconds = solve_end - monitor.start;
    PetscCall(KSPGetIterationNumber(ksp, &result->outer_iterations));
    PetscCall(KSPGetConvergedReason(ksp, &result->reason));

    Vec residual = nullptr;
    PetscCall(VecDuplicate(rhs, &residual));
    PetscCall(MatMult(spline_mass, result->solution, residual));
    PetscCall(VecAYPX(residual, -1.0, rhs));
    PetscCall(VecNorm(residual, NORM_2, &result->residual_norm));
    PetscReal rhs_norm = 0.0;
    PetscCall(VecNorm(rhs, NORM_2, &rhs_norm));
    result->relative_residual_norm = rhs_norm > 0.0 ?
        result->residual_norm / rhs_norm : result->residual_norm;
    PetscCall(VecDestroy(&residual));
    PetscCall(KSPDestroy(&ksp));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode destroy_greville_mass_pc_result(
    GrevilleMassPCSolveResult *result)
{
    PetscFunctionBeginUser;
    if (result != nullptr)
        PetscCall(VecDestroy(&result->solution));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template PetscErrorCode solve_mass_with_greville_tensor_pc<1>(Mat, Vec,
    const std::array<GrevilleMassPreconditioner1D, 1> &,
    const GrevilleMassPCSolveOptions &, GrevilleMassPCSolveResult *);
template PetscErrorCode solve_mass_with_greville_tensor_pc<2>(Mat, Vec,
    const std::array<GrevilleMassPreconditioner1D, 2> &,
    const GrevilleMassPCSolveOptions &, GrevilleMassPCSolveResult *);
template PetscErrorCode solve_mass_with_greville_tensor_pc<3>(Mat, Vec,
    const std::array<GrevilleMassPreconditioner1D, 3> &,
    const GrevilleMassPCSolveOptions &, GrevilleMassPCSolveResult *);

} // namespace lor
