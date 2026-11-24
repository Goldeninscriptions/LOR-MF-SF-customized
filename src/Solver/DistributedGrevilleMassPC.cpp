#include "Solver/DistributedGrevilleMassPC.hpp"

#include <slepceps.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lor
{
namespace
{

struct BandLU
{
    int size = 0;
    int lower = 0;
    int upper = 0;
    std::vector<double> entries;
    int width() const noexcept { return lower + upper + 1; }
    double value(const int row, const int column) const noexcept
    {
        if (column < row - lower || column > row + upper)
            return 0.0;
        return entries[static_cast<std::size_t>(row) * width() +
            column - row + lower];
    }
    double &at(const int row, const int column)
    {
        return entries[static_cast<std::size_t>(row) * width() +
            column - row + lower];
    }
};

struct TridiagonalCholesky
{
    std::vector<double> diagonal;
    std::vector<double> subdiagonal;
};

BandLU factor_collocation(const SparseMatrix1D &matrix)
{
    BandLU factor;
    factor.size = matrix.size();
    for (int row = 0; row < matrix.size(); ++row)
        for (const auto &entry : matrix.rows[row])
        {
            factor.lower = std::max(factor.lower, row - entry.first);
            factor.upper = std::max(factor.upper, entry.first - row);
        }
    factor.entries.assign(static_cast<std::size_t>(factor.size) *
        factor.width(), 0.0);
    for (int row = 0; row < matrix.size(); ++row)
        for (const auto &entry : matrix.rows[row])
            factor.at(row, entry.first) = entry.second;
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon();
    for (int pivot = 0; pivot < factor.size; ++pivot)
    {
        double scale = 0.0;
        for (int column = std::max(0, pivot - factor.lower);
             column <= std::min(factor.size - 1,
                 pivot + factor.upper); ++column)
            scale = std::max(scale,
                std::abs(factor.value(pivot, column)));
        if (!(std::abs(factor.value(pivot, pivot)) >
              tolerance * std::max(1.0, scale)))
            throw std::runtime_error(
                "Distributed Greville band LU encountered a zero pivot");
        for (int row = pivot + 1;
             row <= std::min(factor.size - 1,
                 pivot + factor.lower); ++row)
        {
            factor.at(row, pivot) /= factor.value(pivot, pivot);
            const double multiplier = factor.value(row, pivot);
            for (int column = pivot + 1;
                 column <= std::min(factor.size - 1,
                     pivot + factor.upper); ++column)
                if (column >= row - factor.lower &&
                    column <= row + factor.upper)
                    factor.at(row, column) -= multiplier *
                        factor.value(pivot, column);
        }
    }
    return factor;
}

TridiagonalCholesky factor_mass(const SparseMatrix1D &matrix)
{
    TridiagonalCholesky factor;
    factor.diagonal.resize(matrix.size());
    factor.subdiagonal.resize(std::max(0, matrix.size() - 1));
    factor.diagonal[0] = std::sqrt(matrix.value(0, 0));
    for (int row = 1; row < matrix.size(); ++row)
    {
        factor.subdiagonal[row - 1] = matrix.value(row, row - 1) /
            factor.diagonal[row - 1];
        const double pivot = matrix.value(row, row) -
            factor.subdiagonal[row - 1] * factor.subdiagonal[row - 1];
        if (!(pivot > 0.0))
            throw std::runtime_error(
                "Distributed Greville mass factor is not SPD");
        factor.diagonal[row] = std::sqrt(pivot);
    }
    return factor;
}

void solve_T(const BandLU &factor, double *line)
{
    for (int row = 0; row < factor.size; ++row)
        for (int column = std::max(0, row - factor.lower);
             column < row; ++column)
            line[row] -= factor.value(row, column) * line[column];
    for (int row = factor.size - 1; row >= 0; --row)
    {
        for (int column = row + 1;
             column <= std::min(factor.size - 1,
                 row + factor.upper); ++column)
            line[row] -= factor.value(row, column) * line[column];
        line[row] /= factor.value(row, row);
    }
}

void solve_T_transpose(const BandLU &factor, double *line)
{
    for (int row = 0; row < factor.size; ++row)
    {
        for (int column = std::max(0, row - factor.upper);
             column < row; ++column)
            line[row] -= factor.value(column, row) * line[column];
        line[row] /= factor.value(row, row);
    }
    for (int row = factor.size - 1; row >= 0; --row)
        for (int column = row + 1;
             column <= std::min(factor.size - 1,
                 row + factor.lower); ++column)
            line[row] -= factor.value(column, row) * line[column];
}

void solve_Mg(const TridiagonalCholesky &factor, double *line)
{
    const int size = static_cast<int>(factor.diagonal.size());
    line[0] /= factor.diagonal[0];
    for (int row = 1; row < size; ++row)
        line[row] = (line[row] - factor.subdiagonal[row - 1] *
            line[row - 1]) / factor.diagonal[row];
    line[size - 1] /= factor.diagonal[size - 1];
    for (int row = size - 2; row >= 0; --row)
        line[row] = (line[row] - factor.subdiagonal[row] *
            line[row + 1]) / factor.diagonal[row];
}

struct Partition
{
    int global = 0;
    int rank = 0;
    int ranks = 1;
    std::vector<int> counts;
    std::vector<int> offsets;
    int local_count = 0;
    int local_offset = 0;
};

Partition balanced_partition(const int global, const MPI_Comm comm)
{
    Partition result;
    result.global = global;
    MPI_Comm_rank(comm, &result.rank);
    MPI_Comm_size(comm, &result.ranks);
    result.counts.resize(result.ranks);
    result.offsets.resize(result.ranks);
    for (int rank = 0; rank < result.ranks; ++rank)
    {
        const int begin = static_cast<int>(
            static_cast<long long>(global) * rank / result.ranks);
        const int end = static_cast<int>(
            static_cast<long long>(global) * (rank + 1) / result.ranks);
        result.counts[rank] = end - begin;
        result.offsets[rank] = begin;
    }
    result.local_count = result.counts[result.rank];
    result.local_offset = result.offsets[result.rank];
    return result;
}

struct TransposePlan
{
    MPI_Comm comm = MPI_COMM_NULL;
    int nx = 0;
    int ny = 0;
    Partition x;
    Partition y;
    std::vector<int> forward_send_counts;
    std::vector<int> forward_send_displacements;
    std::vector<int> forward_recv_counts;
    std::vector<int> forward_recv_displacements;
    std::vector<double> send;
    std::vector<double> receive;
    std::vector<double> transposed;
};

TransposePlan make_transpose_plan(const MPI_Comm comm,
    const int nx, const int ny)
{
    TransposePlan plan;
    plan.comm = comm;
    plan.nx = nx;
    plan.ny = ny;
    plan.x = balanced_partition(nx, comm);
    plan.y = balanced_partition(ny, comm);
    const int ranks = plan.x.ranks;
    plan.forward_send_counts.resize(ranks);
    plan.forward_send_displacements.resize(ranks);
    plan.forward_recv_counts.resize(ranks);
    plan.forward_recv_displacements.resize(ranks);
    int send_total = 0;
    int recv_total = 0;
    for (int rank = 0; rank < ranks; ++rank)
    {
        plan.forward_send_displacements[rank] = send_total;
        plan.forward_recv_displacements[rank] = recv_total;
        plan.forward_send_counts[rank] =
            plan.y.local_count * plan.x.counts[rank];
        plan.forward_recv_counts[rank] =
            plan.x.local_count * plan.y.counts[rank];
        send_total += plan.forward_send_counts[rank];
        recv_total += plan.forward_recv_counts[rank];
    }
    plan.send.resize(send_total);
    plan.receive.resize(recv_total);
    plan.transposed.resize(
        static_cast<std::size_t>(plan.x.local_count) * ny);
    return plan;
}

void transpose_forward(TransposePlan &plan,
    const std::vector<double> &original, PetscLogDouble &communication)
{
    for (int destination = 0; destination < plan.x.ranks; ++destination)
    {
        int output = plan.forward_send_displacements[destination];
        for (int local_x = 0;
             local_x < plan.x.counts[destination]; ++local_x)
        {
            const int x = plan.x.offsets[destination] + local_x;
            for (int local_y = 0;
                 local_y < plan.y.local_count; ++local_y)
                plan.send[output++] =
                    original[static_cast<std::size_t>(local_y) * plan.nx + x];
        }
    }
    PetscLogDouble start = 0.0;
    PetscLogDouble end = 0.0;
    PetscTime(&start);
    MPI_Alltoallv(plan.send.data(), plan.forward_send_counts.data(),
        plan.forward_send_displacements.data(), MPI_DOUBLE,
        plan.receive.data(), plan.forward_recv_counts.data(),
        plan.forward_recv_displacements.data(), MPI_DOUBLE, plan.comm);
    PetscTime(&end);
    communication += end - start;
    for (int source = 0; source < plan.y.ranks; ++source)
    {
        int input = plan.forward_recv_displacements[source];
        for (int local_x = 0; local_x < plan.x.local_count; ++local_x)
            for (int local_y = 0; local_y < plan.y.counts[source]; ++local_y)
                plan.transposed[static_cast<std::size_t>(local_x) * plan.ny +
                    plan.y.offsets[source] + local_y] = plan.receive[input++];
    }
}

void transpose_backward(TransposePlan &plan,
    std::vector<double> &original, PetscLogDouble &communication)
{
    // Backward send counts are forward receive counts and vice versa.
    for (int destination = 0; destination < plan.y.ranks; ++destination)
    {
        int output = plan.forward_recv_displacements[destination];
        for (int local_x = 0; local_x < plan.x.local_count; ++local_x)
            for (int local_y = 0;
                 local_y < plan.y.counts[destination]; ++local_y)
                plan.receive[output++] =
                    plan.transposed[static_cast<std::size_t>(local_x) *
                        plan.ny + plan.y.offsets[destination] + local_y];
    }
    PetscLogDouble start = 0.0;
    PetscLogDouble end = 0.0;
    PetscTime(&start);
    MPI_Alltoallv(plan.receive.data(), plan.forward_recv_counts.data(),
        plan.forward_recv_displacements.data(), MPI_DOUBLE,
        plan.send.data(), plan.forward_send_counts.data(),
        plan.forward_send_displacements.data(), MPI_DOUBLE, plan.comm);
    PetscTime(&end);
    communication += end - start;
    for (int source = 0; source < plan.x.ranks; ++source)
    {
        int input = plan.forward_send_displacements[source];
        for (int local_x = 0; local_x < plan.x.counts[source]; ++local_x)
        {
            const int x = plan.x.offsets[source] + local_x;
            for (int local_y = 0; local_y < plan.y.local_count; ++local_y)
                original[static_cast<std::size_t>(local_y) * plan.nx + x] =
                    plan.send[input++];
        }
    }
}

void apply_sparse_lines(const SparseMatrix1D &matrix,
    const int lines, const int stride, const double *input, double *output)
{
    const int size = matrix.size();
    for (int line = 0; line < lines; ++line)
        for (int row = 0; row < size; ++row)
        {
            double value = 0.0;
            for (const auto &entry : matrix.rows[row])
                value += entry.second *
                    input[static_cast<std::size_t>(line) * size * stride +
                        entry.first * stride];
            output[static_cast<std::size_t>(line) * size * stride +
                row * stride] = value;
        }
}

struct TensorMassContext
{
    const SparseMatrix1D *x_mass = nullptr;
    const SparseMatrix1D *y_mass = nullptr;
    const SparseMatrix1D *x_second = nullptr;
    const SparseMatrix1D *y_second = nullptr;
    double first_weight = 1.0;
    double second_weight = 1.0;
    TransposePlan plan;
    std::vector<double> original;
    std::vector<double> work;
    std::vector<double> transposed_work;
    std::vector<double> accumulated;
    PetscLogDouble apply_seconds = 0.0;
    PetscInt applications = 0;
};

PetscErrorCode tensor_mass_mult(Mat matrix, Vec input, Vec output)
{
    PetscFunctionBeginUser;
    TensorMassContext *context = nullptr;
    PetscCall(MatShellGetContext(matrix, &context));
    PetscLogDouble start = 0.0;
    PetscLogDouble end = 0.0;
    PetscCall(PetscTime(&start));
    const PetscScalar *source = nullptr;
    PetscCall(VecGetArrayRead(input, &source));
    for (std::size_t index = 0; index < context->original.size(); ++index)
        context->original[index] = PetscRealPart(source[index]);
    PetscCall(VecRestoreArrayRead(input, &source));
    PetscLogDouble ignored_communication = 0.0;
    const auto apply_product = [&](const SparseMatrix1D &x_factor,
                                   const SparseMatrix1D &y_factor) {
        // Complete x lines are local in the original y-slab distribution.
        apply_sparse_lines(x_factor, context->plan.y.local_count, 1,
            context->original.data(), context->work.data());
        transpose_forward(context->plan, context->work,
            ignored_communication);
        apply_sparse_lines(y_factor, context->plan.x.local_count, 1,
            context->plan.transposed.data(),
            context->transposed_work.data());
        context->plan.transposed.swap(context->transposed_work);
        transpose_backward(context->plan, context->work,
            ignored_communication);
    };
    apply_product(*context->x_mass, *context->y_mass);
    if (context->first_weight != 1.0)
        for (double &value : context->work)
            value *= context->first_weight;
    if (context->x_second != nullptr && context->y_second != nullptr)
    {
        context->accumulated = context->work;
        apply_product(*context->x_second, *context->y_second);
        for (std::size_t index = 0; index < context->work.size(); ++index)
            context->work[index] = context->second_weight *
                context->work[index] + context->accumulated[index];
    }
    PetscScalar *destination = nullptr;
    PetscCall(VecGetArray(output, &destination));
    for (std::size_t index = 0; index < context->work.size(); ++index)
        destination[index] = context->work[index];
    PetscCall(VecRestoreArray(output, &destination));
    PetscCall(PetscTime(&end));
    context->apply_seconds += end - start;
    ++context->applications;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode destroy_tensor_mass(Mat matrix)
{
    PetscFunctionBeginUser;
    TensorMassContext *context = nullptr;
    PetscCall(MatShellGetContext(matrix, &context));
    delete context;
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

namespace
{

struct ParallelPCContext
{
    MPI_Comm comm = MPI_COMM_NULL;
    DistributedGrevillePCMode mode =
        DistributedGrevillePCMode::tensor_transpose;
    BandLU Tx;
    BandLU Ty;
    TridiagonalCholesky Mx;
    TridiagonalCholesky My;
    TransposePlan plan;
    std::vector<double> values;
    DistributedGrevillePCStatistics *statistics = nullptr;
};

void apply_complete_local(const BandLU &T,
    const TridiagonalCholesky &M, double *line)
{
    solve_T_transpose(T, line);
    solve_Mg(M, line);
    solve_T(T, line);
}

void pack_rows(const std::vector<double> &values, const int nx,
    const int local_row, const int rows, std::vector<double> &buffer)
{
    buffer.resize(static_cast<std::size_t>(rows) * nx);
    std::copy_n(values.begin() + static_cast<std::size_t>(local_row) * nx,
        static_cast<std::size_t>(rows) * nx, buffer.begin());
}

void timed_receive(std::vector<double> &buffer, const int count,
    const int source, const int tag, ParallelPCContext &context,
    DistributedGrevillePCApplyRecord &record)
{
    buffer.resize(count);
    PetscLogDouble start = 0.0;
    PetscLogDouble end = 0.0;
    PetscTime(&start);
    MPI_Recv(buffer.data(), count, MPI_DOUBLE, source, tag,
        context.comm, MPI_STATUS_IGNORE);
    PetscTime(&end);
    record.communication_seconds += end - start;
}

void timed_send(const std::vector<double> &buffer, const int destination,
    const int tag, ParallelPCContext &context,
    DistributedGrevillePCApplyRecord &record)
{
    PetscLogDouble start = 0.0;
    PetscLogDouble end = 0.0;
    PetscTime(&start);
    MPI_Send(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE,
        destination, tag, context.comm);
    PetscTime(&end);
    record.communication_seconds += end - start;
    record.communicated_bytes +=
        static_cast<PetscLogDouble>(buffer.size()) * sizeof(double);
}

double previous_value(const ParallelPCContext &context,
    const std::vector<double> &halo, const int halo_start,
    const int global_row, const int x)
{
    if (global_row >= context.plan.y.local_offset)
        return context.values[static_cast<std::size_t>(
            global_row - context.plan.y.local_offset) * context.plan.nx + x];
    return halo[static_cast<std::size_t>(global_row - halo_start) *
        context.plan.nx + x];
}

double next_value(const ParallelPCContext &context,
    const std::vector<double> &halo, const int halo_start,
    const int global_row, const int x)
{
    if (global_row < context.plan.y.local_offset +
        context.plan.y.local_count)
        return context.values[static_cast<std::size_t>(
            global_row - context.plan.y.local_offset) * context.plan.nx + x];
    return halo[static_cast<std::size_t>(global_row - halo_start) *
        context.plan.nx + x];
}

void distributed_T_transpose(ParallelPCContext &context,
    DistributedGrevillePCApplyRecord &record)
{
    const auto &factor = context.Ty;
    const auto &part = context.plan.y;
    const int nx = context.plan.nx;
    std::vector<double> halo;
    std::vector<double> boundary;
    // U^T forward solve.
    if (part.rank > 0)
        timed_receive(halo, factor.upper * nx, part.rank - 1, 410,
            context, record);
    const int previous_start = part.local_offset - factor.upper;
    for (int local = 0; local < part.local_count; ++local)
    {
        const int global = part.local_offset + local;
        for (int x = 0; x < nx; ++x)
        {
            double &value = context.values[
                static_cast<std::size_t>(local) * nx + x];
            for (int column = std::max(0, global - factor.upper);
                 column < global; ++column)
                value -= factor.value(column, global) * previous_value(
                    context, halo, previous_start, column, x);
            value /= factor.value(global, global);
        }
    }
    if (part.rank + 1 < part.ranks)
    {
        pack_rows(context.values, nx,
            part.local_count - factor.upper, factor.upper, boundary);
        timed_send(boundary, part.rank + 1, 410, context, record);
    }
    // L^T backward solve (unit diagonal).
    if (part.rank + 1 < part.ranks)
        timed_receive(halo, factor.lower * nx, part.rank + 1, 411,
            context, record);
    const int next_start = part.local_offset + part.local_count;
    for (int local = part.local_count - 1; local >= 0; --local)
    {
        const int global = part.local_offset + local;
        for (int x = 0; x < nx; ++x)
        {
            double &value = context.values[
                static_cast<std::size_t>(local) * nx + x];
            for (int column = global + 1;
                 column <= std::min(factor.size - 1,
                     global + factor.lower); ++column)
                value -= factor.value(column, global) * next_value(
                    context, halo, next_start, column, x);
        }
    }
    if (part.rank > 0)
    {
        pack_rows(context.values, nx, 0, factor.lower, boundary);
        timed_send(boundary, part.rank - 1, 411, context, record);
    }
}

void distributed_Mg(ParallelPCContext &context,
    DistributedGrevillePCApplyRecord &record)
{
    const auto &factor = context.My;
    const auto &part = context.plan.y;
    const int nx = context.plan.nx;
    std::vector<double> halo;
    std::vector<double> boundary;
    if (part.rank > 0)
        timed_receive(halo, nx, part.rank - 1, 420, context, record);
    for (int local = 0; local < part.local_count; ++local)
    {
        const int global = part.local_offset + local;
        for (int x = 0; x < nx; ++x)
        {
            double &value = context.values[
                static_cast<std::size_t>(local) * nx + x];
            if (global > 0)
            {
                const double previous = local > 0 ?
                    context.values[static_cast<std::size_t>(local - 1) *
                        nx + x] : halo[x];
                value -= factor.subdiagonal[global - 1] * previous;
            }
            value /= factor.diagonal[global];
        }
    }
    if (part.rank + 1 < part.ranks)
    {
        pack_rows(context.values, nx, part.local_count - 1, 1, boundary);
        timed_send(boundary, part.rank + 1, 420, context, record);
    }
    if (part.rank + 1 < part.ranks)
        timed_receive(halo, nx, part.rank + 1, 421, context, record);
    for (int local = part.local_count - 1; local >= 0; --local)
    {
        const int global = part.local_offset + local;
        for (int x = 0; x < nx; ++x)
        {
            double &value = context.values[
                static_cast<std::size_t>(local) * nx + x];
            if (global + 1 < part.global)
            {
                const double next = local + 1 < part.local_count ?
                    context.values[static_cast<std::size_t>(local + 1) *
                        nx + x] : halo[x];
                value -= factor.subdiagonal[global] * next;
            }
            value /= factor.diagonal[global];
        }
    }
    if (part.rank > 0)
    {
        pack_rows(context.values, nx, 0, 1, boundary);
        timed_send(boundary, part.rank - 1, 421, context, record);
    }
}

void distributed_T(ParallelPCContext &context,
    DistributedGrevillePCApplyRecord &record)
{
    const auto &factor = context.Ty;
    const auto &part = context.plan.y;
    const int nx = context.plan.nx;
    std::vector<double> halo;
    std::vector<double> boundary;
    // L forward solve (unit diagonal).
    if (part.rank > 0)
        timed_receive(halo, factor.lower * nx, part.rank - 1, 430,
            context, record);
    const int previous_start = part.local_offset - factor.lower;
    for (int local = 0; local < part.local_count; ++local)
    {
        const int global = part.local_offset + local;
        for (int x = 0; x < nx; ++x)
        {
            double &value = context.values[
                static_cast<std::size_t>(local) * nx + x];
            for (int column = std::max(0, global - factor.lower);
                 column < global; ++column)
                value -= factor.value(global, column) * previous_value(
                    context, halo, previous_start, column, x);
        }
    }
    if (part.rank + 1 < part.ranks)
    {
        pack_rows(context.values, nx,
            part.local_count - factor.lower, factor.lower, boundary);
        timed_send(boundary, part.rank + 1, 430, context, record);
    }
    // U backward solve.
    if (part.rank + 1 < part.ranks)
        timed_receive(halo, factor.upper * nx, part.rank + 1, 431,
            context, record);
    const int next_start = part.local_offset + part.local_count;
    for (int local = part.local_count - 1; local >= 0; --local)
    {
        const int global = part.local_offset + local;
        for (int x = 0; x < nx; ++x)
        {
            double &value = context.values[
                static_cast<std::size_t>(local) * nx + x];
            for (int column = global + 1;
                 column <= std::min(factor.size - 1,
                     global + factor.upper); ++column)
                value -= factor.value(global, column) * next_value(
                    context, halo, next_start, column, x);
            value /= factor.value(global, global);
        }
    }
    if (part.rank > 0)
    {
        pack_rows(context.values, nx, 0, factor.upper, boundary);
        timed_send(boundary, part.rank - 1, 431, context, record);
    }
}

PetscErrorCode apply_parallel_pc(PC pc, Vec input, Vec output)
{
    PetscFunctionBeginUser;
    ParallelPCContext *context = nullptr;
    PetscCall(PCShellGetContext(pc, &context));
    DistributedGrevillePCApplyRecord record;
    record.application = context->statistics == nullptr ? 0 :
        context->statistics->pc_applications + 1;
    PetscLogDouble total_start = 0.0;
    PetscLogDouble stage_start = 0.0;
    PetscLogDouble stage_end = 0.0;
    PetscCall(PetscTime(&total_start));
    const PetscScalar *source = nullptr;
    PetscCall(VecGetArrayRead(input, &source));
    for (std::size_t index = 0; index < context->values.size(); ++index)
        context->values[index] = PetscRealPart(source[index]);
    PetscCall(VecRestoreArrayRead(input, &source));

    // All complete x lines are local in the y-slab layout.
    PetscCall(PetscTime(&stage_start));
    for (int local_y = 0; local_y < context->plan.y.local_count; ++local_y)
        apply_complete_local(context->Tx, context->Mx,
            context->values.data() +
                static_cast<std::size_t>(local_y) * context->plan.nx);
    PetscCall(PetscTime(&stage_end));
    record.local_direction_seconds = stage_end - stage_start;

    PetscCall(PetscTime(&stage_start));
    if (context->mode == DistributedGrevillePCMode::tensor_transpose)
    {
        transpose_forward(context->plan, context->values,
            record.communication_seconds);
        for (int local_x = 0; local_x < context->plan.x.local_count; ++local_x)
            apply_complete_local(context->Ty, context->My,
                context->plan.transposed.data() +
                    static_cast<std::size_t>(local_x) * context->plan.ny);
        transpose_backward(context->plan, context->values,
            record.communication_seconds);
        record.communicated_bytes = static_cast<PetscLogDouble>(
            context->values.size() + context->plan.transposed.size()) *
            sizeof(double);
    }
    else
    {
        distributed_T_transpose(*context, record);
        distributed_Mg(*context, record);
        distributed_T(*context, record);
    }
    PetscCall(PetscTime(&stage_end));
    record.distributed_direction_seconds = stage_end - stage_start;

    PetscScalar *destination = nullptr;
    PetscCall(VecGetArray(output, &destination));
    for (std::size_t index = 0; index < context->values.size(); ++index)
        destination[index] = context->values[index];
    PetscCall(VecRestoreArray(output, &destination));
    PetscCall(PetscTime(&stage_end));
    record.total_seconds = stage_end - total_start;
    if (context->statistics != nullptr)
    {
        ++context->statistics->pc_applications;
        context->statistics->pc_apply_seconds += record.total_seconds;
        context->statistics->pc_communication_seconds +=
            record.communication_seconds;
        context->statistics->communicated_bytes += record.communicated_bytes;
        context->statistics->applications.push_back(record);
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode destroy_parallel_pc(PC pc)
{
    PetscFunctionBeginUser;
    ParallelPCContext *context = nullptr;
    PetscCall(PCShellGetContext(pc, &context));
    delete context;
    PetscFunctionReturn(PETSC_SUCCESS);
}

struct MonitorContext
{
    DistributedGrevillePCStatistics *statistics = nullptr;
    PetscLogDouble start = 0.0;
    bool print = false;
};

PetscErrorCode monitor(KSP ksp, const PetscInt iteration,
    const PetscReal residual, void *raw)
{
    PetscFunctionBeginUser;
    auto *context = static_cast<MonitorContext *>(raw);
    PetscLogDouble now = 0.0;
    PetscCall(PetscTime(&now));
    context->statistics->iterations.push_back(
        {iteration, residual, now - context->start});
    if (context->print)
        PetscCall(PetscPrintf(PetscObjectComm(
            reinterpret_cast<PetscObject>(ksp)),
            "[distributed-greville-cg] iteration=%" PetscInt_FMT
            " residual=%.12e elapsed=%.6e s\n",
            iteration, static_cast<double>(residual), now - context->start));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscLogDouble factor_bytes(const ParallelPCContext &context)
{
    return static_cast<PetscLogDouble>(context.Tx.entries.capacity() +
        context.Ty.entries.capacity() +
        context.Mx.diagonal.capacity() +
        context.Mx.subdiagonal.capacity() +
        context.My.diagonal.capacity() +
        context.My.subdiagonal.capacity()) * sizeof(double);
}

PetscLogDouble workspace_bytes(const ParallelPCContext &context)
{
    return static_cast<PetscLogDouble>(context.values.capacity() +
        context.plan.send.capacity() + context.plan.receive.capacity() +
        context.plan.transposed.capacity()) * sizeof(double);
}

void reduce_statistics(const MPI_Comm comm,
    DistributedGrevillePCStatistics &statistics)
{
    double local_values[7] = {
        statistics.setup_seconds, statistics.solve_seconds,
        statistics.operator_apply_seconds, statistics.pc_apply_seconds,
        statistics.pc_communication_seconds,
        statistics.factor_memory_bytes, statistics.workspace_memory_bytes};
    double maxima[7]{};
    MPI_Allreduce(local_values, maxima, 7, MPI_DOUBLE, MPI_MAX, comm);
    statistics.setup_seconds = maxima[0];
    statistics.solve_seconds = maxima[1];
    statistics.operator_apply_seconds = maxima[2];
    statistics.pc_apply_seconds = maxima[3];
    statistics.pc_communication_seconds = maxima[4];
    statistics.factor_memory_bytes = maxima[5];
    statistics.workspace_memory_bytes = maxima[6];
    double global_bytes = 0.0;
    MPI_Allreduce(&statistics.communicated_bytes, &global_bytes, 1,
        MPI_DOUBLE, MPI_SUM, comm);
    statistics.communicated_bytes = global_bytes;
    for (auto &record : statistics.applications)
    {
        double local[4] = {record.local_direction_seconds,
            record.distributed_direction_seconds,
            record.communication_seconds, record.total_seconds};
        double maximum[4]{};
        MPI_Allreduce(local, maximum, 4, MPI_DOUBLE, MPI_MAX, comm);
        record.local_direction_seconds = maximum[0];
        record.distributed_direction_seconds = maximum[1];
        record.communication_seconds = maximum[2];
        record.total_seconds = maximum[3];
        double bytes = 0.0;
        MPI_Allreduce(&record.communicated_bytes, &bytes, 1,
            MPI_DOUBLE, MPI_SUM, comm);
        record.communicated_bytes = bytes;
    }
}

} // namespace

const char *distributed_greville_pc_mode_name(
    const DistributedGrevillePCMode mode) noexcept
{
    return mode == DistributedGrevillePCMode::tensor_transpose ?
        "tensor_transpose" : "distributed_band";
}

namespace
{

PetscErrorCode create_distributed_tensor_product_2d(const MPI_Comm comm,
    const SparseMatrix1D &x_factor, const SparseMatrix1D &y_factor, Mat *matrix,
    const SparseMatrix1D *x_second = nullptr,
    const SparseMatrix1D *y_second = nullptr,
    const double first_weight = 1.0,
    const double second_weight = 1.0)
{
    PetscFunctionBeginUser;
    PetscCheck(matrix != nullptr, comm, PETSC_ERR_ARG_NULL,
        "Output tensor matrix pointer must not be null");
    auto *context = new TensorMassContext();
    context->x_mass = &x_factor;
    context->y_mass = &y_factor;
    context->x_second = x_second;
    context->y_second = y_second;
    context->first_weight = first_weight;
    context->second_weight = second_weight;
    context->plan = make_transpose_plan(
        comm, x_factor.size(), y_factor.size());
    const PetscInt local_size = static_cast<PetscInt>(
        context->plan.y.local_count) * x_factor.size();
    const PetscInt global_size = static_cast<PetscInt>(x_factor.size()) *
        y_factor.size();
    context->original.resize(local_size);
    context->work.resize(local_size);
    context->transposed_work.resize(context->plan.transposed.size());
    if (x_second != nullptr && y_second != nullptr)
        context->accumulated.resize(local_size);
    PetscCall(MatCreateShell(comm, local_size, local_size,
        global_size, global_size, context, matrix));
    PetscCall(MatShellSetOperation(*matrix, MATOP_MULT,
        reinterpret_cast<void (*)(void)>(tensor_mass_mult)));
    PetscCall(MatShellSetOperation(*matrix, MATOP_DESTROY,
        reinterpret_cast<void (*)(void)>(destroy_tensor_mass)));
    PetscCall(MatSetOption(*matrix, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatSetOption(*matrix, MAT_SPD, PETSC_TRUE));
    PetscFunctionReturn(PETSC_SUCCESS);
}

ParallelPCContext *make_parallel_pc_context(const MPI_Comm comm,
    const DistributedGrevillePCConfiguration2D &configuration)
{
    if (configuration.x_direction == nullptr ||
        configuration.y_direction == nullptr)
        throw std::invalid_argument(
            "Distributed Greville PC directions must not be null");
    auto *context = new ParallelPCContext();
    try
    {
        context->comm = comm;
        context->mode = configuration.mode;
        context->statistics = configuration.statistics;
        context->Tx = factor_collocation(
            configuration.x_direction->collocation);
        context->Ty = factor_collocation(
            configuration.y_direction->collocation);
        context->Mx = factor_mass(
            configuration.x_direction->greville_mass);
        context->My = factor_mass(
            configuration.y_direction->greville_mass);
        context->plan = make_transpose_plan(comm,
            configuration.x_direction->collocation.size(),
            configuration.y_direction->collocation.size());
        const int required_bandwidth = std::max({context->Ty.lower,
            context->Ty.upper, 1});
        if (context->mode == DistributedGrevillePCMode::distributed_band &&
            *std::min_element(context->plan.y.counts.begin(),
                context->plan.y.counts.end()) < required_bandwidth)
            throw std::invalid_argument(
                "Distributed band solve requires every y slab to contain at least the factor bandwidth");
        context->values.resize(static_cast<std::size_t>(
            context->plan.y.local_count) * context->plan.nx);
        if (context->mode == DistributedGrevillePCMode::distributed_band)
        {
            context->plan.send.clear();
            context->plan.receive.clear();
            context->plan.transposed.clear();
        }
    }
    catch (...)
    {
        delete context;
        throw;
    }
    return context;
}

} // namespace

PetscErrorCode create_distributed_tensor_mass_2d(const MPI_Comm comm,
    const GrevilleMassPreconditioner1D &x_direction,
    const GrevilleMassPreconditioner1D &y_direction, Mat *matrix)
{
    PetscFunctionBeginUser;
    PetscCall(create_distributed_tensor_product_2d(comm,
        x_direction.spline_mass, y_direction.spline_mass, matrix));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode create_distributed_tensor_greville_preconditioner_2d(
    const MPI_Comm comm,
    const GrevilleMassPreconditioner1D &x_direction,
    const GrevilleMassPreconditioner1D &y_direction, Mat *matrix)
{
    PetscFunctionBeginUser;
    PetscCall(create_distributed_tensor_product_2d(comm,
        x_direction.transformed_mass, y_direction.transformed_mass, matrix));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode configure_distributed_greville_pc_2d(
    KSP ksp, void *raw_configuration)
{
    PetscFunctionBeginUser;
    PetscCheck(ksp != nullptr, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
        "KSP must not be null");
    const MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(ksp));
    auto *configuration = static_cast<DistributedGrevillePCConfiguration2D *>(
        raw_configuration);
    PetscCheck(configuration != nullptr, comm, PETSC_ERR_ARG_NULL,
        "Distributed Greville PC configuration must not be null");
    ParallelPCContext *context = nullptr;
    try
    {
        context = make_parallel_pc_context(comm, *configuration);
    }
    catch (const std::exception &error)
    {
        SETERRQ(comm, PETSC_ERR_ARG_WRONG, "%s", error.what());
    }
    PetscCall(KSPSetType(ksp, KSPPREONLY));
    PetscCall(KSPSetErrorIfNotConverged(ksp, PETSC_TRUE));
    PC pc = nullptr;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCSHELL));
    PetscCall(PCShellSetContext(pc, context));
    PetscCall(PCShellSetApply(pc, apply_parallel_pc));
    PetscCall(PCShellSetDestroy(pc, destroy_parallel_pc));
    PetscCall(PCShellSetName(pc,
        distributed_greville_pc_mode_name(configuration->mode)));
    PetscFunctionReturn(PETSC_SUCCESS);
}

namespace
{

struct DirectionalExtremes
{
    PetscReal minimum = 0.0;
    PetscReal maximum = 0.0;
    PetscReal minimum_error = 0.0;
    PetscReal maximum_error = 0.0;
    PetscInt minimum_iterations = 0;
    PetscInt maximum_iterations = 0;
    int minimum_reason = 0;
    int maximum_reason = 0;
    PetscLogDouble minimum_time = 0.0;
    PetscLogDouble maximum_time = 0.0;
};

PetscErrorCode create_sequential_sparse_matrix(
    const SparseMatrix1D &source, Mat *matrix)
{
    PetscFunctionBeginUser;
    PetscInt maximum_row = 1;
    for (const auto &row : source.rows)
        maximum_row = std::max<PetscInt>(maximum_row, row.size());
    PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, source.size(), source.size(),
        maximum_row, nullptr, matrix));
    for (PetscInt row = 0; row < source.size(); ++row)
        for (const auto &entry : source.rows[row])
            PetscCall(MatSetValue(*matrix, row, entry.first,
                entry.second, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(*matrix, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*matrix, MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetOption(*matrix, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatSetOption(*matrix, MAT_SPD, PETSC_TRUE));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode solve_directional_extreme(Mat matrix, const bool minimum,
    const PetscReal tolerance, const PetscInt maximum_iterations,
    const PetscInt requested_ncv, PetscReal *value, PetscReal *error,
    PetscInt *iterations, int *reason, PetscLogDouble *solve_time)
{
    PetscFunctionBeginUser;
    PetscInt size = 0;
    PetscCall(MatGetSize(matrix, &size, nullptr));
    EPS eps = nullptr;
    PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
    PetscCall(EPSSetOperators(eps, matrix, nullptr));
    PetscCall(EPSSetProblemType(eps, EPS_HEP));
    PetscCall(EPSSetType(eps, EPSKRYLOVSCHUR));
    PetscCall(EPSSetDimensions(eps, 1,
        std::min(size, requested_ncv), PETSC_DEFAULT));
    PetscCall(EPSSetTolerances(eps, tolerance, maximum_iterations));
    ST st = nullptr;
    PetscCall(EPSGetST(eps, &st));
    if (minimum)
    {
        PetscCall(EPSSetTarget(eps, 0.0));
        PetscCall(EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE));
        PetscCall(STSetType(st, STSINVERT));
        PetscCall(STSetShift(st, 0.0));
        KSP ksp = nullptr;
        PetscCall(STGetKSP(st, &ksp));
        PetscCall(KSPSetType(ksp, KSPPREONLY));
        PC pc = nullptr;
        PetscCall(KSPGetPC(ksp, &pc));
        PetscCall(PCSetType(pc, PCLU));
    }
    else
    {
        PetscCall(EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL));
        PetscCall(STSetType(st, STSHIFT));
    }
    PetscLogDouble start = 0.0;
    PetscLogDouble end = 0.0;
    PetscCall(PetscTime(&start));
    PetscCall(EPSSolve(eps));
    PetscCall(PetscTime(&end));
    *solve_time = end - start;
    EPSConvergedReason convergence = EPS_CONVERGED_ITERATING;
    PetscInt converged = 0;
    PetscCall(EPSGetConverged(eps, &converged));
    PetscCall(EPSGetConvergedReason(eps, &convergence));
    PetscCall(EPSGetIterationNumber(eps, iterations));
    PetscCheck(convergence > 0 && converged > 0, PETSC_COMM_SELF,
        PETSC_ERR_NOT_CONVERGED,
        "Directional Greville EPS extreme did not converge");
    PetscScalar eigenvalue = 0.0;
    PetscCall(EPSGetEigenvalue(eps, 0, &eigenvalue, nullptr));
    *value = PetscRealPart(eigenvalue);
    PetscCall(EPSComputeError(eps, 0, EPS_ERROR_RELATIVE, error));
    *reason = static_cast<int>(convergence);
    PetscCall(EPSDestroy(&eps));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode compute_directional_extremes(const SparseMatrix1D &factor,
    const PetscReal tolerance, const PetscInt maximum_iterations,
    const PetscInt ncv, DirectionalExtremes *result)
{
    PetscFunctionBeginUser;
    Mat matrix = nullptr;
    PetscCall(create_sequential_sparse_matrix(factor, &matrix));
    PetscCall(solve_directional_extreme(matrix, false, tolerance,
        maximum_iterations, ncv, &result->maximum,
        &result->maximum_error, &result->maximum_iterations,
        &result->maximum_reason, &result->maximum_time));
    PetscCall(solve_directional_extreme(matrix, true, tolerance,
        maximum_iterations, ncv, &result->minimum,
        &result->minimum_error, &result->minimum_iterations,
        &result->minimum_reason, &result->minimum_time));
    PetscCall(MatDestroy(&matrix));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

PetscErrorCode compute_tensor_greville_preconditioner_condition_2d(
    const MPI_Comm comm,
    const GrevilleMassPreconditioner1D &x_direction,
    const GrevilleMassPreconditioner1D &y_direction,
    const PetscReal eps_tolerance, const PetscInt max_iterations,
    const PetscInt subspace_dimension, GeneralizedConditionResult *result)
{
    PetscFunctionBeginUser;
    PetscCheck(result != nullptr, comm, PETSC_ERR_ARG_NULL,
        "Tensor Greville condition result must not be null");
    PetscMPIInt rank = 0;
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    double values[8]{};
    PetscInt integers[4]{};
    int reasons[4]{};
    if (rank == 0)
    {
        DirectionalExtremes x;
        DirectionalExtremes y;
        PetscCall(compute_directional_extremes(x_direction.transformed_mass,
            eps_tolerance, max_iterations, subspace_dimension, &x));
        PetscCall(compute_directional_extremes(y_direction.transformed_mass,
            eps_tolerance, max_iterations, subspace_dimension, &y));
        values[0] = x.minimum * y.minimum;
        values[1] = x.maximum * y.maximum;
        values[2] = std::max(x.minimum_error, y.minimum_error);
        values[3] = std::max(x.maximum_error, y.maximum_error);
        values[4] = x.minimum_time + y.minimum_time;
        values[5] = x.maximum_time + y.maximum_time;
        values[6] = values[4] + values[5];
        integers[0] = x.minimum_iterations + y.minimum_iterations;
        integers[1] = x.maximum_iterations + y.maximum_iterations;
        reasons[0] = x.minimum_reason;
        reasons[1] = x.maximum_reason;
    }
    PetscCallMPI(MPI_Bcast(values, 8, MPI_DOUBLE, 0, comm));
    PetscCallMPI(MPI_Bcast(integers, 4, MPIU_INT, 0, comm));
    PetscCallMPI(MPI_Bcast(reasons, 4, MPI_INT, 0, comm));
    *result = GeneralizedConditionResult{};
    result->lambda_min = values[0];
    result->lambda_max = values[1];
    result->condition_number = values[1] / values[0];
    result->min_eigenpair_error = values[2];
    result->max_eigenpair_error = values[3];
    result->min_eps_iterations = integers[0];
    result->max_eps_iterations = integers[1];
    result->min_convergence_reason = reasons[0];
    result->max_convergence_reason = reasons[1];
    result->min_solve_time = values[4];
    result->max_solve_time = values[5];
    result->total_time = values[6];
    result->converged = true;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode solve_mass_with_distributed_greville_pc_2d(Mat spline_mass,
    Vec rhs, const GrevilleMassPreconditioner1D &x_direction,
    const GrevilleMassPreconditioner1D &y_direction,
    const DistributedGrevillePCMode mode,
    const GrevilleMassPCSolveOptions &options,
    DistributedGrevillePCSolveResult *result)
{
    PetscFunctionBeginUser;
    const MPI_Comm comm = PetscObjectComm(
        reinterpret_cast<PetscObject>(spline_mass));
    PetscCheck(result != nullptr, comm, PETSC_ERR_ARG_NULL,
        "Distributed Greville result must not be null");
    PetscCheck(result->solution == nullptr, comm, PETSC_ERR_ARG_WRONGSTATE,
        "Distributed Greville result already owns a solution");
    const int nx = x_direction.spline_mass.size();
    const int ny = y_direction.spline_mass.size();
    auto *context = new ParallelPCContext();
    context->comm = comm;
    context->mode = mode;
    context->statistics = &result->statistics;
    PetscCall(PetscMemoryGetCurrentUsage(
        &result->statistics.process_memory_before_setup_bytes));
    PetscLogDouble setup_start = 0.0;
    PetscLogDouble setup_end = 0.0;
    PetscCall(PetscTime(&setup_start));
    try
    {
        context->Tx = factor_collocation(x_direction.collocation);
        context->Ty = factor_collocation(y_direction.collocation);
        context->Mx = factor_mass(x_direction.greville_mass);
        context->My = factor_mass(y_direction.greville_mass);
        context->plan = make_transpose_plan(comm, nx, ny);
        const int required_bandwidth = std::max({context->Ty.lower,
            context->Ty.upper, 1});
        if (mode == DistributedGrevillePCMode::distributed_band &&
            *std::min_element(context->plan.y.counts.begin(),
                context->plan.y.counts.end()) < required_bandwidth)
            throw std::invalid_argument(
                "Distributed band solve requires every y slab to contain at least the factor bandwidth");
        context->values.resize(
            static_cast<std::size_t>(context->plan.y.local_count) * nx);
        if (mode == DistributedGrevillePCMode::distributed_band)
        {
            context->plan.send.clear();
            context->plan.send.shrink_to_fit();
            context->plan.receive.clear();
            context->plan.receive.shrink_to_fit();
            context->plan.transposed.clear();
            context->plan.transposed.shrink_to_fit();
        }
    }
    catch (const std::exception &error)
    {
        delete context;
        SETERRQ(comm, PETSC_ERR_ARG_WRONG, "%s", error.what());
    }
    PetscCall(PetscTime(&setup_end));
    result->statistics.setup_seconds = setup_end - setup_start;
    result->statistics.factor_memory_bytes = factor_bytes(*context);
    result->statistics.workspace_memory_bytes = workspace_bytes(*context);
    PetscCall(PetscMemoryGetCurrentUsage(
        &result->statistics.process_memory_after_setup_bytes));

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
    PetscCall(PCShellSetApply(pc, apply_parallel_pc));
    PetscCall(PCShellSetDestroy(pc, destroy_parallel_pc));
    PetscCall(PCShellSetName(pc,
        distributed_greville_pc_mode_name(mode)));
    PetscCall(VecDuplicate(rhs, &result->solution));
    PetscCall(VecSet(result->solution, 0.0));
    MonitorContext monitor_context{&result->statistics, 0.0,
        options.print_iteration_monitor};
    TensorMassContext *mass_context = nullptr;
    PetscCall(MatShellGetContext(spline_mass, &mass_context));
    mass_context->apply_seconds = 0.0;
    mass_context->applications = 0;
    PetscCall(PetscTime(&monitor_context.start));
    PetscCall(KSPMonitorSet(ksp, monitor, &monitor_context, nullptr));
    PetscCall(KSPSolve(ksp, rhs, result->solution));
    PetscLogDouble solve_end = 0.0;
    PetscCall(PetscTime(&solve_end));
    result->statistics.solve_seconds = solve_end - monitor_context.start;
    PetscCall(KSPGetIterationNumber(ksp, &result->outer_iterations));
    PetscCall(KSPGetConvergedReason(ksp, &result->reason));
    result->statistics.operator_apply_seconds = mass_context->apply_seconds;
    result->statistics.operator_applications = mass_context->applications;

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
    reduce_statistics(comm, result->statistics);
    PetscCall(KSPDestroy(&ksp));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode destroy_distributed_greville_pc_result(
    DistributedGrevillePCSolveResult *result)
{
    PetscFunctionBeginUser;
    if (result != nullptr)
        PetscCall(VecDestroy(&result->solution));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace lor
