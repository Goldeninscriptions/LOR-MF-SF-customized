#include "Assembly/TruncatedCholeskyTensor.hpp"

#include "Assembly/LexicographicOrdering.hpp"
#include "Mesh/BSplinePatch.hpp"
#include "Mesh/ID.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace lor
{
namespace
{

using Clock = std::chrono::steady_clock;

template <int Dim>
std::array<double, Dim> validate_affine_axis_aligned(
    const BSplinePatch<Dim> &patch)
{
    std::array<double, Dim> lower_parameter{};
    std::array<double, Dim> upper_parameter{};
    for (int direction = 0; direction < Dim; ++direction)
    {
        const auto &knots = patch.knots(direction);
        lower_parameter[direction] = knots[patch.degree(direction)];
        upper_parameter[direction] =
            knots[patch.num_basis(direction)];
    }
    const auto origin = patch.evaluate_geometry(lower_parameter);
    std::array<std::array<double, Dim>, Dim> increments{};
    std::array<double, Dim> lengths{};
    double scale = 1.0;
    for (int direction = 0; direction < Dim; ++direction)
    {
        auto endpoint = lower_parameter;
        endpoint[direction] = upper_parameter[direction];
        const auto mapped = patch.evaluate_geometry(endpoint);
        for (int component = 0; component < Dim; ++component)
        {
            increments[direction][component] =
                mapped[component] - origin[component];
            scale = std::max(scale, std::abs(mapped[component]));
        }
    }
    const double tolerance = 2.0e-11 * scale;
    for (int direction = 0; direction < Dim; ++direction)
    {
        for (int component = 0; component < Dim; ++component)
            if (component != direction &&
                std::abs(increments[direction][component]) > tolerance)
                throw std::invalid_argument(
                    "truncated_cholesky_tensor supports only axis-aligned "
                    "affine rectangles and boxes; this geometry has cross-axis mapping");
        lengths[direction] = increments[direction][direction];
        if (!(lengths[direction] > tolerance))
            throw std::invalid_argument(
                "truncated_cholesky_tensor requires positive axis-aligned physical lengths");
    }

    // Check interior points, not only corners, so a curved mapping with the
    // same corner box cannot pass as affine.
    const double fractions[3] = {0.0, 0.37, 1.0};
    const int samples = Dim == 2 ? 9 : 27;
    for (int flat = 0; flat < samples; ++flat)
    {
        int code = flat;
        auto parameter = lower_parameter;
        std::array<double, Dim> fraction{};
        for (int direction = 0; direction < Dim; ++direction)
        {
            fraction[direction] = fractions[code % 3];
            code /= 3;
            parameter[direction] = lower_parameter[direction] +
                fraction[direction] *
                    (upper_parameter[direction] - lower_parameter[direction]);
        }
        const auto mapped = patch.evaluate_geometry(parameter);
        for (int component = 0; component < Dim; ++component)
        {
            double expected = origin[component];
            for (int direction = 0; direction < Dim; ++direction)
                expected += fraction[direction] *
                    increments[direction][component];
            if (std::abs(mapped[component] - expected) > tolerance)
                throw std::invalid_argument(
                    "truncated_cholesky_tensor supports only affine geometry; "
                    "the supplied patch is curved or non-affine");
        }
    }
    return lengths;
}

template <int Dim>
using TensorIndex = std::array<int, Dim>;

template <int Dim>
void accumulate_tensor_product(const int direction,
    const TensorIndex<Dim> &row,
    const std::array<const SparseMatrix1D *, Dim> &matrices,
    TensorIndex<Dim> &column, const double product,
    const LexicographicOrdering<Dim> &free_ordering,
    std::map<PetscInt, double> &entries)
{
    if (direction == Dim)
    {
        entries[free_ordering.global_index(column)] += product;
        return;
    }
    for (const auto &entry : matrices[direction]->rows[row[direction]])
    {
        column[direction] = entry.first;
        accumulate_tensor_product<Dim>(direction + 1, row, matrices,
            column, product * entry.second, free_ordering, entries);
    }
}

template <int Dim>
std::map<PetscInt, double> tensor_row(const TensorIndex<Dim> &row,
    const std::array<TruncatedDirectionResult1D, Dim> &directions,
    const std::array<double, Dim> &gamma, const bool approximate,
    const LexicographicOrdering<Dim> &free_ordering)
{
    std::map<PetscInt, double> result;
    TensorIndex<Dim> column{};
    for (int derivative_direction = 0;
         derivative_direction < Dim; ++derivative_direction)
    {
        std::array<const SparseMatrix1D *, Dim> matrices{};
        for (int direction = 0; direction < Dim; ++direction)
            if (direction == derivative_direction)
                matrices[direction] = approximate ?
                    &directions[direction].approximate_stiffness :
                    &directions[direction].exact_stiffness;
            else
                matrices[direction] = approximate ?
                    &directions[direction].value_mass.approximate_mass :
                    &directions[direction].value_mass.exact_mass;
        accumulate_tensor_product<Dim>(0, row, matrices, column,
            gamma[derivative_direction], free_ordering, result);
    }
    for (auto iterator = result.begin(); iterator != result.end(); )
        if (iterator->second == 0.0)
            iterator = result.erase(iterator);
        else
            ++iterator;
    return result;
}

std::uint64_t checked_add(const std::uint64_t left,
    const std::uint64_t right)
{
    if (right > std::numeric_limits<std::uint64_t>::max() - left)
        throw std::overflow_error("Tensor nonzero estimate overflowed uint64_t");
    return left + right;
}

std::uint64_t checked_multiply(const std::uint64_t left,
    const std::uint64_t right)
{
    if (left != 0 && right >
        std::numeric_limits<std::uint64_t>::max() / left)
        throw std::overflow_error("Tensor nonzero estimate overflowed uint64_t");
    return left * right;
}

template <int Dim>
std::uint64_t tensor_row_nonzero_upper_bound(
    const TensorIndex<Dim> &row,
    const std::array<TruncatedDirectionResult1D, Dim> &directions)
{
    std::uint64_t sum = 0;
    for (int derivative_direction = 0;
         derivative_direction < Dim; ++derivative_direction)
    {
        std::uint64_t product = 1;
        for (int direction = 0; direction < Dim; ++direction)
        {
            const auto &matrix = direction == derivative_direction ?
                directions[direction].approximate_stiffness :
                directions[direction].value_mass.approximate_mass;
            product = checked_multiply(product,
                static_cast<std::uint64_t>(matrix.rows[row[direction]].size()));
        }
        sum = checked_add(sum, product);
    }
    return sum;
}

template <int Dim>
double verify_tensor_identity(Mat stiffness,
    const std::array<TruncatedDirectionResult1D, Dim> &directions,
    const std::array<double, Dim> &gamma,
    const LexicographicOrdering<Dim> &free_ordering)
{
    PetscInt begin = 0;
    PetscInt end = 0;
    MatGetOwnershipRange(stiffness, &begin, &end);
    long double local_difference = 0.0;
    long double local_reference = 0.0;
    for (PetscInt row = begin; row < end; ++row)
    {
        const auto index = free_ordering.multi_index(row);
        const auto expected = tensor_row<Dim>(
            index, directions, gamma, false, free_ordering);
        PetscInt count = 0;
        const PetscInt *columns = nullptr;
        const PetscScalar *values = nullptr;
        MatGetRow(stiffness, row, &count, &columns, &values);
        std::map<PetscInt, double> actual;
        for (PetscInt entry = 0; entry < count; ++entry)
            actual[columns[entry]] = PetscRealPart(values[entry]);
        MatRestoreRow(stiffness, row, &count, &columns, &values);
        std::map<PetscInt, double> combined = expected;
        for (const auto &entry : actual)
            combined.emplace(entry.first, 0.0);
        for (const auto &entry : combined)
        {
            const auto expected_entry = expected.find(entry.first);
            const auto actual_entry = actual.find(entry.first);
            const long double e = expected_entry == expected.end() ? 0.0 :
                expected_entry->second;
            const long double a = actual_entry == actual.end() ? 0.0 :
                actual_entry->second;
            const long double delta = e - a;
            local_difference += delta * delta;
            local_reference += a * a;
        }
    }
    long double global_difference = 0.0;
    long double global_reference = 0.0;
    MPI_Allreduce(&local_difference, &global_difference, 1,
        MPI_LONG_DOUBLE, MPI_SUM,
        PetscObjectComm(reinterpret_cast<PetscObject>(stiffness)));
    MPI_Allreduce(&local_reference, &global_reference, 1,
        MPI_LONG_DOUBLE, MPI_SUM,
        PetscObjectComm(reinterpret_cast<PetscObject>(stiffness)));
    return std::sqrt(static_cast<double>(global_difference /
        std::max(global_reference,
            std::numeric_limits<long double>::min())));
}

} // namespace

template <int Dim>
PetscErrorCode assemble_truncated_cholesky_tensor(
    const BSplinePatch<Dim> &patch, const ID &ID_map,
    Mat high_order_stiffness, const int truncation_half_bandwidth,
    const bool compute_certificate, const bool verify_identity,
    const PetscLogDouble maximum_memory_bytes,
    TruncatedCholeskyTensorResult<Dim> *result)
{
    PetscFunctionBeginUser;
    const MPI_Comm comm = PetscObjectComm(
        reinterpret_cast<PetscObject>(high_order_stiffness));
    PetscCheck(result != nullptr, comm, PETSC_ERR_ARG_NULL,
        "Truncated tensor result must not be null");
    PetscCheck(result->preconditioner.matrix == nullptr, comm,
        PETSC_ERR_ARG_WRONGSTATE,
        "Truncated tensor result already owns a matrix");
    PetscCheck(maximum_memory_bytes > 0.0 &&
        std::isfinite(maximum_memory_bytes), comm, PETSC_ERR_ARG_OUTOFRANGE,
        "Truncated tensor memory limit must be finite and positive");
    try
    {
        result->physical_lengths = validate_affine_axis_aligned(patch);
        double volume = 1.0;
        for (const double length : result->physical_lengths)
            volume *= length;
        for (int direction = 0; direction < Dim; ++direction)
            result->gamma[direction] = volume /
                (result->physical_lengths[direction] *
                 result->physical_lengths[direction]);

        typename LexicographicOrdering<Dim>::Index free_shape{};
        for (int direction = 0; direction < Dim; ++direction)
        {
            if (patch.degree(direction) != patch.degree(0))
                throw std::invalid_argument(
                    "truncated_cholesky_tensor currently requires equal degree in every direction");
            result->directions[direction] = build_truncated_direction_1d(
                patch.degree(direction), patch.knots(direction),
                truncation_half_bandwidth, compute_certificate);
            free_shape[direction] = patch.num_basis(direction) - 2;
            const auto &value_mass = result->directions[direction].value_mass;
            const auto &derivative_mass =
                result->directions[direction].derivative_mass;
            result->time_1d_mass_assembly_seconds +=
                value_mass.mass_assembly_seconds +
                derivative_mass.mass_assembly_seconds;
            result->time_1d_full_cholesky_seconds +=
                value_mass.full_cholesky_seconds +
                derivative_mass.full_cholesky_seconds;
            result->time_1d_factor_truncation_seconds +=
                value_mass.truncation_seconds +
                derivative_mass.truncation_seconds;
            result->time_truncation_certificate_seconds +=
                value_mass.certificate_seconds +
                derivative_mass.certificate_seconds;
            result->time_1d_reduced_operators_seconds +=
                result->directions[direction].reduced_operator_seconds;
        }
        const LexicographicOrdering<Dim> free_ordering(free_shape);
        if (free_ordering.size() != ID_map.num_free_dofs())
            throw std::invalid_argument(
                "Dirichlet ID map is not the full-boundary tensor interior");
        for (int id = 0; id < ID_map.num_free_dofs(); ++id)
        {
            auto full = free_ordering.multi_index(id);
            for (int direction = 0; direction < Dim; ++direction)
                ++full[direction];
            const LexicographicOrdering<Dim> full_ordering(patch.basis_shape());
            if (ID_map.get_ID(full_ordering.global_index(full)) != id)
                throw std::invalid_argument(
                    "ID ordering is incompatible with tensor assembly");
        }

        PetscInt local_rows = 0;
        PetscInt local_columns = 0;
        PetscInt global_rows = 0;
        PetscInt global_columns = 0;
        PetscCall(MatGetLocalSize(high_order_stiffness,
            &local_rows, &local_columns));
        PetscCall(MatGetSize(high_order_stiffness,
            &global_rows, &global_columns));
        PetscInt begin = 0;
        PetscInt end = 0;
        PetscCall(MatGetOwnershipRange(high_order_stiffness, &begin, &end));

        // This overflow-safe structural preflight precedes all tensor-row map
        // construction.  It intentionally overcounts overlaps among the Dim
        // Kronecker summands and accounts for both AIJ storage and one
        // transient std::map row on every rank.
        std::uint64_t local_upper = 0;
        std::uint64_t local_maximum_row = 0;
        for (PetscInt row = begin; row < end; ++row)
        {
            const std::uint64_t row_upper = tensor_row_nonzero_upper_bound<Dim>(
                free_ordering.multi_index(row), result->directions);
            PetscCheck(row_upper <= static_cast<std::uint64_t>(
                std::numeric_limits<PetscInt>::max()), comm, PETSC_ERR_SUP,
                "A tensor row upper bound exceeds the PetscInt range");
            local_upper = checked_add(local_upper, row_upper);
            local_maximum_row = std::max(local_maximum_row, row_upper);
        }
        int comm_size = 1;
        PetscCallMPI(MPI_Comm_size(comm, &comm_size));
        PetscCheck(local_upper <= std::numeric_limits<std::uint64_t>::max() /
            static_cast<std::uint64_t>(comm_size), comm, PETSC_ERR_SUP,
            "Tensor nonzero estimate is too large for an MPI sum");
        std::uint64_t global_upper = 0;
        std::uint64_t global_transient_upper = 0;
        PetscCallMPI(MPI_Allreduce(&local_upper, &global_upper, 1,
            MPI_UINT64_T, MPI_SUM, comm));
        PetscCallMPI(MPI_Allreduce(&local_maximum_row,
            &global_transient_upper, 1, MPI_UINT64_T, MPI_SUM, comm));
        const long double aij_lower =
            static_cast<long double>(global_upper) *
                (sizeof(PetscScalar) + sizeof(PetscInt)) +
            static_cast<long double>(2) *
                (static_cast<long double>(global_rows) + 1.0L) *
                sizeof(PetscInt);
        constexpr std::size_t map_node_estimate =
            sizeof(std::pair<const PetscInt, double>) + 4 * sizeof(void *);
        const long double conservative = aij_lower +
            static_cast<long double>(global_upper) * sizeof(PetscInt) +
            static_cast<long double>(global_transient_upper) *
                map_node_estimate +
            static_cast<long double>(4) *
                (static_cast<long double>(global_rows) + comm_size) *
                sizeof(PetscInt);
        PetscCheck(conservative <=
            std::numeric_limits<PetscLogDouble>::max(), comm, PETSC_ERR_SUP,
            "Tensor memory estimate overflowed PetscLogDouble");
        result->conservative_memory_estimate_bytes =
            static_cast<PetscLogDouble>(conservative);
        result->memory_limit_bytes = maximum_memory_bytes;
        PetscCheck(result->conservative_memory_estimate_bytes <=
            maximum_memory_bytes, comm, PETSC_ERR_MEM,
            "Truncated tensor preflight conservative estimate %.3f MiB exceeds configured limit %.3f MiB; no tensor rows or PETSc matrix were allocated",
            result->conservative_memory_estimate_bytes / (1024.0 * 1024.0),
            maximum_memory_bytes / (1024.0 * 1024.0));

        if (verify_identity)
        {
            const auto verification_start = Clock::now();
            result->tensor_identity_relative_error =
                verify_tensor_identity<Dim>(high_order_stiffness,
                    result->directions, result->gamma, free_ordering);
            const double local_verification_seconds =
                std::chrono::duration<double>(Clock::now() -
                    verification_start).count();
            PetscCallMPI(MPI_Allreduce(&local_verification_seconds,
                &result->time_tensor_identity_verification_seconds, 1,
                MPI_DOUBLE, MPI_MAX, comm));
            result->tensor_identity_verification_computed = true;
            result->tensor_identity_verification_passed =
                result->tensor_identity_relative_error < 5.0e-11;
            PetscCheck(result->tensor_identity_verification_passed,
                comm, PETSC_ERR_PLIB,
                "Untruncated tensor stiffness disagrees with assembled K (relative Frobenius error %.6e)",
                result->tensor_identity_relative_error);
        }

        const auto tensor_start = Clock::now();
        std::vector<PetscInt> diagonal(local_rows, 0);
        std::vector<PetscInt> off_diagonal(local_rows, 0);
        std::uint64_t local_nonzeros = 0;
        for (PetscInt row = begin; row < end; ++row)
        {
            const auto entries = tensor_row<Dim>(free_ordering.multi_index(row),
                result->directions, result->gamma, true, free_ordering);
            for (const auto &entry : entries)
                if (entry.first >= begin && entry.first < end)
                    ++diagonal[row - begin];
                else
                    ++off_diagonal[row - begin];
            local_nonzeros = checked_add(local_nonzeros,
                static_cast<std::uint64_t>(entries.size()));
        }
        PetscCheck(local_nonzeros <= std::numeric_limits<std::uint64_t>::max() /
            static_cast<std::uint64_t>(comm_size), comm, PETSC_ERR_SUP,
            "Exact tensor nonzero count is too large for an MPI sum");
        std::uint64_t global_nonzeros = 0;
        PetscCallMPI(MPI_Allreduce(&local_nonzeros, &global_nonzeros, 1,
            MPI_UINT64_T, MPI_SUM, comm));
        PetscCheck(global_nonzeros <= static_cast<std::uint64_t>(
            std::numeric_limits<PetscInt>::max()), comm, PETSC_ERR_SUP,
            "Exact tensor nonzero count exceeds the PetscInt range");
        result->preconditioner.predicted_nonzeros =
            static_cast<PetscInt>(global_nonzeros);
        result->preconditioner.estimated_minimum_bytes =
            static_cast<PetscLogDouble>(
                result->preconditioner.predicted_nonzeros) *
                (sizeof(PetscScalar) + sizeof(PetscInt)) +
            2.0 * (static_cast<PetscLogDouble>(global_rows) + 1.0) *
                sizeof(PetscInt);
        PetscCall(PetscPrintf(comm,
            "[truncated tensor allocation] rows=%" PetscInt_FMT
            ", predicted_nnz=%" PetscInt_FMT
            ", estimated_minimum=%.3f MiB, conservative_preflight=%.3f MiB, limit=%.3f MiB\n",
            global_rows, result->preconditioner.predicted_nonzeros,
            result->preconditioner.estimated_minimum_bytes /
                (1024.0 * 1024.0),
            result->conservative_memory_estimate_bytes / (1024.0 * 1024.0),
            maximum_memory_bytes / (1024.0 * 1024.0)));
        PetscCall(MatCreateAIJ(comm, local_rows, local_columns,
            global_rows, global_columns, 0, diagonal.data(),
            0, off_diagonal.data(), &result->preconditioner.matrix));
        PetscCall(MatSetOption(result->preconditioner.matrix,
            MAT_SYMMETRIC, PETSC_TRUE));
        PetscCall(MatSetOption(result->preconditioner.matrix,
            MAT_SPD, PETSC_TRUE));
        PetscCall(MatSetOption(result->preconditioner.matrix,
            MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
        for (PetscInt row = begin; row < end; ++row)
        {
            const auto entries = tensor_row<Dim>(free_ordering.multi_index(row),
                result->directions, result->gamma, true, free_ordering);
            std::vector<PetscInt> columns;
            std::vector<PetscScalar> values;
            columns.reserve(entries.size());
            values.reserve(entries.size());
            for (const auto &entry : entries)
            {
                columns.push_back(entry.first);
                values.push_back(entry.second);
            }
            PetscCall(MatSetValues(result->preconditioner.matrix, 1, &row,
                static_cast<PetscInt>(columns.size()), columns.data(),
                values.data(), INSERT_VALUES));
        }
        PetscCall(MatAssemblyBegin(result->preconditioner.matrix,
            MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(result->preconditioner.matrix,
            MAT_FINAL_ASSEMBLY));
        const double local_tensor_seconds =
            std::chrono::duration<double>(Clock::now() - tensor_start).count();
        PetscCallMPI(MPI_Allreduce(&local_tensor_seconds,
            &result->time_tensor_assembly_seconds, 1,
            MPI_DOUBLE, MPI_MAX, comm));
        MatInfo info{};
        PetscCall(MatGetInfo(result->preconditioner.matrix,
            MAT_GLOBAL_SUM, &info));
        result->preconditioner.allocated_nonzeros = info.nz_allocated;
        result->preconditioner.used_nonzeros = info.nz_used;
        result->preconditioner.allocation_calls = info.mallocs;
        result->preconditioner.memory_bytes = info.memory;
        PetscBool symmetric = PETSC_FALSE;
        PetscCall(MatIsSymmetric(result->preconditioner.matrix,
            1.0e-11, &symmetric));
        result->symmetric = symmetric == PETSC_TRUE;
        Vec diagonal_vector = nullptr;
        PetscCall(MatCreateVecs(result->preconditioner.matrix,
            &diagonal_vector, nullptr));
        PetscCall(MatGetDiagonal(result->preconditioner.matrix,
            diagonal_vector));
        PetscCall(VecMin(diagonal_vector, nullptr,
            &result->minimum_diagonal));
        PetscCall(VecDestroy(&diagonal_vector));
        PetscCheck(result->symmetric && result->minimum_diagonal > 0.0,
            comm, PETSC_ERR_PLIB,
            "Assembled truncated tensor matrix failed symmetry/positive-diagonal checks");

        if (compute_certificate)
        {
            double beta_star = 0.0;
            bool all_below_one = true;
            for (const auto &direction : result->directions)
            {
                beta_star = std::max(beta_star,
                    direction.value_mass.certificate.beta);
                beta_star = std::max(beta_star,
                    direction.derivative_mass.certificate.beta);
                all_below_one = all_below_one &&
                    direction.value_mass.certificate.beta_below_one &&
                    direction.derivative_mass.certificate.beta_below_one;
            }
            result->certificate_bound_available = all_below_one;
            if (all_below_one)
                result->theoretical_kappa_lambda_bound = std::pow(
                    (1.0 + beta_star) / (1.0 - beta_star), 2 * Dim);
        }
    }
    catch (const std::exception &exception)
    {
        SETERRQ(comm, PETSC_ERR_ARG_WRONG, "%s", exception.what());
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

template PetscErrorCode assemble_truncated_cholesky_tensor<2>(
    const BSplinePatch<2> &, const ID &, Mat, int, bool, bool, PetscLogDouble,
    TruncatedCholeskyTensorResult<2> *);
template PetscErrorCode assemble_truncated_cholesky_tensor<3>(
    const BSplinePatch<3> &, const ID &, Mat, int, bool, bool, PetscLogDouble,
    TruncatedCholeskyTensorResult<3> *);

} // namespace lor
