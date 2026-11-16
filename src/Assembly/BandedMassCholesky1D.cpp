#include "Assembly/BandedMassCholesky1D.hpp"

#include "Element/BSplineBasis1D.hpp"
#include "Element/GaussQuadrature.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace lor
{
namespace
{

using Clock = std::chrono::steady_clock;

double elapsed_seconds(const Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

struct LowerBandFactor
{
    int size = 0;
    int half_bandwidth = 0;
    std::vector<double> entries;

    double value(const int row, const int column) const
    {
        if (row < column || row - column > half_bandwidth)
            return 0.0;
        return entries[static_cast<std::size_t>(row) *
            (half_bandwidth + 1) + (row - column)];
    }

    double &lower(const int row, const int column)
    {
        return entries[static_cast<std::size_t>(row) *
            (half_bandwidth + 1) + (row - column)];
    }
};

LowerBandFactor band_cholesky(const SymmetricBandMatrix1D &matrix)
{
    LowerBandFactor factor;
    factor.size = matrix.size();
    factor.half_bandwidth = matrix.half_bandwidth();
    factor.entries.assign(static_cast<std::size_t>(factor.size) *
        (factor.half_bandwidth + 1), 0.0);
    for (int row = 0; row < factor.size; ++row)
    {
        const int first_column = std::max(0, row - factor.half_bandwidth);
        for (int column = first_column; column <= row; ++column)
        {
            double entry = matrix.value(row, column);
            const int first_product = std::max({0,
                row - factor.half_bandwidth,
                column - factor.half_bandwidth});
            for (int inner = first_product; inner < column; ++inner)
                entry -= factor.value(row, inner) *
                    factor.value(column, inner);
            if (row == column)
            {
                if (!(entry > 0.0) || !std::isfinite(entry))
                    throw std::runtime_error(
                        "One-dimensional mass matrix is not positive definite");
                factor.lower(row, column) = std::sqrt(entry);
            }
            else
                factor.lower(row, column) =
                    entry / factor.value(column, column);
        }
    }
    return factor;
}

SparseMatrix1D band_to_sparse(const SymmetricBandMatrix1D &matrix)
{
    SparseMatrix1D result;
    result.rows.resize(matrix.size());
    for (int row = 0; row < matrix.size(); ++row)
        for (int column = std::max(0, row - matrix.half_bandwidth());
             column <= std::min(matrix.size() - 1,
                 row + matrix.half_bandwidth()); ++column)
        {
            const double entry = matrix.value(row, column);
            if (entry != 0.0)
                result.rows[row].emplace_back(column, entry);
        }
    return result;
}

SparseMatrix1D factor_product(const LowerBandFactor &factor)
{
    std::vector<std::map<int, double>> accumulated(factor.size);
    // L L^T is accumulated a column at a time from the nonzeros of L.  Every
    // temporary object is O(N w), never O(N^2).
    for (int column = 0; column < factor.size; ++column)
    {
        std::vector<std::pair<int, double>> active;
        for (int row = column;
             row <= std::min(factor.size - 1,
                 column + factor.half_bandwidth); ++row)
        {
            const double entry = factor.value(row, column);
            if (entry != 0.0)
                active.emplace_back(row, entry);
        }
        for (const auto &left : active)
            for (const auto &right : active)
                accumulated[left.first][right.first] +=
                    left.second * right.second;
    }
    SparseMatrix1D result;
    result.rows.resize(factor.size);
    for (int row = 0; row < factor.size; ++row)
        for (const auto &entry : accumulated[row])
            if (entry.second != 0.0)
                result.rows[row].push_back(entry);
    return result;
}

SymmetricBandMatrix1D assemble_mass(const int degree,
    const std::vector<double> &knots, const bool remove_endpoints)
{
    const int full_size = static_cast<int>(knots.size()) - degree - 1;
    const int offset = remove_endpoints ? 1 : 0;
    const int size = full_size - (remove_endpoints ? 2 : 0);
    if (size <= 0)
        throw std::invalid_argument(
            "One-dimensional mass matrix has no retained basis functions");
    const int bandwidth = std::min(degree, size - 1);
    SymmetricBandMatrix1D mass(size, bandwidth);
    const QuadratureRule1D quadrature = gauss_legendre_rule(degree + 1);
    const BSplineBasis1D basis(degree, knots, quadrature.points);
    const auto &values = basis.N_Gauss();
    const int points = static_cast<int>(quadrature.points.size());
    for (int span = degree; span < full_size; ++span)
    {
        const double left = knots[span];
        const double right = knots[span + 1];
        if (!(right > left))
            continue;
        for (int point = 0; point < points; ++point)
        {
            const double weight = 0.5 * (right - left) *
                quadrature.weights[point];
            for (int local_row = 0; local_row <= degree; ++local_row)
            {
                const int full_row = span - degree + local_row;
                const int row = full_row - offset;
                if (row < 0 || row >= size)
                    continue;
                const std::size_t row_index =
                    (static_cast<std::size_t>(span) * points + point) *
                        (degree + 1) + local_row;
                for (int local_column = 0;
                     local_column <= local_row; ++local_column)
                {
                    const int full_column =
                        span - degree + local_column;
                    const int column = full_column - offset;
                    if (column < 0 || column >= size || row < column)
                        continue;
                    const std::size_t column_index =
                        (static_cast<std::size_t>(span) * points + point) *
                            (degree + 1) + local_column;
                    mass.lower(row, column) += weight *
                        values[row_index] * values[column_index];
                }
            }
        }
    }
    return mass;
}

int actual_half_bandwidth(const SymmetricBandMatrix1D &matrix)
{
    int bandwidth = 0;
    const double scale = [&matrix]() {
        double maximum = 0.0;
        for (int row = 0; row < matrix.size(); ++row)
            maximum = std::max(maximum, std::abs(matrix.value(row, row)));
        return maximum;
    }();
    const double threshold = 64.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, scale);
    for (int row = 0; row < matrix.size(); ++row)
        for (int column = std::max(0,
                 row - matrix.half_bandwidth()); column < row; ++column)
            if (std::abs(matrix.value(row, column)) > threshold)
                bandwidth = std::max(bandwidth, row - column);
    return bandwidth;
}

std::vector<double> solve_lower(const LowerBandFactor &factor,
    const std::vector<double> &right_hand_side)
{
    std::vector<double> solution(factor.size, 0.0);
    for (int row = 0; row < factor.size; ++row)
    {
        double value = right_hand_side[row];
        for (int column = std::max(0,
                 row - factor.half_bandwidth); column < row; ++column)
            value -= factor.value(row, column) * solution[column];
        solution[row] = value / factor.value(row, row);
    }
    return solution;
}

std::vector<double> solve_upper(const LowerBandFactor &factor,
    const std::vector<double> &right_hand_side)
{
    std::vector<double> solution(factor.size, 0.0);
    for (int row = factor.size - 1; row >= 0; --row)
    {
        double value = right_hand_side[row];
        for (int column = row + 1;
             column <= std::min(factor.size - 1,
                 row + factor.half_bandwidth); ++column)
            value -= factor.value(column, row) * solution[column];
        solution[row] = value / factor.value(row, row);
    }
    return solution;
}

TruncationCertificate1D certificate(const LowerBandFactor &complete,
    const LowerBandFactor &truncated)
{
    const int size = complete.size;
    std::vector<double> row_sums(size, 0.0);
    for (int row = 0; row < size; ++row)
        for (int column = std::max(0,
                 row - complete.half_bandwidth); column <= row; ++column)
            row_sums[row] += complete.value(row, column) -
                truncated.value(row, column);

    std::vector<double> signed_row_sums(size, 0.0);
    std::vector<double> signed_ones(size, 0.0);
    for (int index = 0; index < size; ++index)
    {
        const double sign = index % 2 == 0 ? 1.0 : -1.0;
        signed_row_sums[index] = sign * row_sums[index];
        signed_ones[index] = sign;
    }
    std::vector<double> a = solve_lower(complete, signed_row_sums);
    std::vector<double> v = solve_upper(complete, signed_ones);
    for (int index = 0; index < size; ++index)
    {
        const double sign = index % 2 == 0 ? 1.0 : -1.0;
        a[index] *= sign;
        v[index] *= sign;
    }
    std::vector<double> b(size, 0.0);
    for (int row = 0; row < size; ++row)
        for (int column = std::max(0,
                 row - complete.half_bandwidth); column <= row; ++column)
        {
            const double error = complete.value(row, column) -
                truncated.value(row, column);
            b[column] += error * v[row];
        }
    TruncationCertificate1D result;
    result.computed = true;
    result.maximum_row_sum = *std::max_element(a.begin(), a.end());
    result.maximum_column_sum = *std::max_element(b.begin(), b.end());
    // Roundoff may create tiny negative estimates when no entry was dropped.
    result.maximum_row_sum = std::max(0.0, result.maximum_row_sum);
    result.maximum_column_sum = std::max(0.0, result.maximum_column_sum);
    result.beta = std::sqrt(result.maximum_row_sum *
        result.maximum_column_sum);
    result.beta_below_one = result.beta < 1.0;
    return result;
}

TruncatedMassResult1D build_mass(const int degree,
    const std::vector<double> &knots, const bool remove_endpoints,
    const int truncation_half_bandwidth, const bool compute_certificate)
{
    TruncatedMassResult1D result;
    result.degree = degree;
    result.boundary_dofs_removed = remove_endpoints;
    result.truncation_half_bandwidth = truncation_half_bandwidth;
    auto start = Clock::now();
    SymmetricBandMatrix1D mass = assemble_mass(
        degree, knots, remove_endpoints);
    result.mass_assembly_seconds = elapsed_seconds(start);
    result.size = mass.size();
    result.original_half_bandwidth = actual_half_bandwidth(mass);
    if (result.original_half_bandwidth <= 0)
        throw std::invalid_argument(
            "The participating one-dimensional mass matrix has zero half-bandwidth");
    if (truncation_half_bandwidth < 0 ||
        truncation_half_bandwidth >= result.original_half_bandwidth)
        throw std::invalid_argument(
            "trunc_chol_k must be nonnegative and strictly smaller than every "
            "participating one-dimensional mass-matrix half-bandwidth");
    result.exact_mass = band_to_sparse(mass);

    start = Clock::now();
    const LowerBandFactor complete = band_cholesky(mass);
    result.full_cholesky_seconds = elapsed_seconds(start);
    result.complete_factor_nonzeros = 0;
    for (int row = 0; row < complete.size; ++row)
        for (int column = std::max(0,
                 row - complete.half_bandwidth); column <= row; ++column)
            if (complete.value(row, column) != 0.0)
                ++result.complete_factor_nonzeros;

    start = Clock::now();
    LowerBandFactor truncated = complete;
    const int boundary_width = result.original_half_bandwidth;
    result.truncated_factor_nonzeros = result.complete_factor_nonzeros;
    for (int row = 0; row < truncated.size; ++row)
        for (int column = std::max(0,
                 row - truncated.half_bandwidth); column < row; ++column)
        {
            const bool row_boundary = row <= boundary_width ||
                truncated.size - 1 - row <= boundary_width;
            const bool column_boundary = column <= boundary_width ||
                truncated.size - 1 - column <= boundary_width;
            if (row - column > truncation_half_bandwidth &&
                !row_boundary && !column_boundary &&
                truncated.value(row, column) != 0.0)
            {
                truncated.lower(row, column) = 0.0;
                ++result.dropped_factor_entries;
                --result.truncated_factor_nonzeros;
            }
        }
    result.approximate_mass = factor_product(truncated);
    result.truncation_seconds = elapsed_seconds(start);
    const SparseMatrix1D reconstructed = factor_product(complete);
    result.full_reconstruction_relative_error =
        relative_frobenius_difference(result.exact_mass, reconstructed);

    if (compute_certificate)
    {
        start = Clock::now();
        result.certificate = certificate(complete, truncated);
        result.certificate_seconds = elapsed_seconds(start);
    }
    return result;
}

SparseMatrix1D difference_mass_stiffness(const SparseMatrix1D &mass,
    const int original_degree, const std::vector<double> &original_knots)
{
    const int original_basis =
        static_cast<int>(original_knots.size()) - original_degree - 1;
    const int free_size = original_basis - 2;
    std::vector<double> alpha(original_basis - 1, 0.0);
    for (int row = 0; row < original_basis - 1; ++row)
    {
        const double denominator = original_knots[row + original_degree + 1] -
            original_knots[row + 1];
        if (denominator > 0.0)
            alpha[row] = original_degree / denominator;
    }
    std::vector<std::map<int, double>> accumulated(free_size);

    // Each derivative-space row z_j depends on at most two retained control
    // coefficients. Traverse the actual nonzeros of M and scatter the at-most
    // four products D(j,:)^T M(j,l) D(l,:) instead of scanning all free
    // row/column pairs.
    const auto derivative_row = [&alpha, free_size](const int derivative) {
        std::array<std::pair<int, double>, 2> entries{{{-1, 0.0}, {-1, 0.0}}};
        int count = 0;
        // z_j contains +alpha_j*c_{j+1}.
        if (derivative < free_size)
            entries[count++] = {derivative, alpha[derivative]};
        // z_j contains -alpha_j*c_j; c_j is free only for j>=1.
        if (derivative > 0 && derivative - 1 < free_size)
            entries[count++] = {derivative - 1, -alpha[derivative]};
        return std::make_pair(entries, count);
    };
    for (int mass_row = 0; mass_row < mass.size(); ++mass_row)
    {
        const auto left = derivative_row(mass_row);
        for (const auto &mass_entry : mass.rows[mass_row])
        {
            const auto right = derivative_row(mass_entry.first);
            for (int i = 0; i < left.second; ++i)
                for (int j = 0; j < right.second; ++j)
                    accumulated[left.first[i].first][right.first[j].first] +=
                        left.first[i].second * mass_entry.second *
                        right.first[j].second;
        }
    }
    SparseMatrix1D result;
    result.rows.resize(free_size);
    for (int row = 0; row < free_size; ++row)
        for (const auto &entry : accumulated[row])
            result.rows[row].push_back(entry);
    return result;
}

SparseMatrix1D direct_stiffness(const int degree,
    const std::vector<double> &knots)
{
    const int full_size = static_cast<int>(knots.size()) - degree - 1;
    const int size = full_size - 2;
    const QuadratureRule1D quadrature = gauss_legendre_rule(degree + 1);
    const BSplineBasis1D basis(degree, knots, quadrature.points);
    const auto &derivatives = basis.N_der1_Gauss();
    const int points = static_cast<int>(quadrature.points.size());
    std::vector<std::map<int, double>> accumulated(size);
    for (int span = degree; span < full_size; ++span)
    {
        const double left = knots[span];
        const double right = knots[span + 1];
        if (!(right > left))
            continue;
        for (int point = 0; point < points; ++point)
        {
            const double weight = 0.5 * (right - left) *
                quadrature.weights[point];
            for (int local_row = 0; local_row <= degree; ++local_row)
            {
                const int row = span - degree + local_row - 1;
                if (row < 0 || row >= size)
                    continue;
                const std::size_t row_index =
                    (static_cast<std::size_t>(span) * points + point) *
                        (degree + 1) + local_row;
                for (int local_column = 0;
                     local_column <= degree; ++local_column)
                {
                    const int column = span - degree + local_column - 1;
                    if (column < 0 || column >= size)
                        continue;
                    const std::size_t column_index =
                        (static_cast<std::size_t>(span) * points + point) *
                            (degree + 1) + local_column;
                    accumulated[row][column] += weight *
                        derivatives[row_index] * derivatives[column_index];
                }
            }
        }
    }
    SparseMatrix1D result;
    result.rows.resize(size);
    for (int row = 0; row < size; ++row)
        for (const auto &entry : accumulated[row])
            if (entry.second != 0.0)
                result.rows[row].push_back(entry);
    return result;
}

} // namespace

SymmetricBandMatrix1D::SymmetricBandMatrix1D(
    const int size, const int half_bandwidth)
    : size_(size), half_bandwidth_(half_bandwidth),
      entries_(static_cast<std::size_t>(size) *
          (half_bandwidth + 1), 0.0)
{
    if (size <= 0 || half_bandwidth < 0 || half_bandwidth >= size)
        throw std::invalid_argument("Invalid symmetric band-matrix shape");
}

double SymmetricBandMatrix1D::value(int row, int column) const
{
    if (row < 0 || column < 0 || row >= size_ || column >= size_)
        throw std::out_of_range("Band-matrix index is outside the matrix");
    if (row < column)
        std::swap(row, column);
    if (row - column > half_bandwidth_)
        return 0.0;
    return entries_[static_cast<std::size_t>(row) *
        (half_bandwidth_ + 1) + row - column];
}

double &SymmetricBandMatrix1D::lower(const int row, const int column)
{
    if (row < column || row - column > half_bandwidth_ || column < 0 ||
        row >= size_)
        throw std::out_of_range("Invalid lower band-matrix index");
    return entries_[static_cast<std::size_t>(row) *
        (half_bandwidth_ + 1) + row - column];
}

std::size_t SymmetricBandMatrix1D::stored_nonzeros(
    const double tolerance) const
{
    std::size_t count = 0;
    for (int row = 0; row < size_; ++row)
        for (int column = std::max(0, row - half_bandwidth_);
             column <= row; ++column)
            if (std::abs(value(row, column)) > tolerance)
                ++count;
    return count;
}

double SparseMatrix1D::value(const int row, const int column) const
{
    if (row < 0 || row >= size() || column < 0 || column >= size())
        throw std::out_of_range("Sparse one-dimensional index is outside the matrix");
    const auto &entries = rows[row];
    const auto found = std::lower_bound(entries.begin(), entries.end(), column,
        [](const auto &entry, const int target) {
            return entry.first < target;
        });
    return found != entries.end() && found->first == column ?
        found->second : 0.0;
}

std::size_t SparseMatrix1D::nonzeros() const noexcept
{
    std::size_t result = 0;
    for (const auto &row : rows)
        result += row.size();
    return result;
}

TruncatedDirectionResult1D build_truncated_direction_1d(
    const int original_degree, const std::vector<double> &original_knots,
    const int truncation_half_bandwidth, const bool compute_certificate)
{
    if (original_degree <= 1)
        throw std::invalid_argument(
            "truncated_cholesky_tensor requires original degree p >= 2; "
            "p=1 has an untruncatable derivative mass matrix");
    TruncatedDirectionResult1D result;
    result.original_degree = original_degree;
    result.derivative_degree = original_degree - 1;
    result.num_original_basis =
        static_cast<int>(original_knots.size()) - original_degree - 1;
    result.value_mass = build_mass(original_degree, original_knots, true,
        truncation_half_bandwidth, compute_certificate);
    if (original_knots.size() < 3)
        throw std::invalid_argument("Original knot vector cannot be trimmed");
    const std::vector<double> derivative_knots(
        original_knots.begin() + 1, original_knots.end() - 1);
    result.derivative_mass = build_mass(original_degree - 1,
        derivative_knots, false, truncation_half_bandwidth,
        compute_certificate);
    const auto start = Clock::now();
    result.exact_stiffness = difference_mass_stiffness(
        result.derivative_mass.exact_mass, original_degree, original_knots);
    result.approximate_stiffness = difference_mass_stiffness(
        result.derivative_mass.approximate_mass,
        original_degree, original_knots);
    result.reduced_operator_seconds = elapsed_seconds(start);
    const SparseMatrix1D integrated = direct_stiffness(
        original_degree, original_knots);
    result.derivative_identity_relative_error =
        relative_frobenius_difference(integrated, result.exact_stiffness);
    return result;
}

double relative_frobenius_difference(const SparseMatrix1D &first,
    const SparseMatrix1D &second)
{
    if (first.size() != second.size())
        throw std::invalid_argument("Matrix dimensions differ");
    long double difference = 0.0;
    long double reference = 0.0;
    for (int row = 0; row < first.size(); ++row)
    {
        const auto &left = first.rows[row];
        const auto &right = second.rows[row];
        std::size_t i = 0;
        std::size_t j = 0;
        while (i < left.size() || j < right.size())
        {
            long double a = 0.0;
            long double b = 0.0;
            if (j == right.size() ||
                (i < left.size() && left[i].first < right[j].first))
                a = left[i++].second;
            else if (i == left.size() || right[j].first < left[i].first)
                b = right[j++].second;
            else
            {
                a = left[i++].second;
                b = right[j++].second;
            }
            const long double delta = a - b;
            difference += delta * delta;
            reference += a * a;
        }
    }
    return std::sqrt(static_cast<double>(difference /
        std::max(reference, std::numeric_limits<long double>::min())));
}

bool is_symmetric_positive_definite(const SparseMatrix1D &matrix,
    const double symmetry_tolerance)
{
    const int size = matrix.size();
    int bandwidth = 0;
    for (int row = 0; row < size; ++row)
        for (const auto &entry : matrix.rows[row])
        {
            const int column = entry.first;
            if (std::abs(matrix.value(row, column) -
                    matrix.value(column, row)) > symmetry_tolerance)
                return false;
            bandwidth = std::max(bandwidth, std::abs(row - column));
        }
    SymmetricBandMatrix1D band(size, bandwidth);
    for (int row = 0; row < size; ++row)
        for (const auto &entry : matrix.rows[row])
            if (row >= entry.first)
                band.lower(row, entry.first) = entry.second;
    try
    {
        static_cast<void>(band_cholesky(band));
        return true;
    }
    catch (const std::runtime_error &)
    {
        return false;
    }
}

} // namespace lor
