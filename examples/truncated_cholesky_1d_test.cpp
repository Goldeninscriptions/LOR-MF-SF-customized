#include "Assembly/BandedMassCholesky1D.hpp"

#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace
{

std::vector<double> dense(const lor::SparseMatrix1D &matrix)
{
    const int n = matrix.size();
    std::vector<double> result(static_cast<std::size_t>(n) * n, 0.0);
    for (int row = 0; row < n; ++row)
        for (const auto &entry : matrix.rows[row])
            result[static_cast<std::size_t>(row) * n + entry.first] =
                entry.second;
    return result;
}

std::pair<double, double> generalized_eigenvalue_extremes(
    const lor::SparseMatrix1D &exact,
    const lor::SparseMatrix1D &approximate)
{
    const int n = exact.size();
    auto factor = dense(approximate);
    for (int row = 0; row < n; ++row)
        for (int column = 0; column <= row; ++column)
        {
            double value = factor[static_cast<std::size_t>(row) * n + column];
            for (int inner = 0; inner < column; ++inner)
                value -= factor[static_cast<std::size_t>(row) * n + inner] *
                    factor[static_cast<std::size_t>(column) * n + inner];
            if (row == column)
                factor[static_cast<std::size_t>(row) * n + column] =
                    std::sqrt(value);
            else
                factor[static_cast<std::size_t>(row) * n + column] = value /
                    factor[static_cast<std::size_t>(column) * n + column];
        }
    std::vector<double> inverse(static_cast<std::size_t>(n) * n, 0.0);
    for (int column = 0; column < n; ++column)
        for (int row = 0; row < n; ++row)
        {
            double value = row == column ? 1.0 : 0.0;
            for (int inner = 0; inner < row; ++inner)
                value -= factor[static_cast<std::size_t>(row) * n + inner] *
                    inverse[static_cast<std::size_t>(inner) * n + column];
            inverse[static_cast<std::size_t>(row) * n + column] = value /
                factor[static_cast<std::size_t>(row) * n + row];
        }
    const auto a = dense(exact);
    std::vector<double> work(static_cast<std::size_t>(n) * n, 0.0);
    std::vector<double> transformed(static_cast<std::size_t>(n) * n, 0.0);
    for (int row = 0; row < n; ++row)
        for (int column = 0; column < n; ++column)
            for (int inner = 0; inner < n; ++inner)
                work[static_cast<std::size_t>(row) * n + column] +=
                    inverse[static_cast<std::size_t>(row) * n + inner] *
                    a[static_cast<std::size_t>(inner) * n + column];
    for (int row = 0; row < n; ++row)
        for (int column = 0; column < n; ++column)
            for (int inner = 0; inner < n; ++inner)
                transformed[static_cast<std::size_t>(row) * n + column] +=
                    work[static_cast<std::size_t>(row) * n + inner] *
                    inverse[static_cast<std::size_t>(column) * n + inner];
    for (int sweep = 0; sweep < 60 * n * n; ++sweep)
    {
        int p = 0;
        int q = 1;
        double maximum = 0.0;
        for (int row = 0; row < n; ++row)
            for (int column = row + 1; column < n; ++column)
                if (std::abs(transformed[static_cast<std::size_t>(row) * n + column]) > maximum)
                {
                    maximum = std::abs(transformed[static_cast<std::size_t>(row) * n + column]);
                    p = row;
                    q = column;
                }
        if (maximum < 1.0e-12)
            break;
        const double app = transformed[static_cast<std::size_t>(p) * n + p];
        const double aqq = transformed[static_cast<std::size_t>(q) * n + q];
        const double apq = transformed[static_cast<std::size_t>(p) * n + q];
        const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        for (int index = 0; index < n; ++index)
        {
            const double ip = transformed[static_cast<std::size_t>(index) * n + p];
            const double iq = transformed[static_cast<std::size_t>(index) * n + q];
            transformed[static_cast<std::size_t>(index) * n + p] = c * ip - s * iq;
            transformed[static_cast<std::size_t>(index) * n + q] = s * ip + c * iq;
        }
        for (int index = 0; index < n; ++index)
        {
            const double pi = transformed[static_cast<std::size_t>(p) * n + index];
            const double qi = transformed[static_cast<std::size_t>(q) * n + index];
            transformed[static_cast<std::size_t>(p) * n + index] = c * pi - s * qi;
            transformed[static_cast<std::size_t>(q) * n + index] = s * pi + c * qi;
        }
    }
    double minimum = std::numeric_limits<double>::max();
    double maximum = 0.0;
    for (int index = 0; index < n; ++index)
    {
        const double value = transformed[static_cast<std::size_t>(index) * n + index];
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    return {minimum, maximum};
}

void check_certificate(const lor::TruncatedMassResult1D &mass)
{
    if (!mass.certificate.computed || !mass.certificate.beta_below_one)
        return;
    const auto extremes = generalized_eigenvalue_extremes(
        mass.exact_mass, mass.approximate_mass);
    const double beta = mass.certificate.beta;
    const double bound = std::pow((1.0 + beta) / (1.0 - beta), 2.0);
    if (extremes.second / extremes.first > bound * (1.0 + 1.0e-8))
        throw std::runtime_error(
            "Certificate bound failed independent dense spectral check");
}

std::vector<double> open_uniform_knots(const int degree, const int elements)
{
    std::vector<double> knots(degree + 1, 0.0);
    for (int span = 1; span < elements; ++span)
        knots.push_back(static_cast<double>(span) / elements);
    knots.insert(knots.end(), degree + 1, 1.0);
    return knots;
}

// Test-only reproduction of the former quadratic free-row/free-column loop.
// It provides a numerical oracle and a directly comparable n=1024 timing;
// production code must not call this implementation.
lor::SparseMatrix1D legacy_difference_mass_stiffness(
    const lor::SparseMatrix1D &mass, const int degree,
    const std::vector<double> &knots)
{
    const int basis = static_cast<int>(knots.size()) - degree - 1;
    const int free_size = basis - 2;
    std::vector<double> alpha(basis - 1, 0.0);
    for (int row = 0; row < basis - 1; ++row)
    {
        const double denominator = knots[row + degree + 1] - knots[row + 1];
        if (denominator > 0.0)
            alpha[row] = degree / denominator;
    }
    lor::SparseMatrix1D result;
    result.rows.resize(free_size);
    for (int row = 0; row < free_size; ++row)
        for (int column = 0; column < free_size; ++column)
        {
            double entry = 0.0;
            const int derivative_rows[2] = {row, row + 1};
            const double derivative_row_values[2] =
                {alpha[row], -alpha[row + 1]};
            const int derivative_columns[2] = {column, column + 1};
            const double derivative_column_values[2] =
                {alpha[column], -alpha[column + 1]};
            for (int left = 0; left < 2; ++left)
                for (int right = 0; right < 2; ++right)
                    entry += derivative_row_values[left] *
                        mass.value(derivative_rows[left],
                            derivative_columns[right]) *
                        derivative_column_values[right];
            if (entry != 0.0)
                result.rows[row].emplace_back(column, entry);
        }
    return result;
}

void run_case(const int degree, const int elements, const int bandwidth)
{
    const auto result = lor::build_truncated_direction_1d(degree,
        open_uniform_knots(degree, elements), bandwidth, true);
    if (result.derivative_identity_relative_error > 5.0e-12)
        throw std::runtime_error("D^T M^(p-1) D identity failed");
    if (result.value_mass.full_reconstruction_relative_error > 5.0e-12 ||
        result.derivative_mass.full_reconstruction_relative_error > 5.0e-12)
        throw std::runtime_error("Complete band Cholesky reconstruction failed");
    if (!lor::is_symmetric_positive_definite(
            result.value_mass.approximate_mass) ||
        !lor::is_symmetric_positive_definite(
            result.derivative_mass.approximate_mass) ||
        !lor::is_symmetric_positive_definite(
            result.approximate_stiffness))
        throw std::runtime_error("Truncated one-dimensional operator is not SPD");
    check_certificate(result.value_mass);
    check_certificate(result.derivative_mass);
    std::cout << "p=" << degree << " n=" << elements
              << " k=" << bandwidth
              << " identity=" << result.derivative_identity_relative_error
              << " dropped="
              << result.value_mass.dropped_factor_entries +
                    result.derivative_mass.dropped_factor_entries
              << " beta_value=" << result.value_mass.certificate.beta
              << " beta_derivative="
              << result.derivative_mass.certificate.beta << '\n';
}

} // namespace

int main()
{
    try
    {
        run_case(3, 16, 0);
        run_case(3, 16, 1);
        run_case(4, 16, (4 + 1) / 2); // automatic default
        run_case(8, 16, 4);
        run_case(8, 16, 6); // nearest valid derivative-mass bandwidth
        bool invalid_rejected = false;
        try
        {
            (void)lor::build_truncated_direction_1d(3,
                open_uniform_knots(3, 16), 2, true);
        }
        catch (const std::invalid_argument &)
        {
            invalid_rejected = true;
        }
        if (!invalid_rejected)
            throw std::runtime_error("Invalid k >= p-1 was not rejected");
        const auto no_certificate = lor::build_truncated_direction_1d(4,
            open_uniform_knots(4, 16), 2, false);
        if (no_certificate.value_mass.certificate.computed ||
            no_certificate.derivative_mass.certificate.computed)
            throw std::runtime_error("Disabled certificate was computed");
        const auto start = std::chrono::steady_clock::now();
        const auto large = lor::build_truncated_direction_1d(8,
            open_uniform_knots(8, 1024), 4, true);
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        if (large.approximate_stiffness.nonzeros() >
            static_cast<std::size_t>(100 * large.approximate_stiffness.size()))
            throw std::runtime_error("Large 1D case lost sparse scaling");
        const auto legacy_start = std::chrono::steady_clock::now();
        const auto legacy_exact = legacy_difference_mass_stiffness(
            large.derivative_mass.exact_mass, 8,
            open_uniform_knots(8, 1024));
        const auto legacy_approximate = legacy_difference_mass_stiffness(
            large.derivative_mass.approximate_mass, 8,
            open_uniform_knots(8, 1024));
        const double legacy_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - legacy_start).count();
        if (lor::relative_frobenius_difference(
                legacy_exact, large.exact_stiffness) > 1.0e-14 ||
            lor::relative_frobenius_difference(
                legacy_approximate, large.approximate_stiffness) > 1.0e-14)
            throw std::runtime_error(
                "Sparse 1D stiffness differs from quadratic reference");
        std::cout << "large_1d p=8 n=1024 k=4 seconds=" << seconds
                  << " sparse_reduced_seconds="
                  << large.reduced_operator_seconds
                  << " quadratic_reference_seconds=" << legacy_seconds
                  << " nnz=" << large.approximate_stiffness.nonzeros()
                  << " dropped_value="
                  << large.value_mass.dropped_factor_entries
                  << " dropped_derivative="
                  << large.derivative_mass.dropped_factor_entries
                  << " beta_value=" << large.value_mass.certificate.beta
                  << " beta_derivative="
                  << large.derivative_mass.certificate.beta
                  << '\n';
    }
    catch (const std::exception &exception)
    {
        std::cerr << "truncated_cholesky_1d_test: "
                  << exception.what() << '\n';
        return 1;
    }
    return 0;
}
