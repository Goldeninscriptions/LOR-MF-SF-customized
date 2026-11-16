#include "Assembly/BandedMassCholesky1D.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{

std::vector<double> open_uniform_knots(const int degree, const int elements)
{
    std::vector<double> knots(degree + 1, 0.0);
    for (int span = 1; span < elements; ++span)
        knots.push_back(static_cast<double>(span) / elements);
    knots.insert(knots.end(), degree + 1, 1.0);
    return knots;
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
        run_case(3, 16, 1);
        run_case(4, 16, 2);
        run_case(8, 16, 4);
    }
    catch (const std::exception &exception)
    {
        std::cerr << "truncated_cholesky_1d_test: "
                  << exception.what() << '\n';
        return 1;
    }
    return 0;
}
