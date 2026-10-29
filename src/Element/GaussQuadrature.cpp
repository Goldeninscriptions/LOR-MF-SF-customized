#include "Element/GaussQuadrature.hpp"

#include <cmath>
#include <stdexcept>

namespace lor
{

QuadratureRule1D gauss_legendre_rule(const int order)
{
    if (order < 1)
        throw std::invalid_argument("Quadrature order must be positive");

    QuadratureRule1D rule;
    rule.points.resize(order);
    rule.weights.resize(order);
    constexpr double pi = 3.141592653589793238462643383279502884;
    const int half = (order + 1) / 2;

    for (int i = 0; i < half; ++i)
    {
        double root = std::cos(pi * (i + 0.75) / (order + 0.5));
        double previous = 0.0;
        double derivative = 0.0;
        do
        {
            previous = root;
            double p0 = 1.0;
            double p1 = root;
            for (int degree = 2; degree <= order; ++degree)
            {
                const double value =
                    ((2.0 * degree - 1.0) * root * p1 -
                     (degree - 1.0) * p0) / degree;
                p0 = p1;
                p1 = value;
            }
            const double polynomial = (order == 1) ? root : p1;
            const double predecessor = (order == 1) ? 1.0 : p0;
            derivative = order * (root * polynomial - predecessor) /
                (root * root - 1.0);
            root = previous - polynomial / derivative;
        }
        while (std::abs(root - previous) > 1.0e-15);

        const double weight = 2.0 /
            ((1.0 - root * root) * derivative * derivative);
        rule.points[i] = -root;
        rule.points[order - 1 - i] = root;
        rule.weights[i] = weight;
        rule.weights[order - 1 - i] = weight;
    }
    return rule;
}

} // namespace lor

