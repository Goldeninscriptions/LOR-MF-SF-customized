#ifndef LOR_ELEMENT_GAUSS_QUADRATURE_HPP
#define LOR_ELEMENT_GAUSS_QUADRATURE_HPP

#include <vector>

namespace lor
{

struct QuadratureRule1D
{
    std::vector<double> points;
    std::vector<double> weights;
};

QuadratureRule1D gauss_legendre_rule(int order);

} // namespace lor

#endif

