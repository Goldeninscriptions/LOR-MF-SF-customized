#include "Assembly/GrevilleMassPreconditioner.hpp"

#include "Element/BSplineBasis1D.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace lor
{
namespace
{

SparseMatrix1D transpose(const SparseMatrix1D &matrix)
{
    SparseMatrix1D result;
    result.rows.resize(matrix.size());
    for (int row = 0; row < matrix.size(); ++row)
        for (const auto &entry : matrix.rows[row])
        {
            if (entry.first < 0 || entry.first >= matrix.size())
                throw std::invalid_argument(
                    "Sparse matrix column lies outside its square shape");
            result.rows[entry.first].emplace_back(row, entry.second);
        }
    return result;
}

SparseMatrix1D multiply(const SparseMatrix1D &left,
    const SparseMatrix1D &right)
{
    if (left.size() != right.size())
        throw std::invalid_argument(
            "Sparse matrix product requires equal square shapes");
    SparseMatrix1D result;
    result.rows.resize(left.size());
    for (int row = 0; row < left.size(); ++row)
    {
        std::map<int, double> accumulated;
        for (const auto &left_entry : left.rows[row])
        {
            if (left_entry.first < 0 || left_entry.first >= right.size())
                throw std::invalid_argument(
                    "Sparse matrix product encountered an invalid column");
            for (const auto &right_entry : right.rows[left_entry.first])
                accumulated[right_entry.first] +=
                    left_entry.second * right_entry.second;
        }
        for (const auto &entry : accumulated)
            if (entry.second != 0.0)
                result.rows[row].push_back(entry);
    }
    return result;
}

std::vector<double> greville_abscissae(const int degree,
    const std::vector<double> &knots)
{
    if (degree <= 0)
        throw std::invalid_argument(
            "Greville mass preconditioner requires positive degree");
    const int size = static_cast<int>(knots.size()) - degree - 1;
    std::vector<double> points(size, 0.0);
    for (int basis = 0; basis < size; ++basis)
        for (int offset = 1; offset <= degree; ++offset)
            points[basis] += knots[basis + offset] / degree;
    if (!std::is_sorted(points.begin(), points.end()))
        throw std::invalid_argument(
            "Greville points must form a nondecreasing grid");
    for (int index = 0; index + 1 < size; ++index)
        if (!(points[index + 1] > points[index]))
            throw std::invalid_argument(
                "P1 Greville mass requires distinct adjacent points");
    return points;
}

SparseMatrix1D assemble_greville_p1_mass(
    const std::vector<double> &points)
{
    SparseMatrix1D result;
    result.rows.resize(points.size());
    std::vector<std::map<int, double>> rows(points.size());
    for (int cell = 0; cell + 1 < static_cast<int>(points.size()); ++cell)
    {
        const double length = points[cell + 1] - points[cell];
        rows[cell][cell] += length / 3.0;
        rows[cell][cell + 1] += length / 6.0;
        rows[cell + 1][cell] += length / 6.0;
        rows[cell + 1][cell + 1] += length / 3.0;
    }
    for (int row = 0; row < static_cast<int>(points.size()); ++row)
        result.rows[row].assign(rows[row].begin(), rows[row].end());
    return result;
}

SparseMatrix1D assemble_greville_p1_stiffness(
    const std::vector<double> &points)
{
    if (points.size() < 2)
        throw std::invalid_argument(
            "Greville P1 stiffness requires at least two points");
    std::vector<std::map<int, double>> accumulated(points.size());
    for (int element = 0;
         element + 1 < static_cast<int>(points.size()); ++element)
    {
        const double h = points[element + 1] - points[element];
        if (!(h > 0.0))
            throw std::invalid_argument(
                "Greville points must be strictly increasing");
        const double inverse_h = 1.0 / h;
        accumulated[element][element] += inverse_h;
        accumulated[element][element + 1] -= inverse_h;
        accumulated[element + 1][element] -= inverse_h;
        accumulated[element + 1][element + 1] += inverse_h;
    }
    SparseMatrix1D result;
    result.rows.resize(points.size());
    for (int row = 0; row < static_cast<int>(points.size()); ++row)
        for (const auto &entry : accumulated[row])
            if (entry.second != 0.0)
                result.rows[row].push_back(entry);
    return result;
}

SparseMatrix1D assemble_collocation(const int degree,
    const std::vector<double> &knots,
    const std::vector<double> &points)
{
    const BSplineBasis1D basis(degree, knots);
    SparseMatrix1D result;
    result.rows.resize(points.size());
    for (int row = 0; row < static_cast<int>(points.size()); ++row)
    {
        const auto evaluation = basis.evaluate_N(points[row]);
        for (int local = 0; local <= degree; ++local)
        {
            const double value = evaluation.N[local];
            if (value != 0.0)
                result.rows[row].emplace_back(
                    evaluation.first_basis + local, value);
        }
    }
    return result;
}

SparseMatrix1D remove_endpoints(const SparseMatrix1D &matrix)
{
    if (matrix.size() <= 2)
        throw std::invalid_argument(
            "Endpoint restriction leaves no interior degrees of freedom");
    SparseMatrix1D result;
    result.rows.resize(matrix.size() - 2);
    for (int row = 1; row + 1 < matrix.size(); ++row)
        for (const auto &entry : matrix.rows[row])
            if (entry.first > 0 && entry.first + 1 < matrix.size())
                result.rows[row - 1].emplace_back(
                    entry.first - 1, entry.second);
    return result;
}

} // namespace

std::vector<double> open_uniform_knots_1d(
    const int degree, const int elements)
{
    if (degree <= 0 || elements <= 0)
        throw std::invalid_argument(
            "Degree and element count must be positive");
    std::vector<double> knots(degree + 1, 0.0);
    for (int span = 1; span < elements; ++span)
        knots.push_back(static_cast<double>(span) / elements);
    knots.insert(knots.end(), degree + 1, 1.0);
    return knots;
}

GrevilleMassPreconditioner1D build_greville_mass_preconditioner_1d(
    const int degree, const std::vector<double> &knots)
{
    GrevilleMassPreconditioner1D result;
    result.degree = degree;
    result.knots = knots;
    result.elements = 0;
    for (int span = degree;
         span + 1 < static_cast<int>(knots.size()) - degree; ++span)
        if (knots[span + 1] > knots[span])
            ++result.elements;
    result.greville_points = greville_abscissae(degree, knots);
    result.spline_mass = assemble_bspline_mass_1d(degree, knots, false);
    result.greville_mass =
        assemble_greville_p1_mass(result.greville_points);
    result.collocation = assemble_collocation(
        degree, knots, result.greville_points);
    result.transformed_mass = multiply(transpose(result.collocation),
        multiply(result.greville_mass, result.collocation));
    return result;
}

GrevilleMassPreconditioner1D restrict_greville_mass_preconditioner_endpoints(
    const GrevilleMassPreconditioner1D &full)
{
    if (full.greville_points.size() <= 2)
        throw std::invalid_argument(
            "Greville endpoint restriction requires interior points");
    GrevilleMassPreconditioner1D result;
    result.degree = full.degree;
    result.elements = full.elements;
    result.knots = full.knots;
    result.greville_points.assign(full.greville_points.begin() + 1,
        full.greville_points.end() - 1);
    result.spline_mass = remove_endpoints(full.spline_mass);
    result.greville_mass = remove_endpoints(full.greville_mass);
    result.collocation = remove_endpoints(full.collocation);
    result.transformed_mass = remove_endpoints(full.transformed_mass);
    result.spline_stiffness = assemble_bspline_stiffness_1d(
        full.degree, full.knots);
    const auto full_greville_stiffness =
        assemble_greville_p1_stiffness(full.greville_points);
    result.greville_stiffness = remove_endpoints(full_greville_stiffness);
    result.transformed_stiffness = multiply(transpose(result.collocation),
        multiply(result.greville_stiffness, result.collocation));
    return result;
}

SparseMatrix1D sparse_kronecker_product(
    const SparseMatrix1D &slow_factor,
    const SparseMatrix1D &fast_factor)
{
    if (slow_factor.size() <= 0 || fast_factor.size() <= 0)
        throw std::invalid_argument(
            "Kronecker factors must be nonempty");
    const int fast_size = fast_factor.size();
    SparseMatrix1D result;
    result.rows.resize(slow_factor.size() * fast_size);
    for (int slow_row = 0; slow_row < slow_factor.size(); ++slow_row)
        for (int fast_row = 0; fast_row < fast_size; ++fast_row)
        {
            const int row = slow_row * fast_size + fast_row;
            for (const auto &slow_entry : slow_factor.rows[slow_row])
                for (const auto &fast_entry : fast_factor.rows[fast_row])
                    result.rows[row].emplace_back(
                        slow_entry.first * fast_size + fast_entry.first,
                        slow_entry.second * fast_entry.second);
        }
    return result;
}

} // namespace lor
