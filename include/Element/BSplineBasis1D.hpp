#ifndef LOR_ELEMENT_BSPLINE_BASIS_1D_HPP
#define LOR_ELEMENT_BSPLINE_BASIS_1D_HPP

#include <vector>

namespace lor
{

template <int Dim>
class BSplinePatch;

class BSplineBasis1D
{
public:
    // Result of evaluating the p + 1 nonzero basis functions at one arbitrary
    // parameter coordinate xi. The local entry N[i] belongs to the global
    // basis function first_basis + i. The returned object owns N, so the
    // caller may retain or modify it without changing BSplineBasis1D.
    struct PointEvaluation
    {
        int span = -1;
        int first_basis = -1;
        std::vector<double> N;
    };

    // Construct a mathematical one-dimensional B-spline basis without a
    // Gauss table. Use this lightweight form when only evaluate_N(xi) is
    // needed.
    BSplineBasis1D(int degree, const std::vector<double> &knots);

    // Construct the basis and immediately precompute N_Gauss and
    // N_der1_Gauss at the supplied reference Gauss points on every nonempty
    // knot span. The points must use the reference interval [-1,1].
    BSplineBasis1D(int degree, const std::vector<double> &knots,
        const std::vector<double> &gauss_points);

    // Evaluate the active basis functions at an arbitrary parameter xi in the
    // original knot domain. The result contains the knot span, the global
    // index of its first active basis, and the p + 1 values N_i(xi). This
    // performs the recurrence immediately and intentionally does not compute
    // unused derivatives. Repeated element quadrature should use N_Gauss().
    PointEvaluation evaluate_N(double xi) const;

    // Return the precomputed univariate B-spline values at Gauss points
    // without copying:
    //
    //   N_Gauss()[offset(s,q,i)] = B^s_{qi}
    //                              = N_{s-p+i,p}(xi^s_q).
    //
    // The constructor has already initialized this array. Here:
    //   s = actual knot-span index supplied by BSplinePatch::element_spans();
    //   q = index in the constructor's gauss_points argument;
    //   i = local basis index in [0,p];
    //   p = the degree argument supplied to the constructor;
    //   xi^s_q = the qth point mapped from [-1,1] to knot span s.
    //
    // The flattened layout is [s][q][i], with i varying fastest. The caller
    // already knows p and nq = gauss_points.size(), so:
    //
    //   offset(s,q,i) = (s * nq + q) * (p + 1) + i.
    //
    // Storage belonging to zero-length or non-active span indices remains
    // zero. The returned const reference is valid while this basis object is
    // alive. Performance kernels may read contiguous data through
    // N_Gauss().data(), but callers cannot modify, resize, or clear the table.
    const std::vector<double> &N_Gauss() const noexcept { return N_Gauss_; }

    // Return precomputed first derivatives corresponding entry-by-entry to
    // N_Gauss:
    //
    //   N_der1_Gauss()[offset(s,q,i)] = D^s_{qi}
    //                                  = dN_{s-p+i,p}/dxi (xi^s_q).
    //
    // The derivative is with respect to the original knot parameter xi, not
    // the reference coordinate on [-1,1]. N_der1_Gauss() has exactly the same
    // [s][q][i] layout and lifetime rules as N_Gauss(). It also returns a
    // const reference, so no vector or table-data copy is performed.
    //
    // To expose higher derivatives in the future, add private arrays such as
    // N_der2_Gauss_ or N_der3_Gauss_, provide matching const-reference
    // accessors, and fill them inside build_gauss_table() from the private
    // evaluate_derivatives() recurrence. The recurrence itself should remain
    // private so callers cannot bypass table construction.
    const std::vector<double> &N_der1_Gauss() const noexcept
    {
        return N_der1_Gauss_;
    }

private:
    using DerivativeValues = std::vector<std::vector<double>>;

    int degree_ = 0;
    std::vector<double> knots_;
    std::vector<double> N_Gauss_;
    std::vector<double> N_der1_Gauss_;

    static void validate_definition(
        int degree, const std::vector<double> &knots);
    static int num_basis(
        int degree, const std::vector<double> &knots) noexcept;
    static int find_span(
        int degree, const std::vector<double> &knots, double parameter);
    static DerivativeValues evaluate_derivatives(
        int degree, const std::vector<double> &knots,
        double parameter, int span, int maximum_derivative);
    void build_gauss_table(const std::vector<double> &gauss_points);
    static std::size_t table_offset(
        int degree, int number_of_table_points,
        int span, int point, int local_basis) noexcept;

    template <int Dim>
    friend class BSplinePatch;
};

} // namespace lor

#endif

