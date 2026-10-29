#ifndef LOR_MESH_BSPLINE_PATCH_HPP
#define LOR_MESH_BSPLINE_PATCH_HPP

#include <array>
#include <cstddef>
#include <vector>

namespace lor
{

template <int Dim>
class BSplinePatch
{
    static_assert(Dim == 2 || Dim == 3, "BSplinePatch supports only 2D and 3D");

public:
    using Point = std::array<double, Dim>;
    using Index = std::array<int, Dim>;
    using Boundary = std::vector<Index>;
    using Boundaries = std::array<Boundary, 2 * Dim>;

    BSplinePatch(const Index &degrees,
        const std::array<std::vector<double>, Dim> &knots,
        const std::vector<Point> &control_points);

    int degree(int direction) const;
    const std::vector<double> &knots(int direction) const;
    int num_basis(int direction) const;
    Index basis_shape() const noexcept { return basis_shape_; }
    std::size_t num_basis() const noexcept { return control_points_.size(); }

    int num_elements(int direction) const;
    std::size_t num_elements() const noexcept;
    const std::vector<int> &active_spans(int direction) const;
    Index element_spans(std::size_t element) const;

    // Return one independently stored patch boundary as tensor-product basis
    // multi-indices. Boundary IDs are 2*d for the lower side and 2*d+1 for
    // the upper side of parameter direction d.
    const Boundary &boundary(int boundary_id) const;

    // Return all four 2D edges or all six 3D faces. Corner/edge indices appear
    // in every boundary to which they geometrically belong.
    const Boundaries &boundaries() const noexcept { return boundaries_; }

    const Point &control_point(std::size_t index) const;
    Point evaluate_geometry(const std::array<double, Dim> &para) const;
    std::vector<Point> evaluate_geometry_grid(
        const std::array<std::vector<double>, Dim> &para_coors) const;
    void insert_knot(int direction, double knot);
    void uniform_refine(const Index &elements_per_direction);

private:
    Index degrees_;
    std::array<std::vector<double>, Dim> knots_;
    Index basis_shape_{};
    std::vector<Point> control_points_;
    std::array<std::vector<int>, Dim> active_spans_;
    Boundaries boundaries_;

    void validate() const;
    void rebuild_topology();
    void rebuild_boundaries();
};

extern template class BSplinePatch<2>;
extern template class BSplinePatch<3>;

} // namespace lor

#endif

