#ifndef LOR_MESH_LOR_POINT_SET_HPP
#define LOR_MESH_LOR_POINT_SET_HPP

#include "Mesh/BSplinePatch.hpp"

#include <array>
#include <vector>

namespace lor
{

// Numerical controls for the Greville-initialized Demko Remez iteration.
struct DemkoOptions
{
    // Maximum number of outer Remez point-set updates.
    int max_iterations = 50;

    // Minimum samples used on each nonzero knot span to bracket extrema.
    int samples_per_span = 16;

    // Golden-section iterations used to refine each bracketed extremum.
    int extremum_iterations = 80;

    // Allowed relative spread among newly extracted extremum amplitudes.
    double equioscillation_tolerance = 1.0e-9;

    // Relative auxiliary tolerance for point-position changes.
    double point_tolerance = 1.0e-11;

    // Absolute pivot threshold for the B-spline collocation solve.
    double pivot_tolerance = 1.0e-13;
};

// Immutable directional parameter-point set used to construct an LOR mesh.
// The base type also represents an explicitly supplied point set; derived
// classes define named generation strategies such as Greville points.
template <int Dim>
class LORPointSet
{
    static_assert(Dim == 2 || Dim == 3, "LORPointSet supports only 2D and 3D");

public:
    using Index = std::array<int, Dim>;
    using Values = std::array<std::vector<double>, Dim>;

    // Store and validate one strictly increasing xi sequence per direction.
    LORPointSet(const BSplinePatch<Dim> &patch, Values values);
    virtual ~LORPointSet() = default;

    // Return the parameter points in one direction or all directions.
    const std::vector<double> &points(int direction) const;
    const Values &values() const noexcept { return values_; }

    // Return the tensor-grid node count in every parameter direction.
    Index shape() const noexcept { return shape_; }

private:
    Values values_;
    Index shape_{};
};

// Greville abscissae associated one-to-one with the B-spline coefficients.
template <int Dim>
class GrevillePointSet final : public LORPointSet<Dim>
{
public:
    // Generate and validate the directional Greville abscissae of patch.
    explicit GrevillePointSet(const BSplinePatch<Dim> &patch);
};

// Chebyshev--Demko abscissae computed independently in every parameter
// direction by a Greville-initialized Remez iteration.
template <int Dim>
class DemkoPointSet final : public LORPointSet<Dim>
{
public:
    // Compute one Demko point per directional B-spline coefficient. Throws a
    // diagnostic exception if interpolation or equioscillation does not
    // converge under the supplied numerical controls.
    explicit DemkoPointSet(const BSplinePatch<Dim> &patch,
        const DemkoOptions &options = {});
};

extern template class LORPointSet<2>;
extern template class LORPointSet<3>;
extern template class GrevillePointSet<2>;
extern template class GrevillePointSet<3>;
extern template class DemkoPointSet<2>;
extern template class DemkoPointSet<3>;

} // namespace lor

#endif
