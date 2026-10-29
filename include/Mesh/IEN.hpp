#ifndef LOR_MESH_IEN_HPP
#define LOR_MESH_IEN_HPP

#include <cstddef>
#include <vector>

namespace lor
{

template <int Dim>
class BSplinePatch;

template <int Dim>
class LORMesh;

// Read-only element-to-global-basis map shared by assembled, matrix-free, and
// sum-factorized operators. Storage is element-major:
// IEN[e * num_local_basis() + a].
class IEN
{
public:
    virtual ~IEN() = default;

    // Return the global basis/node index associated with local basis a of
    // element e, matching MixPERIGEE's get_IEN(e, a) convention.
    int get_IEN(std::size_t element, int local_basis) const;

    // Return the number of elements and local basis functions represented by
    // this immutable IEN table.
    std::size_t num_elements() const noexcept { return num_elements_; }
    int num_local_basis() const noexcept { return num_local_basis_; }

protected:
    IEN(std::size_t num_elements, int num_local_basis,
        std::vector<int> values);

private:
    std::size_t num_elements_ = 0;
    int num_local_basis_ = 0;
    std::vector<int> values_;
};

// IEN table for active tensor-product B-spline knot-span elements.
template <int Dim>
class BSplineIEN final : public IEN
{
public:
    explicit BSplineIEN(const BSplinePatch<Dim> &patch);
};

// IEN table for tensor-product Q1 cells of the LOR mesh.
template <int Dim>
class LORIEN final : public IEN
{
public:
    explicit LORIEN(const LORMesh<Dim> &mesh);
};

extern template class BSplineIEN<2>;
extern template class BSplineIEN<3>;
extern template class LORIEN<2>;
extern template class LORIEN<3>;

} // namespace lor

#endif
