#include "Solver/ErrorAnalysis.hpp"

#include "Element/BSplineElement.hpp"

#include <cmath>
#include <vector>

namespace lor
{

template <int Dim>
PetscErrorCode compute_l2_error(const BSplinePatch<Dim> &patch,
    const BSplineIEN<Dim> &IEN_map, const ID &ID_map,
    const QuadratureRule1D &quad,
    const std::vector<BSplineBasis1D> &basis_tables,
    Vec solution, const ScalarExactSolution<Dim> &exact,
    PetscReal *l2_error)
{
    PetscFunctionBeginUser;
    PetscCheck(solution != nullptr && l2_error != nullptr,
        PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
        "Solution and L2-error output must not be null");
    PetscCheck(static_cast<bool>(exact), PETSC_COMM_SELF,
        PETSC_ERR_ARG_NULL, "Exact-solution function must not be empty");

    // Access the reduced coefficient vector once for the whole integration.
    const PetscScalar *reduced_solution = nullptr;
    PetscCall(VecGetArrayRead(solution, &reduced_solution));
    std::vector<typename BSplinePatch<Dim>::Point> control_points(
        IEN_map.num_local_basis());
    std::vector<double> local_solution(IEN_map.num_local_basis(), 0.0);
    double squared_error = 0.0;

    // Follow the same element construction and IEN/ID composition as global
    // assembly, then integrate the value error as in MixPERIGEE post_err.
    for (std::size_t element_index = 0;
         element_index < patch.num_elements(); ++element_index)
    {
        for (int local = 0; local < IEN_map.num_local_basis(); ++local)
        {
            const int global_index =
                IEN_map.get_IEN(element_index, local);
            control_points[local] = patch.control_point(global_index);
            const int ID_value = ID_map.get_ID(global_index);
            local_solution[local] = ID_value < 0 ? 0.0 :
                PetscRealPart(reduced_solution[ID_value]);
        }

        const BSplineElement<Dim> element(patch, element_index,
            control_points, quad, basis_tables);
        for (int q = 0; q < element.num_quad_pts(); ++q)
        {
            double numerical = 0.0;
            for (int local = 0; local < element.num_local_basis(); ++local)
                numerical += local_solution[local] * element.N(q, local);
            const double difference = numerical - exact(element.phy_coor(q));
            squared_error += difference * difference * element.det_Jac(q) *
                element.quad_weight(q);
        }
    }

    PetscCall(VecRestoreArrayRead(solution, &reduced_solution));
    *l2_error = std::sqrt(squared_error);
    PetscFunctionReturn(PETSC_SUCCESS);
}

template PetscErrorCode compute_l2_error<2>(
    const BSplinePatch<2> &, const BSplineIEN<2> &, const ID &,
    const QuadratureRule1D &, const std::vector<BSplineBasis1D> &,
    Vec, const ScalarExactSolution<2> &, PetscReal *);
template PetscErrorCode compute_l2_error<3>(
    const BSplinePatch<3> &, const BSplineIEN<3> &, const ID &,
    const QuadratureRule1D &, const std::vector<BSplineBasis1D> &,
    Vec, const ScalarExactSolution<3> &, PetscReal *);

} // namespace lor
