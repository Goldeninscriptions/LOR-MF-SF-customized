#include "Solver/ErrorAnalysis.hpp"

#include "Element/BSplineElement.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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
    PetscCheck(solution != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
        "Solution must not be null");
    const MPI_Comm comm = PetscObjectComm(
        reinterpret_cast<PetscObject>(solution));
    PetscCheck(l2_error != nullptr,
        comm, PETSC_ERR_ARG_NULL,
        "Solution and L2-error output must not be null");
    PetscCheck(static_cast<bool>(exact), comm,
        PETSC_ERR_ARG_NULL, "Exact-solution function must not be empty");

    // Find the smallest contiguous algebraic interval needed by this rank's
    // element support. With full-boundary tensor ID numbering, a slow-direction
    // slab touches its owned rows plus at most p neighboring layers; constrained
    // basis functions carry ID=-1 and do not enlarge this halo.
    PetscInt halo_begin = std::numeric_limits<PetscInt>::max();
    PetscInt halo_end = 0;
    for (std::size_t element_index = IEN_map.local_element_begin();
         element_index < IEN_map.local_element_end(); ++element_index)
        for (int local = 0; local < IEN_map.num_local_basis(); ++local)
        {
            const PetscInt ID_value = ID_map.get_ID(
                IEN_map.get_IEN(element_index, local));
            if (ID_value >= 0)
            {
                halo_begin = std::min(halo_begin, ID_value);
                halo_end = std::max(halo_end, ID_value + 1);
            }
        }
    if (halo_begin == std::numeric_limits<PetscInt>::max())
        halo_begin = halo_end = 0;
    const PetscInt halo_size = halo_end - halo_begin;

    // Scatter only the required owned-and-ghost coefficient interval to a
    // rank-local read-only vector. Public slab-interface DOFs therefore have
    // one algebraic owner but are available on every element rank that needs
    // them, without replicating the complete global solution.
    VecScatter solution_scatter = nullptr;
    Vec halo_solution = nullptr;
    IS source_indices = nullptr;
    IS halo_indices = nullptr;
    PetscCall(ISCreateStride(PETSC_COMM_SELF, halo_size,
        halo_begin, 1, &source_indices));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, halo_size,
        0, 1, &halo_indices));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, halo_size, &halo_solution));
    PetscCall(VecScatterCreate(solution, source_indices,
        halo_solution, halo_indices, &solution_scatter));
    PetscCall(VecScatterBegin(solution_scatter, solution, halo_solution,
        INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(solution_scatter, solution, halo_solution,
        INSERT_VALUES, SCATTER_FORWARD));

    // Access the local halo coefficient vector once for local integration.
    const PetscScalar *reduced_solution = nullptr;
    PetscCall(VecGetArrayRead(halo_solution, &reduced_solution));
    std::vector<typename BSplinePatch<Dim>::Point> control_points(
        IEN_map.num_local_basis());
    std::vector<double> local_solution(IEN_map.num_local_basis(), 0.0);
    PetscReal local_squared_error = 0.0;

    // Follow the same element construction and IEN/ID composition as global
    // assembly, then integrate the value error as in MixPERIGEE post_err.
    for (std::size_t element_index = IEN_map.local_element_begin();
         element_index < IEN_map.local_element_end(); ++element_index)
    {
        for (int local = 0; local < IEN_map.num_local_basis(); ++local)
        {
            const int global_index =
                IEN_map.get_IEN(element_index, local);
            control_points[local] = patch.control_point(global_index);
            const int ID_value = ID_map.get_ID(global_index);
            local_solution[local] = ID_value < 0 ? 0.0 :
                PetscRealPart(reduced_solution[ID_value - halo_begin]);
        }

        const BSplineElement<Dim> element(patch, element_index,
            control_points, quad, basis_tables);
        for (int q = 0; q < element.num_quad_pts(); ++q)
        {
            double numerical = 0.0;
            for (int local = 0; local < element.num_local_basis(); ++local)
                numerical += local_solution[local] * element.N(q, local);
            const double difference = numerical - exact(element.phy_coor(q));
            local_squared_error += difference * difference * element.det_Jac(q) *
                element.quad_weight(q);
        }
    }

    PetscCall(VecRestoreArrayRead(halo_solution, &reduced_solution));
    PetscCall(VecScatterDestroy(&solution_scatter));
    PetscCall(VecDestroy(&halo_solution));
    PetscCall(ISDestroy(&halo_indices));
    PetscCall(ISDestroy(&source_indices));

    // Sum the disjoint element integrals and take the square root only after
    // the global reduction, making the result independent of MPI rank count.
    PetscReal global_squared_error = 0.0;
    PetscCallMPI(MPI_Allreduce(&local_squared_error,
        &global_squared_error, 1, MPIU_REAL, MPI_SUM, comm));
    *l2_error = std::sqrt(global_squared_error);
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
