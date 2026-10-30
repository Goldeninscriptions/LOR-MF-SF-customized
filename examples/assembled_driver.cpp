/*
 * Manufactured solution used by this assembled Laplace driver:
 *
 *   u_exact(x) = product_d sin(pi (x_d-a_d)/(b_d-a_d)),
 *   -Delta u_exact = f(x)
 *                  = [sum_d (pi/(b_d-a_d))^2] u_exact(x).
 *
 * Here [a_d,b_d] is the physical bounding interval in direction d. Therefore
 * u_exact=0 on the complete boundary of an axis-aligned rectangle (2D) or box
 * (3D). The driver verifies this geometry assumption before solving. General
 * curved patches need a different manufactured solution whose trace vanishes
 * on their physical boundary.
 */

#include "Assembly/DirichletBC.hpp"
#include "Assembly/GlobalAssembly.hpp"
#include "Assembly/LexicographicOrdering.hpp"
#include "Element/GaussQuadrature.hpp"
#include "IO/ExperimentIO.hpp"
#include "IO/GeometryInput.hpp"
#include "IO/PatchInput.hpp"
#include "Mesh/ID.hpp"
#include "Mesh/HRefinement.hpp"
#include "Mesh/IEN.hpp"
#include "Mesh/KRefinement.hpp"
#include "Mesh/LORMesh.hpp"
#include "Mesh/LORPointSet.hpp"
#include "Mesh/PRefinement.hpp"
#include "Solver/ConditionNumber.hpp"
#include "Solver/ErrorAnalysis.hpp"
#include "Solver/LinearSolver.hpp"
#include "Solver/ManufacturedSolution.hpp"

#include <slepcsys.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

template <int Dim>
std::array<int, Dim> element_counts(const lor::ExperimentOptions &options)
{
    std::array<int, Dim> counts{};
    for (int direction = 0; direction < Dim; ++direction)
        counts[direction] = options.elements[direction];
    return counts;
}

// Apply exactly one user-selected refinement policy to the minimal geometry
// Patch. Pure p- and h-refinement reject changes belonging to the other policy;
// k-refinement is the only mode that changes degree and mesh together.
template <int Dim>
lor::BSplinePatch<Dim> refine_geometry_patch(
    const lor::BSplinePatch<Dim> &geometry,
    const lor::ExperimentOptions &options)
{
    // Enforce one common input degree before selecting the refinement object.
    const int initial_degree = geometry.degree(0);
    for (int direction = 1; direction < Dim; ++direction)
        if (geometry.degree(direction) != initial_degree)
            throw std::invalid_argument(
                "The paper configuration requires one common degree in all directions");
    const std::array<int, Dim> targets = element_counts<Dim>(options);

    // Pure p-refinement keeps the complete element partition unchanged.
    if (options.refinement_mode == "p_refine")
    {
        for (int direction = 0; direction < Dim; ++direction)
            if (targets[direction] != geometry.num_elements(direction))
                throw std::invalid_argument(
                    "p_refine requires -n to equal the geometry element counts");
        return lor::PRefinement<Dim>(options.degree).apply(geometry);
    }

    // Pure h-refinement keeps the common input polynomial degree unchanged.
    if (options.refinement_mode == "h_refine")
    {
        if (options.degree != initial_degree)
            throw std::invalid_argument(
                "h_refine requires -p to equal the geometry degree");
        return lor::HRefinement<Dim>(targets).apply(geometry);
    }

    // Classical k-refinement elevates first and inserts knots second.
    return lor::KRefinement<Dim>(options.degree, targets).apply(geometry);
}

// Verify that each parametric boundary face has the corresponding constant
// physical coordinate required by the manufactured rectangle/box solution.
template <int Dim>
void verify_axis_aligned_box(const lor::BSplinePatch<Dim> &patch,
    const lor::ManufacturedSolution<Dim> &manufactured)
{
    const lor::LexicographicOrdering<Dim> ordering(patch.basis_shape());
    const auto &lower = manufactured.lower();
    const auto &upper = manufactured.upper();
    for (int direction = 0; direction < Dim; ++direction)
        for (int side = 0; side < 2; ++side)
        {
            const double expected = side == 0 ?
                lower[direction] : upper[direction];
            const double scale = std::max(1.0,
                std::abs(upper[direction] - lower[direction]));
            for (const auto &multi_index :
                patch.boundary(2 * direction + side))
            {
                const double actual = patch.control_point(
                    ordering.global_index(multi_index))[direction];
                if (std::abs(actual - expected) > 1.0e-10 * scale)
                    throw std::invalid_argument(
                        "The manufactured exact solution requires an "
                        "axis-aligned rectangle or box geometry");
            }
        }
}

template <int Dim>
PetscErrorCode run_experiment(const lor::GeometryInput &geometry_input,
    const lor::ExperimentOptions &options)
{
    PetscFunctionBeginUser;

    // Construct the minimal geometry Patch, then form the actual analysis case
    // only through one of the three mathematically nested refinement policies.
    const lor::BSplinePatch<Dim> geometry =
        lor::make_geometry_patch<Dim>(geometry_input);
    const lor::BSplinePatch<Dim> patch =
        refine_geometry_patch(geometry, options);
    const lor::ManufacturedSolution<Dim> manufactured(patch);
    verify_axis_aligned_box(patch, manufactured);

    // Build homogeneous-Dirichlet ID numbering from the patch boundaries.
    const lor::LexicographicOrdering<Dim> ordering(patch.basis_shape());
    const lor::DirichletBC<Dim> dirichlet_BC(
        patch.boundaries(), ordering);
    const lor::ID ID_map(dirichlet_BC.num_global_indices(),
        dirichlet_BC.constrained_global_indices());
    PetscCheck(ID_map.num_free_dofs() > 0, PETSC_COMM_SELF,
        PETSC_ERR_ARG_SIZ,
        "All coefficients are constrained; increase the element counts");

    // Precompute the high-order quadrature basis tables once. The source and
    // exact functions share the manufactured-solution object.
    const lor::QuadratureRule1D assembly_quad =
        lor::gauss_legendre_rule(patch.degree(0) + 1);
    std::vector<lor::BSplineBasis1D> assembly_basis;
    assembly_basis.reserve(Dim);
    for (int direction = 0; direction < Dim; ++direction)
        assembly_basis.emplace_back(patch.degree(direction), patch.knots(direction),
            assembly_quad.points);
    const lor::ScalarSource<Dim> source =
        [&manufactured](const std::array<double, Dim> &coor) {
            return manufactured.source(coor);
        };
    const lor::ScalarExactSolution<Dim> exact =
        [&manufactured](const std::array<double, Dim> &coor) {
            return manufactured.exact(coor);
        };

    // Assemble the high-order Galerkin system.
    const lor::BSplineIEN<Dim> high_IEN(patch);
    lor::GlobalAssemblyResult high_system;
    PetscCall(lor::assemble_global_system<Dim>(patch, high_IEN, ID_map,
        assembly_quad, assembly_basis, source, &high_system));

    // Construct the selected coefficient-aligned LOR points and assemble its
    // Q1 stiffness matrix. Its load is not used to solve the high-order PDE.
    std::unique_ptr<lor::LORPointSet<Dim>> lor_points;
    if (options.lor_points == "greville")
        lor_points = std::make_unique<lor::GrevillePointSet<Dim>>(patch);
    else
        lor_points = std::make_unique<lor::DemkoPointSet<Dim>>(patch);
    const lor::LORMesh<Dim> lor_mesh(patch, *lor_points);
    const lor::LORIEN<Dim> lor_IEN(lor_mesh);
    const lor::QuadratureRule1D lor_quad = lor::gauss_legendre_rule(2);
    lor::GlobalAssemblyResult lor_system;
    PetscCall(lor::assemble_global_system<Dim>(lor_mesh, lor_IEN, ID_map,
        lor_quad, source, &lor_system));

    // Solve the same high-order system first without and then with the LOR
    // preconditioner. Retain both solution vectors for independent L2 errors.
    lor::LinearSolveResult original_solve;
    lor::LinearSolveResult preconditioned_solve;
    PetscCall(lor::solve_unpreconditioned(high_system.stiffness,
        high_system.load, options.lor_pc, options.tolerances,
        &original_solve));
    PetscCall(lor::solve_with_lor(high_system.stiffness,
        lor_system.stiffness, high_system.load, options.lor_pc,
        options.tolerances, &preconditioned_solve));

    // Use a slightly richer rule for the non-polynomial exact-error integral.
    const lor::QuadratureRule1D error_quad =
        lor::gauss_legendre_rule(patch.degree(0) + 3);
    std::vector<lor::BSplineBasis1D> error_basis;
    error_basis.reserve(Dim);
    for (int direction = 0; direction < Dim; ++direction)
        error_basis.emplace_back(patch.degree(direction), patch.knots(direction),
            error_quad.points);
    PetscReal original_error = 0.0;
    PetscReal preconditioned_error = 0.0;
    PetscCall(lor::compute_l2_error<Dim>(patch, high_IEN, ID_map,
        error_quad, error_basis, original_solve.solution, exact,
        &original_error));
    PetscCall(lor::compute_l2_error<Dim>(patch, high_IEN, ID_map,
        error_quad, error_basis, preconditioned_solve.solution, exact,
        &preconditioned_error));

    // Compute true SVD 2-norm condition numbers independently of both KSPs.
    lor::ConditionNumberResult original_condition;
    lor::ConditionNumberResult preconditioned_condition;
    PetscCall(lor::compute_stiffness_condition_numbers(
        high_system.stiffness, lor_system.stiffness,
        options.svd_tolerance, &original_condition,
        &preconditioned_condition));

    // Record actual refined-space metadata and the six numerical diagnostics.
    lor::ExperimentResults results;
    results.refinement_mode = options.refinement_mode;
    results.actual_degree = patch.degree(0);
    for (int direction = 0; direction < Dim; ++direction)
        results.actual_elements.push_back(patch.num_elements(direction));
    results.condition_K = original_condition.condition_number;
    results.iterations_K = original_solve.iterations;
    results.l2_error_K = original_error;
    results.condition_lor_K = preconditioned_condition.condition_number;
    results.iterations_lor_K = preconditioned_solve.iterations;
    results.l2_error_lor_K = preconditioned_error;
    lor::write_experiment_results(options.output_file, results);

    // Provide concise real-time progress information for interactive and
    // batch-script logs.
    PetscCall(PetscPrintf(PETSC_COMM_SELF,
        "Completed refinement=%s, actual p=%d, actual elements=(",
        options.refinement_mode.c_str(), patch.degree(0)));
    for (int direction = 0; direction < Dim; ++direction)
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "%s%d",
            direction == 0 ? "" : ",", patch.num_elements(direction)));
    PetscCall(PetscPrintf(PETSC_COMM_SELF,
        "), mode=%s, point_set=%s\n"
        "  cond_2(K) = %.12e, iterations = %" PetscInt_FMT
        ", L2 error = %.12e\n"
        "  cond_2(K_LOR^{-1}K) = %.12e, iterations = %" PetscInt_FMT
        ", L2 error = %.12e\n"
        "  result file: %s\n",
        lor::lor_pc_mode_name(options.lor_pc), options.lor_points.c_str(),
        static_cast<double>(results.condition_K), results.iterations_K,
        static_cast<double>(results.l2_error_K),
        static_cast<double>(results.condition_lor_K),
        results.iterations_lor_K,
        static_cast<double>(results.l2_error_lor_K),
        options.output_file.c_str()));

    // Destroy all PETSc-owned results after output and error analysis.
    PetscCall(lor::destroy_linear_solve_result(&preconditioned_solve));
    PetscCall(lor::destroy_linear_solve_result(&original_solve));
    PetscCall(lor::destroy_global_system(&lor_system));
    PetscCall(lor::destroy_global_system(&high_system));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

int main(int argc, char **argv)
{
    SlepcInitialize(&argc, &argv, nullptr, nullptr);
    PetscErrorCode error = PETSC_SUCCESS;
    try
    {
        const std::string geometry_file = lor::read_geometry_filename();
        const lor::GeometryInput geometry =
            lor::read_geometry_input(geometry_file);
        const lor::ExperimentOptions options =
            lor::read_experiment_options(geometry.dimension);
        if (geometry.dimension == 2)
            error = run_experiment<2>(geometry, options);
        else if (geometry.dimension == 3)
            error = run_experiment<3>(geometry, options);
        else
            throw std::runtime_error("Only dimensions 2 and 3 are supported");
    }
    catch (const std::exception &exception)
    {
        PetscPrintf(PETSC_COMM_SELF, "Experiment error: %s\n",
            exception.what());
        error = PETSC_ERR_LIB;
    }
    SlepcFinalize();
    return static_cast<int>(error);
}
