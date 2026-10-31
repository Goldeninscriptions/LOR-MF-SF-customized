/*
 * Manufactured solution used by this assembled Laplace experiment:
 *
 *   u_exact(x) = product_d sin(pi (x_d-a_d)/(b_d-a_d)),
 *   -Delta u_exact = f(x)
 *                  = [sum_d (pi/(b_d-a_d))^2] u_exact(x).
 *
 * Here [a_d,b_d] is the physical bounding interval in direction d. The
 * ManufacturedSolution object validates the axis-aligned rectangle/box
 * assumption before its exact and source functions are used.
 */

#include "Solver/AssembledLaplaceExperiment.hpp"

#include "Assembly/DirichletBC.hpp"
#include "Assembly/GlobalAssembly.hpp"
#include "Assembly/LexicographicOrdering.hpp"
#include "Element/GaussQuadrature.hpp"
#include "IO/PatchInput.hpp"
#include "Mesh/DistributedPatch.hpp"
#include "Mesh/ID.hpp"
#include "Mesh/IEN.hpp"
#include "Mesh/LORMesh.hpp"
#include "Mesh/LORPointSet.hpp"
#include "Mesh/Refinement.hpp"
#include "Mesh/TensorSlabPartition.hpp"
#include "Solver/ConditionNumber.hpp"
#include "Solver/ErrorAnalysis.hpp"
#include "Solver/LinearSolver.hpp"
#include "Solver/ManufacturedSolution.hpp"

#include <array>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace lor
{
namespace
{

// Convert a root exception string to the int count required by MPI_Bcast.
int mpi_count(const std::size_t size)
{
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("Experiment status exceeds MPI int count");
    return static_cast<int>(size);
}

// Broadcast a refinement failure before entering the Patch data collectives,
// ensuring every rank exits through the same C++ error path.
void broadcast_root_error(std::string &message, const int root,
    const int rank, const MPI_Comm comm)
{
    int length = rank == root ? mpi_count(message.size()) : 0;
    if (MPI_Bcast(&length, 1, MPI_INT, root, comm) != MPI_SUCCESS)
        throw std::runtime_error("Cannot broadcast refinement status");
    if (rank != root)
        message.resize(static_cast<std::size_t>(length));
    if (MPI_Bcast(message.data(), length, MPI_CHAR, root, comm) != MPI_SUCCESS)
        throw std::runtime_error("Cannot broadcast refinement error");
}

// Create the minimal geometry and execute p/h/k refinement on root only, then
// reconstruct one identical full refined Patch per rank through MPI broadcast.
template <int Dim>
BSplinePatch<Dim> build_distributed_patch(
    const GeometryInput &geometry_input,
    const ExperimentOptions &options, const int root, const MPI_Comm comm)
{
    // Query rank once and reserve root-only Patch storage for the potentially
    // expensive degree-elevation and knot-insertion work.
    int rank = 0;
    if (MPI_Comm_rank(comm, &rank) != MPI_SUCCESS)
        throw std::runtime_error("Cannot query refinement rank");
    std::unique_ptr<BSplinePatch<Dim>> root_patch;
    std::string root_error;

    // Convert IO options to a Mesh request on every rank, but construct and
    // refine control points only on root.
    const RefinementRequest<Dim> request =
        make_refinement_request<Dim>(options);
    if (rank == root)
        try
        {
            const BSplinePatch<Dim> geometry =
                make_geometry_patch<Dim>(geometry_input);
            root_patch = std::make_unique<BSplinePatch<Dim>>(
                apply_refinement<Dim>(geometry, request));
        }
        catch (const std::exception &exception)
        {
            root_error = exception.what();
        }

    // Synchronize root's refinement status before broadcasting valid Patch
    // data; all ranks throw the same diagnostic if construction failed.
    broadcast_root_error(root_error, root, rank, comm);
    if (!root_error.empty())
        throw std::runtime_error(root_error);
    return broadcast_patch<Dim>(root_patch.get(), root, comm);
}

} // namespace

template <int Dim>
PetscErrorCode run_assembled_laplace_experiment(
    const GeometryInput &geometry_input,
    const ExperimentOptions &options, const int root, const MPI_Comm comm)
{
    PetscFunctionBeginUser;

    // Refine once on root, distribute the refined control net, and construct
    // the analytic source/exact solution on the replicated immutable Patch.
    const BSplinePatch<Dim> patch = build_distributed_patch<Dim>(
        geometry_input, options, root, comm);
    const ManufacturedSolution<Dim> manufactured(patch);

    // Build homogeneous-Dirichlet ID numbering from the Patch boundary
    // multi-indices and the one project-wide lexicographic ordering.
    const LexicographicOrdering<Dim> ordering(patch.basis_shape());
    const DirichletBC<Dim> dirichlet_BC(patch.boundaries(), ordering);
    const ID ID_map(dirichlet_BC.num_global_indices(),
        dirichlet_BC.constrained_global_indices());
    PetscCheck(ID_map.num_free_dofs() > 0, comm, PETSC_ERR_ARG_SIZ,
        "All coefficients are constrained; increase the element counts");

    // Precompute one high-order one-dimensional basis table per independent
    // knot vector. The manufactured object supplies both PDE callbacks.
    const QuadratureRule1D assembly_quad =
        gauss_legendre_rule(patch.degree(0) + 1);
    std::vector<BSplineBasis1D> assembly_basis;
    assembly_basis.reserve(Dim);
    for (int direction = 0; direction < Dim; ++direction)
        assembly_basis.emplace_back(patch.degree(direction),
            patch.knots(direction), assembly_quad.points);
    const ScalarSource<Dim> source =
        [&manufactured](const std::array<double, Dim> &coor) {
            return manufactured.source(coor);
        };
    const ScalarExactSolution<Dim> exact =
        [&manufactured](const std::array<double, Dim> &coor) {
            return manufactured.exact(coor);
        };

    // Construct coefficient-aligned LOR points and their sampled physical
    // mesh. This first distributed-Patch stage still replicates LORMesh; local
    // slab+halo storage is deliberately reserved for the next optimization.
    std::unique_ptr<LORPointSet<Dim>> lor_points;
    if (options.lor_points == "greville")
        lor_points = std::make_unique<GrevillePointSet<Dim>>(patch);
    else
        lor_points = std::make_unique<DemkoPointSet<Dim>>(patch);
    const LORMesh<Dim> lor_mesh(patch, *lor_points);

    // Partition high-order elements, LOR cells, and algebraic rows without
    // changing global_index or ID numbering, then bind both on-demand IEN maps.
    const TensorSlabPartition<Dim> partition(
        patch, lor_mesh, ID_map, comm);
    const BSplineIEN<Dim> high_IEN(patch, partition);
    const LORIEN<Dim> lor_IEN(lor_mesh, partition);

    // Print rank-local half-open ranges in rank order for load-balance and
    // public-interface-DOF diagnostics.
    PetscCall(PetscSynchronizedPrintf(comm,
        "[rank %d/%d] ID=[%d,%d), high elements=[%zu,%zu), "
        "LOR cells=[%zu,%zu)\n",
        partition.rank(), partition.size(), partition.owned_ID_begin(),
        partition.owned_ID_end(), partition.high_element_begin(),
        partition.high_element_end(), partition.lor_cell_begin(),
        partition.lor_cell_end()));
    PetscCall(PetscSynchronizedFlush(comm, PETSC_STDOUT));

    // Assemble high-order stiffness/load in one local traversal; PETSc sums
    // shared-interface contributions inserted by different element ranks.
    GlobalMatrixAssemblyResult high_stiffness;
    GlobalVectorAssemblyResult high_load;
    PetscCall(assemble_global_stiffness_load<Dim>(patch, high_IEN,
        ID_map, assembly_quad, assembly_basis, source,
        &high_stiffness, &high_load));

    // Assemble only the Q1 LOR stiffness matrix used by the preconditioner.
    const QuadratureRule1D lor_quad = gauss_legendre_rule(2);
    GlobalMatrixAssemblyResult lor_stiffness;
    PetscCall(assemble_global_stiffness<Dim>(
        lor_mesh, lor_IEN, ID_map, lor_quad, &lor_stiffness));

    // Solve the identical high-order system without and with LOR, retaining
    // both distributed coefficient vectors for independent error integration.
    LinearSolveResult original_solve;
    LinearSolveResult preconditioned_solve;
    PetscCall(solve_unpreconditioned(high_stiffness.matrix,
        high_load.vector, options.lor_pc, options.tolerances,
        &original_solve));
    PetscCall(solve_with_lor(high_stiffness.matrix,
        lor_stiffness.matrix, high_load.vector, options.lor_pc,
        options.tolerances, &preconditioned_solve));

    // Use a richer Gauss rule for the non-polynomial exact-error integrand and
    // reuse its direction-specific B-spline tables for both solutions.
    const QuadratureRule1D error_quad =
        gauss_legendre_rule(patch.degree(0) + 3);
    std::vector<BSplineBasis1D> error_basis;
    error_basis.reserve(Dim);
    for (int direction = 0; direction < Dim; ++direction)
        error_basis.emplace_back(patch.degree(direction),
            patch.knots(direction), error_quad.points);
    PetscReal original_error = 0.0;
    PetscReal preconditioned_error = 0.0;
    PetscCall(compute_l2_error<Dim>(patch, high_IEN, ID_map,
        error_quad, error_basis, original_solve.solution, exact,
        &original_error));
    PetscCall(compute_l2_error<Dim>(patch, high_IEN, ID_map,
        error_quad, error_basis, preconditioned_solve.solution, exact,
        &preconditioned_error));

    // Compute true SVD 2-norm condition numbers independently of both outer
    // KSP convergence histories.
    ConditionNumberResult original_condition;
    ConditionNumberResult preconditioned_condition;
    PetscCall(compute_stiffness_condition_numbers(
        high_stiffness.matrix, lor_stiffness.matrix,
        options.svd_tolerance, options.svd_max_iterations,
        options.svd_subspace_dimension, &original_condition,
        &preconditioned_condition));

    // Collect realized metadata and all six diagnostics; every rank retains
    // identical scalar results while only root attempts filesystem output.
    ExperimentResults results;
    results.refinement_mode = options.refinement_mode;
    results.mpi_ranks = partition.size();
    results.actual_degree = patch.degree(0);
    for (int direction = 0; direction < Dim; ++direction)
        results.actual_elements.push_back(patch.num_elements(direction));
    results.condition_K = original_condition.condition_number;
    results.iterations_K = original_solve.iterations;
    results.l2_error_K = original_error;
    results.condition_lor_K = preconditioned_condition.condition_number;
    results.iterations_lor_K = preconditioned_solve.iterations;
    results.l2_error_lor_K = preconditioned_error;
    std::string output_error;
    if (partition.rank() == root)
        try
        {
            write_experiment_results(options.output_file, results);
        }
        catch (const std::exception &exception)
        {
            output_error = exception.what();
        }

    // Propagate a root filesystem failure but postpone throwing until all
    // ranks have collectively destroyed their PETSc objects.
    broadcast_root_error(output_error, root, partition.rank(), comm);

    // Print one concise run summary only after every rank knows output
    // succeeded; PetscPrintf emits it through the communicator's rank zero.
    if (output_error.empty())
    {
        PetscCall(PetscPrintf(comm,
            "Completed MPI ranks=%d, refinement=%s, actual p=%d, actual elements=(",
            partition.size(), options.refinement_mode.c_str(), patch.degree(0)));
        for (int direction = 0; direction < Dim; ++direction)
            PetscCall(PetscPrintf(comm, "%s%d", direction == 0 ? "" : ",",
                patch.num_elements(direction)));
        PetscCall(PetscPrintf(comm,
            "), mode=%s, point_set=%s\n"
            "  cond_2(K) = %.12e, iterations = %" PetscInt_FMT
            ", L2 error = %.12e\n"
            "  cond_2(K_LOR^{-1}K) = %.12e, iterations = %" PetscInt_FMT
            ", L2 error = %.12e\n"
            "  result file: %s\n",
            lor_pc_mode_name(options.lor_pc), options.lor_points.c_str(),
            static_cast<double>(results.condition_K), results.iterations_K,
            static_cast<double>(results.l2_error_K),
            static_cast<double>(results.condition_lor_K),
            results.iterations_lor_K,
            static_cast<double>(results.l2_error_lor_K),
            options.output_file.c_str()));
    }

    // Release all PETSc-owned result objects after output and post-processing.
    PetscCall(destroy_linear_solve_result(&preconditioned_solve));
    PetscCall(destroy_linear_solve_result(&original_solve));
    PetscCall(destroy_global_matrix(&lor_stiffness));
    PetscCall(destroy_global_vector(&high_load));
    PetscCall(destroy_global_matrix(&high_stiffness));

    // Raise the synchronized output failure only after collective cleanup, so
    // no rank can strand its peers inside Mat/Vec destruction.
    if (!output_error.empty())
        throw std::runtime_error(output_error);
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Emit both dimensions supported by the runtime-dispatched driver.
template PetscErrorCode run_assembled_laplace_experiment<2>(
    const GeometryInput &, const ExperimentOptions &, int, MPI_Comm);
template PetscErrorCode run_assembled_laplace_experiment<3>(
    const GeometryInput &, const ExperimentOptions &, int, MPI_Comm);

} // namespace lor
