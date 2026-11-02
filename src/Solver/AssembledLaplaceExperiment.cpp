/*
 * Manufactured solution used by this assembled Laplace experiment:
 *
 *   u_exact(x) = 1 + sum_d x_d + geometry-compatible bubble(x).
 *
 * The bubble is zero on the physical boundary, so every geometry shares the
 * exactly representable nonzero trace 1+sum_d x_d. The interior sine or curved
 * polynomial term provides a nonzero source and meaningful discretization
 * error while retaining exact strong Dirichlet coefficients.
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

    // Refine once on root and distribute the refined control net. Manufactured
    // source/exact functions are created later only when a solve is requested.
    const BSplinePatch<Dim> patch = build_distributed_patch<Dim>(
        geometry_input, options, root, comm);

    // Build Dirichlet constraint topology from Patch boundary multi-indices;
    // prescribed coefficient values remain separate from the ID=-1 map.
    const LexicographicOrdering<Dim> ordering(patch.basis_shape());
    const DirichletBC<Dim> dirichlet_BC(patch.boundaries(), ordering);
    const ID ID_map(dirichlet_BC.num_global_indices(),
        dirichlet_BC.constrained_global_indices());
    PetscCheck(ID_map.num_free_dofs() > 0, comm, PETSC_ERR_ARG_SIZ,
        "All coefficients are constrained; increase the element counts");

    // Precompute one high-order one-dimensional basis table per independent
    // knot vector for stiffness and, when enabled, load assembly.
    const QuadratureRule1D assembly_quad =
        gauss_legendre_rule(patch.degree(0) + 1);
    std::vector<BSplineBasis1D> assembly_basis;
    assembly_basis.reserve(Dim);
    for (int direction = 0; direction < Dim; ++direction)
        assembly_basis.emplace_back(patch.degree(direction),
            patch.knots(direction), assembly_quad.points);
    std::unique_ptr<ManufacturedSolution<Dim>> manufactured;
    ScalarSource<Dim> source;
    ScalarExactSolution<Dim> exact;
    std::vector<double> dirichlet_coefficients;
    if (options.compute_solution)
    {
        // Instantiate the physical exact/source callbacks only for a requested
        // solve, retaining their owner through subsequent error integration.
        manufactured =
            std::make_unique<ManufacturedSolution<Dim>>(patch);
        const ManufacturedSolution<Dim> *manufactured_view =
            manufactured.get();
        source = [manufactured_view](
            const std::array<double, Dim> &physical_coordinate) {
            return manufactured_view->source(physical_coordinate);
        };
        if (options.compute_l2_error)
            exact = [manufactured_view](
                const std::array<double, Dim> &physical_coordinate) {
                return manufactured_view->exact(physical_coordinate);
            };

        // Build the MixPERIGEE-style nonzero Dirichlet lift separately from
        // ID=-1 topology. Affine physical fields have exact B-spline
        // coefficients obtained by applying the field to control points.
        dirichlet_coefficients.assign(patch.num_basis(), 0.0);
        for (const int global_index :
             dirichlet_BC.constrained_global_indices())
            dirichlet_coefficients[global_index] =
                manufactured->boundary_coefficient(
                    patch.control_point(global_index));
    }

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
        "high assembly halo=[%zu,%zu), LOR cells=[%zu,%zu), "
        "LOR assembly halo=[%zu,%zu)\n",
        partition.rank(), partition.size(), partition.owned_ID_begin(),
        partition.owned_ID_end(), partition.high_element_begin(),
        partition.high_element_end(),
        partition.high_assembly_element_begin(),
        partition.high_assembly_element_end(), partition.lor_cell_begin(),
        partition.lor_cell_end(), partition.lor_assembly_cell_begin(),
        partition.lor_assembly_cell_end()));
    PetscCall(PetscSynchronizedFlush(comm, PETSC_STDOUT));

    // Always assemble the high-order stiffness. Fuse load into the traversal
    // only when the user requested the two linear solves.
    GlobalMatrixAssemblyResult high_stiffness;
    GlobalVectorAssemblyResult high_load;
    if (options.compute_solution)
        PetscCall(assemble_global_stiffness_load_with_dirichlet<Dim>(patch,
            high_IEN, ID_map, assembly_quad, assembly_basis, source,
            dirichlet_coefficients, &high_stiffness, &high_load));
    else
        PetscCall(assemble_global_stiffness<Dim>(patch, high_IEN,
            ID_map, assembly_quad, assembly_basis, &high_stiffness));

    // Assemble only the Q1 LOR stiffness matrix used by the preconditioner.
    const QuadratureRule1D lor_quad = gauss_legendre_rule(2);
    GlobalMatrixAssemblyResult lor_stiffness;
    PetscCall(assemble_global_stiffness<Dim>(
        lor_mesh, lor_IEN, ID_map, lor_quad, &lor_stiffness));

    // Optionally solve the identical high-order system without and with LOR.
    // Iteration counts are a direct byproduct of these two solves.
    LinearSolveResult original_solve;
    LinearSolveResult preconditioned_solve;
    if (options.compute_solution)
    {
        PetscCall(solve_unpreconditioned(high_stiffness.matrix,
            high_load.vector, options.lor_pc, options.tolerances,
            &original_solve));
        PetscCall(solve_with_lor(high_stiffness.matrix,
            lor_stiffness.matrix, high_load.vector, options.lor_pc,
            options.preconditioner_side, options.inner_pc,
            options.tolerances,
            &preconditioned_solve));
    }

    // Optionally build the richer error quadrature and integrate both retained
    // solutions. No error tables are allocated when this feature is disabled.
    PetscReal original_error = 0.0;
    PetscReal preconditioned_error = 0.0;
    if (options.compute_l2_error)
    {
        const QuadratureRule1D error_quad =
            gauss_legendre_rule(patch.degree(0) + 3);
        std::vector<BSplineBasis1D> error_basis;
        error_basis.reserve(Dim);
        for (int direction = 0; direction < Dim; ++direction)
            error_basis.emplace_back(patch.degree(direction),
                patch.knots(direction), error_quad.points);
        PetscCall(compute_l2_error<Dim>(patch, high_IEN, ID_map,
            error_quad, error_basis, original_solve.solution,
            dirichlet_coefficients, exact, &original_error));
        PetscCall(compute_l2_error<Dim>(patch, high_IEN, ID_map,
            error_quad, error_basis, preconditioned_solve.solution,
            dirichlet_coefficients, exact, &preconditioned_error));
    }

    // Optionally invoke the independent GlobalAssembly spectral analysis. A
    // disabled condition-number feature creates no SLEPc or factorization data.
    ConditionNumberResult original_condition;
    ConditionNumberResult preconditioned_condition;
    if (options.compute_condition_number)
        PetscCall(compute_stiffness_condition_numbers(
            high_stiffness.matrix, lor_stiffness.matrix,
            options.preconditioner_side,
            options.svd_tolerance, options.svd_max_iterations,
            options.svd_subspace_dimension, &original_condition,
            &preconditioned_condition));

    // Collect metadata and only the diagnostics whose computations were
    // enabled; root writes a self-describing sparse result record.
    ExperimentResults results;
    results.refinement_mode = options.refinement_mode;
    results.mpi_ranks = partition.size();
    results.actual_degree = patch.degree(0);
    for (int direction = 0; direction < Dim; ++direction)
        results.actual_elements.push_back(patch.num_elements(direction));
    results.computed_solution = options.compute_solution;
    results.computed_l2_error = options.compute_l2_error;
    results.computed_condition_number = options.compute_condition_number;
    results.lor_pc = options.lor_pc;
    results.preconditioner_side = options.preconditioner_side;
    results.inner_pc = options.inner_pc;
    results.tolerances = options.tolerances;
    results.predicted_nonzeros_K = high_stiffness.predicted_nonzeros;
    results.allocated_nonzeros_K = high_stiffness.allocated_nonzeros;
    results.used_nonzeros_K = high_stiffness.used_nonzeros;
    results.allocation_calls_K = high_stiffness.allocation_calls;
    results.memory_bytes_K = high_stiffness.memory_bytes;
    results.estimated_minimum_bytes_K =
        high_stiffness.estimated_minimum_bytes;
    results.predicted_nonzeros_lor = lor_stiffness.predicted_nonzeros;
    results.allocated_nonzeros_lor = lor_stiffness.allocated_nonzeros;
    results.used_nonzeros_lor = lor_stiffness.used_nonzeros;
    results.allocation_calls_lor = lor_stiffness.allocation_calls;
    results.memory_bytes_lor = lor_stiffness.memory_bytes;
    results.estimated_minimum_bytes_lor =
        lor_stiffness.estimated_minimum_bytes;
    if (options.compute_condition_number)
    {
        results.condition_K = original_condition.condition_number;
        results.condition_lor_K = preconditioned_condition.condition_number;
    }
    if (options.compute_solution)
    {
        results.iterations_K = original_solve.iterations;
        results.iterations_lor_K = preconditioned_solve.iterations;
        results.residual_K = original_solve.residual_norm;
        results.relative_residual_K = original_solve.relative_residual_norm;
        results.residual_lor_K = preconditioned_solve.residual_norm;
        results.relative_residual_lor_K =
            preconditioned_solve.relative_residual_norm;
        results.setup_time_K = original_solve.setup_time_seconds;
        results.outer_solve_time_K =
            original_solve.outer_solve_time_seconds;
        results.setup_time_lor_K = preconditioned_solve.setup_time_seconds;
        results.outer_solve_time_lor_K =
            preconditioned_solve.outer_solve_time_seconds;
        results.pc_applications_lor_K =
            preconditioned_solve.pc_applications;
        results.inner_iterations_lor_K =
            preconditioned_solve.inner_iterations;
        results.inner_solve_time_lor_K =
            preconditioned_solve.inner_solve_time_seconds;
    }
    if (options.compute_l2_error)
    {
        results.l2_error_K = original_error;
        results.l2_error_lor_K = preconditioned_error;
    }
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
            "), mode=%s, side=%s, inner_pc=%s, point_set=%s, solve=%s, "
            "L2=%s, condition=%s\n",
            lor_pc_mode_name(options.lor_pc),
            preconditioner_side_name(options.preconditioner_side),
            inner_pc_mode_name(options.inner_pc),
            options.lor_points.c_str(),
            options.compute_solution ? "true" : "false",
            options.compute_l2_error ? "true" : "false",
            options.compute_condition_number ? "true" : "false"));
        PetscCall(PetscPrintf(comm,
            "  K storage: predicted_nnz=%" PetscInt_FMT
            ", used_nnz=%.0f, allocated_nnz=%.0f, mallocs=%.0f, "
            "minimum=%.3f MiB, PETSc_reported=%.3f MiB\n"
            "  K_LOR storage: predicted_nnz=%" PetscInt_FMT
            ", used_nnz=%.0f, allocated_nnz=%.0f, mallocs=%.0f, "
            "minimum=%.3f MiB, PETSc_reported=%.3f MiB\n",
            results.predicted_nonzeros_K, results.used_nonzeros_K,
            results.allocated_nonzeros_K, results.allocation_calls_K,
            results.estimated_minimum_bytes_K / (1024.0 * 1024.0),
            results.memory_bytes_K / (1024.0 * 1024.0),
            results.predicted_nonzeros_lor, results.used_nonzeros_lor,
            results.allocated_nonzeros_lor, results.allocation_calls_lor,
            results.estimated_minimum_bytes_lor / (1024.0 * 1024.0),
            results.memory_bytes_lor / (1024.0 * 1024.0)));
        // Keep console labels consistent with the operator selected for both
        // the solve and optional condition-number analysis.
        const char *preconditioned_operator =
            options.preconditioner_side == PreconditionerSide::left ?
            "K_LOR^{-1}K" : "KK_LOR^{-1}";
        if (options.compute_condition_number)
            PetscCall(PetscPrintf(comm,
                "  cond_2(K) = %.12e\n"
                "  cond_2(%s) = %.12e\n",
                static_cast<double>(results.condition_K),
                preconditioned_operator,
                static_cast<double>(results.condition_lor_K)));
        if (options.compute_solution)
        {
            PetscCall(PetscPrintf(comm,
                "  solve(K): outer_iterations=%" PetscInt_FMT
                ", setup=%.6e s, outer_total=%.6e s\n"
                "  solve(%s): outer_iterations=%" PetscInt_FMT
                ", setup=%.6e s, outer_total=%.6e s\n",
                results.iterations_K,
                static_cast<double>(results.setup_time_K),
                static_cast<double>(results.outer_solve_time_K),
                preconditioned_operator, results.iterations_lor_K,
                static_cast<double>(results.setup_time_lor_K),
                static_cast<double>(results.outer_solve_time_lor_K)));
            if (options.lor_pc == LORPCMode::gmres_shell)
            {
                // Summarize the approximate LOR inverse cost alongside the
                // complete outer solve without adding the nested times.
                const double applications = static_cast<double>(
                    results.pc_applications_lor_K);
                const double average_iterations = applications > 0.0 ?
                    static_cast<double>(results.inner_iterations_lor_K) /
                        applications : 0.0;
                const double average_time = applications > 0.0 ?
                    results.inner_solve_time_lor_K / applications : 0.0;
                PetscCall(PetscPrintf(comm,
                    "  inner CG: applications=%" PetscInt_FMT
                    ", iterations_total=%" PetscInt_FMT
                    ", iterations_average=%.6e, time_total=%.6e s, "
                    "time_average=%.6e s\n",
                    results.pc_applications_lor_K,
                    results.inner_iterations_lor_K, average_iterations,
                    static_cast<double>(results.inner_solve_time_lor_K),
                    average_time));
            }
        }
        if (options.compute_l2_error)
            PetscCall(PetscPrintf(comm,
                "  L2_error(K) = %.12e\n"
                "  L2_error(%s) = %.12e\n",
                static_cast<double>(results.l2_error_K),
                preconditioned_operator,
                static_cast<double>(results.l2_error_lor_K)));
        PetscCall(PetscPrintf(comm, "  result file: %s\n",
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
