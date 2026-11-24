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
#include "Assembly/DerivativeGrevilleStiffness.hpp"
#include "Assembly/GlobalAssembly.hpp"
#include "Assembly/LexicographicOrdering.hpp"
#include "Assembly/TruncatedCholeskyTensor.hpp"
#include "Assembly/GrevilleMassPreconditioner.hpp"
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
#include "Solver/GeneralizedConditionNumber.hpp"
#include "Solver/DistributedGrevilleMassPC.hpp"
#include "Solver/LinearSolver.hpp"
#include "Solver/ManufacturedSolution.hpp"

#include <array>
#include <algorithm>
#include <cmath>
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

// The first explicit DMG implementation is intentionally restricted to an
// axis-aligned affine rectangle.  Validate the complete control net against
// the affine B-spline representation and return its two physical lengths.
std::array<double, 2> axis_aligned_rectangle_lengths(
    const BSplinePatch<2> &patch)
{
    if (patch.degree(0) != patch.degree(1) || patch.degree(0) < 2)
        throw std::invalid_argument(
            "derivative_greville_mass requires equal 2D degrees p>=2");
    const auto shape = patch.basis_shape();
    const LexicographicOrdering<2> ordering(shape);
    const auto gx = strict_greville_points(
        patch.degree(0), patch.knots(0));
    const auto gy = strict_greville_points(
        patch.degree(1), patch.knots(1));
    const auto &origin = patch.control_point(ordering.global_index({0, 0}));
    const auto &right = patch.control_point(
        ordering.global_index({shape[0] - 1, 0}));
    const auto &top = patch.control_point(
        ordering.global_index({0, shape[1] - 1}));
    const double length_x = right[0] - origin[0];
    const double length_y = top[1] - origin[1];
    if (!(length_x > 0.0) || !(length_y > 0.0))
        throw std::invalid_argument(
            "derivative_greville_mass requires positively oriented rectangle");
    const double parameter_x0 = patch.knots(0)[patch.degree(0)];
    const double parameter_x1 =
        patch.knots(0)[patch.num_basis(0)];
    const double parameter_y0 = patch.knots(1)[patch.degree(1)];
    const double parameter_y1 =
        patch.knots(1)[patch.num_basis(1)];
    const double tolerance = 2.0e-11 *
        std::max({1.0, std::abs(length_x), std::abs(length_y)});
    for (int j = 0; j < shape[1]; ++j)
        for (int i = 0; i < shape[0]; ++i)
        {
            const double sx = (gx[i] - parameter_x0) /
                (parameter_x1 - parameter_x0);
            const double sy = (gy[j] - parameter_y0) /
                (parameter_y1 - parameter_y0);
            const auto &point = patch.control_point(
                ordering.global_index({i, j}));
            if (std::abs(point[0] - (origin[0] + length_x * sx)) > tolerance ||
                std::abs(point[1] - (origin[1] + length_y * sy)) > tolerance)
                throw std::invalid_argument(
                    "derivative_greville_mass supports only an axis-aligned "
                    "affine rectangle in the assembled benchmark");
        }
    return {length_x, length_y};
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
    PetscLogDouble case_start = 0.0;
    PetscCall(PetscTime(&case_start));

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

    const bool uses_preconditioner_matrix =
        lor_pc_mode_uses_lor_matrix(options.lor_pc);
    const bool uses_lor_matrix = uses_preconditioner_matrix &&
        options.preconditioner_matrix_source ==
            PreconditionerMatrixSource::lor_q1;

    // Construct Greville/Demko and Q1 topology only when the selected matrix
    // source actually consumes it.  The truncated tensor route has its own
    // one-dimensional data and uses the high-order-only slab constructor.
    std::unique_ptr<LORPointSet<Dim>> lor_points;
    std::unique_ptr<LORMesh<Dim>> lor_mesh;
    if (uses_lor_matrix)
    {
        if (options.lor_points == "greville")
            lor_points = std::make_unique<GrevillePointSet<Dim>>(patch);
        else
            lor_points = std::make_unique<DemkoPointSet<Dim>>(patch);
        lor_mesh = std::make_unique<LORMesh<Dim>>(patch, *lor_points);
    }

    // Partition high-order elements, LOR cells, and algebraic rows without
    // changing global_index or ID numbering, then bind both on-demand IEN maps.
    std::unique_ptr<TensorSlabPartition<Dim>> partition;
    if (uses_lor_matrix)
        partition = std::make_unique<TensorSlabPartition<Dim>>(
            patch, *lor_mesh, ID_map, comm);
    else
        partition = std::make_unique<TensorSlabPartition<Dim>>(
            patch, ID_map, comm);
    const BSplineIEN<Dim> high_IEN(patch, *partition);
    std::unique_ptr<LORIEN<Dim>> lor_IEN;
    if (uses_lor_matrix)
        lor_IEN = std::make_unique<LORIEN<Dim>>(*lor_mesh, *partition);

    // Print rank-local half-open ranges in rank order for load-balance and
    // public-interface-DOF diagnostics.
    PetscCall(PetscSynchronizedPrintf(comm,
        "[rank %d/%d] ID=[%d,%d), high elements=[%zu,%zu), "
        "high assembly halo=[%zu,%zu), LOR cells=[%zu,%zu), "
        "LOR assembly halo=[%zu,%zu)\n",
        partition->rank(), partition->size(), partition->owned_ID_begin(),
        partition->owned_ID_end(), partition->high_element_begin(),
        partition->high_element_end(),
        partition->high_assembly_element_begin(),
        partition->high_assembly_element_end(), partition->lor_cell_begin(),
        partition->lor_cell_end(), partition->lor_assembly_cell_begin(),
        partition->lor_assembly_cell_end()));
    PetscCall(PetscSynchronizedFlush(comm, PETSC_STDOUT));

    // A pure Q1-LOR or matrix-free Greville P spectrum does not consume K.
    // Truncated-tensor P still requires K during its construction.
    const bool pure_independent_p_spectrum =
        options.compute_spectral_condition_number &&
        options.spectral_condition_target ==
            SpectralConditionTarget::preconditioner &&
        !options.compute_solution && !options.compute_condition_number &&
        options.preconditioner_matrix_source !=
            PreconditionerMatrixSource::truncated_cholesky_tensor;
    const bool assemble_high_matrix = !pure_independent_p_spectrum;

    // Assemble the high-order stiffness whenever a requested feature consumes
    // it. Fuse load into the traversal only for the two linear solves.
    GlobalMatrixAssemblyResult high_stiffness;
    GlobalVectorAssemblyResult high_load;
    PetscLogDouble original_assembly_start = 0.0;
    PetscLogDouble original_assembly_end = 0.0;
    PetscLogDouble original_assembly_time = 0.0;
    if (assemble_high_matrix)
    {
        PetscCallMPI(MPI_Barrier(comm));
        PetscCall(PetscTime(&original_assembly_start));
        if (options.compute_solution)
            PetscCall(assemble_global_stiffness_load_with_dirichlet<Dim>(patch,
                high_IEN, ID_map, assembly_quad, assembly_basis, source,
                dirichlet_coefficients, &high_stiffness, &high_load));
        else
            PetscCall(assemble_global_stiffness<Dim>(patch, high_IEN,
                ID_map, assembly_quad, assembly_basis, &high_stiffness));
        PetscCall(PetscTime(&original_assembly_end));
        const PetscLogDouble local_original_assembly_time =
            original_assembly_end - original_assembly_start;
        PetscCallMPI(MPI_Allreduce(&local_original_assembly_time,
            &original_assembly_time, 1, MPI_DOUBLE, MPI_MAX, comm));
    }

    // Assemble K_LOR only for configurations whose preconditioner or spectral
    // diagnostic actually consumes it. Direct K reference modes avoid this
    // setup cost while retaining the common partition and error-analysis path.
    PetscLogDouble preconditioner_construction_start = 0.0;
    PetscLogDouble preconditioner_construction_end = 0.0;
    if (uses_preconditioner_matrix)
    {
        PetscCallMPI(MPI_Barrier(comm));
        PetscCall(PetscTime(&preconditioner_construction_start));
    }
    GlobalMatrixAssemblyResult lor_stiffness;
    if (uses_lor_matrix)
    {
        const QuadratureRule1D lor_quad = gauss_legendre_rule(2);
        PetscCall(assemble_global_stiffness<Dim>(
            *lor_mesh, *lor_IEN, ID_map, lor_quad, &lor_stiffness));
    }
    TruncatedCholeskyTensorResult<Dim> truncated_tensor;
    if (uses_preconditioner_matrix &&
        options.preconditioner_matrix_source ==
            PreconditionerMatrixSource::truncated_cholesky_tensor)
    {
        const int effective_k = options.trunc_chol_k_was_auto ?
            (patch.degree(0) + 1) / 2 : options.trunc_chol_k;
        PetscCall(assemble_truncated_cholesky_tensor<Dim>(patch, ID_map,
            high_stiffness.matrix, effective_k,
            options.compute_truncation_certificate,
            options.verify_tensor_identity,
            options.trunc_chol_boundary_protection,
            options.trunc_chol_max_memory_bytes, &truncated_tensor));
    }
    std::array<GrevilleMassPreconditioner1D, 2> greville_directions;
    Mat greville_preconditioner = nullptr;
    DistributedGrevillePCConfiguration2D greville_configuration;
    if (uses_preconditioner_matrix &&
        options.preconditioner_matrix_source ==
            PreconditionerMatrixSource::greville_mass_tensor)
    {
        PetscCheck(Dim == 2, comm, PETSC_ERR_SUP,
            "greville_mass_tensor currently supports only 2D");
        try
        {
            for (int direction = 0; direction < 2; ++direction)
                greville_directions[direction] =
                    restrict_greville_mass_preconditioner_endpoints(
                        build_greville_mass_preconditioner_1d(
                            patch.degree(direction), patch.knots(direction)));
        }
        catch (const std::exception &error)
        {
            SETERRQ(comm, PETSC_ERR_ARG_WRONG, "%s", error.what());
        }
        PetscCall(create_distributed_tensor_greville_preconditioner_2d(
            comm, greville_directions[0], greville_directions[1],
            &greville_preconditioner));
        greville_configuration.x_direction = &greville_directions[0];
        greville_configuration.y_direction = &greville_directions[1];
        greville_configuration.mode =
            DistributedGrevillePCMode::tensor_transpose;
    }
    DerivativeGreville2DResult derivative_greville;
    if (uses_preconditioner_matrix &&
        options.preconditioner_matrix_source ==
            PreconditionerMatrixSource::derivative_greville_mass)
    {
        if constexpr (Dim == 2)
        {
            std::array<double, 2> lengths{};
            try
            {
                lengths = axis_aligned_rectangle_lengths(patch);
            }
            catch (const std::exception &error)
            {
                SETERRQ(comm, PETSC_ERR_ARG_WRONG, "%s", error.what());
            }
            DerivativeGrevilleAssemblyOptions derivative_options;
            derivative_options.assemble_reference_stiffness = false;
            derivative_options.assemble_exact_derivative_masses = false;
            derivative_options.retain_intermediates = false;
            PetscCall(assemble_derivative_greville_stiffness_2d(
                comm, patch.degree(0), patch.knots(0), patch.knots(1),
                lengths[0], lengths[1], derivative_options,
                &derivative_greville));
        }
        else
            SETERRQ(comm, PETSC_ERR_SUP,
                "derivative_greville_mass assembled integration is 2D only");
    }
    PetscLogDouble preconditioner_construction_time = 0.0;
    if (uses_preconditioner_matrix)
    {
        PetscCall(PetscTime(&preconditioner_construction_end));
        const PetscLogDouble local_preconditioner_construction_time =
            preconditioner_construction_end -
            preconditioner_construction_start;
        PetscCallMPI(MPI_Allreduce(
            &local_preconditioner_construction_time,
            &preconditioner_construction_time, 1, MPI_DOUBLE, MPI_MAX,
            comm));
    }
    Mat preconditioner_matrix = nullptr;
    if (uses_lor_matrix)
        preconditioner_matrix = lor_stiffness.matrix;
    else if (options.preconditioner_matrix_source ==
        PreconditionerMatrixSource::truncated_cholesky_tensor)
        preconditioner_matrix = truncated_tensor.preconditioner.matrix;
    else if (options.preconditioner_matrix_source ==
        PreconditionerMatrixSource::greville_mass_tensor)
        preconditioner_matrix = greville_preconditioner;
    else if (options.preconditioner_matrix_source ==
        PreconditionerMatrixSource::derivative_greville_mass)
        preconditioner_matrix = derivative_greville.preconditioner;

    // Test/debug-only PETSc viewers.  They are inert unless their explicit
    // options are supplied and let the integration test form an independent
    // dense generalized eigensystem and dense SVD on a small matrix.
    if (high_stiffness.matrix != nullptr)
        PetscCall(MatViewFromOptions(high_stiffness.matrix, nullptr,
            "-dense_reference_K_view"));
    if (preconditioner_matrix != nullptr)
        PetscCall(MatViewFromOptions(preconditioner_matrix, nullptr,
            "-dense_reference_P_view"));

    // Optionally solve the identical high-order system without and with the
    // selected LOR-based or direct-K preconditioner.
    LinearSolveResult original_solve;
    LinearSolveResult preconditioned_solve;
    if (options.compute_solution)
    {
        if (options.compute_unpreconditioned_solution)
            PetscCall(solve_unpreconditioned(high_stiffness.matrix,
                high_load.vector, options.lor_pc, options.tolerances,
                &original_solve));
        PetscCall(solve_with_lor(high_stiffness.matrix,
            preconditioner_matrix, high_load.vector, options.lor_pc,
            options.inner_pc,
            options.tolerances, options.fixed_jacobi,
            &preconditioned_solve));
    }

    // Optionally build the richer error quadrature and integrate both retained
    // solutions. No error tables are allocated when this feature is disabled.
    PetscReal original_error = 0.0;
    PetscReal preconditioned_error = 0.0;
    PetscLogDouble l2_error_time = 0.0;
    if (options.compute_l2_error)
    {
        PetscCallMPI(MPI_Barrier(comm));
        PetscLogDouble l2_start = 0.0;
        PetscLogDouble l2_end = 0.0;
        PetscCall(PetscTime(&l2_start));
        const QuadratureRule1D error_quad =
            gauss_legendre_rule(patch.degree(0) + 3);
        std::vector<BSplineBasis1D> error_basis;
        error_basis.reserve(Dim);
        for (int direction = 0; direction < Dim; ++direction)
            error_basis.emplace_back(patch.degree(direction),
                patch.knots(direction), error_quad.points);
        if (options.compute_unpreconditioned_solution)
            PetscCall(compute_l2_error<Dim>(patch, high_IEN, ID_map,
                error_quad, error_basis, original_solve.solution,
                dirichlet_coefficients, exact, &original_error));
        PetscCall(compute_l2_error<Dim>(patch, high_IEN, ID_map,
            error_quad, error_basis, preconditioned_solve.solution,
            dirichlet_coefficients, exact, &preconditioned_error));
        PetscCall(PetscTime(&l2_end));
        const PetscLogDouble local_l2_time = l2_end - l2_start;
        PetscCallMPI(MPI_Allreduce(&local_l2_time, &l2_error_time,
            1, MPI_DOUBLE, MPI_MAX, comm));
    }

    // Optionally invoke the independent GlobalAssembly spectral analysis. A
    // disabled condition-number feature creates no SLEPc or factorization data.
    ConditionNumberResult original_condition;
    ConditionNumberResult preconditioned_condition;
    PetscLogDouble singular_svd_time = 0.0;
    if (options.compute_condition_number)
    {
        PetscCallMPI(MPI_Barrier(comm));
        PetscLogDouble analysis_start = 0.0;
        PetscLogDouble analysis_end = 0.0;
        PetscCall(PetscTime(&analysis_start));
        const ConditioningPreconditioner spectral_preconditioner =
            options.lor_pc == LORPCMode::left_gmres_k_jacobi ?
            ConditioningPreconditioner::stiffness_jacobi :
            ConditioningPreconditioner::exact_lor;
        PetscCall(compute_stiffness_condition_numbers(
            high_stiffness.matrix, preconditioner_matrix,
            options.preconditioner_side,
            spectral_preconditioner,
            options.svd_tolerance, options.svd_max_iterations,
            options.svd_subspace_dimension, &original_condition,
            &preconditioned_condition));
        PetscCall(PetscTime(&analysis_end));
        const PetscLogDouble local_analysis_time =
            analysis_end - analysis_start;
        PetscCallMPI(MPI_Allreduce(&local_analysis_time,
            &singular_svd_time, 1, MPI_DOUBLE, MPI_MAX, comm));
    }

    GeneralizedConditionResult spectral_condition;
    PetscLogDouble spectral_eps_time = 0.0;
    if (options.compute_spectral_condition_number)
    {
        Mat spectral_A = nullptr;
        Mat spectral_M = nullptr;
        switch (options.spectral_condition_target)
        {
        case SpectralConditionTarget::stiffness:
            spectral_A = high_stiffness.matrix;
            break;
        case SpectralConditionTarget::preconditioner:
            spectral_A = preconditioner_matrix;
            break;
        case SpectralConditionTarget::preconditioned:
            spectral_A = high_stiffness.matrix;
            spectral_M = preconditioner_matrix;
            break;
        }
        PetscCheck(spectral_A != nullptr, comm, PETSC_ERR_ARG_NULL,
            "Selected EPS spectral target matrix was not assembled");
        const bool greville_source =
            options.preconditioner_matrix_source ==
                PreconditionerMatrixSource::greville_mass_tensor;
        if (greville_source && options.spectral_condition_target ==
            SpectralConditionTarget::preconditioner)
            PetscCall(compute_tensor_greville_preconditioner_condition_2d(
                comm, greville_directions[0], greville_directions[1],
                options.eps_tolerance, options.eps_max_iterations,
                options.eps_subspace_dimension, &spectral_condition));
        else
            PetscCall(compute_generalized_condition_number(
                spectral_A, spectral_M,
                options.eps_tolerance, options.eps_max_iterations,
                options.eps_subspace_dimension, &spectral_condition,
                greville_source && spectral_M != nullptr ?
                    configure_distributed_greville_pc_2d : nullptr,
                greville_source && spectral_M != nullptr ?
                    &greville_configuration : nullptr));
        spectral_eps_time = spectral_condition.total_time;
    }

    // Collect metadata and only the diagnostics whose computations were
    // enabled; root writes a self-describing sparse result record.
    ExperimentResults results;
    results.refinement_mode = options.refinement_mode;
    results.mpi_ranks = partition->size();
    results.actual_degree = patch.degree(0);
    results.total_free_dofs = ID_map.num_free_dofs();
    for (int direction = 0; direction < Dim; ++direction)
        results.actual_elements.push_back(patch.num_elements(direction));
    results.computed_solution = options.compute_solution;
    results.computed_unpreconditioned_solution =
        options.compute_solution && options.compute_unpreconditioned_solution;
    results.computed_l2_error = options.compute_l2_error;
    results.computed_condition_number = options.compute_condition_number;
    results.computed_spectral_condition_number =
        options.compute_spectral_condition_number;
    results.spectral_condition_target = options.spectral_condition_target;
    results.lor_pc = options.lor_pc;
    results.lor_points = options.lor_points;
    results.preconditioner_matrix_source =
        options.preconditioner_matrix_source;
    results.preconditioner_side = options.preconditioner_side;
    results.inner_pc = options.inner_pc;
    results.tolerances = options.tolerances;
    results.fixed_jacobi = options.fixed_jacobi;
    results.assembled_high_matrix = assemble_high_matrix;
    results.assembled_lor_matrix = uses_preconditioner_matrix;
    results.predicted_nonzeros_K = high_stiffness.predicted_nonzeros;
    results.allocated_nonzeros_K = high_stiffness.allocated_nonzeros;
    results.used_nonzeros_K = high_stiffness.used_nonzeros;
    results.allocation_calls_K = high_stiffness.allocation_calls;
    results.memory_bytes_K = high_stiffness.memory_bytes;
    results.estimated_minimum_bytes_K =
        high_stiffness.estimated_minimum_bytes;
    if (uses_preconditioner_matrix &&
        options.preconditioner_matrix_source ==
            PreconditionerMatrixSource::derivative_greville_mass)
    {
        MatInfo info{};
        PetscCall(MatGetInfo(derivative_greville.preconditioner,
            MAT_GLOBAL_SUM, &info));
        results.predicted_nonzeros_lor =
            derivative_greville.nnz_preconditioner;
        results.allocated_nonzeros_lor = info.nz_allocated;
        results.used_nonzeros_lor = info.nz_used;
        results.allocation_calls_lor = info.mallocs;
        results.memory_bytes_lor = info.memory;
        results.estimated_minimum_bytes_lor =
            info.nz_used * (sizeof(PetscScalar) + sizeof(PetscInt));
    }
    else if (uses_preconditioner_matrix &&
        options.preconditioner_matrix_source !=
            PreconditionerMatrixSource::greville_mass_tensor)
    {
        const GlobalMatrixAssemblyResult &selected_preconditioner =
            uses_lor_matrix ? lor_stiffness :
                truncated_tensor.preconditioner;
        results.predicted_nonzeros_lor =
            selected_preconditioner.predicted_nonzeros;
        results.allocated_nonzeros_lor =
            selected_preconditioner.allocated_nonzeros;
        results.used_nonzeros_lor = selected_preconditioner.used_nonzeros;
        results.allocation_calls_lor =
            selected_preconditioner.allocation_calls;
        results.memory_bytes_lor = selected_preconditioner.memory_bytes;
        results.estimated_minimum_bytes_lor =
            selected_preconditioner.estimated_minimum_bytes;
    }
    if (options.compute_condition_number)
    {
        results.condition_K = original_condition.condition_number;
        results.condition_lor_K = preconditioned_condition.condition_number;
        results.singular_condition_converged =
            preconditioned_condition.converged;
        results.original_singular_condition_converged =
            original_condition.converged;
        results.sigma_min = preconditioned_condition.sigma_min;
        results.sigma_max = preconditioned_condition.sigma_max;
        results.svd_iterations_min =
            preconditioned_condition.iterations_min;
        results.svd_iterations_max =
            preconditioned_condition.iterations_max;
        results.svd_reason_min = preconditioned_condition.reason_min;
        results.svd_reason_max = preconditioned_condition.reason_max;
    }
    if (options.compute_spectral_condition_number)
    {
        results.spectral_condition_converged = spectral_condition.converged;
        results.kappa_lambda = spectral_condition.condition_number;
        results.lambda_min = spectral_condition.lambda_min;
        results.lambda_max = spectral_condition.lambda_max;
        results.lambda_relative_error_min =
            spectral_condition.min_eigenpair_error;
        results.lambda_relative_error_max =
            spectral_condition.max_eigenpair_error;
        results.lambda_min_converged_by_stability =
            spectral_condition.min_converged_by_stability;
        results.lambda_min_stability_relative_span =
            spectral_condition.min_stability_relative_span;
        results.lambda_min_stability_residual_limit =
            spectral_condition.min_stability_residual_limit;
        results.lambda_min_stability_window =
            spectral_condition.min_stability_window;
        results.eps_iterations_min =
            spectral_condition.min_eps_iterations;
        results.eps_iterations_max =
            spectral_condition.max_eps_iterations;
        results.eps_reason_min =
            spectral_condition.min_convergence_reason;
        results.eps_reason_max =
            spectral_condition.max_convergence_reason;
        results.time_eps_lambda_min_s = spectral_condition.min_solve_time;
        results.time_eps_lambda_max_s = spectral_condition.max_solve_time;
    }
    if (options.compute_solution)
    {
        if (options.compute_unpreconditioned_solution)
        {
            results.iterations_K = original_solve.iterations;
            results.residual_K = original_solve.residual_norm;
            results.relative_residual_K = original_solve.relative_residual_norm;
            results.setup_time_K = original_solve.setup_time_seconds;
            results.outer_solve_time_K =
                original_solve.outer_solve_time_seconds;
            results.reason_K = original_solve.reason;
        }
        results.iterations_lor_K = preconditioned_solve.iterations;
        results.residual_lor_K = preconditioned_solve.residual_norm;
        results.relative_residual_lor_K =
            preconditioned_solve.relative_residual_norm;
        results.setup_time_lor_K = preconditioned_solve.setup_time_seconds;
        results.outer_solve_time_lor_K =
            preconditioned_solve.outer_solve_time_seconds;
        results.pc_applications_lor_K =
            preconditioned_solve.pc_applications;
        results.inner_iterations_lor_K =
            preconditioned_solve.inner_iterations;
        results.maximum_inner_iterations_lor_K =
            preconditioned_solve.maximum_inner_iterations;
        results.failed_inner_applications_lor_K =
            preconditioned_solve.failed_inner_applications;
        results.last_inner_reason_lor_K =
            preconditioned_solve.last_inner_reason;
        results.inner_solve_time_lor_K =
            preconditioned_solve.inner_solve_time_seconds;
        results.preconditioner_setup_time_lor_K =
            preconditioned_solve.preconditioner_setup_time_seconds;
        results.reason_lor_K = preconditioned_solve.reason;
    }
    if (options.compute_l2_error)
    {
        if (options.compute_unpreconditioned_solution)
            results.l2_error_K = original_error;
        results.l2_error_lor_K = preconditioned_error;
    }
    results.time_original_matrix_assembly_s = original_assembly_time;
    results.time_preconditioner_matrix_construction_s =
        preconditioner_construction_time;
    results.time_spectral_eps_s = spectral_eps_time;
    results.time_singular_svd_s = singular_svd_time;
    results.time_l2_error_s = l2_error_time;
    if (options.preconditioner_matrix_source ==
        PreconditionerMatrixSource::truncated_cholesky_tensor)
    {
        results.trunc_chol_k_was_auto = options.trunc_chol_k_was_auto;
        results.trunc_chol_k_effective = options.trunc_chol_k_was_auto ?
            (patch.degree(0) + 1) / 2 : options.trunc_chol_k;
        results.trunc_chol_original_degree_p = patch.degree(0);
        results.trunc_chol_mass_degree_q = patch.degree(0) - 1;
        results.trunc_chol_boundary_protection =
            options.trunc_chol_boundary_protection;
        results.trunc_chol_certificate_computed =
            options.compute_truncation_certificate;
        results.trunc_chol_certificate_bound_available =
            truncated_tensor.certificate_bound_available;
        results.trunc_chol_theoretical_kappa_lambda_bound =
            truncated_tensor.theoretical_kappa_lambda_bound;
        results.tensor_identity_relative_error =
            truncated_tensor.tensor_identity_relative_error;
        results.tensor_identity_verification_computed =
            truncated_tensor.tensor_identity_verification_computed;
        results.tensor_identity_verification_passed =
            truncated_tensor.tensor_identity_verification_passed;
        results.trunc_chol_conservative_memory_estimate_bytes =
            truncated_tensor.conservative_memory_estimate_bytes;
        results.trunc_chol_memory_limit_bytes =
            truncated_tensor.memory_limit_bytes;
        results.preconditioner_symmetric = truncated_tensor.symmetric;
        results.preconditioner_minimum_diagonal =
            truncated_tensor.minimum_diagonal;
        for (const auto &direction : truncated_tensor.directions)
        {
            results.trunc_chol_value_original_half_bandwidth.push_back(
                direction.value_mass.original_half_bandwidth);
            results.trunc_chol_value_complete_factor_half_bandwidth.push_back(
                direction.value_mass.complete_factor_half_bandwidth);
            results.trunc_chol_value_truncated_factor_half_bandwidth.push_back(
                direction.value_mass.truncated_factor_half_bandwidth);
            results.trunc_chol_derivative_original_half_bandwidth.push_back(
                direction.derivative_mass.original_half_bandwidth);
            results.trunc_chol_derivative_complete_factor_half_bandwidth.push_back(
                direction.derivative_mass.complete_factor_half_bandwidth);
            results.trunc_chol_derivative_truncated_factor_half_bandwidth.push_back(
                direction.derivative_mass.truncated_factor_half_bandwidth);
            results.trunc_chol_value_complete_factor_nonzeros.push_back(
                direction.value_mass.complete_factor_nonzeros);
            results.trunc_chol_value_truncated_factor_nonzeros.push_back(
                direction.value_mass.truncated_factor_nonzeros);
            results.trunc_chol_value_dropped_factor_entries.push_back(
                direction.value_mass.dropped_factor_entries);
            results.trunc_chol_derivative_complete_factor_nonzeros.push_back(
                direction.derivative_mass.complete_factor_nonzeros);
            results.trunc_chol_derivative_truncated_factor_nonzeros.push_back(
                direction.derivative_mass.truncated_factor_nonzeros);
            results.trunc_chol_derivative_dropped_factor_entries.push_back(
                direction.derivative_mass.dropped_factor_entries);
            results.derivative_identity_max_relative_error = std::max(
                results.derivative_identity_max_relative_error,
                direction.derivative_identity_relative_error);
            if (options.compute_truncation_certificate)
            {
                results.trunc_chol_beta_value_mass.push_back(
                    direction.value_mass.certificate.beta);
                results.trunc_chol_beta_derivative_mass.push_back(
                    direction.derivative_mass.certificate.beta);
                results.trunc_chol_beta_below_one.push_back(
                    direction.value_mass.certificate.beta_below_one &&
                    direction.derivative_mass.certificate.beta_below_one);
                results.trunc_chol_certificate_value_maximum_row_sum.push_back(
                    direction.value_mass.certificate.maximum_row_sum);
                results.trunc_chol_certificate_value_maximum_column_sum.push_back(
                    direction.value_mass.certificate.maximum_column_sum);
                results.trunc_chol_certificate_derivative_maximum_row_sum.push_back(
                    direction.derivative_mass.certificate.maximum_row_sum);
                results.trunc_chol_certificate_derivative_maximum_column_sum.push_back(
                    direction.derivative_mass.certificate.maximum_column_sum);
            }
            results.trunc_chol_dropped_entries +=
                direction.value_mass.dropped_factor_entries +
                direction.derivative_mass.dropped_factor_entries;
            results.trunc_chol_complete_factor_nonzeros +=
                direction.value_mass.complete_factor_nonzeros +
                direction.derivative_mass.complete_factor_nonzeros;
            results.trunc_chol_truncated_factor_nonzeros +=
                direction.value_mass.truncated_factor_nonzeros +
                direction.derivative_mass.truncated_factor_nonzeros;
        }
        results.time_1d_mass_assembly_s =
            truncated_tensor.time_1d_mass_assembly_seconds;
        results.time_1d_full_cholesky_s =
            truncated_tensor.time_1d_full_cholesky_seconds;
        results.time_1d_factor_truncation_s =
            truncated_tensor.time_1d_factor_truncation_seconds;
        results.time_truncation_certificate_s =
            truncated_tensor.time_truncation_certificate_seconds;
        results.time_1d_reduced_operators_s =
            truncated_tensor.time_1d_reduced_operators_seconds;
        results.time_tensor_pc_assembly_s =
            truncated_tensor.time_tensor_assembly_seconds;
        results.time_tensor_memory_preflight_s =
            truncated_tensor.time_tensor_memory_preflight_seconds;
        results.time_tensor_identity_verification_s =
            truncated_tensor.time_tensor_identity_verification_seconds;
    }
    PetscLogDouble case_end = 0.0;
    PetscCall(PetscTime(&case_end));
    const PetscLogDouble local_case_time = case_end - case_start;
    PetscCallMPI(MPI_Allreduce(&local_case_time,
        &results.time_case_total_s, 1, MPI_DOUBLE, MPI_MAX, comm));
    std::string output_error;
    if (partition->rank() == root)
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
    broadcast_root_error(output_error, root, partition->rank(), comm);

    // Print one concise run summary only after every rank knows output
    // succeeded; PetscPrintf emits it through the communicator's rank zero.
    if (output_error.empty())
    {
        PetscCall(PetscPrintf(comm,
            "Completed MPI ranks=%d, refinement=%s, actual p=%d, actual elements=(",
            partition->size(), options.refinement_mode.c_str(), patch.degree(0)));
        for (int direction = 0; direction < Dim; ++direction)
            PetscCall(PetscPrintf(comm, "%s%d", direction == 0 ? "" : ",",
                patch.num_elements(direction)));
        PetscCall(PetscPrintf(comm, "), mode=%s, side=%s",
            lor_pc_mode_name(options.lor_pc),
            preconditioner_side_name(options.preconditioner_side)));
        PetscCall(PetscPrintf(comm, ", matrix_source=%s",
            preconditioner_matrix_source_name(
                options.preconditioner_matrix_source)));
        if (options.lor_pc == LORPCMode::left_gmres_cg ||
            options.lor_pc == LORPCMode::right_gmres_cg ||
            options.lor_pc == LORPCMode::right_fgmres_cg)
            PetscCall(PetscPrintf(comm, ", inner_pc=%s",
                inner_pc_mode_name(options.inner_pc)));
        else if (options.lor_pc ==
                 LORPCMode::left_gmres_fixed_jacobi)
            PetscCall(PetscPrintf(comm,
                ", fixed_jacobi_steps=%" PetscInt_FMT
                ", fixed_jacobi_omega=%.6f",
                options.fixed_jacobi.steps,
                static_cast<double>(options.fixed_jacobi.omega)));
        PetscCall(PetscPrintf(comm,
            ", point_set=%s, solve=%s, L2=%s, condition=%s\n",
            options.lor_points.c_str(),
            options.compute_solution ? "true" : "false",
            options.compute_l2_error ? "true" : "false",
            options.compute_condition_number ? "true" : "false"));
        if (results.assembled_high_matrix)
            PetscCall(PetscPrintf(comm,
                "  K storage: predicted_nnz=%" PetscInt_FMT
                ", used_nnz=%.0f, allocated_nnz=%.0f, mallocs=%.0f, "
                "minimum=%.3f MiB, PETSc_reported=%.3f MiB\n",
                results.predicted_nonzeros_K, results.used_nonzeros_K,
                results.allocated_nonzeros_K, results.allocation_calls_K,
                results.estimated_minimum_bytes_K / (1024.0 * 1024.0),
                results.memory_bytes_K / (1024.0 * 1024.0)));
        else
            PetscCall(PetscPrintf(comm,
                "  K storage: not assembled for P-only EPS mode\n"));
        if (options.preconditioner_matrix_source ==
            PreconditionerMatrixSource::greville_mass_tensor)
            PetscCall(PetscPrintf(comm,
                "  P storage: matrix-free Greville tensor MatShell\n"));
        else if (uses_preconditioner_matrix)
            PetscCall(PetscPrintf(comm,
                "  P storage: predicted_nnz=%" PetscInt_FMT
                ", used_nnz=%.0f, allocated_nnz=%.0f, mallocs=%.0f, "
                "minimum=%.3f MiB, PETSc_reported=%.3f MiB\n",
                results.predicted_nonzeros_lor, results.used_nonzeros_lor,
                results.allocated_nonzeros_lor,
                results.allocation_calls_lor,
                results.estimated_minimum_bytes_lor / (1024.0 * 1024.0),
                results.memory_bytes_lor / (1024.0 * 1024.0)));
        else
            PetscCall(PetscPrintf(comm,
                "  P storage: not assembled for direct-K mode\n"));
        // Distinguish the ideal exact-LOR spectral operator from the actual
        // algorithm applying an exact, variable, or fixed approximate inverse.
        const char *ideal_operator =
            options.lor_pc == LORPCMode::left_gmres_k_jacobi ?
            "D_K^{-1}K" :
            (options.preconditioner_matrix_source ==
                PreconditionerMatrixSource::truncated_cholesky_tensor ?
                (options.preconditioner_side == PreconditionerSide::left ?
                    "P_k^{-1}K" : "KP_k^{-1}") :
            (options.preconditioner_matrix_source ==
                PreconditionerMatrixSource::derivative_greville_mass ?
                (options.preconditioner_side == PreconditionerSide::left ?
                    "P_DMG^{-1}K" : "KP_DMG^{-1}") :
                (options.preconditioner_side == PreconditionerSide::left ?
                    "K_LOR^{-1}K" : "KK_LOR^{-1}")));
        const char *solve_label = lor_pc_mode_name(options.lor_pc);
        if (options.compute_condition_number)
        {
            PetscCall(PetscPrintf(comm, "  SVD status: K=%s, %s=%s\n",
                results.original_singular_condition_converged ?
                    "converged" : "not_converged",
                ideal_operator,
                results.singular_condition_converged ?
                    "converged" : "not_converged"));
            if (results.original_singular_condition_converged)
                PetscCall(PetscPrintf(comm, "  cond_2(K) = %.12e\n",
                    static_cast<double>(results.condition_K)));
            if (results.singular_condition_converged)
                PetscCall(PetscPrintf(comm, "  cond_2(%s) = %.12e\n",
                    ideal_operator,
                    static_cast<double>(results.condition_lor_K)));
        }
        if (options.compute_spectral_condition_number)
        {
            const char *spectral_operator =
                options.spectral_condition_target ==
                    SpectralConditionTarget::stiffness ? "K" :
                (options.spectral_condition_target ==
                    SpectralConditionTarget::preconditioner ? "P" :
                    "P^{-1}K");
            PetscCall(PetscPrintf(comm,
                "  EPS %s: status=%s, time=%.6e s\n",
                spectral_operator,
                results.spectral_condition_converged ?
                    "converged" : "not_converged",
                static_cast<double>(results.time_spectral_eps_s)));
            if (results.spectral_condition_converged)
                PetscCall(PetscPrintf(comm,
                    "    lambda_min=%.12e, lambda_max=%.12e, "
                    "kappa_lambda=%.12e, min_criterion=%s\n",
                    static_cast<double>(results.lambda_min),
                    static_cast<double>(results.lambda_max),
                    static_cast<double>(results.kappa_lambda),
                    results.lambda_min_converged_by_stability ?
                        "ritz_stability" : "residual"));
        }
        if (options.preconditioner_matrix_source ==
            PreconditionerMatrixSource::truncated_cholesky_tensor)
            PetscCall(PetscPrintf(comm,
                "  truncated tensor: k=%d, tensor_identity_status=%s, "
                "1D_derivative_identity=%.6e, dropped=%zu, symmetric=%s, "
                "min_diagonal=%.6e, certificate_bound=%s\n",
                results.trunc_chol_k_effective,
                !results.tensor_identity_verification_computed ?
                    "not_computed" :
                    (results.tensor_identity_verification_passed ?
                        "passed" : "failed"),
                static_cast<double>(
                    results.derivative_identity_max_relative_error),
                results.trunc_chol_dropped_entries,
                results.preconditioner_symmetric ? "true" : "false",
                static_cast<double>(results.preconditioner_minimum_diagonal),
                results.trunc_chol_certificate_bound_available ?
                    "available" : "unavailable"));
        if (options.compute_solution)
        {
            if (options.compute_unpreconditioned_solution)
                PetscCall(PetscPrintf(comm,
                    "  solve(K): outer_iterations=%" PetscInt_FMT
                    ", setup=%.6e s, outer_total=%.6e s\n",
                    results.iterations_K,
                    static_cast<double>(results.setup_time_K),
                    static_cast<double>(results.outer_solve_time_K)));
            else
                PetscCall(PetscPrintf(comm,
                    "  solve(K): not_computed\n"));
            PetscCall(PetscPrintf(comm,
                "  solve(%s): outer_iterations=%" PetscInt_FMT
                ", setup=%.6e s, outer_total=%.6e s\n",
                solve_label, results.iterations_lor_K,
                static_cast<double>(results.setup_time_lor_K),
                static_cast<double>(results.outer_solve_time_lor_K)));
            if (options.lor_pc == LORPCMode::left_gmres_cg ||
                options.lor_pc == LORPCMode::right_gmres_cg ||
                options.lor_pc == LORPCMode::right_fgmres_cg)
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
            else if (options.lor_pc == LORPCMode::right_fgmres_amg ||
                     options.lor_pc == LORPCMode::left_gmres_amg)
                PetscCall(PetscPrintf(comm,
                    "  AMG-only: applications=%" PetscInt_FMT
                    ", CG_iterations=not_applicable, setup=%.6e s, "
                    "apply_total=%.6e s\n",
                    results.pc_applications_lor_K,
                    static_cast<double>(
                        results.preconditioner_setup_time_lor_K),
                    static_cast<double>(results.inner_solve_time_lor_K)));
            else if (options.lor_pc == LORPCMode::right_fgmres_jacobi)
                PetscCall(PetscPrintf(comm,
                    "  Jacobi-only: applications=%" PetscInt_FMT
                    ", CG_iterations=not_applicable, setup=%.6e s, "
                    "apply_total=%.6e s\n",
                    results.pc_applications_lor_K,
                    static_cast<double>(
                        results.preconditioner_setup_time_lor_K),
                    static_cast<double>(results.inner_solve_time_lor_K)));
            else if (options.lor_pc == LORPCMode::left_cg_lu ||
                     options.lor_pc == LORPCMode::right_gmres_lu)
                PetscCall(PetscPrintf(comm,
                    "  LU inverse: applications=%" PetscInt_FMT
                    ", CG_iterations=not_applicable, factor_setup=%.6e s, "
                    "apply_total=%.6e s\n",
                    results.pc_applications_lor_K,
                    static_cast<double>(
                        results.preconditioner_setup_time_lor_K),
                    static_cast<double>(results.inner_solve_time_lor_K)));
            else if (options.lor_pc ==
                     LORPCMode::left_gmres_fixed_jacobi)
                PetscCall(PetscPrintf(comm,
                    "  fixed Jacobi: applications=%" PetscInt_FMT
                    ", steps_per_application=%" PetscInt_FMT
                    ", steps_total=%" PetscInt_FMT
                    ", time_total=%.6e s\n",
                    results.pc_applications_lor_K,
                    options.fixed_jacobi.steps,
                    results.inner_iterations_lor_K,
                    static_cast<double>(results.inner_solve_time_lor_K)));
        }
        if (options.compute_l2_error)
        {
            if (options.compute_unpreconditioned_solution)
                PetscCall(PetscPrintf(comm,
                    "  L2_error(K) = %.12e\n",
                    static_cast<double>(results.l2_error_K)));
            else
                PetscCall(PetscPrintf(comm,
                    "  L2_error(K) = not_computed\n"));
            PetscCall(PetscPrintf(comm,
                "  L2_error(%s) = %.12e\n", solve_label,
                static_cast<double>(results.l2_error_lor_K)));
        }
        PetscCall(PetscPrintf(comm, "  result file: %s\n",
            options.output_file.c_str()));
    }

    // Release all PETSc-owned result objects after output and post-processing.
    PetscCall(destroy_linear_solve_result(&preconditioned_solve));
    PetscCall(destroy_linear_solve_result(&original_solve));
    PetscCall(destroy_derivative_greville_result(&derivative_greville));
    PetscCall(MatDestroy(&greville_preconditioner));
    PetscCall(destroy_global_matrix(&truncated_tensor.preconditioner));
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
