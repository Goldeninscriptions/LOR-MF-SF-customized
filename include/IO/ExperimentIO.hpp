#ifndef LOR_IO_EXPERIMENT_IO_HPP
#define LOR_IO_EXPERIMENT_IO_HPP

#include "Solver/LinearSolver.hpp"

#include <petscsys.h>

#include <string>
#include <vector>

namespace lor
{

struct ExperimentOptions
{
    // Geometry description read by GeometryInput and result-file target.
    std::string geometry_file;
    std::string output_file;

    // Refinement policy selected from p_refine, h_refine, or k_refine.
    std::string refinement_mode = "k_refine";

    // Requested common final degree and final element counts. The active
    // refinement policy validates which entries are allowed to change.
    int degree = 0;
    std::vector<int> elements;

    // LOR point strategy and one of the two supported solver hierarchies.
    std::string lor_points = "greville";
    LORPCMode lor_pc = LORPCMode::cg_lu;

    // KSP stopping controls and independent SLEPc singular-value tolerance.
    SolverTolerances tolerances;
    PetscReal svd_tolerance = 1.0e-8;
};

// Self-describing experiment record used by both direct and batch workflows.
// The two errors compare solutions of the same K u=F system; they may differ
// only because the solver modes terminate at different tolerances.
struct ExperimentResults
{
    // Actual case metadata queried from the refined Patch, never copied
    // blindly from requested command-line targets.
    std::string refinement_mode;
    int actual_degree = 0;
    std::vector<int> actual_elements;

    // Unpreconditioned high-order stiffness result triplet.
    PetscReal condition_K = 0.0;
    PetscInt iterations_K = 0;
    PetscReal l2_error_K = 0.0;

    // Left-LOR-preconditioned operator result triplet.
    PetscReal condition_lor_K = 0.0;
    PetscInt iterations_lor_K = 0;
    PetscReal l2_error_lor_K = 0.0;
};

// Read the experiment command line after PETSc/SLEPc initialization.
std::string read_geometry_filename();
ExperimentOptions read_experiment_options(int dimension);

// Write actual refined-space metadata followed by the six numerical results,
// with one key-value item on every line.
void write_experiment_results(const std::string &filename,
    const ExperimentResults &results);

} // namespace lor

#endif
