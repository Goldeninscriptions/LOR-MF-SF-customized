#include "Assembly/DerivativeGrevilleStiffness.hpp"
#include "Assembly/GrevilleMassPreconditioner.hpp"
#include "derivative_greville_driver_common.hpp"

#include <slepceps.h>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{

std::pair<int, int> parse_elements(const std::string &text)
{
    std::stringstream stream(text);
    std::string first;
    std::string second;
    std::string extra;
    if (!std::getline(stream, first, ',') ||
        !std::getline(stream, second, ',') || std::getline(stream, extra, ','))
        throw std::invalid_argument("-n must have the form Ex,Ey");
    std::size_t consumed_x = 0;
    std::size_t consumed_y = 0;
    const int ex = std::stoi(first, &consumed_x);
    const int ey = std::stoi(second, &consumed_y);
    if (consumed_x != first.size() || consumed_y != second.size() ||
        ex < 1 || ey < 1)
        throw std::invalid_argument("Ex and Ey must be positive");
    return {ex, ey};
}

PetscErrorCode run()
{
    PetscFunctionBeginUser;
    const MPI_Comm comm = PETSC_COMM_WORLD;
    PetscInt degree = 0;
    PetscBool degree_set = PETSC_FALSE;
    PetscBool elements_set = PETSC_FALSE;
    PetscBool compute_spectrum = PETSC_TRUE;
    PetscBool compute_solve = PETSC_TRUE;
    PetscReal length_x = 1.0;
    PetscReal length_y = 1.0;
    PetscReal eps_tolerance = 1.0e-10;
    PetscInt eps_max_iterations = 2000;
    PetscInt eps_ncv = 40;
    char elements_text[128] = "";
    char output_path[PETSC_MAX_PATH_LEN] = "derivative_greville_2d.csv";
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-p",
        &degree, &degree_set));
    PetscCall(PetscOptionsGetString(nullptr, nullptr, "-n",
        elements_text, sizeof(elements_text), &elements_set));
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-Lx",
        &length_x, nullptr));
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-Ly",
        &length_y, nullptr));
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-compute_spectrum",
        &compute_spectrum, nullptr));
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-compute_solve",
        &compute_solve, nullptr));
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-eps_tol",
        &eps_tolerance, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-eps_max_it",
        &eps_max_iterations, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-eps_ncv",
        &eps_ncv, nullptr));
    PetscCall(PetscOptionsGetString(nullptr, nullptr, "-output",
        output_path, sizeof(output_path), nullptr));
    PetscCheck(degree_set && degree >= 2, comm, PETSC_ERR_ARG_WRONG,
        "-p must specify polynomial degree p>=2");
    PetscCheck(elements_set, comm, PETSC_ERR_ARG_WRONG,
        "-n Ex,Ey is required");
    int ex = 0;
    int ey = 0;
    try
    {
        const auto parsed = parse_elements(elements_text);
        ex = parsed.first;
        ey = parsed.second;
    }
    catch (const std::exception &error)
    {
        SETERRQ(comm, PETSC_ERR_ARG_WRONG, "%s", error.what());
    }

    lor::DerivativeGreville2DResult matrices;
    lor::DerivativeGrevilleAssemblyOptions assembly_options;
    PetscCall(lor::assemble_derivative_greville_stiffness_2d(
        comm, degree,
        lor::open_uniform_knots_1d(degree, ex),
        lor::open_uniform_knots_1d(degree, ey),
        length_x, length_y, assembly_options, &matrices));
    derivative_greville_example::AnalysisResult analysis;
    PetscCall(derivative_greville_example::run_analysis(
        matrices.stiffness, matrices.preconditioner,
        compute_spectrum, compute_solve, eps_tolerance,
        eps_max_iterations, eps_ncv, &analysis));

    int rank = 0;
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    if (rank == 0)
    {
        std::ofstream output(output_path);
        PetscCheck(output.good(), comm, PETSC_ERR_FILE_OPEN,
            "Cannot open CSV output %s", output_path);
        output << "dimension,p,Ex,Ey,Lx,Ly,num_dofs_full,num_dofs_free,"
            "nnz_K,nnz_P,nnz_Tx,nnz_Ty,nnz_Dx,nnz_Dy,assembly_K_s,"
            "assembly_P_s,lambda_min,lambda_max,kappa_generalized,"
            "solver_type,preconditioner,outer_iterations,inner_iterations,"
            "true_relative_residual,solve_time_s,l2_error_if_available,status\n";
        output << std::setprecision(17)
            << "2," << degree << ',' << ex << ',' << ey << ','
            << length_x << ',' << length_y << ','
            << matrices.num_full_dofs << ',' << matrices.num_free_dofs << ','
            << matrices.nnz_stiffness << ',' << matrices.nnz_preconditioner << ','
            << matrices.nnz_collocation_x << ',' << matrices.nnz_collocation_y
            << ',' << matrices.nnz_derivative_x << ','
            << matrices.nnz_derivative_y << ','
            << matrices.assembly_stiffness_seconds << ','
            << matrices.assembly_preconditioner_seconds << ',';
        if (analysis.spectrum_computed)
            output << analysis.spectrum.lambda_min << ','
                << analysis.spectrum.lambda_max << ','
                << analysis.spectrum.condition_number << ',';
        else
            output << ",,,";
        output << "cg," << lor::derivative_greville_mass_name << ',';
        if (analysis.solve_computed)
            output << analysis.outer_iterations << ",,"
                << analysis.true_relative_residual << ','
                << analysis.solve_seconds << ",,converged\n";
        else
            output << ",,,,,assembly_only\n";
        PetscCheck(output.good(), comm, PETSC_ERR_FILE_WRITE,
            "Failed while writing CSV output %s", output_path);
    }
    PetscCall(lor::destroy_derivative_greville_result(&matrices));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

int main(int argc, char **argv)
{
    SlepcInitialize(&argc, &argv, nullptr, nullptr);
    const PetscErrorCode error = run();
    SlepcFinalize();
    return error == PETSC_SUCCESS ? 0 : 1;
}
