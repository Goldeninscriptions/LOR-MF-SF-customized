#ifndef LOR_ASSEMBLY_DERIVATIVE_GREVILLE_STIFFNESS_HPP
#define LOR_ASSEMBLY_DERIVATIVE_GREVILLE_STIFFNESS_HPP

#include <petscmat.h>

#include <cstddef>
#include <vector>

namespace lor
{

inline constexpr const char *derivative_greville_mass_name =
    "derivative_greville_mass";

struct DerivativeGrevilleAssemblyOptions
{
    bool assemble_reference_stiffness = true;
    bool assemble_exact_derivative_masses = true;
    bool retain_intermediates = true;
};

struct DerivativeGreville1DResult
{
    int degree = 0;
    int elements = 0;
    int num_full_dofs = 0;
    int num_free_dofs = 0;
    int num_derivative_dofs = 0;
    std::vector<double> knots;
    std::vector<double> derivative_knots;
    std::vector<double> original_greville_points;
    std::vector<double> derivative_greville_points;

    Mat derivative = nullptr;            // D_p
    Mat restriction = nullptr;           // R
    Mat derivative_free = nullptr;       // D_p R
    Mat collocation = nullptr;            // T_{p-1}
    Mat greville_mass = nullptr;          // P1 consistent M_g
    Mat exact_derivative_mass = nullptr;  // exact B-spline M_{p-1}
    Mat sampled_derivative = nullptr;     // B=T D R
    Mat preconditioner = nullptr;         // B^T M_g B
    Mat stiffness = nullptr;              // independently integrated K_s

    PetscLogDouble assembly_stiffness_seconds = 0.0;
    PetscLogDouble assembly_preconditioner_seconds = 0.0;
    PetscInt nnz_stiffness = 0;
    PetscInt nnz_preconditioner = 0;
    PetscInt nnz_derivative = 0;
    PetscInt nnz_collocation = 0;
};

struct DerivativeGreville2DResult
{
    int degree = 0;
    int elements_x = 0;
    int elements_y = 0;
    int num_basis_x = 0;
    int num_basis_y = 0;
    int num_full_dofs = 0;
    int num_free_dofs = 0;
    int num_x_derivative_dofs = 0;
    int num_y_derivative_dofs = 0;
    double length_x = 1.0;
    double length_y = 1.0;
    double x_derivative_weight = 1.0;
    double y_derivative_weight = 1.0;
    std::vector<double> knots_x;
    std::vector<double> knots_y;
    std::vector<double> derivative_knots_x;
    std::vector<double> derivative_knots_y;
    std::vector<double> original_greville_x;
    std::vector<double> original_greville_y;
    std::vector<double> derivative_greville_x;
    std::vector<double> derivative_greville_y;

    Mat derivative_x = nullptr;           // D_x
    Mat derivative_y = nullptr;           // D_y
    Mat restriction = nullptr;            // original-space R
    Mat derivative_x_free = nullptr;      // D_x R
    Mat derivative_y_free = nullptr;      // D_y R
    Mat collocation_x = nullptr;           // mixed (p-1,p) T_x
    Mat collocation_y = nullptr;           // mixed (p,p-1) T_y
    Mat greville_mass_x = nullptr;         // mixed-grid Q1 mass
    Mat greville_mass_y = nullptr;
    Mat exact_derivative_mass_x = nullptr; // exact mixed spline mass
    Mat exact_derivative_mass_y = nullptr;
    Mat sampled_derivative_x = nullptr;    // B_x
    Mat sampled_derivative_y = nullptr;    // B_y
    Mat preconditioner_x = nullptr;
    Mat preconditioner_y = nullptr;
    Mat preconditioner = nullptr;
    Mat stiffness = nullptr;               // independent high-order K_s

    PetscLogDouble assembly_stiffness_seconds = 0.0;
    PetscLogDouble assembly_preconditioner_seconds = 0.0;
    PetscInt nnz_stiffness = 0;
    PetscInt nnz_preconditioner = 0;
    PetscInt nnz_derivative_x = 0;
    PetscInt nnz_derivative_y = 0;
    PetscInt nnz_collocation_x = 0;
    PetscInt nnz_collocation_y = 0;
};

// Public helpers encode the specification's degree and x-fast conventions.
std::vector<double> derivative_space_knots(
    int degree, const std::vector<double> &original_knots);
std::vector<double> strict_greville_points(
    int degree, const std::vector<double> &knots);
int derivative_greville_id_2d(int i, int j, int extent_x);

PetscErrorCode assemble_derivative_greville_stiffness_1d(
    MPI_Comm comm, int degree, const std::vector<double> &knots,
    const DerivativeGrevilleAssemblyOptions &options,
    DerivativeGreville1DResult *result);

PetscErrorCode assemble_derivative_greville_stiffness_2d(
    MPI_Comm comm, int degree,
    const std::vector<double> &knots_x,
    const std::vector<double> &knots_y,
    double length_x, double length_y,
    const DerivativeGrevilleAssemblyOptions &options,
    DerivativeGreville2DResult *result);

PetscErrorCode destroy_derivative_greville_result(
    DerivativeGreville1DResult *result);
PetscErrorCode destroy_derivative_greville_result(
    DerivativeGreville2DResult *result);

} // namespace lor

#endif
