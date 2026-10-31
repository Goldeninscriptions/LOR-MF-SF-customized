#include "Mesh/DistributedPatch.hpp"

#include <limits>
#include <stdexcept>
#include <vector>

namespace lor
{
namespace
{

// Convert a container size to the int count required by the MPI calls used in
// this prototype, rejecting truncation before communication begins.
int mpi_count(const std::size_t size)
{
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("Distributed Patch data exceeds MPI int count");
    return static_cast<int>(size);
}

// Turn an MPI return code into the same C++ failure on the calling rank.
void check_mpi(const int error, const char *operation)
{
    if (error != MPI_SUCCESS)
        throw std::runtime_error(operation);
}

} // namespace

template <int Dim>
BSplinePatch<Dim> broadcast_patch(const BSplinePatch<Dim> *root_patch,
    const int root, const MPI_Comm comm)
{
    // Validate communicator/root metadata consistently before serializing the
    // refined geometry.
    if (comm == MPI_COMM_NULL)
        throw std::invalid_argument("Patch broadcast requires a communicator");
    int rank = 0;
    int size = 0;
    check_mpi(MPI_Comm_rank(comm, &rank), "Cannot query Patch-broadcast rank");
    check_mpi(MPI_Comm_size(comm, &size), "Cannot query Patch-broadcast size");
    if (root < 0 || root >= size)
        throw std::invalid_argument("Patch-broadcast root is outside communicator");
    if ((rank == root) != (root_patch != nullptr))
        throw std::invalid_argument(
            "Only the Patch-broadcast root must provide root_patch");

    // Broadcast the common degree metadata and each independent knot vector.
    typename BSplinePatch<Dim>::Index degrees{};
    std::array<std::vector<double>, Dim> knots;
    if (rank == root)
        for (int direction = 0; direction < Dim; ++direction)
        {
            degrees[direction] = root_patch->degree(direction);
            knots[direction] = root_patch->knots(direction);
        }
    check_mpi(MPI_Bcast(degrees.data(), Dim, MPI_INT, root, comm),
        "Cannot broadcast Patch degrees");
    for (int direction = 0; direction < Dim; ++direction)
    {
        int knot_count = rank == root ? mpi_count(knots[direction].size()) : 0;
        check_mpi(MPI_Bcast(&knot_count, 1, MPI_INT, root, comm),
            "Cannot broadcast Patch knot count");
        if (rank != root)
            knots[direction].resize(static_cast<std::size_t>(knot_count));
        check_mpi(MPI_Bcast(knots[direction].data(), knot_count,
            MPI_DOUBLE, root, comm), "Cannot broadcast Patch knots");
    }

    // Flatten direction-0-fastest control points so one collective transfers
    // the complete refined control net without padding from std::array.
    int control_point_count = rank == root ?
        mpi_count(root_patch->num_basis()) : 0;
    check_mpi(MPI_Bcast(&control_point_count, 1, MPI_INT, root, comm),
        "Cannot broadcast Patch control-point count");
    std::vector<double> flat_control_points(
        static_cast<std::size_t>(control_point_count) * Dim);
    if (rank == root)
        for (int index = 0; index < control_point_count; ++index)
            for (int direction = 0; direction < Dim; ++direction)
                flat_control_points[static_cast<std::size_t>(index) * Dim +
                    direction] = root_patch->control_point(index)[direction];
    check_mpi(MPI_Bcast(flat_control_points.data(),
        mpi_count(flat_control_points.size()), MPI_DOUBLE, root, comm),
        "Cannot broadcast Patch control points");

    // Restore the strongly typed point sequence and let BSplinePatch validate
    // all tensor sizes and rebuild topology on every rank.
    std::vector<typename BSplinePatch<Dim>::Point> control_points(
        static_cast<std::size_t>(control_point_count));
    for (int index = 0; index < control_point_count; ++index)
        for (int direction = 0; direction < Dim; ++direction)
            control_points[index][direction] =
                flat_control_points[static_cast<std::size_t>(index) * Dim +
                    direction];
    return BSplinePatch<Dim>(degrees, knots, control_points);
}

// Emit both dimensions used by the assembled experiment.
template BSplinePatch<2> broadcast_patch<2>(
    const BSplinePatch<2> *, int, MPI_Comm);
template BSplinePatch<3> broadcast_patch<3>(
    const BSplinePatch<3> *, int, MPI_Comm);

} // namespace lor
