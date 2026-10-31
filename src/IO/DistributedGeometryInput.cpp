#include "IO/DistributedGeometryInput.hpp"

#include <limits>
#include <stdexcept>
#include <vector>

namespace lor
{
namespace
{

// Check the int-sized count boundary used by this prototype's MPI collectives.
int mpi_count(const std::size_t size)
{
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("Geometry input exceeds MPI int count");
    return static_cast<int>(size);
}

// Raise a local C++ exception when an MPI collective fails.
void check_mpi(const int error, const char *operation)
{
    if (error != MPI_SUCCESS)
        throw std::runtime_error(operation);
}

// Broadcast a root-generated error string before any rank attempts to consume
// incomplete geometry data.
void broadcast_root_error(std::string &message, const int root,
    const int rank, const MPI_Comm comm)
{
    int length = rank == root ? mpi_count(message.size()) : 0;
    check_mpi(MPI_Bcast(&length, 1, MPI_INT, root, comm),
        "Cannot broadcast geometry-read status");
    if (rank != root)
        message.resize(static_cast<std::size_t>(length));
    check_mpi(MPI_Bcast(message.data(), length, MPI_CHAR, root, comm),
        "Cannot broadcast geometry-read error");
}

} // namespace

GeometryInput read_distributed_geometry_input(const std::string &filename,
    const int root, const MPI_Comm comm)
{
    // Validate the collective context before the root performs filesystem IO.
    if (comm == MPI_COMM_NULL)
        throw std::invalid_argument(
            "Distributed geometry input requires a communicator");
    int rank = 0;
    int size = 0;
    check_mpi(MPI_Comm_rank(comm, &rank),
        "Cannot query distributed-geometry rank");
    check_mpi(MPI_Comm_size(comm, &size),
        "Cannot query distributed-geometry size");
    if (root < 0 || root >= size)
        throw std::invalid_argument(
            "Distributed-geometry root is outside communicator");

    // Parse on root only, then turn any parser exception into a collective
    // error so non-root ranks never wait at a later unmatched broadcast.
    GeometryInput input;
    std::string root_error;
    if (rank == root)
        try
        {
            input = read_geometry_input(filename);
        }
        catch (const std::exception &exception)
        {
            root_error = exception.what();
        }
    broadcast_root_error(root_error, root, rank, comm);
    if (!root_error.empty())
        throw std::runtime_error(root_error);

    // Broadcast the fixed dimension and one common degree per parameter
    // direction before allocating variable-sized arrays.
    check_mpi(MPI_Bcast(&input.dimension, 1, MPI_INT, root, comm),
        "Cannot broadcast geometry dimension");
    if (rank != root)
        input.degrees.resize(static_cast<std::size_t>(input.dimension));
    check_mpi(MPI_Bcast(input.degrees.data(), input.dimension,
        MPI_INT, root, comm), "Cannot broadcast geometry degrees");

    // Broadcast every direction's independent knot vector.
    if (rank != root)
        input.knot_vectors.resize(static_cast<std::size_t>(input.dimension));
    for (int direction = 0; direction < input.dimension; ++direction)
    {
        int count = rank == root ?
            mpi_count(input.knot_vectors[direction].size()) : 0;
        check_mpi(MPI_Bcast(&count, 1, MPI_INT, root, comm),
            "Cannot broadcast geometry knot count");
        if (rank != root)
            input.knot_vectors[direction].resize(
                static_cast<std::size_t>(count));
        check_mpi(MPI_Bcast(input.knot_vectors[direction].data(), count,
            MPI_DOUBLE, root, comm), "Cannot broadcast geometry knots");
    }

    // Flatten the small initial control net for one collective and restore its
    // dynamic point representation on non-root ranks.
    int point_count = rank == root ?
        mpi_count(input.control_points.size()) : 0;
    check_mpi(MPI_Bcast(&point_count, 1, MPI_INT, root, comm),
        "Cannot broadcast geometry control-point count");
    std::vector<double> flat_points(
        static_cast<std::size_t>(point_count) * input.dimension);
    if (rank == root)
        for (int point = 0; point < point_count; ++point)
            for (int direction = 0; direction < input.dimension; ++direction)
                flat_points[static_cast<std::size_t>(point) * input.dimension +
                    direction] = input.control_points[point][direction];
    check_mpi(MPI_Bcast(flat_points.data(), mpi_count(flat_points.size()),
        MPI_DOUBLE, root, comm),
        "Cannot broadcast geometry control points");
    if (rank != root)
        input.control_points.assign(static_cast<std::size_t>(point_count),
            std::vector<double>(static_cast<std::size_t>(input.dimension)));
    for (int point = 0; point < point_count; ++point)
        for (int direction = 0; direction < input.dimension; ++direction)
            input.control_points[point][direction] =
                flat_points[static_cast<std::size_t>(point) * input.dimension +
                    direction];
    return input;
}

} // namespace lor
