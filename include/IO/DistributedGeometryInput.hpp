#ifndef LOR_IO_DISTRIBUTED_GEOMETRY_INPUT_HPP
#define LOR_IO_DISTRIBUTED_GEOMETRY_INPUT_HPP

#include "IO/GeometryInput.hpp"

#include <mpi.h>

#include <string>

namespace lor
{

// Read the minimal geometry file exactly once on root, report any parsing
// failure to every rank, and broadcast the small unrefined GeometryInput.
// filename is used only by root; comm remains owned by MPI/PETSc.
GeometryInput read_distributed_geometry_input(const std::string &filename,
    int root, MPI_Comm comm);

} // namespace lor

#endif
