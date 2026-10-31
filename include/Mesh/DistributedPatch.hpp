#ifndef LOR_MESH_DISTRIBUTED_PATCH_HPP
#define LOR_MESH_DISTRIBUTED_PATCH_HPP

#include "Mesh/BSplinePatch.hpp"

#include <mpi.h>

namespace lor
{

// Broadcast one already refined Patch from root and reconstruct an identical
// immutable full Patch on every rank. Only root may supply root_patch; the
// communicator is borrowed and remains owned by MPI/PETSc.
template <int Dim>
BSplinePatch<Dim> broadcast_patch(const BSplinePatch<Dim> *root_patch,
    int root, MPI_Comm comm);

extern template BSplinePatch<2> broadcast_patch<2>(
    const BSplinePatch<2> *, int, MPI_Comm);
extern template BSplinePatch<3> broadcast_patch<3>(
    const BSplinePatch<3> *, int, MPI_Comm);

} // namespace lor

#endif
