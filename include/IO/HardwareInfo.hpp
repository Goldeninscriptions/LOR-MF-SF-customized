#ifndef LOR_IO_HARDWARE_INFO_HPP
#define LOR_IO_HARDWARE_INFO_HPP

#include <cstddef>
#include <string>

namespace lor
{

// Host CPU capacity used by the serial case planner. One logical processor is
// treated as one possible MPI rank because the requested policy explicitly
// includes simultaneous-multithreading (virtual) cores.
struct HardwareConcurrency
{
    // Number of online/available logical processors reported by the platform.
    int logical_processors = 1;

    // Human-readable operating-system mechanism that supplied the count.
    std::string source;
};

// Detect logical CPU capacity on macOS or Linux without starting MPI. Linux
// honors the calling process's CPU affinity; macOS uses hw.logicalcpu.
HardwareConcurrency detect_hardware_concurrency();

// Select one rank per target_dofs free algebraic unknowns, rounded upward and
// capped by max_ranks. The result is always in [1,max_ranks].
int recommend_mpi_ranks(std::size_t free_dofs, int max_ranks,
    std::size_t target_dofs = 10000);

} // namespace lor

#endif
