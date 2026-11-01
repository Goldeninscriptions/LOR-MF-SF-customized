#include "IO/HardwareInfo.hpp"

#include <thread>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sched.h>
#endif

#include <unistd.h>

namespace lor
{
namespace
{

#if defined(__APPLE__)
// Query the logical CPU count exposed by macOS, including virtual cores.
int macos_logical_processors()
{
    int count = 0;
    std::size_t size = sizeof(count);
    if (sysctlbyname("hw.logicalcpu", &count, &size, nullptr, 0) == 0 &&
        count > 0)
        return count;
    return 0;
}
#elif defined(__linux__)
// Count CPUs in the current Linux affinity mask so containers, cpusets, and
// scheduler bindings reduce the rank limit to resources this process may use.
int linux_affinity_logical_processors()
{
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) != 0)
        return 0;

    int count = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &affinity))
            ++count;
    return count;
}
#endif

} // namespace

HardwareConcurrency detect_hardware_concurrency()
{
    // Prefer the strongest platform-specific query available on each
    // supported operating system.
#if defined(__APPLE__)
    if (const int count = macos_logical_processors(); count > 0)
        return {count, "macOS hw.logicalcpu"};
#elif defined(__linux__)
    if (const int count = linux_affinity_logical_processors(); count > 0)
        return {count, "Linux CPU affinity"};
#endif

    // POSIX online processors provide a common fallback when the preferred
    // platform query is unavailable or fails.
    const long online = sysconf(_SC_NPROCESSORS_ONLN);
    if (online > 0)
        return {static_cast<int>(online), "POSIX online logical processors"};

    // The standard-library hint is the final portable fallback; zero means
    // unknown, in which case a safe serial limit is used.
    const unsigned int concurrent = std::thread::hardware_concurrency();
    if (concurrent > 0)
        return {static_cast<int>(concurrent),
            "C++ hardware_concurrency"};
    return {1, "serial fallback"};
}

int recommend_mpi_ranks(const std::size_t free_dofs, const int max_ranks,
    const std::size_t target_dofs)
{
    // Reject invalid policy inputs rather than silently producing a rank
    // count that no launcher can use.
    if (max_ranks < 1 || target_dofs < 1)
        return 1;

    // Ceiling division implements one rank for each target_dofs unknowns;
    // the subtraction form avoids overflow near size_t's upper limit.
    const std::size_t dof_limited = free_dofs == 0 ? 1 :
        1 + (free_dofs - 1) / target_dofs;
    if (dof_limited >= static_cast<std::size_t>(max_ranks))
        return max_ranks;
    return static_cast<int>(dof_limited);
}

} // namespace lor
