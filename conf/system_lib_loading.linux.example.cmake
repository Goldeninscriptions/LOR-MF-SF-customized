# Example Linux machine configuration for LOR-MF-SF-customized.
# Copy the relevant values into system_lib_loading.cmake on the target machine,
# or pass this file with -DSYSTEM_LIB_LOADING_FILE=/absolute/path/to/file.

set(LOR_PETSC_DIR "/home/xuanming/lib/petsc-3.24.0-opt")
set(LOR_PETSC_ARCH ".")
set(LOR_SLEPC_DIR "/home/xuanming/lib/slepc-3.24.0")
set(LOR_EIGEN3_INCLUDE_DIR "/home/xuanming/lib/eigen-3.4.1")

set(LOR_MPI_C_COMPILER "/home/xuanming/lib/mpich-4.3.2/bin/mpicc")
set(LOR_MPI_CXX_COMPILER "/home/xuanming/lib/mpich-4.3.2/bin/mpicxx")
set(LOR_MPI_INCLUDE_DIRS "/home/xuanming/lib/mpich-4.3.2/include")

# GCC/Clang normally lets CMake discover OpenMP without explicit paths.  Define
# LOR_OPENMP_CXX_INCLUDE_DIR or LOR_OPENMP_CXX_LIBRARY here only when needed.

set(LOR_VERBOSE_MAKEFILE OFF)

