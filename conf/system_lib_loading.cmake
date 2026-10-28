# Machine-local dependency configuration for LOR-MF-SF-customized.
#
# This file describes the current macOS/Apple Silicon workstation.  It should
# contain paths and machine-specific options only; package discovery is kept in
# project_build_setup.cmake so that configuring the project has one entry point.

# -----------------------------------------------------------------------------
# PETSc (required by the first prototype solver)
# -----------------------------------------------------------------------------
set(LOR_PETSC_DIR "/Users/huang/lib/petsc-3.19.2-opt")
set(LOR_PETSC_ARCH ".")

# This PETSc installation was built with Homebrew MPICH.  Using the same
# wrappers avoids compiler/MPI ABI mismatches even though the paper prototype
# is intended to run with exactly one MPI rank.
set(LOR_MPI_C_COMPILER "/opt/homebrew/opt/mpich/bin/mpicc")
set(LOR_MPI_CXX_COMPILER "/opt/homebrew/opt/mpich/bin/mpicxx")
set(LOR_MPI_INCLUDE_DIRS "/opt/homebrew/opt/mpich/include")

# -----------------------------------------------------------------------------
# SLEPc (optional, used by spectral-analysis executables)
# -----------------------------------------------------------------------------
# This installation keeps public headers in the source root and generated
# headers/libraries in a separate architecture directory.
set(LOR_SLEPC_DIR "/Users/huang/lib/slepc-3.19.2-src")
set(LOR_SLEPC_ARCH_DIR
    "/Users/huang/lib/slepc-3.19.2-src/installed-arch-darwin-c-opt")

# -----------------------------------------------------------------------------
# Eigen (optional)
# -----------------------------------------------------------------------------
# PETSc's installation contains Eigen headers.  Core operator code should not
# depend on Eigen unless a later design decision explicitly requires it.
set(LOR_EIGEN3_INCLUDE_DIR
    "/Users/huang/lib/petsc-3.19.2-opt/include/eigen3")

# -----------------------------------------------------------------------------
# OpenMP (optional at configure time)
# -----------------------------------------------------------------------------
# AppleClang needs the Homebrew libomp headers, library, and preprocessing flag.
# Use the stable Homebrew opt symlink rather than a versioned Cellar path.
set(LOR_OPENMP_ROOT "/opt/homebrew/opt/libomp")
set(LOR_OPENMP_CXX_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
set(LOR_OPENMP_CXX_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
set(LOR_OPENMP_CXX_FLAGS "-Xpreprocessor -fopenmp")

# -----------------------------------------------------------------------------
# Build defaults
# -----------------------------------------------------------------------------
set(LOR_VERBOSE_MAKEFILE OFF)

