include_guard(GLOBAL)

include(CMakeParseArguments)

set(LOR_CONF_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Load an explicitly selected machine file first, then the repository-local
# default.  This keeps source files portable without editing build logic.
macro(lor_include_machine_config)
  if(NOT LOR_MACHINE_CONFIG_LOADED)
    set(_lor_machine_candidates)

    if(DEFINED SYSTEM_LIB_LOADING_FILE)
      list(APPEND _lor_machine_candidates "${SYSTEM_LIB_LOADING_FILE}")
    endif()
    if(DEFINED ENV{SYSTEM_LIB_LOADING_FILE})
      list(APPEND _lor_machine_candidates "$ENV{SYSTEM_LIB_LOADING_FILE}")
    endif()

    list(APPEND _lor_machine_candidates
      "${LOR_CONF_DIR}/system_lib_loading.cmake")

    foreach(_lor_candidate IN LISTS _lor_machine_candidates)
      if(_lor_candidate AND EXISTS "${_lor_candidate}")
        message(STATUS "Loading machine configuration: ${_lor_candidate}")
        include("${_lor_candidate}")
        set(LOR_MACHINE_CONFIG_LOADED TRUE)
        break()
      endif()
    endforeach()

    if(NOT LOR_MACHINE_CONFIG_LOADED)
      message(FATAL_ERROR
        "No system_lib_loading.cmake was found. Set SYSTEM_LIB_LOADING_FILE "
        "or create ${LOR_CONF_DIR}/system_lib_loading.cmake.")
    endif()
  endif()
endmacro()

# Call before project().  PETSc was built with these wrappers, so using them is
# an ABI compatibility choice rather than enabling distributed execution.
macro(lor_apply_machine_compilers)
  if(DEFINED LOR_MPI_C_COMPILER)
    if(NOT EXISTS "${LOR_MPI_C_COMPILER}")
      message(FATAL_ERROR "C compiler does not exist: ${LOR_MPI_C_COMPILER}")
    endif()
    set(CMAKE_C_COMPILER "${LOR_MPI_C_COMPILER}" CACHE FILEPATH
      "C compiler compatible with PETSc" FORCE)
  endif()

  if(DEFINED LOR_MPI_CXX_COMPILER)
    if(NOT EXISTS "${LOR_MPI_CXX_COMPILER}")
      message(FATAL_ERROR "C++ compiler does not exist: ${LOR_MPI_CXX_COMPILER}")
    endif()
    set(CMAKE_CXX_COMPILER "${LOR_MPI_CXX_COMPILER}" CACHE FILEPATH
      "C++ compiler compatible with PETSc" FORCE)
  endif()
endmacro()

function(lor_set_project_defaults)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  endif()

  set(CMAKE_CXX_STANDARD 17 PARENT_SCOPE)
  set(CMAKE_CXX_STANDARD_REQUIRED ON PARENT_SCOPE)
  set(CMAKE_CXX_EXTENSIONS OFF PARENT_SCOPE)

  if(DEFINED LOR_VERBOSE_MAKEFILE)
    set(CMAKE_VERBOSE_MAKEFILE "${LOR_VERBOSE_MAKEFILE}" PARENT_SCOPE)
  endif()
endfunction()

# Discover only dependencies requested by a target-level project configuration.
# The function exports availability booleans and creates imported targets:
#   LOR::PETSc, LOR::SLEPc, OpenMP::OpenMP_CXX, Eigen3::Eigen (when requested).
function(lor_setup_dependencies)
  set(options REQUIRE_PETSC USE_SLEPC USE_OPENMP USE_EIGEN)
  cmake_parse_arguments(LOR_DEP "${options}" "" "" ${ARGN})

  list(PREPEND CMAKE_MODULE_PATH "${LOR_CONF_DIR}")
  set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)

  if(LOR_DEP_REQUIRE_PETSC OR LOR_DEP_USE_SLEPC)
    set(PETSC_DIR "${LOR_PETSC_DIR}")
    set(PETSC_ARCH "${LOR_PETSC_ARCH}")
    find_package(PETSc REQUIRED)
    set(LOR_HAS_PETSC TRUE PARENT_SCOPE)
  endif()

  if(LOR_DEP_USE_SLEPC)
    if(NOT DEFINED LOR_SLEPC_DIR)
      message(FATAL_ERROR
        "SLEPc was requested, but LOR_SLEPC_DIR is not defined in the machine configuration.")
    endif()
    set(SLEPC_DIR "${LOR_SLEPC_DIR}")
    if(DEFINED LOR_SLEPC_ARCH_DIR)
      set(SLEPC_ARCH_DIR "${LOR_SLEPC_ARCH_DIR}")
    endif()
    find_package(SLEPc REQUIRED)
    set(LOR_HAS_SLEPC TRUE PARENT_SCOPE)
  endif()

  if(LOR_DEP_USE_EIGEN)
    if(DEFINED LOR_EIGEN3_INCLUDE_DIR AND
       EXISTS "${LOR_EIGEN3_INCLUDE_DIR}/Eigen/Core")
      if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${LOR_EIGEN3_INCLUDE_DIR}")
      endif()
    else()
      find_package(Eigen3 3.3 REQUIRED NO_MODULE)
    endif()
    set(LOR_HAS_EIGEN TRUE PARENT_SCOPE)
  endif()

  if(LOR_DEP_USE_OPENMP)
    if(DEFINED LOR_OPENMP_ROOT)
      set(OpenMP_ROOT "${LOR_OPENMP_ROOT}")
    endif()
    if(DEFINED LOR_OPENMP_CXX_FLAGS)
      set(OpenMP_CXX_FLAGS "${LOR_OPENMP_CXX_FLAGS}")
    endif()
    if(DEFINED LOR_OPENMP_CXX_INCLUDE_DIR)
      set(OpenMP_CXX_INCLUDE_DIR "${LOR_OPENMP_CXX_INCLUDE_DIR}")
    endif()
    if(DEFINED LOR_OPENMP_CXX_LIBRARY)
      set(OpenMP_libomp_LIBRARY "${LOR_OPENMP_CXX_LIBRARY}")
    endif()

    find_package(OpenMP REQUIRED COMPONENTS CXX)
    set(LOR_HAS_OPENMP TRUE PARENT_SCOPE)
  endif()
endfunction()

function(lor_print_build_summary)
  message(STATUS "================ LOR-MF-SF build summary ================")
  message(STATUS "System:          ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
  message(STATUS "Build type:      ${CMAKE_BUILD_TYPE}")
  message(STATUS "C compiler:      ${CMAKE_C_COMPILER}")
  message(STATUS "C++ compiler:    ${CMAKE_CXX_COMPILER}")
  message(STATUS "C++ standard:    ${CMAKE_CXX_STANDARD}")
  message(STATUS "PETSc root:      ${LOR_PETSC_DIR}")
  message(STATUS "SLEPc root:      ${LOR_SLEPC_DIR}")
  message(STATUS "SLEPc arch:      ${LOR_SLEPC_ARCH_DIR}")
  message(STATUS "OpenMP enabled:  ${LOR_HAS_OPENMP}")
  message(STATUS "==========================================================")
endfunction()

