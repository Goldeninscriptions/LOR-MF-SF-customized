# Locate a PETSc installation from PETSC_DIR and PETSC_ARCH.
#
# Result variables:
#   PETSc_FOUND, PETSC_DIR, PETSC_ARCH, PETSC_INC, PETSC_LIB
# Imported target:
#   LOR::PETSc

if(NOT PETSC_DIR)
  set(PETSC_DIR "$ENV{PETSC_DIR}")
endif()
if(NOT DEFINED PETSC_ARCH)
  set(PETSC_ARCH "$ENV{PETSC_ARCH}")
endif()

if(PETSC_ARCH AND NOT PETSC_ARCH STREQUAL ".")
  set(_petsc_arch_dir "${PETSC_DIR}/${PETSC_ARCH}")
else()
  set(_petsc_arch_dir "${PETSC_DIR}")
endif()

find_path(PETSC_BASE_INCLUDE_DIR petscksp.h
  HINTS "${PETSC_DIR}/include"
  NO_DEFAULT_PATH)
find_path(PETSC_ARCH_INCLUDE_DIR petscconf.h
  HINTS "${_petsc_arch_dir}/include"
  NO_DEFAULT_PATH)
find_library(PETSC_LIB
  NAMES petsc
  HINTS "${_petsc_arch_dir}/lib"
  NO_DEFAULT_PATH)

set(PETSC_INC "${PETSC_BASE_INCLUDE_DIR}")
if(PETSC_ARCH_INCLUDE_DIR)
  list(APPEND PETSC_INC "${PETSC_ARCH_INCLUDE_DIR}")
endif()
list(REMOVE_DUPLICATES PETSC_INC)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PETSc
  REQUIRED_VARS PETSC_DIR PETSC_BASE_INCLUDE_DIR PETSC_ARCH_INCLUDE_DIR PETSC_LIB)

if(PETSc_FOUND AND NOT TARGET LOR::PETSc)
  add_library(LOR::PETSc UNKNOWN IMPORTED)
  set_target_properties(LOR::PETSc PROPERTIES
    IMPORTED_LOCATION "${PETSC_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INC}")
endif()

mark_as_advanced(PETSC_BASE_INCLUDE_DIR PETSC_ARCH_INCLUDE_DIR PETSC_LIB)

