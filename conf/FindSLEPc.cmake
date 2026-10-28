# Locate a SLEPc installation compatible with the selected PETSc tree.
#
# Result variables:
#   SLEPc_FOUND, SLEPC_DIR, SLEPC_INC, SLEPC_LIB
# Imported target:
#   LOR::SLEPc

if(NOT SLEPC_DIR)
  set(SLEPC_DIR "$ENV{SLEPC_DIR}")
endif()

if(SLEPC_ARCH_DIR)
  set(_slepc_arch_dir "${SLEPC_ARCH_DIR}")
elseif(PETSC_ARCH AND NOT PETSC_ARCH STREQUAL ".")
  set(_slepc_arch_dir "${SLEPC_DIR}/${PETSC_ARCH}")
else()
  set(_slepc_arch_dir "${SLEPC_DIR}")
endif()

find_path(SLEPC_BASE_INCLUDE_DIR slepceps.h
  HINTS "${SLEPC_DIR}/include"
  NO_DEFAULT_PATH)
find_path(SLEPC_ARCH_INCLUDE_DIR slepcconf.h
  HINTS "${_slepc_arch_dir}/include"
  NO_DEFAULT_PATH)
find_library(SLEPC_LIB
  NAMES slepc
  HINTS "${_slepc_arch_dir}/lib"
  NO_DEFAULT_PATH)

set(SLEPC_INC "${SLEPC_BASE_INCLUDE_DIR}")
if(SLEPC_ARCH_INCLUDE_DIR)
  list(APPEND SLEPC_INC "${SLEPC_ARCH_INCLUDE_DIR}")
endif()
list(REMOVE_DUPLICATES SLEPC_INC)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SLEPc
  REQUIRED_VARS SLEPC_DIR SLEPC_BASE_INCLUDE_DIR SLEPC_ARCH_INCLUDE_DIR SLEPC_LIB)

if(SLEPc_FOUND AND NOT TARGET LOR::SLEPc)
  add_library(LOR::SLEPc UNKNOWN IMPORTED)
  set_target_properties(LOR::SLEPc PROPERTIES
    IMPORTED_LOCATION "${SLEPC_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${SLEPC_INC}"
    INTERFACE_LINK_LIBRARIES "LOR::PETSc")
endif()

mark_as_advanced(SLEPC_BASE_INCLUDE_DIR SLEPC_ARCH_INCLUDE_DIR SLEPC_LIB)

