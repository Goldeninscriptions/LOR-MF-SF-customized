#ifndef LOR_MF_SF_PETSC_COMPAT_HPP
#define LOR_MF_SF_PETSC_COMPAT_HPP

#include <petscversion.h>

#if PETSC_VERSION_LT(3, 19, 0)
#define LOR_PETSC_NULLPTR PETSC_NULL
#else
#define LOR_PETSC_NULLPTR PETSC_NULLPTR
#endif

#endif
