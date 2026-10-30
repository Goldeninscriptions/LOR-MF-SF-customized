cmake_minimum_required(VERSION 3.18)

include(${CMAKE_CURRENT_SOURCE_DIR}/conf/project_build_setup.cmake)
lor_include_machine_config()
lor_apply_machine_compilers()

project(LOR_MF_SF LANGUAGES C CXX)

lor_set_project_defaults()

option(LOR_ENABLE_OPENMP "Enable OpenMP support" ON)
option(LOR_BUILD_EXAMPLES "Build experiment drivers" ON)

set(_lor_dependencies REQUIRE_PETSC USE_SLEPC)
if(LOR_ENABLE_OPENMP)
  list(APPEND _lor_dependencies USE_OPENMP)
endif()
lor_setup_dependencies(${_lor_dependencies})

add_library(lor_core
  src/Mesh/BSplinePatch.cpp
  src/Mesh/ID.cpp
  src/Mesh/IEN.cpp
  src/Mesh/LORMesh.cpp
  src/Mesh/LORPointSet.cpp
  src/Mesh/PRefinement.cpp
  src/Mesh/HRefinement.cpp
  src/Mesh/KRefinement.cpp
  src/IO/GeometryInput.cpp
  src/IO/PatchInput.cpp
  src/Element/BSplineBasis1D.cpp
  src/Element/FEMHex8.cpp
  src/Element/FEMQ1Basis.cpp
  src/Element/FEMQuad4.cpp
  src/Element/GaussQuadrature.cpp
  src/Element/BSplineElement.cpp
  src/Assembly/LexicographicOrdering.cpp
  src/Assembly/DirichletBC.cpp
  src/Assembly/LocalAssembly.cpp
)

target_include_directories(lor_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(lor_core PUBLIC cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(lor_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(LOR_ENABLE_OPENMP)
  target_link_libraries(lor_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(lor_core PUBLIC LOR_USE_OPENMP=1)
endif()

add_library(lor_petsc
  src/Assembly/GlobalAssembly.cpp
)
target_include_directories(lor_petsc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(lor_petsc PUBLIC cxx_std_17)
target_link_libraries(lor_petsc PUBLIC lor_core LOR::PETSc)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(lor_petsc PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(lor_solver
  src/IO/ExperimentIO.cpp
  src/Solver/LinearSolver.cpp
  src/Solver/ConditionNumber.cpp
  src/Solver/ErrorAnalysis.cpp
)
target_include_directories(lor_solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(lor_solver PUBLIC cxx_std_17)
target_link_libraries(lor_solver PUBLIC lor_petsc LOR::SLEPc)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(lor_solver PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(LOR_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

lor_print_build_summary()

