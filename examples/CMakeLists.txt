cmake_minimum_required(VERSION 3.18)

# Support the requested standalone workflow:
#   mkdir -p build && cd build && cmake ../examples && make
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  include(${CMAKE_CURRENT_LIST_DIR}/../conf/project_build_setup.cmake)
  lor_include_machine_config()
  lor_apply_machine_compilers()
  project(LOR_MF_SF_examples LANGUAGES C CXX)
  set(LOR_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/..
                   ${CMAKE_BINARY_DIR}/lor_project EXCLUDE_FROM_ALL)
endif()

add_executable(assembled_driver assembled_driver.cpp)
target_link_libraries(assembled_driver PRIVATE lor_solver)

add_executable(case_planner case_planner.cpp)
target_link_libraries(case_planner PRIVATE lor_core)

add_executable(truncated_cholesky_1d_test truncated_cholesky_1d_test.cpp)
target_link_libraries(truncated_cholesky_1d_test PRIVATE lor_core)

add_executable(greville_mass_preconditioner_test
  greville_mass_preconditioner_test.cpp)
target_link_libraries(greville_mass_preconditioner_test PRIVATE lor_petsc)

add_executable(greville_stiffness_preconditioner_test
  greville_stiffness_preconditioner_test.cpp)
target_link_libraries(greville_stiffness_preconditioner_test PRIVATE lor_petsc)

add_executable(greville_mass_pc_application_test
  greville_mass_pc_application_test.cpp)
target_link_libraries(greville_mass_pc_application_test PRIVATE lor_petsc)

add_executable(distributed_greville_mass_benchmark
  distributed_greville_mass_benchmark.cpp)
target_link_libraries(distributed_greville_mass_benchmark PRIVATE lor_petsc)

add_executable(greville_tensor_condition_driver
  greville_tensor_condition_driver.cpp)
target_link_libraries(greville_tensor_condition_driver PRIVATE lor_petsc)

enable_testing()
add_test(NAME truncated_cholesky_1d COMMAND truncated_cholesky_1d_test)
add_test(NAME greville_mass_preconditioner
  COMMAND greville_mass_preconditioner_test)
add_test(NAME greville_stiffness_preconditioner
  COMMAND greville_stiffness_preconditioner_test)
add_test(NAME greville_mass_pc_application
  COMMAND greville_mass_pc_application_test)
find_program(LOR_MPIEXEC_EXECUTABLE NAMES mpirun mpiexec REQUIRED)
add_test(NAME distributed_greville_tensor_transpose
  COMMAND ${LOR_MPIEXEC_EXECUTABLE} -n 2
          $<TARGET_FILE:distributed_greville_mass_benchmark>
          -p 3 -n 16 -mode tensor_transpose -rtol 1e-10 -max_it 200)
add_test(NAME distributed_greville_band
  COMMAND ${LOR_MPIEXEC_EXECUTABLE} -n 2
          $<TARGET_FILE:distributed_greville_mass_benchmark>
          -p 3 -n 16 -mode distributed_band -rtol 1e-10 -max_it 200)
set_tests_properties(
  distributed_greville_tensor_transpose distributed_greville_band
  PROPERTIES TIMEOUT 60)
add_test(NAME greville_tensor_mass_condition
  COMMAND ${LOR_MPIEXEC_EXECUTABLE} -n 2
          $<TARGET_FILE:greville_tensor_condition_driver>
          -operator_type mass -spectral_target preconditioned
          -p 3 -n 4 -eps_tol 1e-10
          -eps_max_it 1000 -eps_ncv 20 -eigmin_eps_stability_stop false
          -output ${CMAKE_CURRENT_BINARY_DIR}/greville_mass_condition.log)
add_test(NAME greville_tensor_stiffness_condition
  COMMAND ${LOR_MPIEXEC_EXECUTABLE} -n 2
          $<TARGET_FILE:greville_tensor_condition_driver>
          -operator_type stiffness -spectral_target preconditioned
          -p 3 -n 4 -eps_tol 1e-10
          -eps_max_it 1000 -eps_ncv 20 -eigmin_eps_stability_stop false
          -output ${CMAKE_CURRENT_BINARY_DIR}/greville_stiffness_condition.log)
add_test(NAME greville_tensor_mass_preconditioner_condition
  COMMAND $<TARGET_FILE:greville_tensor_condition_driver>
          -operator_type mass -spectral_target preconditioner
          -p 3 -n 4 -eps_tol 1e-10 -eps_max_it 1000 -eps_ncv 20
          -output ${CMAKE_CURRENT_BINARY_DIR}/greville_mass_P_condition.log)
add_test(NAME greville_tensor_stiffness_preconditioner_condition
  COMMAND ${LOR_MPIEXEC_EXECUTABLE} -n 2
          $<TARGET_FILE:greville_tensor_condition_driver>
          -operator_type stiffness -spectral_target preconditioner
          -p 3 -n 4 -eps_tol 1e-10 -eps_max_it 1000 -eps_ncv 20
          -eigmin_st_type sinvert -eigmin_eps_target 0
          -eigmin_eps_target_magnitude -eigmin_eps_stability_stop false
          -output ${CMAKE_CURRENT_BINARY_DIR}/greville_stiffness_P_condition.log)
set_tests_properties(
  greville_tensor_mass_condition greville_tensor_stiffness_condition
  greville_tensor_mass_preconditioner_condition
  greville_tensor_stiffness_preconditioner_condition
  PROPERTIES TIMEOUT 60)
add_test(NAME greville_tensor_eps_integration
  COMMAND ${LOR_MPIEXEC_EXECUTABLE} -n 2 $<TARGET_FILE:assembled_driver>
          -geometry ${CMAKE_CURRENT_LIST_DIR}/../work/input/square_2d.txt
          -refinement k_refine -p 3 -n 4,4 -lor_points greville
          -lor_pc left_cg_lu -preconditioner_side left
          -preconditioner_matrix_source greville_mass_tensor
          -compute_solution false -compute_unpreconditioned_solution false
          -compute_l2_error false -compute_condition_number false
          -compute_spectral_condition_number true
          -spectral_condition_target pinv_k -eps_tol 1e-10
          -eps_max_it 1000 -eps_ncv 20
          -output ${CMAKE_CURRENT_BINARY_DIR}/greville_tensor_eps_test.log)
set_tests_properties(greville_tensor_eps_integration PROPERTIES TIMEOUT 60)
add_test(NAME truncated_tensor_integration
  COMMAND bash ${CMAKE_CURRENT_LIST_DIR}/truncated_tensor_integration_test.sh
          $<TARGET_FILE:assembled_driver> ${LOR_MPIEXEC_EXECUTABLE}
          ${CMAKE_CURRENT_LIST_DIR}/..)
set_tests_properties(truncated_tensor_integration PROPERTIES TIMEOUT 120)
