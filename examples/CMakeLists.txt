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

