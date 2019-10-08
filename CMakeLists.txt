cmake_minimum_required(VERSION 3.11.0)
project(Math VERSION 0.1.0)

set(PROJECT_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test")
set(PROJECT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(UTIL_INCLUDE_DIR "/home/antonio/projects/util" CACHE STRING "path to the util library")

include(CTest)
enable_testing()

add_subdirectory("${PROJECT_TEST_DIR}")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

unset(UTIL_INCLUDE_DIR CACHE)