cmake_minimum_required(VERSION 3.0.0)
project(Math VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(MatrixTest src/matrixTest.cpp)

set_property(TARGET MatrixTest PROPERTY CXX_STANDARD 17 )

target_include_directories(MatrixTest PUBLIC "/home/antonio/projects/util")
target_include_directories(MatrixTest PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
