cmake_minimum_required(VERSION 3.16.0)
project(NeuralNetwork VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

set(PROJECT_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test")
set(PROJECT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

include(CTest)
enable_testing()

FetchContent_Declare(
    ranges
    GIT_REPOSITORY  https://github.com/ericniebler/range-v3.git
    GIT_TAG         0.10.0
)

FetchContent_MakeAvailable(ranges)

add_library(NeuralNetwork INTERFACE)
target_include_directories(NeuralNetwork INTERFACE
    ${PROJECT_INCLUDE_DIR}
)

add_subdirectory(${PROJECT_TEST_DIR})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)