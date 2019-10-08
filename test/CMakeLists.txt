include(FetchContent)
include(GoogleTest)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY  https://github.com/google/googletest.git
    GIT_TAG         release-1.10.0
)

FetchContent_MakeAvailable(googletest)

function(add_gtest_target target_name)
    set_property(TARGET ${target_name} PROPERTY CXX_STANDARD 17)
    target_link_libraries(${target_name} gtest)
endfunction()

function(add_gtest_executable executable_path)
    add_executable(testMain ${executable_path})
    add_gtest_target(testMain)
endfunction()

function(add_gtest test_name test_path)
    add_library(${test_name} ${test_path})
    add_gtest_target(${test_name})

    target_include_directories(${test_name} PUBLIC 
        ${UTIL_INCLUDE_DIR}
        ${PROJECT_INCLUDE_DIR}
    )

    target_link_libraries(testMain ${test_name})
endfunction()

add_gtest_executable(${PROJECT_TEST_DIR}/main.cpp)

add_gtest(MatrixTest ${PROJECT_TEST_DIR}/matrix.cpp)

add_gtest(TensorDataTest ${PROJECT_TEST_DIR}/memory/tensorData.cpp)

gtest_discover_tests(testMain)