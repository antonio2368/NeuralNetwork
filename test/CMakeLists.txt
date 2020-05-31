include(FetchContent)
include(GoogleTest)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY  https://github.com/google/googletest.git
    GIT_TAG         release-1.10.0
)

FetchContent_MakeAvailable(googletest)

function(add_gtest_executable executable_path)
    add_executable(testMain ${executable_path})

    target_link_libraries(testMain PRIVATE NeuralNetwork gtest range-v3)
    target_compile_options(testMain PRIVATE -Werror -Wall -Wextra)
    set_target_properties(testMain
                      PROPERTIES
                      CXX_STANDARD 20
                      CXX_STANDARD_REQUIRED YES
                      CXX_EXTENSIONS NO)
endfunction()

function(add_gtest test_path)
    target_sources(testMain PUBLIC ${test_path})
endfunction()

add_gtest_executable(${PROJECT_TEST_DIR}/main.cpp)

add_gtest(${PROJECT_TEST_DIR}/tensor.cpp)
add_gtest(${PROJECT_TEST_DIR}/initializers/initializers.cpp)
add_gtest(${PROJECT_TEST_DIR}/layers.cpp)
add_gtest(${PROJECT_TEST_DIR}/operations.cpp)
# add_gtest(${PROJECT_TEST_DIR}/utils.cpp)
add_gtest(${PROJECT_TEST_DIR}/shape.cpp)

gtest_discover_tests(testMain)