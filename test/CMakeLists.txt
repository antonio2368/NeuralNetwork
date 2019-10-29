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

    set_property(TARGET testMain PROPERTY CXX_STANDARD 17)
    target_link_libraries(testMain PUBLIC gtest)

    target_compile_options(testMain PRIVATE -Werror -Wall -Wextra)

    target_include_directories(testMain PUBLIC
        ${UTIL_INCLUDE_DIR}
        ${PROJECT_INCLUDE_DIR}
    )
endfunction()

function(add_gtest test_path)
    target_sources(testMain PUBLIC ${test_path})
endfunction()

add_gtest_executable(${PROJECT_TEST_DIR}/main.cpp)

add_gtest(${PROJECT_TEST_DIR}/memory/tensorContainer.cpp)
add_gtest(${PROJECT_TEST_DIR}/tensor.cpp)
add_gtest(${PROJECT_TEST_DIR}/initializers/initializers.cpp)
add_gtest(${PROJECT_TEST_DIR}/layers.cpp)

gtest_discover_tests(testMain)