include(CTest)
enable_testing()

function(add_executable_with_properties executable_name executable_path)
    add_executable(${executable_name} ${executable_path})
    set_property(TARGET ${executable_name} PROPERTY CXX_STANDARD 17)
endfunction()

add_executable_with_properties(MatrixTest ${PROJECT_TEST_DIR}/matrix.cpp)

target_include_directories(MatrixTest PUBLIC 
    ${UTIL_INCLUDE_DIR}
    ${PROJECT_INCLUDE_DIR}
)

add_executable_with_properties(TensorDataTest ${PROJECT_TEST_DIR}/memory/tensorData.cpp)

target_include_directories(TensorDataTest PUBLIC
    ${UTIL_INCLUDE_DIR}
    ${PROJECT_INCLUDE_DIR}
)