# rm -rf * && cmake .. && make -j32
##################################################
cmake_minimum_required(VERSION 3.0)
project(HNNS)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)

set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
message(${INCLUDE_DIR})

include_directories(
    ${INCLUDE_DIR}
)

file(GLOB GRAPH_LIST ${INCLUDE_DIR}/graph/*.hpp)
file(GLOB FLAT_LIST ${INCLUDE_DIR}/flat/*.hpp)
file(GLOB UTILS_LIST ${INCLUDE_DIR}/utils/*.hpp)

foreach(graph_file ${GRAPH_LIST})
message("Graph file: ${graph_file}")
endforeach()

foreach(flat_file ${FLAT_LIST})
message("Flat file: ${flat_file}")
endforeach()

foreach(utils_file ${UTILS_LIST})
message("Utils file: ${utils_file}")
endforeach()

add_library(HNNS SHARED
    ${GRAPH_LIST}
    ${FLAT_LIST}
    ${UTILS_LIST}
    ${INCLUDE_DIR}/distance.hpp
)

set_target_properties(HNNS PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(HNNS PUBLIC OpenMP::OpenMP_CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native -mtune=native -lrt")

##################################################

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)

file(GLOB GPU_SRC_LIST ${INCLUDE_DIR}/gpu/*.cpp ${INCLUDE_DIR}/gpu/*.cu)
file(GLOB GPU_HEADER_LIST ${INCLUDE_DIR}/gpu/*.h ${INCLUDE_DIR}/gpu/*.cuh)

cuda_add_library(GPU SHARED
    ${GPU_SRC_LIST}
    ${GPU_HEADER_LIST}
)
target_link_libraries(GPU
    CUDA::cudart
    CUDA::cublas
    gtest
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)
target_include_directories(GPU PUBLIC
    $<BUILD_INTERFACE:${INCLUDE_DIR}/gpu>)

target_compile_definitions(GPU PUBLIC FINTEGER=int)


##################################################

set(TEST_DIR ${PROJECT_SOURCE_DIR}/test)
message(${TEST_DIR})

# file(GLOB CPU_TEST_LIST ${TEST_DIR}/*.cpp)
set(CPU_TEST_LIST
    ${TEST_DIR}/flat_cpu.cpp
    ${TEST_DIR}/gt.cpp
    ${TEST_DIR}/hnsw_build.cpp
    ${TEST_DIR}/hnsw_get_data.cpp
    ${TEST_DIR}/hnsw_run.cpp
    )
set(GPU_TEST_LIST
    ${TEST_DIR}/flat_gpu.cpp
)

    
foreach(test_file ${CPU_TEST_LIST})
    get_filename_component(TEST_NAME ${test_file} NAME_WE)
    message("Test file: ${test_file}")
    add_executable(${TEST_NAME} ${test_file})
    target_link_libraries(${TEST_NAME} PRIVATE 
        HNNS
    )
endforeach(test_file ${CPU_TEST_LIST})

foreach(test_file ${GPU_TEST_LIST})
    get_filename_component(TEST_NAME ${test_file} NAME_WE)
    message("Test file: ${test_file}")
    add_executable(${TEST_NAME} ${test_file})
    target_link_libraries(${TEST_NAME} PRIVATE 
        GPU
    )
endforeach(test_file ${GPU_TEST_LIST})

