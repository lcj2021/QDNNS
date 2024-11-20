# rm -rf * && cmake .. && make -j32
##################################################
cmake_minimum_required(VERSION 3.0)
project(HNNS)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)
set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
message(${INCLUDE_DIR})

include_directories(/usr/include/)
link_directories(/usr/local/lib)
include_directories(
    ${INCLUDE_DIR}
)

file(GLOB GRAPH_LIST ${INCLUDE_DIR}/graph/*.hpp)
file(GLOB FLAT_LIST ${INCLUDE_DIR}/flat/*.hpp)
file(GLOB UTILS_LIST ${INCLUDE_DIR}/utils/*.hpp)
file(GLOB GPU_SRC_LIST ${INCLUDE_DIR}/gpu/*.cpp ${INCLUDE_DIR}/gpu/*.cu)
file(GLOB GPU_HEADER_LIST ${INCLUDE_DIR}/gpu/*.h ${INCLUDE_DIR}/gpu/*.cuh)

foreach(graph_file ${GRAPH_LIST})
message("Graph file: ${graph_file}")
endforeach()

foreach(flat_file ${FLAT_LIST})
message("Flat file: ${flat_file}")
endforeach()

foreach(utils_file ${UTILS_LIST})
message("Utils file: ${utils_file}")
endforeach()

cuda_add_library(HNNS SHARED
    ${GRAPH_LIST}
    ${FLAT_LIST}
    ${UTILS_LIST}
    ${INCLUDE_DIR}/distance.hpp
    ${GPU_SRC_LIST}
    ${GPU_HEADER_LIST}
)

target_link_libraries(HNNS 
    CUDA::cudart
    CUDA::cublas
    gtest
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    lib_lightgbm.so
)

set_target_properties(HNNS PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(HNNS PUBLIC
    $<BUILD_INTERFACE:${INCLUDE_DIR}/gpu>)

target_compile_definitions(HNNS PUBLIC FINTEGER=int)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native -mtune=native -lrt")

##################################################

set(TEST_DIR ${PROJECT_SOURCE_DIR}/test)
message(${TEST_DIR})

# file(GLOB TEST_LIST ${TEST_DIR}/*.cpp)
set(TEST_LIST
    ${TEST_DIR}/difficulty.cpp
    ${TEST_DIR}/flat_cpu.cpp
    ${TEST_DIR}/flat_gpu.cpp
    ${TEST_DIR}/gt_cpu.cpp
    ${TEST_DIR}/gt_gpu.cpp
    ${TEST_DIR}/hnns.cpp
    ${TEST_DIR}/hnsw_build.cpp
    ${TEST_DIR}/hnsw_get_data.cpp
    ${TEST_DIR}/hnsw_run.cpp
    ${TEST_DIR}/test_dataloader.cpp
)
    
foreach(test_file ${TEST_LIST})
    get_filename_component(TEST_NAME ${test_file} NAME_WE)
    message("Test file: ${test_file}")
    add_executable(${TEST_NAME} ${test_file})
    target_link_libraries(${TEST_NAME} PRIVATE 
        HNNS
    )
endforeach(test_file ${TEST_LIST})

