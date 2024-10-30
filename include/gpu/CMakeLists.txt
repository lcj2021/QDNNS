cmake_minimum_required(VERSION 3.0)
project(gpu)
# set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)

file(GLOB SRC_LIST "*.cpp" "*.cu")
list(FILTER SRC_LIST EXCLUDE REGEX "gtest_gpu.cpp")
file(GLOB HEADER_LIST "*.h" "*.cuh")

cuda_add_library(gpu SHARED
    ${SRC_LIST}
    ${HEADER_LIST}
)
target_link_libraries(gpu
    CUDA::cudart
    CUDA::cublas
    gtest
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

target_include_directories(gpu PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

target_compile_definitions(gpu PRIVATE FINTEGER=int)

# add_executable(gtest_gpu gtest_gpu.cpp)
# target_sources(gtest_gpu PRIVATE
#     ${SRC_LIST}
#     ${HEADER_LIST}
# )
# target_compile_definitions(gtest_gpu PRIVATE FINTEGER=int)
# target_link_libraries(gtest_gpu PUBLIC
#     CUDA::cudart 
#     CUDA::cublas
#     gtest
#     OpenMP::OpenMP_CXX
#     ${CUDA_LIBRARIES}
#     ${BLAS_LIBRARIES}
#     ${LAPACK_LIBRARIES}
#     gpu
# )

add_executable(test_gpu test_gpu.cpp)
target_sources(test_gpu PRIVATE
    ${SRC_LIST}
    ${HEADER_LIST}
)
target_compile_definitions(test_gpu PRIVATE FINTEGER=int)
target_link_libraries(test_gpu PUBLIC
    CUDA::cudart 
    CUDA::cublas
    gtest
    OpenMP::OpenMP_CXX
    ${CUDA_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    gpu
)


# clear && rm CMakeCache.txt && cmake .. && make -j8 && ./flat_gpu imagenet 96 3000 1024 1000