# rm -rf * && cmake .. && make -j32
cmake_minimum_required(VERSION 3.0)
project(HNNS)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)

set(CPU_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
message(${CPU_INCLUDE_DIR})

include_directories(
    ${CPU_INCLUDE_DIR}
)

file(GLOB GRAPH_LIST ${CPU_INCLUDE_DIR}/graph/*.hpp)
file(GLOB FLAT_LIST ${CPU_INCLUDE_DIR}/flat/*.hpp)
file(GLOB UTILS_LIST ${CPU_INCLUDE_DIR}/utils/*.hpp)

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
    ${CPU_INCLUDE_DIR}/distance.hpp
)

set_target_properties(HNNS PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(HNNS PUBLIC OpenMP::OpenMP_CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native -mtune=native -lrt")

##################################################

set(TEST_DIR ${PROJECT_SOURCE_DIR}/test)
message(${TEST_DIR})

file(GLOB TEST_LIST ${TEST_DIR}/*.cpp)

foreach(test_file ${TEST_LIST})
    get_filename_component(TEST_NAME ${test_file} NAME_WE)
    message("Test file: ${test_file}")
    add_executable(${TEST_NAME} ${test_file})
    target_link_libraries(${TEST_NAME} PRIVATE 
        HNNS
    )
endforeach(test_file ${TEST_LIST})

