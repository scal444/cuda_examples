cmake_minimum_required(VERSION 3.14)
project(cuda_examples CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

add_library(capped_distance capped_distance.cc capped_distance.cu)

target_include_directories (capped_distance PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


# ---------------------------------------------------------------------------------------------------------
# Testing stuff
# ---------------------------------------------------------------------------------------------------------
add_subdirectory(googletest)

include_directories(googletest/googletest/include)
include_directories(googletest/googlemock/include)

add_executable(capped_distance_test capped_distance_test.cc)

target_link_libraries(capped_distance_test capped_distance gtest gtest_main gmock gmock_main)
