cmake_minimum_required(VERSION 3.13)
project(berkeley C)

set(CMAKE_C_STANDARD 11)
add_executable(berkeley main.c)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(berkeley ${MPI_C_LIBRARIES})