CMAKE_MINIMUM_REQUIRED (VERSION 3.5.0)
SET(CMAKE_CXX_FLAGS "-g -Wall -Werror -std=c++11")
project(MIRAM VERSION 0.3 LANGUAGES CXX)

find_package (Eigen3 3.3 REQUIRED)

link_directories(${Eigen_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})


find_package(Protobuf REQUIRED)


INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(${Protobuf_INCLUDE_DIRS})

ADD_SUBDIRECTORY(proto)

find_package(MPI)
if(MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
    message([NOTICE] "MPI was found")
endif(MPI_FOUND)


# Set MPICXX as the CXX compiler

SET(CMAKE_CXX_COMPILER mpicxx)

add_subdirectory(utils)
add_subdirectory(test)
add_subdirectory(tools)
add_subdirectory(algorithms)
add_subdirectory(iofiles)


file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
