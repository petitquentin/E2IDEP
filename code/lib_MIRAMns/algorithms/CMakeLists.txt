set(HEADER_LIST_ALGORITHMS "${MIRAM_SOURCE_DIR}/include/algorithms/algorithms.hpp")
set(FILES_LIST_ALGORITHMS "algorithms.cpp")

add_library(algorithms_library ${FILES_LIST_ALGORITHMS} ${HEADER_LIST_ALGORITHMS})
target_include_directories(algorithms_library PRIVATE ../include)
target_link_libraries(algorithms_library PRIVATE ${Eigen_LIBRARIES} Eigen3::Eigen tools_library ${MPI_LIBRARIES})

target_compile_features(algorithms_library PUBLIC cxx_std_11)