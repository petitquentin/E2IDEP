set(IO_HEADER_LIST "${MIRAM_SOURCE_DIR}/include/iofiles/iofiles.hpp")
set(IO_FILES_LIST "ioread.cpp" "iowrite.cpp")

add_library(iofiles_library ${IO_FILES_LIST} ${IO_HEADER_LIST})
target_link_libraries(iofiles_library PRIVATE proto_library ${MPI_LIBRARIES} ${Eigen_LIBRARIES} Eigen3::Eigen)

target_include_directories(iofiles_library PRIVATE ../include)
target_include_directories(iofiles_library PRIVATE ../proto)
target_compile_features(iofiles_library PUBLIC cxx_std_11)