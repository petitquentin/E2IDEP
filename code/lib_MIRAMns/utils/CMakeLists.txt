set(HEADER_LIST "${MIRAM_SOURCE_DIR}/include/utils/utils.hpp")
set(FILES_LIST "utils.cpp")

add_library(utils_library ${FILES_LIST} ${HEADER_LIST})
target_link_libraries(utils_library PUBLIC iofiles_library proto_library ${MPI_LIBRARIES})

target_include_directories(utils_library PRIVATE ../include)
target_include_directories(utils_library PRIVATE ../proto)
target_compile_features(utils_library PUBLIC cxx_std_11)