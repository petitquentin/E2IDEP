set(HEADER_LIST_TOOLS "${MIRAM_SOURCE_DIR}/include/tools/tools.hpp")
set(FILES_LIST_TOOLS "tools.cpp")

add_library(tools_library ${FILES_LIST_TOOLS} ${HEADER_LIST_TOOLS})
target_include_directories(tools_library PRIVATE ../include)
target_link_libraries(tools_library PRIVATE ${Eigen_LIBRARIES} Eigen3::Eigen)
target_compile_features(tools_library PUBLIC cxx_std_11)


file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/python DESTINATION ${CMAKE_CURRENT_BINARY_DIR})