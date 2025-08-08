add_executable(mtx2c mtx2c.cpp)
target_compile_features(mtx2c PRIVATE cxx_std_17)

target_link_libraries(mtx2c PRIVATE utils_library iofiles_library proto_library ${Protobuf_LIBRARIES} ${MPI_LIBRARIES} stdc++fs)
target_include_directories(mtx2c PUBLIC ${PROJECT_SOURCE_DIR}/include)
install(TARGETS mtx2c DESTINATION bin)

add_executable(mtx3c mtx3c.cpp)
target_compile_features(mtx3c PRIVATE cxx_std_17)

target_link_libraries(mtx3c PRIVATE utils_library iofiles_library proto_library ${Protobuf_LIBRARIES} ${MPI_LIBRARIES} stdc++fs)
target_include_directories(mtx3c PUBLIC ${PROJECT_SOURCE_DIR}/include)
install(TARGETS mtx3c DESTINATION bin)

add_executable(mtx3c_dense mtx3c_dense.cpp)
target_compile_features(mtx3c_dense PRIVATE cxx_std_17)

target_link_libraries(mtx3c_dense PRIVATE utils_library iofiles_library proto_library ${Protobuf_LIBRARIES} ${MPI_LIBRARIES} stdc++fs)
target_include_directories(mtx3c_dense PUBLIC ${PROJECT_SOURCE_DIR}/include)
install(TARGETS mtx3c_dense DESTINATION bin)



add_executable(cov cov.cpp)
target_compile_features(cov PRIVATE cxx_std_17)

target_link_libraries(cov PRIVATE utils_library iofiles_library proto_library ${Protobuf_LIBRARIES} ${MPI_LIBRARIES} stdc++fs)
target_include_directories(cov PUBLIC ${PROJECT_SOURCE_DIR}/include)
install(TARGETS cov DESTINATION bin)

add_executable(buildCovariance buildCovariance.cpp)
target_compile_features(buildCovariance PRIVATE cxx_std_17)

target_link_libraries(buildCovariance PRIVATE utils_library iofiles_library proto_library ${Protobuf_LIBRARIES} ${MPI_LIBRARIES} stdc++fs)
target_include_directories(buildCovariance PUBLIC ${PROJECT_SOURCE_DIR}/include)
install(TARGETS buildCovariance DESTINATION bin)


add_executable(seq_miram seq_miram.cpp)
target_compile_features(seq_miram PRIVATE cxx_std_17)

target_link_libraries(seq_miram PRIVATE algorithms_library tools_library utils_library iofiles_library ${Protobuf_LIBRARIES}  stdc++fs)
target_include_directories(seq_miram PUBLIC ${PROJECT_SOURCE_DIR}/include)
install(TARGETS seq_miram DESTINATION bin)

add_executable(dis_miram dis_miram.cpp)
target_compile_features(dis_miram PRIVATE cxx_std_17)

target_link_libraries(dis_miram PRIVATE algorithms_library tools_library utils_library iofiles_library ${Protobuf_LIBRARIES} ${MPI_LIBRARIES} stdc++fs)
target_include_directories(dis_miram PUBLIC ${PROJECT_SOURCE_DIR}/include)
install(TARGETS dis_miram DESTINATION bin)

add_executable(seq_iram seq_iram.cpp)
target_compile_features(seq_iram PRIVATE cxx_std_17)

target_link_libraries(seq_iram PRIVATE algorithms_library tools_library utils_library iofiles_library ${Protobuf_LIBRARIES}  stdc++fs)
target_include_directories(seq_iram PUBLIC ${PROJECT_SOURCE_DIR}/include)
install(TARGETS seq_iram DESTINATION bin)

add_executable(dis_iram dis_iram.cpp)
target_compile_features(dis_iram PRIVATE cxx_std_17)

target_link_libraries(dis_iram PRIVATE algorithms_library tools_library utils_library iofiles_library ${Protobuf_LIBRARIES} ${MPI_LIBRARIES} stdc++fs)
target_include_directories(dis_iram PUBLIC ${PROJECT_SOURCE_DIR}/include)
install(TARGETS dis_iram DESTINATION bin)


add_executable(eigenvectors_save eigenvectors_save.cpp)
target_compile_features(eigenvectors_save PRIVATE cxx_std_17)

target_link_libraries(eigenvectors_save PRIVATE algorithms_library tools_library utils_library iofiles_library ${Protobuf_LIBRARIES} ${MPI_LIBRARIES} stdc++fs)
target_include_directories(eigenvectors_save PUBLIC ${PROJECT_SOURCE_DIR}/include)
install(TARGETS eigenvectors_save DESTINATION bin)


add_executable(scalability_eigenvectors scalability_eigenvectors.cpp)
target_compile_features(scalability_eigenvectors PRIVATE cxx_std_17)

target_link_libraries(scalability_eigenvectors PRIVATE algorithms_library tools_library utils_library iofiles_library ${Protobuf_LIBRARIES} ${MPI_LIBRARIES} stdc++fs)
target_include_directories(scalability_eigenvectors PUBLIC ${PROJECT_SOURCE_DIR}/include)
install(TARGETS scalability_eigenvectors DESTINATION bin)

add_executable(speigenvectors_save speigenvectors_save.cpp)
target_compile_features(speigenvectors_save PRIVATE cxx_std_17)

target_link_libraries(speigenvectors_save PRIVATE algorithms_library tools_library utils_library iofiles_library ${Protobuf_LIBRARIES} ${MPI_LIBRARIES} stdc++fs)
target_include_directories(speigenvectors_save PUBLIC ${PROJECT_SOURCE_DIR}/include)
install(TARGETS speigenvectors_save DESTINATION bin)


add_executable(scalability_speigenvectors scalability_speigenvectors.cpp)
target_compile_features(scalability_speigenvectors PRIVATE cxx_std_17)

target_link_libraries(scalability_speigenvectors PRIVATE algorithms_library tools_library utils_library iofiles_library ${Protobuf_LIBRARIES} ${MPI_LIBRARIES} stdc++fs)
target_include_directories(scalability_speigenvectors PUBLIC ${PROJECT_SOURCE_DIR}/include)
install(TARGETS scalability_speigenvectors DESTINATION bin)
