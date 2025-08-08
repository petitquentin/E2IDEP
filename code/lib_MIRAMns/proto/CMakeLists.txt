INCLUDE(FindProtobuf)
FIND_PACKAGE(Protobuf REQUIRED)
INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})
PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HEADER protoData.proto)
ADD_LIBRARY(proto_library ${PROTO_HEADER} ${PROTO_SRC})
target_link_libraries(proto_library ${PROTOBUF_LIBRARIES})
target_include_directories(proto_library PRIVATE ../include)

target_compile_features(proto_library PUBLIC cxx_std_11)