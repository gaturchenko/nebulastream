cmake_minimum_required(VERSION 3.28)

project(flink_openvino_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenVINO REQUIRED)
find_package(JNI REQUIRED)

add_library(openvino_engine STATIC
        OpenVinoEngine.cpp
)

set_target_properties(openvino_engine PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

target_include_directories(openvino_engine
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(openvino_engine
        PUBLIC
        openvino::runtime
)

add_library(flink_openvino_bridge SHARED
        OpenVinoC.cpp
)

target_include_directories(flink_openvino_bridge
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(flink_openvino_bridge
        PRIVATE
        openvino_engine
)

add_library(flink_openvino_bridge_jni SHARED
        OpenVinoJNI.cpp
)

target_include_directories(flink_openvino_bridge_jni
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${JNI_INCLUDE_DIRS}
)

target_link_libraries(flink_openvino_bridge_jni
        PRIVATE
        flink_openvino_bridge
        ${JNI_LIBRARIES}
)
