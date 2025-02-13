cmake_minimum_required(VERSION 3.14.7)

project(deepslope_demo
  DESCRIPTION "A demo and reference implementation of using DeepSlope in C++ with OpenCV."
  LANGUAGES CXX)

find_package(OpenCV REQUIRED)

add_library(deepslope
  deepslope.h
  deepslope.cpp)
target_link_libraries(deepslope PUBLIC ${OpenCV_LIBRARIES})

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  add_executable(deepslope_example
    example.cpp)
  target_link_libraries(deepslope_example
    PUBLIC
      deepslope)
  set_target_properties(deepslope_example
    PROPERTIES
      OUTPUT_NAME example)
  target_compile_definitions(deepslope_example
    PUBLIC
      "PROJECT_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"")
endif()
