cmake_minimum_required(VERSION 3.14.7)

project(vz)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)

FetchContent_Declare(uikit
  URL "https://github.com/tay10r/uikit/archive/refs/tags/v0.2.0.zip"
  URL_HASH "SHA256=ea6616e3e3dc095a450a36a0dfb43c57c93fe406956f82c2fff5c2bf614b0646")
FetchContent_MakeAvailable(uikit)

FetchContent_Declare(pybind11
  URL "https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.zip"
  URL_HASH "SHA256=d0a116e91f64a4a2d8fb7590c34242df92258a61ec644b79127951e821b47be6")
FetchContent_MakeAvailable(pybind11)

set_target_properties(glad_gles2 PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(imgui_opengl3 PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(imgui_glfw PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(implot PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(vz
  main.cpp)
target_link_libraries(vz PUBLIC uikit::uikit)

install(TARGETS vz
  RUNTIME DESTINATION "."
  LIBRARY DESTINATION "."
  ARCHIVE DESTINATION ".")
