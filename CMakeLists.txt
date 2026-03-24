if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

cmake_minimum_required(VERSION 3.18)
project(HNSW_Lib)
find_program(PYTHON_EXECUTABLE python3 REQUIRED)
set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)

add_library(hnsw_lib SHARED src/hnsw_lib.cpp)

target_compile_options(hnsw_lib PRIVATE -march=armv8.2-a+fp16 -ffast-math)

set_target_properties(hnsw_lib PROPERTIES PREFIX "")
set_target_properties(hnsw_lib PROPERTIES SUFFIX ".so")

target_link_libraries(hnsw_lib PRIVATE pybind11::module)