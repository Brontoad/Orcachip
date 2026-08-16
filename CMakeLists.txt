cmake_minimum_required(VERSION 3.23)
project(Orcachip LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)

# pybind11
find_package(pybind11 REQUIRED CONFIG)

# Add Python module orcachip
pybind11_add_module(orcachip_core 
    src/bindings.cpp
    src/engine/vector_search_engine.cpp
)

# Set include paths
target_include_directories(orcachip_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/engine
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
)

# Select install directory
install(TARGETS orcachip_core DESTINATION orcachip)