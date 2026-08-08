cmake_minimum_required(VERSION 3.23)
project(VectorSearchEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 1. Find Python from active venv
set(Python_FIND_VIRTUALENV FIRST)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)


find_package(pybind11 CONFIG)

# 4. Build extension
pybind11_add_module(vector_search main.cpp)
target_link_libraries(vector_search PRIVATE pybind11::headers)

# 5. Send output binary directly to api/ folder
set_target_properties(vector_search PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../api"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../api"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../api"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../api"
)

