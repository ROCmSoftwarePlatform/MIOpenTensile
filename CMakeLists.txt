
cmake_minimum_required(VERSION 3.5)

project(miopentensile LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(virtualenv)

# Use the virtual-env setup and download package from specified repo:
set( MIOPEN_TENSILE_TAG 315f058dbc34233a81a1ada7ca618137040c5d2f CACHE STRING "Tensile tag to download" )
virtualenv_install("git+https://github.com/ROCmSoftwarePlatform/Tensile.git@${MIOPEN_TENSILE_TAG}")
list(APPEND CMAKE_PREFIX_PATH ${VIRTUALENV_HOME_DIR})

find_package(Tensile REQUIRED)

TensileCreateLibraryFiles(
    "${CMAKE_CURRENT_SOURCE_DIR}/yaml"
    "${CMAKE_CURRENT_BINARY_DIR}/yaml"
    EMBED_LIBRARY miopen_tensile_kernels
    EMBED_KEY     miopen_tensile_kernels
    VAR_PREFIX MIOPEN_TENSILE
    )

add_library(MIOpenTensile src/main.cpp)
target_link_libraries(MIOpenTensile miopen_tensile_kernels)

# find_package( ROCM CONFIG QUIET PATHS /opt/rocm )


