
cmake_minimum_required(VERSION 3.5)

project(miopentensile LANGUAGES CXX)

add_definitions(-std=c++14)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc /opt/rocm/hip /opt/rocm)

find_package( ROCM CONFIG QUIET PATHS /opt/rocm )
include(ROCMInstallTargets)
include(ROCMSetupVersion)

rocm_setup_version(VERSION 1.0)

# Dont build as shared as a workaround
# option( BUILD_SHARED_LIBS "Build as a shared library" ON )

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(virtualenv)

# Workaround broken hcc-config
macro(add_library NAME)
    if(${NAME} STREQUAL "hsa-runtime64")
        _add_library(hsa-runtime64 SHARED IMPORTED)
    else()
        _add_library(${NAME} ${ARGN})
    endif()
endmacro()
find_package(hip)

# Use the virtual-env setup and download package from specified repo:
set( MIOPEN_TENSILE_TAG b9ca88f4334b8091b50f716a0951bd9a5b994127 CACHE STRING "Tensile tag to download" )
virtualenv_install(wheel)
virtualenv_install("git+https://github.com/ROCmSoftwarePlatform/Tensile.git@${MIOPEN_TENSILE_TAG}")
list(APPEND CMAKE_PREFIX_PATH ${VIRTUALENV_HOME_DIR})

find_package(Tensile REQUIRED COMPONENTS HIP LLVM)

# Workaround broke tensile cmake
get_filename_component(COMPILER_PATH ${CMAKE_CXX_COMPILER} DIRECTORY)
set(ENV{PATH} "${hip_BIN_INSTALL_DIR}:${COMPILER_PATH}:$ENV{PATH}")
TensileCreateLibraryFiles(
    "${CMAKE_CURRENT_SOURCE_DIR}/yaml"
    "${CMAKE_CURRENT_BINARY_DIR}/yaml"
    EMBED_LIBRARY miopen_tensile_kernels
    EMBED_KEY     miopen_tensile_kernels
    VAR_PREFIX MIOPEN_TENSILE
    )


add_library(MIOpenTensile SHARED src/gemm_api.cpp)
target_link_libraries(MIOpenTensile PUBLIC hip::host)
target_link_libraries(MIOpenTensile PRIVATE TensileHost miopen_tensile_kernels)

include(ROCMCreatePackage)
rocm_create_package(
    NAME MIOpenTensile
    DESCRIPTION "MIOpen and Tensile integration library"
    MAINTAINER "Paul Fultz II <paul.fultz@amd.com>"
    LDCONFIG
    DEPENDS hip-hcc
)

rocm_install_targets(
  TARGETS MIOpenTensile
  INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

rocm_export_targets(
  TARGETS MIOpenTensile
  DEPENDS
    hip
)

add_subdirectory(test)

