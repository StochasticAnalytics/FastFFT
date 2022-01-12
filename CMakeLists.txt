cmake_minimum_required(VERSION 3.18)
# CMAKE_CUDA_ARCHITECTURES was added in 3.18

project(FastFFT
        LANGUAGES CXX
        VERSION 0.0.0
        HOMEPAGE_URL https://github.com/bHimes/FastFFT
        )

# ---------------------------------------------------------------------------------------
# Whether or not this is the top-level (master) project
# ---------------------------------------------------------------------------------------
#if (NOT DEFINED FASTFFT_IS_MASTER)
#    if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
#        set(FASTFFT_IS_MASTER ON)
#    else ()
#        set(FASTFFT_IS_MASTER OFF)
#    endif ()
#endif ()
include(cmake/settings/PreventInSourceBuilds.cmake)

# ---------------------------------------------------------------------------------------
# Project options
# ---------------------------------------------------------------------------------------
include(cmake/settings/ProjectOptions.cmake)

if (FASTFFT_ENABLE_CUDA)
    # TODO: check for specific version?
    enable_language(CUDA)
    find_package(CUDAToolkit)
endif ()

if (BUILD_SHARED_LIBS)
    message(FATAL_ERROR "FastFFT is only tested as a static library currently, should be added for linking through python. Turn BUILD_SHARED_LIBS OFF")
endif ()

# ---------------------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------------------
include(cmake/settings/ProjectEnvironment.cmake)

# These interfaces sets the project base requirements.
# Targets should not directly modify these since they're common to multiple targets.
add_library(prj_common_option INTERFACE)
add_library(prj_cxx_warnings INTERFACE)

# Make sure everything in the project use C++17.
target_compile_features(prj_common_option INTERFACE cxx_std_17)

# Add the C++ warnings to the C++ compiler.
include(cmake/settings/CompilerWarnings.cmake)
set_cxx_compiler_warnings(prj_cxx_warnings)

# Go fetch the necessary external contents and make them available to the project.
set(FASTFFT_EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext)

include(${PROJECT_SOURCE_DIR}/ext/fftw/fftw.cmake)
include(${PROJECT_SOURCE_DIR}/ext/half/half.cmake)

# ---------------------------------------------------------------------------------------
# Project targets
# ---------------------------------------------------------------------------------------
add_subdirectory(src/fastfft)

# The test framework (using Catch2): noa_tests
if (FASTFFT_BUILD_TESTS)
    add_subdirectory(tests)
else ()
    message(STATUS "Skipping target: FastFFT tests")
endif ()

# TODO: should a target for building and publishing the docs via Jupyter Book be added? 
# (Currently I just run through a shell cmd in tasks.json in vscode)

