message(STATUS "Configuring target: FastFFT")

# ---------------------------------------------------------------------------------------
# Options and libraries
# ---------------------------------------------------------------------------------------

add_library(fastfft_libraries INTERFACE)
add_library(fastfft_options INTERFACE)

target_link_libraries(fastfft_libraries
        INTERFACE
        half-ieee754
        )

# ---------------------------------------------------------------------------------------
# CPU backend
# ---------------------------------------------------------------------------------------
if (FASTFFT_ENABLE_CPU)
    set(FASTFFT_HEADERS ${FASTFFT_HEADERS} ${FASTFFT_CPU_HEADERS})
    set(FASTFFT_SOURCES ${FASTFFT_SOURCES} ${FASTFFT_CPU_SOURCES})

    target_link_libraries(fastfft_libraries
            INTERFACE
            fftw3::float
            )

#    find_package(OpenMP 4.5 REQUIRED)
#    if (FASTFFT_ENABLE_OPENMP)
#        target_link_libraries(fastfft_libraries
#                INTERFACE
#                OpenMP::OpenMP_CXX
#                )
#    endif ()


endif ()

# ---------------------------------------------------------------------------------------
# CUDA backend
# ---------------------------------------------------------------------------------------
if (FASTFFT_ENABLE_CUDA)
    set(FASTFFT_HEADERS ${FASTFFT_HEADERS} ${FASTFFT_CUDA_HEADERS})
    set(FASTFFT_SOURCES ${FASTFFT_SOURCES} ${FASTFFT_CUDA_SOURCES})

    # TODO compilation fails with fastfft_tests when using cufft_static...?
    #      Maybe look here: https://github.com/arrayfire/arrayfire/blob/master/src/backend/cuda/CMakeLists.txt
    target_link_libraries(fastfft_libraries
            INTERFACE
            CUDA::cudart
            CUDA::cufft
            )
endif ()

# ---------------------------------------------------------------------------------------
# The target
# ---------------------------------------------------------------------------------------
add_library(fastfft_static ${FASTFFT_SOURCES} ${FASTFFT_HEADERS})
add_library(fastfft::fastfft_static ALIAS fastfft_static)

target_link_libraries(fastfft_static
        PRIVATE
        prj_common_option
        prj_cxx_warnings
        fastfft_options
        PUBLIC
        fastfft_libraries
        )

# Not sure why, but these need to be set on the target directly
if (FASTFFT_ENABLE_CUDA)
    set_target_properties(fastfft_static
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            # CUDA_RESOLVE_DEVICE_SYMBOLS ON
            `CUDA_ARCHITECTURES` ${FASTFFT_CUDA_ARCH}
            )
endif ()

if (FASTFFT_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(
            RESULT
            result
            OUTPUT
            output)
    if (result)
        set_target_properties(fastfft_static PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
    else ()
        message(SEND_ERROR "IPO is not supported: ${output}")
    endif ()
endif ()

if (FASTFFT_ENABLE_PCH)
    target_precompile_headers(fastfft_static
            PRIVATE
            # Streams:
            <iostream>
            <fstream>
            <string>
            <string_view>

            # Containers:
            <map>
            <unordered_map>
            <vector>
            <array>
            <tuple>

            # Others:
            <cstdint>
            <cctype>
            <cstring>
            <cerrno>
            <cmath>
            <exception>
            <filesystem>
            <thread>
            <utility>
            <algorithm>
            <memory>
            <type_traits>
            <complex>
            <bitset>

            # spdlog
            <spdlog/spdlog.h>
            <spdlog/sinks/stdout_color_sinks.h>
            <spdlog/sinks/basic_file_sink.h>
            <spdlog/fmt/bundled/compile.h>
            <spdlog/fmt/bundled/ranges.h>
            <spdlog/fmt/bundled/os.h>
            <spdlog/fmt/bundled/chrono.h>
            <spdlog/fmt/bundled/color.h>
            )
endif ()

# Set definitions:
target_compile_definitions(fastfft_static
        PUBLIC
        "$<$<CONFIG:DEBUG>:FASTFFT_DEBUG>"
        "$<$<BOOL:${FASTFFT_ENABLE_PROFILER}>:FASTFFT_PROFILE>"
        "$<$<BOOL:${FASTFFT_ENABLE_CUDA}>:FASTFFT_ENABLE_CUDA>"
        "$<$<BOOL:${FASTFFT_ENABLE_TIFF}>:FASTFFT_ENABLE_TIFF>"
        "$<$<BOOL:${FASTFFT_ENABLE_OPENMP}>:FASTFFT_ENABLE_OPENMP>"
        "$<$<BOOL:${FASTFFT_FFTW_USE_THREADS}>:FASTFFT_FFTW_USE_THREADS>"
        )

# Set included directories:
target_include_directories(fastfft_static
        PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
        "$<BUILD_INTERFACE:${FASTFFT_GENERATED_HEADERS_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        )

# ---------------------------------------------------------------------------------------
# API Compatibility - Versioning
# ---------------------------------------------------------------------------------------
configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/settings/Version.h.in"
        "${FASTFFT_GENERATED_HEADERS_DIR}/fastfft/Version.h"
        @ONLY)

# NOTE: Since it is static library only, the SOVERSION doesn't matter much for now.
#
# The library uses the semantic versioning: MAJOR.MINOR.PATCH
# PATCH: Bug fix only. No API changes.
# MINOR: Non-breaking additions to the API (i.e. something was added).
#        This ensure that if someone had built against the previous version, they could simply
#        replace with the new version without having to rebuild their application.
#        When updated, PATCH should be reset to 0.
# MAJOR: Breaking change. Reset MINOR and PATCH to 0.

# In the case of shared library:
# UNIX-based:
#   libfastfft.so                       : NAME LINK (for build-time linker)
#   libfastfft.so.SOVERSION             : SONAME (for runtime loader)
#   libfastfft.so.MAJOR.MINOR.PATCH     : REAL LIBRARY (for human and packages)
#
#   The SONAME is used to specify the build version and API version respectively. Specifying the
#   SOVERSION is important, since CMake defaults it to the VERSION, effectively saying that any
#   MINOR or PATCH should break the API (which can break the runtime loader). The SOVERSION will
#   be set on the fastfft target and in our case will be equal to the PROJECT_VERSION_MAJOR.
#
# Windows:
#   fastfft.dll     : Acts kind of the SONAME
#   fastfft.lib     : Acts kind of the NAME LINK
#   Some version details may be encoded into the binaries (if Makefile or Ninja),
#   but this is usually not used.

set_target_properties(fastfft_static PROPERTIES
        SOVERSION ${PROJECT_VERSION_MAJOR}
        VERSION ${PROJECT_VERSION})

# ---------------------------------------------------------------------------------------
# Symbol visibility
# ---------------------------------------------------------------------------------------
# NOTE: Since it is static library only, ignore the export/import for now.
#
# Visual Studio:
#   Visual Studio hides symbols by default.
#   The attribute __declspec(dllexport) is used when building the library.
#   The attribute __declspec(dllimport) is used when using the library.
# GCC and Clang:
#   GCC and Clang do NOT hide symbols by default.
#   Compiler option -fvisibility=hidden to change default visibility to hidden.
#   Compiler option -fvisibility-inlines-hidden to change visibility of inlined code (including templates).
#   Then, use __attribute__((visibility("default"))) to make something visible visible.

# Hides everything by default - export manually (for Visual Studio, do nothing).
#set_target_properties(fastfft PROPERTIES
#        CMAKE_CXX_VISIBILITY_PRESET hidden
#        CMAKE_VISIBILITY_INLINES_HIDDEN YES)

# Generate the attribute given the compiler and usage (build vs using). CMake does that nicely so
# we don't have to distinguish between the different scenarios.
#   - generates fastfft/API.h
#   - ensure the macro FASTFFT_API is defined. This can be added to classes, free functions and global variables;
#include(GenerateExportHeader)
#generate_export_header(fastfft
#        EXPORT_MACRO_NAME FASTFFT_API
#        EXPORT_FILE_NAME ${FASTFFT_GENERATED_HEADERS_DIR}/fastfft/API.h)

# ---------------------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------------------
# Targets:
#   - <install_path>/lib/libfastfft(b).(a|so)
#   - header location after install: <prefix>/fastfft/*.h
#   - headers can be included by C++ code `#include <fastfft/*.h>`
install(TARGETS half-ieee754 prj_common_option prj_cxx_warnings fastfft_options fastfft_libraries fastfft_static
        EXPORT "${FASTFFT_TARGETS_EXPORT_NAME}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"

        LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        COMPONENT fastfft_runtime
        NAMELINK_COMPONENT fastfft_development

        ARCHIVE
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        COMPONENT fastfft_development

        RUNTIME
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT fastfft_runtime
        )

# Headers:
#   - *.h -> <install_path>/include/fastfft/*.h
foreach (FILE ${FASTFFT_HEADERS})
    get_filename_component(DIR ${FILE} DIRECTORY)
    install(FILES ${FILE} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fastfft/${DIR})
endforeach ()

# Headers:
#   - <build_path>/fastfft_generated_headers/fastfft/Version.h -> <install_path>/include/fastfft/Version.h
#   - <build_path>/fastfft_generated_headers/fastfft/API.h     -> <install_path>/include/fastfft/API.h
install(FILES
        # "${FASTFFT_GENERATED_HEADERS_DIR}/fastfft/API.h"
        "${FASTFFT_GENERATED_HEADERS_DIR}/fastfft/Version.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fastfft")

# Package config:
#   - <install_path>/lib/cmake/fastfft/NoaConfig.cmake
#   - <install_path>/lib/cmake/fastfft/NoaConfigVersion.cmake
install(FILES
        "${FASTFFT_CONFIG_FILE}"
        "${FASTFFT_CONFIG_VERSION_FILE}"
        DESTINATION "${FASTFFT_INSTALL_LIBDIR}")

# Package config:
#   - <install_path>/lib/cmake/Noa/NoaTargets.cmake
install(EXPORT "${FASTFFT_TARGETS_EXPORT_NAME}"
        FILE "${FASTFFT_TARGETS_EXPORT_NAME}.cmake"
        DESTINATION "${FASTFFT_INSTALL_LIBDIR}"
        NAMESPACE "fastfft::")

message("")
