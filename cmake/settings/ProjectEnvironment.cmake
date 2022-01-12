# Sets the project environment.
# Using:
#   - CMAKE_MODULE_PATH
#   - CMAKE_CONFIGURATION_TYPES or CMAKE_BUILD_TYPE
#   - PROJECT_VERSION
#   - FASTFFT_ENABLE_CUDA
#
# Introduces:
#   - CMAKE_INSTALL_LIBDIR      : The standard install directories. See GNUInstallDirs.
#   - CMAKE_INSTALL_BINDIR      : The standard install directories. See GNUInstallDirs.
#   - CMAKE_INSTALL_INCLUDEDIR  : The standard install directories. See GNUInstallDirs.
#   - FASTFFT_GENERATED_DIR         : Where to store generated project CMake files.
#   - FASTFFT_GENERATED_HEADERS_DIR : Where to store generated FastFFT C++ headers.
#   - FASTFFT_INSTALL_LIBDIR        : Where to store installed project CMake files.
#   - FASTFFT_TARGETS_EXPORT_NAME
#   - FASTFFT_CONFIG_FILE
#   - FASTFFT_CONFIG_VERSION_FILE
#
# Created targets:
#   - uninstall                 : Uninstall everything in the install_manifest.txt.

# Add the local modules.
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/find)

# Make sure different configurations don't collide.
set(CMAKE_DEBUG_POSTFIX "d")

# Generate compile_commands.json to make it easier to work with clang based tools.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Get the standard install directories.
include(GNUInstallDirs)

# Generated directories:
set(FASTFFT_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/FastFFT_generated")
set(FASTFFT_GENERATED_HEADERS_DIR "${CMAKE_CURRENT_BINARY_DIR}/FastFFT_generated_headers")

# ---------------------------------------------------------------------------------------
# Build type (default: Release)
# ---------------------------------------------------------------------------------------
# If multi-config, uses CMAKE_CONFIGURATION_TYPES.
# If single-config, uses CMAKE_BUILD_TYPE.
get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if (isMultiConfig)
    set(CMAKE_CONFIGURATION_TYPES "Release;Debug;MinSizeRel;RelWithDebInfo" CACHE STRING "")
    message(STATUS "CMAKE_CONFIGURATION_TYPES (multi-config): ${CMAKE_CONFIGURATION_TYPES}")
else ()
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build.")
    message(STATUS "CMAKE_BUILD_TYPE (single-config): ${CMAKE_BUILD_TYPE}")

    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug" "MinSizeRel" "RelWithDebInfo")
endif ()
message(STATUS "CMAKE_GENERATOR: ${CMAKE_GENERATOR}")
message(STATUS "CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
if (FASTFFT_ENABLE_CUDA)
    message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CMAKE_CUDA_HOST_COMPILER: ${CMAKE_CUDA_HOST_COMPILER}")
    message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
endif ()

# ---------------------------------------------------------------------------------------
# CMake project packaging files
# ---------------------------------------------------------------------------------------
set(FASTFFT_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}/cmake/FastFFT")
set(FASTFFT_TARGETS_EXPORT_NAME "NoaTargets")
set(FASTFFT_CONFIG_VERSION_FILE "${FASTFFT_GENERATED_DIR}/NoaConfigVersion.cmake")
set(FASTFFT_CONFIG_FILE "${FASTFFT_GENERATED_DIR}/NoaConfig.cmake")

include(CMakePackageConfigHelpers)
configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/settings/Config.cmake.in"
        "${FASTFFT_CONFIG_FILE}"
        INSTALL_DESTINATION "${FASTFFT_INSTALL_LIBDIR}")
write_basic_package_version_file(
        "${FASTFFT_CONFIG_VERSION_FILE}"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMajorVersion)
# Since we currently only support STATIC builds, the ConfigVersion is not really useful, is it?.

# ---------------------------------------------------------------------------------------
# Uninstall target
# ---------------------------------------------------------------------------------------
configure_file("${PROJECT_SOURCE_DIR}/cmake/settings/Uninstall.cmake.in"
        "${FASTFFT_GENERATED_DIR}/Uninstall.cmake"
        IMMEDIATE @ONLY)
add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${FASTFFT_GENERATED_DIR}/Uninstall.cmake)

# ---------------------------------------------------------------------------------------
# RPATH
# ---------------------------------------------------------------------------------------
# Always full RPATH (for shared libraries)
#  https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
#if (BUILD_SHARED_LIBS)
#    # use, i.e. don't skip the full RPATH for the build tree
#    set(CMAKE_SKIP_BUILD_RPATH FALSE)
#
#    # when building, don't use the install RPATH already
#    # (but later on when installing)
#    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
#
#    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
#
#    # add the automatically determined parts of the RPATH
#    # which point to directories outside the build tree to the install RPATH
#    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
#
#    # the RPATH to be used when installing, but only if it's not a files directory
#    list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
#    if ("${isSystemDir}" STREQUAL "-1")
#        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
#    endif ()
#endif ()

message("")
