# ProjectSetup.cmake
#
# NOTE: you must include this file BEFORE the project() command for it to have an effect.
#
# Check to make sure we aren't doing an in-source build, throw a fatal error if we are.
#
# Configure project build type (Release, Debug, etc) and provide a list of options to user
# in ccmake and cmake-gui. Don't allow user to select a build type that we don't support.
#
# Collect all build artifacts in standard subdirs (bin and lib), instead of having them
# scattered all over the build directory.
#
# Stores list of build types in PROJECTSETUP_BUILD_TYPES.
#
# # # # # # # # # # # #
# The MIT License (MIT)
#
# Copyright (c) 2019 Stephen Sorley
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# # # # # # # # # # # #
#
cmake_minimum_required(VERSION 3.14)

include_guard(DIRECTORY)

# Don't allow user to build in the source directory.
get_filename_component(srcdir "${CMAKE_SOURCE_DIR}" REALPATH)
get_filename_component(bindir "${CMAKE_BINARY_DIR}" REALPATH)
if(srcdir STREQUAL bindir)
    message(FATAL_ERROR "\
\n\
Don't run CMake in the source directory - please create a new build directory, and invoke cmake from there. \
To clean up after this build attempt, please run the following in your source directory:\
\n\
cmake -E remove -f CMakeCache.txt && cmake -E remove_directory CMakeFiles
    ")
endif()


# Define the list of build types. The first one in the list is the default build type.
set(build_types
    Release
    RelWithDebInfo
    Debug
)
set(PROJECTSETUP_BUILD_TYPES ${build_types}) # Fancy variable name for use outside this file.


# If the user requested a particular build type when they invoked CMake, and it's on the list, use that.
# Otherwise, use the default build type (first in the list).
#
# Note: the matching is case-insensitive - i.e., if the user passes in RELEASE, it will match "Release"
#       from the list, and the value of CMAKE_BUILD_TYPE after this file is run will be "Release".
list(GET build_types 0 selected_type)
if(CMAKE_BUILD_TYPE)
    string(TOLOWER "${CMAKE_BUILD_TYPE}" _val_lower)
    list(TRANSFORM build_types TOLOWER OUTPUT_VARIABLE _list_lower)
    list(FIND _list_lower "${_val_lower}" _idx)
    if(_idx GREATER -1)
        list(GET build_types ${_idx} selected_type)
    else()
        message(WARNING "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} is invalid, using default (${selected_type})")
    endif()
endif()

# Set CMAKE_BUILD_TYPE to selected type, and add help text and drop-down list entries for cmake-gui.
string(REPLACE ";" ", " build_types_help "${build_types}")
set(CMAKE_BUILD_TYPE "${selected_type}" CACHE STRING ${build_types_help} FORCE)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "${build_types}")


# Put build artifacts in standard subdirectories.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")


# Add any extra path hints to help the project() command find compiler executables.
list(APPEND CMAKE_PREFIX_PATH
    /usr/local/cuda # Default cuda install location when using nvidia's Linux .run installer.
)
