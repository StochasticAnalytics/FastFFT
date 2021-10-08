# IDESetup.cmake
#
# NOTE: you must include this file AFTER the project() command for it to have an effect.
#
# Perform any setup specific to IDE's (e.g., Visual Studio, XCode). This involves setting
# the list of configurations and any other variables that affect the look-and-feel inside
# an IDE.
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

# If we aren't targeting a multi-config generator like Visual Studio or XCode, return without
# doing anything.
get_property(int_idesetup_is_multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT int_idesetup_is_multi)
    return()
endif()

# If we have a list of build types from an earlier include of ProjectSetup.cmake, tell the IDE
# to only select from that list of build types.
if(PROJECTSETUP_BUILD_TYPES)
    set(CMAKE_CONFIGURATION_TYPES "${PROJECTSETUP_BUILD_TYPES}" CACHE STRING "Build types available to use in IDE" FORCE)
endif()

# Multi-config generators ignore CMAKE_BUILD_TYPE, so remove it from the cache. This prevents the
# user from thinking they can change it through CMake.
unset(CMAKE_BUILD_TYPE)
unset(CMAKE_BUILD_TYPE CACHE)
