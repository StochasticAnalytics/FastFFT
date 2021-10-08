# AddQtPath.cmake
#
# Search common locations for Qt, and add the most recent version to CMAKE_MODULE_PATH if found.
#
# You still need to call find_package(Qt ...) after including this file. This file just tries to
# ensure that the find_package call will find the newest version of Qt that matches the compiler
# you're building with. This is entirely optional - if no matching version is found, no changes
# are made and find_package works just like it would if you weren't using this file.
#
# Example usage:
#
# include(AddQtPath)
#
# find_package(Qt5 REQUIRED COMPONENTS Widgets)
#
# add_executable(myapp WIN32
#     myapp.cpp
#     mydialog_box.cpp # cmake will detect included ui_*.h header files and run uic for us (AUTOUIC option below).
#     resources.qrc # cmake will run Qt's resource compiler (rcc) on any listed *.qrc files (AUTORCC option below).
# )
#
# target_link_libraries(myapp PRIVATE
#    Qt5::Widgets
# )
#
# set_target_properties(myapp PROPERTIES
#     AUTOMOC TRUE
#     AUTOUIC TRUE
#     AUTORCC TRUE
# )
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

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(bitness _64)
else()
    set(bitness)
endif()

if(CMAKE_HOST_WIN32)
    set(qtpath "C:/Qt")
    if(NOT EXISTS "${qtpath}")
        return()
    endif()

    # Each installed version of Qt will have a subdir with the version number as the name.
    # Ex: Qt v5.12.0 is installed into C:\Qt\5.12.0 on Windows.
    file(GLOB dirs RELATIVE "${qtpath}" "${qtpath}/*.*.*")

    # Find the latest version.
    set(max_ver)
    foreach(dir ${dirs})
        if(IS_DIRECTORY "${qtpath}/${dir}" AND dir MATCHES "[0-9]+\.[0-9]+\.[0-9]+")
            if((NOT max_ver) OR (dir VERSION_GREATER max_ver))
                set(max_ver "${dir}")
            endif()
        endif()
    endforeach()
    if(NOT max_ver)
        return()
    endif()
    string(APPEND qtpath "/${max_ver}")

    # Inside the dir for the latest version, get list of all subdirs - each subdir is an installation
    # for a different compiler/architecture. Looks like this: C:/Qt/5.12.0/msvc2017_64
    if(MSVC)
        set(guess)
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.00) # VS 2015, VS 2017 or newer
            set(guess "${qtpath}/msvc2017${bitness}")
            if(NOT EXISTS ${guess})
                set(guess "${qtpath}/msvc2015${bitness}")
                if(NOT EXISTS ${guess})
                    set(guess)
                endif()
            endif()
        endif()
        if(guess)
            list(APPEND CMAKE_PREFIX_PATH "${guess}")
        endif()

    # TODO: add MinGW, android options if we eventually need to support those.
    endif()

elseif(CMAKE_HOST_APPLE)
    #TODO: add MacOS/iOS support (if needed).
else() #Linux, BSD, Cygwin, etc.

    # Look for manual installations of Qt (prefer these to system installs).
    file(GLOB dirs "$ENV{HOME}/[Qq][Tt]*/*.*.*" /usr/local/[Qq][Tt]*/*.*.* /opt/[Qq][Tt]*/*.*.*)

    set(max_ver)
    set(max_ver_dir)
    foreach(dir ${dirs})
        if(IS_DIRECTORY "${dir}" AND dir MATCHES "[0-9]+\.[0-9]+\.[0-9]+")
            if((NOT max_ver) OR CMAKE_MATCH_0 VERSION_GREATER max_ver)
                set(max_ver     "${CMAKE_MATCH_0}")
                set(max_ver_dir "${dir}")
            endif()
        endif()
    endforeach()
    if(NOT max_ver_dir)
        return()
    endif()

    set(guess "${max_ver_dir}/gcc${bitness}")
    if(EXISTS "${guess}")
        list(APPEND CMAKE_PREFIX_PATH "${guess}")
    endif()
endif()
