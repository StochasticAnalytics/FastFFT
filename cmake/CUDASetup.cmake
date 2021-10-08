# CUDASetup.cmake
#
# Adds various CUDA flags, including those which determine what architectures will be supported by
# the produced binaries. By default, this will produce a fat binary containing binary versions
# for each major supported architecture, plus a PTX version for the newest supported arch to allow
# the code to run on architectures that will be released in the future. The CUDA_MIN_ARCH and
# CUDA_MAX_ARCH options can be used to restrict the range of architectures, if desired.
#
# Searches for CUDA libraries (cublas, cusolver, etc.) from the CUDA toolkit, and provides imported
# targets for them if found.
#
# Imported libs (only most commonly used ones, see bottom section of file for exhaustive list):
#   CUDA::cudart
#   CUDA::cublas
#   CUDA::cufft
#   CUDA::curand
#   CUDA::cusparse
#   CUDA::cusolver
#   CUDA::npp{c,ial,icc,icom,idei,if,ig,im,ist,isu,itc,s}
#   CUDA::nvgraph
#   CUDA::nvjpeg
#
# If not on Windows, static versions of these import libraries are usually also available, just
# add '_static' suffix to name (e.g., CUDA::cublas_static).
#
# On Windows, as of CUDA 10 the only static lib available is CUDA::cudart_static, everything else
# is shared only.
#
# Options:
#   CUDA_MIN_ARCH: minimum architecture to support (default: minimum version allowed by CUDA toolkit)
#   CUDA_MAX_ARCH: maximum architecture to support (default: maximum version allowed by CUDA toolkit)
#   CUDASETUP_VERBOSE: set to TRUE to show messages useful for debugging problems (default: FALSE)
#
# WARNING:
# The lists of supported architectures must be manually updated whenever a new toolkit is released.
# You may also need to update the list of support libs at the bottom of the file, though these change
# less often.
#
# The oldest CUDA toolkit supported by this file: 9.0
#
# The newest CUDA toolkit available when last updated: 10.1 Update 1
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

if(NOT CMAKE_CUDA_COMPILER_VERSION)
    return()
endif()

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "CUDASetup only supports 64-bit builds.")
endif()



if(    CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
    # https://docs.nvidia.com/cuda/archive/11.0/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list
    set(arch_list
        70 # Volta
        75 # Turing
        80 # Ampere
    )
else()
    message(FATAL_ERROR "Current CUDA version (${CMAKE_CUDA_COMPILER_VERSION}) it too old, minimum allowed is 9.0")
endif()



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Code to handle arch selection, and any extra flags besides the CXX flags added in AddFlags.cmake

# Allow user to select minimum and maximum supported architectures.
# Default to covering the entire list of arch's supported by this version of CUDA.
list(GET arch_list 0  default_min_arch)
list(GET arch_list -1 default_max_arch)

set(CUDA_MIN_ARCH ${default_min_arch} CACHE STRING "Minimum CUDA arch to support")
set(CUDA_MAX_ARCH ${default_max_arch} CACHE STRING "Maximum CUDA arch to support")
set_property(CACHE CUDA_MIN_ARCH PROPERTY STRINGS "${arch_list}")
set_property(CACHE CUDA_MAX_ARCH PROPERTY STRINGS "${arch_list}")

if(NOT CUDA_MIN_ARCH IN_LIST arch_list)
    message(FATAL_ERROR "CUDA_MIN_ARCH (${CUDA_MIN_ARCH}) is not supported by CUDA ${CMAKE_CUDA_COMPILER_VERSION}.")
endif()
if(NOT CUDA_MAX_ARCH IN_LIST arch_list)
    message(FATAL_ERROR "CUDA_MAX_ARCH (${CUDA_MAX_ARCH}) is not supported by CUDA ${CMAKE_CUDA_COMPILER_VERSION}.")
endif()
if(CUDA_MIN_ARCH GREATER CUDA_MAX_ARCH)
    message(FATAL_ERROR "CUDA_MIN_ARCH (${CUDA_MIN_ARCH}) cannot be greater than CUDA_MAX_ARCH (${CUDA_MAX_ARCH})")
endif()

# Construct flags to pass to nvcc to produce code for the requested range of architectures.
foreach(arch ${arch_list})
    if(arch LESS "${CUDA_MIN_ARCH}")
        continue()
    endif()
    if(arch GREATER "${CUDA_MAX_ARCH}")
        break()
    endif()

    if(arch LESS "${CUDA_MAX_ARCH}")
        # If this isn't the newest arch, just compile binary GPU code for it.
        string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_${arch},code=sm_${arch}")
    else()
        # If this is the newest arch, compile binary GPU code for it, and also embed PTX code for it so
        # that the code will work on GPU arch's newer than our max supported one (uses JIT compilation).
        string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_${arch},code=[compute_${arch},sm_${arch}]")
    endif()
endforeach()

# Force nvcc to treat headers from CUDA include dir as system headers. If we don't do this, we get tons of
# spam warnings from CUDA's headers when building with newer GCC or Clang.
foreach(incdir ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    string(APPEND CMAKE_CUDA_FLAGS " -isystem \"${incdir}\"")
endforeach()



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Helper functions for finding CUDA libraries.

# Calculate which dirs to search for libs in, from the CUDA compiler and include dirs CMake found.
get_filename_component(bindir "${CMAKE_CUDA_COMPILER}" DIRECTORY)
string(REGEX REPLACE "/+[x648_]*bin[x648_]*(/.*|$)" "" rootdirs "${bindir}")
foreach(incdir ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    string(REGEX REPLACE "/include(/.*|$)" "" incdir "${incdir}")
    list(APPEND rootdirs "${incdir}")
endforeach()
list(REMOVE_DUPLICATES rootdirs)

# Find the shared CUDA library with the given core name (cublas, cufft, etc).
# If found, create an imported target for it, then add the core name to CUDA_all_libs_shared.
#   name: name of library, without lib prefix or suffix, and without 'CUDA::'.
#   ARGN: names of any additional variables to add the core name to (optional)
set(CUDA_all_libs_shared)
function(int_cudasetup_find_lib name)
    find_library(CUDA_${name}_LIBRARY ${name}
        HINTS         ${rootdirs}
        PATH_SUFFIXES lib/x64 lib64 lib
    )
    mark_as_advanced(FORCE CUDA_${name}_LIBRARY)
    if(NOT CUDA_${name}_LIBRARY)
        return()
    endif()

    if(WIN32)
        # On Windows, only use the library if we can also find the DLL (not just the import lib).
        if(NOT CUDA_${name}_DLL)
            set(glob_paths)
            foreach(dir ${rootdirs})
                list(APPEND glob_paths
                    "${dir}/bin/${name}64_*.dll" # Name scheme used in CUDA 9 and 10.
                    "${dir}/bin/${name}_*.dll"   # Alternate name scheme (for future-proofing).
                    "${dir}/bin/${name}.dll"     # Alternate name scheme (for future-proofing).
                )
            endforeach()

            file(GLOB dllfile LIST_DIRECTORIES FALSE ${glob_paths})
            if(dllfile)
                list(GET dllfile 0 dllfile)
            else()
                set(dllfile "CUDA_${name}_DLL-NOTFOUND")
                message(AUTHOR_WARNING
                    "Can't find DLL for ${CUDA_${name}_LIBRARY}, set CUDA_${name}_DLL to fix.")
            endif()
            set(CUDA_${name}_DLL "${dllfile}" CACHE FILEPATH "Path to DLL for ${name} library" FORCE)
            mark_as_advanced(FORCE CUDA_${name}_DLL)
        endif()
        if(CUDA_${name}_DLL)
            add_library(CUDA::${name} SHARED IMPORTED)
            set_target_properties(CUDA::${name} PROPERTIES
                IMPORTED_LOCATION             "${CUDA_${name}_DLL}"
                IMPORTED_IMPLIB               "${CUDA_${name}_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
            )
        endif()
    else()
        add_library(CUDA::${name} SHARED IMPORTED)
        set_target_properties(CUDA::${name} PROPERTIES
            IMPORTED_LOCATION             "${CUDA_${name}_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
        )
    endif()

    # If we successfully found the library, add the name to CUDA_all_libs_shared, and to any other
    # variable names that the caller passed in.
    if(TARGET CUDA::${name})
        foreach(varname CUDA_all_libs_shared ${ARGN})
            list(APPEND ${varname} ${name})
            set(${varname} "${${varname}}" PARENT_SCOPE)
        endforeach()
    endif()
endfunction()

# Find the static CUDA library with the given core name (cublas_static, cufft_static, etc).
# If found, create an imported target for it, then add the core name to CUDA_all_libs_static.
#   name: name of library, without lib prefix or suffix, and without 'CUDA::'.
#   ARGN: names of any additional variables to add the core name to (optional)
set(CUDA_all_libs_static)
function(int_cudasetup_find_lib_static name)
    set(CMAKE_FIND_LIBRARY_PREFIXES "${CMAKE_STATIC_LIBRARY_PREFIX}")
    set(CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_STATIC_LIBRARY_SUFFIX}")
    find_library(CUDA_${name}_LIBRARY ${name}
        HINTS         ${rootdirs}
        PATH_SUFFIXES lib/x64 lib64 lib
    )
    mark_as_advanced(FORCE CUDA_${name}_LIBRARY)
    if(NOT CUDA_${name}_LIBRARY)
        return()
    endif()

    add_library(CUDA::${name} STATIC IMPORTED)
    set_target_properties(CUDA::${name} PROPERTIES
        IMPORTED_LOCATION             "${CUDA_${name}_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    )

    # If we successfully found the library, add the name to CUDA_all_libs_static, and to any other
    # variable names that the caller passed in.
    if(TARGET CUDA::${name})
        foreach(varname CUDA_all_libs_static ${ARGN})
            list(APPEND ${varname} ${name})
            set(${varname} "${${varname}}" PARENT_SCOPE)
        endforeach()
    endif()
endfunction()

# For each of the given CUDA libraries, add all the given libraries as dependencies (link and install).
# Both NAMES and DEPS should be given CUDA library core names (cufft, cublas_static, etc).
#
# int_cudasetup_add_deps(NAMES <CUDA lib names> DEPS <dependencies (also CUDA lib names)>)
function(int_cudasetup_add_deps)
    cmake_parse_arguments(arg "" "" "NAMES;DEPS" ${ARGN})

    if((NOT arg_NAMES) OR (NOT arg_DEPS))
        return()
    endif()

    # Convert core libnames in both lists into their corresponding imported target names.
    list(TRANSFORM arg_NAMES PREPEND "CUDA::")
    list(TRANSFORM arg_DEPS PREPEND "CUDA::")

    # Filter out & ignore any bad names or libraries we couldn't find.
    set(names)
    foreach(name ${arg_NAMES})
        if(TARGET ${name})
            list(APPEND names ${name})
        endif()
    endforeach()

    # Filter out & ignore any bad names or libraries we couldn't find. Also, separate shared and
    # static dependencies out into their own lists (they're handled differently).
    set(static_deps)
    set(shared_deps)
    foreach(dep ${arg_DEPS})
        if(TARGET ${dep})
            get_target_property(type ${dep} TYPE)
            if(type STREQUAL "SHARED_LIBRARY")
                list(APPEND shared_deps ${dep})
            else()
                list(APPEND static_deps ${dep})
            endif()
        endif()
    endforeach()

    # Shared library dependencies get added to IMPORTED_LINK_DEPENDENT_LIBRARIES - not linked to
    # directly, but will help CMake determine proper RPATH's, and will cause the dependency to be
    # automatically installed by InstallDeps.cmake.
    if(shared_deps)
        set_property(TARGET ${names} APPEND PROPERTY
            IMPORTED_LINK_DEPENDENT_LIBRARIES ${shared_deps}
        )
    endif()

    # Static library dependencies get added to INTERFACE_LINK_LIBRARIES, so they're explicitly
    # linked right after the library that depends on them. InstallDeps.cmake will see these
    # dependencies, but will leave them out of the install because they're static libraries.
    if(static_deps)
        set_property(TARGET ${names} APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES ${static_deps}
        )
    endif()

    # For debugging:
    if(CUDASETUP_VERBOSE)
        message(STATUS "${names} deps_shared=${shared_deps} deps_static=${static_deps}")
    endif()
endfunction()



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Find CUDA runtime libraries.

# Shared CUDA runtime. No real reason to use this, the static version should be preferred. One
# possible use is if you are linking to a third-party (not nvidia) CUDA library that was built
# against the shared runtime, and you need to install the shared runtime as part of deployment.
int_cudasetup_find_lib(cudart)

# Static CUDA runtime. Only need to explicitly link to this if you're calling CUDA API or library
# functions from a pure C++ library or application. If at least one file in the target is a CUDA
# file, the CUDA runtime will be pulled in automatically for you, and you don't need this target.
int_cudasetup_find_lib_static(cudart_static)
if(TARGET CUDA::cudart_static AND NOT WIN32)
    # Need to explicitly link to a few extra system libraries on Linux.
    set(extras)
    if(NOT APPLE)
        set(extras -lrt)
    endif()
    set_property(TARGET CUDA::cudart_static APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES -lpthread -ldl ${extras}
    )
endif()



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Find CUDA support libraries (stuff like cuBLAS, cuFFT, etc).

# CU libraries (these are the most commonly used ones).
int_cudasetup_find_lib(cublas)
int_cudasetup_find_lib(cublasLt)
int_cudasetup_find_lib(cufft)
int_cudasetup_find_lib(cufftw)
int_cudasetup_find_lib(curand)
int_cudasetup_find_lib(cusolver)
int_cudasetup_find_lib(cusparse)

int_cudasetup_add_deps(NAMES cublas DEPS cublasLt)
int_cudasetup_add_deps(NAMES cufftw DEPS cufft)

# NPP (NVIDIA Performance Primitives)
set(needc)
int_cudasetup_find_lib(nppc)
int_cudasetup_find_lib(nppial  needc)
int_cudasetup_find_lib(nppicc  needc)
int_cudasetup_find_lib(nppicom needc)
int_cudasetup_find_lib(nppidei needc)
int_cudasetup_find_lib(nppif   needc)
int_cudasetup_find_lib(nppig   needc)
int_cudasetup_find_lib(nppim   needc)
int_cudasetup_find_lib(nppist  needc)
int_cudasetup_find_lib(nppisu  needc)
int_cudasetup_find_lib(nppitc  needc)
int_cudasetup_find_lib(npps    needc)

int_cudasetup_add_deps(NAMES ${needc} DEPS nppc)

# NV libraries (additional libs for very specialized uses)
int_cudasetup_find_lib(nvblas)
int_cudasetup_add_deps(NAMES nvblas DEPS cublas)

int_cudasetup_find_lib(nvgraph)
set(deps curand cusolver)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 10.1)
    list(APPEND deps cublas cusparse)
endif()
int_cudasetup_add_deps(NAMES nvgraph DEPS ${deps})

int_cudasetup_find_lib(nvjpeg)
int_cudasetup_find_lib(nvToolsExt)


# Static versions of everything (not available on Windows as of CUDA 10.1)
set(needos) #add libs to here that need to be linked to the culibos static library.

# CU static libs
set(solver_deps)
int_cudasetup_find_lib_static(cublasLt_static         needos)
int_cudasetup_find_lib_static(cublas_static           needos solver_deps)
int_cudasetup_find_lib_static(cufft_static            needos)
int_cudasetup_find_lib_static(cufft_static_nocallback needos)
int_cudasetup_find_lib_static(cufftw_static           needos)
int_cudasetup_find_lib_static(curand_static           needos)
int_cudasetup_find_lib_static(cusolver_static         needos)
int_cudasetup_find_lib_static(cusparse_static         needos solver_deps)
# -- only needed for cusolver_static
int_cudasetup_find_lib_static(lapack_static                  solver_deps)
int_cudasetup_find_lib_static(metis_static                   solver_deps)

int_cudasetup_add_deps(NAMES cublas_static   DEPS cublasLt_static)
int_cudasetup_add_deps(NAMES cufftw_static   DEPS cufft_static)
int_cudasetup_add_deps(NAMES cusolver_static DEPS ${solver_deps})

# NPP static libs
set(needc)
int_cudasetup_find_lib_static(nppc_static    needos)
int_cudasetup_find_lib_static(nppial_static  needc)
int_cudasetup_find_lib_static(nppicc_static  needc)
int_cudasetup_find_lib_static(nppicom_static needc)
int_cudasetup_find_lib_static(nppidei_static needc)
int_cudasetup_find_lib_static(nppif_static   needc)
int_cudasetup_find_lib_static(nppig_static   needc)
int_cudasetup_find_lib_static(nppim_static   needc)
int_cudasetup_find_lib_static(nppist_static  needc)
int_cudasetup_find_lib_static(nppisu_static  needc)
int_cudasetup_find_lib_static(nppitc_static  needc)
int_cudasetup_find_lib_static(npps_static    needc)

int_cudasetup_add_deps(NAMES ${needc} DEPS nppc_static)

# NV static libs
int_cudasetup_find_lib(nvgraph_static)
set(deps curand_static cusolver_static)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 10.1)
    list(APPEND deps cublas_static cusparse_static)
endif()
int_cudasetup_add_deps(NAMES nvgraph_static DEPS ${deps})

int_cudasetup_find_lib(nvjpeg_static)
int_cudasetup_add_deps(NAMES nvjpeg_static DEPS cudart_static)

# Add culibos to every static lib that needs it (MUST BE LAST)
int_cudasetup_find_lib_static(culibos) # common dependency of almost all toolkit static libs
int_cudasetup_add_deps(NAMES ${needos} DEPS culibos)

# For debugging:
if(CUDASETUP_VERBOSE)
    message(STATUS "CUDA_all_libs_shared = ${CUDA_all_libs_shared}")
    message(STATUS "CUDA_all_libs_static = ${CUDA_all_libs_static}")
endif()
