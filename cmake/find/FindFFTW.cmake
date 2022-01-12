# Find the FFTW library (single, double precision versions, with thread support.).
# This should work both on Linux and Windows.
#
# The following variables will be used:
#   FASTFFT_FFTW_USE_STATIC:      If true, only static libraries are found, otherwise both static and shared.
#   (ENV) FASTFFT_FFTW_LIBRARIES: If set and not empty, the libraries are exclusively searched under this path.
#   (ENV) FASTFFT_FFTW_INCLUDE:   If set and not empty, the headers (i.e. fftw3.h) are exclusively searched under this path.
#
# The following targets are created:
#   fftw3::float
#   fftw3::double
#   fftw3::float_threads    (can be "empty", since fftw3::float can already include the thread support)
#   fftw3::double_threads   (can be "empty", since fftw3::double can already include the thread support)
#   fftw3::float_omp
#   fftw3::double_omp
#
# The following variables are set:
#   FASTFFT_FFTW_FLOAT_FOUND
#   FASTFFT_FFTW_DOUBLE_FOUND
#   FASTFFT_FFTW_FLOAT_THREADS_FOUND
#   FASTFFT_FFTW_DOUBLE_THREADS_FOUND
#   FASTFFT_FFTW_FLOAT_OPENMP_FOUND
#   FASTFFT_FFTW_DOUBLE_OPENMP_FOUND

# Whether to search for static or dynamic libraries.
set(FASTFFT_CMAKE_FIND_LIBRARY_SUFFIXES_OLD ${CMAKE_FIND_LIBRARY_SUFFIXES})
if (FASTFFT_FFTW_USE_STATIC)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
endif ()

if (DEFINED ENV{FASTFFT_FFTW_LIBRARIES} AND NOT $ENV{FASTFFT_FFTW_LIBRARIES} STREQUAL "")
    find_library(
            FASTFFT_FFTW_FLOAT_LIB
            NAMES "fftw3f" libfftw3f-3
            PATHS $ENV{FASTFFT_FFTW_LIBRARIES}
            PATH_SUFFIXES "lib" "lib64"
            NO_DEFAULT_PATH
    )
    find_library(
            FASTFFT_FFTW_FLOAT_THREADS_LIB
            NAMES "fftw3f_threads"
            PATHS $ENV{FASTFFT_FFTW_LIBRARIES}
            PATH_SUFFIXES "lib" "lib64"
            NO_DEFAULT_PATH
    )
    find_library(
            FASTFFT_FFTW_FLOAT_OPENMP_LIB
            NAMES "fftw3f_omp"
            PATHS $ENV{FASTFFT_FFTW_LIBRARIES}
            PATH_SUFFIXES "lib" "lib64"
            NO_DEFAULT_PATH
    )
    find_library(
            FASTFFT_FFTW_DOUBLE_LIB
            NAMES "fftw3" libfftw3-3
            PATHS $ENV{FASTFFT_FFTW_LIBRARIES}
            PATH_SUFFIXES "lib" "lib64"
            NO_DEFAULT_PATH
    )
    find_library(
            FASTFFT_FFTW_DOUBLE_THREADS_LIB
            NAMES "fftw3_threads"
            PATHS $ENV{FASTFFT_FFTW_LIBRARIES}
            PATH_SUFFIXES "lib" "lib64"
            NO_DEFAULT_PATH
    )
    find_library(
            FASTFFT_FFTW_DOUBLE_OPENMP_LIB
            NAMES "fftw3_omp"
            PATHS $ENV{FASTFFT_FFTW_LIBRARIES}
            PATH_SUFFIXES "lib" "lib64"
            NO_DEFAULT_PATH
    )
else ()
    find_library(
            FASTFFT_FFTW_DOUBLE_LIB
            NAMES "fftw3"
            PATHS ${LIB_INSTALL_DIR}
    )
    find_library(
            FASTFFT_FFTW_DOUBLE_THREADS_LIB
            NAMES "fftw3_threads"
            PATHS ${LIB_INSTALL_DIR}
    )
    find_library(
            FASTFFT_FFTW_DOUBLE_OPENMP_LIB
            NAMES "fftw3_omp"
            PATHS ${LIB_INSTALL_DIR}
    )
    find_library(
            FASTFFT_FFTW_FLOAT_LIB
            NAMES "fftw3f"
            PATHS ${LIB_INSTALL_DIR}
    )
    find_library(
            FASTFFT_FFTW_FLOAT_THREADS_LIB
            NAMES "fftw3f_threads"
            PATHS ${LIB_INSTALL_DIR}
    )
    find_library(
            FASTFFT_FFTW_FLOAT_OPENMP_LIB
            NAMES "fftw3f_omp"
            PATHS ${LIB_INSTALL_DIR}
    )
endif ()

if (DEFINED ENV{FASTFFT_FFTW_INCLUDE} AND NOT $ENV{FASTFFT_FFTW_INCLUDE} STREQUAL "")
    find_path(FASTFFT_FFTW_INC
            NAMES "fftw3.h"
            PATHS $ENV{FASTFFT_FFTW_INCLUDE}
            PATH_SUFFIXES "include"
            NO_DEFAULT_PATH
            )
else ()
    find_path(FASTFFT_FFTW_INC
            NAMES "fftw3.h"
            PATHS ${INCLUDE_INSTALL_DIR}
            )
endif ()

# Reset to whatever it was:
set(CMAKE_FIND_LIBRARY_SUFFIXES ${FASTFFT_CMAKE_FIND_LIBRARY_SUFFIXES_OLD})

# Logging:
message(STATUS "FASTFFT_FFTW_FLOAT_LIB: ${FASTFFT_FFTW_FLOAT_LIB}")
message(STATUS "FASTFFT_FFTW_DOUBLE_LIB: ${FASTFFT_FFTW_DOUBLE_LIB}")
message(STATUS "FASTFFT_FFTW_DOUBLE_OPENMP_LIB: ${FASTFFT_FFTW_DOUBLE_OPENMP_LIB}")
message(STATUS "FASTFFT_FFTW_FLOAT_OPENMP_LIB: ${FASTFFT_FFTW_FLOAT_OPENMP_LIB}")
message(STATUS "FASTFFT_FFTW_FLOAT_THREADS_LIB: ${FASTFFT_FFTW_FLOAT_THREADS_LIB}")
message(STATUS "FASTFFT_FFTW_DOUBLE_THREADS_LIB: ${FASTFFT_FFTW_DOUBLE_THREADS_LIB}")

# Targets:
if (FASTFFT_FFTW_FLOAT_LIB)
    set(FASTFFT_FFTW_FLOAT_LIB_FOUND TRUE)
    add_library(fftw3::float INTERFACE IMPORTED)
    set_target_properties(fftw3::float
            PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FASTFFT_FFTW_INC}"
            INTERFACE_LINK_LIBRARIES "${FASTFFT_FFTW_FLOAT_LIB}"
            )
else ()
    set(FASTFFT_FFTW_FLOAT_LIB_FOUND FALSE)
    add_library(fftw3::float INTERFACE)
endif ()

if (FASTFFT_FFTW_DOUBLE_LIB)
    set(FASTFFT_FFTW_DOUBLE_LIB_FOUND TRUE)
    add_library(fftw3::double INTERFACE IMPORTED)
    set_target_properties(fftw3::double
            PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${FASTFFT_FFTW_INC}"
            INTERFACE_LINK_LIBRARIES "${FASTFFT_FFTW_DOUBLE_LIB}"
            )
else ()
    set(FASTFFT_FFTW_DOUBLE_LIB_FOUND FALSE)
    add_library(fftw3::double INTERFACE)
endif ()

if (FASTFFT_FFTW_FLOAT_OPENMP_LIB)
    set(FASTFFT_FFTW_FLOAT_OPENMP_LIB_FOUND TRUE)
    add_library(fftw3::float_omp INTERFACE IMPORTED)
    set_target_properties(fftw3::float_omp
            PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FASTFFT_FFTW_INC}"
            INTERFACE_LINK_LIBRARIES "${FASTFFT_FFTW_FLOAT_OPENMP_LIB}"
            )
else ()
    set(FASTFFT_FFTW_FLOAT_OPENMP_LIB_FOUND FALSE)
    add_library(fftw3::float_omp INTERFACE)
endif ()

if (FASTFFT_FFTW_DOUBLE_OPENMP_LIB)
    set(FASTFFT_FFTW_DOUBLE_OPENMP_LIB_FOUND TRUE)
    add_library(fftw3::double_omp INTERFACE IMPORTED)
    set_target_properties(fftw3::double_omp
            PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FASTFFT_FFTW_INC}"
            INTERFACE_LINK_LIBRARIES "${FASTFFT_FFTW_DOUBLE_OPENMP_LIB}"
            )
else ()
    set(FASTFFT_FFTW_DOUBLE_OPENMP_LIB_FOUND FALSE)
    add_library(fftw3::double_omp INTERFACE)
endif ()

if (FASTFFT_FFTW_FLOAT_THREADS_LIB)
    set(FASTFFT_FFTW_FLOAT_THREADS_LIB_FOUND TRUE)
    add_library(fftw3::float_threads INTERFACE IMPORTED)
    set_target_properties(fftw3::float_threads
            PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FASTFFT_FFTW_INC}"
            INTERFACE_LINK_LIBRARIES "${FASTFFT_FFTW_FLOAT_THREADS_LIB}"
            )
else ()
    set(FASTFFT_FFTW_FLOAT_THREADS_LIB_FOUND FALSE)
    add_library(fftw3::float_threads INTERFACE)
endif ()

if (FASTFFT_FFTW_DOUBLE_THREADS_LIB)
    set(FASTFFT_FFTW_DOUBLE_THREADS_LIB_FOUND TRUE)
    add_library(fftw3::double_threads INTERFACE IMPORTED)
    set_target_properties(fftw3::double_threads
            PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${FASTFFT_FFTW_INC}"
            INTERFACE_LINK_LIBRARIES "${FASTFFT_FFTW_DOUBLE_THREADS_LIB}"
            )
else ()
    set(FASTFFT_FFTW_DOUBLE_THREADS_LIB_FOUND FALSE)
    add_library(fftw3::double_threads INTERFACE)
endif ()

if (FASTFFT_FFTW_INC)
    message(STATUS "FASTFFT_FFTW_INC: ${FASTFFT_FFTW_INC}")
else ()
    message(FATAL_ERROR "Could not find the fftw3.h header on the system.")
endif ()

mark_as_advanced(
        FASTFFT_FFTW_DOUBLE_LIB
        FASTFFT_FFTW_FLOAT_LIB
        FASTFFT_FFTW_DOUBLE_THREADS_LIB
        FASTFFT_FFTW_FLOAT_THREADS_LIB
        FASTFFT_FFTW_DOUBLE_OPENMP_LIB
        FASTFFT_FFTW_FLOAT_OPENMP_LIB
        FASTFFT_FFTW_INC
)
