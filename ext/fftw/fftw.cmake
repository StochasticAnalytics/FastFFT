# Find the necessary FFTW libraries.
message(STATUS "fftw3: searching for existing libraries...")
find_package(FFTW)

# Note: FetchContent is not very practical with non-CMake projects.
#       FFTW added CMake support in 3.3.7 but it seems to be experimental even in 3.3.8.
# TODO: Add support for FetchContent or ExternalProject_Add.

message("")
