# Every single project-specific options.
# Options are CACHE variables (i.e. they are not updated if already set), so they can be
# set from the command line or the cmake-gui.

# =====================================================================================
# General options
# =====================================================================================

option(FASTFFT_ENABLE_WARNINGS "Enable compiler warnings" ON)
option(FASTFFT_ENABLE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(FASTFFT_ENABLE_LTO "Enable Interprocedural Optimization, aka Link Time Optimization (LTO)" OFF)
option(FASTFFT_ENABLE_PCH "Build using precompiled header to speed up compilation time in Debug mode" ON)
option(FASTFFT_BUILD_TESTS "Build unit tests" OFF)

# =====================================================================================
# Backend and Dependencies
# =====================================================================================

# CUDA
# ====

set(FASTFFT_CUDA_ARCH 70 CACHE STRING "Architectures to generate device code for. Default=70")
# Static is required for callbacks in the tests, I think callbacks are deprecated in CUDA > 11.4 TODO: confirm
option(FASTFFT_CUDA_USE_CUFFT_STATIC "Use the cuFFT static library instead of the shared ones" ON)

# CPU
# ===
# No threading yet, and will probably leave to the host program
#option(FASTFFT_ENABLE_OPENMP "Enable multithreading, using OpenMP, on the CPU backend" ON)

# FFTW (see noa/ext/fftw for more details):
# FFTW is only used as a reference for testing, however, this should be removed when reworking the tests b/c GPL3 TODO:
option(FASTFFT_FFTW_USE_EXISTING "Use the installed FFTW3 libraries. If OFF, the libraries are fetched from the web" ON)
option(FASTFFT_FFTW_USE_STATIC "Use the FFTW static libraries instead of the shared ones" OFF)