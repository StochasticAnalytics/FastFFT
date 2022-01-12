## Build options

The build generator is `CMake >= 3.18`. Note that CMake has almost no dependencies and can be
installed without root privileges. If the CUDA backend is built (see below),
the `CUDA toolkit >= 11.0` is required. The supported C++ compilers are `gcc >= 9.3`
and `clang >= 10`.

### Environmental variables

- The FFTW3 libraries are required. Currently, fastfft cannot install it for you but CMake will try to
  find your install in the default locations. However, these variables are also available:
    - `FASTFFT_FFTW_LIBRARIES`: If set and not empty, the libraries are exclusively searched under this
      path.
    - `FASTFFT_FFTW_INCLUDE`: If set and not empty, the headers (i.e. `fftw3.h`) are exclusively
      searched under this path.

### CMake options

Options are CACHE variables (i.e. they are not updated if already set), so they can be set from the
command line or the [cmake-gui](https://cmake.org/cmake/help/latest/manual/cmake-gui.1.html).
Options should be prefixed with `-D` when passed through the command line.

__fastfft specific options:__

- `FASTFFT_ENABLE_WARNINGS`: Enable compiler warnings. `Default = ON`
- `FASTFFT_ENABLE_WARNINGS_AS_ERRORS`: Treat compiler warnings as errors. `Default = OFF`
- `FASTFFT_ENABLE_LTO`: Enable Interprocedural Optimization, aka Link Time Optimization (LTO)
  . `Default = OFF`
- `FASTFFT_ENABLE_PCH`: Build using precompiled header to speed up compilation time in Debug
  mode. `Default = ON`
- `FASTFFT_BUILD_TESTS`: Build the tests. The `fastfft::fastfft_tests` target will be available to the
  project. `Default = OFF`
- `FASTFFT_ENABLE_CUDA`: Generate and build the CUDA GPU backend. `Default = ON`
- `FASTFFT_CUDA_ARCH`: List of architectures to generate device code for. `Default = 70 75 80 86`.


__CMake useful options:__

- `CMAKE_BUILD_TYPE`: The build type. fastfft defaults to `Release`.
- `CMAKE_CXX_COMPILER`: The C++ compiler path can be specified to ensure that the correct host
  compiler is selected or if the compiler is not installed in the default paths and is not found by
  CMake.
- `CMAKE_CUDA_COMPILER`: The CUDA compiler path can be specified to ensure that the correct device
  compiler is selected or if the compiler is not installed in the default paths and is not found by
  CMake. fastfft only supports nvcc as CUDA compiler.
- `CMAKE_CUDA_HOST_COMPILER`: The host compiler path can be specified to ensure that the correct
  host compiler is selected or if the compiler is not installed in the default paths and is not
  found by CMake. If multiple versions of the C++ compiler are available, CMake may select the wrong
  executable. This should be equal to `CMAKE_CXX_COMPILER`.
- `CMAKE_INSTALL_PREFIX`: Install directory used by CMake.

_Note:_  `BUILD_SHARED_LIBS` is not supported. fastfft is a statically linked library

## Build and Install

To build and install the library, as a project of its own, the easiest is probably to use the
command line:

```shell
mkdir fastfft && cd fastfft                                 # (1)
git clone https://github.com/bHimes/FastFFT    # (2)
mkdir _build && cd _build                           # (3)
cmake -DCMAKE_INSTALL_PREFIX=../_install ../fastfft     # (4)
cmake --build . --target install                    # (5)
```

1. Create a directory where to put the source code as well as the build and install directory. We'll
   refer to this directory as `{installation home}`.
2. Once in `{installation home}`, clone the repository.
3. Create and go into the build directory. This has to be outside the source directory that was just
   cloned in, otherwise the project will through an error at you.
4. Sets the fastfft environmental variables if the defaults are not satisfying. Then configure and
   generate the project using CMake. It is usually useful to specify the install directory. This
   will set up fastfft, with its CUDA backend, as well as the tests and benchmarks, in
   `Release` mode. This behavior can be changed by passing the appropriate project options as
   specified in "CMake options". fastfft has a few dependencies (see `ext/`), most of which are entirely
   managed by CMake
   using [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html). Although it
   simplifies most workflows, this step requires a working internet connection to download the
   dependencies.
5. Once the generation is done, CMake can build and install the project.

In this example, the `{installation home}/_install_` directory is organized as followed:

- `lib` contains the library. In debug mode, the library is postfixed with `d`, e.g. `libfastfftd.a`.
- `lib/cmake/fastfft` contains the CMake project packaging files.
- `bin` contains the `fastfft_tests` and `fastfft_benchmarks` executables.
- `include` contains the headers. The directory hierarchy is identical to `src` with the addition of
  some generated headers, like `include/fastfft/Version.h`.

_Note:_ Alternatively, one can use an IDE supporting CMake projects, like CLion and create a new
project from an existing source.
_Note:_ An `install_manifest.txt` is generated, so `make unistall` can be ran to uninstall the
library.

## How to use the library, as dependency, in an external project?

If the library is installed,
[find_package](https://cmake.org/cmake/help/latest/command/find_package.html?highlight=find_package)
can be used. Otherwise, [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html)
is probably the easiest way to consume fastfft, which in its simplest form would look something like:

```cmake
# [...]

include(FetchContent)
FetchContent_Declare(
        fastfft
        GIT_REPOSITORY https://github.com/bHimes/FastFFT
        GIT_TAG v0.0.0
)
FetchContent_MakeAvailable(fastfft)

# [...]
```

The CMake project `fastfft` comes with three main targets; the library aliased to `fastfft::fastfft`, the test
executable, `fastfft::tests`, and the benchmark executable, `fastfft::banchmarks`.
