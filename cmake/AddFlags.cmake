# AddFlags.cmake
#
# Adds project-wide compiler and linker flags.
#
# Options:
#   ADDFLAGS_STRICT_WARNINGS: display type conversion and other verbose warnings (defaults to TRUE)
#   ADDFLAGS_HARDEN: add flags to security harden the code (defaults to TRUE)
#   ADDFLAGS_IPO: add flags to enable interprocedural optimization on release builds (defaults to FALSE)
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

include("${CMAKE_CURRENT_LIST_DIR}/AddFlagsHelpers.cmake")


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# User-configurable options:

option(ADDFLAGS_STRICT_WARNINGS "Display type conversion and other verbose warnings?"        TRUE)
option(ADDFLAGS_HARDEN          "Security harden the code (stack protector, etc)?"           TRUE)
option(ADDFLAGS_IPO             "Enable interprocedural optimization (release builds only)?" FALSE)

mark_as_advanced(FORCE
    ADDFLAGS_HARDEN #Hide by default, only allow advanced users to disable security hardening.
)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Add flags through platform-agnostic CMake interface:

# Don't relink executables when shared libraries they depend on change - it's not necessary.
set(CMAKE_LINK_DEPENDS_NO_SHARED TRUE)

# Hide symbols by default on all platforms (not just on Windows), instead of exposing everything.
get_property(enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
foreach(lang ${enabled_languages})
    set(CMAKE_${lang}_VISIBILITY_PRESET hidden)
endforeach()
set(CMAKE_VISIBILITY_INLINES_HIDDEN TRUE)

# Always compile everything as PIE/PIC (for security, and so that static libs can be used as inputs
# when building shared libs).
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
# Must call check_pie_supported in order for any extra required linker flags to be added to PIE executables.
include(CheckPIESupported)
check_pie_supported()

# Add link flags to set RPATH (on platforms that support it), so that installed components
# can find their bundled libraries in the install package.
#
# Will look first in same directory as component, then "lib" child dir, then "lib" sibling dir.
set(CMAKE_INSTALL_RPATH "\$ORIGIN/:\$ORIGIN/lib/:\$ORIGIN/../lib")

# Use interprocedural optimization for release builds, if supported by toolchain.
if(ADDFLAGS_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT res OUTPUT out)
    if(res)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()
endif()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Add platform-specific flags:

if(MSVC)
    # Visual Studio
    _int_add_flags_compiler(LANGS C CXX FLAGS
        /W3
        /we4013 # Treat "function undefined, assuming extern returning int" warning as an error.
    )

    # If we're not building in strict mode, remove some warnings to reduce noise level.
    if(NOT ADDFLAGS_STRICT_WARNINGS)
        _int_add_flags_compiler(LANGS C CXX FLAGS
            /wd4018 # Disable signed/unsigned mismatch warnings
            /wd4068 # Disable unknown pragma warnings
            /wd4244 # Disable integer type conversion warnings
            /wd4267 # Disable warnings about converting size_t to a smaller type
        )
    endif()

    # Note: the useful hardening flags are all enabled by default on modern versions of Visual Studio.

else()
    # GCC, Clang, Intel, etc.
    _int_add_flags_compiler(LANGS C CXX FLAGS
        # Warning flags.
        -Wall
        -Wextra

        -Wcast-align
        -Wformat-security
        -Wlogical-op
        -Wmissing-declarations
        -Wmissing-format-attribute
        -Wmissing-include-dirs
        -Wpointer-arith
        -Wredundant-decls
        -Wshadow
        -Wundef
        -Wunused
        -Wvla
    )

    _int_add_flags_compiler(LANGS C FLAGS
        -Winit-self
        -Wjump-misses-init
        -Wmissing-prototypes
        -Wnested-externs
        -Wold-style-definition
        -Wstrict-prototypes

        -Werror=implicit-int
        -Werror=implicit-function-declaration
    )

    if(ADDFLAGS_STRICT_WARNINGS)
        _int_add_flags_compiler(LANGS C CXX FLAGS
            -Wconversion
            -Wsign-conversion
        )
    else()
        _int_add_flags_compiler(LANGS C CXX FLAGS
            -Wno-unused-parameter
        )
    endif()

    # Set additional hardening flags, if requested.
    if(ADDFLAGS_HARDEN)
        _int_add_flag_options_compiler(LANGS C CXX FLAG_OPTIONS
            -fstack-protector-strong                      # Preferred, but only supported on GCC 4.9 or newer.
            "-fstack-protector --param=ssp-buffer-size=4" # Fall back to older flag if strong protector isn't supported.
        )

        add_compile_definitions(
            _FORTIFY_SOURCE=2 # Adds additional compile-time and runtime buffer overflow checks.
        )

        _int_add_flags_linker(FLAGS
            # Allow undefined symbols in shared libraries (usually the default, anyway).
            "-Wl,--allow-shlib-undefined"
            # Prune out shared libraries that we link against, but don't actually use.
            "-Wl,--as-needed"
            # Ensure that stack is marked non-executable (same thing as DEP in Visual Studio)
            "-Wl,-z,noexecstack"
            # Enable full RELRO (read-only relocations) - hardens ELF data sections and the GOT (global offsets table)
            # http://tk-blog.blogspot.com/2009/02/relro-not-so-well-known-memory.html
            "-Wl,-z,relro,-z,now"
        )
    endif()

    # Force colorized output, even when output has been redirected via pipe.
    _int_add_flag_options_compiler(LANGS C CXX FLAG_OPTIONS
        -fdiagnostics-color=always # GCC
        -fcolor-diagnostics        # Clang
    )
endif()

# Add extra definitions if we're targeting Windows.
if(WIN32)
    add_compile_definitions(
        # Suppress annoying deprecation warnings about standard C/C++ functions.
        _CRT_SECURE_NO_DEPRECATE
        _CRT_NONSTDC_NO_DEPRECATE
        # Leave out little-used parts of Windows.h (can still get them by including the specific headers you need).
        WIN32_LEAN_AND_MEAN
        VC_EXTRALEAN
        NOMINMAX
        # Explicitly request Windows 7 or newer feature level.
        _WIN32_WINNT=0x0601
    )
endif()
