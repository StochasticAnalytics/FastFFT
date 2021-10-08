# InstallDeps.cmake
#
# Helper functions for installing library dependencies (and copying to the build dir on Windows).
#
# Accepts a list of paths and/or targets. Targets that are static libs or executables are silently
# ignored. Alias targets and import targets are handled naturally, and any interface link
# dependencies of interface targets are followed recursively and added as well.
#
# The "lib dest" and "runtime dest" parameters given to install_deps are allowed to contain
# generator expressions.
#
# On Windows, if given a path to an import library (.lib), this function will attempt to guess the
# path to the DLL file. If at all possible, please define import libraries as SHARED explicitly
# and provide dll path in LOCATION and import lib path in IMPORTED_IMPLIB, so that this guessing
# isn't necessary.
#
# On *nix, all symlinks are resolved before installing the file, so the name of the installed file
# will be the name of the physical file at the end of the symlink chain. If the file is an ELF
# shared library and SONAME is defined, a symlink will also be installed that points to the physical
# file and has the name stored in SONAME.
#
# For example, let's say "libz.so" is passed in, it's a symlink to "libz.so.1.2.11", and the SONAME
# stored in the ELF header is "libz.so.1". This function will install a file named "libz.so.1.2.11",
# and a symlink that points to that file named "libz.so.1".
#
# Functions:
# install_deps([lib dest] [runtime dest] [... lib files or import targets ...])
#     Install the given list of lib dependencies to CMAKE_INSTALL_PREFIX, alongside the project.
#     Can be passed either library paths or imported targets. On Windows, if given an import lib
#     path or an imported library of type UNKNOWN, it will try to guess the path to the DLL. If it
#     can't find the the DLL, it will throw an error.
#
#     By default, install_deps will also copy any DLL's to the build's runtime output directory
#     (CMAKE_RUNTIME_OUTPUT_DIR, usually set to [build dir]/bin). This behavior can be turned off
#     by setting INSTALL_DEPS_COPY_DLL to FALSE (see below).
#
# install_deps_system([lib dest] [runtime dest])
#     Installs any needed system dependencies (like Visual C++ runtime libraries). If OpenMP was
#     found by the project, any required OpenMP libraries are included.
#
#     Just like with install_deps, this will also copy any DLL's to the build directory when on
#     Windows. This behavior can be disabled by setting INSTALL_DEPS_COPY_DLL to FALSE.
#
# Extra variables that modify how the functions work:
# INSTALL_DEPS_COPY_DLL
#   If TRUE (the default), any DLL's installed with install_deps will also be copied to the build
#   directory (specifically, to CMAKE_RUNTIME_OUTPUT_DIR). Set to FALSE to disable DLL copies.
#
# INSTALL_DEPS_AUTO_MODE [NONE|{LIMITED}|ALL]
#   Determines how install_deps() will handle any additional dependencies listed in the
#   IMPORTED_LINK_DEPENDENT_LIBRARIES or INTERFACE_LINK_LIBRARIES properties of the import libs
#   that were explicitly passed in. Note that these properties are recursively expanded out to
#   their fullest extent (so deps of deps of deps can be pulled in).
#      NONE: only the libs explicitly passed into install_deps() will be installed.
#      LIMITED: any additional deps that are import libs with the same namespace as one of the
#               explicitly passed-in libraries are installed.
#      ALL: all additional deps that are import libs are installed, regardless of namespace.
#
#   For example, let's say you called install_deps(Qt5::Widgets), and Qt5::Widgets has Qt5::Core
#   and OpenSSL::OpenSSL listed in its INTERFACE_LINK_LIBRARIES property. Furthermore, let's say
#   OpenSSL::OpenSSL has ZLIB::ZLIB listed in its INTERFACE_LINK_LIBRARIES property.  So our
#   dependency graph looks like this:
#       Qt5::Widgets -> Qt5::Core
#                    -> OpenSSL::OpenSSL -> ZLIB::ZLIB
#   If mode is NONE, only Qt5::Widgets is installed.
#   If mode is LIMITED, Qt5::Widgets and Qt5::Core are installed.
#   If mode is ALL, Qt5::Widgets, Qt5::Core, OpenSSL::OpenSSL, and ZLIB::ZLIB are installed.
#
#   "LIMITED" is the default behavior.
#
# # # # # # # # # # # #
# This file was originally adapted from the mstdlib project (also MIT licensed), found here:
#   https://github.com/Monetra/mstdlib/blob/master/CMakeModules/InstallDepLibs.cmake
#   (Monetra Technologies, LLC)
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


# READELF
# used to extract SONAME from shared libs on ELF platforms.
find_program(READELF
    NAMES readelf
          elfdump  #on Solaris, "elfdump -d" gives very similar output to "readelf -d" on Linux.
    DOC "readelf (unix/ELF only)"
)
mark_as_advanced(FORCE READELF)

# DUMPBIN
# used to get DLL name from import lib on Windows.
# (optional - will try other heuristics to guess DLL when dumpbin not available, plus the whole
#  guessing routine is never even run if the DLL location is stored in the imported library target)
set(dumpbin_hints)
foreach(lang CXX C)
    if(CMAKE_${lang}_COMPILER)
        get_filename_component(dir "${CMAKE_${lang}_COMPILER}" DIRECTORY)
        list(APPEND dumpbin_hints "${dir}")
    endif()
endforeach()
find_program(DUMPBIN
    NAMES dumpbin
    HINTS ${dumpbin_hints}
    DOC "dumpbin (Windows only)"
)
mark_as_advanced(FORCE DUMPBIN)


# Helper function for _install_deps_internal: try to find .dll using path of an import lib (VS or MinGW).
function(_install_deps_get_dll_from_implib out_dll path)
    # Get directory containing import lib, and try to guess root dir of install.
    get_filename_component(imp_dir "${path}" DIRECTORY)
    string(REGEX REPLACE "/[/0-9x_-]*lib[/0-9x_-]*(/.*|$)" "" root_dir "${imp_dir}")

    # Get library name by removing .lib or .dll.a from extension.
    get_filename_component(imp_file "${path}" NAME)
    string(REGEX REPLACE "\.lib$" "" libname "${imp_file}")
    string(REGEX REPLACE "\.dll\.a$" "" libname "${libname}")

    # If dumpbin is available, run it on the import lib to determine the DLL name.
    set(dumpbin_name)
    if(DUMPBIN)
        execute_process(COMMAND ${DUMPBIN} /ARCHIVEMEMBERS ${path}
            OUTPUT_VARIABLE dumpbin_out
            RESULT_VARIABLE res
            ERROR_QUIET
            ENCODING        auto
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(res EQUAL 0)
            string(REGEX MATCH "[: \t]+([^:\n]+\.dll)" dumpbin_out "${dumpbin_out}")
            if(CMAKE_MATCH_1)
                set(dumpbin_name ${CMAKE_MATCH_1})
                string(REGEX REPLACE "[ \t]*/[0-9a-fA-F]*[ \t]*" "" dumpbin_name "${dumpbin_name}")
            endif()
        endif()
    endif()

    # Get alternate library names by removing lib prefix, and/or d, MT, MDd, etc.
    # (These are common suffixes that indicate which visual studio build flags were used).
    string(REGEX REPLACE "^lib" "" nolibname "${libname}")

    set(alt_names)
    string(REGEX REPLACE "M[dDtT]+$" "" alt_name "${libname}")
    list(APPEND alt_names ${alt_name})
    string(REGEX REPLACE "M[dDtT]+$" "" alt_name "${nolibname}")
    list(APPEND alt_names ${alt_name})
    string(REGEX REPLACE "[dD]$" "" alt_name "${libname}")
    list(APPEND alt_names ${alt_name})
    string(REGEX REPLACE "[dD]$" "" alt_name "${nolibname}")
    list(APPEND alt_names ${alt_name})

    string(TOLOWER "${libname}" libname_lower)
    string(TOUPPER "${libname}" libname_upper)

    string(MAKE_C_IDENTIFIER "${libname_upper}" clibname)

    # Make list of possible subdirectory names that might contain DLL's.
    set(suffixes)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        list(APPEND suffixes
            64 bin64 64/bin bin/64 lib64 64/lib lib/64
            x86_64 x86_64/bin bin/x86_64 x86_64/lib lib/x86_64
            x64 binx64 x64/bin bin/x64 libx64 x64/lib lib/x64
        )
    else()
        list(APPEND suffixes
            32 bin32 32/bin bin/32 lib32 32/lib lib/32
            x86 x86/bin bin/x86 x86/lib lib/x86
            Win32 binWin32 Win32/bin bin/Win32 libWin32 Win32/lib lib/Win32
        )
    endif()
    list(APPEND suffixes bin lib)

    # Ask CMake to search for the DLL.
    set(CMAKE_FIND_LIBRARY_SUFFIXES .dll)
    find_library(${clibname}_DLL
        NAMES           ${dumpbin_name} ${libname} ${libname_lower} ${libname_upper} ${nolibname} ${alt_names}
        HINTS           "${imp_dir}"
                        "${root_dir}"
        NO_DEFAULT_PATH
        PATH_SUFFIXES   ${suffixes}
    )
    mark_as_advanced(FORCE ${clibname}_DLL)

    # If found, set result in parent scope. Otherwise, send a fatal error message with instructions
    # on how to manually specify the DLL location.
    if(${clibname}_DLL)
        set(${out_dll} "${${clibname}_DLL}" PARENT_SCOPE)
    else()
        message(FATAL_ERROR
        "install_dep_libs() couldn't find DLL for given import lib \"${path}\" (set manually with -D${clibname}_DLL=...)"
        )
    endif()
endfunction()


# Helper function for _install_deps_internal: convert given list of libs into file paths.
function(_install_deps_get_paths_from_libs lib_dest runtime_dest component out_paths_name out_libs_name prefixes_name)
    set(automode LIMITED)
    if(INSTALL_DEPS_AUTO_MODE)
        string(TOUPPER "${INSTALL_DEPS_AUTO_MODE}" automode)
        if(NOT (INSTALL_DEPS_AUTO_MODE STREQUAL "NONE" OR INSTALL_DEPS_AUTO_MODE STREQUAL "ALL"))
            set(automode LIMITED)
        endif()
    endif()

    set(out_libs)
    foreach(lib IN LISTS ${out_libs_name})
        # Skip empty list elements, as well as "optimized" and "debug" keywords that might be in a <NAME>_LIBRARIES variable.
        if((NOT lib) OR (lib STREQUAL "optimized") OR (lib STREQUAL "debug"))
            continue()
        endif()

        if (TARGET "${lib}")
            # If this is an alias target, get the proper name of the target, then add the result back
            # onto the list of libs to process on the next invocation of this function.
            get_target_property(alias ${lib} ALIASED_TARGET)
            if(alias)
                list(APPEND out_libs "${alias}")
                continue()
            endif()

            get_target_property(is_imported ${lib} IMPORTED)

            # If this is an imported library, and it has dependencies listed using properties, add
            # them to the list of libs too. Will be processed on next invocation of this function.
            #
            # NOTE: this only adds deps that are import libs, filters out any paths or link flags.
            if(is_imported AND NOT automode STREQUAL "NONE")
                foreach(prop IMPORTED_LINK_DEPENDENT_LIBRARIES INTERFACE_LINK_LIBRARIES)
                    get_target_property(dep_libs ${lib} ${prop})
                    if(dep_libs)
                        foreach(lib ${dep_libs})
                            if(lib MATCHES "([^ \t:]+)::[^ \t]+" AND TARGET ${lib})
                                if(automode STREQUAL "ALL" OR "${CMAKE_MATCH_1}" IN_LIST ${prefixes_name})
                                    list(APPEND out_libs "${lib}")
                                endif()
                            endif()
                        endforeach()
                    endif()
                endforeach()
            endif()

            # If this target isn't a shared, module, interface, or unknown imported library target, skip it silently
            # without doing anything.
            get_target_property(type ${lib} TYPE)
            if(NOT type STREQUAL "SHARED_LIBRARY" AND
               NOT type STREQUAL "MODULE_LIBRARY" AND
               NOT type STREQUAL "INTERFACE_LIBRARY" AND
               NOT type STREQUAL "UNKNOWN_LIBRARY")  #UNKNOWN is a special type that only applies to import libraries.
                continue()
            endif()

            # If this target isn't imported, install directly if shared or module, then skip regardless of type.
            if(NOT is_imported)
                if(type STREQUAL "SHARED_LIBRARY" OR type STREQUAL "MODULE_LIBRARY")
                    install(TARGETS ${lib}
                        LIBRARY DESTINATION "${lib_dest}"
                        RUNTIME DESTINATION "${runtime_dest}"
                        ${component}
                    )
                endif()
                continue()
            endif()

            # If this was an imported interface library, nothing left to do after we read from the properties.
            if(type STREQUAL "INTERFACE_LIBRARY")
                continue()
            endif()

            # For shared/module/unknown imported libs, get the imported location (should be DLL, on Windows).
            # Add to list of paths.
            get_target_property(lib_path ${lib} LOCATION)
            if(WIN32 AND NOT lib_path)
                # If there's no known DLL, use the import lib as the path instead.
                get_target_property(lib_path ${lib} IMPORTED_IMPLIB)
            endif()
            if(NOT EXISTS "${lib_path}")
                message(FATAL_ERROR "Target ${lib} given to install_dep_libs() contained bad path ${lib_path}")
            endif()
            list(APPEND ${out_paths_name} "${lib_path}")
        elseif(lib)
            # Handling for if this lib is a file path.
            if (NOT EXISTS "${lib}")
                message(FATAL_ERROR "Path ${lib} given to install_dep_libs() was bad")
            endif()
            list(APPEND ${out_paths_name} "${lib}")
        endif()
    endforeach()

    set(${out_paths_name} "${${out_paths_name}}" PARENT_SCOPE)
    set(${out_libs_name} "${out_libs}" PARENT_SCOPE)
endfunction()


# Helper function for _install_deps_internal: if we're installing Qt runtime libraries, add extra
# libraries and plugins that those runtime libraries depend on.
#
# ARGN == list of qt import libs that the caller explicitly requested that we install.
function(_install_deps_get_qt_extra_paths lib_dest runtime_dest component out_paths_name)
    set(qt_import_libs ${ARGN})
    if((NOT qt_import_libs) OR (NOT TARGET Qt5::Core))
        return()
    endif()

    # Get root dir of Qt5 installation.
    #   Find absolute path to core library.
    get_target_property(qt5_root Qt5::Core LOCATION)
    #   Remove library file from path.
    get_filename_component(qt5_root "${qt5_root}" DIRECTORY)
    #   Remove last directory ("lib") from path.
    get_filename_component(qt5_root "${qt5_root}" DIRECTORY)

    # For each Qt lib we support, install any required plugins directly, and add any required
    # libraries to "to_add" (will get appended onto out_paths at bottom of function).
    set(to_add)

    if("Qt5::Core" IN_LIST qt_import_libs)
        # Install platform plugin. Don't add to out_paths, this has to go in a special subdir,
        # and it's guaranteed not to need symlinks.
        set(plugin)
        if(WIN32)
            set(plugin "qwindows.dll")
        elseif(NOT APPLE) # Linux/X11
            set(plugin "libqxcb.so")
        endif()

        if(plugin)
            set(plugin "${qt5_root}/plugins/platforms/${plugin}")
            if(EXISTS "${plugin}")
                install(FILES "${plugin}"
                    DESTINATION "${runtime_dest}/platforms"
                    ${component}
                )
            endif()
        endif()

        # Install any extra libs needed by Qt5::Core.
        if(NOT WIN32 AND NOT APPLE) # Linux
            # Linux XCB platform plugin requires a few additional libraries that aren't deps
            # of other modules.
            list(APPEND to_add
                libQt5DBus.so
                libQt5XcbQpa.so
            )

            # Qt can't rely on OS libs for internationalization on Linux, because they're not
            # standardized enough. So, Qt is built against one specific version of ICU that must
            # be bundled alongside its libraries. It has to be the exact version bundled with Qt.
            list(APPEND to_add
                libicui18n.so
                libicuuc.so
                libicudata.so
            )
        endif()
    endif()

    if("Qt5::PrintSupport" IN_LIST qt_import_libs)
        set(plugin)
        if(WIN32)
            set(plugin "windowsprintersupport.dll")
        elseif(NOT APPLE) # Linux/X11
            set(plugin "libcupsprintersupport.so")
        endif()

        if(plugin)
            set(plugin "${qt5_root}/plugins/printsupport/${plugin}")
            if(EXISTS "${plugin}")
                install(FILES "${plugin}"
                    DESTINATION "${runtime_dest}/printsupport"
                    ${component}
                )
            endif()
        endif()
    endif()

    # Add prefix to each lib name we want to add, then append to out_paths.
    list(TRANSFORM to_add PREPEND "${qt5_root}/lib/")
    foreach(lib IN LISTS to_add)
        if(EXISTS ${lib})
            list(APPEND ${out_paths_name} ${lib})
        endif()
    endforeach()

    # Push changes to out_paths up to caller's scope.
    set(${out_paths_name} "${${out_paths_name}}" PARENT_SCOPE)
endfunction()


# Helper function for _install_deps_internal: retrieve soname of given lib file. If no soname, returns the path.
function(_install_deps_read_soname outvarname path)
    # Set output variable to empty string - this is what will be returned on an error.
    set(${outvarname} "" PARENT_SCOPE)

    if(NOT READELF) # If this system doesn't provide the readelf command.
        return()
    endif()

    # Read the ELF header from the file.
    execute_process(COMMAND "${READELF}" -d ${path}
        RESULT_VARIABLE res
        OUTPUT_VARIABLE header
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT res EQUAL 0) # If the readelf command returned an error status code.
        return()
    endif()

    # Parse the SONAME out of the header.
    if(READELF MATCHES "readelf")
        # Linux (readelf) format:   0x000000000000000e (SONAME) Library soname: [libssl.so.1.0.0]
        if(NOT header MATCHES "\\(SONAME\\)[^\n]+\\[([^\n]+)\\]")
            return()
        endif()
    else()
        # Solaris (elfdump) format: [8] SONAME 0x49c1 libssl.so.1.0.0
        if(NOT header MATCHES "\\[[0-9]+\\][ \t]+SONAME[ \t]+[x0-9a-fA-F]+[ \t]+([^\n]+)")
            return()
        endif()
    endif()

    string(STRIP "${CMAKE_MATCH_1}" soname)
    set(${outvarname} "${soname}" PARENT_SCOPE)
endfunction()


# Helper function for install_deps.
# _install_deps_internal([lib dest] [runtime dest] [flag to turn file copy on/off] [flag to turn file install on/off] [... lib files or import targets ...]
function(_install_deps_internal lib_dest runtime_dest component do_copy do_install)
    if((NOT do_copy) AND (NOT do_install))
        return()
    endif()

    # Handle default destination for copied DLL's.
    set(copy_dest "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Construct list of library paths.
    set(lib_paths)
    if(NOT lib_dest)
        set(lib_dest lib)
    endif()
    if(NOT runtime_dest)
        set(runtime_dest bin)
    endif()

    set(libs ${ARGN})

    set(qt_import_libs)
    set(prefixes)
    foreach(lib IN LISTS libs)
        if(NOT TARGET ${lib})
            continue()
        endif()

        # If any Qt import libraries were passed in, save them in a separate list.
        if(lib MATCHES "Qt[0-9]+::")
            list(APPEND qt_import_libs "${lib}")
        endif()

        # Get list of import lib prefixes (used to support LIMITED auto mode).
        if(lib MATCHES "([^: \t]+)::[^ \t]+")
            list(APPEND prefixes "${CMAKE_MATCH_1}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES qt_import_libs)
    list(REMOVE_DUPLICATES prefixes)

    # Convert list of libraries into a list of library paths. Will resolve any alias libraries,
    # import libs, and import lib dependencies into full paths. Will also filter out unwanted
    # stuff like static library targets.
    while(libs)
        _install_deps_get_paths_from_libs("${lib_dest}" "${runtime_dest}" "${component}"
            lib_paths
            libs
            prefixes
        )
    endwhile()

    # If we're installing any Qt libs, add any extra plugins and libraries that are required for
    # them to work properly.
    _install_deps_get_qt_extra_paths("${lib_dest}" "${runtime_dest}" "${component}"
        lib_paths
        ${qt_import_libs}
    )

    # Remove any obvious duplicates. This won't catch different symlinks that refer to the same file,
    # or the same files installed on different calls to install_deps(). However, duplicates don't
    # actually hurt anything, they just make cmake's output during an install noisier to look at.
    # So, it's OK that we don't catch 100% of duplicates, since it's just to make things look nicer.
    #
    #   NOTE: if we wanted to catch 100% of duplicates, we could. It would just take more code.
    list(REMOVE_DUPLICATES lib_paths)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Process each library path, install to appropriate location (if not already installed).
    set(sonames)
    foreach(path IN LISTS lib_paths)
        # If this is an empty list element, skip it.
        if(NOT path)
            continue()
        endif()

        # If on Windows, try to replace import libraries with DLL's. Throws fatal error if it can't do it.
        if(WIN32 AND (${path} MATCHES "\.lib$" OR ${path} MATCHES "\.a$"))
           _install_deps_get_dll_from_implib(path "${path}")
        endif()

        # Resolve any symlinks in path to get actual physical name. If relative, evaluate relative to current binary dir.
        get_filename_component(path "${path}" REALPATH BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")

        # Figure out the destination (DLL's and manifest files go to runtime_dest, everything else goes to lib_dest).
        if(path MATCHES "\.[dD][lL][lL]$" OR path MATCHES "\.[mM][aA][nN][iI][fF][eE][sS][tT]$")
            set(dest "${runtime_dest}")
            if(path MATCHES "\.[dD][lL][lL]$")
                set(type PROGRAMS) # install DLL's as executable
            else()
                set(type FILES) # install manifests as a normal non-executable file
            endif()

            # If requested by caller, copy the DLL's to the build dir in addition to installing them.
            # If the file with the same name and timestamp already exists at the destination, nothing will be copied.
            if(do_copy)
                if(CMAKE_CONFIGURATION_TYPES)
                    foreach(conf ${CMAKE_CONFIGURATION_TYPES})
                        file(MAKE_DIRECTORY "${copy_dest}/${conf}")
                        file(COPY "${path}" DESTINATION "${copy_dest}/${conf}")
                    endforeach()
                else()
                    file(COPY "${path}" DESTINATION "${copy_dest}")
                endif()
            endif()
        else()
            set(dest "${lib_dest}")
            set(type FILES)
        endif()

        if(NOT do_install)
            continue()
        endif()

        # Install the file.
        #message(STATUS "will install ${path} to ${dest}")
        install(${type} "${path}" DESTINATION "${dest}" ${component})

        # If the library has a soname that's different than the actual name of the file on disk, add a symlink.
        _install_deps_read_soname(soname "${path}")
        if(soname)
            get_filename_component(libname "${path}" NAME)
            if(NOT soname STREQUAL libname)
                # Generate a relative-path symlink in the build dir (doesn't have to be valid).
                set(tmpdir "${CMAKE_CURRENT_BINARY_DIR}/dep-lib-links")
                file(MAKE_DIRECTORY "${tmpdir}")
                file(REMOVE "${tmpdir}/${soname}") # Remove any old symlink with the same name.
                execute_process(
                    COMMAND           ${CMAKE_COMMAND} -E create_symlink "${libname}" "${soname}"
                    WORKING_DIRECTORY "${tmpdir}"
                    RESULT_VARIABLE   res
                    ERROR_QUIET
                    OUTPUT_QUIET
                )

                if(NOT res EQUAL 0)
                    message(AUTHOR_WARNING "install_deplib: failed to create install symlink for \"${libname}\"")
                    continue()
                endif()

                # Install the symlink to the same directory as the library.
                #message(STATUS "will install ${tmpdir}/${soname} to ${dest}")
                install(${type} "${tmpdir}/${soname}" DESTINATION "${dest}" ${component})
            endif()
        endif()
    endforeach()
endfunction()


# install_deps([lib dest] [runtime dest] [... lib files or import targets ...]
function(install_deps lib_dest runtime_dest)
    # Handle default values for variables that control DLL copying.
    if(NOT DEFINED INSTALL_DEPS_COPY_DLL)
        set(INSTALL_DEPS_COPY_DLL TRUE)
    endif()

    set(libs ${ARGN})
    if(NOT libs)
        return()
    endif()

    # See if the user passed an optional "COMPONENT [component name]" to the install command.
    # If they did, remove those entries from the 'libs' list and add them to the 'component' list.
    set(component)
    list(FIND libs COMPONENT idx)
    if(idx GREATER -1)
        math(EXPR idx_next "${idx} + 1")
        list(GET libs ${idx} ${idx_next} component)
        list(REMOVE_AT libs ${idx} ${idx_next})
    endif()

    # Call internal helper
    _install_deps_internal("${lib_dest}" "${runtime_dest}" "${component}" ${INSTALL_DEPS_COPY_DLL} TRUE ${libs})
endfunction()


# install_deps_system([lib dest] [runtime dest])
function(install_deps_system lib_dest runtime_dest)
    if(OpenMP_FOUND)
        set(CMAKE_INSTALL_OPENMP_LIBRARIES TRUE)
    endif()
    # Install any required system libs, if any (usually just MSVC redistributables on windows).
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE) # tell module not to install, just save to CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS
    include(InstallRequiredSystemLibraries) # sets CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS

    install_deps("${lib_dest}" "${runtime_dest}" ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS})
endfunction()
