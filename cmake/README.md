# CMake-Common

### Project Overview
Contains CMake modules that are shared between my various projects.

These CMake modules contain both common stuff like helpers for setting warning
flags and installing/packaging dependencies, and specialized stuff to handle
specific dependencies like Qt and CUDA.

Requires CMake v3.14 or newer. Tested on Visual Studio 2017 (Windows 10) and
GCC 7.4 (Ubuntu 18.04/WSL).

Currently has no support for macOS-specific packaging (bundles and frameworks),
though this might change if I figure out a way to test it without having to buy
Apple hardware myself.
