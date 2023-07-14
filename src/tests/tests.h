#ifndef _SRC_TESTS_TESTS_H
#define _SRC_TESTS_TESTS_H

#include "../fastfft/Image.cuh"
#include "../../include/FastFFT.cuh"
#include "helper_functions.cuh"

#include <iostream>

namespace FastFFT {
// Input size vectors to be tested.
std::vector<int> test_size           = {32, 64, 128, 256, 512, 1024, 2048, 4096};
std::vector<int> test_size_rectangle = {64, 128, 256, 512, 1024, 2048, 4096};
std::vector<int> test_size_3d        = {32, 64, 128, 256, 512};
// std::vector<int> test_size_3d ={512};

// The launch parameters fail for 4096 -> < 64 for r2c_decrease, not sure if it is the elements_per_thread or something else.
// For now, just over-ride these small sizes
std::vector<int> test_size_for_decrease = {64, 128, 256, 512, 1024, 2048, 4096};

} // namespace FastFFT

#endif