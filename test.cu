#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

// sudo apt-get install libfftw3-dev libfftw3-doc
#include <fftw3.h>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

#include "FastFFT.cu"


int main(int, char**) {
  std::printf("Entering main in tests.cpp\n");

  std::printf("Standard is %i\n",__cplusplus);
  float* real_values;
  float2* complex_values;
  int real_memory_allocated;
  int wanted_x_size = 64;
  int wanted_y_size = 1;
  int wanted_z_size = 1;

  	// first_x_dimension
	if (wanted_x_size % 2 == true) real_memory_allocated =  wanted_x_size / 2 + 1;
	else real_memory_allocated = (wanted_x_size - 1) / 2 + 1;

	real_memory_allocated *= wanted_y_size * wanted_z_size; // other dimensions
	real_memory_allocated *= 2; // room for complex

//  real_values = new float[real_memory_allocated];
	real_values = (float *) fftwf_malloc(sizeof(float) * real_memory_allocated);
	complex_values = (float2*) real_values;  // Set the complex_values to point at the newly allocated real values;

  fftwf_plan plan_fwd = NULL;
	plan_fwd = fftwf_plan_dft_r2c_3d(1, 1, 64, real_values, reinterpret_cast<fftwf_complex*>(complex_values), FFTW_ESTIMATE);
	//plan_bwd = fftwf_plan_dft_c2r_3d(logical_z_dimension, logical_y_dimension, logical_x_dimension, reinterpret_cast<fftwf_complex*>(complex_values), real_values, FFTW_ESTIMATE);

  for (int i = 0; i < 66; i++) { real_values[i] = 1.f ;}
  for (int i = 0; i < 10; i++) { std::printf("Before, i[%d] = %f\n", i, real_values[i]);}
  FastFFT::FourierTransformer FT(FastFFT::FourierTransformer::DataType::fp32);
//  FT.SetInputDimensionsAndType(64,1,1,true, false,(wanted_data_type, wanted_origin);    
	FT.SetInputDimensionsAndType(64,1,1,true, false,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);
	FT.SetOutputDimensionsAndType(64,1,1,true,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);
	FT.SetInputPointer(&real_values[0], false);
	FT.CopyHostToDevice();
  for (int i = 0; i < 66; i++) { real_values[i] = 0.f ;}
  for (int i = 0; i < 10; i++) { std::printf("After zero, i[%d] = %f\n", i, real_values[i]);}
	FT.CopyDeviceToHost(true, true);
  for (int i = 0; i < 10; i++) { std::printf("After copy back, i[%d] = %f\n", i, real_values[i]);}

  delete [] real_values;
  
}
