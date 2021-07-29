#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

#include "FastFFT.cu"

int main(int, char**) {
  std::printf("Entering main in tests.cpp\n");

  
  float one[66];
  for (int i = 0; i < 66; i++) { one[i] = 1.f ;}
  for (int i = 0; i < 10; i++) { std::printf("Before, i[%d] = %f\n", i, one[i]);}
  FastFFT::FourierTransformer FT(FastFFT::FourierTransformer::DataType::fp32);
//  FT.SetInputDimensionsAndType(64,1,1,true, false,(wanted_data_type, wanted_origin);    
	FT.SetInputDimensionsAndType(64,1,1,true, false,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);
	FT.SetOutputDimensionsAndType(64,1,1,true,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);
	FT.SetInputPointer(&one[0], false);
	FT.CopyHostToDevice();
  for (int i = 0; i < 66; i++) { one[i] = 0.f ;}
  for (int i = 0; i < 10; i++) { std::printf("After zero, i[%d] = %f\n", i, one[i]);}
	FT.CopyDeviceToHost(true, true);
  for (int i = 0; i < 10; i++) { std::printf("After copy back, i[%d] = %f\n", i, one[i]);}
  
}
