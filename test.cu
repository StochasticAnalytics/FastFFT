#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>

// sudo apt-get install libfftw3-dev libfftw3-doc
#include <fftw3.h>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

#include "FastFFT.cu"

// To print a message and some number n_to_print complex values to stdout
void print_values_complex(float* input, std::string msg, int n_to_print)
{
  for (int i = 0; i < n_to_print*2; i+=2) { std::cout << msg << i/2 << "  " << input[i] << " " << input[i+1] << std::endl ;}
}

// Return sum of real values
float ReturnSumOfReal(float* input, short4 size)
{
  double temp_sum = 0;
  long address = 0;
  int padding_jump_val = size.w*2 - size.x;
  for (int k = 0; k < size.z; k++)
  {
    for (int j = 0; j < size.y; j++)
    {
      for (int i = 0; i < size.x; i++)
      {
        temp_sum += (input[address]);
        address++;
      }
      address += padding_jump_val;
    }
  }

  return float(temp_sum);
}

// Return the sum of the complex values
float2 ReturnSumOfComplex(float2* input, int n_to_print)
{
  double sum_x = 0;
  double sum_y = 0;

  for (int i = 0; i < n_to_print; i++) 
  {
    sum_x += input[i].x;
    sum_y += input[i].y;
  }
  
  return make_float2(float(sum_x), float(sum_y));  
}


void unit_impulse_test(short4 input_size, short4 output_size)
{
  // Pointers to the arrays on the host -- maybe make this a struct of some sort? I'm sure there is a parallel in cuda, look into cuarray/texture code
  float* host_input;
  float* host_output;
  float2* host_input_complex;
  float2* host_output_complex;
  int host_input_memory_allocated;
  int host_output_memory_allocated;


  // Pointers to the arrays on the device
  float* device_input;
  float* device_output;
  float2* device_input_complex;
  float2* device_output_complex;
  int device_memory_allocated;

  float sum;
  float2 sum_complex;


  // We just make one instance of the FourierTransformer class, with calc type float.
  // For the time being input and output are also float. TODO calc optionally either fp16 or nv_bloat16, TODO inputs at lower precision for bandwidth improvement.
  FastFFT::FourierTransformer FT(FastFFT::FourierTransformer::DataType::fp32);




  // Determine how much memory we need, working with FFTW/CUDA style in place transform padding.
  // Note: there is no reason we really need this, because the xforms will always be out of place. 
  //       For now, this is just in place because all memory in cisTEM is allocated accordingly.
  host_input_memory_allocated = FT.ReturnPaddedMemorySize(input_size);
  host_output_memory_allocated = FT.ReturnPaddedMemorySize(output_size);
  
  // On the device, we will always allocate enough memory for the larger of input/output including the buffer array.
  // Minmize the number of calls to malloc which are slow and can lead to fragmentation.
  device_memory_allocated = std::max(host_input_memory_allocated, host_output_memory_allocated);


  // In your own programs, you will be handling this memory allocation yourself. We'll just make something here.
  // I think fftwf_malloc may potentially create a different alignment than new/delete, but kinda doubt it. For cisTEM consistency...
	host_input = (float *) fftwf_malloc(sizeof(float) * host_input_memory_allocated);
	host_input_complex = (float2*) host_input;  // Set the complex_values to point at the newly allocated real values;
  
  // Make FFTW plans for comparing CPU to GPU xforms.
  // This is nearly verbatim from cisTEM::Image::Allocate - I do not know if FFTW_ESTIMATE is the best option.
  // In cisTEM we almost always use MKL, so this might be worth testing. I always used exhaustive in Matlab/emClarity.
  fftwf_plan plan_fwd = NULL;
  fftwf_plan plan_bwd = NULL;
	plan_fwd = fftwf_plan_dft_r2c_3d(input_size.z, input_size.y, input_size.x, host_input, reinterpret_cast<fftwf_complex*>(host_input_complex), FFTW_ESTIMATE);
  plan_bwd = fftwf_plan_dft_c2r_3d(input_size.z, input_size.y, input_size.x, reinterpret_cast<fftwf_complex*>(host_input_complex), host_input, FFTW_ESTIMATE);
  
  // Set our input host memory to a constant. Then FFT[0] = host_input_memory_allocated
  FT.SetToConstant<float>(host_input, host_input_memory_allocated, 1.0f);
  

  
  // This is similar to creating an FFT/CUFFT plan, so set these up before doing anything on the GPU
	FT.SetInputDimensionsAndType(input_size.x,input_size.y,input_size.z,true, false,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);
	FT.SetOutputDimensionsAndType(input_size.x,input_size.y,input_size.z,true,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);
  
  // Now we want to associate the host memory with the device memory. The method here asks if the host pointer is pinned (in page locked memory) which
  // ensures faster transfer. If false, it will be pinned for you.
	FT.SetInputPointer(&host_input[0], false);
  sum_complex = ReturnSumOfComplex(host_input_complex, FT.input_memory_allocated/2);
  MyFFTDebugAssertTestTrue( sum_complex.x == FT.input_memory_allocated/2 && sum_complex.y == FT.input_memory_allocated/2, "Unit impulse init");

  // This copies the host memory into the device global memory. If needed, it will also allocate the device memory first.
	FT.CopyHostToDevice();
  
  // Now let's do the forward FFT on the host and check that the result is correct.
  fftwf_execute_dft_r2c(plan_fwd, host_input, reinterpret_cast<fftwf_complex*>(host_input_complex));
  // print_values_complex(host_input, "fftw ", FT.output_memory_allocated/2);
  
  sum_complex = ReturnSumOfComplex(host_input_complex, FT.output_memory_allocated/2);
  // std::cout << sum_complex.x << " " << sum_complex.y << std::endl;
  MyFFTDebugAssertTestTrue( sum_complex.x == FT.output_number_of_real_values && sum_complex.y == 0, "FFTW unit impulse forward FFT");
  FT.SetToConstant<float>(host_input, host_input_memory_allocated, 2.0f);

  // Forward FFT ;
  FT.FFT_R2C_Transposed();
  FT.FFT_C2C_WithPadding();
  // in buffer, do not deallocate, do not unpin memory
	FT.CopyDeviceToHost(false, false, false);

  sum_complex = ReturnSumOfComplex(host_input_complex, FT.output_memory_allocated/2);
  // std::cout << sum_complex.x << " " << powf(input_size.x*input_size.y*input_size.z,2) << " " << std::endl;

  // for (int i = 0; i < output_size.w*output_size.x*2; i++) { std::cout << host_input[i] << " "; }
  MyFFTDebugAssertTestTrue( sum_complex.x == FT.output_number_of_real_values && sum_complex.y == 0, "FastFFT unit impulse forward FFT");
  FT.SetToConstant<float>(host_input, host_input_memory_allocated, 2.0f);

  FT.FFT_C2C();
  FT.FFT_C2R_Transposed();
	FT.CopyDeviceToHost(false, true, true);

  // Assuming the outputs are always even dimensions, padding_jump_val is always 2.
  sum = ReturnSumOfReal(host_input, output_size);
  // std::cout << sum << " " << powf(input_size.x*input_size.y*input_size.z,2) << " " << std::endl;
  // for (int i = 0; i < output_size.w*output_size.x*2; i++) { std::cout << host_input[i] << " "; }

  MyFFTDebugAssertTestTrue( sum == powf(input_size.x*input_size.y*input_size.z,2),"FastFFT unit impulse round trip FFT");

  fftwf_free(host_input);
  fftwf_destroy_plan(plan_fwd);
  fftwf_destroy_plan(plan_bwd);

}

int main(int argc, char** argv) {

  std::printf("Entering main in tests.cpp\n");
  std::printf("Standard is %i\n\n",__cplusplus);

  // Input and output dimensions, with simple checks. I'm sure there are better checks on argv.
  short4 input_size;
  short4 output_size;

  constexpr const int n_tests = 4;
  int test_size[n_tests] = {64, 128, 4096, 8192};
  for (int iSize = 0; iSize < n_tests; iSize++) {

    std::cout << std::endl << "Testing " << test_size[iSize] << " x" << std::endl;
    input_size = make_short4(test_size[iSize],test_size[iSize],1,0);
    output_size = make_short4(test_size[iSize],test_size[iSize],1,0);

    unit_impulse_test(input_size, output_size);

  }
  
}
