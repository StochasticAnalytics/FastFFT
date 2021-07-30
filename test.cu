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

void print_values(float* input, std::string msg, int n_to_print)
{
  for (int i = 0; i < n_to_print; i++) { std::cout << msg << i << "  " << input[i] << std::endl ;}
}

void print_values_complex(float* input, std::string msg, int n_to_print)
{
  for (int i = 0; i < n_to_print*2; i+=2) { std::cout << msg << i/2 << "  " << input[i] << " " << input[i+1] << std::endl ;}
}

void print_values_matrix(float* input, std::string msg, int n_row, int n_col)
{
  std::cout << msg << std::endl;
  int padding_val;
  if (n_row % 2 == 0) { padding_val =  2; }
  else { padding_val =  1; }

  int address = (n_row + padding_val) * n_col;

  // print matrix rotated 90 degrees
  for (int y = n_col-1; y >= 0; y--) 
  {
    address -= (n_row+padding_val);
    for (int x = 0; x < n_row; x++)
    {
      std::cout << input[address] << " ";
      address++;
    }
    std::cout << std::endl;
    address -= (n_row+padding_val);
  }
  
}

void ReturnSumOfComplex(float2* input, float2& sum, int n_to_print)
{
  sum.x = 0.f;
  sum.y = 0.f;
  for (int i = 0; i < n_to_print; i++) 
  {
    sum.x += input[i].x;
    sum.y += input[i].y;
  }
  
}
int main(int argc, char** argv) {

  std::printf("Entering main in tests.cpp\n");
  std::printf("Standard is %i\n\n",__cplusplus);

  // Input and output dimensions, with simple checks. I'm sure there are better checks on argv.
  short4 input_size;
  short4 output_size;

  if ( argc != 4 && argc != 7) 
  { 
    std::cout << argc << std::endl;
    std::cout << "Usage: ./tests n_x n_y n_z [optionally 3 larger or smaller sizes, otherwise input_size=output_size]" << std::endl;
    exit(1);
  }
  else
  {
    for (int i = 1; i < 4; i++) { if (atoi(argv[i]) <=0) { std::cout << "Error: " << argv[i] << " is not a positive integer" << std::endl; exit(1);}}
    input_size  = make_short4( atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), 0 );

    if (argc == 7) 
    {
      for (int i = 1; i < 4; i++) { if (atoi(argv[i+3]) <=0) { std::cout << "Error: " << argv[i+3] << " is not a positive integer" << std::endl; exit(1);}}
      output_size = make_short4( atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), 0 );
    }
    else
    {
      output_size = input_size;
    }
  }


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

  float2 sum;


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
  ReturnSumOfComplex(host_input_complex, sum, FT.input_memory_allocated/2);
  MyFFTDebugAssertTestTrue( sum.x == FT.input_memory_allocated/2 && sum.y == FT.input_memory_allocated/2, "Unit impulse init");

  // This copies the host memory into the device global memory. If needed, it will also allocate the device memory first.
	FT.CopyHostToDevice();
  
  // Now let's do the forward FFT on the host and check that the result is correct.
  fftwf_execute_dft_r2c(plan_fwd, host_input, reinterpret_cast<fftwf_complex*>(host_input_complex));
  print_values_complex(host_input, "fftw ", FT.output_memory_allocated/2);
  
  ReturnSumOfComplex(host_input_complex, sum, FT.output_memory_allocated/2);
  std::cout << sum.x << " " << sum.y << std::endl;
  MyFFTDebugAssertTestTrue( sum.x == FT.output_number_of_real_values && sum.y == 0, "FFTW unit impulse forward FFT");
  FT.SetToConstant<float>(host_input, host_input_memory_allocated, 2.0f);

  // FT.SimpleFFT_NoPadding();
  FT.FFT_R2C_Transposed();
  FT.FFT_C2C_WithPadding(true);
	FT.CopyDeviceToHost(false, true, true);

  ReturnSumOfComplex(host_input_complex, sum, FT.output_memory_allocated/2);
  std::cout << sum.x << " " << sum.y << std::endl;
  MyFFTDebugAssertTestTrue( sum.x == FT.output_number_of_real_values  && sum.y == 0,"FastFFT unit impulse forward FFT");

  fftwf_free(host_input);
  fftwf_destroy_plan(plan_fwd);
  fftwf_destroy_plan(plan_bwd);
  
}
