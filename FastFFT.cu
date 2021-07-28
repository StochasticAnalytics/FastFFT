// Insert some license stuff here

// #include <string>
#include <iostream>
#include <string>
#include <stdio.h>
// #include <fstream>
// #include <stdio.h>
// #include <stdlib.h>
#include <cufftdx.hpp>

#include "FastFFT.h"

// This macro is more or less a snippet from the CUDA SDK
#ifndef cudaErr
#define cudaErr(error)                                                                      \
        {                                                                                                   \
            auto status = static_cast<cudaError_t>(error);                                                  \
            if (status != cudaSuccess) {                                                                    \
                std::cerr << cudaGetErrorString(status) << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                                          \
            }                                                                                               \
        }
#endif 


// #ifdef DEBUG
#define MyFFTPrint(...)	{std::cerr << __VA_ARGS__ << sizeof(short) << std::endl;}
#define MyFFTPrintWithDetails(...)	{std::cerr << __VA_ARGS__  << " From: " << __FILE__  << " " << __LINE__  << " " << __PRETTY_FUNCTION__ << std::endl;}
#define MyFFTDebugAssertTrue(cond, msg, ...) {if ((cond) != true) { std::cerr << msg   << std::endl << " Failed Assert at "  << __FILE__  << " " << __LINE__  << " " << __PRETTY_FUNCTION__ << std::endl; exit(-1);}}
#define MyFFTDebugAssertFalse(cond, msg, ...) {if ((cond) == true) { std::cerr << msg  << std::endl << " Failed Assert at "  << __FILE__  << " " << __LINE__  << " " << __PRETTY_FUNCTION__ << std::endl; exit(-1);}}
// #endif " " << __VA_ARGS__ << " Failed Assert at "  << __FILE__  << " " << __LINE__  << " " << __PRETTY_FUNCTION__ << 

#define HEAVY_ERROR_CHECKING_FFT

#ifdef HEAVY_ERROR_CHECKING_FFT
#define checkErrorsAndTimingWithSynchronization(input_stream) { cudaError_t cuda_error = cudaStreamSynchronize(input_stream); if (cuda_error != cudaSuccess) { std::cerr << cudaGetErrorString(cuda_error) << std::endl; MyFFTPrintWithDetails("");} };
#define pre_checkErrorsAndTimingWithSynchronization(input_sream) { cudaErr(cudaGetLastError()); }
#else
#define checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
#define pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
#endif

namespace FastFFT {



FourierTransformer::FourierTransformer(DataType wanted_calc_data_type) 
{


  MyFFTPrint( "Initializing FourierTrasnformer...");
  calc_data_type = wanted_calc_data_type;
  // Plan to allow fp16 and bf16
  MyFFTDebugAssertTrue(calc_data_type == DataType::fp32, "Only F32 is supported at the moment");
  SetDefaults();
}

FourierTransformer::~FourierTransformer() 
{
  Deallocate();
}

void FourierTransformer::SetDefaults()
{
  DataType input_data_type = fp32;
  DataType output_data_type = fp32;

  // booleans to track state, could be bit fields but that seem opaque to me.
  is_in_memory_host_pointer = false;
  is_in_memory_device_pointer = false;

  is_fftw_padded_input = false;
  is_fftw_padded_output = false;
  is_fftw_padded_buffer = false;

  is_set_input_params = false;
  is_set_output_params = false;

  is_host_memory_pinned = false;
}

void FourierTransformer::SetInputDimensionsAndType(size_t input_logical_x_dimension, 
                                                   size_t input_logical_y_dimension, 
                                                   size_t input_logical_z_dimension, 
                                                   bool is_padded_input, 
                                                   bool is_host_memory_pinned, 
                                                   DataType input_data_type,
                                                   OriginType input_origin_type)
{

  MyFFTDebugAssertTrue(input_logical_x_dimension > 0, "Input logical x dimension must be > 0");
  MyFFTDebugAssertTrue(input_logical_y_dimension > 0, "Input logical y dimension must be > 0");
  MyFFTDebugAssertTrue(input_logical_z_dimension > 0, "Input logical z dimension must be > 0");
  MyFFTDebugAssertTrue(is_padded_input, "The input memory must be fftw padded");

  short int w;
  if (is_padded_input)
  {
    if (input_logical_x_dimension % 2 == 0) w = 2;
    else w = 1;
  }
  else w = 0;

  dims_in = make_short4(input_logical_x_dimension, input_logical_y_dimension, input_logical_z_dimension,input_logical_x_dimension + w);

  input_memory_allocated = ReturnPaddedMemorySize(input_logical_x_dimension, input_logical_y_dimension, input_logical_z_dimension);
  this->input_origin_type = input_origin_type;
  is_set_input_params = true;
}

void FourierTransformer::SetOutputDimensionsAndType(size_t output_logical_x_dimension, 
                                                    size_t output_logical_y_dimension, 
                                                    size_t output_logical_z_dimension, 
                                                    bool is_padded_output, 
                                                    DataType output_data_type,
                                                    OriginType output_origin_type)
{
  MyFFTDebugAssertTrue(output_logical_x_dimension > 0, "output logical x dimension must be > 0");
  MyFFTDebugAssertTrue(output_logical_y_dimension > 0, "output logical y dimension must be > 0");
  MyFFTDebugAssertTrue(output_logical_z_dimension > 0, "output logical z dimension must be > 0");
  MyFFTDebugAssertTrue(is_padded_output, "The output memory must be fftw padded");

  short int w;
  if (is_padded_output)
  {
    if (output_logical_x_dimension % 2 == 0) w = 2;
    else w = 1;
  }
  else w = 0;

  dims_out = make_short4(output_logical_x_dimension, output_logical_y_dimension, output_logical_z_dimension,output_logical_x_dimension + w);

  output_memory_allocated = ReturnPaddedMemorySize(output_logical_x_dimension, output_logical_y_dimension, output_logical_z_dimension);

  this->output_origin_type = output_origin_type;
  is_set_output_params = true;
}




void FourierTransformer::SetInputPointer(float* input_pointer, bool is_input_on_device) 
{ 
  MyFFTDebugAssertTrue(calc_data_type == DataType::fp32, "Only F32 is supported at the moment");
  MyFFTDebugAssertTrue(is_set_input_params, "Input parameters not set");

  if ( is_input_on_device) 
  {
    // We'll need a check on compute type, and a conversion if needed prior to this.
    device_pointer_fp32 = input_pointer;
  }
  else
  {
    MyFFTPrintWithDetails("Input pointer is on host memory");
    // host_pointer = static_cast<float*>(host_pointer); 
    host_pointer = input_pointer;
    // pinnedPtr = static_cast<float*>(pinnedPtr); 
  }

  // Check to see if the host memory is pinned.
  if ( ! is_host_memory_pinned)
  {
    pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
    cudaErr(cudaHostRegister(host_pointer, sizeof(float)*input_memory_allocated, cudaHostRegisterDefault));
    checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

    pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
    cudaErr(cudaHostGetDevicePointer( &pinnedPtr, host_pointer, 0));
    checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

    is_host_memory_pinned = true;

  }
  is_in_memory_host_pointer = true;
  
}

void FourierTransformer::CopyHostToDevice()
{
 
	MyFFTDebugAssertTrue(is_in_memory_host_pointer, "Host memory not allocated");
  MyFFTDebugAssertTrue(is_set_output_params, "Output parameters need to be set");
  MyFFTDebugAssertTrue(is_set_input_params, "Input parameters need to be set");
  MyFFTPrintWithDetails("Copying host to device");
  MyFFTPrint(std::to_string(output_memory_allocated) + " bytes of host memory to device");
	if ( ! is_in_memory_device_pointer )
	{
    // Allocate enough for the out of place buffer as well.
    MyFFTPrintWithDetails("Allocating device memory for input pointer");
    pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
		cudaErr(cudaMalloc(&device_pointer_fp32, 2*output_memory_allocated*sizeof(float)));
    checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

		device_pointer_fp32_complex = (float2 *)device_pointer_fp32;

    buffer_fp32 = &device_pointer_fp32[output_memory_allocated];
    buffer_fp32_complex = (float2 *)buffer_fp32;
 
		is_in_memory_device_pointer = true;
	}


  pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
  // This will be too big on the output memory if padded
  cudaErr(cudaMemcpyAsync(device_pointer_fp32, pinnedPtr, output_memory_allocated*sizeof(float),cudaMemcpyDeviceToHost,cudaStreamPerThread));
  bool should_block_until_complete = true;
	if (should_block_until_complete) cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
  checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);


}

void FourierTransformer::CopyDeviceToHost(bool free_gpu_memory, bool unpin_host_memory)
{
 
	MyFFTDebugAssertTrue(is_in_memory_device_pointer, "GPU memory not allocated");


  pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	cudaErr(cudaMemcpyAsync(pinnedPtr, device_pointer_fp32, output_memory_allocated*sizeof(float),cudaMemcpyDeviceToHost,cudaStreamPerThread));
  checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
  // Just set true her for now
  bool should_block_until_complete = true;
	if (should_block_until_complete) cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
  	// TODO add asserts etc.
	if (free_gpu_memory) 
	{ 
		Deallocate();
	}
	if (unpin_host_memory && is_host_memory_pinned)
	{
    pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
		cudaHostUnregister(host_pointer);
    checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
		is_host_memory_pinned = false;
	}

}


void FourierTransformer::Deallocate()
{

  if (is_host_memory_pinned)
	{
		cudaErr(cudaHostUnregister(host_pointer));
		is_host_memory_pinned = false;
	} 
	if (is_in_memory_device_pointer) 
	{
		cudaErr(cudaFree(device_pointer_fp32));
		is_in_memory_device_pointer = false;
	}	
}


} // namespace fast_FFT



