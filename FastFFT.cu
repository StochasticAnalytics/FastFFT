// Insert some license stuff here

// #include <string>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cufftdx.hpp>

#include "FastFFT.h"
#include "FastFFT.cuh"



namespace FastFFT {



  ///////////////////////////////////////////////
  ///////////////////////////////////////////////



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
  UnPinHostMemory();
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

  input_memory_allocated = ReturnPaddedMemorySize(dims_in);
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

  output_memory_allocated = ReturnPaddedMemorySize(dims_out);

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
    host_pointer = input_pointer;
  }

  // Check to see if the host memory is pinned.
  if ( ! is_host_memory_pinned)
  {
    precheck
    cudaErr(cudaHostRegister(host_pointer, sizeof(float)*input_memory_allocated, cudaHostRegisterDefault));
    postcheck

    precheck
    cudaErr(cudaHostGetDevicePointer( &pinnedPtr, host_pointer, 0));
    postcheck

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
    precheck
		cudaErr(cudaMalloc(&device_pointer_fp32, 2*output_memory_allocated*sizeof(float)));
    postcheck

		device_pointer_fp32_complex = (float2 *)device_pointer_fp32;

    buffer_fp32 = &device_pointer_fp32[output_memory_allocated];
    buffer_fp32_complex = (float2 *)buffer_fp32;
 
		is_in_memory_device_pointer = true;
	}


  precheck
  // This will be too big on the output memory if padded
  cudaErr(cudaMemcpyAsync(device_pointer_fp32, pinnedPtr, output_memory_allocated*sizeof(float),cudaMemcpyDeviceToHost,cudaStreamPerThread));
  bool should_block_until_complete = true;
	if (should_block_until_complete) cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
  postcheck


}

void FourierTransformer::CopyDeviceToHost(bool free_gpu_memory, bool unpin_host_memory)
{
 
	MyFFTDebugAssertTrue(is_in_memory_device_pointer, "GPU memory not allocated");


  precheck
	cudaErr(cudaMemcpyAsync(pinnedPtr, device_pointer_fp32, output_memory_allocated*sizeof(float),cudaMemcpyDeviceToHost,cudaStreamPerThread));
  postcheck

  // Just set true her for now
  bool should_block_until_complete = true;
	if (should_block_until_complete) cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
  	// TODO add asserts etc.
	if (free_gpu_memory) { Deallocate();}
  if (unpin_host_memory) { UnPinHostMemory();}


}


void FourierTransformer::Deallocate()
{

	if (is_in_memory_device_pointer) 
	{
    precheck
		cudaErr(cudaFree(device_pointer_fp32));
    postcheck
		is_in_memory_device_pointer = false;
	}	
}

void FourierTransformer::UnPinHostMemory()
{
  if (is_host_memory_pinned)
	{
    precheck
		cudaErr(cudaHostUnregister(host_pointer));
    postcheck
		is_host_memory_pinned = false;
	} 
}


void FourierTransformer::SimpleFFT_NoPadding()
{

  using namespace cufftdx;
	int threadsPerBlock = dims_in.x; // FIXME make sure its a multiple of 32
	int gridDims = 1;

	using FFT = decltype( FFT_64_fp32() + Type<fft_type::r2c>() + Direction<fft_direction::forward>() );
  using complex_type = typename FFT::value_type;
  using scalar_type    = typename complex_type::value_type;

	SimpleFFT_NoPaddingKernel<FFT, complex_type, scalar_type>
  << <gridDims,  FFT::block_dim, FFT::shared_memory_size, cudaStreamPerThread>> > ( (scalar_type*)device_pointer_fp32, (complex_type*)buffer_fp32_complex, dims_in, dims_out);
	cudaStreamSynchronize(cudaStreamPerThread);



}

template<class FFT, class ComplexType, class ScalarType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void SimpleFFT_NoPaddingKernel(ScalarType* real_input, ComplexType* complex_output, short4 dims_in, short4 dims_out)
{

	// Initialize the shared memory, assuming everying matches the input data X size in
	// Check that setting cudaFuncSetSharedMemConfig  to 8byte makes any diff for complex reads
  using complex_type = ComplexType;
  using scalar_type  = ScalarType;

  extern __shared__  complex_type shared_mem[];
  complex_type thread_data[FFT::storage_size];

  io<FFT>::load_r2c(real_input, thread_data, 0);
  FFT().execute(thread_data, shared_mem);
  io<FFT>::store_r2c(thread_data, complex_output,  0);


}

} // namespace fast_FFT



