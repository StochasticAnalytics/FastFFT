// Insert some license stuff here

// #include <string>
#include <iostream>
#include <string>
#include <stdio.h>
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



  //////////////////////////////
  // Kernel definitions
  //////////////////////////////
  template<class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
  __launch_bounds__(FFT::max_threads_per_block) __global__
  void SimpleFFT_NoPaddingKernel(ScalarType * real_input, ComplexType* complex_output, short4 dims_in, short4 dims_out);


  //////////////////////////////////////////////
  // IO functions adapted from the cufftdx examples
  ///////////////////////////////

  template<class FFT>
  struct io 
  {
      using complex_type = typename FFT::value_type;
      using scalar_type  = typename complex_type::value_type;

      static inline __device__ unsigned int stride_size() 
      {
          return cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
      }

      static inline __device__ void load_r2c(const scalar_type* input,
                                        complex_type*      thread_data,
                                        int       		  offset) 
      {
        // Calculate global offset of FFT batch
        //            const unsigned int offset = batch_offset(local_fft_id);
        // Get stride, this shows how elements from batch should be split between threads
        const unsigned int stride = stride_size();
        unsigned int       index  = offset + threadIdx.x;
        for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
        {
        reinterpret_cast<scalar_type*>(thread_data)[i] = input[index];
        index += stride;
        }
      } // load_r2c

      static inline __device__ void store_r2c(const complex_type* thread_data,
                                              complex_type*       output,
                                              int        offset) 
      {
        //            const unsigned int offset = batch_offset_r2c(local_fft_id);
        const unsigned int stride = stride_size();
        unsigned int       index  = offset + threadIdx.x;
        for (unsigned int i = 0; i < FFT::elements_per_thread / 2; i++) 
        {
          output[index] = thread_data[i];
          index += stride;
        }
        constexpr unsigned int threads_per_fft        = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
        // threads_per_fft == 1 means that EPT == SIZE, so we need to store one more element
        constexpr unsigned int values_left_to_store =
        threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
        if (threadIdx.x < values_left_to_store) 
        {
          output[index] = thread_data[FFT::elements_per_thread / 2];
        }
       }// store_r2c



  }; // struct io}











  ///////////////////////////////////////////////
  ///////////////////////////////////////////////

  using namespace cufftdx;
  using FFT_64_r2c          = decltype(Block() + Size<64>() + Type<fft_type::r2c>() +
  Precision<float>() + ElementsPerThread<4>() + FFTsPerBlock<1>() + SM<700>());

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

void FourierTransformer::SimpleFFT_NoPadding()
{

  using namespace cufftdx;
	int threadsPerBlock = dims_in.x; // FIXME make sure its a multiple of 32
	int gridDims = 1;

	using FFT = decltype( FFT_64_r2c() + Direction<fft_direction::forward>() );
	using complex_type = typename FFT::value_type;
	using scalar_type    = typename complex_type::value_type;

	SimpleFFT_NoPaddingKernel<FFT, complex_type, scalar_type><< <gridDims,  FFT::block_dim, FFT::shared_memory_size, cudaStreamPerThread>> > ( (scalar_type*)device_pointer_fp32, (complex_type*)device_pointer_fp32_complex, dims_in, dims_out);
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



