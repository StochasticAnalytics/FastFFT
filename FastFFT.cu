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

void FourierTransformer::CopyDeviceToHost(bool is_in_buffer, bool free_gpu_memory, bool unpin_host_memory)
{
 
	MyFFTDebugAssertTrue(is_in_memory_device_pointer, "GPU memory not allocated");

  float* copy_pointer;
  if (is_in_buffer) copy_pointer = buffer_fp32;
  else copy_pointer = device_pointer_fp32;

  precheck
	cudaErr(cudaMemcpyAsync(pinnedPtr, copy_pointer, output_memory_allocated*sizeof(float),cudaMemcpyDeviceToHost,cudaStreamPerThread));
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

  precheck
	SimpleFFT_NoPaddingKernel<FFT, complex_type, scalar_type>
  << <gridDims,  FFT::block_dim, FFT::shared_memory_size, cudaStreamPerThread>> > ( (scalar_type*)device_pointer_fp32, (complex_type*)buffer_fp32_complex, dims_in, dims_out);
  postcheck


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

void FourierTransformer::FFT_R2C_Transposed()
{

  // TODO add asserts

  // TODO padding or maybe that is a separate funcitno.
	// For the twiddle factors ahead of the P size ffts
	float twiddle_in = -2*PIf/dims_out.x;
	int   Q = dims_out.x / dims_in.x; // FIXME assuming for now this is already divisible


	dim3 threadsPerBlock = dim3(dims_in.x/elements_per_thread_real, 1, 1); // FIXME make sure its a multiple of 32
	dim3 gridDims = dim3(1,dims_in.y, 1); // TODO allow 3d and also confirm this isn't used in any artifacts leftover 

  using FFT = decltype( FFT_64_fp32() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() );
	using complex_type = typename FFT::value_type;
	using scalar_type    = typename complex_type::value_type;

  int shared_mem = dims_in.x*sizeof(scalar_type) + FFT::shared_memory_size;
  precheck
  block_fft_kernel_R2C_Transposed<FFT,complex_type,scalar_type><< <gridDims,  threadsPerBlock, shared_mem, cudaStreamPerThread>> >
  ( (scalar_type *) device_pointer_fp32,  (complex_type*) buffer_fp32_complex, dims_in, dims_out,twiddle_in,Q);
  postcheck

}

template<class FFT, class ComplexType, class ScalarType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_R2C_Transposed(ScalarType* input_values, ComplexType* output_values, short4 dims_in, short4 dims_out, float twiddle_in, int Q)
{

  // Initialize the shared memory, assuming everyting matches the input data X size in
  using complex_type = ComplexType;
  using scalar_type  = ScalarType;

  // The data store is non-coalesced, so don't aggregate the data in shared mem.
	extern __shared__  scalar_type shared_input[];
  complex_type* shared_mem = (complex_type*)&shared_input[dims_in.x];


	// Memory used by FFT
	complex_type twiddle;
  complex_type thread_data[FFT::storage_size];

  // To re-map the thread index to the data ... these really could be short ints, but I don't know how that will perform. TODO benchmark
  // It is also questionable whether storing these vs, recalculating makes more sense.
  int input_MAP[FFT::storage_size];
  // To re-map the decomposed frequency to the full output frequency
  int output_MAP[FFT::storage_size];
  // For a given decomposed fragment
  float twiddle_factor_args[FFT::storage_size];

  // No need to __syncthreads as each thread only accesses its own shared mem anyway
  // multiply Q*dims_out.w because x maps to y in the output transposed FFT
  io<FFT>::load_r2c_shared(&input_values[blockIdx.y*dims_in.w*2], shared_input, thread_data, twiddle_factor_args, twiddle_in, input_MAP, output_MAP, Q);

	// In the first FFT the modifying twiddle factor is 1 so the data are real
	FFT().execute(thread_data, shared_mem);

  io<FFT>::store_r2c_transposed(thread_data, output_values, output_MAP, dims_out.y);

    // For the other fragments we need the initial twiddle
	for (int sub_fft = 1; sub_fft < Q; sub_fft++)
	{

	    io<FFT>::copy_from_shared(shared_input, thread_data, input_MAP);

    printf("I SHOULD NOT BE HERA\n");
		// cufftDX expects packed real data for a real xform, but we modify with a complex twiddle factor.
		// to get around this, split the complex fft into the sum of the real and imaginary parts
		for (int i = 0; i < FFT::elements_per_thread; i++)
		{
			// Pre shift with twiddle
			__sincosf(twiddle_factor_args[i]*sub_fft,&twiddle.y,&twiddle.x);
			thread_data[i] *= twiddle;
		  // increment the output map. 
			output_MAP[i]++;
		}

		FFT().execute(thread_data, shared_mem);

    io<FFT>::store_r2c_transposed(thread_data, output_values, output_MAP, dims_out.y);

	}


} // end of block_fft_kernel_R2C_Transposed

void FourierTransformer::FFT_C2C_WithPadding(bool forward_transform)
{

	// This is the first set of 1d ffts when the input data are real valued, accessing the strided dimension. Since we need the full length, it will actually run a C2C xform


	float twiddle_in;
	int Q;
	int shared_mem;
	dim3 threadsPerBlock;
	dim3 gridDims;

  using FFT = decltype( FFT_64_fp32() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() );
	using complex_type = typename FFT::value_type;
	using scalar_type    = typename complex_type::value_type;


  // For the twiddle factors ahead of the P size ffts
  twiddle_in = -2*PIf/dims_out.y;
  Q = dims_out.y / dims_in.y; // FIXME assuming for now this is already divisible
    // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
    // will be executed on block level. Shared memory is required for co-operation between threads.

  threadsPerBlock = dim3(dims_in.y/elements_per_thread_complex, 1, 1); // FIXME make sure its a multiple of 32
  gridDims = dim3(1,dims_out.w,1);

  // Aggregate the transformed frequency data in shared memory so that we can write to global coalesced.
  shared_mem = dims_out.y*sizeof(complex_type) + dims_in.y*sizeof(complex_type) + FFT::shared_memory_size;

  // When it is the output dims being smaller, may need a logical or different method
  precheck
  block_fft_kernel_C2C_WithPadding<FFT,complex_type><< <gridDims,  threadsPerBlock, shared_mem, cudaStreamPerThread>> >
  ( (complex_type*)buffer_fp32_complex,  (complex_type*)device_pointer_fp32_complex, dims_in, dims_out,twiddle_in,Q);
  postcheck

}

template<class FFT, class ComplexType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_WithPadding(ComplexType* input_values, ComplexType* output_values, short4 dims_in, short4 dims_out, float twiddle_in, int Q)
{

//	// Initialize the shared memory, assuming everyting matches the input data X size in
  using complex_type = ComplexType;

	extern __shared__  complex_type shared_input_complex[]; // Storage for the input data that is re-used each blcok
	complex_type* shared_output = (complex_type*)&shared_input_complex[dims_in.y]; // storage for the coalesced output data. This may grow too large, 
	complex_type* shared_mem = (complex_type*)&shared_output[dims_out.y];


	// Memory used by FFT
	complex_type twiddle;
  complex_type thread_data[FFT::storage_size];

  // To re-map the thread index to the data
  int input_MAP[FFT::storage_size];
  // To re-map the decomposed frequency to the full output frequency
  int output_MAP[FFT::storage_size];
  // For a given decomposed fragment
  float twiddle_factor_args[FFT::storage_size];

  // No need to __syncthreads as each thread only accesses its own shared mem anyway
  io<FFT>::load_shared(&input_values[blockIdx.y*dims_out.y], shared_input_complex, thread_data, twiddle_factor_args, twiddle_in, input_MAP, output_MAP, Q, 1);


	// In the first FFT the modifying twiddle factor is 1 so the data are reeal
	FFT().execute(thread_data, shared_mem);

	io<FFT>::store(thread_data,shared_output,output_MAP,1);


    // For the other fragments we need the initial twiddle
	for (int sub_fft = 1; sub_fft < Q; sub_fft++)
	{

	  io<FFT>::copy_from_shared(shared_input_complex, thread_data, input_MAP);

		for (int i = 0; i < FFT::elements_per_thread; i++)
		{
			// Pre shift with twiddle
			__sincosf(twiddle_factor_args[i]*sub_fft,&twiddle.y,&twiddle.x);
			thread_data[i] *= twiddle;
		    // increment the output map. Note this only works for the leading non-zero case
			output_MAP[i]++;
		}

		FFT().execute(thread_data, shared_mem);

		io<FFT>::store(thread_data,shared_output,output_MAP,1);

	}

  // TODO confirm this is needed
	__syncthreads();

	// Now that the memory output can be coalesced send to global
  // FIXME is this actually coalced?
	// int this_idx;
	for (int sub_fft = 0; sub_fft < Q; sub_fft++)
	{
    io<FFT>::store_coalesced(shared_output, &output_values[blockIdx.y * dims_out.y], sub_fft, dims_in.y);
		// for (int i = 0; i < FFT::elements_per_thread; i++)
		// {
		// 	this_idx = input_MAP[i] + dims_in.x*sub_fft;
		// 	if (this_idx < dims_out.w)
		// 	{
		// 		output_values[blockIdx.y * dims_out.w + this_idx] = shared_output[this_idx];
		// 	}
		// }
	}


} // end of block_fft_kernel_C2C_WithPadding

} // namespace fast_FFT



