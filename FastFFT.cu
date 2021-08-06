// Insert some license stuff here

// #include <string>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cufftdx.hpp>


#include "FastFFT.cuh"



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

  fft_status = 0;
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

 
  dims_in = make_short4(input_logical_x_dimension, input_logical_y_dimension, input_logical_z_dimension,0);

  input_memory_allocated = ReturnPaddedMemorySize(dims_in);
  input_number_of_real_values = dims_in.x*dims_in.y*dims_in.z;

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


  dims_out = make_short4(output_logical_x_dimension, output_logical_y_dimension, output_logical_z_dimension,0);

  output_memory_allocated = ReturnPaddedMemorySize(dims_out);
  output_number_of_real_values = dims_out.x*dims_out.y*dims_out.z;

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
  // MyFFTPrintWithDetails("Copying host to device");
  // MyFFTPrint(std::to_string(output_memory_allocated) + " bytes of host memory to device");
	if ( ! is_in_memory_device_pointer )
	{
    // Allocate enough for the out of place buffer as well.
    // MyFFTPrintWithDetails("Allocating device memory for input pointer");
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

	using FFT = decltype( FFT_64_fp32() + Direction<fft_direction::forward>() + Type<fft_type::c2c>());
  using complex_type = typename FFT::value_type;
  using scalar_type    = typename complex_type::value_type;
  cudaError_t error_code = cudaSuccess;
  auto workspace = make_workspace<FFT>(error_code);

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

template<class FFT>
void FourierTransformer::FFT_R2C_Transposed_t()
{

  LaunchParams LP = SetLaunchParameters(elements_per_thread_complex);

  using complex_type = typename FFT::value_type;
  using scalar_type = typename complex_type::value_type;
  int shared_mem = LP.mem_offsets.shared_input*sizeof(scalar_type) + FFT::shared_memory_size;

  precheck
  block_fft_kernel_R2C_Transposed<FFT,complex_type,scalar_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
  ( (scalar_type*) device_pointer_fp32,  (complex_type*) buffer_fp32_complex, LP.mem_offsets, LP.twiddle_in,LP.Q);
  postcheck
}

void FourierTransformer::FFT_R2C_Transposed()
{
  MyFFTDebugAssertTrue(fft_status == 0, "fft status must be 0 (real space) for R2C");

  int device, arch;
  GetCudaDeviceArch( device, arch );

  switch (dims_in.x)
  {
    case 64: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<64>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_R2C_Transposed_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<64>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_R2C_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<64>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_R2C_Transposed_t<FFT>(); break;}
      }
      break; }

    case 128: {
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_R2C_Transposed_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_R2C_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_R2C_Transposed_t<FFT>(); break;}
      }
      break; }
   
  }
  fft_status = 1;
}

template<class FFT, class ComplexType, class ScalarType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_R2C_Transposed(ScalarType* input_values, ComplexType* output_values, Offsets mem_offsets, float twiddle_in, int Q)
{
  // Initialize the shared memory, assuming everyting matches the input data X size in
  using complex_type = ComplexType;
  using scalar_type  = ScalarType;

  // The data store is non-coalesced, so don't aggregate the data in shared mem.
	extern __shared__  scalar_type shared_input[];
  complex_type* shared_mem = (complex_type*)&shared_input[mem_offsets.shared_input];


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
  io<FFT>::load_r2c_shared(&input_values[blockIdx.y*mem_offsets.pixel_pitch_input], shared_input, thread_data, twiddle_factor_args, twiddle_in, input_MAP, output_MAP, Q);

	// In the first FFT the modifying twiddle factor is 1 so the data are real
	FFT().execute(thread_data, shared_mem);
  
  io<FFT>::store_r2c_transposed(thread_data, output_values, output_MAP, mem_offsets.pixel_pitch_output);

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

    io<FFT>::store_r2c_transposed(thread_data, output_values, output_MAP, mem_offsets.pixel_pitch_output);

	}


} // end of block_fft_kernel_R2C_Transposed

template <class FFT>
void FourierTransformer::FFT_C2C_WithPadding_t()
{

  LaunchParams LP = SetLaunchParameters(elements_per_thread_complex);

	using complex_type = typename FFT::value_type;
  cudaError_t error_code = cudaSuccess;
  auto workspace = make_workspace<FFT>(error_code);


  int shared_mem;
  // Aggregate the transformed frequency data in shared memory so that we can write to global coalesced.
  shared_mem = LP.mem_offsets.shared_output*sizeof(complex_type) + LP.mem_offsets.shared_input*sizeof(complex_type) + FFT::shared_memory_size;
  // When it is the output dims being smaller, may need a logical or different method
  precheck
  block_fft_kernel_C2C_WithPadding<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
  ( (complex_type*)buffer_fp32_complex,  (complex_type*)device_pointer_fp32_complex, LP.mem_offsets, LP.twiddle_in,LP.Q);
  postcheck


}
void FourierTransformer::FFT_C2C_WithPadding()
{

	// This is the first set of 1d ffts when the input data are real valued, accessing the strided dimension. Since we need the full length, it will actually run a C2C xform
  MyFFTDebugAssertTrue(fft_status == 1, "fft status must be 1 (partial forward)");

  int device, arch;
  GetCudaDeviceArch( device, arch );

  switch (dims_in.x)
  {
    case 64: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<64>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_C2C_WithPadding_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<64>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_C2C_WithPadding_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<64>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_C2C_WithPadding_t<FFT>(); break;}
      }
      break; }

    case 128: {
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_C2C_WithPadding_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_C2C_WithPadding_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_C2C_WithPadding_t<FFT>(); break;}
      }
      break; }
   
  }

  // Relies on the debug assert above
  fft_status = 2;

}

template<class FFT, class ComplexType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_WithPadding(ComplexType* input_values, ComplexType* output_values, Offsets mem_offsets, float twiddle_in, int Q)
{

//	// Initialize the shared memory, assuming everyting matches the input data X size in
  using complex_type = ComplexType;

	extern __shared__  complex_type shared_input_complex[]; // Storage for the input data that is re-used each blcok
	complex_type* shared_output = (complex_type*)&shared_input_complex[mem_offsets.shared_input]; // storage for the coalesced output data. This may grow too large, 
	complex_type* shared_mem = (complex_type*)&shared_output[mem_offsets.shared_output];


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
  io<FFT>::load_shared(&input_values[blockIdx.y*mem_offsets.pixel_pitch_input], shared_input_complex, thread_data, twiddle_factor_args, twiddle_in, input_MAP, output_MAP, Q);


	// In the first FFT the modifying twiddle factor is 1 so the data are reeal
	FFT().execute(thread_data, shared_mem);

	io<FFT>::store(thread_data,shared_output,output_MAP);

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

		io<FFT>::store(thread_data,shared_output,output_MAP);

	}

  // TODO confirm this is needed
	__syncthreads();

	// Now that the memory output can be coalesced send to global
  // FIXME is this actually coalced?
	// int this_idx;
	for (int sub_fft = 0; sub_fft < Q; sub_fft++)
	{
    io<FFT>::store_coalesced(shared_output, &output_values[blockIdx.y * mem_offsets.pixel_pitch_output], sub_fft*mem_offsets.shared_input);
	}


} // end of block_fft_kernel_C2C_WithPadding


template<class FFT>
void FourierTransformer::FFT_C2C_t()
{
  LaunchParams LP = SetLaunchParameters(elements_per_thread_complex);

	using complex_type = typename FFT::value_type;
	using scalar_type    = typename complex_type::value_type;
  cudaError_t error_code = cudaSuccess;
  auto workspace = make_workspace<FFT>(error_code);

  // Aggregate the transformed frequency data in shared memory so that we can write to global coalesced.
  int shared_mem = FFT::shared_memory_size;
  // When it is the output dims being smaller, may need a logical or different method
  precheck
  block_fft_kernel_C2C<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
  ( (complex_type*)device_pointer_fp32_complex,  (complex_type*)buffer_fp32_complex, LP.mem_offsets, workspace);
  postcheck

}

void FourierTransformer::FFT_C2C()
{

  MyFFTDebugAssertTrue(fft_status == 2, "For now, FFT_C2C only works for a full inverse transform along the transposed X dimension")

  int device, arch;
  GetCudaDeviceArch( device, arch );

  switch (dims_in.x)
  {
    case 64: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<64>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2c>() + SM<700>());  FFT_C2C_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<64>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2c>() + SM<750>());  FFT_C2C_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<64>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2c>() + SM<800>());  FFT_C2C_t<FFT>(); break;}
      }
      break; }

    case 128: {
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2c>() + SM<700>());  FFT_C2C_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2c>() + SM<750>());  FFT_C2C_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2c>() + SM<800>());  FFT_C2C_t<FFT>(); break;}
      }
      break; }
   
  }

  // Relies on the debug assert above
  fft_status = 3;
}

template<class FFT, class ComplexType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C(ComplexType* input_values, ComplexType* output_values, Offsets mem_offsets, typename FFT::workspace_type workspace)
{

//	// Initialize the shared memory, assuming everyting matches the input data X size in
  using complex_type = ComplexType;

	extern __shared__  complex_type shared_mem[]; // Storage for the input data that is re-used each blcok


	// Memory used by FFT
  complex_type thread_data[FFT::storage_size];

  // No need to __syncthreads as each thread only accesses its own shared mem anyway
  io<FFT>::load(&input_values[blockIdx.y*mem_offsets.pixel_pitch_input],  thread_data);


	// In the first FFT the modifying twiddle factor is 1 so the data are reeal
	FFT().execute(thread_data, shared_mem, workspace);

	io<FFT>::store(thread_data ,&output_values[blockIdx.y*mem_offsets.pixel_pitch_output]);


} // end of block_fft_kernel_C2C

template <class FFT>
void FourierTransformer::FFT_C2R_Transposed_t()
{
  LaunchParams LP = SetLaunchParameters(elements_per_thread_complex);

	using complex_type = typename FFT::value_type;
	using scalar_type    = typename complex_type::value_type;
  cudaError_t error_code = cudaSuccess;
  auto workspace = make_workspace<FFT>(error_code);


  // Aggregate the transformed frequency data in shared memory so that we can write to global coalesced.
  int shared_mem = FFT::shared_memory_size;
	precheck
	block_fft_kernel_C2R_Transformed<FFT, complex_type, scalar_type><< <LP.gridDims, LP.threadsPerBlock, FFT::shared_memory_size, cudaStreamPerThread>> >
	( (complex_type*)buffer_fp32_complex, (scalar_type*)device_pointer_fp32, LP.mem_offsets, workspace);
  postcheck
}

void FourierTransformer::FFT_C2R_Transposed()
{

  MyFFTDebugAssertTrue(fft_status == 3, "status must be 3");

  int device, arch;
  GetCudaDeviceArch( device, arch );

  switch (dims_in.x)
  {
    case 64: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<64>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<700>());  FFT_C2R_Transposed_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<64>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<750>());  FFT_C2R_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<64>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<800>());  FFT_C2R_Transposed_t<FFT>(); break;}
      }
      break; }

    case 128: {
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<700>());  FFT_C2R_Transposed_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<750>());  FFT_C2R_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<800>());  FFT_C2R_Transposed_t<FFT>(); break;}
      }
      break; }
   
  }


}

template<class FFT, class ComplexType, class ScalarType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2R_Transformed(ComplexType* input_values, ScalarType* output_values, Offsets mem_offsets, typename FFT::workspace_type workspace)
{

	using complex_type = ComplexType;
	using scalar_type  = ScalarType;

	extern __shared__  complex_type shared_mem[];


  complex_type thread_data[FFT::storage_size];

  io<FFT>::load_c2r_transposed(&input_values[blockIdx.y], thread_data, mem_offsets.pixel_pitch_input);

  // For loop zero the twiddles don't need to be computed
  FFT().execute(thread_data, shared_mem, workspace);

  io<FFT>::store_c2r(thread_data, &output_values[blockIdx.y*mem_offsets.pixel_pitch_output]);

} // end of block_fft_kernel_C2R_Transposed

} // namespace fast_FFT



