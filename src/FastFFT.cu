// Insert some license stuff here

// #include <string>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cufftdx.hpp>


#include "FastFFT.cuh"



namespace FastFFT {


template <class ComputeType, class InputType, class OutputType>
FourierTransformer<ComputeType, InputType, OutputType>::FourierTransformer() 
{
  SetDefaults();

  // This assumption precludes the use of a packed _half2 that is really RRII layout for two arrays of __half.
  // TODO could is_real_valued_input be constexpr?
  if constexpr(std::is_same< InputType, __half2>::value || std::is_same< InputType,float2>::value)
  {
    is_real_valued_input = false;
  }
  else
  {
    is_real_valued_input = true;
  }
  
}

template <class ComputeType, class InputType, class OutputType>
FourierTransformer<ComputeType, InputType, OutputType>::~FourierTransformer() 
{
  Deallocate();
  UnPinHostMemory();
  SetDefaults();
}

template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::SetDefaults()
{

  // booleans to track state, could be bit fields but that seem opaque to me.
  is_in_memory_host_pointer = false;
  is_in_memory_device_pointer = false;

  is_fftw_padded_input = false;
  is_fftw_padded_output = false;

  is_set_input_params = false;
  is_set_output_params = false;
  is_size_validated = false;
  is_set_input_pointer = false;

  is_host_memory_pinned = false;


}

template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::Deallocate()
{
	if (is_in_memory_device_pointer) 
	{
    precheck
		cudaErr(cudaFree(d_ptr.position_space));
    postcheck
		is_in_memory_device_pointer = false;
	}	
}

template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::UnPinHostMemory()
{
  if (is_host_memory_pinned)
	{
    precheck
		cudaErr(cudaHostUnregister(host_pointer));
    postcheck
		is_host_memory_pinned = false;
	} 
}


template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::SetInputDimensionsAndType(size_t input_logical_x_dimension, 
                                                   size_t input_logical_y_dimension, 
                                                   size_t input_logical_z_dimension, 
                                                   bool is_padded_input, 
                                                   bool is_host_memory_pinned, 
                                                   OriginType input_origin_type)
{

  MyFFTDebugAssertTrue(input_logical_x_dimension > 0, "Input logical x dimension must be > 0");
  MyFFTDebugAssertTrue(input_logical_y_dimension > 0, "Input logical y dimension must be > 0");
  MyFFTDebugAssertTrue(input_logical_z_dimension > 0, "Input logical z dimension must be > 0");

  dims_in = make_short4(input_logical_x_dimension, input_logical_y_dimension, input_logical_z_dimension,0);
  is_fftw_padded_input = is_padded_input; // Note: Must be set before ReturnPaddedMemorySize
  MyFFTRunTimeAssertTrue(is_fftw_padded_input, "Support for input arrays that are not FFTW padded needs to be implemented."); // FIXME

  input_memory_allocated = ReturnPaddedMemorySize(dims_in);
  input_number_non_padding_values = dims_in.x*dims_in.y*dims_in.z;

  this->input_origin_type = input_origin_type;
  is_set_input_params = true;
}

template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::SetOutputDimensionsAndType(size_t output_logical_x_dimension, 
                                                    size_t output_logical_y_dimension, 
                                                    size_t output_logical_z_dimension, 
                                                    bool is_padded_output, 
                                                    OriginType output_origin_type)
{
  MyFFTDebugAssertTrue(is_set_input_params, "Please set the input paramters first.")
  MyFFTDebugAssertTrue(output_logical_x_dimension > 0, "output logical x dimension must be > 0");
  MyFFTDebugAssertTrue(output_logical_y_dimension > 0, "output logical y dimension must be > 0");
  MyFFTDebugAssertTrue(output_logical_z_dimension > 0, "output logical z dimension must be > 0");
  MyFFTDebugAssertTrue(is_fftw_padded_input == is_padded_output, "If the input data are FFTW padded, so must the output.");

  dims_out = make_short4(output_logical_x_dimension, output_logical_y_dimension, output_logical_z_dimension,0);

  output_memory_allocated = ReturnPaddedMemorySize(dims_out);
  output_number_non_padding_values = dims_out.x*dims_out.y*dims_out.z;

  this->output_origin_type = output_origin_type;
  is_set_output_params = true;
}


template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::SetInputPointer(InputType* input_pointer, bool is_input_on_device) 
{ 
  MyFFTDebugAssertTrue(is_set_input_params, "Input parameters not set");

  if ( is_input_on_device) 
  {
    // We'll need a check on compute type, and a conversion if needed prior to this.
    d_ptr.position_space = input_pointer;
  }
  else
  {
    host_pointer = input_pointer;
  }

  // Check to see if the host memory is pinned.
  if ( ! is_host_memory_pinned)
  {
    precheck
    cudaErr(cudaHostRegister((void *)host_pointer, sizeof(InputType)*input_memory_allocated, cudaHostRegisterDefault));
    postcheck

    precheck
    cudaErr(cudaHostGetDevicePointer( &pinnedPtr, host_pointer, 0));
    postcheck

    is_host_memory_pinned = true;
  }
  is_in_memory_host_pointer = true;
  
  is_set_input_pointer = true;
}



template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::CopyHostToDevice()
{
 
  CheckDimensions();
	MyFFTDebugAssertTrue(is_in_memory_host_pointer, "Host memory not allocated");

  // MyFFTPrintWithDetails("Copying host to device");
  // MyFFTPrint(std::to_string(output_memory_allocated) + " bytes of host memory to device");
	if ( ! is_in_memory_device_pointer )
	{

    // Allocate enough for the out of place buffer as well.
    // MyFFTPrintWithDetails("Allocating device memory for input pointer");
    precheck
		cudaErr(cudaMalloc(&d_ptr.position_space, compute_memory_allocated * sizeof(ComputeType)));
    postcheck

    size_t buffer_address;
    if (is_real_valued_input) buffer_address = compute_memory_allocated/2 ;
    else buffer_address = compute_memory_allocated/4; 

    if constexpr(std::is_same< decltype(d_ptr.momentum_space), __half2>::value )
    {
      d_ptr.momentum_space = (__half2 *)d_ptr.position_space;
      d_ptr.position_space_buffer = &d_ptr.position_space[buffer_address];
      d_ptr.momentum_space_buffer = (__half2 *)d_ptr.position_space_buffer;
    }
    else
    {
      d_ptr.momentum_space = (float2 *)d_ptr.position_space;
      d_ptr.position_space_buffer = &d_ptr.position_space[buffer_address]; // compute 
      d_ptr.momentum_space_buffer = (float2 *)d_ptr.position_space_buffer;
    }


 
		is_in_memory_device_pointer = true;
	}


  precheck
  cudaErr(cudaMemcpyAsync(d_ptr.position_space, pinnedPtr, input_memory_allocated * sizeof(InputType),cudaMemcpyDeviceToHost,cudaStreamPerThread));
  postcheck
  is_in_buffer_memory = false;

  // TODO r/n assuming InputType is _half, _half2, float, or _float2 (real, complex, real, complex) need to handle other types and convert
  bool should_block_until_complete = true;
	if (should_block_until_complete) cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

}

template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::CopyDeviceToHost( bool free_gpu_memory, bool unpin_host_memory)
{
 
  CheckDimensions();
	MyFFTDebugAssertTrue(is_in_memory_device_pointer, "GPU memory not allocated");

  ComputeType* copy_pointer;
  if (is_in_buffer_memory) copy_pointer = d_ptr.position_space_buffer;
  else copy_pointer = d_ptr.position_space;

  // FIXME this is assuming the input type matches the compute type.
  precheck
	cudaErr(cudaMemcpyAsync(pinnedPtr, copy_pointer, input_memory_allocated*sizeof(InputType),cudaMemcpyDeviceToHost,cudaStreamPerThread));
  postcheck

  // Just set true her for now
  bool should_block_until_complete = true;
	if (should_block_until_complete) cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
  	// TODO add asserts etc.
	if (free_gpu_memory) { Deallocate();}
  if (unpin_host_memory) { UnPinHostMemory();}

}

template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::CopyDeviceToHost(OutputType* output_pointer, bool free_gpu_memory, bool unpin_host_memory)
{
 
  CheckDimensions();
	MyFFTDebugAssertTrue(is_in_memory_device_pointer, "GPU memory not allocated");
  // Assuming the output is not pinned, TODO change to optionally maintain as host_input as well.
  OutputType* tmpPinnedPtr;
  precheck
  // FIXME this is assuming output type is the same as compute type.
  cudaErr(cudaHostRegister(output_pointer, sizeof(OutputType)*output_memory_allocated, cudaHostRegisterDefault));
  postcheck
  
  precheck
  cudaErr(cudaHostGetDevicePointer( &tmpPinnedPtr, output_pointer, 0));
  postcheck
  if (is_in_buffer_memory)
  {
    precheck
    cudaErr(cudaMemcpyAsync(tmpPinnedPtr, d_ptr.position_space_buffer, output_memory_allocated*sizeof(OutputType),cudaMemcpyDeviceToHost,cudaStreamPerThread));
    postcheck
  }
  else
  {
    precheck
    cudaErr(cudaMemcpyAsync(tmpPinnedPtr, d_ptr.position_space, output_memory_allocated*sizeof(OutputType),cudaMemcpyDeviceToHost,cudaStreamPerThread));
    postcheck
  }


  // Just set true her for now
  bool should_block_until_complete = true;
  if (should_block_until_complete) cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

  precheck
  cudaErr(cudaHostUnregister(tmpPinnedPtr));
  postcheck

	if (free_gpu_memory) { Deallocate();}
  if (unpin_host_memory) { UnPinHostMemory();}

}



template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::FwdFFT(bool swap_real_space_quadrants, bool transpose_output)
{
  CheckDimensions();
  static constexpr bool use_thread_method = false;
  bool do_forward_transform = true;

  // SetPrecisionAndExectutionMethod(KernelType kernel_type, bool do_forward_transform, bool use_thread_method)
  switch (transform_dimension)
  {
    case 1: {
        // FIXME there is some redundancy in specifying _decomposed and use_thread_method
        // Note: the only time the non-transposed method should be used is for 1d data.
        if constexpr (use_thread_method)
        {
          if (is_real_valued_input) SetPrecisionAndExectutionMethod(r2c_decomposed, do_forward_transform); //FFT_R2C_decomposed(transpose_output);
          else SetPrecisionAndExectutionMethod(c2c_decomposed, do_forward_transform);
        }
        else
        {
          if (is_real_valued_input) SetPrecisionAndExectutionMethod<false>(r2c_transposed, do_forward_transform); //FFT_R2C_decomposed(transpose_output);
          else SetPrecisionAndExectutionMethod(c2c, do_forward_transform);
        }

        break;
    }
    case 2: {
      switch (size_change_type)
      {
        case none: {
          // FIXME there is some redundancy in specifying _decomposed and use_thread_method
          // Note: the only time the non-transposed method should be used is for 1d data.
          if (use_thread_method)
          {
            SetPrecisionAndExectutionMethod(r2c_decomposed_transposed, do_forward_transform);
            SetPrecisionAndExectutionMethod(c2c_decomposed, do_forward_transform);
          }
          else
          {
            SetPrecisionAndExectutionMethod(r2c_transposed, do_forward_transform);
            SetPrecisionAndExectutionMethod(c2c,            do_forward_transform);
          }
          break;
        }
        case increase: {
          SetPrecisionAndExectutionMethod(r2c_transposed, do_forward_transform);
          SetPrecisionAndExectutionMethod(c2c_padded,     do_forward_transform);          
          // FFT_R2C_WithPadding(transpose_output);
          // FFT_C2C_WithPadding(swap_real_space_quadrants);
          break;
        }
        case decrease: {
          MyFFTRunTimeAssertTrue(false, "Size reduction not yet supported");
          break;
        }
      }
      break; // case 2
    }
    case 3: {
      // Not yet supported
      MyFFTRunTimeAssertTrue(false, "3D FFT not yet supported");
      break;
    }
  }


}

template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::InvFFT(bool transpose_output)
{
  CheckDimensions();
  bool use_thread_method = false;
  bool do_forward_transform = false;

  switch (transform_dimension)
  {
    case 1: {
      if (is_real_valued_input) SetPrecisionAndExectutionMethod(c2r_decomposed, do_forward_transform); //FFT_R2C_decomposed(transpose_output);
      else SetPrecisionAndExectutionMethod(c2c_decomposed, do_forward_transform);

      break;
    }
    case 2: {
      switch (size_change_type)
      {
        case none: {
          // FIXME there is some redundancy in specifying _decomposed and use_thread_method
          // Note: the only time the non-transposed method should be used is for 1d data.
          if (use_thread_method)
          {
            SetPrecisionAndExectutionMethod(c2c_decomposed,            do_forward_transform);
            SetPrecisionAndExectutionMethod(c2r_decomposed_transposed, do_forward_transform);
          }
          else
          {
            SetPrecisionAndExectutionMethod(c2c,            do_forward_transform);
            SetPrecisionAndExectutionMethod(c2r_transposed, do_forward_transform);
          }          
          // // FFT_C2C(false);
          // // FFT_C2R_Transposed();
          // FFT_C2C_decomposed(false);
          // FFT_C2R_decomposed(true);
          break;
        }
        case increase: {
          SetPrecisionAndExectutionMethod(c2c,            do_forward_transform);
          SetPrecisionAndExectutionMethod(c2r_transposed, do_forward_transform);           
          // FFT_C2C(false);
          // FFT_C2R_Transposed();
          break;
        }
        case decrease: {
          // not defined;
          break;
        }
      }
      break; // case 2
    }
    case 3: {
      // Not yet supported
      MyFFTRunTimeAssertTrue(false, "3D FFT not yet supported");
      break;
    }
  }


}

template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::CrossCorrelate(float2* image_to_search, bool swap_real_space_quadrants)
{
  CheckDimensions();

  // Set the member pointer to the passed pointer
  d_ptr.image_to_search = image_to_search;

  switch (transform_dimension)
  {
    case 1: {
      MyFFTRunTimeAssertTrue(false, "1D FFT Cross correlation not yet supported");
      break;
    }
    case 2: {
      switch (size_change_type)
      {
        case none: {
          MyFFTRunTimeAssertTrue(false, "2D FFT Cross correlation without size change not yet supported");

          // FFT_R2C_decomposed(true);
          // FFT_C2C_decomposed_ConjMul_C2C(image_to_search, false);
          // FFT_C2R_decomposed(true);
        }
        case increase: {
          SetPrecisionAndExectutionMethod(r2c_transposed,   true);
          SetPrecisionAndExectutionMethod(xcorr_transposed, true);
          SetPrecisionAndExectutionMethod(c2r_transposed,   false);    
          // FFT_R2C_WithPadding();   
          // FFT_C2C_WithPadding_ConjMul_C2C(image_to_search, swap_real_space_quadrants);  
          // FFT_C2R_Transposed();
          break;
        }
        case decrease: {
          MyFFTRunTimeAssertTrue(false, "2D FFT Cross correlation without size decrease not yet supported");
          break;
        }
      }
      break; // case 2
    }
    case 3: {
      // Not yet supported
      MyFFTRunTimeAssertTrue(false, "3D FFT not yet supported");
      break;
    }
  }


}

template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::CrossCorrelate(__half2* image_to_search, bool swap_real_space_quadrants)
{
  CheckDimensions();

  // Set the member pointer to the passed pointer
  d_ptr.image_to_search = image_to_search;
  switch (transform_dimension)
  {
    case 1: {
      MyFFTRunTimeAssertTrue(false, "1D FFT Cross correlation not yet supported");
      break;
    }
    case 2: {
      switch (size_change_type)
      {
        case none: {
          MyFFTRunTimeAssertTrue(false, "2D FFT Cross correlation without size change not yet supported");
        }
        case increase: {
          SetPrecisionAndExectutionMethod(r2c_transposed,   true);
          SetPrecisionAndExectutionMethod(xcorr_transposed, true);
          SetPrecisionAndExectutionMethod(c2r_transposed,   false);     
          // FFT_R2C_WithPadding();
    
          // FFT_C2C_WithPadding_ConjMul_C2C(image_to_search, swap_real_space_quadrants);
    
          // FFT_C2R_Transposed();
          break;
        }
        case decrease: {
          MyFFTRunTimeAssertTrue(false, "3D FFT not yet supported");
          break;
        }
      }
      break; // case 2
    }
    case 3: {
      // Not yet supported
      MyFFTRunTimeAssertTrue(false, "3D FFT not yet supported");
      break;
    }
  }


}
////////////////////////////////////////////////////
/// END PUBLIC METHODS
////////////////////////////////////////////////////

template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::CheckDimensions()
{
  // This should be run inside any public method call to ensure things ar properly setup.

  // TODO - runtime asserts would be better as these are breaking errors that are under user control.
  // check to see if there is any measurable penalty for this.
  if ( ! is_size_validated )
  {
    MyFFTDebugAssertTrue(is_set_input_params, "Input parameters not set");
    MyFFTDebugAssertTrue(is_set_output_params, "Output parameters not set");
    MyFFTDebugAssertTrue(is_set_input_pointer, "The input data pointer is not set");

  
    if (dims_out.x > dims_in.x || dims_out.y > dims_in.y || dims_out.z > dims_in.z)
    {
      // For now we must pad in all dimensions, this is not needed and should be lifted. FIXME
      MyFFTDebugAssertTrue(dims_out.x >= dims_in.x, "If padding, all dimensions must be >=, x out < x in");
      MyFFTDebugAssertTrue(dims_out.y >= dims_in.y, "If padding, all dimensions must be >=, y out < y in");
      MyFFTDebugAssertTrue(dims_out.z >= dims_in.z, "If padding, all dimensions must be >=, z out < z in");
  
      size_change_type = increase;
    }
    else if (dims_out.x < dims_in.x || dims_out.y < dims_in.y || dims_out.z < dims_in.z)
    {
      MyFFTRunTimeAssertTrue( false, "Trimming (subset of output points) is yet to be implemented.");
      size_change_type = decrease;
    }
    else if (dims_out.x == dims_in.x && dims_out.y == dims_in.y && dims_out.z == dims_in.z)
    {
      size_change_type = none;
    }
    else
    {
      // TODO: if this is relaxed, the dimensionality check below will be invalid.
      MyFFTRunTimeAssertTrue( false, "Currently all dimensions must either increase, decrease or stay the same.");
    }

    // check for dimensionality
    // Note: this is predicated on the else clause ensuring all dimensions behave the same way w.r.t. size change.
    if (dims_in.z == 1 && dims_out.z == 1)
    {
      if (dims_in.y == 1 && dims_out.y == 1) 
      {
        transform_dimension = 1;
      }
      else 
      {
        transform_dimension = 2;
      }
    }
    else 
    {
      transform_dimension = 3;
    }
  
    is_size_validated = true;
  }

}

////////////////////////////////////////////////////
/// Transform kernels
////////////////////////////////////////////////////

// R2C_decomposed

template<class FFT, class ComplexType, class ScalarType>
__global__
void thread_fft_kernel_R2C_decomposed(const ScalarType*  __restrict__ input_values, ComplexType* __restrict__  output_values, Offsets mem_offsets, float twiddle_in, int Q)
{

  using complex_type = ComplexType;
  using scalar_type  = ScalarType;
  // The data store is non-coalesced, so don't aggregate the data in shared mem.
	extern __shared__  complex_type shared_mem[];

	// Memory used by FFT - for Thread() type, FFT::storage_size == FFT::elements_per_thread == size_of<FFT>::value
  complex_type thread_data[FFT::storage_size];
 
  io_thread<FFT>::load_r2c(&input_values[blockIdx.y*mem_offsets.pixel_pitch_input], thread_data, Q);

  // We then have Q FFTs of size size_of<FFT>::value (P in the paper)
	FFT().execute(thread_data);

  // Now we need to aggregate each of the Q transforms into each output block of size P
  
  io_thread<FFT>::remap_decomposed_segments(thread_data, shared_mem, twiddle_in, Q, mem_offsets.pixel_pitch_output);


  io_thread<FFT>::store_r2c(shared_mem, &output_values[blockIdx.y*mem_offsets.pixel_pitch_output], Q, mem_offsets.shared_output);

 
} // end of thread_fft_kernel_R2C

// R2C_decomposed_transposed

template<class FFT, class ComplexType, class ScalarType>
__global__
void thread_fft_kernel_R2C_decomposed_transposed(const ScalarType*  __restrict__ input_values, ComplexType* __restrict__  output_values, Offsets mem_offsets, float twiddle_in, int Q)
{

  using complex_type = ComplexType;
  using scalar_type  = ScalarType;
  // The data store is non-coalesced, so don't aggregate the data in shared mem.
	extern __shared__  complex_type shared_mem[];

	// Memory used by FFT - for Thread() type, FFT::storage_size == FFT::elements_per_thread == size_of<FFT>::value
  complex_type thread_data[FFT::storage_size];
 
  io_thread<FFT>::load_r2c(&input_values[blockIdx.y*mem_offsets.pixel_pitch_input], thread_data, Q);

  // We then have Q FFTs of size size_of<FFT>::value (P in the paper)
	FFT().execute(thread_data);

  // Now we need to aggregate each of the Q transforms into each output block of size P
  io_thread<FFT>::remap_decomposed_segments(thread_data, shared_mem, twiddle_in, Q, mem_offsets.shared_output);


  io_thread<FFT>::store_r2c_transposed(shared_mem, &output_values[blockIdx.y], Q, mem_offsets.pixel_pitch_output, mem_offsets.shared_output);

 
} // end of thread_fft_kernel_R2C_transposed

// R2C

template<class FFT, class ComplexType, class ScalarType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_R2C(const ScalarType* __restrict__ input_values, ComplexType*  __restrict__  output_values, Offsets mem_offsets, typename FFT::workspace_type workspace)
{
  // Initialize the shared memory, assuming everyting matches the input data X size in
  using complex_type = ComplexType;
  using scalar_type  = ScalarType;

  // The data store is non-coalesced, so don't aggregate the data in shared mem.
	extern __shared__  complex_type shared_mem[];


	// Memory used by FFT
  complex_type thread_data[FFT::storage_size];


  // No need to __syncthreads as each thread only accesses its own shared mem anyway
  // multiply Q*dims_out.w because x maps to y in the output transposed FFT
  io<FFT>::load_r2c(&input_values[blockIdx.y*mem_offsets.pixel_pitch_input], thread_data);

	// In the first FFT the modifying twiddle factor is 1 so the data are real
	FFT().execute(thread_data, shared_mem, workspace);
  
  io<FFT>::store_r2c_transposed(thread_data, output_values, mem_offsets.pixel_pitch_output);

 
} // end of block_fft_kernel_R2C

// R2C_WithPadding

template<class FFT, class ComplexType, class ScalarType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_R2C_WithPadding(const ScalarType* __restrict__  input_values, ComplexType* __restrict__  output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace)
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
  int output_MAP[FFT::storage_size];
  float twiddle_factor_args[FFT::storage_size];

  // No need to __syncthreads as each thread only accesses its own shared mem anyway
  // multiply Q*dims_out.w because x maps to y in the output transposed FFT
  io<FFT>::load_r2c_shared(&input_values[blockIdx.y*mem_offsets.pixel_pitch_input], shared_input, thread_data, twiddle_factor_args, twiddle_in, input_MAP, output_MAP, Q);

	// We unroll the first and last loops.
  // In the first FFT the modifying twiddle factor is 1 so the data are real
	FFT().execute(thread_data, shared_mem, workspace);  
  io<FFT>::store_r2c_transposed(thread_data, output_values, output_MAP, mem_offsets.pixel_pitch_output);

    // For the other fragments we need the initial twiddle
	for (int sub_fft = 1; sub_fft < Q-1; sub_fft++)
	{

	  io<FFT>::copy_from_shared(shared_input, thread_data, input_MAP);
		for (int i = 0; i < FFT::elements_per_thread; i++)
		{
			// Pre shift with twiddle
			__sincosf(twiddle_factor_args[i]*sub_fft,&twiddle.y,&twiddle.x);
			thread_data[i] *= twiddle;
		  // increment the output mapping. 
			output_MAP[i]++;
		}

		FFT().execute(thread_data, shared_mem, workspace);
    io<FFT>::store_r2c_transposed(thread_data, output_values, output_MAP, mem_offsets.pixel_pitch_output);
	}

  // For the last fragment we need to also do a bounds check.
  io<FFT>::copy_from_shared(shared_input, thread_data, input_MAP);
  for (int i = 0; i < FFT::elements_per_thread; i++)
  {
    // Pre shift with twiddle
    __sincosf(twiddle_factor_args[i]*(Q-1),&twiddle.y,&twiddle.x);
    thread_data[i] *= twiddle;
    // increment the output mapping. 
    output_MAP[i]++;
  }

  FFT().execute(thread_data, shared_mem, workspace);
  io<FFT>::store_r2c_transposed(thread_data, output_values, output_MAP, mem_offsets.pixel_pitch_output, mem_offsets.shared_output);
	


} // end of block_fft_kernel_R2C_WithPadding

// decomposed with conj multiplication

template<class FFT, class invFFT, class ComplexType>
__global__
void thread_fft_kernel_C2C_decomposed_ConjMul(const ComplexType* __restrict__ image_to_search, const ComplexType* __restrict__  input_values, ComplexType* __restrict__  output_values, Offsets mem_offsets, float twiddle_in, int Q)
{


  using complex_type = ComplexType;
  // The data store is non-coalesced, so don't aggregate the data in shared mem.
	extern __shared__  complex_type shared_mem[];

	// Memory used by FFT - for Thread() type, FFT::storage_size == FFT::elements_per_thread == size_of<FFT>::value
  complex_type thread_data[FFT::storage_size];
 
  io_thread<FFT>::load_c2c(&input_values[blockIdx.y*mem_offsets.pixel_pitch_input], thread_data, Q);

  // We then have Q FFTs of size size_of<FFT>::value (P in the paper)
	FFT().execute(thread_data);

  // Now we need to aggregate each of the Q transforms into each output block of size P
  io_thread<FFT>::remap_decomposed_segments(thread_data, shared_mem, twiddle_in, Q, mem_offsets.pixel_pitch_output);

  // 
  io_thread<invFFT>::load_shared_and_conj_multiply(&image_to_search[blockIdx.y*mem_offsets.pixel_pitch_input], shared_mem, thread_data, Q);

	invFFT().execute(thread_data);

  // Now we need to aggregate each of the Q transforms into each output block of size P
  io_thread<invFFT>::remap_decomposed_segments(thread_data, shared_mem, -twiddle_in, Q, mem_offsets.pixel_pitch_output);

  io_thread<invFFT>::store_c2c(shared_mem, &output_values[blockIdx.y*mem_offsets.pixel_pitch_output], Q);

}

// C2C with conj multiplication

template<class FFT, class invFFT, class ComplexType>
__launch_bounds__(invFFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_WithPadding_ConjMul_C2C(const ComplexType* __restrict__ image_to_search, const ComplexType*  __restrict__ input_values, ComplexType*  __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace_fwd, typename invFFT::workspace_type workspace_inv)
{

//	// Initialize the shared memory, assuming everyting matches the input data X size in
  using complex_type = ComplexType;

	__shared__ complex_type shared_mem[invFFT::shared_memory_size/sizeof(complex_type)]; // Storage for the input data that is re-used each blcok

  complex_type thread_data[FFT::storage_size];


  io<FFT>::load(&input_values[blockIdx.y*mem_offsets.pixel_pitch_input], thread_data, mem_offsets.shared_input);

	// In the first FFT the modifying twiddle factor is 1 so the data are reeal
	FFT().execute(thread_data, shared_mem, workspace_fwd);


  io<invFFT>::load_shared_and_conj_multiply(&image_to_search[blockIdx.y*mem_offsets.pixel_pitch_input], thread_data);

  invFFT().execute(thread_data, shared_mem, workspace_inv);

  io<invFFT>::store(thread_data, &output_values[blockIdx.y * mem_offsets.pixel_pitch_output]);



} // end of block_fft_kernel_C2C_WithPadding_ConjMul_C2C

template<class FFT, class invFFT, class ComplexType>
__launch_bounds__(invFFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_WithPadding_ConjMul_C2C_SwapRealSpaceQuadrants(const ComplexType* __restrict__ image_to_search, const ComplexType*  __restrict__ input_values, ComplexType*  __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace_fwd, typename invFFT::workspace_type workspace_inv)
{

//	// Initialize the shared memory, assuming everyting matches the input data X size in
  using complex_type = ComplexType;

	__shared__ complex_type shared_mem[invFFT::shared_memory_size/sizeof(complex_type)]; // Storage for the input data that is re-used each blcok

  complex_type thread_data[FFT::storage_size];


  io<FFT>::load(&input_values[blockIdx.y*mem_offsets.pixel_pitch_input], thread_data, mem_offsets.shared_input);

	// In the first FFT the modifying twiddle factor is 1 so the data are reeal
	FFT().execute(thread_data, shared_mem, workspace_fwd);

  // Swap real space quadrants using a phase shift by N/2 pixels 
  const unsigned int  stride = io<invFFT>::stride_size();
  int logical_y;
  for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
  {
    logical_y = threadIdx.x+ i*stride;
    if ( logical_y >= mem_offsets.pixel_pitch_output/2) logical_y -= mem_offsets.pixel_pitch_output;
    if ( (int(blockIdx.y) + logical_y) % 2 != 0) thread_data[i] *= -1.f; // FIXME TYPE
  }

  io<invFFT>::load_shared_and_conj_multiply(&image_to_search[blockIdx.y*mem_offsets.pixel_pitch_input], thread_data);

  invFFT().execute(thread_data, shared_mem, workspace_inv);

  io<invFFT>::store(thread_data, &output_values[blockIdx.y * mem_offsets.pixel_pitch_output]);



} // 

// C2C with Padding

template<class FFT, class ComplexType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_WithPadding(const ComplexType*  __restrict__ input_values, ComplexType*  __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace)
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
	FFT().execute(thread_data, shared_mem, workspace);

	// 
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

		FFT().execute(thread_data, shared_mem, workspace);

    io<FFT>::store(thread_data,shared_output,output_MAP);


	}

  // TODO confirm this is needed
	__syncthreads();

	// Now that the memory output can be coalesced send to global
  // FIXME is this actually coalced?
	for (int sub_fft = 0; sub_fft < Q; sub_fft++)
	{
    io<FFT>::store_coalesced(shared_output, &output_values[blockIdx.y * mem_offsets.pixel_pitch_output], sub_fft*mem_offsets.shared_input);
	}


} // end of block_fft_kernel_C2C_WithPadding

template<class FFT, class ComplexType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_WithPadding_SwapRealSpaceQuadrants(const ComplexType*  __restrict__  input_values, ComplexType*  __restrict__  output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace)
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
	FFT().execute(thread_data, shared_mem, workspace);

	// 
  io<FFT>::store_and_swap_quadrants(thread_data,shared_output,output_MAP,mem_offsets.pixel_pitch_input/2);

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

		FFT().execute(thread_data, shared_mem, workspace);
    io<FFT>::store_and_swap_quadrants(thread_data,shared_output,output_MAP,mem_offsets.pixel_pitch_input/2);


	}

  // TODO confirm this is needed
	__syncthreads();

	// Now that the memory output can be coalesced send to global
  // FIXME is this actually coalced?
	for (int sub_fft = 0; sub_fft < Q; sub_fft++)
	{
    io<FFT>::store_coalesced(shared_output, &output_values[blockIdx.y * mem_offsets.pixel_pitch_output], sub_fft*mem_offsets.shared_input);
	}


} // end of block_fft_kernel_C2C_WithPadding_SwapRealSpaceQuadrants

// C2C

template<class FFT, class ComplexType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C(const ComplexType*  __restrict__  input_values, ComplexType*  __restrict__  output_values, Offsets mem_offsets, typename FFT::workspace_type workspace)
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

// C2C decomposed

template<class FFT, class ComplexType>
__global__
void thread_fft_kernel_C2C_decomposed(const ComplexType* __restrict__  input_values, ComplexType* __restrict__  output_values, Offsets mem_offsets, float twiddle_in, int Q)
{


  using complex_type = ComplexType;
  // The data store is non-coalesced, so don't aggregate the data in shared mem.
	extern __shared__  complex_type shared_mem[];

	// Memory used by FFT - for Thread() type, FFT::storage_size == FFT::elements_per_thread == size_of<FFT>::value
  complex_type thread_data[FFT::storage_size];
 
  io_thread<FFT>::load_c2c(&input_values[blockIdx.y*mem_offsets.pixel_pitch_input], thread_data, Q);

  // We then have Q FFTs of size size_of<FFT>::value (P in the paper)
	FFT().execute(thread_data);

  // Now we need to aggregate each of the Q transforms into each output block of size P
  io_thread<FFT>::remap_decomposed_segments(thread_data, shared_mem, twiddle_in, Q, mem_offsets.pixel_pitch_output);


  io_thread<FFT>::store_c2c(shared_mem, &output_values[blockIdx.y*mem_offsets.pixel_pitch_output], Q);

}

// C2R transposed

template<class FFT, class ComplexType, class ScalarType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2R_Transposed(const ComplexType* __restrict__  input_values, ScalarType*  __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace)
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

// C2R decomposed

template<class FFT, class ComplexType, class ScalarType>
__global__
void thread_fft_kernel_C2R_decomposed(const ComplexType*  __restrict__ input_values, ScalarType*  __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q)
{
  using complex_type = ComplexType;
  using scalar_type  = ScalarType;

  // The data store is non-coalesced, so don't aggregate the data in shared mem.
	extern __shared__  scalar_type shared_mem_C2R_decomposed[];

	// Memory used by FFT - for Thread() type, FFT::storage_size == FFT::elements_per_thread == size_of<FFT>::value
  complex_type thread_data[FFT::storage_size];
 

  io_thread<FFT>::load_c2r(&input_values[blockIdx.y*mem_offsets.pixel_pitch_input], thread_data, Q, mem_offsets.pixel_pitch_input);

  // We then have Q FFTs of size size_of<FFT>::value (P in the paper)
	FFT().execute(thread_data);

  // Now we need to aggregate each of the Q transforms into each output block of size P
  io_thread<FFT>::remap_decomposed_segments_c2r(thread_data, shared_mem_C2R_decomposed, twiddle_in, Q);

  io_thread<FFT>::store_c2r(shared_mem_C2R_decomposed, &output_values[blockIdx.y*mem_offsets.pixel_pitch_output],Q);
}

template<class FFT, class ComplexType, class ScalarType>
__global__
void thread_fft_kernel_C2R_decomposed_transposed(const ComplexType*  __restrict__ input_values, ScalarType*  __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q)
{

  using complex_type = ComplexType;
  using scalar_type  = ScalarType;

  // The data store is non-coalesced, so don't aggregate the data in shared mem.
	extern __shared__  scalar_type shared_mem_transposed[];

	// Memory used by FFT - for Thread() type, FFT::storage_size == FFT::elements_per_thread == size_of<FFT>::value
  complex_type thread_data[FFT::storage_size];
 

  io_thread<FFT>::load_c2r_transposed(&input_values[blockIdx.y], thread_data, Q, mem_offsets.pixel_pitch_input, mem_offsets.pixel_pitch_output/2);

  // We then have Q FFTs of size size_of<FFT>::value (P in the paper)
	FFT().execute(thread_data);

  // Now we need to aggregate each of the Q transforms into each output block of size P
  io_thread<FFT>::remap_decomposed_segments_c2r(thread_data, shared_mem_transposed, twiddle_in, Q);

  io_thread<FFT>::store_c2r(shared_mem_transposed, &output_values[blockIdx.y*mem_offsets.pixel_pitch_output],Q);

}


template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::ClipIntoTopLeft()
{
  // TODO add some checks and logic.

  // Assuming we are calling this from R2C_Transposed and that the launch bounds are not set.
  dim3 threadsPerBlock;
  dim3 gridDims;

  threadsPerBlock = dim3(512,1,1);
  gridDims = dim3( (dims_out.x + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

  const short4 area_to_clip_from = make_short4(dims_in.x, dims_in.y, dims_in.w*2, dims_out.w*2);

  precheck
  clip_into_top_left_kernel<float, float><< < gridDims, threadsPerBlock, 0, cudaStreamPerThread >> >
  (d_ptr.position_space, d_ptr.position_space, area_to_clip_from);
  postcheck
}
 
template<typename InputType, typename OutputType>
__global__ void clip_into_top_left_kernel(InputType*  input_values, OutputType* output_values, short4 dims )
{

  int x = blockIdx.x*blockDim.x + threadIdx.x;
  if (x > dims.w) return; // Out of bounds. 

  // dims.w is the pitch of the output array
  if (blockIdx.y > dims.y) { output_values[blockIdx.y * dims.w + x] = OutputType(0); return; }

  if (threadIdx.x > dims.x) { output_values[blockIdx.y * dims.w + x] = OutputType(0); return; }
  else 
  {
    // dims.z is the pitch of the output array
    output_values[blockIdx.y * dims.w + x] = input_values[blockIdx.y * dims.z + x];
    return;
  }
} // end of clip_into_top_left_kernel

template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::ClipIntoReal(int wanted_coordinate_of_box_center_x, int wanted_coordinate_of_box_center_y, int wanted_coordinate_of_box_center_z)
{
  // TODO add some checks and logic.

  // Assuming we are calling this from R2C_Transposed and that the launch bounds are not set.
  dim3 threadsPerBlock;
  dim3 gridDims;
  int3 wanted_center = make_int3(wanted_coordinate_of_box_center_x, wanted_coordinate_of_box_center_y, wanted_coordinate_of_box_center_z);
  threadsPerBlock = dim3(32,32,1);
  gridDims = dim3( (dims_out.x + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (dims_out.y + threadsPerBlock.y - 1) / threadsPerBlock.y, 
                   1);

  const short4 area_to_clip_from = make_short4(dims_in.x, dims_in.y, dims_in.w*2, dims_out.w*2);
  float wanted_padding_value = 0.f;
  
  precheck
  clip_into_real_kernel<float, float><< < gridDims, threadsPerBlock, 0, cudaStreamPerThread >> >
  (d_ptr.position_space, d_ptr.position_space, dims_in, dims_out,wanted_center, wanted_padding_value);
  postcheck

}
// Modified from GpuImage::ClipIntoRealKernel
template<typename InputType, typename OutputType>
__global__ void clip_into_real_kernel(InputType* real_values_gpu,
                                      OutputType* other_image_real_values_gpu,
                                      short4 dims, 
                                      short4 other_dims,
                                      int3 wanted_coordinate_of_box_center, 
                                      OutputType wanted_padding_value)
{
  int3 other_coord = make_int3(blockIdx.x*blockDim.x + threadIdx.x,
                               blockIdx.y*blockDim.y + threadIdx.y,
                               blockIdx.z);

  int3 coord = make_int3(0, 0, 0); 

  if (other_coord.x < other_dims.x &&
      other_coord.y < other_dims.y &&
      other_coord.z < other_dims.z)
  {

    coord.z = dims.z/2 + wanted_coordinate_of_box_center.z + 
    other_coord.z - other_dims.z/2;

    coord.y = dims.y/2 + wanted_coordinate_of_box_center.y + 
    other_coord.y - other_dims.y/2;

    coord.x = dims.x + wanted_coordinate_of_box_center.x + 
    other_coord.x - other_dims.x;

    if (coord.z < 0 || coord.z >= dims.z || 
        coord.y < 0 || coord.y >= dims.y ||
        coord.x < 0 || coord.x >= dims.x)
    {
      other_image_real_values_gpu[ d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims) ] = wanted_padding_value;
    }
    else
    {
      other_image_real_values_gpu[ d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims) ] = 
      real_values_gpu[ d_ReturnReal1DAddressFromPhysicalCoord(coord, dims) ];
    }

  } // end of bounds check

} // end of ClipIntoRealKernel

template <class ComputeType, class InputType, class OutputType>
template <bool use_thread_method>
void FourierTransformer<ComputeType, InputType, OutputType>::SetPrecisionAndExectutionMethod(KernelType kernel_type, bool do_forward_transform)
{
  // For kernels with fwd and inv transforms, we want to not set the direction yet.
  

  static const bool is_half = std::is_same_v<ComputeType, __half>;
  static const bool is_float = std::is_same_v<ComputeType, float>;
  static_assert( is_half || is_float , "FourierTransformer::SetPrecisionAndExectutionMethod: Unsupported ComputeType");


  if constexpr (use_thread_method)
  {
    using FFT = decltype(Thread() + Size<32>() + Precision<ComputeType>());
    SelectSizeAndType<FFT>(kernel_type, do_forward_transform);

  }
  else
  {
    using FFT = decltype(Block() + Precision<ComputeType>() + ElementsPerThread<elements_per_thread_complex>()  + FFTsPerBlock<1>());
    SelectSizeAndType<FFT>(kernel_type, do_forward_transform);
  }
  

}

template <class ComputeType, class InputType, class OutputType>
template <class FFT_base>
void FourierTransformer<ComputeType, InputType, OutputType>::SelectSizeAndType(KernelType kernel_type, bool do_forward_transform)
{

  // For all cases, we want to get the device arch. TODO consider other device properties here. (Or should that go in the kernel launch?)
  int device, arch;
  GetCudaDeviceArch( device, arch );

  if constexpr (detail::is_operator<fft_operator::thread, FFT_base>::value)
  {
    GetTransformSize_thread(kernel_type, size_of<FFT_base>::value);
    switch (arch)
    {
      case 700: { using FFT = decltype(FFT_base() + SM<700>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
      case 750: { using FFT = decltype(FFT_base() + SM<750>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
      case 800: { using FFT = decltype(FFT_base() + SM<800>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
    }
  }
  else
  {
    GetTransformSize(kernel_type);

    switch (transform_size)
    {
      case 64: {
        switch (arch)
        {
          case 700: { using FFT = decltype(FFT_base()  + Size<64>()  + SM<700>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 750: { using FFT = decltype(FFT_base()  + Size<64>()  + SM<750>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 800: { using FFT = decltype(FFT_base()  + Size<64>()  + SM<800>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
        }
        break; }
  
      case 128: {
        switch (arch)
        {
          case 700: { using FFT = decltype(FFT_base()  + Size<128>()  + SM<700>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 750: { using FFT = decltype(FFT_base()  + Size<128>()  + SM<750>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 800: { using FFT = decltype(FFT_base()  + Size<128>()  + SM<800>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
        }
        break; }
  
      case 256: {
        switch (arch)
        {
          case 700: { using FFT = decltype(FFT_base()  + Size<256>()  + SM<700>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 750: { using FFT = decltype(FFT_base()  + Size<256>()  + SM<750>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 800: { using FFT = decltype(FFT_base()  + Size<256>()  + SM<800>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
        }
        break; } 
  
      case 512: {
        switch (arch)
        {
          case 700: { using FFT = decltype(FFT_base()  + Size<512>()  + SM<700>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 750: { using FFT = decltype(FFT_base()  + Size<512>()  + SM<750>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 800: { using FFT = decltype(FFT_base()  + Size<512>()  + SM<800>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
        }
        break; } 
  
      // case 768: {
      //   switch (arch)
      //   {
      //     case 700: { using FFT = decltype(FFT_base()  + Size<768>()  + SM<700>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
      //     case 750: { using FFT = decltype(FFT_base()  + Size<768>()  + SM<750>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
      //     case 800: { using FFT = decltype(FFT_base()  + Size<768>()  + SM<800>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
      //   }
      // break; } 
  
      case 1024: {
        switch (arch)
        {
          case 700: { using FFT = decltype(FFT_base()  + Size<1024>()  + SM<700>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 750: { using FFT = decltype(FFT_base()  + Size<1024>()  + SM<750>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 800: { using FFT = decltype(FFT_base()  + Size<1024>()  + SM<800>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
        }
      break; } 
  
      // case 1536: {
      //   switch (arch)
      //   {
      //     case 700: { using FFT = decltype(FFT_base()  + Size<1536>()  + SM<700>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
      //     // case 750: { using FFT = decltype(FFT_base()  + Size<1536>()  + SM<750>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
      //     case 800: { using FFT = decltype(FFT_base()  + Size<1536>()  + SM<800>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
      //   }
      // break; }    
  
      case 2048: {
        switch (arch)
        {
          case 700: { using FFT = decltype(FFT_base()  + Size<2048>()  + SM<700>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 750: { using FFT = decltype(FFT_base()  + Size<2048>()  + SM<750>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 800: { using FFT = decltype(FFT_base()  + Size<2048>()  + SM<800>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
        }
        break; } 
  
  
      case 4096: {
        switch (arch)
        {
          case 700: { using FFT = decltype(FFT_base()  + Size<4096>()  + SM<700>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          // case 750: { using FFT = decltype(FFT_base()  + Size<4096>()  + SM<750>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 800: { using FFT = decltype(FFT_base()  + Size<4096>()  + SM<800>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
        }
        break; }  
  
      case 8192: {
        switch (arch)
        {
          case 700: { using FFT = decltype(FFT_base()  + Size<8192>()  + SM<700>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 800: { using FFT = decltype(FFT_base()  + Size<8192>()  + SM<800>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
        }
        break; } 
    }

  }

}

template <class ComputeType, class InputType, class OutputType>
template <class FFT_base_arch, bool use_thread_method>
void FourierTransformer<ComputeType, InputType, OutputType>::SetAndLaunchKernel(KernelType kernel_type, bool do_forward_transform)
{

  using complex_type = typename FFT_base_arch::value_type;
	using scalar_type    = typename complex_type::value_type;

  complex_type* complex_input;
  complex_type* complex_output;
  scalar_type*  scalar_input;
  scalar_type*  scalar_output;

  // Make sure we are in the right chunk of the memory pool.
  if (is_in_buffer_memory) 
  {
    complex_input  = (complex_type*)d_ptr.momentum_space_buffer;
    complex_output = (complex_type*)d_ptr.momentum_space;

    scalar_input   = (scalar_type*)d_ptr.position_space_buffer;
    scalar_output  = (scalar_type*)d_ptr.position_space;

    is_in_buffer_memory = false;
  }
  else
  {
    complex_input  = (complex_type*)d_ptr.momentum_space;
    complex_output = (complex_type*)d_ptr.momentum_space_buffer;

    scalar_input   = (scalar_type*)d_ptr.position_space;
    scalar_output  = (scalar_type*)d_ptr.position_space_buffer;

    is_in_buffer_memory = true;
  }

  
  if constexpr (detail::is_operator<fft_operator::thread, FFT_base_arch>::value)
  {
    switch (kernel_type)
    {
      case r2c_decomposed: {

        using FFT = decltype( FFT_base_arch() + Direction<fft_direction::forward>() + Type<fft_type::c2c>() ); 

        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, r2c_decomposed);  
      
        int shared_mem = LP.mem_offsets.shared_output * sizeof(complex_type);
      
        precheck
        thread_fft_kernel_R2C_decomposed<FFT,complex_type,scalar_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
        (scalar_input, complex_output, LP.mem_offsets, LP.twiddle_in, LP.Q);
        postcheck
            
        break; 
      }

      case r2c_decomposed_transposed: {

        using FFT = decltype( FFT_base_arch() + Direction<fft_direction::forward>() + Type<fft_type::c2c>() ); 

        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, r2c_decomposed_transposed);
      
        int shared_mem = LP.mem_offsets.shared_output * sizeof(complex_type);

        precheck
        thread_fft_kernel_R2C_decomposed_transposed<FFT,complex_type,scalar_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
        (scalar_input, complex_output, LP.mem_offsets, LP.twiddle_in, LP.Q);
        postcheck

        break; 
      }
    case c2r_decomposed: {

      // Note that unlike the block C2R we require a C2C sub xform.
      using FFT = decltype(FFT_base_arch() + Direction<fft_direction::inverse>() + Type<fft_type::c2c>());
      // TODO add completeness check.

      LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2r_decomposed);
      int shared_memory = LP.mem_offsets.shared_output * sizeof(scalar_type);
      precheck
      thread_fft_kernel_C2R_decomposed<FFT, complex_type, scalar_type><< <LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
      (complex_input, complex_output, LP.mem_offsets, LP.twiddle_in, LP.Q);
      postcheck

      break; 
    }
    case c2r_decomposed_transposed: {  
      // Note that unlike the block C2R we require a C2C sub xform.
      using FFT = decltype(FFT_base_arch() + Direction<fft_direction::inverse>() + Type<fft_type::c2c>());

      LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2r_decomposed_transposed);
      int shared_memory = LP.mem_offsets.shared_output * sizeof(scalar_type);
  
      precheck
      thread_fft_kernel_C2R_decomposed_transposed<FFT, complex_type, scalar_type><< <LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
      (complex_input, complex_output, LP.mem_offsets, LP.twiddle_in, LP.Q);
      postcheck
      

      break; 
    } 
    case xcorr_decomposed: {

      using    FFT = decltype( FFT_base_arch() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() );  
      using invFFT = decltype( FFT_base_arch() + Type<fft_type::c2c>() + Direction<fft_direction::inverse>() ); 

      LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, xcorr_decomposed);

      int shared_mem = LP.mem_offsets.shared_output * sizeof(complex_type);
      
      // FIXME
      bool swap_real_space_quadrants = false;

      if (swap_real_space_quadrants)
      {
        MyFFTRunTimeAssertTrue(false, "decomposed xcorr with swap real space quadrants is not implemented.");
        // precheck
        // block_fft_kernel_C2C_WithPadding_ConjMul_C2C_SwapRealSpaceQuadrants<FFT,invFFT, complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
        // ( (complex_type*) image_to_search, (complex_type*)  d_ptr.momentum_space_buffer,  (complex_type*) d_ptr.momentum_space, LP.mem_offsets, LP.twiddle_in,LP.Q, workspace_fwd, workspace_inv);
        // postcheck
      }
      else
      {
        // the image_to_search pointer is set during call to CrossCorrelate,
        precheck
        thread_fft_kernel_C2C_decomposed_ConjMul<FFT, invFFT, complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
        ( (complex_type *)d_ptr.image_to_search, complex_input, complex_output, LP.mem_offsets, LP.twiddle_in,LP.Q);
        postcheck
      }
      
      break; 
    }
    case c2c_decomposed: {
      using FFT_nodir = decltype(FFT_base_arch() + Type<fft_type::c2c>() );

      LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2c_decomposed, do_forward_transform);

      if (do_forward_transform)
      {
        using FFT = decltype( FFT_nodir() + Direction<fft_direction::forward>() );
        int shared_mem = LP.mem_offsets.shared_output * sizeof(complex_type);
    
        precheck
        thread_fft_kernel_C2C_decomposed<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
        ( complex_input, complex_output, LP.mem_offsets, LP.twiddle_in, LP.Q);
        postcheck
      }
      else
      {
    
        using FFT = decltype( FFT_nodir() + Direction<fft_direction::inverse>() );
        int shared_mem = LP.mem_offsets.shared_output * sizeof(complex_type);
    
        precheck
        thread_fft_kernel_C2C_decomposed<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
        ( complex_input, complex_output, LP.mem_offsets, LP.twiddle_in, LP.Q);
        postcheck
      }
    }
    
    break; 
    }    
  }
  else
  {
    switch (kernel_type)
    {
      case r2c_transposed: {
  
        using FFT = decltype( FFT_base_arch() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() );  
        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, r2c_transposed);
  
        cudaError_t error_code = cudaSuccess;
        auto workspace = make_workspace<FFT>(error_code);
  
        switch (size_change_type)
        {
          case decrease: {
            MyFFTRunTimeAssertTrue(false, "r2c_transposed with decreasing size is not yet implemented.");
            break;
          }
          case increase: {
            int shared_mem = LP.mem_offsets.shared_input*sizeof(scalar_type) + FFT::shared_memory_size;
  
            precheck
            block_fft_kernel_R2C_WithPadding<FFT,complex_type,scalar_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
            ( scalar_input, complex_output, LP.mem_offsets, LP.twiddle_in,LP.Q, workspace);
            postcheck          
            break;
          }
          case none: {
            int shared_mem = FFT::shared_memory_size;
  
            // cudaErr(cudaSetDevice(0));
            //  cudaErr(cudaFuncSetCacheConfig( (void*)block_fft_kernel_R2C<FFT,complex_type,scalar_type>,cudaFuncCachePreferShared ));
            //  cudaFuncSetSharedMemConfig ( (void*)block_fft_kernel_R2C<FFT,complex_type,scalar_type>, cudaSharedMemBankSizeEightByte );
      
            precheck
            block_fft_kernel_R2C<FFT,complex_type,scalar_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
            (scalar_input, complex_output, LP.mem_offsets, workspace);
            postcheck          
            break;
          }
        }
  
        break; 
      }
  
      case c2c_padded: {
  
        using FFT = decltype(FFT_base_arch() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() );  
  
        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2c_padded);
        
        cudaError_t error_code = cudaSuccess;
        auto workspace = make_workspace<FFT>(error_code);
        
          // cudaErr(cudaFuncSetCacheConfig( (void*)block_fft_kernel_C2C_WithPadding<FFT,complex_type>,cudaFuncCachePreferShared ));
          // cudaFuncSetSharedMemConfig ( (void*)block_fft_kernel_C2C_WithPadding<FFT,complex_type>, cudaSharedMemBankSizeEightByte );
      
        int shared_mem;
        // Aggregate the transformed frequency data in shared memory so that we can write to global coalesced.
        shared_mem = LP.mem_offsets.shared_output*sizeof(complex_type) + LP.mem_offsets.shared_input*sizeof(complex_type) + FFT::shared_memory_size;
        // When it is the output dims being smaller, may need a logical or different method
        //FIXME
        bool swap_real_space_quadrants = false;
        if (swap_real_space_quadrants)
        {
          precheck
          block_fft_kernel_C2C_WithPadding_SwapRealSpaceQuadrants<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
          ( complex_input, complex_output, LP.mem_offsets, LP.twiddle_in,LP.Q, workspace);
          postcheck
        }
        else
        {
          precheck
          block_fft_kernel_C2C_WithPadding<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
          ( complex_input, complex_output, LP.mem_offsets, LP.twiddle_in,LP.Q, workspace);
          postcheck
        }
             
        // do something
        break; 
      }
      case c2c: {
  
        using FFT_nodir = decltype( FFT_base_arch() + Type<fft_type::c2c>() ); 
  
        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2c, do_forward_transform);
        
          
        if (do_forward_transform)
        {   
          using FFT = decltype( FFT_nodir() + Direction<fft_direction::forward>() );
          cudaError_t error_code = cudaSuccess;
          auto workspace = make_workspace<FFT>(error_code);
          int shared_mem = FFT::shared_memory_size;
          precheck
          block_fft_kernel_C2C<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
          ( (complex_type*)d_ptr.momentum_space_buffer,  (complex_type*)d_ptr.momentum_space, LP.mem_offsets, workspace);
          postcheck
      
          is_in_buffer_memory = false;
        }
        else
        {
          using FFT = decltype( FFT_nodir() + Direction<fft_direction::inverse>() );
          cudaError_t error_code = cudaSuccess;
          auto workspace = make_workspace<FFT>(error_code);
          int shared_mem = FFT::shared_memory_size;
          precheck
          block_fft_kernel_C2C<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
          ( (complex_type*)d_ptr.momentum_space,  (complex_type*)d_ptr.momentum_space_buffer, LP.mem_offsets, workspace);
          postcheck
        }
  
      
        // do something
        break; 
      }
      case c2r_transposed: {
  
        using FFT = decltype(FFT_base_arch() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() ); 
  
        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2r_transposed);
      
        cudaError_t error_code = cudaSuccess;
        auto workspace = make_workspace<FFT>(error_code);
        cudaErr(error_code);
  
        precheck
        block_fft_kernel_C2R_Transposed<FFT, complex_type, scalar_type><< <LP.gridDims, LP.threadsPerBlock, FFT::shared_memory_size, cudaStreamPerThread>> >
        ( complex_input, scalar_output, LP.mem_offsets, workspace);
        postcheck
  
        break; 
      }
      case xcorr_transposed: {
  
        using FFT    = decltype( FFT_base_arch() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() ); 
        using invFFT = decltype( FFT_base_arch() + Type<fft_type::c2c>() + Direction<fft_direction::inverse>() ); 
          
        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, xcorr_transposed);
  
        cudaError_t error_code = cudaSuccess;
        auto workspace_fwd = make_workspace<FFT>(error_code); // presumably larger of the two
        cudaErr(error_code);
        error_code = cudaSuccess;
        auto workspace_inv = make_workspace<invFFT>(error_code); // presumably larger of the two
        cudaErr(error_code);
  
        int shared_mem = invFFT::shared_memory_size;
        
          // cudaErr(cudaFuncSetCacheConfig( (void*)block_fft_kernel_C2C_WithPadding<FFT,complex_type>,cudaFuncCachePreferShared ));
          // cudaFuncSetSharedMemConfig ( (void*)block_fft_kernel_C2C_WithPadding<FFT,complex_type>, cudaSharedMemBankSizeEightByte );  
        // FIXME
        bool swap_real_space_quadrants = false;   
        if (swap_real_space_quadrants)
        {
          precheck
          block_fft_kernel_C2C_WithPadding_ConjMul_C2C_SwapRealSpaceQuadrants<FFT,invFFT, complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
          ( (complex_type *)d_ptr.image_to_search, complex_input, complex_output, LP.mem_offsets, LP.twiddle_in,LP.Q, workspace_fwd, workspace_inv);
          postcheck
        }
        else
        {
          precheck
          block_fft_kernel_C2C_WithPadding_ConjMul_C2C<FFT, invFFT, complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
          ( (complex_type *)d_ptr.image_to_search, complex_input, complex_output , LP.mem_offsets, LP.twiddle_in,LP.Q, workspace_fwd, workspace_inv);
          postcheck
        }
            
        // do something
        break; 
      }
      default:
        // throw something
        break;
  
    }
  }


    



  // 
}


} // namespace fast_FFT



