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

  is_size_validated = false;

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

void FourierTransformer::CheckDimensions()
{
  // This should be run inside any public method call to ensure things ar properly setup.
  if ( ! is_size_validated )
  {
    MyFFTDebugAssertTrue(is_set_input_params, "Input parameters not set");
    MyFFTDebugAssertTrue(is_set_output_params, "Output parameters not set");
  
    if (dims_out.x > dims_in.x || dims_out.y > dims_in.y || dims_out.z > dims_in.z)
    {
      // For now we must pad in all dimensions, this is not needed and should be lifted. FIXME
      MyFFTRunTimeAssertTrue(dims_out.x >= dims_in.x, "If padding, all dimensions must be >=, x out < x in");
      MyFFTRunTimeAssertTrue(dims_out.y >= dims_in.y, "If padding, all dimensions must be >=, y out < y in");
      MyFFTRunTimeAssertTrue(dims_out.z >= dims_in.z, "If padding, all dimensions must be >=, z out < z in");
  
      size_change_type = increase;
    }
  
    MyFFTRunTimeAssertFalse(dims_out.x < dims_in.x || dims_out.y < dims_in.y || dims_out.z < dims_in.z, "Trimming (subset of output points) is yet to be implemented.");
  
    if (dims_out.x == dims_in.x && dims_out.y == dims_in.y && dims_out.z == dims_in.z)
    {
      size_change_type = none;
    }

    // check for dimensionality
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
  cudaErr(cudaMemcpyAsync(device_pointer_fp32, pinnedPtr, input_memory_allocated*sizeof(float),cudaMemcpyDeviceToHost,cudaStreamPerThread));
  bool should_block_until_complete = true;
	if (should_block_until_complete) cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
  postcheck

  is_in_buffer_memory = false;


}

void FourierTransformer::CopyDeviceToHost( bool free_gpu_memory, bool unpin_host_memory)
{
 
	MyFFTDebugAssertTrue(is_in_memory_device_pointer, "GPU memory not allocated");

  float* copy_pointer;
  if (is_in_buffer_memory) copy_pointer = buffer_fp32;
  else copy_pointer = device_pointer_fp32;

  precheck
	cudaErr(cudaMemcpyAsync(pinnedPtr, copy_pointer, input_memory_allocated*sizeof(float),cudaMemcpyDeviceToHost,cudaStreamPerThread));
  postcheck

  // Just set true her for now
  bool should_block_until_complete = true;
	if (should_block_until_complete) cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
  	// TODO add asserts etc.
	if (free_gpu_memory) { Deallocate();}
  if (unpin_host_memory) { UnPinHostMemory();}


}


void FourierTransformer::CopyDeviceToHost(float* output_pointer, bool free_gpu_memory, bool unpin_host_memory)
{
 
	MyFFTDebugAssertTrue(is_in_memory_device_pointer, "GPU memory not allocated");

  float* copy_pointer;
  if (is_in_buffer_memory) copy_pointer = buffer_fp32;
  else copy_pointer = device_pointer_fp32;

  // Assuming the output is not pinned, TODO change to optionally maintain as host_input as well.
  float* tmpPinnedPtr;
  precheck
  cudaErr(cudaHostRegister(output_pointer, sizeof(float)*output_memory_allocated, cudaHostRegisterDefault));
  postcheck
  
  precheck
  cudaErr(cudaHostGetDevicePointer( &tmpPinnedPtr, output_pointer, 0));
  postcheck
  
  precheck
	cudaErr(cudaMemcpyAsync(tmpPinnedPtr, copy_pointer, output_memory_allocated*sizeof(float),cudaMemcpyDeviceToHost,cudaStreamPerThread));
  postcheck

  // Just set true her for now
  bool should_block_until_complete = true;
  if (should_block_until_complete) cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

  precheck
  cudaErr(cudaHostUnregister(tmpPinnedPtr));
  postcheck

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


void FourierTransformer::FwdFFT(bool swap_real_space_quadrants)
{
  CheckDimensions();

  switch (transform_dimension)
  {
    case 1: {
      FFT_R2C_decomposed();
      break;
    }
    case 2: {
      switch (size_change_type)
      {
        case none: {
          FFT_R2C_Transposed();
          FFT_C2C(true);
          break;
        }
        case increase: {
          FFT_R2C_WithPadding_Transposed();
          FFT_C2C_WithPadding(swap_real_space_quadrants);
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


void FourierTransformer::InvFFT()
{
  CheckDimensions();

  switch (transform_dimension)
  {
    case 1: {
      // In progress
      break;
    }
    case 2: {
      switch (size_change_type)
      {
        case none: {
          FFT_C2C(false);
          FFT_C2R_Transposed();
          break;
        }
        case increase: {
          FFT_C2C(false);
          FFT_C2R_Transposed();
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

void FourierTransformer::CrossCorrelate(float2* image_to_search, bool swap_real_space_quadrants)
{
  CheckDimensions();
  // Checks on input ft type

  switch (transform_dimension)
  {
    case 1: {
      // In progress
      break;
    }
    case 2: {
      switch (size_change_type)
      {
        case none: {
          // not defined
        }
        case increase: {
    
          FFT_R2C_WithPadding_Transposed();
    
          FFT_C2C_WithPadding_ConjMul_C2C(image_to_search, swap_real_space_quadrants);
    
          FFT_C2R_Transposed();
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
  
////////////////////////////////////////////////////
/// END PUBLIC METHODS
////////////////////////////////////////////////////


template<class FFT> void FourierTransformer::FFT_R2C_decomposed_t()
{

  // Note unlike block transforms, we get the transform size here, it must be before LaunchParams. TODO add logical checks
  GetTransformSize_thread(dims_in.x, size_of<FFT>::value);
  LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, r2c_decomposed);

  using complex_type = typename FFT::value_type;
  using scalar_type = typename complex_type::value_type;


  int shared_mem = LP.mem_offsets.shared_output * sizeof(complex_type);

  // cudaErr(cudaSetDevice(0));
  //  cudaErr(cudaFuncSetCacheConfig( (void*)block_fft_kernel_R2C_Transposed<FFT,complex_type,scalar_type>,cudaFuncCachePreferShared ));
  //  cudaFuncSetSharedMemConfig ( (void*)block_fft_kernel_R2C_Transposed<FFT,complex_type,scalar_type>, cudaSharedMemBankSizeEightByte );

  precheck
  thread_fft_kernel_R2C_decomposed<FFT,complex_type,scalar_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
  ((scalar_type*) device_pointer_fp32,  (complex_type*) buffer_fp32_complex, LP.mem_offsets, LP.twiddle_in, LP.Q);
  postcheck

  is_in_buffer_memory = true;
}

void FourierTransformer::FFT_R2C_decomposed()
{

  int device, arch;
  GetCudaDeviceArch( device, arch );
  // Get transform size is called in FFT_R2C_decomposed_t() for thread based xform.

  switch (arch)
  {
    case 700: { using FFT = decltype(FFT_thread_base() + Direction<fft_direction::forward>() + Type<fft_type::c2c>() + SM<700>());  FFT_R2C_decomposed_t<FFT>(); break;}
    case 750: { using FFT = decltype(FFT_thread_base() + Direction<fft_direction::forward>() + Type<fft_type::c2c>() + SM<750>());  FFT_R2C_decomposed_t<FFT>(); break;}
    case 800: { using FFT = decltype(FFT_thread_base() + Direction<fft_direction::forward>() + Type<fft_type::c2c>() + SM<800>());  FFT_R2C_decomposed_t<FFT>(); break;}
  }
  
}

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
  complex_type twiddle;
 
  // Each thread reads in the input data at stride = Q
  unsigned int index  = threadIdx.x + blockIdx.y*mem_offsets.pixel_pitch_input;
  for (unsigned int i = 0; i < size_of<FFT>::value; i++) 
  {
    thread_data[i].x = input_values[index];
    thread_data[i].y = scalar_type(0);
    index += Q;
  }
  
  // We then have Q FFTs of size size_of<FFT>::value (P in the paper)
	FFT().execute(thread_data);

  // Now we need to aggregate each of the Q transforms into each output block of size P
  
  // Unroll the first loop and initialize the shared mem. 
  index = threadIdx.x * size_of<FFT>::value;
  twiddle_in *= threadIdx.x; // twiddle factor arg now just needs to multiplied by K = (index + i)
  for (unsigned int i = 0; i < size_of<FFT>::value; i++)
  {
    __sincosf( twiddle_in * (index + i) ,&twiddle.y,&twiddle.x);
    twiddle *= thread_data[i];
    if (index + i < mem_offsets.pixel_pitch_output) shared_mem[index +  i] = twiddle;
  }
  
  for (unsigned int sub_fft = 1; sub_fft < Q; sub_fft++)
  {
    // wrap around, 0 --> 1, Q-1 --> 0 etc.
    index = ((threadIdx.x + sub_fft) % Q) * size_of<FFT>::value;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++)
    {
      __sincosf( twiddle_in * (index + i) ,&twiddle.y,&twiddle.x);
      twiddle *= thread_data[i];
      if (index + i < mem_offsets.pixel_pitch_output) shared_mem[index +  i] += twiddle;
    }
  }
  

  // Each thread reads in the input data at stride = mem_offsets.Q
  index  = threadIdx.x;
  for (unsigned int i = 0; i < size_of<FFT>::value/2; i++) 
  {
    output_values[index + blockIdx.y*mem_offsets.pixel_pitch_input] = shared_mem[index];
    index += Q;
  }
  if (index < mem_offsets.pixel_pitch_output)
  {
    output_values[index + blockIdx.y*mem_offsets.pixel_pitch_input] = shared_mem[index];
  }

 
} // end of block_fft_kernel_R2C

template<class FFT>
void FourierTransformer::FFT_R2C_Transposed_t()
{

  LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, r2c_transposed);

  using complex_type = typename FFT::value_type;
  using scalar_type = typename complex_type::value_type;
  cudaError_t error_code = cudaSuccess;
  auto workspace = make_workspace<FFT>(error_code);

  int shared_mem = FFT::shared_memory_size;

  // cudaErr(cudaSetDevice(0));
   cudaErr(cudaFuncSetCacheConfig( (void*)block_fft_kernel_R2C_Transposed<FFT,complex_type,scalar_type>,cudaFuncCachePreferShared ));
   cudaFuncSetSharedMemConfig ( (void*)block_fft_kernel_R2C_Transposed<FFT,complex_type,scalar_type>, cudaSharedMemBankSizeEightByte );

  precheck
  block_fft_kernel_R2C_Transposed<FFT,complex_type,scalar_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
  ((scalar_type*) device_pointer_fp32,  (complex_type*) buffer_fp32_complex, LP.mem_offsets, workspace);
  postcheck

  is_in_buffer_memory = true;
}

void FourierTransformer::FFT_R2C_Transposed()
{

  int device, arch;
  GetCudaDeviceArch( device, arch );
  GetTransformSize(dims_in.x);

  switch (transform_size)
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
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_R2C_Transposed_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_R2C_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_R2C_Transposed_t<FFT>(); break;}
      }
      break; }

    case 256: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<256>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_R2C_Transposed_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<256>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_R2C_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<256>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_R2C_Transposed_t<FFT>(); break;}
      }
      break; } 

    case 512: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<512>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_R2C_Transposed_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<512>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_R2C_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<512>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_R2C_Transposed_t<FFT>(); break;}
      }
      break; } 

    // case 768: {
    //   switch (arch)
    //   {
    //     case 700: { using FFT = decltype(FFT_base()  + Size<768>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_R2C_Transposed_t<FFT>(); break;}
    //     case 750: { using FFT = decltype(FFT_base()  + Size<768>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_R2C_Transposed_t<FFT>(); break;}
    //     case 800: { using FFT = decltype(FFT_base()  + Size<768>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_R2C_Transposed_t<FFT>(); break;}
    //   }
    // break; } 

    case 1024: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<1024>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_R2C_Transposed_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<1024>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_R2C_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<1024>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_R2C_Transposed_t<FFT>(); break;}
      }
    break; } 

    // case 1536: {
    //   switch (arch)
    //   {
    //     case 700: { using FFT = decltype(FFT_base()  + Size<1536>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_R2C_Transposed_t<FFT>(); break;}
    //     // case 750: { using FFT = decltype(FFT_base()  + Size<1536>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_R2C_Transposed_t<FFT>(); break;}
    //     case 800: { using FFT = decltype(FFT_base()  + Size<1536>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_R2C_Transposed_t<FFT>(); break;}
    //   }
    // break; }    

    case 2048: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<2048>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_R2C_Transposed_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<2048>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_R2C_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<2048>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_R2C_Transposed_t<FFT>(); break;}
      }
      break; } 


    case 4096: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<4096>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_R2C_Transposed_t<FFT>(); break;}
        // case 750: { using FFT = decltype(FFT_base()  + Size<4096>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_R2C_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<4096>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_R2C_Transposed_t<FFT>(); break;}
      }
      break; }  

    case 8192: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<8192>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_R2C_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<8192>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_R2C_Transposed_t<FFT>(); break;}
      }
      break; } 
  }
}

template<class FFT, class ComplexType, class ScalarType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_R2C_Transposed(const ScalarType* __restrict__ input_values, ComplexType*  __restrict__  output_values, Offsets mem_offsets, typename FFT::workspace_type workspace)
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

 
} // end of block_fft_kernel_R2C_Transposed

template<class FFT>
void FourierTransformer::FFT_R2C_WithPadding_Transposed_t()
{

  LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, r2c_transposed);

  using complex_type = typename FFT::value_type;
  using scalar_type = typename complex_type::value_type;
  cudaError_t error_code = cudaSuccess;
  auto workspace = make_workspace<FFT>(error_code);
  cudaErr(error_code);
  int shared_mem = LP.mem_offsets.shared_input*sizeof(scalar_type) + FFT::shared_memory_size;

  precheck
  block_fft_kernel_R2C_WithPadding_Transposed<FFT,complex_type,scalar_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
  ( (scalar_type*) device_pointer_fp32,  (complex_type*) buffer_fp32_complex, LP.mem_offsets, LP.twiddle_in,LP.Q, workspace);
  postcheck

  is_in_buffer_memory = true;
}

void FourierTransformer::FFT_R2C_WithPadding_Transposed()
{

  int device, arch;
  GetCudaDeviceArch( device, arch );
  GetTransformSize(dims_in.x);

  switch (transform_size)
  {
    case 64: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<64>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<64>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<64>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
      }
      break; }

    case 128: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
      }
      break; }

    case 256: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<256>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<256>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<256>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
      }
      break; } 

    case 512: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<512>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<512>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<512>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
      }
      break; } 

    // case 768: {
    //   switch (arch)
    //   {
    //     case 700: { using FFT = decltype(FFT_base()  + Size<768>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
    //     case 750: { using FFT = decltype(FFT_base()  + Size<768>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
    //     case 800: { using FFT = decltype(FFT_base()  + Size<768>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
    //   }
    // break; } 

    case 1024: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<1024>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<1024>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<1024>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
      }
    break; } 

    // case 1536: {
    //   switch (arch)
    //   {
    //     case 700: { using FFT = decltype(FFT_base()  + Size<1536>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
    //     // case 750: { using FFT = decltype(FFT_base()  + Size<1536>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
    //     case 800: { using FFT = decltype(FFT_base()  + Size<1536>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
    //   }
    // break; }   

    case 2048: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<2048>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<2048>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<2048>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
      }
      break; } 

    case 4096: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<4096>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
        // case 750: { using FFT = decltype(FFT_base()  + Size<4096>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<4096>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
      }
      break; }  

    case 8192: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<8192>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<8192>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_R2C_WithPadding_Transposed_t<FFT>(); break;}
      }
      break; } 
  }
}

template<class FFT, class ComplexType, class ScalarType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_R2C_WithPadding_Transposed(const ScalarType* __restrict__  input_values, ComplexType* __restrict__  output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace)
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
	


} // end of block_fft_kernel_R2C_WithPadding_Transposed

template<class FFT, class invFFT> 
void FourierTransformer::FFT_C2C_WithPadding_ConjMul_C2C_t(float2* image_to_search, bool swap_real_space_quadrants)
{
  
  LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, xcorr_transposed);

  // Assuming invFFT is >= in size to FFT and both are C2C
	using complex_type = typename FFT::value_type;
  cudaError_t error_code = cudaSuccess;
  auto workspace_fwd = make_workspace<FFT>(error_code); // presumably larger of the two
  cudaErr(error_code);
  error_code = cudaSuccess;
  auto workspace_inv = make_workspace<invFFT>(error_code); // presumably larger of the two
  cudaErr(error_code);
  int shared_mem = invFFT::shared_memory_size;

  // cudaErr(cudaFuncSetCacheConfig( (void*)block_fft_kernel_C2C_WithPadding<FFT,complex_type>,cudaFuncCachePreferShared ));
  // cudaFuncSetSharedMemConfig ( (void*)block_fft_kernel_C2C_WithPadding<FFT,complex_type>, cudaSharedMemBankSizeEightByte );


  if (swap_real_space_quadrants)
  {
    precheck
    block_fft_kernel_C2C_WithPadding_ConjMul_C2C_SwapRealSpaceQuadrants<FFT,invFFT, complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
    ( (complex_type*) image_to_search, (complex_type*)  buffer_fp32_complex,  (complex_type*) device_pointer_fp32_complex, LP.mem_offsets, LP.twiddle_in,LP.Q, workspace_fwd, workspace_inv);
    postcheck
  }
  else
  {
    precheck
    block_fft_kernel_C2C_WithPadding_ConjMul_C2C<FFT, invFFT, complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
    (  (complex_type*) image_to_search, (complex_type*)  buffer_fp32_complex,  (complex_type*) device_pointer_fp32_complex, LP.mem_offsets, LP.twiddle_in,LP.Q, workspace_fwd, workspace_inv);
    postcheck
  }

  is_in_buffer_memory = false;

}

void FourierTransformer::FFT_C2C_WithPadding_ConjMul_C2C(float2* image_to_search, bool swap_real_space_quadrants)
{

	// This is the first set of 1d ffts when the input data are real valued, accessing the strided dimension. Since we need the full length, it will actually run a C2C xform

  int device, arch;
  GetCudaDeviceArch( device, arch );
  GetTransformSize(dims_out.y);

  switch (transform_size)
  {
    case 64: {
      using FFT_noarch    = decltype(FFT_base()  + Size<64>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>());
      using invFFT_noarch = decltype(FFT_base()  + Size<64>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2c>());       
        switch (arch)
        {
          case 700: { using FFT = decltype(FFT_noarch() + SM<700>()); using invFFT = decltype(invFFT_noarch() + SM<700>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
          case 750: { using FFT = decltype(FFT_noarch() + SM<750>()); using invFFT = decltype(invFFT_noarch() + SM<750>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
          case 800: { using FFT = decltype(FFT_noarch() + SM<800>()); using invFFT = decltype(invFFT_noarch() + SM<800>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
        }
        break;
      }

    case 128: {
      using FFT_noarch    = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>());
      using invFFT_noarch = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2c>());
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_noarch() + SM<700>()); using invFFT = decltype(invFFT_noarch() + SM<700>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
        case 750: { using FFT = decltype(FFT_noarch() + SM<750>()); using invFFT = decltype(invFFT_noarch() + SM<750>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
        case 800: { using FFT = decltype(FFT_noarch() + SM<800>()); using invFFT = decltype(invFFT_noarch() + SM<800>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
      }
      break;
    }
 
    case 256: {
      using FFT_noarch    = decltype(FFT_base()  + Size<256>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>());
      using invFFT_noarch = decltype(FFT_base()  + Size<256>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2c>());
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_noarch() + SM<700>()); using invFFT = decltype(invFFT_noarch() + SM<700>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
        case 750: { using FFT = decltype(FFT_noarch() + SM<750>()); using invFFT = decltype(invFFT_noarch() + SM<750>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
        case 800: { using FFT = decltype(FFT_noarch() + SM<800>()); using invFFT = decltype(invFFT_noarch() + SM<800>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
      }
      break;
    }

    case 512: {
      using FFT_noarch    = decltype(FFT_base()  + Size<512>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>());
      using invFFT_noarch = decltype(FFT_base()  + Size<512>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2c>());
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_noarch() + SM<700>()); using invFFT = decltype(invFFT_noarch() + SM<700>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
        case 750: { using FFT = decltype(FFT_noarch() + SM<750>()); using invFFT = decltype(invFFT_noarch() + SM<750>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
        case 800: { using FFT = decltype(FFT_noarch() + SM<800>()); using invFFT = decltype(invFFT_noarch() + SM<800>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
      }
      break;
    }

    // case 768: {
    //   using FFT_noarch    = decltype(FFT_base()  + Size<768>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>());
    //   using invFFT_noarch = decltype(FFT_base()  + Size<768>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2c>());      
    //   switch (arch)
    //   {
    //     case 700: { using FFT = decltype(FFT_noarch() + SM<700>()); using invFFT = decltype(invFFT_noarch() + SM<700>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
    //     case 750: { using FFT = decltype(FFT_noarch() + SM<750>()); using invFFT = decltype(invFFT_noarch() + SM<750>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
    //     case 800: { using FFT = decltype(FFT_noarch() + SM<800>()); using invFFT = decltype(invFFT_noarch() + SM<800>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}        
    //   }
    // break; } 

    case 1024: {
      using FFT_noarch    = decltype(FFT_base()  + Size<1024>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>());
      using invFFT_noarch = decltype(FFT_base()  + Size<1024>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2c>());
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_noarch() + SM<700>()); using invFFT = decltype(invFFT_noarch() + SM<700>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
        case 750: { using FFT = decltype(FFT_noarch() + SM<750>()); using invFFT = decltype(invFFT_noarch() + SM<750>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
        case 800: { using FFT = decltype(FFT_noarch() + SM<800>()); using invFFT = decltype(invFFT_noarch() + SM<800>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}        
      }
    break; } 

    // case 1536: {
    //   using FFT_noarch    = decltype(FFT_base()  + Size<1536>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>());
    //   using invFFT_noarch = decltype(FFT_base()  + Size<1536>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2c>());
    //   switch (arch)
    //   {
    //     case 700: { using FFT = decltype(FFT_noarch() + SM<700>()); using invFFT = decltype(invFFT_noarch() + SM<700>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
    //     // case 750: { using FFT = decltype(FFT_noarch() + SM<750>()); using invFFT = decltype(invFFT_noarch() + SM<750>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
    //     case 800: { using FFT = decltype(FFT_noarch() + SM<800>()); using invFFT = decltype(invFFT_noarch() + SM<800>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}        
    //   }
    // break; }    

    case 2048: {
      using FFT_noarch    = decltype(FFT_base()  + Size<2048>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>());
      using invFFT_noarch = decltype(FFT_base()  + Size<2048>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2c>());
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_noarch() + SM<700>()); using invFFT = decltype(invFFT_noarch() + SM<700>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
        case 750: { using FFT = decltype(FFT_noarch() + SM<750>()); using invFFT = decltype(invFFT_noarch() + SM<750>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
        case 800: { using FFT = decltype(FFT_noarch() + SM<800>()); using invFFT = decltype(invFFT_noarch() + SM<800>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
      }
      break;
    }

    case 4096: {
      using FFT_noarch    = decltype(FFT_base()  + Size<4096>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>());
      using invFFT_noarch = decltype(FFT_base()  + Size<4096>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2c>());
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_noarch() + SM<700>()); using invFFT = decltype(invFFT_noarch() + SM<700>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
        case 750: { using FFT = decltype(FFT_noarch() + SM<750>()); using invFFT = decltype(invFFT_noarch() + SM<750>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
        case 800: { using FFT = decltype(FFT_noarch() + SM<800>()); using invFFT = decltype(invFFT_noarch() + SM<800>()); FFT_C2C_WithPadding_ConjMul_C2C_t<FFT, invFFT>(image_to_search,swap_real_space_quadrants); break;}
      }
      break;
    }
     
  } // end of switch on dims_in.y

  // Relies on the debug assert above

}

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

template <class FFT>
void FourierTransformer::FFT_C2C_WithPadding_t(bool swap_real_space_quadrants)
{

  LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2c_padded);


	using complex_type = typename FFT::value_type;
  cudaError_t error_code = cudaSuccess;
  auto workspace = make_workspace<FFT>(error_code);

  // cudaErr(cudaFuncSetCacheConfig( (void*)block_fft_kernel_C2C_WithPadding<FFT,complex_type>,cudaFuncCachePreferShared ));
  // cudaFuncSetSharedMemConfig ( (void*)block_fft_kernel_C2C_WithPadding<FFT,complex_type>, cudaSharedMemBankSizeEightByte );

  int shared_mem;
  // Aggregate the transformed frequency data in shared memory so that we can write to global coalesced.
  shared_mem = LP.mem_offsets.shared_output*sizeof(complex_type) + LP.mem_offsets.shared_input*sizeof(complex_type) + FFT::shared_memory_size;
  // When it is the output dims being smaller, may need a logical or different method
  if (swap_real_space_quadrants)
  {
    precheck
    block_fft_kernel_C2C_WithPadding_SwapRealSpaceQuadrants<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
    ( (complex_type*)  buffer_fp32_complex,  (complex_type*) device_pointer_fp32_complex, LP.mem_offsets, LP.twiddle_in,LP.Q, workspace);
    postcheck
  }
  else
  {
    precheck
    block_fft_kernel_C2C_WithPadding<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
    ( (complex_type*)  buffer_fp32_complex,  (complex_type*) device_pointer_fp32_complex, LP.mem_offsets, LP.twiddle_in,LP.Q, workspace);
    postcheck
  }


  is_in_buffer_memory = false;


}
void FourierTransformer::FFT_C2C_WithPadding(bool swap_real_space_quadrants)
{

	// This is the first set of 1d ffts when the input data are real valued, accessing the strided dimension. Since we need the full length, it will actually run a C2C xform

  int device, arch;
  GetCudaDeviceArch( device, arch );
  GetTransformSize(dims_in.y);

  switch (transform_size)
  {
    case 64: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<64>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<64>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<64>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
      }
      break; }

    case 128: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
      }
      break; }
 
    case 256: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<256>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<256>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<256>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
      }
      break; } 

    case 512: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<512>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<512>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<512>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
      }
      break; } 

    // case 768: {
    //   switch (arch)
    //   {
    //     case 700: { using FFT = decltype(FFT_base()  + Size<768>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
    //     case 750: { using FFT = decltype(FFT_base()  + Size<768>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
    //     case 800: { using FFT = decltype(FFT_base()  + Size<768>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
    //   }
    // break; } 

    case 1024: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<1024>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<1024>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<1024>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
      }
    break; } 

    // case 1536: {
    //   switch (arch)
    //   {
    //     case 700: { using FFT = decltype(FFT_base()  + Size<1536>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
    //     // case 750: { using FFT = decltype(FFT_base()  + Size<1536>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
    //     case 800: { using FFT = decltype(FFT_base()  + Size<1536>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
    //   }
    // break; }   

    case 2048: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<2048>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<2048>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<2048>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
      }
      break; } 

    case 4096: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<4096>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
        // case 750: { using FFT = decltype(FFT_base()  + Size<4096>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<750>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<4096>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
      }
      break; }  

    case 8192: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<8192>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<700>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<8192>() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() + SM<800>());  FFT_C2C_WithPadding_t<FFT>(swap_real_space_quadrants); break;}
      }
      break; }    
  }


}

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

template<class FFT_nodir>
void FourierTransformer::FFT_C2C_t( bool do_forward_transform )
{
  LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2c);

  
  if (do_forward_transform)
  {   
    using FFT = decltype( FFT_nodir() + Direction<fft_direction::forward>() );
    using complex_type = typename FFT::value_type;
    using scalar_type    = typename complex_type::value_type;
    cudaError_t error_code = cudaSuccess;
    auto workspace = make_workspace<FFT>(error_code);
    int shared_mem = FFT::shared_memory_size;
    precheck
    block_fft_kernel_C2C<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
    ( (complex_type*)buffer_fp32_complex,  (complex_type*)device_pointer_fp32_complex, LP.mem_offsets, workspace);
    postcheck

    is_in_buffer_memory = false;
  }
  else
  {
    using FFT = decltype( FFT_nodir() + Direction<fft_direction::inverse>() );
    using complex_type = typename FFT::value_type;
    using scalar_type    = typename complex_type::value_type;
    cudaError_t error_code = cudaSuccess;
    auto workspace = make_workspace<FFT>(error_code);
    int shared_mem = FFT::shared_memory_size;
    precheck
    block_fft_kernel_C2C<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
    ( (complex_type*)device_pointer_fp32_complex,  (complex_type*)buffer_fp32_complex, LP.mem_offsets, workspace);
    postcheck

    is_in_buffer_memory = true;
  }
  

}

void FourierTransformer::FFT_C2C( bool do_forward_transform )
{

  int device, arch;
  GetCudaDeviceArch( device, arch );
  GetTransformSize(dims_out.y);

  switch (transform_size)
  {
    case 64: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<64>() + Type<fft_type::c2c>() + SM<700>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<64>() + Type<fft_type::c2c>() + SM<750>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<64>() + Type<fft_type::c2c>() + SM<800>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
      }
      break; }

    case 128: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<128>() + Type<fft_type::c2c>() + SM<700>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<128>() + Type<fft_type::c2c>() + SM<750>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<128>() + Type<fft_type::c2c>() + SM<800>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
      }
      break; }
     
    case 256: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<256>() + Type<fft_type::c2c>() + SM<700>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<256>() + Type<fft_type::c2c>() + SM<750>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<256>() + Type<fft_type::c2c>() + SM<800>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
      }   
      break; } 

    case 512: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<512>() + Type<fft_type::c2c>() + SM<700>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<512>() + Type<fft_type::c2c>() + SM<750>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<512>() + Type<fft_type::c2c>() + SM<800>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
      }
      break; } 

    // case 768: {
    //   switch (arch)
    //   {
    //     case 700: { using FFT = decltype(FFT_base()  + Size<768>() + Type<fft_type::c2c>() + SM<700>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
    //     case 750: { using FFT = decltype(FFT_base()  + Size<768>() + Type<fft_type::c2c>() + SM<750>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
    //     case 800: { using FFT = decltype(FFT_base()  + Size<768>() + Type<fft_type::c2c>() + SM<800>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
    //   }
    // break; } 

    case 1024: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<1024>() + Type<fft_type::c2c>() + SM<700>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<1024>() + Type<fft_type::c2c>() + SM<750>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<1024>() + Type<fft_type::c2c>() + SM<800>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
      }
    break; } 

    // case 1536: {
    //   switch (arch)
    //   {
    //     case 700: { using FFT = decltype(FFT_base()  + Size<1536>() + Type<fft_type::c2c>() + SM<700>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
    //     // case 750: { using FFT = decltype(FFT_base()  + Size<1536>() + Type<fft_type::c2c>() + SM<750>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
    //     case 800: { using FFT = decltype(FFT_base()  + Size<1536>() + Type<fft_type::c2c>() + SM<800>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
    //   }
    // break; }  


    case 2048: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<2048>() + Type<fft_type::c2c>() + SM<700>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<2048>() + Type<fft_type::c2c>() + SM<750>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<2048>() + Type<fft_type::c2c>() + SM<800>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
      }
      break; }       

    case 4096: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<4096>() + Type<fft_type::c2c>() + SM<700>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
        // case 750: { using FFT = decltype(FFT_base()  + Size<4096>() + Type<fft_type::c2c>() + SM<750>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<4096>() + Type<fft_type::c2c>() + SM<800>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
      }
      break; }  

    case 8192: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<8192>() + Type<fft_type::c2c>() + SM<700>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<8192>() + Type<fft_type::c2c>() + SM<800>());  FFT_C2C_t<FFT>(do_forward_transform); break;}
      }
      break; }      
  }


}

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

template <class FFT>
void FourierTransformer::FFT_C2R_Transposed_t()
{
  LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2r_transposed);

	using complex_type = typename FFT::value_type;
	using scalar_type    = typename complex_type::value_type;
  cudaError_t error_code = cudaSuccess;
  auto workspace = make_workspace<FFT>(error_code);
  cudaErr(error_code);

  if (is_in_buffer_memory)
  {
    precheck
    block_fft_kernel_C2R_Transformed<FFT, complex_type, scalar_type><< <LP.gridDims, LP.threadsPerBlock, FFT::shared_memory_size, cudaStreamPerThread>> >
    ( (complex_type*)buffer_fp32_complex, (scalar_type*)device_pointer_fp32, LP.mem_offsets, workspace);
    postcheck
    is_in_buffer_memory = false;
  }
  else
  {
    precheck
    block_fft_kernel_C2R_Transformed<FFT, complex_type, scalar_type><< <LP.gridDims, LP.threadsPerBlock, FFT::shared_memory_size, cudaStreamPerThread>> >
    ( (complex_type*)device_pointer_fp32, (scalar_type*)buffer_fp32_complex, LP.mem_offsets, workspace);
    postcheck
    is_in_buffer_memory = true;
  }


}

void FourierTransformer::FFT_C2R_Transposed()
{


  int device, arch;
  GetCudaDeviceArch( device, arch );
  GetTransformSize(dims_out.x);

  switch (transform_size)
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
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<700>());  FFT_C2R_Transposed_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<750>());  FFT_C2R_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<128>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<800>());  FFT_C2R_Transposed_t<FFT>(); break;}
      }
      break; }
       
    case 256: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<256>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<700>());  FFT_C2R_Transposed_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<256>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<750>());  FFT_C2R_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<256>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<800>());  FFT_C2R_Transposed_t<FFT>(); break;}
      }
      break; } 

    case 512: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<512>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<700>());  FFT_C2R_Transposed_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<512>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<750>());  FFT_C2R_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<512>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<800>());  FFT_C2R_Transposed_t<FFT>(); break;}
      }
      break; } 

    // case 768: {
    //   switch (arch)
    //   {
    //     case 700: { using FFT = decltype(FFT_base()  + Size<768>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<700>());  FFT_C2R_Transposed_t<FFT>(); break;}
    //     case 750: { using FFT = decltype(FFT_base()  + Size<768>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<750>());  FFT_C2R_Transposed_t<FFT>(); break;}
    //     case 800: { using FFT = decltype(FFT_base()  + Size<768>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<800>());  FFT_C2R_Transposed_t<FFT>(); break;}
    //   }
    // break; } 

    case 1024: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<1024>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<700>());  FFT_C2R_Transposed_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<1024>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<750>());  FFT_C2R_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<1024>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<800>());  FFT_C2R_Transposed_t<FFT>(); break;}
      }
    break; } 

    // case 1536: {
    //   switch (arch)
    //   {
    //     case 700: { using FFT = decltype(FFT_base()  + Size<1536>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<700>());  FFT_C2R_Transposed_t<FFT>(); break;}
    //     // case 750: { using FFT = decltype(FFT_base()  + Size<1536>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<750>());  FFT_C2R_Transposed_t<FFT>(); break;}
    //     case 800: { using FFT = decltype(FFT_base()  + Size<1536>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<800>());  FFT_C2R_Transposed_t<FFT>(); break;}
    //   }
    // break; }  

    case 2048: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<2048>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<700>());  FFT_C2R_Transposed_t<FFT>(); break;}
        case 750: { using FFT = decltype(FFT_base()  + Size<2048>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<750>());  FFT_C2R_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<2048>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<800>());  FFT_C2R_Transposed_t<FFT>(); break;}
      }
      break; } 
      
    case 4096: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<4096>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<700>());  FFT_C2R_Transposed_t<FFT>(); break;}
        // case 750: { using FFT = decltype(FFT_base()  + Size<4096>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<750>());  FFT_C2R_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<4096>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<800>());  FFT_C2R_Transposed_t<FFT>(); break;}
      }
      break; }  

    case 8192: {
      switch (arch)
      {
        case 700: { using FFT = decltype(FFT_base()  + Size<8192>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<700>());  FFT_C2R_Transposed_t<FFT>(); break;}
        case 800: { using FFT = decltype(FFT_base()  + Size<8192>() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() + SM<800>());  FFT_C2R_Transposed_t<FFT>(); break;}
      }
      break; }      
     
  }


}

template<class FFT, class ComplexType, class ScalarType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2R_Transformed(const ComplexType* __restrict__  input_values, ScalarType*  __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace)
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


void FourierTransformer::ClipIntoTopLeft()
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
  (device_pointer_fp32, device_pointer_fp32, area_to_clip_from);
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


void FourierTransformer::ClipIntoReal(int wanted_coordinate_of_box_center_x, int wanted_coordinate_of_box_center_y, int wanted_coordinate_of_box_center_z)
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
  (device_pointer_fp32, device_pointer_fp32, dims_in, dims_out,wanted_center, wanted_padding_value);
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

} // namespace fast_FFT



