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
  GetCudaDeviceProps( device_properties );
  // exit(0);
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
  is_in_memory_host_pointer = false; // To track allocation of host side memory
  is_in_memory_device_pointer = false; // To track allocation of device side memory.
  is_in_buffer_memory = false; // To track whether the current result is in dev_ptr.position_space or dev_ptr.position_space_buffer (momemtum space/ momentum space buffer respectively.)
  transform_stage_completed = none;
  
  is_host_memory_pinned = false; // Specified in the constructor. Assuming host memory won't be pinned for many applications.
  
  is_fftw_padded_input = false; // Padding for in place r2c transforms
  is_fftw_padded_output = false; // Currently the output state will match the input state, otherwise it is an error.
  
  is_real_valued_input = true; // This is determined by the input type. If it is a float2 or __half2, then it is assumed to be a complex valued input function.
  
  is_set_input_params = false; // Yes, yes, "are" set.
  is_set_output_params = false;
  is_size_validated = false; // Defaults to false, set after both input/output dimensions are set and checked.
  is_set_input_pointer = false; // May be on the host of the device.


  compute_memory_allocated = 0;


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
void FourierTransformer<ComputeType, InputType, OutputType>::SetForwardFFTPlan(size_t input_logical_x_dimension,  size_t input_logical_y_dimension,  size_t input_logical_z_dimension, 
                                                                               size_t output_logical_x_dimension, size_t output_logical_y_dimension, size_t output_logical_z_dimension,
                                                                               bool is_padded_input, 
                                                                               bool is_host_memory_pinned, 
                                                                               OriginType input_origin_type)
{

  MyFFTDebugAssertTrue(input_logical_x_dimension > 0, "Input logical x dimension must be > 0");
  MyFFTDebugAssertTrue(input_logical_y_dimension > 0, "Input logical y dimension must be > 0");
  MyFFTDebugAssertTrue(input_logical_z_dimension > 0, "Input logical z dimension must be > 0");
  MyFFTDebugAssertTrue(output_logical_x_dimension > 0, "output logical x dimension must be > 0");
  MyFFTDebugAssertTrue(output_logical_y_dimension > 0, "output logical y dimension must be > 0");
  MyFFTDebugAssertTrue(output_logical_z_dimension > 0, "output logical z dimension must be > 0");

  fwd_dims_in  = make_short4(input_logical_x_dimension, input_logical_y_dimension, input_logical_z_dimension,0);
  fwd_dims_out = make_short4(output_logical_x_dimension, output_logical_y_dimension, output_logical_z_dimension,0);

  is_fftw_padded_input = is_padded_input; // Note: Must be set before ReturnPaddedMemorySize
  MyFFTRunTimeAssertTrue(is_fftw_padded_input, "Support for input arrays that are not FFTW padded needs to be implemented."); // FIXME

  // ReturnPaddedMemorySize also sets FFTW padding etc.
  input_memory_allocated = ReturnPaddedMemorySize(fwd_dims_in);
  fwd_output_memory_allocated = ReturnPaddedMemorySize(fwd_dims_out); // sets .w and also increases compute_memory_allocated if needed. 

  // The compute memory allocated is the max of all possible sizes.

  this->input_origin_type = input_origin_type;
  is_set_input_params = true;
}

template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::SetInverseFFTPlan(size_t input_logical_x_dimension,  size_t input_logical_y_dimension,  size_t input_logical_z_dimension, 
                                                                               size_t output_logical_x_dimension, size_t output_logical_y_dimension, size_t output_logical_z_dimension,
                                                                               bool is_padded_output, 
                                                                               OriginType output_origin_type)
{
  MyFFTDebugAssertTrue(is_set_input_params, "Please set the input paramters first.")
  MyFFTDebugAssertTrue(output_logical_x_dimension > 0, "output logical x dimension must be > 0");
  MyFFTDebugAssertTrue(output_logical_y_dimension > 0, "output logical y dimension must be > 0");
  MyFFTDebugAssertTrue(output_logical_z_dimension > 0, "output logical z dimension must be > 0");
  MyFFTDebugAssertTrue(is_fftw_padded_input == is_padded_output, "If the input data are FFTW padded, so must the output.");

  inv_dims_in  = make_short4(input_logical_x_dimension, input_logical_y_dimension, input_logical_z_dimension,0);
  inv_dims_out = make_short4(output_logical_x_dimension, output_logical_y_dimension, output_logical_z_dimension,0);

  ReturnPaddedMemorySize(inv_dims_in); // sets .w and also increases compute_memory_allocated if needed. 
  inv_output_memory_allocated = ReturnPaddedMemorySize(inv_dims_out);
  // The compute memory allocated is the max of all possible sizes.

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
 
  SetDimensions(CopyFromHost);
	MyFFTDebugAssertTrue(is_in_memory_host_pointer, "Host memory not allocated");

  // FIXME switch to stream ordered malloc
	if ( ! is_in_memory_device_pointer )
	{
    // Allocate enough for the out of place buffer as well.
    // MyFFTDebugPrintWithDetails("Allocating device memory for input pointer");
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
  cudaErr(cudaMemcpyAsync(d_ptr.position_space, pinnedPtr, memory_size_to_copy * sizeof(InputType),cudaMemcpyDeviceToHost,cudaStreamPerThread));
  postcheck
  // TODO r/n assuming InputType is _half, _half2, float, or _float2 (real, complex, real, complex) need to handle other types and convert
  bool should_block_until_complete = true; // FIXME after switching to stream ordered malloc this will not be needed.
	if (should_block_until_complete) cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

}

template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::CopyDeviceToHost( bool free_gpu_memory, bool unpin_host_memory, int n_elements_to_copy)
{
 
  SetDimensions(CopyToHost);  
  if (n_elements_to_copy != 0) memory_size_to_copy = n_elements_to_copy;
  
  // std::cout << "N elements " << n_elements_to_copy << " memory to copy " << memory_size_to_copy <<  std::endl;
	MyFFTDebugAssertTrue(is_in_memory_device_pointer, "GPU memory not allocated");
  ComputeType* copy_pointer;
  if (is_in_buffer_memory) copy_pointer = d_ptr.position_space_buffer;
  else copy_pointer = d_ptr.position_space;

  // FIXME this is assuming the input type matches the compute type.
  precheck
	cudaErr(cudaMemcpyAsync(pinnedPtr, copy_pointer, memory_size_to_copy * sizeof(InputType),cudaMemcpyDeviceToHost,cudaStreamPerThread));
  postcheck

  // Just set true her for now
  bool should_block_until_complete = true;
	if (should_block_until_complete) cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

  	// TODO add asserts etc.
	if (free_gpu_memory) { Deallocate();}

  if (unpin_host_memory) { UnPinHostMemory();}


}

template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::CopyDeviceToHost(OutputType* output_pointer, bool free_gpu_memory, bool unpin_host_memory, int n_elements_to_copy)
{
 
  SetDimensions(CopyToHost);
  if (n_elements_to_copy != 0) memory_size_to_copy = n_elements_to_copy;
	MyFFTDebugAssertTrue(is_in_memory_device_pointer, "GPU memory not allocated");
  // Assuming the output is not pinned, TODO change to optionally maintain as host_input as well.
  OutputType* tmpPinnedPtr;
  precheck
  // FIXME this is assuming output type is the same as compute type.
  cudaErr(cudaHostRegister(output_pointer, sizeof(OutputType)*memory_size_to_copy, cudaHostRegisterDefault));
  postcheck
  
  precheck
  cudaErr(cudaHostGetDevicePointer( &tmpPinnedPtr, output_pointer, 0));
  postcheck
  if (is_in_buffer_memory)
  {
    precheck
    cudaErr(cudaMemcpyAsync(tmpPinnedPtr, d_ptr.position_space_buffer, memory_size_to_copy*sizeof(OutputType),cudaMemcpyDeviceToHost,cudaStreamPerThread));
    postcheck
  }
  else
  {
    precheck
    cudaErr(cudaMemcpyAsync(tmpPinnedPtr, d_ptr.position_space, memory_size_to_copy*sizeof(OutputType),cudaMemcpyDeviceToHost,cudaStreamPerThread));
    postcheck
  }


  // Just set true her for now
  bool should_block_until_complete = true;
  if (should_block_until_complete) cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

  precheck
  cudaErr(cudaHostUnregister(output_pointer));
  postcheck

	if (free_gpu_memory) { Deallocate();}
  if (unpin_host_memory) { UnPinHostMemory();}

}



template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::FwdFFT(bool swap_real_space_quadrants, bool transpose_output)
{
  SetDimensions(FwdTransform);
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
          transform_stage_completed = TransformStageCompleted::fwd;

        }
        else
        {
          if (is_real_valued_input) 
          {
            switch (fwd_size_change_type)
            {
              case SizeChangeType::no_change:{ SetPrecisionAndExectutionMethod<false>(r2c_none_XY); break; }
              case SizeChangeType::decrease: { SetPrecisionAndExectutionMethod<false>(r2c_decrease); break; }
              case SizeChangeType::increase: { SetPrecisionAndExectutionMethod<false>(r2c_increase); break; }
              default: { MyFFTDebugAssertTrue(false, "Invalid size change type"); }
            }
          }
          else
          {
            switch (fwd_size_change_type)
            {
              case SizeChangeType::no_change:{ SetPrecisionAndExectutionMethod<false>(c2c_fwd_none); break; }
              case SizeChangeType::decrease: { SetPrecisionAndExectutionMethod<false>(c2c_fwd_decrease); break; }
              case SizeChangeType::increase: { SetPrecisionAndExectutionMethod<false>(c2c_fwd_increase); break; }
              default: { MyFFTDebugAssertTrue(false, "Invalid size change type"); }
            }
          }
          transform_stage_completed = TransformStageCompleted::fwd;
        }

        break;
    }
    case 2: {
      switch (fwd_size_change_type)
      {
        case no_change: {
          // FIXME there is some redundancy in specifying _decomposed and use_thread_method
          // Note: the only time the non-transposed method should be used is for 1d data.
          if (use_thread_method)
          {
            SetPrecisionAndExectutionMethod(r2c_decomposed_transposed, do_forward_transform);
            transform_stage_completed = TransformStageCompleted::fwd; // technically not complete, needed for copy on validation of partial fft.
            SetPrecisionAndExectutionMethod(c2c_decomposed, do_forward_transform);
          }
          else
          {
            SetPrecisionAndExectutionMethod(r2c_none_XY);
            transform_stage_completed = TransformStageCompleted::fwd; // technically not complete, needed for copy on validation of partial fft.
            SetPrecisionAndExectutionMethod(c2c_fwd_none);
          }
          break;
        }
        case increase: {
          SetPrecisionAndExectutionMethod(r2c_increase);
          transform_stage_completed = TransformStageCompleted::fwd; // technically not complete, needed for copy on validation of partial fft.
          SetPrecisionAndExectutionMethod(c2c_fwd_increase);   

          break;
        }
        case decrease: {

          SetPrecisionAndExectutionMethod(r2c_decrease);

          transform_stage_completed = TransformStageCompleted::fwd; // technically not complete, needed for copy on validation of partial fft.
          SetPrecisionAndExectutionMethod(c2c_fwd_decrease); 
 
          break;
        }
      }
      break; // case 2
    }
    case 3: {
      switch (fwd_size_change_type)
      {
        case no_change: {
          SetPrecisionAndExectutionMethod(r2c_none_XZ);
          transform_stage_completed = TransformStageCompleted::fwd; // technically not complete, needed for copy on validation of partial fft.
          SetPrecisionAndExectutionMethod(c2c_fwd_none_Z);
          SetPrecisionAndExectutionMethod(c2c_fwd_none);
          break;
        }
        case increase: {
          SetPrecisionAndExectutionMethod(r2c_increase);
          transform_stage_completed = TransformStageCompleted::fwd; // technically not complete, needed for copy on validation of partial fft.
          SetPrecisionAndExectutionMethod(c2c_fwd_increase_Z);   
          break;
        }
        case decrease: {
          // Not yet supported
          MyFFTRunTimeAssertTrue(false, "3D FFT fwd no change not yet supported");
          break;
        }
      } 
    }
  }


}

template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::InvFFT(bool transpose_output)
{
  SetDimensions(InvTransform);
  constexpr const bool use_thread_method = false;
  bool do_forward_transform = false;

  switch (transform_dimension)
  {
    case 1: {

              // FIXME there is some redundancy in specifying _decomposed and use_thread_method
        // Note: the only time the non-transposed method should be used is for 1d data.
        if constexpr (use_thread_method)
        {
          if (is_real_valued_input) SetPrecisionAndExectutionMethod(c2r_decomposed, do_forward_transform); //FFT_R2C_decomposed(transpose_output);
          else SetPrecisionAndExectutionMethod(c2c_decomposed, do_forward_transform);
          transform_stage_completed = TransformStageCompleted::inv;

        }
        else
        {
          if (is_real_valued_input) 
          {
            switch (inv_size_change_type)
            {
              case SizeChangeType::no_change:{ SetPrecisionAndExectutionMethod<false>(c2r_none_XY); break; }
              case SizeChangeType::decrease: { SetPrecisionAndExectutionMethod<false>(c2r_decrease); break; }
              case SizeChangeType::increase: { SetPrecisionAndExectutionMethod<false>(c2r_increase); break; }
              default: { MyFFTDebugAssertTrue(false, "Invalid size change type"); }
            }
          }
          else
          {
            switch (inv_size_change_type)
            {
              case SizeChangeType::no_change:{ SetPrecisionAndExectutionMethod<false>(c2c_inv_none); break; }
              case SizeChangeType::decrease: { SetPrecisionAndExectutionMethod<false>(c2c_inv_decrease); break; }
              case SizeChangeType::increase: { SetPrecisionAndExectutionMethod<false>(c2c_inv_increase); break; }
              default: { MyFFTDebugAssertTrue(false, "Invalid size change type"); }
            }
          }
          transform_stage_completed = TransformStageCompleted::inv;
        }

        break;
    }
    case 2: {
      switch (inv_size_change_type)
      {
        case no_change: {
          // FIXME there is some redundancy in specifying _decomposed and use_thread_method
          // Note: the only time the non-transposed method should be used is for 1d data.
          if (use_thread_method)
          {
            SetPrecisionAndExectutionMethod(c2c_decomposed,            do_forward_transform);
            transform_stage_completed = TransformStageCompleted::inv; // technically not complete, needed for copy on validation of partial fft.
            SetPrecisionAndExectutionMethod(c2r_decomposed_transposed, do_forward_transform);

          }
          else
          {
            SetPrecisionAndExectutionMethod(c2c_inv_none);
            transform_stage_completed = TransformStageCompleted::inv; // technically not complete, needed for copy on validation of partial fft.
            SetPrecisionAndExectutionMethod(c2r_none_XY);

          }          
          break;
        }
        case increase: {
          SetPrecisionAndExectutionMethod(c2c_inv_increase);
          transform_stage_completed = TransformStageCompleted::inv; // technically not complete, needed for copy on validation of partial fft.
          SetPrecisionAndExectutionMethod(c2r_increase); 
          
          // FFT_C2C(false);
          // FFT_C2R_Transposed();
          break;
        }
        case decrease: {
          SetPrecisionAndExectutionMethod(c2c_inv_decrease);
          transform_stage_completed = TransformStageCompleted::inv; // technically not complete, needed for copy on validation of partial fft.
          SetPrecisionAndExectutionMethod(c2r_decrease); 
          break;
        }
        default: {
          MyFFTDebugAssertTrue(false, "Invalid size change type");
        }
      } // switch on inv size change type
      break; // case 2
    }
    case 3: {
      switch (inv_size_change_type) 
      {
        case no_change: {
          SetPrecisionAndExectutionMethod(c2c_inv_none);
          transform_stage_completed = TransformStageCompleted::inv; // technically not complete, needed for copy on validation of partial fft.
          SetPrecisionAndExectutionMethod(c2c_inv_none_Z);
          SetPrecisionAndExectutionMethod(c2r_none);
          break;
        }
        case increase: {
          SetPrecisionAndExectutionMethod(r2c_increase);
          transform_stage_completed = TransformStageCompleted::fwd; // technically not complete, needed for copy on validation of partial fft.
          SetPrecisionAndExectutionMethod(c2c_fwd_increase_Z);   
          break;
        }
        case decrease: {
          // Not yet supported
          MyFFTRunTimeAssertTrue(false, "3D FFT inv no decrease not yet supported");
          break;
        }
        default: {
          MyFFTDebugAssertTrue(false, "Invalid dimension");
          break;
        }
      }
    }
  }


}

template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::CrossCorrelate(float2* image_to_search, bool swap_real_space_quadrants)
{

  // Set the member pointer to the passed pointer
  d_ptr.image_to_search = image_to_search;
 

  switch (transform_dimension)
  {
   

    case 1: {
     

      MyFFTRunTimeAssertTrue(false, "1D FFT Cross correlation not yet supported");
      break;
    }
    case 2: {
     
      switch (fwd_size_change_type)
      {
          case no_change: {
           

            SetDimensions(FwdTransform);
            SetPrecisionAndExectutionMethod(r2c_none_XY,   true);
            switch (inv_size_change_type)
            {
              case no_change: {
                MyFFTRunTimeAssertTrue(false, "2D FFT Cross correlation no change/ nochange not yet supported");
                break;
              }
              case increase: {
                MyFFTRunTimeAssertTrue(false, "2D FFT Cross correlation no change/increase not yet supported");
                break;
              }
              case decrease: {
                           
                SetPrecisionAndExectutionMethod(xcorr_fwd_none_inv_decrease, true);
                SetPrecisionAndExectutionMethod(c2r_decrease,   false);
                break;
              }
              default: {
                MyFFTDebugAssertTrue(false, "Invalid size change type");
                break;
              }
            } // switch on inv size change type
            break;
          } // case fwd no change
          case increase: {
           
            SetDimensions(FwdTransform);
            SetPrecisionAndExectutionMethod(r2c_increase,   true);
            switch (inv_size_change_type)
            {
              case no_change: {
               

                SetPrecisionAndExectutionMethod(xcorr_fwd_increase_inv_none, true);
                SetPrecisionAndExectutionMethod(c2r_none_XY,   false);
                transform_stage_completed = TransformStageCompleted::inv;
                break;
              }
              case increase: {
                // I don't see where increase increase makes any sense
                // FIXME add a check on this in the validation step.
                MyFFTRunTimeAssertTrue(false, "2D FFT Cross correlation with fwd and inv size increase is not supported");
                break;
              }
              case decrease: {
                // with FwdTransform set, call c2c
                // Set InvTransform
                // Call new kernel that handles the conj mul inv c2c trimmed, and inv c2r in one go.
                MyFFTRunTimeAssertTrue(false, "2D FFT Cross correlation with fwd increase and inv size decrease is a work in progress");

                break;
              }
              default: {
                MyFFTRunTimeAssertTrue(false, "Invalid size change type");
              }
            } // switch on inv size change type
      
            // FFT_R2C_WithPadding();   
            // FFT_C2C_INCREASE_ConjMul_C2C(image_to_search, swap_real_space_quadrants);  
            // FFT_C2R_Transposed();
            break;
          }
          case decrease: {

            SetDimensions(FwdTransform);
            SetPrecisionAndExectutionMethod(r2c_decrease,   true);
            switch (inv_size_change_type)
            {
              case no_change: {
                SetPrecisionAndExectutionMethod(xcorr_fwd_increase_inv_none, true);
                SetPrecisionAndExectutionMethod(c2r_none_XY,   false); // TODO the output could be smaller
                transform_stage_completed = TransformStageCompleted::inv;
                break;
              }
              case increase: {

                MyFFTRunTimeAssertTrue(false, "2D FFT Cross correlation with fwd and inv size increase is not supported");
                break;
              }
              case decrease: {

                MyFFTRunTimeAssertTrue(false, "2D FFT Cross correlation with fwd decrease and inv size decrease is a work in progress");
                break;
              }
              default: {
                MyFFTRunTimeAssertTrue(false, "Invalid inv size change type");
              }
            break;
          }
          break;
        } // case decrease
        default: {
          MyFFTRunTimeAssertTrue(false, "Invalid fwd size change type");
        }
      } // switch on fwd size change type

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

  // Set the member pointer to the passed pointer
  d_ptr.image_to_search = image_to_search;
  switch (transform_dimension)
  {
    case 1: {
      MyFFTRunTimeAssertTrue(false, "1D FFT Cross correlation not yet supported");
      break;
    }
    case 2: {
      switch (fwd_size_change_type)
      {
        case no_change: {
          MyFFTRunTimeAssertTrue(false, "2D FFT Cross correlation without size change not yet supported");
          break;
        }
        case increase: {
          SetDimensions(FwdTransform);
          SetPrecisionAndExectutionMethod(r2c_increase,   true);

          switch (inv_size_change_type)
          {
            case no_change: {
              SetPrecisionAndExectutionMethod(xcorr_fwd_increase_inv_none, true);
              SetPrecisionAndExectutionMethod(c2r_none_XY,   false); // TODO the output could be smaller
              transform_stage_completed = TransformStageCompleted::inv;

              break;
            }
            case increase: {
              // I don't see where increase increase makes any sense
              // FIXME add a check on this in the validation step.
              MyFFTRunTimeAssertTrue(false, "2D FFT Cross correlation with fwd and inv size increase is not supported");
              break;
            }
            case decrease: {
              // with FwdTransform set, call c2c
              // Set InvTransform
              // Call new kernel that handles the conj mul inv c2c trimmed, and inv c2r in one go.
              MyFFTRunTimeAssertTrue(false, "2D FFT Cross correlation with fwd increase and inv size decrease is a work in progress");

              break;
            }
          } // inv size change type
        } // case fwd_size_change = increase
        case decrease: {
          MyFFTRunTimeAssertTrue(false, "2D FFT Cross correlation without size decrease not yet supported");
          break;
        }
      } // fwd size change type
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
void FourierTransformer<ComputeType, InputType, OutputType>::ValidateDimensions()
{
  // TODO - runtime asserts would be better as these are breaking errors that are under user control.
  // check to see if there is any measurable penalty for this.

  MyFFTDebugAssertTrue(is_set_input_params, "Input parameters not set");
  MyFFTDebugAssertTrue(is_set_output_params, "Output parameters not set");
  MyFFTDebugAssertTrue(is_set_input_pointer, "The input data pointer is not set");

  MyFFTRunTimeAssertTrue( fwd_dims_out.x == inv_dims_in.x &&
                          fwd_dims_out.y == inv_dims_in.y &&
                          fwd_dims_out.z == inv_dims_in.z, "Error in validating the dimension: Currently all fwd out should match inv in.");

  // Validate the forward transform
  if (fwd_dims_out.x > fwd_dims_in.x || fwd_dims_out.y > fwd_dims_in.y || fwd_dims_out.z > fwd_dims_in.z)
  {
    // For now we must pad in all dimensions, this is not needed and should be lifted. FIXME
    MyFFTDebugAssertTrue(fwd_dims_out.x >= fwd_dims_in.x, "If padding, all dimensions must be >=, x out < x in");
    MyFFTDebugAssertTrue(fwd_dims_out.y >= fwd_dims_in.y, "If padding, all dimensions must be >=, y out < y in");
    MyFFTDebugAssertTrue(fwd_dims_out.z >= fwd_dims_in.z, "If padding, all dimensions must be >=, z out < z in");

    fwd_size_change_type = increase;
  }
  else if (fwd_dims_out.x < fwd_dims_in.x || fwd_dims_out.y < fwd_dims_in.y || fwd_dims_out.z < fwd_dims_in.z)
  {
    // For now we must pad in all dimensions, this is not needed and should be lifted. FIXME
    MyFFTDebugAssertTrue(fwd_dims_out.x <= fwd_dims_in.x, "If padding, all dimensions must be <=, x out > x in");
    MyFFTDebugAssertTrue(fwd_dims_out.y <= fwd_dims_in.y, "If padding, all dimensions must be <=, y out > y in");
    MyFFTDebugAssertTrue(fwd_dims_out.z <= fwd_dims_in.z, "If padding, all dimensions must be <=, z out > z in");

    fwd_size_change_type = decrease;
  }
  else if (fwd_dims_out.x == fwd_dims_in.x && fwd_dims_out.y == fwd_dims_in.y && fwd_dims_out.z == fwd_dims_in.z)
  {
    fwd_size_change_type = no_change;
  }
  else
  {
    // TODO: if this is relaxed, the dimensionality check below will be invalid.
    MyFFTRunTimeAssertTrue( false, "Error in validating fwd plan: Currently all dimensions must either increase, decrease or stay the same.");
  }

  // Validate the inverse transform
  if (inv_dims_out.x > inv_dims_in.x || inv_dims_out.y > inv_dims_in.y || inv_dims_out.z > inv_dims_in.z)
  {
    // For now we must pad in all dimensions, this is not needed and should be lifted. FIXME
    MyFFTDebugAssertTrue(inv_dims_out.x >= inv_dims_in.x, "If padding, all dimensions must be >=, x out < x in");
    MyFFTDebugAssertTrue(inv_dims_out.y >= inv_dims_in.y, "If padding, all dimensions must be >=, y out < y in");
    MyFFTDebugAssertTrue(inv_dims_out.z >= inv_dims_in.z, "If padding, all dimensions must be >=, z out < z in");

    inv_size_change_type = increase;
  }
  else if (inv_dims_out.x < inv_dims_in.x || inv_dims_out.y < inv_dims_in.y || inv_dims_out.z < inv_dims_in.z)
  {
    inv_size_change_type = decrease;
  }
  else if (inv_dims_out.x == inv_dims_in.x && inv_dims_out.y == inv_dims_in.y && inv_dims_out.z == inv_dims_in.z)
  {
    inv_size_change_type = no_change;
  }
  else
  {
    // TODO: if this is relaxed, the dimensionality check below will be invalid.
    MyFFTRunTimeAssertTrue( false, "Error in validating inv plan: Currently all dimensions must either increase, decrease or stay the same.");
  }

  // check for dimensionality
  // Note: this is predicated on the else clause ensuring all dimensions behave the same way w.r.t. size change.
  if (fwd_dims_in.z == 1 && fwd_dims_out.z == 1)
  {
    MyFFTRunTimeAssertTrue(inv_dims_in.z == 1 && inv_dims_out.z == 1, "Fwd/Inv dimensionality may not change from 1d,2d,3d (z dimension)");
    if (fwd_dims_in.y == 1 && fwd_dims_out.y == 1) 
    {
      MyFFTRunTimeAssertTrue(inv_dims_in.y == 1 && inv_dims_out.y == 1, "Fwd/Inv dimensionality may not change from 1d,2d,3d (y dimension)");
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
template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::SetDimensions(DimensionCheckType check_op_type)
{
  // This should be run inside any public method call to ensure things ar properly setup.
  if ( ! is_size_validated ) ValidateDimensions();

  switch (check_op_type)
  {
    case CopyFromHost: {
      MyFFTDebugAssertTrue(transform_stage_completed == none, "When copying from host, the transform stage should be none, something has gone wrong.");
      memory_size_to_copy = input_memory_allocated;
      break;
    }

    case CopyToHost: {
      // FIXME currently there is no check that the right amount of memory is allocated on the host side array.
      switch (transform_stage_completed)
      {
        case no_change: {
          memory_size_to_copy = input_memory_allocated;
          break;
        }
        case fwd: {
          memory_size_to_copy = fwd_output_memory_allocated; 
          break;
        }
        case inv: {
          memory_size_to_copy = inv_output_memory_allocated;
          break;
        }
      } // switch transform_stage_completed
      break;
    } // case CopToHose

    case FwdTransform: {
      MyFFTDebugAssertTrue(transform_stage_completed == none || transform_stage_completed == inv, "When doing a forward transform, the transform stage completed should be none, something has gone wrong.");
      break;
    }

    case InvTransform: {
      MyFFTDebugAssertTrue(transform_stage_completed == fwd, "When doing an inverse transform, the transform stage completed should be fwd, something has gone wrong.");
      break;
    }
  } // end switch on operation type  

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
 
  io_thread<FFT>::load_r2c(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)], thread_data, Q);

  // We then have Q FFTs of size size_of<FFT>::value (P in the paper)
  FFT().execute(thread_data);

  // Now we need to aggregate each of the Q transforms into each output block of size P  
  io_thread<FFT>::remap_decomposed_segments(thread_data, shared_mem, twiddle_in, Q, mem_offsets.physical_x_output);


  io_thread<FFT>::store_r2c(shared_mem, &output_values[Return1DFFTAddress(mem_offsets.physical_x_output)], Q, mem_offsets.physical_x_output);

 
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
 
  io_thread<FFT>::load_r2c(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)], thread_data, Q);

    // We then have Q FFTs of size size_of<FFT>::value (P in the paper)
    FFT().execute(thread_data);

  // Now we need to aggregate each of the Q transforms into each output block of size P
  io_thread<FFT>::remap_decomposed_segments(thread_data, shared_mem, twiddle_in, Q, mem_offsets.physical_x_output);


  io_thread<FFT>::store_r2c_transposed_xy(shared_mem, &output_values[ ReturnZplane(blockDim.y, mem_offsets.physical_x_output)], Q, gridDim.y, mem_offsets.physical_x_output);

 
} // end of thread_fft_kernel_R2C_transposed

// R2C

template<class FFT, class ComplexType, class ScalarType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_R2C_NONE_XY(const ScalarType* __restrict__ input_values, ComplexType*  __restrict__  output_values, Offsets mem_offsets, typename FFT::workspace_type workspace)
{
  // Initialize the shared memory, assuming everyting matches the input data X size in
  using complex_type = ComplexType;
  using scalar_type  = ScalarType;

  // The data store is non-coalesced, so don't aggregate the data in shared mem.
	extern __shared__  complex_type shared_mem[];


	// Memory used by FFT
  complex_type thread_data[FFT::storage_size];


  // No need to __syncthreads as each thread only accesses its own shared mem anyway
  // multiply Q*fwd_dims_out.w because x maps to y in the output transposed FFT
  io<FFT>::load_r2c(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)], thread_data);
  // io<FFT>::load_r2c(&input_values[blockIdx.y*mem_offsets.physical_x_input], thread_data);

  // In the first FFT the modifying twiddle factor is 1 so the data are real
  FFT().execute(thread_data, shared_mem, workspace);

  io<FFT>::store_r2c_transposed_xy(thread_data, &output_values[ ReturnZplane(gridDim.y, mem_offsets.physical_x_output)], gridDim.y);

 
} // end of block_fft_kernel_R2C_NONE_XY

template<class FFT, class ComplexType, class ScalarType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_R2C_NONE_XZ(const ScalarType*  __restrict__ input_values, ComplexType* __restrict__  output_values, Offsets mem_offsets, typename FFT::workspace_type workspace)
{
  // Initialize the shared memory, assuming everyting matches the input data X size in
  using complex_type = ComplexType;
  using scalar_type  = ScalarType;

  // The data store is non-coalesced, so don't aggregate the data in shared mem.
	extern __shared__  complex_type shared_mem[];

	// Memory used by FFT
  complex_type thread_data[FFT::storage_size];

  io<FFT>::load_r2c(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)], thread_data);

  FFT().execute(thread_data, shared_mem, workspace);

  // Transpose XZ, so the proper Z dimension now comes from X
  io<FFT>::store_r2c_transposed_xz(thread_data, output_values);

 
} // end of block_fft_kernel_R2C_NONE_XY

template<class FFT, class ComplexType, class ScalarType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_R2C_INCREASE_XY(const ScalarType* __restrict__  input_values, ComplexType* __restrict__  output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace)
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
  // multiply Q*fwd_dims_out.w because x maps to y in the output transposed FFT
  io<FFT>::load_r2c_shared(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)], shared_input, thread_data, twiddle_factor_args, twiddle_in, input_MAP, output_MAP, Q);

  // We unroll the first and last loops.
  // In the first FFT the modifying twiddle factor is 1 so the data are real
  FFT().execute(thread_data, shared_mem, workspace);  

  io<FFT>::store_r2c_transposed_xy(thread_data, &output_values[ ReturnZplane(blockDim.y, mem_offsets.physical_x_output)], output_MAP, gridDim.y);


  // For the other fragments we need the initial twiddle
  for (int sub_fft = 1; sub_fft < Q-1; sub_fft++)
  {

      io<FFT>::copy_from_shared(shared_input, thread_data, input_MAP);
      for (int i = 0; i < FFT::elements_per_thread; i++)
      {
        // Pre shift with twiddle
        SINCOS(twiddle_factor_args[i]*sub_fft,&twiddle.y,&twiddle.x);
        thread_data[i] *= twiddle;
        // increment the output mapping. 
        output_MAP[i]++;
      }
  
      FFT().execute(thread_data, shared_mem, workspace);


    io<FFT>::store_r2c_transposed_xy(thread_data, &output_values[ ReturnZplane(blockDim.y, mem_offsets.physical_x_output)], output_MAP, gridDim.y);
  }

  // For the last fragment we need to also do a bounds check.
  io<FFT>::copy_from_shared(shared_input, thread_data, input_MAP);
  for (int i = 0; i < FFT::elements_per_thread; i++)
  {
    // Pre shift with twiddle
    SINCOS(twiddle_factor_args[i]*(Q-1),&twiddle.y,&twiddle.x);
    thread_data[i] *= twiddle;
    // increment the output mapping. 
    output_MAP[i]++;
  }

  FFT().execute(thread_data, shared_mem, workspace);

  io<FFT>::store_r2c_transposed_xy(thread_data, &output_values[ ReturnZplane(blockDim.y, mem_offsets.physical_x_output)], output_MAP, gridDim.y, mem_offsets.physical_x_output);

} // end of block_fft_kernel_R2C_INCREASE_XY

// __launch_bounds__(FFT::max_threads_per_block)  we don't know this because it is threadDim.x * threadDim.z - this could be templated if it affects performance significantly
template<class FFT, class ComplexType, class ScalarType>
__global__
void block_fft_kernel_R2C_DECREASE_XY(const ScalarType* __restrict__  input_values, ComplexType* __restrict__  output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace)
{
  // Initialize the shared memory, assuming everyting matches the input data X size in
  using complex_type = ComplexType;
  using scalar_type  = ScalarType;

  // The shared memory is used for storage, shuffling and fft ops at different stages and includes room for bank padding.
	extern __shared__  complex_type shared_mem[];

  complex_type thread_data[FFT::storage_size];

  // Load in natural order
  io<FFT>::load_r2c_shared_and_pad(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)], shared_mem);

  // DIT shuffle, bank conflict free
  io<FFT>::copy_from_shared(shared_mem, thread_data, Q);

  // The FFT operator has no idea we are using threadIdx.z to get multiple sub transforms, so we need to 
  // segment the shared memory it accesses to avoid conflicts.
  constexpr const unsigned int fft_shared_mem_num_elements = FFT::shared_memory_size / sizeof(complex_type);
  FFT().execute(thread_data, &shared_mem[fft_shared_mem_num_elements * threadIdx.z], workspace);
  __syncthreads();

  // Full twiddle multiply and store in natural order in shared memory
  io<FFT>::reduce_block_fft(thread_data, shared_mem, twiddle_in, Q);
 
  // Reduce from shared memory into registers, ending up with only P valid outputs.
  io<FFT>::store_r2c_reduced(thread_data, &output_values[ mem_offsets.physical_x_output * threadIdx.z ], gridDim.y, mem_offsets.physical_x_output);

} // end of block_fft_kernel_R2C_DECREASE_XY

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
 
  io_thread<FFT>::load_c2c(&input_values[Return1DFFTAddress(size_of<FFT>::value)*Q], thread_data, Q);

  // We then have Q FFTs of size size_of<FFT>::value (P in the paper)
  FFT().execute(thread_data);

  // Now we need to aggregate each of the Q transforms into each output block of size P
  io_thread<FFT>::remap_decomposed_segments(thread_data, shared_mem, twiddle_in, Q, size_of<FFT>::value*Q);
  
  #if DEBUG_FFT_STAGE > 3
    io_thread<invFFT>::load_shared_and_conj_multiply(&image_to_search[Return1DFFTAddress(size_of<FFT>::value*Q)], shared_mem, thread_data, Q);
  #endif

  #if DEBUG_FFT_STAGE > 4
	  invFFT().execute(thread_data);
    // Now we need to aggregate each of the Q transforms into each output block of size P
    io_thread<invFFT>::remap_decomposed_segments(thread_data, shared_mem, -twiddle_in, Q,size_of<FFT>::value*Q);
  #endif

  io_thread<invFFT>::store_c2c(shared_mem, &output_values[Return1DFFTAddress(size_of<FFT>::value*Q)], Q);

}

// C2C with conj multiplication

template<class FFT, class invFFT, class ComplexType>
__launch_bounds__(invFFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_FWD_INCREASE_INV_NONE_ConjMul(const ComplexType* __restrict__ image_to_search, const ComplexType*  __restrict__ input_values, ComplexType*  __restrict__ output_values, 
                                                        Offsets mem_offsets, int apparent_Q, typename FFT::workspace_type workspace_fwd, typename invFFT::workspace_type workspace_inv)
{

  //	// Initialize the shared memory, assuming everyting matches the input data X size in
  using complex_type = ComplexType;

	// __shared__ complex_type shared_mem[invFFT::shared_memory_size/sizeof(complex_type)]; // Storage for the input data that is re-used each blcok
	extern __shared__ complex_type shared_mem[]; // Storage for the input data that is re-used each blcok

  complex_type thread_data[FFT::storage_size];

  // For simplicity, we explicitly zeropad the input data to the size of the FFT.
  // It may be worth trying to use threadIdx.z as in the DECREASE methods.
  // Until then, this 
  io<FFT>::load(&input_values[Return1DFFTAddress(size_of<FFT>::value / apparent_Q)], thread_data, size_of<FFT>::value / apparent_Q);

  // In the first FFT the modifying twiddle factor is 1 so the data are reeal
  FFT().execute(thread_data, shared_mem, workspace_fwd);

  #if DEBUG_FFT_STAGE > 3
    //  * apparent_Q
    io<invFFT>::load_shared_and_conj_multiply(&image_to_search[Return1DFFTAddress(size_of<FFT>::value)], thread_data);
  #endif

  #if DEBUG_FFT_STAGE > 4
    invFFT().execute(thread_data, shared_mem, workspace_inv);
  #endif

  //  * apparent_Q
  io<invFFT>::store(thread_data, &output_values[Return1DFFTAddress(size_of<FFT>::value)]);


} // end of block_fft_kernel_C2C_FWD_INCREASE_INV_NONE_ConjMul

template<class FFT, class invFFT, class ComplexType>
__launch_bounds__(invFFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_FWD_INCREASE_INV_NONE_ConjMul_SwapRealSpaceQuadrants(const ComplexType* __restrict__ image_to_search, const ComplexType*  __restrict__ input_values, ComplexType*  __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace_fwd, typename invFFT::workspace_type workspace_inv)
{

  //	// Initialize the shared memory, assuming everyting matches the input data X size in
  using complex_type = ComplexType;

	// __shared__ complex_type shared_mem[invFFT::shared_memory_size/sizeof(complex_type)]; // Storage for the input data that is re-used each blcok
	extern __shared__ complex_type shared_mem[]; // Storage for the input data that is re-used each blcok

  complex_type thread_data[FFT::storage_size];


  io<FFT>::load(&input_values[Return1DFFTAddress(size_of<FFT>::value)], thread_data, size_of<FFT>::value);

  // In the first FFT the modifying twiddle factor is 1 so the data are reeal
  FFT().execute(thread_data, shared_mem, workspace_fwd);

  #if DEBUG_FFT_STAGE > 3
    // Swap real space quadrants using a phase shift by N/2 pixels 
    const unsigned int  stride = io<invFFT>::stride_size();
    int logical_y;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
      logical_y = threadIdx.x+ i*stride;
      // FIXME, not sure the physical_x_output is updated to replace the previous terms appropriately. This is supposed to be setting the conjugate terms.
      if ( logical_y >= mem_offsets.physical_x_output) logical_y -= mem_offsets.physical_x_output;
      if ( (int(blockIdx.y) + logical_y) % 2 != 0) thread_data[i] *= -1.f; // FIXME TYPE
    }

    io<invFFT>::load_shared_and_conj_multiply(&image_to_search[Return1DFFTAddress(size_of<FFT>::value * Q)], thread_data);
  #endif


  #if DEBUG_FFT_STAGE > 4
    invFFT().execute(thread_data, shared_mem, workspace_inv);
  #endif

  io<invFFT>::store(thread_data, &output_values[Return1DFFTAddress(size_of<FFT>::value * Q)]);



} // 

template<class FFT, class invFFT, class ComplexType>
__global__
void _INV_DECREASE_ConjMul( const ComplexType* __restrict__ image_to_search, const ComplexType* __restrict__  input_values, ComplexType* __restrict__  output_values, 
                                                Offsets mem_offsets, float twiddle_in, int apparent_Q, typename FFT::workspace_type workspace_fwd, typename invFFT::workspace_type workspace_inv)
{
  
    using complex_type = ComplexType;

    extern __shared__  complex_type shared_mem[]; 
  
    complex_type thread_data[FFT::storage_size];
  
    // Load in natural order
    io<FFT>::load(&input_values[Return1DFFTAddress(size_of<FFT>::value)], thread_data);

    // io<FFT>::load_c2c_shared_and_pad(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)], shared_mem);
  
    // // DIT shuffle, bank conflict free
    // io<FFT>::copy_from_shared(shared_mem, thread_data, Q);
  
    // constexpr const unsigned int fft_shared_mem_num_elements = FFT::shared_memory_size / sizeof(complex_type);
    // FFT().execute(thread_data, &shared_mem[fft_shared_mem_num_elements * threadIdx.z], workspace_fwd);
    // __syncthreads();
    FFT().execute(thread_data, shared_mem, workspace_fwd);


    // // Full twiddle multiply and store in natural order in shared memory
    // io<FFT>::reduce_block_fft(thread_data, shared_mem, twiddle_in, Q);

    #if DEBUG_FFT_STAGE > 3
      // Load in imageFFT to search
      io<invFFT>::load_shared_and_conj_multiply(&image_to_search[Return1DFFTAddress(size_of<FFT>::value)], thread_data);
    #endif

    #if DEBUG_FFT_STAGE > 4
      // Run the inverse FFT
      // invFFT().execute(thread_data, &shared_mem[fft_shared_mem_num_elements * threadIdx.z], workspace_inv);
      invFFT().execute(thread_data, shared_mem, workspace_inv);

    #endif

    // // The reduced store considers threadIdx.z to ignore extra threads
    // io<invFFT>::store_c2c_reduced(thread_data, &output_values[blockIdx.y * gridDim.y]);
    #if DEBUG_FFT_STAGE < 5
      // There is no size reduction for this debug stage, so we need to use the pixel_pitch of the input array.
      io<invFFT>::store(thread_data, &output_values[Return1DFFTAddress(size_of<FFT>::value)]);
    #else
      // In the current simplified version of the kernel, I am not using any transform decomposition (this is because of the difficulties with resrved threadIdx.x/y in the cufftdx lib)
      // So the full thing is calculated and only truncated on output.
      io<invFFT>::store(thread_data, &output_values[Return1DFFTAddress(size_of<FFT>::value / apparent_Q)], size_of<FFT>::value / apparent_Q);
    #endif


} // end of _INV_DECREASE_ConjMul

// C2C

template<class FFT, class ComplexType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_NONE(const ComplexType*  __restrict__  input_values, ComplexType*  __restrict__  output_values, Offsets mem_offsets, typename FFT::workspace_type workspace)
{

 //	// Initialize the shared memory, assuming everyting matches the input data X size in
  using complex_type = ComplexType;

	extern __shared__  complex_type shared_mem[]; // Storage for the input data that is re-used each blcok


	// Memory used by FFT
  complex_type thread_data[FFT::storage_size];

  // No need to __syncthreads as each thread only accesses its own shared mem anyway
  io<FFT>::load(&input_values[Return1DFFTAddress(size_of<FFT>::value)],  thread_data);

  // Since the memory ops are super straightforward this is an okay compromise.
	FFT().execute(thread_data, shared_mem, workspace);

	io<FFT>::store(thread_data ,&output_values[Return1DFFTAddress(size_of<FFT>::value)]);


} // end of block_fft_kernel_C2C_NONE

// C2C decomposed

// __launch_bounds__(FFT::max_threads_per_block)  we don't know this because it is threadDim.x * threadDim.z - this could be templated if it affects performance significantly
template<class FFT, class ComplexType>
__global__
void block_fft_kernel_C2C_DECREASE(const ComplexType* __restrict__  input_values, ComplexType*  __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace)
{
  //	// Initialize the shared memory, assuming everyting matches the input data X size in
  using complex_type = ComplexType;

  extern __shared__  complex_type shared_mem[]; 

  complex_type thread_data[FFT::storage_size];

  // Load in natural order
  io<FFT>::load_c2c_shared_and_pad(&input_values[Return1DFFTAddress(size_of<FFT>::value * Q)], shared_mem);

  // DIT shuffle, bank conflict free
  io<FFT>::copy_from_shared(shared_mem, thread_data, Q);


  constexpr const unsigned int fft_shared_mem_num_elements = FFT::shared_memory_size / sizeof(complex_type);
  FFT().execute(thread_data, &shared_mem[fft_shared_mem_num_elements * threadIdx.z], workspace);
  __syncthreads();

  // Full twiddle multiply and store in natural order in shared memory
  io<FFT>::reduce_block_fft(thread_data, shared_mem, twiddle_in, Q);

  // Reduce from shared memory into registers, ending up with only P valid outputs.
  io<FFT>::store_c2c_reduced(thread_data, &output_values[Return1DFFTAddress(size_of<FFT>::value)]);


}

// __launch_bounds__(FFT::max_threads_per_block)  we don't know this because it is threadDim.x * threadDim.z - this could be templated if it affects performance significantly
template<class FFT, class ComplexType>
__global__
void block_fft_kernel_C2C_INCREASE(const ComplexType*  __restrict__ input_values, ComplexType*  __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace)
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
  io<FFT>::load_shared(&input_values[Return1DFFTAddress(size_of<FFT>::value)], shared_input_complex, thread_data, twiddle_factor_args, twiddle_in, input_MAP, output_MAP, Q);

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
			SINCOS(twiddle_factor_args[i]*sub_fft,&twiddle.y,&twiddle.x);
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
    io<FFT>::store_coalesced(shared_output, &output_values[Return1DFFTAddress(size_of<FFT>::value * Q)], sub_fft*mem_offsets.shared_input);
	}


} // end of block_fft_kernel_C2C_INCREASE

template<class FFT, class ComplexType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_INCREASE_SwapRealSpaceQuadrants(const ComplexType*  __restrict__  input_values, ComplexType*  __restrict__  output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace)
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
  io<FFT>::load_shared(&input_values[Return1DFFTAddress(size_of<FFT>::value)], shared_input_complex, thread_data, twiddle_factor_args, twiddle_in, input_MAP, output_MAP, Q);

	// In the first FFT the modifying twiddle factor is 1 so the data are reeal
	FFT().execute(thread_data, shared_mem, workspace);

	// FIXME I have not confirmed on switch to physical_x_output that this represents the index of the first negative frequency in Y as it should.
  io<FFT>::store_and_swap_quadrants(thread_data,shared_output,output_MAP,size_of<FFT>::value * Q);

    // For the other fragments we need the initial twiddle
	for (int sub_fft = 1; sub_fft < Q; sub_fft++)
	{

	  io<FFT>::copy_from_shared(shared_input_complex, thread_data, input_MAP);

		for (int i = 0; i < FFT::elements_per_thread; i++)
		{
			// Pre shift with twiddle
			SINCOS(twiddle_factor_args[i]*sub_fft,&twiddle.y,&twiddle.x);
			thread_data[i] *= twiddle;
		    // increment the output map. Note this only works for the leading non-zero case
			output_MAP[i]++;
		}

		FFT().execute(thread_data, shared_mem, workspace);
    io<FFT>::store_and_swap_quadrants(thread_data,shared_output,output_MAP,size_of<FFT>::value * Q);


	}

  // TODO confirm this is needed
	__syncthreads();

	// Now that the memory output can be coalesced send to global
  // FIXME is this actually coalced?
	for (int sub_fft = 0; sub_fft < Q; sub_fft++)
	{
    io<FFT>::store_coalesced(shared_output, &output_values[Return1DFFTAddress(size_of<FFT>::value * Q)], sub_fft*mem_offsets.shared_input);
	}


} // end of block_fft_kernel_C2C_INCREASE_SwapRealSpaceQuadrants

template<class FFT, class ComplexType>
__global__
void thread_fft_kernel_C2C_decomposed(const ComplexType* __restrict__  input_values, ComplexType* __restrict__  output_values, Offsets mem_offsets, float twiddle_in, int Q)
{


  using complex_type = ComplexType;
  // The data store is non-coalesced, so don't aggregate the data in shared mem.
	extern __shared__  complex_type shared_mem[];

	// Memory used by FFT - for Thread() type, FFT::storage_size == FFT::elements_per_thread == size_of<FFT>::value
  complex_type thread_data[FFT::storage_size];
 
  io_thread<FFT>::load_c2c(&input_values[Return1DFFTAddress(size_of<FFT>::value)], thread_data, Q);

  // We then have Q FFTs of size size_of<FFT>::value (P in the paper)
  FFT().execute(thread_data);

  // Now we need to aggregate each of the Q transforms into each output block of size P
  io_thread<FFT>::remap_decomposed_segments(thread_data, shared_mem, twiddle_in, Q, size_of<FFT>::value * Q);

  io_thread<FFT>::store_c2c(shared_mem, &output_values[Return1DFFTAddress(size_of<FFT>::value * Q)], Q);

}


template<class FFT, class ComplexType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_NONE_XY(const ComplexType* __restrict__  input_values, ComplexType* __restrict__  output_values, Offsets mem_offsets, typename FFT::workspace_type workspace)
{

  //	// Initialize the shared memory, assuming everyting matches the input data X size in
   using complex_type = ComplexType;
 
   extern __shared__  complex_type shared_mem[]; // Storage for the input data that is re-used each blcok
 
   // Memory used by FFT
   complex_type thread_data[FFT::storage_size];
 
   // No need to __syncthreads as each thread only accesses its own shared mem anyway
   io<FFT>::load_Z(&input_values[ Return1DFFTAddress_YX() ], thread_data, gridDim.y * gridDim.z );
 
   // Since the memory ops are super straightforward this is an okay compromise.
   FFT().execute(thread_data, shared_mem, workspace);
   
   io<FFT>::store_Z(thread_data, &output_values[ Return1DFFTAddress_YX_to_XY() ], gridDim.y * gridDim.z);
  
}

template<class FFT, class ComplexType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_NONE_XYZ(const ComplexType* __restrict__  input_values, ComplexType* __restrict__  output_values, Offsets mem_offsets, typename FFT::workspace_type workspace)
{

  //	// Initialize the shared memory, assuming everyting matches the input data X size in
   using complex_type = ComplexType;
 
   extern __shared__  complex_type shared_mem[]; // Storage for the input data that is re-used each blcok
 
   // Memory used by FFT
   complex_type thread_data[FFT::storage_size];
 
   // No need to __syncthreads as each thread only accesses its own shared mem anyway
   io<FFT>::load_Z(&input_values[ Return1DFFTColumn_XZ_transpose(size_of<FFT>::value) ], thread_data );
 
   // Since the memory ops are super straightforward this is an okay compromise.
   FFT().execute(thread_data, shared_mem, workspace);
 
   io<FFT>::store_Z(thread_data, &output_values[ Return1DFFTColumn_XZ_to_XY() ], gridDim.y * gridDim.z);
  
}

template<class FFT, class ComplexType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_INCREASE_Z(const ComplexType* __restrict__  input_values, ComplexType*  __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace)
{
  // //	// Initialize the shared memory, assuming everyting matches the input data X size in
  //  using complex_type = ComplexType;
 
  //  extern __shared__  complex_type shared_input_complex[]; // Storage for the input data that is re-used each blcok
  //  complex_type* shared_mem = (complex_type*)&shared_input_complex[size_of<FFT>::value];
 
 
  //  // Memory used by FFT
  //  complex_type twiddle;
  //  complex_type thread_data[FFT::storage_size];
 
  //  // To re-map the thread index to the data
  //  int input_MAP[FFT::storage_size];
  //  // To re-map the decomposed frequency to the full output frequency
  //  int output_MAP[FFT::storage_size];
  //  // For a given decomposed fragment
  //  float twiddle_factor_args[FFT::storage_size];
 
  //  // mem_offsets.pixel_pitch_output is the number of elements nw * ny;
  // //  io<FFT>::load_shared_Z(&input_values[blockIdx.y*mem_offsets.pixel_pitch_input], shared_input_complex, thread_data, twiddle_factor_args, twiddle_in, input_MAP, output_MAP, Q, mem_offsets.pixel_pitch_output);
 
  //  // In the first FFT the modifying twiddle factor is 1 so the data are reeal
  //  FFT().execute(thread_data, shared_mem, workspace);
 
  //  // 
  //  io<FFT>::store(thread_data, output_values, output_MAP);
 
  //    // For the other fragments we need the initial twiddle
  //  for (int sub_fft = 1; sub_fft < Q; sub_fft++)
  //  {
 
  //    io<FFT>::copy_from_shared(shared_input_complex, thread_data, input_MAP);
 
  //    for (int i = 0; i < FFT::elements_per_thread; i++)
  //    {
  //      // Pre shift with twiddle
  //      SINCOS(twiddle_factor_args[i]*sub_fft,&twiddle.y,&twiddle.x);
  //      thread_data[i] *= twiddle;
  //        // increment the output map. Note this only works for the leading non-zero case
  //      output_MAP[i]++;
  //    }
 
  //    FFT().execute(thread_data, shared_mem, workspace);
 
  //    io<FFT>::store(thread_data, output_values, output_MAP);
 
  //  }
 

  
} // end of block_fft_kernel_C2C_INCREASE_Z 


template<class FFT, class ComplexType, class ScalarType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2R_NONE(const ComplexType*  __restrict__ input_values, ScalarType*  __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace)
{

	using complex_type = ComplexType;
	using scalar_type  = ScalarType;

	extern __shared__  complex_type shared_mem[];


  complex_type thread_data[FFT::storage_size];

  io<FFT>::load_c2r(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)], thread_data);

  // For loop zero the twiddles don't need to be computed
  FFT().execute(thread_data, shared_mem, workspace);
  
  io<FFT>::store_c2r(thread_data, &output_values[Return1DFFTAddress(mem_offsets.physical_x_output)], size_of<FFT>::value);

}

template<class FFT, class ComplexType, class ScalarType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2R_NONE_XY(const ComplexType* __restrict__  input_values, ScalarType*  __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace)
{

	using complex_type = ComplexType;
	using scalar_type  = ScalarType;

	extern __shared__  complex_type shared_mem[];


  complex_type thread_data[FFT::storage_size];

  io<FFT>::load_c2r_transposed(&input_values[ReturnZplane(gridDim.y, mem_offsets.physical_x_input)], thread_data, gridDim.y);

  // For loop zero the twiddles don't need to be computed
  FFT().execute(thread_data, shared_mem, workspace);
  
  io<FFT>::store_c2r(thread_data, &output_values[Return1DFFTAddress(mem_offsets.physical_x_output)], size_of<FFT>::value);

} // end of block_fft_kernel_C2R_NONE_XY

template<class FFT, class ComplexType, class ScalarType>
__global__
void block_fft_kernel_C2R_DECREASE_XY(const ComplexType*  __restrict__ input_values, ScalarType*  __restrict__ output_values, Offsets mem_offsets, const float twiddle_in, const unsigned int Q, typename FFT::workspace_type workspace)
{

	using complex_type = ComplexType;
	using scalar_type  = ScalarType;

	extern __shared__  complex_type shared_mem[];

  complex_type thread_data[FFT::storage_size];


  io<FFT>::load_c2r_transposed(&input_values[ReturnZplane(gridDim.y, mem_offsets.physical_x_input)], thread_data, gridDim.y);
  
  // For loop zero the twiddles don't need to be computed
  FFT().execute(thread_data, shared_mem, workspace);

  io<FFT>::store_c2r(thread_data, &output_values[Return1DFFTAddress(mem_offsets.physical_x_output)], size_of<FFT>::value);

  // // Load transposed data into shared memory in natural order.
  // io<FFT>::load_c2r_shared_and_pad(&input_values[blockIdx.y], shared_mem, mem_offsets.physical_x_input);

  // // DIT shuffle, bank conflict free
  // io<FFT>::copy_from_shared(shared_mem, thread_data, Q);

  // constexpr const unsigned int fft_shared_mem_num_elements = FFT::shared_memory_size / sizeof(complex_type);
  // FFT().execute(thread_data, &shared_mem[fft_shared_mem_num_elements * threadIdx.z], workspace);
  // __syncthreads();

  // // Full twiddle multiply and store in natural order in shared memory
  // io<FFT>::reduce_block_fft(thread_data, shared_mem, twiddle_in, Q);

  // // Reduce from shared memory into registers, ending up with only P valid outputs.
  // io<FFT>::store_c2r_reduced(thread_data, &output_values[Return1DFFTAddress(mem_offsets.physical_x_output)]);


} // end of block_fft_kernel_C2R_DECREASE_XY

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
 

  io_thread<FFT>::load_c2r(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)], thread_data, Q, mem_offsets.physical_x_input);

  // We then have Q FFTs of size size_of<FFT>::value (P in the paper)
  FFT().execute(thread_data);

  // Now we need to aggregate each of the Q transforms into each output block of size P
  io_thread<FFT>::remap_decomposed_segments_c2r(thread_data, shared_mem_C2R_decomposed, twiddle_in, Q);

  io_thread<FFT>::store_c2r(shared_mem_C2R_decomposed, &output_values[Return1DFFTAddress(mem_offsets.physical_x_output)],Q);
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
 

  io_thread<FFT>::load_c2r_transposed(&input_values[ReturnZplane(blockDim.y, mem_offsets.physical_x_input)], thread_data, Q, gridDim.y, mem_offsets.physical_x_input);

  // We then have Q FFTs of size size_of<FFT>::value (P in the paper)
  FFT().execute(thread_data);

  // Now we need to aggregate each of the Q transforms into each output block of size P
  io_thread<FFT>::remap_decomposed_segments_c2r(thread_data, shared_mem_transposed, twiddle_in, Q);

  io_thread<FFT>::store_c2r(shared_mem_transposed, &output_values[Return1DFFTAddress(mem_offsets.physical_x_output)],Q);

}

// FIXME assumed FWD 
template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::ClipIntoTopLeft()
{
  // TODO add some checks and logic.

  // Assuming we are calling this from R2C_Transposed and that the launch bounds are not set.
  dim3 threadsPerBlock;
  dim3 gridDims;

  threadsPerBlock = dim3(512,1,1);
  gridDims = dim3( (fwd_dims_out.x + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

  const short4 area_to_clip_from = make_short4(fwd_dims_in.x, fwd_dims_in.y, fwd_dims_in.w*2, fwd_dims_out.w*2);

  precheck
  clip_into_top_left_kernel<float, float><< < gridDims, threadsPerBlock, 0, cudaStreamPerThread >> >
  (d_ptr.position_space, d_ptr.position_space, area_to_clip_from);
  postcheck
}

// FIXME assumed FWD 
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
  gridDims = dim3( (fwd_dims_out.x + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (fwd_dims_out.y + threadsPerBlock.y - 1) / threadsPerBlock.y, 
                   1);

  const short4 area_to_clip_from = make_short4(fwd_dims_in.x, fwd_dims_in.y, fwd_dims_in.w*2, fwd_dims_out.w*2);
  float wanted_padding_value = 0.f;
  
  precheck
  clip_into_real_kernel<float, float><< < gridDims, threadsPerBlock, 0, cudaStreamPerThread >> >
  (d_ptr.position_space, d_ptr.position_space, fwd_dims_in, fwd_dims_out,wanted_center, wanted_padding_value);
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
                               blockIdx.y*gridDim.y + threadIdx.y,
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
    using FFT = decltype( Block() + Precision<ComputeType>()  + FFTsPerBlock<1>() );
    SelectSizeAndType<FFT>(kernel_type, do_forward_transform);
  }
  

}

template <class ComputeType, class InputType, class OutputType>
template <class FFT_base>
void FourierTransformer<ComputeType, InputType, OutputType>::SelectSizeAndType(KernelType kernel_type, bool do_forward_transform)
{


  if constexpr (detail::is_operator<fft_operator::thread, FFT_base>::value)
  {
    GetTransformSize_thread(kernel_type, size_of<FFT_base>::value);
    switch (device_properties.device_arch)
    {
      case 700: { using FFT = decltype(FFT_base() + SM<700>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
      case 750: { using FFT = decltype(FFT_base() + SM<750>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
      case 800: { using FFT = decltype(FFT_base() + SM<800>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
    }
  }
  else
  {
    GetTransformSize(kernel_type);

    switch (transform_size.P)
    {
      case 16: {
        elements_per_thread_complex = elements_per_thread_16;
        switch (device_properties.device_arch)
        {
          case 700: { using FFT = decltype(FFT_base()  + Size<16>()  + SM<700>() + ElementsPerThread<elements_per_thread_16>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 750: { using FFT = decltype(FFT_base()  + Size<16>()  + SM<750>() + ElementsPerThread<elements_per_thread_16>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 800: { using FFT = decltype(FFT_base()  + Size<16>()  + SM<800>() + ElementsPerThread<elements_per_thread_16>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
        }
        break; }

      case 32: {
        elements_per_thread_complex = elements_per_thread_32;
        switch (device_properties.device_arch)
        {
          case 700: { using FFT = decltype(FFT_base()  + Size<32>()  + SM<700>() + ElementsPerThread<elements_per_thread_32>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 750: { using FFT = decltype(FFT_base()  + Size<32>()  + SM<750>() + ElementsPerThread<elements_per_thread_32>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 800: { using FFT = decltype(FFT_base()  + Size<32>()  + SM<800>() + ElementsPerThread<elements_per_thread_32>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
        }
        break; }

      case 64: {
        elements_per_thread_complex = elements_per_thread_64;
        switch (device_properties.device_arch)
        {
          case 700: { using FFT = decltype(FFT_base()  + Size<64>()  + SM<700>() + ElementsPerThread<elements_per_thread_64>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 750: { using FFT = decltype(FFT_base()  + Size<64>()  + SM<750>() + ElementsPerThread<elements_per_thread_64>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 800: { using FFT = decltype(FFT_base()  + Size<64>()  + SM<800>() + ElementsPerThread<elements_per_thread_64>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
        }
        break; }
  
      case 128: {
        elements_per_thread_complex = elements_per_thread_128;
        switch (device_properties.device_arch)
        {
          case 700: { using FFT = decltype(FFT_base()  + Size<128>()  + SM<700>() + ElementsPerThread<elements_per_thread_128>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 750: { using FFT = decltype(FFT_base()  + Size<128>()  + SM<750>() + ElementsPerThread<elements_per_thread_128>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 800: { using FFT = decltype(FFT_base()  + Size<128>()  + SM<800>() + ElementsPerThread<elements_per_thread_128>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
        }
        break; }
  
      case 256: {
        elements_per_thread_complex = elements_per_thread_256;
        switch (device_properties.device_arch)
        {
          case 700: { using FFT = decltype(FFT_base()  + Size<256>()  + SM<700>() + ElementsPerThread<elements_per_thread_256>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 750: { using FFT = decltype(FFT_base()  + Size<256>()  + SM<750>() + ElementsPerThread<elements_per_thread_256>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 800: { using FFT = decltype(FFT_base()  + Size<256>()  + SM<800>() + ElementsPerThread<elements_per_thread_256>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
        }
        break; } 
  
      case 512: {
        elements_per_thread_complex = elements_per_thread_512;
        switch (device_properties.device_arch)
        {
          case 700: { using FFT = decltype(FFT_base()  + Size<512>()  + SM<700>() + ElementsPerThread<elements_per_thread_512>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 750: { using FFT = decltype(FFT_base()  + Size<512>()  + SM<750>() + ElementsPerThread<elements_per_thread_512>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 800: { using FFT = decltype(FFT_base()  + Size<512>()  + SM<800>() + ElementsPerThread<elements_per_thread_512>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
        }
        break; } 
  
      // case 768: {
      //   switch (device_properties.device_arch)
      //   {
      //     case 700: { using FFT = decltype(FFT_base()  + Size<768>()  + SM<700>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
      //     case 750: { using FFT = decltype(FFT_base()  + Size<768>()  + SM<750>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
      //     case 800: { using FFT = decltype(FFT_base()  + Size<768>()  + SM<800>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
      //   }
      // break; } 
  
      case 1024: {
        elements_per_thread_complex = elements_per_thread_1024;
        switch (device_properties.device_arch)
        {
          case 700: { using FFT = decltype(FFT_base()  + Size<1024>()  + SM<700>() + ElementsPerThread<elements_per_thread_1024>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 750: { using FFT = decltype(FFT_base()  + Size<1024>()  + SM<750>() + ElementsPerThread<elements_per_thread_1024>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 800: { using FFT = decltype(FFT_base()  + Size<1024>()  + SM<800>() + ElementsPerThread<elements_per_thread_1024>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
        }
      break; } 
  
      // case 1536: {
      //   switch (device_properties.device_arch)
      //   {
      //     case 700: { using FFT = decltype(FFT_base()  + Size<1536>()  + SM<700>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
      //     // case 750: { using FFT = decltype(FFT_base()  + Size<1536>()  + SM<750>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
      //     case 800: { using FFT = decltype(FFT_base()  + Size<1536>()  + SM<800>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
      //   }
      // break; }    
  
      case 2048: {
        elements_per_thread_complex = elements_per_thread_2048;
        switch (device_properties.device_arch)
        {
          case 700: { using FFT = decltype(FFT_base()  + Size<2048>()  + SM<700>() + ElementsPerThread<elements_per_thread_2048>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 750: { using FFT = decltype(FFT_base()  + Size<2048>()  + SM<750>() + ElementsPerThread<elements_per_thread_2048>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 800: { using FFT = decltype(FFT_base()  + Size<2048>()  + SM<800>() + ElementsPerThread<elements_per_thread_2048>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
        }
        break; } 
  
  
      case 4096: {
        elements_per_thread_complex = elements_per_thread_4096;
        switch (device_properties.device_arch)
        {
          case 700: { using FFT = decltype(FFT_base()  + Size<4096>()  + SM<700>() + ElementsPerThread<elements_per_thread_4096>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          // case 750: { using FFT = decltype(FFT_base()  + Size<4096>()  + SM<750>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 800: { using FFT = decltype(FFT_base()  + Size<4096>()  + SM<800>() + ElementsPerThread<elements_per_thread_4096>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
        }
        break; }  
  
      case 8192: {
        elements_per_thread_complex = elements_per_thread_8192;
        switch (device_properties.device_arch)
        {
          case 700: { using FFT = decltype(FFT_base()  + Size<8192>()  + SM<700>() + ElementsPerThread<elements_per_thread_8192>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
          case 800: { using FFT = decltype(FFT_base()  + Size<8192>()  + SM<800>() + ElementsPerThread<elements_per_thread_8192>());  SetAndLaunchKernel<FFT>(kernel_type, do_forward_transform); break;}
        }
        break; } 

      default: {
        MyFFTRunTimeAssertTrue(false, "FFT size not supported");
      }
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
        CheckSharedMemory(shared_mem, device_properties);
        cudaErr(cudaFuncSetAttribute((void*)thread_fft_kernel_R2C_decomposed<FFT,complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem));        
        #if DEBUG_FFT_STAGE > 0
          precheck
          thread_fft_kernel_R2C_decomposed<FFT,complex_type,scalar_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
          (scalar_input, complex_output, LP.mem_offsets, LP.twiddle_in, LP.Q);
          postcheck
        #else
          is_in_buffer_memory = ! is_in_buffer_memory;
        #endif    
        break; 
      }

      case r2c_decomposed_transposed: {

        using FFT = decltype( FFT_base_arch() + Direction<fft_direction::forward>() + Type<fft_type::c2c>() ); 

        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, r2c_decomposed_transposed);
      
        int shared_mem = LP.mem_offsets.shared_output * sizeof(complex_type);
        CheckSharedMemory(shared_mem, device_properties);
        cudaErr(cudaFuncSetAttribute((void*)thread_fft_kernel_R2C_decomposed_transposed<FFT,complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem));        
        #if DEBUG_FFT_STAGE > 0
          precheck
          thread_fft_kernel_R2C_decomposed_transposed<FFT,complex_type,scalar_type><< <LP.gridDims,  LP.threadsPerBlock, shared_mem, cudaStreamPerThread>> >
          (scalar_input, complex_output, LP.mem_offsets, LP.twiddle_in, LP.Q);
          postcheck
        #else
          is_in_buffer_memory = ! is_in_buffer_memory;
        #endif

        break; 
      }
    case c2r_decomposed: {

      // Note that unlike the block C2R we require a C2C sub xform.
      using FFT = decltype(FFT_base_arch() + Direction<fft_direction::inverse>() + Type<fft_type::c2c>());
      // TODO add completeness check.

      LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2r_decomposed);
      int shared_memory = LP.mem_offsets.shared_output * sizeof(scalar_type);
      CheckSharedMemory(shared_memory, device_properties);
      cudaErr(cudaFuncSetAttribute((void*)thread_fft_kernel_C2R_decomposed<FFT,complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));        
      #if DEBUG_FFT_STAGE > 6
        precheck
        thread_fft_kernel_C2R_decomposed<FFT, complex_type, scalar_type><< <LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
        (complex_input, complex_output, LP.mem_offsets, LP.twiddle_in, LP.Q);
        postcheck
      #else
        is_in_buffer_memory = ! is_in_buffer_memory;
      #endif

      break; 
    }
    case c2r_decomposed_transposed: {  
      // Note that unlike the block C2R we require a C2C sub xform.
      using FFT = decltype(FFT_base_arch() + Direction<fft_direction::inverse>() + Type<fft_type::c2c>());

      LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2r_decomposed_transposed);
      int shared_memory = LP.mem_offsets.shared_output * sizeof(scalar_type);
      CheckSharedMemory(shared_memory, device_properties);
      cudaErr(cudaFuncSetAttribute((void*)thread_fft_kernel_C2R_decomposed_transposed<FFT,complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));        
      #if DEBUG_FFT_STAGE > 6
        precheck
        thread_fft_kernel_C2R_decomposed_transposed<FFT, complex_type, scalar_type><< <LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
        (complex_input, complex_output, LP.mem_offsets, LP.twiddle_in, LP.Q);
        postcheck
      #else
        is_in_buffer_memory = ! is_in_buffer_memory;
      #endif
    
      break; 
    } 
    case xcorr_decomposed: {

      using    FFT = decltype( FFT_base_arch() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() );  
      using invFFT = decltype( FFT_base_arch() + Type<fft_type::c2c>() + Direction<fft_direction::inverse>() ); 

      LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, xcorr_decomposed);

      int shared_memory = LP.mem_offsets.shared_output * sizeof(complex_type);
      CheckSharedMemory(shared_memory, device_properties);

      // FIXME
      bool swap_real_space_quadrants = false;

      if (swap_real_space_quadrants)
      {
        MyFFTRunTimeAssertTrue(false, "decomposed xcorr with swap real space quadrants is not implemented.");
        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_FWD_INCREASE_INV_NONE_ConjMul_SwapRealSpaceQuadrants<FFT,complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));        

        // precheck
        // block_fft_kernel_C2C_FWD_INCREASE_INV_NONE_ConjMul_SwapRealSpaceQuadrants<FFT,invFFT, complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
        // ( (complex_type*) image_to_search, (complex_type*)  d_ptr.momentum_space_buffer,  (complex_type*) d_ptr.momentum_space, LP.mem_offsets, LP.twiddle_in,LP.Q, workspace_fwd, workspace_inv);
        // postcheck
      }
      else
      {
        cudaErr(cudaFuncSetAttribute((void*)thread_fft_kernel_C2C_decomposed_ConjMul<FFT,complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));        

        #if DEBUG_FFT_STAGE > 2
          // the image_to_search pointer is set during call to CrossCorrelate,
          precheck
          thread_fft_kernel_C2C_decomposed_ConjMul<FFT, invFFT, complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
          ( (complex_type *)d_ptr.image_to_search, complex_input, complex_output, LP.mem_offsets, LP.twiddle_in,LP.Q);
          postcheck
        #else
          is_in_buffer_memory = ! is_in_buffer_memory;
        #endif
      }
      
      break; 
    }
    case c2c_decomposed: {
      using FFT_nodir = decltype(FFT_base_arch() + Type<fft_type::c2c>() );

      LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2c_decomposed, do_forward_transform);

      if (do_forward_transform)
      {
        using FFT = decltype( FFT_nodir() + Direction<fft_direction::forward>() );
        int shared_memory = LP.mem_offsets.shared_output * sizeof(complex_type);
        CheckSharedMemory(shared_memory, device_properties);
        cudaErr(cudaFuncSetAttribute((void*)thread_fft_kernel_C2C_decomposed<FFT,complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));        
        #if DEBUG_FFT_STAGE > 2
          precheck
          thread_fft_kernel_C2C_decomposed<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
          ( complex_input, complex_output, LP.mem_offsets, LP.twiddle_in, LP.Q);
          postcheck
        #else
          is_in_buffer_memory = ! is_in_buffer_memory;
        #endif
      }
      else
      {
    
        using FFT = decltype( FFT_nodir() + Direction<fft_direction::inverse>() );
        int shared_memory = LP.mem_offsets.shared_output * sizeof(complex_type);
        CheckSharedMemory(shared_memory, device_properties);
        cudaErr(cudaFuncSetAttribute((void*)thread_fft_kernel_C2C_decomposed<FFT,complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));        
        #if DEBUG_FFT_STAGE > 4
          precheck
          thread_fft_kernel_C2C_decomposed<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
          ( complex_input, complex_output, LP.mem_offsets, LP.twiddle_in, LP.Q);
          postcheck
        #else
          is_in_buffer_memory = ! is_in_buffer_memory;  
        #endif
      }
    }
    
    break; 
    }    
  }
  else // Block
  {
    switch (kernel_type)
    {
      case r2c_none_XY: {

        using FFT = decltype( FFT_base_arch() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() );  
        cudaError_t error_code = cudaSuccess;
        auto workspace = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;
        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, r2c_none_XY);

        int shared_memory = FFT::shared_memory_size;
        CheckSharedMemory(shared_memory, device_properties);
        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_R2C_NONE_XY<FFT,complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));        
   
        // cudaErr(cudaSetDevice(0));
        //  cudaErr(cudaFuncSetCacheConfig( (void*)block_fft_kernel_R2C_NONE_XY<FFT,complex_type,scalar_type>,cudaFuncCachePreferShared ));
        //  cudaFuncSetSharedMemConfig ( (void*)block_fft_kernel_R2C_NONE_XY<FFT,complex_type,scalar_type>, cudaSharedMemBankSizeEightByte );

        #if DEBUG_FFT_STAGE > 0
          precheck
          block_fft_kernel_R2C_NONE_XY<FFT,complex_type,scalar_type><< <LP.gridDims,  LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
          (scalar_input, complex_output, LP.mem_offsets, workspace);
          postcheck 
        #else
          is_in_buffer_memory = ! is_in_buffer_memory;
        #endif


           
        break;
      }

      case r2c_none_XZ: {

        using FFT = decltype( FFT_base_arch() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() );  
        cudaError_t error_code = cudaSuccess;
        auto workspace = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;
        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, r2c_none_XZ);

        int shared_memory = FFT::shared_memory_size;
        CheckSharedMemory(shared_memory, device_properties);
        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_R2C_NONE_XZ<FFT,complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));        
   
        // cudaErr(cudaSetDevice(0));
        //  cudaErr(cudaFuncSetCacheConfig( (void*)block_fft_kernel_R2C_NONE_XZ<FFT,complex_type,scalar_type>,cudaFuncCachePreferShared ));
        //  cudaFuncSetSharedMemConfig ( (void*)block_fft_kernel_R2C_NONE_XZ<FFT,complex_type,scalar_type>, cudaSharedMemBankSizeEightByte );


        #if DEBUG_FFT_STAGE > 0
          precheck
          block_fft_kernel_R2C_NONE_XZ<FFT,complex_type,scalar_type><< <LP.gridDims,  LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
          (scalar_input, complex_output, LP.mem_offsets, workspace);
          postcheck 
        #else
          is_in_buffer_memory = ! is_in_buffer_memory;
        #endif


           
        break;
      }      
      
      case r2c_decrease: {

        using FFT = decltype( FFT_base_arch() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() );  
        cudaError_t error_code = cudaSuccess;
        auto workspace = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;
        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, r2c_decrease);

        // the shared mem is mixed between storage, shuffling and FFT. For this kernel we need to add padding to avoid bank conlicts (N/32)
        int shared_memory = std::max( FFT::shared_memory_size * LP.threadsPerBlock.z, (LP.mem_offsets.shared_input + LP.mem_offsets.shared_input/32) * (unsigned int)sizeof(complex_type));

        CheckSharedMemory(shared_memory, device_properties);
        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_R2C_DECREASE_XY<FFT,complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));  

        #if DEBUG_FFT_STAGE > 0
          precheck
          block_fft_kernel_R2C_DECREASE_XY<FFT,complex_type,scalar_type><< <LP.gridDims,  LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
          ( scalar_input, complex_output, LP.mem_offsets, LP.twiddle_in,LP.Q, workspace);
          postcheck 
        #else
          is_in_buffer_memory = ! is_in_buffer_memory;
        #endif
        break;
      }

      case r2c_increase: {
        using FFT = decltype( FFT_base_arch() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() );  
        cudaError_t error_code = cudaSuccess;
        auto workspace = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;
        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, r2c_increase);

        int shared_memory = LP.mem_offsets.shared_input*sizeof(scalar_type) + FFT::shared_memory_size;

        CheckSharedMemory(shared_memory, device_properties);
        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_R2C_INCREASE_XY<FFT,complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));        
    
        #if DEBUG_FFT_STAGE > 0
          precheck
          block_fft_kernel_R2C_INCREASE_XY<FFT,complex_type,scalar_type><< <LP.gridDims,  LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
          ( scalar_input, complex_output, LP.mem_offsets, LP.twiddle_in,LP.Q, workspace);
          postcheck 
        #else
          is_in_buffer_memory = ! is_in_buffer_memory;
        #endif

        break;
      }

      case c2c_fwd_none: {

        using FFT = decltype( FFT_base_arch() + Direction<fft_direction::forward>() + Type<fft_type::c2c>() ); 
  
        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2c_fwd_none);

        cudaError_t error_code = cudaSuccess;
        auto workspace = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;
        int shared_memory = FFT::shared_memory_size;

        #if DEBUG_FFT_STAGE > 2
          CheckSharedMemory(shared_memory, device_properties);
          cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_NONE<FFT,complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));        
          precheck
          block_fft_kernel_C2C_NONE<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
          ( complex_input,  complex_output, LP.mem_offsets, workspace);
          postcheck
        #else
          // Since we skip the memory ops, unlike the other kernels, we need to flip the buffer pinter
          is_in_buffer_memory = ! is_in_buffer_memory;
        #endif

        break;
      }

      case c2c_fwd_none_Z: {

        using FFT = decltype( FFT_base_arch() + Direction<fft_direction::forward>() + Type<fft_type::c2c>() ); 
  
        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2c_fwd_none_Z);

        cudaError_t error_code = cudaSuccess;
        auto workspace = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;
        int shared_memory = FFT::shared_memory_size;

        #if DEBUG_FFT_STAGE > 1
          CheckSharedMemory(shared_memory, device_properties);
          cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_NONE_XYZ<FFT,complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory)); 
          precheck
          block_fft_kernel_C2C_NONE_XYZ<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
          ( complex_input,  complex_output, LP.mem_offsets, workspace);
          postcheck
        #else
          // Since we skip the memory ops, unlike the other kernels, we need to flip the buffer pinter
          is_in_buffer_memory = ! is_in_buffer_memory;
        #endif

        break;
      }

      case c2c_fwd_decrease: {

        using FFT = decltype( FFT_base_arch() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() );  
        cudaError_t error_code = cudaSuccess;
        auto workspace = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;
        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2c_fwd_decrease);

        #if DEBUG_FFT_STAGE > 2
          // the shared mem is mixed between storage, shuffling and FFT. For this kernel we need to add padding to avoid bank conlicts (N/32)
          // For decrease methods, the shared_input > shared_output
          int shared_memory = std::max( FFT::shared_memory_size * LP.threadsPerBlock.z, (LP.mem_offsets.shared_input + LP.mem_offsets.shared_input/32) * (unsigned int)sizeof(complex_type));

          CheckSharedMemory(shared_memory, device_properties);
          cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_DECREASE<FFT,complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));        
      
          precheck
          block_fft_kernel_C2C_DECREASE<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
          ( complex_input, complex_output, LP.mem_offsets, LP.twiddle_in,LP.Q, workspace);
          postcheck 
        #else
          // Since we skip the memory ops, unlike the other kernels, we need to flip the buffer pinter
          is_in_buffer_memory = ! is_in_buffer_memory;
        #endif

        break;
      }
      case c2c_fwd_increase: {
  
        using FFT = decltype(FFT_base_arch() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() );  
  
        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2c_fwd_increase);
        
        cudaError_t error_code = cudaSuccess;
        auto workspace = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;        
        // cudaErr(cudaFuncSetCacheConfig( (void*)block_fft_kernel_C2C_INCREASE<FFT,complex_type>,cudaFuncCachePreferShared ));
          // cudaFuncSetSharedMemConfig ( (void*)block_fft_kernel_C2C_INCREASE<FFT,complex_type>, cudaSharedMemBankSizeEightByte )
        #if DEBUG_FFT_STAGE > 2
          int shared_memory;
          // Aggregate the transformed frequency data in shared memory so that we can write to global coalesced.
          shared_memory = LP.mem_offsets.shared_output*sizeof(complex_type) + LP.mem_offsets.shared_input*sizeof(complex_type) + FFT::shared_memory_size;

          CheckSharedMemory(shared_memory, device_properties);
    
          // std::cout << "shared_memory " << shared_memory << std::endl;
          // When it is the output dims being smaller, may need a logical or different method
          //FIXME
          bool swap_real_space_quadrants = false;
          if (swap_real_space_quadrants)
          {
            MyFFTRunTimeAssertTrue(false, "c2c_fwd_increase with swap_real_space_quadrants == true, is not yet implemented.");
            cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_INCREASE_SwapRealSpaceQuadrants<FFT,complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));

            precheck
            block_fft_kernel_C2C_INCREASE_SwapRealSpaceQuadrants<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
            ( complex_input, complex_output, LP.mem_offsets, LP.twiddle_in,LP.Q, workspace);
            postcheck
          }
          else
          {
            cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_INCREASE<FFT,complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));

            precheck
            block_fft_kernel_C2C_INCREASE<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
            ( complex_input, complex_output, LP.mem_offsets, LP.twiddle_in,LP.Q, workspace);
            postcheck
          }
        #else
          // Since we skip the memory ops, unlike the other kernels, we need to flip the buffer pinter
          is_in_buffer_memory = ! is_in_buffer_memory;
        #endif
             
        // do something
        break; 
      }

      case c2c_fwd_increase_Z: {
  
        using FFT = decltype(FFT_base_arch() + Direction<fft_direction::forward>()+ Type<fft_type::c2c>() );  
  
        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2c_fwd_increase_Z);
        
        cudaError_t error_code = cudaSuccess;
        auto workspace = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;        
        // cudaErr(cudaFuncSetCacheConfig( (void*)block_fft_kernel_C2C_INCREASE<FFT,complex_type>,cudaFuncCachePreferShared ));
          // cudaFuncSetSharedMemConfig ( (void*)block_fft_kernel_C2C_INCREASE<FFT,complex_type>, cudaSharedMemBankSizeEightByte )
        #if DEBUG_FFT_STAGE > 2
          int shared_memory;
          // Aggregate the transformed frequency data in shared memory so that we can write to global coalesced.
          shared_memory = size_of<FFT>::value*sizeof(complex_type) + FFT::shared_memory_size;

          CheckSharedMemory(shared_memory, device_properties);
    
          // std::cout << "shared_memory " << shared_memory << std::endl;
          // When it is the output dims being smaller, may need a logical or different method
          //FIXME
          bool swap_real_space_quadrants = false;
          if (swap_real_space_quadrants)
          {
            MyFFTRunTimeAssertTrue(false, "c2c_fwd_increase with swap_real_space_quadrants == true, is not yet implemented.");

          }
          else
          {
            cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_INCREASE_Z<FFT,complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));

            precheck
            block_fft_kernel_C2C_INCREASE_Z<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
            ( complex_input, complex_output, LP.mem_offsets, LP.twiddle_in, LP.Q, workspace);
            postcheck
          }
        #else
          // Since we skip the memory ops, unlike the other kernels, we need to flip the buffer pinter
          is_in_buffer_memory = ! is_in_buffer_memory;
        #endif
             
        // do something
        break; 
      }

      case c2c_inv_none: {

        using FFT = decltype( FFT_base_arch() + Type<fft_type::c2c>() + Direction<fft_direction::inverse>() );

  
        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2c_inv_none);

        cudaError_t error_code = cudaSuccess;
        auto workspace = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;        
        
          int shared_memory = FFT::shared_memory_size;

          CheckSharedMemory(shared_memory, device_properties);
          cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_NONE<FFT,complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));        
        #if DEBUG_FFT_STAGE > 4  
          precheck
          block_fft_kernel_C2C_NONE<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
          ( complex_input, complex_output, LP.mem_offsets, workspace);
          postcheck
        #else
          // Since we skip the memory ops, unlike the other kernels, we need to flip the buffer pinter
          is_in_buffer_memory = ! is_in_buffer_memory;         
        #endif
        
      
        // do something
        break; 
      }
      case c2c_inv_none_Z: {

        using FFT = decltype( FFT_base_arch() + Direction<fft_direction::inverse>() + Type<fft_type::c2c>() ); 
  
        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2c_inv_none_Z);

        cudaError_t error_code = cudaSuccess;
        auto workspace = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;
        int shared_memory = FFT::shared_memory_size;
        #if DEBUG_FFT_STAGE > 5
          CheckSharedMemory(shared_memory, device_properties);
          cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_NONE_XY<FFT,complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory)); 
          precheck
          block_fft_kernel_C2C_NONE_XY<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
          ( complex_input,  complex_output, LP.mem_offsets, workspace);
          postcheck
        #else
          // Since we skip the memory ops, unlike the other kernels, we need to flip the buffer pinter
          is_in_buffer_memory = ! is_in_buffer_memory;
        #endif

        break;
      }

      case c2c_inv_decrease: {

        using FFT = decltype( FFT_base_arch() + Direction<fft_direction::inverse>()+ Type<fft_type::c2c>() );  
        cudaError_t error_code = cudaSuccess;
        auto workspace = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;
        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2c_inv_decrease);

        #if DEBUG_FFT_STAGE > 4
          // the shared mem is mixed between storage, shuffling and FFT. For this kernel we need to add padding to avoid bank conlicts (N/32)
          // For decrease methods, the shared_input > shared_output
          int shared_memory = std::max( FFT::shared_memory_size * LP.threadsPerBlock.z, (LP.mem_offsets.shared_input + LP.mem_offsets.shared_input/32) * (unsigned int)sizeof(complex_type));

          CheckSharedMemory(shared_memory, device_properties);
          cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_DECREASE<FFT,complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));        
      
          precheck
          block_fft_kernel_C2C_DECREASE<FFT,complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
          ( complex_input, complex_output, LP.mem_offsets, LP.twiddle_in,LP.Q, workspace);
          postcheck 
        #else
          // Since we skip the memory ops, unlike the other kernels, we need to flip the buffer pinter
          is_in_buffer_memory = ! is_in_buffer_memory;
        #endif

        break;
      }

      case c2c_inv_increase: {
        MyFFTRunTimeAssertTrue(false, "c2c_inv_increase is not yet implemented.");

        #if DEBUG_FFT_STAGE > 4
        #else
          // Since we skip the memory ops, unlike the other kernels, we need to flip the buffer pinter
          is_in_buffer_memory = ! is_in_buffer_memory;
        #endif

        break;
      }

      case c2r_none: {
  
        using FFT = decltype(FFT_base_arch() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() ); 
  
        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2r_none);
      
        cudaError_t error_code = cudaSuccess;
        auto workspace = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;        cudaErr(error_code);

        int shared_memory = FFT::shared_memory_size;

        CheckSharedMemory(shared_memory, device_properties);
        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2R_NONE<FFT,complex_type, scalar_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));  
        #if DEBUG_FFT_STAGE > 6
          precheck
          block_fft_kernel_C2R_NONE<FFT, complex_type, scalar_type><< <LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
          ( complex_input, scalar_output, LP.mem_offsets, workspace);
          postcheck
        #else
          is_in_buffer_memory = ! is_in_buffer_memory;
        #endif
  
        break; 
      }

      case c2r_none_XY: {
  
        using FFT = decltype(FFT_base_arch() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() ); 
  
        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2r_none_XY);
      
        cudaError_t error_code = cudaSuccess;
        auto workspace = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;        cudaErr(error_code);

        int shared_memory = FFT::shared_memory_size;

        CheckSharedMemory(shared_memory, device_properties);
        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2R_NONE_XY<FFT,complex_type, scalar_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));  
        #if DEBUG_FFT_STAGE > 6
          precheck
          block_fft_kernel_C2R_NONE_XY<FFT, complex_type, scalar_type><< <LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
          ( complex_input, scalar_output, LP.mem_offsets, workspace);
          postcheck
        #else
          is_in_buffer_memory = ! is_in_buffer_memory;
        #endif
  
        break; 
      }

      case c2r_decrease: {
        using FFT = decltype(FFT_base_arch() + Direction<fft_direction::inverse>()+ Type<fft_type::c2r>() ); 
  
        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, c2r_decrease);
      
        cudaError_t error_code = cudaSuccess;
        auto workspace = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;        cudaErr(error_code);

        int shared_memory = std::max( FFT::shared_memory_size * LP.gridDims.z , (LP.mem_offsets.shared_input + LP.mem_offsets.shared_input/32) * (unsigned int)sizeof(complex_type));

        CheckSharedMemory(shared_memory, device_properties);
        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2R_DECREASE_XY<FFT,complex_type, scalar_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory)); 


        #if DEBUG_FFT_STAGE > 6
          precheck
          block_fft_kernel_C2R_DECREASE_XY<FFT, complex_type, scalar_type><< <LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
          ( complex_input, scalar_output, LP.mem_offsets, LP.twiddle_in, LP.Q, workspace);
          postcheck

          transform_stage_completed = TransformStageCompleted::inv;
        #else
          is_in_buffer_memory = ! is_in_buffer_memory;  
        #endif
   
        break;
      }

      case c2r_increase: {
        MyFFTRunTimeAssertTrue(false, "c2r_increase is not yet implemented.");
        break;
      }

      case xcorr_fwd_increase_inv_none: {
  
        using FFT    = decltype( FFT_base_arch() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() ); 
        using invFFT = decltype( FFT_base_arch() + Type<fft_type::c2c>() + Direction<fft_direction::inverse>() ); 
          
        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, xcorr_fwd_increase_inv_none);

        cudaError_t error_code = cudaSuccess;
        auto workspace_fwd = make_workspace<FFT>(error_code); // presumably larger of the two
        cudaErr(error_code);
        error_code = cudaSuccess;
        auto workspace_inv = make_workspace<invFFT>(error_code); // presumably larger of the two
        cudaErr(error_code);
  
        int shared_memory = invFFT::shared_memory_size;
        CheckSharedMemory(shared_memory, device_properties);
          // cudaErr(cudaFuncSetCacheConfig( (void*)block_fft_kernel_C2C_INCREASE<FFT,complex_type>,cudaFuncCachePreferShared ));
          // cudaFuncSetSharedMemConfig ( (void*)block_fft_kernel_C2C_INCREASE<FFT,complex_type>, cudaSharedMemBankSizeEightByte );  
        // FIXME
        #if DEBUG_FFT_STAGE > 2
          bool swap_real_space_quadrants = false;   
          if (swap_real_space_quadrants)
          {
            MyFFTRunTimeAssertTrue(false, "Swapping real space quadrants is not yet implemented.");
            // cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_FWD_INCREASE_INV_NONE_ConjMul_SwapRealSpaceQuadrants<FFT,invFFT, complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));        

            // precheck
            // block_fft_kernel_C2C_FWD_INCREASE_INV_NONE_ConjMul_SwapRealSpaceQuadrants<FFT,invFFT, complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
            // ( (complex_type *)d_ptr.image_to_search, complex_input, complex_output, LP.mem_offsets, LP.twiddle_in,LP.Q, workspace_fwd, workspace_inv);
            // postcheck
          }
          else
          {

            cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_FWD_INCREASE_INV_NONE_ConjMul<FFT, invFFT, complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));        
            precheck

            // Right now, because of the n_threads == size_of<FFT> requirement, we are explicitly zero padding, so we need to send an "apparent Q" to know the input size.
            // Could send the actual size, but later when converting to use the transform decomp with different sized FFTs this will be a more direct conversion.
            int apperent_Q = size_of<FFT>::value / fwd_dims_in.y;
   
            block_fft_kernel_C2C_FWD_INCREASE_INV_NONE_ConjMul<FFT, invFFT, complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
            ( (complex_type *)d_ptr.image_to_search, complex_input, complex_output , LP.mem_offsets,apperent_Q, workspace_fwd, workspace_inv);
            postcheck
          }
        #else
          is_in_buffer_memory = ! is_in_buffer_memory;
        #endif
            
        // do something
        break; 
      }

      case xcorr_fwd_none_inv_decrease: {
        using FFT    = decltype( FFT_base_arch() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() ); 
        using invFFT = decltype( FFT_base_arch() + Type<fft_type::c2c>() + Direction<fft_direction::inverse>() ); 
          
        LaunchParams LP = SetLaunchParameters(elements_per_thread_complex, xcorr_fwd_none_inv_decrease);
  
        cudaError_t error_code = cudaSuccess;
        auto workspace_fwd = make_workspace<FFT>(error_code); // presumably larger of the two
        cudaErr(error_code);
        error_code = cudaSuccess;
        auto workspace_inv = make_workspace<invFFT>(error_code); // presumably larger of the two
        cudaErr(error_code);
  
        // Max shared memory needed to store the full 1d fft remaining on the forward transform
        unsigned int shared_memory = FFT::shared_memory_size + (unsigned int)sizeof(complex_type) * LP.mem_offsets.physical_x_input;
        // shared_memory = std::max( shared_memory, std::max( invFFT::shared_memory_size * LP.threadsPerBlock.z, (LP.mem_offsets.shared_input + LP.mem_offsets.shared_input/32) * (unsigned int)sizeof(complex_type)));

        CheckSharedMemory(shared_memory, device_properties);
 
        
          // cudaErr(cudaFuncSetCacheConfig( (void*)block_fft_kernel_C2C_INCREASE<FFT,complex_type>,cudaFuncCachePreferShared ));
          // cudaFuncSetSharedMemConfig ( (void*)block_fft_kernel_C2C_INCREASE<FFT,complex_type>, cudaSharedMemBankSizeEightByte );  
        // FIXME
        #if DEBUG_FFT_STAGE > 2


          bool swap_real_space_quadrants = false;   
          if (swap_real_space_quadrants)
          {
            // cudaErr(cudaFuncSetAttribute((void*)_INV_DECREASE_ConjMul_SwapRealSpaceQuadrants<FFT,invFFT, complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));        
            MyFFTDebugAssertFalse(swap_real_space_quadrants, "Swap real space quadrants not yet implemented in xcorr_fwd_none_inv_decrease.");

            // precheck
            // _INV_DECREASE_ConjMul_SwapRealSpaceQuadrants<FFT,invFFT, complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
            // ( (complex_type *)d_ptr.image_to_search, complex_input, complex_output, LP.mem_offsets, LP.twiddle_in,LP.Q, workspace_fwd, workspace_inv);
            // postcheck
          }
          else
          {
            cudaErr(cudaFuncSetAttribute((void*)_INV_DECREASE_ConjMul<FFT, invFFT, complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));        
            // Right now, because of the n_threads == size_of<FFT> requirement, we are explicitly zero padding, so we need to send an "apparent Q" to know the input size.
            // Could send the actual size, but later when converting to use the transform decomp with different sized FFTs this will be a more direct conversion.
            int apparent_Q = size_of<FFT>::value / inv_dims_out.y;
            precheck
            _INV_DECREASE_ConjMul<FFT, invFFT, complex_type><< <LP.gridDims,  LP.threadsPerBlock, shared_memory, cudaStreamPerThread>> >
            ( (complex_type *)d_ptr.image_to_search, complex_input, complex_output , LP.mem_offsets, LP.twiddle_in, apparent_Q, workspace_fwd, workspace_inv);
            postcheck
          }
          transform_stage_completed = TransformStageCompleted::fwd;

        #else
          is_in_buffer_memory = ! is_in_buffer_memory;
        #endif
            
        // do something
        break; 
      } // end case xcorr_fwd_none_inv_decrease    
      default:
        // throw something
        break;
  
    }
  }


    



  // 
} // end set and launc kernel

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Some helper functions that are annoyingly long to have in the header.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::GetTransformSize(KernelType kernel_type)
{
  // Set member variable transform_size.N (.P .L .Q)

  if ( IsR2CType(kernel_type) )
  {
    AssertDivisibleAndFactorOf2( std::max(fwd_dims_in.x, fwd_dims_out.x), std::min(fwd_dims_in.x, fwd_dims_out.x) );
  }
  else if ( IsC2RType(kernel_type) )
  {
    // FIXME
    if (kernel_type == c2r_decrease)
    {
      AssertDivisibleAndFactorOf2( std::max(inv_dims_in.x, inv_dims_out.x), std::max(inv_dims_in.x, inv_dims_out.x) );
    }
    else
    {
      AssertDivisibleAndFactorOf2( std::max(inv_dims_in.x, inv_dims_out.x), std::min(inv_dims_in.x, inv_dims_out.x) );
    }
  }
  else
  {
    // C2C type
    if ( IsForwardType( kernel_type) )
    {
      switch (transform_dimension)
      {
        case 1: { AssertDivisibleAndFactorOf2( std::max(fwd_dims_in.x, fwd_dims_out.x),  std::min(fwd_dims_in.x, fwd_dims_out.x) ); break; }
        case 2: { 
          if (kernel_type == xcorr_fwd_increase_inv_none)
          {
            // FIXME
            AssertDivisibleAndFactorOf2( std::max(fwd_dims_in.y, fwd_dims_out.y),  std::max(fwd_dims_in.y, fwd_dims_out.y) ); 
          }
          else
          {
            AssertDivisibleAndFactorOf2( std::max(fwd_dims_in.y, fwd_dims_out.y),  std::min(fwd_dims_in.y, fwd_dims_out.y) ); 
          }
          break; 
        }
        case 3:  {
          if (IsTransormAlongZ(kernel_type)) 
          {
            AssertDivisibleAndFactorOf2( std::max(fwd_dims_in.z, fwd_dims_out.z),  std::min(fwd_dims_in.z, fwd_dims_out.z) ); 
          }
          else
          {
            AssertDivisibleAndFactorOf2( std::max(fwd_dims_in.y, fwd_dims_out.y),  std::min(fwd_dims_in.y, fwd_dims_out.y) ); 
          }
          
          break;
        }

        default: { MyFFTDebugAssertTrue(false, "ERROR: Invalid transform dimension for c2c fwd type.\n"); }
      }
    }
    else
    {
      switch (transform_dimension)
      {
        case 1: { AssertDivisibleAndFactorOf2( std::max(inv_dims_in.x, inv_dims_out.x),  std::min(inv_dims_in.x, inv_dims_out.x) ); break; }
        case 2: { 
          if (kernel_type == xcorr_fwd_none_inv_decrease)
          {
            // FIXME, for now using full transform
            AssertDivisibleAndFactorOf2( std::max(inv_dims_in.y, inv_dims_out.y),  std::max(inv_dims_in.y, inv_dims_out.y) );
          }
          else
          {
            AssertDivisibleAndFactorOf2( std::max(inv_dims_in.y, inv_dims_out.y),  std::min(inv_dims_in.y, inv_dims_out.y) ); 
          }
          break; 
        }
        case 3:  {
          if (IsTransormAlongZ(kernel_type)) 
          {
            AssertDivisibleAndFactorOf2( std::max(inv_dims_in.z, inv_dims_out.z),  std::min(inv_dims_in.z, inv_dims_out.z) ); 
          }
          else
          {
            AssertDivisibleAndFactorOf2( std::max(inv_dims_in.y, inv_dims_out.y),  std::min(inv_dims_in.y, inv_dims_out.y) ); 
          }
          
          break;
        }

        default: { MyFFTDebugAssertTrue(false, "ERROR: Invalid transform dimension for c2c inverse type.\n"); }
      }
    }


  }

} // end GetTransformSize function

template <class ComputeType, class InputType, class OutputType>
void FourierTransformer<ComputeType, InputType, OutputType>::GetTransformSize_thread(KernelType kernel_type, int thread_fft_size)
{

  transform_size.P = thread_fft_size;
  
  switch (kernel_type)
  {
    case r2c_decomposed:
      transform_size.N = fwd_dims_in.x;
      break;
    case r2c_decomposed_transposed:
      transform_size.N = fwd_dims_in.x;
      break; 
    case c2c_decomposed:
    // FIXME fwd vs inv
      if (fwd_dims_in.y == 1) transform_size.N = fwd_dims_in.x;
      else transform_size.N = fwd_dims_in.y;
      break;
    case c2r_decomposed:
      transform_size.N = inv_dims_out.x;
      break;
    case c2r_decomposed_transposed:
      transform_size.N = inv_dims_out.x;
      break;
    case xcorr_decomposed:
        // FIXME fwd vs inv
      if (fwd_dims_in.y == 1) transform_size.N = fwd_dims_out.x; // FIXME should probably throw an error for now.
      else transform_size.N = fwd_dims_out.y; // does fwd_dims_in make sense?

      break;
    default:
      std::cerr << "Function GetTransformSize_thread does not recognize the kernel type ( " << KernelName[kernel_type] << " )" << std::endl;
      exit(-1);
  }

  if (transform_size.N % transform_size.P != 0) { std::cerr << "Thread based decompositions must factor by thread_fft_size (" << thread_fft_size << ") in the current implmentations." << std::endl; exit(-1); }
  transform_size.Q = transform_size.N / transform_size.P;
} // end GetTransformSize_thread function

template <class ComputeType, class InputType, class OutputType>
LaunchParams FourierTransformer<ComputeType, InputType, OutputType>::SetLaunchParameters(const int& ept, KernelType kernel_type, bool do_forward_transform)
{
  /*
    Assuming:
    1) r2c/c2r imply forward/inverse transform. 
       c2c_padded implies forward transform.
    2) for 2d or 3d transforms the x/y dimensions are transposed in momentum space during store on the 1st set of 1ds transforms.
    3) if 1d then z = y = 1.

    threadsPerBlock = size/threads_per_fft (for thread based transforms)
                    = size of fft ( for block based transforms ) NOTE: Something in cufftdx seems to be very picky about this. Launching > threads seem to cause problems.
    gridDims = number of 1d FFTs, placed on blockDim perpendicular
    shared_input/output = number of elements reserved in dynamic shared memory. TODO add a check on minimal (48k) and whether this should be increased (depends on Arch)
    physical_x_input/output = number of elements along the fast (x) dimension, depends on fftw padding && whether the memory is currently transposed in x/y
    twiddle_in = +/- 2*PI/Largest dimension : + for the inverse transform
    Q = number of sub-transforms
  */
  LaunchParams L;

  // This is the same for all kernels as set in AssertDivisibleAndFactorOf2()
  L.Q = transform_size.Q;

  // Set the twiddle factor, only differ in sign between fwd/inv transforms.
  // For mixed kernels (eg. xcorr_* the size type is defined by where the size change happens.
  // FIXME fwd_increase (oversampling) xcorr -> inv decrease (peak search) is a likely algorithm, that will not fit with this logic.
  SizeChangeType size_change_type;
  if ( IsForwardType(kernel_type) ) 
  {
    size_change_type = fwd_size_change_type;
    L.twiddle_in = L.twiddle_in = -2*PIf/transform_size.N ;
  }
  else 
  {
    size_change_type = inv_size_change_type;
    L.twiddle_in = L.twiddle_in = 2*PIf/transform_size.N ;
  }

  // Set the thread block dimensions
  if ( IsThreadType(kernel_type) ) {
      L.threadsPerBlock = dim3(transform_size.Q, 1, 1);
  }
  else {
    if (size_change_type == decrease) {
      L.threadsPerBlock = dim3(transform_size.P/ept, 1, transform_size.Q);
    }
    else {
      // In the current xcorr methods that have INCREASE, explicit zero padding is used, so this will be overridden (overrode?) with transform_size.N
      L.threadsPerBlock = dim3(transform_size.P/ept, 1, 1);
    }
  }

  
  // Set the shared mem sizes, which depend on the size_change_type
  switch (size_change_type)
  {
    case no_change: {
      // no shared memory is needed outside that for the FFT itself.
      // For C2C kernels of size_type increase, the shared output may be reset below in order to store for coalesced global writes.
      L.mem_offsets.shared_input  = 0;
      L.mem_offsets.shared_output = 0;
      break;
    }
    case decrease: {
      // Prior to reduction, we must be able to store the full transform. An alternate algorithm with multiple reads would relieve this dependency and 
      // may be worth considering if L2 cache residence on Ampere is an effective way to reduce repeated Globabl memory access.
      // Note: that this shared memory is not static, in the sense that it is used both for temporory fast storage, as well as the calculation of the FFT. The max of those two requirments is calculated per kernel.
      L.mem_offsets.shared_input  = transform_size.N;
      if (IsR2CType(kernel_type)) { L.mem_offsets.shared_output = 0; } 
      else { L.mem_offsets.shared_output = transform_size.N; } // TODO this line is just from case increase, haven't thought about it.
      break;
    }
    case increase: {
      // We want to re-use the input memory as we loop over construction of the full FFT. This shared memory is independent of the
      // memory used for the FFT itself.
      L.mem_offsets.shared_input  = transform_size.P;
      if (IsR2CType(kernel_type)) { L.mem_offsets.shared_output = 0; } 
      else { L.mem_offsets.shared_output = transform_size.N; }
      // Note: This is overwritten in the C2C methods as it depends on 1d vs 2d and fwd vs inv.
      break;
    }
    default: {
      MyFFTDebugAssertTrue(false, "Unknown size_change_type ( " + std::to_string(size_change_type) + " )");
    }
  } // switch on size change

  
  // Set the grid dimensions and pixel pitch
  if (IsR2CType(kernel_type)) 
  {
    L.gridDims = dim3(1, fwd_dims_in.y, fwd_dims_in.z); 
    L.mem_offsets.physical_x_input = fwd_dims_in.w*2; // scalar type, natural 
    L.mem_offsets.physical_x_output = fwd_dims_out.w;

  } 
  else if (IsC2RType(kernel_type)) 
  {
    // This is always the last op, so if there is a size change, it will have happened once on C2C, reducing the number of blocks
    L.gridDims = dim3(1, inv_dims_out.y, inv_dims_out.z);
    L.mem_offsets.physical_x_input = inv_dims_in.w;
    L.mem_offsets.physical_x_output = inv_dims_out.w*2;      
  }
  else // C2C type
  {
    // All dimensions have the same physical x dims for C2C transforms
    if ( IsForwardType(kernel_type) ) 
    {
      // If 1d, this is implicitly a complex valued input, s.t. fwd_dims_in.x = fwd_dims_in.w.) But if fftw_padding is allowed false this may not be true.
      L.mem_offsets.physical_x_input =  fwd_dims_in.w; 
      L.mem_offsets.physical_x_output = fwd_dims_out.w;
    }
    else
    {
      L.mem_offsets.physical_x_input =  inv_dims_in.w;
      L.mem_offsets.physical_x_output = inv_dims_out.w;
    }

    switch (transform_dimension)
    {
      case 1: {
        L.gridDims = dim3(1, 1, 1);
        break;
      }
      case 2: {
        if ( IsForwardType(kernel_type) ) 
        {
          L.gridDims = dim3(1, fwd_dims_out.w, fwd_dims_out.z);
        }
        else
        {
          L.gridDims = dim3(1, inv_dims_out.w, inv_dims_out.z);
        }
        break;
      }
      case 3: {
        if (IsTransormAlongZ(kernel_type))
        {
          // When transforming along the Z-dimension, The Z grid dimensions for a 3d kernel are used to indicate the transposed x coordinate.
          if ( IsForwardType(kernel_type) ) 
          {
            L.gridDims = dim3(1, fwd_dims_out.y, fwd_dims_out.w);
          }
          else
          {
            L.gridDims = dim3(1, inv_dims_out.y, inv_dims_out.w);
          }
        }
        else
        {
          if ( IsForwardType(kernel_type) ) 
          {
            L.gridDims = dim3(1, fwd_dims_out.w, fwd_dims_out.z);
          }
          else
          {
            L.gridDims = dim3(1, inv_dims_out.w, inv_dims_out.z);
          }
        }          
        break;
      } // 3 dimensional case
      default: {
        MyFFTDebugAssertTrue(false, "Unknown transform_dimension ( " + std::to_string(transform_dimension) + " )");
      }
    }
  }

  // FIXME
  // Some shared memory over-rides
  if (  kernel_type == c2c_inv_decrease || kernel_type == c2c_inv_increase )
  {
    L.mem_offsets.shared_output = inv_dims_out.y;
  }

  // FIXME
  // Some xcorr overrides TODO try the DECREASE approcae
  if (kernel_type == xcorr_fwd_increase_inv_none)
  {
    // FIXME not correct for 3D
    L.threadsPerBlock = dim3(transform_size.N/ept, 1, 1);
  }

  if (kernel_type == xcorr_fwd_none_inv_decrease)
  {
    // FIXME not correct for 3D

    L.threadsPerBlock = dim3(transform_size.N/ept, 1, 1);
    // FIXME
    L.gridDims = dim3(1, fwd_dims_out.w, 1);
    L.mem_offsets.physical_x_input = inv_dims_in.y;
    L.mem_offsets.physical_x_output = inv_dims_out.y;
  }
 
  return L;
}

} // namespace fast_FFT



