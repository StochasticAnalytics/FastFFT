// Insert some license stuff here

#ifndef fast_FFT_H_
#define fast_FFT_H_



// #include "/groups/himesb/git/cufftdx/example/block_io.hpp"
// #include "/groups/himesb/git/cufftdx/example/common.hpp"
// #include <iostream>

namespace FastFFT {

  constexpr const float PIf = 3.14159275358979323846f;

  typedef
	struct __align__(8) _Offsets{
    short shared_input;
    short shared_output;
    short pixel_pitch_input;
    short pixel_pitch_output;
  } Offsets;

  typedef 
  struct __align__(32) _LaunchParams{
    int Q;
    float twiddle_in;
    dim3 gridDims;
    dim3 threadsPerBlock;
    Offsets mem_offsets;
  } LaunchParams;

  

  template<typename I, typename C>
  struct DevicePointers {
    // Use this to catch unsupported input/ compute types and throw exception.
    int* position_space = nullptr;
    int* position_space_buffer = nullptr;
    int* momentum_space = nullptr;
    int* momentum_space_buffer = nullptr;
  };

  // Input real, compute single-precision
  template<>
  struct DevicePointers<float*, float*> {
    float*  position_space;
    float*  position_space_buffer;
    float2* momentum_space;
    float2* momentum_space_buffer;

  };

  // Input real, compute half-precision FP16
  template<>
  struct DevicePointers<__half*, __half*> {
    __half*  position_space;
    __half*  position_space_buffer;
    __half2* momentum_space;
    __half2* momentum_space_buffer;
  };

  // Input complex, compute single-precision
  template<>
  struct DevicePointers<float2*, float*> {
    float2*  position_space;
    float2*  position_space_buffer;
    float2*  momentum_space;
    float2*  momentum_space_buffer;
  };

  // Input complex, compute half-precision FP16
  template<>
  struct DevicePointers<__half2*, __half*> {
    __half2*  position_space;
    __half2*  position_space_buffer;
    __half2*  momentum_space;
    __half2*  momentum_space_buffer;

  };


  

template <class ComputeType = float, class InputType = float, class OutputType = float>
class FourierTransformer {

public:

  // Used to specify input/calc/output data types
  enum DataType { int4_2, uint8, int8, uint16, int16, fp16, bf16, tf32, uint32, int32, fp32};
  enum OriginType { natural, centered, quadrant_swapped}; // Used to specify the origin of the data



  short  padding_jump_val;
  int input_memory_allocated;
  int output_memory_allocated;
  int compute_memory_allocated;
  int input_number_non_padding_values; // not used, but set in constructor
  int output_number_non_padding_values;// not used, but set in constructor


  ///////////////////////////////////////////////
  // Initialization functions
  ///////////////////////////////////////////////

  FourierTransformer();
  // FourierTransformer(const FourierTransformer &); // Copy constructor
  virtual ~FourierTransformer();

  // This is pretty similar to an FFT plan, I should probably make it align with CufftPlan
  void SetInputDimensionsAndType(size_t input_logical_x_dimension, 
                                 size_t input_logical_y_dimension, 
                                 size_t input_logical_z_dimension, 
                                 bool is_padded_input, 
                                 bool is_host_memory_pinned, 
                                 OriginType input_origin_type);
  
  void SetOutputDimensionsAndType(size_t output_logical_x_dimension, 
                                  size_t output_logical_y_dimension, 
                                  size_t output_logical_z_dimension, 
                                  bool is_padded_output, 
                                  OriginType output_origin_type);

  // For the time being, the caller is responsible for having the memory allocated for any of these input/output pointers.
  void SetInputPointer(InputType* input_pointer, bool is_input_on_device);


  ///////////////////////////////////////////////
  // Public actions:
  // ALL public actions should call ::CheckDimensions() to ensure the meta data are properly intialized.
  // this ensures the prior three methods have been called and are valid.
  ///////////////////////////////////////////////

  void CopyHostToDevice();
  // By default we are blocking with a stream sync until complete for simplicity. This is overkill and should FIXME.
  void CopyDeviceToHost(bool free_gpu_memory, bool unpin_host_memory);
  // When the size changes, we need a new host pointer
  void CopyDeviceToHost(OutputType* output_pointer, bool free_gpu_memory = true, bool unpin_host_memory = true);
  // FFT calls

  void FwdFFT(bool swap_real_space_quadrants = false, bool transpose_output = true);
  void InvFFT(bool transpose_output = true);
  void CrossCorrelate(float2* image_to_search, bool swap_real_space_quadrants);
  void CrossCorrelate(__half2* image_to_search, bool swap_real_space_quadrants);

  void ClipIntoTopLeft();
  void ClipIntoReal(int wanted_coordinate_of_box_center_x, int wanted_coordinate_of_box_center_y, int wanted_coordinate_of_box_center_z);

  // For all real valued inputs, assumed for any InputType that is not float2 or __half2

  int inline ReturnInputMemorySize() { return input_memory_allocated; }
  int inline ReturnOutputMemorySize() { return output_memory_allocated; }

  template<typename T, bool is_on_host = true>
  void SetToConstant(T* input_pointer, int N_values, const T& wanted_value)
  {
    if (is_on_host) 
    {
      for (int i = 0; i < N_values; i++)
      {
        input_pointer[i] = wanted_value;
      }
    }
    else
    {
      exit(-1);
    }
  }

  // Input is real or complex inferred from InputType
  DevicePointers<InputType*, ComputeType*> d_ptr;

private:


  OriginType input_origin_type;
  OriginType output_origin_type;

  // booleans to track state, could be bit fields but that seem opaque to me.
  bool is_in_memory_host_pointer; // To track allocation of host side memory
  bool is_in_memory_device_pointer; // To track allocation of device side memory.
  bool is_in_buffer_memory; // To track whether the current result is in dev_ptr.position_space or dev_ptr.position_space_buffer (momemtum space/ momentum space buffer respectively.)


  bool is_host_memory_pinned; // Specified in the constructor. Assuming host memory won't be pinned for many applications.

  bool is_fftw_padded_input; // Padding for in place r2c transforms
  bool is_fftw_padded_output; // Currently the output state will match the input state, otherwise it is an error.

  bool is_real_valued_input; // This is determined by the input type. If it is a float2 or __half2, then it is assumed to be a complex valued input function.

  bool is_set_input_params; // Yes, yes, "are" set.
  bool is_set_output_params;
  bool is_size_validated; // Defaults to false, set after both input/output dimensions are set and checked.
  bool is_set_input_pointer; // May be on the host of the device.

  int  transform_dimension; // 1,2,3d.
  int  transform_size; // Size of the 1d (sub) FFT which is <= the size of the full transform determined by the larger of the input/output dimensions. 
  int  transform_divisor; // full size/ transform size = Q the number of sub-transforms.
  enum SizeChangeType { increase, decrease, none }; // Assumed to be the same for all dimesnions. This may be relaxed later.
  SizeChangeType size_change_type;

  short4 dims_in;
  short4 dims_out;

  InputType* host_pointer;
  InputType* pinnedPtr;

  void Deallocate();
  void UnPinHostMemory();

  void SetDefaults();
  void CheckDimensions();


  enum KernelType { r2c_decomposed, r2c_decomposed_transposed, r2c_transposed, c2c_padded, c2c, c2c_decomposed, c2r_transposed,
                    c2r_decomposed, c2r_decomposed_transposed,  xcorr_transposed, xcorr_decomposed}; // Used to specify the origin of the data
    
  std::vector<std::string> 
        KernelName{ "r2c_decomposed", "r2c_decomposed_transposed", "r2c_transposed", "c2c_padded", "c2c", "c2c_decomposed", "c2r_transposed",
                    "c2r_decomposed", """c2r_decomposed_transposed",  "xcorr_transposed", "xcorr_decomposed"};

  void GetTransformSize(KernelType kernel_type)
  {
    int input_size;

    switch (kernel_type)
    {
      case r2c_transposed:
        input_size = dims_in.x;
        break;
      case xcorr_decomposed:
        input_size = dims_out.y;
        break;
      case c2c_padded:
        input_size = dims_in.y;
        break;
      case c2c;
        input_size = dims_out.y;
        break;
      case c2r_transposed:
        input_size = dims_out.x;
        break;
      default:
        std::cerr << "Function GetTransformSize does not recognize the kernel type ( " << KernelName[kernel_type] << " )" << std::endl;
        exit(-1);
    }


    if ( abs(fmod(log2(float(input_size)), 1)) < 1e-6 ) 
    {
      // For the time being, this also implies a block transform rather than a thread transform.
      transform_divisor = 1;
      transform_size = input_size;
      // TODO for larger sizes, below
      // transform_size = input_size / transform_divisor;
    }
    else 
    {
      if ( abs(fmod(log2(float(input_size)/3), 1)) < 1e-6) 
      {
        transform_divisor = 3;
        transform_size = input_size / transform_divisor;
      }
      else
      {
        std::cerr << "The input dimension must factor into powers of two, with at most one factor of three." << std::endl;
        exit(-1);
      }
    }
  };

  inline void GetTransformSize_thread(KernelType kernel_type, int thread_fft_size)
  {
        int input_size;

    switch (kernel_type)
    {
      case r2c_decomposed:
        input_size = dims_in.x;
        break;
      case r2c_decomposed_transposed:
        input_size = dims_in.x;
        break;
      case c2c_decomposed:
        if (dims_in.y == 1) input_size = dims_in.x;
        else input_size = dims_in.y;
        break;
      case c2r_decomposed:
        input_size = dims_out.x;
        break;
      case c2r_decomposed_transposed:
        input_size = dims_out.x;
        break;
      case xcorr_decomposed:
        input_size = dims_out.y;
        break;
      default:
        std::cerr << "Function GetTransformSize_thread does not recognize the kernel type ( " << KernelName[kernel_type] << " )" << std::endl;
        exit(-1);
    }

    if (input_size % thread_fft_size != 0) { std::cerr << "Thread based decompositions must factor by thread_fft_size (" << thread_fft_size << ") in the current implmentations." << std::endl; exit(-1); }
    transform_divisor = input_size / thread_fft_size;
    transform_size = thread_fft_size;
  };



  // TODO: not sure this should be inlined. (Probably ignored by the compiler anyway.)
  inline LaunchParams SetLaunchParameters(const int& ept, KernelType kernel_type, bool do_forward_transform = true)
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
      pixel_pitch_input/output = number of elements along the fast (x) dimension, depends on fftw padding && whether the memory is currently transposed in x/y
      twiddle_in = +/- 2*PI/Largest dimension : + for the inverse transform
      Q = number of sub-transforms
    */
    LaunchParams L;
    switch (kernel_type)
    {
      case r2c_decomposed: 
        // This is also fine for a 1d transform
        L.threadsPerBlock = dim3(transform_divisor, 1, 1);
        L.gridDims = dim3(1, dims_in.y, 1); 
        L.mem_offsets.shared_input = 0;
        L.mem_offsets.shared_output = dims_out.w; 
        L.mem_offsets.pixel_pitch_input = dims_in.w*2; // scalar type, natural 
        L.mem_offsets.pixel_pitch_output = dims_out.w; // complex type
        L.twiddle_in = -2*PIf/dims_in.x ;
        L.Q =  transform_divisor; //  (dims_in / transform size)
        break;

      case r2c_decomposed_transposed: 
        // The logic in this kernel would cause a segfault if the output is transposed in 1d
        if (transform_dimension == 1) { std::cerr << "r2c_decomposed_transposed is not supported for 1d transforms." << std::endl; exit(-1); }
      
        L.threadsPerBlock = dim3(transform_divisor, 1, 1);
        L.gridDims = dim3(1, dims_in.y, 1); 
        L.mem_offsets.shared_input = 0;  
        L.mem_offsets.shared_output = dims_out.w;
        L.mem_offsets.pixel_pitch_input = dims_in.w*2; // scalar type, natural 
        L.mem_offsets.pixel_pitch_output = dims_out.y; // complex type
        L.twiddle_in = -2*PIf/dims_in.x ;
        L.Q =  transform_divisor; //  (dims_in / transform size)
        break;        

      case r2c_transposed:
        // The logic in this kernel would cause a segfault if the output is transposed in 1d
        if (transform_dimension == 1) { std::cerr << "r2c_decomposed_transposed is not supported for 1d transforms." << std::endl; exit(-1); }

        L.threadsPerBlock = dim3(transform_size/ept, 1, 1);
        L.gridDims = dim3(transform_divisor, dims_in.y, 1); 
        L.mem_offsets.shared_input = dims_in.x;
        L.mem_offsets.shared_output = dims_out.w; // used in bounds check.
        L.mem_offsets.pixel_pitch_input = dims_in.w*2; // scalar type, natural 
        L.mem_offsets.pixel_pitch_output = dims_out.y; // complex type, transposed
        L.twiddle_in = -2*PIf/dims_out.x;
        L.Q = dims_out.x / dims_in.x; 
        break;

      case c2c_padded:
        // This is implicitly a forward transform
        switch (transform_dimension)
        {
          case 1: {         
            // If 1d, this is implicitly a complex valued input, s.t. dims_in.x = dims_in.w.) But if fftw_padding is allowed false this may not be true.
            L.threadsPerBlock = dim3(transform_size/ept, 1, 1);
            L.gridDims = dim3(transform_divisor, 1, 1);
            L.mem_offsets.shared_input = dims_in.x;
            L.mem_offsets.shared_output = dims_out.w;
            L.mem_offsets.pixel_pitch_input = dims_in.w; // complex ype, natural
            L.mem_offsets.pixel_pitch_output = dims_out.w; // complex type, natural
            L.twiddle_in = -2*PIf/dims_out.x;
            L.Q = dims_out.x / dims_in.x;
            break;
          }
          case 2: {
            L.threadsPerBlock = dim3(transform_size/ept, 1, 1); 
            L.gridDims = dim3(transform_divisor, dims_out.w, 1);
            L.mem_offsets.shared_input = dims_in.y;
            L.mem_offsets.shared_output = dims_out.y;
            L.mem_offsets.pixel_pitch_input = dims_out.y;
            L.mem_offsets.pixel_pitch_output = dims_out.y;
            L.twiddle_in = -2*PIf/dims_in.y;
            L.Q = dims_out.y / dims_in.y; // FIXME assuming for now this is already divisible
            break;
          }
          case 3: {
            // Not implemented
            std::cerr << "3d c2c_padded not implemented" << std::endl;
            exit(-1);
            break;
          }
        } // end switch on transform dimension
        // If inverse we need to negate the twidddle factor.
        if ( ! do_forward_transform)  L.twiddle_in  = -L.twiddle_in ;
        break; // case c2c_padded

      case c2c:
        switch (transform_dimension)
        {
          case 1: {  
            // If 1d, this is implicitly a complex valued input, s.t. dims_in.x = dims_in.w.) But if fftw_padding is allowed false this may not be true.
            L.threadsPerBlock = dim3(transform_size/ept, 1, 1);
            L.gridDims = dim3(transform_divisor, 1, 1);
            L.mem_offsets.pixel_pitch_input = dims_in.w;
            L.mem_offsets.pixel_pitch_output = dims_out.w;
            L.twiddle_in = -2*PIf/dims_out.x;
            L.Q = dims_out.x / dims_in.x; // should be 1
            break;
          }
          case 2: {             
            L.threadsPerBlock = dim3(transform_size/ept, 1, 1); 
            L.gridDims = dim3(transform_divisor, dims_out.w, 1);
            L.mem_offsets.pixel_pitch_input = dims_out.y;
            L.mem_offsets.pixel_pitch_output = dims_out.y;
            L.twiddle_in = -2*PIf/dims_out.y;
            L.Q = dims_out.y / dims_in.y; // should be 1
            break;
          }
          case 3: {
            // Not implemented
            std::cerr << "3d c2c not implemented" << std::endl;
            exit(-1);
            break;
          }
        } // end switch on transform dimension
        L.mem_offsets.shared_input = 0;
        L.mem_offsets.shared_output = 0;  
        // If inverse we need to negate the twidddle factor.
        if ( ! do_forward_transform)  L.twiddle_in  = -L.twiddle_in ; 
        break; // case c2c

      case c2c_decomposed: 
        L.threadsPerBlock = dim3(transform_divisor, 1, 1);
        L.Q =  transform_divisor; //  (dims_in / transform size)
        switch (transform_dimension)
        {
          case 1: {
            L.gridDims = dim3(1, 1, 1); 
            L.mem_offsets.shared_input = 0;
            L.mem_offsets.shared_output = dims_out.x; 
            L.mem_offsets.pixel_pitch_input = dims_out.x; // scalar type, natural 
            L.mem_offsets.pixel_pitch_output = dims_out.x; // complex type
            L.twiddle_in = 2*PIf/dims_out.x;            
            break;
          }
          case 2: {
            L.gridDims = dim3(1, dims_out.w, 1); 
            L.mem_offsets.shared_input = 0;
            L.mem_offsets.shared_output = dims_out.y; 
            L.mem_offsets.pixel_pitch_input = dims_out.y; // scalar type, natural 
            L.mem_offsets.pixel_pitch_output = dims_out.y; // complex type
            L.twiddle_in = 2*PIf/dims_out.y;
            break;
          }
          case 3: {
            // Not implemented
            std::cerr << "3d c2c_decomposed not implemented" << std::endl;
            exit(-1);
            break;
          }

        } // end switch on transform dimension
        if ( ! do_forward_transform) L.twiddle_in = - L.twiddle_in ;
      break;
      
      case c2r_transposed:
        if (transform_dimension == 1) { std::cerr << "c2r_transposed is not supported for 1d transforms." << std::endl; exit(-1); }

        L.twiddle_in = 2*PIf/dims_out.y;
        L.Q = 1; // Already full size - FIXME when working out limited number of output pixels  
        L.threadsPerBlock = dim3(transform_size/ept, 1, 1); 
        L.gridDims = dim3(transform_divisor, dims_out.y, 1);
        L.mem_offsets.shared_input = 0;
        L.mem_offsets.shared_output = 0;
        L.mem_offsets.pixel_pitch_input = dims_out.y;
        L.mem_offsets.pixel_pitch_output = dims_out.w*2;      
        break;

      case c2r_decomposed:
        if (transform_dimension == 2 || transform_dimension == 3) { std::cerr << "c2r_decomposed is not supported for transposed xforms, implied by 2d/3d." << std::endl; exit(-1); }

        L.twiddle_in = 2*PIf/dims_out.x;
        L.Q = transform_divisor; //  (dims_in / transform size)
        L.threadsPerBlock = dim3(transform_divisor, 1, 1); 
        L.gridDims = dim3(1, dims_out.y, 1);
        L.mem_offsets.shared_input = 0;
        L.mem_offsets.shared_output = dims_out.x;
        L.mem_offsets.pixel_pitch_input = dims_out.w;
        L.mem_offsets.pixel_pitch_output = dims_out.w*2;      
      break; 
      
      case c2r_decomposed_transposed:
        if (transform_dimension == 1) { std::cerr << "c2r_decomposed_transposed is not supported for 1d transforms." << std::endl; exit(-1); }
        L.twiddle_in = 2*PIf/dims_out.x;
        L.Q = transform_divisor; //  (dims_in / transform size)
        L.threadsPerBlock = dim3(transform_divisor, 1, 1); 
        L.gridDims = dim3(1, dims_out.y, 1);
        L.mem_offsets.shared_input = 0;
        L.mem_offsets.shared_output = dims_out.x;
        L.mem_offsets.pixel_pitch_input = dims_out.y;
        L.mem_offsets.pixel_pitch_output = dims_out.w*2;      
      break; 

      case xcorr_transposed:
        if (transform_dimension == 1 || transform_dimension == 3) { std::cerr << "xcorr_transposed is not supported for 1d/3d yet." << std::endl; exit(-1); } // FIXME

      // Cross correlation case
      // The added complexity, in instructions and shared memory usage outweigh the cost of just running the full length C2C on the forward.
        L.threadsPerBlock = dim3(transform_size/ept, 1, 1); 
        L.gridDims = dim3(transform_divisor, dims_out.w, 1);
        L.mem_offsets.shared_input = dims_in.y;
        L.mem_offsets.shared_output = dims_out.y;
        L.mem_offsets.pixel_pitch_input = dims_out.y;
        L.mem_offsets.pixel_pitch_output = dims_out.y;

        L.twiddle_in = -2*PIf/dims_out.y;
        L.Q = dims_out.y / dims_in.y; // FIXME assuming for now this is already divisible

        break;

      case xcorr_decomposed:
        if (transform_dimension == 1 || transform_dimension == 3) { std::cerr << "xcorr_decomposed is not supported for 1d/3d yet." << std::endl; exit(-1); } // FIXME

        L.threadsPerBlock = dim3(transform_divisor, 1, 1);
        L.Q =  transform_divisor; //  (dims_in / transform size)

        L.gridDims = dim3(1, dims_out.w, 1); 
        L.mem_offsets.shared_input = 0;
        L.mem_offsets.shared_output = dims_out.y; 
        L.mem_offsets.pixel_pitch_input = dims_out.y; // scalar type, natural 
        L.mem_offsets.pixel_pitch_output = dims_out.y; // complex type
        L.twiddle_in = -2*PIf/dims_out.y; // this is negated on the inverse xform in the kernel
        break;
       
      default:
        std::cerr << "ERROR: Unrecognized fft_status" << std::endl;
        exit(-1);
        
    }
    return L;
  }

  inline int ReturnPaddedMemorySize(short4 & wanted_dims)
  {
    // Assumes a) SetInputDimensionsAndType has been called and is_fftw_padded is set before this call. (Currently RuntimeAssert to die if false) FIXME
    int wanted_memory = 0;

    if (is_real_valued_input)
    {
      if (wanted_dims.x % 2 == 0) { padding_jump_val = 2; wanted_memory = wanted_dims.x / 2 + 1;}
      else { padding_jump_val = 1 ; wanted_memory = (wanted_dims.x - 1) / 2 + 1;}
    
      wanted_memory *= wanted_dims.y * wanted_dims.z; // other dimensions
      wanted_memory *= 2; // room for complex
      wanted_dims.w = (wanted_dims.x + padding_jump_val) / 2; // number of complex elements in the X dimesnions after FFT.
      compute_memory_allocated = 2 * wanted_memory; // scaling by 2 making room for the buffer.
    }
    else
    {    
      wanted_memory = wanted_dims.x * wanted_dims.y * wanted_dims.z;
      wanted_dims.w = wanted_dims.x; // pitch is constant
      // We allocate using sizeof(ComputeType) which is either __half or float, so we need an extra factor of 2
      // Note: this must be considered when setting the address of the buffer memory based on the address of the regular memory.
      compute_memory_allocated = 4 * wanted_memory; 
    }
    return wanted_memory;
  }

  void FFT_R2C_decomposed(bool transpose_output = true);
  void FFT_R2C(bool transpose_output = true); // non-transposed is not implemented and will fail at runtime.
  void FFT_R2C_WithPadding(bool transpose_output = true) ;// non-transposed is not implemented and will fail at runtime.
  void FFT_C2C_WithPadding(bool swap_real_space_quadrants = false);
  void FFT_C2C( bool do_forward_transform );
  void FFT_C2C_decomposed( bool do_forward_transform );
  void FFT_C2R_Transposed();
  void FFT_C2R_decomposed(bool transpose_output = true);


  void FFT_C2C_WithPadding_ConjMul_C2C(float2* image_to_search, bool swap_real_space_quadrants = false);
  void FFT_C2C_decomposed_ConjMul_C2C(float2* image_to_search, bool swap_real_space_quadrants = false);


  template<class FFT> void FFT_R2C_decomposed_t(bool transpose_output);
  template<class FFT> void FFT_R2C_t(bool transpose_output);
  template<class FFT> void FFT_R2C_WithPadding_t(bool transpose_output);
  template<class FFT> void FFT_C2C_WithPadding_t(bool swap_real_space_quadrants);
  template<class FFT> void FFT_C2C_t( bool do_forward_transform );
  template<class FFT> void FFT_C2C_decomposed_t( bool do_forward_transform );
  template<class FFT> void FFT_C2R_Transposed_t();
  template<class FFT> void FFT_C2R_decomposed_t(bool transpose_output);


  template<class FFT, class invFFT> void FFT_C2C_WithPadding_ConjMul_C2C_t(float2* image_to_search, bool swap_real_space_quadrants);
  template<class FFT, class invFFT> void FFT_C2C_decomposed_ConjMul_C2C_t(float2* image_to_search, bool swap_real_space_quadrants);

  // 1. 
  // First call passed from a public transform function, selects block or thread and the transform precision.
  void SetPrecisionAndExectutionMethod(KernelType kernel_type, bool do_forward_transform, bool use_thread_method);

  // 2.
  // Second call, sets size of the transform kernel, selects the appropriate GPU arch
  template <class FFT_base>
  void SelectSizeAndType(KernelType kernel_type, bool do_forward_transform);

  // 3.
  // Third call, sets the input and output dimensions and type
  template <class FFT_base_arch>
  void SetAndLaunchKernel(KernelType kernel_type, bool do_forward_transform);



};




} // namespace fast_FFT



#endif
