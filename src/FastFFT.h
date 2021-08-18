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



class FourierTransformer 
  
{

public:

  // Used to specify input/calc/output data types
  enum DataType { int4_2, uint8, int8, uint16, int16, fp16, bf16, tf32, uint32, int32, fp32};
  enum OriginType { natural, centered, quadrant_swapped}; // Used to specify the origin of the data
  short  padding_jump_val;
  int input_memory_allocated;
  int output_memory_allocated;
  int input_number_of_real_values;
  int output_number_of_real_values;

  FourierTransformer(DataType wanted_calc_data_type);
  // FourierTransformer(const FourierTransformer &); // Copy constructor
  virtual ~FourierTransformer();

  // This is pretty similar to an FFT plan, I should probably make it align with CufftPlan
  void SetInputDimensionsAndType(size_t input_logical_x_dimension, 
                                size_t input_logical_y_dimension, 
                                size_t input_logical_z_dimension, 
                                bool is_padded_input, 
                                bool is_host_memory_pinned, 
                                DataType input_data_type,
                                OriginType input_origin_type);
  
  void SetOutputDimensionsAndType(size_t output_logical_x_dimension, 
                                  size_t output_logical_y_dimension, 
                                  size_t output_logical_z_dimension, 
                                  bool is_padded_output, 
                                  DataType output_data_type,
                                  OriginType output_origin_type);

  // For the time being, the caller is responsible for having the memory allocated for any of these input/output pointers.
  void SetInputPointer(float* input_pointer, bool is_input_on_device);

  void CopyHostToDevice();
  // By default we are blocking with a stream sync until complete for simplicity. This is overkill and should FIXME.
  void CopyDeviceToHost(bool free_gpu_memory, bool unpin_host_memory);
  // When the size changes, we need a new host pointer
  void CopyDeviceToHost(float* output_pointer, bool free_gpu_memory = true, bool unpin_host_memory = true);



  // FFT calls

  // 1:1 no resizing or anything fancy.


  void FwdFFT(bool swap_real_space_quadrants = false, bool transpose_output = true);
  void InvFFT(bool transpose_output = true);
  void CrossCorrelate(float2* image_to_search, bool swap_real_space_quadrants);
  void ClipIntoTopLeft();
  void ClipIntoReal(int wanted_coordinate_of_box_center_x, int wanted_coordinate_of_box_center_y, int wanted_coordinate_of_box_center_z);

  
  inline int ReturnPaddedMemorySize(short4 & wanted_dims) 
  {
    int wanted_memory = 0;
    if (wanted_dims.x % 2 == 0) { padding_jump_val = 2; wanted_memory = wanted_dims.x / 2 + 1;}
    else { padding_jump_val = 1 ; wanted_memory = (wanted_dims.x - 1) / 2 + 1;}

    wanted_memory *= wanted_dims.y * wanted_dims.z; // other dimensions
    wanted_memory *= 2; // room for complex
    wanted_dims.w = (wanted_dims.x + padding_jump_val) / 2; // number of complex elements in the X dimesnions after FFT.
    return wanted_memory;
  };

  template<typename T, bool is_on_host = true>
  void SetToConstant(T* input_pointer, int N_values, const T wanted_value)
  {
    if (is_on_host) 
    {
      for (int i = 0; i < N_values; i++)
      {
        input_pointer[i] = (T)wanted_value;
      }
    }
    else
    {
      exit(-1);
    }
  }

// TODO move back to private and provide method to return a pointer to the memory if requested.
    float* device_pointer_fp32; float2* device_pointer_fp32_complex;
  float* buffer_fp32; float2* buffer_fp32_complex;
  __half* device_pointer_fp16; __half2* device_pointer_fp16_complex;

private:


  DataType input_date_type;
  DataType calc_data_type;
  DataType output_data_type;

  OriginType input_origin_type;
  OriginType output_origin_type;

  // booleans to track state, could be bit fields but that seem opaque to me.
  bool is_in_memory_host_pointer;
  bool is_in_memory_device_pointer;
  bool is_in_buffer_memory;

  bool is_host_memory_pinned;

  bool is_fftw_padded_input;
  bool is_fftw_padded_output;
  bool is_fftw_padded_buffer;

  bool is_size_validated;
  int  transform_dimension;
  int  transform_size;
  int  transform_divisor;
  enum SizeChangeType { increase, decrease, none };
  SizeChangeType size_change_type;

  bool is_set_input_params;
  bool is_set_output_params;

  short4 dims_in;
  short4 dims_out;

  float* host_pointer;
  float* pinnedPtr;



  void Deallocate();
  void UnPinHostMemory();

  void SetDefaults();
  void CheckDimensions();

  inline void GetTransformSize(int input_dimension)
  {
    if ( abs(fmod(log2(float(input_dimension)), 1)) < 1e-6 ) 
    {
      transform_divisor = 1;
      transform_size = input_dimension;
      // TODO for larger sizes, below
      // transform_size = input_dimension / transform_divisor;
    }
    else 
    {
      if ( abs(fmod(log2(float(input_dimension)/3), 1)) < 1e-6) 
      {
        transform_divisor = 3;
        transform_size = input_dimension / transform_divisor;
      }
      else
      {
        std::cerr << "The input dimension must factor into powers of two, with at most one factor of three." << std::endl;
        exit(-1);
      }
    }
  };

  inline void GetTransformSize_thread(int input_dimension, int thread_fft_size)
  {
    if (input_dimension % thread_fft_size != 0) { std::cerr << "Thread based decompositions must factor by thread_fft_size (" << thread_fft_size << ") in the current implmentations." << std::endl; exit(-1); }
    transform_divisor = input_dimension / thread_fft_size;
    transform_size = thread_fft_size;
  };


  enum KernelType { r2c_decomposed, r2c_decomposed_transposed, r2c_transposed, c2c_padded, c2c, c2c_decomposed, c2r_transposed,
                    c2r_decomposed, c2r_decomposed_transposed,  xcorr_transposed}; // Used to specify the origin of the data
  inline LaunchParams SetLaunchParameters(const int& ept, KernelType kernel_type)
  {
    LaunchParams L;
    switch (kernel_type)
    {
      case r2c_decomposed: 

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
        // The only read from the input array is in this blcok
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
        L.threadsPerBlock = dim3(transform_size/ept, 1, 1); 
        L.gridDims = dim3(transform_divisor, dims_out.w, 1);
        L.mem_offsets.shared_input = dims_in.y;
        L.mem_offsets.shared_output = dims_out.y;
        L.mem_offsets.pixel_pitch_input = dims_out.y;
        L.mem_offsets.pixel_pitch_output = dims_out.y;

        L.twiddle_in = -2*PIf/dims_out.y;
        L.Q = dims_out.y / dims_in.y; // FIXME assuming for now this is already divisible

        break;
      case c2c:
        L.threadsPerBlock = dim3(transform_size/ept, 1, 1); 
        L.gridDims = dim3(transform_divisor, dims_out.w, 1);
        L.mem_offsets.shared_input = 0;
        L.mem_offsets.shared_output = 0;
        L.mem_offsets.pixel_pitch_input = dims_out.y;
        L.mem_offsets.pixel_pitch_output = dims_out.y;
        L.twiddle_in = -2*PIf/dims_out.y;
        L.Q = 1; // Already full size - FIXME when working out limited number of output pixels       
        break;

      case c2c_decomposed: 
        L.threadsPerBlock = dim3(transform_divisor, 1, 1);
        L.gridDims = dim3(1, dims_out.w, 1); 
        L.mem_offsets.shared_input = 0;
        L.mem_offsets.shared_output = dims_out.y; 
        L.mem_offsets.pixel_pitch_input = dims_out.y; // scalar type, natural 
        L.mem_offsets.pixel_pitch_output = dims_out.y; // complex type
        L.twiddle_in = -2*PIf/dims_out.y ;
        L.Q =  transform_divisor; //  (dims_in / transform size)
      break;
      
      case c2r_transposed:
        L.twiddle_in = -2*PIf/dims_out.y;
        L.Q = 1; // Already full size - FIXME when working out limited number of output pixels  
        L.threadsPerBlock = dim3(transform_size/ept, 1, 1); 
        L.gridDims = dim3(transform_divisor, dims_out.y, 1);
        L.mem_offsets.shared_input = 0;
        L.mem_offsets.shared_output = 0;
        L.mem_offsets.pixel_pitch_input = dims_out.y;
        L.mem_offsets.pixel_pitch_output = dims_out.w*2;      
        break;

      case c2r_decomposed:
        L.twiddle_in = -2*PIf/dims_out.x;
        L.Q = transform_divisor; //  (dims_in / transform size)
        L.threadsPerBlock = dim3(transform_divisor, 1, 1); 
        L.gridDims = dim3(1, dims_out.y, 1);
        L.mem_offsets.shared_input = 0;
        L.mem_offsets.shared_output = dims_out.x;
        L.mem_offsets.pixel_pitch_input = dims_out.w;
        L.mem_offsets.pixel_pitch_output = dims_out.w*2;      
      break; 
      
      case c2r_decomposed_transposed:
        L.twiddle_in = -2*PIf/dims_out.x;
        L.Q = 1; // Already full size - FIXME when working out limited number of output pixels  
        L.threadsPerBlock = dim3(transform_divisor, 1, 1); 
        L.gridDims = dim3(1, dims_out.y, 1);
        L.mem_offsets.shared_input = 0;
        L.mem_offsets.shared_output = dims_out.x;
        L.mem_offsets.pixel_pitch_input = dims_out.y;
        L.mem_offsets.pixel_pitch_output = dims_out.w*2;      
      break; 

      case xcorr_transposed:
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
      default:
        std::cerr << "ERROR: Unrecognized fft_status" << std::endl;
        exit(-1);
        
    }
    std::cout << "Transform Divisor " << L.Q << " " << transform_divisor << std::endl;
    return L;
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

  template<class FFT> void FFT_R2C_decomposed_t(bool transpose_output);
  template<class FFT> void FFT_R2C_t(bool transpose_output);
  template<class FFT> void FFT_R2C_WithPadding_t(bool transpose_output);
  template<class FFT> void FFT_C2C_WithPadding_t(bool swap_real_space_quadrants);
  template<class FFT> void FFT_C2C_t( bool do_forward_transform );
  template<class FFT> void FFT_C2C_decomposed_t( bool do_forward_transform );
  template<class FFT> void FFT_C2R_Transposed_t();
  template<class FFT> void FFT_C2R_decomposed_t(bool transpose_output);


  template<class FFT, class invFFT> void FFT_C2C_WithPadding_ConjMul_C2C_t(float2* image_to_search, bool swap_real_space_quadrants);



};




} // namespace fast_FFT



#endif
