// Insert some license stuff here

#ifndef fast_FFT_H_
#define fast_FFT_H_



// #include "/groups/himesb/git/cufftdx/example/block_io.hpp"
// #include "/groups/himesb/git/cufftdx/example/common.hpp"
// #include <iostream>

namespace FastFFT {


class FourierTransformer 
  
{

public:

  // Used to specify input/calc/output data types
  enum DataType { int4_2, uint8, int8, uint16, int16, fp16, bf16, tf32, uint32, int32, fp32};
  enum OriginType { natural, centered, quadrant_swapped}; // Used to specify the origin of the data

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

  // void SetInputPointer(int16* input_pointer, bool is_input_on_device);
  void SetInputPointer(float* input_pointer, bool is_input_on_device);
  void CopyHostToDevice();
  void CopyDeviceToHost(bool free_gpu_memory, bool unpin_host_memory);
  void Deallocate();

  // FFT calls

  // 1:1 no resizing or anything fancy.
  void SimpleFFT_NoPadding();


  inline int ReturnPaddedMemorySize(short4 & wanted_dims) 
  {
    int wanted_memory = 0;
    wanted_dims.w = wanted_dims.x;
    if (wanted_dims.x % 2 == 0) { wanted_dims.w +=2; wanted_memory = wanted_dims.x / 2 + 1;}
    else { wanted_dims.w += 1 ; wanted_memory = (wanted_dims.x - 1) / 2 + 1;}

    wanted_memory *= wanted_dims.y * wanted_dims.z; // other dimensions
    wanted_memory *= 2; // room for complex
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
private:


  DataType input_date_type;
  DataType calc_data_type;
  DataType output_data_type;

  OriginType input_origin_type;
  OriginType output_origin_type;

  // booleans to track state, could be bit fields but that seem opaque to me.
  bool is_in_memory_host_pointer;
  bool is_in_memory_device_pointer;

  bool is_host_memory_pinned;

  bool is_fftw_padded_input;
  bool is_fftw_padded_output;
  bool is_fftw_padded_buffer;

  bool is_set_input_params;
  bool is_set_output_params;

  short4 dims_in;
  short4 dims_out;

  float* host_pointer;
  float* pinnedPtr;
  float* device_pointer_fp32; float2* device_pointer_fp32_complex;
  float* buffer_fp32; float2* buffer_fp32_complex;
  __half* device_pointer_fp16; __half2* device_pointer_fp16_complex;

  int input_memory_allocated;
  int output_memory_allocated;

  void SetDefaults();



};




} // namespace fast_FFT



#endif