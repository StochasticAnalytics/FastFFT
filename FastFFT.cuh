// Utilites for FastFFT.cu that we don't need the host to know about (FastFFT.h)
#include "FastFFT.h"

// “This software contains source code provided by NVIDIA Corporation.” Much of it is modfied as noted at relevant function definitions.


//
// 0 - no checks
// 1 - basic checks without blocking
// 2 - full checks, including blocking


#define HEAVYERRORCHECKING_FFT 

// #ifdef DEBUG
#define MyFFTPrint(...)	{std::cerr << __VA_ARGS__  << std::endl;}
#define MyFFTPrintWithDetails(...)	{std::cerr << __VA_ARGS__  << " From: " << __FILE__  << " " << __LINE__  << " " << __PRETTY_FUNCTION__ << std::endl;}
#define MyFFTDebugAssertTrue(cond, msg, ...) {if ((cond) != true) { std::cerr << msg   << std::endl << " Failed Assert at "  << __FILE__  << " " << __LINE__  << " " << __PRETTY_FUNCTION__ << std::endl; exit(-1);}}
#define MyFFTDebugAssertFalse(cond, msg, ...) {if ((cond) == true) { std::cerr << msg  << std::endl << " Failed Assert at "  << __FILE__  << " " << __LINE__  << " " << __PRETTY_FUNCTION__ << std::endl; exit(-1);}}
#define MyFFTDebugAssertTestTrue(cond, msg, ...) {if ((cond) != true) { std::cerr <<  "    Test " << msg << " FAILED!"  << std::endl << "  at "  << __FILE__  << " " << __LINE__  << " " << __PRETTY_FUNCTION__ << std::endl; exit(-1);} else { std::cerr << "    Test " << msg << " passed!" << std::endl;}}
#define MyFFTDebugAssertTestFalse(cond, msg, ...) {if ((cond) == true) { std::cerr<<  "    Test " << msg << " FAILED!"  << std::endl  << " at "  << __FILE__  << " " << __LINE__  << " " << __PRETTY_FUNCTION__ << std::endl; exit(-1);} else { std::cerr << "    Test " << msg << " passed!" << std::endl;}}

#define MyFFTRunTimeAssertTrue(cond, msg, ...) {if ((cond) != true) { std::cerr << msg   << std::endl << " Failed Assert at "  << __FILE__  << " " << __LINE__  << " " << __PRETTY_FUNCTION__ << std::endl; exit(-1);}}
#define MyFFTRunTimeAssertFalse(cond, msg, ...) {if ((cond) == true) { std::cerr << msg  << std::endl << " Failed Assert at "  << __FILE__  << " " << __LINE__  << " " << __PRETTY_FUNCTION__ << std::endl; exit(-1);}}

// Note we are using std::cerr b/c the wxWidgets apps running in cisTEM are capturing std::cout
#ifndef HEAVYERRORCHECKING_FFT 
#define postcheck
#define cudaErr(...)
#define precheck
#else
#define postcheck { cudaError_t error = cudaStreamSynchronize(cudaStreamPerThread); if (error != cudaSuccess) { std::cerr << cudaGetErrorString(error) << std::endl; MyFFTPrintWithDetails("");} };
#define cudaErr(error) { auto status = static_cast<cudaError_t>(error); if (status != cudaSuccess) { std::cerr << cudaGetErrorString(status) << std::endl; MyFFTPrintWithDetails("");} };
#define precheck { cudaErr(cudaGetLastError()); }
#endif


namespace FastFFT {

// GetCudaDeviceArch from https://github.com/mnicely/cufft_examples/blob/master/Common/cuda_helper.h
void GetCudaDeviceArch( int &device, int &arch ) {
  int major;
  int minor;
  cudaErr( cudaGetDevice( &device ) );

  cudaErr( cudaDeviceGetAttribute( &major, cudaDevAttrComputeCapabilityMajor, device ) );
  cudaErr( cudaDeviceGetAttribute( &minor, cudaDevAttrComputeCapabilityMinor, device ) );

  arch = major * 100 + minor * 10;
}

//////////////////////
// Base FFT kerenel types, direction (r2c, c2r, c2c) and direction are ommited, to be applied in the method calling afull kernel
using namespace cufftdx;

constexpr const int elements_per_thread_real = 8;
constexpr const int elements_per_thread_complex = 8;
constexpr const uint device_arch = 700;

// All transforms are 
using FFT_base   = decltype(Block() + Precision<float>() + ElementsPerThread<elements_per_thread_complex>()  + FFTsPerBlock<1>()  );



//////////////////////////////
// Kernel definitions
//////////////////////////////

template<class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_R2C_Transposed(ScalarType* input_values, ComplexType* output_values, Offsets mem_offsets, typename FFT::workspace_type workspace);

template<class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_R2C_WithPadding_Transposed(ScalarType* input_values, ComplexType* output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace);

template<class FFT, class ComplexType = typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_WithPadding(ComplexType* input_values, ComplexType* output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace);

template<class FFT, class ComplexType = typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C(ComplexType* input_values, ComplexType* output_values, Offsets mem_offsets, typename FFT::workspace_type workspace);

template<class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2R_Transformed(ComplexType* input_values, ScalarType* output_values, Offsets mem_offsets, typename FFT::workspace_type workspace);



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
                                         complex_type*      thread_data) 
  {
    const unsigned int stride = stride_size();
    unsigned int       index  = threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
      thread_data[i].x = input[index];
      thread_data[i].y = 0.0f;
      index += stride;
    }
  } // load_r2c 

  static inline __device__ void store_r2c(const complex_type* thread_data,
                                          complex_type*       output,
                                          int                 offset) 
  {
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

  // Since we can make repeated use of the same shared memory for each sub-fft
  // we use this method to load into shared mem instead of directly to registers
  // TODO set this up for async mem load
  static inline __device__ void load_shared(const complex_type* input,
                                            complex_type*       shared_input,
                                            complex_type*       thread_data,
                                            float* 	            twiddle_factor_args,
                                            float				        twiddle_in,
                                            int*				        input_map,
                                            int*				        output_map,
                                            int				          Q)
  {
    const unsigned int stride = stride_size();
    unsigned int       index  =  threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++)
    {
      input_map[i] = index;
      output_map[i] = Q*index;
      twiddle_factor_args[i] = twiddle_in * index;
      thread_data[i] = input[index];
      shared_input[index] = thread_data[i];
      index += stride;
    }

  } // load_shared

  // Since we can make repeated use of the same shared memory for each sub-fft
  // we use this method to load into shared mem instead of directly to registers
  // TODO set this up for async mem load - alternatively, load to registers then copy but leave in register for firt compute
  static inline __device__ void load_r2c_shared(const scalar_type*  input,
                                                scalar_type*        shared_input,
                                                complex_type*       thread_data,
                                                float* 	            twiddle_factor_args,
                                                float				        twiddle_in,
                                                int*				        input_map,
                                                int*				        output_map,
                                                int				          Q) 
  {
    const unsigned int stride = stride_size();
    unsigned int       index  = threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
      // if (blockIdx.y == 0) ("blck %i index %i \n", Q*index, index);

      input_map[i] = index;
      output_map[i] = Q*index;
      twiddle_factor_args[i] = twiddle_in * index;
      thread_data[i].x = input[index];
      thread_data[i].y = 0.0f;
      shared_input[index] =  thread_data[i].x;
      index += stride;
    }

  } // load_r2c_shared}

  // when using load_shared || load_r2c_shared, we need then copy from shared mem into the registers.
  // notice we still need the packed complex values for the xform.
  static inline __device__ void copy_from_shared(const scalar_type* shared_input,
                                                 complex_type*		  thread_data,
                                                 int*			        	input_map)
  {
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++)
    {
      thread_data[i].x = shared_input[input_map[i]];
      thread_data[i].y = 0.0f;
    }
  } // copy_from_shared

  static inline __device__ void copy_from_shared(const complex_type* shared_input_complex,
                                                 complex_type*		   thread_data,
                                                 int*				         input_map)
  {
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++)
    {
      thread_data[i] = shared_input_complex[input_map[i]];
    }
  } // copy_from_shared

  static inline __device__ void store_r2c_transposed(const complex_type* thread_data,
                                                     complex_type*       output,
                                                     int                 pixel_pitch) 
  {
    const unsigned int stride = stride_size();
    unsigned int       index  = threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread / 2; i++) 
    {
      // output map is thread local, so output_MAP[i] gives the x-index in the non-transposed array and blockIdx.y gives the y-index
      output[index*pixel_pitch + blockIdx.y] = thread_data[i];
      index += stride;
    }
    constexpr unsigned int threads_per_fft        = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
    constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
    constexpr unsigned int values_left_to_store = threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
    if (threadIdx.x < values_left_to_store)
    {
     output[index*pixel_pitch + blockIdx.y] =  thread_data[FFT::elements_per_thread / 2];
    }
  } // store_r2c_transposed

  static inline __device__ void store_r2c_transposed(const complex_type* thread_data,
                                                     complex_type*       output,
                                                     int*	               output_MAP,
                                                     int                 pixel_pitch) 
  {
    const unsigned int stride = stride_size();
    for (unsigned int i = 0; i < FFT::elements_per_thread / 2; i++) 
    {
      // output map is thread local, so output_MAP[i] gives the x-index in the non-transposed array and blockIdx.y gives the y-index
      output[output_MAP[i]*pixel_pitch + blockIdx.y] = thread_data[i];
      // if (blockIdx.y == 32) printf("from store transposed %i , val %f %f\n", output_MAP[i], thread_data[i].x, thread_data[i].y);

    }
    constexpr unsigned int threads_per_fft        = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
    constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
    constexpr unsigned int values_left_to_store = threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
    if (threadIdx.x < values_left_to_store)
    {
      output[output_MAP[FFT::elements_per_thread / 2]*pixel_pitch + blockIdx.y] =  thread_data[FFT::elements_per_thread / 2];
    }
  } // store_r2c_transposed


  static inline __device__ void load_c2r_transposed(const complex_type* input,
                                                    complex_type*       thread_data,
                                                    int        		      pixel_pitch) 
  {
    const unsigned int stride = stride_size();
    unsigned int       index  =  threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread / 2; i++) 
    {
      thread_data[i] = input[pixel_pitch*(int)index];
      index += stride;
    }
    constexpr unsigned int threads_per_fft       = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
    constexpr unsigned int output_values_to_load = (cufftdx::size_of<FFT>::value / 2) + 1;
    // threads_per_fft == 1 means that EPT == SIZE, so we need to load one more element
    constexpr unsigned int values_left_to_load = threads_per_fft == 1 ? 1 : (output_values_to_load % threads_per_fft);
    if (threadIdx.x < values_left_to_load) 
    {
      thread_data[FFT::elements_per_thread / 2] = input[pixel_pitch*(int)index];
    }
  } // load_c2r_transposed

  // this may benefit from asynchronous execution
  static inline __device__ void load(const complex_type* input,
                                     complex_type*       thread_data) 
  {
    const unsigned int stride = stride_size();
    unsigned int       index  = threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
      thread_data[i] = input[index];
      // if (blockIdx.y == 0) printf("block %i , val %f %f\n", index, input[index].x, input[index].y);

      index += stride;
    }
  }
  
  static inline __device__ void store(const complex_type* thread_data,
                                      complex_type*       output) 
  {
    const unsigned int  stride = stride_size();
    unsigned int       index  = threadIdx.x;
    complex_type phase_shifted;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
      // If no kernel based changes are made to source_idx, this will be the same as the original index value
      // TEMP HACK TO Swap quadrants
      // phase_shift =  (index + blockIdx.y) % 2 == 0 ? 1.f : -1.f;
      phase_shifted = thread_data[i];
      if ((index + blockIdx.y) % 2 != 0) phase_shifted *= -1.f;

      output[index] = phase_shifted; //thread_data[i];
      // if (blockIdx.y == 1) printf("block iyt %i , val %f %f\n", index, thread_data[i].x,thread_data[i].y);

      index += stride;
    }
  } // store


  static inline __device__ void store(const complex_type* thread_data,
                                      complex_type*       output,
                                      int*				        source_idx) 
  {
    const unsigned int  stride = stride_size();
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
      // If no kernel based changes are made to source_idx, this will be the same as the original index value
      output[source_idx[i]] = thread_data[i];
    }
  } // store

  static inline __device__ void store(const complex_type* thread_data,
                                      complex_type*       output,
                                      int*				        source_idx,
                                      int	                output_stride,
                                      int				          memory_limit) 
  {
    //            const unsigned int offset = batch_offset(local_fft_id);
    const unsigned int stride = stride_size();
    //            unsigned int       index  = offset + threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
      // If no kernel based changes are made to source_idx, this will be the same as the original index value
      if (source_idx[i] < memory_limit) output[source_idx[i]*output_stride] = thread_data[i];
      //                output[index] = thread_data[i];
      //                index += stride;
    }
  } // store

  static inline __device__ void store_coalesced(const complex_type* shared_output,
                                                complex_type*       global_output,
                                                int 				        offset)
  {

    const unsigned int stride = stride_size();
    unsigned int       index  =  offset + threadIdx.x;
    unsigned int phase_shift;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++)
    {
      global_output[index] = shared_output[index] ;
      index += stride;
    }
  } // store_coalesced
//
  static inline __device__ void store_transposed(const complex_type* thread_data,
                                              complex_type*       output,
                                              int*				        output_map,
                                              int*                rotated_offset,
                                              int				          memory_limit) 
  {
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
      // If no kernel based changes are made to source_idx, this will be the same as the original index value
      if (output_map[i] < memory_limit) output[rotated_offset[1]*output_map[i] + rotated_offset[0]] = thread_data[i];
    }
  } // store_transposed

  static inline __device__ void store_c2r(const complex_type* thread_data,
                                          scalar_type*        output) 
  {
    const unsigned int stride = stride_size();
    unsigned int       index  = threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
      output[index] = reinterpret_cast<const scalar_type*>(thread_data)[i];
      index += stride;
    }
  }
}; // struct io}


} // namespace FastFFT


