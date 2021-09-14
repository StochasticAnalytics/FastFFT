// Utilites for FastFFT.cu that we don't need the host to know about (FastFFT.h)
#include "FastFFT.h"

#ifndef Fast_FFT_cuh_
#define Fast_FFT_cuh_

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
#define cudaErr
#define precheck
#else
#define postcheck { cudaErr(cudaPeekAtLastError()); cudaError_t error = cudaStreamSynchronize(cudaStreamPerThread); cudaErr(error); };
#define cudaErr(error) { auto status = static_cast<cudaError_t>(error); if (status != cudaSuccess) { std::cerr << cudaGetErrorString(status) << " :-> "; MyFFTPrintWithDetails("");} };
#define precheck { cudaErr(cudaGetLastError()); }
#endif

inline void checkCudaErr(cudaError_t err) 
{ 
  if (err != cudaSuccess) 
  { 
    std::cerr << cudaGetErrorString(err) << " :-> " << std::endl;
    MyFFTPrintWithDetails(" ");
  } 
};

#define USEFASTSINCOS
// The __sincosf doesn't appear to be the problem with accuracy, likely just the extra additions, but it probably also is less flexible with other types. I don't see a half precision equivalent.
#ifdef USEFASTSINCOS
__device__ __forceinline__ void SINCOS(float arg, float* s, float* c)
{
  __sincosf(arg ,s,c);
}
#else
__device__ __forceinline__ void SINCOS(float arg, float* s, float* c)
{
  sincos(arg ,s,c);
}
#endif


namespace FastFFT {

  __device__ __forceinline__ int
  d_ReturnReal1DAddressFromPhysicalCoord(int3 coords, short4 img_dims)
  {
    return ( (((int)coords.z*(int)img_dims.y + coords.y) * (int)img_dims.w * 2)  + (int)coords.x) ;
  }

  // Complex a * conj b multiplication
  template <typename ComplexType, typename ScalarType>
  static __device__ __host__ inline auto ComplexConjMulAndScale(const ComplexType a, const ComplexType b, ScalarType s) -> decltype(b)
  {
      ComplexType c;
      c.x = s * (a.x * b.x + a.y * b.y);
      c.y = s * (a.y * b.x - a.x * b.y) ;
      return c;
  }

// GetCudaDeviceArch from https://github.com/mnicely/cufft_examples/blob/master/Common/cuda_helper.h
void GetCudaDeviceProps( DeviceProps& dp ) {
  int major;
  int minor;

  cudaErr( cudaGetDevice( &dp.device_id ) );
  cudaErr( cudaDeviceGetAttribute( &major, cudaDevAttrComputeCapabilityMajor, dp.device_id ) );
  cudaErr( cudaDeviceGetAttribute( &minor, cudaDevAttrComputeCapabilityMinor, dp.device_id ) );

  dp.device_arch = major * 100 + minor * 10;

  MyFFTRunTimeAssertTrue(dp.device_arch == 700 || dp.device_arch == 750 || dp.device_arch == 800, "FastFFT currently only supports compute capability [7.0, 7.5, 8.0].");


  cudaErr( cudaDeviceGetAttribute( &dp.max_shared_memory_per_block, cudaDevAttrMaxSharedMemoryPerBlock, dp.device_id ) );
  cudaErr( cudaDeviceGetAttribute( &dp.max_shared_memory_per_SM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dp.device_id) );
  cudaErr( cudaDeviceGetAttribute( &dp.max_registers_per_block, cudaDevAttrMaxRegistersPerBlock, dp.device_id ) );
  cudaErr( cudaDeviceGetAttribute( &dp.max_persisting_L2_cache_size, cudaDevAttrMaxPersistingL2CacheSize, dp.device_id) );
}
void CheckSharedMemory(int& memory_requested, DeviceProps& dp) {

  // Depends on GetCudaDeviceProps having been called, which should be happening in the constructor.
  // Throw an error if requesting more than allowed, otherwise, we'll set to requested and let the rest be L1 Cache.
  MyFFTRunTimeAssertFalse(memory_requested > dp.max_shared_memory_per_SM, "The shared memory requested is greater than permitted for this arch.") 
  // if (memory_requested > dp.max_shared_memory_per_block) { memory_requested = dp.max_shared_memory_per_block; }
}

//////////////////////
// Base FFT kerenel types, direction (r2c, c2r, c2c) and direction are ommited, to be applied in the method calling afull kernel
using namespace cufftdx;

// TODO this probably needs to depend on the size of the xform, at least small vs large.
constexpr const int elements_per_thread_real = 8;
constexpr const int elements_per_thread_complex = 8;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// FFT kernels
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////
// Block FFT based Kernel definitions
//////////////////////////////

/////////////
// R2C
/////////////


template<class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_R2C_NONE(const ScalarType*  __restrict__ input_values, ComplexType* __restrict__  output_values, Offsets mem_offsets, typename FFT::workspace_type workspace);

template<class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_R2C_INCREASE(const ScalarType*  __restrict__ input_values, ComplexType* __restrict__  output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace);


/////////////
// C2C
/////////////

template<class FFT, class ComplexType = typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_INCREASE(const ComplexType* __restrict__  input_values, ComplexType*  __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace);

template<class FFT, class ComplexType = typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_WithPadding_SwapRealSpaceQuadrants(const ComplexType* __restrict__  input_values, ComplexType* __restrict__  output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace);

template<class FFT, class invFFT, class ComplexType = typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_WithPadding_ConjMul_C2C( const ComplexType* __restrict__ image_to_search, const ComplexType* __restrict__  input_values, ComplexType* __restrict__  output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace_fwd, typename invFFT::workspace_type workspace_inv);

template<class FFT, class invFFT, class ComplexType = typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_WithPadding_ConjMul_C2C_SwapRealSpaceQuadrants( const ComplexType* __restrict__ image_to_search, const ComplexType* __restrict__  input_values, ComplexType* __restrict__  output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace_fwd, typename invFFT::workspace_type workspace_inv);

template<class FFT, class ComplexType = typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_NONE(const ComplexType* __restrict__  input_values, ComplexType* __restrict__  output_values, Offsets mem_offsets, typename FFT::workspace_type workspace);

/////////////
// C2R 
/////////////


template<class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2R_NONE(const ComplexType*  __restrict__ input_values, ScalarType*  __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace);



//////////////////////////////
// Thread FFT based Kernel definitions
//////////////////////////////

/////////////
// R2C
/////////////

template<class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__global__
void thread_fft_kernel_R2C_decomposed(const ScalarType*  __restrict__ input_values, ComplexType* __restrict__  output_values, Offsets mem_offsets, float twiddle_in, int Q);

template<class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__global__
void thread_fft_kernel_R2C_decomposed_transposed(const ScalarType*  __restrict__ input_values, ComplexType* __restrict__  output_values, Offsets mem_offsets, float twiddle_in, int Q);



/////////////
// C2C
/////////////


template<class FFT, class ComplexType = typename FFT::value_type>
__global__
void thread_fft_kernel_C2C_decomposed(const ComplexType* __restrict__  input_values, ComplexType* __restrict__  output_values, Offsets mem_offsets, float twiddle_in, int Q);

template<class FFT, class invFFT, class ComplexType = typename FFT::value_type>
__global__
void thread_fft_kernel_C2C_decomposed_ConjMul(const ComplexType* __restrict__ image_to_search, const ComplexType* __restrict__  input_values, ComplexType* __restrict__  output_values, Offsets mem_offsets, float twiddle_in, int Q);

/////////////
// C2R 
/////////////

template<class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__global__
void thread_fft_kernel_C2R_decomposed(const ComplexType*  __restrict__ input_values, ScalarType*  __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q);

template<class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__global__
void thread_fft_kernel_C2R_decomposed_transposed(const ComplexType*  __restrict__ input_values, ScalarType*  __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// End FFT Kernels
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



template<class InputType, class OutputType> 
 __global__ void clip_into_top_left_kernel( InputType*  input_values, OutputType*  output_values, const short4 dims );

 // Modified from GpuImage::ClipIntoRealKernel
template<typename InputType, typename OutputType>
__global__ void clip_into_real_kernel(InputType* real_values_gpu,
                                      OutputType* other_image_real_values_gpu,
                                      short4 dims, 
                                      short4 other_dims,
                                      int3 wanted_coordinate_of_box_center, 
                                      OutputType wanted_padding_value);

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

  static inline __device__ void load_shared(const complex_type* input,
                                            complex_type*       shared_input,
                                            complex_type*       thread_data,
                                            float* 	            twiddle_factor_args,
                                            float				        twiddle_in,
                                            int*				        input_map,
                                            int*				        output_map,
                                            int				          Q,
                                            int                 number_of_elements)
  {
    const unsigned int stride = stride_size();
    unsigned int       index  =  threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++)
    {
      if (index < number_of_elements)
      {
        input_map[i] = index;
        output_map[i] = Q*index;
        twiddle_factor_args[i] = twiddle_in * index;
        thread_data[i] = input[index];
        shared_input[index] = thread_data[i];
        index += stride;
      }
      else
      {
        input_map[i] = -9999; // ignore this in subsequent ops
      }
    }

  } // load_shared, MORE THREADS THAN ELEMENTS

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

  static inline __device__ void load_shared_and_conj_multiply(const complex_type*  image_to_search,
                                                              complex_type*  thread_data)
  {
    const unsigned int stride = stride_size();
    unsigned int       index  = threadIdx.x;
    complex_type c;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++)
    {
      c.x =  (thread_data[i].x * image_to_search[index].x + thread_data[i].y * image_to_search[index].y);
      c.y =  (thread_data[i].y * image_to_search[index].x - thread_data[i].x * image_to_search[index].y) ;
      // a * conj b
      thread_data[i] = c;//ComplexConjMulAndScale<complex_type, scalar_type>(thread_data[i], image_to_search[index], 1.0f);
      index += stride;
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

  static inline __device__ void store_r2c_transposed(const complex_type* thread_data,
                                                      complex_type*       output,
                                                      int*	              output_MAP,
                                                      int                 pixel_pitch,
                                                      int                 memory_limit) 
  {
    const unsigned int stride = stride_size();
    for (unsigned int i = 0; i <= FFT::elements_per_thread / 2; i++) 
    {
      // output map is thread local, so output_MAP[i] gives the x-index in the non-transposed array and blockIdx.y gives the y-index
      // if (blockIdx.y == 1) printf("index, pitch, blcok, address %i, %i, %i, %i\n", output_MAP[i], pixel_pitch, memory_limit, output_MAP[i]*pixel_pitch + blockIdx.y);

      if (output_MAP[i] < memory_limit) output[output_MAP[i]*pixel_pitch + blockIdx.y] = thread_data[i];
      // if (blockIdx.y == 32) printf("from store transposed %i , val %f %f\n", output_MAP[i], thread_data[i].x, thread_data[i].y);

    }
    // constexpr unsigned int threads_per_fft        = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
    // constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
    // constexpr unsigned int values_left_to_store = threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
    // if (threadIdx.x < values_left_to_store)
    // {
    //   printf("index, pitch, blcok, address %i, %i, %i, %i\n", output_MAP[FFT::elements_per_thread / 2], pixel_pitch, blockIdx.y, output_MAP[FFT::elements_per_thread / 2]*pixel_pitch + blockIdx.y);
    //   if (output_MAP[FFT::elements_per_thread / 2] < memory_limit) output[output_MAP[FFT::elements_per_thread / 2]*pixel_pitch + blockIdx.y] =  thread_data[FFT::elements_per_thread / 2];
    // }
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

    // this may benefit from asynchronous execution
    static inline __device__ void load(const complex_type* input,
                                       complex_type*       thread_data,
                                       int	               last_index_to_load) 
  {
    const unsigned int stride = stride_size();
    unsigned int       index  = threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
      if (index < last_index_to_load) thread_data[i] = input[index];
      else {thread_data[i].x = scalar_type(0); thread_data[i].y = scalar_type(0); } 
      index += stride;
    }
  }
  

  static inline __device__ void store_and_swap_quadrants(const complex_type* thread_data,
                                                         complex_type*       output,
                                                         int				         first_negative_index) 
  {
    const unsigned int  stride = stride_size();
    unsigned int        index  = threadIdx.x;
    complex_type phase_shift;
    int logical_y;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
      // If no kernel based changes are made to source_idx, this will be the same as the original index value
      phase_shift = thread_data[i];
      logical_y = index;
      if ( logical_y >= first_negative_index) logical_y -= 2*first_negative_index;
      if ( (int(blockIdx.y) + logical_y) % 2 != 0) phase_shift *= -1.f; 
      output[index] = phase_shift;
      index += stride;
    }
  } // store_and_swap_quadrants

  static inline __device__ void store_and_swap_quadrants(const complex_type* thread_data,
                                 complex_type*       output,
                                 int*				         source_idx,
                                 int				         first_negative_index) 
  {
    const unsigned int  stride = stride_size();
    complex_type phase_shift;
    int logical_y;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
      // If no kernel based changes are made to source_idx, this will be the same as the original index value
      phase_shift = thread_data[i];
      logical_y = source_idx[i];
      if ( logical_y >= first_negative_index) logical_y -= 2*first_negative_index;
      if ( (int(blockIdx.y) + logical_y) % 2 != 0) phase_shift *= -1.f; 
      output[source_idx[i]] = phase_shift;
    }
  } // store_and_swap_quadrants



  static inline __device__ void store(const complex_type* thread_data,
                                      complex_type*       output) 
  {
    const unsigned int  stride = stride_size();
    unsigned int       index  = threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {

      output[index] = thread_data[i];

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

  static inline __device__ void store_subset(const complex_type* thread_data,
                                      complex_type*       output,
                                      int*				        source_idx)                                    
  {
    const unsigned int  stride = stride_size();
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
      // If no kernel based changes are made to source_idx, this will be the same as the original index value
      if (source_idx[i] >= 0) output[source_idx[i]] = thread_data[i];
    }

  } // store, MORE THREADS THAN ELEMENTS

  static inline __device__ void store_coalesced(const complex_type* shared_output,
                                                complex_type*       global_output,
                                                int 				        offset)
  {

    const unsigned int stride = stride_size();
    unsigned int       index  =  offset + threadIdx.x;
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


template<class FFT>
struct io_thread 
{
  using complex_type = typename FFT::value_type;
  using scalar_type  = typename complex_type::value_type;


  static inline __device__ void load_r2c(const scalar_type* input,
                                         complex_type*      thread_data,
                                         const int          stride)  
  {
    unsigned int index  = threadIdx.x;
    for (unsigned int i = 0; i < size_of<FFT>::value; i++) 
    {
      thread_data[i].x = input[index];
      thread_data[i].y = scalar_type(0);
      index += stride;
    }
  } // load_r2c


  static inline __device__ void store_r2c(const complex_type* shared_output,
                                          complex_type*       output,
                                          const int           stride,
                                          const int           memory_limit)   
  {
    // Each thread reads in the input data at stride = mem_offsets.Q
    unsigned int index  = threadIdx.x;
    for (unsigned int i = 0; i < size_of<FFT>::value/2; i++) 
    {
      output[index] = shared_output[index];
      index += stride;
    }
    if (index < memory_limit)
    {
      output[index] = shared_output[index];
    }
  } // store_r2c

  static inline __device__ void store_r2c_transposed(const complex_type* shared_output,
                                                     complex_type*       output,
                                                     int                 stride,
                                                     int                 pixel_pitch,
                                                     int                 memory_limit)   
  {
    // Each thread reads in the input data at stride = mem_offsets.Q
    unsigned int index  = threadIdx.x;
    for (unsigned int i = 0; i < size_of<FFT>::value/2; i++) 
    {
      output[index*pixel_pitch] = shared_output[index];
      index += stride;
    }
    if (index < memory_limit)
    {
      output[index*pixel_pitch] = shared_output[index];
    }
  } // store_r2c_transposed


  static inline __device__ void remap_decomposed_segments(const complex_type* thread_data,
                                                          complex_type*       shared_output,
                                                          float               twiddle_in,
                                                          int                 Q,
                                                          int                 memory_limit)  // mem_offsets.pixel_pitch_output
  {
     // Unroll the first loop and initialize the shared mem. 
     complex_type twiddle;
     int index = threadIdx.x * size_of<FFT>::value;
     twiddle_in *= threadIdx.x; // twiddle factor arg now just needs to multiplied by K = (index + i) 
     for (unsigned int i = 0; i < size_of<FFT>::value; i++)
     {
       SINCOS( twiddle_in * (index + i) ,&twiddle.y,&twiddle.x);
       twiddle *= thread_data[i];
       if (index + i < memory_limit) shared_output[index +  i] = twiddle;
     } 
     __syncthreads(); // make sure all the shared mem is initialized to the starting value. There should be no contention as every thread is working on its own block of memory. 

     for (unsigned int sub_fft = 1; sub_fft < Q; sub_fft++)
     {
       // wrap around, 0 --> 1, Q-1 --> 0 etc.
       index = ((threadIdx.x + sub_fft) % Q) * size_of<FFT>::value;
       for (unsigned int i = 0; i < FFT::elements_per_thread; i++)
       {
         SINCOS( twiddle_in * (index + i) ,&twiddle.y,&twiddle.x);
         twiddle *= thread_data[i];
         if (index + i < memory_limit) 
         {
           atomicAdd_block(&shared_output[index +  i].x, twiddle.x);
           atomicAdd_block(&shared_output[index +  i].y, twiddle.y);
         }
       }
     }
     __syncthreads();
  }  // remap_decomposed_segments (r2c specialized with memory limit)                                              


  static inline __device__ void load_c2c(const complex_type* input,
                                         complex_type*      thread_data,
                                         const int          stride)  
  {
    unsigned int index  = threadIdx.x;
    for (unsigned int i = 0; i < size_of<FFT>::value; i++) 
    {
      thread_data[i] = input[index];
      index += stride;
    }
  } // load_c2c


  static inline __device__ void store_c2c(const complex_type* shared_output,
                                          complex_type*       output,
                                          const int           stride)   
  {
    // Each thread reads in the input data at stride = mem_offsets.Q
    unsigned int index  = threadIdx.x;
    for (unsigned int i = 0; i < size_of<FFT>::value; i++) 
    {
      output[index] = shared_output[index];
      index += stride;
    }
  } // store_c2c

  static inline __device__ void remap_decomposed_segments(const complex_type* thread_data,
                                                          complex_type*       shared_output,
                                                          float               twiddle_in,
                                                          int                 Q)  
  {
    // Unroll the first loop and initialize the shared mem. 
    complex_type twiddle;
    int index = threadIdx.x * size_of<FFT>::value;
    twiddle_in *= threadIdx.x; // twiddle factor arg now just needs to multiplied by K = (index + i) 
    for (unsigned int i = 0; i < size_of<FFT>::value; i++)
    {
      SINCOS( twiddle_in * (index + i) ,&twiddle.y,&twiddle.x);
      twiddle *= thread_data[i];
      shared_output[index +  i] = twiddle;
    }  
    __syncthreads(); // make sure all the shared mem is initialized to the starting value. There should be no contention as every thread is working on its own block of memory. 

    for (unsigned int sub_fft = 1; sub_fft < Q; sub_fft++)
    {
      // wrap around, 0 --> 1, Q-1 --> 0 etc.
      index = ((threadIdx.x + sub_fft) % Q) * size_of<FFT>::value;
      for (unsigned int i = 0; i < FFT::elements_per_thread; i++)
      {
        SINCOS( twiddle_in * (index + i) ,&twiddle.y,&twiddle.x);
        twiddle *= thread_data[i];
        atomicAdd_block(&shared_output[index +  i].x, twiddle.x);
        atomicAdd_block(&shared_output[index +  i].y, twiddle.y);
      }
    }
    __syncthreads();
  }  // remap_decomposed_segments (c2c specialized no explicit memory checks)  

 
  static inline __device__ void load_c2r(const complex_type* input,
                                         complex_type*       thread_data,
                                         const int           stride,
                                         const int           memory_limit)   
  {
    // Each thread reads in the input data at stride = mem_offsets.Q
    unsigned int index  = threadIdx.x;
    unsigned int offset = 2*memory_limit - 2;
    for (unsigned int i = 0; i < size_of<FFT>::value; i++) 
    {
      if (index <  memory_limit)
      {
        thread_data[i] = input[index];
      }
      else
      {
        // assuming even dimension
        // FIXME shouldn't need to read in from global for an even stride
        thread_data[i] = input[offset - index];
        thread_data[i].y = -thread_data[i].y; // conjugate
      }
      index += stride;


    }
  } // store_r2c_transposed

  // FIXME as above
  static inline __device__ void load_c2r_transposed(const complex_type* input,
                                                    complex_type*       thread_data,
                                                    int                 stride,
                                                    int                 pixel_pitch,
                                                    int                 memory_limit)   
  {
    // Each thread reads in the input data at stride = mem_offsets.Q
    unsigned int index  = threadIdx.x;
    // unsigned int offset = 2*memory_limit - 2;
    for (unsigned int i = 0; i < size_of<FFT>::value; i++) 
    {
      if (index <  memory_limit)
      {
        thread_data[i] = input[index*pixel_pitch];
      }
      else
      {
        // input[2*memory_limit - index - 2];
        // assuming even dimension
        // FIXME shouldn't need to read in from global for an even stride
        thread_data[i] = input[(2*memory_limit - index)*pixel_pitch];
        thread_data[i].y = -thread_data[i].y; // conjugate
      }
      index += stride;
    }
  } // load c2r_transposed

  static inline __device__ void remap_decomposed_segments_c2r(const complex_type* thread_data,
                                                              scalar_type*       shared_output,
                                                              scalar_type         twiddle_in,
                                                              int                 Q)  
  {
    // Unroll the first loop and initialize the shared mem. 
    complex_type twiddle;
    int index = threadIdx.x * size_of<FFT>::value;
    twiddle_in *= threadIdx.x; // twiddle factor arg now just needs to multiplied by K = (index + i) 
    for (unsigned int i = 0; i < size_of<FFT>::value; i++)
    {
      // printf("remap tid, index + i %i %i\n", threadIdx.x, index+i);
      SINCOS( twiddle_in * (index + i) ,&twiddle.y,&twiddle.x);
      shared_output[index +  i] = (twiddle.x*thread_data[i].x - twiddle.y*thread_data[i].y); // assuming the output is real, only the real parts add, so don't bother with the complex
    }  
    __syncthreads(); // make sure all the shared mem is initialized to the starting value. There should be no contention as every thread is working on its own block of memory. 

    for (unsigned int sub_fft = 1; sub_fft < Q; sub_fft++)
    {
      // wrap around, 0 --> 1, Q-1 --> 0 etc.
      index = ((threadIdx.x + sub_fft) % Q) * size_of<FFT>::value;

      for (unsigned int i = 0; i < size_of<FFT>::value; i++)
      {
        // if (threadIdx.x == 32) printf("remap tid, subfft, q, index + i %i %i %i %i\n", threadIdx.x,sub_fft, Q, index+i);

      SINCOS( twiddle_in * (index + i) ,&twiddle.y,&twiddle.x);
      atomicAdd_block(&shared_output[index +  i], twiddle.x*thread_data[i].x - twiddle.y*thread_data[i].y);
      }
    }
    __syncthreads();
  }  // remap_decomposed_segments (c2c specialized no explicit memory checks)  

  static inline __device__ void store_c2r(const scalar_type* shared_output,
                                          scalar_type*       output,
                                          const int           stride)   
  {
    // Each thread reads in the input data at stride = mem_offsets.Q
    unsigned int index  = threadIdx.x;
    for (unsigned int i = 0; i < size_of<FFT>::value; i++) 
    {
      output[index] = shared_output[index];
      index += stride;
    }
  } // store_c2c


  static inline __device__ void load_shared_and_conj_multiply(const complex_type*  image_to_search,
                                                              const complex_type*  shared_mem,
                                                              complex_type*        thread_data,
                                                              const int            stride)
  {
    unsigned int       index  = threadIdx.x;
    complex_type c;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++)
    {
      c.x =  (shared_mem[index].x * image_to_search[index].x + shared_mem[index].y * image_to_search[index].y);
      c.y =  (shared_mem[index].y * image_to_search[index].x - shared_mem[index].x * image_to_search[index].y) ;
      // a * conj b
      thread_data[i] = c;//ComplexConjMulAndScale<complex_type, scalar_type>(thread_data[i], image_to_search[index], 1.0f);
      index += stride;
    }
    __syncthreads();
  } // load_shared_and_conj_multiply



}; // struct thread_io

} // namespace FastFFT


#endif // Fast_FFT_cuh_
