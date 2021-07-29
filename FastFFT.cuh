// Utilites for FastFFT.cu that we don't need the host to know about (FastFFT.h)

// “This software contains source code provided by NVIDIA Corporation.” Much of it is modfied as noted at relevant function definitions.


//
// 0 - no checks
// 1 - basic checks without blocking
// 2 - full checks, including blocking


#define HEAVYERRORCHECKING_FFT 

// #ifdef DEBUG
#define MyFFTPrint(...)	{std::cerr << __VA_ARGS__ << sizeof(short) << std::endl;}
#define MyFFTPrintWithDetails(...)	{std::cerr << __VA_ARGS__  << " From: " << __FILE__  << " " << __LINE__  << " " << __PRETTY_FUNCTION__ << std::endl;}
#define MyFFTDebugAssertTrue(cond, msg, ...) {if ((cond) != true) { std::cerr << msg   << std::endl << " Failed Assert at "  << __FILE__  << " " << __LINE__  << " " << __PRETTY_FUNCTION__ << std::endl; exit(-1);}}
#define MyFFTDebugAssertFalse(cond, msg, ...) {if ((cond) == true) { std::cerr << msg  << std::endl << " Failed Assert at "  << __FILE__  << " " << __LINE__  << " " << __PRETTY_FUNCTION__ << std::endl; exit(-1);}}


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

//////////////////////
// Base FFT kerenel types, direction (r2c, c2r, c2c) and direction are ommited, to be applied in the method calling afull kernel
using namespace cufftdx;

constexpr const int elements_per_thread_real = 8;
constexpr const int elements_per_thread_complex = 8;

using FFT_64_fp32         = decltype(Block() + Size<64>()  + Precision<float>() + ElementsPerThread<elements_per_thread_real>() + FFTsPerBlock<1>() + SM<700>());
//////////////////////////////
// Kernel definitions
//////////////////////////////
template<class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void SimpleFFT_NoPaddingKernel(ScalarType * real_input, ComplexType* complex_output, short4 dims_in, short4 dims_out);



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
                                         complex_type*      thread_data,
                                         int       		      offset) 
  {
    const unsigned int stride = stride_size();
    unsigned int       index  = offset + threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
      reinterpret_cast<scalar_type*>(thread_data)[i] = input[index];
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
                                            float* 	            twiddle_factor_args,
                                            float				        twiddle_in,
                                            int*				        input_map,
                                            int*				        output_map,
                                            int				          Q,
                                            int       		      input_stride)
  {
    const unsigned int stride = stride_size();
    unsigned int       index  =  threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++)
    {
      input_map[i] = index;
      output_map[i] = Q*index;
      twiddle_factor_args[i] = twiddle_in * index;
      shared_input[index] = input[index*input_stride];
      index += stride;
    }

  } // load_shared

  // Since we can make repeated use of the same shared memory for each sub-fft
  // we use this method to load into shared mem instead of directly to registers
  // TODO set this up for async mem load - alternatively, load to registers then copy but leave in register for firt compute
  static inline __device__ void load_r2c_shared(const scalar_type*  input,
                                                scalar_type*        shared_input,
                                                float* 	            twiddle_factor_args,
                                                float				        twiddle_in,
                                                int*				        input_map,
                                                int*				        output_map,
                                                int				          Q,
                                                int       		      input_stride) 
  {
    const unsigned int stride = stride_size();
    unsigned int       index  = threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
      input_map[i] = index;
      output_map[i] = Q*index;
      twiddle_factor_args[i] = twiddle_in * index;
      shared_input[index] = input[index*input_stride];
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


  static inline __device__ void store_r2c_rotated(const complex_type* thread_data,
                                                  complex_type*       output,
                                                  int*	              rotated_offset) 
  {
    const unsigned int stride = stride_size();
    unsigned int       index  = threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread / 2; i++) 
    {
      output[rotated_offset[1]*(int)index + rotated_offset[0]] = thread_data[i];
      index += stride;
    }
    constexpr unsigned int threads_per_fft        = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
    constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
    constexpr unsigned int values_left_to_store = threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
    if (threadIdx.x < values_left_to_store)
    {
      output[rotated_offset[1]*(int)index + rotated_offset[0]] = thread_data[FFT::elements_per_thread / 2];
    }
  } // store_r2c_rotated


  static inline __device__ void load_c2r(const complex_type* input,
                                         complex_type*       thread_data,
                                         int                 offset,
                                         int*				         source_idx = NULL) 
  {
    const unsigned int stride = stride_size();
    unsigned int       index  = offset + threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread / 2; i++) 
    {
      thread_data[i] = input[index];
      if (source_idx != NULL) source_idx[i] = index;
      index += stride;
    }
    constexpr unsigned int threads_per_fft       = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
    constexpr unsigned int output_values_to_load = (cufftdx::size_of<FFT>::value / 2) + 1;
    // threads_per_fft == 1 means that EPT == SIZE, so we need to load one more element
    constexpr unsigned int values_left_to_load = threads_per_fft == 1 ? 1 : (output_values_to_load % threads_per_fft);
    if (threadIdx.x < values_left_to_load) 
    {
      thread_data[FFT::elements_per_thread / 2] = input[index];
      if (source_idx != NULL) source_idx[FFT::elements_per_thread / 2] = index;
    }
  } // load_c2r

  static inline __device__ void load_c2r_rotated(const complex_type* input,
                                                 complex_type*       thread_data,
                                                 int*        		     rotated_offset) 
  {
    const unsigned int stride = stride_size();
    unsigned int       index  =  threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread / 2; i++) 
    {
      thread_data[i] = input[rotated_offset[1]*(int)index + rotated_offset[0]];
      index += stride;
    }
    constexpr unsigned int threads_per_fft       = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
    constexpr unsigned int output_values_to_load = (cufftdx::size_of<FFT>::value / 2) + 1;
    // threads_per_fft == 1 means that EPT == SIZE, so we need to load one more element
    constexpr unsigned int values_left_to_load = threads_per_fft == 1 ? 1 : (output_values_to_load % threads_per_fft);
    if (threadIdx.x < values_left_to_load) 
    {
    thread_data[FFT::elements_per_thread / 2] = input[rotated_offset[1]*(int)index + rotated_offset[0]];
    }
  } // load_c2r_rotated

}; // struct io}


} // namespace FastFFT


