// Utilites for FastFFT.cu that we don't need the host to know about (FastFFT.h)
#include "FastFFT.h"

#ifndef Fast_FFT_cuh_
#define Fast_FFT_cuh_

#include "cufftdx/include/cufftdx.hpp"

// “This software contains source code provided by NVIDIA Corporation.” Much of it is modfied as noted at relevant function definitions.

// When defined Turns on synchronization based checking for all FFT kernels as well as cudaErr macros
// Defined in the Makefile when DEBUG_STAGE is not equal 8 (the default, not partial transforms.)
// #define HEAVYERRORCHECKING_FFT

// Various levels of debuging conditions and prints
// #define FFT_DEBUG_LEVEL 0

// #define forceforce( type )  __nv_is_extended_device_lambda_closure_type( type )
//FIXME: change to constexpr func
template <typename K>
constexpr inline bool IS_IKF_t( ) {
    if constexpr ( std::is_final_v<K> ) {
        return true;
    }
    else {
        return false;
    }
};

// clang-format off

#if FFT_DEBUG_LEVEL < 1

#define MyFFTDebugPrintWithDetails(...)
#define MyFFTDebugAssertTrue(cond, msg, ...)
#define MyFFTDebugAssertFalse(cond, msg, ...)
#define MyFFTDebugAssertTestTrue(cond, msg, ...)
#define MyFFTDebugAssertTestFalse(cond, msg, ...)

#else
// Minimally define asserts that check state variables and setup.
#define MyFFTDebugAssertTrue(cond, msg, ...) { if ( (cond) != true ) { std::cerr << msg << std::endl << " Failed Assert at " << __FILE__ << " " << __LINE__ << " " << __PRETTY_FUNCTION__ << std::endl; exit(-1); } }
#define MyFFTDebugAssertFalse(cond, msg, ...) { if ( (cond) == true ) { std::cerr << msg << std::endl  << " Failed Assert at " << __FILE__ << " " << __LINE__ << " " << __PRETTY_FUNCTION__ << std::endl;  exit(-1); } }                                                                                                    
                                                                                                                    
#endif

#if FFT_DEBUG_LEVEL > 1
// Turn on checkpoints in the testing functions.
#define MyFFTDebugAssertTestTrue(cond, msg, ...)  { if ( (cond) != true ) { std::cerr << "    Test " << msg << " FAILED!" << std::endl  << "  at " << __FILE__ << " " << __LINE__ << " " << __PRETTY_FUNCTION__ << std::endl;exit(-1); } else { std::cerr << "    Test " << msg << " passed!" << std::endl; }}
#define MyFFTDebugAssertTestFalse(cond, msg, ...)  { if ( (cond) == true ) {  std::cerr << "    Test " << msg << " FAILED!" << std::endl   << " at " << __FILE__ << " " << __LINE__ << " " << __PRETTY_FUNCTION__ << std::endl;   exit(-1);  } else {  std::cerr << "    Test " << msg << " passed!" << std::endl;  } }

#endif

#if FFT_DEBUG_LEVEL == 2
#define MyFFTDebugPrintWithDetails(...)
#endif

#if FFT_DEBUG_LEVEL == 3
// More verbose debug info
#define MyFFTDebugPrint(...) \
    { std::cerr << __VA_ARGS__ << std::endl; }
#define MyFFTDebugPrintWithDetails(...) \
    { std::cerr << __VA_ARGS__ << " From: " << __FILE__ << " " << __LINE__ << " " << __PRETTY_FUNCTION__ << std::endl; }

#endif

#if FFT_DEBUG_LEVEL == 4
// More verbose debug info + state info
#define MyFFTDebugPrint(...) { PrintState( );  std::cerr << __VA_ARGS__ << std::endl; }
#define MyFFTDebugPrintWithDetails(...)  { PrintState( ); std::cerr << __VA_ARGS__ << " From: " << __FILE__ << " " << __LINE__ << " " << __PRETTY_FUNCTION__ << std::endl; }

#endif

// Always in use
#define MyFFTPrint(...) { std::cerr << __VA_ARGS__ << std::endl; }
#define MyFFTPrintWithDetails(...) { std::cerr << __VA_ARGS__ << " From: " << __FILE__ << " " << __LINE__ << " " << __PRETTY_FUNCTION__ << std::endl; }
#define MyFFTRunTimeAssertTrue(cond, msg, ...) { if ( (cond) != true ) { std::cerr << msg << std::endl << " Failed Assert at " << __FILE__ << " " << __LINE__ << " " << __PRETTY_FUNCTION__ << std::endl;exit(-1); }  }
#define MyFFTRunTimeAssertFalse(cond, msg, ...) { if ( (cond) == true ) {std::cerr << msg << std::endl << " Failed Assert at " << __FILE__ << " " << __LINE__ << " " << __PRETTY_FUNCTION__ << std::endl;exit(-1);  } }                                                                                                               



// I use the same things in cisTEM, so check for them. FIXME, get rid of defines and also find a better sharing mechanism.
#ifndef cudaErr
// Note we are using std::cerr b/c the wxWidgets apps running in cisTEM are capturing std::cout
// If I leave cudaErr blank when HEAVYERRORCHECKING_FFT is not defined, I get some reports/warnings about unused or unreferenced variables. I suspect the performance hit is very small so just leave this on.
// The real cost is in the synchronization of in pre/postcheck.
#define cudaErr(error) { auto status = static_cast<cudaError_t>(error); if ( status != cudaSuccess ) { std::cerr << cudaGetErrorString(status) << " :-> "; MyFFTPrintWithDetails(""); } };
#endif

#ifndef postcheck
    #ifndef precheck
        #ifndef HEAVYERRORCHECKING_FFT
            #define postcheck
            #define precheck
        #else
            #define postcheck  { cudaErr(cudaPeekAtLastError( )); cudaError_t error = cudaStreamSynchronize(cudaStreamPerThread); cudaErr(error); };
            #define precheck { cudaErr(cudaGetLastError( )); }
        #endif
    #endif
#endif  

// clang-format on

inline void checkCudaErr(cudaError_t err) {
    if ( err != cudaSuccess ) {
        std::cerr << cudaGetErrorString(err) << " :-> " << std::endl;
        MyFFTPrintWithDetails(" ");
    }
};

#define USEFASTSINCOS
// The __sincosf doesn't appear to be the problem with accuracy, likely just the extra additions, but it probably also is less flexible with other types. I don't see a half precision equivalent.
#ifdef USEFASTSINCOS
__device__ __forceinline__ void SINCOS(float arg, float* s, float* c) {
    __sincosf(arg, s, c);
}
#else
__device__ __forceinline__ void SINCOS(float arg, float* s, float* c) {
    sincos(arg, s, c);
}
#endif

namespace FastFFT {

__device__ __forceinline__ int
d_ReturnReal1DAddressFromPhysicalCoord(int3 coords, short4 img_dims) {
    return ((((int)coords.z * (int)img_dims.y + coords.y) * (int)img_dims.w * 2) + (int)coords.x);
}

static constexpr const int XZ_STRIDE = 16;

static constexpr const int          bank_size    = 32;
static constexpr const int          bank_padded  = bank_size + 1;
static constexpr const unsigned int ubank_size   = 32;
static constexpr const unsigned int ubank_padded = ubank_size + 1;

__device__ __forceinline__ int GetSharedMemPaddedIndex(const int index) {
    return (index % bank_size) + ((index / bank_size) * bank_padded);
}

__device__ __forceinline__ int GetSharedMemPaddedIndex(const unsigned int index) {
    return (index % ubank_size) + ((index / ubank_size) * ubank_padded);
}

// Return the address of the 1D transform index 0
static __device__ __forceinline__ unsigned int Return1DFFTAddress(const unsigned int pixel_pitch) {
    return pixel_pitch * (blockIdx.y + blockIdx.z * gridDim.y);
}

// Return the address of the 1D transform index 0. Right now testing for a stride of 2, but this could be modifiable if it works.
static __device__ __forceinline__ unsigned int Return1DFFTAddress_strided_Z(const unsigned int pixel_pitch) {
    // In the current condition, threadIdx.z is either 0 || 1, and gridDim.z = size_z / 2
    // index into a 2D tile in the XZ plane, for output in the ZX transposed plane (for coalsced write.)
    return pixel_pitch * (blockIdx.y + (XZ_STRIDE * blockIdx.z + threadIdx.z) * gridDim.y);
}

// Return the address of the 1D transform index 0
static __device__ __forceinline__ unsigned int ReturnZplane(const unsigned int NX, const unsigned int NY) {
    return (blockIdx.z * NX * NY);
}

// Return the address of the 1D transform index 0
static __device__ __forceinline__ unsigned int Return1DFFTAddress_Z(const unsigned int NY) {
    return blockIdx.y + (blockIdx.z * NY);
}

// Return the address of the 1D transform index 0
static __device__ __forceinline__ unsigned int Return1DFFTColumn_XYZ_transpose(const unsigned int NX) {
    // NX should be size_of<FFT>::value for this method. Should this be templated?
    // presumably the XZ axis is alread transposed on the forward, used to index into this state. Indexs in (ZY)' plane for input, to be transposed and permuted to output.'
    return NX * (XZ_STRIDE * (blockIdx.y + gridDim.y * blockIdx.z) + threadIdx.z);
}

// Return the address of the 1D transform index 0
static __device__ __forceinline__ unsigned int Return1DFFTAddress_XZ_transpose(const unsigned int X) {
    return blockIdx.z + gridDim.z * (blockIdx.y + X * gridDim.y);
}

// Return the address of the 1D transform index 0
static __device__ __forceinline__ unsigned int Return1DFFTAddress_XZ_transpose_strided_Z(const unsigned int IDX) {
    // return (XZ_STRIDE*blockIdx.z + (X % XZ_STRIDE)) + (XZ_STRIDE*gridDim.z) * ( blockIdx.y + (X / XZ_STRIDE) * gridDim.y );
    // (IDX % XZ_STRIDE) -> transposed x coordinate in tile
    // ((blockIdx.z*XZ_STRIDE) -> tile offest in physical X (with above gives physical X out (transposed Z))
    // (XZ_STRIDE*gridDim.z) -> n elements in physical X (transposed Z)
    // above * blockIdx.y -> offset in physical Y (transposed Y)
    // (IDX / XZ_STRIDE) -> n elements physical Z (transposed X)
    return ((IDX % XZ_STRIDE) + (blockIdx.z * XZ_STRIDE)) + (XZ_STRIDE * gridDim.z) * (blockIdx.y + (IDX / XZ_STRIDE) * gridDim.y);
}

static __device__ __forceinline__ unsigned int Return1DFFTAddress_XZ_transpose_strided_Z(const unsigned int IDX, const unsigned int Q, const unsigned int sub_fft) {
    // return (XZ_STRIDE*blockIdx.z + (X % XZ_STRIDE)) + (XZ_STRIDE*gridDim.z) * ( blockIdx.y + (X / XZ_STRIDE) * gridDim.y );
    // (IDX % XZ_STRIDE) -> transposed x coordinate in tile
    // ((blockIdx.z*XZ_STRIDE) -> tile offest in physical X (with above gives physical X out (transposed Z))
    // (XZ_STRIDE*gridDim.z) -> n elements in physical X (transposed Z)
    // above * blockIdx.y -> offset in physical Y (transposed Y)
    // (IDX / XZ_STRIDE) -> n elements physical Z (transposed X)
    return ((IDX % XZ_STRIDE) + (blockIdx.z * XZ_STRIDE)) + (XZ_STRIDE * gridDim.z) * (blockIdx.y + ((IDX / XZ_STRIDE) * Q + sub_fft) * gridDim.y);
}

// Return the address of the 1D transform index 0
static __device__ __forceinline__ unsigned int Return1DFFTAddress_YZ_transpose_strided_Z(const unsigned int IDX) {
    // return (XZ_STRIDE*blockIdx.z + (X % XZ_STRIDE)) + (XZ_STRIDE*gridDim.z) * ( blockIdx.y + (X / XZ_STRIDE) * gridDim.y );
    return ((IDX % XZ_STRIDE) + (blockIdx.y * XZ_STRIDE)) + (gridDim.y * XZ_STRIDE) * (blockIdx.z + (IDX / XZ_STRIDE) * gridDim.z);
}

// Return the address of the 1D transform index 0
static __device__ __forceinline__ unsigned int Return1DFFTAddress_YZ_transpose_strided_Z(const unsigned int IDX, const unsigned int Q, const unsigned int sub_fft) {
    // return (XZ_STRIDE*blockIdx.z + (X % XZ_STRIDE)) + (XZ_STRIDE*gridDim.z) * ( blockIdx.y + (X / XZ_STRIDE) * gridDim.y );
    return ((IDX % XZ_STRIDE) + (blockIdx.y * XZ_STRIDE)) + (gridDim.y * XZ_STRIDE) * (blockIdx.z + ((IDX / XZ_STRIDE) * Q + sub_fft) * gridDim.z);
}

// Return the address of the 1D transform index 0
static __device__ __forceinline__ unsigned int Return1DFFTColumn_XZ_to_XY( ) {
    // return blockIdx.y + gridDim.y * ( blockIdx.z + gridDim.z * X);
    return blockIdx.y + gridDim.y * blockIdx.z;
}

static __device__ __forceinline__ unsigned int Return1DFFTAddress_YX_to_XY( ) {
    return blockIdx.z + gridDim.z * blockIdx.y;
}

static __device__ __forceinline__ unsigned int Return1DFFTAddress_YX( ) {
    return Return1DFFTColumn_XZ_to_XY( );
}

// Complex a * conj b multiplication
template <typename ComplexType, typename ScalarType>
static __device__ __host__ inline auto ComplexConjMulAndScale(const ComplexType a, const ComplexType b, ScalarType s) -> decltype(b) {
    ComplexType c;
    c.x = s * (a.x * b.x + a.y * b.y);
    c.y = s * (a.y * b.x - a.x * b.y);
    return c;
}

// GetCudaDeviceArch from https://github.com/mnicely/cufft_examples/blob/master/Common/cuda_helper.h
void GetCudaDeviceProps(DeviceProps& dp);

void CheckSharedMemory(int& memory_requested, DeviceProps& dp);
void CheckSharedMemory(unsigned int& memory_requested, DeviceProps& dp);

using namespace cufftdx;

// TODO this probably needs to depend on the size of the xform, at least small vs large.
constexpr const int elements_per_thread_16   = 4;
constexpr const int elements_per_thread_32   = 8;
constexpr const int elements_per_thread_64   = 8;
constexpr const int elements_per_thread_128  = 8;
constexpr const int elements_per_thread_256  = 8;
constexpr const int elements_per_thread_512  = 8;
constexpr const int elements_per_thread_1024 = 8;
constexpr const int elements_per_thread_2048 = 8;
constexpr const int elements_per_thread_4096 = 8;
constexpr const int elements_per_thread_8192 = 16;

namespace KernelFunction {

    // Define an enum for different functors
    // Intra Kernel Function Type
    enum IKF_t { NOOP, CONJ_MUL};

// Maybe a better way to check , but using keyword final to statically check for non NONE types
template <class T, int N_ARGS, IKF_t U>
class my_functor {};

template <class T>
class my_functor<T, 0, IKF_t::NOOP> {
  public:
    __device__ __forceinline__
            T
            operator( )( ) {
        printf("really specific NOOP\n");
        return 0;
    }
};

template <class T>
class my_functor<T, 2, IKF_t::CONJ_MUL> final {
  public:
    __device__ __forceinline__
            T
            operator( )(float& template_fft_x, float& template_fft_y, const float& target_fft_x, const float& target_fft_y) {
        // Is there a better way than declaring this variable each time?
        float tmp      = (template_fft_x * target_fft_x + template_fft_y * target_fft_y);
        template_fft_y = (template_fft_y * target_fft_x - template_fft_x * target_fft_y);
        template_fft_x = tmp;
    }
};

} // namespace KernelFunction

// constexpr const std::map<unsigned int, unsigned int> elements_per_thread = {
//     {16, 4}, {"GPU", 15}, {"RAM", 20},
// };
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// FFT kernels
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////
// BLOCK FFT based Kernel definitions
////////////////////////////////////////

/* 

transpose definitions in the kernel names refer to the physical axes in memory, which may not match the logical axes if following a previous transpose.
    2 letters indicate a swap of the axes specified
    3 letters indicate a permutation. E.g./ XZY, x -> Z, z -> Y, y -> X
R2C and C2R kernels are named as:
<cufftdx transform method>_fft_kernel_< fft type >_< size change >_< transpose axes >

C2C additionally specify direction and may specify an operation.
<cufftdx transform method>_fft_kernel_< fft type >_< direction >_< size change >_< transpose axes >_< operation in between round trip kernels >

*/

/////////////
// R2C
/////////////

/*
  For these kernels the XY transpose is intended for 2d transforms, while the XZ is for 3d transforms.
*/

template <class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_R2C_NONE_XY(const ScalarType* __restrict__ input_values, ComplexType* __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace);

// XZ_STRIDE ffts/block via threadIdx.x, notice launch bounds. Creates partial coalescing.
template <class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_R2C_NONE_XZ(const ScalarType* __restrict__ input_values, ComplexType* __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace);

template <class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_R2C_INCREASE_XY(const ScalarType* __restrict__ input_values, ComplexType* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace);

// XZ_STRIDE ffts/block via threadIdx.x, notice launch bounds. Creates partial coalescing.
template <class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_R2C_INCREASE_XZ(const ScalarType* __restrict__ input_values, ComplexType* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace);

// __launch_bounds__(FFT::max_threads_per_block)  we don't know this because it is threadDim.x * threadDim.z - this could be templated if it affects performance significantly
template <class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__global__ void block_fft_kernel_R2C_DECREASE_XY(const ScalarType* __restrict__ input_values, ComplexType* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace);

/////////////
// C2C
/////////////

template <class FFT, class ComplexType = typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_INCREASE(const ComplexType* __restrict__ input_values, ComplexType* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace);

// __launch_bounds__(FFT::max_threads_per_block)  we don't know this because it is threadDim.x * threadDim.z - this could be templated if it affects performance significantly
template <class FFT, class ComplexType = typename FFT::value_type>
__global__ void block_fft_kernel_C2C_DECREASE(const ComplexType* __restrict__ input_values, ComplexType* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace);

template <class FFT, class ComplexType = typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_WithPadding_SwapRealSpaceQuadrants(const ComplexType* __restrict__ input_values, ComplexType* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace);

template <class FFT, class invFFT, class ComplexType = typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_FWD_INCREASE_INV_NONE_ConjMul(const ComplexType* __restrict__ image_to_search, const ComplexType* __restrict__ input_values, ComplexType* __restrict__ output_values,
                                                                Offsets mem_offsets, int Q, typename FFT::workspace_type workspace_fwd, typename invFFT::workspace_type workspace_inv);

template <class FFT, class invFFT, class ComplexType = typename FFT::value_type, class PreOpType, class IntraOpType, class PostOpType>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_FWD_INCREASE_OP_INV_NONE(const ComplexType* __restrict__ image_to_search, const ComplexType* __restrict__ input_values, ComplexType* __restrict__ output_values,
                                                           Offsets mem_offsets, int Q, typename FFT::workspace_type workspace_fwd, typename invFFT::workspace_type workspace_inv,
                                                           PreOpType pre_op_lambda, IntraOpType intra_op_lambda, PostOpType post_op_lambda);

template <class FFT, class invFFT, class ComplexType = typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_FWD_INCREASE_INV_NONE_ConjMul_SwapRealSpaceQuadrants(const ComplexType* __restrict__ image_to_search, const ComplexType* __restrict__ input_values, ComplexType* __restrict__ output_values,
                                                                                       Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace_fwd, typename invFFT::workspace_type workspace_inv);

template <class FFT, class invFFT, class ComplexType = typename FFT::value_type>
__global__ void block_fft_kernel_C2C_FWD_NONE_INV_DECREASE_ConjMul(const ComplexType* __restrict__ image_to_search, const ComplexType* __restrict__ input_values, ComplexType* __restrict__ output_values,
                                                                   Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace_fwd, typename invFFT::workspace_type workspace_inv);

template <class FFT, class ComplexType = typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_NONE(const ComplexType* __restrict__ input_values, ComplexType* __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace);

template <class FFT, class ComplexType = typename FFT::value_type>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_NONE_XZ(const ComplexType* __restrict__ input_values, ComplexType* __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace);

template <class FFT, class ComplexType = typename FFT::value_type>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_NONE_XYZ(const ComplexType* __restrict__ input_values, ComplexType* __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace);

template <class FFT, class ComplexType = typename FFT::value_type>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_INCREASE_XYZ(const ComplexType* __restrict__ input_values, ComplexType* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace);
/////////////
// C2R
/////////////

template <class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2R_NONE(const ComplexType* __restrict__ input_values, ScalarType* __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace);

template <class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2R_NONE_XY(const ComplexType* __restrict__ input_values, ScalarType* __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace);

// __launch_bounds__(FFT::max_threads_per_block)  we don't know this because it is threadDim.x * threadDim.z - this could be templated if it affects performance significantly
template <class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__global__ void block_fft_kernel_C2R_DECREASE_XY(const ComplexType* __restrict__ input_values, ScalarType* __restrict__ output_values, Offsets mem_offsets, const float twiddle_in, const unsigned int Q, typename FFT::workspace_type workspace);

//////////////////////////////
// Thread FFT based Kernel definitions
//////////////////////////////

/////////////
// R2C
/////////////

template <class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__global__ void thread_fft_kernel_R2C_decomposed(const ScalarType* __restrict__ input_values, ComplexType* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q);

template <class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__global__ void thread_fft_kernel_R2C_decomposed_transposed(const ScalarType* __restrict__ input_values, ComplexType* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q);

/////////////
// C2C
/////////////

template <class FFT, class ComplexType = typename FFT::value_type>
__global__ void thread_fft_kernel_C2C_decomposed(const ComplexType* __restrict__ input_values, ComplexType* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q);

template <class FFT, class invFFT, class ComplexType = typename FFT::value_type>
__global__ void thread_fft_kernel_C2C_decomposed_ConjMul(const ComplexType* __restrict__ image_to_search, const ComplexType* __restrict__ input_values, ComplexType* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q);

/////////////
// C2R
/////////////

template <class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__global__ void thread_fft_kernel_C2R_decomposed(const ComplexType* __restrict__ input_values, ScalarType* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q);

template <class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__global__ void thread_fft_kernel_C2R_decomposed_transposed(const ComplexType* __restrict__ input_values, ScalarType* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// End FFT Kernels
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class InputType, class OutputType>
__global__ void clip_into_top_left_kernel(InputType* input_values, OutputType* output_values, const short4 dims);

// Modified from GpuImage::ClipIntoRealKernel
template <typename InputType, typename OutputType>
__global__ void clip_into_real_kernel(InputType*  real_values_gpu,
                                      OutputType* other_image_real_values_gpu,
                                      short4      dims,
                                      short4      other_dims,
                                      int3        wanted_coordinate_of_box_center,
                                      OutputType  wanted_padding_value);

//////////////////////////////////////////////
// IO functions adapted from the cufftdx examples
///////////////////////////////

template <class FFT>
struct io {
    using complex_type = typename FFT::value_type;
    using scalar_type  = typename complex_type::value_type;

    static inline __device__ unsigned int stride_size( ) {
        return cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
    }

    static inline __device__ void load_r2c(const scalar_type* input,
                                           complex_type*      thread_data) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            thread_data[i].x = input[index];
            thread_data[i].y = 0.0f;
            index += stride;
        }
    }

    static inline __device__ void store_r2c(const complex_type* thread_data,
                                            complex_type*       output,
                                            int                 offset) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = offset + threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            output[index] = thread_data[i];
            index += stride;
        }
        constexpr unsigned int threads_per_fft        = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
        // threads_per_fft == 1 means that EPT == SIZE, so we need to store one more element
        constexpr unsigned int values_left_to_store =
                threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
        if ( threadIdx.x < values_left_to_store ) {
            output[index] = thread_data[FFT::elements_per_thread / 2];
        }
    }

    // Since we can make repeated use of the same shared memory for each sub-fft
    // we use this method to load into shared mem instead of directly to registers
    // TODO set this up for async mem load
    static inline __device__ void load_shared(const complex_type* input,
                                              complex_type*       shared_input,
                                              complex_type*       thread_data,
                                              float*              twiddle_factor_args,
                                              float               twiddle_in,
                                              int*                input_map,
                                              int*                output_map,
                                              int                 Q) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            input_map[i]           = index;
            output_map[i]          = Q * index;
            twiddle_factor_args[i] = twiddle_in * index;
            thread_data[i]         = input[index];
            shared_input[index]    = thread_data[i];
            index += stride;
        }
    }

    // Since we can make repeated use of the same shared memory for each sub-fft
    // we use this method to load into shared mem instead of directly to registers
    // TODO set this up for async mem load
    static inline __device__ void load_shared(const complex_type* input,
                                              complex_type*       shared_input,
                                              complex_type*       thread_data,
                                              float*              twiddle_factor_args,
                                              float               twiddle_in) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            twiddle_factor_args[i] = twiddle_in * index;
            thread_data[i]         = input[index];
            shared_input[index]    = thread_data[i];
            index += stride;
        }
    }

    static inline __device__ void load_shared(const complex_type* input,
                                              complex_type*       shared_input,
                                              complex_type*       thread_data,
                                              float*              twiddle_factor_args,
                                              float               twiddle_in,
                                              int*                input_map,
                                              int*                output_map,
                                              int                 Q,
                                              int                 number_of_elements) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            if ( index < number_of_elements ) {
                input_map[i]           = index;
                output_map[i]          = Q * index;
                twiddle_factor_args[i] = twiddle_in * index;
                thread_data[i]         = input[index];
                shared_input[index]    = thread_data[i];
                index += stride;
            }
            else {
                input_map[i] = -9999; // ignore this in subsequent ops
            }
        }
    }

    // Since we can make repeated use of the same shared memory for each sub-fft
    // we use this method to load into shared mem instead of directly to registers
    // TODO set this up for async mem load - alternatively, load to registers then copy but leave in register for firt compute
    static inline __device__ void load_r2c_shared(const scalar_type* input,
                                                  scalar_type*       shared_input,
                                                  complex_type*      thread_data,
                                                  float*             twiddle_factor_args,
                                                  float              twiddle_in,
                                                  int*               input_map,
                                                  int*               output_map,
                                                  int                Q) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // if (blockIdx.y == 0) ("blck %i index %i \n", Q*index, index);
            input_map[i]           = index;
            output_map[i]          = Q * index;
            twiddle_factor_args[i] = twiddle_in * index;
            thread_data[i].x       = input[index];
            thread_data[i].y       = 0.0f;
            shared_input[index]    = thread_data[i].x;
            index += stride;
        }
    }

    // Since we can make repeated use of the same shared memory for each sub-fft
    // we use this method to load into shared mem instead of directly to registers
    // TODO set this up for async mem load - alternatively, load to registers then copy but leave in register for firt compute
    static inline __device__ void load_r2c_shared(const scalar_type* input,
                                                  scalar_type*       shared_input,
                                                  complex_type*      thread_data,
                                                  float*             twiddle_factor_args,
                                                  float              twiddle_in) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            twiddle_factor_args[i] = twiddle_in * index;
            thread_data[i].x       = input[index];
            thread_data[i].y       = 0.0f;
            shared_input[index]    = thread_data[i].x;
            index += stride;
        }
    }

    static inline __device__ void load_r2c_shared_and_pad(const scalar_type* input,
                                                          complex_type*      shared_mem) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x + (threadIdx.z * size_of<FFT>::value);
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            shared_mem[GetSharedMemPaddedIndex(index)] = complex_type(input[index], 0.f);
            index += stride;
        }
        __syncthreads( );
    }

    static inline __device__ void copy_from_shared(const complex_type* shared_mem,
                                                   complex_type*       thread_data,
                                                   const unsigned int  Q) {
        const unsigned int stride = stride_size( ) * Q; // I think the Q is needed, but double check me TODO
        unsigned int       index  = (threadIdx.x * Q) + threadIdx.z;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            thread_data[i] = shared_mem[GetSharedMemPaddedIndex(index)];
            index += stride;
        }
        __syncthreads( ); // FFT().execute is setup to reuse the shared mem, so we need to sync here. Optionally, we could allocate more shared mem and remove this sync
    }

    // Note that unlike most functions in this file, this one does not have a
    // const decorator on the thread mem, as we want to modify it with the twiddle factors
    // before reducing the full shared mem space.
    static inline __device__ void reduce_block_fft(complex_type*      thread_data,
                                                   complex_type*      shared_mem,
                                                   const float        twiddle_in,
                                                   const unsigned int Q) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x + (threadIdx.z * size_of<FFT>::value);
        complex_type       twiddle;
        // In the first loop, all threads participate and write back to natural order in shared memory
        // while also updating with the full size twiddle factor.
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // ( index * threadIdx.z) == ( k % P * n2 )
            SINCOS(twiddle_in * (index * threadIdx.z), &twiddle.y, &twiddle.x);
            thread_data[i] *= twiddle;

            shared_mem[GetSharedMemPaddedIndex(index)] = thread_data[i];
            index += stride;
        }
        __syncthreads( );

        // Now we reduce the shared memory into the first block of size P
        // Reuse index
        for ( index = 2; index <= Q; index *= 2 ) {
            // Some threads drop out each loop
            if ( threadIdx.z % index == 0 ) {
                for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                    thread_data[i] += shared_mem[GetSharedMemPaddedIndex(threadIdx.x + (i * stride) + (index / 2 * size_of<FFT>::value))];
                }
            } // end if condition
            // All threads can reach this point
            __syncthreads( );
        }
    }

    static inline __device__ void store_r2c_reduced(const complex_type* thread_data,
                                                    complex_type*       output,
                                                    const unsigned int  pixel_pitch,
                                                    const unsigned int  memory_limit) {
        if ( threadIdx.z == 0 ) {
            // Finally we write out the first size_of<FFT>::values to global
            const unsigned int stride = stride_size( );
            unsigned int       index  = threadIdx.x;
            for ( unsigned int i = 0; i <= FFT::elements_per_thread / 2; i++ ) {
                if ( index < memory_limit ) {
                    // transposed index.
                    output[index * pixel_pitch + blockIdx.y] = thread_data[i];
                }
                index += stride;
            }
        }
    }

    // when using load_shared || load_r2c_shared, we need then copy from shared mem into the registers.
    // notice we still need the packed complex values for the xform.
    static inline __device__ void copy_from_shared(const scalar_type* shared_input,
                                                   complex_type*      thread_data,
                                                   int*               input_map) {
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            thread_data[i].x = shared_input[input_map[i]];
            thread_data[i].y = 0.0f;
        }
    }

    static inline __device__ void copy_from_shared(const complex_type* shared_input_complex,
                                                   complex_type*       thread_data,
                                                   int*                input_map) {
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            thread_data[i] = shared_input_complex[input_map[i]];
        }
    }

    static inline __device__ void copy_from_shared(const scalar_type* shared_input,
                                                   complex_type*      thread_data) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            thread_data[i].x = shared_input[index];
            thread_data[i].y = 0.0f;
            index += stride;
        }
    }

    static inline __device__ void copy_from_shared(const complex_type* shared_input_complex,
                                                   complex_type*       thread_data) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            thread_data[i] = shared_input_complex[index];
            index += stride;
        }
    }

    static inline __device__ void load_shared_and_conj_multiply(const complex_type* image_to_search,
                                                                complex_type*       thread_data) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        complex_type       c;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            c.x = (thread_data[i].x * image_to_search[index].x + thread_data[i].y * image_to_search[index].y);
            c.y = (thread_data[i].y * image_to_search[index].x - thread_data[i].x * image_to_search[index].y);
            // a * conj b
            thread_data[i] = c; //ComplexConjMulAndScale<complex_type, scalar_type>(thread_data[i], image_to_search[index], 1.0f);
            index += stride;
        }
    }

    // TODO: set user lambda to default = false, then get rid of other load_shared
    template <class FunctionType>
    static inline __device__ void load_shared(const complex_type* image_to_search,
                                              complex_type*       thread_data,
                                              FunctionType        intra_op_lambda = nullptr) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        if constexpr ( IS_IKF_t<FunctionType>( ) ) {
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                intra_op_lambda(thread_data[i].x, thread_data[i].y, image_to_search[index].x, image_to_search[index].y); //ComplexConjMulAndScale<complex_type, scalar_type>(thread_data[i], image_to_search[index], 1.0f);
                index += stride;
            }
        }
        else {
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                // a * conj b
                thread_data[i] = thread_data[i], image_to_search[index]; //ComplexConjMulAndScale<complex_type, scalar_type>(thread_data[i], image_to_search[index], 1.0f);
                index += stride;
            }
        }
    }

    // Now we need send to shared mem and transpose on the way
    // TODO: fix bank conflicts later.
    static inline __device__ void transpose_r2c_in_shared_XZ(complex_type* shared_mem,
                                                             complex_type* thread_data) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            shared_mem[threadIdx.z + index * XZ_STRIDE] = thread_data[i];
            index += stride;
        }
        constexpr unsigned int threads_per_fft        = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
        constexpr unsigned int values_left_to_store   = threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
        if ( threadIdx.x < values_left_to_store ) {
            shared_mem[threadIdx.z + index * XZ_STRIDE] = thread_data[FFT::elements_per_thread / 2];
        }
        __syncthreads( );
    }

    // Now we need send to shared mem and transpose on the way
    // TODO: fix bank conflicts later.
    static inline __device__ void transpose_in_shared_XZ(complex_type* shared_mem,
                                                         complex_type* thread_data) {
        const unsigned int stride = io<FFT>::stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // return (XZ_STRIDE*blockIdx.z + threadIdx.z) + (XZ_STRIDE*gridDim.z) * ( blockIdx.y + X * gridDim.y );
            // XZ_STRIDE == XZ_STRIDE
            shared_mem[threadIdx.z + index * XZ_STRIDE] = thread_data[i];
            index += stride;
        }
        __syncthreads( );
    }

    static inline __device__ void store_r2c_transposed_xz(const complex_type* thread_data,
                                                          complex_type*       output) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            output[Return1DFFTAddress_XZ_transpose(index)] = thread_data[i];
            index += stride;
        }
        constexpr unsigned int threads_per_fft        = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
        constexpr unsigned int values_left_to_store   = threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
        if ( threadIdx.x < values_left_to_store ) {
            output[Return1DFFTAddress_XZ_transpose(index)] = thread_data[FFT::elements_per_thread / 2];
        }
        __syncthreads( );
    }

    // Store a transposed tile, made up of contiguous (full) FFTS
    static inline __device__ void store_r2c_transposed_xz_strided_Z(const complex_type* shared_mem,
                                                                    complex_type*       output) {
        const unsigned int     stride                 = stride_size( );
        constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
        unsigned int           index                  = threadIdx.x + threadIdx.z * output_values_to_store;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            output[Return1DFFTAddress_XZ_transpose_strided_Z(index)] = shared_mem[index];
            index += stride;
        }
        constexpr unsigned int threads_per_fft      = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        constexpr unsigned int values_left_to_store = threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
        if ( threadIdx.x < values_left_to_store ) {
            output[Return1DFFTAddress_XZ_transpose_strided_Z(index)] = shared_mem[index];
        }
        __syncthreads( );
    }

    // Store a transposed tile, made up of non-contiguous (strided partial) FFTS
    //
    static inline __device__ void store_r2c_transposed_xz_strided_Z(const complex_type* shared_mem,
                                                                    complex_type*       output,
                                                                    const unsigned int  Q,
                                                                    const unsigned int  sub_fft) {
        const unsigned int     stride                 = stride_size( );
        constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
        unsigned int           index                  = threadIdx.x + threadIdx.z * output_values_to_store;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            output[Return1DFFTAddress_XZ_transpose_strided_Z(index, Q, sub_fft)] = shared_mem[index];
            index += stride;
        }
        constexpr unsigned int threads_per_fft      = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        constexpr unsigned int values_left_to_store = threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
        if ( threadIdx.x < values_left_to_store ) {
            output[Return1DFFTAddress_XZ_transpose_strided_Z(index, Q, sub_fft)] = shared_mem[index];
        }
        __syncthreads( );
    }

    static inline __device__ void store_transposed_xz_strided_Z(const complex_type* shared_mem,
                                                                complex_type*       output) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x + threadIdx.z * cufftdx::size_of<FFT>::value;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            output[Return1DFFTAddress_XZ_transpose_strided_Z(index)] = shared_mem[index];
            index += stride;
        }
        __syncthreads( );
    }

    static inline __device__ void store_r2c_transposed_xy(const complex_type* thread_data,
                                                          complex_type*       output,
                                                          int                 pixel_pitch) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            // output map is thread local, so output_MAP[i] gives the x-index in the non-transposed array and blockIdx.y gives the y-index
            output[index * pixel_pitch + blockIdx.y] = thread_data[i];
            index += stride;
        }
        constexpr unsigned int threads_per_fft        = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
        constexpr unsigned int values_left_to_store   = threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
        if ( threadIdx.x < values_left_to_store ) {
            output[index * pixel_pitch + blockIdx.y] = thread_data[FFT::elements_per_thread / 2];
        }
    }

    static inline __device__ void store_r2c_transposed_xy(const complex_type* thread_data,
                                                          complex_type*       output,
                                                          int*                output_MAP,
                                                          int                 pixel_pitch) {
        const unsigned int stride = stride_size( );
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            // output map is thread local, so output_MAP[i] gives the x-index in the non-transposed array and blockIdx.y gives the y-index
            output[output_MAP[i] * pixel_pitch + blockIdx.y] = thread_data[i];
            // if (blockIdx.y == 32) printf("from store transposed %i , val %f %f\n", output_MAP[i], thread_data[i].x, thread_data[i].y);
        }
        constexpr unsigned int threads_per_fft        = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
        constexpr unsigned int values_left_to_store   = threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
        if ( threadIdx.x < values_left_to_store ) {
            output[output_MAP[FFT::elements_per_thread / 2] * pixel_pitch + blockIdx.y] = thread_data[FFT::elements_per_thread / 2];
        }
    }

    static inline __device__ void store_r2c_transposed_xy(const complex_type* thread_data,
                                                          complex_type*       output,
                                                          int*                output_MAP,
                                                          int                 pixel_pitch,
                                                          int                 memory_limit) {
        const unsigned int stride = stride_size( );
        for ( unsigned int i = 0; i <= FFT::elements_per_thread / 2; i++ ) {
            // output map is thread local, so output_MAP[i] gives the x-index in the non-transposed array and blockIdx.y gives the y-index
            // if (blockIdx.y == 1) printf("index, pitch, blcok, address %i, %i, %i, %i\n", output_MAP[i], pixel_pitch, memory_limit, output_MAP[i]*pixel_pitch + blockIdx.y);

            if ( output_MAP[i] < memory_limit )
                output[output_MAP[i] * pixel_pitch + blockIdx.y] = thread_data[i];
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
    }

    static inline __device__ void load_c2r(const complex_type* input,
                                           complex_type*       thread_data) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            thread_data[i] = input[index];
            index += stride;
        }
        constexpr unsigned int threads_per_fft       = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        constexpr unsigned int output_values_to_load = (cufftdx::size_of<FFT>::value / 2) + 1;
        // threads_per_fft == 1 means that EPT == SIZE, so we need to load one more element
        constexpr unsigned int values_left_to_load = threads_per_fft == 1 ? 1 : (output_values_to_load % threads_per_fft);
        if ( threadIdx.x < values_left_to_load ) {
            thread_data[FFT::elements_per_thread / 2] = input[index];
        }
    }

    static inline __device__ void load_c2r_transposed(const complex_type* input,
                                                      complex_type*       thread_data,
                                                      unsigned int        pixel_pitch) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            thread_data[i] = input[(pixel_pitch * index) + blockIdx.y];
            index += stride;
        }
        constexpr unsigned int threads_per_fft       = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        constexpr unsigned int output_values_to_load = (cufftdx::size_of<FFT>::value / 2) + 1;
        // threads_per_fft == 1 means that EPT == SIZE, so we need to load one more element
        constexpr unsigned int values_left_to_load = threads_per_fft == 1 ? 1 : (output_values_to_load % threads_per_fft);
        if ( threadIdx.x < values_left_to_load ) {
            thread_data[FFT::elements_per_thread / 2] = input[(pixel_pitch * index) + blockIdx.y];
        }
    }

    static inline __device__ void load_c2r_shared_and_pad(const complex_type* input,
                                                          complex_type*       shared_mem,
                                                          const unsigned int  pixel_pitch) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x + (threadIdx.z * size_of<FFT>::value);
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            shared_mem[GetSharedMemPaddedIndex(index)] = input[pixel_pitch * index];
            index += stride;
        }
        constexpr unsigned int threads_per_fft       = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        constexpr unsigned int output_values_to_load = (cufftdx::size_of<FFT>::value / 2) + 1;
        // threads_per_fft == 1 means that EPT == SIZE, so we need to load one more element
        constexpr unsigned int values_left_to_load = threads_per_fft == 1 ? 1 : (output_values_to_load % threads_per_fft);
        if ( threadIdx.x < values_left_to_load ) {
            shared_mem[GetSharedMemPaddedIndex(index)] = input[pixel_pitch * index];
        }
        __syncthreads( );
    }

    // this may benefit from asynchronous execution
    static inline __device__ void load(const complex_type* input,
                                       complex_type*       thread_data) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            thread_data[i] = input[index];
            // if (blockIdx.y == 0) printf("block %i , val %f %f\n", index, input[index].x, input[index].y);

            index += stride;
        }
    }

    // this may benefit from asynchronous execution
    static inline __device__ void load(const complex_type* input,
                                       complex_type*       thread_data,
                                       int                 last_index_to_load) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            if ( index < last_index_to_load )
                thread_data[i] = input[index];
            else
                thread_data[i] = complex_type(0.0f, 0.0f);
            index += stride;
        }
    }

    //  TODO: set pre_op_lambda to default=false and get rid of other load
    template <class FunctionType>
    static inline __device__ void load(const complex_type* input,
                                       complex_type*       thread_data,
                                       int                 last_index_to_load,
                                       FunctionType        pre_op_lambda = nullptr) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        if constexpr ( IS_IKF_t<FunctionType>( ) ) {
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                if ( index < last_index_to_load )
                    thread_data[i] = pre_op_lambda(input[index]);
                else
                    thread_data[i] = pre_op_lambda(complex_type(0.0f, 0.0f));
                index += stride;
            }
        }
        else {
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                if ( index < last_index_to_load )
                    thread_data[i] = input[index];
                else
                    thread_data[i] = complex_type(0.0f, 0.0f);
                index += stride;
            }
        }
    }

    static inline __device__ void store_and_swap_quadrants(const complex_type* thread_data,
                                                           complex_type*       output,
                                                           int                 first_negative_index) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        complex_type       phase_shift;
        int                logical_y;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // If no kernel based changes are made to source_idx, this will be the same as the original index value
            phase_shift = thread_data[i];
            logical_y   = index;
            if ( logical_y >= first_negative_index )
                logical_y -= 2 * first_negative_index;
            if ( (int(blockIdx.y) + logical_y) % 2 != 0 )
                phase_shift *= -1.f;
            output[index] = phase_shift;
            index += stride;
        }
    }

    static inline __device__ void store_and_swap_quadrants(const complex_type* thread_data,
                                                           complex_type*       output,
                                                           int*                source_idx,
                                                           int                 first_negative_index) {
        const unsigned int stride = stride_size( );
        complex_type       phase_shift;
        int                logical_y;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // If no kernel based changes are made to source_idx, this will be the same as the original index value
            phase_shift = thread_data[i];
            logical_y   = source_idx[i];
            if ( logical_y >= first_negative_index )
                logical_y -= 2 * first_negative_index;
            if ( (int(blockIdx.y) + logical_y) % 2 != 0 )
                phase_shift *= -1.f;
            output[source_idx[i]] = phase_shift;
        }
    }

    template <class FunctionType = std::nullptr_t>
    static inline __device__ void store(const complex_type* thread_data,
                                        complex_type*       output,
                                        FunctionType        post_op_lambda = nullptr) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        if constexpr ( IS_IKF_t<FunctionType>( ) ) {
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                output[index] = post_op_lambda(thread_data[i]);
                index += stride;
            }
        }
        else {
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                output[index] = thread_data[i];
                index += stride;
            }
        }
    }

    static inline __device__ void store(const complex_type* thread_data,
                                        complex_type*       output,
                                        const unsigned int  Q,
                                        const unsigned int  sub_fft) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            output[index * Q + sub_fft] = thread_data[i];
            index += stride;
        }
    }

    static inline __device__ void store_Z(const complex_type* shared_mem,
                                          complex_type*       output) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x + threadIdx.z * size_of<FFT>::value;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            output[Return1DFFTAddress_YZ_transpose_strided_Z(index)] = shared_mem[index];

            index += stride;
        }
    }

    static inline __device__ void store_Z(const complex_type* shared_mem,
                                          complex_type*       output,
                                          const unsigned int  Q,
                                          const unsigned int  sub_fft) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x + threadIdx.z * size_of<FFT>::value;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            output[Return1DFFTAddress_YZ_transpose_strided_Z(index, Q, sub_fft)] = shared_mem[index];
            index += stride;
        }
        __syncthreads( );
    }

    static inline __device__ void store(const complex_type* thread_data,
                                        complex_type*       output,
                                        unsigned int        memory_limit) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            if ( index < memory_limit )
                output[index] = thread_data[i];
            index += stride;
        }
    }

    static inline __device__ void store(const complex_type* thread_data,
                                        complex_type*       output,
                                        int*                source_idx) {
        const unsigned int stride = stride_size( );
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // If no kernel based changes are made to source_idx, this will be the same as the original index value
            output[source_idx[i]] = thread_data[i];
        }
    }

    static inline __device__ void store_subset(const complex_type* thread_data,
                                               complex_type*       output,
                                               int*                source_idx) {
        const unsigned int stride = stride_size( );
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // If no kernel based changes are made to source_idx, this will be the same as the original index value
            if ( source_idx[i] >= 0 )
                output[source_idx[i]] = thread_data[i];
        }
    }

    static inline __device__ void store_coalesced(const complex_type* shared_output,
                                                  complex_type*       global_output,
                                                  int                 offset) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = offset + threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            global_output[index] = shared_output[index];
            index += stride;
        }
    }

    static inline __device__ void load_c2c_shared_and_pad(const complex_type* input,
                                                          complex_type*       shared_mem) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x + (threadIdx.z * size_of<FFT>::value);
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            shared_mem[GetSharedMemPaddedIndex(index)] = input[index];
            index += stride;
        }
        __syncthreads( );
    }

    static inline __device__ void store_c2c_reduced(const complex_type* thread_data,
                                                    complex_type*       output) {
        if ( threadIdx.z == 0 ) {
            // Finally we write out the first size_of<FFT>::values to global
            const unsigned int stride = stride_size( );
            unsigned int       index  = threadIdx.x + (threadIdx.z * size_of<FFT>::value);
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                if ( index < size_of<FFT>::value ) {
                    // transposed index.
                    output[index] = thread_data[i];
                }
                index += stride;
            }
        }
    }

    static inline __device__ void store_c2r_reduced(const complex_type* thread_data,
                                                    scalar_type*        output) {
        if ( threadIdx.z == 0 ) {
            // Finally we write out the first size_of<FFT>::values to global
            const unsigned int stride = stride_size( );
            unsigned int       index  = threadIdx.x + (threadIdx.z * size_of<FFT>::value);
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                if ( index < size_of<FFT>::value ) {
                    // transposed index.
                    output[index] = reinterpret_cast<const scalar_type*>(thread_data)[i];
                }
                index += stride;
            }
        }
    }

    static inline __device__ void store_transposed(const complex_type* thread_data,
                                                   complex_type*       output,
                                                   int*                output_map,
                                                   int*                rotated_offset,
                                                   int                 memory_limit) {
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // If no kernel based changes are made to source_idx, this will be the same as the original index value
            if ( output_map[i] < memory_limit )
                output[rotated_offset[1] * output_map[i] + rotated_offset[0]] = thread_data[i];
        }
    }

    static inline __device__ void store_c2r(const complex_type* thread_data,
                                            scalar_type*        output) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            output[index] = reinterpret_cast<const scalar_type*>(thread_data)[i];
            index += stride;
        }
    }

    static inline __device__ void store_c2r(const complex_type* thread_data,
                                            scalar_type*        output,
                                            unsigned int        memory_limit) {
        const unsigned int stride = stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // TODO: does reinterpret_cast<const scalar_type*>(thread_data)[i] make more sense than just thread_data[i].x??
            if ( index < memory_limit )
                output[index] = reinterpret_cast<const scalar_type*>(thread_data)[i];
            index += stride;
        }
    }
}; // struct io}

template <class FFT>
struct io_thread {
    using complex_type = typename FFT::value_type;
    using scalar_type  = typename complex_type::value_type;

    static inline __device__ void load_r2c(const scalar_type* input,
                                           complex_type*      thread_data,
                                           const int          stride) {
        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < size_of<FFT>::value; i++ ) {
            thread_data[i].x = input[index];
            thread_data[i].y = scalar_type(0);
            index += stride;
        }
    }

    static inline __device__ void store_r2c(const complex_type* shared_output,
                                            complex_type*       output,
                                            const int           stride,
                                            const int           memory_limit) {
        // Each thread reads in the input data at stride = mem_offsets.Q
        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < size_of<FFT>::value / 2; i++ ) {
            output[index] = shared_output[index];
            index += stride;
        }
        if ( index < memory_limit ) {
            output[index] = shared_output[index];
        }
    }

    static inline __device__ void store_r2c_transposed_xy(const complex_type* shared_output,
                                                          complex_type*       output,
                                                          int                 stride,
                                                          int                 pixel_pitch,
                                                          int                 memory_limit) {
        // Each thread reads in the input data at stride = mem_offsets.Q
        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < size_of<FFT>::value / 2; i++ ) {
            output[index * pixel_pitch] = shared_output[index];
            index += stride;
        }
        if ( index < memory_limit ) {
            output[index * pixel_pitch] = shared_output[index];
        }
    }

    static inline __device__ void remap_decomposed_segments(const complex_type* thread_data,
                                                            complex_type*       shared_output,
                                                            float               twiddle_in,
                                                            int                 Q,
                                                            int                 memory_limit) {
        // Unroll the first loop and initialize the shared mem.
        complex_type twiddle;
        int          index = threadIdx.x * size_of<FFT>::value;
        twiddle_in *= threadIdx.x; // twiddle factor arg now just needs to multiplied by K = (index + i)
        for ( unsigned int i = 0; i < size_of<FFT>::value; i++ ) {
            SINCOS(twiddle_in * (index + i), &twiddle.y, &twiddle.x);
            twiddle *= thread_data[i];
            if ( index + i < memory_limit )
                shared_output[index + i] = twiddle;
        }
        __syncthreads( ); // make sure all the shared mem is initialized to the starting value. There should be no contention as every thread is working on its own block of memory.

        for ( unsigned int sub_fft = 1; sub_fft < Q; sub_fft++ ) {
            // wrap around, 0 --> 1, Q-1 --> 0 etc.
            index = ((threadIdx.x + sub_fft) % Q) * size_of<FFT>::value;
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                SINCOS(twiddle_in * (index + i), &twiddle.y, &twiddle.x);
                twiddle *= thread_data[i];
                if ( index + i < memory_limit ) {
                    atomicAdd_block(&shared_output[index + i].x, twiddle.x);
                    atomicAdd_block(&shared_output[index + i].y, twiddle.y);
                }
            }
        }
        __syncthreads( );
    }

    static inline __device__ void load_c2c(const complex_type* input,
                                           complex_type*       thread_data,
                                           const int           stride) {
        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < size_of<FFT>::value; i++ ) {
            thread_data[i] = input[index];
            index += stride;
        }
    }

    static inline __device__ void store_c2c(const complex_type* shared_output,
                                            complex_type*       output,
                                            const int           stride) {
        // Each thread reads in the input data at stride = mem_offsets.Q
        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < size_of<FFT>::value; i++ ) {
            output[index] = shared_output[index];
            index += stride;
        }
    }

    static inline __device__ void remap_decomposed_segments(const complex_type* thread_data,
                                                            complex_type*       shared_output,
                                                            float               twiddle_in,
                                                            int                 Q) {
        // Unroll the first loop and initialize the shared mem.
        complex_type twiddle;
        int          index = threadIdx.x * size_of<FFT>::value;
        twiddle_in *= threadIdx.x; // twiddle factor arg now just needs to multiplied by K = (index + i)
        for ( unsigned int i = 0; i < size_of<FFT>::value; i++ ) {
            SINCOS(twiddle_in * (index + i), &twiddle.y, &twiddle.x);
            twiddle *= thread_data[i];
            shared_output[index + i] = twiddle;
        }
        __syncthreads( ); // make sure all the shared mem is initialized to the starting value. There should be no contention as every thread is working on its own block of memory.

        for ( unsigned int sub_fft = 1; sub_fft < Q; sub_fft++ ) {
            // wrap around, 0 --> 1, Q-1 --> 0 etc.
            index = ((threadIdx.x + sub_fft) % Q) * size_of<FFT>::value;
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                SINCOS(twiddle_in * (index + i), &twiddle.y, &twiddle.x);
                twiddle *= thread_data[i];
                atomicAdd_block(&shared_output[index + i].x, twiddle.x);
                atomicAdd_block(&shared_output[index + i].y, twiddle.y);
            }
        }
        __syncthreads( );
    }

    static inline __device__ void load_c2r(const complex_type* input,
                                           complex_type*       thread_data,
                                           const int           stride,
                                           const int           memory_limit) {
        // Each thread reads in the input data at stride = mem_offsets.Q
        unsigned int index  = threadIdx.x;
        unsigned int offset = 2 * memory_limit - 2;
        for ( unsigned int i = 0; i < size_of<FFT>::value; i++ ) {
            if ( index < memory_limit ) {
                thread_data[i] = input[index];
            }
            else {
                // assuming even dimension
                // FIXME shouldn't need to read in from global for an even stride
                thread_data[i]   = input[offset - index];
                thread_data[i].y = -thread_data[i].y; // conjugate
            }
            index += stride;
        }
    }

    // FIXME as above
    static inline __device__ void load_c2r_transposed(const complex_type* input,
                                                      complex_type*       thread_data,
                                                      int                 stride,
                                                      int                 pixel_pitch,
                                                      int                 memory_limit) {
        // Each thread reads in the input data at stride = mem_offsets.Q
        unsigned int index = threadIdx.x;
        // unsigned int offset = 2*memory_limit - 2;
        for ( unsigned int i = 0; i < size_of<FFT>::value; i++ ) {
            if ( index < memory_limit ) {
                thread_data[i] = input[index * pixel_pitch];
            }
            else {
                // input[2*memory_limit - index - 2];
                // assuming even dimension
                // FIXME shouldn't need to read in from global for an even stride
                thread_data[i]   = input[(2 * memory_limit - index) * pixel_pitch];
                thread_data[i].y = -thread_data[i].y; // conjugate
            }
            index += stride;
        }
    }

    static inline __device__ void remap_decomposed_segments_c2r(const complex_type* thread_data,
                                                                scalar_type*        shared_output,
                                                                scalar_type         twiddle_in,
                                                                int                 Q) {
        // Unroll the first loop and initialize the shared mem.
        complex_type twiddle;
        int          index = threadIdx.x * size_of<FFT>::value;
        twiddle_in *= threadIdx.x; // twiddle factor arg now just needs to multiplied by K = (index + i)
        for ( unsigned int i = 0; i < size_of<FFT>::value; i++ ) {
            SINCOS(twiddle_in * (index + i), &twiddle.y, &twiddle.x);
            shared_output[index + i] = (twiddle.x * thread_data[i].x - twiddle.y * thread_data[i].y); // assuming the output is real, only the real parts add, so don't bother with the complex
        }
        __syncthreads( ); // make sure all the shared mem is initialized to the starting value. There should be no contention as every thread is working on its own block of memory.

        for ( unsigned int sub_fft = 1; sub_fft < Q; sub_fft++ ) {
            // wrap around, 0 --> 1, Q-1 --> 0 etc.
            index = ((threadIdx.x + sub_fft) % Q) * size_of<FFT>::value;

            for ( unsigned int i = 0; i < size_of<FFT>::value; i++ ) {
                // if (threadIdx.x == 32) printf("remap tid, subfft, q, index + i %i %i %i %i\n", threadIdx.x,sub_fft, Q, index+i);
                SINCOS(twiddle_in * (index + i), &twiddle.y, &twiddle.x);
                atomicAdd_block(&shared_output[index + i], twiddle.x * thread_data[i].x - twiddle.y * thread_data[i].y);
            }
        }
        __syncthreads( );
    }

    static inline __device__ void store_c2r(const scalar_type* shared_output,
                                            scalar_type*       output,
                                            const int          stride) {
        // Each thread reads in the input data at stride = mem_offsets.Q
        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < size_of<FFT>::value; i++ ) {
            output[index] = shared_output[index];
            index += stride;
        }
    }

    static inline __device__ void load_shared_and_conj_multiply(const complex_type* image_to_search,
                                                                const complex_type* shared_mem,
                                                                complex_type*       thread_data,
                                                                const int           stride) {
        unsigned int index = threadIdx.x;
        complex_type c;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            c.x = (shared_mem[index].x * image_to_search[index].x + shared_mem[index].y * image_to_search[index].y);
            c.y = (shared_mem[index].y * image_to_search[index].x - shared_mem[index].x * image_to_search[index].y);
            // a * conj b
            thread_data[i] = c; //ComplexConjMulAndScale<complex_type, scalar_type>(thread_data[i], image_to_search[index], 1.0f);
            index += stride;
        }
        __syncthreads( );
    }
}; // struct thread_io

} // namespace FastFFT

#endif // Fast_FFT_cuh_
