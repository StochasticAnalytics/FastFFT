// Utilites for FastFFT.cu that we don't need the host to know about (FastFFT.h)
#include "FastFFT.h"

#ifndef __INCLUDE_FAST_FFT_CUH__
#define __INCLUDE_FAST_FFT_CUH__

// #define USE_FOLDED_R2C_C2R
#define USE_FOLDED_C2R
#define C2R_BUFFER_LINES


// cudaErr(cudaFuncSetSharedMemConfig((const void*)block_fft_kernel_C2R_DECREASE_XY<FFT, data_buffer_t, data_io_t>, cudaSharedMemBankSizeEightByte));

#include "detail/detail.cuh"
#include <cuda_fp16.h>

// “This software contains source code provided by NVIDIA Corporation.”
// This is located in include/cufftdx*
// Please review the license in the cufftdx directory.

// #define forceforce( type )  __nv_is_extended_device_lambda_closure_type( type )
//FIXME: change to constexpr func

namespace FastFFT {

template <bool flag = false>
inline void static_assert_invalid_loop_limit( ) { static_assert(flag, "loop limit lt 1"); }

using namespace cufftdx;

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

template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_R2C_NONE_XY(const InputData_t* __restrict__ input_values,
                                          OutputData_t* __restrict__ output_values,
                                          Offsets                      mem_offsets,
                                          typename FFT::workspace_type workspace);

// XZ_STRIDE ffts/block via threadIdx.x, notice launch bounds. Creates partial coalescing.
template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_R2C_NONE_XZ(const InputData_t* __restrict__ input_values,
                                          OutputData_t* __restrict__ output_values,
                                          Offsets                      mem_offsets,
                                          typename FFT::workspace_type workspace);

template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_R2C_INCREASE_XY(const InputData_t* __restrict__ input_values,
                                              OutputData_t* __restrict__ output_values,
                                              Offsets                      mem_offsets,
                                              float                        twiddle_in,
                                              const int                    Q,
                                              const int                    SignalLength,
                                              typename FFT::workspace_type workspace);

// XZ_STRIDE ffts/block via threadIdx.x, notice launch bounds. Creates partial coalescing.
template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_R2C_INCREASE_XZ(const InputData_t* __restrict__ input_values,
                                              OutputData_t* __restrict__ output_values,
                                              Offsets                      mem_offsets,
                                              float                        twiddle_in,
                                              int                          Q,
                                              typename FFT::workspace_type workspace);

// __launch_bounds__(FFT::max_threads_per_block)  we don't know this because it is threadDim.x * threadDim.z - this could be templated if it affects performance significantly
template <class FFT, class InputData_t, class OutputData_t>
__global__ void block_fft_kernel_R2C_DECREASE_XY(const InputData_t* __restrict__ input_values,
                                                 OutputData_t* __restrict__ output_values,
                                                 Offsets                      mem_offsets,
                                                 float                        twiddle_in,
                                                 int                          Q,
                                                 typename FFT::workspace_type workspace);

/////////////
// C2C
/////////////

template <class FFT, class ComplexData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_INCREASE(const ComplexData_t* __restrict__ input_values,
                                           ComplexData_t* __restrict__ output_values,
                                           Offsets                      mem_offsets,
                                           float                        twiddle_in,
                                           int                          Q,
                                           const unsigned int           SignalLength,
                                           typename FFT::workspace_type workspace);

// __launch_bounds__(FFT::max_threads_per_block)  we don't know this because it is threadDim.x * threadDim.z - this could be templated if it affects performance significantly
template <class FFT, class ComplexData_t>
__global__ void block_fft_kernel_C2C_DECREASE(const ComplexData_t* __restrict__ input_values,
                                              ComplexData_t* __restrict__ output_values,
                                              Offsets                      mem_offsets,
                                              float                        twiddle_in,
                                              int                          Q,
                                              typename FFT::workspace_type workspace);

template <class FFT, class ComplexData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_WithPadding_SwapRealSpaceQuadrants(const ComplexData_t* __restrict__ input_values,
                                                                     ComplexData_t* __restrict__ output_values,
                                                                     Offsets                      mem_offsets,
                                                                     float                        twiddle_in,
                                                                     int                          Q,
                                                                     typename FFT::workspace_type workspace);

template <class ExternalImage_t, class FFT, class invFFT, class ComplexData_t, class PreOpType, class IntraOpType, class PostOpType>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_FWD_INCREASE_OP_INV_NONE(const ExternalImage_t* __restrict__ image_to_search,
                                                           const ComplexData_t* __restrict__ input_values,
                                                           ComplexData_t* __restrict__ output_values,
                                                           Offsets                         mem_offsets,
                                                           typename FFT::workspace_type    workspace_fwd,
                                                           typename invFFT::workspace_type workspace_inv,
                                                           PreOpType                       pre_op_functor,
                                                           IntraOpType                     intra_op_functor,
                                                           PostOpType                      post_op_functor);

template <class ExternalImage_t, class FFT, class invFFT, class ComplexData_t>
__global__ void block_fft_kernel_C2C_FWD_NONE_INV_DECREASE_ConjMul(const ExternalImage_t* __restrict__ image_to_search,
                                                                   const ComplexData_t* __restrict__ input_values,
                                                                   ComplexData_t* __restrict__ output_values,
                                                                   Offsets                         mem_offsets,
                                                                   float                           twiddle_in,
                                                                   int                             Q,
                                                                   typename FFT::workspace_type    workspace_fwd,
                                                                   typename invFFT::workspace_type workspace_inv);

template <class FFT, class ComplexData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_NONE(const ComplexData_t* __restrict__ input_values,
                                       ComplexData_t* __restrict__ output_values,
                                       Offsets                      mem_offsets,
                                       typename FFT::workspace_type workspace);

template <class FFT, class ComplexData_t>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_NONE_XZ(const ComplexData_t* __restrict__ input_values,
                                          ComplexData_t* __restrict__ output_values,
                                          Offsets                      mem_offsets,
                                          typename FFT::workspace_type workspace);

template <class FFT, class ComplexData_t>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_NONE_XYZ(const ComplexData_t* __restrict__ input_values,
                                           ComplexData_t* __restrict__ output_values,
                                           Offsets                      mem_offsets,
                                           typename FFT::workspace_type workspace);

template <class FFT, class ComplexData_t>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_INCREASE_XYZ(const ComplexData_t* __restrict__ input_values,
                                               ComplexData_t* __restrict__ output_values,
                                               Offsets                      mem_offsets,
                                               float                        twiddle_in,
                                               int                          Q,
                                               typename FFT::workspace_type workspace);

/////////////
// C2R
/////////////

template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2R_NONE(const InputData_t* __restrict__ input_values,
                                       OutputData_t* __restrict__ output_values,
                                       Offsets                      mem_offsets,
                                       typename FFT::workspace_type workspace);

template <class FFT, unsigned int MAX_TPB, class InputData_t, class OutputData_t, unsigned int n_ffts = 1>
__launch_bounds__(MAX_TPB) __global__
        void block_fft_kernel_C2R_NONE_XY(const InputData_t* __restrict__ input_values,
                                          OutputData_t* __restrict__ output_values,
                                          Offsets                      mem_offsets,
                                          typename FFT::workspace_type workspace);

// __launch_bounds__(FFT::max_threads_per_block)  we don't know this because it is threadDim.x * threadDim.z - this could be templated if it affects performance significantly
template <class FFT, class InputData_t, class OutputData_t>
__global__ void block_fft_kernel_C2R_DECREASE_XY(const InputData_t* __restrict__ input_values,
                                                 OutputData_t* __restrict__ output_values,
                                                 Offsets                      mem_offsets,
                                                 float                        twiddle_in,
                                                 int                          Q,
                                                 typename FFT::workspace_type workspace);

template <class PositionSpaceType, class OutputBaseType>
__global__ void clip_into_top_left_kernel(PositionSpaceType* input_values,
                                          OutputBaseType*    output_values,
                                          const short4       dims);

// Modified from GpuImage::ClipIntoRealKernel
template <typename PositionSpaceType, typename OutputBaseType>
__global__ void clip_into_real_kernel(PositionSpaceType* real_values_gpu,
                                      OutputBaseType*    other_image_real_values_gpu,
                                      short4             dims,
                                      short4             other_dims,
                                      int3               wanted_coordinate_of_box_center,
                                      OutputBaseType     wanted_padding_value);

// TODO: This would be much cleaner if we could first go from complex_compute_t -> float 2 then do conversions
// I think since this would be a compile time decision, it would be fine, but it would be good to confirm.
template <class FFT, typename SetTo_t, typename GetFrom_t>
inline __device__ SetTo_t convert_if_needed(const GetFrom_t* __restrict__ ptr, const unsigned int idx) {
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    // For now, we are assuming (as everywhere else) that compute precision is never double
    // and may in the future be _half. But is currently only float.
    // FIXME: THis should be caught earlier I think.
    if constexpr ( std::is_same_v<complex_compute_t, double> ) {
        static_no_doubles( );
    }
    if constexpr ( std::is_same_v<complex_compute_t, __half> ) {
        static_no_half_support_yet( );
    }

    if constexpr ( std::is_same_v<std::decay_t<GetFrom_t>, complex_compute_t> || std::is_same_v<std::decay_t<GetFrom_t>, float2> ) {
        if constexpr ( std::is_same_v<SetTo_t, scalar_compute_t> || std::is_same_v<SetTo_t, float> ) {
            // In this case we assume we have a real valued result, packed into the first half of the complex array
            // TODO: think about cases where we may hit this block unintentionally and how to catch this
            return std::move(reinterpret_cast<const SetTo_t*>(ptr)[idx]);
        }
        else if constexpr ( std::is_same_v<SetTo_t, __half> ) {
            // In this case we assume we have a real valued result, packed into the first half of the complex array
            // TODO: think about cases where we may hit this block unintentionally and how to catch this
            return std::move(__float2half_rn(reinterpret_cast<const float*>(ptr)[idx]));
        }
        else if constexpr ( std::is_same_v<SetTo_t, __half2> ) {
            // Note: we will eventually need a similar hase for __nv_bfloat16
            // I think I may need to strip the const news for this to work
            if constexpr ( std::is_same_v<GetFrom_t, complex_compute_t> ) {
                return std::move(__floats2half2_rn((ptr[idx]).real( ), 0.f));
            }
            else {
                return std::move(__floats2half2_rn(static_cast<const float*>(ptr)[idx], 0.f));
            }
        }
        else if constexpr ( std::is_same_v<std::decay_t<GetFrom_t>, complex_compute_t> && std::is_same_v<std::decay_t<SetTo_t>, complex_compute_t> ) {
            // return std::move(static_cast<const SetTo_t*>(ptr)[idx]);
            float2 t = (reinterpret_cast<const float2*>(ptr)[idx]); // FIXME
            return std::move(complex_compute_t{t.x, t.y});
        }
        else if constexpr ( std::is_same_v<std::decay_t<GetFrom_t>, complex_compute_t> && std::is_same_v<std::decay_t<SetTo_t>, float2> ) {
            // return std::move(static_cast<const SetTo_t*>(ptr)[idx]);
            // return std::move(SetTo_t{(reinterpret_cast<const float2*>(ptr)[idx]).real( ), (reinterpret_cast<const float2*>(ptr)[idx]).imag( )});
            return std::move(SetTo_t{(reinterpret_cast<const float2*>(ptr)[idx]).x, (reinterpret_cast<const float2*>(ptr)[idx]).y});
        }

        else if constexpr ( std::is_same_v<std::decay_t<GetFrom_t>, float2> && std::is_same_v<std::decay_t<SetTo_t>, complex_compute_t> ) {
            // return std::move(static_cast<const SetTo_t*>(ptr)[idx]);
            return std::move(SetTo_t{(ptr[idx]).x, (ptr[idx]).y});
        }
        else if constexpr ( std::is_same_v<std::decay_t<GetFrom_t>, float2> && std::is_same_v<std::decay_t<SetTo_t>, float2> ) {
            // return std::move(static_cast<const SetTo_t*>(ptr)[idx]);
            return std::move((ptr[idx]));
        }
        else {
            static_no_match( );
        }
    }
    else if constexpr ( std::is_same_v<std::decay_t<GetFrom_t>, scalar_compute_t> || std::is_same_v<std::decay_t<GetFrom_t>, float> ) {
        if constexpr ( std::is_same_v<SetTo_t, scalar_compute_t> || std::is_same_v<SetTo_t, float> ) {
            // In this case we assume we have a real valued result, packed into the first half of the complex array
            // TODO: think about cases where we may hit this block unintentionally and how to catch this
            return std::move((static_cast<const SetTo_t*>(ptr)[idx]));
        }
        else if constexpr ( std::is_same_v<SetTo_t, __half> ) {
            // In this case we assume we have a real valued result, packed into the first half of the complex array
            // TODO: think about cases where we may hit this block unintentionally and how to catch this
            return std::move(__float2half_rn((static_cast<const float*>(ptr)[idx])));
        }
        else if constexpr ( std::is_same_v<SetTo_t, __half2> ) {
            // Here we assume we are reading a real value and placeing it in a complex array. Could this go sideways?
            return std::move(__floats2half2_rn((static_cast<const float*>(ptr)[idx]), 0.f));
        }
        else if constexpr ( std::is_same_v<std::decay_t<SetTo_t>, complex_compute_t> || std::is_same_v<std::decay_t<SetTo_t>, float2> ) {
            // Here we assume we are reading a real value and placeing it in a complex array. Could this go sideways?

            return std::move(SetTo_t{(static_cast<const float*>(ptr)[idx]), 0.f});
        }
        else {
            static_no_match( );
        }
    }
    else if constexpr ( std::is_same_v<std::decay_t<GetFrom_t>, __half> ) {
        if constexpr ( std::is_same_v<SetTo_t, scalar_compute_t> || std::is_same_v<SetTo_t, float> ) {
            return std::move(__half2float((ptr[idx])));
        }
        else if constexpr ( std::is_same_v<SetTo_t, __half> ) {
            // In this case we assume we have a real valued result, packed into the first half of the complex array
            // TODO: think about cases where we may hit this block unintentionally and how to catch this
            return std::move((static_cast<const SetTo_t*>(ptr)[idx]));
        }
        else if constexpr ( std::is_same_v<SetTo_t, __half2> ) {
            // Here we assume we are reading a real value and placeing it in a complex array. Could this go sideways?
            // FIXME: For some reason CUDART_ZERO_FP16 is not defined even with cuda_fp16.h included
            return std::move(__halves2half2((static_cast<const __half*>(ptr)[idx]), __ushort_as_half((unsigned short)0x0000U)));
        }
        else if constexpr ( std::is_same_v<std::decay_t<SetTo_t>, complex_compute_t> || std::is_same_v<std::decay_t<SetTo_t>, float2> ) {
            // Here we assume we are reading a real value and placeing it in a complex array. Could this go sideways?
            return std::move(SetTo_t{__half2float((static_cast<const __half*>(ptr)[idx])), 0.f});
        }
        else {
            static_no_match( );
        }
    }
    else if constexpr ( std::is_same_v<std::decay_t<GetFrom_t>, __half2> ) {
        if constexpr ( std::is_same_v<SetTo_t, scalar_compute_t> || std::is_same_v<SetTo_t, float> || std::is_same_v<SetTo_t, __half> ) {
            // In this case we assume we have a real valued result, packed into the first half of the complex array
            // TODO: think about cases where we may hit this block unintentionally and how to catch this
            return std::move((reinterpret_cast<const SetTo_t*>(ptr)[idx]));
        }
        else if constexpr ( std::is_same_v<SetTo_t, __half2> ) {
            // Here we assume we are reading a real value and placeing it in a complex array. Could this go sideways?
            // FIXME: For some reason CUDART_ZERO_FP16 is not defined even with cuda_fp16.h included
            return std::move((static_cast<const SetTo_t*>(ptr)[idx]));
        }
        else if constexpr ( std::is_same_v<std::decay_t<SetTo_t>, complex_compute_t> || std::is_same_v<std::decay_t<SetTo_t>, float2> ) {
            // Here we assume we are reading a real value and placeing it in a complex array. Could this go sideways?
            return std::move(SetTo_t{__low2float((static_cast<const __half2*>(ptr)[idx])), __high2float((static_cast<const __half2*>(ptr)[idx]))});
        }
        else {
            static_no_match( );
        }
    }
    else {
        static_no_match( );
    }
}

template <bool flag = false>
inline void static_assert_allowed_read_size( ) { static_assert(flag, "static_assert_allowed_read_size"); }

// Allows for testing on smaller arrays for easier readouts.
#define DEBUG_WARP_TILER

template <class FFT, unsigned int max_threads_per_block = FFT::max_threads_per_block, unsigned int n_coalesced_ffts = 1>
struct WarpTiler {
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    // Drafting for C2R XY specifically

    // Describe in terms of 2D as 3D is handled as a stack of 2D

    // Some steps are substantially simplified if all threads that are producers are also consumers. This means - whoops only half the threads will be consumers then
    // we need blockDim.x >= 32, and with EPT restricted to pow 2, min size is 64
    // For these smaller arrays, L2 cache hits are far more likely, can be encouraged further > 8.0 or alternatively handled
    // by a intra-block smem transpose if the thread fft api is used. In any case, for the current project, we mostly care about large
    // FFts anyway

    // We have an array of M complex valued FFTs, all with their 0th index along the fast dimension (for this example)
    // The FFT dimension is N and for C2R in a natural layout we have N/2 + 1 complex inputs per FFT
    // blockDim.x is fixed to be N / ept (FFT::elements_per_thread) = FFT::stride
    // For C2R we have ept/2+1 = input_ept, and have at most that many read cycles/FFT, while many threads are excluded on the last read (We may be able to fix that with this method)
    // Each read cycle has blockDim.x reads of contigous threads and each thread has an array T[FFT::storage_size] where the data is expected to be T[i] = data[tid.x + i*FFT::stride]

    // For block methods _XY the data are transposed in memory and so the reads are not coalesced
    // Normally, we have gridDim.y blocks, but we reduce that here by n_coalesced_ffts

    // We read in values in 4byte / thread to maximized coalescing and also b/c I think that is all a warp shuffle supports (though I could be wrong on that.)

    // The goal is to have n_coalesced_ffts as large as possible without killing occupancy due to register pressure

    // each read cycle, a warp will be have to read n_coalseced_ffts * 2 4byte chunks leaving warp_size (32) / n_coalseced_ffts * 2 complex elements read for each FFT

    // We require that blockDim.x = FFT::stride = n_expected reads / cycle be >= this value. EPT is restricted to a power of 2, so if it is > than it must be so by at least 2 and it will be a nice whole number

    // FIXME: need to add checks on data layout (RIRI vs RRII)
    // FIXME: need to add checks on data sizes and allow more flexiblity
    static constexpr unsigned int read_bytes = 4;
    static_assert(sizeof(complex_compute_t) == 2 * read_bytes, "warp_tiler requires complex_compute_t to be 8 bytes atm.");
    static_assert(sizeof(scalar_compute_t) == read_bytes, "warp_tiler requires scalar_compute_t to be 4 bytes atm.");

    static constexpr unsigned int expected_reads_per_cycle = FFT::stride;
    static constexpr unsigned int warp_size                = 32; // I think this is true for all arch as of 2024 FIXME: confirm

    // overkill for now, but for handling half later
    static constexpr unsigned int n_4byte_reads_per_complex_element = sizeof(scalar_compute_t) == read_bytes ? 2 : 0;

    static_assert(n_4byte_reads_per_complex_element != 0, "static_assert_allowed_read_size");

    static constexpr bool         strided_XY           = true; // for later configuration;
    static constexpr unsigned int tile_thread_dim_fast = n_coalesced_ffts * n_4byte_reads_per_complex_element;
#ifdef DEBUG_WARP_TILER
    static constexpr unsigned int tile_thread_dim_strided = std::min(expected_reads_per_cycle, warp_size) / tile_thread_dim_fast;
#else
    static constexpr unsigned int tile_thread_dim_strided = warp_size / tile_thread_dim_fast;
    static_assert(expected_reads_per_cycle >= warp_size, "Min blockDim.x 32 for this optimization");
#endif

    static constexpr unsigned int n_tile_reads_per_cycle = expected_reads_per_cycle / tile_thread_dim_strided;

    // Some threads may not have a valid read in a given cycle, but we other threads won't know that.
    // Even a thread without a valid read on the tile, may be a consumer through a warp shuffle, so we can't exclude it with logic.
    // Instead, use a set value to communicate this
    static constexpr float invalid_read_v =
            sizeof(scalar_compute_t) == read_bytes ? scalar_compute_t{-std::numeric_limits<float>::max( )} : 0;
    static_assert(invalid_read_v != 0, "invalid invalid_read_v");

    const unsigned int lane_idx; // We may have threads coming from x/y so we get the lane ide from special reg, which could be a tiny bit slower, but more dependible than : threadIdx.x & 31;

    // This should be safe b/c if blockDim.x >= 32 we should have blockDim.y = 1, For now, I have an assert prior to the kernel launch to save overhead
    // MyFFTDebugAssertTestTrue(blockDim.x* blockDim.y == 0 || blockDim.x * blockDim.y == 32, "Incorrect partitioning of x/y threads for warp tiler.");
    const unsigned int tile_idx = threadIdx.x / warp_size;

    // TOOD: rename and make invertable
    const unsigned int logical_idx_in_tile_strided = lane_idx / tile_thread_dim_fast;
    const unsigned int logical_idx_in_tile_fast    = lane_idx % tile_thread_dim_fast;

    // in the normal approach this will be threadIdx.x + i*FFT::stride (which we'll need to define the consumer thread)
    // I think this will be at most 2 instuctions IMAD, but maybe it is better to calc an intial value and increment it
    __device__ __forceinline__ unsigned int data_index_to_read_1d(const unsigned int i, const unsigned int i_read) {
        if constexpr ( FFT::stride >= 32 )
            return (logical_idx_in_tile_strided + tile_idx * warp_size) + i_read * tile_thread_dim_strided + i * FFT::stride;
        else
            return logical_idx_in_tile_strided + i_read * tile_thread_dim_strided + i * FFT::stride;
    }

    __device__ __forceinline__ unsigned int data_index_to_read_1d_shared( ) {
        if constexpr ( FFT::stride >= 32 )
            return logical_idx_in_tile_strided + threadIdx.y * tile_thread_dim_strided + tile_idx * warp_size * blockDim.y;
        else
            return logical_idx_in_tile_strided + threadIdx.y * tile_thread_dim_strided;
    }

    __device__ __forceinline__ unsigned int thread_data_linear_idx_real_part(const unsigned int i) {
        // The consumer's i_fft (index of coalesced FFT) is just logical_idx_in_tile_fast (FIXME, may be C2R XY specific here)
        return (n_4byte_reads_per_complex_element * FFT::storage_size * logical_idx_in_tile_fast / 2) + n_4byte_reads_per_complex_element * i;
    }

    static __device__ __forceinline__ unsigned int thread_data_idx_real_part_fft_0(const unsigned int i) {
        return n_4byte_reads_per_complex_element * i;
    }

    static constexpr unsigned int thread_data_fft_stride = n_4byte_reads_per_complex_element * FFT::storage_size;

    // This can't be right, which means data_index_to_read_1d must be wrong
    __device__ __forceinline__ int get_producer_thread_tile_index_y(const unsigned int i_read) {
        // A consumer thread will be wanting the data from input_data[threadIdx.x + i*FFT::stride]
        // The producer thread that fetched this index was (logical_idx_in_tile_strided + tile_idx * warp_size) +  i_read * n_tile_reads_per_cycle + i * FFT::stride;
        // Solving for logical_idx_in_tile_strided = threadIdx.x - tile_idx * warp_size - i_read * n_tile_reads_per_cycle

        // This will only be a valid index for a return value >= 0 < n_tile_reads_per_cycle which determines if we are an active consumer thread
        if constexpr ( FFT::stride >= 32 )
            return threadIdx.x - tile_idx * warp_size - i_read * tile_thread_dim_strided;
        else
            return threadIdx.x - i_read * tile_thread_dim_strided;
    }

    __device__ __forceinline__ unsigned int get_my_fft_idx( ) {
        return logical_idx_in_tile_fast / 2;
    }

    // the real part will be at the even (logical_idx_in_tile_fast, thread_tile_index_y)
    // The imaginary part will just be + 1
    __device__ __forceinline__ unsigned int get_producer_thread_re_lane_idx(const unsigned int i_read) {
        int producer_thread_tile_index_y = get_producer_thread_tile_index_y(i_read);
        if ( producer_thread_tile_index_y < 0 || producer_thread_tile_index_y >= tile_thread_dim_strided ) {
            return warp_size;
        }
        else {
            // just a conversion from a 2d index in the tile to a 1d index == producers lane
            return producer_thread_tile_index_y * tile_thread_dim_fast;
        }
    }

    __device__ WarpTiler(const unsigned int lane_idx) : lane_idx{lane_idx} { };

    // Now every thread gets the producer value
    // Every thread reads with a shfl_sync to a register copied value
    // if (producer lane < warp_size) -> write to thread array
    // maybe warpsync?
    // Every thread reads again, but + 1
    // if (producer lane)
};

template <class FFT, unsigned int max_threads_per_block = FFT::max_threads_per_block, unsigned int n_coalesced_ffts = 1>
struct io {

    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    /*  For dealing with R2C and C2R we previously included all of the following code in each method:
            constexpr unsigned int threads_per_fft        = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
            constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
            // threads_per_fft == 1 means that EPT == SIZE, so we need to store one more element
            constexpr unsigned int values_left_to_store =
                    threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
        Since we limit FFTs to be power of 2 and also ept, as long as size >= ept the calculation will always yield 1
        Then if ( threadIdx.x  == 0 )     = if ( threadIDx.x == 0 )

        This means that for N x N/2+1 transform, we have N write ops where only 1 thread / warp is writing.
        We could try to aggregate these values in shared and then write out, but this would depend on the number of blocks
        fitting into an SM, and then one of those blocks waiting for all to finish before writing out. I can imagine this ending up 
        being a pessimation in addition to complicating the code.

        FIXME: that if we use the cufftdx::packed during the transform, we could efficiently handle the 0 and N/2 partial transforms at stage 2 
        if we know that stage 1 was r2c and similarly between stage 6 and stage 7. 
    */
    static_assert(size_of<FFT>::value >= FFT::elements_per_thread, "FFT size must be greater than elements per thread");
    static_assert(real_fft_mode_of<FFT>::value == real_mode::normal || real_fft_mode_of<FFT>::value == real_mode::folded,
                  "R2C or C2Rs may only be normal or folded, packed layout is not supported");

    // R2C/C2R can be normal (full), folded (packed into an n/2 + 1 complex) or packed (packed into a n/2 with the real part of the last in the imag part of the first)
    // This *ONLY* works for 1D transforms

    template <typename data_io_t>
    static inline __device__ void load_r2c(const data_io_t*   input,
                                           complex_compute_t* thread_data) {

        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            thread_data[i] = convert_if_needed<FFT, complex_compute_t>(input, index);
            index += FFT::stride;
        }
    }

    static inline __device__ void store_r2c(const complex_compute_t* __restrict__ thread_data,
                                            complex_compute_t* __restrict__ output,
                                            int offset) {

        unsigned int index = offset + threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            output[index] = thread_data[i];
            index += FFT::stride;
        }
        if ( threadIdx.x == 0 ) {
            output[index] = thread_data[FFT::elements_per_thread / 2];
        }
    }

    // Since we can make repeated use of the same shared memory for each sub-fft
    // we use this method to load into shared mem instead of directly to registers
    // TODO set this up for async mem load
    static inline __device__ void load_shared(const complex_compute_t* __restrict__ input,
                                              complex_compute_t* __restrict__ shared_input,
                                              complex_compute_t* __restrict__ thread_data,
                                              float* __restrict__ twiddle_factor_args,
                                              float twiddle_in,
                                              int*  input_map,
                                              int* __restrict__ output_map,
                                              int Q) {

        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            input_map[i]           = index;
            output_map[i]          = Q * index;
            twiddle_factor_args[i] = twiddle_in * index;
            thread_data[i]         = input[index];
            shared_input[index]    = thread_data[i];
            index += FFT::stride;
        }
    }

    // Since we can make repeated use of the same shared memory for each sub-fft
    // we use this method to load into shared mem instead of directly to registers
    // TODO set this up for async mem load
    template <typename data_io_t>
    static inline __device__ void load_shared(const data_io_t* __restrict__ input,
                                              complex_compute_t* __restrict__ shared_input,
                                              complex_compute_t* __restrict__ thread_data,
                                              float* __restrict__ twiddle_factor_args,
                                              float              twiddle_in,
                                              const unsigned int SignalLength) {

        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            twiddle_factor_args[i] = twiddle_in * index;
            if ( index < SignalLength )
                thread_data[i] = convert_if_needed<FFT, complex_compute_t>(input, index);
            else
                thread_data[i] = complex_compute_t{0, 0};
            shared_input[index] = thread_data[i];
            index += FFT::stride;
        }
    }

    static inline __device__ void load_shared(const complex_compute_t* __restrict__ input,
                                              complex_compute_t* __restrict__ shared_input,
                                              complex_compute_t* __restrict__ thread_data,
                                              float* __restrict__ twiddle_factor_args,
                                              float twiddle_in,
                                              int* __restrict__ input_map,
                                              int* __restrict__ output_map,
                                              int Q,
                                              int number_of_elements) {

        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            if ( index < number_of_elements ) {
                input_map[i]           = index;
                output_map[i]          = Q * index;
                twiddle_factor_args[i] = twiddle_in * index;
                thread_data[i]         = input[index];
                shared_input[index]    = thread_data[i];
                index += FFT::stride;
            }
            else {
                input_map[i] = -9999; // ignore this in subsequent ops
            }
        }
    }

    // Since we can make repeated use of the same shared memory for each sub-fft
    // we use this method to load into shared mem instead of directly to registers
    // TODO set this up for async mem load - alternatively, load to registers then copy but leave in register for firt compute
    template <typename data_io_t>
    static inline __device__ void load_r2c_shared(const data_io_t* __restrict__ input,
                                                  scalar_compute_t* __restrict__ shared_input,
                                                  complex_compute_t* __restrict__ thread_data,
                                                  float* __restrict__ twiddle_factor_args,
                                                  float twiddle_in,
                                                  int* __restrict__ input_map,
                                                  int* __restrict__ output_map,
                                                  int          Q,
                                                  unsigned int SignalLength) {

        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {

            input_map[i]           = index;
            output_map[i]          = Q * index;
            twiddle_factor_args[i] = twiddle_in * index;
            if ( index < SignalLength ) {
                thread_data[i] = convert_if_needed<FFT, complex_compute_t>(input, index);
            }
            else {
                thread_data[i] = complex_compute_t{0, 0};
            }
            shared_input[index] = thread_data[i].x;

            index += FFT::stride;
        }
    }

    // Since we can make repeated use of the same shared memory for each sub-fft
    // we use this method to load into shared mem instead of directly to registers
    // TODO set this up for async mem load - alternatively, load to registers then copy but leave in register for firt compute
    template <typename data_io_t>
    static inline __device__ void load_r2c_shared(const data_io_t* __restrict__ input,
                                                  scalar_compute_t* __restrict__ shared_input,
                                                  complex_compute_t* __restrict__ thread_data,
                                                  float* __restrict__ twiddle_factor_args,
                                                  float     twiddle_in,
                                                  const int SignalLength) {

        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            twiddle_factor_args[i] = twiddle_in * index;
            if ( index < SignalLength )
                thread_data[i] = convert_if_needed<FFT, complex_compute_t>(input, index);
            else
                thread_data[i] = complex_compute_t{0, 0};
            shared_input[index] = thread_data[i].x;
            index += FFT::stride;
        }
    }

    template <typename data_io_t>
    static inline __device__ void load_r2c_shared_and_pad(const data_io_t* __restrict__ input,
                                                          complex_compute_t* __restrict__ shared_mem) {

        unsigned int index = threadIdx.x + (threadIdx.y * size_of<FFT>::value);
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            shared_mem[GetSharedMemPaddedIndex(index)] = convert_if_needed<FFT, complex_compute_t>(input, index);
            index += FFT::stride;
        }
        __syncthreads( );
    }

    static inline __device__ void copy_from_shared(const complex_compute_t* __restrict__ shared_mem,
                                                   complex_compute_t* __restrict__ thread_data,
                                                   const unsigned int Q) {
        const unsigned int stride = FFT::stride * Q; // I think the Q is needed, but double check me TODO
        unsigned int       index  = (threadIdx.x * Q) + threadIdx.y;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            thread_data[i] = shared_mem[GetSharedMemPaddedIndex(index)];
            index += stride;
        }
        __syncthreads( ); // FFT().execute is setup to reuse the shared mem, so we need to sync here. Optionally, we could allocate more shared mem and remove this sync
    }

    // Note that unlike most functions in this file, this one does not have a
    // const decorator on the thread mem, as we want to modify it with the twiddle factors
    // before reducing the full shared mem space.
    static inline __device__ void reduce_block_fft(complex_compute_t* __restrict__ thread_data,
                                                   complex_compute_t* __restrict__ shared_mem,
                                                   const float        twiddle_in,
                                                   const unsigned int Q) {

        unsigned int      index = threadIdx.x + (threadIdx.y * size_of<FFT>::value);
        complex_compute_t twiddle;
        // In the first loop, all threads participate and write back to natural order in shared memory
        // while also updating with the full size twiddle factor.
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // ( index * threadIdx.y) == ( k % P * n2 )
            SINCOS(twiddle_in * (index * threadIdx.y), &twiddle.y, &twiddle.x);
            thread_data[i] *= twiddle;

            shared_mem[GetSharedMemPaddedIndex(index)] = thread_data[i];
            index += FFT::stride;
        }
        __syncthreads( );

        // Now we reduce the shared memory into the first block of size P
        // Reuse index
        for ( index = 2; index <= Q; index *= 2 ) {
            // Some threads drop out each loop
            if ( threadIdx.y % index == 0 ) {
                for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                    thread_data[i] += shared_mem[GetSharedMemPaddedIndex(threadIdx.x + (i * FFT::stride) + (index / 2 * size_of<FFT>::value))];
                }
            } // end if condition
            // All threads can reach this point
            __syncthreads( );
        }
    }

    template <typename data_io_t>
    static inline __device__ void store_r2c_reduced(const complex_compute_t* __restrict__ thread_data,
                                                    data_io_t* __restrict__ output,
                                                    const unsigned int pixel_pitch,
                                                    const unsigned int memory_limit) {
        if ( threadIdx.y == 0 ) {
            // Finally we write out the first size_of<FFT>::values to global

            unsigned int index = threadIdx.x;
            for ( unsigned int i = 0; i <= FFT::elements_per_thread / 2; i++ ) {
                if ( index < memory_limit ) {
                    // transposed index.
                    output[index * pixel_pitch + blockIdx.y] = convert_if_needed<FFT, data_io_t>(thread_data, i);
                }
                index += FFT::stride;
            }
        }
    }

    // when using lod_shared || load_r2c_shared, we need then copy from shared mem into the registers.
    // notice we still need the packed complex values for the xform.
    static inline __device__ void copy_from_shared(const scalar_compute_t* __restrict__ shared_input,
                                                   complex_compute_t* __restrict__ thread_data,
                                                   int* input_map) {
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            thread_data[i].x = shared_input[input_map[i]];
            thread_data[i].y = 0.0f;
        }
    }

    static inline __device__ void copy_from_shared(const complex_compute_t* __restrict__ shared_input_complex,
                                                   complex_compute_t* __restrict__ thread_data,
                                                   int* __restrict__ input_map) {
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            thread_data[i] = shared_input_complex[input_map[i]];
        }
    }

    static inline __device__ void copy_from_shared(const scalar_compute_t* __restrict__ shared_input,
                                                   complex_compute_t* __restrict__ thread_data) {

        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            thread_data[i].x = shared_input[index];
            thread_data[i].y = 0.0f;
            index += FFT::stride;
        }
    }

    static inline __device__ void copy_from_shared(const complex_compute_t* __restrict__ shared_input_complex,
                                                   complex_compute_t* __restrict__ thread_data) {

        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            thread_data[i] = shared_input_complex[index];
            index += FFT::stride;
        }
    }

    template <class ExternalImage_t>
    static inline __device__ void load_shared_and_conj_multiply(ExternalImage_t* __restrict__ image_to_search,
                                                                complex_compute_t* __restrict__ thread_data) {

        unsigned int      index = threadIdx.x;
        complex_compute_t c;
        if constexpr ( std::is_same_v<ExternalImage_t, __half2> ) {
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                c.x            = (thread_data[i].x * __low2float(image_to_search[index]) + thread_data[i].y * __high2float(image_to_search[index].y));
                c.y            = (thread_data[i].y * __low2float(image_to_search[index]) - thread_data[i].x * __high2float(image_to_search[index].y));
                thread_data[i] = c;
                index += FFT::stride;
            }
        }
        else if constexpr ( std::is_same_v<ExternalImage_t, float2> ) {
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                c.x            = (thread_data[i].x * image_to_search[index].x + thread_data[i].y * image_to_search[index].y);
                c.y            = (thread_data[i].y * image_to_search[index].x - thread_data[i].x * image_to_search[index].y);
                thread_data[i] = c;
                index += FFT::stride;
            }
        }
        else {
            static_assert_type_name(image_to_search);
        }
    }

    // TODO: set user lambda to default = false, then get rid of other load_shared
    template <typename ExternalImage_t, class FunctionType = std::nullptr_t>
    static inline __device__ void load_external_data(const ExternalImage_t* __restrict__ image_to_search,
                                                     complex_compute_t* __restrict__ thread_data,
                                                     FunctionType intra_op_functor = nullptr) {

        unsigned int index = threadIdx.x;
        if constexpr ( IS_IKF_t<FunctionType>( ) ) {
            if constexpr ( std::is_same_v<ExternalImage_t, __half2*> ) {
                for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                    intra_op_functor(thread_data[i].x, thread_data[i].y, __low2float(image_to_search[index]), __high2float(image_to_search[index]));
                    index += FFT::stride;
                }
            }
            else {
                for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                    intra_op_functor(thread_data[i].x, thread_data[i].y, image_to_search[index].x, image_to_search[index].y);
                    index += FFT::stride;
                }
            }
        }
        else {
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                // a * conj b
                thread_data[i] = thread_data[i], image_to_search[index];
                index += FFT::stride;
            }
        }
    }

    // Now we need send to shared mem and transpose on the way
    // TODO: fix bank conflicts later.
    static inline __device__ void transpose_r2c_in_shared_XZ(complex_compute_t* __restrict__ shared_mem,
                                                             complex_compute_t* __restrict__ thread_data) {

        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            shared_mem[threadIdx.y + index * XZ_STRIDE] = thread_data[i];
            index += FFT::stride;
        }
        if ( threadIdx.x == 0 ) {
            shared_mem[threadIdx.y + index * XZ_STRIDE] = thread_data[FFT::elements_per_thread / 2];
        }
        __syncthreads( );
    }

    // Now we need send to shared mem and transpose on the way
    // TODO: fix bank conflicts later.
    static inline __device__ void transpose_in_shared_XZ(complex_compute_t* __restrict__ shared_mem,
                                                         complex_compute_t* __restrict__ thread_data) {
        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // return (XZ_STRIDE*blockIdx.z + threadIdx.y) + (XZ_STRIDE*gridDim.z) * ( blockIdx.y + X * gridDim.y );
            // XZ_STRIDE == XZ_STRIDE
            shared_mem[threadIdx.y + index * XZ_STRIDE] = thread_data[i];
            index += FFT::stride;
        }
        __syncthreads( );
    }

    static inline __device__ void store_r2c_transposed_xz(const complex_compute_t* __restrict__ thread_data,
                                                          complex_compute_t* __restrict__ output) {

        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            output[Return1DFFTAddress_XZ_transpose(index)] = thread_data[i];
            index += FFT::stride;
        }
        if ( threadIdx.x == 0 ) {
            output[Return1DFFTAddress_XZ_transpose(index)] = thread_data[FFT::elements_per_thread / 2];
        }
        __syncthreads( );
    }

    // Store a transposed tile, made up of contiguous (full) FFTS
    template <typename data_io_t>
    static inline __device__ void store_r2c_transposed_xz_strided_Z(const complex_compute_t* __restrict__ shared_mem,
                                                                    data_io_t* __restrict__ output) {
        constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
        unsigned int           index                  = threadIdx.x + threadIdx.y * output_values_to_store;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            output[Return1DFFTAddress_XZ_transpose_strided_Z(index)] = convert_if_needed<FFT, data_io_t>(shared_mem, index);
            index += FFT::stride;
        }

        if ( threadIdx.x == 0 ) {
            output[Return1DFFTAddress_XZ_transpose_strided_Z(index)] = convert_if_needed<FFT, data_io_t>(shared_mem, index);
        }
        __syncthreads( );
    }

    // Store a transposed tile, made up of non-contiguous (strided partial) FFTS
    //
    template <typename data_io_t>
    static inline __device__ void store_r2c_transposed_xz_strided_Z(const complex_compute_t* __restrict__ shared_mem,
                                                                    data_io_t* __restrict__ output,
                                                                    const unsigned int Q,
                                                                    const unsigned int sub_fft) {
        constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
        unsigned int           index                  = threadIdx.x + threadIdx.y * output_values_to_store;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            output[Return1DFFTAddress_XZ_transpose_strided_Z(index, Q, sub_fft)] = convert_if_needed<FFT, data_io_t>(shared_mem, index);
            index += FFT::stride;
        }
        if ( threadIdx.x == 0 ) {
            output[Return1DFFTAddress_XZ_transpose_strided_Z(index, Q, sub_fft)] = convert_if_needed<FFT, data_io_t>(shared_mem, index);
        }
        __syncthreads( );
    }

    template <typename data_io_t>
    static inline __device__ void store_transposed_xz_strided_Z(const complex_compute_t* __restrict__ shared_mem,
                                                                data_io_t* __restrict__ output) {

        unsigned int index = threadIdx.x + threadIdx.y * cufftdx::size_of<FFT>::value;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            output[Return1DFFTAddress_XZ_transpose_strided_Z(index)] = convert_if_needed<FFT, data_io_t>(shared_mem, index);
            index += FFT::stride;
        }
        __syncthreads( );
    }

    template <typename data_io_t>
    static inline __device__ void store_r2c_transposed_xy(const complex_compute_t* __restrict__ thread_data,
                                                          data_io_t* __restrict__ output,
                                                          int pixel_pitch) {

        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            // output map is thread local, so output_MAP[i] gives the x-index in the non-transposed array and blockIdx.y gives the y-index
            output[index * pixel_pitch + blockIdx.y] = convert_if_needed<FFT, data_io_t>(thread_data, i);
            index += FFT::stride;
        }
        if ( threadIdx.x == 0 ) {
            output[index * pixel_pitch + blockIdx.y] = convert_if_needed<FFT, data_io_t>(thread_data, FFT::elements_per_thread / 2);
        }
    }

    template <typename data_io_t>
    static inline __device__ void store_r2c_transposed_xy(const complex_compute_t* __restrict__ thread_data,
                                                          data_io_t* __restrict__ output,
                                                          int* output_MAP,
                                                          int  pixel_pitch) {

        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            // output map is thread local, so output_MAP[i] gives the x-index in the non-transposed array and blockIdx.y gives the y-index
            output[Return1DFFTAddress_transpose_XY(output_MAP[i], pixel_pitch)] = convert_if_needed<FFT, data_io_t>(thread_data, i);
            // if (blockIdx.y == 32) printf("from store transposed %i , val %f %f\n", output_MAP[i], thread_data[i].x, thread_data[i].y);
        }
        if ( threadIdx.x == 0 ) {
            output[Return1DFFTAddress_transpose_XY(output_MAP[FFT::elements_per_thread / 2], pixel_pitch)] = convert_if_needed<FFT, data_io_t>(thread_data, FFT::elements_per_thread / 2);
        }
    }

    template <typename data_io_t>
    static inline __device__ void load_c2r(const data_io_t* __restrict__ input,
                                           complex_compute_t* __restrict__ thread_data) {

        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            thread_data[i] = convert_if_needed<FFT, complex_compute_t>(input, index);
            index += FFT::stride;
        }

        if ( threadIdx.x == 0 ) {
            thread_data[FFT::elements_per_thread / 2] = convert_if_needed<FFT, complex_compute_t>(input, index);
        }
    }

    template <typename data_io_t>
    static inline __device__ void load_c2r_transposed(const data_io_t* __restrict__ input,
                                                      complex_compute_t* __restrict__ thread_data,
                                                      const unsigned int pixel_pitch,
                                                      const unsigned int SignalLength = FFT::input_length) {
        // Note: for c2r FFT::input_ept = N/2+1 whether folded or not, so this block would also work for normal c2r it also allows us to specify a non-fft signal length at the cost of extra branching
        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::input_ept; i++ ) {
            if ( index < SignalLength ) {
                thread_data[i] = convert_if_needed<FFT, complex_compute_t>(input, Return1DFFTAddress_transpose_XY(index, pixel_pitch));
                index += FFT::stride;
            }
        }
    }

#if defined(C2R_BUFFER_LINES) && defined(USE_FOLDED_C2R)
    // EnableIf<real_fft_mode_of<FFT>::value == real_mode::folded>

    template <typename data_io_t>
    static inline __device__ void load_c2r_transposed_coalesced(const data_io_t* __restrict__ input,
                                                                complex_compute_t* __restrict__ thread_data,
                                                                complex_compute_t* __restrict__ smem_buffer,
                                                                const unsigned int pixel_pitch,
                                                                const unsigned int SignalLength = FFT::input_length) {

        // We normally have FFT::stride threads in x (blockDim.x) and nFFTs as gridDim.y
        // To help with coalescing, we add n_coalesced_ffts threads in y (blockDim.y) reduce gridDim.y by the same
        // To avoid bank conflicts we organize this 2d tile by tid.x x tid.y
        // 0 0 0 0 4 4 4 4   x
        // 1 1 1 1 5 5 5 5
        // 2 2 2 2 6 6 6 6
        // 3 3 3 3 7 7 7 7
        //
        // 0 1 2 3 0 1 2 3

        // unsigned int index = threadIdx.x;
        // for ( unsigned int i = 0; i < FFT::input_ept; i++ ) {
        //     if ( index < SignalLength ) {
        //         thread_data[i] = convert_if_needed<FFT, complex_compute_t>(input, Return1DFFTAddress_transpose_XY(index, pixel_pitch));
        //         index += FFT::stride;
        //     }
        // }/

        unsigned int index = threadIdx.x;
        // 0 1 2 3 0 1 2 3 + 4 * (0 0 0 0 1 1 1 1) = 0 1 2 3 4 5 6 7 ...
        unsigned int       x_prime = threadIdx.y + n_coalesced_ffts * (threadIdx.x / n_coalesced_ffts);
        const unsigned int fft_idx = threadIdx.x % n_coalesced_ffts; // change to fft_idx

        // n_coalesced_ffts will read the 0th index of each FFT, so we want to offset the write into smem so we write into banks 0 1 2 3
        // the next thread in this warp will be reading at the 4th index
        const unsigned int smem_idx = x_prime + fft_idx * FFT::stride;
        for ( unsigned int i = 0; i < FFT::input_ept; i++ ) {

            if ( x_prime < SignalLength )
                smem_buffer[smem_idx] = convert_if_needed<FFT, complex_compute_t>(input, x_prime * pixel_pitch + fft_idx + blockIdx.y * n_coalesced_ffts);
            __syncthreads( );

            if ( index < SignalLength )
                thread_data[i] = convert_if_needed<FFT, complex_compute_t>(smem_buffer, threadIdx.x + threadIdx.y * FFT::stride);
            __syncthreads( );

            x_prime += FFT::stride;
            index += FFT::stride;
        }
    }

#endif

    static inline __device__ void
    load_c2r_shared_and_pad(const complex_compute_t* __restrict__ input,
                            complex_compute_t* __restrict__ shared_mem,
                            const unsigned int pixel_pitch) {

        unsigned int index = threadIdx.x + (threadIdx.y * size_of<FFT>::value);
        for ( unsigned int i = 0; i < FFT::elements_per_thread / 2; i++ ) {
            shared_mem[GetSharedMemPaddedIndex(index)] = input[pixel_pitch * index];
            index += FFT::stride;
        }

        if ( threadIdx.x == 0 ) {
            shared_mem[GetSharedMemPaddedIndex(index)] = input[pixel_pitch * index];
        }
        __syncthreads( );
    }

    // this may benefit from asynchronous execution
    template <typename data_io_t>
    static inline __device__ void load(const data_io_t* __restrict__ input,
                                       complex_compute_t* __restrict__ thread_data) {

        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            thread_data[i] = convert_if_needed<FFT, complex_compute_t>(input, index);
            index += FFT::stride;
        }
    }

    //  TODO: set pre_op_functor to default=false and get rid of other load
    template <typename data_io_t, class FunctionType = std::nullptr_t>
    static inline __device__ void load(const data_io_t* __restrict__ input,
                                       complex_compute_t* __restrict__ thread_data,
                                       int          last_index_to_load,
                                       FunctionType pre_op_functor = nullptr) {

        unsigned int index = threadIdx.x;
        // FIXME: working out how to use these functors and this is NOT what is intended
        if constexpr ( IS_IKF_t<FunctionType>( ) ) {
            float2 temp;
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                if ( index < last_index_to_load ) {
                    temp           = pre_op_functor(convert_if_needed<FFT, float2>(input, index));
                    thread_data[i] = convert_if_needed<FFT, complex_compute_t>(&temp, 0);
                }
                else {
                    // thread_data[i] = complex_compute_t{0.0f, 0.0f};
                    temp           = pre_op_functor(float2{0.0f, 0.0f});
                    thread_data[i] = convert_if_needed<FFT, complex_compute_t>(&temp, 0);
                }

                index += FFT::stride;
            }
        }
        else {
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                if ( index < last_index_to_load )
                    thread_data[i] = convert_if_needed<FFT, complex_compute_t>(input, index);
                else
                    thread_data[i] = complex_compute_t{0.0f, 0.0f};
                index += FFT::stride;
            }
        }
    }

    static inline __device__ void store_and_swap_quadrants(const complex_compute_t* __restrict__ thread_data,
                                                           complex_compute_t* __restrict__ output,
                                                           int first_negative_index) {

        unsigned int      index = threadIdx.x;
        complex_compute_t phase_shift;
        int               logical_y;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // If no kernel based changes are made to source_idx, this will be the same as the original index value
            phase_shift = thread_data[i];
            logical_y   = index;
            if ( logical_y >= first_negative_index )
                logical_y -= 2 * first_negative_index;
            if ( (int(blockIdx.y) + logical_y) % 2 != 0 )
                phase_shift *= -1.f;
            output[index] = phase_shift;
            index += FFT::stride;
        }
    }

    static inline __device__ void store_and_swap_quadrants(const complex_compute_t* __restrict__ thread_data,
                                                           complex_compute_t* __restrict__ output,
                                                           int* __restrict__ source_idx,
                                                           int first_negative_index) {

        complex_compute_t phase_shift;
        int               logical_y;
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

    template <typename data_io_t, class FunctionType = std::nullptr_t>
    static inline __device__ void store(const complex_compute_t* __restrict__ thread_data,
                                        data_io_t* __restrict__ output,
                                        FunctionType post_op_functor = nullptr) {

        unsigned int index = threadIdx.x;
        if constexpr ( IS_IKF_t<FunctionType>( ) ) {
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                output[index] = post_op_functor(convert_if_needed<FFT, data_io_t>(thread_data, i));
                index += FFT::stride;
            }
        }
        else {
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                output[index] = convert_if_needed<FFT, data_io_t>(thread_data, i);
                index += FFT::stride;
            }
        }
    }

    template <typename data_io_t>
    static inline __device__ void store(const complex_compute_t* __restrict__ thread_data,
                                        data_io_t* __restrict__ output,
                                        const unsigned int Q,
                                        const unsigned int sub_fft) {

        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            output[index * Q + sub_fft] = convert_if_needed<FFT, data_io_t>(thread_data, i);
            index += FFT::stride;
        }
    }

    template <typename data_io_t>
    static inline __device__ void store_Z(const complex_compute_t* __restrict__ shared_mem,
                                          data_io_t* __restrict__ output) {

        unsigned int index = threadIdx.x + threadIdx.y * size_of<FFT>::value;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            output[Return1DFFTAddress_YZ_transpose_strided_Z(index)] = convert_if_needed<FFT, data_io_t>(shared_mem, index);

            index += FFT::stride;
        }
    }

    template <typename data_io_t>
    static inline __device__ void store_Z(const complex_compute_t* __restrict__ shared_mem,
                                          data_io_t* __restrict__ output,
                                          const unsigned int Q,
                                          const unsigned int sub_fft) {

        unsigned int index = threadIdx.x + threadIdx.y * size_of<FFT>::value;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            output[Return1DFFTAddress_YZ_transpose_strided_Z(index, Q, sub_fft)] = convert_if_needed<FFT, data_io_t>(shared_mem, index);
            index += FFT::stride;
        }
        __syncthreads( );
    }

    template <unsigned int memory_limit, typename data_io_t>
    static inline __device__ void store(const complex_compute_t* __restrict__ thread_data,
                                        data_io_t* __restrict__ output) {

        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {

            if ( index < memory_limit )
                output[index] = convert_if_needed<FFT, data_io_t>(thread_data, i);

            index += FFT::stride;
        }
    }

    template <typename data_io_t>
    static inline __device__ void store_subset(const complex_compute_t* __restrict__ thread_data,
                                               data_io_t* __restrict__ output,
                                               int* __restrict__ source_idx) {

        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // If no kernel based changes are made to source_idx, this will be the same as the original index value
            if ( source_idx[i] >= 0 )
                output[source_idx[i]] = convert_if_needed<FFT, data_io_t>(thread_data, i);
        }
    }

    template <typename data_io_t>
    static inline __device__ void store_coalesced(const complex_compute_t* __restrict__ shared_output,
                                                  data_io_t* __restrict__ global_output,
                                                  int offset) {

        unsigned int index = offset + threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            global_output[index] = convert_if_needed<FFT, data_io_t>(shared_output, index);
            index += FFT::stride;
        }
    }

    template <typename data_io_t>
    static inline __device__ void load_c2c_shared_and_pad(const data_io_t* __restrict__ input,
                                                          complex_compute_t* __restrict__ shared_mem) {

        unsigned int index = threadIdx.x + (threadIdx.y * size_of<FFT>::value);
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            shared_mem[GetSharedMemPaddedIndex(index)] = convert_if_needed<FFT, complex_compute_t>(input, index);
            index += FFT::stride;
        }
        __syncthreads( );
    }

    template <typename data_io_t>
    static inline __device__ void store_c2c_reduced(const complex_compute_t* __restrict__ thread_data,
                                                    data_io_t* __restrict__ output) {
        if ( threadIdx.y == 0 ) {
            // Finally we write out the first size_of<FFT>::values to global

            unsigned int index = threadIdx.x + (threadIdx.y * size_of<FFT>::value);
            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                if ( index < size_of<FFT>::value ) {
                    // transposed index.
                    output[index] = convert_if_needed<FFT, data_io_t>(thread_data, i);
                }
                index += FFT::stride;
            }
        }
    }

    template <typename data_io_t>
    static inline __device__ void store_c2r_reduced(const complex_compute_t* __restrict__ thread_data,
                                                    data_io_t* __restrict__ output) {

#ifdef USE_FOLDED_C2R
        static_assert(real_fft_mode_of<FFT>::value == real_mode::folded, "Folded C2R only supported for folded real FFTs");
        constexpr auto inner_loop_limit = sizeof(complex_compute_t) / sizeof(data_io_t);
        unsigned int   index            = (threadIdx.x * inner_loop_limit) + (threadIdx.y * size_of<FFT>::value);
        if ( threadIdx.y == 0 ) {
            for ( int i = 0; i < FFT::output_ept; ++i ) {
                for ( int j = 0; j < inner_loop_limit; ++j ) {
                    output[index + j] = convert_if_needed<FFT, data_io_t>(thread_data, i * inner_loop_limit + j);
                }
                index += inner_loop_limit * FFT::stride;
            }
        }

#else

        if ( threadIdx.y == 0 ) {
            // Finally we write out the first size_of<FFT>::values to global

            unsigned int index = threadIdx.x + (threadIdx.y * size_of<FFT>::value);

            for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
                if ( index < size_of<FFT>::value ) {
                    // transposed index.
                    output[index] = convert_if_needed<FFT, data_io_t>(thread_data, i);
                }
                index += FFT::stride;
            }
        }
#endif
    }

    template <typename data_io_t>
    static inline __device__ void store_transposed(const complex_compute_t* __restrict__ thread_data,
                                                   data_io_t* __restrict__ output,
                                                   int* __restrict__ output_map,
                                                   int* __restrict__ rotated_offset,
                                                   int memory_limit) {
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // If no kernel based changes are made to source_idx, this will be the same as the original index value
            if ( output_map[i] < memory_limit )
                output[rotated_offset[1] * output_map[i] + rotated_offset[0]] = convert_if_needed<FFT, data_io_t>(thread_data, i);
        }
    }

    template <typename data_io_t>
    static inline __device__
            EnableIf<IsAllowedRealType<data_io_t>>
            store_c2r(const complex_compute_t* __restrict__ thread_data,
                      data_io_t* __restrict__ output,
                      const unsigned int SignalLength = size_of<FFT>::value) {

#ifdef USE_FOLDED_C2R
        static_assert(real_fft_mode_of<FFT>::value == real_mode::folded, "Folded C2R only supported for folded real FFTs");

        // Currently convert_to<> assumes if we are going from a complex float to a scalar half, that we can contract only the real part.
        // Currently we assume we are always going to have an inner_loop limit of 2
        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::input_ept; i++ ) {
            if ( index < SignalLength / 2 ) {
                if constexpr ( std::is_same_v<data_io_t, __half> ) {
                    reinterpret_cast<__half2*>(output)[index] = __float22half2_rn(reinterpret_cast<const float2*>(thread_data)[i]);
                }
                else {
                    reinterpret_cast<float2*>(output)[index] = reinterpret_cast<const float2*>(thread_data)[i];
                }
            }
            index += FFT::stride;
        }
#else
        unsigned int index = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // TODO: does reinterpret_cast<const scalar_compute_t*>(thread_data)[i] make more sense than just thread_data[i].x??
            if ( index < SignalLength ) {
                output[index] = convert_if_needed<FFT, data_io_t>(thread_data, i);
                index += FFT::stride;
            }
        }
#endif
    }
};

} // namespace FastFFT

#endif // Fast_FFT_cuh_
