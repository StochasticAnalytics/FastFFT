#ifndef __INCLUDE_DETAIL_DEVICE_FUNCTIONS_H__
#define __INCLUDE_DETAIL_DEVICE_FUNCTIONS_H__

#include <cufftdx/include/cufftdx.hpp>
#include <cufftdx/include/operators/direction.hpp>

namespace FastFFT {

template <bool flag = false>
inline void static_assert_invalid_cufftdx_direction( ) { static_assert(flag, "static_assert_invalid_cufftdx_direction"); }

template <class FFT, typename T>
constexpr T _i2pi_div_P( ) {
    // This is the twiddle factor for the FULL xform, which uses float(-2.0 * pi_v<double> / double(transform_size.N));
    // We only know P = N/Q at compile time, but it still makes sense to calculate this.
    if constexpr ( cufftdx::direction_of<FFT>::value == cufftdx::fft_direction::forward ) {
        return T{-2.0 * pi_v<double> / double(cufftdx::size_of<FFT>::value)};
    }
    else if constexpr ( cufftdx::direction_of<FFT>::value == cufftdx::fft_direction::inverse ) {
        return T{2.0 * pi_v<double> / double(cufftdx::size_of<FFT>::value)};
    }
    else {
        static_assert_invalid_cufftdx_direction( );
    }
}

template <class FFT, typename T>
inline float _i2pi_div_N(float Q) {
    return float(_i2pi_div_P<FFT, T>( )) / Q;
}

// Complex a * conj b multiplication
template <typename ComplexType, typename ScalarType>
static __device__ __host__ inline auto ComplexConjMulAndScale(const ComplexType a, const ComplexType b, ScalarType s) -> decltype(b) {

    ComplexType c;
    c.x = s * (a.x * b.x + a.y * b.y);
    c.y = s * (a.y * b.x - a.x * b.y);
    return c;
}

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

__device__ __forceinline__ unsigned int get_lane_id( ) {
    unsigned ret;
    asm volatile("mov.u32 %0, %laneid;"
                 : "=r"(ret));
    return ret;
}

/**
 * @brief Get the warp id from the special register. WARNING: This is not necessarily the same as threadIdx.x % warpSize
 * 
 * This value can change from instruction issue to issue, but is usefule for debugging
 * 
 * @return __device__ 
 */
__device__ __forceinline__ unsigned int get_warp_id( ) {
    unsigned ret;
    asm volatile("mov.u32 %0, %warpid;"
                 : "=r"(ret));
    return ret;
}

// FIXME: warp size should be deined based on arch at comiple time at constants.h
constexpr unsigned int replaceme_warpSize        = 32;
constexpr unsigned int replaceme_warpSize_minus1 = 31;

/**
 * @brief Returns the lane id of the current thread. Calculated as threadIdx.x & (warpSize - 1)
 * 
 * This should be the same as threadIdx.x % warpSize
 * 
 * @return __device__ 
 */
__device__ __forceinline__ unsigned int calc_lane_id( ) {

    return threadIdx.x & replaceme_warpSize_minus1;
}

__device__ __forceinline__ int
d_ReturnReal1DAddressFromPhysicalCoord(int3 coords, short4 img_dims) {
    return ((((int)coords.z * (int)img_dims.y + coords.y) * (int)img_dims.w * 2) + (int)coords.x);
}

// constexpr bool allow_c2c_cache_sets = false;
// constexpr bool allow_c2r_cache_sets = false;
// constexpr bool allow_r2c_cache_sets = false;

// // Wrapper that allows cleaner ifdefing
// template <class FFT>
// void SetCudaFuncCache(const void* func, cudaFuncCache wantedFuncCache) {
//     using namespace cufftdx;

//     if constexpr ( type_of<FFT>::value == fft_type::c2c && allow_c2c_cache_sets )
//         cudaErr(cudaFuncSetCacheConfig(func, wantedFuncCache));
//     else if constexpr ( type_of<FFT>::value == fft_type::c2r && allow_c2r_cache_sets )
//         cudaErr(cudaFuncSetCacheConfig(func, wantedFuncCache));
//     else if constexpr ( type_of<FFT>::value == fft_type::r2c && allow_r2c_cache_sets )
//         cudaErr(cudaFuncSetCacheConfig(func, wantedFuncCache));
// };

} // namespace FastFFT

#endif // __INCLUDE_DETAIL_DEVICE_FUNCTIONS_H__