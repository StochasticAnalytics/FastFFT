#ifndef __INCLUDE_DETAIL_CONCEPTS_H__
#define __INCLUDE_DETAIL_CONCEPTS_H__

#include <type_traits>
#include "functors.h"

namespace FastFFT {

// To limit which kernels are instantiated, define a set of constants for the FFT method to be used at compile time.
constexpr int Generic_Fwd_FFT           = 1;
constexpr int Generic_Inv_FFT           = 2;
constexpr int Generic_Fwd_Image_Inv_FFT = 3;

template <bool, typename T = void>
struct EnableIfT {};

template <typename T>
struct EnableIfT<true, T> { using Type = T; };

template <bool cond, typename T = void>
using EnableIf = typename EnableIfT<cond, T>::Type;

template <typename IntraOpType>
constexpr bool HasIntraOpFunctor = IS_IKF_t<IntraOpType>( );

template <typename IntraOpType, int FFT_ALGO_t>
constexpr bool IfAppliesIntraOpFunctor_HasIntraOpFunctor = (FFT_ALGO_t != Generic_Fwd_Image_Inv_FFT || (FFT_ALGO_t == Generic_Fwd_Image_Inv_FFT && HasIntraOpFunctor<IntraOpType>));

} // namespace FastFFT

template <typename T>
constexpr bool IsComplexType = (std::is_same_v<T, float2> || std::is_same_v<T, __half2>);

template <typename... Args>
constexpr bool IsPointerOrNullPtrType = (... && (std::is_same<Args, nullptr_t>::value || std::is_pointer_v<std::decay_t<Args>>));

template <typename... Args>
constexpr bool IsAllowedRealType = (... && (std::is_same_v<Args, __half> || std::is_same_v<Args, float>));

template <typename... Args>
constexpr bool IsAllowedComplexType = (... && (std::is_same_v<Args, __half2> || std::is_same_v<Args, float2>));

template <typename... Args>
constexpr bool IsAllowedInputType = (... && (std::is_same_v<Args, __half> || std::is_same_v<Args, float> || std::is_same_v<Args, __half2> || std::is_same_v<Args, float2>));

#endif // __INCLUDE_DETAIL_CONCEPTS_H__