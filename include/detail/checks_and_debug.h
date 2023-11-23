#ifndef __INCLUDE_DETAILS_CHECKS_AND_DEBUG_H__
#define __INCLUDE_DETAILS_CHECKS_AND_DEBUG_H__

#include "types.h"

namespace FastFFT {
// hacky and non-conclusive way to trouble shoot mismatched types in function calls
template <typename T>
__device__ __host__ inline void static_assert_type_name(T v) {

    if constexpr ( std::is_pointer_v<T> ) {
        static_assert(! std::is_same_v<T, int*>, "int*");
        static_assert(! std::is_same_v<T, int2*>, "int2*");
        static_assert(! std::is_same_v<T, int3*>, "int3*");
        static_assert(! std::is_same_v<T, int4*>, "int4*");
        static_assert(! std::is_same_v<T, float*>, "float*");
        static_assert(! std::is_same_v<T, float2*>, "float2*");
        static_assert(! std::is_same_v<T, float3*>, "float3*");
        static_assert(! std::is_same_v<T, float4*>, "float4*");
        static_assert(! std::is_same_v<T, double*>, "double*");
        static_assert(! std::is_same_v<T, __half*>, "__half*");
        static_assert(! std::is_same_v<T, __half2*>, "__half2*");
        static_assert(! std::is_same_v<T, nullptr_t>, "nullptr_t");
    }
    else {
        static_assert(! std::is_same_v<T, int>, "int");
        static_assert(! std::is_same_v<T, int2>, "int2");
        static_assert(! std::is_same_v<T, int3>, "int3");
        static_assert(! std::is_same_v<T, int4>, "int4");
        static_assert(! std::is_same_v<T, float>, "float");
        static_assert(! std::is_same_v<T, float2>, "float2");
        static_assert(! std::is_same_v<T, float3>, "float3");
        static_assert(! std::is_same_v<T, float4>, "float4");
        static_assert(! std::is_same_v<T, double>, "double");
        static_assert(! std::is_same_v<T, __half>, "__half");
        static_assert(! std::is_same_v<T, __half2>, "__half2");
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
#define MyFFTDebugAssertTrue(cond, msg, ...) { if ( (cond) != true ) { std::cerr << msg << std::endl << " Failed Assert at " << __FILE__ << " " << __LINE__ << " " << __PRETTY_FUNCTION__ << std::endl; std::abort(); } }
#define MyFFTDebugAssertFalse(cond, msg, ...) { if ( (cond) == true ) { std::cerr << msg << std::endl  << " Failed Assert at " << __FILE__ << " " << __LINE__ << " " << __PRETTY_FUNCTION__ << std::endl;  std::abort(); } }                                                                                                    
                                                                                                                    
#endif

#if FFT_DEBUG_LEVEL > 1
// Turn on checkpoints in the testing functions.
#define MyFFTDebugAssertTestTrue(cond, msg, ...)  { if ( (cond) != true ) { std::cerr << "    Test " << msg << " FAILED!" << std::endl  << "  at " << __FILE__ << " " << __LINE__ << " " << __PRETTY_FUNCTION__ << std::endl;std::abort(); } else { std::cerr << "    Test " << msg << " passed!" << std::endl; }}
#define MyFFTDebugAssertTestFalse(cond, msg, ...)  { if ( (cond) == true ) {  std::cerr << "    Test " << msg << " FAILED!" << std::endl   << " at " << __FILE__ << " " << __LINE__ << " " << __PRETTY_FUNCTION__ << std::endl;   std::abort();  } else {  std::cerr << "    Test " << msg << " passed!" << std::endl;  } }

#endif

#if FFT_DEBUG_LEVEL == 2
#define MyFFTDebugPrintWithDetails(...)
#endif

#if FFT_DEBUG_LEVEL == 3
// More verbose debug info
#define MyFFTDebugPrint(...) { std::cerr << __VA_ARGS__ << std::endl; }
#define MyFFTDebugPrintWithDetails(...) { std::cerr << __VA_ARGS__ << " From: " << __FILE__ << " " << __LINE__ << " " << __PRETTY_FUNCTION__ << std::endl; }
#endif

#if FFT_DEBUG_LEVEL == 4
// More verbose debug info + state info
#define MyFFTDebugPrint(...) { FastFFT::FourierTransformer::PrintState( );  std::cerr << __VA_ARGS__ << std::endl; }
#define MyFFTDebugPrintWithDetails(...)  { FastFFT::FourierTransformer::PrintState( ); std::cerr << __VA_ARGS__ << " From: " << __FILE__ << " " << __LINE__ << " " << __PRETTY_FUNCTION__ << std::endl; }

#endif

// Always in use
#define MyFFTPrint(...) { std::cerr << __VA_ARGS__ << std::endl; }
#define MyFFTPrintWithDetails(...) { std::cerr << __VA_ARGS__ << " From: " << __FILE__ << " " << __LINE__ << " " << __PRETTY_FUNCTION__ << std::endl; }
#define MyFFTRunTimeAssertTrue(cond, msg, ...) { if ( (cond) != true ) { std::cerr << msg << std::endl << " Failed Assert at " << __FILE__ << " " << __LINE__ << " " << __PRETTY_FUNCTION__ << std::endl;std::abort(); }  }
#define MyFFTRunTimeAssertFalse(cond, msg, ...) { if ( (cond) == true ) {std::cerr << msg << std::endl << " Failed Assert at " << __FILE__ << " " << __LINE__ << " " << __PRETTY_FUNCTION__ << std::endl;std::abort();  } }                                                                                                               



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
            #define postcheck  { cudaErr(cudaPeekAtLastError( )); cudaError_t error = cudaStreamSynchronize(cudaStreamPerThread); cudaErr(error) }
            #define precheck { cudaErr(cudaGetLastError( )) }
        #endif
    #endif
#endif


inline void checkCudaErr(cudaError_t err) {
    if ( err != cudaSuccess ) {
        std::cerr << cudaGetErrorString(err) << " :-> " << std::endl;
        MyFFTPrintWithDetails(" ");
    }
};

// clang-format on

} // namespace FastFFT

#endif