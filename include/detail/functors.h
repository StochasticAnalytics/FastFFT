#ifndef __INCLUDE_DETAILS_FUNCTORS_H__
#define __INCLUDE_DETAILS_FUNCTORS_H__

// TODO: doc and namespace
template <typename K>
constexpr inline bool IS_IKF_t( ) {
    if constexpr ( std::is_final_v<K> ) {
        return true;
    }
    else {
        return false;
    }
};

namespace FastFFT {
namespace KernelFunction {

// Define an enum for different functors
// Intra Kernel Function Type
enum IKF_t { NOOP,
             CONJ_MUL };

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
        // This is target * conj(template)
        float tmp      = (template_fft_x * target_fft_x + template_fft_y * target_fft_y);
        template_fft_y = (template_fft_x * target_fft_y - template_fft_y * target_fft_x);
        template_fft_x = tmp;
    }
};

} // namespace KernelFunction
} // namespace FastFFT

#endif