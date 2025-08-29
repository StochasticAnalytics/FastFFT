# Modifying cufftdx for use in FastFFT

## Overview

Existing versions of cufftdx have routines optimized to use the ptx register tid.y (threadIdx.y in cuda) to allow for implicit batching of 1d transforms. This also requires rearrangments in shared memory. FastFFT needs threadIdx.y simlarly to handle transform decomposition effictively.

1. We need to modify the assembly code to replace all instances of tid.y with tid.z.
    - This allows us to still use the implicit batching mode on tid.z, with the caveat being tid.z is more constrained in CUDA (iir something like 64 or 128)
    - Use setup.sh to do this
2. The directory structure of each version is different to, so setup.sh will create a makefile stub at <version>.mk that can be included in make.
3. There are several files outside of the assembly that reference threadIdx.y (CUDA not PTX) that we also need to modify. This is less automated, so below are detailed instructions that hopefully work going forward as well.
4. There is a shared memory api for memory fold in R2C/C2R that is not supported. We block those with static assers.

## Modifications

1. First do a simple grep to look for files that need to be modified.

```bash
# assuming we have already run setup.sh for version 24.08.0
file_path=$(awk -F "nvidia-srcs/" '{print $2}' nvidia-mathdx-24.08.0.mk)
# Look through recursively ignoring binary and get just unique file names
grep -ril threadIdx.y $file_path/ | sort -u
# Hopefully this list isn't too long and looks something like this

# nvidia-mathdx-24.08.0/nvidia/mathdx/24.08/include/cufftdx/include/detail/fft_execution.hpp
# nvidia-mathdx-24.08.0/nvidia/mathdx/24.08/include/cufftdx/include/detail/processing/fft_block_postprocess.hpp
# nvidia-mathdx-24.08.0/nvidia/mathdx/24.08/include/cufftdx/include/detail/processing/fft_block_preprocess.hpp
# nvidia-mathdx-24.08.0/nvidia/mathdx/24.08/include/cufftdx/include/detail/processing/postprocess_fold.hpp
# nvidia-mathdx-24.08.0/nvidia/mathdx/24.08/include/cufftdx/include/detail/processing/preprocess_fold.hpp    
```

2. Modification for threadIdx.y / implicit batching

```bash
#fft_execution.hpp
# Since we do not want to break the batching, we add a new template parameter to two functions
#   registers_to_shared_impl, shared_to_registers
#   registers_to_shared_impl, registers_to_shared
```

```cpp

    private:
        template<unsigned int FPB, unsigned int N, class T>
        inline __device__ void shared_to_registers_impl(T* shared_memory, T* thread_data) {
            unsigned int batch_offset = threadIdx.y * N;

    template<bool Bluestein, class V>
    inline __device__ void shared_to_registers(void* shared_memory, V* thread_data) {
        using input_t = typename this_type::input_type;
        shared_to_registers_impl<this_type::input_length>();

    }
```

```bash
# becomes
```

```cpp
    private:
        template<unsigned int FPB, unsigned int N, class T>
        inline __device__ void shared_to_registers_impl(T* shared_memory, T* thread_data) {
            unsigned int batch_offset;
            if constexpr (FPB > 1) {
                // Note: I had previously used base_type::this_fft_size_v rather than N, but haven't checked yet if this is still okay, so leaving a 
                // possibly redundant spec.
                batch_offset = threadIdx.y * N;
            } 
            else {
                batch_offset = 0;
            }

    // ffts_per_block is constexpr on FFT declartion
        template<bool Bluestein, class V>
        inline __device__ void shared_to_registers(void* shared_memory, V* thread_data) {
            using input_t = typename this_type::input_type;
            shared_to_registers_impl<ffts_per_block, this_type::input_length>();}
```

3. Modifications to handle the shared memory api

- preprocess_fold.hpp

```cpp
// define this just after the includes. This is a method we use to throw static asserts that are under compile time ifs (constexpr if). 
// The static_assert can't just be placed in a constexpr else because it would be evalueated.
template <bool flag = false>
inline void static_no_shared_api( ) { static_assert(flag, "shared api is being accesed!"); }

// Now preprocess_fold_c2r is modified to allow the shared API if we are using more than one fft_per_block, which is to maintain compatibilty, but won't work in current FastFFT. So we break it after this block. The idea is to make sure the code is maintainable.
// 

if constexpr (FFT::ffts_per_block > 1) {
    if constexpr (IsShared) {
        smem += (threadIdx.y * FFT::input_length);
    }
    else {
        smem += (threadIdx.y * pre<FFT>::half_complex_size);
    }
} 

if constexpr (IsShared) {
    static_no_shared_api();
    // If Shared API is used, there is no need to store data
    // from registers to shared, as it's already in there
    smem += pre<FFT>::half_complex_size;
}

```

### postprocess_fold.hpp

```cpp
template<class FFT, class ComplexType>
inline __device__ auto postprocess_fold_r2c(ComplexType* rmem, ComplexType* smem) -> CUFFTDX_STD::enable_if_t<size_of<FFT>::value != FFT::elements_per_thread> {
    ComplexType  twiddle;
    unsigned int index = threadIdx.x;

    const unsigned int shared_offset = threadIdx.y * post<FFT>::half_complex_size;
    smem += shared_offset;

    /// code
}

// becomes

template<class FFT, class ComplexType>
inline __device__ auto postprocess_fold_r2c(ComplexType* rmem, ComplexType* smem) -> CUFFTDX_STD::enable_if_t<size_of<FFT>::value != FFT::elements_per_thread> {
    ComplexType  twiddle;
    unsigned int index = threadIdx.x;

    if constexpr (FFT::ffts_per_block > 1) {
        smem +=  (threadIdx.y * post<FFT>::half_complex_size);
    } 

    /// CODE 
}
}

```

### fft_block_preprocess.hpp and fft_block_postprocess.hpp (similar pattern)

```cpp

ComplexType* smem_fft_batch = smem + (threadIdx.y * FFT::input_length);

// becomes

ComplexType* smem_fft_batch;
if constexpr (FFT::ffts_per_block > 1) {
    smem_fft_batch = smem + (threadIdx.y * FFT::input_length);
} 
else {
    smem_fft_batch = smem;
} 

```


