// Insert some license stuff here

#ifndef _INCLUDE_FASTFFT_H
#define _INCLUDE_FASTFFT_H

#include <chrono>
#include <random>
#include <iostream>

// Forward declaration so we can leave the inclusion of cuda_fp16.h to FastFFT.cu
struct __half;
struct __half2;
// #include <cuda_fp16.h>

#ifndef ENABLE_FastFFT // ifdef being used in cisTEM that defines these
#if __cplusplus >= 202002L
#include <numbers>
using namespace std::numbers;
#else
#if __cplusplus < 201703L
#message "C++ is " __cplusplus
#error "C++17 or later required"
#else
template <typename _Tp>
// inline constexpr _Tp pi_v = _Enable_if_floating<_Tp>(3.141592653589793238462643383279502884L);
inline constexpr _Tp pi_v = 3.141592653589793238462643383279502884L;
#endif // __cplusplus require > 17
#endif // __cplusplus 20 support
#endif // enable FastFFT

#include "../src/fastfft/types.cuh"

// For testing/debugging it is convenient to execute and have print functions for partial transforms.
// These will go directly in the kernels and also in the helper Image.cuh definitions for PrintArray.
// The number refers to the number of 1d FFTs performed,
// Fwd 0, 1, 2, 3( none, x, z, original y)
// 4 intermediate ops, like conj multiplication
// Inv 5, 6, 7 ( original y, z, x)
// Defined in make by setting environmental variable  FFT_DEBUG_STAGE

// #include <iostream>
/*

Some of the more relevant notes about extended lambdas.
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-lambda

The enclosing function for the extended lambda must be named and its address can be taken. If the enclosing function is a class member, then the following conditions must be satisfied:

    All classes enclosing the member function must have a name.
    The member function must not have private or protected access within its parent class.
    All enclosing classes must not have private or protected access within their respective parent classes.


If the enclosing function is an instantiation of a function template or a member function template, and/or the function is a member of a class template, the template(s) must satisfy the following constraints:

    The template must have at most one variadic parameter, and it must be listed last in the template parameter list.
    The template parameters must be named.
    The template instantiation argument types cannot involve types that are either local to a function (except for closure types for extended lambdas), or are private or protected class members.
#define IS_EXT_LAMBDA( type )  __nv_is_extended_device_lambda_closure_type( type ) 


*/
namespace FastFFT {

// For debugging

inline void PrintVectorType(int3 input) {
    std::cerr << "(x,y,z) " << input.x << " " << input.y << " " << input.z << std::endl;
}

inline void PrintVectorType(int4 input) {
    std::cerr << "(x,y,z,w) " << input.x << " " << input.y << " " << input.z << " " << input.w << std::endl;
}

inline void PrintVectorType(dim3 input) {
    std::cerr << "(x,y,z) " << input.x << " " << input.y << " " << input.z << std::endl;
}

inline void PrintVectorType(short3 input) {
    std::cerr << "(x,y,z) " << input.x << " " << input.y << " " << input.z << std::endl;
}

inline void PrintVectorType(short4 input) {
    std::cerr << "(x,y,z,w) " << input.x << " " << input.y << " " << input.z << " " << input.w << std::endl;
}

typedef struct __align__(32) _DeviceProps {
    int device_id;
    int device_arch;
    int max_shared_memory_per_block;
    int max_shared_memory_per_SM;
    int max_registers_per_block;
    int max_persisting_L2_cache_size;
}

DeviceProps;

typedef struct __align__(8) _FFT_Size {
    // Following Sorensen & Burrus 1993 for clarity
    short N; // N : 1d FFT size
    short L; // L : number of non-zero output/input points
    short P; // P >= L && N % P == 0 : The size of the sub-FFT used to compute the full transform. Currently also must be a power of 2.
    short Q; // Q = N/P : The number of sub-FFTs used to compute the full transform
}

FFT_Size;

typedef struct __align__(8) _Offsets {
    unsigned short shared_input;
    unsigned short shared_output;
    unsigned short physical_x_input;
    unsigned short physical_x_output;
}

Offsets;

typedef struct __align__(64) _LaunchParams {
    int     Q;
    float   twiddle_in;
    dim3    gridDims;
    dim3    threadsPerBlock;
    Offsets mem_offsets;
}

LaunchParams;

template <typename I, typename C>
struct DevicePointers {
    // Use this to catch unsupported input/ compute types and throw exception.
    int* position_space        = nullptr;
    int* position_space_buffer = nullptr;
    int* momentum_space        = nullptr;
    int* momentum_space_buffer = nullptr;
    int* image_to_search       = nullptr;
};

// Input real, compute single-precision
template <>
struct DevicePointers<float*, float*> {
    float*  position_space;
    float*  position_space_buffer;
    float2* momentum_space;
    float2* momentum_space_buffer;
    float2* image_to_search;
};

// Input real, compute half-precision FP16
template <>
struct DevicePointers<__half*, __half*> {
    __half*  position_space;
    __half*  position_space_buffer;
    __half2* momentum_space;
    __half2* momentum_space_buffer;
    __half2* image_to_search;
};

// Input complex, compute single-precision
template <>
struct DevicePointers<float2*, float*> {
    float2* position_space;
    float2* position_space_buffer;
    float2* momentum_space;
    float2* momentum_space_buffer;
    float2* image_to_search;
};

// Input complex, compute half-precision FP16
template <>
struct DevicePointers<__half2*, __half*> {
    __half2* position_space;
    __half2* position_space_buffer;
    __half2* momentum_space;
    __half2* momentum_space_buffer;
    __half2* image_to_search;
};

template <class ComputeType = float, class InputType = float, class OutputType = float, int Rank = 2>
class FourierTransformer {

  private:

  public:
    // Using the enum directly from python is not something I've figured out yet. Just make simple methods.
    inline void SetOriginTypeNatural(bool set_input_type = true) {
        if ( set_input_type )
            input_origin_type = OriginType::natural;
        else
            output_origin_type = OriginType::natural;
    }

    inline void SetOriginTypeCentered(bool set_input_type = true) {
        if ( set_input_type )
            input_origin_type = OriginType::centered;
        else
            output_origin_type = OriginType::centered;
    }

    inline void SetOriginTypeQuadrantSwapped(bool set_input_type = true) {
        if ( set_input_type )
            input_origin_type = OriginType::quadrant_swapped;
        else
            output_origin_type = OriginType::quadrant_swapped;
    }

    short padding_jump_val;
    int   input_memory_allocated;
    int   fwd_output_memory_allocated;
    int   inv_output_memory_allocated;
    int   compute_memory_allocated;
    int   memory_size_to_copy;

    ///////////////////////////////////////////////
    // Initialization functions
    ///////////////////////////////////////////////

    FourierTransformer( );
    // FourierTransformer(const FourierTransformer &); // Copy constructor
    virtual ~FourierTransformer( );

    // This is pretty similar to an FFT plan, I should probably make it align with CufftPlan
    void SetForwardFFTPlan(size_t input_logical_x_dimension, size_t input_logical_y_dimension, size_t input_logical_z_dimension,
                           size_t output_logical_x_dimension, size_t output_logical_y_dimension, size_t output_logical_z_dimension,
                           bool is_padded_output = true);

    void SetInverseFFTPlan(size_t input_logical_x_dimension, size_t input_logical_y_dimension, size_t input_logical_z_dimension,
                           size_t output_logical_x_dimension, size_t output_logical_y_dimension, size_t output_logical_z_dimension,
                           bool is_padded_output = true);

    // For the time being, the caller is responsible for having the memory allocated for any of these input/output pointers.
    void SetInputPointer(InputType* input_pointer, bool is_input_on_device);
    // When passing in a pointer from python (cupy or pytorch) it is a long, and needs to be cast to input type.
    // For now, we are assuming memory ops are all handled in the python code.
    void SetInputPointer(long input_pointer);
    void SetCallerPinnedInputPointer(InputType* input_pointer);

    ///////////////////////////////////////////////
    // Public actions:
    // ALL public actions should call ::CheckDimensions() to ensure the meta data are properly intialized.
    // this ensures the prior three methods have been called and are valid.
    ///////////////////////////////////////////////
    inline void Wait( ) { cudaStreamSynchronize(cudaStreamPerThread); };

    void CopyHostToDevceAndSynchronize(InputType* input_pointer, int n_elements_to_copy = 0);
    void CopyHostToDevice(InputType* input_pointer, int n_elements_to_copy = 0);
    // If int n_elements_to_copy = 0 the appropriate size will be determined by the state of the transform completed (none, fwd, inv.)
    // For partial increase/decrease transforms, needed for testing, this will be invalid, so specify the int n_elements_to_copy.
    // When the size changes, we need a new host pointer
    void CopyDeviceToHostAndSynchronize(OutputType* output_pointer, bool free_gpu_memory = true, int n_elements_to_copy = 0);
    void CopyDeviceToHost(OutputType* output_pointer, bool free_gpu_memory = true, int n_elements_to_copy = 0);

    // Ideally, in addition to position/momentum space (buffer) ponters, there would also be a input pointer, which may point
    // to a gpu address that is from an external process or to the FastFFT buffer space. This way, when calling CopyHostToDevice,
    // that input is set to the FastFFT buffer space, data is copied and the first Fwd kernels are called as they are currently.
    // This would also allow the input pointer to point to a different address than the FastFFT buffer only accessed on initial kernel
    // calls and read only. In turn we could skip the device to device transfer we are doing in the following method.
    void CopyDeviceToDeviceFromNonOwningAddress(InputType* input_pointer, int n_elements_to_copy = 0);

    // Here we may be copying input data type from another GPU buffer, OR output data type to another GPU buffer.
    // Check in these methods that the types match
    template <class TransferDataType>
    void CopyDeviceToDeviceAndSynchronize(TransferDataType* input_pointer, bool free_gpu_memory = true, int n_elements_to_copy = 0);
    template <class TransferDataType>
    void CopyDeviceToDevice(TransferDataType* input_pointer, bool free_gpu_memory = true, int n_elements_to_copy = 0);
    // FFT calls

    //   void FwdFFT(bool swap_real_space_quadrants = false, bool transpose_output = true);
    //   void InvFFT(bool transpose_output = true);
    // void CrossCorrelate(float2* image_to_search, bool swap_real_space_quadrants);
    // void CrossCorrelate(__half2* image_to_search, bool swap_real_space_quadrants);

    // could float2* be replaced with decltype(DevicePointers.momentum_space)
    template <class PreOpType = std::nullptr_t, class IntraOpType = std::nullptr_t, class PostOpType = std::nullptr_t>
    void Generic_Fwd_Image_Inv(float2* data, PreOpType pre_op = nullptr, IntraOpType intra_op = nullptr, PostOpType post_op = nullptr);

    template <class PreOpType = std::nullptr_t, class IntraOpType = std::nullptr_t>
    void Generic_Fwd(PreOpType pre_op = nullptr, IntraOpType intra_op = nullptr);

    template <class IntraOpType = std::nullptr_t, class PostOpType = std::nullptr_t>
    void Generic_Inv(IntraOpType intra_op = nullptr, PostOpType post_op = nullptr);

    // Alias for FwdFFT, is there any overhead?
    template <class PreOpType = std::nullptr_t, class IntraOpType = std::nullptr_t>
    void FwdFFT(PreOpType pre_op = nullptr, IntraOpType intra_op = nullptr) { Generic_Fwd<PreOpType, IntraOpType>(pre_op, intra_op); }

    template <class IntraOpType = std::nullptr_t, class PostOpType = std::nullptr_t>
    void InvFFT(IntraOpType intra_op = nullptr, PostOpType post_op = nullptr) { Generic_Inv<IntraOpType, PostOpType>(intra_op, post_op); }

    void ClipIntoTopLeft( );
    void ClipIntoReal(int wanted_coordinate_of_box_center_x, int wanted_coordinate_of_box_center_y, int wanted_coordinate_of_box_center_z);

    // For all real valued inputs, assumed for any InputType that is not float2 or __half2

    int inline ReturnInputMemorySize( ) { return input_memory_allocated; }

    int inline ReturnFwdOutputMemorySize( ) { return fwd_output_memory_allocated; }

    int inline ReturnInvOutputMemorySize( ) { return inv_output_memory_allocated; }

    short4 inline ReturnFwdInputDimensions( ) { return fwd_dims_in; }

    short4 inline ReturnFwdOutputDimensions( ) { return fwd_dims_out; }

    short4 inline ReturnInvInputDimensions( ) { return inv_dims_in; }

    short4 inline ReturnInvOutputDimensions( ) { return inv_dims_out; }

    template <typename T, bool is_on_host = true>
    void SetToConstant(T* input_pointer, int N_values, const T& wanted_value) {
        if ( is_on_host ) {
            for ( int i = 0; i < N_values; i++ ) {
                input_pointer[i] = wanted_value;
            }
        }
        else {
            exit(-1);
        }
    }

    template <typename T, bool is_on_host = true>
    void SetToRandom(T* input_pointer, int N_values, const T& wanted_mean, const T& wanted_stddev) {
        std::random_device rd;
        std::mt19937       rng(rd( ));
        const uint64_t     seed = std::chrono::high_resolution_clock::now( ).time_since_epoch( ).count( );
        rng.seed(seed);

        if ( is_on_host ) {
            for ( int i = 0; i < N_values; i++ ) {
                input_pointer[i] = std::normal_distribution<T>{wanted_mean, wanted_stddev}(rng);
            }
        }
        else {
            exit(-1);
        }
    }

    // Input is real or complex inferred from InputType
    DevicePointers<InputType*, ComputeType*> d_ptr;

    void PrintState( ) {
        std::cerr << "================================================================" << std::endl;
        std::cerr << "Device Properties: " << std::endl;
        std::cerr << "================================================================" << std::endl;

        std::cerr << "Device idx: " << device_properties.device_id << std::endl;
        std::cerr << "max_shared_memory_per_block: " << device_properties.max_shared_memory_per_block << std::endl;
        std::cerr << "max_shared_memory_per_SM: " << device_properties.max_shared_memory_per_SM << std::endl;
        std::cerr << "max_registers_per_block: " << device_properties.max_registers_per_block << std::endl;
        std::cerr << "max_persisting_L2_cache_size: " << device_properties.max_persisting_L2_cache_size << std::endl;
        std::cerr << std::endl;

        std::cerr << "State Variables:\n"
                  << std::endl;
        std::cerr << "is_in_memory_host_pointer " << is_in_memory_host_pointer << std::endl;
        std::cerr << "is_in_memory_device_pointer " << is_in_memory_device_pointer << std::endl;
        std::cerr << "is_in_buffer_memory " << is_in_buffer_memory << std::endl;
        std::cerr << "is_fftw_padded_input " << is_fftw_padded_input << std::endl;
        std::cerr << "is_fftw_padded_output " << is_fftw_padded_output << std::endl;
        std::cerr << "is_real_valued_input " << is_real_valued_input << std::endl;
        std::cerr << "is_set_input_params " << is_set_input_params << std::endl;
        std::cerr << "is_set_output_params " << is_set_output_params << std::endl;
        std::cerr << "is_size_validated " << is_size_validated << std::endl;
        std::cerr << "is_set_input_pointer " << is_set_input_pointer << std::endl;
        std::cerr << std::endl;

        std::cerr << "Size variables:\n"
                  << std::endl;
        std::cerr << "transform_size.N " << transform_size.N << std::endl;
        std::cerr << "transform_size.L " << transform_size.L << std::endl;
        std::cerr << "transform_size.P " << transform_size.P << std::endl;
        std::cerr << "transform_size.Q " << transform_size.Q << std::endl;
        std::cerr << "fwd_dims_in.x,y,z ";
        PrintVectorType(fwd_dims_in);
        std::cerr << std::endl;
        std::cerr << "fwd_dims_out.x,y,z ";
        PrintVectorType(fwd_dims_out);
        std::cerr << std::endl;
        std::cerr << "inv_dims_in.x,y,z ";
        PrintVectorType(inv_dims_in);
        std::cerr << std::endl;
        std::cerr << "inv_dims_out.x,y,z ";
        PrintVectorType(inv_dims_out);
        std::cerr << std::endl;
        std::cerr << std::endl;

        std::cerr << "Misc:\n"
                  << std::endl;
        std::cerr << "compute_memory_allocated " << compute_memory_allocated << std::endl;
        std::cerr << "memory size to copy " << memory_size_to_copy << std::endl;
        std::cerr << "fwd_size_change_type " << SizeChangeName[fwd_size_change_type] << std::endl;
        std::cerr << "inv_size_change_type " << SizeChangeName[inv_size_change_type] << std::endl;
        std::cerr << "transform stage complete " << TransformStageCompletedName[transform_stage_completed] << std::endl;
        std::cerr << "input_origin_type " << OriginType::name[input_origin_type] << std::endl;
        std::cerr << "output_origin_type " << OriginType::name[output_origin_type] << std::endl;

    }; // PrintState()

    // private:

    DeviceProps      device_properties;
    OriginType::Enum input_origin_type;
    OriginType::Enum output_origin_type;

    // booleans to track state, could be bit fields but that seem opaque to me.
    bool is_in_memory_host_pointer; // To track allocation of host side memory
    bool is_in_memory_device_pointer; // To track allocation of device side memory.
    bool is_in_buffer_memory; // To track whether the current result is in dev_ptr.position_space or dev_ptr.position_space_buffer (momemtum space/ momentum space buffer respectively.)

    bool is_fftw_padded_input; // Padding for in place r2c transforms
    bool is_fftw_padded_output; // Currently the output state will match the input state, otherwise it is an error.

    bool is_real_valued_input; // This is determined by the input type. If it is a float2 or __half2, then it is assumed to be a complex valued input function.

    bool is_set_input_params; // Yes, yes, "are" set.
    bool is_set_output_params;
    bool is_size_validated; // Defaults to false, set after both input/output dimensions are set and checked.
    bool is_set_input_pointer; // May be on the host of the device.

    int      transform_dimension; // 1,2,3d.
    FFT_Size transform_size;
    int      elements_per_thread_complex; // Set depending on the kernel and size of the transform.

    std::vector<std::string> SizeChangeName{"increase", "decrease", "no_change"};

    std::vector<std::string> TransformStageCompletedName{"", "", "", "", "", // padding of 5
                                                         "", "", "", "", "", // padding of 5
                                                         "none", "fwd", "inv"};

    std::vector<std::string> DimensionCheckName{"CopyFromHost", "CopyToHost", "FwdTransform", "InvTransform"};

    bool is_from_python_call;
    bool is_owner_of_memory;

    SizeChangeType::Enum fwd_size_change_type;
    SizeChangeType::Enum inv_size_change_type;

    TransformStageCompleted::Enum transform_stage_completed;

    // dims_in may change during calculation, depending on padding, but is reset after each call.
    short4 dims_in;
    short4 dims_out;

    short4 fwd_dims_in;
    short4 fwd_dims_out;
    short4 inv_dims_in;
    short4 inv_dims_out;

    InputType* host_pointer;
    InputType* pinnedPtr;

    void Deallocate( );
    void UnPinHostMemory( );

    void SetDefaults( );
    void ValidateDimensions( );
    void SetDimensions(DimensionCheckType::Enum check_op_type);

    void SetDevicePointers(bool should_allocate_buffer_memory);

    /*
    IMPORTANT: if you add a kernel, you need to modify
      1) enum KernelType
      2) KernelName: this is positionally dependent on KernelType
      3) If appropriate:
        a) IsThreadType()
        b) IsR2CType()
        c) IsC2RType()
        d) IsForwardType()
        e) IsTransformAlongZ()
  */

    enum KernelType { r2c_decomposed, // Thread based, full length.
                      r2c_decomposed_transposed, // Thread based, full length, transposed.
                      r2c_none_XY,
                      r2c_none_XZ,
                      r2c_decrease,
                      r2c_increase,
                      r2c_increase_XZ,
                      c2c_fwd_none,
                      c2c_fwd_none_Z,
                      c2c_fwd_decrease,
                      c2c_fwd_increase,
                      c2c_fwd_increase_Z,
                      c2c_inv_none,
                      c2c_inv_none_XZ,
                      c2c_inv_none_Z,
                      c2c_inv_decrease,
                      c2c_inv_increase,
                      c2c_decomposed,
                      c2r_decomposed,
                      c2r_decomposed_transposed,
                      c2r_none,
                      c2r_none_XY,
                      c2r_decrease,
                      c2r_increase,
                      xcorr_fwd_increase_inv_none, //  (e.g. template matching)
                      xcorr_fwd_decrease_inv_none, // (e.g. Fourier cropping)
                      xcorr_fwd_none_inv_decrease, // (e.g. movie/particle translational search)
                      xcorr_fwd_decrease_inv_decrease, // (e.g. bandlimit, xcorr, translational search)
                      xcorr_decomposed,
                      generic_fwd_increase_op_inv_none };

    // WARNING this is flimsy and prone to breaking, you must ensure the order matches the KernelType enum.
    std::vector<std::string>
            KernelName{"r2c_decomposed",
                       "r2c_decomposed_transposed",
                       "r2c_none_XY", "r2c_none_XZ",
                       "r2c_decrease", "r2c_increase", "r2c_increase_XZ",
                       "c2c_fwd_none", "c2c_fwd_none_Z", "c2c_fwd_increase", "c2c_fwd_increase", "c2c_fwd_increase_Z",
                       "c2c_inv_none", "c2c_inv_none_XZ", "c2c_inv_none_Z", "c2c_inv_increase", "c2c_inv_increase",
                       "c2c_decomposed",
                       "c2r_decomposed",
                       "c2r_decomposed_transposed",
                       "c2r_none", "c2r_none_XY", "c2r_decrease", "c2r_increase",
                       "xcorr_fwd_increase_inv_none",
                       "xcorr_fwd_decrease_inv_none",
                       "xcorr_fwd_none_inv_decrease",
                       "xcorr_fwd_decrease_inv_decrease",
                       "xcorr_decomposed",
                       "generic_fwd_increase_op_inv_none"};

    inline bool IsThreadType(KernelType kernel_type) {
        if ( kernel_type == r2c_decomposed || kernel_type == r2c_decomposed_transposed ||
             kernel_type == c2c_decomposed ||
             kernel_type == c2r_decomposed || kernel_type == c2r_decomposed_transposed || kernel_type == xcorr_decomposed ) {
            return true;
        }

        else if ( kernel_type == r2c_none_XY || kernel_type == r2c_none_XZ ||
                  kernel_type == r2c_decrease || kernel_type == r2c_increase || kernel_type == r2c_increase_XZ ||
                  kernel_type == c2c_fwd_none || c2c_fwd_none_Z ||
                  kernel_type == c2c_fwd_decrease ||
                  kernel_type == c2c_fwd_increase || kernel_type == c2c_fwd_increase_Z ||
                  kernel_type == c2c_inv_none || kernel_type == c2c_inv_none_XZ || kernel_type == c2c_inv_none_Z ||
                  kernel_type == c2c_inv_decrease || kernel_type == c2c_inv_increase ||
                  kernel_type == c2r_none || kernel_type == c2r_none_XY || kernel_type == c2r_decrease || kernel_type == c2r_increase ||
                  kernel_type == xcorr_fwd_increase_inv_none || kernel_type == xcorr_fwd_decrease_inv_none || kernel_type == xcorr_fwd_none_inv_decrease || kernel_type == xcorr_fwd_decrease_inv_decrease ||
                  kernel_type == generic_fwd_increase_op_inv_none ) {
            return false;
        }
        else {
            std::cerr << "Function IsThreadType does not recognize the kernel type ( " << KernelName[kernel_type] << " )" << std::endl;
            exit(-1);
        }
    };

    inline bool IsR2CType(KernelType kernel_type) {
        if ( kernel_type == r2c_decomposed || kernel_type == r2c_decomposed_transposed ||
             kernel_type == r2c_none_XY || kernel_type == r2c_none_XZ ||
             kernel_type == r2c_decrease || kernel_type == r2c_increase || kernel_type == r2c_increase_XZ ) {
            return true;
        }
        else
            return false;
    }

    inline bool IsC2RType(KernelType kernel_type) {
        if ( kernel_type == c2r_decomposed || kernel_type == c2r_decomposed_transposed ||
             kernel_type == c2r_none || kernel_type == c2r_none_XY || kernel_type == c2r_decrease || kernel_type == c2r_increase ) {
            return true;
        }
        else
            return false;
    }

    // This is used to set the sign of the twiddle factor for decomposed kernels, whether threaded, or part of a block fft.
    // For mixed kernels (eg. xcorr_* the size type is defined by where the size change happens.
    inline bool IsForwardType(KernelType kernel_type) {
        if ( kernel_type == r2c_decomposed || kernel_type == r2c_decomposed_transposed ||
             kernel_type == r2c_none_XY || kernel_type == r2c_none_XZ ||
             kernel_type == r2c_decrease || kernel_type == r2c_increase || kernel_type == r2c_increase_XZ ||
             kernel_type == c2c_fwd_none || kernel_type == c2c_fwd_none_Z || kernel_type == c2c_fwd_increase_Z ||
             kernel_type == c2c_fwd_decrease ||
             kernel_type == c2c_fwd_increase ||
             kernel_type == xcorr_fwd_decrease_inv_none || kernel_type == xcorr_fwd_increase_inv_none ||
             kernel_type == generic_fwd_increase_op_inv_none )

        {
            return true;
        }
        else
            return false;
    }

    inline bool IsTransormAlongZ(KernelType kernel_type) {
        if ( kernel_type == c2c_fwd_none_Z || kernel_type == c2c_fwd_increase_Z ||
             kernel_type == c2c_inv_none_Z ) {
            return true;
        }
        else
            return false;
    }

    inline bool IsRank3(KernelType kernel_type) {
        if ( kernel_type == r2c_none_XZ || kernel_type == r2c_increase_XZ ||
             kernel_type == c2c_fwd_increase_Z || kernel_type == c2c_inv_none_XZ ||
             kernel_type == c2c_fwd_none_Z || kernel_type == c2c_inv_none_Z ) {
            return true;
        }
        else
            return false;
    }

    inline void AssertDivisibleAndFactorOf2(int full_size_transform, int number_non_zero_inputs_or_outputs) {
        // FIXME: This function could be named more appropriately.
        transform_size.N = full_size_transform;
        transform_size.L = number_non_zero_inputs_or_outputs;
        // FIXME: in principle, transform_size.L should equal number_non_zero_inputs_or_outputs and transform_size.P only needs to be >= and satisfy other requirements, e.g. power of two (currently.)
        transform_size.P = number_non_zero_inputs_or_outputs;

        if ( transform_size.N % transform_size.P == 0 ) {
            transform_size.Q = transform_size.N / transform_size.P;
        }
        else {
            std::cerr << "Array size " << transform_size.N << " is not divisible by wanted output size " << transform_size.P << std::endl;
            exit(1);
        }

        if ( abs(fmod(log2(float(transform_size.P)), 1)) > 1e-6 ) {
            std::cerr << "Wanted output size " << transform_size.P << " is not a power of 2." << std::endl;
            exit(1);
        }
    }

    void         GetTransformSize(KernelType kernel_type);
    void         GetTransformSize_thread(KernelType kernel_type, int thread_fft_size);
    LaunchParams SetLaunchParameters(const int& ept, KernelType kernel_type, bool do_forward_transform = true);

    inline int ReturnPaddedMemorySize(short4& wanted_dims) {
        // Assumes a) SetInputDimensionsAndType has been called and is_fftw_padded is set before this call. (Currently RuntimeAssert to die if false) FIXME
        int wanted_memory = 0;

        if ( is_real_valued_input ) {
            if ( wanted_dims.x % 2 == 0 ) {
                padding_jump_val = 2;
                wanted_memory    = wanted_dims.x / 2 + 1;
            }
            else {
                padding_jump_val = 1;
                wanted_memory    = (wanted_dims.x - 1) / 2 + 1;
            }

            wanted_memory *= wanted_dims.y * wanted_dims.z; // other dimensions
            wanted_memory *= 2; // room for complex
            wanted_dims.w            = (wanted_dims.x + padding_jump_val) / 2; // number of complex elements in the X dimesnions after FFT.
            compute_memory_allocated = std::max(compute_memory_allocated, 2 * wanted_memory); // scaling by 2 making room for the buffer.
        }
        else {
            wanted_memory = wanted_dims.x * wanted_dims.y * wanted_dims.z;
            wanted_dims.w = wanted_dims.x; // pitch is constant
            // We allocate using sizeof(ComputeType) which is either __half or float, so we need an extra factor of 2
            // Note: this must be considered when setting the address of the buffer memory based on the address of the regular memory.
            compute_memory_allocated = std::max(compute_memory_allocated, 4 * wanted_memory);
        }
        return wanted_memory;
    }

    template <class FFT, class invFFT>
    void FFT_C2C_WithPadding_ConjMul_C2C_t(float2* image_to_search, bool swap_real_space_quadrants);
    template <class FFT, class invFFT>
    void FFT_C2C_decomposed_ConjMul_C2C_t(float2* image_to_search, bool swap_real_space_quadrants);

    // 1.
    // First call passed from a public transform function, selects block or thread and the transform precision.
    template <bool use_thread_method = false, class PreOpType = std::nullptr_t, class IntraOpType = std::nullptr_t, class PostOpType = std::nullptr_t> // bool is just used as a dummy type
    void SetPrecisionAndExectutionMethod(KernelType kernel_type, bool do_forward_transform = true, PreOpType pre_op_functor = nullptr, IntraOpType intra_op_functor = nullptr, PostOpType post_op_functor = nullptr);

    // 2. // TODO: remove this now that the functors are working
    // Check to see if any intra kernel functions are wanted, and if so set the appropriate device pointers.
    template <class FFT_base, class PreOpType, class IntraOpType, class PostOpType>
    void SetIntraKernelFunctions(KernelType kernel_type, bool do_forward_transform, PreOpType pre_op_functor, IntraOpType intra_op_functor, PostOpType post_op_functor);

    // 3.
    // Second call, sets size of the transform kernel, selects the appropriate GPU arch

    // template <class FFT_base, class PreOpType, class IntraOpType, class PostOpType>
    // void SelectSizeAndType(KernelType kernel_type, bool do_forward_transform, PreOpType pre_op_functor, IntraOpType intra_op_functor, PostOpType post_op_functor);
    // This allows us to iterate through a set of constexpr sizes passed as a template parameter pack. The value is in providing a means to have different size packs
    // for different fft configurations, eg. 2d vs 3d
    template <class FFT_base, class PreOpType, class IntraOpType, class PostOpType>
    void SelectSizeAndType(KernelType kernel_type, bool do_forward_transform, PreOpType pre_op_functor, IntraOpType intra_op_functor, PostOpType post_op_functor);

    template <class FFT_base, class PreOpType, class IntraOpType, class PostOpType, unsigned int SizeValue, unsigned int Ept, unsigned int... OtherValues>
    void SelectSizeAndType(KernelType kernel_type, bool do_forward_transform, PreOpType pre_op_functor, IntraOpType intra_op_functor, PostOpType post_op_functor);

    // 3.
    // Third call, sets the input and output dimensions and type
    template <class FFT_base_arch, class PreOpType, class IntraOpType, class PostOpType>
    void SetAndLaunchKernel(KernelType kernel_type, bool do_forward_transform, PreOpType pre_op_functor, IntraOpType intra_op_functor, PostOpType post_op_functor);

    void PrintLaunchParameters(LaunchParams LP) {
        std::cerr << "Launch parameters: " << std::endl;
        std::cerr << "  Threads per block: ";
        PrintVectorType(LP.threadsPerBlock);
        std::cerr << "  Grid dimensions: ";
        PrintVectorType(LP.gridDims);
        std::cerr << "  Q: " << LP.Q << std::endl;
        std::cerr << "  Twiddle in: " << LP.twiddle_in << std::endl;
        std::cerr << "  shared input: " << LP.mem_offsets.shared_input << std::endl;
        std::cerr << "  shared output (memlimit in r2c): " << LP.mem_offsets.shared_output << std::endl;
        std::cerr << "  physical_x_input: " << LP.mem_offsets.physical_x_input << std::endl;
        std::cerr << "  physical_x_output: " << LP.mem_offsets.physical_x_output << std::endl;
    };

}; // class Fourier Transformer

} // namespace FastFFT

#endif
