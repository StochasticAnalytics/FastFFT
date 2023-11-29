// Insert some license stuff here

#ifndef __INCLUDE_FASTFFT_H_
#define __INCLUDE_FASTFFT_H_

#include "detail/detail.h"

// TODO: When recompiling a changed debug type, make has no knowledge and the -B flag must be passed.
//       Save the most recent state and have make query that to determine if the -B flag is needed.
// For testing/debugging it is convenient to execute and have print functions for partial transforms.
// These will go directly in the kernels and also in the helper Image.cuh definitions for PrintArray.
// The number refers to the number of 1d FFTs performed,
// Fwd 0, 1, 2, 3( none, x, z, original y)
// 4 intermediate ops, like conj multiplication
// Inv 5, 6, 7 ( original y, z, x)
// Defined in make by setting environmental variable  FFT_DEBUG_STAGE

namespace FastFFT {

// TODO: this may be expanded, for now it is to be used in the case where we have
// packed the real values of a c2r into the first half of the complex array.
// The output type pointer needs to be cast to the correct type AND posibly converted
template <bool flag = false>
inline void static_no_match( ) { static_assert(flag, "no match"); }

template <bool flag = false>
inline void static_no_doubles( ) { static_assert(flag, "no doubles are allowed"); }

template <bool flag = false>
inline void static_no_half_support_yet( ) { static_assert(flag, "no __half support yet"); }

// TODO: Device pointers are probably no longer needed, and only depend on InputType
// TODO: Add a method to set the cuda stream
template <class C, class I, class OI>
struct DevicePointers {
    // Use this to catch unsupported input/ compute types and throw exception.
    void* buffer_1;
    void* buffer_2;
};

template <>
struct DevicePointers<float*, float*, float*> {
    float2* buffer_1{ };
    float2* buffer_2{ };
};

template <>
struct DevicePointers<float*, __half*, float*> {
    __half2* buffer_1{ };
    __half2* buffer_2{ };
};

template <>
struct DevicePointers<float*, __half*, __half*> {
    __half2* buffer_1{ };
    __half2* buffer_2{ };
};

/**
 * @brief Construct a new Fourier Transformer< Compute Type,  Input Type,  OtherImageType ,  Rank>:: Fourier Transformer object
 * 
 * 
 * @tparam ComputeBaseType - float. Support for ieee half precision is not yet implemented.
 * @tparam InputType - __half or float for real valued input, __half2 or float2 for complex valued input images.
 * @tparam OtherImageType - __half or float. Actual type depends on position/momentum space representation.
 * @tparam Rank - only 2,3 supported. Support for 3d is partial
 */
template <class ComputeBaseType = float, class InputType = float, class OtherImageType = float, int Rank = 2>
class FourierTransformer {

  public:
    ///////////////////////////////////////////////
    // Initialization functions
    ///////////////////////////////////////////////

    FourierTransformer( );
    ~FourierTransformer( );

    // For now, we do not want to allow Copy or Move
    FourierTransformer(const FourierTransformer&)            = delete;
    FourierTransformer& operator=(const FourierTransformer&) = delete;
    FourierTransformer(FourierTransformer&&)                 = delete;
    FourierTransformer& operator=(FourierTransformer&&)      = delete;

    // This is pretty similar to an FFT plan, I should probably make it align with CufftPlan
    void SetForwardFFTPlan(size_t input_logical_x_dimension, size_t input_logical_y_dimension, size_t input_logical_z_dimension,
                           size_t output_logical_x_dimension, size_t output_logical_y_dimension, size_t output_logical_z_dimension,
                           bool is_padded_output = true);

    void SetInverseFFTPlan(size_t input_logical_x_dimension, size_t input_logical_y_dimension, size_t input_logical_z_dimension,
                           size_t output_logical_x_dimension, size_t output_logical_y_dimension, size_t output_logical_z_dimension,
                           bool is_padded_output = true);

    // When passing in a pointer from python (cupy or pytorch) it is a long, and needs to be cast to input type.
    // For now, we are assuming memory ops are all handled in the python code.
    void SetInputPointerFromPython(long input_pointer);

    // template <class InputDataPtr_t,
    //           class OutputDataPtr_t,
    //           class ImageToSearchPtr_t>
    // void SetDataPointers(InputDataPtr_t input_ptr, OutputDataPtr_t output_ptr, ImageToSearchPtr_t image_to_search_ptr);

    ///////////////////////////////////////////////
    // Public actions:
    // ALL public actions should call ::CheckDimensions() to ensure the meta data are properly intialized.
    // this ensures the prior three methods have been called and are valid.
    ///////////////////////////////////////////////
    inline void Wait( ) {
        cudaStreamSynchronize(cudaStreamPerThread);
    };

    auto inline GetDeviceBufferPointer( ) {
        return d_ptr.buffer_1;
    }

    void CopyHostToDeviceAndSynchronize(InputType* input_pointer, int n_elements_to_copy = 0);
    void CopyHostToDevice(InputType* input_pointer, int n_elements_to_copy = 0);

    void CopyDeviceToHostAndSynchronize(InputType* input_pointer, int n_elements_to_copy = 0) {
        SetDimensions(DimensionCheckType::CopyToHost);
        int n_to_actually_copy = (n_elements_to_copy > 0) ? n_elements_to_copy : memory_size_to_copy_;
        if ( is_in_buffer_memory ) {
            std::cerr << " in buffer memory " << std::endl;
            std::cerr << "  buffer_1 " << d_ptr.buffer_1 << std::endl;
            std::cerr << "  buffer_2 " << d_ptr.buffer_2 << std::endl;
            std::cerr << " input_ptr " << input_pointer << std::endl;
            std::cerr << "copying " << n_to_actually_copy << " elements" << std::endl;
            cudaErr(cudaMemcpyAsync(input_pointer, d_ptr.buffer_1, n_to_actually_copy * sizeof(InputType), cudaMemcpyDeviceToHost, cudaStreamPerThread));
        }
        else {
            std::cerr << " not in buffer memory " << std::endl;
            std::cerr << "  buffer_1 " << d_ptr.buffer_1 << std::endl;
            std::cerr << "  buffer_2 " << d_ptr.buffer_2 << std::endl;
            std::cerr << " input_ptr " << input_pointer << std::endl;
            std::cerr << "copying " << n_to_actually_copy << " elements" << std::endl;

            cudaErr(cudaMemcpyAsync(input_pointer, d_ptr.buffer_2, n_to_actually_copy * sizeof(InputType), cudaMemcpyDeviceToHost, cudaStreamPerThread));
        }
        cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
    };

    // Using the enum directly from python is not something I've figured out yet. Just make simple methods.
    // FIXME: these are not currently used, and perhaps are not needed.
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

    // FFT calls

    // TODO: when picking up tomorrow, remove default values for input pointers and move EnableIf to the declarations for the generic functions and instantiate these from fastFFT.cus
    // Following this
    // Alias for FwdFFT, is there any overhead?
    template <class PreOpType   = nullptr_t,
              class IntraOpType = nullptr_t>
    void FwdFFT(InputType*  input_ptr,
                PreOpType   pre_op   = nullptr,
                IntraOpType intra_op = nullptr);

    template <class IntraOpType = nullptr_t,
              class PostOpType  = nullptr_t>
    void InvFFT(InputType*  input_ptr,
                IntraOpType intra_op = nullptr,
                PostOpType  post_op  = nullptr);

    template <class PreOpType   = nullptr_t,
              class IntraOpType = nullptr_t,
              class PostOpType  = nullptr_t>
    void FwdImageInvFFT(InputType*      input_ptr,
                        OtherImageType* image_to_search,
                        PreOpType       pre_op   = nullptr,
                        IntraOpType     intra_op = nullptr,
                        PostOpType      post_op  = nullptr) {
        Generic_Fwd_Image_Inv<PreOpType, IntraOpType, PostOpType>(input_ptr, image_to_search, pre_op, intra_op, post_op);
    }

    void ClipIntoTopLeft(InputType* input_ptr);
    void ClipIntoReal(InputType*, int wanted_coordinate_of_box_center_x, int wanted_coordinate_of_box_center_y, int wanted_coordinate_of_box_center_z);

    // For all real valued inputs, assumed for any InputType that is not float2 or __half2

    int inline ReturnInputMemorySize( ) {
        return input_memory_wanted_;
    }

    int inline ReturnFwdOutputMemorySize( ) {
        return fwd_output_memory_wanted_;
    }

    int inline ReturnInvOutputMemorySize( ) {
        return inv_output_memory_wanted_;
    }

    short4 inline ReturnFwdInputDimensions( ) {
        return fwd_dims_in;
    }

    short4 inline ReturnFwdOutputDimensions( ) {
        return fwd_dims_out;
    }

    short4 inline ReturnInvInputDimensions( ) {
        return inv_dims_in;
    }

    short4 inline ReturnInvOutputDimensions( ) {
        return inv_dims_out;
    }

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
        // std::cerr << "is_in_memory_device_pointer " << is_in_memory_device_pointer << std::endl; // FIXME: switched to is_pointer_in_device_memory(d_ptr.buffer_1) defined in FastFFT.cuh
        std::cerr << "is_in_buffer_memory " << is_in_buffer_memory << std::endl;
        std::cerr << "is_fftw_padded_input " << is_fftw_padded_input << std::endl;
        std::cerr << "is_fftw_padded_output " << is_fftw_padded_output << std::endl;
        std::cerr << "is_real_valued_input " << IsAllowedRealType<InputType> << std::endl;
        std::cerr << "is_set_input_params " << is_set_input_params << std::endl;
        std::cerr << "is_set_output_params " << is_set_output_params << std::endl;
        std::cerr << "is_size_validated " << is_size_validated << std::endl;
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
        std::cerr << "compute_memory_wanted_ " << compute_memory_wanted_ << std::endl;
        std::cerr << "memory size to copy " << memory_size_to_copy_ << std::endl;
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
    bool is_in_buffer_memory; // To track whether the current result is in dev_ptr.position_space or dev_ptr.position_space_buffer (momemtum space/ momentum space buffer respectively.)

    bool is_fftw_padded_input; // Padding for in place r2c transforms
    bool is_fftw_padded_output; // Currently the output state will match the input state, otherwise it is an error.

    bool is_set_input_params; // Yes, yes, "are" set.
    bool is_set_output_params;
    bool is_size_validated; // Defaults to false, set after both input/output dimensions are set and checked.

    int      transform_dimension; // 1,2,3d.
    FFT_Size transform_size;
    int      elements_per_thread_complex; // Set depending on the kernel and size of the transform.

    std::vector<std::string> SizeChangeName{"increase", "decrease", "no_change"};

    std::vector<std::string> TransformStageCompletedName{"", "", "", "", "", // padding of 5
                                                         "", "", "", "", "", // padding of 5
                                                         "none", "fwd", "inv"};

    std::vector<std::string> DimensionCheckName{"CopyFromHost", "CopyToHost", "FwdTransform", "InvTransform"};

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

    void Deallocate( );

    void SetDefaults( );
    void ValidateDimensions( );
    void SetDimensions(DimensionCheckType::Enum check_op_type);

    inline int ReturnPaddedMemorySize(short4& wanted_dims) {
        // FIXME: Assumes a) SetInputDimensionsAndType has been called and is_fftw_padded is set before this call. (Currently RuntimeAssert to die if false)
        int           wanted_memory_n_elements             = 0;
        constexpr int scale_compute_base_type_to_full_type = 2;
        // The odd sized block is probably not needed.
        if constexpr ( IsAllowedRealType<InputType> ) {
            if ( wanted_dims.x % 2 == 0 ) {
                padding_jump_val_        = 2;
                wanted_memory_n_elements = wanted_dims.x / 2 + 1;
            }
            else {
                padding_jump_val_        = 1;
                wanted_memory_n_elements = (wanted_dims.x - 1) / 2 + 1;
            }

            wanted_memory_n_elements *= wanted_dims.y * wanted_dims.z; // other dimensions
            wanted_dims.w = (wanted_dims.x + padding_jump_val_) / 2; // number of complex elements in the X dimesnions after FFT.
        }
        else if constexpr ( IsAllowedComplexType<InputType> ) {
            wanted_memory_n_elements = wanted_dims.x * wanted_dims.y * wanted_dims.z;
            wanted_dims.w            = wanted_dims.x; // pitch is constant
        }
        else {
            constexpr InputType a;
            static_assert_type_name(a);
        }

        // Here wanted_memory_n_elements contains enough memory for in-place real/complex transforms.
        // We need to to scale it up as we use sizeof(compute_base_type) when allocating.
        wanted_memory_n_elements *= scale_compute_base_type_to_full_type;

        // FIXME: For 3d tranforms we need either need an additional buffer space or we will have to do an extra device to
        // device copy for simple forward/inverse transforms. For now, we'll do the extra copy to keep buffer assignments easier.
        // if constepxr (Rank == 3) {
        //     wanted_memory_n_elements *= 2;
        // }

        // The total compute_memory_wanted_ will depend on the largest image in the forward/inverse transform plans,
        // so we need to keep track of the largest value.
        compute_memory_wanted_ = std::max(compute_memory_wanted_, wanted_memory_n_elements);
        return wanted_memory_n_elements;
    }

    template <class FFT, class invFFT>
    void FFT_C2C_WithPadding_ConjMul_C2C_t(float2* image_to_search, bool swap_real_space_quadrants);
    template <class FFT, class invFFT>
    void FFT_C2C_decomposed_ConjMul_C2C_t(float2* image_to_search, bool swap_real_space_quadrants);

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

    // TODO: start hiding things that should not be public

  private:
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

    /*
    MEANING of KERNEL TYPE NAMES:
    
    - r2c and c2r are for real valued input/output images

    - any kernel with "decomposed" is a thread based routine (not currently supported)

    - if a kernel is part of a size change routine it is specified as none/increase/decrease
    
    - if 2 axes are specified, those dimensions are transposed.
        - 1d - this is meaningless
        - 2d - should always be XY
        - 3d - should always be XZ

    - if 3 axes are specified, those dimensions are permuted XYZ (only 3d)

    - any c2c FWD method without an axes is a terminal stage of a forward transform
    - any c2c INV method without an axes is a initial stage of an inverse transform

 */

    // FIXME: in the execution blocks, we should have some check that the correct direction is implemented.
    // Or better yet, have this templated and
    enum KernelType { r2c_decomposed, // 1D fwd
                      r2c_decomposed_transposed, // 2d fwd 1st stage
                      r2c_none_XY, // 1d fwd  //  2d fwd 1st stage
                      r2c_none_XZ, // 3d fwd 1st stage
                      r2c_decrease_XY,
                      r2c_increase_XY,
                      r2c_increase_XZ,
                      c2c_fwd_none, // 1d complex valued input, or final stage of Fwd 2d or 3d
                      c2c_fwd_none_XYZ,
                      c2c_fwd_decrease,
                      c2c_fwd_increase,
                      c2c_fwd_increase_XYZ,
                      c2c_inv_none,
                      c2c_inv_none_XZ,
                      c2c_inv_none_XYZ,
                      c2c_inv_decrease,
                      c2c_inv_increase,
                      c2c_fwd_decomposed,
                      c2c_inv_decomposed,
                      c2r_decomposed,
                      c2r_decomposed_transposed,
                      c2r_none,
                      c2r_none_XY,
                      c2r_decrease_XY,
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
                       "r2c_none_XY",
                       "r2c_none_XZ",
                       "r2c_decrease_XY",
                       "r2c_increase_XY",
                       "r2c_increase_XZ",
                       "c2c_fwd_none",
                       "c2c_fwd_none_XYZ",
                       "c2c_fwd_decrease",
                       "c2c_fwd_increase",
                       "c2c_fwd_increase_XYZ",
                       "c2c_inv_none",
                       "c2c_inv_none_XZ",
                       "c2c_inv_none_XYZ",
                       "c2c_inv_decrease",
                       "c2c_inv_increase",
                       "c2c_fwd_decomposed",
                       "c2c_inv_decomposed",
                       "c2r_decomposed",
                       "c2r_decomposed_transposed",
                       "c2r_none",
                       "c2r_none_XY",
                       "c2r_decrease_XY",
                       "c2r_increase",
                       "xcorr_fwd_increase_inv_none",
                       "xcorr_fwd_decrease_inv_none",
                       "xcorr_fwd_none_inv_decrease",
                       "xcorr_fwd_decrease_inv_decrease",
                       "xcorr_decomposed",
                       "generic_fwd_increase_op_inv_none"};

    // All in a column so it is obvious if a "==" is missing which will of course break the or co nditions and
    // always evaluate true.
    inline bool IsThreadType(KernelType kernel_type) {
        if ( kernel_type == r2c_decomposed ||
             kernel_type == r2c_decomposed_transposed ||
             kernel_type == c2c_fwd_decomposed ||
             kernel_type == c2c_inv_decomposed ||
             kernel_type == c2r_decomposed ||
             kernel_type == c2r_decomposed_transposed ||
             kernel_type == xcorr_decomposed ) {
            return true;
        }
        else {
            if ( kernel_type == r2c_none_XY ||
                 kernel_type == r2c_none_XZ ||
                 kernel_type == r2c_decrease_XY ||
                 kernel_type == r2c_increase_XY ||
                 kernel_type == r2c_increase_XZ ||
                 kernel_type == c2c_fwd_none ||
                 kernel_type == c2c_fwd_none_XYZ ||
                 kernel_type == c2c_fwd_decrease ||
                 kernel_type == c2c_fwd_increase ||
                 kernel_type == c2c_fwd_increase_XYZ ||
                 kernel_type == c2c_inv_none ||
                 kernel_type == c2c_inv_none_XZ ||
                 kernel_type == c2c_inv_none_XYZ ||
                 kernel_type == c2c_inv_decrease ||
                 kernel_type == c2c_inv_increase ||
                 kernel_type == c2r_none ||
                 kernel_type == c2r_none_XY ||
                 kernel_type == c2r_decrease_XY ||
                 kernel_type == c2r_increase ||
                 kernel_type == xcorr_fwd_increase_inv_none ||
                 kernel_type == xcorr_fwd_decrease_inv_none ||
                 kernel_type == xcorr_fwd_none_inv_decrease ||
                 kernel_type == xcorr_fwd_decrease_inv_decrease ||
                 kernel_type == generic_fwd_increase_op_inv_none ) {
                return false;
            }
            else {
                std::cerr << "Function IsThreadType does not recognize the kernel type ( " << KernelName[kernel_type] << " )" << std::endl;
                exit(-1);
            }
        }
    };

    inline bool IsR2CType(KernelType kernel_type) {
        if ( kernel_type == r2c_decomposed || kernel_type == r2c_decomposed_transposed ||
             kernel_type == r2c_none_XY || kernel_type == r2c_none_XZ ||
             kernel_type == r2c_decrease_XY || kernel_type == r2c_increase_XY || kernel_type == r2c_increase_XZ ) {
            return true;
        }
        else
            return false;
    }

    inline bool IsC2RType(KernelType kernel_type) {
        if ( kernel_type == c2r_decomposed || kernel_type == c2r_decomposed_transposed ||
             kernel_type == c2r_none || kernel_type == c2r_none_XY || kernel_type == c2r_decrease_XY || kernel_type == c2r_increase ) {
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
             kernel_type == r2c_decrease_XY || kernel_type == r2c_increase_XY || kernel_type == r2c_increase_XZ ||
             kernel_type == c2c_fwd_none || kernel_type == c2c_fwd_none_XYZ || kernel_type == c2c_fwd_increase_XYZ ||
             kernel_type == c2c_fwd_decrease ||
             kernel_type == c2c_fwd_increase ||
             kernel_type == xcorr_fwd_decrease_inv_none || kernel_type == xcorr_fwd_increase_inv_none ||
             kernel_type == generic_fwd_increase_op_inv_none ) {
            return true;
        }
        else
            return false;
    }

    void         GetTransformSize(KernelType kernel_type);
    void         GetTransformSize_thread(KernelType kernel_type, int thread_fft_size);
    LaunchParams SetLaunchParameters(const int& ept, KernelType kernel_type);

    // 1.
    // First call passed from a public transform function, selects block or thread and the transform precision.
    template <int FFT_ALGO_t, bool use_thread_method = false, class PreOpType = std::nullptr_t, class IntraOpType = std::nullptr_t, class PostOpType = std::nullptr_t>
    EnableIf<IfAppliesIntraOpFunctor_HasIntraOpFunctor<IntraOpType, FFT_ALGO_t>>
    SetPrecisionAndExectutionMethod(InputType*      input_ptr,
                                    OtherImageType* other_image_ptr,
                                    KernelType      kernel_type,
                                    PreOpType       pre_op_functor   = nullptr,
                                    IntraOpType     intra_op_functor = nullptr,
                                    PostOpType      post_op_functor  = nullptr);

    // 2. // TODO: remove this now that the functors are working
    // Check to see if any intra kernel functions are wanted, and if so set the appropriate device pointers.
    template <int FFT_ALGO_t, class FFT_base, class PreOpType, class IntraOpType, class PostOpType>
    void SetIntraKernelFunctions(InputType* input_ptr, OtherImageType* other_image_ptr, KernelType kernel_type, PreOpType pre_op_functor, IntraOpType intra_op_functor, PostOpType post_op_functor);

    // 3.
    // Second call, sets size of the transform kernel, selects the appropriate GPU arch

    // template <class FFT_base, class PreOpType, class IntraOpType, class PostOpType>
    // void SelectSizeAndType(KernelType kernel_type, bool do_forward_transform, PreOpType pre_op_functor, IntraOpType intra_op_functor, PostOpType post_op_functor);
    // This allows us to iterate through a set of constexpr sizes passed as a template parameter pack. The value is in providing a means to have different size packs
    // for different fft configurations, eg. 2d vs 3d
    template <int FFT_ALGO_t, class FFT_base, class PreOpType, class IntraOpType, class PostOpType>
    void SelectSizeAndType(InputType* input_ptr, OtherImageType* other_image_ptr, KernelType kernel_type, PreOpType pre_op_functor, IntraOpType intra_op_functor, PostOpType post_op_functor);

    template <int FFT_ALGO_t, class FFT_base, class PreOpType, class IntraOpType, class PostOpType, unsigned int SizeValue, unsigned int Ept, unsigned int... OtherValues>
    void SelectSizeAndType(InputType* input_ptr, OtherImageType* other_image_ptr, KernelType kernel_type, PreOpType pre_op_functor, IntraOpType intra_op_functor, PostOpType post_op_functor);

    // 3.
    // Third call, sets the input and output dimensions and type
    template <int FFT_ALGO_t, class FFT_base_arch, class PreOpType, class IntraOpType, class PostOpType>
    void SetAndLaunchKernel(InputType* input_ptr, OtherImageType* other_image_ptr, KernelType kernel_type, PreOpType pre_op_functor, IntraOpType intra_op_functor, PostOpType post_op_functor);

    short padding_jump_val_;
    int   input_memory_wanted_;
    int   fwd_output_memory_wanted_;
    int   inv_output_memory_wanted_;
    int   compute_memory_wanted_;
    int   memory_size_to_copy_;

    bool input_data_is_on_device;
    bool output_data_is_on_device;
    bool external_image_is_on_device;
    void AllocateBufferMemory( );

    template <class PreOpType,
              class IntraOpType,
              class PostOpType>
    EnableIf<HasIntraOpFunctor<IntraOpType> && IsAllowedInputType<InputType, OtherImageType>>
    Generic_Fwd_Image_Inv(InputType*      input_ptr,
                          OtherImageType* image_to_search,
                          PreOpType       pre_op,
                          IntraOpType     intra_op,
                          PostOpType      post_op);

    template <class PreOpType,
              class IntraOpType>
    EnableIf<IsAllowedInputType<InputType>>
    Generic_Fwd(InputType*  input_ptr,
                PreOpType   pre_op,
                IntraOpType intra_op);

    template <class IntraOpType,
              class PostOpType>
    EnableIf<IsAllowedInputType<InputType>>
    Generic_Inv(InputType*  input_ptr,
                IntraOpType intra_op,
                PostOpType  post_op);

    inline bool IsTransormAlongZ(KernelType kernel_type) {
        if ( kernel_type == c2c_fwd_none_XYZ || kernel_type == c2c_fwd_increase_XYZ ||
             kernel_type == c2c_inv_none_XYZ ) {
            return true;
        }
        else
            return false;
    }

    inline bool IsRank3(KernelType kernel_type) {
        if ( kernel_type == r2c_none_XZ || kernel_type == r2c_increase_XZ ||
             kernel_type == c2c_fwd_increase_XYZ || kernel_type == c2c_inv_none_XZ ||
             kernel_type == c2c_fwd_none_XYZ || kernel_type == c2c_inv_none_XYZ ) {
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

    // Input is real or complex inferred from InputType
    DevicePointers<ComputeBaseType*, InputType*, OtherImageType*> d_ptr;
    // Check to make sure we haven't fouled up the explicit instantiation of DevicePointers

}; // class Fourier Transformer

} // namespace FastFFT

#endif
