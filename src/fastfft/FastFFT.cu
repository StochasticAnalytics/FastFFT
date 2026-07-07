// Insert some license stuff here

// #include <string>
#include <iostream>
#include <string>
#include <stdio.h>

#include "../../include/FastFFT.cuh"

#ifndef FFT_DEBUG_STAGE
#error "FFT_DEBUG_STAGE must be defined"
#endif

#ifndef FFT_DEBUG_LEVEL
#error "FFT_DEBUG_LEVEL must be defined"
#endif

#ifndef c2r_multiplier
#define c2r_multiplier 4
#endif

#ifndef minBlocksPerMultiprocessor
#define minBlocksPerMultiprocessor 3
#endif

// The cufftdx library code assumes that both shared_memory and shared_memory_input are aligned to 128 bits for optimal memory operations.
// rather than saying extern __shared__ __align__(16) value_type shared_mem[]; we define the following instead.
#define FastFFT_SMEM extern __shared__ __align__(16)

// clang-format off
#define FastFFT_PRINT_DUMP(LP) \
    std::cerr << "\nFastFFT Debug Dump\n"; \
    PrintState(); \
    PrintLaunchParameters(LP); \
    std::cerr << "Smem :" << shared_memory << std::endl; \
    std::cerr << "Ept: " << FFT::elements_per_thread << std::endl; \
    std::cerr << "Shared memory size: " << FFT::shared_memory_size << std::endl;

// clang-format on
namespace FastFFT {

// If we are using the suggested ept we need to scale some down that otherwise requiest too many registers as cufftdx is unaware of the higher order transforms.
template <class Description>
struct check_and_set_ept {
    static_assert(is_fft<Description>::value, "Description is not a cuFFTDx FFT description");
// Get the existing elements per thread

// FIXME: The same logic should be added to USE_SUPPLIED_EPT
// FIXME: THE SAME logic should be applied if USE_FOLDED_R2C is implemented
#ifdef USE_FOLDED_C2R
#ifdef C2R_BUFFER_LINES

    // FIXME: for now assuming size 64 so set to 2 to get 32 threads, may not be needed (min ept must actually be 4) revert
    static constexpr unsigned int using_ept = size_of<Description>::value < 256 ? 4 : size_of<Description>::value < 1024 ? 8
                                                                              : size_of<Description>::value < 4096       ? 16
                                                                                                                         : 32;

    using check_ept = std::conditional_t<type_of<Description>::value == fft_type::c2r,
                                         cufftdx::replace_t<Description, ElementsPerThread<using_ept>>,
                                         Description>;
#else
    using check_ept = std::conditional_t<type_of<Description>::value == fft_type::c2r,
                                         cufftdx::replace_t<Description, ElementsPerThread<std::min(32u, Description::elements_per_thread* c2r_multiplier)>>,
                                         Description>;
#endif // buffer lines
#else
    using check_ept = Description;
#endif

  public:
    using type = check_ept;
};
template <class Description>
using check_ept_t = typename check_and_set_ept<Description>::type;

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
                                                           PostOpType                      post_op_functor) {

    // Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    FastFFT_SMEM complex_compute_t shared_mem[];

    complex_compute_t thread_data[FFT::storage_size];

    // All threads must participate in the FFT due to shared memory fences, so there
    // is no way to merge kernels w/ different sized FFTs, so we are forced to explicitly
    // zero-pad the input on the forward.
    io<FFT>::load(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)],
                  thread_data,
                  mem_offsets.physical_x_input,
                  pre_op_functor);

    // In the first FFT the modifying twiddle factor is 1 so the data are reeal
    FFT( ).execute(thread_data, shared_mem, workspace_fwd);

#if FFT_DEBUG_STAGE > 3
    //  * apparent_Q
    io<invFFT>::load_external_data(&image_to_search[Return1DFFTAddress(size_of<invFFT>::value)],
                                   thread_data,
                                   intra_op_functor);
#endif

#if FFT_DEBUG_STAGE > 4
    invFFT( ).execute(thread_data, shared_mem, workspace_inv);
    io<invFFT>::store(thread_data, &output_values[Return1DFFTAddress(size_of<invFFT>::value)], post_op_functor);
#else
    // Do not do the post op lambda if the invFFT is not used.
    io<invFFT>::store(thread_data, &output_values[Return1DFFTAddress(size_of<invFFT>::value)]);
#endif
}

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::FourierTransformer( ) {
    SetDefaults( );
    GetCudaDeviceProps(device_properties);
    // FIXME: assert on OtherImageType being a complex type
    static_assert(std::is_same_v<ComputeBaseType, float>, "Compute base type must be float");
    static_assert(Rank == 2 || Rank == 3, "Only 2D and 3D FFTs are supported");

    // exit(0);
    // This assumption precludes the use of a packed _half2 that is really RRII layout for two arrays of __half.
    static_assert(IsAllowedRealType<PositionSpaceType> || IsAllowedComplexType<PositionSpaceType>, "Input type must be either float or __half, support for complex input types is not yet implemented.");

    // Make sure an explicit specializtion for the device pointers is available
    static_assert(! std::is_same_v<decltype(d_ptr.buffer_1), std::nullptr_t>, "Device pointer type not specialized");
#ifdef FastFFT_DEBUG_BUILD_TIME
    std::cerr << "Init with  FastFFT object using code build on " << __DATE__ << " " << __TIME__ << std::endl;
#endif
}

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::~FourierTransformer( ) {
    Deallocate( );
    SetDefaults( );
}

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::SetDefaults( ) {

    // booleans to track state, could be bit fields but that seem opaque to me.
    current_buffer            = fastfft_external_input;
    transform_stage_completed = 0;

    fwd_implicit_dimension_change = false;
    inv_implicit_dimension_change = false;

    is_fftw_padded_input  = false; // Padding for in place r2c transforms
    is_fftw_padded_output = false; // Currently the output state will match the input state, otherwise it is an error.

    is_set_input_params  = false; // Yes, yes, "are" set.
    is_set_output_params = false;

    input_data_is_on_device     = false;
    output_data_is_on_device    = false;
    external_image_is_on_device = false;

    compute_memory_wanted_ = 0;
}

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::Deallocate( ) {

    if ( is_pointer_in_device_memory(d_ptr.buffer_1) ) {
        precheck;
        cudaErr(cudaFreeAsync(d_ptr.buffer_1, cudaStreamPerThread));
        postcheck;

        // For now sync so the state variable is accurate. We don't do much allcoation/deallocation so this is not a big deal.
        cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
        n_bytes_allocated = 0;
    }
}

/**
 * @brief Create a forward FFT plan. 
 * Buffer memory is allocated on the latter of creating forward/inverse plans.
 * Data may be copied to this buffer and used directly 
 * 
 * @tparam ComputeBaseType 
 * @tparam PositionSpaceType 
 * @tparam OtherImageType 
 * @tparam Rank 
 * @param input_logical_x_dimension 
 * @param input_logical_y_dimension 
 * @param input_logical_z_dimension 
 * @param output_logical_x_dimension 
 * @param output_logical_y_dimension 
 * @param output_logical_z_dimension 
 * @param is_padded_input 
 */

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::SetForwardFFTPlan(size_t input_logical_x_dimension,
                                                                                                     size_t input_logical_y_dimension,
                                                                                                     size_t input_logical_z_dimension,
                                                                                                     size_t output_logical_x_dimension,
                                                                                                     size_t output_logical_y_dimension,
                                                                                                     size_t output_logical_z_dimension,
                                                                                                     bool   is_padded_input) {

    fwd_dims_in  = make_short4(input_logical_x_dimension, input_logical_y_dimension, input_logical_z_dimension, 0);
    fwd_dims_out = make_short4(output_logical_x_dimension, output_logical_y_dimension, output_logical_z_dimension, 0);
    ValidateDimensions(fwd_dims_in, fwd_dims_out, true);

    is_fftw_padded_input = is_padded_input;
    MyFFTRunTimeAssertTrue(is_fftw_padded_input, "Support for input arrays that are not FFTW padded needs to be implemented."); // FIXME

    // ReturnPaddedMemorySize also sets FFTW padding etc.
    input_memory_wanted_ = ReturnPaddedMemorySize(fwd_dims_in);

    // sets .w and also increases compute_memory_wanted_ if needed.
    fwd_output_memory_wanted_ = ReturnPaddedMemorySize(fwd_dims_out);

    // The compute memory allocated is the max of all possible sizes.

    this->input_origin_type = OriginType::natural;
    is_set_input_params     = true;

    // Only run when both input and output params are set
    if ( is_set_output_params ) {
        AllocateBufferMemory( );
    }
}

/**
 * @brief Create an inverse FFT plan. 
 * Buffer memory is allocated on the latter of creating forward/inverse plans.
 * Data may be copied to this buffer and used directly 
 * 
 * @tparam ComputeBaseType 
 * @tparam PositionSpaceType 
 * @tparam OtherImageType 
 * @tparam Rank 
 * @param input_logical_x_dimension 
 * @param input_logical_y_dimension 
 * @param input_logical_z_dimension 
 * @param output_logical_x_dimension 
 * @param output_logical_y_dimension 
 * @param output_logical_z_dimension 
 * @param is_padded_output 
 */
template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::SetInverseFFTPlan(size_t input_logical_x_dimension,
                                                                                                     size_t input_logical_y_dimension,
                                                                                                     size_t input_logical_z_dimension,
                                                                                                     size_t output_logical_x_dimension,
                                                                                                     size_t output_logical_y_dimension,
                                                                                                     size_t output_logical_z_dimension,
                                                                                                     bool   is_padded_output) {

    MyFFTDebugAssertTrue(is_fftw_padded_input == is_padded_output, "If the input data are FFTW padded, so must the output.");

    inv_dims_in  = make_short4(input_logical_x_dimension, input_logical_y_dimension, input_logical_z_dimension, 0);
    inv_dims_out = make_short4(output_logical_x_dimension, output_logical_y_dimension, output_logical_z_dimension, 0);
    ValidateDimensions(inv_dims_in, inv_dims_out, false);

    ReturnPaddedMemorySize(inv_dims_in); // sets .w and also increases compute_memory_wanted_ if needed.
    inv_output_memory_wanted_ = ReturnPaddedMemorySize(inv_dims_out);
    // The compute memory allocated is the max of all possible sizes.

    this->output_origin_type = OriginType::natural;
    is_set_output_params     = true;
    // Only run when both input and output params are set
    if ( is_set_input_params ) {
        AllocateBufferMemory( );
    }
}

/**
 * @brief Private method to allocate memory for the internal FastFFT buffer.
 * 
 * @tparam ComputeBaseType 
 * @tparam PositionSpaceType 
 * @tparam OtherImageType 
 * @tparam Rank 
 */
template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::AllocateBufferMemory( ) {
    MyFFTDebugAssertTrue(is_set_input_params && is_set_output_params, "Input and output parameters must be set before allocating buffer memory");

    MyFFTDebugAssertTrue(compute_memory_wanted_ > 0, "Compute memory already allocated");

    // Allocate enough for the out of place buffer as well.
    constexpr size_t compute_memory_scalar = 2;
    // To get the address of the second buffer we want half of the number of ComputeType, not ComputeBaseType elements
    constexpr size_t buffer_address_scalar = 2;
    precheck;
    cudaErr(cudaMallocAsync(&d_ptr.buffer_1, compute_memory_scalar * compute_memory_wanted_ * sizeof(ComputeBaseType), cudaStreamPerThread));
    postcheck;
    n_bytes_allocated = compute_memory_scalar * compute_memory_wanted_ * sizeof(ComputeBaseType);
    // cudaMallocAsync returns the pointer immediately, even though the allocation has not yet completed, so we
    // should be fine to go on and point our secondary buffer to the correct location.
    d_ptr.buffer_2 = &d_ptr.buffer_1[compute_memory_wanted_ / buffer_address_scalar];
}

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::ZeroBufferMemory( ) {
    MyFFTDebugAssertTrue(is_set_input_params && is_set_output_params, "Input and output parameters must be set before allocating buffer memory");
    MyFFTDebugAssertTrue(n_bytes_allocated > 0, "No  memory has been allocated");
    // Allocate enough for the out of place buffer as well.

    precheck;
    cudaErr(cudaMemsetAsync(d_ptr.buffer_1, 0, n_bytes_allocated, cudaStreamPerThread));
    postcheck;
}

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::SetInputPointerFromPython(long input_pointer) {

    MyFFTRunTimeAssertFalse(true, "This needs to be re-implemented.");
    //         MyFFTDebugAssertTrue(is_set_input_params, "Input parameters not set");

    // // The assumption for now is that access from python wrappers have taken care of device/host xfer
    // // and the passed pointer is in device memory.
    // // TODO: I should probably have a state variable to track is_python_call
    // d_ptr.position_space        = reinterpret_cast<PositionSpaceType*>(input_pointer);

    // // These are normally set on CopyHostToDevice
    // SetDevicePointers( );
}

// FIXME: see header file for comments
template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::CopyHostToDeviceAndSynchronize(PositionSpaceType* input_pointer, int n_elements_to_copy) {
    CopyHostToDevice(input_pointer, n_elements_to_copy);
    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
}

// FIXME: see header file for comments
template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::CopyHostToDevice(PositionSpaceType* input_pointer, int n_elements_to_copy) {
    MyFFTDebugAssertFalse(input_data_is_on_device, "External input pointer is on device, cannot copy from host");
    MyFFTRunTimeAssertTrue(false, "This method is being removed.");
    SetDimensions(DimensionCheckType::CopyFromHost);

    precheck;
    cudaErr(cudaMemcpyAsync(d_ptr.buffer_1, input_pointer, memory_size_to_copy_ * sizeof(PositionSpaceType), cudaMemcpyHostToDevice, cudaStreamPerThread));
    postcheck;
}

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
template <class PreOpType,
          class IntraOpType>
void FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::FwdFFT(PositionSpaceType* input_ptr,
                                                                                          PositionSpaceType* output_ptr,
                                                                                          PreOpType          pre_op,
                                                                                          IntraOpType        intra_op) {

    transform_stage_completed = 0;
    current_buffer            = fastfft_external_input;
    // Keep track of the device side pointer used when called
    d_ptr.external_input  = input_ptr;
    d_ptr.external_output = output_ptr;
    Generic_Fwd(pre_op, intra_op);
}

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
template <class IntraOpType,
          class PostOpType>
void FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::InvFFT(PositionSpaceType* input_ptr,
                                                                                          PositionSpaceType* output_ptr,
                                                                                          IntraOpType        intra_op,
                                                                                          PostOpType         post_op) {
    transform_stage_completed = 4;
    current_buffer            = fastfft_external_input;
    // Keep track of the device side pointer used when called
    d_ptr.external_input  = input_ptr;
    d_ptr.external_output = output_ptr;

    Generic_Inv(intra_op, post_op);
}

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>

template <class PreOpType,
          class IntraOpType,
          class PostOpType>
void FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::FwdImageInvFFT(PositionSpaceType* input_ptr,
                                                                                                  OtherImageType*    image_to_search,
                                                                                                  PositionSpaceType* output_ptr,
                                                                                                  PreOpType          pre_op,
                                                                                                  IntraOpType        intra_op,
                                                                                                  PostOpType         post_op) {
    transform_stage_completed = 0;
    current_buffer            = fastfft_external_input;
    // Keep track of the device side pointer used when called
    d_ptr.external_input  = input_ptr;
    d_ptr.external_output = output_ptr;

    Generic_Fwd_Image_Inv<PreOpType, IntraOpType, PostOpType>(image_to_search, pre_op, intra_op, post_op);
}

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
template <class PreOpType,
          class IntraOpType>
EnableIf<IsAllowedPositionSpaceType<PositionSpaceType>>
FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::Generic_Fwd(PreOpType   pre_op_functor,
                                                                                          IntraOpType intra_op_functor) {

    SetDimensions(DimensionCheckType::FwdTransform);

    // TODO: extend me
    MyFFTRunTimeAssertFalse(fwd_implicit_dimension_change, "Implicit dimension change not yet supported for FwdFFT");
    MyFFTRunTimeAssertFalse(inv_implicit_dimension_change, "Implicit dimension change not yet supported for FwdFFT");

    // SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(<Generic_Inv_FFT, KernelType kernel_type, bool  bool use_thread_method)
    if constexpr ( Rank == 1 ) {

        if constexpr ( IsAllowedRealType<PositionSpaceType> ) {
            switch ( fwd_size_change_type ) {
                case SizeChangeType::no_change: {
                    SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, r2c_none_XY, pre_op_functor, intra_op_functor);
                    transform_stage_completed = 1;
                    break;
                }
                case SizeChangeType::decrease: {
                    SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, r2c_decrease_XY, pre_op_functor, intra_op_functor);
                    transform_stage_completed = 1;
                    break;
                }
                case SizeChangeType::increase: {
                    SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, r2c_increase_XY, pre_op_functor, intra_op_functor);
                    transform_stage_completed = 1;
                    break;
                }
                default: {
                    MyFFTDebugAssertTrue(false, "Invalid size change type");
                }
            }
        }
        else {
            switch ( fwd_size_change_type ) {
                case SizeChangeType::no_change: {
                    MyFFTDebugAssertTrue(false, "Complex input images are not yet supported"); // FIXME:
                    SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, c2c_fwd_none, pre_op_functor, intra_op_functor);
                    transform_stage_completed = 1;
                    break;
                }
                case SizeChangeType::decrease: {
                    SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, c2c_fwd_decrease, pre_op_functor, intra_op_functor);
                    transform_stage_completed = 1;
                    break;
                }
                case SizeChangeType::increase: {
                    SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, c2c_fwd_increase, pre_op_functor, intra_op_functor);
                    transform_stage_completed = 1;
                    break;
                }
                default: {
                    MyFFTDebugAssertTrue(false, "Invalid size change type");
                }
            }
        }
    }
    else if constexpr ( Rank == 2 ) {
        switch ( fwd_size_change_type ) {
            case SizeChangeType::no_change: {
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, r2c_none_XY, pre_op_functor, intra_op_functor);
                transform_stage_completed = 1;
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, c2c_fwd_none, pre_op_functor, intra_op_functor);
                transform_stage_completed = 3;
                break;
            }
            case SizeChangeType::increase: {
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, r2c_increase_XY, pre_op_functor, intra_op_functor);
                transform_stage_completed = 1;
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, c2c_fwd_increase, pre_op_functor, intra_op_functor);
                transform_stage_completed = 3;
                break;
            }
            case SizeChangeType::decrease: {
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, r2c_decrease_XY, pre_op_functor, intra_op_functor);
                transform_stage_completed = 1;
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, c2c_fwd_decrease, pre_op_functor, intra_op_functor);
                transform_stage_completed = 3;
                break;
            }
        }
    }
    else if constexpr ( Rank == 3 ) {
        switch ( fwd_size_change_type ) {
            case SizeChangeType::no_change: {
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, r2c_none_XZ, pre_op_functor, intra_op_functor);
                transform_stage_completed = 1;
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, c2c_fwd_none_XYZ, pre_op_functor, intra_op_functor);
                transform_stage_completed = 2;
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, c2c_fwd_none, pre_op_functor, intra_op_functor);
                transform_stage_completed = 3;
                break;
            }
            case SizeChangeType::increase: {
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, r2c_increase_XZ, pre_op_functor, intra_op_functor);
                transform_stage_completed = 1;
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, c2c_fwd_increase_XYZ, pre_op_functor, intra_op_functor);
                transform_stage_completed = 2;
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, c2c_fwd_increase, pre_op_functor, intra_op_functor);
                transform_stage_completed = 3;
                break;
            }
            case SizeChangeType::decrease: {
                // Not yet supported
                MyFFTRunTimeAssertTrue(false, "3D FFT fwd decrease not yet supported");
                break;
            }
        }
    }
    else {
        MyFFTDebugAssertTrue(false, "Invalid rank");
    }
}

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
template <class IntraOpType,
          class PostOpType>
EnableIf<IsAllowedPositionSpaceType<PositionSpaceType>>
FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::Generic_Inv(IntraOpType intra_op,
                                                                                          PostOpType  post_op) {

    SetDimensions(DimensionCheckType::InvTransform);
    MyFFTRunTimeAssertFalse(fwd_implicit_dimension_change, "Implicit dimension change not yet supported for InvFFT");
    MyFFTRunTimeAssertFalse(inv_implicit_dimension_change, "Implicit dimension change not yet supported for InvFFT");

    switch ( Rank ) {
        case 1: {

            if constexpr ( IsAllowedRealType<PositionSpaceType> ) {
                switch ( inv_size_change_type ) {
                    case SizeChangeType::no_change: {
                        SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2r_none_XY, intra_op, post_op);
                        transform_stage_completed = 5;
                        break;
                    }
                    case SizeChangeType::decrease: {
                        SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2r_decrease_XY, intra_op, post_op);
                        transform_stage_completed = 5;
                        break;
                    }
                    case SizeChangeType::increase: {
                        SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2r_increase, intra_op, post_op);
                        transform_stage_completed = 5;
                        break;
                    }
                    default: {
                        MyFFTDebugAssertTrue(false, "Invalid size change type");
                    }
                }
            }
            else {
                switch ( inv_size_change_type ) {
                    case SizeChangeType::no_change: {
                        SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2c_inv_none, intra_op, post_op);
                        transform_stage_completed = 5;
                        break;
                    }
                    case SizeChangeType::decrease: {
                        SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2c_inv_decrease, intra_op, post_op);
                        transform_stage_completed = 5;
                        break;
                    }
                    case SizeChangeType::increase: {
                        SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2c_inv_increase, intra_op, post_op);
                        transform_stage_completed = 5;
                        break;
                    }
                    default: {
                        MyFFTDebugAssertTrue(false, "Invalid size change type");
                    }
                }
            }
            break;
        }
        case 2: {
            switch ( inv_size_change_type ) {
                case SizeChangeType::no_change: {
                    SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2c_inv_none, intra_op, post_op);
                    transform_stage_completed = 5;
                    SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2r_none_XY, intra_op, post_op);
                    transform_stage_completed = 7;
                    break;
                }
                case SizeChangeType::increase: {
                    SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2c_inv_increase, intra_op, post_op);
                    transform_stage_completed = 5;
                    SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2r_increase, intra_op, post_op);
                    transform_stage_completed = 7;
                    break;
                }
                case SizeChangeType::decrease: {
                    SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2c_inv_decrease, intra_op, post_op);
                    transform_stage_completed = 5;
                    SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2r_decrease_XY, intra_op, post_op);
                    transform_stage_completed = 7;
                    break;
                }
                default: {
                    MyFFTDebugAssertTrue(false, "Invalid size change type");
                    break;
                }
            } // switch on inv size change type
            break; // case 2
        }
        case 3: {
            switch ( inv_size_change_type ) {
                case SizeChangeType::no_change: {
                    SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2c_inv_none_XZ, intra_op, post_op);
                    transform_stage_completed = 5;
                    SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2c_inv_none_XYZ, intra_op, post_op);
                    transform_stage_completed = 6;
                    SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2r_none, intra_op, post_op);
                    transform_stage_completed = 7;
                    break;
                }
                case SizeChangeType::increase: {
                    MyFFTRunTimeAssertFalse(true, "3D FFT inv increase not yet supported");
                    SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, r2c_increase_XY, intra_op, post_op);
                    // SetPrecisionAndExectutionMethod<Generic_Inv_FFT>( nullptr, c2c_fwd_increase_XYZ);
                    break;
                }
                case SizeChangeType::decrease: {
                    // Not yet supported
                    MyFFTRunTimeAssertTrue(false, "3D FFT inv no decrease not yet supported");
                    break;
                }
                default: {
                    MyFFTDebugAssertTrue(false, "Invalid dimension");
                    break;
                }
            } // switch on inv size change type
        }
    }
}

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
template <class PreOpType,
          class IntraOpType,
          class PostOpType>
EnableIf<HasIntraOpFunctor<IntraOpType> && IsAllowedPositionSpaceType<PositionSpaceType, OtherImageType>>
FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::Generic_Fwd_Image_Inv(OtherImageType* image_to_search_ptr,
                                                                                                    PreOpType       pre_op_functor,
                                                                                                    IntraOpType     intra_op_functor,
                                                                                                    PostOpType      post_op_functor) {

    // Set the member pointer to the passed pointer
    SetDimensions(DimensionCheckType::FwdTransform);

    switch ( Rank ) {
        case 1: {
            MyFFTRunTimeAssertTrue(false, "1D FFT Cross correlation not yet supported");
            break;
        }
        case 2: {
            switch ( fwd_size_change_type ) {
                case SizeChangeType::no_change: {
                    SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(image_to_search_ptr, r2c_none_XY, pre_op_functor, intra_op_functor, post_op_functor);
                    transform_stage_completed = 1;
                    switch ( inv_size_change_type ) {
                        case SizeChangeType::no_change: {
                            MyFFTRunTimeAssertTrue(false, "2D FFT generic lambda no change/nochange not yet supported");
                            break;
                        }
                        case SizeChangeType::increase: {
                            MyFFTRunTimeAssertTrue(false, "2D FFT generic lambda no change/increase not yet supported");
                            break;
                        }
                        case SizeChangeType::decrease: {
                            SetPrecisionAndExectutionMethod<Generic_Fwd_Image_Inv_FFT>(image_to_search_ptr, xcorr_fwd_none_inv_decrease, pre_op_functor, intra_op_functor, post_op_functor);
                            transform_stage_completed = 5;
                            SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(image_to_search_ptr, c2r_decrease_XY, pre_op_functor, intra_op_functor, post_op_functor);
                            transform_stage_completed = 7;
                            break;
                        }
                        default: {
                            MyFFTDebugAssertTrue(false, "Invalid size change type");
                            break;
                        }
                    } // switch on inv size change type
                    break;
                } // case fwd no change
                case SizeChangeType::increase: {
                    SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(image_to_search_ptr, r2c_increase_XY, pre_op_functor, intra_op_functor, post_op_functor);
                    transform_stage_completed = 1;
                    switch ( inv_size_change_type ) {
                        case SizeChangeType::no_change: {
                            SetPrecisionAndExectutionMethod<Generic_Fwd_Image_Inv_FFT>(image_to_search_ptr, generic_fwd_increase_op_inv_none, pre_op_functor, intra_op_functor, post_op_functor);
                            transform_stage_completed = 5;
                            SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(image_to_search_ptr, c2r_none_XY, pre_op_functor, intra_op_functor, post_op_functor);
                            transform_stage_completed = 7;
                            break;
                        }

                        case SizeChangeType::increase: {
                            // I don't see where increase increase makes any sense
                            // FIXME add a check on this in the validation step.
                            MyFFTRunTimeAssertTrue(false, "2D FFT Cross correlation with fwd and inv size increase is not supported");
                            break;
                        }
                        case SizeChangeType::decrease: {
                            // with FwdTransform set, call c2c
                            // Set InvTransform
                            // Call new kernel that handles the conj mul inv c2c trimmed, and inv c2r in one go.
                            MyFFTRunTimeAssertTrue(false, "2D FFT Cross correlation with fwd increase and inv size decrease is a work in progress");

                            break;
                        }
                        default: {
                            MyFFTRunTimeAssertTrue(false, "Invalid size change type");
                        }
                    } // switch on inv size change type
                    break;
                }
                case SizeChangeType::decrease: {

                    SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(image_to_search_ptr, r2c_decrease_XY, pre_op_functor, intra_op_functor, post_op_functor);
                    transform_stage_completed = 1;
                    switch ( inv_size_change_type ) {
                        case SizeChangeType::no_change: {
                            SetPrecisionAndExectutionMethod<Generic_Fwd_Image_Inv_FFT>(image_to_search_ptr, generic_fwd_increase_op_inv_none, pre_op_functor, intra_op_functor, post_op_functor);
                            transform_stage_completed = 5;
                            SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(image_to_search_ptr, c2r_none_XY, pre_op_functor, intra_op_functor, post_op_functor);
                            transform_stage_completed = 7;
                            break;
                        }
                        case SizeChangeType::increase: {

                            MyFFTRunTimeAssertTrue(false, "2D FFT Cross correlation with fwd and inv size increase is not supported");
                            break;
                        }
                        case SizeChangeType::decrease: {

                            MyFFTRunTimeAssertTrue(false, "2D FFT Cross correlation with fwd decrease and inv size decrease is a work in progress");
                            break;
                        }
                        default: {
                            MyFFTRunTimeAssertTrue(false, "Invalid inv size change type");
                        } break;
                    }
                    break;
                } // case decrease
                default: {
                    MyFFTRunTimeAssertTrue(false, "Invalid fwd size change type");
                }

            } // switch on fwd size change type
            break; // case dimension 2
        }
        case 3: {
            switch ( fwd_size_change_type ) {
                case SizeChangeType::no_change: {
                    MyFFTDebugAssertTrue(false, "3D FFT Cross correlation fwd no change not yet supported");
                    switch ( inv_size_change_type ) {
                        case SizeChangeType::no_change: {
                            break;
                        }
                        case SizeChangeType::increase: {
                            MyFFTDebugAssertTrue(false, "3D FFT Cross correlation with fwd and inv size increase is not supported");
                            break;
                        }
                        case SizeChangeType::decrease: {
                            MyFFTDebugAssertTrue(false, "3D FFT Cross correlation with fwd and inv size decrease is not supported");
                            break;
                        }
                    }

                    break;
                }

                case SizeChangeType::increase: {
                    SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(image_to_search_ptr, r2c_increase_XZ, pre_op_functor, intra_op_functor, post_op_functor);
                    transform_stage_completed = 1;
                    SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(image_to_search_ptr, c2c_fwd_increase_XYZ, pre_op_functor, intra_op_functor, post_op_functor);
                    transform_stage_completed = 2;
                    switch ( inv_size_change_type ) {
                        case SizeChangeType::no_change: {
                            // TODO: will need a kernel for generic_fwd_increase_op_inv_none_XZ
                            SetPrecisionAndExectutionMethod<Generic_Fwd_Image_Inv_FFT>(image_to_search_ptr, generic_fwd_increase_op_inv_none, pre_op_functor, intra_op_functor, post_op_functor);
                            transform_stage_completed = 5;
                            // SetPrecisionAndExectutionMethod<Generic_Fwd_Image_Inv_FFT>( image_to_search_ptr, c2c_inv_none_XZ);
                            SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(image_to_search_ptr, c2c_inv_none_XYZ, pre_op_functor, intra_op_functor, post_op_functor);
                            transform_stage_completed = 6;
                            SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(image_to_search_ptr, c2r_none, pre_op_functor, intra_op_functor, post_op_functor);
                            transform_stage_completed = 7;
                            break;
                        }
                        case SizeChangeType::increase: {
                            MyFFTDebugAssertTrue(false, "3D FFT Cross correlation with fwd and inv size increase is not supported");
                            break;
                        }
                        case SizeChangeType::decrease: {
                            MyFFTDebugAssertTrue(false, "3D FFT Cross correlation with fwd and inv size decrease is not supported");
                            break;
                        }
                        default: {
                            MyFFTRunTimeAssertTrue(false, "Invalid inv size change type");
                        }
                    }
                    break;
                }
                case SizeChangeType::decrease: {
                    MyFFTDebugAssertTrue(false, "3D FFT Cross correlation fwd decrease not yet supported");
                    switch ( inv_size_change_type ) {
                        case SizeChangeType::no_change: {
                            break;
                        }
                        case SizeChangeType::increase: {
                            MyFFTDebugAssertTrue(false, "3D FFT Cross correlation with fwd and inv size increase is not supported");
                            break;
                        }
                        case SizeChangeType::decrease: {
                            MyFFTDebugAssertTrue(false, "3D FFT Cross correlation with fwd and inv size decrease is not supported");
                            break;
                        }
                    }
                    break;
                }
                default: {
                    MyFFTRunTimeAssertTrue(false, "Invalid fwd size change type");
                }
            }
        }
        default: {
            MyFFTRunTimeAssertTrue(false, "Invalid dimension");
        }
    } // switch on transform dimension
}

////////////////////////////////////////////////////
/// END PUBLIC METHODS
////////////////////////////////////////////////////
template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::ValidateDimensions(short4& dims_in, short4& dims_out, const bool is_fwd_not_inv) {

    /* Basic conditions on allowed (supported sizes)
        - Transforms must be power of 2
        - We need to confirm that this build includes the requested size
        - Signal length may be non-power-2 for forward increase, as long as fwd output is an allowed size > signal length
        - Current allowed cases require the fwd output size = inv input size as size changes are only from signal -> fourier or fourier -> signal
        - Currently only allowing square or cubic images
    */

    // Validate the forward transform
    if ( dims_out.x > dims_in.x || dims_out.y > dims_in.y || dims_out.z > dims_in.z ) {
        // For now we must pad in all dimensions, this is not needed and should be lifted. FIXME
        MyFFTRunTimeAssertTrue(dims_out.x >= dims_in.x, "If padding, all dimensions must be >=, x out < x in");
        MyFFTRunTimeAssertTrue(dims_out.y >= dims_in.y, "If padding, all dimensions must be >=, y out < y in");
        MyFFTRunTimeAssertTrue(dims_out.z >= dims_in.z, "If padding, all dimensions must be >=, z out < z in");
        MyFFTRunTimeAssertTrue(IsAPowerOfTwo(dims_out.x) && IsAPowerOfTwo(dims_out.y) && IsAPowerOfTwo(dims_out.z), "Output dimensions must be a power of 2");

        if ( is_fwd_not_inv ) {
            fwd_size_change_type          = SizeChangeType::increase;
            fwd_implicit_dimension_change = ! (IsAPowerOfTwo(dims_out.x) && IsAPowerOfTwo(dims_out.y) && IsAPowerOfTwo(dims_out.z));
        }
        else {
            inv_size_change_type = SizeChangeType::increase;
            MyFFTRunTimeAssertTrue(IsAPowerOfTwo(dims_in.x) && IsAPowerOfTwo(dims_in.y) && IsAPowerOfTwo(dims_in.z), "Input dimensions must be a power of 2");
        }
    }

    else if ( dims_out.x < dims_in.x || dims_out.y < dims_in.y || dims_out.z < dims_in.z ) {
        // For now we must pad in all dimensions, this is not needed and should be lifted. FIXME
        MyFFTRunTimeAssertTrue(dims_out.x <= dims_in.x, "If padding, all dimensions must be <=, x out > x in");
        MyFFTRunTimeAssertTrue(dims_out.y <= dims_in.y, "If padding, all dimensions must be <=, y out > y in");
        MyFFTRunTimeAssertTrue(dims_out.z <= dims_in.z, "If padding, all dimensions must be <=, z out > z in");

        MyFFTRunTimeAssertTrue(IsAPowerOfTwo(dims_in.x) && IsAPowerOfTwo(dims_in.y) && IsAPowerOfTwo(dims_in.z), "Input dimensions must be a power of 2");
        MyFFTRunTimeAssertTrue(IsAPowerOfTwo(dims_out.x) && IsAPowerOfTwo(dims_out.y) && IsAPowerOfTwo(dims_out.z), "Output dimensions must be a power of 2");

        if ( is_fwd_not_inv )
            fwd_size_change_type = SizeChangeType::decrease;
        else
            inv_size_change_type = SizeChangeType::decrease;
    }
    else if ( dims_out.x == dims_in.x && dims_out.y == dims_in.y && dims_out.z == dims_in.z ) {
        MyFFTRunTimeAssertTrue(IsAPowerOfTwo(dims_in.x) && IsAPowerOfTwo(dims_in.y) && IsAPowerOfTwo(dims_in.z), "Input dimensions must be a power of 2");
        MyFFTRunTimeAssertTrue(IsAPowerOfTwo(dims_out.x) && IsAPowerOfTwo(dims_out.y) && IsAPowerOfTwo(dims_out.z), "Output dimensions must be a power of 2");

        if ( is_fwd_not_inv )
            fwd_size_change_type = SizeChangeType::no_change;
        else
            inv_size_change_type = SizeChangeType::no_change;
    }
    else {
        // TODO: if this is relaxed, the dimensionality check below will be invalid.
        MyFFTRunTimeAssertTrue(false, "Error in validating fwd plan: Currently all dimensions must either increase, decrease or stay the same.");
    }

    // Validate the FFT sizes the build supports
    std::array<bool, 6> valid_sizes = {false, false, false, false, false, false};
    // The signal length can be < the FFT size
    for ( auto& size : sizes_in_this_build ) {
        if ( size == dims_in.x || (dims_out.x > dims_in.x) )
            valid_sizes[0] = true;
        if ( size == dims_in.y || (dims_out.y > dims_in.y || dims_in.y == 1) )
            valid_sizes[1] = true;
        if ( size == dims_in.z || (dims_out.z > dims_in.z || dims_in.z == 1) )
            valid_sizes[2] = true;
        if ( size == dims_out.x )
            valid_sizes[3] = true;
        if ( size == dims_out.y )
            valid_sizes[4] = true;
        if ( size == dims_out.z || dims_out.z == 1 )
            valid_sizes[5] = true;
    }
    for ( int i = 0; i < 6; i++ ) {
        if ( is_fwd_not_inv ) {
            MyFFTRunTimeAssertTrue(valid_sizes[i], "Invalid size fwd");
        }
        else {
            MyFFTRunTimeAssertTrue(valid_sizes[i], "Invalid size inv");
        }
    }

    // We can only run this onces both input and output are set, right now we are setting after... FIXME
    if ( is_set_input_params || is_set_output_params ) {
        MyFFTRunTimeAssertTrue(fwd_dims_out.x == inv_dims_in.x &&
                                       fwd_dims_out.y == inv_dims_in.y &&
                                       fwd_dims_out.z == inv_dims_in.z,
                               "Error in validating the dimension: Currently all fwd out should match inv in.");

        if ( Rank == 2 ) {
            MyFFTRunTimeAssertTrue(fwd_dims_in.y == fwd_dims_in.x && fwd_dims_out.y == fwd_dims_out.x && inv_dims_in.y == inv_dims_in.x && inv_dims_out.y == inv_dims_out.x, "Only square images are supported currently (y dimension)");
        }
        else if ( Rank == 3 ) {
            // We need to enforce cubic dimensions currently
            MyFFTRunTimeAssertTrue(fwd_dims_in.y == fwd_dims_in.x && fwd_dims_out.y == fwd_dims_out.x && fwd_dims_in.z == fwd_dims_in.x && fwd_dims_out.z == fwd_dims_out.x &&
                                           inv_dims_in.y == inv_dims_in.x && inv_dims_out.y == inv_dims_out.x && inv_dims_in.z == inv_dims_in.x && inv_dims_out.z == inv_dims_out.x,
                                   "Only cubic images are supported currently (z dimension)");
            constexpr unsigned int max_3d_size = 512;
            MyFFTRunTimeAssertFalse(fwd_dims_in.z > max_3d_size ||
                                            fwd_dims_out.z > max_3d_size ||
                                            inv_dims_in.z > max_3d_size ||
                                            inv_dims_out.z > max_3d_size ||
                                            fwd_dims_in.y > max_3d_size ||
                                            fwd_dims_out.y > max_3d_size ||
                                            inv_dims_in.y > max_3d_size ||
                                            inv_dims_out.y > max_3d_size ||
                                            fwd_dims_in.x > max_3d_size ||
                                            fwd_dims_out.x > max_3d_size ||
                                            inv_dims_in.x > max_3d_size ||
                                            inv_dims_out.x > max_3d_size,
                                    "Error in validating the dimension: Currently all dimensions must be <= 512 for 3d transforms.");
        }
    }
}

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::SetDimensions(DimensionCheckType::Enum check_op_type) {

    switch ( check_op_type ) {
        case DimensionCheckType::CopyFromHost: {
            // MyFFTDebugAssertTrue(transform_stage_completed == none, "When copying from host, the transform stage should be none, something has gone wrong.");
            // FIXME: is this the right thing to do? Maybe this should be explicitly "reset" when the input image is "refereshed."
            memory_size_to_copy_ = input_memory_wanted_;
            break;
        }

        case DimensionCheckType::CopyToHost: {
            if ( transform_stage_completed == 0 ) {
                memory_size_to_copy_ = input_memory_wanted_;
            }
            else if ( transform_stage_completed < 5 ) {
                memory_size_to_copy_ = fwd_output_memory_wanted_;
            }
            else {
                memory_size_to_copy_ = inv_output_memory_wanted_;
            }
        } // switch transform_stage_completed
        break;

    } // end switch on operation type
}

////////////////////////////////////////////////////
/// Transform kernels
////////////////////////////////////////////////////

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::CopyAndSynchronize(bool               to_host,
                                                                                                      PositionSpaceType* input_pointer,
                                                                                                      int                n_elements_to_copy) {
    SetDimensions(DimensionCheckType::CopyToHost);
    int n_to_actually_copy = (n_elements_to_copy > 0) ? n_elements_to_copy : memory_size_to_copy_;

    cudaMemcpyKind copy_kind = (to_host) ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;

    MyFFTDebugAssertTrue(n_to_actually_copy > 0, "Error in CopyDeviceToHostAndSynchronize: n_elements_to_copy must be > 0");
    if ( to_host )
        MyFFTDebugAssertTrue(is_pointer_in_memory_and_registered(input_pointer), "Error in CopyDeviceToHostAndSynchronize: input_pointer must be in memory and registered");

    switch ( current_buffer ) {
        case fastfft_external_input: {
            MyFFTDebugAssertTrue(is_pointer_in_device_memory(d_ptr.external_input), "Error in CopyDeviceToHostAndSynchronize: input_pointer must be in device memory");
            cudaErr(cudaMemcpyAsync(input_pointer, d_ptr.external_input, n_to_actually_copy * sizeof(PositionSpaceType), copy_kind, cudaStreamPerThread));
            break;
        }
        case fastfft_external_output: {
            MyFFTDebugAssertTrue(is_pointer_in_device_memory(d_ptr.external_input), "Error in CopyDeviceToHostAndSynchronize: input_pointer must be in device memory");
            cudaErr(cudaMemcpyAsync(input_pointer, d_ptr.external_output, n_to_actually_copy * sizeof(PositionSpaceType), copy_kind, cudaStreamPerThread));
            break;
        }
        // If we are in the internal buffers, our data is ComputeBaseType
        case fastfft_internal_buffer_1: {
            MyFFTDebugAssertTrue(is_pointer_in_device_memory(d_ptr.buffer_1), "Error in CopyDeviceToHostAndSynchronize: input_pointer must be in device memory");
            if ( sizeof(ComputeBaseType) != sizeof(PositionSpaceType) )
                std::cerr << "\n\tWarning: CopyDeviceToHostAndSynchronize: sizeof(ComputeBaseType) != sizeof(PositionSpaceType) - this may be a problem\n\n";
            cudaErr(cudaMemcpyAsync(input_pointer, d_ptr.buffer_1, n_to_actually_copy * sizeof(ComputeBaseType), copy_kind, cudaStreamPerThread));
            break;
        }
        case fastfft_internal_buffer_2: {
            MyFFTDebugAssertTrue(is_pointer_in_device_memory(d_ptr.buffer_2), "Error in CopyDeviceToHostAndSynchronize: input_pointer must be in device memory");
            if ( sizeof(ComputeBaseType) != sizeof(PositionSpaceType) )
                std::cerr << "\n\tWarning: CopyDeviceToHostAndSynchronize: sizeof(ComputeBaseType) != sizeof(PositionSpaceType) - this may be a problem\n\n";
            cudaErr(cudaMemcpyAsync(input_pointer, d_ptr.buffer_2, n_to_actually_copy * sizeof(ComputeBaseType), copy_kind, cudaStreamPerThread));
            break;
        }
        default: {
            MyFFTDebugAssertTrue(false, "Error in CopyDeviceToHostAndSynchronize: current_buffer must be one of fastfft_external_input, fastfft_internal_buffer_1, fastfft_internal_buffer_2");
        }
    }

    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
};

template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_R2C_NONE_XY(const InputData_t* __restrict__ input_values,
                                          OutputData_t* __restrict__ output_values,
                                          Offsets                      mem_offsets,
                                          typename FFT::workspace_type workspace) {
    // Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    // The data store is non-coalesced, so don't aggregate the data in shared mem.
    FastFFT_SMEM complex_compute_t shared_mem[];

    // Memory used by FFT
    complex_compute_t thread_data[FFT::storage_size];

    // No need to __syncthreads as each thread only accesses its own shared mem anyway
    // multiply Q*fwd_dims_out.w because x maps to y in the output transposed FFT
    io<FFT>::load_r2c(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)], thread_data);
    // io<FFT>::load_r2c(&input_values[blockIdx.y*mem_offsets.physical_x_input], thread_data);

    // In the first FFT the modifying twiddle factor is 1 so the data are real
    FFT( ).execute(thread_data, shared_mem, workspace);

    io<FFT>::store_r2c_transposed_xy(thread_data,
                                     &output_values[ReturnZplane(gridDim.y, mem_offsets.physical_x_output)],
                                     gridDim.y);
}

// 2 ffts/block via threadIdx.x, notice launch bounds. Creates partial coalescing.

template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_R2C_NONE_XZ(const InputData_t* __restrict__ input_values,
                                          OutputData_t* __restrict__ output_values,
                                          Offsets                      mem_offsets,
                                          typename FFT::workspace_type workspace) {
    // Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    // The data store is non-coalesced, so don't aggregate the data in shared mem.
    FastFFT_SMEM complex_compute_t shared_mem[];

    // Memory used by FFT
    complex_compute_t thread_data[FFT::storage_size];

    io<FFT>::load_r2c(&input_values[Return1DFFTAddress_strided_Z(mem_offsets.physical_x_input)], thread_data);

    constexpr const unsigned int n_compute_elements = FFT::shared_memory_size / sizeof(complex_compute_t);
    FFT( ).execute(thread_data, &shared_mem[threadIdx.y * n_compute_elements], workspace);
    __syncthreads( ); // TODO: is this needed?

    // memory is at least large enough to hold the output with padding. synchronizing
    io<FFT>::transpose_r2c_in_shared_XZ(shared_mem, thread_data);

    // Transpose XZ, so the proper Z dimension now comes from X
    io<FFT>::store_r2c_transposed_xz_strided_Z(shared_mem, output_values);
}

template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_R2C_INCREASE_XY(const InputData_t* __restrict__ input_values,
                                              OutputData_t* __restrict__ output_values,
                                              Offsets                      mem_offsets,
                                              float                        twiddle_in,
                                              const int                    Q,
                                              const int                    SignalLength,
                                              typename FFT::workspace_type workspace) {
    // Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    // The data store is non-coalesced, so don't aggregate the data in shared mem.
    FastFFT_SMEM scalar_compute_t shared_input[];
    complex_compute_t*            shared_mem = (complex_compute_t*)&shared_input[mem_offsets.shared_input];

    // Memory used by FFT
    complex_compute_t twiddle;
    complex_compute_t thread_data[FFT::storage_size];

    // To re-map the thread index to the data ... these really could be short ints, but I don't know how that will perform. TODO benchmark
    // It is also questionable whether storing these vs, recalculating makes more sense.
    int   input_MAP[FFT::storage_size];
    int   output_MAP[FFT::storage_size];
    float twiddle_factor_args[FFT::storage_size];

    // No need to __syncthreads as each thread only accesses its own shared mem anyway
    // multiply Q*fwd_dims_out.w because x maps to y in the output transposed FFT
    io<FFT>::load_r2c_shared(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)],
                             shared_input,
                             thread_data,
                             twiddle_factor_args,
                             twiddle_in,
                             input_MAP,
                             output_MAP,
                             Q,
                             SignalLength);

    // We unroll the first and last loops.
    // In the first FFT the modifying twiddle factor is 1 so the data are real
    FFT( ).execute(thread_data, shared_mem, workspace);

    io<FFT>::store_r2c_transposed_xy(thread_data,
                                     &output_values[ReturnZplane(blockDim.y, mem_offsets.physical_x_output)],
                                     output_MAP,
                                     gridDim.y);

    // For the other fragments we need the initial twiddle
    for ( int sub_fft = 1; sub_fft < Q - 1; sub_fft++ ) {
        io<FFT>::copy_from_shared(shared_input, thread_data, input_MAP);
        for ( int i = 0; i < FFT::elements_per_thread; i++ ) {
            // Pre shift with twiddle
            SINCOS(twiddle_factor_args[i] * sub_fft, &twiddle.y, &twiddle.x);
            thread_data[i] *= twiddle;
            // increment the output mapping.
            output_MAP[i]++;
        }
        FFT( ).execute(thread_data, shared_mem, workspace);

        io<FFT>::store_r2c_transposed_xy(thread_data,
                                         &output_values[ReturnZplane(blockDim.y, mem_offsets.physical_x_output)],
                                         output_MAP,
                                         gridDim.y);
    }

    // For the last fragment we need to also do a bounds check.
    io<FFT>::copy_from_shared(shared_input, thread_data, input_MAP);
    for ( int i = 0; i < FFT::elements_per_thread; i++ ) {
        // Pre shift with twiddle
        SINCOS(twiddle_factor_args[i] * (Q - 1), &twiddle.y, &twiddle.x);
        thread_data[i] *= twiddle;
        // increment the output mapping.
        output_MAP[i]++;
    }

    FFT( ).execute(thread_data, shared_mem, workspace);

    io<FFT>::store_r2c_transposed_xy(thread_data,
                                     &output_values[ReturnZplane(blockDim.y, mem_offsets.physical_x_output)],
                                     output_MAP,
                                     gridDim.y);
}

template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_R2C_INCREASE_XZ(const InputData_t* __restrict__ input_values,
                                              OutputData_t* __restrict__ output_values,
                                              Offsets                      mem_offsets,
                                              float                        twiddle_in,
                                              int                          Q,
                                              typename FFT::workspace_type workspace) {

    // Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    // The data store is non-coalesced, so don't aggregate the data in shared mem.
    FastFFT_SMEM scalar_compute_t shared_input[];
    complex_compute_t*            shared_mem = (complex_compute_t*)&shared_input[XZ_STRIDE * mem_offsets.shared_input];

    // Memory used by FFT
    complex_compute_t twiddle;
    complex_compute_t thread_data[FFT::storage_size];

    float twiddle_factor_args[FFT::storage_size];
    // Note: Q is used to calculate the strided output, which in this use, will end up being an offest in Z, so
    // we multiply by the NXY physical mem size of the OUTPUT array (which will be ZY') Then in the sub_fft loop, instead of adding one
    // we add NXY
    io<FFT>::load_r2c_shared(&input_values[Return1DFFTAddress_strided_Z(mem_offsets.physical_x_input)],
                             &shared_input[threadIdx.y * mem_offsets.shared_input],
                             thread_data,
                             twiddle_factor_args,
                             twiddle_in);

    FFT( ).execute(thread_data, &shared_mem[threadIdx.y * FFT::shared_memory_size / sizeof(complex_compute_t)], workspace);
    __syncthreads( );
    // Now we have a partial strided output due to the transform decomposition. In the 2D case we either write it out, or coalsece it in to shared memory
    // until we have the full output. Here, we are working on a tile, so we can transpose the data, and write it out partially coalesced.

    io<FFT>::transpose_r2c_in_shared_XZ(shared_mem, thread_data);
    io<FFT>::store_r2c_transposed_xz_strided_Z(shared_mem, output_values, Q, 0);

    // Now we need to loop over the remaining fragments.
    // For the other fragments we need the initial twiddle
    for ( int sub_fft = 1; sub_fft < Q; sub_fft++ ) {
        io<FFT>::copy_from_shared(&shared_input[threadIdx.y * mem_offsets.shared_input], thread_data);
        for ( int i = 0; i < FFT::elements_per_thread; i++ ) {
            // Pre shift with twiddle
            SINCOS(twiddle_factor_args[i] * sub_fft, &twiddle.y, &twiddle.x);
            thread_data[i] *= twiddle;
            // increment the output mapping.
        }

        FFT( ).execute(thread_data, &shared_mem[threadIdx.y * FFT::shared_memory_size / sizeof(complex_compute_t)], workspace); // FIXME the workspace is probably not going to work with the batched, look at the examples to see what to do.
        __syncthreads( );
        io<FFT>::transpose_r2c_in_shared_XZ(shared_mem, thread_data);
        io<FFT>::store_r2c_transposed_xz_strided_Z(shared_mem, output_values, Q, sub_fft);
    }

    // // For the last fragment we need to also do a bounds check. FIXME where does this happen
    // io<FFT>::copy_from_shared(&shared_input[threadIdx.y * mem_offsets.shared_input], thread_data);
    // for (int i = 0; i < FFT::elements_per_thread; i++) {
    //     // Pre shift with twiddle
    //     SINCOS(twiddle_factor_args[i]*(Q-1),&twiddle.y,&twiddle.x);
    //     thread_data[i] *= twiddle;
    //     // increment the output mapping.
    // }

    // FFT().execute(thread_data, &shared_mem[threadIdx.y * FFT::shared_memory_size/sizeof(complex_compute_t)], workspace); // FIXME the workspace is not setup for tiled approach
    // __syncthreads();
    // io<FFT>::transpose_r2c_in_shared_XZ(shared_mem, thread_data);
    // io<FFT>::store_r2c_transposed_xz_strided_Z(shared_mem, output_values, Q, 0);
}

// __launch_bounds__(FFT::max_threads_per_block)  we don't know this because it is threadDim.x * threadDim.z - this could be templated if it affects performance significantly
template <class FFT, class InputData_t, class OutputData_t>
__global__ void block_fft_kernel_R2C_DECREASE_XY(const InputData_t* __restrict__ input_values,
                                                 OutputData_t* __restrict__ output_values,
                                                 Offsets                      mem_offsets,
                                                 float                        twiddle_in,
                                                 int                          Q,
                                                 typename FFT::workspace_type workspace) {
    // Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    // The shared memory is used for storage, shuffling and fft ops at different stages and includes room for bank padding.
    FastFFT_SMEM complex_compute_t shared_mem[];

    complex_compute_t thread_data[FFT::storage_size];

    // Load in natural order
    io<FFT>::load_r2c_shared_and_pad(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)], shared_mem);

    // DIT shuffle, bank conflict free
    io<FFT>::copy_from_shared(shared_mem, thread_data, Q);

    // The FFT operator has no idea we are using threadIdx.y to get multiple sub transforms, so we need to
    // segment the shared memory it accesses to avoid conflicts.
    constexpr const unsigned int fft_shared_mem_num_elements = FFT::shared_memory_size / sizeof(complex_compute_t);
    FFT( ).execute(thread_data, &shared_mem[fft_shared_mem_num_elements * threadIdx.y], workspace);
    __syncthreads( );

    // Full twiddle multiply and store in natural order in shared memory
    io<FFT>::reduce_block_fft(thread_data, shared_mem, twiddle_in, Q);

    // Reduce from shared memory into registers, ending up with only P valid outputs.
    io<FFT>::store_r2c_reduced(thread_data, &output_values[mem_offsets.physical_x_output * threadIdx.y], gridDim.y, mem_offsets.physical_x_output);
}

template <class ExternalImage_t, class FFT, class invFFT, class ComplexData_t>
__global__ void block_fft_kernel_C2C_FWD_NONE_INV_DECREASE_ConjMul(const ExternalImage_t* __restrict__ image_to_search,
                                                                   const ComplexData_t* __restrict__ input_values,
                                                                   ComplexData_t* __restrict__ output_values,
                                                                   Offsets                         mem_offsets,
                                                                   float                           twiddle_in,
                                                                   int                             apparent_Q,
                                                                   typename FFT::workspace_type    workspace_fwd,
                                                                   typename invFFT::workspace_type workspace_inv) {

    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    FastFFT_SMEM complex_compute_t shared_mem[];

    complex_compute_t thread_data[FFT::storage_size];

    // Load in natural order
    io<FFT>::load(&input_values[Return1DFFTAddress(size_of<FFT>::value)], thread_data);

    // io<FFT>::load_c2c_shared_and_pad(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)], shared_mem);

    // // DIT shuffle, bank conflict free
    // io<FFT>::copy_from_shared(shared_mem, thread_data, Q);

    // constexpr const unsigned int fft_shared_mem_num_elements = FFT::shared_memory_size / sizeof(complex_compute_t);
    // FFT().execute(thread_data, &shared_mem[fft_shared_mem_num_elements * threadIdx.y], workspace_fwd);
    // __syncthreads();
    FFT( ).execute(thread_data, shared_mem, workspace_fwd);

    // // Full twiddle multiply and store in natural order in shared memory
    // io<FFT>::reduce_block_fft(thread_data, shared_mem, twiddle_in, Q);

#if FFT_DEBUG_STAGE > 3
    // Load in imageFFT to search
    io<invFFT>::load_shared_and_conj_multiply(&image_to_search[Return1DFFTAddress(size_of<FFT>::value)], thread_data);
#endif

#if FFT_DEBUG_STAGE > 4
    // Run the inverse FFT
    // invFFT().execute(thread_data, &shared_mem[fft_shared_mem_num_elements * threadIdx.y], workspace_inv);
    invFFT( ).execute(thread_data, shared_mem, workspace_inv);

#endif

// // The reduced store considers threadIdx.y to ignore extra threads
// io<invFFT>::store_c2c_reduced(thread_data, &output_values[blockIdx.y * gridDim.y]);
#if FFT_DEBUG_STAGE < 5
    // There is no size reduction for this debug stage, so we need to use the pixel_pitch of the input array.
    io<invFFT>::store(thread_data, &output_values[Return1DFFTAddress(size_of<FFT>::value)]);
#else
    // In the current simplified version of the kernel, I am not using any transform decomposition (this is because of the difficulties with resrved threadIdx.x/y in the cufftdx lib)
    // So the full thing is calculated and only truncated on output.
    io<invFFT>::store(thread_data,
                      &output_values[Return1DFFTAddress(size_of<FFT>::value / apparent_Q)],
                      size_of<FFT>::value / apparent_Q);
#endif
}

// C2C

template <class FFT, class ComplexData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_NONE(const ComplexData_t* __restrict__ input_values,
                                       ComplexData_t* __restrict__ output_values,
                                       Offsets                      mem_offsets,
                                       typename FFT::workspace_type workspace) {
    // Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    FastFFT_SMEM complex_compute_t shared_mem[]; // Storage for the input data that is re-used each blcok

    // Memory used by FFT
    complex_compute_t thread_data[FFT::storage_size];

    // No need to __syncthreads as each thread only accesses its own shared mem anyway
    io<FFT>::load(&input_values[Return1DFFTAddress(size_of<FFT>::value)], thread_data);

    // Since the memory ops are super straightforward this is an okay compromise.
    FFT( ).execute(thread_data, shared_mem, workspace);

    io<FFT>::store<size_of<FFT>::value>(thread_data,
                                        &output_values[Return1DFFTAddress(size_of<FFT>::value)]);
}

template <class FFT, class ComplexData_t>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_NONE_XZ(const ComplexData_t* __restrict__ input_values,
                                          ComplexData_t* __restrict__ output_values,
                                          Offsets                      mem_offsets,
                                          typename FFT::workspace_type workspace) {

    //	// Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    FastFFT_SMEM complex_compute_t shared_mem[]; // Storage for the input data that is re-used each blcok

    // Memory used by FFT
    complex_compute_t thread_data[FFT::storage_size];

    // No need to __syncthreads as each thread only accesses its own shared mem anyway
    io<FFT>::load(&input_values[Return1DFFTAddress_strided_Z(size_of<FFT>::value)], thread_data);

    FFT( ).execute(thread_data, &shared_mem[threadIdx.y * FFT::shared_memory_size / sizeof(complex_compute_t)], workspace);
    __syncthreads( );

    // Now we need to transpose in shared mem, fix bank conflicts later. TODO
    {
        const unsigned int stride = io<FFT>::stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // return (XZ_STRIDE*blockIdx.z + threadIdx.y) + (XZ_STRIDE*gridDim.z) * ( blockIdx.y + X * gridDim.y );
            // XZ_STRIDE == blockDim.z
            shared_mem[threadIdx.y + index * XZ_STRIDE] = thread_data[i];
            index += stride;
        }
    }
    __syncthreads( );

    // Transpose XZ, so the proper Z dimension now comes from X
    io<FFT>::store_transposed_xz_strided_Z(shared_mem, output_values);
}

// __launch_bounds__(FFT::max_threads_per_block)  we don't know this because it is threadDim.x * threadDim.z - this could be templated if it affects performance significantly
template <class FFT, class ComplexData_t>
__global__ void block_fft_kernel_C2C_DECREASE(const ComplexData_t* __restrict__ input_values,
                                              ComplexData_t* __restrict__ output_values,
                                              Offsets                      mem_offsets,
                                              float                        twiddle_in,
                                              int                          Q,
                                              typename FFT::workspace_type workspace) {
    //	// Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    FastFFT_SMEM complex_compute_t shared_mem[];

    complex_compute_t thread_data[FFT::storage_size];

    // Load in natural order
    io<FFT>::load_c2c_shared_and_pad(&input_values[Return1DFFTAddress(size_of<FFT>::value * Q)], shared_mem);

    // DIT shuffle, bank conflict free
    io<FFT>::copy_from_shared(shared_mem, thread_data, Q);

    constexpr const unsigned int fft_shared_mem_num_elements = FFT::shared_memory_size / sizeof(complex_compute_t);
    FFT( ).execute(thread_data, &shared_mem[fft_shared_mem_num_elements * threadIdx.y], workspace);
    __syncthreads( );

    // Full twiddle multiply and store in natural order in shared memory
    io<FFT>::reduce_block_fft(thread_data, shared_mem, twiddle_in, Q);

    // Reduce from shared memory into registers, ending up with only P valid outputs.
    io<FFT>::store_c2c_reduced(thread_data, &output_values[Return1DFFTAddress(size_of<FFT>::value)]);
}

// __launch_bounds__(FFT::max_threads_per_block)  we don't know this because it is threadDim.x * threadDim.z - this could be templated if it affects performance significantly
template <class FFT, class ComplexData_t>
__global__ void block_fft_kernel_C2C_INCREASE(const ComplexData_t* __restrict__ input_values,
                                              ComplexData_t* __restrict__ output_values,
                                              Offsets                      mem_offsets,
                                              float                        twiddle_in,
                                              int                          Q,
                                              const unsigned int           SignalLength,
                                              typename FFT::workspace_type workspace) {
    // Initialize the shared memory, assuming everyting matches the input data X size in

    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    FastFFT_SMEM complex_compute_t shared_input_complex[]; // Storage for the input data that is re-used each blcok
    // storage for the coalesced output data. This may grow too large,
    // FIXME: could be signal length, but that that also might interfere with coalescing
    complex_compute_t* shared_output = (complex_compute_t*)&shared_input_complex[mem_offsets.shared_input];
    complex_compute_t* shared_mem    = (complex_compute_t*)&shared_output[mem_offsets.shared_output];

    // Memory used by FFT
    complex_compute_t thread_data[FFT::storage_size];
    float             twiddle_factor_args[FFT::storage_size];
    complex_compute_t twiddle;

    // No need to __syncthreads as each thread only accesses its own shared mem anyway
    io<FFT>::load_shared(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)],
                         shared_input_complex,
                         thread_data,
                         twiddle_factor_args,
                         twiddle_in,
                         SignalLength); // FIXME: currently zero padding to size_of<FFT>::value on load, but could use less shared at SignalLenght, if so, would need to modify copy_from_shared.

    FFT( ).execute(thread_data, shared_mem, workspace);
    io<FFT>::store(thread_data, shared_output, Q, 0);

    // For the other fragments we need the initial twiddle
    for ( int sub_fft = 1; sub_fft < Q; sub_fft++ ) {
        io<FFT>::copy_from_shared(shared_input_complex, thread_data);

        for ( int i = 0; i < FFT::elements_per_thread; i++ ) {
            // Pre shift with twiddle
            SINCOS(twiddle_factor_args[i] * sub_fft, &twiddle.y, &twiddle.x);
            thread_data[i] *= twiddle;
        }
        FFT( ).execute(thread_data, shared_mem, workspace);
        io<FFT>::store(thread_data, shared_output, Q, sub_fft);
    }
    __syncthreads( );

    // Now that the memory output can be coalesced send to global
    // FIXME: is this actually coalced?
    for ( int sub_fft = 0; sub_fft < Q; sub_fft++ ) {
        io<FFT>::store_coalesced(shared_output,
                                 &output_values[Return1DFFTAddress(mem_offsets.physical_x_output)],
                                 sub_fft * mem_offsets.shared_input); // FIXME: if we shrink shared_input == SignalLength then this should be size_of<FFT>::value
    }
}

template <class FFT, class ComplexData_t>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_NONE_XYZ(const ComplexData_t* __restrict__ input_values,
                                           ComplexData_t* __restrict__ output_values,
                                           Offsets                      mem_offsets,
                                           typename FFT::workspace_type workspace) {

    //	// Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    FastFFT_SMEM complex_compute_t shared_mem[]; // Storage for the input data that is re-used each blcok

    // Memory used by FFT
    complex_compute_t thread_data[FFT::storage_size];

    // No need to __syncthreads as each thread only accesses its own shared mem anyway
    io<FFT>::load(&input_values[Return1DFFTColumn_XYZ_transpose(size_of<FFT>::value)], thread_data);

    FFT( ).execute(thread_data, &shared_mem[threadIdx.y * FFT::shared_memory_size / sizeof(complex_compute_t)], workspace);
    __syncthreads( );

    io<FFT>::transpose_in_shared_XZ(shared_mem, thread_data);

    io<FFT>::store_Z(shared_mem, output_values);
}

template <class FFT, class ComplexData_t>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_INCREASE_XYZ(const ComplexData_t* __restrict__ input_values,
                                               ComplexData_t* __restrict__ output_values,
                                               Offsets                      mem_offsets,
                                               float                        twiddle_in,
                                               int                          Q,
                                               typename FFT::workspace_type workspace) {

    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    FastFFT_SMEM complex_compute_t shared_input_complex[]; // Storage for the input data that is re-used each blcok
    complex_compute_t*             shared_mem = (complex_compute_t*)&shared_input_complex[XZ_STRIDE * mem_offsets.shared_input]; // storage for computation and transposition (alternating)

    // Memory used by FFT
    complex_compute_t thread_data[FFT::storage_size];
    complex_compute_t twiddle;
    float             twiddle_factor_args[FFT::storage_size];

    // No need to __syncthreads as each thread only accesses its own shared mem anyway
    io<FFT>::load_shared(&input_values[Return1DFFTColumn_XYZ_transpose(size_of<FFT>::value)],
                         &shared_input_complex[threadIdx.y * mem_offsets.shared_input],
                         thread_data,
                         twiddle_factor_args,
                         twiddle_in);

    FFT( ).execute(thread_data, &shared_mem[threadIdx.y * FFT::shared_memory_size / sizeof(complex_compute_t)], workspace);
    __syncthreads( );

    io<FFT>::transpose_in_shared_XZ(shared_mem, thread_data);
    io<FFT>::store_Z(shared_mem, output_values, Q, 0);

    // For the other fragments we need the initial twiddle
    for ( int sub_fft = 1; sub_fft < Q; sub_fft++ ) {
        io<FFT>::copy_from_shared(&shared_input_complex[threadIdx.y * mem_offsets.shared_input], thread_data);
        for ( int i = 0; i < FFT::elements_per_thread; i++ ) {
            // Pre shift with twiddle
            SINCOS(twiddle_factor_args[i] * sub_fft, &twiddle.y, &twiddle.x);
            thread_data[i] *= twiddle;
        }
        FFT( ).execute(thread_data, &shared_mem[threadIdx.y * FFT::shared_memory_size / sizeof(complex_compute_t)], workspace);
        io<FFT>::transpose_in_shared_XZ(shared_mem, thread_data);
        io<FFT>::store_Z(shared_mem, output_values, Q, sub_fft);
    }
}

template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2R_NONE(const InputData_t* __restrict__ input_values,
                                       OutputData_t* __restrict__ output_values,
                                       Offsets                      mem_offsets,
                                       typename FFT::workspace_type workspace) {

    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    FastFFT_SMEM complex_compute_t shared_mem[];

    complex_compute_t thread_data[FFT::storage_size];

    io<FFT>::load_c2r(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)], thread_data);

    // For loop zero the twiddles don't need to be computed
    FFT( ).execute(thread_data, shared_mem, workspace);

    io<FFT>::store_c2r(thread_data, &output_values[Return1DFFTAddress(mem_offsets.physical_x_output)]);
}

template <class FFT, unsigned int MAX_TPB, class InputData_t, class OutputData_t, unsigned int n_ffts>
__launch_bounds__(MAX_TPB) __global__
        void block_fft_kernel_C2R_NONE_XY(const InputData_t* __restrict__ input_values,
                                          OutputData_t* __restrict__ output_values,
                                          Offsets                      mem_offsets,
                                          typename FFT::workspace_type workspace) {

    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    FastFFT_SMEM complex_compute_t shared_mem[];

    complex_compute_t thread_data[FFT::storage_size];

    // Total concurent FFTs, n-1 in shared and then on in thread data
    if constexpr ( n_ffts > 1 && sizeof(scalar_compute_t) == 4 && size_of<FFT>::value > 16 ) {
#ifdef C2R_BUFFER_LINES

        io<FFT, MAX_TPB, n_ffts>::load_c2r_transposed_coalesced(&input_values[ReturnZplane(gridDim.y * n_ffts, mem_offsets.physical_x_input)],
                                                                thread_data,
                                                                shared_mem,
                                                                gridDim.y * n_ffts);

        FFT( ).execute(thread_data, &shared_mem[FFT::shared_memory_size / sizeof(complex_compute_t) * threadIdx.y], workspace);
        // To cut down on total smem needed split these
        // FIXME: I don't think there is any way to get this to work, b/c the FFT().exectute method inserts barrier.sync ptx, which afaik behave the same as __syncthreads
        // const unsigned int eve_odd = threadIdx.y % 2;
        // if ( eve_odd == 0 ) {
        //     FFT( ).execute(thread_data, &shared_mem[FFT::shared_memory_size / sizeof(complex_compute_t) * (threadIdx.y / 2)], workspace);
        // }
        // __syncthreads( );
        // if ( eve_odd == 1 ) {
        //     FFT( ).execute(thread_data, &shared_mem[FFT::shared_memory_size / sizeof(complex_compute_t) * (threadIdx.y / 2)], workspace);
        // }
        // __syncthreads( );

#else
        static_assert(n_ffts == 1, "C2R_BUFFER_LINES must be enabled, should not get hereonly for n_ffts == 1");
#endif
    }
    else {

        io<FFT>::load_c2r_transposed(&input_values[ReturnZplane(gridDim.y, mem_offsets.physical_x_input)],
                                     thread_data,
                                     gridDim.y);
        // // For loop zero the twiddles don't need to be computed
        FFT( ).execute(thread_data, shared_mem, workspace);
    }

#ifdef C2R_BUFFER_LINES
    // normally  Return1DFFTAddress(mem_offsets.physical_x_output) = pixel_pitch * (blockIdx.y + blockIdx.z * gridDim.y)
    // we reduce the number of blocks in Y by n_ffts, hysical_x = fft_idx + blockIdx.y * n_coalesced_ffts;
    io<FFT>::store_c2r(thread_data,
                       &output_values[mem_offsets.physical_x_output * (blockIdx.y * n_ffts + threadIdx.y)]); // FIXME only 2d

#else
    io<FFT>::store_c2r(thread_data, &output_values[Return1DFFTAddress(mem_offsets.physical_x_output)]);
#endif
}

template <class FFT, class InputData_t, class OutputData_t>
__global__ void block_fft_kernel_C2R_DECREASE_XY(const InputData_t* __restrict__ input_values,
                                                 OutputData_t* __restrict__ output_values,
                                                 Offsets                      mem_offsets,
                                                 float                        twiddle_in,
                                                 int                          Q,
                                                 typename FFT::workspace_type workspace) {

    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    FastFFT_SMEM complex_compute_t shared_mem[];

    complex_compute_t thread_data[FFT::storage_size];

    io<FFT>::load_c2r_transposed(&input_values[ReturnZplane(gridDim.y, mem_offsets.physical_x_input)], thread_data, gridDim.y);

    // For loop zero the twiddles don't need to be computed
    FFT( ).execute(thread_data, shared_mem, workspace);

    io<FFT>::store_c2r(thread_data, &output_values[Return1DFFTAddress(mem_offsets.physical_x_output)]);
}

// FIXME assumed FWD
template <class PositionSpaceType, class OtherImageType>
__global__ void clip_into_top_left_kernel(PositionSpaceType* input_values,
                                          PositionSpaceType* output_values,
                                          const short4       dims) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if ( x > dims.w )
        return; // Out of bounds.

    // dims.w is the pitch of the output array
    if ( blockIdx.y > dims.y ) {
        output_values[blockIdx.y * dims.w + x] = OtherImageType(0);
        return;
    }

    if ( threadIdx.x > dims.x ) {
        output_values[blockIdx.y * dims.w + x] = OtherImageType(0);
        return;
    }
    else {
        // dims.z is the pitch of the output array
        output_values[blockIdx.y * dims.w + x] = input_values[blockIdx.y * dims.z + x];
        return;
    }
}

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::ClipIntoTopLeft(PositionSpaceType* input_ptr) {
    // TODO add some checks and logic.

    // Assuming we are calling this from R2C_Transposed and that the launch bounds are not set.
    dim3 local_threadsPerBlock = dim3(512, 1, 1);
    dim3 local_gridDims        = dim3((fwd_dims_out.x + local_threadsPerBlock.x - 1) / local_threadsPerBlock.x, 1, 1);

    const short4 area_to_clip_from = make_short4(fwd_dims_in.x, fwd_dims_in.y, fwd_dims_in.w * 2, fwd_dims_out.w * 2);

    precheck;
    clip_into_top_left_kernel<PositionSpaceType, PositionSpaceType><<<local_gridDims, local_threadsPerBlock, 0, cudaStreamPerThread>>>(input_ptr, (PositionSpaceType*)d_ptr.buffer_1, area_to_clip_from);
    postcheck;
    current_buffer = fastfft_internal_buffer_1;
}

// Modified from GpuImage::ClipIntoRealKernel
template <typename PositionSpaceType, typename OtherImageType>
__global__ void clip_into_real_kernel(PositionSpaceType* real_values_gpu,
                                      PositionSpaceType* other_image_real_values_gpu,
                                      short4             dims,
                                      short4             other_dims,
                                      int3               wanted_coordinate_of_box_center,
                                      PositionSpaceType  wanted_padding_value) {
    int3 other_coord = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                                 blockIdx.y * gridDim.y + threadIdx.y,
                                 blockIdx.z);

    int3 coord = make_int3(0, 0, 0);

    if ( other_coord.x < other_dims.x &&
         other_coord.y < other_dims.y &&
         other_coord.z < other_dims.z ) {

        coord.z = dims.z / 2 + wanted_coordinate_of_box_center.z +
                  other_coord.z - other_dims.z / 2;

        coord.y = dims.y / 2 + wanted_coordinate_of_box_center.y +
                  other_coord.y - other_dims.y / 2;

        coord.x = dims.x + wanted_coordinate_of_box_center.x +
                  other_coord.x - other_dims.x;

        if ( coord.z < 0 || coord.z >= dims.z ||
             coord.y < 0 || coord.y >= dims.y ||
             coord.x < 0 || coord.x >= dims.x ) {
            other_image_real_values_gpu[d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims)] = wanted_padding_value;
        }
        else {
            other_image_real_values_gpu[d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims)] =
                    real_values_gpu[d_ReturnReal1DAddressFromPhysicalCoord(coord, dims)];
        }

    } // end of bounds check
}

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::ClipIntoReal(PositionSpaceType* input_ptr,
                                                                                                int                wanted_coordinate_of_box_center_x,
                                                                                                int                wanted_coordinate_of_box_center_y,
                                                                                                int                wanted_coordinate_of_box_center_z) {
    // TODO add some checks and logic.

    // Assuming we are calling this from R2C_Transposed and that the launch bounds are not set.
    dim3 threadsPerBlock;
    dim3 gridDims;
    int3 wanted_center = make_int3(wanted_coordinate_of_box_center_x, wanted_coordinate_of_box_center_y, wanted_coordinate_of_box_center_z);
    threadsPerBlock    = dim3(32, 32, 1);
    gridDims           = dim3((fwd_dims_out.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                              (fwd_dims_out.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
                              1);

    const short4 area_to_clip_from    = make_short4(fwd_dims_in.x, fwd_dims_in.y, fwd_dims_in.w * 2, fwd_dims_out.w * 2);
    float        wanted_padding_value = 0.f;

    precheck;
    clip_into_real_kernel<PositionSpaceType, PositionSpaceType><<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(input_ptr, (PositionSpaceType*)d_ptr.buffer_1, fwd_dims_in, fwd_dims_out, wanted_center, wanted_padding_value);
    postcheck;
    current_buffer = fastfft_internal_buffer_1;
}

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
template <int FFT_ALGO_t, class PreOpType, class IntraOpType, class PostOpType>
EnableIf<IfAppliesIntraOpFunctor_HasIntraOpFunctor<IntraOpType, FFT_ALGO_t>>
FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::SetPrecisionAndExectutionMethod(OtherImageType* other_image_ptr,
                                                                                                              KernelType      kernel_type,
                                                                                                              PreOpType       pre_op_functor,
                                                                                                              IntraOpType     intra_op_functor,
                                                                                                              PostOpType      post_op_functor) {
    // For kernels with fwd and inv transforms, we want to not set the direction yet.

    static const bool is_half  = std::is_same_v<ComputeBaseType, __half>; // FIXME: This should be done in the constructor
    static const bool is_float = std::is_same_v<ComputeBaseType, float>;
    static_assert(is_half || is_float, "FourierTransformer::SetPrecisionAndExectutionMethod: Unsupported ComputeBaseType");
    if constexpr ( FFT_ALGO_t == Generic_Fwd_Image_Inv_FFT ) {
        static_assert(IS_IKF_t<IntraOpType>( ), "FourierTransformer::SetPrecisionAndExectutionMethod: Unsupported IntraOpType");
    }

    using FFT = decltype(Block( ) + Precision<ComputeBaseType>( ) + FFTsPerBlock<1>( ));
    SetIntraKernelFunctions<FFT_ALGO_t, FFT, PreOpType, IntraOpType, PostOpType>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
}

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
template <int FFT_ALGO_t, class FFT_base, class PreOpType, class IntraOpType, class PostOpType>
void FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::SetIntraKernelFunctions(OtherImageType* other_image_ptr,
                                                                                                           KernelType      kernel_type,
                                                                                                           PreOpType       pre_op_functor,
                                                                                                           IntraOpType     intra_op_functor,
                                                                                                           PostOpType      post_op_functor) {

    SelectSizeAndType<FFT_ALGO_t, FFT_base, PreOpType, IntraOpType, PostOpType, FastFFT_build_sizes>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
}

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
template <int FFT_ALGO_t, class FFT_base, class PreOpType, class IntraOpType, class PostOpType, unsigned int... SizeValues>
void FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::SelectSizeAndType(OtherImageType* other_image_ptr,
                                                                                                     KernelType      kernel_type,
                                                                                                     PreOpType       pre_op_functor,
                                                                                                     IntraOpType     intra_op_functor,
                                                                                                     PostOpType      post_op_functor) {

    (SelectSizeAndTypeWithFold<FFT_ALGO_t, FFT_base, PreOpType, IntraOpType, PostOpType, SizeValues>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor), ...);
}

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
template <int FFT_ALGO_t, class FFT_base, class PreOpType, class IntraOpType, class PostOpType, unsigned int SizeValue>
void FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::SelectSizeAndTypeWithFold(OtherImageType* other_image_ptr,
                                                                                                             KernelType      kernel_type,
                                                                                                             PreOpType       pre_op_functor,
                                                                                                             IntraOpType     intra_op_functor,
                                                                                                             PostOpType      post_op_functor) {

    // Use recursion to step through the allowed sizes.
    GetTransformSize(kernel_type);

    constexpr unsigned int Ept = SizeValue < 16 ? 4 : SizeValue < 4096 ? 8
                                                                       : 16;
    // Note: the size of the input/output may not match the size of the transform, i.e. transform_size.L <= transform_size.P
    if ( SizeValue == transform_size.P ) {
        switch ( device_properties.device_arch ) {
            case 700: {
                if constexpr ( SizeValue <= 4096 ) {
                    using FFT = decltype(FFT_base( ) + Size<SizeValue>( ) + SM<700>( ) + ElementsPerThread<Ept>( ));
                    SetAndLaunchKernel<FFT_ALGO_t, FFT, PreOpType, IntraOpType, PostOpType>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
                }
                break;
            }
            case 750: {

                if constexpr ( SizeValue <= 4096 ) {
                    using FFT = decltype(FFT_base( ) + Size<SizeValue>( ) + SM<750>( ) + ElementsPerThread<Ept>( ));
                    SetAndLaunchKernel<FFT_ALGO_t, FFT, PreOpType, IntraOpType, PostOpType>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
                }
                break;
            }
            case 800: {

                using FFT = decltype(FFT_base( ) + Size<SizeValue>( ) + SM<800>( ) + ElementsPerThread<Ept>( ));
                SetAndLaunchKernel<FFT_ALGO_t, FFT, PreOpType, IntraOpType, PostOpType>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
                break;
            }
            case 860: {

                // TODO: confirm that this is needed (over 860) which at the time just redirects to 700
                //       if maintining this, we could save some time on compilation by combining with the 700 case
                using FFT = decltype(FFT_base( ) + Size<SizeValue>( ) + SM<700>( ) + ElementsPerThread<Ept>( ));
                SetAndLaunchKernel<FFT_ALGO_t, FFT, PreOpType, IntraOpType, PostOpType>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
                break;
            }
            case 890: {

                // TODO: confirm that this is needed (over 860) which at the time just redirects to 700
                //       if maintining this, we could save some time on compilation by combining with the 700 case
                // FIXME: on migrating to cufftDx 1.1.1

                using FFT = decltype(FFT_base( ) + Size<SizeValue>( ) + SM<700>( ) + ElementsPerThread<Ept>( ));
                SetAndLaunchKernel<FFT_ALGO_t, FFT, PreOpType, IntraOpType, PostOpType>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
                break;
            }
            case 900: {

                using FFT = decltype(FFT_base( ) + Size<SizeValue>( ) + SM<900>( ) + ElementsPerThread<Ept>( ));
                SetAndLaunchKernel<FFT_ALGO_t, FFT, PreOpType, IntraOpType, PostOpType>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
                break;
            }
            case 1000: {

                // Blackwell data center (sm_100); 10.3 and 11.0 are normalized to 1000 in GetCudaDeviceProps
                using FFT = decltype(FFT_base( ) + Size<SizeValue>( ) + SM<1000>( ) + ElementsPerThread<Ept>( ));
                SetAndLaunchKernel<FFT_ALGO_t, FFT, PreOpType, IntraOpType, PostOpType>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
                break;
            }
            case 1200: {

                // Blackwell GeForce/RTX (sm_120, e.g. RTX 5090); 12.1 is normalized to 1200 in GetCudaDeviceProps
                using FFT = decltype(FFT_base( ) + Size<SizeValue>( ) + SM<1200>( ) + ElementsPerThread<Ept>( ));
                SetAndLaunchKernel<FFT_ALGO_t, FFT, PreOpType, IntraOpType, PostOpType>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
                break;
            }
            default: {
                MyFFTRunTimeAssertTrue(false, "Unsupported architecture" + std::to_string(device_properties.device_arch));
                break;
            }
        }
    }
}

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
template <int FFT_ALGO_t, class FFT_base_arch, class PreOpType, class IntraOpType, class PostOpType>
void FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::SetAndLaunchKernel(OtherImageType* other_image_ptr,
                                                                                                      KernelType      kernel_type,
                                                                                                      PreOpType       pre_op_functor,
                                                                                                      IntraOpType     intra_op_functor,
                                                                                                      PostOpType      post_op_functor) {

    // Used to determine shared memory requirements
    using complex_compute_t = typename FFT_base_arch::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;
    // Determined by PositionSpaceType as complex version, i.e., half half2 or float float2
    using data_buffer_t = std::remove_pointer_t<decltype(d_ptr.buffer_1)>;
    // Allowed half, float (real type image) half2 float2 (complex type image) so with typical case
    // as real valued image, data_io_t != data_buffer_t
    using data_io_t = std::remove_pointer_t<decltype(d_ptr.external_input)>;
    // Could match data_io_t, but need not, will be converted in kernels to match complex_compute_t as needed.
    using external_image_t = OtherImageType;

    // If the user passed in a different exerternal pointer for the image, set it here, and if not,
    // we just alias the external input
    data_io_t*      external_output_ptr;
    buffer_location aliased_output_buffer;
    if ( d_ptr.external_output != nullptr ) {
        external_output_ptr   = d_ptr.external_output;
        aliased_output_buffer = fastfft_external_output;
    }
    else {
        external_output_ptr   = d_ptr.external_input;
        aliased_output_buffer = fastfft_external_input;
    }
    // if constexpr (detail::is_operator<fft_operator::thread, FFT_base_arch>::value) {
    if constexpr ( detail::has_any_block_operator<FFT_base_arch>::value ) {
        switch ( kernel_type ) {
            case r2c_none_XY: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_FFT ) {

#ifdef USE_FOLDED_R2C_C2R
                    using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::r2c>( ) + RealFFTOptions<complex_layout::natural, real_mode::folded>);
#else
                    using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::c2c>( ));
#endif
                    cudaError_t        error_code = cudaSuccess;
                    auto               workspace  = make_workspace<FFT>(error_code);
                    const LaunchParams LP         = SetLaunchParameters(r2c_none_XY, FFT::elements_per_thread, 1, 1);

                    int shared_memory = FFT::shared_memory_size;
                    CheckSharedMemory(shared_memory, device_properties);

#if FFT_DEBUG_STAGE > 0
                    cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_R2C_NONE_XY<FFT, data_io_t, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));

                    // If impl a round trip, the output will need to be data_io_t
                    if constexpr ( Rank == 1 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_R2C_NONE_XY<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input,
                                reinterpret_cast<data_buffer_t*>(external_output_ptr),
                                LP.mem_offsets,
                                workspace);
                        postcheck;
                        current_buffer = aliased_output_buffer;
                    }
                    else if constexpr ( Rank == 2 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_R2C_NONE_XY<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input,
                                d_ptr.buffer_1,
                                LP.mem_offsets,
                                workspace);
                        postcheck;
                    }
                    current_buffer = fastfft_internal_buffer_1;
#endif
                }
                break;
            }

            case r2c_none_XZ: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_FFT ) {
                    if constexpr ( Rank == 3 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");

#ifdef USE_FOLDED_R2C_C2R
                        using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + RealFFTOptions<complex_layout::natural, real_mode::folded>);
#else
                        using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::c2c>( ));
#endif
                        cudaError_t        error_code = cudaSuccess;
                        auto               workspace  = make_workspace<FFT>(error_code);
                        const LaunchParams LP         = SetLaunchParameters(r2c_none_XZ, FFT::elements_per_thread, 1, XZ_STRIDE);

                        int shared_memory = std::max(LP.threadsPerBlock.y * FFT::shared_memory_size, LP.threadsPerBlock.y * LP.mem_offsets.physical_x_output * (unsigned int)sizeof(complex_compute_t));
                        CheckSharedMemory(shared_memory, device_properties);

#if FFT_DEBUG_STAGE > 0
                        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_R2C_NONE_XZ<FFT, data_io_t, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                        precheck;
                        block_fft_kernel_R2C_NONE_XZ<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input,
                                d_ptr.buffer_1,
                                LP.mem_offsets,
                                workspace);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_1;
#endif
                    }
                }
                break;
            }

            case r2c_decrease_XY: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_FFT ) {
#ifdef USE_FOLDED_R2C_C2R
                    using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + RealFFTOptions<complex_layout::natural, real_mode::folded>);
#else
                    using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::c2c>( ));
#endif

                    cudaError_t        error_code = cudaSuccess;
                    auto               workspace  = make_workspace<FFT>(error_code);
                    const LaunchParams LP         = SetLaunchParameters(r2c_decrease_XY, FFT::elements_per_thread, 1, 1);

                    // the shared mem is mixed between storage, shuffling and FFT. For this kernel we need to add padding to avoid bank conlicts (N/32)
                    int shared_memory = std::max(FFT::shared_memory_size * LP.threadsPerBlock.y, (LP.mem_offsets.shared_input + LP.mem_offsets.shared_input / 32) * (unsigned int)sizeof(complex_compute_t));

                    CheckSharedMemory(shared_memory, device_properties);
#if FFT_DEBUG_STAGE > 0

                    cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_R2C_DECREASE_XY<FFT, data_io_t, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                    if constexpr ( Rank == 1 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_R2C_DECREASE_XY<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input,
                                reinterpret_cast<data_buffer_t*>(external_output_ptr),
                                LP.mem_offsets,
                                _i2pi_div_N<FFT, float>(LP.transform_size.Q),
                                LP.transform_size.Q,
                                workspace);
                        postcheck;
                        current_buffer = aliased_output_buffer;
                    }
                    else if constexpr ( Rank == 2 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_R2C_DECREASE_XY<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input,
                                d_ptr.buffer_1,
                                LP.mem_offsets,
                                _i2pi_div_N<FFT, float>(LP.transform_size.Q),
                                LP.transform_size.Q,
                                workspace);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_1;
                    }
#endif
                }
                break;
            }

            case r2c_increase_XY: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_FFT ) {
#ifdef USE_FOLDED_R2C_C2R
                    using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::r2c>( ) + RealFFTOptions<complex_layout::natural, real_mode::folded>);
#else
                    using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::c2c>( ));
#endif
                    cudaError_t        error_code = cudaSuccess;
                    auto               workspace  = make_workspace<FFT>(error_code);
                    const LaunchParams LP         = SetLaunchParameters(r2c_increase_XY, FFT::elements_per_thread, 1, 1);

                    int shared_memory = LP.mem_offsets.shared_input * sizeof(scalar_compute_t) + FFT::shared_memory_size;

                    CheckSharedMemory(shared_memory, device_properties);
#if FFT_DEBUG_STAGE > 0
                    cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_R2C_INCREASE_XY<FFT, data_io_t, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                    if constexpr ( Rank == 1 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_R2C_INCREASE_XY<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input,
                                reinterpret_cast<data_buffer_t*>(external_output_ptr),
                                LP.mem_offsets,
                                _i2pi_div_N<FFT, float>(LP.transform_size.Q),
                                LP.transform_size.Q,
                                transform_size.L,
                                workspace);
                        postcheck;
                        current_buffer = aliased_output_buffer;
                    }
                    else if constexpr ( Rank == 2 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_R2C_INCREASE_XY<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input,
                                d_ptr.buffer_1,
                                LP.mem_offsets,
                                _i2pi_div_N<FFT, float>(LP.transform_size.Q),
                                LP.transform_size.Q, // TODO: transform_size.Q
                                transform_size.L,
                                workspace);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_1;
                    }
#endif
                }
                break;
            }

            case r2c_increase_XZ: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_FFT ) {
                    if constexpr ( Rank == 3 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
#ifdef USE_FOLDED_R2C_C2R
                        using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::r2c>( ) + RealFFTOptions<complex_layout::natural, real_mode::folded>);
#else
                        using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::c2c>( ));
#endif
                        cudaError_t        error_code = cudaSuccess;
                        auto               workspace  = make_workspace<FFT>(error_code); // FIXME: I don't think this is right when XZ_STRIDE is used
                        const LaunchParams LP         = SetLaunchParameters(r2c_increase_XZ, FFT::elements_per_thread, 1, XZ_STRIDE);

                        // We need shared memory to hold the input array(s) that is const through the kernel.
                        // We alternate using additional shared memory for the computation and the transposition of the data.
                        int shared_memory = std::max(XZ_STRIDE * FFT::shared_memory_size, LP.mem_offsets.physical_x_output / LP.transform_size.Q * (unsigned int)sizeof(complex_compute_t));
                        shared_memory += XZ_STRIDE * LP.mem_offsets.shared_input * (unsigned int)sizeof(scalar_compute_t);

                        CheckSharedMemory(shared_memory, device_properties);
#if FFT_DEBUG_STAGE > 0
                        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_R2C_INCREASE_XZ<FFT, data_io_t, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                        precheck;
                        block_fft_kernel_R2C_INCREASE_XZ<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input,
                                d_ptr.buffer_1,
                                LP.mem_offsets,
                                _i2pi_div_N<FFT, float>(LP.transform_size.Q),
                                LP.transform_size.Q,
                                workspace);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_1;
#endif
                    }
                }
                break;
            }

            case c2c_fwd_none: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_FFT ) {
                    using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::c2c>( ));

                    const LaunchParams LP = SetLaunchParameters(c2c_fwd_none, FFT::elements_per_thread, 1, 1);

                    cudaError_t      error_code    = cudaSuccess;
                    DebugUnused auto workspace     = make_workspace<FFT>(error_code);
                    DebugUnused int  shared_memory = FFT::shared_memory_size;

#if FFT_DEBUG_STAGE > 2
                    CheckSharedMemory(shared_memory, device_properties);
                    cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_NONE<FFT, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                    if constexpr ( Rank == 1 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_C2C_NONE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input,
                                reinterpret_cast<data_buffer_t*>(external_output_ptr),
                                LP.mem_offsets,
                                workspace);
                        postcheck;
                    }
                    else if constexpr ( Rank == 2 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_1, "current_buffer != fastfft_internal_buffer_1");
                        precheck;
                        block_fft_kernel_C2C_NONE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.buffer_1,
                                reinterpret_cast<data_buffer_t*>(external_output_ptr),
                                LP.mem_offsets,
                                workspace);
                        postcheck;
                    }
                    else if constexpr ( Rank == 3 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_2, "current_buffer != fastfft_internal_buffer_2");
                        precheck;
                        block_fft_kernel_C2C_NONE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.buffer_2,
                                reinterpret_cast<data_buffer_t*>(external_output_ptr),
                                LP.mem_offsets,
                                workspace);
                        postcheck;
                    }
                    current_buffer = aliased_output_buffer;
#endif
                }
                break;
            }

            case c2c_fwd_none_XYZ: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_FFT ) {
                    if constexpr ( Rank == 3 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_1, "current_buffer != fastfft_internal_buffer_1");

                        using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::c2c>( ));

                        const LaunchParams LP = SetLaunchParameters(c2c_fwd_none_XYZ, FFT::elements_per_thread, 1, XZ_STRIDE);

                        cudaError_t error_code    = cudaSuccess;
                        auto        workspace     = make_workspace<FFT>(error_code);
                        int         shared_memory = std::max(XZ_STRIDE * FFT::shared_memory_size, size_of<FFT>::value * (unsigned int)sizeof(complex_compute_t) * XZ_STRIDE);

#if FFT_DEBUG_STAGE > 1

                        CheckSharedMemory(shared_memory, device_properties);
                        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_NONE_XYZ<FFT, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                        precheck;
                        block_fft_kernel_C2C_NONE_XYZ<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.buffer_1,
                                d_ptr.buffer_2,
                                LP.mem_offsets,
                                workspace);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_2;
#endif
                    }
                }
                break;
            }

            case c2c_fwd_decrease: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_FFT ) {
                    using FFT                     = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::c2c>( ));
                    cudaError_t        error_code = cudaSuccess;
                    auto               workspace  = make_workspace<FFT>(error_code);
                    const LaunchParams LP         = SetLaunchParameters(c2c_fwd_decrease, FFT::elements_per_thread, 1, 1);

#if FFT_DEBUG_STAGE > 2
                    // the shared mem is mixed between storage, shuffling and FFT. For this kernel we need to add padding to avoid bank conlicts (N/32)
                    // For decrease methods, the shared_input > shared_output
                    int shared_memory = std::max(FFT::shared_memory_size * LP.threadsPerBlock.y, (LP.mem_offsets.shared_input + LP.mem_offsets.shared_input / 32) * (unsigned int)sizeof(complex_compute_t));

                    CheckSharedMemory(shared_memory, device_properties);
                    cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_DECREASE<FFT, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));

                    if constexpr ( Rank == 1 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_C2C_DECREASE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input,
                                reinterpret_cast<data_buffer_t*>(external_output_ptr),
                                LP.mem_offsets,
                                _i2pi_div_N<FFT, float>(LP.transform_size.Q),
                                LP.transform_size.Q,
                                workspace);
                        postcheck;
                    }
                    else if constexpr ( Rank == 2 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_1, "current_buffer != fastfft_internal_buffer_1");
                        precheck;
                        block_fft_kernel_C2C_DECREASE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.buffer_1,
                                reinterpret_cast<data_buffer_t*>(external_output_ptr),
                                LP.mem_offsets,
                                _i2pi_div_N<FFT, float>(LP.transform_size.Q),
                                LP.transform_size.Q,
                                workspace);
                        postcheck;
                    }
                    // Rank 3 not yet implemented
                    current_buffer = aliased_output_buffer;
#endif
                }
                break;
            }

            case c2c_fwd_increase_XYZ: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_FFT ) {
                    if constexpr ( Rank == 3 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_1, "current_buffer != fastfft_internal_buffer_1");
                        using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::c2c>( ));

                        const LaunchParams LP = SetLaunchParameters(c2c_fwd_increase_XYZ, FFT::elements_per_thread, 1, XZ_STRIDE);

                        cudaError_t error_code = cudaSuccess;
                        auto        workspace  = make_workspace<FFT>(error_code);

                        // We need shared memory to hold the input array(s) that is const through the kernel.
                        // We alternate using additional shared memory for the computation and the transposition of the data.
                        int shared_memory = std::max(XZ_STRIDE * FFT::shared_memory_size, XZ_STRIDE * LP.mem_offsets.physical_x_output / LP.transform_size.Q * (unsigned int)sizeof(complex_compute_t));
                        shared_memory += XZ_STRIDE * LP.mem_offsets.shared_input * (unsigned int)sizeof(complex_compute_t);

#if FFT_DEBUG_STAGE > 1

                        CheckSharedMemory(shared_memory, device_properties);
                        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_INCREASE_XYZ<FFT, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                        precheck;
                        block_fft_kernel_C2C_INCREASE_XYZ<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.buffer_1,
                                d_ptr.buffer_2,
                                LP.mem_offsets,
                                _i2pi_div_N<FFT, float>(LP.transform_size.Q),
                                LP.transform_size.Q,
                                workspace);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_2;
#endif
                    }
                }
                break;
            }

            case c2c_fwd_increase: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_FFT ) {
                    using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::c2c>( ));

                    const LaunchParams LP = SetLaunchParameters(c2c_fwd_increase, FFT::elements_per_thread, 1, 1);

                    cudaError_t error_code    = cudaSuccess;
                    auto        workspace     = make_workspace<FFT>(error_code);
                    int         shared_memory = FFT::shared_memory_size + (unsigned int)sizeof(complex_compute_t) * (LP.mem_offsets.shared_input + LP.mem_offsets.shared_output);
#if FFT_DEBUG_STAGE > 2
                    CheckSharedMemory(shared_memory, device_properties);
                    cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_INCREASE<FFT, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));

                    if constexpr ( Rank == 1 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_C2C_INCREASE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input, reinterpret_cast<data_buffer_t*>(external_output_ptr),
                                LP.mem_offsets,
                                _i2pi_div_N<FFT, float>(LP.transform_size.Q),
                                LP.transform_size.Q,
                                LP.transform_size.L,
                                workspace);
                        postcheck;
                    }
                    else if constexpr ( Rank == 2 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_1, "current_buffer != fastfft_internal_buffer_1");
                        precheck;
                        block_fft_kernel_C2C_INCREASE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.buffer_1,
                                reinterpret_cast<data_buffer_t*>(external_output_ptr),
                                LP.mem_offsets,
                                _i2pi_div_N<FFT, float>(LP.transform_size.Q),
                                LP.transform_size.Q,
                                LP.transform_size.L,
                                workspace);
                        postcheck;
                    }
                    else if constexpr ( Rank == 3 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_2, "current_buffer != fastfft_internal_buffer_2");
                        precheck;
                        block_fft_kernel_C2C_INCREASE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.buffer_2,
                                reinterpret_cast<data_buffer_t*>(external_output_ptr),
                                LP.mem_offsets,
                                _i2pi_div_N<FFT, float>(LP.transform_size.Q),
                                LP.transform_size.Q,
                                LP.transform_size.L,
                                workspace);
                        postcheck;
                    }
                    current_buffer = aliased_output_buffer;
#endif
                }
                break;
            }

            case c2c_inv_none: {
                if constexpr ( FFT_ALGO_t == Generic_Inv_FFT ) {
                    using FFT = decltype(FFT_base_arch( ) + Type<fft_type::c2c>( ) + Direction<fft_direction::inverse>( ));

                    const LaunchParams LP = SetLaunchParameters(c2c_inv_none, FFT::elements_per_thread, 1, 1);

                    cudaError_t error_code = cudaSuccess;
                    auto        workspace  = make_workspace<FFT>(error_code);

                    int shared_memory = FFT::shared_memory_size;
                    CheckSharedMemory(shared_memory, device_properties);
#if FFT_DEBUG_STAGE > 4
                    cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_NONE<FFT, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                    if constexpr ( Rank == 1 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_C2C_NONE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                reinterpret_cast<data_buffer_t*>(d_ptr.external_input),
                                external_output_ptr,
                                LP.mem_offsets,
                                workspace);
                        postcheck;
                        current_buffer = aliased_output_buffer;
                    }
                    else if constexpr ( Rank == 2 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_C2C_NONE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                reinterpret_cast<data_buffer_t*>(d_ptr.external_input),
                                d_ptr.buffer_1,
                                LP.mem_offsets,
                                workspace);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_1;
                    }
#endif

                    // do something
                }
                break;
            }

            case c2c_inv_none_XZ: {
                if constexpr ( FFT_ALGO_t == Generic_Inv_FFT ) {
                    if constexpr ( Rank == 3 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        using FFT = decltype(FFT_base_arch( ) + Type<fft_type::c2c>( ) + Direction<fft_direction::inverse>( ));

                        const LaunchParams LP = SetLaunchParameters(c2c_inv_none_XZ, FFT::elements_per_thread, 1, XZ_STRIDE);

                        cudaError_t error_code = cudaSuccess;
                        auto        workspace  = make_workspace<FFT>(error_code);

                        int shared_memory = std::max(FFT::shared_memory_size * XZ_STRIDE, size_of<FFT>::value * (unsigned int)sizeof(complex_compute_t) * XZ_STRIDE);

                        CheckSharedMemory(shared_memory, device_properties);
#if FFT_DEBUG_STAGE > 4
                        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_NONE_XZ<FFT, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                        precheck;
                        block_fft_kernel_C2C_NONE_XZ<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                reinterpret_cast<data_buffer_t*>(d_ptr.external_input),
                                d_ptr.buffer_1,
                                LP.mem_offsets,
                                workspace);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_1;
#endif
                    }
                    // do something
                }
                break;
            }

            case c2c_inv_none_XYZ: {
                if constexpr ( FFT_ALGO_t == Generic_Inv_FFT ) {
                    if constexpr ( Rank == 3 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_1, "current_buffer != fastfft_internal_buffer_1");
                        using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::inverse>( ) + Type<fft_type::c2c>( ));

                        const LaunchParams LP = SetLaunchParameters(c2c_inv_none_XYZ, FFT::elements_per_thread, 1, XZ_STRIDE);

                        cudaError_t error_code    = cudaSuccess;
                        auto        workspace     = make_workspace<FFT>(error_code);
                        int         shared_memory = std::max(XZ_STRIDE * FFT::shared_memory_size, size_of<FFT>::value * (unsigned int)sizeof(complex_compute_t) * XZ_STRIDE);

#if FFT_DEBUG_STAGE > 5
                        CheckSharedMemory(shared_memory, device_properties);
                        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_NONE_XYZ<FFT, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                        precheck;
                        block_fft_kernel_C2C_NONE_XYZ<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.buffer_1,
                                d_ptr.buffer_2,
                                LP.mem_offsets,
                                workspace);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_2;
#endif
                    }
                }
                break;
            }

            case c2c_inv_decrease: {
                if constexpr ( FFT_ALGO_t == Generic_Inv_FFT ) {
                    using FFT                     = decltype(FFT_base_arch( ) + Direction<fft_direction::inverse>( ) + Type<fft_type::c2c>( ));
                    cudaError_t        error_code = cudaSuccess;
                    auto               workspace  = make_workspace<FFT>(error_code);
                    const LaunchParams LP         = SetLaunchParameters(c2c_inv_decrease, FFT::elements_per_thread, 1, 1);

#if FFT_DEBUG_STAGE > 4
                    // the shared mem is mixed between storage, shuffling and FFT. For this kernel we need to add padding to avoid bank conlicts (N/32)
                    // For decrease methods, the shared_input > shared_output
                    int shared_memory = std::max(FFT::shared_memory_size * LP.threadsPerBlock.y, (LP.mem_offsets.shared_input + LP.mem_offsets.shared_input / 32) * (unsigned int)sizeof(complex_compute_t));

                    CheckSharedMemory(shared_memory, device_properties);
                    cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_DECREASE<FFT, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));

                    if constexpr ( Rank == 1 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_C2C_DECREASE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                reinterpret_cast<data_buffer_t*>(d_ptr.external_input),
                                external_output_ptr, LP.mem_offsets,
                                _i2pi_div_N<FFT, float>(LP.transform_size.Q),
                                LP.transform_size.Q,
                                workspace);
                        postcheck;
                        current_buffer = aliased_output_buffer;
                    }
                    else if constexpr ( Rank == 2 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_C2C_DECREASE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                reinterpret_cast<data_buffer_t*>(d_ptr.external_input),
                                d_ptr.buffer_1,
                                LP.mem_offsets,
                                _i2pi_div_N<FFT, float>(LP.transform_size.Q),
                                LP.transform_size.Q,
                                workspace);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_1;
                    }
                    // 3D not yet implemented
#endif
                }
                break;
            }

            case c2c_inv_increase: {
                if constexpr ( FFT_ALGO_t == Generic_Inv_FFT ) {
                    MyFFTRunTimeAssertTrue(false, "c2c_inv_increase is not yet implemented.");

#if FFT_DEBUG_STAGE > 4
// TODO;
#endif
                }
                break;
            }

            case c2r_none: {
                if constexpr ( FFT_ALGO_t == Generic_Inv_FFT ) {
#ifdef USE_FOLDED_C2R
                    using c2r_folded_options = RealFFTOptions<complex_layout::natural, real_mode::folded>;
                    using extended_base      = decltype(FFT_base_arch( ) + Direction<fft_direction::inverse>( ) + Type<fft_type::c2r>( ) + c2r_folded_options( ));
                    using FFT                = check_ept_t<extended_base>;

#else
                    using FFT = check_ept_t<decltype(FFT_base_arch( ) + Direction<fft_direction::inverse>( ) + Type<fft_type::c2r>( ))>;
#endif

                    const LaunchParams LP = SetLaunchParameters(c2r_none, FFT::elements_per_thread, 1, 1);

                    cudaError_t error_code = cudaSuccess;
                    auto        workspace  = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;        cudaErr(error_code);

                    int shared_memory = FFT::shared_memory_size;

                    CheckSharedMemory(shared_memory, device_properties);
#if FFT_DEBUG_STAGE > 6
                    cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2R_NONE<FFT, data_buffer_t, data_io_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));

                    if constexpr ( Rank == 1 ) {
                        // TODO:
                        // precheck;
                        // block_fft_kernel_C2R_NONE<FFT, data_buffer_t, data_io_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(external_input, external_data_ptr, LP.mem_offsets, workspace);
                        // postcheck;
                    }
                    else if constexpr ( Rank == 2 ) {
                        // TODO:
                        // precheck;
                        // block_fft_kernel_C2R_NONE<FFT, data_buffer_t, data_io_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                        //         intra_complex_input, external_data_ptr, LP.mem_offsets, workspace);
                        // postcheck;
                    }
                    else if constexpr ( Rank == 3 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_2, "current_buffer != fastfft_internal_buffer_2");
                        precheck;
                        block_fft_kernel_C2R_NONE<FFT, data_buffer_t, data_io_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.buffer_2,
                                external_output_ptr,
                                LP.mem_offsets,
                                workspace);
                        postcheck;
                        current_buffer = aliased_output_buffer;
                    }

#endif
                }
                break;
            }

            case c2r_none_XY: {
                if constexpr ( FFT_ALGO_t == Generic_Inv_FFT ) {
#ifdef USE_FOLDED_C2R
                    using c2r_folded_options = RealFFTOptions<complex_layout::natural, real_mode::folded>;
                    using extended_base      = decltype(FFT_base_arch( ) + Direction<fft_direction::inverse>( ) + Type<fft_type::c2r>( ) + c2r_folded_options( ));
                    using FFT                = check_ept_t<extended_base>;

#else
                    using FFT = check_ept_t<decltype(FFT_base_arch( ) + Direction<fft_direction::inverse>( ) + Type<fft_type::c2r>( ))>;
#endif

                    cudaError_t error_code = cudaSuccess;
                    auto        workspace  = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;        cudaErr(error_code);

                    int shared_memory = FFT::shared_memory_size;
#ifdef C2R_BUFFER_LINES
                    // neads to be chosen so that 16 / n_buffer_lines >= FFT::stride (blockDim.x)
                    constexpr unsigned int min_buffer_lines = std::max(16u / FFT::stride, 2u);
                    constexpr unsigned int n_buffer_lines   = size_of<FFT>::value < 32u ? 1 : size_of<FFT>::value < 128u ? std::max(min_buffer_lines, 2u)
                                                                                      : size_of<FFT>::value < 512        ? std::max(min_buffer_lines, 4u)
                                                                                                                         : 8; //std::max(min_buffer_lines, 4u);
                    // For the shared impl, we need more threads in y each will workin on the subfft
                    constexpr unsigned int max_threads_per_block = FFT::max_threads_per_block * n_buffer_lines;
                    static_assert(max_threads_per_block < 1025, "C2R_BUFFER_LINES resulting in too many threads per block.");

                    // Add enough shared mem to swap on each read
                    // FIXME: I don't think there is any way to get this to work, b/c the FFT().exectute method inserts barrier.sync ptx, which afaik behave the same as __syncthreads
                    // So we can't have some threads wait their turn to do the FFT because they will never reach the sync point.
                    // shared_memory = std::max(size_t(FFT::shared_memory_size * n_buffer_lines / 2), size_t(FFT::stride * n_buffer_lines * sizeof(complex_compute_t)));
                    shared_memory = std::max(size_t(FFT::shared_memory_size * n_buffer_lines), size_t(FFT::stride * n_buffer_lines * sizeof(complex_compute_t)));

#else

                    constexpr unsigned int n_buffer_lines        = 1;
                    constexpr unsigned int max_threads_per_block = FFT::max_threads_per_block;
#endif

                    const LaunchParams LP = SetLaunchParameters(c2r_none_XY, FFT::elements_per_thread, n_buffer_lines, 1);

                    CheckSharedMemory(shared_memory, device_properties);

#if FFT_DEBUG_STAGE > 6
                    // cudaErr(cudaFuncSetCacheConfig((void*)block_fft_kernel_C2R_NONE_XY<FFT, max_threads_per_block, data_buffer_t, data_io_t, n_buffer_lines>, cudaFuncCachePreferShared));
                    cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2R_NONE_XY<FFT, max_threads_per_block, data_buffer_t, data_io_t, n_buffer_lines>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));

                    if constexpr ( Rank == 1 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_C2R_NONE_XY<FFT, max_threads_per_block, data_buffer_t, data_io_t, n_buffer_lines><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                reinterpret_cast<data_buffer_t*>(d_ptr.external_input),
                                external_output_ptr,
                                LP.mem_offsets,
                                workspace);
                        postcheck;
                    }
                    else if constexpr ( Rank == 2 ) {
                        // This could be the last step after a partial InvFFT or partial FwdImgInv, and right now, the only way to tell the difference is if the
                        // current buffer is set correctly.
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_1 || current_buffer == fastfft_internal_buffer_2, "current_buffer != fastfft_internal_buffer_1/2");
                        if ( current_buffer == fastfft_internal_buffer_1 ) {
                            // Presumably this is intended to be the second step of an InvFFT
                            precheck;
                            block_fft_kernel_C2R_NONE_XY<FFT, max_threads_per_block, data_buffer_t, data_io_t, n_buffer_lines><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                    d_ptr.buffer_1,
                                    external_output_ptr,
                                    LP.mem_offsets,
                                    workspace);
                            postcheck;
                        }
                        else if ( current_buffer == fastfft_internal_buffer_2 ) {
                            // Presumably this is intended to be the last step in a FwdImgInv
                            precheck;
                            block_fft_kernel_C2R_NONE_XY<FFT, max_threads_per_block, data_buffer_t, data_io_t, n_buffer_lines><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                    d_ptr.buffer_2,
                                    external_output_ptr,
                                    LP.mem_offsets,
                                    workspace);
                            postcheck;
                        }
                        // DebugAssert will fail if current_buffer is not set correctly to end up in an else clause
                    }
                    current_buffer = aliased_output_buffer;
                    // 3D not yet implemented
#endif
                }
                break;
            }

            case c2r_decrease_XY: {
                if constexpr ( FFT_ALGO_t == Generic_Inv_FFT ) {
#ifdef USE_FOLDED_C2R
                    using c2r_folded_options = RealFFTOptions<complex_layout::natural, real_mode::folded>;
                    using extended_base      = decltype(FFT_base_arch( ) + Direction<fft_direction::inverse>( ) + Type<fft_type::c2r>( ) + c2r_folded_options( ));
                    using FFT                = check_ept_t<extended_base>;

#else
                    using FFT                                    = check_ept_t<decltype(FFT_base_arch( ) + Direction<fft_direction::inverse>( ) + Type<fft_type::c2r>( ))>;
#endif

                    const LaunchParams LP = SetLaunchParameters(c2r_decrease_XY, FFT::elements_per_thread, 1, 1);

                    cudaError_t error_code = cudaSuccess;
                    auto        workspace  = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;        cudaErr(error_code);

                    int shared_memory = std::max(FFT::shared_memory_size * LP.gridDims.z, (LP.mem_offsets.shared_input + LP.mem_offsets.shared_input / 32) * (unsigned int)sizeof(complex_compute_t));

                    CheckSharedMemory(shared_memory, device_properties);

                    // size_t size = min(int(device_properties.L2_cache_size * 1.0), device_properties.max_persisting_L2_cache_size);
                    // cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); // set-aside 3/4 of L2 cache for persisting accesses or the max allowed

                    // size_t window_size = std::min(device_properties.accessPolicyMaxWindowSize, int(n_bytes_allocated / 2)); // Select minimum of user defined num_bytes and max window size.

                    // cudaStreamAttrValue stream_attribute; // Stream level attributes data structure
                    // stream_attribute.accessPolicyWindow.num_bytes = window_size; // Number of bytes for persistence access
                    // stream_attribute.accessPolicyWindow.hitRatio  = 1.0; // Hint for cache hit ratio
                    // stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Persistence Property
                    // stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming; // Type of access property on cache miss

#if FFT_DEBUG_STAGE > 6
                    // cudaErr(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte)); //cudaSharedMemBankSizeDefault
                    // cudaErr(cudaFuncSetCacheConfig((void*)block_fft_kernel_C2R_DECREASE_XY<FFT, data_buffer_t, data_io_t>, cudaFuncCachePreferL1)); //
                    // cudaErr(cudaFuncSetSharedMemConfig((const void*)block_fft_kernel_C2R_DECREASE_XY<FFT, data_buffer_t, data_io_t>, cudaSharedMemBankSizeEightByte));
                    // SetCudaFuncCache((const void*)block_fft_kernel_C2R_DECREASE_XY<FFT, data_buffer_t, data_io_t>, cudaFuncCachePreferL1);

                    if constexpr ( Rank == 1 ) {
                        precheck;
                        block_fft_kernel_C2R_DECREASE_XY<FFT, data_buffer_t, data_io_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input,
                                external_output_ptr,
                                LP.mem_offsets,
                                _i2pi_div_N<FFT, float>(LP.transform_size.Q),
                                LP.transform_size.Q,
                                workspace);
                        postcheck;
                    }
                    else if constexpr ( Rank == 2 ) {
                        // This could be the last step after a partial InvFFT or partial FwdImgInv, and right now, the only way to tell the difference is if the
                        // current buffer is set correctly.
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_1 || current_buffer == fastfft_internal_buffer_2, "current_buffer != fastfft_internal_buffer_1/2");
                        if ( current_buffer == fastfft_internal_buffer_1 ) {
                            // stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(d_ptr.buffer_1); // Global Memory data pointer
                            // cudaStreamSetAttribute(cudaStreamPerThread, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
                            precheck;
                            block_fft_kernel_C2R_DECREASE_XY<FFT, data_buffer_t, data_io_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                    d_ptr.buffer_1,
                                    external_output_ptr,
                                    LP.mem_offsets,
                                    _i2pi_div_N<FFT, float>(LP.transform_size.Q),
                                    LP.transform_size.Q,
                                    workspace);
                            postcheck;
                        }
                        else if ( current_buffer == fastfft_internal_buffer_2 ) {
                            // stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(d_ptr.buffer_2); // Global Memory data pointer
                            // cudaStreamSetAttribute(cudaStreamPerThread, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
                            precheck;
                            block_fft_kernel_C2R_DECREASE_XY<FFT, data_buffer_t, data_io_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                    d_ptr.buffer_2,
                                    external_output_ptr,
                                    LP.mem_offsets,
                                    _i2pi_div_N<FFT, float>(LP.transform_size.Q),
                                    LP.transform_size.Q,
                                    workspace);
                            postcheck;
                        }
                        // stream_attribute.accessPolicyWindow.num_bytes = 0; // Setting the window size to 0 disable it
                        // cudaStreamSetAttribute(cudaStreamPerThread, cudaStreamAttributeAccessPolicyWindow, &stream_attribute); // Overwrite the access policy attribute to a CUDA Stream
                        // cudaCtxResetPersistingL2Cache( );
                    }
                    current_buffer = aliased_output_buffer;
                    // cudaErr(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeDefault)); //
#endif
                }
                break;
            }

            case c2r_increase: {
                if constexpr ( FFT_ALGO_t == Generic_Inv_FFT ) {
                    MyFFTRunTimeAssertTrue(false, "c2r_increase is not yet implemented.");
                }
                break;
            }

            case xcorr_fwd_none_inv_decrease: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_Image_Inv_FFT ) {
                    if constexpr ( Rank == 2 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_1, "current_buffer != fastfft_internal_buffer_1");
                        using FFT    = decltype(FFT_base_arch( ) + Type<fft_type::c2c>( ) + Direction<fft_direction::forward>( ));
                        using invFFT = decltype(FFT_base_arch( ) + Type<fft_type::c2c>( ) + Direction<fft_direction::inverse>( ));

                        const LaunchParams LP = SetLaunchParameters(xcorr_fwd_none_inv_decrease, FFT::elements_per_thread, 1, 1);

                        cudaError_t error_code    = cudaSuccess;
                        auto        workspace_fwd = make_workspace<FFT>(error_code); // presumably larger of the two
                        cudaErr(error_code);
                        error_code         = cudaSuccess;
                        auto workspace_inv = make_workspace<invFFT>(error_code); // presumably larger of the two
                        cudaErr(error_code);

                        // Max shared memory needed to store the full 1d fft remaining on the forward transform
                        unsigned int shared_memory = FFT::shared_memory_size + (unsigned int)sizeof(complex_compute_t) * LP.mem_offsets.physical_x_input;
                        // shared_memory = std::max( shared_memory, std::max( invFFT::shared_memory_size * LP.threadsPerBlock.y, (LP.mem_offsets.shared_input + LP.mem_offsets.shared_input/32) * (unsigned int)sizeof(complex_compute_t)));

                        CheckSharedMemory(shared_memory, device_properties);
#if FFT_DEBUG_STAGE > 2

                        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_FWD_NONE_INV_DECREASE_ConjMul<external_image_t, FFT, invFFT, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                        // Right now, because of the n_threads == size_of<FFT> requirement, we are explicitly zero padding, so we need to send an "apparent Q" to know the input size.
                        // Could send the actual size, but later when converting to use the transform decomp with different sized FFTs this will be a more direct conversion.
                        int apparent_Q = size_of<FFT>::value / inv_dims_out.y;
                        precheck;
                        block_fft_kernel_C2C_FWD_NONE_INV_DECREASE_ConjMul<external_image_t, FFT, invFFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                (external_image_t*)other_image_ptr,
                                d_ptr.buffer_1,
                                d_ptr.buffer_2,
                                LP.mem_offsets,
                                _i2pi_div_N<FFT, float>(LP.transform_size.Q),
                                apparent_Q,
                                workspace_fwd,
                                workspace_inv);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_2;
#endif
                    }
                }
                // do something

                break;
            }

            case generic_fwd_increase_op_inv_none: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_Image_Inv_FFT ) {
                    // revert : with 16 ept this method uses 128 reg/thread which limits to 2 blocks /sm, testing 8
                    using mod_base = cufftdx::replace_t<FFT_base_arch, ElementsPerThread<16>>;
                    // For convenience, we are explicitly zero-padding. This is lazy. FIXME
                    using FFT    = decltype(mod_base( ) + Type<fft_type::c2c>( ) + Direction<fft_direction::forward>( ));
                    using invFFT = decltype(mod_base( ) + Type<fft_type::c2c>( ) + Direction<fft_direction::inverse>( ));

                    const LaunchParams LP = SetLaunchParameters(generic_fwd_increase_op_inv_none, FFT::elements_per_thread, 1, 1);

                    cudaError_t error_code    = cudaSuccess;
                    auto        workspace_fwd = make_workspace<FFT>(error_code); // presumably larger of the two
                    cudaErr(error_code);
                    error_code         = cudaSuccess;
                    auto workspace_inv = make_workspace<invFFT>(error_code); // presumably larger of the two
                    cudaErr(error_code);

                    int shared_memory = invFFT::shared_memory_size;
                    CheckSharedMemory(shared_memory, device_properties);

                    // __nv_is_extended_device_lambda_closure_type(type);
                    // __nv_is_extended_host_device_lambda_closure_type(type)
                    if constexpr ( IS_IKF_t<IntraOpType>( ) ) {
// FIXME
#if FFT_DEBUG_STAGE > 2

                        // Right now, because of the n_threads == size_of<FFT> requirement, we are explicitly zero padding, so we need to send an "apparent Q" to know the input size.
                        // Could send the actual size, but later when converting to use the transform decomp with different sized FFTs this will be a more direct conversion.
                        int apparent_Q = size_of<FFT>::value / fwd_dims_in.y;
                        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_FWD_INCREASE_OP_INV_NONE<external_image_t, FFT, invFFT, data_buffer_t, PreOpType, IntraOpType, PostOpType>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                        if constexpr ( Rank == 2 ) {
                            MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_1, "current_buffer != fastfft_internal_buffer_1");
                            precheck;
                            block_fft_kernel_C2C_FWD_INCREASE_OP_INV_NONE<external_image_t, FFT, invFFT, data_buffer_t, PreOpType, IntraOpType, PostOpType><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                    (external_image_t*)other_image_ptr,
                                    d_ptr.buffer_1,
                                    d_ptr.buffer_2,
                                    LP.mem_offsets,
                                    workspace_fwd,
                                    workspace_inv,
                                    pre_op_functor,
                                    intra_op_functor,
                                    post_op_functor);
                            postcheck;
                            current_buffer = fastfft_internal_buffer_2;
                        }
                        else if constexpr ( Rank == 3 ) {
                            MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_2, "current_buffer != fastfft_internal_buffer_2");
                            precheck;
                            block_fft_kernel_C2C_FWD_INCREASE_OP_INV_NONE<external_image_t, FFT, invFFT, data_buffer_t, PreOpType, IntraOpType, PostOpType><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                    (external_image_t*)other_image_ptr,
                                    d_ptr.buffer_2,
                                    d_ptr.buffer_1,
                                    LP.mem_offsets,
                                    workspace_fwd,
                                    workspace_inv,
                                    pre_op_functor,
                                    intra_op_functor,
                                    post_op_functor);
                            postcheck;
                            current_buffer = fastfft_internal_buffer_1;
                        }

#endif
                    }

                    // do something
                }
                break;
            }
            default: {
                MyFFTDebugAssertFalse(true, "Unsupported transform stage.");
                break;
            }

        } // kernel type switch.
    }
    else {
        static_no_thread_fft_support_yet( );
    }

    //
} // end set and launch kernel

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Some helper functions that are annoyingly long to have in the header.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::GetTransformSize(KernelType kernel_type) {
    // Set member variable transform_size.N (.P .L .Q)

    if ( IsR2CType(kernel_type) ) {
        AssertDivisibleAndFactorOf2(kernel_type, std::max(fwd_dims_in.x, fwd_dims_out.x), std::min(fwd_dims_in.x, fwd_dims_out.x));
    }
    else if ( IsC2RType(kernel_type) ) {
        // FIXME
        if ( kernel_type == c2r_decrease_XY ) {
            AssertDivisibleAndFactorOf2(kernel_type, std::max(inv_dims_in.x, inv_dims_out.x), std::max(inv_dims_in.x, inv_dims_out.x));
        }
        else {
            AssertDivisibleAndFactorOf2(kernel_type, std::max(inv_dims_in.x, inv_dims_out.x), std::min(inv_dims_in.x, inv_dims_out.x));
        }
    }
    else {
        // C2C type
        if ( IsForwardType(kernel_type) ) {
            switch ( Rank ) {
                case 1: {
                    AssertDivisibleAndFactorOf2(kernel_type, std::max(fwd_dims_in.x, fwd_dims_out.x), std::min(fwd_dims_in.x, fwd_dims_out.x));
                    break;
                }
                case 2: {
                    if ( kernel_type == generic_fwd_increase_op_inv_none ) {
                        // FIXME
                        AssertDivisibleAndFactorOf2(kernel_type, std::max(fwd_dims_in.y, fwd_dims_out.y), std::max(fwd_dims_in.y, fwd_dims_out.y));
                    }
                    else {
                        AssertDivisibleAndFactorOf2(kernel_type, std::max(fwd_dims_in.y, fwd_dims_out.y), std::min(fwd_dims_in.y, fwd_dims_out.y));
                    }
                    break;
                }
                case 3: {
                    if ( IsTransormAlongZ(kernel_type) ) {
                        AssertDivisibleAndFactorOf2(kernel_type, std::max(fwd_dims_in.z, fwd_dims_out.z), std::min(fwd_dims_in.z, fwd_dims_out.z));
                    }
                    else {
                        AssertDivisibleAndFactorOf2(kernel_type, std::max(fwd_dims_in.y, fwd_dims_out.y), std::min(fwd_dims_in.y, fwd_dims_out.y));
                    }

                    break;
                }

                default: {
                    std::cerr << "Transform dimension: " << Rank << std::endl;
                    MyFFTDebugAssertTrue(false, "ERROR: Invalid transform dimension for c2c fwd type.\n");
                }
            }
        }
        else {
            switch ( Rank ) {
                case 1: {
                    AssertDivisibleAndFactorOf2(kernel_type, std::max(inv_dims_in.x, inv_dims_out.x), std::min(inv_dims_in.x, inv_dims_out.x));
                    break;
                }
                case 2: {
                    if ( kernel_type == xcorr_fwd_none_inv_decrease ) {
                        // FIXME, for now using full transform
                        AssertDivisibleAndFactorOf2(kernel_type, std::max(inv_dims_in.y, inv_dims_out.y), std::max(inv_dims_in.y, inv_dims_out.y));
                    }
                    else {
                        AssertDivisibleAndFactorOf2(kernel_type, std::max(inv_dims_in.y, inv_dims_out.y), std::min(inv_dims_in.y, inv_dims_out.y));
                    }
                    break;
                }
                case 3: {
                    if ( IsTransormAlongZ(kernel_type) ) {
                        AssertDivisibleAndFactorOf2(kernel_type, std::max(inv_dims_in.z, inv_dims_out.z), std::min(inv_dims_in.z, inv_dims_out.z));
                    }
                    else {
                        AssertDivisibleAndFactorOf2(kernel_type, std::max(inv_dims_in.y, inv_dims_out.y), std::min(inv_dims_in.y, inv_dims_out.y));
                    }

                    break;
                }

                default: {
                    MyFFTDebugAssertTrue(false, "ERROR: Invalid transform dimension for c2c inverse type.\n");
                }
            }
        }
    }

} // end GetTransformSize function

template <class ComputeBaseType, class PositionSpaceType, class OtherImageType, int Rank>
LaunchParams FourierTransformer<ComputeBaseType, PositionSpaceType, OtherImageType, Rank>::SetLaunchParameters(KernelType         kernel_type,
                                                                                                               const unsigned int ept,
                                                                                                               const unsigned int stride_y,
                                                                                                               const unsigned int stride_z) {
    /*
    Assuming:
    1) r2c/c2r imply forward/inverse transform. 
       c2c_padded implies forward transform.
    2) for 2d or 3d transforms the x/y dimensions are transposed in momentum space during store on the 1st set of 1ds transforms.
    3) if 1d then z = y = 1.

    threadsPerBlock = size/threads_per_fft (for thread based transforms)
                    = size of fft ( for block based transforms ) NOTE: Something in cufftdx seems to be very picky about this. Launching > threads seem to cause problems.
    gridDims = number of 1d FFTs, placed on blockDim perpendicular
    shared_input/output = number of elements reserved in dynamic shared memory. TODO add a check on minimal (48k) and whether this should be increased (depends on Arch)
    physical_x_input/output = number of elements along the fast (x) dimension, depends on fftw padding && whether the memory is currently transposed in x/y
    twiddle_in = +/- 2*PI/Largest dimension : + for the inverse transform
    Q = number of sub-transforms
  */

    // Some member variables are copied here so that we can be sure they are not changed and can be used as __grid_constants__ in the kernel.
    LaunchParams L;

    L.transform_size = transform_size;
    // Set some default values so we can ensure we hit some actual branch
    constexpr unsigned int unset_value = std::numeric_limits<unsigned int>::max( );
    L.mem_offsets.shared_input         = unset_value;
    L.mem_offsets.shared_output        = unset_value;
    L.mem_offsets.physical_x_input     = unset_value;
    L.mem_offsets.physical_x_output    = unset_value;
    L.gridDims                         = dim3(0, 0, 0);
    L.threadsPerBlock                  = dim3(0, 0, 0);

    SizeChangeType::Enum size_change_type;
    if ( IsForwardType(kernel_type) ) {
        size_change_type = fwd_size_change_type;
    }
    else {
        size_change_type = inv_size_change_type;
    }

    // Set the shared mem sizes, which depend on the size_change_type
    // FIXME: this should also depend on any coalescing, but that is currently handled in one way (XZ_STRIDE) for 3d transorms,
    // and per kernel / with defines in the call site for experimental 2d use.
    switch ( size_change_type ) {
        case SizeChangeType::no_change: {
            // no shared memory is needed outside that for the FFT itself.
            // For C2C kernels of size_type increase, the shared output may be reset below in order to store for coalesced global writes.
            L.mem_offsets.shared_input  = 0;
            L.mem_offsets.shared_output = 0;
            break;
        }
        case SizeChangeType::decrease: {
            // Prior to reduction, we must be able to store the full transform. An alternate algorithm with multiple reads would relieve this dependency and
            // may be worth considering if L2 cache residence on Ampere is an effective way to reduce repeated Globabl memory access.
            // Note: that this shared memory is not static, in the sense that it is used both for temporory fast storage, as well as the calculation of the FFT. The max of those two requirments is calculated per kernel.
            L.mem_offsets.shared_input = transform_size.N;
            if ( IsR2CType(kernel_type) ) {
                L.mem_offsets.shared_output = 0;
            }
            else {
                // FIXME: this line is just from case increase, haven't thought about it.
                L.mem_offsets.shared_output = transform_size.N;
            }
            break;
        }
        case SizeChangeType::increase: {
            // We want to re-use the input memory as we loop over construction of the full FFT. This shared memory is independent of the
            // memory used for the FFT itself.
            L.mem_offsets.shared_input = transform_size.P;
            if ( IsR2CType(kernel_type) ) {
                L.mem_offsets.shared_output = 0;
            }
            else {
                L.mem_offsets.shared_output = transform_size.N;
            }
            // Note: This is overwritten in the C2C methods as it depends on 1d vs 2d and fwd vs inv.
            break;
        }
        default: {
            MyFFTDebugAssertTrue(false, "Unknown size_change_type ( " + std::to_string(size_change_type) + " )");
        }
    } // switch on size change

    // Set the grid dimensions and pixel pitch
    if ( IsR2CType(kernel_type) ) {
        L.gridDims                     = dim3(1, fwd_dims_in.y, fwd_dims_in.z);
        L.mem_offsets.physical_x_input = fwd_dims_in.w * 2; // scalar type, natural
        // Note: for those algs that transpose, the new physical x is still the respective input logical dimension
        if ( Is_XY_Transposed(kernel_type) ) {
            L.mem_offsets.physical_x_output = fwd_dims_in.y;
        }
        else if ( Is_XZ_Transposed(kernel_type) ) {
            L.mem_offsets.physical_x_output = fwd_dims_in.z;
        }
        else {
            L.mem_offsets.physical_x_output = fwd_dims_out.w;
        }
    }

    else if ( IsC2RType(kernel_type) ) {
        // This is always the last op, so if there is a size change, it will have happened once on C2C, reducing the number of blocks
        if constexpr ( Rank == 2 ) {
            MyFFTDebugAssertTrue(Is_XY_Transposed(kernel_type), "ERROR: Rank 2 C2R should always be transposed.");
            // For 2d cases we are still transposed XY
            L.gridDims                      = dim3(1, inv_dims_out.y, 1);
            L.mem_offsets.physical_x_input  = inv_dims_out.y;
            L.mem_offsets.physical_x_output = inv_dims_out.w * 2;
        }
        else {
            MyFFTDebugAssertTrue(! Is_XY_Transposed(kernel_type) && ! Is_XY_Transposed(kernel_type) && ! IsTransormAlongZ(kernel_type), "ERROR: Rank 3 C2R should always be transposed.");
            // For 3d cases we have already returned all logical axes to match physical
            L.gridDims                      = dim3(1, inv_dims_out.y, inv_dims_out.z);
            L.mem_offsets.physical_x_input  = inv_dims_in.w;
            L.mem_offsets.physical_x_output = inv_dims_out.w * 2;
        }
    }
    else // C2C type
    {
        // FIXME: this currently (and correctly) assumes that the input/output data are real, and so we must be
        // RETURN
        static_assert(Rank != 1, "Rank 1 C2C not yet implemented."); // L.gridDims = dim3(1, 1, 1);
        static_assert(std::is_same_v<PositionSpaceType, float> || std::is_same_v<PositionSpaceType, __half>, "PositionSpaceType must be real valued");
        switch ( transform_stage_completed ) {
            case 0: {
                // We currently are only supporting real valued input/output so this should not happen.
                MyFFTDebugAssertTrue(false, "ERROR: transform_stage_completed == 0 should not happen.");
                break;
            }
            case 1: {
                // if 2d XY are transposed, if 3d XZ are transposed.
                if constexpr ( Rank == 2 ) {
                    L.gridDims                      = dim3(1, fwd_dims_out.w, 1);
                    L.mem_offsets.physical_x_input  = fwd_dims_in.y;
                    L.mem_offsets.physical_x_output = fwd_dims_out.y;
                }
                else {
                    MyFFTDebugAssertTrue(IsTransormAlongZ(kernel_type), "ERROR: transform_stage_completed == 1 should only happen for 3d transforms.");
                    L.gridDims                      = dim3(1, fwd_dims_in.y, fwd_dims_out.w);
                    L.mem_offsets.physical_x_input  = fwd_dims_in.z;
                    L.mem_offsets.physical_x_output = fwd_dims_in.y;
                }

                break;
            }
            case 2: {
                // This should only ever be set for a 3d transform
                MyFFTDebugAssertTrue(Rank == 3, "ERROR: transform_stage_completed == 2 should only happen for 3d transforms.");
                MyFFTDebugAssertTrue(! Is_XY_Transposed(kernel_type) && ! IsTransormAlongZ(kernel_type), "ERROR: transform_stage_completed == 2 should only happen for 3d transforms.");
                L.gridDims                      = dim3(1, fwd_dims_out.w, fwd_dims_out.z);
                L.mem_offsets.physical_x_input  = fwd_dims_in.y;
                L.mem_offsets.physical_x_output = fwd_dims_out.y;
                break;
            }
            case 3: {
                // There are no cases currently where this should happen here (but is used in copy ops)
                MyFFTDebugAssertTrue(false, "ERROR: transform_stage_completed == 3 should not happen.");
                break;
            }
            case 4: {
                // This should be set only when calling InvFFT and in all cases we should have the y dim input
                L.mem_offsets.physical_x_input = inv_dims_in.y;
                // For 2d we handle the transpose on C2R, for 3d we should be transposing physical XZ, logical y z
                if constexpr ( Rank == 2 ) {
                    L.mem_offsets.physical_x_output = inv_dims_out.y;
                    L.gridDims                      = dim3(1, inv_dims_in.w, 1);
                }
                else {
                    MyFFTDebugAssertTrue(Is_XZ_Transposed(kernel_type), "ERROR: transform_stage_completed == 4 should only happen for 3d transforms.");
                    L.mem_offsets.physical_x_output = inv_dims_in.z; // note dims in as we haven't done anything to the logical z dimension yet
                    L.gridDims                      = dim3(1, inv_dims_in.w, inv_dims_in.z);
                }
                break;
            }
            case 5: {
                //
                if constexpr ( Rank == 2 ) {
                    // Currently with real valued input/output this should not happen.
                    MyFFTDebugAssertTrue(false, "ERROR: transform_stage_completed == 5 should not happen for Rank 2.");
                }
                else {
                    MyFFTDebugAssertTrue(IsTransormAlongZ(kernel_type), "ERROR: transform_stage_completed == 5 should only happen for 3d transforms.");
                    L.gridDims                      = dim3(1, inv_dims_in.w, inv_dims_out.y);
                    L.mem_offsets.physical_x_input  = inv_dims_in.z;
                    L.mem_offsets.physical_x_output = inv_dims_in.w; // we write out the logical x to the physical N/2+1
                }
                break;
            }
            case 6: {

                // Currently with real valued input/output this should not happen.
                MyFFTDebugAssertTrue(false, "ERROR: transform_stage_completed == 6 should not happen for Rank 2 or 3 c2c with real valued input/output.");
                break;
            }
            case 7: {
                // Currently with real valued input/output this should not happen.
                MyFFTDebugAssertTrue(false, "ERROR: transform_stage_completed == 7 should not happen for Rank 2 or 3 c2c with real valued input/output.");
                break;
            }
            default: {
                MyFFTDebugAssertTrue(false, "ERROR: transform_stage_completed is not a valid value.");
            }
        }
    }

    // Now handle setting block dim
    if ( IsRoundTripType(kernel_type) ) {
        L.threadsPerBlock = dim3(transform_size.N / ept, 1, 1);
        if ( IsDecreaseSizeType(kernel_type) ) {
            // FIXME: I'm not sure what was supposed to be happending here and it needs to be reviewd.
            L.gridDims                      = dim3(1, fwd_dims_out.w, 1);
            L.mem_offsets.physical_x_input  = inv_dims_in.y;
            L.mem_offsets.physical_x_output = inv_dims_out.y;
        }
        else {
            // FIXME: I'm not sure what was supposed to be happending here and it needs to be reviewd.
            L.gridDims                      = dim3(1, fwd_dims_out.w, 1);
            L.mem_offsets.physical_x_input  = fwd_dims_in.y;
            L.mem_offsets.physical_x_output = inv_dims_out.y;
        }
    }
    else {
        if ( size_change_type == SizeChangeType::decrease ) {
            L.threadsPerBlock = dim3(transform_size.P / ept, transform_size.Q, 1);
        }
        else {
            // In the current xcorr methods that have INCREASE, explicit zero padding is used, so this will be overridden (overrode?) with transform_size.N
            L.threadsPerBlock = dim3(transform_size.P / ept, 1, 1);
        }
    }

    if ( stride_y > 1 ) {
        L.threadsPerBlock.y = stride_y;
        L.gridDims.y /= stride_y;
    }

    if ( stride_z > 1 ) {
        L.threadsPerBlock.y = stride_z;
        L.gridDims.z /= stride_z;
    }

    // FIXME
    // Some shared memory over-rides
    if ( kernel_type == c2c_inv_decrease || kernel_type == c2c_inv_increase ) {
        L.mem_offsets.shared_output = inv_dims_out.y;
    }

    // Maybe these could be debug asserts, but I think the cost is minimal. TOOD: see if there is any measurable performance hit.
    MyFFTRunTimeAssertFalse(L.mem_offsets.physical_x_input == unset_value, "physical_x_input not set");
    MyFFTRunTimeAssertFalse(L.mem_offsets.physical_x_output == unset_value, "physical_x_output not set");
    MyFFTRunTimeAssertFalse(L.mem_offsets.shared_input == unset_value, "shared_input not set");
    MyFFTRunTimeAssertFalse(L.mem_offsets.shared_output == unset_value, "shared_output not set");
    MyFFTRunTimeAssertFalse(L.threadsPerBlock.x == 0, "threadsPerBlock.x not set");
    MyFFTRunTimeAssertFalse(L.threadsPerBlock.y == 0, "threadsPerBlock.y not set");
    MyFFTRunTimeAssertFalse(L.threadsPerBlock.z == 0, "threadsPerBlock.z not set");
    MyFFTRunTimeAssertFalse(L.gridDims.x == 0, "gridDims.x not set");
    MyFFTRunTimeAssertFalse(L.gridDims.y == 0, "gridDims.y not set");
    MyFFTRunTimeAssertFalse(L.gridDims.z == 0, "gridDims.z not set");

    return L;
}

void GetCudaDeviceProps(DeviceProps& dp) {
    int major = 0;
    int minor = 0;

    cudaErr(cudaGetDevice(&dp.device_id));
    cudaErr(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dp.device_id));
    cudaErr(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dp.device_id));

    dp.device_arch = major * 100 + minor * 10;

    // Newer architectures are normalized to the nearest (lower) arch with an explicit dispatch case in
    // SelectSizeAndTypeWithFold. This mirrors cufftdx's own record forwarding (e.g. sm_121 -> sm_120 records).
    //   9.x  (Hopper)                     -> 900
    //   10.x, 11.x (Blackwell datacenter) -> 1000
    //   12.x (Blackwell GeForce/RTX)      -> 1200
    if ( dp.device_arch > 890 ) {
        if ( dp.device_arch >= 1200 )
            dp.device_arch = 1200;
        else if ( dp.device_arch >= 1000 )
            dp.device_arch = 1000;
        else
            dp.device_arch = 900;
    }

    MyFFTRunTimeAssertTrue(dp.device_arch == 700 || dp.device_arch == 750 || dp.device_arch == 800 || dp.device_arch == 860 || dp.device_arch == 890 || dp.device_arch == 900 || dp.device_arch == 1000 || dp.device_arch == 1200,
                           "FastFFT currently only supports compute capability [7.0, 7.5, 8.0, 8.6, 8.9, 9.0+].");

    cudaErr(cudaDeviceGetAttribute(&dp.max_shared_memory_per_block, cudaDevAttrMaxSharedMemoryPerBlock, dp.device_id));
    cudaErr(cudaDeviceGetAttribute(&dp.max_shared_memory_per_SM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dp.device_id));
    cudaErr(cudaDeviceGetAttribute(&dp.max_registers_per_block, cudaDevAttrMaxRegistersPerBlock, dp.device_id));
    cudaErr(cudaDeviceGetAttribute(&dp.max_persisting_L2_cache_size, cudaDevAttrMaxPersistingL2CacheSize, dp.device_id));
    cudaErr(cudaDeviceGetAttribute(&dp.L2_cache_size, cudaDevAttrL2CacheSize, dp.device_id));
    cudaErr(cudaDeviceGetAttribute(&dp.accessPolicyMaxWindowSize, cudaDevAttrMaxAccessPolicyWindowSize, dp.device_id));
}

void CheckSharedMemory(int& memory_requested, DeviceProps& dp) {
    // Depends on GetCudaDeviceProps having been called, which should be happening in the constructor.
    // Throw an error if requesting more than allowed, otherwise, we'll set to requested and let the rest be L1 Cache.
    MyFFTRunTimeAssertFalse(memory_requested > dp.max_shared_memory_per_SM, "The shared memory requested is greater than permitted for this arch.");
    // if (memory_requested > dp.max_shared_memory_per_block) { memory_requested = dp.max_shared_memory_per_block; }
}

void CheckSharedMemory(unsigned int& memory_requested, DeviceProps& dp) {
    // Depends on GetCudaDeviceProps having been called, which should be happening in the constructor.
    // Throw an error if requesting more than allowed, otherwise, we'll set to requested and let the rest be L1 Cache.
    MyFFTRunTimeAssertFalse(memory_requested > dp.max_shared_memory_per_SM, "The shared memory requested is greater than permitted for this arch.");
    // if (memory_requested > dp.max_shared_memory_per_block) { memory_requested = dp.max_shared_memory_per_block; }
}

using namespace FastFFT::KernelFunction;
// my_functor, IKF_t

// 2d explicit instantiations

// TODO: Pass in functor types
// TODO: Take another look at the explicit NOOP vs nullptr and determine if it is really needed
#define INSTANTIATE(COMPUTE_BASE_TYPE, INPUT_TYPE, OTHER_IMAGE_TYPE, RANK)                                                                                                                       \
    template class FourierTransformer<COMPUTE_BASE_TYPE, INPUT_TYPE, OTHER_IMAGE_TYPE, RANK>;                                                                                                    \
                                                                                                                                                                                                 \
    template void FourierTransformer<COMPUTE_BASE_TYPE, INPUT_TYPE, OTHER_IMAGE_TYPE, RANK>::FwdFFT<std::nullptr_t,                                                                              \
                                                                                                    std::nullptr_t>(INPUT_TYPE*, INPUT_TYPE*, std::nullptr_t, std::nullptr_t);                   \
                                                                                                                                                                                                 \
    template void FourierTransformer<COMPUTE_BASE_TYPE, INPUT_TYPE, OTHER_IMAGE_TYPE, RANK>::InvFFT<std::nullptr_t,                                                                              \
                                                                                                    std::nullptr_t>(INPUT_TYPE*, INPUT_TYPE*, std::nullptr_t, std::nullptr_t);                   \
                                                                                                                                                                                                 \
    template void FourierTransformer<COMPUTE_BASE_TYPE, INPUT_TYPE, OTHER_IMAGE_TYPE, RANK>::FwdFFT<my_functor<float, 0, IKF_t::NOOP>,                                                           \
                                                                                                    my_functor<float, 0, IKF_t::NOOP>>(INPUT_TYPE*,                                              \
                                                                                                                                       INPUT_TYPE*,                                              \
                                                                                                                                       my_functor<float, 0, IKF_t::NOOP>,                        \
                                                                                                                                       my_functor<float, 0, IKF_t::NOOP>);                       \
                                                                                                                                                                                                 \
    template void FourierTransformer<COMPUTE_BASE_TYPE, INPUT_TYPE, OTHER_IMAGE_TYPE, RANK>::InvFFT<my_functor<float, 0, IKF_t::NOOP>,                                                           \
                                                                                                    my_functor<float, 0, IKF_t::NOOP>>(INPUT_TYPE*,                                              \
                                                                                                                                       INPUT_TYPE*,                                              \
                                                                                                                                       my_functor<float, 0, IKF_t::NOOP>,                        \
                                                                                                                                       my_functor<float, 0, IKF_t::NOOP>);                       \
                                                                                                                                                                                                 \
    template void FourierTransformer<COMPUTE_BASE_TYPE, INPUT_TYPE, OTHER_IMAGE_TYPE, RANK>::FwdImageInvFFT<my_functor<float, 0, IKF_t::NOOP>,                                                   \
                                                                                                            my_functor<float, 4, IKF_t::CONJ_MUL>,                                               \
                                                                                                            my_functor<float, 0, IKF_t::NOOP>>(INPUT_TYPE*,                                      \
                                                                                                                                               OTHER_IMAGE_TYPE*,                                \
                                                                                                                                               INPUT_TYPE*,                                      \
                                                                                                                                               my_functor<float, 0, IKF_t::NOOP>,                \
                                                                                                                                               my_functor<float, 4, IKF_t::CONJ_MUL>,            \
                                                                                                                                               my_functor<float, 0, IKF_t::NOOP>);               \
    template void FourierTransformer<COMPUTE_BASE_TYPE, INPUT_TYPE, OTHER_IMAGE_TYPE, RANK>::FwdImageInvFFT<my_functor<float, 2, IKF_t::SCALE>,                                                  \
                                                                                                            my_functor<float, 4, IKF_t::CONJ_MUL>,                                               \
                                                                                                            my_functor<float, 0, IKF_t::NOOP>>(INPUT_TYPE*,                                      \
                                                                                                                                               OTHER_IMAGE_TYPE*,                                \
                                                                                                                                               INPUT_TYPE*,                                      \
                                                                                                                                               my_functor<float, 2, IKF_t::SCALE>,               \
                                                                                                                                               my_functor<float, 4, IKF_t::CONJ_MUL>,            \
                                                                                                                                               my_functor<float, 0, IKF_t::NOOP>);               \
    template void FourierTransformer<COMPUTE_BASE_TYPE, INPUT_TYPE, OTHER_IMAGE_TYPE, RANK>::FwdImageInvFFT<my_functor<float, 0, IKF_t::NOOP>,                                                   \
                                                                                                            my_functor<float, 4, IKF_t::CONJ_MUL_THEN_SCALE>,                                    \
                                                                                                            my_functor<float, 0, IKF_t::NOOP>>(INPUT_TYPE*,                                      \
                                                                                                                                               OTHER_IMAGE_TYPE*,                                \
                                                                                                                                               INPUT_TYPE*,                                      \
                                                                                                                                               my_functor<float, 0, IKF_t::NOOP>,                \
                                                                                                                                               my_functor<float, 4, IKF_t::CONJ_MUL_THEN_SCALE>, \
                                                                                                                                               my_functor<float, 0, IKF_t::NOOP>);

INSTANTIATE(float, float, float2, 2);
INSTANTIATE(float, __half, float2, 2);
INSTANTIATE(float, float, __half2, 2);
INSTANTIATE(float, __half, __half2, 2);
#ifdef FastFFT_3d_instantiation
INSTANTIATE(float, float, float2, 3);
INSTANTIATE(float, __half, __half2, 3);
#endif
#undef INSTANTIATE

} // namespace FastFFT
