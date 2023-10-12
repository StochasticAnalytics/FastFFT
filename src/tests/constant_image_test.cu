
#include "tests.h"

template <int Rank>
bool const_image_test(std::vector<int>& size) {

    bool              all_passed = true;
    std::vector<bool> init_passed(size.size( ), true);
    std::vector<bool> FFTW_passed(size.size( ), true);
    std::vector<bool> FastFFT_forward_passed(size.size( ), true);
    std::vector<bool> FastFFT_roundTrip_passed(size.size( ), true);

    for ( int n = 0; n < size.size( ); n++ ) {

        short4 input_size;
        short4 output_size;
        long   full_sum = long(size[n]);
        if ( Rank == 3 ) {
            input_size  = make_short4(size[n], size[n], size[n], 0);
            output_size = make_short4(size[n], size[n], size[n], 0);
            full_sum    = full_sum * full_sum * full_sum * full_sum * full_sum * full_sum;
        }
        else {
            input_size  = make_short4(size[n], size[n], 1, 0);
            output_size = make_short4(size[n], size[n], 1, 0);
            full_sum    = full_sum * full_sum * full_sum * full_sum;
        }

        float sum;

        Image<float, float2> host_input(input_size);
        Image<float, float2> host_output(output_size);
        Image<float, float2> device_output(output_size);

        // Pointers to the arrays on the host -- maybe make this a struct of some sort? I'm sure there is a parallel in cuda, look into cuarray/texture code

        // We just make one instance of the FourierTransformer class, with calc type float.
        // For the time being input and output are also float. TODO calc optionally either fp16 or nv_bloat16, TODO inputs at lower precision for bandwidth improvement.
        FastFFT::FourierTransformer<float, float, float, Rank> FT;

        // This is similar to creating an FFT/CUFFT plan, so set these up before doing anything on the GPU
        FT.SetForwardFFTPlan(input_size.x, input_size.y, input_size.z, output_size.x, output_size.y, output_size.z);
        FT.SetInverseFFTPlan(output_size.x, output_size.y, output_size.z, output_size.x, output_size.y, output_size.z);

        // The padding (dims.w) is calculated based on the setup
        short4 dims_in  = FT.ReturnFwdInputDimensions( );
        short4 dims_out = FT.ReturnFwdOutputDimensions( );

        // Determine how much memory we need, working with FFTW/CUDA style in place transform padding.
        // Note: there is no reason we really need this, because the xforms will always be out of place.
        //       For now, this is just in place because all memory in cisTEM is allocated accordingly.
        host_input.real_memory_allocated  = FT.ReturnInputMemorySize( );
        host_output.real_memory_allocated = FT.ReturnInvOutputMemorySize( );

        // On the device, we will always allocate enough memory for the larger of input/output including the buffer array.
        // Minmize the number of calls to malloc which are slow and can lead to fragmentation.
        device_output.real_memory_allocated = std::max(host_input.real_memory_allocated, host_output.real_memory_allocated);

        // In your own programs, you will be handling this memory allocation yourself. We'll just make something here.
        // I think fftwf_malloc may potentially create a different alignment than new/delete, but kinda doubt it. For cisTEM consistency...
        bool set_fftw_plan = true;
        host_input.Allocate(set_fftw_plan);
        host_output.Allocate(set_fftw_plan);

        // Set our input host memory to a constant. Then FFT[0] = host_input_memory_allocated
        FT.SetToConstant(host_output.real_values, host_output.real_memory_allocated, 1.0f);

        // Now we want to associate the host memory with the device memory. The method here asks if the host pointer is pinned (in page locked memory) which
        // ensures faster transfer. If false, it will be pinned for you.
        FT.SetInputPointer(host_output.real_values, false);
        sum = host_output.ReturnSumOfReal(host_output.real_values, dims_out);

        if ( sum != long(dims_in.x) * long(dims_in.y) * long(dims_in.z) ) {
            all_passed     = false;
            init_passed[n] = false;
        }

        // MyFFTDebugAssertTestTrue( sum == dims_out.x*dims_out.y*dims_out.z,"Unit impulse Init ");

        // This copies the host memory into the device global memory. If needed, it will also allocate the device memory first.
        FT.CopyHostToDevice(host_output.real_values);

        host_output.FwdFFT( );

        bool test_passed = true;
        for ( long index = 1; index < host_output.real_memory_allocated / 2; index++ ) {
            if ( host_output.complex_values[index].x != 0.0f && host_output.complex_values[index].y != 0.0f ) {
                std::cout << host_output.complex_values[index].x << " " << host_output.complex_values[index].y << " " << std::endl;
                test_passed = false;
            }
        }
        if ( host_output.complex_values[0].x != (float)dims_out.x * (float)dims_out.y * (float)dims_out.z )
            test_passed = false;

        if ( test_passed == false ) {
            all_passed     = false;
            FFTW_passed[n] = false;
        }
        // MyFFTDebugAssertTestTrue( test_passed, "FFTW unit impulse forward FFT");

        // Just to make sure we don't get a false positive, set the host memory to some undesired value.
        FT.SetToConstant(host_output.real_values, host_output.real_memory_allocated, 2.0f);

        // This method will call the regular FFT kernels given the input/output dimensions are equal when the class is instantiated.
        // bool swap_real_space_quadrants = false;
        FT.FwdFFT( );

        // in buffer, do not deallocate, do not unpin memory
        FT.CopyDeviceToHostAndSynchronize(host_output.real_values, false);
        test_passed = true;
        // FIXME: centralized test conditions
        for ( long index = 1; index < host_output.real_memory_allocated / 2; index++ ) {
            if ( host_output.complex_values[index].x != 0.0f && host_output.complex_values[index].y != 0.0f ) {
                test_passed = false;
            } // std::cout << host_output.complex_values[index].x  << " " << host_output.complex_values[index].y << " " );}
        }
        if ( host_output.complex_values[0].x != (float)dims_out.x * (float)dims_out.y * (float)dims_out.z )
            test_passed = false;

        bool continue_debugging;
        // We don't want this to break compilation of other tests, so only check at runtime.
        if constexpr ( FFT_DEBUG_STAGE < 5 ) {
            continue_debugging = debug_partial_fft<FFT_DEBUG_STAGE, Rank>(host_output, dims_in, dims_out, dims_in, dims_out, __LINE__);
        }

        if ( test_passed == false ) {
            all_passed                = false;
            FastFFT_forward_passed[n] = false;
        }
        // MyFFTDebugAssertTestTrue( test_passed, "FastFFT unit impulse forward FFT");
        FT.SetToConstant(host_input.real_values, host_input.real_memory_allocated, 2.0f);

        FT.InvFFT( );
        FT.CopyDeviceToHostAndSynchronize(host_output.real_values, true);

        if constexpr ( FFT_DEBUG_STAGE > 4 ) {
            continue_debugging = debug_partial_fft<FFT_DEBUG_STAGE, Rank>(host_output, dims_in, dims_out, dims_in, dims_out, __LINE__);
        }

        // Assuming the outputs are always even dimensions, padding_jump_val is always 2.
        sum = host_output.ReturnSumOfReal(host_output.real_values, dims_out, true);

        if ( sum != full_sum ) {
            all_passed                  = false;
            FastFFT_roundTrip_passed[n] = false;
        }
        MyFFTDebugAssertTestTrue(sum == full_sum, "FastFFT constant image round trip for size " + std::to_string(dims_in.x));
    } // loop over sizes

    if ( all_passed ) {
        if ( Rank == 3 )
            std::cout << "    All 3d const_image tests passed!" << std::endl;
        else
            std::cout << "    All 2d const_image tests passed!" << std::endl;
    }
    else {
        for ( int n = 0; n < size.size( ); n++ ) {
            if ( ! init_passed[n] )
                std::cout << "    Initialization failed for size " << size[n] << std::endl;
            if ( ! FFTW_passed[n] )
                std::cout << "    FFTW failed for size " << size[n] << std::endl;
            if ( ! FastFFT_forward_passed[n] )
                std::cout << "    FastFFT failed for forward transform size " << size[n] << std::endl;
            if ( ! FastFFT_roundTrip_passed[n] )
                std::cout << "    FastFFT failed for roundtrip transform size " << size[n] << std::endl;
        }
    }
    return all_passed;
}

int main(int argc, char** argv) {

    std::string test_name;
    // Default to running all tests
    bool run_2d_unit_tests = false;
    bool run_3d_unit_tests = false;

    const std::string_view text_line = "constant image";
    FastFFT::CheckInputArgs(argc, argv, text_line, run_2d_unit_tests, run_3d_unit_tests);

    if ( run_2d_unit_tests ) {
        if ( ! const_image_test<2>(FastFFT::test_size) )
            return 1;
    }

    if ( run_3d_unit_tests ) {
        if ( ! const_image_test<3>(FastFFT::test_size_3d) )
            return 1;
        // if (! unit_impulse_test(test_size_3d, true, true)) return 1;
    }

    return 0;
};