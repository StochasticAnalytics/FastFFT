
#include "tests.h"

template <int Rank>
bool unit_impulse_test(std::vector<int> size, bool do_increase_size) {

    bool              all_passed = true;
    std::vector<bool> init_passed(size.size( ), true);
    std::vector<bool> FFTW_passed(size.size( ), true);
    std::vector<bool> FastFFT_forward_passed(size.size( ), true);
    std::vector<bool> FastFFT_roundTrip_passed(size.size( ), true);

    short4 input_size;
    short4 output_size;
    for ( int iSize = 0; iSize < size.size( ) - 1; iSize++ ) {
        int oSize = iSize + 1;
        while ( oSize < size.size( ) ) {

            // std::cout << std::endl << "Testing padding from  " << size[iSize] << " to " << size[oSize] << std::endl;
            if ( do_increase_size ) {
                if ( Rank == 3 ) {
                    input_size  = make_short4(size[iSize], size[iSize], size[iSize], 0);
                    output_size = make_short4(size[oSize], size[oSize], size[oSize], 0);
                }
                else {
                    input_size  = make_short4(size[iSize], size[iSize], 1, 0);
                    output_size = make_short4(size[oSize], size[oSize], 1, 0);
                }
            }
            else {
                if ( Rank == 3 ) {
                    output_size = make_short4(size[iSize], size[iSize], size[iSize], 0);
                    input_size  = make_short4(size[oSize], size[oSize], size[oSize], 0);
                }
                else {
                    output_size = make_short4(size[iSize], size[iSize], 1, 0);
                    input_size  = make_short4(size[oSize], size[oSize], 1, 0);
                }
            }

            float sum;

            Image<float, float2> host_input(input_size);
            Image<float, float2> host_output(output_size);
            Image<float, float2> device_output(output_size);

            // We just make one instance of the FourierTransformer class, with calc type float.
            // For the time being input and output are also float. TODO calc optionally either fp16 or nv_bloat16, TODO inputs at lower precision for bandwidth improvement.
            FastFFT::FourierTransformer<float, float, float, Rank> FT;
            // This is similar to creating an FFT/CUFFT plan, so set these up before doing anything on the GPU
            FT.SetForwardFFTPlan(input_size.x, input_size.y, input_size.z, output_size.x, output_size.y, output_size.z, true, false);
            FT.SetInverseFFTPlan(output_size.x, output_size.y, output_size.z, output_size.x, output_size.y, output_size.z, true);

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

            // Now we want to associate the host memory with the device memory. The method here asks if the host pointer is pinned (in page locked memory) which
            // ensures faster transfer. If false, it will be pinned for you.
            FT.SetInputPointer(host_input.real_values, false);

            // Set a unit impulse at the center of the input array.
            FT.SetToConstant(host_input.real_values, host_input.real_memory_allocated, 0.0f);
            FT.SetToConstant(host_output.real_values, host_output.real_memory_allocated, 0.0f);

            sum = host_output.ReturnSumOfReal(host_output.real_values, dims_out);
            // host_input.real_values[ dims_in.y/2 * (dims_in.x+host_input.padding_jump_value) + dims_in.x/2] = 1.0f;
            // short4 wanted_center = make_short4(0,0,0,0);
            // ClipInto(host_input.real_values, host_output.real_values, dims_in ,  dims_out,  wanted_center, 0.f);

            // FT.SetToConstant(host_input.real_values, host_input.real_memory_allocated, 0.0f);
            host_input.real_values[0]  = 1.0f;
            host_output.real_values[0] = 1.0f;

            sum = host_output.ReturnSumOfReal(host_output.real_values, dims_out);
            if ( sum != 1 ) {
                all_passed         = false;
                init_passed[iSize] = false;
            }

            // MyFFTDebugAssertTestTrue( sum == 1,"Unit impulse Init ");

            // This copies the host memory into the device global memory. If needed, it will also allocate the device memory first.
            FT.CopyHostToDevice( );

            host_output.FwdFFT( );

            host_output.fftw_epsilon = host_output.ReturnSumOfComplexAmplitudes(host_output.complex_values, host_output.real_memory_allocated / 2);
            // std::cout << "host " << host_output.fftw_epsilon << " " << host_output.real_memory_allocated<< std::endl;

            host_output.fftw_epsilon -= (host_output.real_memory_allocated / 2);
            if ( std::abs(host_output.fftw_epsilon) > 1e-8 ) {
                all_passed         = false;
                FFTW_passed[iSize] = false;
            }

            // MyFFTDebugAssertTestTrue( std::abs(host_output.fftw_epsilon) < 1e-8 , "FFTW unit impulse forward FFT");

            // Just to make sure we don't get a false positive, set the host memory to some undesired value.
            FT.SetToConstant(host_output.real_values, host_output.real_memory_allocated, 2.0f);

            // This method will call the regular FFT kernels given the input/output dimensions are equal when the class is instantiated.
            bool swap_real_space_quadrants = true;

            FT.FwdFFT( );

            if ( do_increase_size ) {
                FT.CopyDeviceToHost(host_output.real_values, false, false);

#if FFT_DEBUG_STAGE == 0
                PrintArray(host_output.real_values, dims_in.x, dims_in.y, dims_in.z, dims_in.w);
                MyTestPrintAndExit("stage 0 ");
#elif FFT_DEBUG_STAGE == 1
                // If we are doing a fwd increase, the data will have only been expanded along the (transposed) X dimension at this point
                // So the (apparent) X is dims_in.y not dims_out.y
                if ( Rank == 3 ) {
                    std::cout << " in 3d print " << std::endl;
                    PrintArray(host_output.complex_values, dims_in.z, dims_in.y, dims_out.w);
                }
                else
                    PrintArray(host_output.complex_values, dims_in.y, dims_in.z, dims_out.w);

                MyTestPrintAndExit("stage 1 ");
#elif FFT_DEBUG_STAGE == 2
                // If we are doing a fwd increase, the data will have only been expanded along the (transposed) X dimension at this point
                // So the (apparent) X is dims_in.y not dims_out.y
                PrintArray(host_output.complex_values, dims_in.y, dims_out.z, dims_out.w);
                MyTestPrintAndExit("stage 2 ");
#elif FFT_DEBUG_STAGE == 3
                // Now the array is fully expanded to dims_out, but still transposed
                PrintArray(host_output.complex_values, dims_out.y, dims_out.z, dims_out.w);
                MyTestPrintAndExit("stage 3 ");
#endif
                sum = host_output.ReturnSumOfComplexAmplitudes(host_output.complex_values, host_output.real_memory_allocated / 2);
            }
            else {
                FT.CopyDeviceToHost(false, false, FT.ReturnInputMemorySize( ));
#if FFT_DEBUG_STAGE == 0
                PrintArray(host_input.real_values, dims_in.x, dims_in.y, dims_in.z, dims_in.w);
                MyTestPrintAndExit("stage 0 ");
#elif FFT_DEBUG_STAGE == 1
                // If we are doing a fwd increase, the data will have only been expanded along the (transposed) X dimension at this point
                // So the (apparent) X is dims_in.y not dims_out.y
                PrintArray(host_input.complex_values, dims_in.y, dims_in.z, dims_out.w);
                MyTestPrintAndExit("stage 1 ");
#elif FFT_DEBUG_STAGE == 2
                // If we are doing a fwd increase, the data will have only been expanded along the (transposed) X dimension at this point
                // So the (apparent) X is dims_in.y not dims_out.y
                PrintArray(host_input.complex_values, dims_in.y, dims_out.z, dims_out.w);
                MyTestPrintAndExit("stage 2 ");
#elif FFT_DEBUG_STAGE == 3
                // Now the array is fully expanded to dims_out, but still transposed
                PrintArray(host_input.complex_values, dims_out.y, dims_out.z, dims_out.w);
                MyTestPrintAndExit("stage 3 ");
#endif
                sum = host_input.ReturnSumOfComplexAmplitudes(host_input.complex_values, host_input.real_memory_allocated / 2);
            }

            sum -= (host_output.real_memory_allocated / 2);

            // std::cout << "sum " << sum << std::endl;
            // std::cout << "FFT Unit Impulse Forward FFT: " << sum <<  " epsilon " << host_output.fftw_epsilon << std::endl;
            // std::cout << "epsilon " << abs(sum - host_output.fftw_epsilon) << std::endl;
            if ( abs(sum) > 1e-8 ) {
                all_passed                    = false;
                FastFFT_forward_passed[iSize] = false;
            }

            // MyFFTDebugAssertTestTrue( abs(sum - host_output.fftw_epsilon) < 1e-8, "FastFFT unit impulse forward FFT");
            FT.SetToConstant(host_output.real_values, host_output.real_memory_allocated, 2.0f);

            FT.InvFFT( );
            FT.CopyDeviceToHost(host_output.real_values, true, true);

#if FFT_DEBUG_STAGE == 5
            PrintArray(host_output.complex_values, dims_out.y, dims_out.z, dims_out.w);
            MyTestPrintAndExit("stage 5 ");
#endif
#if FFT_DEBUG_STAGE == 6
            PrintArray(host_output.complex_values, dims_out.y, dims_out.z, dims_out.w);
            MyTestPrintAndExit("stage 6 ");
#elif FFT_DEBUG_STAGE == 7
            PrintArray(host_output.real_values, dims_out.x, dims_out.y, dims_out.z, dims_out.w);
            MyTestPrintAndExit("stage 7 ");
#elif FFT_DEBUG_STAGE > 7
            // No debug, keep going
#else
            MyTestPrintAndExit(" This block is only valid for FFT_DEBUG_STAGE == 3 || 4 ");
#endif

            sum = host_output.ReturnSumOfReal(host_output.real_values, dims_out);
            if ( sum != dims_out.x * dims_out.y * dims_out.z ) {
                all_passed                      = false;
                FastFFT_roundTrip_passed[iSize] = false;
            }

            // std::cout << "size in/out " << dims_in.x << ", " << dims_out.x << std::endl;
            // MyFFTDebugAssertTestTrue( sum == dims_out.x*dims_out.y*dims_out.z,"FastFFT unit impulse round trip FFT");

            oSize++;
        } // while loop over pad to size
    } // for loop over pad from size

    if ( all_passed ) {
        if ( ! do_increase_size )
            std::cout << "    All rank " << Rank << " size_decrease unit impulse tests passed!" << std::endl;
        else
            std::cout << "    All rank " << Rank << " size_increase unit impulse tests passed!" << std::endl;
    }
    else {
        for ( int n = 0; n < size.size( ); n++ ) {
            if ( ! init_passed[n] )
                std::cout << "    Initialization failed for size " << size[n] << " rank " << Rank << std::endl;
            if ( ! FFTW_passed[n] )
                std::cout << "    FFTW failed for size " << size[n] << " rank " << Rank << std::endl;
            if ( ! FastFFT_forward_passed[n] )
                std::cout << "    FastFFT failed for forward transform size " << size[n] << " rank " << Rank << std::endl;
            if ( ! FastFFT_roundTrip_passed[n] )
                std::cout << "    FastFFT failed for roundtrip transform size " << size[n] << " rank " << Rank << std::endl;
        }
    }
    return all_passed;
}

int main(int argc, char** argv) {

    std::string test_name;
    // Default to running all tests
    bool run_2d_unit_tests = false;
    bool run_3d_unit_tests = false;

    const std::string_view text_line = "unit impulse";
    FastFFT::CheckInputArgs(argc, argv, text_line, run_2d_unit_tests, run_3d_unit_tests);

    constexpr bool do_increase_size = true;
    // TODO: size decrease
    if ( run_2d_unit_tests ) {
        if ( ! unit_impulse_test<2>(FastFFT::test_size, do_increase_size) )
            return 1;
    }

    if ( run_3d_unit_tests ) {
        // FIXME: tests are failing for 3d
        if ( ! unit_impulse_test<3>(FastFFT::test_size_3d, do_increase_size) )
            return 1;
        // if (! unit_impulse_test(test_size_3d, true, true)) return 1;
    }

    return 0;
};