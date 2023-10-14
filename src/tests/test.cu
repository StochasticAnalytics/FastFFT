#include "tests.h"

// Define an enum for size change type to indecate a decrease, no change or increase

// The Fourier transform of a constant should be a unit impulse, and on back fft, without normalization, it should be a constant * N.
// It is assumed the input/output have the same dimension (i.e. no padding)

template <int Rank>
bool random_image_test(std::vector<int> size, bool do_3d = false) {

    bool              all_passed = true;
    std::vector<bool> init_passed(size.size( ), true);
    std::vector<bool> FFTW_passed(size.size( ), true);
    std::vector<bool> FastFFT_forward_passed(size.size( ), true);
    std::vector<bool> FastFFT_roundTrip_passed(size.size( ), true);

    for ( int n = 0; n < size.size( ); n++ ) {

        short4 input_size;
        short4 output_size;
        long   full_sum = long(size[n]);
        if ( do_3d ) {
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
        Image<float, float2> host_copy(output_size);
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
        host_copy.real_memory_allocated   = FT.ReturnInvOutputMemorySize( );

        // On the device, we will always allocate enough memory for the larger of input/output including the buffer array.
        // Minmize the number of calls to malloc which are slow and can lead to fragmentation.
        device_output.real_memory_allocated = std::max(host_input.real_memory_allocated, host_output.real_memory_allocated);

        // In your own programs, you will be handling this memory allocation yourself. We'll just make something here.
        // I think fftwf_malloc may potentially create a different alignment than new/delete, but kinda doubt it. For cisTEM consistency...
        bool set_fftw_plan = true;
        host_input.Allocate(set_fftw_plan);
        host_output.Allocate(set_fftw_plan);
        host_copy.Allocate(set_fftw_plan);

        // Set our input host memory to a constant. Then FFT[0] = host_input_memory_allocated
        FT.SetToRandom(host_output.real_values, host_output.real_memory_allocated, 0.0f, 1.0f);

        // Now we want to associate the host memory with the device memory. The method here asks if the host pointer is pinned (in page locked memory) which
        // ensures faster transfer. If false, it will be pinned for you.
        FT.SetInputPointer(host_output.real_values, false);

        // This copies the host memory into the device global memory. If needed, it will also allocate the device memory first.
        FT.CopyHostToDevice(host_output.real_values);

#if FFT_DEBUG_STAGE > 0
        host_output.FwdFFT( );
#endif

        for ( long i = 0; i < host_output.real_memory_allocated / 2; i++ ) {
            host_copy.complex_values[i] = host_output.complex_values[i];
        }

        // This method will call the regular FFT kernels given the input/output dimensions are equal when the class is instantiated.
        bool swap_real_space_quadrants = false;
        FT.FwdFFT( );

        // in buffer, do not deallocate, do not unpin memory
        FT.CopyDeviceToHostAndSynchronize(host_output.real_values, false);
        bool test_passed = true;

#if FFT_DEBUG_STAGE == 0
        PrintArray(host_output.real_values, dims_out.x, dims_in.y, dims_in.z, dims_out.w);
        PrintArray(host_copy.real_values, dims_out.x, dims_in.y, dims_in.z, dims_out.w);
        MyTestPrintAndExit("stage 0 ");
#elif FFT_DEBUG_STAGE == 1
        std::cout << " For random_image_test partial transforms aren't supported, b/c we need to compare to the cpu output." << std::endl;
        MyTestPrintAndExit("stage 1 ");
#elif FFT_DEBUG_STAGE == 2
        std::cout << " For random_image_test partial transforms aren't supported, b/c we need to compare to the cpu output." << std::endl;
        MyTestPrintAndExit("stage 2 ");
#elif FFT_DEBUG_STAGE == 3
        PrintArray(host_output.complex_values, dims_in.y, dims_out.w, dims_out.z);
        PrintArray(host_copy.complex_values, dims_in.y, dims_out.w, dims_out.z);

        //   std::cout << "Distance between FastFFT and CPU: " << distance << std::endl;
        MyTestPrintAndExit("stage 3 ");
#endif

        double distance = 0.0;
        for ( long index = 0; index < host_output.real_memory_allocated / 2; index++ ) {
            distance += sqrt((host_output.complex_values[index].x - host_copy.complex_values[index].x) * (host_output.complex_values[index].x - host_copy.complex_values[index].x) +
                             (host_output.complex_values[index].y - host_copy.complex_values[index].y) * (host_output.complex_values[index].y - host_copy.complex_values[index].y));
        }
        distance /= (host_output.real_memory_allocated / 2);

        std::cout << "Distance between FastFFT and CPU: " << distance << std::endl;
        exit(0);
        if ( test_passed == false ) {
            all_passed                = false;
            FastFFT_forward_passed[n] = false;
        }
        // MyFFTDebugAssertTestTrue( test_passed, "FastFFT unit impulse forward FFT");
        FT.SetToConstant(host_input.real_values, host_input.real_memory_allocated, 2.0f);

        FT.InvFFT( );
        FT.CopyDeviceToHostAndSynchronize(host_output.real_values, true);

#if FFT_DEBUG_STAGE == 4
        PrintArray(host_output.complex_values, dims_out.y, dims_out.w, dims_out.z);
        MyTestPrintAndExit("stage 4 ");
#elif FFT_DEBUG_STAGE == 5
        PrintArray(host_output.complex_values, dims_out.y, dims_out.w, dims_out.z);
        MyTestPrintAndExit("stage 5 ");
#elif FFT_DEBUG_STAGE == 6
        if ( do_3d ) {
            std::cout << " in 3d print inv " << dims_out.w << "w" << std::endl;
            PrintArray(host_output.complex_values, dims_out.w, dims_out.y, dims_out.z);
        }
        else
            PrintArray(host_output.complex_values, dims_out.y, dims_out.w, dims_out.z);
        MyTestPrintAndExit("stage 6 ");
#elif FFT_DEBUG_STAGE == 7
        PrintArray(host_output.real_values, dims_out.x, dims_out.y, dims_out.z, dims_out.w);
        MyTestPrintAndExit("stage 7 ");
#elif FFT_DEBUG_STAGE > 7
        // No debug, keep going
#else
        MyTestPrintAndExit(" This block is only valid for FFT_DEBUG_STAGE == 4, 5, 7 ");
#endif

        // Assuming the outputs are always even dimensions, padding_jump_val is always 2.
        sum = host_output.ReturnSumOfReal(host_output.real_values, dims_out, true);

        if ( sum != full_sum ) {
            all_passed                  = false;
            FastFFT_roundTrip_passed[n] = false;
        }
        MyFFTDebugAssertTestTrue(sum == full_sum, "FastFFT constant image round trip for size " + std::to_string(dims_in.x));
    } // loop over sizes

    if ( all_passed ) {
        if ( do_3d )
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

template <int Rank>
void run_oned(std::vector<int> size) {

    // Override the size to be one dimensional in x
    std::cout << "Running one-dimensional tests\n"
              << std::endl;

    for ( int n : size ) {
        short4 input_size  = make_short4(n, 1, 1, 0);
        short4 output_size = make_short4(n, 1, 1, 0);

        Image<float, float2>  FT_input(input_size);
        Image<float, float2>  FT_output(output_size);
        Image<float2, float2> FT_input_complex(input_size);
        Image<float2, float2> FT_output_complex(output_size);

        // We just make one instance of the FourierTransformer class, with calc type float.
        // For the time being input and output are also float. TODO calc optionally either fp16 or nv_bloat16, TODO inputs at lower precision for bandwidth improvement.
        FastFFT::FourierTransformer<float, float, float, Rank> FT;
        FastFFT::FourierTransformer<float, float2, float2>     FT_complex;

        // This is similar to creating an FFT/CUFFT plan, so set these up before doing anything on the GPU
        FT.SetForwardFFTPlan(input_size.x, input_size.y, input_size.z, output_size.x, output_size.y, output_size.z);
        FT.SetInverseFFTPlan(output_size.x, output_size.y, output_size.z, output_size.x, output_size.y, output_size.z);

        FT_complex.SetForwardFFTPlan(input_size.x, input_size.y, input_size.z, output_size.x, output_size.y, output_size.z);
        FT_complex.SetInverseFFTPlan(output_size.x, output_size.y, output_size.z, output_size.x, output_size.y, output_size.z);

        FT_input.real_memory_allocated  = FT.ReturnInputMemorySize( );
        FT_output.real_memory_allocated = FT.ReturnInvOutputMemorySize( );

        FT_input_complex.real_memory_allocated  = FT_complex.ReturnInputMemorySize( );
        FT_output_complex.real_memory_allocated = FT_complex.ReturnInvOutputMemorySize( );

        bool set_fftw_plan = true;
        FT_input.Allocate(set_fftw_plan);
        FT_output.Allocate(set_fftw_plan);

        FT_input_complex.Allocate( );
        FT_output_complex.Allocate( );

        // Now we want to associate the host memory with the device memory. The method here asks if the host pointer is pinned (in page locked memory) which
        // ensures faster transfer. If false, it will be pinned for you.
        FT.SetInputPointer(FT_input.real_values, false);
        FT_complex.SetInputPointer(FT_input_complex.complex_values, false);

        FT.SetToConstant(FT_input.real_values, FT_input.real_memory_allocated, 1.f);

        // Set a unit impulse at the center of the input array.
        // FT.SetToConstant(FT_input.real_values, FT_input.real_memory_allocated, 1.0f);
        float2 const_val = make_float2(1.0f, 0.0f);
        FT_complex.SetToConstant<float2>(FT_input_complex.complex_values, FT_input.real_memory_allocated, const_val);
        for ( int i = 0; i < 10; i++ ) {
            std::cout << FT_input_complex.complex_values[i].x << "," << FT_input_complex.complex_values[i].y << std::endl;
        }

        FT.CopyHostToDevice(FT_input.real_values);
        FT_complex.CopyHostToDevice(FT_input_complex.complex_values);
        cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

        // Set the outputs to a clearly wrong answer.
        FT.SetToConstant(FT_output.real_values, FT_input.real_memory_allocated, 2.0f);
        const_val = make_float2(2.0f, 2.0f);
        FT_complex.SetToConstant<float2>(FT_output_complex.complex_values, FT_output.real_memory_allocated, const_val);

        FT_input.FwdFFT( );

        bool transpose_output          = false;
        bool swap_real_space_quadrants = false;
        FT.FwdFFT( );
        FT_complex.FwdFFT( );

        FT.CopyDeviceToHostAndSynchronize(FT_output.real_values, false, false);
        FT_complex.CopyDeviceToHostAndSynchronize(FT_output_complex.real_values, false, false);

        FT_input.InvFFT( );

        for ( int i = 0; i < 5; ++i ) {
            std::cout << "FFTW inv " << FT_input.real_values[i] << std::endl;
        }
        std::cout << std::endl;

        FT.InvFFT( );
        FT_complex.InvFFT( );
        FT.CopyDeviceToHostAndSynchronize(FT_output.real_values, true);
        FT_complex.CopyDeviceToHostAndSynchronize(FT_output_complex.real_values, true);

        for ( int i = 0; i < 10; i++ ) {
            std::cout << "Ft inv " << FT_output.real_values[i] << std::endl;
        }
        for ( int i = 0; i < 10; i++ ) {
            std::cout << "Ft complex inv " << FT_output_complex.real_values[i].x << "," << FT_output_complex.real_values[i].y << std::endl;
        }
    }
}

int main(int argc, char** argv) {

    using SCT = FastFFT::SizeChangeType::Enum;

    if ( argc != 2 ) {
        return 1;
    }
    std::string test_name = argv[1];
    std::printf("Standard is %li\n\n", __cplusplus);

    // Input size vectors to be tested.
    std::vector<int> test_size           = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    std::vector<int> test_size_rectangle = {64, 128, 256, 512, 1024, 2048, 4096};
    std::vector<int> test_size_3d        = {32, 64, 128, 256, 512};
    // std::vector<int> test_size_3d ={512};

    // The launch parameters fail for 4096 -> < 64 for r2c_decrease, not sure if it is the elements_per_thread or something else.
    // For now, just over-ride these small sizes
    std::vector<int> test_size_for_decrease = {64, 128, 256, 512, 1024, 2048, 4096};

    // If we get here, all tests passed.
    return 0;
};
