#include "Image.cu"
#include "FastFFT.cu"
#include <cufft.h>
#include <cufftXt.h>

// The Fourier transform of a constant should be a unit impulse, and on back fft, without normalization, it should be a constant * N.
// It is assumed the input/output have the same dimension (i.e. no padding)
void const_image_test(short4 input_size, short4 output_size)
{

  MyFFTRunTimeAssertTrue(input_size.x == output_size.x && input_size.y == output_size.y && input_size.z == output_size.z, "Input/output size mismatch");

  bool test_passed = true;
  long address = 0;
  float sum;
  float2 sum_complex;

  Image< float, float2 > host_input(input_size);
  Image< float, float2 > host_output(output_size);
  Image< float, float2 > device_output(output_size);


    // Pointers to the arrays on the host -- maybe make this a struct of some sort? I'm sure there is a parallel in cuda, look into cuarray/texture code

  // We just make one instance of the FourierTransformer class, with calc type float.
  // For the time being input and output are also float. TODO calc optionally either fp16 or nv_bloat16, TODO inputs at lower precision for bandwidth improvement.
  FastFFT::FourierTransformer FT(FastFFT::FourierTransformer::DataType::fp32);
  
  
  // Determine how much memory we need, working with FFTW/CUDA style in place transform padding.
  // Note: there is no reason we really need this, because the xforms will always be out of place. 
  //       For now, this is just in place because all memory in cisTEM is allocated accordingly.
  host_input.real_memory_allocated = FT.ReturnPaddedMemorySize(input_size);
  host_output.real_memory_allocated = FT.ReturnPaddedMemorySize(output_size);
  
  // On the device, we will always allocate enough memory for the larger of input/output including the buffer array.
  // Minmize the number of calls to malloc which are slow and can lead to fragmentation.
  device_output.real_memory_allocated = std::max(host_input.real_memory_allocated, host_output.real_memory_allocated);
  
  
  // In your own programs, you will be handling this memory allocation yourself. We'll just make something here.
  // I think fftwf_malloc may potentially create a different alignment than new/delete, but kinda doubt it. For cisTEM consistency...
  bool set_fftw_plan = true;
  host_input.Allocate(set_fftw_plan);
  host_output.Allocate(set_fftw_plan);

    
  // Set our input host memory to a constant. Then FFT[0] = host_input_memory_allocated
  FT.SetToConstant<float>(host_output.real_values, host_output.real_memory_allocated, 1.0f);

    
  // This is similar to creating an FFT/CUFFT plan, so set these up before doing anything on the GPU
  FT.SetInputDimensionsAndType(input_size.x,input_size.y,input_size.z,true, false,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);
  FT.SetOutputDimensionsAndType(output_size.x,output_size.y,output_size.z,true,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);
  
  // Now we want to associate the host memory with the device memory. The method here asks if the host pointer is pinned (in page locked memory) which
  // ensures faster transfer. If false, it will be pinned for you.
  FT.SetInputPointer(host_output.real_values, false);
  sum = ReturnSumOfReal(host_output.real_values, output_size);
  MyFFTDebugAssertTestTrue( sum == output_size.x*output_size.y*output_size.z,"Unit impulse Init ");
  
  // This copies the host memory into the device global memory. If needed, it will also allocate the device memory first.
  FT.CopyHostToDevice();
    
  host_output.FwdFFT();
  
  test_passed = true;
  for (long index = 1; index < host_output.real_memory_allocated/2; index++)
  {
    if (host_output.complex_values[index].x != 0.0f && host_output.complex_values[index].y != 0.0f) { std::cout << host_output.complex_values[index].x  << " " << host_output.complex_values[index].y << " " << std::endl; test_passed = false;}
  }
  if (host_output.complex_values[0].x != (float)output_size.x * (float)output_size.y * (float)output_size.z) test_passed = false;
  
  MyFFTDebugAssertTestTrue( test_passed, "FFTW unit impulse forward FFT");
  
  // Just to make sure we don't get a false positive, set the host memory to some undesired value.
  FT.SetToConstant<float>(host_output.real_values, host_output.real_memory_allocated, 2.0f);
  
  // This method will call the regular FFT kernels given the input/output dimensions are equal when the class is instantiated.
  bool swap_real_space_quadrants = false;
  FT.FwdFFT(swap_real_space_quadrants);
  
  // in buffer, do not deallocate, do not unpin memory
  FT.CopyDeviceToHost( false, false);
  test_passed = true;
  for (long index = 1; index < host_output.real_memory_allocated/2; index++)
  {
    if (host_output.complex_values[index].x != 0.0f && host_output.complex_values[index].y != 0.0f) test_passed = false;
  }
  if (host_output.complex_values[0].x != (float)output_size.x * (float)output_size.y * (float)output_size.z) test_passed = false;
  
  MyFFTDebugAssertTestTrue( test_passed, "FastFFT unit impulse forward FFT");
  FT.SetToConstant<float>(host_input.real_values, host_input.real_memory_allocated, 2.0f);
  

  FT.InvFFT();
  FT.CopyDeviceToHost( true, true);
  
  // Assuming the outputs are always even dimensions, padding_jump_val is always 2.
  sum = ReturnSumOfReal(host_output.real_values, output_size);
  
  MyFFTDebugAssertTestTrue( sum == powf(input_size.x*input_size.y*input_size.z,2),"FastFFT unit impulse round trip FFT");
  

}

void unit_impulse_test(short4 input_size, short4 output_size)
{

  bool test_passed = true;
  long address = 0;

  float sum;
  float2 sum_complex;

  Image< float, float2 > host_input(input_size);
  Image< float, float2 > host_output(output_size);
  Image< float, float2 > device_output(output_size);
  

  // We just make one instance of the FourierTransformer class, with calc type float.
  // For the time being input and output are also float. TODO calc optionally either fp16 or nv_bloat16, TODO inputs at lower precision for bandwidth improvement.
  FastFFT::FourierTransformer FT(FastFFT::FourierTransformer::DataType::fp32);
  
  
  // Determine how much memory we need, working with FFTW/CUDA style in place transform padding.
  // Note: there is no reason we really need this, because the xforms will always be out of place. 
  //       For now, this is just in place because all memory in cisTEM is allocated accordingly.
  host_input.real_memory_allocated = FT.ReturnPaddedMemorySize(input_size);
  host_output.real_memory_allocated = FT.ReturnPaddedMemorySize(output_size);
  
  // On the device, we will always allocate enough memory for the larger of input/output including the buffer array.
  // Minmize the number of calls to malloc which are slow and can lead to fragmentation.
  device_output.real_memory_allocated = std::max(host_input.real_memory_allocated, host_output.real_memory_allocated);
  
  
  // In your own programs, you will be handling this memory allocation yourself. We'll just make something here.
  // I think fftwf_malloc may potentially create a different alignment than new/delete, but kinda doubt it. For cisTEM consistency...
  bool set_fftw_plan = true;
  host_input.Allocate(set_fftw_plan);
  host_output.Allocate(set_fftw_plan);
  

  // This is similar to creating an FFT/CUFFT plan, so set these up before doing anything on the GPU
  FT.SetInputDimensionsAndType(input_size.x,input_size.y,input_size.z,true, false,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);
  FT.SetOutputDimensionsAndType(output_size.x,output_size.y,output_size.z,true,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);
  
  // Now we want to associate the host memory with the device memory. The method here asks if the host pointer is pinned (in page locked memory) which
  // ensures faster transfer. If false, it will be pinned for you.
  FT.SetInputPointer(host_input.real_values, false);
  
  // Set a unit impulse at the center of the input array.
  FT.SetToConstant<float>(host_input.real_values, host_input.real_memory_allocated, 0.0f);
  FT.SetToConstant<float>(host_output.real_values, host_output.real_memory_allocated, 0.0f);

  host_input.real_values[ input_size.y/2 * (input_size.x+host_input.padding_jump_value) + input_size.x/2] = 1.0f;
  short4 wanted_center = make_short4(0,0,0,0);
  ClipInto(host_input.real_values, host_output.real_values, input_size ,  output_size,  wanted_center, 0.f);
  // for (int x = 0; x < 128; x++)
  // {
  //   int n=0;
  //   std::cout << x << "[ ";
  //   for (int y = 0; y < 128; y++)
  //   {  
  //     if (host_output[x + y*130]== 1) {host_output[x + y*130] = x; host_output[1 +x + y*130] = y;  n++;}
  //     std::cout << host_output[x + y*130]<< " ";
     
  //   }
  //   std::cout << "] " << n << std::endl;
  // }

  // Now set to origin and then phase swap quadrants
  FT.SetToConstant<float>(host_input.real_values, host_input.real_memory_allocated, 0.0f);
  host_input.real_values[0] = 1.0f;

  sum = ReturnSumOfReal(host_output.real_values, output_size);
  MyFFTDebugAssertTestTrue( sum == 1,"Unit impulse Init ");
  
  // This copies the host memory into the device global memory. If needed, it will also allocate the device memory first.
  FT.CopyHostToDevice();

  host_output.FwdFFT();
  
  host_output.fftw_epsilon = ReturnSumOfComplexAmplitudes(host_output.complex_values, host_output.real_memory_allocated/2);  
  std::cout << "host" << host_output.fftw_epsilon << " " << host_output.real_memory_allocated<< std::endl;

  host_output.fftw_epsilon -= (host_output.real_memory_allocated/2 );
  MyFFTDebugAssertTestTrue( std::abs(host_output.fftw_epsilon) < 1e-8 , "FFTW unit impulse forward FFT");
  
  // Just to make sure we don't get a false positive, set the host memory to some undesired value.
  FT.SetToConstant<float>(host_output.real_values, host_output.real_memory_allocated, 2.0f);
  
  // This method will call the regular FFT kernels given the input/output dimensions are equal when the class is instantiated.
  bool swap_real_space_quadrants = true;
  FT.FwdFFT(swap_real_space_quadrants);
  
  // do not deallocate, do not unpin memory

  FT.CopyDeviceToHost(host_output.real_values, false, false);

  // int n=0;
  // for (int x = 0; x <  host_output.size.y ; x++)
  // {
    
  //   std::cout << x << "[ ";
  //   for (int y = 0; y < host_output.size.w; y++)
  //   {  
  //     std::cout << host_output.complex_values[x + y*host_output.size.y].x << "," << host_output.complex_values[x + y*host_output.size.y].y << " ";
  //     n++;
  //     if (n == 32) {n = 0; std::cout << std::endl ;} // line wrapping
  //   }
  //   std::cout << "] " << std::endl;
  //   n = 0;
  // }

  sum = ReturnSumOfComplexAmplitudes(host_output.complex_values, host_output.real_memory_allocated/2); 
  std::cout << sum << " " << host_output.real_memory_allocated<< std::endl;

  sum -= (host_output.real_memory_allocated/2 );


  std::cout << "FFT Unit Impulse Forward FFT: " << sum <<  " epsilon" << host_output.fftw_epsilon << std::endl;
  MyFFTDebugAssertTestTrue( abs(sum - host_output.fftw_epsilon) < 1e-8, "FastFFT unit impulse forward FFT");
  FT.SetToConstant<float>(host_output.real_values, host_output.real_memory_allocated, 2.0f);
  

  FT.InvFFT();
  FT.CopyDeviceToHost(host_output.real_values, true, true);
  // for (int x = 0; x < 128; x++)
  // {
  //   int n=0;
  //   std::cout << x << "[ ";
  //   for (int y = 0; y < 128; y++)
  //   {  
  //     std::cout << host_output[x + y*130]<< " ";
  //   }
  //   std::cout << "] " << n << std::endl;
  // }
  // Assuming the outputs are always even dimensions, padding_jump_val is always 2.
  sum = ReturnSumOfReal(host_output.real_values, output_size);
  MyFFTDebugAssertTestTrue( sum == output_size.x*output_size.y*output_size.z,"FastFFT unit impulse round trip FFT");
  



}

void compare_libraries(short4 input_size, short4 output_size)
{

  bool set_padding_callback = true;
  bool set_conjMult_callback = true;
  if (input_size.x == output_size.x && input_size.y == output_size.y && input_size.z == output_size.z) set_padding_callback = false;
  bool test_passed = true;
  long address = 0;

  float sum;
  float2 sum_complex;

  Image< float, float2 > FT_input(input_size);
  Image< float, float2 > FT_output(output_size);
  Image< float, float2 > cuFFT_input(input_size);
  Image< float, float2 > cuFFT_output(output_size);

  Image< float, float2> target_search_image(output_size);


   // We just make one instance of the FourierTransformer class, with calc type float.
  // For the time being input and output are also float. TODO calc optionally either fp16 or nv_bloat16, TODO inputs at lower precision for bandwidth improvement.
  FastFFT::FourierTransformer FT(FastFFT::FourierTransformer::DataType::fp32);
    // Create an instance to copy memory also for the cufft tests.
  FastFFT::FourierTransformer cuFFT(FastFFT::FourierTransformer::DataType::fp32);
  FastFFT::FourierTransformer targetFT(FastFFT::FourierTransformer::DataType::fp32);


  FT_input.real_memory_allocated = FT.ReturnPaddedMemorySize(input_size);
  FT_output.real_memory_allocated = FT.ReturnPaddedMemorySize(output_size);
  
  cuFFT_input.real_memory_allocated = cuFFT.ReturnPaddedMemorySize(input_size);
  cuFFT_output.real_memory_allocated = cuFFT.ReturnPaddedMemorySize(output_size);

  target_search_image.real_memory_allocated = targetFT.ReturnPaddedMemorySize(output_size);

  bool set_fftw_plan = false;
  FT_input.Allocate(set_fftw_plan);
  FT_output.Allocate(set_fftw_plan);

  cuFFT_input.Allocate(set_fftw_plan);
  cuFFT_output.Allocate(set_fftw_plan);

  target_search_image.Allocate(set_fftw_plan);
  



  // This is similar to creating an FFT/CUFFT plan, so set these up before doing anything on the GPU
  FT.SetInputDimensionsAndType(input_size.x,input_size.y,input_size.z,true, false,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);
  FT.SetOutputDimensionsAndType(output_size.x,output_size.y,output_size.z,true,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);
  cuFFT.SetInputDimensionsAndType(input_size.x,input_size.y,input_size.z,true, false,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);
  cuFFT.SetOutputDimensionsAndType(output_size.x,output_size.y,output_size.z,true,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);
  
  targetFT.SetInputDimensionsAndType(input_size.x,input_size.y,input_size.z,true, false,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);
  targetFT.SetOutputDimensionsAndType(output_size.x,output_size.y,output_size.z,true,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);

  // Now we want to associate the host memory with the device memory. The method here asks if the host pointer is pinned (in page locked memory) which
  // ensures faster transfer. If false, it will be pinned for you.
  FT.SetInputPointer(FT_input.real_values, false);
  cuFFT.SetInputPointer(cuFFT_input.real_values, false);
  targetFT.SetInputPointer(target_search_image.real_values, false);

  // Set a unit impulse at the center of the input array.
  FT.SetToConstant<float>(FT_input.real_values, FT_input.real_memory_allocated, 1.0f);
  FT.SetToConstant<float>(cuFFT_input.real_values, cuFFT_input.real_memory_allocated, 1.0f);
  FT.SetToConstant<float>(target_search_image.real_values, target_search_image.real_memory_allocated, 1.0f);


  
  // This copies the host memory into the device global memory. If needed, it will also allocate the device memory first.
  FT.CopyHostToDevice();
  cuFFT.CopyHostToDevice();
  targetFT.CopyHostToDevice();

  cuFFT_output.create_timing_events();
  
  cuFFT_input.MakeCufftPlan();
  cuFFT_output.MakeCufftPlan();


  const int n_loops = 10000;
  cuFFT_output.record_start();
  for (int i = 0; i < n_loops; ++i)
  {
    FT.FwdFFT();
    FT.InvFFT();
    // FT.CrossCorrelate(targetFT.device_pointer_fp32_complex, false);
  }
  cuFFT_output.record_stop();
  cuFFT_output.synchronize();
  cuFFT_output.print_time("FastFFT");

  set_padding_callback = false;
  if (set_padding_callback) 
  {
    precheck
    cufftReal* overlap_pointer;
    overlap_pointer = cuFFT.device_pointer_fp32;
    cuFFT_output.SetClipIntoCallback(overlap_pointer, cuFFT_input.size.x, cuFFT_input.size.y, cuFFT_input.size.w*2);
    postcheck
  }

  if (set_conjMult_callback)
  {
    precheck
    // FIXME scaling factor
    cuFFT_output.SetComplexConjMultiplyAndLoadCallBack( (cufftComplex *) targetFT.device_pointer_fp32_complex, 1.0f);
    postcheck
  }

  cuFFT_output.record_start();
  for (int i = 0; i < n_loops; ++i)
  {
    cuFFT.ClipIntoTopLeft();
    // cuFFT.ClipIntoReal(input_size.x/2, input_size.y/2, input_size.z/2);

    precheck
    cudaErr(cufftExecR2C(cuFFT_output.cuda_plan_forward, (cufftReal*)cuFFT.device_pointer_fp32, (cufftComplex*)cuFFT.device_pointer_fp32_complex));
    postcheck

    precheck
    cudaErr(cufftExecC2R(cuFFT_output.cuda_plan_inverse, (cufftComplex*)cuFFT.device_pointer_fp32_complex, (cufftReal*)cuFFT.device_pointer_fp32));
    postcheck
  }
  cuFFT_output.record_stop();
  cuFFT_output.synchronize();
  cuFFT_output.print_time("cuFFT");

}

int main(int argc, char** argv) {

  std::printf("Entering main in tests.cpp\n");
  std::printf("Standard is %i\n\n",__cplusplus);

  // Input and output dimensions, with simple checks. I'm sure there are better checks on argv.
  short4 input_size;
  short4 output_size;

  constexpr const int n_tests = 5;
  const int test_size[n_tests] = {64, 128, 256, 512, 4096};
  // for (int iSize = 0; iSize < n_tests; iSize++) {

  //   std::cout << std::endl << "Testing constant image size " << test_size[iSize] << " x" << std::endl;
  //   input_size = make_short4(test_size[iSize],test_size[iSize],1,0);
  //   output_size = make_short4(test_size[iSize],test_size[iSize],1,0);

  //   const_image_test(input_size, output_size);

  // }


  // for (int iSize = 0; iSize < n_tests - 1; iSize++) {
  //   int oSize = iSize + 1;
  //   while (oSize < n_tests)
  //   {
  //     std::cout << std::endl << "Testing padding from  " << test_size[iSize] << " to " << test_size[oSize] << std::endl;
  //     input_size = make_short4(test_size[iSize],test_size[iSize],1,0);
  //     output_size = make_short4(test_size[oSize],test_size[oSize],1,0);
  
  //     unit_impulse_test(input_size, output_size);
  //     oSize++;
  //   }
  // }


  // for (int iSize = 0; iSize < n_tests; iSize++) {

  //   std::cout << std::endl << "Testing cufft comparison " << test_size[iSize] << " x" << std::endl;
  //   input_size = make_short4(test_size[iSize],test_size[iSize],1,0);
  //   output_size = make_short4(test_size[iSize],test_size[iSize],1,0);

  //   compare_libraries(input_size, output_size);

  // }

  for (int iSize = 0; iSize < n_tests - 1; iSize++) {
    int oSize = iSize + 1;
    while (oSize < n_tests)
    {
      std::cout << std::endl << "Testing padding from  " << test_size[iSize] << " to " << test_size[oSize] << std::endl;
      input_size = make_short4(test_size[iSize],test_size[iSize],1,0);
      output_size = make_short4(test_size[oSize],test_size[oSize],1,0);
  
      compare_libraries(input_size, output_size);
      oSize++;
    }
  }
  
}
