#include "Image.cu"
#include "../src/FastFFT.cu"
#include <cufft.h>
#include <cufftXt.h>

// The Fourier transform of a constant should be a unit impulse, and on back fft, without normalization, it should be a constant * N.
// It is assumed the input/output have the same dimension (i.e. no padding)
void const_image_test(std::vector<int> size)
{

  bool all_passed = true;
  std::vector<bool> init_passed(size.size(), true);
  std::vector<bool> FFTW_passed(size.size(), true);
  std::vector<bool> FastFFT_forward_passed(size.size(), true);
  std::vector<bool> FastFFT_roundTrip_passed(size.size(), true);

  for (int n = 0; n < size.size() ; n++)
  {

    short4 input_size = make_short4(size[n],size[n],1,0);
    short4 output_size = make_short4(size[n],size[n],1,0);

    bool test_passed = true;
    long address = 0;
    float sum;
    const float acceptable_epsilon = 1e-4;
    float2 sum_complex;

    Image< float, float2 > host_input(input_size);
    Image< float, float2 > host_output(output_size);
    Image< float, float2 > device_output(output_size);


      // Pointers to the arrays on the host -- maybe make this a struct of some sort? I'm sure there is a parallel in cuda, look into cuarray/texture code

    // We just make one instance of the FourierTransformer class, with calc type float.
    // For the time being input and output are also float. TODO calc optionally either fp16 or nv_bloat16, TODO inputs at lower precision for bandwidth improvement.
    FastFFT::FourierTransformer<float, float, float> FT;
    
    // This is similar to creating an FFT/CUFFT plan, so set these up before doing anything on the GPU
    FT.SetInputDimensionsAndType(input_size.x,input_size.y,input_size.z,true, false, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);
    FT.SetOutputDimensionsAndType(output_size.x,output_size.y,output_size.z,true, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);

      // The padding (dims.w) is calculated based on the setup
    short4 dims_in = FT.ReturnInputDimensions();
    short4 dims_out = FT.ReturnOutputDimensions();

    // Determine how much memory we need, working with FFTW/CUDA style in place transform padding.
    // Note: there is no reason we really need this, because the xforms will always be out of place. 
    //       For now, this is just in place because all memory in cisTEM is allocated accordingly.
    host_input.real_memory_allocated = FT.ReturnInputMemorySize();
    host_output.real_memory_allocated = FT.ReturnOutputMemorySize();
    
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

      

    
    // Now we want to associate the host memory with the device memory. The method here asks if the host pointer is pinned (in page locked memory) which
    // ensures faster transfer. If false, it will be pinned for you.
    FT.SetInputPointer(host_output.real_values, false);
    sum = ReturnSumOfReal(host_output.real_values, dims_out);
    if (sum != dims_out.x*dims_out.y*dims_out.z) {all_passed = false; init_passed[n] = false;}

    // MyFFTDebugAssertTestTrue( sum == dims_out.x*dims_out.y*dims_out.z,"Unit impulse Init ");
    
    // This copies the host memory into the device global memory. If needed, it will also allocate the device memory first.
    FT.CopyHostToDevice();
      
    host_output.FwdFFT();
    
    test_passed = true;
    for (long index = 1; index < host_output.real_memory_allocated/2; index++)
    {
      if (host_output.complex_values[index].x != 0.0f && host_output.complex_values[index].y != 0.0f) { std::cout << host_output.complex_values[index].x  << " " << host_output.complex_values[index].y << " " << std::endl; test_passed = false;}
    }
    if (host_output.complex_values[0].x != (float)dims_out.x * (float)dims_out.y * (float)dims_out.z) test_passed = false;
    // for (int i = 0; i < 10; i++)
    // {
    //   std::cout << "FFTW unit " << host_output.complex_values[i].x << " " << host_output.complex_values[i].y << std::endl;
    // }
    if (test_passed == false) {all_passed = false; FFTW_passed[n] = false;}
    // MyFFTDebugAssertTestTrue( test_passed, "FFTW unit impulse forward FFT");
    
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
      if (host_output.complex_values[index].x != 0.0f && host_output.complex_values[index].y != 0.0f) {test_passed = false;} // std::cout << host_output.complex_values[index].x  << " " << host_output.complex_values[index].y << " " << std::endl;}
    }
    if (host_output.complex_values[0].x != (float)dims_out.x * (float)dims_out.y * (float)dims_out.z) test_passed = false;
    // int n=0;
    // for (int x = 0; x <  host_output.size.y ; x++)
    // {
      
    //   std::cout << x << "[ ";
    //   for (int y = 0; y < host_output.size.w; y++)
    //   {  
    //     std::cout << host_output.complex_values[x + y*host_output.size.y].x << "," << host_output.complex_values[x + y*host_output.size.y].y << " ";
    //     n++;
    //     if (n == 34) {n = 0; std::cout << std::endl ;} // line wrapping
    //   }
    //   std::cout << "] " << std::endl;
    //   n = 0;
    // }

    if (test_passed == false) {all_passed = false; FastFFT_forward_passed[n] = false;}
    // MyFFTDebugAssertTestTrue( test_passed, "FastFFT unit impulse forward FFT");
    FT.SetToConstant<float>(host_input.real_values, host_input.real_memory_allocated, 2.0f);
    

    FT.InvFFT();
    FT.CopyDeviceToHost( true, true);
    
    // Assuming the outputs are always even dimensions, padding_jump_val is always 2.
    sum = ReturnSumOfReal(host_output.real_values, dims_out);

    // COMPLEX TODO make these functions.
    //   int n=0;
    //   for (int x = 0; x <  host_output.size.y ; x++)
    // {
      
    //   std::cout << x << "[ ";
    //   for (int y = 0; y < host_output.size.w; y++)
    //   {  
    //     std::cout << host_output.complex_values[x + y*host_output.size.y].x << "," << host_output.complex_values[x + y*host_output.size.y].y << " ";
    //     n++;
    //     if (n == 34) {n = 0; std::cout << std::endl ;} // line wrapping
    //   }
    //   std::cout << "] " << std::endl;
    //   n = 0;
    // }
      // REAL
    //  int n=0;
    // for (int x = 0; x <  host_output.size.x ; x++)
    // {
      
    //   std::cout << x << "[ ";
    //   for (int y = 0; y < host_output.size.y; y++)
    //   {  
    //     std::cout << host_output.real_values[x + y*host_output.size.w*2] <<  " ";
    //     n++;
    //     if (n == 32) {n = 0; std::cout << std::endl ;} // line wrapping
    //   }
    //   std::cout << "] " << std::endl;
    //   n = 0;
    // } 
    if (sum != powf(dims_in.x*dims_in.y*dims_in.z,2)) {all_passed = false; FastFFT_roundTrip_passed[n] = false;}
    // MyFFTDebugAssertTestTrue( sum == powf(dims_in.x*dims_in.y*dims_in.z,2),"FastFFT constant image round trip failed for size");
  }
  
  if (all_passed)
  {
    std::cout << "    All const_image tests passed!" << std::endl;
  }
  else  
  {
    for (int n = 0; n < size.size() ; n++)
    {
      if ( ! init_passed[n] ) std::cout << "    Initialization failed for size " << size[n] << std::endl;
      if ( ! FFTW_passed[n] ) std::cout << "    FFTW failed for size " << size[n] << std::endl;
      if ( ! FastFFT_forward_passed[n] ) std::cout << "    FastFFT failed for forward transform size " << size[n] << std::endl;
      if ( ! FastFFT_roundTrip_passed[n] ) std::cout << "    FastFFT failed for roundtrip transform size " << size[n] << std::endl;

    }
  }
}

void unit_impulse_test(std::vector<int>size)
{

  bool all_passed = true;
  std::vector<bool> init_passed(size.size(), true);
  std::vector<bool> FFTW_passed(size.size(), true);
  std::vector<bool> FastFFT_forward_passed(size.size(), true);
  std::vector<bool> FastFFT_roundTrip_passed(size.size(), true);

  for (int iSize = 0; iSize < size.size() - 1 ; iSize++)
  {
    int oSize = iSize + 1;
    while (oSize < size.size())
    {
      std::cout << std::endl << "Testing padding from  " << size[iSize] << " to " << size[oSize] << std::endl;
      short4 input_size = make_short4(size[iSize],size[iSize],1,0);
      short4 output_size = make_short4(size[oSize],size[oSize],1,0);   


  bool test_passed = true;
  long address = 0;

  float sum;
  float2 sum_complex;

  Image< float, float2 > host_input(input_size);
  Image< float, float2 > host_output(output_size);
  Image< float, float2 > device_output(output_size);
  

  // We just make one instance of the FourierTransformer class, with calc type float.
  // For the time being input and output are also float. TODO calc optionally either fp16 or nv_bloat16, TODO inputs at lower precision for bandwidth improvement.
  FastFFT::FourierTransformer<float, float, float> FT;
  // This is similar to creating an FFT/CUFFT plan, so set these up before doing anything on the GPU
  FT.SetInputDimensionsAndType(input_size.x,input_size.y,input_size.z,true, false, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);
  FT.SetOutputDimensionsAndType(output_size.x,output_size.y,output_size.z,true, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural); 
 
  // The padding (dims.w) is calculated based on the setup
  short4 dims_in = FT.ReturnInputDimensions();
  short4 dims_out = FT.ReturnOutputDimensions();
  // Determine how much memory we need, working with FFTW/CUDA style in place transform padding.
  // Note: there is no reason we really need this, because the xforms will always be out of place. 
  //       For now, this is just in place because all memory in cisTEM is allocated accordingly.
  host_input.real_memory_allocated = FT.ReturnInputMemorySize();
  host_output.real_memory_allocated = FT.ReturnOutputMemorySize();



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
  FT.SetToConstant<float>(host_input.real_values, host_input.real_memory_allocated, 0.0f);
  FT.SetToConstant<float>(host_output.real_values, host_output.real_memory_allocated, 0.0f);

  sum = ReturnSumOfReal(host_output.real_values, dims_out);
  // host_input.real_values[ dims_in.y/2 * (dims_in.x+host_input.padding_jump_value) + dims_in.x/2] = 1.0f;
  // short4 wanted_center = make_short4(0,0,0,0);
  // ClipInto(host_input.real_values, host_output.real_values, dims_in ,  dims_out,  wanted_center, 0.f);

  // FT.SetToConstant<float>(host_input.real_values, host_input.real_memory_allocated, 0.0f);
  host_input.real_values[0] = 1.0f;
  host_output.real_values[0] = 1.0f;


  sum = ReturnSumOfReal(host_output.real_values, dims_out);
  if (sum != 1) {all_passed = false; init_passed[iSize] = false;}

  // MyFFTDebugAssertTestTrue( sum == 1,"Unit impulse Init ");
  
  // This copies the host memory into the device global memory. If needed, it will also allocate the device memory first.
  FT.CopyHostToDevice();

  host_output.FwdFFT();
  
  host_output.fftw_epsilon = ReturnSumOfComplexAmplitudes(host_output.complex_values, host_output.real_memory_allocated/2);  
  // std::cout << "host " << host_output.fftw_epsilon << " " << host_output.real_memory_allocated<< std::endl;

  host_output.fftw_epsilon -= (host_output.real_memory_allocated/2 );
  if (std::abs(host_output.fftw_epsilon) > 1e-8 ) {all_passed = false; FFTW_passed[iSize] = false;}

  // MyFFTDebugAssertTestTrue( std::abs(host_output.fftw_epsilon) < 1e-8 , "FFTW unit impulse forward FFT");
  
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
  // std::cout << sum << " " << host_output.real_memory_allocated<< std::endl;

  sum -= (host_output.real_memory_allocated/2 );


  // std::cout << "FFT Unit Impulse Forward FFT: " << sum <<  " epsilon " << host_output.fftw_epsilon << std::endl;
  // std::cout << "epsilon " << abs(sum - host_output.fftw_epsilon) << std::endl;
  if (abs(sum - host_output.fftw_epsilon) > 1e-8) {all_passed = false; FastFFT_forward_passed[iSize] = false;}

  // MyFFTDebugAssertTestTrue( abs(sum - host_output.fftw_epsilon) < 1e-8, "FastFFT unit impulse forward FFT");
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
  sum = ReturnSumOfReal(host_output.real_values, dims_out);
  if (sum != dims_out.x*dims_out.y*dims_out.z) {all_passed = false; FastFFT_roundTrip_passed[iSize] = false;}

  // MyFFTDebugAssertTestTrue( sum == dims_out.x*dims_out.y*dims_out.z,"FastFFT unit impulse round trip FFT");
    oSize++;
    } // while loop over pad to size
  } // for loop over pad from size


  if (all_passed)
  {
    std::cout << "    All unit impulse tests passed!" << std::endl;
  }
  else  
  {
    for (int n = 0; n < size.size() ; n++)
    {
      if ( ! init_passed[n] ) std::cout << "    Initialization failed for size " << size[n] << std::endl;
      if ( ! FFTW_passed[n] ) std::cout << "    FFTW failed for size " << size[n] << std::endl;
      if ( ! FastFFT_forward_passed[n] ) std::cout << "    FastFFT failed for forward transform size " << size[n] << std::endl;
      if ( ! FastFFT_roundTrip_passed[n] ) std::cout << "    FastFFT failed for roundtrip transform size " << size[n] << std::endl;

    }
  }

}

void compare_libraries(short4 input_size, short4 output_size)
{

  bool set_padding_callback = false; // the padding callback is slower than pasting in b/c the read size of the pointers is larger than the actual data. do not use.
  bool set_conjMult_callback = true;
  if (input_size.x == output_size.x && input_size.y == output_size.y && input_size.z == output_size.z) 
  {
    // Also will change the path called in FastFFT to just be fwd/inv xform.
    set_conjMult_callback = false;
  }

  bool test_passed = true;
  long address = 0;

  float sum;
  float2 sum_complex;

  Image< float, float2 > FT_input(input_size);
  Image< float, float2 > FT_output(output_size);
  Image< float, float2 > cuFFT_input(input_size);
  Image< float, float2 > cuFFT_output(output_size);

  Image< float, float2> target_search_image(output_size);
  Image< float, float2> positive_control(output_size);


   // We just make one instance of the FourierTransformer class, with calc type float.
  // For the time being input and output are also float. TODO calc optionally either fp16 or nv_bloat16, TODO inputs at lower precision for bandwidth improvement.
  FastFFT::FourierTransformer<float, float, float> FT;
    // Create an instance to copy memory also for the cufft tests.
  FastFFT::FourierTransformer<float, float, float> cuFFT;
  FastFFT::FourierTransformer<float, float, float> targetFT;

  // This is similar to creating an FFT/CUFFT plan, so set these up before doing anything on the GPU
  FT.SetInputDimensionsAndType(input_size.x,input_size.y,input_size.z,true, false, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);
  FT.SetOutputDimensionsAndType(output_size.x,output_size.y,output_size.z,true, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);
  cuFFT.SetInputDimensionsAndType(input_size.x,input_size.y,input_size.z,true, false, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);
  cuFFT.SetOutputDimensionsAndType(output_size.x,output_size.y,output_size.z,true, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);
  
  targetFT.SetInputDimensionsAndType(output_size.x,output_size.y,output_size.z,true, false, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);
  targetFT.SetOutputDimensionsAndType(output_size.x,output_size.y,output_size.z,true, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);


  FT_input.real_memory_allocated = FT.ReturnInputMemorySize();
  FT_output.real_memory_allocated = FT.ReturnOutputMemorySize();
  
  cuFFT_input.real_memory_allocated = cuFFT.ReturnInputMemorySize();
  cuFFT_output.real_memory_allocated = cuFFT.ReturnOutputMemorySize();

  target_search_image.real_memory_allocated = targetFT.ReturnInputMemorySize();
  positive_control.real_memory_allocated = targetFT.ReturnOutputMemorySize();


  bool set_fftw_plan = false;
  FT_input.Allocate(set_fftw_plan);
  FT_output.Allocate(set_fftw_plan);

  cuFFT_input.Allocate(set_fftw_plan);
  cuFFT_output.Allocate(set_fftw_plan);

  target_search_image.Allocate(true);
  positive_control.Allocate(true);


  // Now we want to associate the host memory with the device memory. The method here asks if the host pointer is pinned (in page locked memory) which
  // ensures faster transfer. If false, it will be pinned for you.
  FT.SetInputPointer(FT_input.real_values, false);
  cuFFT.SetInputPointer(cuFFT_input.real_values, false);
  targetFT.SetInputPointer(target_search_image.real_values, false);

  // Set a unit impulse at the center of the input array.
  FT.SetToConstant<float>(FT_input.real_values, FT_input.real_memory_allocated, 0.0f);
  FT.SetToConstant<float>(cuFFT_input.real_values, cuFFT_input.real_memory_allocated, 0.0f);
  FT.SetToConstant<float>(FT_output.real_values, FT_input.real_memory_allocated, 0.0f);
  FT.SetToConstant<float>(cuFFT_output.real_values, cuFFT_input.real_memory_allocated, 0.0f);
  FT.SetToConstant<float>(target_search_image.real_values, target_search_image.real_memory_allocated, 0.0f);
  FT.SetToConstant<float>(positive_control.real_values, target_search_image.real_memory_allocated, 0.0f);


  // Place these values at the origin of the image and after convolution, should be at 0,0,0.
  float testVal_1 = 2.0f;
  float testVal_2 = 3.0f;
  FT_input.real_values[0] = testVal_1;
  cuFFT_input.real_values[0] = testVal_1;
  target_search_image.real_values[0] = testVal_2;//target_search_image.size.w*2*target_search_image.size.y/2 + target_search_image.size.x/2] = testVal_2;
  positive_control.real_values[0] = testVal_1;//target_search_image.size.w*2*target_search_image.size.y/2 + target_search_image.size.x/2] = testVal_1;

  // Transform the target on the host prior to transfer.
  target_search_image.FwdFFT();

  // This copies the host memory into the device global memory. If needed, it will also allocate the device memory first.
  FT.CopyHostToDevice();
  cuFFT.CopyHostToDevice();
  targetFT.CopyHostToDevice();
  // Wait on the transfers to finish.
  cudaErr(cudaStreamSynchronize(cudaStreamPerThread));  

  // Positive control on the host.
  positive_control.FwdFFT();
  positive_control.MultiplyConjugateImage(target_search_image.complex_values);
  positive_control.InvFFT();


  // address = 0;
  // test_passed = true;
  // for (int z = 1; z <  positive_control.size.z ; z++)
  // {   
  //   for (int y = 1; y < positive_control.size.y; y++)
  //   {  
  //     for (int x = 1; x < positive_control.size.x; x++)
  //     {
  //       if (positive_control.real_values[address] != 0.0f) test_passed = false;
  //     }
  //   }
  // }
  // if (test_passed) 
  // {
  //   if (positive_control.real_values[address] == positive_control.size.x*positive_control.size.y*positive_control.size.z*testVal_1*testVal_2)
  //   {
  //     std::cout << "Test passed for FFTW positive control.\n" << std::endl;
  //   }
  //   else
  //   {
  //     std::cout << "Test failed for FFTW positive control. Value at zero is  " << positive_control.real_values[address] << std::endl;
  //   }
  // }
  // else
  // {
  //   std::cout << "Test failed for positive control, non-zero values found away from the origin." << std::endl;
  // }


  cuFFT_output.create_timing_events(); 
  cuFFT_input.MakeCufftPlan();
  cuFFT_output.MakeCufftPlan();

  //////////////////////////////////////////
  //////////////////////////////////////////
  // Warm up and check for accuracy
  if (set_conjMult_callback)
  {
    FT.CrossCorrelate(targetFT.d_ptr.momentum_space_buffer, false);
  }
  else
  {
    FT.FwdFFT();
    FT.InvFFT();
  }
  FT.CopyDeviceToHost(FT_output.real_values,false, false);

  // address = 0;
  // test_passed = true;
  // for (int z = 1; z <  FT_output.size.z ; z++)
  // {   
  //   for (int y = 1; y < FT_output.size.y; y++)
  //   {  
  //     for (int x = 1; x < FT_output.size.x; x++)
  //     {
  //       if (FT_output.real_values[address] != 0.0f) test_passed = false;
  //     }
  //   }
  // }
  // if (test_passed) 
  // {
  //   if (FT_output.real_values[address] == FT_output.size.x*FT_output.size.y*FT_output.size.z*testVal_1*testVal_2)
  //   {
  //     std::cout << "Test passed for FastFFT positive control.\n" << std::endl;
  //   }
  //   else
  //   {
  //     std::cout << "Test failed for FastFFT positive control. Value at zero is  " << FT_output.real_values[address] << std::endl;
  //   }
  // }
  // else
  // {
  //   std::cout << "Test failed for FastFFT control, non-zero values found away from the origin." << std::endl;
  // }

  //////////////////////////////////////////
  // //////////////////////////////////////////
  // int n = 0;
  // for (int x = 0; x <  FT_output.size.x ; x++)
  // {
    
  //   std::cout << x << "[ ";
  //   for (int y = 0; y < FT_output.size.y; y++)
  //   {  
  //     std::cout << FT_output.real_values[x + y*FT_output.size.w*2] << " ";
  //     n++;
  //     if (n == 32) {n = 0; std::cout << std::endl ;} // line wrapping
  //   }
  //   std::cout << "] " << std::endl;
  //   n = 0;
  // }


  const int n_loops = 10000;
  cuFFT_output.record_start();
  for (int i = 0; i < n_loops; ++i)
  {
    if (set_conjMult_callback)
    {
      FT.CrossCorrelate(targetFT.d_ptr.momentum_space_buffer, false);
    }
    else
    {
      FT.FwdFFT();
      FT.InvFFT();
    }
  }
  cuFFT_output.record_stop();
  cuFFT_output.synchronize();
  cuFFT_output.print_time("FastFFT");
  float FastFFT_time = cuFFT_output.elapsed_gpu_ms;

  if (set_padding_callback) 
  {
    precheck
    cufftReal* overlap_pointer;
    overlap_pointer = cuFFT.d_ptr.position_space;
    cuFFT_output.SetClipIntoCallback(overlap_pointer, cuFFT_input.size.x, cuFFT_input.size.y, cuFFT_input.size.w*2);
    postcheck
  }

  if (set_conjMult_callback)
  {
    precheck
    // FIXME scaling factor
    cuFFT_output.SetComplexConjMultiplyAndLoadCallBack( (cufftComplex *) targetFT.d_ptr.momentum_space_buffer, 1.0f);
    postcheck
  }


  //////////////////////////////////////////
  //////////////////////////////////////////
  // Warm up and check for accuracy
  cuFFT.ClipIntoTopLeft();
  // cuFFT.ClipIntoReal(cuFFT_output.size.x/2, cuFFT_output.size.y/2, cuFFT_output.size.z/2);

  cuFFT.CopyDeviceToHost(cuFFT_output.real_values,false, false);
  // cuFFT.ClipIntoReal(input_size.x/2, input_size.y/2, input_size.z/2);
  precheck
  cudaErr(cufftExecR2C(cuFFT_output.cuda_plan_forward, (cufftReal*)cuFFT.d_ptr.position_space, (cufftComplex*)cuFFT.d_ptr.momentum_space_buffer));
  postcheck
  precheck
  cudaErr(cufftExecC2R(cuFFT_output.cuda_plan_inverse, (cufftComplex*)cuFFT.d_ptr.momentum_space_buffer, (cufftReal*)cuFFT.d_ptr.position_space));
  postcheck  
  cuFFT.CopyDeviceToHost(cuFFT_output.real_values,false, false);

  // address = 0;
  // test_passed = true;
  // for (int z = 1; z <  cuFFT_output.size.z ; z++)
  // {   
  //   for (int y = 1; y < cuFFT_output.size.y; y++)
  //   {  
  //     for (int x = 1; x < cuFFT_output.size.x; x++)
  //     {
  //       if (cuFFT_output.real_values[address] != 0.0f) test_passed = false;
  //     }
  //   }
  // }
  // if (test_passed) 
  // {
  //   if (cuFFT_output.real_values[address] == cuFFT_output.size.x*cuFFT_output.size.y*cuFFT_output.size.z*testVal_1*testVal_2)
  //   {
  //     std::cout << "Test passed for cuFFT positive control.\n" << std::endl;
  //   }
  //   else
  //   {
  //     std::cout << "Test failed for cuFFT positive control. Value at zero is  " << cuFFT_output.real_values[address] << std::endl;
  //   }
  // }
  // else
  // {
  //   std::cout << "Test failed for cuFFT control, non-zero values found away from the origin." << std::endl;
  // }
  //////////////////////////////////////////
  //////////////////////////////////////////
  // n = 0;
  // for (int x = 0; x <  cuFFT_output.size.x ; x++)
  // {
    
  //   std::cout << x << "[ ";
  //   for (int y = 0; y < cuFFT_output.size.y; y++)
  //   {  
  //     std::cout << cuFFT_output.real_values[x + y*cuFFT_output.size.w*2] << " ";
  //     n++;
  //     if (n == 32) {n = 0; std::cout << std::endl ;} // line wrapping
  //   }
  //   std::cout << "] " << std::endl;
  //   n = 0;
  // }

  cuFFT_output.record_start();
  for (int i = 0; i < n_loops; ++i)
  {
    if (set_conjMult_callback) cuFFT.ClipIntoTopLeft();
    // cuFFT.ClipIntoReal(input_size.x/2, input_size.y/2, input_size.z/2);

    precheck
    cudaErr(cufftExecR2C(cuFFT_output.cuda_plan_forward, (cufftReal*)cuFFT.d_ptr.position_space, (cufftComplex*)cuFFT.d_ptr.momentum_space_buffer));
    postcheck

    precheck
    cudaErr(cufftExecC2R(cuFFT_output.cuda_plan_inverse, (cufftComplex*)cuFFT.d_ptr.momentum_space_buffer, (cufftReal*)cuFFT.d_ptr.position_space));
    postcheck
  }
  cuFFT_output.record_stop();
  cuFFT_output.synchronize();
  cuFFT_output.print_time("cuFFT");

  std::cout << "Ratio cuFFT/FastFFT : " << cuFFT_output.elapsed_gpu_ms/FastFFT_time << std::endl;

}

void run_oned(std::vector<int> size)
{

  // Override the size to be one dimensional in x
  std::cout << "Running one-dimensional tests\n" << std::endl;

  bool test_passed = true;
  long address = 0;

  float sum;
  float2 sum_complex;

  for (int n : size)
  {
    short4 input_size = make_short4(n,1,1,0);
    short4 output_size = make_short4(n,1,1,0);

    Image< float, float2 > FT_input(input_size);
    Image< float, float2 > FT_output(output_size);
    Image< float2, float2 > FT_input_complex(input_size);
    Image< float2, float2 > FT_output_complex(output_size);

    // We just make one instance of the FourierTransformer class, with calc type float.
    // For the time being input and output are also float. TODO calc optionally either fp16 or nv_bloat16, TODO inputs at lower precision for bandwidth improvement.
    FastFFT::FourierTransformer<float, float, float> FT;
    FastFFT::FourierTransformer<float, float2, float2> FT_complex;

    // This is similar to creating an FFT/CUFFT plan, so set these up before doing anything on the GPU
    FT.SetInputDimensionsAndType(input_size.x,input_size.y,input_size.z,true, false, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);
    FT.SetOutputDimensionsAndType(output_size.x,output_size.y,output_size.z,true, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);

    FT_complex.SetInputDimensionsAndType(input_size.x,input_size.y,input_size.z,true, false, FastFFT::FourierTransformer<float, float2 ,float2>::OriginType::natural);
    FT_complex.SetOutputDimensionsAndType(output_size.x,output_size.y,output_size.z,true, FastFFT::FourierTransformer<float, float2 ,float2>::OriginType::natural);

    FT_input.real_memory_allocated = FT.ReturnInputMemorySize();
    FT_output.real_memory_allocated = FT.ReturnOutputMemorySize();

    FT_input_complex.real_memory_allocated = FT_complex.ReturnInputMemorySize();
    FT_output_complex.real_memory_allocated = FT_complex.ReturnOutputMemorySize();
    std::cout << "Allocated " << FT_input_complex.real_memory_allocated << " bytes for input.\n";
    std::cout << "Allocated complex " << FT_output_complex.real_memory_allocated << " bytes for input.\n";

    bool set_fftw_plan = true;
    FT_input.Allocate(set_fftw_plan);
    FT_output.Allocate(set_fftw_plan);

    FT_input_complex.Allocate();
    FT_output_complex.Allocate();



    // Now we want to associate the host memory with the device memory. The method here asks if the host pointer is pinned (in page locked memory) which
    // ensures faster transfer. If false, it will be pinned for you.
    FT.SetInputPointer(FT_input.real_values, false);
    FT_complex.SetInputPointer(FT_input_complex.complex_values, false);


    FT.SetToConstant<float>(FT_input.real_values, FT_input.real_memory_allocated, 1.f);

    // Set a unit impulse at the center of the input array.
    // FT.SetToConstant<float>(FT_input.real_values, FT_input.real_memory_allocated, 1.0f);
    float2 const_val = make_float2(1.0f,0.0f);
    FT_complex.SetToConstant<float2>(FT_input_complex.complex_values, FT_input.real_memory_allocated, const_val);
    for (int i=0; i<10; i++)
    {
      std::cout << FT_input_complex.complex_values[i].x << "," << FT_input_complex.complex_values[i].y << std::endl;
    }


    FT.CopyHostToDevice();
    FT_complex.CopyHostToDevice();
    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));  

        // Set the outputs to a clearly wrong answer.
        FT.SetToConstant<float>(FT_output.real_values, FT_input.real_memory_allocated, 2.0f);
        const_val = make_float2(2.0f,2.0f);
        FT_complex.SetToConstant<float2>(FT_output_complex.complex_values, FT_output.real_memory_allocated, const_val);

    FT_input.FwdFFT();

    for (int i = 0; i < 5; ++i) std::cout << "FFTW fwd " << FT_input.real_values[i] << std::endl;
    std::cout << std::endl;


    bool transpose_output = false;
    bool swap_real_space_quadrants = false;
    FT.FwdFFT(swap_real_space_quadrants, transpose_output);
    FT_complex.FwdFFT(swap_real_space_quadrants, transpose_output);

    FT.CopyDeviceToHost(FT_output.real_values, false, false);
    FT_complex.CopyDeviceToHost(FT_output_complex.real_values, false, false);

    for (int i = 0; i < 10; ++i) {std::cout << "FT fwd " << FT_output.real_values[i] << std::endl;}
    for (int i = 0; i < 10; ++i) {std::cout << "FT complex fwd "<< FT_output_complex.real_values[i].x << "," << FT_output_complex.real_values[i].y << std::endl;}
  

    FT_input.InvFFT();

    for (int i = 0; i < 5; ++i) {std::cout << "FFTW inv " << FT_input.real_values[i] << std::endl;}
    std::cout << std::endl;



    FT.InvFFT(transpose_output);
    FT_complex.InvFFT(transpose_output);
    FT.CopyDeviceToHost(FT_output.real_values, true, true);
    FT_complex.CopyDeviceToHost(FT_output_complex.real_values, true, true);

    for (int i = 0; i < 10; i++) {std::cout << "Ft inv " << FT_output.real_values[i] << std::endl;}
    for (int i = 0; i < 10; i++) {std::cout << "Ft complex inv " << FT_output_complex.real_values[i].x << "," << FT_output_complex.real_values[i].y << std::endl;}


  }


}
int main(int argc, char** argv) {

  std::printf("Entering main in tests.cpp\n");
  std::printf("Standard is %i\n\n",__cplusplus);


  bool run_validation_tests;
  bool run_performance_tests;

  if (argc > 1)
  {
    run_validation_tests = false;
    run_performance_tests = true;
    std::cout << "Running performance tests.\n";
  }
  else
  {
    run_validation_tests = true;
    run_performance_tests = false;
    std::cout << "Running validation tests.\n";
  }
  // Input and output dimensions, with simple checks. I'm sure there are better checks on argv.
  short4 input_size;
  short4 output_size;

  std::vector<int> test_size = { 64, 128, 256, 512, 1024, 2048, 4096};

  std::vector<int> test_sizes =  {32};//,64,128,256,320,480,512,544,608,768,1024,1056,1536,2048,2560,3072,3584,4096,5120,6144};

  if (run_validation_tests)  {

    // change onde these to just report the pass/fail.
    // run_oned(test_sizes);
    // exit(0);



    const_image_test(test_size);
    unit_impulse_test(test_size);


  } // end of validation tests


  if (run_performance_tests) {

    #ifdef HEAVYERRORCHECKING_FFT
      std::cout << "Running performance tests with heavy error checking.\n";
      std::cout << "This doesn't make sense as the synchronizations are invalidating.\n";
      // exit(1);
    #endif

    for (int iSize = 0; iSize < test_size.size(); iSize++) {

      std::cout << std::endl << "Testing cufft comparison " << test_size[iSize] << " x" << std::endl;
      input_size = make_short4(test_size[iSize],test_size[iSize],1,0);
      output_size = make_short4(test_size[iSize],test_size[iSize],1,0);

      compare_libraries(input_size, output_size);

    }
 
    for (int iSize = 0; iSize < test_size.size() - 1; iSize++) {
      int oSize = iSize + 1;
      while (oSize < test_size.size())
      {
        std::cout << std::endl << "Testing padding from  " << test_size[iSize] << " to " << test_size[oSize] << std::endl;
        input_size = make_short4(test_size[iSize],test_size[iSize],1,0);
        output_size = make_short4(test_size[oSize],test_size[oSize],1,0);
    
        compare_libraries(input_size, output_size);
        oSize++;
      }
    }
  }

}
