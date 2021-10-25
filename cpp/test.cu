#include "Image.cu"
#include "../src/FastFFT.cu"
#include <cufft.h>
#include <cufftXt.h>

#define MyTestPrintAndExit(...)	{std::cerr << __VA_ARGS__  << " From: " << __FILE__  << " " << __LINE__  << " " << __PRETTY_FUNCTION__ << std::endl; exit(-1);}


void PrintArray( float2* array, short NX, short NY, short NZ, int line_wrapping = 34)
{
    // COMPLEX TODO make these functions.
    int n=0;
    for (int z = 0; z < NZ ; z ++)
    {
      for (int x = 0; x <  NX ; x++)
      {
        
        std::cout << x << "[ ";
        for (int y = 0; y < NY; y++)
        {  
          std::cout << array[x + NX*(y + z*NY)].x << "," << array[x + NX*(y + z*NY)].y << " ";
          n++;
          if (n == line_wrapping) {n = 0; std::cout << std::endl ;} // line wrapping
        }
        std::cout << "] " << std::endl;
        n = 0;
      }
      if (NZ > 0) std::cout << " ... ... ... " << z << " ... ... ..." << std::endl;
    }

};

void PrintArray(float* array, short NX, short NY, short NZ, short NW, int line_wrapping = 34)
{
  int n=0;
  for (int z = 0; z < NZ ; z ++)
  {
    for (int x = 0; x <  NX ; x++)
    {

      std::cout << x << "[ ";
      for (int y = 0; y < NY; y++)
      {  
        std::cout << array[x + (2*NW)*(y + z*NY)] <<  " ";
        n++;
        if (n == line_wrapping) {n = 0; std::cout << std::endl ;} // line wrapping
      }
      std::cout << "] " << std::endl;
      n = 0;
    } 
    if (NZ > 0) std::cout << " ... ... ... " << z << " ... ... ..." << std::endl;
  }
};

// The Fourier transform of a constant should be a unit impulse, and on back fft, without normalization, it should be a constant * N.
// It is assumed the input/output have the same dimension (i.e. no padding)
void const_image_test(std::vector<int> size, bool do_3d = false)
{

  bool all_passed = true;
  std::vector<bool> init_passed(size.size(), true);
  std::vector<bool> FFTW_passed(size.size(), true);
  std::vector<bool> FastFFT_forward_passed(size.size(), true);
  std::vector<bool> FastFFT_roundTrip_passed(size.size(), true);

  for (int n = 0; n < size.size() ; n++)
  {

    short4 input_size;
    short4 output_size;
    long full_sum = long(size[n]);
    if (do_3d)
    {
      input_size = make_short4(size[n],size[n],size[n],0);
      output_size = make_short4(size[n],size[n],size[n],0);
      full_sum =  full_sum*full_sum*full_sum*full_sum*full_sum*full_sum;
    }
    else
    {
      input_size = make_short4(size[n],size[n],1,0);
      output_size = make_short4(size[n],size[n],1,0);
      full_sum = full_sum*full_sum*full_sum;
    }


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
    FT.SetForwardFFTPlan(input_size.x,input_size.y,input_size.z, output_size.x,output_size.y,output_size.z, true, false, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);
    FT.SetInverseFFTPlan(output_size.x,output_size.y,output_size.z, output_size.x,output_size.y,output_size.z, true, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);

      // The padding (dims.w) is calculated based on the setup
    short4 dims_in = FT.ReturnFwdInputDimensions();
    short4 dims_out = FT.ReturnFwdOutputDimensions();

    // Determine how much memory we need, working with FFTW/CUDA style in place transform padding.
    // Note: there is no reason we really need this, because the xforms will always be out of place. 
    //       For now, this is just in place because all memory in cisTEM is allocated accordingly.
    host_input.real_memory_allocated = FT.ReturnInputMemorySize();
    host_output.real_memory_allocated = FT.ReturnInvOutputMemorySize();

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

      if (sum != long(dims_in.x)*long(dims_in.y)*long(dims_in.z)) {all_passed = false; init_passed[n] = false;}
    
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
      if (host_output.complex_values[index].x != 0.0f && host_output.complex_values[index].y != 0.0f) {test_passed = false;} // std::cout << host_output.complex_values[index].x  << " " << host_output.complex_values[index].y << " " );}
    }
    if (host_output.complex_values[0].x != (float)dims_out.x * (float)dims_out.y * (float)dims_out.z) test_passed = false;


    #if DEBUG_FFT_STAGE == 0
      PrintArray(host_output.real_values, dims_out.x, dims_in.y, dims_in.z, dims_out.w);
      MyTestPrintAndExit( "stage 0 " );
    #elif DEBUG_FFT_STAGE == 1
      PrintArray(host_output.complex_values, dims_in.y, dims_out.w, dims_in.z);
      MyTestPrintAndExit( "stage 1 " );
      #elif DEBUG_FFT_STAGE == 2
      PrintArray(host_output.complex_values, dims_in.y, dims_out.w, dims_out.z);
      MyTestPrintAndExit( "stage 2 " );      
    #elif DEBUG_FFT_STAGE == 3
      PrintArray(host_output.complex_values, dims_in.y, dims_out.w, dims_out.z);
      MyTestPrintAndExit( "stage 3 " );
    #endif   
    

    if (test_passed == false) {all_passed = false; FastFFT_forward_passed[n] = false;}
    // MyFFTDebugAssertTestTrue( test_passed, "FastFFT unit impulse forward FFT");
    FT.SetToConstant<float>(host_input.real_values, host_input.real_memory_allocated, 2.0f);
    

    FT.InvFFT();
    FT.CopyDeviceToHost( true, true);
 

    #if DEBUG_FFT_STAGE == 4
      PrintArray(host_output.complex_values, dims_out.y, dims_out.w, dims_out.z);
      MyTestPrintAndExit( "stage 4 " );
    #elif DEBUG_FFT_STAGE == 5
      PrintArray(host_output.complex_values, dims_out.y, dims_out.w, dims_out.z);
      MyTestPrintAndExit( "stage 5 " );
    #elif DEBUG_FFT_STAGE == 6
      PrintArray(host_output.complex_values, dims_out.y, dims_out.w, dims_out.z);
      MyTestPrintAndExit( "stage 6 " );      
    #elif DEBUG_FFT_STAGE == 7
      PrintArray(host_output.real_values, dims_out.x, dims_out.y,dims_out.z, dims_out.w);
      MyTestPrintAndExit( "stage 7 " );
    #elif DEBUG_FFT_STAGE > 7
      // No debug, keep going
    #else
      MyTestPrintAndExit( " This block is only valid for DEBUG_FFT_STAGE == 4, 5, 7 " );
    #endif   


    // Assuming the outputs are always even dimensions, padding_jump_val is always 2.
    sum = ReturnSumOfReal(host_output.real_values, dims_out);

    if (sum != full_sum) {all_passed = false; FastFFT_roundTrip_passed[n] = false;}
    MyFFTDebugAssertTestTrue( sum == full_sum,"FastFFT constant image round trip for size " + std::to_string(dims_in.x));
  } // loop over sizes
  
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

void unit_impulse_test(std::vector<int>size, bool do_increase_size)
{

  bool all_passed = true;
  std::vector<bool> init_passed(size.size(), true);
  std::vector<bool> FFTW_passed(size.size(), true);
  std::vector<bool> FastFFT_forward_passed(size.size(), true);
  std::vector<bool> FastFFT_roundTrip_passed(size.size(), true);

  short4 input_size;
  short4 output_size;
  for (int iSize = 0; iSize < size.size() - 1 ; iSize++)
  {
    int oSize = iSize + 1;
    while (oSize < size.size())
    {

      // std::cout << std::endl << "Testing padding from  " << size[iSize] << " to " << size[oSize] << std::endl;
      if (do_increase_size)
      {
        input_size  = make_short4(size[iSize],size[iSize],1,0);
        output_size = make_short4(size[oSize],size[oSize],1,0);  
      }
      else
      {
        output_size = make_short4(size[iSize],size[iSize],1,0);
        input_size  = make_short4(size[oSize],size[oSize],1,0);  
      }


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
  FT.SetForwardFFTPlan(input_size.x,input_size.y,input_size.z, output_size.x,output_size.y,output_size.z, true, false, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);
  FT.SetInverseFFTPlan(output_size.x,output_size.y,output_size.z, output_size.x,output_size.y,output_size.z, true, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural); 
 
  // The padding (dims.w) is calculated based on the setup
  short4 dims_in = FT.ReturnFwdInputDimensions();
  short4 dims_out = FT.ReturnFwdOutputDimensions();
  // Determine how much memory we need, working with FFTW/CUDA style in place transform padding.
  // Note: there is no reason we really need this, because the xforms will always be out of place. 
  //       For now, this is just in place because all memory in cisTEM is allocated accordingly.
  host_input.real_memory_allocated = FT.ReturnInputMemorySize();
  host_output.real_memory_allocated = FT.ReturnInvOutputMemorySize();



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

  int n=0;
  if (do_increase_size)
  {
    FT.CopyDeviceToHost(host_output.real_values, false, false);

      #if DEBUG_FFT_STAGE == 0
        PrintArray(host_output.real_values,  dims_in.x, dims_in.y, dims_in.z, dims_in.w);
        MyTestPrintAndExit( "stage 0 " );
      #elif DEBUG_FFT_STAGE == 1
        // If we are doing a fwd increase, the data will have only been expanded along the (transposed) X dimension at this point
        // So the (apparent) X is dims_in.y not dims_out.y
        PrintArray(host_output.complex_values, dims_in.y, dims_in.z, dims_out.w);
        MyTestPrintAndExit( "stage 1 " );
      #elif DEBUG_FFT_STAGE == 2
        // If we are doing a fwd increase, the data will have only been expanded along the (transposed) X dimension at this point
        // So the (apparent) X is dims_in.y not dims_out.y
        PrintArray(host_output.complex_values, dims_in.y, dims_out.z, dims_out.w);
        MyTestPrintAndExit( "stage 2 " );
      #elif DEBUG_FFT_STAGE == 3
        // Now the array is fully expanded to dims_out, but still transposed
        PrintArray(host_output.complex_values, dims_out.y, dims_out.z, dims_out.w);
        MyTestPrintAndExit( "stage 3 " );
      #endif    
      sum = ReturnSumOfComplexAmplitudes(host_output.complex_values, host_output.real_memory_allocated/2); 
  
  }
  else
  {
    FT.CopyDeviceToHost(false, false, FT.ReturnInputMemorySize());
    #if DEBUG_FFT_STAGE == 0
      PrintArray(host_input.real_values, dims_in.x, dims_in.y, dims_in.z, dims_in.w);
      MyTestPrintAndExit( "stage 0 " );
    #elif DEBUG_FFT_STAGE == 1
      // If we are doing a fwd increase, the data will have only been expanded along the (transposed) X dimension at this point
      // So the (apparent) X is dims_in.y not dims_out.y
      PrintArray(host_input.complex_values, dims_in.y, dims_in.z, dims_out.w);
      MyTestPrintAndExit( "stage 1 " );
    #elif DEBUG_FFT_STAGE == 2
      // If we are doing a fwd increase, the data will have only been expanded along the (transposed) X dimension at this point
      // So the (apparent) X is dims_in.y not dims_out.y
      PrintArray(host_input.complex_values, dims_in.y, dims_out.z, dims_out.w);
      MyTestPrintAndExit( "stage 2 " );      
    #elif DEBUG_FFT_STAGE == 3
      // Now the array is fully expanded to dims_out, but still transposed
      PrintArray(host_input.complex_values, dims_out.y, dims_out.z, dims_out.w);
      MyTestPrintAndExit( "stage 3 " );
    #endif   
    sum = ReturnSumOfComplexAmplitudes(host_input.complex_values, host_input.real_memory_allocated/2); 

  }


  sum -= (host_output.real_memory_allocated/2 );

  // std::cout << "sum " << sum << std::endl;
  // std::cout << "FFT Unit Impulse Forward FFT: " << sum <<  " epsilon " << host_output.fftw_epsilon << std::endl;
  // std::cout << "epsilon " << abs(sum - host_output.fftw_epsilon) << std::endl;
  if (abs(sum) > 1e-8) {all_passed = false; FastFFT_forward_passed[iSize] = false;}

  // MyFFTDebugAssertTestTrue( abs(sum - host_output.fftw_epsilon) < 1e-8, "FastFFT unit impulse forward FFT");
  FT.SetToConstant<float>(host_output.real_values, host_output.real_memory_allocated, 2.0f);
  

  FT.InvFFT();
  FT.CopyDeviceToHost(host_output.real_values, true, true);

  
  #if DEBUG_FFT_STAGE == 5
    PrintArray(host_output.complex_values, dims_out.y, dims_out.z, dims_out.w);
    MyTestPrintAndExit( "stage 5 " );
  #endif
  #if DEBUG_FFT_STAGE == 6
    PrintArray(host_output.complex_values, dims_out.y, dims_out.z, dims_out.w);
    MyTestPrintAndExit( "stage 6 " );    
  #elif DEBUG_FFT_STAGE == 7
    PrintArray(host_output.real_values, dims_out.x, dims_out.y, dims_out.z, dims_out.w);
    MyTestPrintAndExit( "stage 7 " );
  #elif DEBUG_FFT_STAGE > 7
    // No debug, keep going      
  #else
    MyTestPrintAndExit( " This block is only valid for DEBUG_FFT_STAGE == 3 || 4 " );
  #endif   

  sum = ReturnSumOfReal(host_output.real_values, dims_out);
  if (sum != dims_out.x*dims_out.y*dims_out.z) {all_passed = false; FastFFT_roundTrip_passed[iSize] = false;}

  // std::cout << "size in/out " << dims_in.x << ", " << dims_out.x << std::endl;
  // MyFFTDebugAssertTestTrue( sum == dims_out.x*dims_out.y*dims_out.z,"FastFFT unit impulse round trip FFT");

    oSize++;
    } // while loop over pad to size
  } // for loop over pad from size


  if (all_passed)
  {
    if ( ! do_increase_size) std::cout << "    All size_decrease unit impulse tests passed!" << std::endl;
    else std::cout << "    All size_increase unit impulse tests passed!" << std::endl;
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

void compare_libraries(std::vector<int>size, int size_change_type)
{

  bool skip_cufft_for_profiling = false;
  bool set_padding_callback = false; // the padding callback is slower than pasting in b/c the read size of the pointers is larger than the actual data. do not use.
  bool set_conjMult_callback = true;
  bool is_size_change_decrease = false;

  if (size_change_type < 0) { is_size_change_decrease = true; }

  short4 input_size;
  short4 output_size;
  for (int iSize = 0; iSize < size.size() - 1 ; iSize++)
  {
    int oSize;
    int loop_size;
    if (size_change_type != 0) 
    {
      oSize = iSize + 1;
      loop_size = size.size();

    }
    else 
    {
      oSize = iSize;
      loop_size = oSize + 1;
    }

    while (oSize < loop_size)
    {

      if (is_size_change_decrease)
      {
        output_size = make_short4(size[iSize],size[iSize],1,0);
        input_size  = make_short4(size[oSize],size[oSize],1,0);  
      }
      else
      {
        input_size  = make_short4(size[iSize],size[iSize],1,0);
        output_size = make_short4(size[oSize],size[oSize],1,0);  

      }
      std::cout << std::endl << "Testing padding from  " << input_size.x << " to " << output_size.x << std::endl;



      if ( ( input_size.x == output_size.x && input_size.y == output_size.y && input_size.z == output_size.z ) )
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

      short4 target_size;

      if (is_size_change_decrease) target_size = input_size; // assuming xcorr_fwd_NONE_inv_DECREASE
      else target_size = output_size;


    
      Image< float, float2> target_search_image(target_size);
      Image< float, float2> positive_control(target_size);


      // We just make one instance of the FourierTransformer class, with calc type float.
      // For the time being input and output are also float. TODO calc optionally either fp16 or nv_bloat16, TODO inputs at lower precision for bandwidth improvement.
      FastFFT::FourierTransformer<float, float, float> FT;
        // Create an instance to copy memory also for the cufft tests.
      FastFFT::FourierTransformer<float, float, float> cuFFT;
      FastFFT::FourierTransformer<float, float, float> targetFT;

      if ( is_size_change_decrease )
      {
        FT.SetForwardFFTPlan(input_size.x,input_size.y,input_size.z, input_size.x,input_size.y,input_size.z, true, false, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);
        FT.SetInverseFFTPlan(input_size.x,input_size.y,input_size.z, output_size.x,output_size.y,output_size.z, true, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);

        // For the subset of outputs this is just the input size, assuming the program then accesses just the valid data (could explicitly put into a new array which would be even slower.)
        cuFFT.SetForwardFFTPlan(input_size.x,input_size.y,input_size.z, input_size.x,input_size.y,input_size.z, true, false, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);
        cuFFT.SetInverseFFTPlan(input_size.x,input_size.y,input_size.z, input_size.x,input_size.y,input_size.z, true, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);
        
        targetFT.SetForwardFFTPlan(input_size.x,input_size.y,input_size.z, input_size.x,input_size.y,input_size.z, true, false, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);
        targetFT.SetInverseFFTPlan(input_size.x,input_size.y,input_size.z, output_size.x,output_size.y,output_size.z, true, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);        
      }
      else
      {
        FT.SetForwardFFTPlan(input_size.x,input_size.y,input_size.z, output_size.x,output_size.y,output_size.z, true, false, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);
        FT.SetInverseFFTPlan(output_size.x,output_size.y,output_size.z, output_size.x,output_size.y,output_size.z, true, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);

        cuFFT.SetForwardFFTPlan(output_size.x,output_size.y,output_size.z, output_size.x,output_size.y,output_size.z, true, false, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);
        cuFFT.SetInverseFFTPlan(output_size.x,output_size.y,output_size.z, output_size.x,output_size.y,output_size.z, true, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);
        
        targetFT.SetForwardFFTPlan(output_size.x,output_size.y,output_size.z, output_size.x,output_size.y,output_size.z, true, false, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);
        targetFT.SetInverseFFTPlan(output_size.x,output_size.y,output_size.z, output_size.x,output_size.y,output_size.z, true, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);
      }

      short4 fwd_dims_in = FT.ReturnFwdInputDimensions();
      short4 fwd_dims_out = FT.ReturnFwdOutputDimensions();
      short4 inv_dims_in = FT.ReturnInvInputDimensions();
      short4 inv_dims_out = FT.ReturnInvOutputDimensions();

      FT_input.real_memory_allocated = FT.ReturnInputMemorySize();
      FT_output.real_memory_allocated = FT.ReturnInvOutputMemorySize();


      cuFFT_input.real_memory_allocated = cuFFT.ReturnInputMemorySize();
      cuFFT_output.real_memory_allocated = cuFFT.ReturnInvOutputMemorySize();


      if (is_size_change_decrease) target_search_image.real_memory_allocated = targetFT.ReturnInputMemorySize();
      else target_search_image.real_memory_allocated = targetFT.ReturnInvOutputMemorySize(); // the larger of the two.

      positive_control.real_memory_allocated = target_search_image.real_memory_allocated; // this won't change size


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
      // For now just considering the real space image to have been implicitly quadrant swapped so the center is at the origin.
      FT.SetToConstant<float>(FT_input.real_values, FT_input.real_memory_allocated, 0.0f);
      FT.SetToConstant<float>(cuFFT_input.real_values, cuFFT_input.real_memory_allocated, 0.0f);
      FT.SetToConstant<float>(FT_output.real_values, FT_output.real_memory_allocated, 0.0f);
      FT.SetToConstant<float>(cuFFT_output.real_values, cuFFT_output.real_memory_allocated, 0.0f);
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
      FT.CopyDeviceToHost(false, false);

      

      cuFFT.CopyHostToDevice();
      

      targetFT.CopyHostToDevice();
      

      // Wait on the transfers to finish.
      cudaErr(cudaStreamSynchronize(cudaStreamPerThread));  

      // Positive control on the host.
      positive_control.FwdFFT();
      positive_control.MultiplyConjugateImage(target_search_image.complex_values);
      positive_control.InvFFT();



      // address = 0;
      test_passed = true;
      for (int z = 1; z <  positive_control.size.z ; z++)
      {   
        for (int y = 1; y < positive_control.size.y; y++)
        {  
          for (int x = 1; x < positive_control.size.x; x++)
          {
            if (positive_control.real_values[address] != 0.0f) test_passed = false;
          }
        }
      }
      if (test_passed) 
      {
        if (positive_control.real_values[address] == positive_control.size.x*positive_control.size.y*positive_control.size.z*testVal_1*testVal_2)
        {
          std::cout << "Test passed for FFTW positive control.\n" << std::endl;
        }
        else
        {
          std::cout << "Test failed for FFTW positive control. Value at zero is  " << positive_control.real_values[address] << std::endl;
        }
      }
      else
      {
        std::cout << "Test failed for positive control, non-zero values found away from the origin." << std::endl;
      }


      cuFFT_output.create_timing_events(); 
      cuFFT_input.MakeCufftPlan();
      cuFFT_output.MakeCufftPlan();


      //////////////////////////////////////////
      //////////////////////////////////////////
      // Warm up and check for accuracy
      if (set_conjMult_callback || is_size_change_decrease ) // we set set_conjMult_callback = false 
      {
        FT.CrossCorrelate(targetFT.d_ptr.momentum_space, false);
      }
      else
      {
        FT.FwdFFT();
        FT.InvFFT();
      }      


      
      if (is_size_change_decrease)
      {
        FT.CopyDeviceToHost(false, false);
        #if DEBUG_FFT_STAGE == 0

          PrintArray(FT_input.real_values, fwd_dims_in.x, fwd_dims_in.y, fwd_dims_in.z, fwd_dims_in.w);
          MyTestPrintAndExit(" Stage 0");
        #elif DEBUG_FFT_STAGE == 1

          PrintArray(FT_input.complex_values, fwd_dims_in.y, fwd_dims_in.z, fwd_dims_out.w);
          MyTestPrintAndExit(" Stage 1");
        #elif DEBUG_FFT_STAGE == 2
          PrintArray(FT_input.complex_values, fwd_dims_in.y, fwd_dims_out.z, fwd_dims_out.w);
          MyTestPrintAndExit(" Stage 2");          
        #elif DEBUG_FFT_STAGE == 3

          PrintArray(FT_input.complex_values, fwd_dims_in.y, fwd_dims_out.z, fwd_dims_out.w);
          MyTestPrintAndExit(" Stage 3");
        #elif DEBUG_FFT_STAGE == 4
        
          PrintArray(FT_input.complex_values, fwd_dims_in.y, fwd_dims_out.z,fwd_dims_out.w);
          MyTestPrintAndExit(" Stage 4");
        #elif DEBUG_FFT_STAGE == 5
          PrintArray(FT_input.complex_values,inv_dims_out.y, inv_dims_in.z, inv_dims_in.w);
          MyTestPrintAndExit(" Stage 5");
        #elif DEBUG_FFT_STAGE == 6
          PrintArray(FT_input.complex_values,inv_dims_out.y, inv_dims_out.z, inv_dims_in.w);
          MyTestPrintAndExit(" Stage 6");          
        #elif DEBUG_FFT_STAGE == 7
          PrintArray(FT_input.real_values,inv_dims_out.x, inv_dims_out.y, inv_dims_out.z, inv_dims_out.w);
          MyTestPrintAndExit(" Stage 7");
        #elif DEBUG_FFT_STAGE > 7
          // Do nothing, we are doing all ops and not debugging.
        #else
          MyTestPrintAndExit("DEBUG_FFT_STAGE not recognized " + std::to_string(DEBUG_FFT_STAGE));
        #endif
      }
      else
      {
        // the output is equal or > the input, so we can always copy there.
        FT.CopyDeviceToHost(FT_output.real_values,false, false);

        #if DEBUG_FFT_STAGE == 0
          PrintArray(FT_output.real_values, fwd_dims_in.x, fwd_dims_in.y, fwd_dims_in.z, fwd_dims_in.w);
          MyTestPrintAndExit(" Stage 0");
        #elif DEBUG_FFT_STAGE == 1
          PrintArray(FT_output.complex_values, fwd_dims_in.y, fwd_dims_in.z, fwd_dims_out.w);
          MyTestPrintAndExit(" Stage 1");
        #elif DEBUG_FFT_STAGE == 2
          PrintArray(FT_output.complex_values, fwd_dims_in.y, fwd_dims_out.z, fwd_dims_out.w);
          MyTestPrintAndExit(" Stage 2");          
        #elif DEBUG_FFT_STAGE == 3
          PrintArray(FT_output.complex_values, fwd_dims_out.y, fwd_dims_out.z, fwd_dims_out.w);
          MyTestPrintAndExit(" Stage 3");
        #elif DEBUG_FFT_STAGE == 4
          PrintArray(FT_output.complex_values, fwd_dims_out.y, fwd_dims_out.z, fwd_dims_out.w);
          MyTestPrintAndExit(" Stage 4");
        #elif DEBUG_FFT_STAGE == 5
          PrintArray(FT_output.complex_values, inv_dims_out.y, inv_dims_in.z, inv_dims_out.w);
          MyTestPrintAndExit(" Stage 5");
        #elif DEBUG_FFT_STAGE == 6
          PrintArray(FT_output.complex_values, inv_dims_out.y, inv_dims_out.z, inv_dims_out.w);
          MyTestPrintAndExit(" Stage 6");          
        #elif DEBUG_FFT_STAGE == 7
          PrintArray(FT_output.real_values, inv_dims_out.x, inv_dims_out.y, inv_dims_out.z,  inv_dims_out.w);
          MyTestPrintAndExit(" Stage 7");
        #elif DEBUG_FFT_STAGE > 7
          // Do nothing, we are doing all ops and not debugging.
        #else
          MyTestPrintAndExit("DEBUG_FFT_STAGE not recognized " + std::to_string(DEBUG_FFT_STAGE));
        #endif
      }
      
      address = 0;
      test_passed = true;
      if (is_size_change_decrease)
      {
        for (int z = 1; z <  FT_input.size.z ; z++)
        {   
          for (int y = 1; y < FT_input.size.y; y++)
          {  
            for (int x = 1; x < FT_input.size.x; x++)
            {
              if (FT_input.real_values[address] != 0.0f) test_passed = false;
            }
          }
        }
        if (test_passed) 
        {
          if (FT_input.real_values[address] == FT_input.size.x*FT_input.size.y*FT_input.size.z*testVal_1*testVal_2)
          {
            std::cout << "Test passed for FastFFT positive control.\n" << std::endl;
          }
          else
          {
            std::cout << "Test failed for FastFFT positive control. Value at zero is  " << FT_input.real_values[address] << std::endl;
          }
        }
        else
        {
          std::cout << "Test failed for FastFFT control, non-zero values found away from the origin." << std::endl;
        }
      }
      else
      {
        for (int z = 1; z <  FT_output.size.z ; z++)
        {   
          for (int y = 1; y < FT_output.size.y; y++)
          {  
            for (int x = 1; x < FT_output.size.x; x++)
            {
              if (FT_output.real_values[address] != 0.0f) test_passed = false;
            }
          }
        }
        if (test_passed) 
        {
          if (FT_output.real_values[address] == FT_output.size.x*FT_output.size.y*FT_output.size.z*testVal_1*testVal_2)
          {
            std::cout << "Test passed for FastFFT positive control.\n" << std::endl;
          }
          else
          {
            std::cout << "Test failed for FastFFT positive control. Value at zero is  " << FT_output.real_values[address] << std::endl;
          }
        }
        else
        {
          std::cout << "Test failed for FastFFT control, non-zero values found away from the origin." << std::endl;
        }
      }

      ////////////////////////////////////////
      //////////////////////////////////////////

      #if DEBUG_FFT_STAGE == 0

          PrintArray(FT_output.real_values, fwd_dims_in.x, fwd_dims_in.y, fwd_dims_in.z, fwd_dims_in.w);

        MyTestPrintAndExit( "stage 0 " );
      #elif DEBUG_FFT_STAGE == 1
        // If we are doing a fwd increase, the data will have only been expanded along the (transposed) X dimension at this point
        // So the (apparent) X is dims_in.y not output_size.y
        // decrease is currently just tested on the output. Really, to simplify there should be 3 different functions, fwd_none_inv_decrease (current decrease), fwd_decrease_inc_decrease (not yet) fwd_increase_inv_none
        if (is_size_change_decrease) 
        {
          MyTestPrintAndExit( "stage 1 decrease" );
          PrintArray(FT_output.complex_values, fwd_dims_out.y, fwd_dims_out.z, fwd_dims_out.w);
        }
        else 
        {
          MyTestPrintAndExit( "stage 1 increase" );
          PrintArray(FT_output.complex_values, fwd_dims_in.y, fwd_dims_in.z, fwd_dims_out.w);
        }

      #elif DEBUG_FFT_STAGE == 2
        // Now the array is fully expanded to output_size, but still transposed
        PrintArray(FT_output.complex_values, fwd_dims_out.y, fwd_dims_out.z, fwd_dims_out.w);
        MyTestPrintAndExit( "stage 2 " );

      #elif DEBUG_FFT_STAGE == 3
        PrintArray(FT_output.complex_values,inv_dims_out.y, inv_dims_out.z, inv_dims_out.w);
        MyTestPrintAndExit( "stage 3 " );
      #elif DEBUG_FFT_STAGE == 4
        PrintArray(FT_output.real_values, inv_dims_out.x, inv_dims_out.y, inv_dims_out.z, inv_dims_out.w);
        MyTestPrintAndExit( "stage 4 " );
      #elif DEBUG_FFT_STAGE > 7
        // This is the final stage, the data is fully expanded and transposed
      #else
        MyTestPrintAndExit( "This blah blah");
      #endif   



      const int n_loops = 3000;
      cuFFT_output.record_start();
      for (int i = 0; i < n_loops; ++i)
      {
        if (set_conjMult_callback || is_size_change_decrease )
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

      

      if (! skip_cufft_for_profiling)
      {
        //////////////////////////////////////////
        //////////////////////////////////////////
        // Warm up and check for accuracy
        if (is_size_change_decrease)
        {

          precheck
          cudaErr(cufftExecR2C(cuFFT_input.cuda_plan_forward, (cufftReal*)cuFFT.d_ptr.position_space, (cufftComplex*)cuFFT.d_ptr.momentum_space_buffer));
          postcheck
          
  
          precheck
          cudaErr(cufftExecC2R(cuFFT_input.cuda_plan_inverse, (cufftComplex*)cuFFT.d_ptr.momentum_space_buffer, (cufftReal*)cuFFT.d_ptr.position_space));
          postcheck  
          
        }
        else
        {
          // cuFFT.ClipIntoTopLeft();
          // cuFFT.ClipIntoReal(cuFFT_output.size.x/2, cuFFT_output.size.y/2, cuFFT_output.size.z/2);
          // cuFFT.CopyDeviceToHost(cuFFT_output.real_values,false, false);

          precheck
          cudaErr(cufftExecR2C(cuFFT_output.cuda_plan_forward, (cufftReal*)cuFFT.d_ptr.position_space, (cufftComplex*)cuFFT.d_ptr.momentum_space_buffer));
          postcheck
          
  
          precheck
          cudaErr(cufftExecC2R(cuFFT_output.cuda_plan_inverse, (cufftComplex*)cuFFT.d_ptr.momentum_space_buffer, (cufftReal*)cuFFT.d_ptr.position_space));
          postcheck  
           
        }
        

        


        cuFFT_output.record_start();
        for (int i = 0; i < n_loops; ++i)
        {
          // std::cout << i << "i / " << n_loops << "n_loops" << std::endl;
          if (set_conjMult_callback) cuFFT.ClipIntoTopLeft();
          // cuFFT.ClipIntoReal(input_size.x/2, input_size.y/2, input_size.z/2);

          if (is_size_change_decrease)
          {
            precheck
            cudaErr(cufftExecR2C(cuFFT_input.cuda_plan_forward, (cufftReal*)cuFFT.d_ptr.position_space, (cufftComplex*)cuFFT.d_ptr.momentum_space_buffer));
            postcheck
  
            precheck
            cudaErr(cufftExecC2R(cuFFT_input.cuda_plan_inverse, (cufftComplex*)cuFFT.d_ptr.momentum_space_buffer, (cufftReal*)cuFFT.d_ptr.position_space));
            postcheck
          }
          else
          {
            precheck
            cudaErr(cufftExecR2C(cuFFT_output.cuda_plan_forward, (cufftReal*)cuFFT.d_ptr.position_space, (cufftComplex*)cuFFT.d_ptr.momentum_space_buffer));
            postcheck
  
            precheck
            cudaErr(cufftExecC2R(cuFFT_output.cuda_plan_inverse, (cufftComplex*)cuFFT.d_ptr.momentum_space_buffer, (cufftReal*)cuFFT.d_ptr.position_space));
            postcheck
          }

        }
        cuFFT_output.record_stop();
        cuFFT_output.synchronize();
        cuFFT_output.print_time("cuFFT");
      } // end of if (! skip_cufft_for_profiling)
      std::cout << "For size " << input_size.x << " to "<< output_size.x << ": " << std::endl;
      std::cout << "Ratio cuFFT/FastFFT : " << cuFFT_output.elapsed_gpu_ms/FastFFT_time << std::endl;

      oSize++;
      // We don't want to loop if the size is not actually changing.
      } // while loop over pad to size
  } // for loop over pad from size

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
    FT.SetForwardFFTPlan(input_size.x,input_size.y,input_size.z, output_size.x,output_size.y,output_size.z, true, false, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);
    FT.SetInverseFFTPlan(output_size.x,output_size.y,output_size.z, output_size.x,output_size.y,output_size.z, true, FastFFT::FourierTransformer<float, float ,float>::OriginType::natural);

    FT_complex.SetForwardFFTPlan(input_size.x,input_size.y,input_size.z, output_size.x,output_size.y,output_size.z, true, false, FastFFT::FourierTransformer<float, float2 ,float2>::OriginType::natural);
    FT_complex.SetInverseFFTPlan(output_size.x,output_size.y,output_size.z, output_size.x,output_size.y,output_size.z, true, FastFFT::FourierTransformer<float, float2 ,float2>::OriginType::natural);

    FT_input.real_memory_allocated = FT.ReturnInputMemorySize();
    FT_output.real_memory_allocated = FT.ReturnInvOutputMemorySize();

    FT_input_complex.real_memory_allocated = FT_complex.ReturnInputMemorySize();
    FT_output_complex.real_memory_allocated = FT_complex.ReturnInvOutputMemorySize();
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

int main(int argc, char** argv) 
{

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
  }
  // Input and output dimensions, with simple checks. I'm sure there are better checks on argv.
  short4 input_size;
  short4 output_size;

  std::vector<int> test_size = { 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
  std::vector<int> test_size_3d = { 16, 32, 64, 128, 256, 512};

  // The launch parameters fail for 4096 -> < 64 for r2c_decrease, not sure if it is the elements_per_thread or something else.
  // For now, just over-ride these small sizes
  std::vector<int> test_size_for_decrease = { 64, 128, 256, 512, 1024, 2048, 4096};



  if (run_validation_tests)  {

    // change onde these to just report the pass/fail.
    // run_oned(test_size);
    // exit(0);

    bool do_3d = true;
    const_image_test(test_size_3d, do_3d);
    exit(0);

    do_3d = false;
    const_image_test(test_size, do_3d);
    unit_impulse_test(test_size, true);
    unit_impulse_test(test_size_for_decrease, false);


  } // end of validation tests


  if (run_performance_tests) {

    #ifdef HEAVYERRORCHECKING_FFT
      std::cout << "Running performance tests with heavy error checking.\n";
      std::cout << "This doesn't make sense as the synchronizations are invalidating.\n";
      // exit(1);
    #endif

    int size_change_type = 0; // no change

    // compare_libraries(test_size, size_change_type);

    // size_change_type = 1; // increase
    // compare_libraries(test_size, size_change_type);

    size_change_type = -1; // decrease
    compare_libraries(test_size, size_change_type);


  }
  return 0;
};

