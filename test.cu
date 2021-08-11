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

    // Pointers to the arrays on the host -- maybe make this a struct of some sort? I'm sure there is a parallel in cuda, look into cuarray/texture code
  float* host_input;
  float* host_output;
  float2* host_input_complex;
  float2* host_output_complex;
  int host_input_memory_allocated;
  int host_output_memory_allocated;
  
  
  // Pointers to the arrays on the device
  float* device_input;
  float* device_output;
  float2* device_input_complex;
  float2* device_output_complex;
  int device_memory_allocated;
  int padding_jump_value;

  float fftw_epsilon;
  float sum;
  float2 sum_complex;
  
  if (output_size.x % 2 == 0) padding_jump_value = 2;
  else padding_jump_value = 1;
  // We just make one instance of the FourierTransformer class, with calc type float.
  // For the time being input and output are also float. TODO calc optionally either fp16 or nv_bloat16, TODO inputs at lower precision for bandwidth improvement.
  FastFFT::FourierTransformer FT(FastFFT::FourierTransformer::DataType::fp32);
  
  
  // Determine how much memory we need, working with FFTW/CUDA style in place transform padding.
  // Note: there is no reason we really need this, because the xforms will always be out of place. 
  //       For now, this is just in place because all memory in cisTEM is allocated accordingly.
  host_input_memory_allocated = FT.ReturnPaddedMemorySize(input_size);
  host_output_memory_allocated = FT.ReturnPaddedMemorySize(output_size);
  
  // On the device, we will always allocate enough memory for the larger of input/output including the buffer array.
  // Minmize the number of calls to malloc which are slow and can lead to fragmentation.
  device_memory_allocated = std::max(host_input_memory_allocated, host_output_memory_allocated);
  
  
  // In your own programs, you will be handling this memory allocation yourself. We'll just make something here.
  // I think fftwf_malloc may potentially create a different alignment than new/delete, but kinda doubt it. For cisTEM consistency...
  host_input = (float *) fftwf_malloc(sizeof(float) * host_input_memory_allocated);
  host_input_complex = (float2*) host_input;  // Set the complex_values to point at the newly allocated real values;

  host_output = (float *) fftwf_malloc(sizeof(float) * host_output_memory_allocated);
  host_output_complex = (float2*) host_output;  // Set the complex_values to point at the newly allocated real values;


  // Make FFTW plans for comparing CPU to GPU xforms.
  // This is nearly verbatim from cisTEM::Image::Allocate - I do not know if FFTW_ESTIMATE is the best option.
  // In cisTEM we almost always use MKL, so this might be worth testing. I always used exhaustive in Matlab/emClarity.
  fftwf_plan plan_fwd = NULL;
  fftwf_plan plan_bwd = NULL;
  plan_fwd = fftwf_plan_dft_r2c_3d(output_size.z, output_size.y, output_size.x, host_input, reinterpret_cast<fftwf_complex*>(host_input_complex), FFTW_ESTIMATE);
  plan_bwd = fftwf_plan_dft_c2r_3d(output_size.z, output_size.y, output_size.x, reinterpret_cast<fftwf_complex*>(host_input_complex), host_input, FFTW_ESTIMATE);
  

  // This is similar to creating an FFT/CUFFT plan, so set these up before doing anything on the GPU
  FT.SetInputDimensionsAndType(input_size.x,input_size.y,input_size.z,true, false,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);
  FT.SetOutputDimensionsAndType(output_size.x,output_size.y,output_size.z,true,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);
  
  // Now we want to associate the host memory with the device memory. The method here asks if the host pointer is pinned (in page locked memory) which
  // ensures faster transfer. If false, it will be pinned for you.
  FT.SetInputPointer(&host_input[0], false);
  
  // Set a unit impulse at the center of the input array.
  FT.SetToConstant<float>(host_input, host_input_memory_allocated, 0.0f);
  FT.SetToConstant<float>(host_output, host_output_memory_allocated, 0.0f);

  host_input[ input_size.y/2 * (input_size.x+padding_jump_value) + input_size.x/2] = 1.0f;
  short4 wanted_center = make_short4(0,0,0,0);
  ClipInto(host_input, host_output, input_size ,  output_size,  wanted_center, 0.f);
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
  FT.SetToConstant<float>(host_input, host_input_memory_allocated, 0.0f);
  host_input[0] = 1.0f;

  sum = ReturnSumOfReal(host_output, output_size);
  MyFFTDebugAssertTestTrue( sum == 1,"Unit impulse Init ");
  
  // This copies the host memory into the device global memory. If needed, it will also allocate the device memory first.
  FT.CopyHostToDevice();

  // Now let's do the forward FFT on the host and check that the result is correct.
  fftwf_execute_dft_r2c(plan_fwd, host_output, reinterpret_cast<fftwf_complex*>(host_output_complex));
  
  fftw_epsilon = ReturnSumOfComplexAmplitudes(host_output_complex, host_output_memory_allocated/2);  

  fftw_epsilon -= (host_output_memory_allocated/2 );
  MyFFTDebugAssertTestTrue( std::abs(fftw_epsilon < 1e-8) , "FFTW unit impulse forward FFT");
  
  // Just to make sure we don't get a false positive, set the host memory to some undesired value.
  FT.SetToConstant<float>(host_output, host_output_memory_allocated, 2.0f);
  
  // This method will call the regular FFT kernels given the input/output dimensions are equal when the class is instantiated.
  bool swap_real_space_quadrants = true;
  FT.FwdFFT(swap_real_space_quadrants);
  
  // do not deallocate, do not unpin memory

  FT.CopyDeviceToHost(host_output, false, false);

  // for (int i = 0; i < host_output_memory_allocated/2; i++){ 
  //   if (host_output_complex[i].x != 0.0f || host_output_complex[i].y != 0.0f){ std::cout << "Value at " << i <<  " is " << host_output_complex[i].x << " " << host_output_complex[i].y << std::endl; }
  // // }
  // for (int x = 0; x < 128; x++)
  // {
  //   int n=0;
  //   std::cout << x << "[ ";
  //   for (int y = 0; y < 65; y++)
  //   {  
  //     std::cout << host_output_complex[x + y*128].x << "," << host_output_complex[x + y*128].y << " ";
  //     if (host_output_complex[x + y*128].x == 3) { n++;}
  //   }
  //   std::cout << "] " << n << std::endl;
  // }

  sum = ReturnSumOfComplexAmplitudes(host_output_complex, host_output_memory_allocated/2); 
  sum -= (host_output_memory_allocated/2 );


  
  MyFFTDebugAssertTestTrue( abs(sum - fftw_epsilon) < 1e-8, "FastFFT unit impulse forward FFT");
  FT.SetToConstant<float>(host_output, host_output_memory_allocated, 2.0f);
  

  FT.InvFFT();
  FT.CopyDeviceToHost(host_output, true, true);
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
  sum = ReturnSumOfReal(host_output, output_size);
  MyFFTDebugAssertTestTrue( sum == output_size.x*output_size.y*output_size.z,"FastFFT unit impulse round trip FFT");
  

  fftwf_destroy_plan(plan_fwd);
  fftwf_destroy_plan(plan_bwd);
  fftwf_free(host_output);
  fftwf_free(host_input);


}

void compare_libraries(short4 input_size, short4 output_size)
{


  bool test_passed = true;
  long address = 0;

    // Pointers to the arrays on the host -- maybe make this a struct of some sort? I'm sure there is a parallel in cuda, look into cuarray/texture code
  float* host_input_ft;
  float* host_input_cufft;
  float2* host_input_ft_complex;
  float2* host_input_cufft_complex;
  int host_input_memory_allocated;
  int host_output_memory_allocated;
  
  
  // Pointers to the arrays on the device
  float* device_input;
  float* device_output;
  float2* device_input_complex;
  float2* device_output_complex;
  int device_memory_allocated;
  int padding_jump_value;

  float fftw_epsilon;
  float sum;
  float2 sum_complex;
  
  if (output_size.x % 2 == 0) padding_jump_value = 2;
  else padding_jump_value = 1;

  cufftHandle cuda_plan_forward;
	cufftHandle cuda_plan_inverse;
	size_t	cuda_plan_worksize_forward;
	size_t	cuda_plan_worksize_inverse;

  cudaEvent_t startEvent { nullptr };
  cudaEvent_t stopEvent { nullptr };
  float       elapsed_gpu_ms {};


  cudaErr(cudaEventCreate( &startEvent, cudaEventBlockingSync));
  cudaErr(cudaEventCreate( &stopEvent, cudaEventBlockingSync ));

   
  std::cout << "Allocating a 2d cufft Plan" << std::endl;

  cudaErr(cufftCreate(&cuda_plan_forward));
  cudaErr(cufftCreate(&cuda_plan_inverse));

  cudaErr(cufftSetStream(cuda_plan_forward, cudaStreamPerThread));
  cudaErr(cufftSetStream(cuda_plan_inverse, cudaStreamPerThread));

  int rank = 2; int iBatch = 1;
  long long int* fftDims = new long long int[rank];
  long long int*inembed = new long long int[rank];
  long long int*onembed = new long long int[rank];

  fftDims[0] = output_size.y;
  fftDims[1] = output_size.x;

  inembed[0] = output_size.y;
  inembed[1] = output_size.w;

  onembed[0] = output_size.y;
  onembed[1] = output_size.w;

  cudaErr(cufftXtMakePlanMany(cuda_plan_forward, rank, fftDims,
    NULL, NULL, NULL, CUDA_R_32F,
    NULL, NULL, NULL, CUDA_C_32F, iBatch, &cuda_plan_worksize_forward, CUDA_C_32F));
    cudaErr(cufftXtMakePlanMany(cuda_plan_inverse, rank, fftDims,
    NULL, NULL, NULL, CUDA_C_32F,
    NULL, NULL, NULL, CUDA_R_32F, iBatch, &cuda_plan_worksize_inverse, CUDA_R_32F));

    delete [] fftDims;
    delete [] inembed;
    delete [] onembed;

    // cudaErr(cufftExecR2C(this->cuda_plan_forward, (cufftReal*)real_values_gpu, (cufftComplex*)complex_values_gpu));
    // cudaErr(cufftExecC2R(this->cuda_plan_inverse, (cufftComplex*)complex_values_gpu, (cufftReal*)real_values_gpu));

 // We just make one instance of the FourierTransformer class, with calc type float.
  // For the time being input and output are also float. TODO calc optionally either fp16 or nv_bloat16, TODO inputs at lower precision for bandwidth improvement.
  FastFFT::FourierTransformer FT(FastFFT::FourierTransformer::DataType::fp32);
    // Create an instance to copy memory also for the cufft tests.
  FastFFT::FourierTransformer cuFFT(FastFFT::FourierTransformer::DataType::fp32);

  
  // Determine how much memory we need, working with FFTW/CUDA style in place transform padding.
  // Note: there is no reason we really need this, because the xforms will always be out of place. 
  //       For now, this is just in place because all memory in cisTEM is allocated accordingly.
  host_input_memory_allocated = FT.ReturnPaddedMemorySize(input_size);
  host_output_memory_allocated = FT.ReturnPaddedMemorySize(output_size);
  
  // On the device, we will always allocate enough memory for the larger of input/output including the buffer array.
  // Minmize the number of calls to malloc which are slow and can lead to fragmentation.
  device_memory_allocated = std::max(host_input_memory_allocated, host_output_memory_allocated);
  
  
  // In your own programs, you will be handling this memory allocation yourself. We'll just make something here.
  // I think fftwf_malloc may potentially create a different alignment than new/delete, but kinda doubt it. For cisTEM consistency...
  host_input_ft = (float *) fftwf_malloc(sizeof(float) * host_input_memory_allocated);
  host_input_ft_complex = (float2*) host_input_ft;  // Set the complex_values to point at the newly allocated real values;

  host_input_cufft = (float *) fftwf_malloc(sizeof(float) * host_output_memory_allocated);
  host_input_cufft_complex = (float2*) host_input_cufft;  // Set the complex_values to point at the newly allocated real values;

  

  // This is similar to creating an FFT/CUFFT plan, so set these up before doing anything on the GPU
  FT.SetInputDimensionsAndType(input_size.x,input_size.y,input_size.z,true, false,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);
  FT.SetOutputDimensionsAndType(output_size.x,output_size.y,output_size.z,true,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);
  cuFFT.SetInputDimensionsAndType(output_size.x,output_size.y,output_size.z,true, false,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);
  cuFFT.SetOutputDimensionsAndType(output_size.x,output_size.y,output_size.z,true,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);  
  // Now we want to associate the host memory with the device memory. The method here asks if the host pointer is pinned (in page locked memory) which
  // ensures faster transfer. If false, it will be pinned for you.
  FT.SetInputPointer(&host_input_ft[0], false);
  cuFFT.SetInputPointer(&host_input_cufft[0], false);

  // Set a unit impulse at the center of the input array.
  FT.SetToConstant<float>(host_input_ft, host_input_memory_allocated, 1.0f);
  FT.SetToConstant<float>(host_input_cufft, host_output_memory_allocated, 1.0f);


  
  // This copies the host memory into the device global memory. If needed, it will also allocate the device memory first.
  FT.CopyHostToDevice();
  cuFFT.CopyHostToDevice();




  const int n_loops = 10000;
  cudaEventRecord( startEvent );
  for (int i = 0; i < n_loops; ++i)
  {
    FT.FwdFFT();
    FT.InvFFT();
  }
  cudaEventRecord( stopEvent );
  cudaEventSynchronize( stopEvent );
  cudaErr(cudaEventElapsedTime( &elapsed_gpu_ms, startEvent, stopEvent ));
  std::printf( "Time on FastFFT %0.2f ms\n", elapsed_gpu_ms );

  cudaEventRecord( startEvent );
  for (int i = 0; i < n_loops; ++i)
  {
    cudaErr(cufftExecR2C(cuda_plan_forward, (cufftReal*)cuFFT.device_pointer_fp32, (cufftComplex*)cuFFT.device_pointer_fp32_complex));
    cudaErr(cufftExecC2R(cuda_plan_inverse, (cufftComplex*)cuFFT.device_pointer_fp32_complex, (cufftReal*)cuFFT.device_pointer_fp32));
  }
  cudaEventRecord( stopEvent );
  cudaEventSynchronize( stopEvent );
  cudaErr(cudaEventElapsedTime( &elapsed_gpu_ms, startEvent, stopEvent ));
  std::printf( "Time on cuFFT %0.2f ms\n", elapsed_gpu_ms );

}

int main(int argc, char** argv) {

  std::printf("Entering main in tests.cpp\n");
  std::printf("Standard is %i\n\n",__cplusplus);

  // Input and output dimensions, with simple checks. I'm sure there are better checks on argv.
  short4 input_size;
  short4 output_size;

  constexpr const int n_tests = 5;
  int test_size[n_tests] = {64, 128, 256, 512, 4096};
  for (int iSize = 0; iSize < n_tests; iSize++) {

    std::cout << std::endl << "Testing " << test_size[iSize] << " x" << std::endl;
    input_size = make_short4(test_size[iSize],test_size[iSize],1,0);
    output_size = make_short4(test_size[iSize],test_size[iSize],1,0);
    compare_libraries( input_size,  output_size);

    // const_image_test(input_size, output_size);

  }
  exit(0);

  for (int iSize = 0; iSize < n_tests - 1; iSize++) {
    int oSize = iSize + 1;
    while (oSize < n_tests)
    {
      std::cout << std::endl << "Testing padding from   " << test_size[iSize] << " to " << test_size[oSize] << std::endl;
      input_size = make_short4(test_size[iSize],test_size[iSize],1,0);
      output_size = make_short4(test_size[oSize],test_size[oSize],1,0);
  
      compare_libraries( input_size,  output_size);
      exit(0);
      // unit_impulse_test(input_size, output_size);
      oSize++;

    }


  }
  
}
