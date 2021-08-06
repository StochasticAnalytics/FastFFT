#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>

// sudo apt-get install libfftw3-dev libfftw3-doc
#include <fftw3.h>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

#include "FastFFT.cu"

// To print a message and some number n_to_print complex values to stdout
void print_values_complex(float* input, std::string msg, int n_to_print)
{
  for (int i = 0; i < n_to_print*2; i+=2) { std::cout << msg << i/2 << "  " << input[i] << " " << input[i+1] << std::endl ;}
}

// Return sum of real values
float ReturnSumOfReal(float* input, short4 size)
{
  double temp_sum = 0;
  long address = 0;
  int padding_jump_val = size.w*2 - size.x;
  for (int k = 0; k < size.z; k++)
  {
    for (int j = 0; j < size.y; j++)
    {
      for (int i = 0; i < size.x; i++)
      {
        temp_sum += (input[address]);
        address++;
      }
      address += padding_jump_val;
    }
  }

  return float(temp_sum);
}

// Return the sum of the complex values
float2 ReturnSumOfComplex(float2* input, int n_to_print)
{
  double sum_x = 0;
  double sum_y = 0;

  for (int i = 0; i < n_to_print; i++) 
  {
    sum_x += input[i].x;
    sum_y += input[i].y;
  }
  
  return make_float2(float(sum_x), float(sum_y));  
}

void ClipInto(const float* array_to_paste, float* array_to_paste_into, short4 size_from, short4 size_into, short4 wanted_center, float wanted_padding_value)
{


	long pixel_counter = 0;

	int kk;
	int k;
	int kk_logi;

	int jj;
	int jj_logi;
	int j;

	int ii;
	int ii_logi;
	int i;

	double junk;

  short4 center_to_paste_into = make_short4(size_into.x/2, size_into.y/2, size_into.z/2, 0);
  short4 center_to_paste = make_short4(size_from.x/2, size_from.y/2, size_from.z/2, 0);
  int padding_jump_value;

  if (size_into.x % 2 == 0) padding_jump_value = 2;
  else padding_jump_value = 1;

  for (kk = 0; kk < size_into.z; kk++)
  {
    kk_logi = kk - center_to_paste_into.z;
    k = center_to_paste.z + wanted_center.z + kk_logi;

    for (jj = 0; jj < size_into.y; jj++)
    {
      jj_logi = jj - center_to_paste_into.y;
      j = center_to_paste.y + wanted_center.y + jj_logi;

      for (ii = 0; ii < size_into.x; ii++)
      {
        ii_logi = ii - center_to_paste_into.x;
        i = center_to_paste.x + wanted_center.x + ii_logi;

        if (k < 0 || k >= size_from.z || j < 0 || j >= size_from.y || i < 0 || i >= size_from.x)
        {
          array_to_paste_into[pixel_counter] = wanted_padding_value;
        }
        else
        {
          array_to_paste_into[pixel_counter] = array_to_paste[ k*(size_from.w*2 *size_from.y) + j*(size_from.x+padding_jump_value) + i];
        }

        pixel_counter++;
      }

      pixel_counter+=padding_jump_value;
    }
  }
	


} // end of clip into

void unit_impulse_test(short4 input_size, short4 output_size)
{
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

  float sum;
  float2 sum_complex;


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
	host_output_complex = (float2*) host_input;  // Set the complex_values to point at the newly allocated real values;
  
  // Make FFTW plans for comparing CPU to GPU xforms.
  // This is nearly verbatim from cisTEM::Image::Allocate - I do not know if FFTW_ESTIMATE is the best option.
  // In cisTEM we almost always use MKL, so this might be worth testing. I always used exhaustive in Matlab/emClarity.
  fftwf_plan plan_fwd = NULL;
  fftwf_plan plan_bwd = NULL;
	plan_fwd = fftwf_plan_dft_r2c_3d(output_size.z, output_size.y, output_size.x, host_input, reinterpret_cast<fftwf_complex*>(host_input_complex), FFTW_ESTIMATE);
  plan_bwd = fftwf_plan_dft_c2r_3d(output_size.z, output_size.y, output_size.x, reinterpret_cast<fftwf_complex*>(host_input_complex), host_input, FFTW_ESTIMATE);
  
  // Set our input host memory to a constant. Then FFT[0] = host_input_memory_allocated
  FT.SetToConstant<float>(host_input, host_input_memory_allocated, 1.0f);
  
  // short4 wanted_center = make_short4(input_size.x/2, input_size.y/2, input_size.z/2, 0);
  short4 wanted_center = make_short4(0,0,0, 0);
  ClipInto(host_input, host_output, input_size, output_size, wanted_center, 0.0f);
  // int padding_jump_value;
  // if (output_size.x % 2 == 0) padding_jump_value = 2;
  // else padding_jump_value = 1;
  // for (int i = 0; i < output_size.x; i++) 
  // { 
  //   for (int j = 0; j < output_size.y; j++)
  //   {
  //     std::cout << host_output[i + j*(padding_jump_value+output_size.x)] << " "; 
  //   }
  //   std::cout << std::endl; 
  // }
  // exit(-1);

  
  // This is similar to creating an FFT/CUFFT plan, so set these up before doing anything on the GPU
	FT.SetInputDimensionsAndType(output_size.x,output_size.y,output_size.z,true, false,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);
	FT.SetOutputDimensionsAndType(output_size.x,output_size.y,output_size.z,true,FastFFT::FourierTransformer::DataType::fp32, FastFFT::FourierTransformer::OriginType::natural);
  
  // Now we want to associate the host memory with the device memory. The method here asks if the host pointer is pinned (in page locked memory) which
  // ensures faster transfer. If false, it will be pinned for you.
	FT.SetInputPointer(&host_output[0], false);
  sum = ReturnSumOfReal(host_output, output_size);
  MyFFTDebugAssertTestTrue( sum == input_size.x*input_size.y*input_size.z,"Unit impulse Init ");

  // This copies the host memory into the device global memory. If needed, it will also allocate the device memory first.
	FT.CopyHostToDevice();
  
  // Now let's do the forward FFT on the host and check that the result is correct.
  fftwf_execute_dft_r2c(plan_fwd, host_output, reinterpret_cast<fftwf_complex*>(host_output_complex));
  print_values_complex(host_output, "fftw ", 5);
  
  sum_complex = ReturnSumOfComplex(host_output_complex, FT.output_memory_allocated/2);
  std::cout << sum_complex.x << " " << sum_complex.y << std::endl;
  MyFFTDebugAssertTestTrue( sum_complex.x == FT.input_number_of_real_values && sum_complex.y == 0, "FFTW unit impulse forward FFT");
  FT.SetToConstant<float>(host_output, host_output_memory_allocated, 2.0f);

  // Forward FFT ;
  // FT.FFT_R2C_WithPadding_Transposed();
  // FT.FFT_C2C_WithPadding();
  FT.FwdFFT();

  // in buffer, do not deallocate, do not unpin memory
	FT.CopyDeviceToHost(false, false, false);
  sum_complex = ReturnSumOfComplex(host_output_complex, FT.output_memory_allocated/2);
  // std::cout << sum_complex.x << " " << powf(input_size.x*input_size.y*input_size.z,2) << " " << std::endl;

  // for (int i = 0; i < output_size.w*output_size.x*2; i++) { std::cout << host_input[i] << " "; }
  MyFFTDebugAssertTestTrue( sum_complex.x == FT.output_number_of_real_values && sum_complex.y == 0, "FastFFT unit impulse forward FFT");
  FT.SetToConstant<float>(host_input, host_input_memory_allocated, 2.0f);

  // FT.FFT_C2C();
  // FT.FFT_C2R_Transposed();
  FT.InvFFT();

	FT.CopyDeviceToHost(false, true, true);

  // Assuming the outputs are always even dimensions, padding_jump_val is always 2.
  sum = ReturnSumOfReal(host_input, output_size);
  // std::cout << sum << " " << powf(input_size.x*input_size.y*input_size.z,2) << " " << std::endl;
  // for (int i = 0; i < output_size.w*output_size.x*2; i++) { std::cout << host_input[i] << " "; }

  MyFFTDebugAssertTestTrue( sum == powf(input_size.x*input_size.y*input_size.z,2),"FastFFT unit impulse round trip FFT");

  fftwf_free(host_input);
  fftwf_destroy_plan(plan_fwd);
  fftwf_destroy_plan(plan_bwd);

}

int main(int argc, char** argv) {

  std::printf("Entering main in tests.cpp\n");
  std::printf("Standard is %i\n\n",__cplusplus);

  // Input and output dimensions, with simple checks. I'm sure there are better checks on argv.
  short4 input_size;
  short4 output_size;

  constexpr const int n_tests = 4;
  int test_size[n_tests] = {64, 128, 256, 512};
  // for (int iSize = 0; iSize < n_tests; iSize++) {

  //   std::cout << std::endl << "Testing " << test_size[iSize] << " x" << std::endl;
  //   input_size = make_short4(test_size[iSize],test_size[iSize],1,0);
  //   output_size = make_short4(test_size[iSize],test_size[iSize],1,0);

  //   unit_impulse_test(input_size, output_size);

  // }

  for (int iSize = 0; iSize < n_tests - 1; iSize++) {
    int oSize = iSize + 1;
    while (oSize > iSize)
    {
      std::cout << std::endl << "Testing padding from   " << test_size[iSize] << " to " << test_size[oSize] << std::endl;
      input_size = make_short4(test_size[iSize],test_size[iSize],1,0);
      output_size = make_short4(test_size[oSize],test_size[oSize],1,0);
  
      unit_impulse_test(input_size, output_size);
    }


  }
  
}
