// Collection of helper functions for test.cu

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>

// sudo apt-get install libfftw3-dev libfftw3-doc
#include <fftw3.h>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include <cufft.h>
#include <cufftXt.h>

// A simple class to represent image objects needed for testing FastFFT. 



template<class wanted_real_type, class wanted_complex_type >
class Image {

  public:

    Image();
    Image(short4 wanted_size);
    ~Image();

    wanted_real_type* real_values;
    wanted_complex_type* complex_values;
    bool* clipIntoMask;

    short4 size;
    int real_memory_allocated;
    int padding_jump_value;
  
    float fftw_epsilon;


    bool is_in_memory;
    bool is_fftw_planned;
    bool is_in_real_space;
    bool is_cufft_planned;


    void Allocate( );
    void Allocate(bool plan_fftw);
    void FwdFFT();
    void InvFFT();

    // Make FFTW plans for comparing CPU to GPU xforms.
    // This is nearly verbatim from cisTEM::Image::Allocate - I do not know if FFTW_ESTIMATE is the best option.
    // In cisTEM we almost always use MKL, so this might be worth testing. I always used exhaustive in Matlab/emClarity.
    fftwf_plan plan_fwd = NULL;
    fftwf_plan plan_bwd = NULL;

    cufftHandle cuda_plan_forward;
    cufftHandle cuda_plan_inverse;
    size_t	cuda_plan_worksize_forward;
    size_t	cuda_plan_worksize_inverse;
  
    cudaEvent_t startEvent { nullptr };
    cudaEvent_t stopEvent { nullptr };
    float       elapsed_gpu_ms {};

    inline void create_timing_events() {   
      cudaEventCreate( &startEvent, cudaEventBlockingSync);
      cudaEventCreate( &stopEvent, cudaEventBlockingSync );
    }

    inline void record_start() { cudaEventRecord( startEvent ); }
    inline void record_stop()  { cudaEventRecord( stopEvent ); }
    inline void synchronize() { cudaEventSynchronize( stopEvent ); }
    inline void print_time(std::string msg, bool print_out = true) {  cudaEventElapsedTime( &elapsed_gpu_ms, startEvent, stopEvent ) ; if (print_out) {std::cout << "Time on " << msg << " " << elapsed_gpu_ms << " ms" << std::endl;} }
    void MakeCufftPlan();
    void MakeCufftPlan3d();

    void SetClipIntoMask(short4 input_size, short4 output_size); bool is_set_clip_into_mask = false;
    void SetClipIntoCallback(cufftReal* image_to_insert, int image_to_insert_size_x, int image_to_insert_size_y,int image_to_insert_pitch);
    void SetComplexConjMultiplyAndLoadCallBack(cufftComplex* search_image_FT, cufftReal FT_normalization_factor);
    void MultiplyConjugateImage( wanted_complex_type* other_image);


  

  private:

};


// To print a message and some number n_to_print complex values to stdout
void print_values_complex(float* input, std::string msg, int n_to_print)
{
  for (int i = 0; i < n_to_print*2; i+=2) { std::cout << msg << i/2 << "  " << input[i] << " " << input[i+1] << std::endl ;}
}

// Return sum of real values
float ReturnSumOfReal(float* input, short4 size, bool print_val = false)
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

        temp_sum += double(input[address]);
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

// Return the sum of the complex values
float ReturnSumOfComplexAmplitudes(float2* input, int n_to_print)
{
  // We want to asses the error in the FFT at single/half precision, but to not add 
  // extra error for the use double here.
  double sum = 0;
  double x;
  double y;

  for (int i = 0; i < n_to_print; i++) 
  {
    x = double (input[i].x);
    y = double (input[i].y);
    sum += sqrt(x*x + y*y);
  }
  
  return sum;
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

//#define HEAVYERRORCHECKING_IMG


// Note we are using std::cerr b/c the wxWidgets apps running in cisTEM are capturing std::cout
#ifndef HEAVYERRORCHECKING_IMG
#define postcheck_img
#define cudaErr_img(error) { auto status = static_cast<cudaError_t>(error); {;} };
#define precheck_img
#else
#define MyFFTPrintWithDetails(...)	{std::cerr << __VA_ARGS__  << " From: " << __FILE__  << " " << __LINE__  << " " << __PRETTY_FUNCTION__ << std::endl;}
#define postcheck_img { cudaError_t error = cudaStreamSynchronize(cudaStreamPerThread); if (error != cudaSuccess) { std::cerr << cudaGetErrorString(error) << std::endl; MyFFTPrintWithDetails("");} };
#define cudaErr_img(error) { auto status = static_cast<cudaError_t>(error); if (status != cudaSuccess) { std::cerr << cudaGetErrorString(status) << std::endl; MyFFTPrintWithDetails("");} };
#define precheck_img { cudaErr_img(cudaGetLastError()); }
#endif