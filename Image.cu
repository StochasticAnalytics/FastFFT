#include "Image.cuh"

template < class wanted_real_type, class wanted_complex_type >
Image<wanted_real_type, wanted_complex_type>::Image(short4 wanted_size)
{

  size = wanted_size;

  if (wanted_size.x % 2 == 0) padding_jump_value = 2;
  else padding_jump_value = 1;

  is_in_memory = false;
  is_in_real_space = true;
  is_cufft_planned = false;
  is_fftw_planned = false;



}

template < class wanted_real_type, class wanted_complex_type >
Image<wanted_real_type, wanted_complex_type>::~Image()
{
  if (is_in_memory) fftwf_free(real_values);
  if (is_fftw_planned)
  {
    fftwf_destroy_plan(plan_fwd);
    fftwf_destroy_plan(plan_bwd);
  }
  if (is_cufft_planned)
  {
    (cufftDestroy(cuda_plan_inverse));
    (cufftDestroy(cuda_plan_forward));
  }
}

// template < class wanted_real_type, class wanted_complex_type >
// Image<class wanted_real_type, class wanted_complex_type >::Image()
// {

// }

template < class wanted_real_type, class wanted_complex_type >
void Image<wanted_real_type, wanted_complex_type>::Allocate(bool set_fftw_plan)
{
  real_values = (wanted_real_type *) fftwf_malloc(sizeof(wanted_real_type) * real_memory_allocated);
  complex_values = (wanted_complex_type*) real_values;  // Set the complex_values to point at the newly allocated real values;

  // This will only work for single precision, should probably add a check on this, but for now rely on the user to make sure they are using single precision.
  if (set_fftw_plan)
  {
    plan_fwd = fftwf_plan_dft_r2c_3d(size.z, size.y, size.x, real_values, reinterpret_cast<fftwf_complex*>(complex_values), FFTW_ESTIMATE);
    plan_bwd = fftwf_plan_dft_c2r_3d(size.z, size.y, size.x, reinterpret_cast<fftwf_complex*>(complex_values), real_values, FFTW_ESTIMATE);
    is_fftw_planned = true;
  }


  is_in_memory = true;
}


template < class wanted_real_type, class wanted_complex_type >
void Image<wanted_real_type, wanted_complex_type>::FwdFFT()
{
  if (is_fftw_planned)
  {
    // Now let's do the forward FFT on the host and check that the result is correct.
    fftwf_execute_dft_r2c(plan_fwd, real_values, reinterpret_cast<fftwf_complex*>(complex_values));
  }
  else {std::cout << "Error: FFTW plan not set up." << std::endl; exit(1);}

  is_in_real_space = false;

}

template < class wanted_real_type, class wanted_complex_type >
void Image<wanted_real_type, wanted_complex_type>::InvFFT()
{
  if (is_fftw_planned)
  {
    // Now let's do the forward FFT on the host and check that the result is correct.
    fftwf_execute_dft_c2r(plan_bwd, reinterpret_cast<fftwf_complex*>(complex_values), real_values);
  }
  else {std::cout << "Error: FFTW plan not set up." << std::endl; exit(1);}

  is_in_real_space = true;
}

template < class wanted_real_type, class wanted_complex_type >
void Image<wanted_real_type, wanted_complex_type>::MakeCufftPlan()
{

  // TODO for alternate precisions.

  std::cout << "Allocating a 2d cufft Plan" << std::endl;

  cufftCreate(&cuda_plan_forward);
  cufftCreate(&cuda_plan_inverse);

  cufftSetStream(cuda_plan_forward, cudaStreamPerThread);
  cufftSetStream(cuda_plan_inverse, cudaStreamPerThread);

  int rank = 2; int iBatch = 1;
  long long int* fftDims = new long long int[rank];
  long long int*inembed = new long long int[rank];
  long long int*onembed = new long long int[rank];

  fftDims[0] = size.y;
  fftDims[1] = size.x;

  inembed[0] = size.y;
  inembed[1] = size.w;

  onembed[0] = size.y;
  onembed[1] = size.w;

  (cufftXtMakePlanMany(cuda_plan_forward, rank, fftDims,
    NULL, NULL, NULL, CUDA_R_32F,
    NULL, NULL, NULL, CUDA_C_32F, iBatch, &cuda_plan_worksize_forward, CUDA_C_32F));
    (cufftXtMakePlanMany(cuda_plan_inverse, rank, fftDims,
    NULL, NULL, NULL, CUDA_C_32F,
    NULL, NULL, NULL, CUDA_R_32F, iBatch, &cuda_plan_worksize_inverse, CUDA_R_32F));

    delete [] fftDims;
    delete [] inembed;
    delete [] onembed;

    is_cufft_planned = true;
}
