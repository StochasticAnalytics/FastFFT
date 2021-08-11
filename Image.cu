#include "Image.cuh"

template < class wanted_real_type, class wanted_complex_type >
Image<wanted_real_type, wanted_complex_type>::Image(short4 wanted_size)
{

  size = wanted_size;

  if (wanted_size.x % 2 == 0) padding_jump_value = 2;
  else padding_jump_value = 1;

  size.w = (size.x + padding_jump_value) / 2;

  is_in_memory = false;
  is_in_real_space = true;
  is_cufft_planned = false;
  is_fftw_planned = false;



}

template < class wanted_real_type, class wanted_complex_type >
Image<wanted_real_type, wanted_complex_type>::~Image()
{
  if (is_in_memory) 
  {
    fftwf_free(real_values);
     is_in_memory = false;
  } 
  if (is_fftw_planned)
  {
    fftwf_destroy_plan(plan_fwd);
    fftwf_destroy_plan(plan_bwd);
    is_fftw_planned = false;
  }
  if (is_cufft_planned)
  {
    cudaErr_img(cufftDestroy(cuda_plan_inverse));
    cudaErr_img(cufftDestroy(cuda_plan_forward));
    is_cufft_planned = false;
  }

  if (is_set_clip_into_mask)
  {
    cudaErr_img(cudaFree(clipIntoMask));
    is_set_clip_into_mask = false;
  }
}

template < class wanted_real_type, class wanted_complex_type >
void Image<wanted_real_type, wanted_complex_type>::SetClipIntoMask(short4 input_size, short4 output_size)
{
  // Allocate the mask
  int pjv;
  int address = 0;
  int n_values = output_size.w*2*output_size.y;
  bool* tmpMask = new bool[n_values];

  precheck_img
  cudaErr_img(cudaMalloc(&clipIntoMask, (n_values)*sizeof(bool)));
  postcheck_img

  if (output_size.x % 2 == 0) pjv = 2;
  else pjv = 1;

  for (int j = 0 ; j < output_size.y ; j++)
  {
    for (int i = 0 ; i < output_size.x ; i++)
    {
      if (i < input_size.x && j < input_size.y) tmpMask[address] = true;
      else tmpMask[address] = false;
      address++;
    }
    tmpMask[address] = false; 
    address++;
    if (pjv > 1) {tmpMask[address] = false;  address++;}
  }


  cudaErr_img(cudaMemcpyAsync(clipIntoMask, tmpMask, n_values*sizeof(bool),cudaMemcpyHostToDevice,cudaStreamPerThread));
  cudaStreamSynchronize(cudaStreamPerThread);

  delete [] tmpMask;
  is_set_clip_into_mask = true;


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

typedef struct _CB_realLoadAndClipInto_params
{
  bool* mask;
	cufftReal*	target;

} CB_realLoadAndClipInto_params;


static __device__ cufftReal CB_realLoadAndClipInto(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr);

static __device__ cufftReal CB_realLoadAndClipInto(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr)
{

	 CB_realLoadAndClipInto_params* my_params = (CB_realLoadAndClipInto_params *)callerInfo;

  if (my_params->mask[offset])
  {
    return my_params->target[offset];

  }
  else
  {
    return 0.0f;
  }



}

__device__ cufftCallbackLoadR d_realLoadAndClipInto = CB_realLoadAndClipInto;

template < class wanted_real_type, class wanted_complex_type >
void Image<wanted_real_type, wanted_complex_type>::SetClipIntoCallback(cufftReal* image_to_insert, int image_to_insert_size_x, int image_to_insert_size_y,int image_to_insert_pitch)
{


  // // First make the mask
  short4 wanted_size = make_short4(image_to_insert_size_x, image_to_insert_size_y, 1, image_to_insert_pitch);
  SetClipIntoMask(wanted_size, size );

  if (!is_cufft_planned) {std::cout << "Cufft plan must be made before setting callback function." << std::endl; exit(-1);}

  cufftCallbackLoadR h_realLoadAndClipInto;
  CB_realLoadAndClipInto_params* d_params;
  CB_realLoadAndClipInto_params h_params;

  precheck_img
  h_params.target = (cufftReal *)image_to_insert;
  h_params.mask = (bool*) clipIntoMask;
  cudaErr_img(cudaMalloc((void **)&d_params,sizeof(CB_realLoadAndClipInto_params)));
  postcheck_img

  precheck_img
  cudaErr_img(cudaMemcpyAsync(d_params, &h_params, sizeof(CB_realLoadAndClipInto_params), cudaMemcpyHostToDevice, cudaStreamPerThread));
  postcheck_img

  precheck_img
  cudaErr_img(cudaMemcpyFromSymbol(&h_realLoadAndClipInto,d_realLoadAndClipInto, sizeof(h_realLoadAndClipInto)));
  postcheck_img

  precheck_img
  cudaErr_img(cudaStreamSynchronize(cudaStreamPerThread));
  postcheck_img

  precheck_img
  cudaErr_img(cufftXtSetCallback(cuda_plan_forward, (void **)&h_realLoadAndClipInto, CUFFT_CB_LD_REAL, (void **)&d_params));
  postcheck_img


}