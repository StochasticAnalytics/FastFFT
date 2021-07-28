/*
 * DFTbyDecomposition.cpp
 *
 *  Created on: Oct 21, 2020
 *      Author: himesb
 */

#include "gpu_core_headers.h"

#include "/groups/himesb/cufftdx/include/cufftdx.hpp"
// block_io depends on fp16. Both included from version () on 2020 Oct 30 as IO here may break on future changes

#include "fp16_common.hpp"
#include "block_io.hpp"

//#include "block_io.hpp"

//#include "/groups/himesb/cufftdx/example/common.hpp"

using namespace cufftdx;

const int test_size = 4096;
const int pre_padded_size = 512;
const int ffts_per_block_padded = 1;//test_size / pre_padded_size;
// Elements per thread must be [2,32]
const int ept_r = 8;
const int ept_c = 8;
// FFts per block. Might be able to re-use twiddles but prob more mem intensive. TODO test me and also evaluate memory size
const int ffts_per_block = 1; // 1 is the default, higher numbers don't work yet. Might be to do with padding. FIXME

__global__ void DFT_R2C_WithPaddingKernel(cufftReal* input_values, cufftComplex* output_values, int4 dims_in, int4 dims_out, float C);
__global__ void DFT_C2C_WithPaddingKernel_strided(cufftComplex* input_values, int4 dims_in, int4 dims_out, float C);
__global__ void DFT_R2C_WithPaddingKernel_strided(cufftReal* input_values, cufftComplex* output_values, int4 dims_in, int4 dims_out, float C);
__global__ void DFT_C2C_WithPaddingKernel(cufftComplex* input_values, int4 dims_in, int4 dims_out, float C);
__global__ void DFT_C2C_WithPaddingKernel_rdx2(cufftComplex* input_values, int4 dims_in, int4 dims_out, float C);

template<class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_R2C_WithPadding(ScalarType* input_values, ComplexType* output_values, int4 dims_in, int4 dims_out, bool rotate, float twiddle_in, int Q);

template<class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_R2C_WithPadding_rotated(ScalarType* input_values, ComplexType* output_values, int4 dims_in, int4 dims_out, bool rotate, float twiddle_in, int Q);

template<class FFT, class ComplexType= typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_WithPadding(ComplexType* input_values, ComplexType* output_values, int4 dims_in, int4 dims_out, bool rotate, float twiddle_in, int Q);

template<class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_R2C_WithPadding_strided(ScalarType* input_values, ComplexType* output_values, int4 dims_in, int4 dims_out, bool rotate, float twiddle_in, int Q);

template<class FFT, class ComplexType = typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_WithPadding_strided(ComplexType* input_values, ComplexType* output_values, int4 dims_in, int4 dims_out, bool rotate, float twiddle_in, int Q);

template<class FFT>
//__launch_bounds__(FFT::max_threads_per_block) __global__
__global__ void block_fft_kernel_C2C(typename FFT::input_type* input_values, typename FFT::output_type* output_values, int4 dims_in, int4 dims_out, float twid_constant, int n_sectors);


template<class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_R2C_rotate(ScalarType* input_values, ComplexType* output_values, int4 dims_in, int4 dims_out, typename FFT::workspace_type workspace, bool rotate);

//template<class FFT>
//__global__ void block_fft_kernel_R2C_rotate(float* input_values, typename FFT::output_type* output_values, int4 dims_in, int4 dims_out, typename FFT::workspace_type workspace);
template<class FFT, class ComplexType = typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_rotate(ComplexType* input_values, ComplexType* output_values, int4 dims_in, int4 dims_out, typename FFT::workspace_type workspace, bool rotate);

template<class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2R_rotate(ComplexType* input_values, ScalarType* output_values, int4 dims_in, int4 dims_out, typename FFT::workspace_type workspace, bool rotate);


using FFT_256          = decltype(Block() + Size<256>() + Type<fft_type::c2c>() +
                     Precision<float>() + ElementsPerThread<2>() + FFTsPerBlock<1>() + SM<700>());
using FFT_16          = decltype(Block() + Size<16>() + Type<fft_type::c2c>() +
                     Precision<float>() + ElementsPerThread<2>() + FFTsPerBlock<1>() + SM<700>());
using FFT_4096_r2c   = decltype(Block() + Size<test_size>() + Type<fft_type::r2c>() +
                     Precision<float>() + ElementsPerThread<ept_r>() + FFTsPerBlock<ffts_per_block>() + SM<700>());
using FFT_4096_c2c   = decltype(Block() + Size<test_size>() + Type<fft_type::c2c>() +
                     Precision<float>() + ElementsPerThread<ept_c>() + FFTsPerBlock<ffts_per_block>() + SM<700>());
using FFT_4096_c2r   = decltype(Block() + Size<test_size>() + Type<fft_type::c2r>() +
                     Precision<float>() + ElementsPerThread<ept_r>() + FFTsPerBlock<ffts_per_block>() + SM<700>());


using FFT_512_r2c   = decltype(Block() + Size<pre_padded_size>() + Type<fft_type::r2c>() +
                     Precision<float>() + ElementsPerThread<ept_r>() + FFTsPerBlock<ffts_per_block_padded>() + SM<700>());
using FFT_512_c2c   = decltype(Block() + Size<pre_padded_size>() + Type<fft_type::c2c>() +
                     Precision<float>() + ElementsPerThread<ept_c>() + FFTsPerBlock<ffts_per_block_padded>() + SM<700>());
using FFT_512_c2r   = decltype(Block() + Size<pre_padded_size>() + Type<fft_type::c2r>() +
                     Precision<float>() + ElementsPerThread<ept_r>() + FFTsPerBlock<ffts_per_block_padded>() + SM<700>());

DFTbyDecomposition::DFTbyDecomposition() // @suppress("Class members should be properly initialized")
{
	is_set_gpu_images = false;
	is_set_twiddles = false;
	is_allocated_rotated_buffer = false;
//	is_set_outputs = false;
}

DFTbyDecomposition::~DFTbyDecomposition()
{
	if (is_set_twiddles)
	{
		cudaErr(cudaFree(twiddles));
	}
	if (is_allocated_rotated_buffer)
	{
		cudaErr(cudaFree(d_rotated_buffer));

	}
//	if (is_set_outputs)
//	{
//		cudaErr(cudaFree(output_real));
//		cudaErr(cudaFree(output_imag));
//	}
}

DFTbyDecomposition::DFTbyDecomposition(const DFTbyDecomposition &other)
{
	// TODO Auto-generated constructor stub

}

DFTbyDecomposition& DFTbyDecomposition::operator=(
		const DFTbyDecomposition &other) {
	// TODO Auto-generated method stub
}

void DFTbyDecomposition::InitTestCase(int wanted_input_size_x, int wanted_input_size_y, int wanted_output_size_x, int wanted_output_size_y)
{
	dims_input = make_int2(wanted_input_size_x, wanted_input_size_y);
	dims_output = make_int2(wanted_output_size_x, wanted_output_size_y);

	// In practice we'll give a pointer to the arrays in some GpuImages
}

void DFTbyDecomposition::SetGpuImages(Image& cpu_input, Image& cpu_output)
{

	// Should be in real space, TODO add check
	input_image.CopyFromCpuImage(cpu_input);
	input_image.CopyHostToDevice();

	if (&cpu_output != &cpu_input)
	{
		wxPrintf("Initializing output image\n");
		// Initialize to Fourier space
		output_image.CopyFromCpuImage(cpu_output);
		output_image.CopyHostToDevice();

//		output_image.Allocate((int)dims_output.x, (int)dims_output.y, 1, false);
//		output_image.Zeros();
	}
	else
	{
		output_image = input_image;
	}

	wxPrintf("Sizes in init %d %d in and %d %d out\n",input_image.dims.x, input_image.dims.y, output_image.dims.x, output_image.dims.y);

	is_set_gpu_images = true;

}

void DFTbyDecomposition::AllocateRotatedBuffer()
{
	MyAssertTrue(is_set_gpu_images,"Gpu images must be set before allocating a buffer");

	cudaErr(cudaMalloc(&d_rotated_buffer, sizeof(float)*output_image.real_memory_allocated));

	is_allocated_rotated_buffer = true;
}


void DFTbyDecomposition::DFT_R2C_WithPadding()
{

	// FIXME when adding real space complex images
	MyAssertTrue( input_image.is_in_memory_gpu, "Input image is in not on the GPU!");
	MyAssertTrue( output_image.is_in_memory_gpu, "Output image is in not on the GPU!");



	int threadsPerBlock = input_image.dims.x; // FIXME make sure its a multiple of 32
	int gridDims = input_image.dims.y;
//	dim3 gridDims = dim3((output_image.dims.w/2 + threadsPerBlock - 1) / threadsPerBlock,
//					  	1, 1);
//  output_image.dims.y
	int shared_mem = sizeof(float)*input_image.dims.x;
	float C = -2*PIf/output_image.dims.x;
	DFT_R2C_WithPaddingKernel<< <gridDims, threadsPerBlock, shared_mem, cudaStreamPerThread>> > ( input_image.real_values_gpu,  output_image.complex_values_gpu, input_image.dims, output_image.dims, C);
	cudaStreamSynchronize(cudaStreamPerThread);



}

__global__ void DFT_R2C_WithPaddingKernel(cufftReal* input_values, cufftComplex* output_values, int4 dims_in, int4 dims_out, float C)
{

//	// Initialize the shared memory, assuming everying matches the input data X size in
	extern __shared__ float s[];
	// Avoid N*k type conversion and multiplication
	float* data = s;
//	float* coeff= (float*)&data[dims_in.x];


	int x = threadIdx.x;
	int pixel_out = (dims_out.w/2)*blockIdx.x;


	data[x] = __ldg((const float *)&input_values[dims_in.w*blockIdx.x + x]);
	__syncthreads();
//
//	 Loop over N updating the actual twiddle value along the way. This might lead to accuracy problems.
	float sum_real;
	float sum_imag;
	float twi_r;
	float twi_i;
	float coeff;

	for (int k = threadIdx.x; k < dims_out.w/2; k+=blockDim.x)
	{
		coeff = C*(float)k;
		sum_real = 0.0f;
		sum_imag = 0.0f;
		for (int n = 0; n < dims_in.x; n++)
		{
			__sincosf(coeff*n,&twi_i,&twi_r);
			sum_real = __fmaf_rn(data[n],twi_r,sum_real);
			sum_imag = __fmaf_rn(data[n],twi_i,sum_imag);
		}

		// Not sure if an async write, or storage to a shared mem temp would be faster.
		output_values[pixel_out + k].x = sum_real;
		output_values[pixel_out + k].y = sum_imag;
	}


	return;

}


void DFTbyDecomposition::DFT_C2C_WithPadding()
{

	// FIXME when adding real space complex images
	MyAssertTrue( input_image.is_in_memory_gpu, "Input image is in not on the GPU!");
	MyAssertTrue( output_image.is_in_memory_gpu, "Output image is in not on the GPU!");


	int threadsPerBlock = input_image.dims.x; // FIXME make sure its a multiple of 32
	int gridDims = output_image.dims.w/2;

	int shared_mem = sizeof(cufftComplex)*input_image.dims.x;

	float C = -2*PIf/output_image.dims.x;
	DFT_C2C_WithPaddingKernel<< <gridDims, threadsPerBlock, shared_mem, cudaStreamPerThread>> > ( output_image.complex_values_gpu, input_image.dims, output_image.dims, C);
	cudaStreamSynchronize(cudaStreamPerThread);



}

__global__ void DFT_C2C_WithPaddingKernel(cufftComplex* inplace_image, int4 dims_in, int4 dims_out, float C)
{

	// Initialize the shared memory, assuming everying matches the input data X size in
	// Check that setting cudaFuncSetSharedMemConfig  to 8byte makes any diff for complex reads
	extern __shared__ cufftComplex c[];
	cufftComplex* data = c;


	int x = threadIdx.x;
	int pixel_out = (dims_out.w/2)*blockIdx.x;

	data[x] = __ldg((const cufftComplex *)&inplace_image[pixel_out + x]);
	__syncthreads();
//
//	 Loop over N updating the actual twiddle value along the way. This might lead to accuracy problems.
	cufftComplex sum;
	float twi_r;
	float twi_i;
	float coeff;
	float tmp;

	for (int k = threadIdx.x; k < dims_out.w/2; k+=blockDim.x)
	{
		coeff = C*(float)k;
		sum.x = 0.0f;
		sum.y = 0.0f;
		for (int n = 0; n < dims_in.y; n++)
		{
			__sincosf(coeff*n,&twi_i,&twi_r);
			tmp = data[n].x * twi_i;
			sum.x += __fmaf_rn(data[n].x, twi_r, -twi_i * data[n].y);
			sum.y += __fmaf_rn(data[n].y, twi_r, tmp);
		}

		// Not sure if an async write, or storage to a shared mem temp would be faster.
//		inplace_image[pixel_out + k].x = sum_real;
//		inplace_image[pixel_out + k].y = sum_imag;
		inplace_image[pixel_out + k] = sum;
	}



	return;

}


void DFTbyDecomposition::DFT_R2C_WithPadding_strided()
{

	// FIXME when adding real space complex images
	MyAssertTrue( input_image.is_in_memory_gpu, "Input image is in not on the GPU!");
	MyAssertTrue( output_image.is_in_memory_gpu, "Output image is in not on the GPU!");



	int threadsPerBlock = input_image.dims.y; // FIXME make sure its a multiple of 32
	int gridDims = input_image.dims.x;
//	dim3 gridDims = dim3((output_image.dims.w/2 + threadsPerBlock - 1) / threadsPerBlock,
//					  	1, 1);
//  output_image.dims.y
	int shared_mem = sizeof(float)*input_image.dims.y;
	float C = -2*PIf/output_image.dims.y;
	DFT_R2C_WithPaddingKernel_strided<< <gridDims, threadsPerBlock, shared_mem, cudaStreamPerThread>> > ( input_image.real_values_gpu,  output_image.complex_values_gpu, input_image.dims, output_image.dims, C);
	cudaStreamSynchronize(cudaStreamPerThread);



}

__global__ void DFT_R2C_WithPaddingKernel_strided(cufftReal* input_values, cufftComplex* output_values, int4 dims_in, int4 dims_out, float C)
{

//	// Initialize the shared memory, assuming everying matches the input data X size in
	extern __shared__ float s[];
	// Avoid N*k type conversion and multiplication
	float* data = s;
//	float* coeff= (float*)&data[dims_in.x];


	int y = threadIdx.x;
	int pixel_in = blockIdx.x + y * (dims_in.w);

	data[y] = __ldg((const cufftReal *)&input_values[pixel_in]);
	__syncthreads();
//

//
//	 Loop over N updating the actual twiddle value along the way. This might lead to accuracy problems.
	float sum_real;
	float sum_imag;
	float twi_r;
	float twi_i;
	float coeff;

	for (int k = threadIdx.x; k < dims_out.y; k+=blockDim.x)
	{
		coeff = C*(float)k;
		sum_real = 0.0f;
		sum_imag = 0.0f;
		for (int n = 0; n < dims_in.x; n++)
		{
			__sincosf(coeff*n,&twi_i,&twi_r);
			sum_real = __fmaf_rn(data[n],twi_r,sum_real);
			sum_imag = __fmaf_rn(data[n],twi_i,sum_imag);
		}

		// Not sure if an async write, or storage to a shared mem temp would be faster.
		// Not sure if an async write, or storage to a shared mem temp would be faster.
		output_values[blockIdx.x + k * (dims_out.w/2)].x = sum_real;
		output_values[blockIdx.x + k * (dims_out.w/2)].y = sum_imag;
	}


	return;

}


void DFTbyDecomposition::DFT_C2C_WithPadding_strided()
{

	// FIXME when adding real space complex images
	MyAssertTrue( input_image.is_in_memory_gpu, "Input image is in not on the GPU!");
	MyAssertTrue( output_image.is_in_memory_gpu, "Output image is in not on the GPU!");


	int threadsPerBlock = input_image.dims.y; // FIXME make sure its a multiple of 32
	int gridDims = output_image.dims.w/2;

	int shared_mem = sizeof(cufftComplex)*input_image.dims.y;

	float C = -2*PIf/output_image.dims.y;
	DFT_C2C_WithPaddingKernel_strided<< <gridDims, threadsPerBlock, shared_mem, cudaStreamPerThread>> > ( output_image.complex_values_gpu, input_image.dims, output_image.dims, C);
	cudaStreamSynchronize(cudaStreamPerThread);



}

__global__ void DFT_C2C_WithPaddingKernel_strided(cufftComplex* inplace_image, int4 dims_in, int4 dims_out, float C)
{

	// Initialize the shared memory, assuming everying matches the input data X size in
	// Check that setting cudaFuncSetSharedMemConfig  to 8byte makes any diff for complex reads
	extern __shared__ cufftComplex c[];
	cufftComplex* data = c;


	int y = threadIdx.x;
	int pixel_in = blockIdx.x + y * (dims_out.w/2);


	data[y] = __ldg((const cufftComplex *)&inplace_image[pixel_in]);
	__syncthreads();
//
//	 Loop over N updating the actual twiddle value along the way. This might lead to accuracy problems.
	float sum_real;
	float sum_imag;
	float twi_r;
	float twi_i;
	float coeff;
	float tmp;

	for (int k = threadIdx.x; k < dims_out.y; k+=blockDim.x)
	{
		coeff = C*(float)k;
		sum_real = 0.0f;
		sum_imag = 0.0f;
		for (int n = 0; n < dims_in.y; n++)
		{
			__sincosf(coeff*n,&twi_i,&twi_r);
			tmp = data[n].x * twi_i;
			sum_real += __fmaf_rn(data[n].x, twi_r, -twi_i * data[n].y);
			sum_imag += __fmaf_rn(data[n].y, twi_r, tmp);
		}

		// Not sure if an async write, or storage to a shared mem temp would be faster.
		inplace_image[blockIdx.x + k * (dims_out.w/2)].x = sum_real;
		inplace_image[blockIdx.x + k * (dims_out.w/2)].y = sum_imag;
	}


	return;

}

void DFTbyDecomposition::DFT_C2C_WithPadding_rdx2()
{

	// FIXME when adding real space complex images
	MyAssertTrue( input_image.is_in_memory_gpu, "Input image is in not on the GPU!");
	MyAssertTrue( output_image.is_in_memory_gpu, "Output image is in not on the GPU!");


	int threadsPerBlock = input_image.dims.x; // FIXME make sure its a multiple of 32
	int gridDims = output_image.dims.w/2;

	int shared_mem = sizeof(cufftComplex)*input_image.dims.x;

	float C = -2*PIf/output_image.dims.x*2;
	DFT_C2C_WithPaddingKernel_rdx2<< <gridDims, threadsPerBlock, shared_mem, cudaStreamPerThread>> > ( output_image.complex_values_gpu, input_image.dims, output_image.dims, C);
	cudaStreamSynchronize(cudaStreamPerThread);



}

__global__ void DFT_C2C_WithPaddingKernel_rdx2(cufftComplex* inplace_image, int4 dims_in, int4 dims_out, float C)
{

	// Initialize the shared memory, assuming everying matches the input data X size in
	// Check that setting cudaFuncSetSharedMemConfig  to 8byte makes any diff for complex reads
	extern __shared__ cufftComplex c[];
	cufftComplex* data = c;


	int x = threadIdx.x;
	int pixel_out = (dims_out.w/2)*blockIdx.x;

	data[x] = __ldg((const cufftComplex *)&inplace_image[pixel_out + x]);
	__syncthreads();
//
//	 Loop over N updating the actual twiddle value along the way. This might lead to accuracy problems.
	cufftComplex sum;
	cufftComplex eve;
	float twi_r;
	float twi_i;
	float coeff;
	float tmp;

	for (int k = threadIdx.x; k < dims_out.w/4; k+=blockDim.x)
	{
		// get the even DFT
		coeff = C*(float)k;
		sum.x = 0.0f;
		sum.y = 0.0f;
		for (int n = 0; n < dims_in.y; n+=2)
		{
			__sincosf(coeff*n,&twi_i,&twi_r);
			tmp = data[n].x * twi_i;
			sum.x += __fmaf_rn(data[n].x, twi_r, -twi_i * data[n].y);
			sum.y += __fmaf_rn(data[n].y, twi_r, tmp);
		}

		eve = sum;

		// get the odd DFT
		sum.x = 0.0f;
		sum.y = 0.0f;
		for (int n = 1; n < dims_in.y; n+=2)
		{
			__sincosf(coeff*n,&twi_i,&twi_r);
			tmp = data[n].x * twi_i;
			sum.x += __fmaf_rn(data[n].x, twi_r, -twi_i * data[n].y);
			sum.y += __fmaf_rn(data[n].y, twi_r, tmp);
		}

		// Get the twiddle for the combined radix
		__sincosf(coeff/2.0f,&twi_i,&twi_r);
		// Multiply the odd
		tmp = sum.x * twi_i;
		sum.x = __fmaf_rn(sum.x, twi_r, -twi_i * sum.y);
		sum.y = __fmaf_rn(sum.y, twi_r, tmp);

		inplace_image[pixel_out + k].x = eve.x + sum.x;
		inplace_image[pixel_out + k].y = eve.y + sum.y;

		inplace_image[pixel_out + k + dims_out.w/4].x = eve.x - sum.x;
		inplace_image[pixel_out + k + dims_out.w/4].y = eve.y - sum.y;

	}



	return;

}

void DFTbyDecomposition::FFT_R2C_WithPadding_strided(bool rotate)
{

	// This is the first set of 1d ffts when the input data are real valued, accessing the strided dimension. Since we need the full length, it will actually run a C2C xform

	// FIXME when adding real space complex images
	MyAssertTrue( input_image.is_in_memory_gpu, "Input image is in not on the GPU!");
	MyAssertTrue( output_image.is_in_memory_gpu, "Output image is in not on the GPU!");



    // For now consider the simplest launch params, where one input element is handled per thread.
    MyAssertFalse(input_image.dims.y % ept_c, "The elements per thread is not a divisor of the input y-dimension.");
	dim3 threadsPerBlock = dim3(input_image.dims.y / ept_c,1,1); // FIXME make sure its a multiple of 32
	dim3 gridDims = dim3(1,1,input_image.dims.x);

	// For the twiddle factors ahead of the P size ffts
	float twiddle_in = -2*PIf/output_image.dims.y;
	int   Q = output_image.dims.y / input_image.dims.y; // FIXME assuming for now this is already divisible

    // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
    // will be executed on block level. Shared memory is required for co-operation between threads.


	using FFT = decltype( FFT_512_c2c() + Direction<fft_direction::forward>() );
//	wxPrintf("FFT::block_dim %d %d %d\n", FFT::block_dim.x,FFT::block_dim.y,FFT::block_dim.z );
	using complex_type = typename FFT::value_type;
	using scalar_type  = typename FFT::value_type::value_type;

	// We need to make sure there is also enough memory for the input fft
//	int shared_mem = output_image.dims.w/2*sizeof(complex_type) + input_image.dims.w*sizeof(scalar_type) + FFT::shared_memory_size;
	int shared_mem = sizeof(scalar_type)*input_image.dims.y + FFT::shared_memory_size;

//	wxPrintf("Shared mem %d max threads per block %d and just fft %d\n",shared_mem,FFT::max_threads_per_block,FFT::shared_memory_size);

	block_fft_kernel_R2C_WithPadding_strided<FFT,complex_type,scalar_type><< <gridDims,  threadsPerBlock, shared_mem, cudaStreamPerThread>> >
	( (scalar_type*)input_image.real_values_gpu,  (complex_type*)output_image.complex_values_gpu, input_image.dims, output_image.dims, rotate,twiddle_in,Q);

	cudaErr(cudaPeekAtLastError());
	cudaErr(cudaDeviceSynchronize());

}

template<class FFT, class ComplexType, class ScalarType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_R2C_WithPadding_strided(ScalarType* input_values, ComplexType* output_values, int4 dims_in, int4 dims_out, bool rotate, float twiddle_in, int Q)
{

//	// Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_type = ComplexType;
    using scalar_type  = ScalarType;

	extern __shared__  scalar_type shared_input[];
	complex_type* shared_mem    = (complex_type*)&shared_input[dims_in.y];

	complex_type* shifted_output = (complex_type*)&output_values[blockIdx.z];
	scalar_type* shifted_input  = (scalar_type*)&input_values[blockIdx.z];

	// Memory used by FFT
	complex_type twiddle;
    complex_type thread_data[FFT::storage_size];

    // To re-map the thread index to the data
    int input_MAP[FFT::storage_size];
    // To re-map the decomposed frequency to the full output frequency
    int output_MAP[FFT::storage_size];
    // For a given decomposed fragment
    float twiddle_factor_args[FFT::storage_size];



	bah_io::io<FFT>::load_r2c_shared(shifted_input, shared_input, twiddle_factor_args,
								 	 twiddle_in,input_MAP,output_MAP,Q, dims_in.w);
    bah_io::io<FFT>::copy_from_shared(shared_input, thread_data, input_MAP);


	FFT().execute(thread_data, shared_mem);

	bah_io::io<FFT>::store(thread_data, shifted_output, output_MAP, dims_out.w/2);



    // For the other fragments we need the initial twiddle
	for (int sub_fft = 1; sub_fft < Q; sub_fft++)
	{

	    bah_io::io<FFT>::copy_from_shared(shared_input, thread_data, input_MAP);

		for (int i = 0; i < FFT::elements_per_thread; i++)
		{
			// Pre shift with twiddle
			__sincosf(twiddle_factor_args[i]*sub_fft,&twiddle.y,&twiddle.x);
			thread_data[i] *= twiddle;
			output_MAP[i]++;
		}


		FFT().execute(thread_data, shared_mem);

		bah_io::io<FFT>::store(thread_data, shifted_output, output_MAP, dims_out.w/2);

	}


	return;

}


void DFTbyDecomposition::FFT_R2C_WithPadding(bool rotate)
{

	// This is the first set of 1d ffts when the input data are real valued, accessing the strided dimension. Since we need the full length, it will actually run a C2C xform

	// FIXME when adding real space complex images
	MyAssertTrue( input_image.is_in_memory_gpu, "Input image is in not on the GPU!");
	MyAssertTrue( output_image.is_in_memory_gpu, "Output image is in not on the GPU!");
	if (rotate) MyAssertTrue( is_allocated_rotated_buffer, "Output image is in not on the GPU!");

    // For now consider the simplest launch params, where one input element is handled per thread.
    MyAssertFalse(input_image.dims.x % ept_c, "The elements per thread is not a divisor of the input y-dimension.");

	// For the twiddle factors ahead of the P size ffts
	float twiddle_in = -2*PIf/output_image.dims.x;
	int   Q = output_image.dims.x / input_image.dims.x; // FIXME assuming for now this is already divisible
    // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
    // will be executed on block level. Shared memory is required for co-operation between threads.


	dim3 threadsPerBlock = dim3(input_image.dims.x/ept_r, 1, 1); // FIXME make sure its a multiple of 32
	dim3 gridDims = dim3(1,1, input_image.dims.y);

	using FFT = decltype( FFT_512_c2c() + Direction<fft_direction::forward>() );
//	wxPrintf("FFT::block_dim %d %d %d TPB %d MAX %d\n", FFT::block_dim.x,FFT::block_dim.y,FFT::block_dim.z,threadsPerBlock.x,FFT::max_threads_per_block);
//	exit(-1);
	using complex_type = typename FFT::value_type;
	using scalar_type    = typename complex_type::value_type;


	if (rotate)
	{
		int shared_mem = input_image.dims.w*sizeof(scalar_type) + FFT::shared_memory_size;
		block_fft_kernel_R2C_WithPadding_rotated<FFT,complex_type,scalar_type><< <gridDims,  threadsPerBlock, shared_mem, cudaStreamPerThread>> >
		( (scalar_type *)input_image.real_values_gpu,   (complex_type*)d_rotated_buffer, input_image.dims, output_image.dims, rotate,twiddle_in,Q);
	//
		cudaErr(cudaPeekAtLastError());
		cudaErr(cudaDeviceSynchronize());
	}
	else
	{
		// remap partial xforms to full xform in shared memory for coalesced output
		int shared_mem = output_image.dims.w/2*sizeof(complex_type) + input_image.dims.w*sizeof(scalar_type) + FFT::shared_memory_size;
		block_fft_kernel_R2C_WithPadding<FFT,complex_type,scalar_type><< <gridDims,  threadsPerBlock, shared_mem, cudaStreamPerThread>> >
		( (scalar_type *)input_image.real_values_gpu,  (complex_type*)output_image.complex_values_gpu, input_image.dims, output_image.dims, rotate,twiddle_in,Q);
	//
	//	cudaErr(cudaPeekAtLastError());
	//	cudaErr(cudaDeviceSynchronize());
	}





}

template<class FFT, class ComplexType, class ScalarType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_R2C_WithPadding(ScalarType* input_values, ComplexType* output_values, int4 dims_in, int4 dims_out, bool rotate, float twiddle_in, int Q)
{

//	// Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_type = ComplexType;
    using scalar_type  = ScalarType;

	extern __shared__  scalar_type shared_input[];
	complex_type* shared_output = (complex_type*)&shared_input[dims_in.x];
	complex_type* shared_mem    = (complex_type*)&shared_output[dims_out.w/2];



	// Memory used by FFT
	complex_type twiddle;
    complex_type thread_data[FFT::storage_size];

    // To re-map the thread index to the data
    int input_MAP[FFT::storage_size];
    // To re-map the decomposed frequency to the full output frequency
    int output_MAP[FFT::storage_size];
    // For a given decomposed fragment
    float twiddle_factor_args[FFT::storage_size];

    // No need to __syncthreads as each thread only accesses its own shared mem anyway
    bah_io::io<FFT>::load_r2c_shared(&input_values[blockIdx.z*dims_in.w], shared_input, twiddle_factor_args, twiddle_in, input_MAP, output_MAP, Q, 1);
    bah_io::io<FFT>::copy_from_shared(shared_input, thread_data, input_MAP);


	// In the first FFT the modifying twiddle factor is 1 so the data are reeal
	FFT().execute(thread_data, shared_mem);

	bah_io::io<FFT>::store(thread_data,shared_output,output_MAP,1, dims_out.w/2);


    // For the other fragments we need the initial twiddle
	for (int sub_fft = 1; sub_fft < Q; sub_fft++)
	{

	    bah_io::io<FFT>::copy_from_shared(shared_input, thread_data, input_MAP);


		// cufftDX expects packed real data for a real xform, but we modify with a complex twiddle factor.
		// to get around this, split the complex fft into the sum of the real and imaginary parts
		for (int i = 0; i < FFT::elements_per_thread; i++)
		{
			// Pre shift with twiddle
			__sincosf(twiddle_factor_args[i]*sub_fft,&twiddle.y,&twiddle.x);
			thread_data[i] *= twiddle;
		    // increment the output map. Note this only works for the leading non-zero case
			output_MAP[i]++;
		}

		FFT().execute(thread_data, shared_mem);

		bah_io::io<FFT>::store(thread_data,shared_output,output_MAP,1, dims_out.w/2);

	}
//
	__syncthreads();

	// Now that the memory output can be coalesced send to global
	int this_idx;
	for (int sub_fft = 0; sub_fft < Q; sub_fft++)
	{
		for (int i = 0; i < FFT::elements_per_thread; i++)
		{
			this_idx = input_MAP[i] + dims_in.x*sub_fft;
			if (this_idx < dims_out.w/2)
			{
				output_values[blockIdx.z * dims_out.w/2 + this_idx] = shared_output[this_idx];
			}
		}
	}

    if (rotate)
    {
        // blockIdx.z + (dims_out.w/2 - index - 1)*dims_out.y
        int rotated_offset[2] = {(int)blockIdx.z + (dims_out.w/2 - 1)*dims_out.y, -int(dims_out.y)};
        bah_io::io<FFT>::store_r2c_rotated(thread_data, output_values, rotated_offset);


    }
    else
    {
        bah_io::io<FFT>::store_r2c(thread_data, &output_values[ffts_per_block*blockIdx.z * dims_in.w/2], dims_out.w/2*threadIdx.y);
    }

	return;

}

template<class FFT, class ComplexType, class ScalarType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_R2C_WithPadding_rotated(ScalarType* input_values, ComplexType* output_values, int4 dims_in, int4 dims_out, bool rotate, float twiddle_in, int Q)
{

//	// Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_type = ComplexType;
    using scalar_type  = ScalarType;

	extern __shared__  scalar_type shared_input[];
	complex_type* shared_mem    = (complex_type*)&shared_input[dims_in.x];



	// Memory used by FFT
	complex_type twiddle;
    complex_type thread_data[FFT::storage_size];

    // To re-map the thread index to the data
    int input_MAP[FFT::storage_size];
    // To re-map the decomposed frequency to the full output frequency
    int output_MAP[FFT::storage_size];
    // For a given decomposed fragment
    float twiddle_factor_args[FFT::storage_size];

    // bockIdx.z = y = x'


    // No need to __syncthreads as each thread only accesses its own shared mem anyway
    bah_io::io<FFT>::load_r2c_shared(&input_values[blockIdx.z*dims_in.w], shared_input, twiddle_factor_args, twiddle_in, input_MAP, output_MAP, Q, 1);
    bah_io::io<FFT>::copy_from_shared(shared_input, thread_data, input_MAP);


	// In the first FFT the modifying twiddle factor is 1 so the data are reeal
	FFT().execute(thread_data, shared_mem);

    // blockIdx.z + (dims_out.w/2 - index - 1)*dims_out.y
    int rotated_offset[2] = {(int)blockIdx.z + (dims_out.w/2 - 1)*dims_out.y, -int(dims_out.y)};
    bah_io::io<FFT>::store_rotated(thread_data, output_values, output_MAP, rotated_offset, dims_out.w/2);

    // For the other fragments we need the initial twiddle
	for (int sub_fft = 1; sub_fft < Q; sub_fft++)
	{

	    bah_io::io<FFT>::copy_from_shared(shared_input, thread_data, input_MAP);


		// cufftDX expects packed real data for a real xform, but we modify with a complex twiddle factor.
		// to get around this, split the complex fft into the sum of the real and imaginary parts
		for (int i = 0; i < FFT::elements_per_thread; i++)
		{
			// Pre shift with twiddle
			__sincosf(twiddle_factor_args[i]*sub_fft,&twiddle.y,&twiddle.x);
			thread_data[i] *= twiddle;
		    // increment the output map. Note this only works for the leading non-zero case
			output_MAP[i]++;
		}

		FFT().execute(thread_data, shared_mem);

	    bah_io::io<FFT>::store_rotated(thread_data, output_values, output_MAP, rotated_offset,dims_out.w/2);

	}
//


	return;

}
void DFTbyDecomposition::FFT_C2C_WithPadding_strided(bool rotate)
{

	// This is the first set of 1d ffts when the input data are real valued, accessing the strided dimension. Since we need the full length, it will actually run a C2C xform

	// FIXME when adding real space complex images
	MyAssertTrue( input_image.is_in_memory_gpu, "Input image is in not on the GPU!");
	MyAssertTrue( output_image.is_in_memory_gpu, "Output image is in not on the GPU!");



    // For now consider the simplest launch params, where one input element is handled per thread.
    MyAssertFalse(input_image.dims.y % ept_c, "The elements per thread is not a divisor of the input y-dimension.");
	dim3 threadsPerBlock = dim3(input_image.dims.y / ept_c,1,1); // FIXME make sure its a multiple of 32
	dim3 gridDims = dim3(1,1,output_image.dims.w/2);

	// For the twiddle factors ahead of the P size ffts
	float twiddle_in = -2*PIf/output_image.dims.y;
	int   Q = output_image.dims.y / input_image.dims.y; // FIXME assuming for now this is already divisible

    // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
    // will be executed on block level. Shared memory is required for co-operation between threads.


	using FFT = decltype( FFT_512_c2c() + Direction<fft_direction::forward>() );
//	wxPrintf("FFT::block_dim %d %d %d\n", FFT::block_dim.x,FFT::block_dim.y,FFT::block_dim.z );
	using complex_type = typename FFT::value_type;

	// We need to make sure there is also enough memory for the input fft
//	int shared_mem = output_image.dims.w/2*sizeof(complex_type) + input_image.dims.w*sizeof(scalar_type) + FFT::shared_memory_size;
	int shared_mem = sizeof(complex_type)*input_image.dims.y + FFT::shared_memory_size;

//	wxPrintf("Shared mem %d max threads per block %d and just fft %d\n",shared_mem,FFT::max_threads_per_block,FFT::shared_memory_size);

	block_fft_kernel_C2C_WithPadding_strided<FFT,complex_type><< <gridDims,  threadsPerBlock, shared_mem, cudaStreamPerThread>> >
	( (complex_type*)output_image.complex_values_gpu,  (complex_type*)output_image.complex_values_gpu, input_image.dims, output_image.dims, rotate,twiddle_in,Q);

//	cudaErr(cudaPeekAtLastError());
//	cudaErr(cudaDeviceSynchronize());

}

template<class FFT, class ComplexType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_WithPadding_strided(ComplexType* input_values, ComplexType* output_values, int4 dims_in, int4 dims_out, bool rotate, float twiddle_in, int Q)
{

//	// Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_type = ComplexType;

	extern __shared__  complex_type shared_input_complex[];
	complex_type* shared_mem    = (complex_type*)&shared_input_complex[dims_in.y];

	complex_type* shifted_output = (complex_type*)&output_values[blockIdx.z];
	complex_type* shifted_input  = (complex_type*)&input_values[blockIdx.z];

	// Memory used by FFT
	complex_type twiddle;
    complex_type thread_data[FFT::storage_size];

    // To re-map the thread index to the data
    int input_MAP[FFT::storage_size];
    // To re-map the decomposed frequency to the full output frequency
    int output_MAP[FFT::storage_size];
    // For a given decomposed fragment
    float twiddle_factor_args[FFT::storage_size];



	bah_io::io<FFT>::load_shared(shifted_input, shared_input_complex, twiddle_factor_args,
								 twiddle_in,input_MAP,output_MAP,Q, dims_out.w/2);
    bah_io::io<FFT>::copy_from_shared(shared_input_complex, thread_data, input_MAP);


	FFT().execute(thread_data, shared_mem);

	bah_io::io<FFT>::store(thread_data, shifted_output, output_MAP, dims_out.w/2);



    // For the other fragments we need the initial twiddle
	for (int sub_fft = 1; sub_fft < Q; sub_fft++)
	{

	    bah_io::io<FFT>::copy_from_shared(shared_input_complex, thread_data, input_MAP);

		for (int i = 0; i < FFT::elements_per_thread; i++)
		{
			// Pre shift with twiddle
			__sincosf(twiddle_factor_args[i]*sub_fft,&twiddle.y,&twiddle.x);
			thread_data[i] *= twiddle;
			output_MAP[i]++;

		}


		FFT().execute(thread_data, shared_mem);

		bah_io::io<FFT>::store(thread_data, shifted_output, output_MAP, dims_out.w/2);

	}


	return;

}

void DFTbyDecomposition::FFT_C2C_WithPadding(bool rotate)
{

	// This is the first set of 1d ffts when the input data are real valued, accessing the strided dimension. Since we need the full length, it will actually run a C2C xform

	// FIXME when adding real space complex images
	MyAssertTrue( input_image.is_in_memory_gpu, "Input image is in not on the GPU!");
	MyAssertTrue( output_image.is_in_memory_gpu, "Output image is in not on the GPU!");


	float twiddle_in;
	int Q;
	int shared_mem;
	dim3 threadsPerBlock;
	dim3 gridDims;

	using FFT = decltype( FFT_512_c2c() + Direction<fft_direction::forward>() );
//	wxPrintf("FFT::block_dim %d %d %d\n", FFT::block_dim.x,FFT::block_dim.y,FFT::block_dim.z );
	using complex_type = typename FFT::value_type;
	using scalar_type    = typename complex_type::value_type;

	if (rotate)
	{
		// For the twiddle factors ahead of the P size ffts
		twiddle_in = -2*PIf/output_image.dims.y;
		Q = output_image.dims.y / input_image.dims.y; // FIXME assuming for now this is already divisible
	    // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
	    // will be executed on block level. Shared memory is required for co-operation between threads.

		threadsPerBlock = dim3(input_image.dims.y/ept_r, 1, 1); // FIXME make sure its a multiple of 32
		gridDims = dim3(1,1, output_image.dims.w/2);

		shared_mem = output_image.dims.y*sizeof(complex_type) + input_image.dims.y*sizeof(complex_type) + FFT::shared_memory_size;

	}
	else
	{
		// For the twiddle factors ahead of the P size ffts
		float twiddle_in = -2*PIf/output_image.dims.x;
		int   Q = output_image.dims.x / input_image.dims.x; // FIXME assuming for now this is already divisible
	    // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
	    // will be executed on block level. Shared memory is required for co-operation between threads.

		threadsPerBlock = dim3(input_image.dims.x/ept_r, 1, 1); // FIXME make sure its a multiple of 32
		gridDims = dim3(1,1, output_image.dims.y);

		shared_mem = output_image.dims.w/2*sizeof(complex_type) + input_image.dims.x*sizeof(complex_type) + FFT::shared_memory_size;

	}



	if (rotate)
	{
		block_fft_kernel_C2C_WithPadding<FFT,complex_type><< <gridDims,  threadsPerBlock, shared_mem, cudaStreamPerThread>> >
		( (complex_type*)d_rotated_buffer,  (complex_type*)output_image.complex_values_gpu, input_image.dims, output_image.dims, rotate,twiddle_in,Q);
	}
	else
	{
		block_fft_kernel_C2C_WithPadding<FFT,complex_type><< <gridDims,  threadsPerBlock, shared_mem, cudaStreamPerThread>> >
		( (complex_type *)output_image.complex_values_gpu,  (complex_type*)output_image.complex_values_gpu, input_image.dims, output_image.dims, rotate,twiddle_in,Q);
	}

//
	cudaErr(cudaPeekAtLastError());
	cudaErr(cudaDeviceSynchronize());


}

template<class FFT, class ComplexType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_WithPadding(ComplexType* input_values, ComplexType* output_values, int4 dims_in, int4 dims_out, bool rotate, float twiddle_in, int Q)
{

//	// Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_type = ComplexType;

	extern __shared__  complex_type shared_input_complex[];
	complex_type* shared_output;
	complex_type* shared_mem;

	if (rotate) shared_output = (complex_type*)&shared_input_complex[dims_in.y];
	else shared_output = (complex_type*)&shared_input_complex[dims_in.x];

	if (rotate) shared_mem = (complex_type*)&shared_output[dims_out.y];
	else shared_mem = (complex_type*)&shared_output[dims_out.w/2];

	int memory_bounds;
	if (rotate) memory_bounds = dims_out.y;
	else memory_bounds = dims_out.w/2;



	// Memory used by FFT
	complex_type twiddle;
    complex_type thread_data[FFT::storage_size];

    // To re-map the thread index to the data
    int input_MAP[FFT::storage_size];
    // To re-map the decomposed frequency to the full output frequency
    int output_MAP[FFT::storage_size];
    // For a given decomposed fragment
    float twiddle_factor_args[FFT::storage_size];

    // No need to __syncthreads as each thread only accesses its own shared mem anyway
    bah_io::io<FFT>::load_shared(&input_values[blockIdx.z*dims_out.w/2], shared_input_complex, twiddle_factor_args, twiddle_in, input_MAP, output_MAP, Q, 1);
    bah_io::io<FFT>::copy_from_shared(shared_input_complex, thread_data, input_MAP);


	// In the first FFT the modifying twiddle factor is 1 so the data are reeal
	FFT().execute(thread_data, shared_mem);

	bah_io::io<FFT>::store(thread_data,shared_output,output_MAP,1, memory_bounds);


    // For the other fragments we need the initial twiddle
	for (int sub_fft = 1; sub_fft < Q; sub_fft++)
	{

	    bah_io::io<FFT>::copy_from_shared(shared_input_complex, thread_data, input_MAP);


		// cufftDX expects packed real data for a real xform, but we modify with a complex twiddle factor.
		// to get around this, split the complex fft into the sum of the real and imaginary parts
		for (int i = 0; i < FFT::elements_per_thread; i++)
		{
			// Pre shift with twiddle
			__sincosf(twiddle_factor_args[i]*sub_fft,&twiddle.y,&twiddle.x);
			thread_data[i] *= twiddle;
		    // increment the output map. Note this only works for the leading non-zero case
			output_MAP[i]++;
		}

		FFT().execute(thread_data, shared_mem);

		bah_io::io<FFT>::store(thread_data,shared_output,output_MAP,1, memory_bounds);

	}
//
	__syncthreads();

	// Now that the memory output can be coalesced send to global
	int this_idx;
	for (int sub_fft = 0; sub_fft < Q; sub_fft++)
	{
		for (int i = 0; i < FFT::elements_per_thread; i++)
		{
			this_idx = input_MAP[i] + dims_in.x*sub_fft;
			if (this_idx < memory_bounds)
			{
				output_values[blockIdx.z * dims_out.w/2 + this_idx] = shared_output[this_idx];
			}
		}
	}



	return;

}

void DFTbyDecomposition::FFT_R2C_rotate(bool rotate)
{

	// This is the first set of 1d ffts when the input data are real valued, accessing the strided dimension. Since we need the full length, it will actually run a C2C xform

	// FIXME when adding real space complex images
	MyAssertTrue( input_image.is_in_memory_gpu, "Input image is in not on the GPU!");
	MyAssertTrue( is_allocated_rotated_buffer, "Output image is in not on the GPU!");


//	dim3 threadsPerBlock = dim3(test_size/ept_r, 1, 1); // FIXME make sure its a multiple of 32
	// NY R2C Transforms of size NX
	dim3 gridDims = dim3(1,1,(input_image.dims.y + ffts_per_block - 1)/ffts_per_block);

	using FFT = decltype( FFT_4096_r2c() + Direction<fft_direction::forward>() );
//	wxPrintf("FFT::block_dim %d %d %d\n", FFT::block_dim.x,FFT::block_dim.y,FFT::block_dim.z );
	using complex_type = typename FFT::value_type;
	using scalar_type    = typename complex_type::value_type;

//	wxPrintf("In R2C the advised EPT (%d) and ffts per block (%d)\n",FFT::elements_per_thread,FFT::suggested_ffts_per_block);

//    for (int i=0+4094; i < 5+4094; i++)
//    {
//    	input_image.printVal("val",i);
//    }
//    for (int i=4090+4094; i < 4098+4094; i++)
//    {
//    	input_image.printVal("val",i);
//    }
	cudaError_t error_code = cudaSuccess;
	auto workspace = make_workspace<FFT>(error_code);
	block_fft_kernel_R2C_rotate<FFT,complex_type,scalar_type><< <gridDims,  FFT::block_dim, FFT::shared_memory_size, cudaStreamPerThread>> >
	( (scalar_type *)input_image.real_values_gpu,  (complex_type*)d_rotated_buffer, input_image.dims, output_image.dims, workspace, rotate);

//	cudaErr(cudaPeekAtLastError());
//	cudaErr(cudaDeviceSynchronize());


}

template<class FFT, class ComplexType, class ScalarType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_R2C_rotate(ScalarType* input_values, ComplexType* output_values, int4 dims_in, int4 dims_out, typename FFT::workspace_type workspace, bool rotate)
{

	// FIXME using exact sizes so every thread and every block is included. Need overflow checks
	if (ffts_per_block*blockIdx.z > dims_in.y-ffts_per_block) return;
//	// Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_type = ComplexType;
    using scalar_type  = ScalarType;

	extern __shared__  complex_type shared_mem[];
    complex_type thread_data[FFT::storage_size];



    bah_io::io<FFT>::load_r2c(&input_values[ffts_per_block*blockIdx.z * dims_in.w], thread_data, dims_in.w*threadIdx.y);



    FFT().execute(thread_data, shared_mem, workspace);

    // index gives us x in the unrotated line, and blockIdx.x*dims_in.w gives us y
    // x' is = y, and y' = dims_in.w/2 - x - 1

    if (rotate)
    {
        // blockIdx.z + (dims_out.w/2 - index - 1)*dims_out.y
        int rotated_offset[2] = {(int)blockIdx.z + (dims_out.w/2 - 1)*dims_out.y, -int(dims_out.y)};
        bah_io::io<FFT>::store_r2c_rotated(thread_data, output_values, rotated_offset);


    }
    else
    {
        bah_io::io<FFT>::store_r2c(thread_data, &output_values[ffts_per_block*blockIdx.z * dims_in.w/2], dims_out.w/2*threadIdx.y);
    }

	return;

}

void DFTbyDecomposition::FFT_C2C_rotate(bool rotate, bool forward_transform)
{

	// This is the first set of 1d ffts when the input data are real valued, accessing the strided dimension. Since we need the full length, it will actually run a C2C xform

	// FIXME when adding real space complex images
	MyAssertTrue( is_allocated_rotated_buffer, "Input image is in not on the GPU!");
	MyAssertTrue( output_image.is_in_memory_gpu, "Output Image is not on the GPU");

//	dim3 threadsPerBlock = dim3(test_size/ept_c,1,1);
	// The rotated image now has size NY x NW/2
	dim3 gridDims = dim3(1,1,(output_image.dims.w/2+ffts_per_block-1)/ffts_per_block);



		if (forward_transform)
		{
		    using FFT = decltype(FFT_4096_c2c() + Direction<fft_direction::forward>() );
		    cudaError_t error_code = cudaSuccess;
		    auto workspace = make_workspace<FFT>(error_code);
		    using complex_type = typename FFT::value_type;

		    // On the forward the input is in the buffer, do an out of place transform and put back into the roiginal memory
			block_fft_kernel_C2C_rotate<FFT, complex_type><< <gridDims, FFT::block_dim, FFT::shared_memory_size, cudaStreamPerThread>> >
			( (complex_type*)d_rotated_buffer, (complex_type*)output_image.complex_values_gpu, input_image.dims, output_image.dims, workspace, rotate);
//		    cudaErr(cudaPeekAtLastError());
//		    cudaErr(cudaDeviceSynchronize());


		}
		else
		{
		    using FFT = decltype(FFT_4096_c2c() + Direction<fft_direction::inverse>() );
		    cudaError_t error_code = cudaSuccess;
		    auto workspace = make_workspace<FFT>(error_code);
		    using complex_type = typename FFT::value_type;

			// On the inverse, do out of place and put back into the bufffer
			block_fft_kernel_C2C_rotate<FFT, complex_type><< <gridDims,  FFT::block_dim, FFT::shared_memory_size, cudaStreamPerThread>> >
			( (complex_type*)output_image.complex_values_gpu, (complex_type*)d_rotated_buffer, input_image.dims, output_image.dims, workspace, rotate);
//		    cudaErr(cudaPeekAtLastError());
//		    cudaErr(cudaDeviceSynchronize());


		}




}

template<class FFT, class ComplexType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2C_rotate(ComplexType* input_values, ComplexType* output_values, int4 dims_in, int4 dims_out, typename FFT::workspace_type workspace, bool rotate)
{

	// FIXME using exact sizes so every thread and every block is included. Need overflow checks
	if (ffts_per_block*blockIdx.z > dims_out.w/2-ffts_per_block) return;
//	// Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_type = ComplexType;

	extern __shared__  complex_type shared_mem[];

    complex_type thread_data[FFT::storage_size];
    int source_idx[FFT::storage_size];


    if (rotate)
    {
        bah_io::io<FFT>::load(&input_values[ffts_per_block*blockIdx.z*dims_out.y], thread_data, source_idx, 1, (int)threadIdx.y);

    }
    else
    {
        bah_io::io<FFT>::load(&input_values[ffts_per_block*blockIdx.z], thread_data, source_idx, dims_out.w/2, (int)threadIdx.y);

    }




    FFT().execute(thread_data, shared_mem, workspace);

    if (rotate)
    {
        bah_io::io<FFT>::store(thread_data, &output_values[ffts_per_block*blockIdx.z], source_idx, 1);

//		index = threadIdx.x;
//		for (int i = 0; i < FFT::elements_per_thread; i++)
//		{
//			output_values[blockIdx.x*dims_out.y + index] = thread_data[i];
//			index += stride;
//		}
    }
    else
    {
        bah_io::io<FFT>::store(thread_data, &output_values[ffts_per_block*blockIdx.z], source_idx, dims_out.w/2);
    }


	return;

}

void DFTbyDecomposition::FFT_C2R_rotate(bool rotate)
{

	// This is the first set of 1d ffts when the input data are real valued, accessing the strided dimension. Since we need the full length, it will actually run a C2C xform

	// FIXME when adding real space complex images
	MyAssertTrue( is_allocated_rotated_buffer, "Input image is in not on the GPU!");
	MyAssertTrue( input_image.is_in_memory_gpu, "Output image is in not on the GPU!");

	// Elements per thread must be [2,32]

//	dim3 threadsPerBlock = dim3(test_size/ept_r,1,1); // FIXME make sure its a multiple of 32
	dim3 gridDims = dim3(1,1,(output_image.dims.y+ffts_per_block-1)/ffts_per_block);


	using FFT = decltype(FFT_4096_c2r() + Direction<fft_direction::inverse>() );
	cudaError_t error_code = cudaSuccess;
	auto workspace = make_workspace<FFT>(error_code);
	using complex_type = typename FFT::value_type;
	using scalar_type    = typename complex_type::value_type;

	// On the inverse, do out of place and put back into the bufffer
	block_fft_kernel_C2R_rotate<FFT, complex_type, scalar_type><< <gridDims,  FFT::block_dim, FFT::shared_memory_size, cudaStreamPerThread>> >
	( (complex_type*)d_rotated_buffer, (scalar_type*)output_image.real_values_gpu, input_image.dims, output_image.dims, workspace, rotate);
//	cudaErr(cudaPeekAtLastError());
//	cudaErr(cudaDeviceSynchronize());
//
//	output_image.MultiplyByConstant(1/4096);
//	cudaErr(cudaPeekAtLastError());
//	cudaErr(cudaDeviceSynchronize());
//    for (int i=0+4094; i < 5+4094; i++)
//    {
//    	output_image.printVal("val out",i);
//    }
//    for (int i=4090+4094; i < 4098+4094; i++)
//    {
//    	output_image.printVal("val out",i);
//    }



}

template<class FFT, class ComplexType, class ScalarType>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_C2R_rotate(ComplexType* input_values, ScalarType* output_values, int4 dims_in, int4 dims_out, typename FFT::workspace_type workspace, bool rotate)
{

	// FIXME using exact sizes so every thread and every block is included. Need overflow checks
	if (ffts_per_block*blockIdx.z > dims_out.y - ffts_per_block) return;
//	// Initialize the shared memory, assuming everyting matches the input data X size in
	//	// Initialize the shared memory, assuming everyting matches the input data X size in
	using complex_type = ComplexType;
	using scalar_type  = ScalarType;

	extern __shared__  complex_type shared_mem[];


    complex_type thread_data[FFT::storage_size];
	int index = threadIdx.x;
	constexpr int half_idx = FFT::elements_per_thread / 2;
	constexpr int stride =  (size_of<FFT>::value / FFT::elements_per_thread);
    if (rotate)
    {


		// inputs are NY xforms of length NW/2, read in strided and rotate
		// blockIdx.x is Y in the rotated frame
    	// blockIdx.z + dims_out.y * (dims_out.w/2 - index - 1)
    	int rotated_offset[2] = {blockIdx.z + dims_out.y * (dims_out.w/2 - 1), -(int)dims_out.y};
        bah_io::io<FFT>::load_c2r_rotated(input_values, thread_data, rotated_offset);

//		for (int i = 0; i < half_idx; i++)
//		{
//			if (rotate) thread_data[i] = __ldg((const double*)&input_values[blockIdx.z + dims_out.y * (dims_out.w/2 - index - 1)]);
//			else thread_data[i] = __ldg((const double*)&input_values[blockIdx.z * dims_out.w/2 + index]);
//			index += stride;
//		}
//
//		constexpr unsigned int threads_per_fft       = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
//		constexpr unsigned int output_values_to_load = (cufftdx::size_of<FFT>::value / 2) + 1;
//		// threads_per_fft == 1 means that EPT == SIZE, so we need to load one more element
//		constexpr unsigned int values_left_to_load =
//			threads_per_fft == 1 ? 1 : (output_values_to_load % threads_per_fft);
//		if (threadIdx.x < values_left_to_load)
//		{
//			thread_data[half_idx] = __ldg((const double*)&input_values[blockIdx.z + dims_out.y * (dims_out.w/2 - index - 1)]);
////			else thread_data[half_idx] = __ldg((const double*)&input_values[blockIdx.z * dims_out.w/2 + index]);
//
//		}
	}
    else
    {
        bah_io::io<FFT>::load_c2r(&input_values[ffts_per_block*blockIdx.z * dims_out.w/2], thread_data, dims_out.w/2*threadIdx.y);
    }
//	bah_io::io<FFT>::load(&input_values[blockIdx.z * dims_out.y], thread_data, 1);


    // For loop zero the twiddles don't need to be computed
    FFT().execute(thread_data, shared_mem, workspace);

    if (rotate)
    {
        bah_io::io<FFT>::store_c2r(thread_data, &output_values[ffts_per_block*blockIdx.z * dims_out.w],dims_out.w * threadIdx.y);

//		index = threadIdx.x;
//		for (int i = 0; i < FFT::elements_per_thread; i++)
//			{
//				output_values[index + blockIdx.z*dims_out.w] = reinterpret_cast<const scalar_type*>(thread_data)[i];
//				index += stride;
//			}
    }
    else
    {
        bah_io::io<FFT>::store_c2r(thread_data, &output_values[ffts_per_block*blockIdx.z * dims_out.w],dims_out.w * threadIdx.y);
    }

	return;

}


template<class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_kernel_r2cT(ScalarType* input_data, ComplexType* output_data)
{

	// FIXME using exact sizes so every thread and every block is included. Need overflow checks
    using complex_type = ComplexType;
    using scalar_type = ScalarType;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];


    int source_idx[FFT::storage_size];

        for (int i = 0; i < FFT::elements_per_thread; i++)
        {
        	source_idx[i] = threadIdx.x + i * (size_of<FFT>::value / FFT::elements_per_thread);
        	reinterpret_cast<scalar_type*>(thread_data)[i] = __ldg((const float*)&input_data[blockIdx.x * 4098 + source_idx[i]]);
        }

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Load data from global memory to registers
//    example::io<FFT>::load_r2c(input_data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Save results
//    example::io<FFT>::store_r2c(thread_data, output_data, local_fft_id);
}

// In this example a one-dimensional real-to-complex transform is performed by a CUDA block.
//
// One block is run, it calculates two 128-point R2C float precision FFTs.
// Data is generated on host, copied to device buffer, and then results are copied back to host.
// Notice different sizes of input and output buffer, and R2C load and store operations in the kernel.
//template<unsigned int Arch>
int DFTbyDecomposition::test_main() {
    using namespace cufftdx;
    unsigned int Arch = 700;
    wxPrintf("in simple\n");
    // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
    // will be executed on block level. Shared memory is required for co-operation between threads.
    using FFT          = decltype(Block() + Size<4096>() + Type<fft_type::r2c>() + Direction<fft_direction::forward>() +
                         Precision<float>() + ElementsPerThread<8>() + FFTsPerBlock<1>() + SM<700>());
    using complex_type = typename FFT::value_type;
    using real_type    = typename complex_type::value_type;

    // Allocate managed memory for input/output
    real_type* input_data;

    auto       input_size       =2* FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    auto       input_size_bytes = input_size * sizeof(real_type);

    float* cpu_input = new float[input_size];
    for (size_t i = 0; i < input_size; i++) {
    	cpu_input[i] = float(i);
    }

    cudaErr(cudaMalloc(&input_data, input_size_bytes));
    cudaErr(cudaMemcpy( input_data, cpu_input,input_size_bytes,cudaMemcpyHostToDevice));
//    cudaErr(cudaMallocManaged(&input_data, input_size_bytes));
//    for (size_t i = 0; i < input_size; i++) {
//        input_data[i] = float(i);
//    }
    MyPrintWithDetails("");
    complex_type* output_data;
    auto          output_size       = FFT::ffts_per_block * 2*(cufftdx::size_of<FFT>::value / 2 + 1);
    auto          output_size_bytes = output_size * sizeof(complex_type);
    cudaErr(cudaMallocManaged(&output_data, output_size_bytes));
    MyPrintWithDetails("");
//    std::cout << "input [1st FFT]:\n";
//    for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
//        std::cout << input_data[i] << std::endl;
//    }
    std::cout << "Block dim" << FFT::block_dim.x << FFT::block_dim.y << FFT::block_dim.z << std::endl;
    MyPrintWithDetails("");
wxPrintf("Size of float %d  and size of real_typ%d\n",sizeof(float),sizeof(real_type));
    // Invokes kernel with FFT::block_dim threads in CUDA block
	real_type* dummy_ptr = reinterpret_cast<real_type*>(input_image.real_values_gpu);
//    block_fft_kernel_r2cT<FFT><<<2, FFT::block_dim, FFT::shared_memory_size>>>(input_data, output_data);
    block_fft_kernel_r2cT<FFT><<<2, FFT::block_dim, FFT::shared_memory_size>>>(dummy_ptr, output_data);

    cudaErr(cudaPeekAtLastError());
    cudaErr(cudaDeviceSynchronize());

    MyPrintWithDetails("");

    std::cout << "output [1st FFT]:\n";
    for (size_t i = 0; i < (cufftdx::size_of<FFT>::value / 2 + 1); i++) {
        std::cout << output_data[i].x << " " << output_data[i].y << std::endl;
        wxPrintf("%3.3f %3.3f\n",output_data[i].x,output_data[i].y);
    }
    MyPrintWithDetails("");

    std::cout << "arch" << Arch << std::endl;
    std::cout << "max threads" << FFT::max_threads_per_block << std::endl;
    cudaErr(cudaFree(input_data));
    cudaErr(cudaFree(output_data));
    std::cout << "Success" << std::endl;

    return 0;
}

//template<unsigned int Arch>
//struct simple_block_fft_r2c_functor {
//    void operator()() { return simple_block_fft_r2c<Arch>(); }
//};
//
//int DFTbyDecomposition::test_main()
//{
//	wxPrintf("In main\n");
//    return example::sm_runner<simple_block_fft_r2c_functor>();
//}

