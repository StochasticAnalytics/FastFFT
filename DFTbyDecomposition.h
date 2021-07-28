/*
 * DFTbyDecomposition.h
 *
 *  Created on: Oct 21, 2020
 *      Author: himesb
 */


#ifndef SRC_GPU_DFTBYDECOMPOSITION_H_
#define SRC_GPU_DFTBYDECOMPOSITION_H_

class DFTbyDecomposition
{

public:

	DFTbyDecomposition();
	virtual ~DFTbyDecomposition();
	DFTbyDecomposition(const DFTbyDecomposition &other);
	DFTbyDecomposition& operator=(const DFTbyDecomposition &other);

	// These properties are for testing and should not be in the final class
	StopWatch timer;
	GpuImage baseline_image;
	GpuImage input_image;
	GpuImage output_image;
	int2 dims_input;
	int2 dims_output;

	int numel;
	// These will point to the memory in the GpuImages, but won't have memory allocated for them.
	cufftReal* d_input_image_real;
	cufftComplex* d_input_image_complex;
	cufftComplex* d_rotated_buffer;
	bool is_set_gpu_images;
	bool is_allocated_rotated_buffer;

	float* twiddles;
	bool is_set_twiddles;
	int shared_mem;



	// To be used in development where a fake image is created.
	void InitTestCase(int wanted_input_size_x, int wanted_input_size_y, int wanted_output_size_x, int wanted_output_size_y);
	void SetGpuImages(Image& cpu_input, Image& cpu_output);
	void AllocateRotatedBuffer();
	void DFT_R2C_WithPadding();
	void DFT_C2C_WithPadding();
	void DFT_R2C_WithPadding_strided();
	void DFT_C2C_WithPadding_strided();
	void DFT_C2C_WithPadding_rdx2();
	void FFT_R2C_WithPadding(bool rotate);
	void FFT_C2C_WithPadding(bool rotate);
	void FFT_R2C_WithPadding_strided(bool do_rotate);
	void FFT_C2C_WithPadding_strided(bool do_rotate);
	void FFT_R2C_rotate(bool rotate);
	void FFT_C2C_rotate(bool rotate, bool forward_transform);
	void FFT_C2R_rotate(bool rotate);


	int test_main();











};

#endif /* SRC_GPU_DFTBYDECOMPOSITION_H_ */
