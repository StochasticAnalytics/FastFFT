# Basic usage principles

## Overview

The FastFFT library is built to optimize a subset of discrete Fourier transforms, most of which involve padding/cropping, *i.e.*, size changes. While a substantial effort has been made to maintain parallels between FastFFT and other popular libraries like cuFFT and FFTW3, the ability to use variable input/output sizes and input/compute/output variable types complicates the interface to some degree.

## cuFFT plans, a review

Creating a plan is a fundamental process in dealing with fft libraries. We'll use the process in cuFFT for illustration.

### Creating a plan

In cufft, the first step to library access is to create a "handle" to a plan, *i.e.*, a pointer, which is needed for both forward and inverse transforms. Not required, *per se*, but highly recommended is to place that plan in a specific cuda stream. A cudaStream allows for fine grained control of a queue of work. All processes in the FastFFT library are placed into the cudaStreamPerThread stream, a special, non-synchronzing stream that has a unique idea for every unique host thread, permitting easy thread saftey without explicit management by using common Host thread management strategies like openMP.

```cpp
    cufftCreate(&cuda_plan_forward);
    cufftCreate(&cuda_plan_inverse);

    cufftSetStream(cuda_plan_forward, cudaStreamPerThread);
    cufftSetStream(cuda_plan_inverse, cudaStreamPerThread);

    // The parallel in Fast FFT would be to create an empty FourierTransformer object, e.g.
    // The template arguments are: ComputeBaseType, InputType, OutputBaseType, Rank
    FastFFT::FourierTransformer<float, float, float, 2> FT;
```

```{note}
The current implementation only supports float for all three stages and dimensions (rank) of 2,3. This is under active development.

Also note that the compute and output are **base** types, while the input is the **full** type.

* The input may be a real (__half or float) or complex (__half2 or float2) image and this depends only on the input data.
* The output may be a real or complex image or a complex FFT, which depends on the FFT algorithm(s) actually called by the user.
* The compute type is always a complex image, and this is a limitation of the cuFFT library.
    * This is the reason for full/base type distinction.
* Any algorithms that couple a second image to one of the intra process functors assumes the image type to have the same base type as the input, and the full type depends on the stage it is used.
    * For example, the correlation functor assumes the input is a real image, and the second image is a complex image, and the output is a real image.
```

The next step in the cuFFT library is to actually create the plan itself, which requires informing the cuda driver of several parameters.

* Similar to the templated declarion of the FastFFT::FourierTransformer object, the input/compute/output datatypes are all specified.

* Batched and strided FFT's are allowed in cuFFT while in FastFFT we are currently only supporting plans for stride of 1 and individual
transforms, although using the library in a batched manner would be trivial to add.

* Perhaps the most significant difference to note, is that in cuFFT, the dinmensionality of the transform is fixed, while in FastFFT the input and output sizes may, and likely should be different.

```cpp

    cufftXtMakePlanMany(cuda_plan_forward, rank, fftDims,
                         NULL, 1, 1, CUDA_R_32F,
                         NULL, 1, 1, CUDA_C_32F, iBatch, &cuda_plan_worksize_forward, CUDA_C_32F);

    FT.SetForwardFFTPlan(input_size.x, input_size.y, input_size.z, input_size.x, input_size.y, input_size.z);
    FT.SetInverseFFTPlan(input_size.x, input_size.y, input_size.z, output_size.x, output_size.y, output_size.z);
```

```{note}
Both SetForwardFFTPlan and SetInverseFFTPlan must be called prior to using any FFT algorithms, and the buffer memory used by FastFFT is allocated automatically on the latter of the two calls.
```

* Create an FourierTransformer object
* Set the forward and inverse plans
  * If these are not set, the ValidateDimensions function will throw an error in debug mode
* Optionally Set input and/or output pointers
  * This will instruct FastFFT to read/write on relevant transforms to and from these external memory buffers
  * Otherwise, data must be manually copied to/from the FastFFT buffers using the relevant methods
    * FastFFT::CopyDeviceToDevice
    * FastFFT::CopyDeviceToHost
    * FastFFT::CopyHostToDevice
    * FastFFT::CopyDeviceToDeviceFromNonOwningAddress
      * If the input/output pointers are not set and the FastFFT buffers are not allocated, the will be allocated on the first call to either of the latter two functions
