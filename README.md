# FastFFT


## Project Summary

#### Goals:
This project aims to accelerate a specific subset of Fast Fourier Transforms (FFTs) especially important image processing, my particular flavor of which is  high-resolution phase-contrast cryo-Electron Microscopy. The gist is to take advantage of the cufftdx library from Nvidia, currently in early access, as well as algorithmic ideas from *Sorensen et. al* [[1]](#1). I became aware of [VkFFT](https://github.com/DTolm/VkFFT) after I had finished my initial [proof of principle](https://github.com/bHimes/cisTEM_downstream_bah/blob/DFT/src/gpu/DFTbyDecomposition.cu) experiments with cufftdx in Oct 2020. It may be entirely possible to do the same things using that library, though I haven't had a change to look through the source.

An additional experiment tied to this project, is the development of the scientific manuscript "live" alongside the development of the code. While most of my projects are pretty safe from being scooped, this one is decidedly "scoopable" as it should have broad impact, and also shouldn't be to hard to implement once the info is presented.

#### Design:

The FourierTransformer class, in the FastFFT namespace may be used in your cpp/cuda project by including the *header.* The basic usage is as follows:

- The input/output data pointers, size and type are set. (analagous to the cufftXtMakePlanMany)
- Class methods are provided to handle memory allocation on the gpu as well as transfer to/from host/device.
- Simple methods for FwdFFT, InvFFT, CrossCorrelation etc. are public and call the correct combination of substranforms based on the data and padding/trimming wanted.



---

## Abstract

The Fast Fourier transform is one of the most widely used and heavily optimized algorithms in digital signal processing. cryoEM makes heavy use of the FFT as it can accelerate convolution operations used for image alignment, and also in reconstruction algorithms that operate most accurately in Fourier space. While FFT libraries like FFTW and cuFFT provide routines for highly-optimized general purpose multi-dimensional FFTs; however, they overlook several use-cases where only a subset of the input or output points are required. We show here algorithms based on transform decomposition are well suited to the memory hierarchy on moden GPUs, and can be implemented using the cufftdx header library to accelerate several important algorithms by factors of 3-10x over Nvidia’s cuFFT library. These include movie-frame alignment, image resampling via Fourier cropping, 2d and 3d template matching, and subtomogram averaging and alignment.

## Introduction

#### The Fourier Transform

The Fourier Trasnform is a mathematical operation that converts a function between two dual spaces, for example, time and frequency, or position and momentum :exclamation:cite. A function that is well localized in postion (commonly "real" space) will be delocalized in momentum (commonly "Fourier" or "K" space) and vice-versa. Converting between different representations of the same function has many practical uses; of particular interest to the author is in image filtering. [:exclamation:Provide some examples, low/high pass, frequency marching algs etc.] [:exclamation: provide equation, maybe explain different conventions (wolfram is useful.)]

#### The discrete Fourier Trasnform

The discrete Fourier Transform (DFT) extends the operation of the Fourier Transform to a band-limited sequence of evenly spaced samples of a continous function :exclamation:cite. 
  - equation
  - list properties (And brief example with a few citations for where each is capitalized on.)
    - linearity
    - sinc interpolation
    - Parsevals
    - Convolution theorem
    - Fourier Slice theorem

#### the fast (discrete) Fourier Transform

In looking at [:exclamation:DFT equation above] it is clear that the DFT requires O(n^2) complex exponential, multiplications and additions. The fast Fourier Transform (FFT) reduces the compuational complexity to O(Nlog_2(N)) with the most efficient algorithm, the split-radix FFT requiring just 4Nlog_2(N) -6N+8 operations :exclamation:cite. While discussions on the FFT center on computational complexity from the standpoint of floating point operations, on modern GPU hardware they are memory-bandwidth limited, with an estimated 80% of the computational run-time being dedicated to moving data on and off chip :exclamation:cite.

#### exploiting zero values

- Concept, reduce ops, but especially i/o
- Mention pruning
- Introduce transform decomposition (Sorensen)
- Pictorial explanation for the major benefactors, also list estimate of ops. 
  - Movie alignment
  - 2D TM
  - 3D TM
  - Subtomogram averaging


## Theory

#### The DFT and FFT

Fast Fourier Transform (FFT) is a mathematical operation that transforms a function <img src="https://render.githubusercontent.com/render/math?math=X(n)"> from the real-valued plane into the complex-valued plane. The function <img src="https://render.githubusercontent.com/render/math?math=f(x)"> is a function of <img src="https://render.githubusercontent.com/render/math?math=x"> and is often a function of the real-valued signal <img src="https://render.githubusercontent.com/render/math?math=x"> or a function of the complex-valued signal <img src="https://render.githubusercontent.com/render/math?math=x + i\cdot y">. The FFT is an optimized algorithm for copying the discrete Fourier transform (DFT) defined as: 

<img src="https://render.githubusercontent.com/render/math?math=X(k) = \sum_{n=0}^{N-1} x(n) \exp\left( -2\pi i k n \right)">  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Eq. (1)

( I think this paragraph is probably belonging in the introduction )

The FFT is useful in image analysis as it can help to isolate repetitive features of a given size. It is also useful in signal processing as it can be used to perform convolution operations. Multidimensional FFTs are fully seperable, such that in the simpliest case, FFTs in higher dimensions are composed of a sequence of 1D FFTs. For example, in two dimensions, the FFT of a 2D image is composed of the 1D FFTs of the rows and columns. A naive implementation is compuatationally ineffecient due to the strided memory access pattern:exclamation:cite. One solution is to transpose the data after the FFT and then transpose the result back :exclamation:cite. This requires extra trips to and from main memory. Another solution is to decompose the 2D transform using vector-radix FFTs :exclamation:cite.

#### Summary of GPU memory hierarchy as it pertains to this work

#### Graphic to illustrate how the implicit transpose and real-space shifting reduces memory access for forward padding

#### Maths for further reduced memory access and computation using transform decomposition on sparse inputs

#### Maths for partial outputs

#### 



## Results

### Basic performance

#### Comparing 2D FFTs using cufftdx w/o reordering

By design, the cufft library from Nvidia returns an FFT in the natural order [TODO check term] which requires two transpose operations, which in many cases seem to be optimized to reduce global memory access via vector-radix transforms. This means that the numbers we compare here are not quite apples to apples, as the result from FastFFT is transposed in memory in the Fourier domain.

##### Table 1: FFT/iFFT pairs

| 2D square size | cufft/FastFFT runtime (10k iterations) |
| --- | ---- |
| 64 | 2.34 |
| 128 | 2.39 |
| 256 | 2.06 |
| 512 | 1.20 |
| 1024 | 0.92 |
| 4096 | 1.17 | 

:biohazard: None of the kernels are even remotely optimized at this point, they have only been assembled and tested to pass expected behavior for FFTs of constant functions, unit impulse functions, and basic convolution ops.

#### Comparing 2D FFT based convolution 

##### Table 2: zero padded convolution of 4096 pixel sq. image

| 2D square size kernel size | cufft/FastFFT runtime (10k iterations) |
| --- | ---- |
| 64 | 2.79 |
| 128 | 2.81 |
| 256 | 2.71 |
| 512 | 2.66 |
| 1024 | 2.48 |

:biohazard: None of the kernels are even remotely optimized at this point, they have only been assembled and tested to pass expected behavior for FFTs of constant functions, unit impulse functions, and basic convolution ops.

- Movie alignment expense (Pre/post process and alignment percentages.)

  :soon: 
- Input movies are sparse (speed-up)
- Peak search is limited, so IFFT is sped up
- 2D template matching, 
- zero padding of template
- Improved kernel fusion
- 3D template matching/ subtomogram averaging.
- Zero padding
- Error analysis for using half precision. Which applications might it be useful for


<img src="https://render.githubusercontent.com/render/math?math=\sum_{n=0}^{N-1} e^{i \pi} = -1">

## Discussion

We’ve assessed a few specific sources of image degradation, buy the tool could capture others.
Part of regular workflow or use in data collection
Simple standalone tool to use - can run on common facility computers

## Conclusion

## References

<a id="1">[1]</a> 
H. V. Sorensen and C. S. Burrus, "Efficient computation of the DFT with only a subset of input or output points," in IEEE Transactions on Signal Processing, vol. 41, no. 3, pp. 1184-1200, March 1993, doi: 10.1109/78.205723.


