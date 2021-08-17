# FastFFT


## Broad strokes

This project aims to accelerate a specific set of FFTs important for cryo-Electron Microscopy. It does so by taking advantage of the cufftdx library from Nvidia, currently in early access as well as algorithmic ideas from *Sorensen et. al* [[1]](#1). An additional experimental aspect is the development of the scientific manuscript "live" alongside the development of the code. 



---

## Abstract

The Fast Fourier transform is one of the most widely used and heavily optimized algorithms in digital signal processing. cryoEM makes heavy use of the FFT as it can accelerate convolution operations and reconstruction algorithms operate most accurately in Fourier space thanks to the Fourier slice theorem. While FFT libraries like FFTW and cuFFT provide highly optimized generic multi-dimensional FFT, these plans ignore several scenarios in cryoEM where only a subset of the input or output points are required. We show here how the cuFFTdx header library can be used to accelerate several important algorithms by factors of 3-10x over Nvidia’s cuFFT library. These include movie alignment, 2d and 3d template matching and subtomogram averaging.

## Introduction


- What is the DFT and why is it used in cryoEM
- What is the FFT and how does it accelerate (mem explanation as well.)
- Reference Sorensen and give  a quick recap.
- Pictorial explanation for the major benefactors.
  - Movie alignment
  - 2D TM
  - 3D TM
  - Subtomogram averaging
- In addition to reduced ops, there are also memory advantages to operating on transposed data (for all problems) and especially for points from (4) as a direct write of uncoalesced values can be reduced by carefully choosing the axis.
- Possibility of half-precision (also how to get around excessive padding due to power of 2 constraint, and that cuFFTdx is limited to 4096)


## Theory

Fast Fourier Transform (FFT) is a mathematical operation that transforms a function $f(x)$ from the real-valued plane into the complex-valued plane. The function <img src="https://render.githubusercontent.com/render/math?math=f(x)"> is a function of <img src="https://render.githubusercontent.com/render/math?math=x"> and is often a function of the real-valued signal <img src="https://render.githubusercontent.com/render/math?math=x"> or a function of the complex-valued signal <img src="https://render.githubusercontent.com/render/math?math=x + i\cdot y">. The FFT is an optimized algorithm for copying the discrete Fourier transform (DFT) defined as: 

<img src="https://render.githubusercontent.com/render/math?math=X(k) = \sum_{n=0}^{N-1} x(n) \exp\left( -2\pi i k n \right)">  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Eq. (1)




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


