
# FastFFT


---

(MS-label)=
## Abstract

The Fast Fourier transform is one of the most widely used and heavily optimized algorithms in digital signal processing. cryo-EM makes heavy use of the FFT as it can accelerate convolution operations used for image alignment, and also in reconstruction algorithms that operate most accurately in Fourier space. FFT libraries like FFTW and cuFFT provide routines for highly-optimized general purpose multi-dimensional FFTs; however, they overlook several use-cases where only a subset of the input or output points are required. We show here algorithms based on transform decomposition are well suited to the memory hierarchy on moden GPUs, and can be implemented using the cufftdx header library to accelerate several important algorithms by factors of 3-10x over Nvidia’s cuFFT library. These include movie-frame alignment, image resampling via Fourier cropping, 2d and 3d template matching, and subtomogram averaging and alignment.

## Introduction


The Discrete Fourier Transform (DFT) and linear filtering, *e.g.* convolution, are among the most common operations in digital signal processing. It is assumed that the reader has basic familiarity with Fourier Analysis and it's applications in their respective fields; we will focus here on digital image processing for convenience. For a detailed introduction to the reader is referred to the free book by Smith {cite:p}`smith_mathematics_2008`. 


#### The discrete Fourier Trasnform

The discrete Fourier Transform (DFT) extends the operation of the Fourier Transform to a band-limited sequence of evenly spaced samples of a continous function. In one dimension, it is defined for a sequence of N samples $x(n)$ as:

```{math}
: label : dft_equation
X(k) = \sum_{n=0}^{N-1} x(n) \exp\left( -2\pi i k n \right) 
```

  - ⚠️ TODO: list properties (And brief example with a few citations, preferably specific to cryo-EM where each is capitalized on.)
    - linearity
    - Parsevals
    - Convolution theorem
    - sinc interpolation
    - Fourier Slice theorem
    - 

#### the fast (discrete) Fourier Transform

In looking at [⚠️ DFT equation above] it is clear that the DFT requires $ O(N^2) $
 complex exponential function evaluations, multiplications, and additions. The fast Fourier Transform (FFT) reduces the compuational complexity to $ O(Nlog_2{N}) $
 with the most efficient algorithm, the split-radix FFT requiring just $ 4Nlog_2{N} - 6N  $. The Cooley-Tukey algorithm {cite:p}`cooley_algorithm_1965` was published little more than a decade after the first digitial computers became available. As is often the case in science, their discovery was really a re-discovery; the divide and conquer approach that underpins the FFT was already known to Gauss as early as 1805, predating Fourier's work itself! {cite:p}`heideman_gauss_1985` 

 ```{epigraph}
This story of the FFT can be used to give one incentive to investigate not
only new and novel approaches, but to occasionally look over old papers and see the variety of tricks and clever ideas which were used when computing was, by itself, a laborious chore which gave clever people great incentive to develop efficient methods. Perhaps among the ideas discarded before the days of electronic computers, we may find more seeds of new
algorithms.

-- James W. Cooley {cite}`cooley_re-discovery_1987`
```

This present work itself follows from this same spirit of re-discovery; presently with respect to ideas discarded before the days of efficient graphics processing units (GPUs), rather than electronic computers on a whole.

⚠️ Segue to include notes from FFTW - before last PP, something something FFTW is an example of dev since then - as those authors note, pruning something something, note on arithmetic vs caches (cite actual FFTW paper) something something.

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

Fast Fourier Transform (FFT) is a mathematical operation that transforms a function $ X(n) $ from the real-valued plane into the complex-valued plane. The function $ f(x) $ is a function of $ x $ and is often a function of the real-valued signal $ x $ or a function of the complex-valued signal $ x + i\cdot y $. The FFT is an optimized algorithm for copying the discrete Fourier transform (DFT) defined as: 


( I think this paragraph is probably belonging in the introduction )

The FFT is useful in image analysis as it can help to isolate repetitive features of a given size. It is also useful in signal processing as it can be used to perform convolution operations. Multidimensional FFTs are fully seperable, such that in the simpliest case, FFTs in higher dimensions are composed of a sequence of 1D FFTs. For example, in two dimensions, the FFT of a 2D image is composed of the 1D FFTs of the rows and columns. A naive implementation is compuatationally ineffecient due to the strided memory access pattern⚠️. One solution is to transpose the data after the FFT and then transpose the result back ⚠️. This requires extra trips to and from main memory. Another solution is to decompose the 2D transform using vector-radix FFTs ⚠️.

#### Summary of GPU memory hierarchy as it pertains to this work

Working with a subset of input or output points for an FFT exposes several possibilites to exploit fine grained control of the memory hierarchy available on modern Nvidia GPUs. 


#### Graphic to illustrate how the implicit transpose and real-space shifting reduces memory access for forward padding

#### Maths for further reduced memory access and computation using transform decomposition on sparse inputs

#### Maths for partial outputs

#### 



## Results

### Basic performance

#### Comparing 2D FFTs using cufftdx w/o reordering

By design, the cufft library from Nvidia returns an FFT in the natural order [TODO check term] which requires two transpose operations, which in many cases seem to be optimized to reduce global memory access via vector-radix transforms. This means that the numbers we compare here are not quite apples to apples, as the result from FastFFT is transposed in memory in the Fourier domain.

##### Table 1: FFT/iFFT pairs

| 2D square size | cufft/FastFFT runtime (10k iterations) | after re-org of code  |
| --- | ---- | ----|
| 64 | 2.34 |  1.17 |
| 128 | 2.39 | 1.38 | 
| 256 | 2.06 | 1.25 |
| 512 | 1.20 | 0.92 |
| 1024 | 0.92 | 0.65 |
| 4096 | 1.17 | 1.14 |

🍍 None of the kernels are even remotely optimized at this point, they have only been assembled and tested to pass expected behavior for FFTs of constant functions, unit impulse functions, and basic convolution ops.
🍍 The relative decrease in performance on re-org seems to  be partially due to less optimization in the kernels at compile time (kernels themselves did not change) and given the trend with size, mainly due to overhead in launch? The question is whether the trade off in ease in reading the code is worth it.

:plus: Add in results for using the decomposition method for non-powers of two, more computation but fewer local memory accesses vs Bluesteins.

#### Comparing 2D FFT based convolution 

##### Table 2: zero padded convolution of 4096 pixel sq. image

| 2D square size kernel size | cufft/FastFFT runtime (10k iterations) | after re-org of code  |
| --- | ---- | ---- |
| 64 | 2.79 | 2.6 |
| 128 | 2.81 | 2.6 |
| 256 | 2.71 | 2.5 |
| 512 | 2.66 | 2.4 |
| 1024 | 2.48 | 2.1 |

🍍 None of the kernels are even remotely optimized at this point, they have only been assembled and tested to pass expected behavior for FFTs of constant functions, unit impulse functions, and basic convolution ops.
🍍 See note on previous table. The relative perf hit is not nearly as dramatic as in the previous table; however it is still about 10% which is a tough pill to swallow.

- Movie alignment expense (Pre/post process and alignment percentages.)

  :soon: 
  
#### Input movies are sparse (speed-up)

The largest speed up for a subset of input points is had when those non-zero points are contiguous, however, the reduced memory transfers even for non-contiguous values is still dope. The low-exposure imaging conditions used in HR-cryoEM generally results in ~10% non-zeros. ⚠️ Something about EER format.

#### Movies have a predictable shift range

The shifts between subsequent exposures in cryoEM movie frames is relatively small, on the order of tens of Angstroms (usually 5-20 pixels) in early frames, down to angstroms (0.5-2 pixels) in later frames. Using this prior knowledge, we can use only a limited subset of the output pixels, dramatically decreasing runtime in the translational search.

#### 2D template matching

Typical use case is to search an image of size ~ 4K x 4K over ~ 2 million cross correlations, with a template that is ~ 384-512 pixels square. Each of these searches requires padding the template to the size of the search image, taking the FFT, conj mul and iFFT. We currently optimize this by transforming the search image in single-precision, storing in FP16 to improve reads. Convert from FP16 and conj mul in single precision using cuFFT callbacks on the load, convert back to FP16 on the store with a store call back. Here we improve on this by ....

#### 3D template matching

Pad something like 128 --> 512^3 over X chunks. Far fewer orientations than 2DTM (number). 

#### Subtomogram averaging

While the total zero padding is less (1.5x to 2x) than 3D template matching, there are typically many more cross-correlations in a given project. Like movie alignment, subtomogram averaging also benefits in both directions, as the translational search allows for a strictly limited set of output points to find the shifts. These combined lead to X speedup.


## Discussion


## Conclusion

The FFT has many applications in signal and image processing, such that improving its efficiency for even a subset of problems can still have a broad impact. Here we demonstrate how fairly old algorithmic ideas, transform decomposition, can be mapped to new computational hardware to achieve substantial computational acceleration. The use cases presented in the results have typical run-times on the order of hours to days, such that even fractional speed ups, much less 2-5 fold acceleration are substantial. Gimme some oreos now.


## References

```{bibliography}
:style: unsrt
```

