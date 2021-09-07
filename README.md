# FastFFT

<img src="https://render.githubusercontent.com/render/math?math=%2Bfft">

## Project Summary

#### Goals:
This project aims to accelerate a specific subset of Fast Fourier Transforms (FFTs) especially important image processing, my particular flavor of which is  high-resolution phase-contrast cryo-Electron Microscopy. The gist is to take advantage of the cufftdx library from Nvidia, currently in early access, as well as algorithmic ideas from *Sorensen et. al* [[1]](#1). I became aware of [VkFFT](https://github.com/DTolm/VkFFT) after I had finished my initial [proof of principle](https://github.com/bHimes/cisTEM_downstream_bah/blob/DFT/src/gpu/DFTbyDecomposition.cu) experiments with cufftdx in Oct 2020. It may be entirely possible to do the same things using that library, though I haven't had a chance to look through the source.

An additional experiment tied to this project, is the development of the scientific manuscript "live" alongside the development of the code. While most of my projects are pretty safe from being scooped, this one is decidedly "scoopable" as it should have broad impact, and also shouldn't be to hard to implement once the info is presented.

#### Design:

The FourierTransformer class, in the FastFFT namespace may be used in your cpp/cuda project by including the *header.* The basic usage is as follows:

- The input/output data pointers, size and type are set. (analagous to the cufftXtMakePlanMany)
- Class methods are provided to handle memory allocation on the gpu as well as transfer to/from host/device.
- Simple methods for FwdFFT, InvFFT, CrossCorrelation etc. are public and call the correct combination of substranforms based on the data and padding/trimming wanted.


#### Manuscript:

Link to book via github pages
