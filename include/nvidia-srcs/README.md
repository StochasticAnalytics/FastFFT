# Customized cuFFTDx (MathDx) sources for FastFFT

FastFFT builds against a locally *customized* copy of NVIDIA's cuFFTDx (shipped in the MathDx
package). The NVIDIA trees are not tracked in git (see `.gitignore`); they are recreated from the
official tarballs with `update_cufftdx.sh`.

## Quick start (new machine or new mathdx version)

```bash
cd include/nvidia-srcs
./update_cufftdx.sh apply 25.06.1 cuda13     # download, tid-swap the PTX, patch headers, emit .mk
# then in build/Makefile select the version:
#   include ../include/nvidia-srcs/nvidia-mathdx-25.06.1.mk
```

`apply` is idempotent and ends with `verify`, which re-checks all invariants. If NVIDIA's download
URL scheme changes, download the tarball manually from https://developer.nvidia.com/cufftdx-downloads
and run `./update_cufftdx.sh apply <version> --tarball <file>`.

## What we change relative to pristine cuFFTDx (and why)

cuFFTDx uses `threadIdx.y` (PTX register `tid.y`) to implicitly batch several 1d FFTs per block.
FastFFT needs `threadIdx.y` for its transform-decomposition kernels and always declares
`ffts_per_block == 1`, so:

1. **PTX database** (`cufftdx/database/*.inc`, ~370 files): every `tid.y` -> `tid.z`
   (implicit batching moves to `tid.z`, which is more constrained but unused by FastFFT).
   Done mechanically by the script, with count verification and a pristine backup at
   `database.bak`.
2. **Header shared-memory offsets** (10 sites in 6 headers): every batch offset computed from
   `threadIdx.y` is wrapped in `if constexpr (FFT::ffts_per_block > 1)`, so a single-FFT block
   never consults `threadIdx.y`. In `fft_execution.hpp` this threads an `FPB` template parameter
   through `shared_to_registers_impl`. In `preprocess_fold.hpp` the unsupported shared-memory API
   is blocked with a deferred `static_assert` (`static_no_shared_api`).
   Known intentional exception: `smem[threadIdx.y]` in `postprocess_r2c_packed`
   (`fft_block_postprocess.hpp`) stays pristine; FastFFT never uses the packed real layout.
3. **`database/detail/block_fft.hpp`**: SM86 records forward to SM80 records (instead of SM70)
   to support larger FFT sizes on 8.6.

The header changes live as a reviewable patch in `patches/nvidia-mathdx-<version>-fastfft-headers.patch`.
For a new mathdx version the script falls back to the newest patch (with fuzz); failed hunks leave
`.rej` files and `verify` prints every remaining unguarded `threadIdx.y` site that needs porting by
hand. After porting, generate a fresh patch for the new version and commit it to `patches/`.

## Architecture support of the embedded versions

| Tree | cuFFTDx | Supported SM | Status |
|---|---|---|---|
| `nvidia-mathdx-25.06.1` | 1.5.1 | 70, 72, 75, 80, 86, 87, 89, 90, 100, 101, 103, 110, 120, 121 | **active** (RTX 5090 = sm_120 supported) |
| `nvidia-cufft-11.5.0-cufftdx-1.4.0` | 1.4.0 | 70-90 | pristine, unused |
| `nvidia-mathdx-24.08.0` | 1.2.x | 70-90 | customized, previous |
| `cufftdx-rv1` | 1.1.0 EA | 70-90 | customized, historical |
| `cufftdx_pre-release-with-8.6-hacks` | 1.0.0 EA | 70-86 | historical |

Note: CUDA 13.x can no longer generate code for sm_70 (Volta); the `SM<700>` dispatch cases still
compile (they only select database records) but sm_70 gencode targets require CUDA 12.x.

`setup.sh` / `setup.md` are the older manual notes that `update_cufftdx.sh` replaces; kept for
reference.
