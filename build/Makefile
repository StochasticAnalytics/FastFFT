NVCC=nvcc

NVCC_FLAGS=-ccbin=icpc -t 4 
NVCC_FLAGS+=--gpu-architecture=sm_70 -gencode=arch=compute_70,code=compute_70 
# NVCC_FLAGS+= -gencode arch=compute_70, code=lto_70
#-ptxas-options=-v
#NVCC_FLAGS+=--gpu-architecture=sm_80 -gencode=arch=compute_86,code=compute_86 --generate-code arch=compute_70,code=sm_70 --generate-code arch=compute_75,code=sm_75 --generate-code arch=compute_80,code=sm_80
NVCC_FLAGS+=--default-stream per-thread -m64 -O3 --use_fast_math --extra-device-vectorization -std=c++17  --cudart=static -Xptxas --warn-on-local-memory-usage,--warn-on-spills, --generate-line-info -Xcompiler=-std=c++17
# Experimental flag to allow relaxed constexpr host call in device code
NVCC_FLAGS+=--expt-relaxed-constexpr
# For testing
NVCC_FLAGS+=-DCUFFTDX_DISABLE_RUNTIME_ASSERTS

CUFFTDX_INCLUDE_DIR=$(FastFFT_cufftdx_dir)/include

$(info $$CUFFTDX_INCLUDE_DIR is [${CUFFTDX_INCLUDE_DIR}])



CUDA_BIN_DIR=$(shell dirname `which $(NVCC)`)
CUDA_INCLUDE_DIR=$(CUDA_BIN_DIR)/../include
# NVRTC_DEFINES=-DCUDA_INCLUDE_DIR="\"$(CUDA_INCLUDE_DIR)\"" -DCUFFTDX_INCLUDE_DIRS="\"$(CUFFTDX_INCLUDE_DIR)\""

#SRCS=$(filter-out nvrtc_*.cu, $(wildcard *.cu))
#TARGETS=$(patsubst %.cu,%,$(SRCS))
SRCS=test.cu
TARGETS=test

# Debug level determines various asserts and print macros defined in FastFFT.cuh

ifeq (${FFT_DEBUG_LEVEL},)
debug_level=0
$(info $$FFT_DEBUG_LEVEL is not defined, setting to 0)
else
debug_level=${FFT_DEBUG_LEVEL}
$(info $$debug_level is  [${FFT_DEBUG_LEVEL}])
endif
NVCC_FLAGS+=-DFFT_DEBUG_LEVEL=${debug_level}

# For testing/debugging it is convenient to execute and have print functions for partial transforms.
# These will go directly in the kernels and also in the helper Image.cuh definitions for PrintArray.
# The number refers to the number of 1d FFTs performed, 
# Fwd 0, 1, 2, 3( none, x, z, original y)
# 4 intermediate ops, like conj multiplication
# Inv 5, 6, 7 ( original y, z, x)

ifeq (${DEBUG_STAGE},)
debug_stage=8
$(info $$DEBUG_STAGE is not defined, setting to 8)
else
debug_stage=${DEBUG_STAGE}
$(info $$debug_stage is  [${DEBUG_STAGE}])
endif

ifneq (${debug_stage}, 8) 
		NVCC_FLAGS+=-DHEAVYERRORCHECKING_FFT
endif

all:
	$(NVCC) -dc $(NVCC_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) -DDEBUG_FFT_STAGE=${debug_stage} -o test.o -c ../cpp/test.cu
	$(NVCC) $(NVCC_FLAGS) -o test.app test.o -lfftw3f -lcufft_static -lculibos -lcudart_static -lrt



