NVCC=nvcc

NVCC_FLAGS=-ccbin=icpc -t 4
NVCC_FLAGS+=--gpu-architecture=sm_70 -gencode=arch=compute_70,code=compute_70 
#NVCC_FLAGS+=--gpu-architecture=sm_86 -gencode=arch=compute_86,code=compute_86 --generate-code arch=compute_70,code=sm_70 --generate-code arch=compute_75,code=sm_75 --generate-code arch=compute_80,code=sm_80
NVCC_FLAGS+=--default-stream per-thread -m64 -O3 --use_fast_math --extra-device-vectorization -std=c++17  -Xptxas --warn-on-local-memory-usage,--warn-on-spills, --generate-line-info -Xcompiler=-std=c++17

CUFFTDX_INCLUDE_DIR=$(FastFFT_cufftdx_dir)/include

$(info $$CUFFTDX_INCLUDE_DIR is [${CUFFTDX_INCLUDE_DIR}])


CUDA_BIN_DIR=$(shell dirname `which $(NVCC)`)
CUDA_INCLUDE_DIR=$(CUDA_BIN_DIR)/../include
NVRTC_DEFINES=-DCUDA_INCLUDE_DIR="\"$(CUDA_INCLUDE_DIR)\"" -DCUFFTDX_INCLUDE_DIRS="\"$(CUFFTDX_INCLUDE_DIR)\""

#SRCS=$(filter-out nvrtc_*.cu, $(wildcard *.cu))
#TARGETS=$(patsubst %.cu,%,$(SRCS))
SRCS=test.cu
TARGETS=test

$(TARGETS): %: %.cu
	$(NVCC) -o $@.app $< $(NVCC_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) -lfftw3f

$(NVRTC_TARGETS): %: %.cu
	$(NVCC) -o $@ $< $(NVCC_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) $(NVRTC_DEFINES) -lnvrtc -lcuda 

.PHONY: all clean

all: $(TARGETS) $(NVRTC_TARGETS)
	$(echo $(NVRTC_TARGETS))

clean:
	rm -f $(TARGETS) $(NVRTC_TARGETS)

.DEFAULT_GOAL := all
