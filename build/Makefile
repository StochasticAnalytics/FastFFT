# TODO: nvc++ ?
NVCC=nvcc
# TODO: test with gcc and clang
NVCC_FLAGS=-ccbin=g++ -t 8 

# TODO: static may need to be added later or independently just for benchmarks, while libs to link through python are probably going to need to be dynamic.
NVCC_FLAGS+=--cudart=static
NVCC_FLAGS+=--default-stream per-thread -m64 -O3 --use_fast_math --extra-device-vectorization --extended-lambda --Wext-lambda-captures-this -std=c++17 -Xptxas --warn-on-local-memory-usage,--warn-on-spills, --generate-line-info -Xcompiler=-std=c++17
# Experimental flag to allow relaxed constexpr host call in device code
NVCC_FLAGS+=--expt-relaxed-constexpr
# For testing, particularly until the cufftdx::BlockDim() operator is enabled
NVCC_FLAGS+=-DCUFFTDX_DISABLE_RUNTIME_ASSERTS

# Gencode arguments, only supporting Volta or newer
# SMS ?= 70 75 80 86
# In initial dev, only compile for 70 or 86 depending on which workstation I'm on, b/c it is faster.
SMS ?=  70 86

# Need to 
# GENCODE_FLAGS := -gencode arch=compute_86,code=lto_86
# GENCODE_FLAGS := -gencode arch=compute_86,code=lto_86

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))


# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=sm_$(HIGHEST_SM)
endif
endif # GENCODE_FLAGS

# FIXME: this is set in my own bashrc. Until cufftdx comes out of early release, this is fine.
CUFFTDX_INCLUDE_DIR := ../include
$(info $$CUFFTDX_INCLUDE_DIR is [${CUFFTDX_INCLUDE_DIR}])

# TODO: check on cuda version (building with 11.3.1 r/n)
CUDA_BIN_DIR=$(shell dirname `which $(NVCC)`)
CUDA_INCLUDE_DIR=$(CUDA_BIN_DIR)/../include



# Debug level determines various asserts and print macros defined in FastFFT.cuh These should only be set when building tests and developing.
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

ifeq (${FFT_DEBUG_STAGE},)
debug_stage=8
$(info $$FFT_DEBUG_STAGE is not defined, setting to 8)
else
debug_stage=${FFT_DEBUG_STAGE}
$(info $$debug_stage is  [${FFT_DEBUG_STAGE}])
endif

# Track what the last build did so that the pre-commit hook can skip re-building if there is no debug included.
remove_debug_file=0
ifeq (${debug_stage}, 8) 
	ifneq (${debug_level},0)
		remove_debug_file=1
	endif
else
	remove_debug_file=1
ifeq (${HEAVYERRORCHECKING_FFT},)
# If HEAVYERRORCHECKING_FFT is not already asked for, then add it anytime debug_stage < 8 (partial FFTs)
	NVCC_FLAGS+=-DHEAVYERRORCHECKING_FFT
endif
endif

LIB_DIR=../src/fastfft
TEST_DIR=../src/tests
# Assuming make is being run from build
TESTS_BUILD_DIR := ./tests
LIB_BUILD_DIR := ./lib
BENCHMARK_BUILD_DIR := ./benchmarks

LIBRARY_BASE_NAME := libFastFFT


LIBRARY_OBJS=$(patsubst %.cu,$(TESTS_BUILD_DIR)/%.o,$(notdir $(LIBRARY_SRCS)))
TARGETS=tests/test1
all: tests/test1
ifeq (${remove_debug_file}, 0)
	touch built_without_debug.txt
else
	rm -f built_without_debug.txt
endif

# External libraries used for testing
EXTERNAL_LIBS= -lfftw3f -lcufft_static -lculibos -lcudart_static -lrt

# tests/test1: tests/test.o lib/libFastFFT.a
# 	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -L/scratch/salina/git/FastFFT/build -o $@ $^ -lFastFFT $(EXTERNAL_LIBS)

tests/test1: tests/linkedTest.o tests/test.o lib/FastFFT.o lib/Image.o 
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) -DFFT_DEBUG_STAGE=${debug_stage} -o $@ $^ $(EXTERNAL_LIBS)

tests/linkedTest.o: tests/test.o lib/FastFFT.o lib/Image.o 
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) -DFFT_DEBUG_STAGE=${debug_stage} --device-link -o $@ $^ 

# TODO: examples?
tests/test.o: ../src/tests/test.cu
	@mkdir -p $(TESTS_BUILD_DIR)
	$(NVCC) -dc $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) -DFFT_DEBUG_STAGE=${debug_stage} -o $@ $<


# # Make a static archive
# lib/libFastFFT.a: lib/FastFFT.o lib/Image.o lib/libFastFFT.o
# 	$(NVCC) $(NVCC_FLAGS) --lib --output-file $@ $^ $(EXTERNAL_LIBS)

# # Link the core lib together.
# # It is debatable whether the helper image class should really be in here as it is really just needed for testing and examples. I can imagine if 
# # someone had the static lib distribution and wanted to test a simple app without compiling everything, they might want it in there?
# lib/libFastFFT.o: ../src/fastfft/FastFFT.o ../src/fastfft/Image.o
# 	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) -DFFT_DEBUG_STAGE=${debug_stage} --device-link -o $@ $^ $(EXTERNAL_LIBS)
	

# # For some reason, I was getting an expansion that resulted in Image.o being built twice, once from Image.cu and once form FastFFT.cu, which obviously failed.
# # This is a hack to get around that. I don't know why it was happening, but it was only happening when I was building the library.
lib/FastFFT.o: ../src/fastfft/FastFFT.cu
	@mkdir -p $(LIB_BUILD_DIR)
	$(NVCC) -dc $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) -DFFT_DEBUG_STAGE=${debug_stage} -o $@ $<  

lib/Image.o: ../src/fastfft/Image.cu
	@mkdir -p $(LIB_BUILD_DIR)
	$(NVCC) -dc $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) -DFFT_DEBUG_STAGE=${debug_stage} -o $@ $<  


# remove all but the executables.
clean:
	rm -f tests/*.o lib/*.o *.a *.o
	rm -f $(TEST_TARGETS)


# clean out everything.
clobber:
ifneq ($(TESTS_BUILD_DIR),)
	rm -f $(TESTS_BUILD_DIR)/*
	rmdir $(TESTS_BUILD_DIR)
endif

.DEFAULT_GOAL := all
