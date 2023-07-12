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


SRCS=test.cu
TARGETS=test

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


# Assuming make is being run from build
TESTS_BUILD_DIR := ./tests
EXAMPLE_BUILD_DIR := ./examples
BENCHMARK_BUILD_DIR := ./benchmarks

LIBRARY_NAME := libFastFFT.a
# Because I am linking test.cu.cpp -> test.cu as a cheap trick to get auto formatting in vs code, we can't include cpp in the src list
LIBRARY_SRCS:= ../src/fastfft/Image.cu ../src/fastfft/FastFFT.cu
TEST_SRCS := ../src/tests/test.cu 
# LIBRARY_OBJS := ../src/fastfft/Image.o ../src/fastfft/FastFFT.o
# LIBRARY_OBJS=$(patsubst %.cu,$(TESTS_BUILD_DIR)/%.o,$(notdir $(LIBRARY_SRCS)))
LIBRARY_OBJS=$(patsubst %.cu,$(TESTS_BUILD_DIR)/%.o,$(notdir $(LIBRARY_SRCS)))
TEST_OBJS=$(patsubst %.cu,$(TESTS_BUILD_DIR)/%.o,$(notdir $(TEST_SRCS)))

TEST_TARGETS=$(patsubst %.cu,$(TESTS_BUILD_DIR)/%, $(notdir $(TEST_SRCS)))




printit:
	@echo test SRCS $(TEST_SRCS)
	@echo test TARGETS $(TEST_TARGETS)
	@echo test OBJS $(TEST_OBJS)
	@echo lib SRCS $(LIBRARY_SRCS)
	@echo lib TARGETS $(LIBRARY_TARGET)
	@echo lib OBJECTS $(LIBRARY_OBJS)


all: $(TEST_TARGETS)
ifeq (${remove_debug_file}, 0)
	touch built_without_debug.txt
else
	rm -f built_without_debug.txt
endif

# External libraries used for testing
EXTERNAL_LIBS= -lfftw3f -lcufft_static -lculibos -lcudart_static -lrt

$(TEST_TARGETS): $(LIBRARY_NAME) $(TEST_OBJS) 
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -L/scratch/salina/git/FastFFT/build -o $@ $(TEST_OBJS) -lFastFFT $(EXTERNAL_LIBS)

# TODO: examples?
$(TEST_OBJS): $(TEST_SRCS)
	@mkdir -p $(TESTS_BUILD_DIR)
	$(NVCC) -dc $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) -DFFT_DEBUG_STAGE=${debug_stage} -o $@ $<


# Make a static archive
$(LIBRARY_NAME): $(LIBRARY_NAME:%.a=%.o)
	$(NVCC) $(NVCC_FLAGS) --lib --output-file libFastFFT.a $(LIBRARY_OBJS) libFastFFT.o

# Link the core lib together.
# It is debatable whether the helper image class should really be in here as it is really just needed for testing and examples. I can imagine if 
# someone had the static lib distribution and wanted to test a simple app without compiling everything, they might want it in there?
$(LIBRARY_NAME:%.a=%.o): $(LIBRARY_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) -DFFT_DEBUG_STAGE=${debug_stage} --device-link -o $@ $(LIBRARY_OBJS) $(EXTERNAL_LIBS)
	

# # For some reason, I was getting an expansion that resulted in Image.o being built twice, once from Image.cu and once form FastFFT.cu, which obviously failed.
# # This is a hack to get around that. I don't know why it was happening, but it was only happening when I was building the library.
$(TESTS_BUILD_DIR)/$(notdir %.o): ../src/fastfft/%.cu
	@mkdir -p $(TESTS_BUILD_DIR)
	$(NVCC) -dc $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) -DFFT_DEBUG_STAGE=${debug_stage} $< -o $@ 




# remove all but the executables.
clean:
	rm -f $(TESTS_BUILD_DIR)/*.* 
	rm -f $(TEST_TARGETS)


# clean out everything.
clobber:
ifneq ($(TESTS_BUILD_DIR),)
	rm -f $(TESTS_BUILD_DIR)/*
	rmdir $(TESTS_BUILD_DIR)
endif

.DEFAULT_GOAL := all
