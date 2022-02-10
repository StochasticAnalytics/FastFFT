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
SMS ?=  80 86

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
CUFFTDX_INCLUDE_DIR := $(FastFFT_cufftdx_dir)/include
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

ifeq (${DEBUG_STAGE},)
debug_stage=8
$(info $$DEBUG_STAGE is not defined, setting to 8)
else
debug_stage=${DEBUG_STAGE}
$(info $$debug_stage is  [${DEBUG_STAGE}])
endif

ifneq (${HEAVYERRORCHECKING_FFT},)
NVCC_FLAGS+=-DHEAVYERRORCHECKING_FFT
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

SRC_DIR  := ../src

TEST_SRCS=$(shell find $(SRC_DIR)/tests -name '*.cu' -or -name '*.cpp')
EXAMPLE_SRCS=$(shell find $(SRC_DIR)/cpp -name '*.cu' -or -name '*.cpp')
BENCHMARK_SRCS=$(shell find $(SRC_DIR)/benchmarks -name '*.cu' -or -name '*.cpp')


TEST_TARGETS=$(patsubst $(SRC_DIR)/tests/%.cu,$(TESTS_BUILD_DIR)/%,$(TEST_SRCS))
EXAMPLE_TARGETS=$(patsubst %.cpp,%,$(EXAMPLE_SRCS))
BENCHMARK_TARGETS=$(patsubst %.cu,%,$(BENCHMARK_SRCS))

TEST_OBJS=$(patsubst %,%.o,$(TEST_TARGETS))
EXAMPLE_OBJS=$(patsubst %,%.o,$(EXAMPLE_TARGETS))
BENCHMARK_OBJS=$(patsubst %,%.o,$(BENCHMARK_TARGETS))

# printit:
# 	echo This is $(TEST_SRCS)
# 	echo $(TEST_TARGETS)
# 	echo $(TEST_OBJS)


all: $(TEST_TARGETS)
ifeq (${remove_debug_file}, 0)
	touch built_without_debug.txt
else
	rm -f built_without_debug.txt
endif

$(TEST_TARGETS): $(TEST_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $< -lfftw3f -lcufft_static -lculibos -lcudart_static -lrt

$(TEST_OBJS): $(TEST_SRCS)
	mkdir -p $(TESTS_BUILD_DIR)
	$(NVCC) -dc $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) -DDEBUG_FFT_STAGE=${debug_stage} -o $@ $<

# .PHONY: all examples tests benchmarks clean

# all: build

# build: simpleCUFFT_callback

# check.deps:
# ifeq ($(SAMPLE_ENABLED),0)
# 	@echo "Sample will be waived due to the above missing dependencies"
# else
# 	@echo "Sample is ready - all dependencies have been met"
# endif

# simpleCUFFT_callback.o:simpleCUFFT_callback.cu
# 	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

# simpleCUFFT_callback: simpleCUFFT_callback.o
# 	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
# 	$(EXEC) mkdir -p ../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)
# 	$(EXEC) cp $@ ../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)

# all:
# 	$(NVCC) -dc $(NVCC_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) -DDEBUG_FFT_STAGE=${debug_stage} -o test.o -c ../cpp/test.cu
# 	$(NVCC) $(NVCC_FLAGS) -o test.app test.o -lfftw3f -lcufft_static -lculibos -lcudart_static -lrt


# SRCS=$(filter-out nvrtc_*.cu, $(wildcard *.cu))
# TARGETS=$(patsubst %.cu,%,$(SRCS))

# NVRTC_SRCS=$(wildcard nvrtc_*.cu)
# NVRTC_TARGETS=$(patsubst %.cu,%,$(NVRTC_SRCS))

# $(TARGETS): %: %.cu
# 	$(NVCC) -o $@ $< $(NVCC_FLAGS) -I$(CUFFTDX_INCLUDE_DIR)

# $(NVRTC_TARGETS): %: %.cu
# 	$(NVCC) -o $@ $< $(NVCC_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) $(NVRTC_DEFINES) -lnvrtc -lcuda


# remove all but the executables.
clean:
	rm -f $(TESTS_BUILD_DIR)/*.*
	rm -f $(TESTS_BUILD_DIR)/$(TEST_TARGETS)


# clean out everything.
clobber:
ifneq ($(TESTS_BUILD_DIR),)
	rm -f $(TESTS_BUILD_DIR)/*
	rmdir $(TESTS_BUILD_DIR)
endif

.DEFAULT_GOAL := all
