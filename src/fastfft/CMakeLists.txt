# ---------------------------------------------------------------------------------------
# The library source and header files
# ---------------------------------------------------------------------------------------
set(FASTFFT_HEADERS
        # ${PROJECT_SOURCE_DIR}Session.h

        ${PROJECT_SOURCE_DIR}/src/cpp/Image.cuh

        )

set(FASTFFT_SOURCES
        ${PROJECT_SOURCE_DIR}/tests/test.cu

        ${PROJECT_SOURCE_DIR}/src/cpp/Image.cu
      
        )

set(FASTFFT_CPU_HEADERS
        # fastfft::cpu::fft
        # cpu/fft/Transforms.h
        # cpu/fft/Plan.h
        # cpu/fft/Resize.h
        # cpu/fft/Remap.h
        # cpu/fft/Filters.h

        )

set(FASTFFT_CPU_SOURCES
        # fastfft::cpu::fft
        # cpu/fft/Plan.cpp
        # cpu/fft/Filters.cpp
        # cpu/fft/Remap.cpp
        # cpu/fft/Resize.cpp

        )

set(FASTFFT_CUDA_HEADERS
        # fastfft::cuda
        
        ${PROJECT_SOURCE_DIR}/include/FastFFT.h
        ${PROJECT_SOURCE_DIR}/include/FastFFT.cuh

        )

set(FASTFFT_CUDA_SOURCES
        # fastfft::cuda::fft

        ${PROJECT_SOURCE_DIR}/src/fastfft/FastFFT.cu
        ${PROJECT_SOURCE_DIR}/tests/test.cu

        )

include(${PROJECT_SOURCE_DIR}/cmake/targets/fastfft.cmake)