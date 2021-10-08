cmake_minimum_required(VERSION 3.14)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cpp/)

message(STATUS "Defined cxx compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CMake source dir: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMake binary dir: ${CMAKE_BINARY_DIR}")


include(ProjectSetup)


project(  FastFFT 
          VERSION 0.1 
          LANGUAGES CXX CUDA
)

include(InstallDeps)
include(AddFlags)
include(CUDASetup)
include(CheckLanguage)

message(STATUS "Project source dir: ${PROJECT_SOURCE_DIR}")
message(STATUS "CUDA_MIN_ARCH: ${CUDA_MIN_ARCH}")


check_language(CUDA)

foreach(lang CXX CUDA)
  set(CMAKE_${lang}_STANDARD 17)
  set(CMAKE_${lang}_STANDARD_REQUIRED ON)
  set(CMAKE_${lang}_STANDARD_EXTENSIONS OFF)
endforeach()

find_package(OpenMP QUIET COMPONENTS CXX)

add_executable(test
               ${PROJECT_SOURCE_DIR}/cpp/test.cu
)

target_link_libraries(test PRIVATE
  fftw3f
  CUDA::cufft_static 
  CUDA::culibos 
  CUDA::cudart_static 
  rt
)

install(TARGETS
  test
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin 
)




set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


