cmake_minimum_required(VERSION 3.28)
project(hfpathsim_gpu LANGUAGES CXX CUDA)

# Find required packages
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures
# CUDA 12.0 supports up to sm_90, CUDA 12.8+ needed for sm_100/120 (Blackwell)
# Check CUDA version and set appropriate architectures
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
    # Full RTX 5090 (Blackwell) support
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90 100 120)
elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
    # CUDA 12.0-12.7: up to Ada/Hopper
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
else()
    # Older CUDA: Ampere only
    set(CMAKE_CUDA_ARCHITECTURES 80 86)
endif()

# CUDA compile options
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -O3
    --use_fast_math
    -Xcompiler -fPIC
)

# Disable LTO to avoid GCC/nvcc version mismatch issues
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-lto")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fno-lto")

# Source files
set(CUDA_SOURCES
    kernels/vogler_transfer.cu
    kernels/fading.cu
    kernels/signal_proc.cu
    kernels/vh_rf_chain.cu
    kernels/watterson.cu
    kernels/agc.cu
    kernels/noise.cu
    kernels/resampler.cu
    kernels/dispersion.cu
)

# CPU fallback sources (compiled with OpenMP)
set(CPU_SOURCES
    kernels/vh_rf_chain_cpu.cpp
    kernels/watterson_cpu.cpp
    kernels/agc_cpu.cpp
    kernels/noise_cpu.cpp
    kernels/resampler_cpu.cpp
    kernels/dispersion_cpu.cpp
)

# Find OpenMP for CPU fallback
find_package(OpenMP)

# Create CUDA library
add_library(hfpathsim_cuda STATIC ${CUDA_SOURCES})
target_include_directories(hfpathsim_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(hfpathsim_cuda
    CUDA::cudart
    CUDA::cufft
    CUDA::curand
)
set_target_properties(hfpathsim_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Create CPU fallback library (with OpenMP and SIMD optimizations)
add_library(hfpathsim_cpu STATIC ${CPU_SOURCES})
target_include_directories(hfpathsim_cpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_options(hfpathsim_cpu PRIVATE
    -O3
    -march=native
    -ffast-math
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(hfpathsim_cpu PUBLIC OpenMP::OpenMP_CXX)
endif()
set_target_properties(hfpathsim_cpu PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Create pybind11 module
pybind11_add_module(_hfpathsim_gpu bindings.cpp)
target_link_libraries(_hfpathsim_gpu PRIVATE
    hfpathsim_cuda
    hfpathsim_cpu
    CUDA::cudart
    CUDA::cufft
    CUDA::curand
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_hfpathsim_gpu PRIVATE OpenMP::OpenMP_CXX)
endif()

# Install the module
install(TARGETS _hfpathsim_gpu
    LIBRARY DESTINATION hfpathsim/gpu
)
