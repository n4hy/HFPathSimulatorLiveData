cmake_minimum_required(VERSION 3.28)
project(hfpathsim_gpu LANGUAGES CXX CUDA)

# Find required packages
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures (RTX 5090 is Blackwell / compute 12.0)
# Also include common architectures for compatibility
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90 100 120)

# CUDA compile options
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -O3
    --use_fast_math
    -Xcompiler -fPIC
)

# Source files
set(CUDA_SOURCES
    kernels/vogler_transfer.cu
    kernels/fading.cu
    kernels/signal_proc.cu
)

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

# Create pybind11 module
pybind11_add_module(_hfpathsim_gpu bindings.cpp)
target_link_libraries(_hfpathsim_gpu PRIVATE
    hfpathsim_cuda
    CUDA::cudart
    CUDA::cufft
)

# Install the module
install(TARGETS _hfpathsim_gpu
    LIBRARY DESTINATION hfpathsim/gpu
)
