#!/bin/bash
# Build script for HF Path Simulator GPU module
#
# Compiles the native CUDA _hfpathsim_gpu module using CMake.
# Requires: CUDA Toolkit 12.x, pybind11, CMake >= 3.28
#
# Usage: ./scripts/build_gpu.sh [Release|Debug]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GPU_DIR="$PROJECT_ROOT/src/hfpathsim/gpu"
BUILD_TYPE="${1:-Release}"

echo "=== HF Path Simulator GPU Module Build ==="
echo "Build type: $BUILD_TYPE"
echo "GPU source: $GPU_DIR"

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA toolkit not found. Please install CUDA 12.x"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
echo "CUDA version: $CUDA_VERSION"

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found. Please install CMake >= 3.28"
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -1 | sed 's/cmake version //')
echo "CMake version: $CMAKE_VERSION"

# Get Python executable and pybind11 directory
PYTHON_EXE=$(which python3)
if [ -z "$PYTHON_EXE" ]; then
    echo "Error: python3 not found"
    exit 1
fi

PYBIND11_DIR=$($PYTHON_EXE -m pybind11 --cmakedir 2>/dev/null)
if [ -z "$PYBIND11_DIR" ]; then
    echo "Error: pybind11 not found. Install with: pip install pybind11"
    exit 1
fi

echo "Python: $PYTHON_EXE"
echo "pybind11: $PYBIND11_DIR"

# Create build directory
BUILD_DIR="$GPU_DIR/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo ""
echo "=== Configuring with CMake ==="
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DPython3_EXECUTABLE="$PYTHON_EXE" \
    -Dpybind11_DIR="$PYBIND11_DIR"

# Build
echo ""
echo "=== Building ==="
cmake --build . --parallel $(nproc)

# Copy the built module to the gpu directory
echo ""
echo "=== Installing module ==="
MODULE_NAME=$(find . -name "_hfpathsim_gpu*.so" -type f | head -1)
if [ -n "$MODULE_NAME" ]; then
    cp "$MODULE_NAME" "$GPU_DIR/"
    echo "Installed: $GPU_DIR/$(basename $MODULE_NAME)"

    # Also copy to parent for easier import
    cp "$MODULE_NAME" "$GPU_DIR/../"
    echo "Installed: $GPU_DIR/../$(basename $MODULE_NAME)"
else
    echo "Error: Built module not found!"
    exit 1
fi

echo ""
echo "=== Build Complete ==="
echo ""
echo "Test with:"
echo "  python3 -c \"from hfpathsim.gpu._hfpathsim_gpu import get_device_info; print(get_device_info())\""
