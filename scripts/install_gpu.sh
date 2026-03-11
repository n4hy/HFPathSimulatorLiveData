#!/bin/bash
# Install the GPU module to site-packages
#
# This script copies the built _hfpathsim_gpu module to the Python
# site-packages directory for system-wide installation.
#
# Usage: ./scripts/install_gpu.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GPU_DIR="$PROJECT_ROOT/src/hfpathsim/gpu"

echo "=== Installing HF Path Simulator GPU Module ==="

# Find the built module
MODULE_PATH=$(find "$GPU_DIR" -maxdepth 1 -name "_hfpathsim_gpu*.so" -type f | head -1)

if [ -z "$MODULE_PATH" ]; then
    echo "Error: GPU module not found. Run build_gpu.sh first."
    exit 1
fi

MODULE_NAME=$(basename "$MODULE_PATH")
echo "Found module: $MODULE_NAME"

# Get Python site-packages directory
PYTHON_EXE=$(which python3)
SITE_PACKAGES=$($PYTHON_EXE -c "import site; print(site.getsitepackages()[0])")

if [ -z "$SITE_PACKAGES" ]; then
    echo "Error: Could not determine site-packages directory"
    exit 1
fi

INSTALL_DIR="$SITE_PACKAGES/hfpathsim/gpu"
echo "Install directory: $INSTALL_DIR"

# Check if hfpathsim is installed
if [ ! -d "$SITE_PACKAGES/hfpathsim" ]; then
    echo "Warning: hfpathsim not installed in site-packages."
    echo "Install with: pip install -e ."
    echo ""
    echo "For development, the module is already available in:"
    echo "  $GPU_DIR/$MODULE_NAME"
    exit 0
fi

# Create gpu directory if needed
mkdir -p "$INSTALL_DIR"

# Copy module
cp "$MODULE_PATH" "$INSTALL_DIR/"
echo "Installed: $INSTALL_DIR/$MODULE_NAME"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Verify with:"
echo "  python3 -c \"from hfpathsim.gpu import get_device_info; print(get_device_info())\""
