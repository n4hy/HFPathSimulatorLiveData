#!/bin/bash
# HF Path Simulator Docker entrypoint script

set -e

# Print startup info
echo "=========================================="
echo "HF Path Simulator"
echo "=========================================="
echo "Python: $(python --version)"
echo "NumPy: $(python -c 'import numpy; print(numpy.__version__)')"

# Check GPU availability
if [ "${HFPATHSIM_USE_GPU:-1}" = "1" ]; then
    echo "GPU Mode: Enabled"
    if python -c "from hfpathsim.gpu import is_available; print('GPU Available:', is_available())" 2>/dev/null; then
        :
    else
        echo "Warning: GPU module not available, falling back to CPU"
        export HFPATHSIM_USE_GPU=0
    fi
else
    echo "GPU Mode: Disabled (CPU-only)"
fi

# Check for CUDA
if [ "${HFPATHSIM_USE_GPU:-1}" = "1" ]; then
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA Devices:"
        nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null || echo "  (unable to query)"
    fi
fi

echo "=========================================="

# Execute command
exec "$@"
