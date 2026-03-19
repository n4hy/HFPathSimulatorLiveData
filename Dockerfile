# HF Path Simulator - GPU-enabled Docker image
# Multi-stage build for CUDA support

# ==============================================================================
# Stage 1: CUDA Build Environment
# ==============================================================================
FROM nvidia/cuda:12.4-devel-ubuntu22.04 AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python build dependencies
RUN pip install --no-cache-dir \
    pybind11>=2.12 \
    numpy>=1.26 \
    cmake>=3.28

# Copy source code
WORKDIR /build
COPY . .

# Build native CUDA module
RUN if [ -f "CMakeLists.txt" ]; then \
        mkdir -p build && cd build && \
        cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
            -DPYTHON_EXECUTABLE=/opt/venv/bin/python && \
        make -j$(nproc); \
    fi

# Install Python package
RUN pip install --no-cache-dir ".[api,gpu]"

# ==============================================================================
# Stage 2: Runtime Image
# ==============================================================================
FROM nvidia/cuda:12.4-runtime-ubuntu22.04

# Labels
LABEL maintainer="HFPathSim Contributors"
LABEL org.opencontainers.image.title="HF Path Simulator"
LABEL org.opencontainers.image.description="HF ionospheric channel simulator with GPU acceleration"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/hfpathsim/hfpathsim"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy built CUDA module if exists
COPY --from=builder /build/build/*.so /opt/venv/lib/python3.11/site-packages/hfpathsim/gpu/ 2>/dev/null || true

# Create non-root user
RUN useradd --create-home --shell /bin/bash hfpathsim
USER hfpathsim
WORKDIR /home/hfpathsim

# Copy entrypoint and healthcheck scripts
COPY --chown=hfpathsim:hfpathsim docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY --chown=hfpathsim:hfpathsim docker/healthcheck.sh /usr/local/bin/healthcheck.sh

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HFPATHSIM_USE_GPU=1
ENV CUDA_VISIBLE_DEVICES=0

# Expose ports
# 8000 - REST API
# 5555 - ZMQ input
# 5556 - ZMQ output
EXPOSE 8000 5555 5556

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["/usr/local/bin/healthcheck.sh"]

# Default command - start API server
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["hfpathsim-server", "--host", "0.0.0.0", "--port", "8000"]
