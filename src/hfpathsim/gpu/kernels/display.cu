/**
 * CUDA kernels for GUI display computations.
 *
 * Optimized operations for real-time spectrum/waterfall displays:
 * - dB conversion (10*log10, 20*log10)
 * - FFT shift
 * - Moving average smoothing
 * - Peak hold with decay
 * - 2D normalization for scattering display
 */

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <vector>

// ============================================================================
// dB Conversion Kernels
// ============================================================================

/**
 * Convert magnitude to dB: 20*log10(|x| + eps)
 * For voltage/amplitude quantities.
 */
__global__ void magnitude_to_db_kernel(
    const float* __restrict__ mag,
    float* __restrict__ db,
    int N,
    float eps,
    float min_db
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float val = mag[idx] + eps;
    float db_val = 20.0f * log10f(val);
    db[idx] = fmaxf(db_val, min_db);
}

/**
 * Convert power to dB: 10*log10(x + eps)
 * For power quantities.
 */
__global__ void power_to_db_kernel(
    const float* __restrict__ power,
    float* __restrict__ db,
    int N,
    float eps,
    float min_db
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float val = power[idx] + eps;
    float db_val = 10.0f * log10f(val);
    db[idx] = fmaxf(db_val, min_db);
}

/**
 * Convert complex magnitude to dB: 20*log10(|re + j*im| + eps)
 */
__global__ void complex_to_db_kernel(
    const float* __restrict__ real,
    const float* __restrict__ imag,
    float* __restrict__ db,
    int N,
    float eps,
    float min_db
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float re = real[idx];
    float im = imag[idx];
    float mag = sqrtf(re * re + im * im) + eps;
    float db_val = 20.0f * log10f(mag);
    db[idx] = fmaxf(db_val, min_db);
}

// ============================================================================
// FFT Shift Kernel
// ============================================================================

/**
 * In-place FFT shift: swap left and right halves.
 * For odd N, the center element stays in place.
 */
__global__ void fftshift_kernel(
    float* __restrict__ data,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = N / 2;

    if (idx >= half) return;

    // Swap data[idx] with data[idx + half]
    int other = idx + half + (N & 1);  // Account for odd N
    if (other < N) {
        float tmp = data[idx];
        data[idx] = data[other];
        data[other] = tmp;
    }
}

/**
 * Out-of-place FFT shift (safer, no race conditions).
 */
__global__ void fftshift_outofplace_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int half = N / 2;
    int shifted_idx;

    if (idx < half) {
        shifted_idx = idx + half + (N & 1);
    } else {
        shifted_idx = idx - half;
    }

    output[shifted_idx] = input[idx];
}

// ============================================================================
// Moving Average / Smoothing Kernels
// ============================================================================

/**
 * Simple moving average with configurable window size.
 * Uses symmetric window centered on each sample.
 */
__global__ void moving_average_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int window_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int half_win = window_size / 2;
    float sum = 0.0f;
    int count = 0;

    for (int k = -half_win; k <= half_win; k++) {
        int i = idx + k;
        if (i >= 0 && i < N) {
            sum += input[i];
            count++;
        }
    }

    output[idx] = sum / (float)count;
}

/**
 * Exponential moving average (IIR smoothing).
 * alpha = smoothing factor (0-1), higher = less smoothing
 */
__global__ void exponential_smooth_kernel(
    const float* __restrict__ current,
    float* __restrict__ smoothed,
    int N,
    float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // EMA: smoothed[n] = alpha * current[n] + (1-alpha) * smoothed[n-1]
    // But this is element-wise, not time-series
    // This kernel updates smoothed with weighted average of current
    smoothed[idx] = alpha * current[idx] + (1.0f - alpha) * smoothed[idx];
}

// ============================================================================
// Peak Hold Kernels
// ============================================================================

/**
 * Peak hold with decay: tracks maximum values with slow decay.
 * peak[i] = max(peak[i] - decay_rate, current[i])
 */
__global__ void peak_hold_kernel(
    const float* __restrict__ current,
    float* __restrict__ peak,
    int N,
    float decay_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float decayed = peak[idx] - decay_rate;
    peak[idx] = fmaxf(decayed, current[idx]);
}

/**
 * Combined: update peak hold and compute average in one pass.
 * Useful when both are needed for display.
 */
__global__ void peak_and_average_kernel(
    const float* __restrict__ current,
    float* __restrict__ peak,
    float* __restrict__ avg_accum,
    int N,
    float decay_rate,
    float avg_weight
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Peak hold
    float decayed = peak[idx] - decay_rate;
    peak[idx] = fmaxf(decayed, current[idx]);

    // Accumulate for average (weighted)
    avg_accum[idx] += current[idx] * avg_weight;
}

// ============================================================================
// 2D Operations for Scattering Display
// ============================================================================

/**
 * 2D dB conversion with peak normalization and clipping.
 * Combined operation for scattering function display.
 *
 * Output: (10*log10(S + eps) - max_db) clipped to [min_clip, 0], then normalized to [0, 1]
 */
__global__ void scattering_normalize_kernel(
    const float* __restrict__ S,
    float* __restrict__ S_norm,
    int rows,
    int cols,
    float eps,
    float max_db,      // Pre-computed max in dB
    float min_clip_db  // e.g., -40 dB
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;

    // Convert to dB
    float db_val = 10.0f * log10f(S[idx] + eps);

    // Normalize to peak
    db_val = db_val - max_db;

    // Clip to range [min_clip_db, 0]
    db_val = fmaxf(db_val, min_clip_db);
    db_val = fminf(db_val, 0.0f);

    // Normalize to [0, 1]
    S_norm[idx] = (db_val - min_clip_db) / (-min_clip_db);
}

/**
 * Find maximum value in array (parallel reduction).
 */
__global__ void find_max_kernel(
    const float* __restrict__ data,
    float* __restrict__ max_out,
    int N
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load to shared memory
    sdata[tid] = (idx < N) ? data[idx] : -FLT_MAX;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        atomicMax((int*)max_out, __float_as_int(sdata[0]));
    }
}

/**
 * 2D transpose for display (row-major to column-major or vice versa).
 */
__global__ void transpose_2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

// ============================================================================
// State Structure
// ============================================================================

struct DisplayGPUState {
    // Buffers for spectrum display
    float* d_spectrum;
    float* d_spectrum_db;
    float* d_peak_hold;
    float* d_avg_accum;
    int spectrum_size;

    // Buffers for scattering display
    float* d_scattering;
    float* d_scattering_norm;
    float* d_max_val;
    int scatter_rows;
    int scatter_cols;

    // Smoothing buffer
    float* d_smooth_buf;
    int smooth_size;

    int max_spectrum_size;
    int max_scatter_size;
    bool initialized;
};

// ============================================================================
// C Interface
// ============================================================================

extern "C" {

/**
 * Initialize display GPU state.
 */
void* init_display_gpu(int max_spectrum_size, int max_scatter_rows, int max_scatter_cols) {
    DisplayGPUState* state = new DisplayGPUState;
    memset(state, 0, sizeof(DisplayGPUState));

    state->max_spectrum_size = max_spectrum_size;
    state->max_scatter_size = max_scatter_rows * max_scatter_cols;

    // Allocate spectrum buffers
    cudaMalloc(&state->d_spectrum, max_spectrum_size * sizeof(float));
    cudaMalloc(&state->d_spectrum_db, max_spectrum_size * sizeof(float));
    cudaMalloc(&state->d_peak_hold, max_spectrum_size * sizeof(float));
    cudaMalloc(&state->d_avg_accum, max_spectrum_size * sizeof(float));
    cudaMalloc(&state->d_smooth_buf, max_spectrum_size * sizeof(float));

    // Initialize peak hold to very negative values
    cudaMemset(state->d_peak_hold, 0, max_spectrum_size * sizeof(float));
    cudaMemset(state->d_avg_accum, 0, max_spectrum_size * sizeof(float));

    // Allocate scattering buffers
    cudaMalloc(&state->d_scattering, state->max_scatter_size * sizeof(float));
    cudaMalloc(&state->d_scattering_norm, state->max_scatter_size * sizeof(float));
    cudaMalloc(&state->d_max_val, sizeof(float));

    state->spectrum_size = 0;
    state->scatter_rows = 0;
    state->scatter_cols = 0;
    state->smooth_size = 0;
    state->initialized = true;

    return state;
}

/**
 * Convert magnitude array to dB.
 */
int magnitude_to_db_gpu(
    void* state_ptr,
    const float* mag,
    float* db,
    int N,
    float eps,
    float min_db
) {
    DisplayGPUState* state = (DisplayGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (N > state->max_spectrum_size) return -2;

    // Copy input to device
    cudaMemcpy(state->d_spectrum, mag, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    magnitude_to_db_kernel<<<blocks, threads>>>(
        state->d_spectrum, state->d_spectrum_db, N, eps, min_db
    );
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(db, state->d_spectrum_db, N * sizeof(float), cudaMemcpyDeviceToHost);

    return 0;
}

/**
 * Convert power array to dB.
 */
int power_to_db_gpu(
    void* state_ptr,
    const float* power,
    float* db,
    int N,
    float eps,
    float min_db
) {
    DisplayGPUState* state = (DisplayGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (N > state->max_spectrum_size) return -2;

    cudaMemcpy(state->d_spectrum, power, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    power_to_db_kernel<<<blocks, threads>>>(
        state->d_spectrum, state->d_spectrum_db, N, eps, min_db
    );
    cudaDeviceSynchronize();

    cudaMemcpy(db, state->d_spectrum_db, N * sizeof(float), cudaMemcpyDeviceToHost);

    return 0;
}

/**
 * FFT shift (out-of-place).
 */
int fftshift_gpu(
    void* state_ptr,
    const float* input,
    float* output,
    int N
) {
    DisplayGPUState* state = (DisplayGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (N > state->max_spectrum_size) return -2;

    cudaMemcpy(state->d_spectrum, input, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    fftshift_outofplace_kernel<<<blocks, threads>>>(
        state->d_spectrum, state->d_spectrum_db, N
    );
    cudaDeviceSynchronize();

    cudaMemcpy(output, state->d_spectrum_db, N * sizeof(float), cudaMemcpyDeviceToHost);

    return 0;
}

/**
 * Moving average smoothing.
 */
int moving_average_gpu(
    void* state_ptr,
    const float* input,
    float* output,
    int N,
    int window_size
) {
    DisplayGPUState* state = (DisplayGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (N > state->max_spectrum_size) return -2;

    cudaMemcpy(state->d_spectrum, input, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    moving_average_kernel<<<blocks, threads>>>(
        state->d_spectrum, state->d_spectrum_db, N, window_size
    );
    cudaDeviceSynchronize();

    cudaMemcpy(output, state->d_spectrum_db, N * sizeof(float), cudaMemcpyDeviceToHost);

    return 0;
}

/**
 * Update peak hold with decay.
 */
int peak_hold_gpu(
    void* state_ptr,
    const float* current,
    float* peak,
    int N,
    float decay_rate
) {
    DisplayGPUState* state = (DisplayGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (N > state->max_spectrum_size) return -2;

    // Copy current values to device
    cudaMemcpy(state->d_spectrum, current, N * sizeof(float), cudaMemcpyHostToDevice);

    // Copy existing peak values to device (or initialize)
    cudaMemcpy(state->d_peak_hold, peak, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    peak_hold_kernel<<<blocks, threads>>>(
        state->d_spectrum, state->d_peak_hold, N, decay_rate
    );
    cudaDeviceSynchronize();

    // Copy updated peak back
    cudaMemcpy(peak, state->d_peak_hold, N * sizeof(float), cudaMemcpyDeviceToHost);

    return 0;
}

/**
 * Exponential smoothing (in-place update of smoothed buffer).
 */
int exponential_smooth_gpu(
    void* state_ptr,
    const float* current,
    float* smoothed,
    int N,
    float alpha
) {
    DisplayGPUState* state = (DisplayGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (N > state->max_spectrum_size) return -2;

    cudaMemcpy(state->d_spectrum, current, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(state->d_smooth_buf, smoothed, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    exponential_smooth_kernel<<<blocks, threads>>>(
        state->d_spectrum, state->d_smooth_buf, N, alpha
    );
    cudaDeviceSynchronize();

    cudaMemcpy(smoothed, state->d_smooth_buf, N * sizeof(float), cudaMemcpyDeviceToHost);

    return 0;
}

/**
 * Normalize scattering function for display.
 * Converts to dB, normalizes to peak, clips, and scales to [0,1].
 */
int normalize_scattering_gpu(
    void* state_ptr,
    const float* S,
    float* S_norm,
    int rows,
    int cols,
    float min_clip_db
) {
    DisplayGPUState* state = (DisplayGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (rows * cols > state->max_scatter_size) return -2;

    int total = rows * cols;

    // Copy input to device
    cudaMemcpy(state->d_scattering, S, total * sizeof(float), cudaMemcpyHostToDevice);

    // Find max value in dB
    // First convert to dB on CPU to find max (simpler than parallel reduction)
    float max_db = -1000.0f;
    for (int i = 0; i < total; i++) {
        float db = 10.0f * log10f(S[i] + 1e-10f);
        if (db > max_db) max_db = db;
    }

    // Normalize on GPU
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    scattering_normalize_kernel<<<blocks, threads>>>(
        state->d_scattering, state->d_scattering_norm,
        rows, cols, 1e-10f, max_db, min_clip_db
    );
    cudaDeviceSynchronize();

    cudaMemcpy(S_norm, state->d_scattering_norm, total * sizeof(float), cudaMemcpyDeviceToHost);

    return 0;
}

/**
 * 2D transpose.
 */
int transpose_2d_gpu(
    void* state_ptr,
    const float* input,
    float* output,
    int rows,
    int cols
) {
    DisplayGPUState* state = (DisplayGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (rows * cols > state->max_scatter_size) return -2;

    int total = rows * cols;

    cudaMemcpy(state->d_scattering, input, total * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((cols + 15) / 16, (rows + 15) / 16);
    transpose_2d_kernel<<<blocks, threads>>>(
        state->d_scattering, state->d_scattering_norm, rows, cols
    );
    cudaDeviceSynchronize();

    cudaMemcpy(output, state->d_scattering_norm, total * sizeof(float), cudaMemcpyDeviceToHost);

    return 0;
}

/**
 * Reset peak hold buffer.
 */
void reset_peak_hold_gpu(void* state_ptr, int N, float initial_value) {
    DisplayGPUState* state = (DisplayGPUState*)state_ptr;
    if (!state || !state->initialized) return;

    // Fill with initial value
    std::vector<float> init_data(N, initial_value);
    cudaMemcpy(state->d_peak_hold, init_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);
}

/**
 * Check if GPU is available for display operations.
 */
bool is_display_gpu_available() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count > 0;
}

/**
 * Check if this state is using GPU (always true for GPU state).
 */
bool is_display_using_gpu(void* state_ptr) {
    DisplayGPUState* state = (DisplayGPUState*)state_ptr;
    return state && state->initialized;
}

/**
 * Free display GPU state.
 */
void free_display_gpu(void* state_ptr) {
    DisplayGPUState* state = (DisplayGPUState*)state_ptr;
    if (!state) return;

    if (state->d_spectrum) cudaFree(state->d_spectrum);
    if (state->d_spectrum_db) cudaFree(state->d_spectrum_db);
    if (state->d_peak_hold) cudaFree(state->d_peak_hold);
    if (state->d_avg_accum) cudaFree(state->d_avg_accum);
    if (state->d_smooth_buf) cudaFree(state->d_smooth_buf);
    if (state->d_scattering) cudaFree(state->d_scattering);
    if (state->d_scattering_norm) cudaFree(state->d_scattering_norm);
    if (state->d_max_val) cudaFree(state->d_max_val);

    delete state;
}

} // extern "C"
