/**
 * CUDA kernels for ionospheric dispersion filtering.
 *
 * Implements chirp all-pass filter for frequency-dependent group delay.
 * Uses FFT-based overlap-save convolution for efficiency.
 *
 * Theory:
 *   Linear dispersion tau(f) = tau_0 + d*(f - f_c) produces quadratic phase.
 *   Impulse response: h(t) = exp(j*pi*t^2/d) / sqrt(j*d)
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define PI 3.14159265358979323846f
#define TWO_PI 6.28318530717958647692f

// ============================================================================
// Chirp Filter Generation Kernel
// ============================================================================

/**
 * Generate chirp impulse response for dispersion filter.
 * h(t) = exp(j*pi*t^2/d_chirp) * window
 */
__global__ void generate_chirp_filter_kernel(
    cuFloatComplex* filter,
    int filter_len,
    float d_chirp,  // Chirp rate parameter (s^2)
    float fs,       // Sample rate
    float window_alpha  // Tukey window parameter
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= filter_len) return;

    int half = filter_len / 2;
    float t = (float)(idx - half) / fs;

    // Chirp phase
    float phase = PI * t * t / d_chirp;

    // Complex exponential
    float re = cosf(phase);
    float im = sinf(phase);

    // Tukey window
    float win = 1.0f;
    float pos = (float)idx / (float)(filter_len - 1);
    if (pos < window_alpha / 2.0f) {
        win = 0.5f * (1.0f + cosf(TWO_PI * (pos / window_alpha - 0.5f)));
    } else if (pos > 1.0f - window_alpha / 2.0f) {
        win = 0.5f * (1.0f + cosf(TWO_PI * ((pos - 1.0f) / window_alpha + 0.5f)));
    }

    filter[idx] = make_cuFloatComplex(re * win, im * win);
}

/**
 * Normalize filter for unit energy.
 */
__global__ void normalize_filter_kernel(
    cuFloatComplex* filter,
    int filter_len,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= filter_len) return;

    filter[idx] = make_cuFloatComplex(
        cuCrealf(filter[idx]) * scale,
        cuCimagf(filter[idx]) * scale
    );
}

/**
 * Compute sum of squared magnitudes (for reduction).
 */
__global__ void compute_energy_kernel(
    const cuFloatComplex* filter,
    float* energy,
    int filter_len
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (idx < filter_len) {
        cuFloatComplex c = filter[idx];
        val = cuCrealf(c) * cuCrealf(c) + cuCimagf(c) * cuCimagf(c);
    }
    sdata[tid] = val;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(energy, sdata[0]);
    }
}

// ============================================================================
// Overlap-Save Convolution Kernels
// ============================================================================

/**
 * Complex multiplication for FFT convolution.
 */
__global__ void complex_multiply_kernel(
    const cuFloatComplex* __restrict__ a,
    const cuFloatComplex* __restrict__ b,
    cuFloatComplex* __restrict__ c,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    cuFloatComplex va = a[idx];
    cuFloatComplex vb = b[idx];

    c[idx] = make_cuFloatComplex(
        cuCrealf(va) * cuCrealf(vb) - cuCimagf(va) * cuCimagf(vb),
        cuCrealf(va) * cuCimagf(vb) + cuCimagf(va) * cuCrealf(vb)
    );
}

/**
 * Extract valid samples from overlap-save convolution.
 */
__global__ void extract_valid_kernel(
    const cuFloatComplex* __restrict__ input,
    cuFloatComplex* __restrict__ output,
    int overlap,
    int valid_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= valid_len) return;

    output[idx] = input[idx + overlap];
}

// ============================================================================
// Dispersion Filter State
// ============================================================================

struct DispersionGPUState {
    // Sample rate
    float sample_rate;

    // Current dispersion coefficient
    float d_us_per_MHz;
    float d_chirp;  // Converted to s^2

    // Filter
    cuFloatComplex* d_filter_time;
    cuFloatComplex* d_filter_freq;
    int filter_len;

    // FFT plan
    cufftHandle fft_plan;
    int fft_size;
    int overlap;
    int valid_per_block;

    // Work buffers
    cuFloatComplex* d_input_padded;
    cuFloatComplex* d_input_freq;
    cuFloatComplex* d_output_freq;
    cuFloatComplex* d_output_time;

    // Energy computation
    float* d_energy;

    int max_samples;
    bool initialized;
    bool filter_valid;
};

// ============================================================================
// C Interface
// ============================================================================

extern "C" {

/**
 * Initialize dispersion filter.
 */
void* init_dispersion_gpu(
    float sample_rate,
    int max_samples
) {
    DispersionGPUState* state = new DispersionGPUState;
    memset(state, 0, sizeof(DispersionGPUState));

    state->sample_rate = sample_rate;
    state->max_samples = max_samples;
    state->d_us_per_MHz = 0.0f;
    state->filter_valid = false;

    // Compute FFT size (power of 2, at least 4x max filter length)
    // Typical dispersion filter: ~500-2000 taps
    int min_fft = 4096;
    state->fft_size = min_fft;
    while (state->fft_size < max_samples + 2048) {
        state->fft_size *= 2;
    }

    state->filter_len = 2001;  // Odd, centered
    state->overlap = state->filter_len - 1;
    state->valid_per_block = state->fft_size - state->overlap;

    // Allocate GPU memory
    cudaMalloc(&state->d_filter_time, state->fft_size * sizeof(cuFloatComplex));
    cudaMalloc(&state->d_filter_freq, state->fft_size * sizeof(cuFloatComplex));
    cudaMalloc(&state->d_input_padded, state->fft_size * sizeof(cuFloatComplex));
    cudaMalloc(&state->d_input_freq, state->fft_size * sizeof(cuFloatComplex));
    cudaMalloc(&state->d_output_freq, state->fft_size * sizeof(cuFloatComplex));
    cudaMalloc(&state->d_output_time, state->fft_size * sizeof(cuFloatComplex));
    cudaMalloc(&state->d_energy, sizeof(float));

    // Initialize to zero
    cudaMemset(state->d_filter_time, 0, state->fft_size * sizeof(cuFloatComplex));
    cudaMemset(state->d_input_padded, 0, state->fft_size * sizeof(cuFloatComplex));

    // Create FFT plan
    cufftPlan1d(&state->fft_plan, state->fft_size, CUFFT_C2C, 1);

    state->initialized = true;
    return state;
}

/**
 * Set dispersion coefficient and regenerate filter.
 */
int set_dispersion_coefficient_gpu(
    void* state_ptr,
    float d_us_per_MHz
) {
    DispersionGPUState* state = (DispersionGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;

    // Skip if unchanged
    if (fabsf(d_us_per_MHz - state->d_us_per_MHz) < 0.001f && state->filter_valid) {
        return 0;
    }

    state->d_us_per_MHz = d_us_per_MHz;

    // Handle zero/negligible dispersion
    if (fabsf(d_us_per_MHz) < 0.01f) {
        // Delta function filter
        cudaMemset(state->d_filter_time, 0, state->fft_size * sizeof(cuFloatComplex));
        cuFloatComplex one = make_cuFloatComplex(1.0f, 0.0f);
        cudaMemcpy(state->d_filter_time + state->filter_len / 2, &one,
                   sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

        // FFT of filter
        cufftExecC2C(state->fft_plan, state->d_filter_time, state->d_filter_freq, CUFFT_FORWARD);
        cudaDeviceSynchronize();

        state->filter_valid = true;
        return 0;
    }

    // Convert units: us/MHz -> s^2
    // d in us/MHz -> s/Hz = 1e-12
    // For chirp: h(t) = exp(j*pi*t^2/d_chirp)
    state->d_chirp = d_us_per_MHz * 1e-12f;

    // Compute filter length based on dispersion spread
    float bandwidth = state->sample_rate / 2.0f;
    float spread_s = fabsf(d_us_per_MHz) * (bandwidth / 1e6f) * 1e-6f;
    int needed_len = (int)ceilf(spread_s * state->sample_rate * 4.0f);
    needed_len = (needed_len | 1);  // Ensure odd

    // Clamp filter length
    state->filter_len = std::min(needed_len, state->fft_size / 2);
    state->filter_len = std::max(state->filter_len, 17);
    if (state->filter_len % 2 == 0) state->filter_len++;

    state->overlap = state->filter_len - 1;
    state->valid_per_block = state->fft_size - state->overlap;

    // Generate chirp filter
    cudaMemset(state->d_filter_time, 0, state->fft_size * sizeof(cuFloatComplex));

    int threads = 256;
    int blocks = (state->filter_len + threads - 1) / threads;

    generate_chirp_filter_kernel<<<blocks, threads>>>(
        state->d_filter_time,
        state->filter_len,
        state->d_chirp,
        state->sample_rate,
        0.2f  // Tukey window alpha
    );
    cudaDeviceSynchronize();

    // Compute energy for normalization
    cudaMemset(state->d_energy, 0, sizeof(float));
    compute_energy_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        state->d_filter_time, state->d_energy, state->filter_len
    );
    cudaDeviceSynchronize();

    float energy;
    cudaMemcpy(&energy, state->d_energy, sizeof(float), cudaMemcpyDeviceToHost);

    if (energy > 0) {
        float scale = 1.0f / sqrtf(energy);
        normalize_filter_kernel<<<blocks, threads>>>(
            state->d_filter_time, state->filter_len, scale
        );
        cudaDeviceSynchronize();
    }

    // FFT of filter
    cufftExecC2C(state->fft_plan, state->d_filter_time, state->d_filter_freq, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    state->filter_valid = true;
    return 0;
}

/**
 * Apply dispersion to signal.
 */
int apply_dispersion_gpu(
    void* state_ptr,
    const float* input_real,
    const float* input_imag,
    int n_samples,
    float* output_real,
    float* output_imag
) {
    DispersionGPUState* state = (DispersionGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (!state->filter_valid) return -2;
    if (n_samples > state->max_samples) return -3;

    // Handle trivial case
    if (fabsf(state->d_us_per_MHz) < 0.01f) {
        cudaMemcpy(output_real, input_real, n_samples * sizeof(float), cudaMemcpyHostToHost);
        cudaMemcpy(output_imag, input_imag, n_samples * sizeof(float), cudaMemcpyHostToHost);
        return 0;
    }

    int threads = 256;

    // For short signals, use single-block convolution
    if (n_samples <= state->valid_per_block) {
        // Pad input to FFT size
        cudaMemset(state->d_input_padded, 0, state->fft_size * sizeof(cuFloatComplex));

        // Copy input (interleaved real/imag)
        std::vector<cuFloatComplex> h_input(n_samples);
        for (int i = 0; i < n_samples; i++) {
            h_input[i] = make_cuFloatComplex(input_real[i], input_imag[i]);
        }
        cudaMemcpy(state->d_input_padded + state->overlap, h_input.data(),
                   n_samples * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

        // FFT of input
        cufftExecC2C(state->fft_plan, state->d_input_padded, state->d_input_freq, CUFFT_FORWARD);

        // Multiply spectra
        int blocks = (state->fft_size + threads - 1) / threads;
        complex_multiply_kernel<<<blocks, threads>>>(
            state->d_input_freq, state->d_filter_freq, state->d_output_freq, state->fft_size
        );

        // IFFT
        cufftExecC2C(state->fft_plan, state->d_output_freq, state->d_output_time, CUFFT_INVERSE);
        cudaDeviceSynchronize();

        // Extract valid samples and copy to host
        std::vector<cuFloatComplex> h_output(n_samples);
        cudaMemcpy(h_output.data(), state->d_output_time + state->overlap,
                   n_samples * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

        // Scale by FFT size and copy to output
        float scale = 1.0f / state->fft_size;
        for (int i = 0; i < n_samples; i++) {
            output_real[i] = cuCrealf(h_output[i]) * scale;
            output_imag[i] = cuCimagf(h_output[i]) * scale;
        }

        return 0;
    }

    // Overlap-save for longer signals
    std::vector<cuFloatComplex> h_input(n_samples);
    for (int i = 0; i < n_samples; i++) {
        h_input[i] = make_cuFloatComplex(input_real[i], input_imag[i]);
    }

    std::vector<cuFloatComplex> h_output(n_samples);
    int output_pos = 0;
    int input_pos = 0;

    while (output_pos < n_samples) {
        // Clear and fill input buffer
        cudaMemset(state->d_input_padded, 0, state->fft_size * sizeof(cuFloatComplex));

        int samples_to_copy = std::min(state->valid_per_block, n_samples - input_pos);
        if (samples_to_copy > 0) {
            cudaMemcpy(state->d_input_padded + state->overlap, h_input.data() + input_pos,
                       samples_to_copy * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
        }

        // FFT
        cufftExecC2C(state->fft_plan, state->d_input_padded, state->d_input_freq, CUFFT_FORWARD);

        // Multiply
        int blocks = (state->fft_size + threads - 1) / threads;
        complex_multiply_kernel<<<blocks, threads>>>(
            state->d_input_freq, state->d_filter_freq, state->d_output_freq, state->fft_size
        );

        // IFFT
        cufftExecC2C(state->fft_plan, state->d_output_freq, state->d_output_time, CUFFT_INVERSE);
        cudaDeviceSynchronize();

        // Copy valid portion
        int valid_to_copy = std::min(state->valid_per_block, n_samples - output_pos);
        std::vector<cuFloatComplex> h_block(valid_to_copy);
        cudaMemcpy(h_block.data(), state->d_output_time + state->overlap,
                   valid_to_copy * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

        float scale = 1.0f / state->fft_size;
        for (int i = 0; i < valid_to_copy; i++) {
            h_output[output_pos + i] = make_cuFloatComplex(
                cuCrealf(h_block[i]) * scale,
                cuCimagf(h_block[i]) * scale
            );
        }

        input_pos += state->valid_per_block;
        output_pos += valid_to_copy;
    }

    // Copy to output arrays
    for (int i = 0; i < n_samples; i++) {
        output_real[i] = cuCrealf(h_output[i]);
        output_imag[i] = cuCimagf(h_output[i]);
    }

    return 0;
}

/**
 * Apply inverse dispersion (compression).
 */
int apply_inverse_dispersion_gpu(
    void* state_ptr,
    const float* input_real,
    const float* input_imag,
    int n_samples,
    float* output_real,
    float* output_imag
) {
    DispersionGPUState* state = (DispersionGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;

    // Temporarily negate coefficient
    float orig_d = state->d_us_per_MHz;
    set_dispersion_coefficient_gpu(state_ptr, -orig_d);

    int ret = apply_dispersion_gpu(state_ptr, input_real, input_imag, n_samples,
                                   output_real, output_imag);

    // Restore original coefficient
    set_dispersion_coefficient_gpu(state_ptr, orig_d);

    return ret;
}

/**
 * Get current filter length.
 */
int get_dispersion_filter_length(void* state_ptr) {
    DispersionGPUState* state = (DispersionGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    return state->filter_len;
}

/**
 * Check if using GPU.
 */
bool is_dispersion_using_gpu(void* state_ptr) {
    return state_ptr != nullptr;
}

/**
 * Reset dispersion filter.
 */
void reset_dispersion_gpu(void* state_ptr) {
    DispersionGPUState* state = (DispersionGPUState*)state_ptr;
    if (!state || !state->initialized) return;

    state->filter_valid = false;
    state->d_us_per_MHz = 0.0f;
}

/**
 * Free dispersion filter resources.
 */
void free_dispersion_gpu(void* state_ptr) {
    DispersionGPUState* state = (DispersionGPUState*)state_ptr;
    if (!state) return;

    if (state->d_filter_time) cudaFree(state->d_filter_time);
    if (state->d_filter_freq) cudaFree(state->d_filter_freq);
    if (state->d_input_padded) cudaFree(state->d_input_padded);
    if (state->d_input_freq) cudaFree(state->d_input_freq);
    if (state->d_output_freq) cudaFree(state->d_output_freq);
    if (state->d_output_time) cudaFree(state->d_output_time);
    if (state->d_energy) cudaFree(state->d_energy);

    if (state->fft_plan) cufftDestroy(state->fft_plan);

    delete state;
}

} // extern "C"
