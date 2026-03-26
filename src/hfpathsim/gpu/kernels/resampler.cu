/**
 * CUDA kernels for polyphase resampling.
 *
 * Implements GPU-accelerated rational rate conversion using polyphase
 * filterbank structure for efficient sample rate conversion.
 *
 * Supports:
 * - Integer upsampling (L:1)
 * - Integer downsampling (1:M)
 * - Rational resampling (L:M)
 * - Arbitrary resampling via cubic interpolation
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.14159265358979323846f
#define TWO_PI 6.28318530717958647692f

// ============================================================================
// Polyphase Filter Bank Kernels
// ============================================================================

/**
 * Polyphase upsampler kernel.
 * Each thread computes one output sample.
 */
__global__ void polyphase_upsample_kernel(
    const float* __restrict__ input_real,
    const float* __restrict__ input_imag,
    float* __restrict__ output_real,
    float* __restrict__ output_imag,
    const float* __restrict__ filter,
    int input_len,
    int output_len,
    int upsample_factor,
    int filter_len,
    int taps_per_phase
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= output_len) return;

    int phase = out_idx % upsample_factor;
    int in_base = out_idx / upsample_factor;
    int half_taps = taps_per_phase / 2;
    int filter_center = filter_len / 2;

    float re = 0.0f, im = 0.0f;

    for (int k = -half_taps; k <= half_taps; k++) {
        int in_idx = in_base - k;
        if (in_idx >= 0 && in_idx < input_len) {
            int filt_idx = filter_center + k * upsample_factor + phase;
            if (filt_idx >= 0 && filt_idx < filter_len) {
                float h = filter[filt_idx] * upsample_factor;
                re += input_real[in_idx] * h;
                im += input_imag[in_idx] * h;
            }
        }
    }

    output_real[out_idx] = re;
    output_imag[out_idx] = im;
}

/**
 * Polyphase downsampler kernel.
 * Each thread computes one output sample.
 */
__global__ void polyphase_downsample_kernel(
    const float* __restrict__ input_real,
    const float* __restrict__ input_imag,
    float* __restrict__ output_real,
    float* __restrict__ output_imag,
    const float* __restrict__ filter,
    int input_len,
    int output_len,
    int downsample_factor,
    int filter_len
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= output_len) return;

    int in_center = out_idx * downsample_factor;
    int half_len = filter_len / 2;

    float re = 0.0f, im = 0.0f;

    for (int k = 0; k < filter_len; k++) {
        int in_idx = in_center - half_len + k;
        if (in_idx >= 0 && in_idx < input_len) {
            float h = filter[k];
            re += input_real[in_idx] * h;
            im += input_imag[in_idx] * h;
        }
    }

    output_real[out_idx] = re;
    output_imag[out_idx] = im;
}

/**
 * Arbitrary resampler using cubic interpolation.
 * Useful when L/M ratio is impractical.
 */
__global__ void cubic_resample_kernel(
    const float* __restrict__ input_real,
    const float* __restrict__ input_imag,
    float* __restrict__ output_real,
    float* __restrict__ output_imag,
    int input_len,
    int output_len,
    float rate_ratio  // input_rate / output_rate
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= output_len) return;

    // Compute input position for this output sample
    float in_pos = (float)out_idx * rate_ratio;
    int in_idx = (int)floorf(in_pos);
    float frac = in_pos - (float)in_idx;

    // Cubic interpolation coefficients (Catmull-Rom)
    float t = frac;
    float t2 = t * t;
    float t3 = t2 * t;

    float c0 = -0.5f * t3 + t2 - 0.5f * t;
    float c1 = 1.5f * t3 - 2.5f * t2 + 1.0f;
    float c2 = -1.5f * t3 + 2.0f * t2 + 0.5f * t;
    float c3 = 0.5f * t3 - 0.5f * t2;

    // Sample indices (clamped)
    int i0 = max(0, min(in_idx - 1, input_len - 1));
    int i1 = max(0, min(in_idx, input_len - 1));
    int i2 = max(0, min(in_idx + 1, input_len - 1));
    int i3 = max(0, min(in_idx + 2, input_len - 1));

    // Interpolate
    output_real[out_idx] = c0 * input_real[i0] + c1 * input_real[i1] +
                           c2 * input_real[i2] + c3 * input_real[i3];
    output_imag[out_idx] = c0 * input_imag[i0] + c1 * input_imag[i1] +
                           c2 * input_imag[i2] + c3 * input_imag[i3];
}

// ============================================================================
// Resampler State Structure
// ============================================================================

struct ResamplerGPUState {
    // Filter coefficients
    float* d_filter;
    int filter_len;
    int taps_per_phase;

    // Configuration
    int upsample_factor;
    int downsample_factor;
    float input_rate;
    float output_rate;

    // Buffers
    float* d_input_real;
    float* d_input_imag;
    float* d_output_real;
    float* d_output_imag;
    float* d_temp_real;
    float* d_temp_imag;

    int max_input_samples;
    int max_output_samples;

    // State for block processing
    float* h_history_real;
    float* h_history_imag;
    int history_len;

    bool initialized;
};

// ============================================================================
// Filter Design Helper
// ============================================================================

static void design_lowpass_filter(float* filter, int len, float cutoff, int upsample) {
    int half = len / 2;

    // Sinc lowpass with Hamming window
    for (int i = 0; i < len; i++) {
        float x = (float)(i - half);
        float h;

        if (x == 0.0f) {
            h = 2.0f * cutoff;
        } else {
            h = sinf(TWO_PI * cutoff * x) / (PI * x);
        }

        // Hamming window
        float w = 0.54f - 0.46f * cosf(TWO_PI * (float)i / (float)(len - 1));
        filter[i] = h * w;
    }

    // Normalize for unity gain at DC
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        sum += filter[i];
    }
    if (sum > 0) {
        for (int i = 0; i < len; i++) {
            filter[i] /= sum;
        }
    }
}

// ============================================================================
// C Interface
// ============================================================================

extern "C" {

/**
 * Initialize resampler.
 */
void* init_resampler_gpu(
    float input_rate,
    float output_rate,
    int max_input_samples
) {
    ResamplerGPUState* state = new ResamplerGPUState;
    memset(state, 0, sizeof(ResamplerGPUState));

    state->input_rate = input_rate;
    state->output_rate = output_rate;
    state->max_input_samples = max_input_samples;

    // Compute rational approximation
    float ratio = output_rate / input_rate;
    if (ratio >= 1.0f) {
        // Upsampling
        state->upsample_factor = (int)roundf(ratio);
        state->downsample_factor = 1;
        if (fabsf(ratio - state->upsample_factor) > 0.01f) {
            // Non-integer, use larger factors
            state->upsample_factor = (int)roundf(ratio * 100);
            state->downsample_factor = 100;
        }
    } else {
        // Downsampling
        state->upsample_factor = 1;
        state->downsample_factor = (int)roundf(1.0f / ratio);
        if (fabsf(1.0f / ratio - state->downsample_factor) > 0.01f) {
            state->upsample_factor = 100;
            state->downsample_factor = (int)roundf(100.0f / ratio);
        }
    }

    // Design filter
    state->taps_per_phase = 16;
    state->filter_len = state->upsample_factor * state->taps_per_phase + 1;

    float* h_filter = new float[state->filter_len];
    float cutoff = 0.5f / (float)fmax(state->upsample_factor, state->downsample_factor);
    design_lowpass_filter(h_filter, state->filter_len, cutoff, state->upsample_factor);

    cudaMalloc(&state->d_filter, state->filter_len * sizeof(float));
    cudaMemcpy(state->d_filter, h_filter, state->filter_len * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_filter;

    // Allocate buffers
    state->max_output_samples = (int)ceilf(max_input_samples * output_rate / input_rate) + 1024;
    int temp_size = max_input_samples * state->upsample_factor + state->filter_len;

    cudaMalloc(&state->d_input_real, max_input_samples * sizeof(float));
    cudaMalloc(&state->d_input_imag, max_input_samples * sizeof(float));
    cudaMalloc(&state->d_output_real, state->max_output_samples * sizeof(float));
    cudaMalloc(&state->d_output_imag, state->max_output_samples * sizeof(float));
    cudaMalloc(&state->d_temp_real, temp_size * sizeof(float));
    cudaMalloc(&state->d_temp_imag, temp_size * sizeof(float));

    // History buffer for continuity
    state->history_len = state->filter_len;
    state->h_history_real = new float[state->history_len]();
    state->h_history_imag = new float[state->history_len]();

    state->initialized = true;
    return state;
}

/**
 * Resample signal.
 */
int resample_gpu(
    void* state_ptr,
    const float* input_real,
    const float* input_imag,
    int n_input,
    float* output_real,
    float* output_imag,
    int* n_output
) {
    ResamplerGPUState* state = (ResamplerGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_input > state->max_input_samples) return -2;

    // Copy input to device
    cudaMemcpy(state->d_input_real, input_real, n_input * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(state->d_input_imag, input_imag, n_input * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;

    if (state->upsample_factor > 1 && state->downsample_factor == 1) {
        // Pure upsampling
        int out_len = n_input * state->upsample_factor;
        int blocks = (out_len + threads - 1) / threads;

        polyphase_upsample_kernel<<<blocks, threads>>>(
            state->d_input_real, state->d_input_imag,
            state->d_output_real, state->d_output_imag,
            state->d_filter,
            n_input, out_len,
            state->upsample_factor,
            state->filter_len,
            state->taps_per_phase
        );

        *n_output = out_len;

    } else if (state->upsample_factor == 1 && state->downsample_factor > 1) {
        // Pure downsampling
        int out_len = n_input / state->downsample_factor;
        int blocks = (out_len + threads - 1) / threads;

        polyphase_downsample_kernel<<<blocks, threads>>>(
            state->d_input_real, state->d_input_imag,
            state->d_output_real, state->d_output_imag,
            state->d_filter,
            n_input, out_len,
            state->downsample_factor,
            state->filter_len
        );

        *n_output = out_len;

    } else if (state->upsample_factor > 1 && state->downsample_factor > 1) {
        // Rational resampling: upsample then downsample
        int temp_len = n_input * state->upsample_factor;
        int out_len = temp_len / state->downsample_factor;

        // Upsample
        int blocks1 = (temp_len + threads - 1) / threads;
        polyphase_upsample_kernel<<<blocks1, threads>>>(
            state->d_input_real, state->d_input_imag,
            state->d_temp_real, state->d_temp_imag,
            state->d_filter,
            n_input, temp_len,
            state->upsample_factor,
            state->filter_len,
            state->taps_per_phase
        );

        // Downsample
        int blocks2 = (out_len + threads - 1) / threads;
        polyphase_downsample_kernel<<<blocks2, threads>>>(
            state->d_temp_real, state->d_temp_imag,
            state->d_output_real, state->d_output_imag,
            state->d_filter,
            temp_len, out_len,
            state->downsample_factor,
            state->filter_len
        );

        *n_output = out_len;

    } else {
        // No resampling needed (1:1)
        cudaMemcpy(state->d_output_real, state->d_input_real, n_input * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(state->d_output_imag, state->d_input_imag, n_input * sizeof(float), cudaMemcpyDeviceToDevice);
        *n_output = n_input;
    }

    cudaDeviceSynchronize();

    // Copy output to host
    cudaMemcpy(output_real, state->d_output_real, *n_output * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_imag, state->d_output_imag, *n_output * sizeof(float), cudaMemcpyDeviceToHost);

    return 0;
}

/**
 * Arbitrary rate resample using cubic interpolation.
 */
int resample_arbitrary_gpu(
    void* state_ptr,
    const float* input_real,
    const float* input_imag,
    int n_input,
    float* output_real,
    float* output_imag,
    int n_output
) {
    ResamplerGPUState* state = (ResamplerGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_input > state->max_input_samples) return -2;
    if (n_output > state->max_output_samples) return -3;

    cudaMemcpy(state->d_input_real, input_real, n_input * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(state->d_input_imag, input_imag, n_input * sizeof(float), cudaMemcpyHostToDevice);

    float rate_ratio = (float)n_input / (float)n_output;
    int threads = 256;
    int blocks = (n_output + threads - 1) / threads;

    cubic_resample_kernel<<<blocks, threads>>>(
        state->d_input_real, state->d_input_imag,
        state->d_output_real, state->d_output_imag,
        n_input, n_output, rate_ratio
    );

    cudaDeviceSynchronize();

    cudaMemcpy(output_real, state->d_output_real, n_output * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_imag, state->d_output_imag, n_output * sizeof(float), cudaMemcpyDeviceToHost);

    return 0;
}

/**
 * Get expected output length for given input.
 */
int get_output_length_gpu(void* state_ptr, int n_input) {
    ResamplerGPUState* state = (ResamplerGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;

    if (state->upsample_factor > 1 && state->downsample_factor == 1) {
        return n_input * state->upsample_factor;
    } else if (state->upsample_factor == 1 && state->downsample_factor > 1) {
        return n_input / state->downsample_factor;
    } else {
        return (n_input * state->upsample_factor) / state->downsample_factor;
    }
}

/**
 * Check if using GPU.
 */
bool is_resampler_using_gpu(void* state_ptr) {
    return state_ptr != nullptr;
}

/**
 * Reset resampler state.
 */
void reset_resampler_gpu(void* state_ptr) {
    ResamplerGPUState* state = (ResamplerGPUState*)state_ptr;
    if (!state || !state->initialized) return;

    memset(state->h_history_real, 0, state->history_len * sizeof(float));
    memset(state->h_history_imag, 0, state->history_len * sizeof(float));
}

/**
 * Free resampler resources.
 */
void free_resampler_gpu(void* state_ptr) {
    ResamplerGPUState* state = (ResamplerGPUState*)state_ptr;
    if (!state) return;

    if (state->d_filter) cudaFree(state->d_filter);
    if (state->d_input_real) cudaFree(state->d_input_real);
    if (state->d_input_imag) cudaFree(state->d_input_imag);
    if (state->d_output_real) cudaFree(state->d_output_real);
    if (state->d_output_imag) cudaFree(state->d_output_imag);
    if (state->d_temp_real) cudaFree(state->d_temp_real);
    if (state->d_temp_imag) cudaFree(state->d_temp_imag);
    if (state->h_history_real) delete[] state->h_history_real;
    if (state->h_history_imag) delete[] state->h_history_imag;

    delete state;
}

} // extern "C"
