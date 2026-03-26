/**
 * CUDA kernels for AGC (Automatic Gain Control) and Limiter.
 *
 * Implements GPU-accelerated:
 * - Per-sample envelope detection with attack/release
 * - Hang AGC with configurable hold time
 * - Soft-knee gain computation
 * - Hard/soft/cubic limiting
 *
 * Note: AGC has inherent sequential dependency (IIR filter), so we use
 * a hybrid approach: envelope detection in parallel blocks, gain
 * smoothing sequentially, then gain application in parallel.
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>

#define THREADS_PER_BLOCK 256

/**
 * Compute envelope (magnitude) of complex signal.
 */
__global__ void compute_envelope(
    const cuFloatComplex* signal,
    float* envelope,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float re = signal[idx].x;
    float im = signal[idx].y;
    envelope[idx] = sqrtf(re * re + im * im);
}

/**
 * Sequential envelope smoothing with attack/release.
 *
 * This kernel must run with a single thread due to IIR dependencies.
 * It processes samples in chunks for efficiency.
 */
__global__ void smooth_envelope_sequential(
    const float* raw_envelope,
    float* smoothed_envelope,
    float* state_envelope,
    float* state_peak,
    int* state_hold,
    float alpha_attack,
    float alpha_release,
    int hold_samples,
    bool hang_agc,
    int N
) {
    // Single-threaded kernel
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float env = *state_envelope;
    float peak = *state_peak;
    int hold = *state_hold;

    for (int i = 0; i < N; i++) {
        float raw = raw_envelope[i];

        if (raw > env) {
            // Attack
            env += alpha_attack * (raw - env);
            peak = env;
            hold = hold_samples;
        } else {
            // Hold or release
            if (hang_agc && hold > 0) {
                hold--;
                // Keep envelope at peak during hold
            } else {
                // Release
                env += alpha_release * (raw - env);
            }
        }

        smoothed_envelope[i] = env;
    }

    // Save state
    *state_envelope = env;
    *state_peak = peak;
    *state_hold = hold;
}

/**
 * Compute gain values from smoothed envelope.
 */
__global__ void compute_agc_gains(
    const float* smoothed_envelope,
    float* gain_db,
    float target_level_linear,
    float max_gain_db,
    float min_gain_db,
    float soft_knee_db,
    float* state_gain_db,
    int N
) {
    // Single-threaded for gain smoothing continuity
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float current_gain = *state_gain_db;
    float target_db = 20.0f * log10f(target_level_linear);

    for (int i = 0; i < N; i++) {
        float env = smoothed_envelope[i];
        float desired_gain;

        if (env > 1e-10f) {
            float env_db = 20.0f * log10f(env);
            desired_gain = target_db - env_db;
        } else {
            desired_gain = max_gain_db;
        }

        // Apply soft knee
        float diff = fabsf(desired_gain - current_gain);
        if (diff < soft_knee_db) {
            // In knee region - smooth transition
            current_gain += 0.1f * (desired_gain - current_gain);
        } else {
            current_gain = desired_gain;
        }

        // Clamp gain to range
        if (current_gain > max_gain_db) current_gain = max_gain_db;
        if (current_gain < min_gain_db) current_gain = min_gain_db;

        gain_db[i] = current_gain;
    }

    *state_gain_db = current_gain;
}

/**
 * Apply gain to complex signal (parallel).
 */
__global__ void apply_gain(
    const cuFloatComplex* input,
    cuFloatComplex* output,
    const float* gain_db,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float gain_linear = powf(10.0f, gain_db[idx] / 20.0f);

    output[idx].x = input[idx].x * gain_linear;
    output[idx].y = input[idx].y * gain_linear;
}

/**
 * Hard limiter (parallel).
 */
__global__ void apply_hard_limiter(
    const cuFloatComplex* input,
    cuFloatComplex* output,
    float threshold,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float re = input[idx].x;
    float im = input[idx].y;
    float mag = sqrtf(re * re + im * im);

    if (mag > threshold && mag > 1e-10f) {
        float scale = threshold / mag;
        output[idx].x = re * scale;
        output[idx].y = im * scale;
    } else {
        output[idx] = input[idx];
    }
}

/**
 * Soft limiter using tanh (parallel).
 */
__global__ void apply_soft_limiter(
    const cuFloatComplex* input,
    cuFloatComplex* output,
    float threshold,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float re = input[idx].x;
    float im = input[idx].y;
    float mag = sqrtf(re * re + im * im);

    if (mag > 1e-10f) {
        float normalized = mag / threshold;
        float limited = tanhf(normalized) * threshold;
        float scale = limited / mag;
        output[idx].x = re * scale;
        output[idx].y = im * scale;
    } else {
        output[idx] = input[idx];
    }
}

/**
 * Cubic soft limiter (parallel).
 */
__global__ void apply_cubic_limiter(
    const cuFloatComplex* input,
    cuFloatComplex* output,
    float threshold,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float re = input[idx].x;
    float im = input[idx].y;
    float mag = sqrtf(re * re + im * im);

    if (mag > 1e-10f) {
        float normalized = mag / threshold;
        float limited_norm;

        if (normalized < 1.0f) {
            limited_norm = normalized;  // Linear below threshold
        } else {
            // Cubic compression above threshold
            float x = normalized - 1.0f;
            limited_norm = 1.0f + x / (1.0f + x * x);
        }

        float limited = limited_norm * threshold;
        float scale = limited / mag;
        output[idx].x = re * scale;
        output[idx].y = im * scale;
    } else {
        output[idx] = input[idx];
    }
}

// Host wrapper structures and functions

extern "C" {

/**
 * AGC GPU state.
 */
struct AGCGPUState {
    // Device memory
    cuFloatComplex* input_dev;
    cuFloatComplex* output_dev;
    float* envelope_dev;
    float* smoothed_dev;
    float* gain_db_dev;

    // State (device)
    float* state_envelope_dev;
    float* state_peak_dev;
    int* state_hold_dev;
    float* state_gain_db_dev;

    // Parameters
    float alpha_attack;
    float alpha_release;
    int hold_samples;
    bool hang_agc;
    float target_level_linear;
    float max_gain_db;
    float min_gain_db;
    float soft_knee_db;
    int max_samples;

    bool initialized;
};

/**
 * Limiter GPU state.
 */
struct LimiterGPUState {
    cuFloatComplex* input_dev;
    cuFloatComplex* output_dev;
    float threshold;
    int mode;  // 0=hard, 1=soft, 2=cubic
    int max_samples;
    bool initialized;
};

/**
 * Initialize AGC GPU processor.
 */
void* init_agc_gpu(
    float sample_rate,
    float attack_time_ms,
    float release_time_ms,
    float hold_time_ms,
    bool hang_agc,
    float target_level_db,
    float max_gain_db,
    float min_gain_db,
    float soft_knee_db,
    int max_samples
) {
    AGCGPUState* state = new AGCGPUState;
    state->initialized = false;
    state->max_samples = max_samples;

    // Compute coefficients
    float tau_attack = attack_time_ms / 1000.0f;
    float tau_release = release_time_ms / 1000.0f;

    state->alpha_attack = 1.0f - expf(-1.0f / (sample_rate * tau_attack));
    state->alpha_release = 1.0f - expf(-1.0f / (sample_rate * tau_release));
    state->hold_samples = (int)(hold_time_ms / 1000.0f * sample_rate);
    state->hang_agc = hang_agc;
    state->target_level_linear = powf(10.0f, target_level_db / 20.0f);
    state->max_gain_db = max_gain_db;
    state->min_gain_db = min_gain_db;
    state->soft_knee_db = soft_knee_db;

    // Allocate device memory
    cudaError_t err;
    err = cudaMalloc(&state->input_dev, max_samples * sizeof(cuFloatComplex));
    if (err != cudaSuccess) { delete state; return nullptr; }

    err = cudaMalloc(&state->output_dev, max_samples * sizeof(cuFloatComplex));
    if (err != cudaSuccess) goto cleanup_1;

    err = cudaMalloc(&state->envelope_dev, max_samples * sizeof(float));
    if (err != cudaSuccess) goto cleanup_2;

    err = cudaMalloc(&state->smoothed_dev, max_samples * sizeof(float));
    if (err != cudaSuccess) goto cleanup_3;

    err = cudaMalloc(&state->gain_db_dev, max_samples * sizeof(float));
    if (err != cudaSuccess) goto cleanup_4;

    err = cudaMalloc(&state->state_envelope_dev, sizeof(float));
    if (err != cudaSuccess) goto cleanup_5;

    err = cudaMalloc(&state->state_peak_dev, sizeof(float));
    if (err != cudaSuccess) goto cleanup_6;

    err = cudaMalloc(&state->state_hold_dev, sizeof(int));
    if (err != cudaSuccess) goto cleanup_7;

    err = cudaMalloc(&state->state_gain_db_dev, sizeof(float));
    if (err != cudaSuccess) goto cleanup_8;

    // Initialize state
    {
        float zero = 0.0f;
        int zero_int = 0;
        cudaMemcpy(state->state_envelope_dev, &zero, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(state->state_peak_dev, &zero, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(state->state_hold_dev, &zero_int, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(state->state_gain_db_dev, &zero, sizeof(float), cudaMemcpyHostToDevice);
    }

    state->initialized = true;
    return state;

cleanup_8: cudaFree(state->state_hold_dev);
cleanup_7: cudaFree(state->state_peak_dev);
cleanup_6: cudaFree(state->state_envelope_dev);
cleanup_5: cudaFree(state->gain_db_dev);
cleanup_4: cudaFree(state->smoothed_dev);
cleanup_3: cudaFree(state->envelope_dev);
cleanup_2: cudaFree(state->output_dev);
cleanup_1: cudaFree(state->input_dev);
    delete state;
    return nullptr;
}

/**
 * Process samples through AGC.
 */
int process_agc_gpu(
    void* state_ptr,
    const float* input_real,
    const float* input_imag,
    int n_samples,
    float* output_real,
    float* output_imag
) {
    AGCGPUState* state = (AGCGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_samples > state->max_samples) return -2;

    // Pack input and copy to device
    cuFloatComplex* input_host = new cuFloatComplex[n_samples];
    for (int i = 0; i < n_samples; i++) {
        input_host[i].x = input_real[i];
        input_host[i].y = input_imag[i];
    }
    cudaMemcpy(state->input_dev, input_host, n_samples * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    delete[] input_host;

    int threads = THREADS_PER_BLOCK;
    int blocks = (n_samples + threads - 1) / threads;

    // Step 1: Compute envelope (parallel)
    compute_envelope<<<blocks, threads>>>(state->input_dev, state->envelope_dev, n_samples);

    // Step 2: Smooth envelope (sequential due to IIR)
    smooth_envelope_sequential<<<1, 1>>>(
        state->envelope_dev,
        state->smoothed_dev,
        state->state_envelope_dev,
        state->state_peak_dev,
        state->state_hold_dev,
        state->alpha_attack,
        state->alpha_release,
        state->hold_samples,
        state->hang_agc,
        n_samples
    );

    // Step 3: Compute gain (sequential for continuity)
    compute_agc_gains<<<1, 1>>>(
        state->smoothed_dev,
        state->gain_db_dev,
        state->target_level_linear,
        state->max_gain_db,
        state->min_gain_db,
        state->soft_knee_db,
        state->state_gain_db_dev,
        n_samples
    );

    // Step 4: Apply gain (parallel)
    apply_gain<<<blocks, threads>>>(state->input_dev, state->output_dev, state->gain_db_dev, n_samples);

    cudaDeviceSynchronize();

    // Copy output back
    cuFloatComplex* output_host = new cuFloatComplex[n_samples];
    cudaMemcpy(output_host, state->output_dev, n_samples * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n_samples; i++) {
        output_real[i] = output_host[i].x;
        output_imag[i] = output_host[i].y;
    }
    delete[] output_host;

    return 0;
}

/**
 * Get current AGC gain.
 */
float get_agc_gain_db(void* state_ptr) {
    AGCGPUState* state = (AGCGPUState*)state_ptr;
    if (!state || !state->initialized) return 0.0f;

    float gain_db;
    cudaMemcpy(&gain_db, state->state_gain_db_dev, sizeof(float), cudaMemcpyDeviceToHost);
    return gain_db;
}

/**
 * Reset AGC state.
 */
void reset_agc_gpu(void* state_ptr) {
    AGCGPUState* state = (AGCGPUState*)state_ptr;
    if (!state || !state->initialized) return;

    float zero = 0.0f;
    int zero_int = 0;
    cudaMemcpy(state->state_envelope_dev, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(state->state_peak_dev, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(state->state_hold_dev, &zero_int, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(state->state_gain_db_dev, &zero, sizeof(float), cudaMemcpyHostToDevice);
}

/**
 * Free AGC GPU resources.
 */
void free_agc_gpu(void* state_ptr) {
    AGCGPUState* state = (AGCGPUState*)state_ptr;
    if (!state) return;

    if (state->initialized) {
        cudaFree(state->input_dev);
        cudaFree(state->output_dev);
        cudaFree(state->envelope_dev);
        cudaFree(state->smoothed_dev);
        cudaFree(state->gain_db_dev);
        cudaFree(state->state_envelope_dev);
        cudaFree(state->state_peak_dev);
        cudaFree(state->state_hold_dev);
        cudaFree(state->state_gain_db_dev);
    }

    delete state;
}

// ============================================================================
// Limiter Functions
// ============================================================================

/**
 * Initialize limiter GPU processor.
 */
void* init_limiter_gpu(float threshold_db, int mode, int max_samples) {
    LimiterGPUState* state = new LimiterGPUState;
    state->initialized = false;
    state->max_samples = max_samples;
    state->threshold = powf(10.0f, threshold_db / 20.0f);
    state->mode = mode;

    cudaError_t err;
    err = cudaMalloc(&state->input_dev, max_samples * sizeof(cuFloatComplex));
    if (err != cudaSuccess) { delete state; return nullptr; }

    err = cudaMalloc(&state->output_dev, max_samples * sizeof(cuFloatComplex));
    if (err != cudaSuccess) {
        cudaFree(state->input_dev);
        delete state;
        return nullptr;
    }

    state->initialized = true;
    return state;
}

/**
 * Process samples through limiter.
 */
int process_limiter_gpu(
    void* state_ptr,
    const float* input_real,
    const float* input_imag,
    int n_samples,
    float* output_real,
    float* output_imag
) {
    LimiterGPUState* state = (LimiterGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_samples > state->max_samples) return -2;

    // Copy input to device
    cuFloatComplex* input_host = new cuFloatComplex[n_samples];
    for (int i = 0; i < n_samples; i++) {
        input_host[i].x = input_real[i];
        input_host[i].y = input_imag[i];
    }
    cudaMemcpy(state->input_dev, input_host, n_samples * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    delete[] input_host;

    int threads = THREADS_PER_BLOCK;
    int blocks = (n_samples + threads - 1) / threads;

    // Apply limiter based on mode
    switch (state->mode) {
        case 0:  // Hard
            apply_hard_limiter<<<blocks, threads>>>(
                state->input_dev, state->output_dev, state->threshold, n_samples);
            break;
        case 1:  // Soft (tanh)
            apply_soft_limiter<<<blocks, threads>>>(
                state->input_dev, state->output_dev, state->threshold, n_samples);
            break;
        case 2:  // Cubic
            apply_cubic_limiter<<<blocks, threads>>>(
                state->input_dev, state->output_dev, state->threshold, n_samples);
            break;
        default:
            // Pass through
            cudaMemcpy(state->output_dev, state->input_dev,
                       n_samples * sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice);
    }

    cudaDeviceSynchronize();

    // Copy output back
    cuFloatComplex* output_host = new cuFloatComplex[n_samples];
    cudaMemcpy(output_host, state->output_dev, n_samples * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n_samples; i++) {
        output_real[i] = output_host[i].x;
        output_imag[i] = output_host[i].y;
    }
    delete[] output_host;

    return 0;
}

/**
 * Update limiter parameters.
 */
void set_limiter_params_gpu(void* state_ptr, float threshold_db, int mode) {
    LimiterGPUState* state = (LimiterGPUState*)state_ptr;
    if (!state || !state->initialized) return;

    state->threshold = powf(10.0f, threshold_db / 20.0f);
    state->mode = mode;
}

/**
 * Free limiter GPU resources.
 */
void free_limiter_gpu(void* state_ptr) {
    LimiterGPUState* state = (LimiterGPUState*)state_ptr;
    if (!state) return;

    if (state->initialized) {
        cudaFree(state->input_dev);
        cudaFree(state->output_dev);
    }

    delete state;
}

} // extern "C"
