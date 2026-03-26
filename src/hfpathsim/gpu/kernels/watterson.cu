/**
 * CUDA kernels for Watterson HF channel model.
 *
 * Implements GPU-accelerated tapped delay line with:
 * - Independent Rayleigh/Rician fading per tap
 * - Gaussian, Flat, or Jakes Doppler spectrum
 * - Block-rate fading updates with smooth interpolation
 * - Phase-continuous delay lines
 *
 * Reference: Watterson, C.C., et al., "Experimental confirmation of an HF
 * channel model," IEEE Trans. Comm. Tech., Dec. 1970.
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <curand_kernel.h>
#include <cmath>

#define PI 3.14159265358979323846f
#define TWO_PI 6.28318530717958647692f
#define INV_SQRT2 0.7071067811865476f

/**
 * Doppler spectrum types.
 */
enum DopplerSpectrumType {
    DOPPLER_GAUSSIAN = 0,
    DOPPLER_FLAT = 1,
    DOPPLER_JAKES = 2
};

/**
 * Watterson tap parameters (device-side).
 */
struct WattersonTapDevice {
    int delay_samples;
    float amplitude;
    float doppler_spread_hz;
    int spectrum_type;
    bool is_rician;
    float k_factor_linear;
};

/**
 * Initialize cuRAND states for fading generation.
 */
__global__ void init_watterson_rng(
    curandState* states,
    unsigned long seed,
    int n_taps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_taps) return;

    curand_init(seed, idx, 0, &states[idx]);
}

/**
 * Generate fading coefficients for all taps.
 *
 * Each tap gets an independent complex Gaussian sample that will be
 * filtered by its Doppler filter in the time domain.
 */
__global__ void generate_tap_fading(
    curandState* rng_states,
    const WattersonTapDevice* taps,
    const float* doppler_filters,      // [n_taps * filter_len]
    const cuFloatComplex* noise_buffers,  // [n_taps * filter_len]
    cuFloatComplex* new_noise,         // [n_taps] - one new sample per tap
    cuFloatComplex* fading_out,        // [n_taps] - filtered fading coefficients
    int n_taps,
    int filter_len
) {
    int tap = blockIdx.x * blockDim.x + threadIdx.x;
    if (tap >= n_taps) return;

    // Generate new complex Gaussian noise for this tap
    float2 gauss = curand_normal2(&rng_states[tap]);
    new_noise[tap] = make_cuFloatComplex(gauss.x * INV_SQRT2, gauss.y * INV_SQRT2);

    // Convolve noise buffer with Doppler filter
    const cuFloatComplex* noise_buf = noise_buffers + tap * filter_len;
    const float* filter = doppler_filters + tap * filter_len;

    float sum_re = 0.0f, sum_im = 0.0f;
    for (int k = 0; k < filter_len; k++) {
        float h = filter[k];
        sum_re += noise_buf[k].x * h;
        sum_im += noise_buf[k].y * h;
    }

    // Apply tap amplitude and Rician adjustment
    float scatter_coeff = 1.0f;
    float direct_re = 0.0f;

    if (taps[tap].is_rician && taps[tap].k_factor_linear > 0.0f) {
        float k = taps[tap].k_factor_linear;
        scatter_coeff = sqrtf(1.0f / (1.0f + k));
        direct_re = sqrtf(k / (1.0f + k)) * taps[tap].amplitude;
    }

    float amp = taps[tap].amplitude;
    fading_out[tap] = make_cuFloatComplex(
        direct_re + scatter_coeff * sum_re * amp,
        scatter_coeff * sum_im * amp
    );
}

/**
 * Shift noise buffers and insert new samples.
 */
__global__ void update_noise_buffers(
    cuFloatComplex* noise_buffers,  // [n_taps * filter_len]
    const cuFloatComplex* new_noise,
    int n_taps,
    int filter_len
) {
    int tap = blockIdx.x * blockDim.x + threadIdx.x;
    if (tap >= n_taps) return;

    cuFloatComplex* buf = noise_buffers + tap * filter_len;

    // Shift buffer left
    for (int k = 0; k < filter_len - 1; k++) {
        buf[k] = buf[k + 1];
    }

    // Insert new sample at end
    buf[filter_len - 1] = new_noise[tap];
}

/**
 * Process samples through Watterson TDL with interpolated fading.
 *
 * Uses gain interpolation across the block to avoid discontinuities.
 */
__global__ void process_watterson_tdl(
    const cuFloatComplex* input,       // Input samples [n_samples]
    cuFloatComplex* output,            // Output samples [n_samples]
    const cuFloatComplex* delay_bufs,  // Delay buffers [n_taps * max_delay]
    const WattersonTapDevice* taps,
    const cuFloatComplex* old_gains,   // Previous block fading [n_taps]
    const cuFloatComplex* new_gains,   // Current block fading [n_taps]
    int n_samples,
    int n_taps,
    int max_delay
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= n_samples) return;

    // Interpolation weight (0 at start, 1 at end)
    float t = (float)sample_idx / (float)n_samples;

    cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);

    for (int tap = 0; tap < n_taps; tap++) {
        int delay = taps[tap].delay_samples;

        // Get delayed sample
        cuFloatComplex delayed;
        int src_idx = sample_idx - delay;
        if (src_idx >= 0) {
            delayed = input[src_idx];
        } else {
            // Read from delay buffer
            int buf_idx = max_delay + src_idx;  // Negative offset from end
            if (buf_idx >= 0 && buf_idx < max_delay) {
                delayed = delay_bufs[tap * max_delay + buf_idx];
            } else {
                delayed = make_cuFloatComplex(0.0f, 0.0f);
            }
        }

        // Interpolate fading gain
        cuFloatComplex g_old = old_gains[tap];
        cuFloatComplex g_new = new_gains[tap];
        cuFloatComplex gain = make_cuFloatComplex(
            g_old.x * (1.0f - t) + g_new.x * t,
            g_old.y * (1.0f - t) + g_new.y * t
        );

        // Apply gain: out += delayed * gain
        sum.x += delayed.x * gain.x - delayed.y * gain.y;
        sum.y += delayed.x * gain.y + delayed.y * gain.x;
    }

    output[sample_idx] = sum;
}

/**
 * Update delay buffers for next block.
 */
__global__ void update_delay_buffers(
    const cuFloatComplex* input,
    cuFloatComplex* delay_bufs,  // [n_taps * max_delay]
    int n_samples,
    int n_taps,
    int max_delay
) {
    int tap = blockIdx.x * blockDim.x + threadIdx.x;
    if (tap >= n_taps) return;

    cuFloatComplex* buf = delay_bufs + tap * max_delay;

    // Copy last max_delay samples from input to delay buffer
    int copy_start = n_samples - max_delay;
    if (copy_start < 0) {
        // Input shorter than max_delay - shift existing and append
        int shift = n_samples;
        for (int i = 0; i < max_delay - shift; i++) {
            buf[i] = buf[i + shift];
        }
        for (int i = 0; i < n_samples; i++) {
            buf[max_delay - shift + i] = input[i];
        }
    } else {
        // Input longer than max_delay - take last max_delay samples
        for (int i = 0; i < max_delay; i++) {
            buf[i] = input[copy_start + i];
        }
    }
}

// Host wrapper structures and functions

extern "C" {

/**
 * Watterson channel GPU state.
 */
struct WattersonGPUState {
    // Device memory
    curandState* rng_states;
    WattersonTapDevice* taps_dev;
    float* doppler_filters_dev;
    cuFloatComplex* noise_buffers_dev;
    cuFloatComplex* new_noise_dev;
    cuFloatComplex* delay_buffers_dev;
    cuFloatComplex* old_gains_dev;
    cuFloatComplex* new_gains_dev;
    cuFloatComplex* input_dev;
    cuFloatComplex* output_dev;

    // Host-side copies for pybind
    WattersonTapDevice* taps_host;

    // Parameters
    int n_taps;
    int max_taps;
    int max_delay;
    int max_samples;
    int filter_len;
    float sample_rate;

    bool initialized;
};

/**
 * Compute Doppler filter coefficients on host.
 */
static void compute_doppler_filter(
    float* filter,
    int filter_len,
    float doppler_spread_hz,
    int spectrum_type,
    float update_rate
) {
    int half = filter_len / 2;
    float f_norm = doppler_spread_hz / update_rate;

    float sum_sq = 0.0f;

    for (int i = 0; i < filter_len; i++) {
        float t = (float)(i - half);
        float h = 0.0f;

        if (spectrum_type == DOPPLER_GAUSSIAN) {
            float sigma = (f_norm > 0.0f) ? fmaxf(1.0f, 1.0f / (TWO_PI * f_norm)) : filter_len / 4.0f;
            h = expf(-0.5f * (t / sigma) * (t / sigma));
        } else if (spectrum_type == DOPPLER_FLAT) {
            float arg = 2.0f * f_norm * t;
            h = (fabsf(arg) < 1e-10f) ? 1.0f : sinf(PI * arg) / (PI * arg);
        } else if (spectrum_type == DOPPLER_JAKES) {
            float arg = 2.0f * f_norm * t;
            float sinc = (fabsf(arg) < 1e-10f) ? 1.0f : sinf(PI * arg) / (PI * arg);
            h = sinc * cosf(PI * f_norm * t);
        }

        filter[i] = h;
        sum_sq += h * h;
    }

    // Normalize for unit variance output
    float norm = 1.0f / sqrtf(sum_sq);
    for (int i = 0; i < filter_len; i++) {
        filter[i] *= norm;
    }
}

/**
 * Initialize Watterson GPU processor.
 */
void* init_watterson_gpu(
    float sample_rate,
    int max_taps,
    int max_delay_samples,
    int max_samples_per_block,
    unsigned long seed
) {
    WattersonGPUState* state = new WattersonGPUState;
    state->initialized = false;

    state->sample_rate = sample_rate;
    state->max_taps = max_taps;
    state->max_delay = max_delay_samples;
    state->max_samples = max_samples_per_block;
    state->filter_len = 32;  // Fixed filter length for now
    state->n_taps = 0;

    // Allocate device memory
    cudaError_t err;

    err = cudaMalloc(&state->rng_states, max_taps * sizeof(curandState));
    if (err != cudaSuccess) { delete state; return nullptr; }

    err = cudaMalloc(&state->taps_dev, max_taps * sizeof(WattersonTapDevice));
    if (err != cudaSuccess) goto cleanup_1;

    err = cudaMalloc(&state->doppler_filters_dev, max_taps * state->filter_len * sizeof(float));
    if (err != cudaSuccess) goto cleanup_2;

    err = cudaMalloc(&state->noise_buffers_dev, max_taps * state->filter_len * sizeof(cuFloatComplex));
    if (err != cudaSuccess) goto cleanup_3;

    err = cudaMalloc(&state->new_noise_dev, max_taps * sizeof(cuFloatComplex));
    if (err != cudaSuccess) goto cleanup_4;

    err = cudaMalloc(&state->delay_buffers_dev, max_taps * max_delay_samples * sizeof(cuFloatComplex));
    if (err != cudaSuccess) goto cleanup_5;

    err = cudaMalloc(&state->old_gains_dev, max_taps * sizeof(cuFloatComplex));
    if (err != cudaSuccess) goto cleanup_6;

    err = cudaMalloc(&state->new_gains_dev, max_taps * sizeof(cuFloatComplex));
    if (err != cudaSuccess) goto cleanup_7;

    err = cudaMalloc(&state->input_dev, max_samples_per_block * sizeof(cuFloatComplex));
    if (err != cudaSuccess) goto cleanup_8;

    err = cudaMalloc(&state->output_dev, max_samples_per_block * sizeof(cuFloatComplex));
    if (err != cudaSuccess) goto cleanup_9;

    // Allocate host-side tap storage
    state->taps_host = new WattersonTapDevice[max_taps];

    // Initialize RNG states
    {
        int threads = 256;
        int blocks = (max_taps + threads - 1) / threads;
        init_watterson_rng<<<blocks, threads>>>(state->rng_states, seed, max_taps);
        cudaDeviceSynchronize();
    }

    // Zero-initialize buffers
    cudaMemset(state->noise_buffers_dev, 0, max_taps * state->filter_len * sizeof(cuFloatComplex));
    cudaMemset(state->delay_buffers_dev, 0, max_taps * max_delay_samples * sizeof(cuFloatComplex));
    cudaMemset(state->old_gains_dev, 0, max_taps * sizeof(cuFloatComplex));
    cudaMemset(state->new_gains_dev, 0, max_taps * sizeof(cuFloatComplex));

    state->initialized = true;
    return state;

    // Cleanup on error
cleanup_9: cudaFree(state->input_dev);
cleanup_8: cudaFree(state->new_gains_dev);
cleanup_7: cudaFree(state->old_gains_dev);
cleanup_6: cudaFree(state->delay_buffers_dev);
cleanup_5: cudaFree(state->new_noise_dev);
cleanup_4: cudaFree(state->noise_buffers_dev);
cleanup_3: cudaFree(state->doppler_filters_dev);
cleanup_2: cudaFree(state->taps_dev);
cleanup_1: cudaFree(state->rng_states);
    delete state;
    return nullptr;
}

/**
 * Configure Watterson taps.
 */
int configure_watterson_taps_gpu(
    void* state_ptr,
    const int* delays,
    const float* amplitudes,
    const float* doppler_spreads,
    const int* spectrum_types,
    const int* is_rician,
    const float* k_factors,
    int n_taps,
    float update_rate
) {
    WattersonGPUState* state = (WattersonGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_taps > state->max_taps) return -2;

    state->n_taps = n_taps;

    // Build tap structures on host
    for (int i = 0; i < n_taps; i++) {
        state->taps_host[i].delay_samples = delays[i];
        state->taps_host[i].amplitude = amplitudes[i];
        state->taps_host[i].doppler_spread_hz = doppler_spreads[i];
        state->taps_host[i].spectrum_type = spectrum_types[i];
        state->taps_host[i].is_rician = (is_rician[i] != 0);
        state->taps_host[i].k_factor_linear = k_factors[i];
    }

    // Copy taps to device
    cudaMemcpy(state->taps_dev, state->taps_host,
               n_taps * sizeof(WattersonTapDevice), cudaMemcpyHostToDevice);

    // Compute and upload Doppler filters
    float* filters_host = new float[n_taps * state->filter_len];
    for (int i = 0; i < n_taps; i++) {
        compute_doppler_filter(
            filters_host + i * state->filter_len,
            state->filter_len,
            doppler_spreads[i],
            spectrum_types[i],
            update_rate
        );
    }
    cudaMemcpy(state->doppler_filters_dev, filters_host,
               n_taps * state->filter_len * sizeof(float), cudaMemcpyHostToDevice);
    delete[] filters_host;

    // Initialize noise buffers with complex Gaussian
    cuFloatComplex* noise_init = new cuFloatComplex[n_taps * state->filter_len];
    for (int i = 0; i < n_taps * state->filter_len; i++) {
        float re = (float)rand() / RAND_MAX - 0.5f;
        float im = (float)rand() / RAND_MAX - 0.5f;
        noise_init[i] = make_cuFloatComplex(re * sqrtf(12.0f) * INV_SQRT2,
                                            im * sqrtf(12.0f) * INV_SQRT2);
    }
    cudaMemcpy(state->noise_buffers_dev, noise_init,
               n_taps * state->filter_len * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    delete[] noise_init;

    return 0;
}

/**
 * Process samples through Watterson channel.
 */
int process_watterson_gpu(
    void* state_ptr,
    const float* input_real,
    const float* input_imag,
    int n_samples,
    float* output_real,
    float* output_imag
) {
    WattersonGPUState* state = (WattersonGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_samples > state->max_samples) return -2;
    if (state->n_taps == 0) return -3;

    int n_taps = state->n_taps;
    int threads = 256;

    // Pack input and copy to device
    cuFloatComplex* input_host = new cuFloatComplex[n_samples];
    for (int i = 0; i < n_samples; i++) {
        input_host[i] = make_cuFloatComplex(input_real[i], input_imag[i]);
    }
    cudaMemcpy(state->input_dev, input_host, n_samples * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    delete[] input_host;

    // Save old gains for interpolation
    cudaMemcpy(state->old_gains_dev, state->new_gains_dev,
               n_taps * sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice);

    // Generate new fading coefficients
    int tap_blocks = (n_taps + threads - 1) / threads;
    generate_tap_fading<<<tap_blocks, threads>>>(
        state->rng_states,
        state->taps_dev,
        state->doppler_filters_dev,
        state->noise_buffers_dev,
        state->new_noise_dev,
        state->new_gains_dev,
        n_taps,
        state->filter_len
    );

    // Update noise buffers
    update_noise_buffers<<<tap_blocks, threads>>>(
        state->noise_buffers_dev,
        state->new_noise_dev,
        n_taps,
        state->filter_len
    );

    // Process samples through TDL
    int sample_blocks = (n_samples + threads - 1) / threads;
    process_watterson_tdl<<<sample_blocks, threads>>>(
        state->input_dev,
        state->output_dev,
        state->delay_buffers_dev,
        state->taps_dev,
        state->old_gains_dev,
        state->new_gains_dev,
        n_samples,
        n_taps,
        state->max_delay
    );

    // Update delay buffers
    update_delay_buffers<<<tap_blocks, threads>>>(
        state->input_dev,
        state->delay_buffers_dev,
        n_samples,
        n_taps,
        state->max_delay
    );

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
 * Get current fading gains for visualization.
 */
int get_watterson_gains_gpu(
    void* state_ptr,
    float* gains_real,
    float* gains_imag,
    int max_taps
) {
    WattersonGPUState* state = (WattersonGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;

    int n = (state->n_taps < max_taps) ? state->n_taps : max_taps;

    cuFloatComplex* gains_host = new cuFloatComplex[n];
    cudaMemcpy(gains_host, state->new_gains_dev, n * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        gains_real[i] = gains_host[i].x;
        gains_imag[i] = gains_host[i].y;
    }
    delete[] gains_host;

    return n;
}

/**
 * Reset Watterson channel state.
 */
void reset_watterson_gpu(void* state_ptr) {
    WattersonGPUState* state = (WattersonGPUState*)state_ptr;
    if (!state || !state->initialized) return;

    cudaMemset(state->delay_buffers_dev, 0,
               state->max_taps * state->max_delay * sizeof(cuFloatComplex));
    cudaMemset(state->old_gains_dev, 0, state->max_taps * sizeof(cuFloatComplex));
    cudaMemset(state->new_gains_dev, 0, state->max_taps * sizeof(cuFloatComplex));
}

/**
 * Free Watterson GPU resources.
 */
void free_watterson_gpu(void* state_ptr) {
    WattersonGPUState* state = (WattersonGPUState*)state_ptr;
    if (!state) return;

    if (state->initialized) {
        cudaFree(state->rng_states);
        cudaFree(state->taps_dev);
        cudaFree(state->doppler_filters_dev);
        cudaFree(state->noise_buffers_dev);
        cudaFree(state->new_noise_dev);
        cudaFree(state->delay_buffers_dev);
        cudaFree(state->old_gains_dev);
        cudaFree(state->new_gains_dev);
        cudaFree(state->input_dev);
        cudaFree(state->output_dev);

        delete[] state->taps_host;
    }

    delete state;
}

} // extern "C"
