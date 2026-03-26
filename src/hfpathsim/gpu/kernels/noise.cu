/**
 * CUDA kernels for noise generation.
 *
 * Implements GPU-accelerated:
 * - AWGN (Additive White Gaussian Noise)
 * - Atmospheric noise (Hall model with impulsive characteristics)
 * - Impulse noise (man-made interference)
 * - Colored noise (shaped by arbitrary spectrum)
 *
 * Uses cuRAND for efficient parallel random number generation.
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <curand_kernel.h>
#include <cufft.h>
#include <cmath>

#define PI 3.14159265358979323846f
#define TWO_PI 6.28318530717958647692f
#define INV_SQRT2 0.7071067811865476f

/**
 * Initialize cuRAND states.
 */
__global__ void init_noise_rng(
    curandState* states,
    unsigned long seed,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    curand_init(seed, idx, 0, &states[idx]);
}

/**
 * Generate complex AWGN samples.
 *
 * Generates samples with variance = noise_power.
 * Complex Gaussian: (N(0, sigma) + j*N(0, sigma)) where sigma^2 = noise_power/2
 */
__global__ void generate_awgn(
    curandState* states,
    cuFloatComplex* noise,
    float noise_power,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Variance for each component: noise_power / 2
    float sigma = sqrtf(noise_power / 2.0f);

    float2 gauss = curand_normal2(&states[idx]);
    noise[idx] = make_cuFloatComplex(gauss.x * sigma, gauss.y * sigma);
}

/**
 * Generate atmospheric noise using Hall model.
 *
 * Atmospheric noise has impulsive characteristics with heavy-tailed
 * amplitude distribution. We model this using a mixture of Gaussian
 * and impulsive components.
 *
 * Parameters:
 *   vd: Voltage deviation ratio (impulsiveness parameter, 0-1)
 *   noise_power: Total noise power
 */
__global__ void generate_atmospheric_noise(
    curandState* states,
    cuFloatComplex* noise,
    float noise_power,
    float vd,  // Voltage deviation ratio (0 = Gaussian, 1 = very impulsive)
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Generate uniform random to decide if this sample is impulsive
    float u = curand_uniform(&states[idx]);

    // Impulsive probability increases with vd
    float p_impulse = vd * 0.1f;  // Max 10% impulses at vd=1

    float sigma;
    if (u < p_impulse) {
        // Impulsive sample: use log-normal envelope
        float log_sigma = 1.0f + 2.0f * vd;  // Higher vd = wider spread
        float log_mean = logf(sqrtf(noise_power)) + log_sigma * log_sigma;
        float log_val = curand_normal(&states[idx]) * log_sigma + log_mean;
        sigma = expf(log_val);
    } else {
        // Gaussian background
        sigma = sqrtf(noise_power / 2.0f);
    }

    float2 gauss = curand_normal2(&states[idx]);
    noise[idx] = make_cuFloatComplex(gauss.x * sigma, gauss.y * sigma);
}

/**
 * Generate impulse noise (man-made interference).
 *
 * Models periodic or random impulses like power line interference,
 * switching noise, or ignition noise.
 *
 * Parameters:
 *   impulse_rate: Average impulses per sample
 *   impulse_amplitude: Peak amplitude of impulses
 *   noise_floor: Background noise floor
 */
__global__ void generate_impulse_noise(
    curandState* states,
    cuFloatComplex* noise,
    float impulse_rate,
    float impulse_amplitude,
    float noise_floor,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Background Gaussian noise
    float sigma = sqrtf(noise_floor / 2.0f);
    float2 gauss = curand_normal2(&states[idx]);
    float re = gauss.x * sigma;
    float im = gauss.y * sigma;

    // Add impulse with Poisson probability
    float u = curand_uniform(&states[idx]);
    if (u < impulse_rate) {
        // Generate impulse with random phase
        float phase = curand_uniform(&states[idx]) * TWO_PI;
        // Amplitude with some random variation
        float amp = impulse_amplitude * (0.5f + curand_uniform(&states[idx]));
        re += amp * cosf(phase);
        im += amp * sinf(phase);
    }

    noise[idx] = make_cuFloatComplex(re, im);
}

/**
 * Add noise to signal (in-place addition on device).
 */
__global__ void add_noise_to_signal(
    cuFloatComplex* signal,
    const cuFloatComplex* noise,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    signal[idx].x += noise[idx].x;
    signal[idx].y += noise[idx].y;
}

/**
 * Apply spectrum shaping to noise (frequency domain multiplication).
 */
__global__ void shape_noise_spectrum(
    cuFloatComplex* noise_spectrum,
    const float* spectrum_shape,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float shape = spectrum_shape[idx];
    noise_spectrum[idx].x *= shape;
    noise_spectrum[idx].y *= shape;
}

/**
 * Normalize noise to target power.
 */
__global__ void normalize_noise_power(
    cuFloatComplex* noise,
    float current_power,
    float target_power,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float scale = sqrtf(target_power / current_power);
    noise[idx].x *= scale;
    noise[idx].y *= scale;
}

// Host wrapper structures and functions

extern "C" {

/**
 * Noise generator GPU state.
 */
struct NoiseGenGPUState {
    curandState* rng_states;
    cuFloatComplex* noise_buffer;
    cuFloatComplex* spectrum_buffer;
    float* spectrum_shape;
    cufftHandle fft_plan;
    cufftHandle ifft_plan;

    int max_samples;
    float sample_rate;
    bool has_spectrum_shape;
    bool initialized;
};

/**
 * Initialize noise generator.
 */
void* init_noise_gen_gpu(
    float sample_rate,
    int max_samples,
    unsigned long seed
) {
    NoiseGenGPUState* state = new NoiseGenGPUState;
    state->initialized = false;
    state->max_samples = max_samples;
    state->sample_rate = sample_rate;
    state->has_spectrum_shape = false;

    cudaError_t err;

    err = cudaMalloc(&state->rng_states, max_samples * sizeof(curandState));
    if (err != cudaSuccess) { delete state; return nullptr; }

    err = cudaMalloc(&state->noise_buffer, max_samples * sizeof(cuFloatComplex));
    if (err != cudaSuccess) goto cleanup_1;

    err = cudaMalloc(&state->spectrum_buffer, max_samples * sizeof(cuFloatComplex));
    if (err != cudaSuccess) goto cleanup_2;

    err = cudaMalloc(&state->spectrum_shape, max_samples * sizeof(float));
    if (err != cudaSuccess) goto cleanup_3;

    // Create FFT plans for colored noise
    if (cufftPlan1d(&state->fft_plan, max_samples, CUFFT_C2C, 1) != CUFFT_SUCCESS)
        goto cleanup_4;

    if (cufftPlan1d(&state->ifft_plan, max_samples, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
        cufftDestroy(state->fft_plan);
        goto cleanup_4;
    }

    // Initialize RNG states
    {
        int threads = 256;
        int blocks = (max_samples + threads - 1) / threads;
        init_noise_rng<<<blocks, threads>>>(state->rng_states, seed, max_samples);
        cudaDeviceSynchronize();
    }

    state->initialized = true;
    return state;

cleanup_4: cudaFree(state->spectrum_shape);
cleanup_3: cudaFree(state->spectrum_buffer);
cleanup_2: cudaFree(state->noise_buffer);
cleanup_1: cudaFree(state->rng_states);
    delete state;
    return nullptr;
}

/**
 * Generate AWGN samples.
 */
int generate_awgn_gpu(
    void* state_ptr,
    float noise_power,
    int n_samples,
    float* noise_real,
    float* noise_imag
) {
    NoiseGenGPUState* state = (NoiseGenGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_samples > state->max_samples) return -2;

    int threads = 256;
    int blocks = (n_samples + threads - 1) / threads;

    generate_awgn<<<blocks, threads>>>(
        state->rng_states,
        state->noise_buffer,
        noise_power,
        n_samples
    );

    cudaDeviceSynchronize();

    // Copy back
    cuFloatComplex* host = new cuFloatComplex[n_samples];
    cudaMemcpy(host, state->noise_buffer, n_samples * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n_samples; i++) {
        noise_real[i] = host[i].x;
        noise_imag[i] = host[i].y;
    }
    delete[] host;

    return 0;
}

/**
 * Generate atmospheric noise samples.
 */
int generate_atmospheric_gpu(
    void* state_ptr,
    float noise_power,
    float vd,
    int n_samples,
    float* noise_real,
    float* noise_imag
) {
    NoiseGenGPUState* state = (NoiseGenGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_samples > state->max_samples) return -2;

    int threads = 256;
    int blocks = (n_samples + threads - 1) / threads;

    generate_atmospheric_noise<<<blocks, threads>>>(
        state->rng_states,
        state->noise_buffer,
        noise_power,
        vd,
        n_samples
    );

    cudaDeviceSynchronize();

    // Copy back
    cuFloatComplex* host = new cuFloatComplex[n_samples];
    cudaMemcpy(host, state->noise_buffer, n_samples * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n_samples; i++) {
        noise_real[i] = host[i].x;
        noise_imag[i] = host[i].y;
    }
    delete[] host;

    return 0;
}

/**
 * Generate impulse noise samples.
 */
int generate_impulse_gpu(
    void* state_ptr,
    float impulse_rate,
    float impulse_amplitude,
    float noise_floor,
    int n_samples,
    float* noise_real,
    float* noise_imag
) {
    NoiseGenGPUState* state = (NoiseGenGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_samples > state->max_samples) return -2;

    int threads = 256;
    int blocks = (n_samples + threads - 1) / threads;

    generate_impulse_noise<<<blocks, threads>>>(
        state->rng_states,
        state->noise_buffer,
        impulse_rate,
        impulse_amplitude,
        noise_floor,
        n_samples
    );

    cudaDeviceSynchronize();

    // Copy back
    cuFloatComplex* host = new cuFloatComplex[n_samples];
    cudaMemcpy(host, state->noise_buffer, n_samples * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n_samples; i++) {
        noise_real[i] = host[i].x;
        noise_imag[i] = host[i].y;
    }
    delete[] host;

    return 0;
}

/**
 * Set spectrum shape for colored noise.
 */
int set_noise_spectrum_shape_gpu(
    void* state_ptr,
    const float* spectrum_shape,
    int n_bins
) {
    NoiseGenGPUState* state = (NoiseGenGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_bins > state->max_samples) return -2;

    cudaMemcpy(state->spectrum_shape, spectrum_shape,
               n_bins * sizeof(float), cudaMemcpyHostToDevice);
    state->has_spectrum_shape = true;

    return 0;
}

/**
 * Generate colored (shaped spectrum) noise.
 */
int generate_colored_noise_gpu(
    void* state_ptr,
    float noise_power,
    int n_samples,
    float* noise_real,
    float* noise_imag
) {
    NoiseGenGPUState* state = (NoiseGenGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_samples > state->max_samples) return -2;
    if (!state->has_spectrum_shape) return -3;

    int threads = 256;
    int blocks = (n_samples + threads - 1) / threads;

    // Generate white noise
    generate_awgn<<<blocks, threads>>>(
        state->rng_states,
        state->noise_buffer,
        1.0f,  // Unit power, will normalize later
        n_samples
    );

    // FFT to frequency domain
    cufftExecC2C(state->fft_plan, state->noise_buffer, state->spectrum_buffer, CUFFT_FORWARD);

    // Apply spectrum shaping
    shape_noise_spectrum<<<blocks, threads>>>(
        state->spectrum_buffer,
        state->spectrum_shape,
        n_samples
    );

    // IFFT back to time domain
    cufftExecC2C(state->ifft_plan, state->spectrum_buffer, state->noise_buffer, CUFFT_INVERSE);

    // Normalize
    float scale = 1.0f / sqrtf((float)n_samples);
    cudaDeviceSynchronize();

    // Copy and scale
    cuFloatComplex* host = new cuFloatComplex[n_samples];
    cudaMemcpy(host, state->noise_buffer, n_samples * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    // Compute current power and normalize to target
    float power_sum = 0.0f;
    for (int i = 0; i < n_samples; i++) {
        float re = host[i].x * scale;
        float im = host[i].y * scale;
        power_sum += re * re + im * im;
    }
    float current_power = power_sum / n_samples;
    float norm_scale = sqrtf(noise_power / current_power);

    for (int i = 0; i < n_samples; i++) {
        noise_real[i] = host[i].x * scale * norm_scale;
        noise_imag[i] = host[i].y * scale * norm_scale;
    }
    delete[] host;

    return 0;
}

/**
 * Add noise to signal (convenience function).
 */
int add_noise_gpu(
    void* state_ptr,
    float* signal_real,
    float* signal_imag,
    int noise_type,  // 0=AWGN, 1=atmospheric, 2=impulse
    float param1,    // noise_power or impulse_rate
    float param2,    // vd or impulse_amplitude
    float param3,    // unused or noise_floor
    int n_samples
) {
    NoiseGenGPUState* state = (NoiseGenGPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_samples > state->max_samples) return -2;

    int threads = 256;
    int blocks = (n_samples + threads - 1) / threads;

    // Generate noise based on type
    switch (noise_type) {
        case 0:  // AWGN
            generate_awgn<<<blocks, threads>>>(
                state->rng_states, state->noise_buffer, param1, n_samples);
            break;
        case 1:  // Atmospheric
            generate_atmospheric_noise<<<blocks, threads>>>(
                state->rng_states, state->noise_buffer, param1, param2, n_samples);
            break;
        case 2:  // Impulse
            generate_impulse_noise<<<blocks, threads>>>(
                state->rng_states, state->noise_buffer, param1, param2, param3, n_samples);
            break;
        default:
            return -3;
    }

    cudaDeviceSynchronize();

    // Add to signal (on host for simplicity)
    cuFloatComplex* host = new cuFloatComplex[n_samples];
    cudaMemcpy(host, state->noise_buffer, n_samples * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n_samples; i++) {
        signal_real[i] += host[i].x;
        signal_imag[i] += host[i].y;
    }
    delete[] host;

    return 0;
}

/**
 * Reset RNG state (for reproducibility).
 */
void reset_noise_gen_gpu(void* state_ptr, unsigned long seed) {
    NoiseGenGPUState* state = (NoiseGenGPUState*)state_ptr;
    if (!state || !state->initialized) return;

    int threads = 256;
    int blocks = (state->max_samples + threads - 1) / threads;
    init_noise_rng<<<blocks, threads>>>(state->rng_states, seed, state->max_samples);
    cudaDeviceSynchronize();
}

/**
 * Free noise generator resources.
 */
void free_noise_gen_gpu(void* state_ptr) {
    NoiseGenGPUState* state = (NoiseGenGPUState*)state_ptr;
    if (!state) return;

    if (state->initialized) {
        cudaFree(state->rng_states);
        cudaFree(state->noise_buffer);
        cudaFree(state->spectrum_buffer);
        cudaFree(state->spectrum_shape);
        cufftDestroy(state->fft_plan);
        cufftDestroy(state->ifft_plan);
    }

    delete state;
}

} // extern "C"
