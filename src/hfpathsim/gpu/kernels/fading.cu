/**
 * CUDA kernels for Gaussian scatter fading model.
 *
 * Implements time-varying fading based on ITU-R F.1487 channel statistics.
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <curand_kernel.h>
#include <math.h>

#define PI 3.14159265358979323846

/**
 * Initialize cuRAND states for fading generation.
 */
__global__ void init_rng_states(
    curandState* states,
    unsigned long seed,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    curand_init(seed, idx, 0, &states[idx]);
}

/**
 * Generate complex Gaussian noise for fading.
 */
__global__ void generate_gaussian_noise(
    curandState* states,
    cuFloatComplex* noise,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Generate two independent Gaussian samples
    float2 gauss = curand_normal2(&states[idx]);

    // Complex Gaussian: (N(0,1) + j*N(0,1)) / sqrt(2)
    noise[idx] = make_cuFloatComplex(
        gauss.x * 0.7071067811865476f,
        gauss.y * 0.7071067811865476f
    );
}

/**
 * Apply Gaussian Doppler spectrum filter.
 *
 * Shapes the noise spectrum with a Gaussian profile centered at DC.
 */
__global__ void apply_doppler_filter(
    cuFloatComplex* spectrum,    // Noise spectrum [N]
    const float* freq,           // Frequency axis [N]
    float doppler_spread,        // Two-sided Doppler spread (Hz)
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Gaussian Doppler spectrum
    float sigma = doppler_spread / 2.35482f;  // FWHM to sigma
    float filter = expf(-0.5f * powf(freq[idx] / sigma, 2));

    spectrum[idx] = make_cuFloatComplex(
        cuCrealf(spectrum[idx]) * filter,
        cuCimagf(spectrum[idx]) * filter
    );
}

/**
 * Apply exponential delay spread filter.
 *
 * Creates frequency-selective fading from multipath delay spread.
 */
__global__ void apply_delay_filter(
    cuFloatComplex* H,           // Transfer function [N]
    float delay_spread_sec,      // Delay spread in seconds
    float sample_rate,           // Sample rate in Hz
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Compute normalized frequency
    float f_norm = (float)idx / (float)N;
    if (f_norm > 0.5f) f_norm -= 1.0f;
    float freq = f_norm * sample_rate;

    // Exponential delay profile in frequency domain
    // H_delay(f) = 1 / (1 + j*2*pi*f*tau)
    float omega_tau = 2.0f * PI * freq * delay_spread_sec;
    float denom = 1.0f + omega_tau * omega_tau;

    cuFloatComplex delay_tf = make_cuFloatComplex(
        1.0f / denom,
        -omega_tau / denom
    );

    // Apply to transfer function
    float re = cuCrealf(H[idx]) * cuCrealf(delay_tf) -
               cuCimagf(H[idx]) * cuCimagf(delay_tf);
    float im = cuCrealf(H[idx]) * cuCimagf(delay_tf) +
               cuCimagf(H[idx]) * cuCrealf(delay_tf);

    H[idx] = make_cuFloatComplex(re, im);
}

/**
 * Combine multiple propagation modes.
 *
 * Sums contributions from different ionospheric paths with
 * relative delays and amplitudes.
 */
__global__ void combine_modes(
    const cuFloatComplex* H_base,    // Base transfer function [N]
    cuFloatComplex* H_total,          // Output combined TF [N]
    const float* mode_amplitudes,     // Relative amplitudes [n_modes]
    const float* mode_delays_sec,     // Mode delays [n_modes]
    const float* freq,                // Frequency axis [N]
    int N,
    int n_modes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);

    for (int m = 0; m < n_modes; m++) {
        // Phase shift from mode delay
        float phase = -2.0f * PI * freq[idx] * mode_delays_sec[m];
        float cos_p = cosf(phase);
        float sin_p = sinf(phase);

        // Apply amplitude and phase to base TF
        float amp = mode_amplitudes[m];
        cuFloatComplex shifted = make_cuFloatComplex(
            amp * (cuCrealf(H_base[idx]) * cos_p - cuCimagf(H_base[idx]) * sin_p),
            amp * (cuCrealf(H_base[idx]) * sin_p + cuCimagf(H_base[idx]) * cos_p)
        );

        sum = make_cuFloatComplex(
            cuCrealf(sum) + cuCrealf(shifted),
            cuCimagf(sum) + cuCimagf(shifted)
        );
    }

    H_total[idx] = sum;
}

/**
 * Compute scattering function S(tau, nu).
 *
 * The scattering function describes the power distribution in
 * the delay-Doppler domain.
 */
__global__ void compute_scattering_function(
    float* S,                    // Output scattering function [N_delay x N_doppler]
    const float* delay_axis,     // Delay axis [N_delay]
    const float* doppler_axis,   // Doppler axis [N_doppler]
    float delay_spread,          // Delay spread parameter
    float doppler_spread,        // Doppler spread parameter
    int N_delay,
    int N_doppler
) {
    int d_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int f_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (d_idx >= N_delay || f_idx >= N_doppler) return;

    float tau = delay_axis[d_idx];
    float nu = doppler_axis[f_idx];

    // Separable scattering function:
    // S(tau, nu) = P_delay(tau) * P_doppler(nu)

    // Exponential delay profile
    float P_delay = expf(-tau / delay_spread);
    if (tau < 0) P_delay = 0.0f;

    // Gaussian Doppler profile
    float P_doppler = expf(-0.5f * powf(nu / doppler_spread, 2));

    int out_idx = f_idx * N_delay + d_idx;
    S[out_idx] = P_delay * P_doppler;
}

/**
 * Normalize scattering function to unit power.
 */
__global__ void normalize_scattering(
    float* S,
    float max_val,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    S[idx] /= max_val;
}

// Host functions

extern "C" {

/**
 * Fading generator state structure.
 */
struct FadingState {
    curandState* rng_states;
    cuFloatComplex* noise_buffer;
    cuFloatComplex* H_faded;
    float* freq_axis;
    int N;
    bool initialized;
};

/**
 * Initialize fading generator.
 */
void* init_fading_generator(int N, unsigned long seed) {
    FadingState* state = new FadingState;
    state->N = N;
    state->initialized = false;

    // Allocate GPU memory
    cudaMalloc(&state->rng_states, N * sizeof(curandState));
    cudaMalloc(&state->noise_buffer, N * sizeof(cuFloatComplex));
    cudaMalloc(&state->H_faded, N * sizeof(cuFloatComplex));
    cudaMalloc(&state->freq_axis, N * sizeof(float));

    // Initialize RNG
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    init_rng_states<<<blocks, threads>>>(state->rng_states, seed, N);

    cudaDeviceSynchronize();
    state->initialized = true;

    return state;
}

/**
 * Generate faded transfer function.
 */
int generate_faded_channel(
    void* state_ptr,
    const float* H_base_real,
    const float* H_base_imag,
    float doppler_spread,
    float delay_spread,
    float sample_rate,
    float* H_out_real,
    float* H_out_imag
) {
    FadingState* state = (FadingState*)state_ptr;
    if (!state || !state->initialized) return -1;

    int N = state->N;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Copy base TF to device
    cuFloatComplex* H_base_dev;
    cudaMalloc(&H_base_dev, N * sizeof(cuFloatComplex));

    // Interleave real/imag
    cuFloatComplex* H_temp = new cuFloatComplex[N];
    for (int i = 0; i < N; i++) {
        H_temp[i] = make_cuFloatComplex(H_base_real[i], H_base_imag[i]);
    }
    cudaMemcpy(H_base_dev, H_temp, N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    delete[] H_temp;

    // Generate noise
    generate_gaussian_noise<<<blocks, threads>>>(
        state->rng_states, state->noise_buffer, N
    );

    // Apply fading
    cudaMemcpy(state->H_faded, H_base_dev, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice);

    apply_delay_filter<<<blocks, threads>>>(
        state->H_faded, delay_spread, sample_rate, N
    );

    // Copy result back
    cuFloatComplex* H_result = new cuFloatComplex[N];
    cudaMemcpy(H_result, state->H_faded, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        H_out_real[i] = cuCrealf(H_result[i]);
        H_out_imag[i] = cuCimagf(H_result[i]);
    }

    delete[] H_result;
    cudaFree(H_base_dev);

    return 0;
}

/**
 * Free fading generator resources.
 */
void free_fading_generator(void* state_ptr) {
    FadingState* state = (FadingState*)state_ptr;
    if (!state) return;

    cudaFree(state->rng_states);
    cudaFree(state->noise_buffer);
    cudaFree(state->H_faded);
    cudaFree(state->freq_axis);

    delete state;
}

} // extern "C"
