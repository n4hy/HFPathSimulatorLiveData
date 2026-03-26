/**
 * CUDA kernels for Vogler-Hoffmeyer RF processing chain.
 *
 * Implements GPU-accelerated:
 * - Polyphase resampling (upsample/downsample)
 * - RF mixing (up/down conversion)
 * - Tapped delay line with time-varying fading
 *
 * Designed for real-time HF channel simulation at MHz sample rates.
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <curand_kernel.h>
#include <math.h>

#define PI 3.14159265358979323846f
#define TWO_PI 6.28318530717958647692f

// ============================================================================
// Polyphase Resampler Kernels
// ============================================================================

/**
 * Polyphase upsampler kernel.
 * Efficiently upsamples by integer factor using polyphase filter bank.
 *
 * Each thread handles one output sample.
 */
__global__ void polyphase_upsample(
    const cuFloatComplex* __restrict__ input,
    cuFloatComplex* __restrict__ output,
    const float* __restrict__ filter,  // Prototype lowpass filter [filter_len]
    int input_len,
    int output_len,
    int upsample_factor,
    int filter_len,
    int taps_per_phase  // filter_len / upsample_factor
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= output_len) return;

    // Determine which phase and input sample
    int phase = out_idx % upsample_factor;
    int in_base = out_idx / upsample_factor;
    int half_taps = taps_per_phase / 2;
    int filter_center = filter_len / 2;

    // Accumulate filtered output
    float re = 0.0f, im = 0.0f;

    // Polyphase decomposition with proper center alignment
    for (int k = -half_taps; k <= half_taps; k++) {
        int in_idx = in_base - k;
        if (in_idx >= 0 && in_idx < input_len) {
            // Filter coefficient - center-aligned for this phase
            int filt_idx = filter_center + k * upsample_factor + phase;
            if (filt_idx >= 0 && filt_idx < filter_len) {
                float h = filter[filt_idx] * upsample_factor;  // Gain correction
                re += cuCrealf(input[in_idx]) * h;
                im += cuCimagf(input[in_idx]) * h;
            }
        }
    }

    output[out_idx] = make_cuFloatComplex(re, im);
}

/**
 * Polyphase downsampler kernel.
 * Efficiently downsamples by integer factor with anti-aliasing.
 *
 * Each thread handles one output sample.
 */
__global__ void polyphase_downsample(
    const cuFloatComplex* __restrict__ input,
    cuFloatComplex* __restrict__ output,
    const float* __restrict__ filter,  // Anti-aliasing lowpass [filter_len]
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
            re += cuCrealf(input[in_idx]) * h;
            im += cuCimagf(input[in_idx]) * h;
        }
    }

    output[out_idx] = make_cuFloatComplex(re, im);
}

// ============================================================================
// RF Mixer Kernels
// ============================================================================

/**
 * Mix signal up to RF carrier frequency.
 * output[n] = input[n] * exp(j * 2 * pi * f_carrier * t[n])
 */
__global__ void mix_up_to_rf(
    const cuFloatComplex* __restrict__ input,
    cuFloatComplex* __restrict__ output,
    float carrier_freq_hz,
    float sample_rate_hz,
    float phase_offset,  // Initial phase for continuity
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float t = (float)idx / sample_rate_hz;
    float phase = TWO_PI * carrier_freq_hz * t + phase_offset;

    float cos_p = cosf(phase);
    float sin_p = sinf(phase);

    cuFloatComplex in = input[idx];
    output[idx] = make_cuFloatComplex(
        cuCrealf(in) * cos_p - cuCimagf(in) * sin_p,
        cuCrealf(in) * sin_p + cuCimagf(in) * cos_p
    );
}

/**
 * Mix signal down from RF carrier frequency.
 * output[n] = input[n] * exp(-j * 2 * pi * f_carrier * t[n])
 */
__global__ void mix_down_from_rf(
    const cuFloatComplex* __restrict__ input,
    cuFloatComplex* __restrict__ output,
    float carrier_freq_hz,
    float sample_rate_hz,
    float phase_offset,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float t = (float)idx / sample_rate_hz;
    float phase = TWO_PI * carrier_freq_hz * t + phase_offset;

    float cos_p = cosf(phase);
    float sin_p = -sinf(phase);  // Negative for down-conversion

    cuFloatComplex in = input[idx];
    output[idx] = make_cuFloatComplex(
        cuCrealf(in) * cos_p - cuCimagf(in) * sin_p,
        cuCrealf(in) * sin_p + cuCimagf(in) * cos_p
    );
}

// ============================================================================
// Tapped Delay Line Channel Kernel
// ============================================================================

/**
 * Process samples through tapped delay line with time-varying fading.
 *
 * Implements Vogler-Hoffmeyer channel model:
 * - Multiple taps with different delays
 * - AR(1) fading process per tap (Gaussian correlation)
 * - Doppler shift per tap
 * - Rician (K-factor) or Rayleigh fading
 *
 * Each thread block processes a chunk of samples.
 * Uses shared memory for delay line buffer.
 */
__global__ void tdl_channel_process(
    const cuFloatComplex* __restrict__ input,
    cuFloatComplex* __restrict__ output,
    cuFloatComplex* __restrict__ tap_states,  // AR(1) state per tap [n_taps]
    curandState* __restrict__ rng_states,     // RNG state per tap [n_taps]
    const int* __restrict__ tap_delays,       // Delay in samples [n_taps]
    const float* __restrict__ tap_amplitudes, // Tap amplitude [n_taps]
    const float* __restrict__ tap_doppler_hz, // Doppler shift per tap [n_taps]
    float rho,                // AR(1) correlation coefficient
    float innovation_coeff,   // sqrt(1 - rho^2)
    float k_factor,           // Rician K-factor (0 = Rayleigh)
    float sample_rate_hz,
    float time_offset,        // Starting time for phase continuity
    int n_samples,
    int n_taps,
    int max_delay             // Maximum tap delay for buffer sizing
) {
    extern __shared__ cuFloatComplex delay_buffer[];

    int tid = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x;
    int global_idx = block_start + tid;

    // Load shared delay buffer: history + current block samples
    // Structure: [history: max_delay samples] [current: blockDim.x samples]

    // Load history (samples before this block)
    if (tid < max_delay) {
        int hist_idx = block_start - max_delay + tid;
        if (hist_idx >= 0) {
            delay_buffer[tid] = input[hist_idx];
        } else {
            delay_buffer[tid] = make_cuFloatComplex(0.0f, 0.0f);
        }
    }

    // Load current block samples into shared memory
    if (global_idx < n_samples) {
        delay_buffer[max_delay + tid] = input[global_idx];
    } else {
        delay_buffer[max_delay + tid] = make_cuFloatComplex(0.0f, 0.0f);
    }
    __syncthreads();

    if (global_idx >= n_samples) return;

    // Current time for Doppler phase
    float t = time_offset + (float)global_idx / sample_rate_hz;

    // Direct and scatter coefficients from K-factor
    float direct_coeff = sqrtf(k_factor / (k_factor + 1.0f));
    float scatter_coeff = sqrtf(1.0f / (k_factor + 1.0f));

    // Accumulate output from all taps
    float out_re = 0.0f, out_im = 0.0f;

    for (int tap = 0; tap < n_taps; tap++) {
        int delay = tap_delays[tap];
        float amplitude = tap_amplitudes[tap];
        float doppler = tap_doppler_hz[tap];

        // Get delayed sample
        cuFloatComplex delayed_sample;
        int delayed_idx = global_idx - delay;
        if (delayed_idx >= 0 && delayed_idx < n_samples) {
            // Calculate offset relative to what's in shared memory
            int shared_idx = delayed_idx - (block_start - max_delay);
            if (shared_idx >= 0 && shared_idx < max_delay + (int)blockDim.x) {
                // Sample is in shared memory
                delayed_sample = delay_buffer[shared_idx];
            } else {
                // Sample in global memory (rare edge case)
                delayed_sample = input[delayed_idx];
            }
        } else {
            delayed_sample = make_cuFloatComplex(0.0f, 0.0f);
        }

        // Update AR(1) fading state for this tap
        // Use first thread of block to update tap states (serialize per block)
        cuFloatComplex fading;
        if (tid == 0) {
            cuFloatComplex state = tap_states[tap];

            // Generate complex Gaussian innovation
            float2 gauss = curand_normal2(&rng_states[tap]);
            cuFloatComplex z = make_cuFloatComplex(
                gauss.x * 0.7071067811865476f,
                gauss.y * 0.7071067811865476f
            );

            // AR(1) update: C[n] = rho * C[n-1] + sqrt(1-rho^2) * z
            state = make_cuFloatComplex(
                rho * cuCrealf(state) + innovation_coeff * cuCrealf(z),
                rho * cuCimagf(state) + innovation_coeff * cuCimagf(z)
            );

            tap_states[tap] = state;
        }
        __syncthreads();

        fading = tap_states[tap];

        // Rician fading: direct path on first tap
        cuFloatComplex fading_gain;
        if (tap == 0 && k_factor > 0.0f) {
            fading_gain = make_cuFloatComplex(
                direct_coeff + scatter_coeff * cuCrealf(fading),
                scatter_coeff * cuCimagf(fading)
            );
        } else {
            fading_gain = make_cuFloatComplex(
                scatter_coeff * cuCrealf(fading),
                scatter_coeff * cuCimagf(fading)
            );
        }

        // Doppler phase shift
        float doppler_phase = TWO_PI * doppler * t;
        float cos_d = cosf(doppler_phase);
        float sin_d = sinf(doppler_phase);

        // Combined tap gain: amplitude * fading * exp(j*doppler_phase)
        float gain_re = amplitude * (cuCrealf(fading_gain) * cos_d - cuCimagf(fading_gain) * sin_d);
        float gain_im = amplitude * (cuCrealf(fading_gain) * sin_d + cuCimagf(fading_gain) * cos_d);

        // Apply to delayed sample
        out_re += cuCrealf(delayed_sample) * gain_re - cuCimagf(delayed_sample) * gain_im;
        out_im += cuCrealf(delayed_sample) * gain_im + cuCimagf(delayed_sample) * gain_re;
    }

    output[global_idx] = make_cuFloatComplex(out_re, out_im);
}

/**
 * Initialize RNG states for TDL taps.
 */
__global__ void init_tdl_rng(
    curandState* states,
    unsigned long seed,
    int n_taps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_taps) return;

    curand_init(seed, idx, 0, &states[idx]);
}

// ============================================================================
// Complete RF Chain Processing
// ============================================================================

/**
 * Pre-generate AR(1) fading coefficients for all samples.
 * This allows proper per-sample fading updates.
 */
__global__ void generate_fading_coefficients(
    cuFloatComplex* fading_out,       // [n_samples * n_taps]
    cuFloatComplex* tap_states,       // [n_taps] - updated in place
    curandState* rng_states,          // [n_taps]
    float rho,
    float innovation_coeff,
    int n_samples,
    int n_taps
) {
    int tap = blockIdx.x;
    if (tap >= n_taps) return;

    // Each block handles one tap sequentially
    cuFloatComplex state = tap_states[tap];
    curandState rng = rng_states[tap];

    for (int n = 0; n < n_samples; n++) {
        // Generate complex Gaussian innovation
        float2 gauss = curand_normal2(&rng);
        cuFloatComplex z = make_cuFloatComplex(
            gauss.x * 0.7071067811865476f,
            gauss.y * 0.7071067811865476f
        );

        // AR(1) update
        state = make_cuFloatComplex(
            rho * cuCrealf(state) + innovation_coeff * cuCrealf(z),
            rho * cuCimagf(state) + innovation_coeff * cuCimagf(z)
        );

        // Store for this sample
        fading_out[n * n_taps + tap] = state;
    }

    // Save state for next block
    tap_states[tap] = state;
    rng_states[tap] = rng;
}

/**
 * Simplified TDL kernel that uses pre-generated fading coefficients.
 */
__global__ void tdl_channel_with_fading(
    const cuFloatComplex* __restrict__ input,
    cuFloatComplex* __restrict__ output,
    const cuFloatComplex* __restrict__ fading,  // [n_samples * n_taps]
    const int* __restrict__ tap_delays,
    const float* __restrict__ tap_amplitudes,
    const float* __restrict__ tap_doppler_hz,
    float k_factor,
    float sample_rate_hz,
    float time_offset,
    int n_samples,
    int n_taps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;

    float t = time_offset + (float)idx / sample_rate_hz;

    float direct_coeff = sqrtf(k_factor / (k_factor + 1.0f));
    float scatter_coeff = sqrtf(1.0f / (k_factor + 1.0f));

    float out_re = 0.0f, out_im = 0.0f;

    for (int tap = 0; tap < n_taps; tap++) {
        int delay = tap_delays[tap];
        float amplitude = tap_amplitudes[tap];
        float doppler = tap_doppler_hz[tap];

        // Get delayed sample
        int delayed_idx = idx - delay;
        cuFloatComplex delayed_sample;
        if (delayed_idx >= 0) {
            delayed_sample = input[delayed_idx];
        } else {
            delayed_sample = make_cuFloatComplex(0.0f, 0.0f);
        }

        // Get pre-generated fading
        cuFloatComplex fading_state = fading[idx * n_taps + tap];

        // Fading gain
        cuFloatComplex fading_gain;
        if (tap == 0 && k_factor > 0.0f) {
            fading_gain = make_cuFloatComplex(
                direct_coeff + scatter_coeff * cuCrealf(fading_state),
                scatter_coeff * cuCimagf(fading_state)
            );
        } else {
            fading_gain = make_cuFloatComplex(
                scatter_coeff * cuCrealf(fading_state),
                scatter_coeff * cuCimagf(fading_state)
            );
        }

        // Doppler phase shift
        float doppler_phase = TWO_PI * doppler * t;
        float cos_d = cosf(doppler_phase);
        float sin_d = sinf(doppler_phase);

        // Combined gain
        float gain_re = amplitude * (cuCrealf(fading_gain) * cos_d - cuCimagf(fading_gain) * sin_d);
        float gain_im = amplitude * (cuCrealf(fading_gain) * sin_d + cuCimagf(fading_gain) * cos_d);

        // Apply to delayed sample
        out_re += cuCrealf(delayed_sample) * gain_re - cuCimagf(delayed_sample) * gain_im;
        out_im += cuCrealf(delayed_sample) * gain_im + cuCimagf(delayed_sample) * gain_re;
    }

    output[idx] = make_cuFloatComplex(out_re, out_im);
}

/**
 * VH RF Chain state structure.
 */
struct VHRFChainState {
    // Buffers
    cuFloatComplex* input_buffer;
    cuFloatComplex* rf_buffer;
    cuFloatComplex* output_buffer;
    cuFloatComplex* fading_buffer;  // Pre-generated fading [max_rf_samples * 16]

    // Resampler filters
    float* upsample_filter;
    float* downsample_filter;
    int upsample_filter_len;
    int downsample_filter_len;

    // TDL state
    cuFloatComplex* tap_states;
    curandState* rng_states;
    int* tap_delays;
    float* tap_amplitudes;
    float* tap_doppler;

    // Parameters
    int input_rate;
    int rf_rate;
    int upsample_factor;
    int max_input_samples;
    int max_rf_samples;
    int n_taps;
    int max_delay;
    float carrier_freq;
    float rho;
    float innovation_coeff;
    float k_factor;

    // Phase tracking for continuity
    float mixer_phase;
    float time_offset;

    bool initialized;
};

extern "C" {

/**
 * Initialize VH RF chain processor.
 */
void* init_vh_rf_chain(
    int input_rate,
    int rf_rate,
    int max_input_samples,
    float carrier_freq_hz,
    float coherence_time_sec,  // For AR(1) correlation
    float k_factor,
    unsigned long seed
) {
    VHRFChainState* state = new VHRFChainState;

    state->input_rate = input_rate;
    state->rf_rate = rf_rate;
    state->upsample_factor = rf_rate / input_rate;
    state->max_input_samples = max_input_samples;
    state->carrier_freq = carrier_freq_hz;
    state->k_factor = k_factor;
    state->mixer_phase = 0.0f;
    state->time_offset = 0.0f;
    state->initialized = false;

    // Compute AR(1) coefficient from coherence time
    // rho = exp(-dt / tau_c) where dt = 1/rf_rate
    float dt = 1.0f / rf_rate;
    state->rho = expf(-dt / coherence_time_sec);
    state->innovation_coeff = sqrtf(1.0f - state->rho * state->rho);

    state->max_rf_samples = max_input_samples * state->upsample_factor;

    // Allocate buffers
    cudaMalloc(&state->input_buffer, max_input_samples * sizeof(cuFloatComplex));
    cudaMalloc(&state->rf_buffer, state->max_rf_samples * sizeof(cuFloatComplex));
    cudaMalloc(&state->output_buffer, max_input_samples * sizeof(cuFloatComplex));
    cudaMalloc(&state->fading_buffer, state->max_rf_samples * 16 * sizeof(cuFloatComplex));

    // Design resampler filters (simple windowed sinc)
    // In practice, these should be precomputed and passed in
    int filter_len = state->upsample_factor * 16 + 1;  // 16 taps per phase
    state->upsample_filter_len = filter_len;
    state->downsample_filter_len = filter_len;

    float* h_filter = new float[filter_len];
    float cutoff = 0.5f / state->upsample_factor;
    int half = filter_len / 2;

    for (int i = 0; i < filter_len; i++) {
        float x = (float)(i - half);
        if (x == 0.0f) {
            h_filter[i] = 2.0f * cutoff;
        } else {
            h_filter[i] = sinf(TWO_PI * cutoff * x) / (PI * x);
        }
        // Hamming window
        h_filter[i] *= 0.54f - 0.46f * cosf(TWO_PI * i / (filter_len - 1));
    }

    cudaMalloc(&state->upsample_filter, filter_len * sizeof(float));
    cudaMalloc(&state->downsample_filter, filter_len * sizeof(float));
    cudaMemcpy(state->upsample_filter, h_filter, filter_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(state->downsample_filter, h_filter, filter_len * sizeof(float), cudaMemcpyHostToDevice);

    delete[] h_filter;

    // Default single tap (will be configured later)
    state->n_taps = 1;
    state->max_delay = 1;

    cudaMalloc(&state->tap_states, 16 * sizeof(cuFloatComplex));  // Max 16 taps
    cudaMalloc(&state->rng_states, 16 * sizeof(curandState));
    cudaMalloc(&state->tap_delays, 16 * sizeof(int));
    cudaMalloc(&state->tap_amplitudes, 16 * sizeof(float));
    cudaMalloc(&state->tap_doppler, 16 * sizeof(float));

    // Initialize RNG
    int threads = 256;
    int blocks = (16 + threads - 1) / threads;
    init_tdl_rng<<<blocks, threads>>>(state->rng_states, seed, 16);

    // Initialize tap states with unit magnitude random phase (not zero!)
    // This ensures immediate output even with Rayleigh fading
    cuFloatComplex h_tap_states[16];
    for (int i = 0; i < 16; i++) {
        float phase = (float)(seed + i * 12345) / 4294967295.0f * TWO_PI;
        h_tap_states[i] = make_cuFloatComplex(cosf(phase), sinf(phase));
    }
    cudaMemcpy(state->tap_states, h_tap_states, 16 * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    // Default tap configuration
    int default_delay = 0;
    float default_amp = 1.0f;
    float default_doppler = 0.0f;
    cudaMemcpy(state->tap_delays, &default_delay, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(state->tap_amplitudes, &default_amp, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(state->tap_doppler, &default_doppler, sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    state->initialized = true;

    return state;
}

/**
 * Configure TDL taps.
 */
int configure_vh_taps(
    void* state_ptr,
    const int* delays,
    const float* amplitudes,
    const float* doppler_hz,
    int n_taps
) {
    VHRFChainState* state = (VHRFChainState*)state_ptr;
    if (!state || !state->initialized || n_taps > 16) return -1;

    state->n_taps = n_taps;

    // Find max delay
    int max_d = 0;
    for (int i = 0; i < n_taps; i++) {
        if (delays[i] > max_d) max_d = delays[i];
    }
    state->max_delay = max_d + 1;

    cudaMemcpy(state->tap_delays, delays, n_taps * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(state->tap_amplitudes, amplitudes, n_taps * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(state->tap_doppler, doppler_hz, n_taps * sizeof(float), cudaMemcpyHostToDevice);

    return 0;
}

/**
 * Process samples through complete VH RF chain.
 *
 * 1. Upsample to RF rate
 * 2. Mix up to RF carrier
 * 3. Apply TDL channel model
 * 4. Mix down from RF carrier
 * 5. Downsample back to baseband
 */
int process_vh_rf_chain(
    void* state_ptr,
    const float* input_real,
    const float* input_imag,
    int n_samples,
    float* output_real,
    float* output_imag
) {
    VHRFChainState* state = (VHRFChainState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_samples > state->max_input_samples) return -2;

    int rf_samples = n_samples * state->upsample_factor;
    int threads = 256;

    // Copy input to device
    cuFloatComplex* h_input = new cuFloatComplex[n_samples];
    for (int i = 0; i < n_samples; i++) {
        h_input[i] = make_cuFloatComplex(input_real[i], input_imag[i]);
    }
    cudaMemcpy(state->input_buffer, h_input, n_samples * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    delete[] h_input;

    // Step 1: Upsample
    int taps_per_phase = state->upsample_filter_len / state->upsample_factor;
    int blocks = (rf_samples + threads - 1) / threads;
    polyphase_upsample<<<blocks, threads>>>(
        state->input_buffer,
        state->rf_buffer,
        state->upsample_filter,
        n_samples,
        rf_samples,
        state->upsample_factor,
        state->upsample_filter_len,
        taps_per_phase
    );

    // Temporary buffer for RF processing
    cuFloatComplex* rf_temp;
    cudaMalloc(&rf_temp, rf_samples * sizeof(cuFloatComplex));

    // Step 2: Mix up to RF
    mix_up_to_rf<<<blocks, threads>>>(
        state->rf_buffer,
        rf_temp,
        state->carrier_freq,
        (float)state->rf_rate,
        state->mixer_phase,
        rf_samples
    );

    // Step 3: Generate fading coefficients (one block per tap)
    generate_fading_coefficients<<<state->n_taps, 1>>>(
        state->fading_buffer,
        state->tap_states,
        state->rng_states,
        state->rho,
        state->innovation_coeff,
        rf_samples,
        state->n_taps
    );
    cudaDeviceSynchronize();  // Ensure fading is generated before TDL reads it

    // Step 4: Apply TDL channel with pre-generated fading
    tdl_channel_with_fading<<<blocks, threads>>>(
        rf_temp,
        state->rf_buffer,
        state->fading_buffer,
        state->tap_delays,
        state->tap_amplitudes,
        state->tap_doppler,
        state->k_factor,
        (float)state->rf_rate,
        state->time_offset,
        rf_samples,
        state->n_taps
    );

    // Step 5: Mix down from RF
    mix_down_from_rf<<<blocks, threads>>>(
        state->rf_buffer,
        rf_temp,
        state->carrier_freq,
        (float)state->rf_rate,
        state->mixer_phase,
        rf_samples
    );

    // Step 6: Downsample
    int out_blocks = (n_samples + threads - 1) / threads;
    polyphase_downsample<<<out_blocks, threads>>>(
        rf_temp,
        state->output_buffer,
        state->downsample_filter,
        rf_samples,
        n_samples,
        state->upsample_factor,
        state->downsample_filter_len
    );

    // Update phase for next block
    float block_time = (float)n_samples / state->input_rate;
    state->mixer_phase += TWO_PI * state->carrier_freq * block_time;
    state->mixer_phase = fmodf(state->mixer_phase, TWO_PI);
    state->time_offset += block_time;

    // Copy output back
    cuFloatComplex* h_output = new cuFloatComplex[n_samples];
    cudaMemcpy(h_output, state->output_buffer, n_samples * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n_samples; i++) {
        output_real[i] = cuCrealf(h_output[i]);
        output_imag[i] = cuCimagf(h_output[i]);
    }

    delete[] h_output;
    cudaFree(rf_temp);

    return 0;
}

/**
 * Debug: Test upsample only.
 * Returns upsampled power.
 */
float debug_test_upsample(void* state_ptr, const float* input_real, const float* input_imag, int n_samples) {
    VHRFChainState* state = (VHRFChainState*)state_ptr;
    if (!state || !state->initialized) return -1.0f;

    int rf_samples = n_samples * state->upsample_factor;
    int threads = 256;
    int blocks = (rf_samples + threads - 1) / threads;
    int taps_per_phase = state->upsample_filter_len / state->upsample_factor;

    // Copy input
    cuFloatComplex* h_input = new cuFloatComplex[n_samples];
    for (int i = 0; i < n_samples; i++) {
        h_input[i] = make_cuFloatComplex(input_real[i], input_imag[i]);
    }
    cudaMemcpy(state->input_buffer, h_input, n_samples * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    delete[] h_input;

    // Upsample
    polyphase_upsample<<<blocks, threads>>>(
        state->input_buffer, state->rf_buffer, state->upsample_filter,
        n_samples, rf_samples, state->upsample_factor,
        state->upsample_filter_len, taps_per_phase
    );
    cudaDeviceSynchronize();

    // Copy back and compute power
    cuFloatComplex* h_rf = new cuFloatComplex[rf_samples];
    cudaMemcpy(h_rf, state->rf_buffer, rf_samples * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    float power = 0.0f;
    for (int i = 0; i < rf_samples; i++) {
        float re = cuCrealf(h_rf[i]);
        float im = cuCimagf(h_rf[i]);
        power += re*re + im*im;
    }
    delete[] h_rf;

    return power / rf_samples;
}

/**
 * Debug: Test upsample + downsample (no RF processing).
 */
float debug_test_resample_roundtrip(void* state_ptr, const float* input_real, const float* input_imag, int n_samples) {
    VHRFChainState* state = (VHRFChainState*)state_ptr;
    if (!state || !state->initialized) return -1.0f;

    int rf_samples = n_samples * state->upsample_factor;
    int threads = 256;
    int taps_per_phase = state->upsample_filter_len / state->upsample_factor;

    // Copy input
    cuFloatComplex* h_input = new cuFloatComplex[n_samples];
    for (int i = 0; i < n_samples; i++) {
        h_input[i] = make_cuFloatComplex(input_real[i], input_imag[i]);
    }
    cudaMemcpy(state->input_buffer, h_input, n_samples * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    delete[] h_input;

    // Upsample
    int blocks_up = (rf_samples + threads - 1) / threads;
    polyphase_upsample<<<blocks_up, threads>>>(
        state->input_buffer, state->rf_buffer, state->upsample_filter,
        n_samples, rf_samples, state->upsample_factor,
        state->upsample_filter_len, taps_per_phase
    );

    // Downsample directly (no RF processing)
    int blocks_down = (n_samples + threads - 1) / threads;
    polyphase_downsample<<<blocks_down, threads>>>(
        state->rf_buffer, state->output_buffer, state->downsample_filter,
        rf_samples, n_samples, state->upsample_factor,
        state->downsample_filter_len
    );
    cudaDeviceSynchronize();

    // Copy output and compute power
    cuFloatComplex* h_output = new cuFloatComplex[n_samples];
    cudaMemcpy(h_output, state->output_buffer, n_samples * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    float power = 0.0f;
    for (int i = 0; i < n_samples; i++) {
        float re = cuCrealf(h_output[i]);
        float im = cuCimagf(h_output[i]);
        power += re*re + im*im;
    }
    delete[] h_output;

    return power / n_samples;
}

/**
 * Debug: Get filter coefficients.
 */
int debug_get_filter(void* state_ptr, float* filter_out, int max_len) {
    VHRFChainState* state = (VHRFChainState*)state_ptr;
    if (!state || !state->initialized) return -1;

    int len = state->upsample_filter_len;
    if (len > max_len) len = max_len;

    cudaMemcpy(filter_out, state->upsample_filter, len * sizeof(float), cudaMemcpyDeviceToHost);
    return len;
}

/**
 * Debug: Test upsample + mix up + mix down + downsample (no TDL).
 */
float debug_test_mixer_roundtrip(void* state_ptr, const float* input_real, const float* input_imag, int n_samples) {
    VHRFChainState* state = (VHRFChainState*)state_ptr;
    if (!state || !state->initialized) return -1.0f;

    int rf_samples = n_samples * state->upsample_factor;
    int threads = 256;
    int taps_per_phase = state->upsample_filter_len / state->upsample_factor;

    // Copy input
    cuFloatComplex* h_input = new cuFloatComplex[n_samples];
    for (int i = 0; i < n_samples; i++) {
        h_input[i] = make_cuFloatComplex(input_real[i], input_imag[i]);
    }
    cudaMemcpy(state->input_buffer, h_input, n_samples * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    delete[] h_input;

    cuFloatComplex* rf_temp;
    cudaMalloc(&rf_temp, rf_samples * sizeof(cuFloatComplex));

    int blocks = (rf_samples + threads - 1) / threads;

    // Upsample
    polyphase_upsample<<<blocks, threads>>>(
        state->input_buffer, state->rf_buffer, state->upsample_filter,
        n_samples, rf_samples, state->upsample_factor,
        state->upsample_filter_len, taps_per_phase
    );

    // Mix up
    mix_up_to_rf<<<blocks, threads>>>(
        state->rf_buffer, rf_temp,
        state->carrier_freq, (float)state->rf_rate, 0.0f, rf_samples
    );

    // Mix down directly (no TDL)
    mix_down_from_rf<<<blocks, threads>>>(
        rf_temp, state->rf_buffer,
        state->carrier_freq, (float)state->rf_rate, 0.0f, rf_samples
    );

    // Downsample
    int out_blocks = (n_samples + threads - 1) / threads;
    polyphase_downsample<<<out_blocks, threads>>>(
        state->rf_buffer, state->output_buffer, state->downsample_filter,
        rf_samples, n_samples, state->upsample_factor, state->downsample_filter_len
    );
    cudaDeviceSynchronize();

    // Compute output power
    cuFloatComplex* h_output = new cuFloatComplex[n_samples];
    cudaMemcpy(h_output, state->output_buffer, n_samples * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    float power = 0.0f;
    for (int i = 0; i < n_samples; i++) {
        float re = cuCrealf(h_output[i]);
        float im = cuCimagf(h_output[i]);
        power += re*re + im*im;
    }
    delete[] h_output;
    cudaFree(rf_temp);

    return power / n_samples;
}

/**
 * Debug: Test full chain but with TDL bypass (unity gain passthrough).
 */
/**
 * Debug: Get fading coefficient statistics.
 */
void debug_get_fading_stats(void* state_ptr, int n_samples, float* mean_mag, float* mean_magsq) {
    VHRFChainState* state = (VHRFChainState*)state_ptr;
    if (!state || !state->initialized) {
        *mean_mag = -1.0f;
        *mean_magsq = -1.0f;
        return;
    }

    int rf_samples = n_samples * state->upsample_factor;

    // Generate fading coefficients
    generate_fading_coefficients<<<state->n_taps, 1>>>(
        state->fading_buffer,
        state->tap_states,
        state->rng_states,
        state->rho,
        state->innovation_coeff,
        rf_samples,
        state->n_taps
    );
    cudaDeviceSynchronize();

    // Copy back and compute stats
    int total = rf_samples * state->n_taps;
    cuFloatComplex* h_fading = new cuFloatComplex[total];
    cudaMemcpy(h_fading, state->fading_buffer, total * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    float sum_mag = 0.0f, sum_magsq = 0.0f;
    for (int i = 0; i < total; i++) {
        float re = cuCrealf(h_fading[i]);
        float im = cuCimagf(h_fading[i]);
        float mag = sqrtf(re*re + im*im);
        sum_mag += mag;
        sum_magsq += re*re + im*im;
    }

    *mean_mag = sum_mag / total;
    *mean_magsq = sum_magsq / total;

    delete[] h_fading;
}

/**
 * Debug: Test TDL output with specific fading value.
 */
float debug_test_tdl_constant_fading(void* state_ptr, const float* input_real, const float* input_imag,
                                     int n_samples, float fading_re, float fading_im) {
    VHRFChainState* state = (VHRFChainState*)state_ptr;
    if (!state || !state->initialized) return -1.0f;

    int rf_samples = n_samples * state->upsample_factor;
    int threads = 256;
    int taps_per_phase = state->upsample_filter_len / state->upsample_factor;

    // Copy input
    cuFloatComplex* h_input = new cuFloatComplex[n_samples];
    for (int i = 0; i < n_samples; i++) {
        h_input[i] = make_cuFloatComplex(input_real[i], input_imag[i]);
    }
    cudaMemcpy(state->input_buffer, h_input, n_samples * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    delete[] h_input;

    cuFloatComplex* rf_temp;
    cudaMalloc(&rf_temp, rf_samples * sizeof(cuFloatComplex));

    int blocks = (rf_samples + threads - 1) / threads;

    // Upsample
    polyphase_upsample<<<blocks, threads>>>(
        state->input_buffer, state->rf_buffer, state->upsample_filter,
        n_samples, rf_samples, state->upsample_factor,
        state->upsample_filter_len, taps_per_phase
    );

    // Mix up
    mix_up_to_rf<<<blocks, threads>>>(
        state->rf_buffer, rf_temp,
        state->carrier_freq, (float)state->rf_rate, 0.0f, rf_samples
    );

    // Set constant fading
    int total = rf_samples * state->n_taps;
    cuFloatComplex* h_fading = new cuFloatComplex[total];
    for (int i = 0; i < total; i++) {
        h_fading[i] = make_cuFloatComplex(fading_re, fading_im);
    }
    cudaMemcpy(state->fading_buffer, h_fading, total * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    delete[] h_fading;

    // Apply TDL
    tdl_channel_with_fading<<<blocks, threads>>>(
        rf_temp, state->rf_buffer, state->fading_buffer,
        state->tap_delays, state->tap_amplitudes, state->tap_doppler,
        state->k_factor, (float)state->rf_rate, 0.0f, rf_samples, state->n_taps
    );

    // Mix down
    mix_down_from_rf<<<blocks, threads>>>(
        state->rf_buffer, rf_temp,
        state->carrier_freq, (float)state->rf_rate, 0.0f, rf_samples
    );

    // Downsample
    int out_blocks = (n_samples + threads - 1) / threads;
    polyphase_downsample<<<out_blocks, threads>>>(
        rf_temp, state->output_buffer, state->downsample_filter,
        rf_samples, n_samples, state->upsample_factor, state->downsample_filter_len
    );
    cudaDeviceSynchronize();

    cuFloatComplex* h_output = new cuFloatComplex[n_samples];
    cudaMemcpy(h_output, state->output_buffer, n_samples * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    float power = 0.0f;
    for (int i = 0; i < n_samples; i++) {
        float re = cuCrealf(h_output[i]);
        float im = cuCimagf(h_output[i]);
        power += re*re + im*im;
    }
    delete[] h_output;
    cudaFree(rf_temp);

    return power / n_samples;
}

float debug_test_tdl_unity(void* state_ptr, const float* input_real, const float* input_imag, int n_samples) {
    VHRFChainState* state = (VHRFChainState*)state_ptr;
    if (!state || !state->initialized) return -1.0f;

    int rf_samples = n_samples * state->upsample_factor;
    int threads = 256;
    int taps_per_phase = state->upsample_filter_len / state->upsample_factor;

    // Copy input
    cuFloatComplex* h_input = new cuFloatComplex[n_samples];
    for (int i = 0; i < n_samples; i++) {
        h_input[i] = make_cuFloatComplex(input_real[i], input_imag[i]);
    }
    cudaMemcpy(state->input_buffer, h_input, n_samples * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    delete[] h_input;

    cuFloatComplex* rf_temp;
    cudaMalloc(&rf_temp, rf_samples * sizeof(cuFloatComplex));

    int blocks = (rf_samples + threads - 1) / threads;

    // Upsample
    polyphase_upsample<<<blocks, threads>>>(
        state->input_buffer, state->rf_buffer, state->upsample_filter,
        n_samples, rf_samples, state->upsample_factor,
        state->upsample_filter_len, taps_per_phase
    );

    // Mix up
    mix_up_to_rf<<<blocks, threads>>>(
        state->rf_buffer, rf_temp,
        state->carrier_freq, (float)state->rf_rate, 0.0f, rf_samples
    );

    // TDL bypass: just copy (unity gain)
    cudaMemcpy(state->rf_buffer, rf_temp, rf_samples * sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice);

    // Mix down
    mix_down_from_rf<<<blocks, threads>>>(
        state->rf_buffer, rf_temp,
        state->carrier_freq, (float)state->rf_rate, 0.0f, rf_samples
    );

    // Downsample
    int out_blocks = (n_samples + threads - 1) / threads;
    polyphase_downsample<<<out_blocks, threads>>>(
        rf_temp, state->output_buffer, state->downsample_filter,
        rf_samples, n_samples, state->upsample_factor, state->downsample_filter_len
    );
    cudaDeviceSynchronize();

    cuFloatComplex* h_output = new cuFloatComplex[n_samples];
    cudaMemcpy(h_output, state->output_buffer, n_samples * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    float power = 0.0f;
    for (int i = 0; i < n_samples; i++) {
        float re = cuCrealf(h_output[i]);
        float im = cuCimagf(h_output[i]);
        power += re*re + im*im;
    }
    delete[] h_output;
    cudaFree(rf_temp);

    return power / n_samples;
}

/**
 * Free VH RF chain resources.
 */
void free_vh_rf_chain(void* state_ptr) {
    VHRFChainState* state = (VHRFChainState*)state_ptr;
    if (!state) return;

    if (state->initialized) {
        cudaFree(state->input_buffer);
        cudaFree(state->rf_buffer);
        cudaFree(state->output_buffer);
        cudaFree(state->fading_buffer);
        cudaFree(state->upsample_filter);
        cudaFree(state->downsample_filter);
        cudaFree(state->tap_states);
        cudaFree(state->rng_states);
        cudaFree(state->tap_delays);
        cudaFree(state->tap_amplitudes);
        cudaFree(state->tap_doppler);
    }

    delete state;
}

/**
 * Debug: Report power at each stage of the RF chain.
 * Returns array of 7 floats: input, after_up, after_mixup, fading, after_tdl, after_mixdown, output
 */
void debug_power_stages(void* state_ptr, const float* input_real, const float* input_imag,
                        int n_samples, float* powers_out) {
    VHRFChainState* state = (VHRFChainState*)state_ptr;
    if (!state || !state->initialized) {
        for (int i = 0; i < 7; i++) powers_out[i] = -1.0f;
        return;
    }

    int rf_samples = n_samples * state->upsample_factor;
    int threads = 256;
    int taps_per_phase = state->upsample_filter_len / state->upsample_factor;

    // Copy input and compute input power
    cuFloatComplex* h_input = new cuFloatComplex[n_samples];
    float input_power = 0.0f;
    for (int i = 0; i < n_samples; i++) {
        h_input[i] = make_cuFloatComplex(input_real[i], input_imag[i]);
        input_power += input_real[i] * input_real[i] + input_imag[i] * input_imag[i];
    }
    powers_out[0] = input_power / n_samples;
    cudaMemcpy(state->input_buffer, h_input, n_samples * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    delete[] h_input;

    cuFloatComplex* rf_temp;
    cudaMalloc(&rf_temp, rf_samples * sizeof(cuFloatComplex));
    int blocks = (rf_samples + threads - 1) / threads;

    // Step 1: Upsample
    polyphase_upsample<<<blocks, threads>>>(
        state->input_buffer, state->rf_buffer, state->upsample_filter,
        n_samples, rf_samples, state->upsample_factor,
        state->upsample_filter_len, taps_per_phase
    );
    cudaDeviceSynchronize();

    // Check power after upsample
    cuFloatComplex* h_rf = new cuFloatComplex[rf_samples];
    cudaMemcpy(h_rf, state->rf_buffer, rf_samples * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    float up_power = 0.0f;
    for (int i = 0; i < rf_samples; i++) {
        up_power += cuCrealf(h_rf[i]) * cuCrealf(h_rf[i]) + cuCimagf(h_rf[i]) * cuCimagf(h_rf[i]);
    }
    powers_out[1] = up_power / rf_samples;

    // Step 2: Mix up
    mix_up_to_rf<<<blocks, threads>>>(
        state->rf_buffer, rf_temp,
        state->carrier_freq, (float)state->rf_rate, 0.0f, rf_samples
    );
    cudaDeviceSynchronize();

    // Check power after mix up
    cudaMemcpy(h_rf, rf_temp, rf_samples * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    float mixup_power = 0.0f;
    for (int i = 0; i < rf_samples; i++) {
        mixup_power += cuCrealf(h_rf[i]) * cuCrealf(h_rf[i]) + cuCimagf(h_rf[i]) * cuCimagf(h_rf[i]);
    }
    powers_out[2] = mixup_power / rf_samples;

    // Step 3: Generate fading
    generate_fading_coefficients<<<state->n_taps, 1>>>(
        state->fading_buffer,
        state->tap_states,
        state->rng_states,
        state->rho,
        state->innovation_coeff,
        rf_samples,
        state->n_taps
    );
    cudaDeviceSynchronize();

    // Check fading power
    int total_fading = rf_samples * state->n_taps;
    cuFloatComplex* h_fading = new cuFloatComplex[total_fading];
    cudaMemcpy(h_fading, state->fading_buffer, total_fading * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    float fading_power = 0.0f;
    for (int i = 0; i < total_fading; i++) {
        fading_power += cuCrealf(h_fading[i]) * cuCrealf(h_fading[i]) + cuCimagf(h_fading[i]) * cuCimagf(h_fading[i]);
    }
    powers_out[3] = fading_power / total_fading;
    delete[] h_fading;

    // Step 4: Apply TDL
    tdl_channel_with_fading<<<blocks, threads>>>(
        rf_temp, state->rf_buffer, state->fading_buffer,
        state->tap_delays, state->tap_amplitudes, state->tap_doppler,
        state->k_factor, (float)state->rf_rate, 0.0f, rf_samples, state->n_taps
    );
    cudaDeviceSynchronize();

    // Check power after TDL
    cudaMemcpy(h_rf, state->rf_buffer, rf_samples * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    float tdl_power = 0.0f;
    for (int i = 0; i < rf_samples; i++) {
        tdl_power += cuCrealf(h_rf[i]) * cuCrealf(h_rf[i]) + cuCimagf(h_rf[i]) * cuCimagf(h_rf[i]);
    }
    powers_out[4] = tdl_power / rf_samples;

    // Step 5: Mix down
    mix_down_from_rf<<<blocks, threads>>>(
        state->rf_buffer, rf_temp,
        state->carrier_freq, (float)state->rf_rate, 0.0f, rf_samples
    );
    cudaDeviceSynchronize();

    // Check power after mix down
    cudaMemcpy(h_rf, rf_temp, rf_samples * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    float mixdown_power = 0.0f;
    for (int i = 0; i < rf_samples; i++) {
        mixdown_power += cuCrealf(h_rf[i]) * cuCrealf(h_rf[i]) + cuCimagf(h_rf[i]) * cuCimagf(h_rf[i]);
    }
    powers_out[5] = mixdown_power / rf_samples;

    delete[] h_rf;

    // Step 6: Downsample
    int out_blocks = (n_samples + threads - 1) / threads;
    polyphase_downsample<<<out_blocks, threads>>>(
        rf_temp, state->output_buffer, state->downsample_filter,
        rf_samples, n_samples, state->upsample_factor, state->downsample_filter_len
    );
    cudaDeviceSynchronize();

    // Check output power
    cuFloatComplex* h_output = new cuFloatComplex[n_samples];
    cudaMemcpy(h_output, state->output_buffer, n_samples * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    float output_power = 0.0f;
    for (int i = 0; i < n_samples; i++) {
        output_power += cuCrealf(h_output[i]) * cuCrealf(h_output[i]) + cuCimagf(h_output[i]) * cuCimagf(h_output[i]);
    }
    powers_out[6] = output_power / n_samples;
    delete[] h_output;

    cudaFree(rf_temp);
}

/**
 * Reset VH RF chain state (clear fading states, reset phase).
 */
void reset_vh_rf_chain(void* state_ptr) {
    VHRFChainState* state = (VHRFChainState*)state_ptr;
    if (!state || !state->initialized) return;

    state->mixer_phase = 0.0f;
    state->time_offset = 0.0f;

    cudaMemset(state->tap_states, 0, 16 * sizeof(cuFloatComplex));
}

} // extern "C"
