/**
 * Optimized CPU implementation of Vogler-Hoffmeyer RF processing chain.
 *
 * Uses OpenMP for parallelization and compiler auto-vectorization.
 * Designed as fallback when CUDA is not available.
 *
 * Compile with: -O3 -march=native -fopenmp -ffast-math
 */

#include <cmath>
#include <cstring>
#include <random>
#include <complex>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

// Use aligned memory for SIMD
#ifdef __AVX__
#define ALIGN_BYTES 32
#else
#define ALIGN_BYTES 16
#endif

#define PI 3.14159265358979323846
#define TWO_PI 6.28318530717958647692

// Aligned allocation helpers
inline void* aligned_alloc_impl(size_t alignment, size_t size) {
#ifdef _MSC_VER
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    posix_memalign(&ptr, alignment, size);
    return ptr;
#endif
}

inline void aligned_free_impl(void* ptr) {
#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/**
 * VH RF Chain CPU state structure.
 */
struct VHRFChainCPUState {
    // Buffers (aligned for SIMD)
    float* input_real;
    float* input_imag;
    float* rf_real;
    float* rf_imag;
    float* output_real;
    float* output_imag;

    // Resampler filter
    float* filter;
    int filter_len;
    int taps_per_phase;

    // TDL state
    float* tap_state_real;  // AR(1) state per tap
    float* tap_state_imag;
    int* tap_delays;
    float* tap_amplitudes;
    float* tap_doppler;

    // RNG per tap
    std::mt19937* rngs;
    std::normal_distribution<float>* normal_dist;

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

    // Phase tracking
    double mixer_phase;
    double time_offset;

    bool initialized;
};

extern "C" {

/**
 * Polyphase upsample - CPU optimized.
 */
static void polyphase_upsample_cpu(
    const float* __restrict__ in_real,
    const float* __restrict__ in_imag,
    float* __restrict__ out_real,
    float* __restrict__ out_imag,
    const float* __restrict__ filter,
    int input_len,
    int output_len,
    int upsample_factor,
    int filter_len,
    int taps_per_phase
) {
    int half_taps = taps_per_phase / 2;
    int filter_center = filter_len / 2;

    #pragma omp parallel for schedule(static)
    for (int out_idx = 0; out_idx < output_len; out_idx++) {
        int phase = out_idx % upsample_factor;
        int in_base = out_idx / upsample_factor;

        float re = 0.0f, im = 0.0f;

        // Polyphase filtering with proper center alignment
        for (int k = -half_taps; k <= half_taps; k++) {
            int in_idx = in_base - k;
            if (in_idx >= 0 && in_idx < input_len) {
                int filt_idx = filter_center + k * upsample_factor + phase;
                if (filt_idx >= 0 && filt_idx < filter_len) {
                    float h = filter[filt_idx] * upsample_factor;
                    re += in_real[in_idx] * h;
                    im += in_imag[in_idx] * h;
                }
            }
        }

        out_real[out_idx] = re;
        out_imag[out_idx] = im;
    }
}

/**
 * Polyphase downsample - CPU optimized.
 */
static void polyphase_downsample_cpu(
    const float* __restrict__ in_real,
    const float* __restrict__ in_imag,
    float* __restrict__ out_real,
    float* __restrict__ out_imag,
    const float* __restrict__ filter,
    int input_len,
    int output_len,
    int downsample_factor,
    int filter_len
) {
    int half_len = filter_len / 2;
    #pragma omp parallel for schedule(static)
    for (int out_idx = 0; out_idx < output_len; out_idx++) {
        int in_center = out_idx * downsample_factor;

        float re = 0.0f, im = 0.0f;

        for (int k = 0; k < filter_len; k++) {
            int in_idx = in_center - half_len + k;
            if (in_idx >= 0 && in_idx < input_len) {
                float h = filter[k];
                re += in_real[in_idx] * h;
                im += in_imag[in_idx] * h;
            }
        }

        out_real[out_idx] = re;
        out_imag[out_idx] = im;
    }
}

/**
 * RF mixer - CPU optimized with vectorization hints.
 */
static void mix_rf_cpu(
    const float* __restrict__ in_real,
    const float* __restrict__ in_imag,
    float* __restrict__ out_real,
    float* __restrict__ out_imag,
    float carrier_freq_hz,
    float sample_rate_hz,
    double phase_offset,
    int N,
    bool up  // true = mix up, false = mix down
) {
    double phase_inc = TWO_PI * carrier_freq_hz / sample_rate_hz;
    float sign = up ? 1.0f : -1.0f;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        double phase = phase_offset + phase_inc * i;
        float cos_p = (float)cos(phase);
        float sin_p = sign * (float)sin(phase);

        float re = in_real[i];
        float im = in_imag[i];

        out_real[i] = re * cos_p - im * sin_p;
        out_imag[i] = re * sin_p + im * cos_p;
    }
}

/**
 * TDL channel process - CPU optimized.
 *
 * Processes samples through tapped delay line with AR(1) fading.
 */
static void tdl_channel_cpu(
    const float* __restrict__ in_real,
    const float* __restrict__ in_imag,
    float* __restrict__ out_real,
    float* __restrict__ out_imag,
    float* tap_state_real,
    float* tap_state_imag,
    std::mt19937* rngs,
    std::normal_distribution<float>* normal_dist,
    const int* tap_delays,
    const float* tap_amplitudes,
    const float* tap_doppler,
    float rho,
    float innovation_coeff,
    float k_factor,
    float sample_rate_hz,
    double time_offset,
    int n_samples,
    int n_taps
) {
    float direct_coeff = sqrtf(k_factor / (k_factor + 1.0f));
    float scatter_coeff = sqrtf(1.0f / (k_factor + 1.0f));
    float inv_sqrt2 = 0.7071067811865476f;

    // Process sample by sample (AR(1) requires sequential update)
    for (int n = 0; n < n_samples; n++) {
        float t = (float)(time_offset + (double)n / sample_rate_hz);

        float sum_re = 0.0f, sum_im = 0.0f;

        for (int tap = 0; tap < n_taps; tap++) {
            // Update AR(1) fading state
            float z_re = (*normal_dist)(rngs[tap]) * inv_sqrt2;
            float z_im = (*normal_dist)(rngs[tap]) * inv_sqrt2;

            tap_state_real[tap] = rho * tap_state_real[tap] + innovation_coeff * z_re;
            tap_state_imag[tap] = rho * tap_state_imag[tap] + innovation_coeff * z_im;

            // Get delayed sample
            int delay = tap_delays[tap];
            int delayed_idx = n - delay;
            float del_re = 0.0f, del_im = 0.0f;
            if (delayed_idx >= 0) {
                del_re = in_real[delayed_idx];
                del_im = in_imag[delayed_idx];
            }

            // Fading gain
            float fading_re, fading_im;
            if (tap == 0 && k_factor > 0.0f) {
                fading_re = direct_coeff + scatter_coeff * tap_state_real[tap];
                fading_im = scatter_coeff * tap_state_imag[tap];
            } else {
                fading_re = scatter_coeff * tap_state_real[tap];
                fading_im = scatter_coeff * tap_state_imag[tap];
            }

            // Doppler phase
            float doppler_phase = (float)(TWO_PI * tap_doppler[tap] * t);
            float cos_d = cosf(doppler_phase);
            float sin_d = sinf(doppler_phase);

            // Combined gain
            float amp = tap_amplitudes[tap];
            float gain_re = amp * (fading_re * cos_d - fading_im * sin_d);
            float gain_im = amp * (fading_re * sin_d + fading_im * cos_d);

            // Apply to delayed sample
            sum_re += del_re * gain_re - del_im * gain_im;
            sum_im += del_re * gain_im + del_im * gain_re;
        }

        out_real[n] = sum_re;
        out_imag[n] = sum_im;
    }
}

/**
 * Initialize VH RF chain CPU processor.
 */
void* init_vh_rf_chain_cpu(
    int input_rate,
    int rf_rate,
    int max_input_samples,
    float carrier_freq_hz,
    float coherence_time_sec,
    float k_factor,
    unsigned long seed
) {
    VHRFChainCPUState* state = new VHRFChainCPUState;

    state->input_rate = input_rate;
    state->rf_rate = rf_rate;
    state->upsample_factor = rf_rate / input_rate;
    state->max_input_samples = max_input_samples;
    state->max_rf_samples = max_input_samples * state->upsample_factor;
    state->carrier_freq = carrier_freq_hz;
    state->k_factor = k_factor;
    state->mixer_phase = 0.0;
    state->time_offset = 0.0;
    state->initialized = false;

    // AR(1) coefficient
    float dt = 1.0f / rf_rate;
    state->rho = expf(-dt / coherence_time_sec);
    state->innovation_coeff = sqrtf(1.0f - state->rho * state->rho);

    // Allocate aligned buffers
    size_t input_size = max_input_samples * sizeof(float);
    size_t rf_size = state->max_rf_samples * sizeof(float);

    state->input_real = (float*)aligned_alloc_impl(ALIGN_BYTES, input_size);
    state->input_imag = (float*)aligned_alloc_impl(ALIGN_BYTES, input_size);
    state->rf_real = (float*)aligned_alloc_impl(ALIGN_BYTES, rf_size);
    state->rf_imag = (float*)aligned_alloc_impl(ALIGN_BYTES, rf_size);
    state->output_real = (float*)aligned_alloc_impl(ALIGN_BYTES, input_size);
    state->output_imag = (float*)aligned_alloc_impl(ALIGN_BYTES, input_size);

    // Design resampler filter
    state->filter_len = state->upsample_factor * 16 + 1;
    state->taps_per_phase = 16;
    state->filter = (float*)aligned_alloc_impl(ALIGN_BYTES, state->filter_len * sizeof(float));

    float cutoff = 0.5f / state->upsample_factor;
    int half = state->filter_len / 2;

    for (int i = 0; i < state->filter_len; i++) {
        float x = (float)(i - half);
        if (x == 0.0f) {
            state->filter[i] = 2.0f * cutoff;
        } else {
            state->filter[i] = sinf(TWO_PI * cutoff * x) / (PI * x);
        }
        // Hamming window
        state->filter[i] *= 0.54f - 0.46f * cosf(TWO_PI * i / (state->filter_len - 1));
    }

    // Tap state (max 16 taps)
    state->n_taps = 1;
    state->max_delay = 1;
    state->tap_state_real = (float*)aligned_alloc_impl(ALIGN_BYTES, 16 * sizeof(float));
    state->tap_state_imag = (float*)aligned_alloc_impl(ALIGN_BYTES, 16 * sizeof(float));
    state->tap_delays = (int*)aligned_alloc_impl(ALIGN_BYTES, 16 * sizeof(int));
    state->tap_amplitudes = (float*)aligned_alloc_impl(ALIGN_BYTES, 16 * sizeof(float));
    state->tap_doppler = (float*)aligned_alloc_impl(ALIGN_BYTES, 16 * sizeof(float));

    // Initialize tap states with unit magnitude random phase (not zero!)
    // This ensures immediate output even with Rayleigh fading
    for (int i = 0; i < 16; i++) {
        double phase = (double)(seed + i * 12345) / 4294967295.0 * TWO_PI;
        state->tap_state_real[i] = (float)cos(phase);
        state->tap_state_imag[i] = (float)sin(phase);
    }

    // Initialize RNGs
    state->rngs = new std::mt19937[16];
    state->normal_dist = new std::normal_distribution<float>(0.0f, 1.0f);
    for (int i = 0; i < 16; i++) {
        state->rngs[i].seed(seed + i);
    }

    // Default single tap
    state->tap_delays[0] = 0;
    state->tap_amplitudes[0] = 1.0f;
    state->tap_doppler[0] = 0.0f;

    state->initialized = true;
    return state;
}

/**
 * Configure TDL taps - CPU version.
 */
int configure_vh_taps_cpu(
    void* state_ptr,
    const int* delays,
    const float* amplitudes,
    const float* doppler_hz,
    int n_taps
) {
    VHRFChainCPUState* state = (VHRFChainCPUState*)state_ptr;
    if (!state || !state->initialized || n_taps > 16) return -1;

    state->n_taps = n_taps;

    int max_d = 0;
    for (int i = 0; i < n_taps; i++) {
        if (delays[i] > max_d) max_d = delays[i];
    }
    state->max_delay = max_d + 1;

    memcpy(state->tap_delays, delays, n_taps * sizeof(int));
    memcpy(state->tap_amplitudes, amplitudes, n_taps * sizeof(float));
    memcpy(state->tap_doppler, doppler_hz, n_taps * sizeof(float));

    return 0;
}

/**
 * Process samples through complete VH RF chain - CPU version.
 */
int process_vh_rf_chain_cpu(
    void* state_ptr,
    const float* input_real,
    const float* input_imag,
    int n_samples,
    float* output_real,
    float* output_imag
) {
    VHRFChainCPUState* state = (VHRFChainCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_samples > state->max_input_samples) return -2;

    int rf_samples = n_samples * state->upsample_factor;

    // Copy input
    memcpy(state->input_real, input_real, n_samples * sizeof(float));
    memcpy(state->input_imag, input_imag, n_samples * sizeof(float));

    // Allocate temp buffers for RF processing
    float* rf_temp_real = (float*)aligned_alloc_impl(ALIGN_BYTES, rf_samples * sizeof(float));
    float* rf_temp_imag = (float*)aligned_alloc_impl(ALIGN_BYTES, rf_samples * sizeof(float));

    // Step 1: Upsample
    polyphase_upsample_cpu(
        state->input_real, state->input_imag,
        state->rf_real, state->rf_imag,
        state->filter,
        n_samples, rf_samples,
        state->upsample_factor,
        state->filter_len,
        state->taps_per_phase
    );

    // Step 2: Mix up to RF
    mix_rf_cpu(
        state->rf_real, state->rf_imag,
        rf_temp_real, rf_temp_imag,
        state->carrier_freq,
        (float)state->rf_rate,
        state->mixer_phase,
        rf_samples,
        true  // up
    );

    // Step 3: TDL channel
    tdl_channel_cpu(
        rf_temp_real, rf_temp_imag,
        state->rf_real, state->rf_imag,
        state->tap_state_real, state->tap_state_imag,
        state->rngs, state->normal_dist,
        state->tap_delays, state->tap_amplitudes, state->tap_doppler,
        state->rho, state->innovation_coeff, state->k_factor,
        (float)state->rf_rate,
        state->time_offset,
        rf_samples,
        state->n_taps
    );

    // Step 4: Mix down from RF
    mix_rf_cpu(
        state->rf_real, state->rf_imag,
        rf_temp_real, rf_temp_imag,
        state->carrier_freq,
        (float)state->rf_rate,
        state->mixer_phase,
        rf_samples,
        false  // down
    );

    // Step 5: Downsample
    polyphase_downsample_cpu(
        rf_temp_real, rf_temp_imag,
        state->output_real, state->output_imag,
        state->filter,
        rf_samples, n_samples,
        state->upsample_factor,
        state->filter_len
    );

    // Update phase for next block
    double block_time = (double)n_samples / state->input_rate;
    state->mixer_phase += TWO_PI * state->carrier_freq * block_time;
    state->mixer_phase = fmod(state->mixer_phase, TWO_PI);
    state->time_offset += block_time;

    // Copy output
    memcpy(output_real, state->output_real, n_samples * sizeof(float));
    memcpy(output_imag, state->output_imag, n_samples * sizeof(float));

    aligned_free_impl(rf_temp_real);
    aligned_free_impl(rf_temp_imag);

    return 0;
}

/**
 * Free VH RF chain CPU resources.
 */
void free_vh_rf_chain_cpu(void* state_ptr) {
    VHRFChainCPUState* state = (VHRFChainCPUState*)state_ptr;
    if (!state) return;

    if (state->initialized) {
        aligned_free_impl(state->input_real);
        aligned_free_impl(state->input_imag);
        aligned_free_impl(state->rf_real);
        aligned_free_impl(state->rf_imag);
        aligned_free_impl(state->output_real);
        aligned_free_impl(state->output_imag);
        aligned_free_impl(state->filter);
        aligned_free_impl(state->tap_state_real);
        aligned_free_impl(state->tap_state_imag);
        aligned_free_impl(state->tap_delays);
        aligned_free_impl(state->tap_amplitudes);
        aligned_free_impl(state->tap_doppler);

        delete[] state->rngs;
        delete state->normal_dist;
    }

    delete state;
}

/**
 * Reset VH RF chain CPU state.
 */
void reset_vh_rf_chain_cpu(void* state_ptr) {
    VHRFChainCPUState* state = (VHRFChainCPUState*)state_ptr;
    if (!state || !state->initialized) return;

    state->mixer_phase = 0.0;
    state->time_offset = 0.0;

    memset(state->tap_state_real, 0, 16 * sizeof(float));
    memset(state->tap_state_imag, 0, 16 * sizeof(float));
}

} // extern "C"
