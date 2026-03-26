/**
 * CPU fallback implementation for Watterson HF channel model.
 *
 * Optimized C++ implementation using OpenMP for parallelization.
 * Used when CUDA is not available.
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

#define PI 3.14159265358979323846
#define TWO_PI 6.28318530717958647692
#define INV_SQRT2 0.7071067811865476

/**
 * Doppler spectrum types.
 */
enum DopplerSpectrumTypeCPU {
    DOPPLER_GAUSSIAN_CPU = 0,
    DOPPLER_FLAT_CPU = 1,
    DOPPLER_JAKES_CPU = 2
};

/**
 * Watterson tap parameters.
 */
struct WattersonTapCPU {
    int delay_samples;
    float amplitude;
    float doppler_spread_hz;
    int spectrum_type;
    bool is_rician;
    float k_factor_linear;
};

/**
 * Watterson channel CPU state.
 */
struct WattersonCPUState {
    // Tap configuration
    std::vector<WattersonTapCPU> taps;
    int n_taps;
    int max_taps;
    int max_delay;
    int max_samples;
    int filter_len;
    float sample_rate;

    // Doppler filters [n_taps][filter_len]
    std::vector<std::vector<float>> doppler_filters;

    // Noise buffers [n_taps][filter_len]
    std::vector<std::vector<std::complex<float>>> noise_buffers;

    // Delay buffers [n_taps][max_delay]
    std::vector<std::vector<std::complex<float>>> delay_buffers;

    // Fading gains (previous and current)
    std::vector<std::complex<float>> old_gains;
    std::vector<std::complex<float>> new_gains;

    // RNG per tap
    std::vector<std::mt19937> rngs;
    std::normal_distribution<float> normal_dist;

    bool initialized;
};

extern "C" {

/**
 * Compute Doppler filter coefficients.
 */
static void compute_doppler_filter_cpu(
    std::vector<float>& filter,
    float doppler_spread_hz,
    int spectrum_type,
    float update_rate
) {
    int filter_len = filter.size();
    int half = filter_len / 2;
    float f_norm = doppler_spread_hz / update_rate;

    float sum_sq = 0.0f;

    for (int i = 0; i < filter_len; i++) {
        float t = (float)(i - half);
        float h = 0.0f;

        if (spectrum_type == DOPPLER_GAUSSIAN_CPU) {
            float sigma = (f_norm > 0.0f) ? std::max(1.0f, 1.0f / (float)(TWO_PI * f_norm)) : filter_len / 4.0f;
            h = expf(-0.5f * (t / sigma) * (t / sigma));
        } else if (spectrum_type == DOPPLER_FLAT_CPU) {
            float arg = 2.0f * f_norm * t;
            h = (fabsf(arg) < 1e-10f) ? 1.0f : sinf((float)PI * arg) / ((float)PI * arg);
        } else if (spectrum_type == DOPPLER_JAKES_CPU) {
            float arg = 2.0f * f_norm * t;
            float sinc = (fabsf(arg) < 1e-10f) ? 1.0f : sinf((float)PI * arg) / ((float)PI * arg);
            h = sinc * cosf((float)PI * f_norm * t);
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
 * Initialize Watterson CPU processor.
 */
void* init_watterson_cpu(
    float sample_rate,
    int max_taps,
    int max_delay_samples,
    int max_samples_per_block,
    unsigned long seed
) {
    WattersonCPUState* state = new WattersonCPUState;

    state->sample_rate = sample_rate;
    state->max_taps = max_taps;
    state->max_delay = max_delay_samples;
    state->max_samples = max_samples_per_block;
    state->filter_len = 32;
    state->n_taps = 0;
    state->initialized = false;

    // Pre-allocate vectors
    state->taps.resize(max_taps);
    state->doppler_filters.resize(max_taps);
    state->noise_buffers.resize(max_taps);
    state->delay_buffers.resize(max_taps);
    state->old_gains.resize(max_taps);
    state->new_gains.resize(max_taps);
    state->rngs.resize(max_taps);

    for (int i = 0; i < max_taps; i++) {
        state->doppler_filters[i].resize(state->filter_len, 0.0f);
        state->noise_buffers[i].resize(state->filter_len, std::complex<float>(0.0f, 0.0f));
        state->delay_buffers[i].resize(max_delay_samples, std::complex<float>(0.0f, 0.0f));
        state->old_gains[i] = std::complex<float>(0.0f, 0.0f);
        state->new_gains[i] = std::complex<float>(0.0f, 0.0f);
        state->rngs[i].seed(seed + i);
    }

    state->initialized = true;
    return state;
}

/**
 * Configure Watterson taps.
 */
int configure_watterson_taps_cpu(
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
    WattersonCPUState* state = (WattersonCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_taps > state->max_taps) return -2;

    state->n_taps = n_taps;

    for (int i = 0; i < n_taps; i++) {
        state->taps[i].delay_samples = delays[i];
        state->taps[i].amplitude = amplitudes[i];
        state->taps[i].doppler_spread_hz = doppler_spreads[i];
        state->taps[i].spectrum_type = spectrum_types[i];
        state->taps[i].is_rician = (is_rician[i] != 0);
        state->taps[i].k_factor_linear = k_factors[i];

        // Compute Doppler filter for this tap
        compute_doppler_filter_cpu(
            state->doppler_filters[i],
            doppler_spreads[i],
            spectrum_types[i],
            update_rate
        );

        // Initialize noise buffer with complex Gaussian
        for (int k = 0; k < state->filter_len; k++) {
            float re = state->normal_dist(state->rngs[i]) * (float)INV_SQRT2;
            float im = state->normal_dist(state->rngs[i]) * (float)INV_SQRT2;
            state->noise_buffers[i][k] = std::complex<float>(re, im);
        }
    }

    return 0;
}

/**
 * Process samples through Watterson channel.
 */
int process_watterson_cpu(
    void* state_ptr,
    const float* input_real,
    const float* input_imag,
    int n_samples,
    float* output_real,
    float* output_imag
) {
    WattersonCPUState* state = (WattersonCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_samples > state->max_samples) return -2;
    if (state->n_taps == 0) return -3;

    int n_taps = state->n_taps;
    int filter_len = state->filter_len;

    // Save old gains
    for (int tap = 0; tap < n_taps; tap++) {
        state->old_gains[tap] = state->new_gains[tap];
    }

    // Generate new fading coefficients for each tap
    for (int tap = 0; tap < n_taps; tap++) {
        // Generate new noise sample
        float z_re = state->normal_dist(state->rngs[tap]) * (float)INV_SQRT2;
        float z_im = state->normal_dist(state->rngs[tap]) * (float)INV_SQRT2;

        // Shift noise buffer and insert new sample
        for (int k = 0; k < filter_len - 1; k++) {
            state->noise_buffers[tap][k] = state->noise_buffers[tap][k + 1];
        }
        state->noise_buffers[tap][filter_len - 1] = std::complex<float>(z_re, z_im);

        // Convolve with Doppler filter
        float sum_re = 0.0f, sum_im = 0.0f;
        for (int k = 0; k < filter_len; k++) {
            float h = state->doppler_filters[tap][k];
            sum_re += state->noise_buffers[tap][k].real() * h;
            sum_im += state->noise_buffers[tap][k].imag() * h;
        }

        // Apply Rician adjustment
        float scatter_coeff = 1.0f;
        float direct_re = 0.0f;

        if (state->taps[tap].is_rician && state->taps[tap].k_factor_linear > 0.0f) {
            float k = state->taps[tap].k_factor_linear;
            scatter_coeff = sqrtf(1.0f / (1.0f + k));
            direct_re = sqrtf(k / (1.0f + k)) * state->taps[tap].amplitude;
        }

        float amp = state->taps[tap].amplitude;
        state->new_gains[tap] = std::complex<float>(
            direct_re + scatter_coeff * sum_re * amp,
            scatter_coeff * sum_im * amp
        );
    }

    // Process samples through TDL with interpolated fading
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < n_samples; n++) {
        float t = (float)n / (float)n_samples;  // Interpolation weight

        float sum_re = 0.0f, sum_im = 0.0f;

        for (int tap = 0; tap < n_taps; tap++) {
            int delay = state->taps[tap].delay_samples;

            // Get delayed sample
            float del_re = 0.0f, del_im = 0.0f;
            int src_idx = n - delay;

            if (src_idx >= 0) {
                del_re = input_real[src_idx];
                del_im = input_imag[src_idx];
            } else {
                // Read from delay buffer
                int buf_idx = state->max_delay + src_idx;
                if (buf_idx >= 0 && buf_idx < state->max_delay) {
                    del_re = state->delay_buffers[tap][buf_idx].real();
                    del_im = state->delay_buffers[tap][buf_idx].imag();
                }
            }

            // Interpolate fading gain
            std::complex<float> g_old = state->old_gains[tap];
            std::complex<float> g_new = state->new_gains[tap];
            float gain_re = g_old.real() * (1.0f - t) + g_new.real() * t;
            float gain_im = g_old.imag() * (1.0f - t) + g_new.imag() * t;

            // Apply gain: out += delayed * gain
            sum_re += del_re * gain_re - del_im * gain_im;
            sum_im += del_re * gain_im + del_im * gain_re;
        }

        output_real[n] = sum_re;
        output_imag[n] = sum_im;
    }

    // Update delay buffers
    for (int tap = 0; tap < n_taps; tap++) {
        int copy_start = n_samples - state->max_delay;

        if (copy_start < 0) {
            // Input shorter than max_delay - shift existing and append
            int shift = n_samples;
            for (int i = 0; i < state->max_delay - shift; i++) {
                state->delay_buffers[tap][i] = state->delay_buffers[tap][i + shift];
            }
            for (int i = 0; i < n_samples; i++) {
                state->delay_buffers[tap][state->max_delay - shift + i] =
                    std::complex<float>(input_real[i], input_imag[i]);
            }
        } else {
            // Input longer than max_delay - take last max_delay samples
            for (int i = 0; i < state->max_delay; i++) {
                state->delay_buffers[tap][i] =
                    std::complex<float>(input_real[copy_start + i], input_imag[copy_start + i]);
            }
        }
    }

    return 0;
}

/**
 * Get current fading gains.
 */
int get_watterson_gains_cpu(
    void* state_ptr,
    float* gains_real,
    float* gains_imag,
    int max_taps
) {
    WattersonCPUState* state = (WattersonCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;

    int n = std::min(state->n_taps, max_taps);
    for (int i = 0; i < n; i++) {
        gains_real[i] = state->new_gains[i].real();
        gains_imag[i] = state->new_gains[i].imag();
    }

    return n;
}

/**
 * Reset Watterson channel state.
 */
void reset_watterson_cpu(void* state_ptr) {
    WattersonCPUState* state = (WattersonCPUState*)state_ptr;
    if (!state || !state->initialized) return;

    for (int tap = 0; tap < state->max_taps; tap++) {
        std::fill(state->delay_buffers[tap].begin(), state->delay_buffers[tap].end(),
                  std::complex<float>(0.0f, 0.0f));
        state->old_gains[tap] = std::complex<float>(0.0f, 0.0f);
        state->new_gains[tap] = std::complex<float>(0.0f, 0.0f);
    }
}

/**
 * Free Watterson CPU resources.
 */
void free_watterson_cpu(void* state_ptr) {
    WattersonCPUState* state = (WattersonCPUState*)state_ptr;
    if (!state) return;

    delete state;
}

} // extern "C"
