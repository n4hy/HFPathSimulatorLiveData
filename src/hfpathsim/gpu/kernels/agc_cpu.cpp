/**
 * CPU fallback implementation for AGC and Limiter.
 *
 * Optimized C++ implementation using OpenMP for parallelization.
 */

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * AGC CPU state.
 */
struct AGCCPUState {
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

    // State
    float envelope;
    float peak_envelope;
    int hold_counter;
    float gain_db;

    // Work buffers
    std::vector<float> envelope_buf;
    std::vector<float> gain_buf;

    bool initialized;
};

/**
 * Limiter CPU state.
 */
struct LimiterCPUState {
    float threshold;
    int mode;  // 0=hard, 1=soft, 2=cubic
    int max_samples;
    bool initialized;
};

extern "C" {

/**
 * Initialize AGC CPU processor.
 */
void* init_agc_cpu(
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
    AGCCPUState* state = new AGCCPUState;
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

    // Initialize state
    state->envelope = 0.0f;
    state->peak_envelope = 0.0f;
    state->hold_counter = 0;
    state->gain_db = 0.0f;

    // Allocate work buffers
    state->envelope_buf.resize(max_samples);
    state->gain_buf.resize(max_samples);

    state->initialized = true;
    return state;
}

/**
 * Process samples through AGC.
 */
int process_agc_cpu(
    void* state_ptr,
    const float* input_real,
    const float* input_imag,
    int n_samples,
    float* output_real,
    float* output_imag
) {
    AGCCPUState* state = (AGCCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_samples > state->max_samples) return -2;

    float target_db = 20.0f * log10f(state->target_level_linear);

    // Step 1: Compute envelope (parallel)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_samples; i++) {
        float re = input_real[i];
        float im = input_imag[i];
        state->envelope_buf[i] = sqrtf(re * re + im * im);
    }

    // Step 2: Smooth envelope and compute gain (sequential)
    float env = state->envelope;
    float peak = state->peak_envelope;
    int hold = state->hold_counter;
    float gain = state->gain_db;

    for (int i = 0; i < n_samples; i++) {
        float raw = state->envelope_buf[i];

        // Envelope smoothing with attack/release
        if (raw > env) {
            // Attack
            env += state->alpha_attack * (raw - env);
            peak = env;
            hold = state->hold_samples;
        } else {
            // Hold or release
            if (state->hang_agc && hold > 0) {
                hold--;
            } else {
                // Release
                env += state->alpha_release * (raw - env);
            }
        }

        // Compute desired gain
        float desired_gain;
        if (env > 1e-10f) {
            float env_db = 20.0f * log10f(env);
            desired_gain = target_db - env_db;
        } else {
            desired_gain = state->max_gain_db;
        }

        // Apply soft knee
        float diff = fabsf(desired_gain - gain);
        if (diff < state->soft_knee_db) {
            gain += 0.1f * (desired_gain - gain);
        } else {
            gain = desired_gain;
        }

        // Clamp gain
        if (gain > state->max_gain_db) gain = state->max_gain_db;
        if (gain < state->min_gain_db) gain = state->min_gain_db;

        state->gain_buf[i] = gain;
    }

    // Save state
    state->envelope = env;
    state->peak_envelope = peak;
    state->hold_counter = hold;
    state->gain_db = gain;

    // Step 3: Apply gain (parallel)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_samples; i++) {
        float gain_linear = powf(10.0f, state->gain_buf[i] / 20.0f);
        output_real[i] = input_real[i] * gain_linear;
        output_imag[i] = input_imag[i] * gain_linear;
    }

    return 0;
}

/**
 * Get current AGC gain.
 */
float get_agc_gain_db_cpu(void* state_ptr) {
    AGCCPUState* state = (AGCCPUState*)state_ptr;
    if (!state || !state->initialized) return 0.0f;
    return state->gain_db;
}

/**
 * Reset AGC state.
 */
void reset_agc_cpu(void* state_ptr) {
    AGCCPUState* state = (AGCCPUState*)state_ptr;
    if (!state || !state->initialized) return;

    state->envelope = 0.0f;
    state->peak_envelope = 0.0f;
    state->hold_counter = 0;
    state->gain_db = 0.0f;
}

/**
 * Free AGC CPU resources.
 */
void free_agc_cpu(void* state_ptr) {
    AGCCPUState* state = (AGCCPUState*)state_ptr;
    if (!state) return;

    delete state;
}

// ============================================================================
// Limiter Functions
// ============================================================================

/**
 * Initialize limiter CPU processor.
 */
void* init_limiter_cpu(float threshold_db, int mode, int max_samples) {
    LimiterCPUState* state = new LimiterCPUState;
    state->threshold = powf(10.0f, threshold_db / 20.0f);
    state->mode = mode;
    state->max_samples = max_samples;
    state->initialized = true;
    return state;
}

/**
 * Process samples through limiter.
 */
int process_limiter_cpu(
    void* state_ptr,
    const float* input_real,
    const float* input_imag,
    int n_samples,
    float* output_real,
    float* output_imag
) {
    LimiterCPUState* state = (LimiterCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_samples > state->max_samples) return -2;

    float threshold = state->threshold;
    int mode = state->mode;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_samples; i++) {
        float re = input_real[i];
        float im = input_imag[i];
        float mag = sqrtf(re * re + im * im);

        if (mag > 1e-10f) {
            float limited_mag;

            switch (mode) {
                case 0:  // Hard
                    limited_mag = (mag > threshold) ? threshold : mag;
                    break;

                case 1:  // Soft (tanh)
                    limited_mag = tanhf(mag / threshold) * threshold;
                    break;

                case 2:  // Cubic
                {
                    float normalized = mag / threshold;
                    float limited_norm;
                    if (normalized < 1.0f) {
                        limited_norm = normalized;
                    } else {
                        float x = normalized - 1.0f;
                        limited_norm = 1.0f + x / (1.0f + x * x);
                    }
                    limited_mag = limited_norm * threshold;
                    break;
                }

                default:
                    limited_mag = mag;
            }

            float scale = limited_mag / mag;
            output_real[i] = re * scale;
            output_imag[i] = im * scale;
        } else {
            output_real[i] = re;
            output_imag[i] = im;
        }
    }

    return 0;
}

/**
 * Update limiter parameters.
 */
void set_limiter_params_cpu(void* state_ptr, float threshold_db, int mode) {
    LimiterCPUState* state = (LimiterCPUState*)state_ptr;
    if (!state || !state->initialized) return;

    state->threshold = powf(10.0f, threshold_db / 20.0f);
    state->mode = mode;
}

/**
 * Free limiter CPU resources.
 */
void free_limiter_cpu(void* state_ptr) {
    LimiterCPUState* state = (LimiterCPUState*)state_ptr;
    if (!state) return;

    delete state;
}

} // extern "C"
