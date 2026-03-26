/**
 * CPU fallback implementation for polyphase resampling.
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

#define PI 3.14159265358979323846
#define TWO_PI 6.28318530717958647692

/**
 * Resampler CPU state.
 */
struct ResamplerCPUState {
    // Filter coefficients
    std::vector<float> filter;
    int filter_len;
    int taps_per_phase;

    // Configuration
    int upsample_factor;
    int downsample_factor;
    float input_rate;
    float output_rate;

    // Buffers
    std::vector<float> temp_real;
    std::vector<float> temp_imag;

    int max_input_samples;
    int max_output_samples;

    // State for block processing
    std::vector<float> history_real;
    std::vector<float> history_imag;
    int history_len;

    bool initialized;
};

/**
 * Design lowpass filter with Hamming window.
 */
static void design_lowpass_filter(float* filter, int len, float cutoff) {
    int half = len / 2;

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

    // Normalize
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

/**
 * Polyphase upsampling.
 */
static void polyphase_upsample_cpu(
    const float* input_real,
    const float* input_imag,
    float* output_real,
    float* output_imag,
    const float* filter,
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
}

/**
 * Polyphase downsampling.
 */
static void polyphase_downsample_cpu(
    const float* input_real,
    const float* input_imag,
    float* output_real,
    float* output_imag,
    const float* filter,
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
                re += input_real[in_idx] * h;
                im += input_imag[in_idx] * h;
            }
        }

        output_real[out_idx] = re;
        output_imag[out_idx] = im;
    }
}

/**
 * Cubic interpolation resampling.
 */
static void cubic_resample_cpu(
    const float* input_real,
    const float* input_imag,
    float* output_real,
    float* output_imag,
    int input_len,
    int output_len,
    float rate_ratio
) {
    #pragma omp parallel for schedule(static)
    for (int out_idx = 0; out_idx < output_len; out_idx++) {
        float in_pos = (float)out_idx * rate_ratio;
        int in_idx = (int)floorf(in_pos);
        float frac = in_pos - (float)in_idx;

        // Catmull-Rom coefficients
        float t = frac;
        float t2 = t * t;
        float t3 = t2 * t;

        float c0 = -0.5f * t3 + t2 - 0.5f * t;
        float c1 = 1.5f * t3 - 2.5f * t2 + 1.0f;
        float c2 = -1.5f * t3 + 2.0f * t2 + 0.5f * t;
        float c3 = 0.5f * t3 - 0.5f * t2;

        int i0 = std::max(0, std::min(in_idx - 1, input_len - 1));
        int i1 = std::max(0, std::min(in_idx, input_len - 1));
        int i2 = std::max(0, std::min(in_idx + 1, input_len - 1));
        int i3 = std::max(0, std::min(in_idx + 2, input_len - 1));

        output_real[out_idx] = c0 * input_real[i0] + c1 * input_real[i1] +
                               c2 * input_real[i2] + c3 * input_real[i3];
        output_imag[out_idx] = c0 * input_imag[i0] + c1 * input_imag[i1] +
                               c2 * input_imag[i2] + c3 * input_imag[i3];
    }
}

extern "C" {

/**
 * Initialize resampler.
 */
void* init_resampler_cpu(
    float input_rate,
    float output_rate,
    int max_input_samples
) {
    ResamplerCPUState* state = new ResamplerCPUState;
    state->input_rate = input_rate;
    state->output_rate = output_rate;
    state->max_input_samples = max_input_samples;
    state->initialized = false;

    // Compute rational approximation
    float ratio = output_rate / input_rate;
    if (ratio >= 1.0f) {
        state->upsample_factor = (int)roundf(ratio);
        state->downsample_factor = 1;
        if (fabsf(ratio - state->upsample_factor) > 0.01f) {
            state->upsample_factor = (int)roundf(ratio * 100);
            state->downsample_factor = 100;
        }
    } else {
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
    state->filter.resize(state->filter_len);

    float cutoff = 0.5f / (float)std::max(state->upsample_factor, state->downsample_factor);
    design_lowpass_filter(state->filter.data(), state->filter_len, cutoff);

    // Allocate buffers
    state->max_output_samples = (int)ceilf(max_input_samples * output_rate / input_rate) + 1024;
    int temp_size = max_input_samples * state->upsample_factor + state->filter_len;

    state->temp_real.resize(temp_size);
    state->temp_imag.resize(temp_size);

    state->history_len = state->filter_len;
    state->history_real.resize(state->history_len, 0.0f);
    state->history_imag.resize(state->history_len, 0.0f);

    state->initialized = true;
    return state;
}

/**
 * Resample signal.
 */
int resample_cpu(
    void* state_ptr,
    const float* input_real,
    const float* input_imag,
    int n_input,
    float* output_real,
    float* output_imag,
    int* n_output
) {
    ResamplerCPUState* state = (ResamplerCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_input > state->max_input_samples) return -2;

    if (state->upsample_factor > 1 && state->downsample_factor == 1) {
        // Pure upsampling
        int out_len = n_input * state->upsample_factor;
        polyphase_upsample_cpu(
            input_real, input_imag,
            output_real, output_imag,
            state->filter.data(),
            n_input, out_len,
            state->upsample_factor,
            state->filter_len,
            state->taps_per_phase
        );
        *n_output = out_len;

    } else if (state->upsample_factor == 1 && state->downsample_factor > 1) {
        // Pure downsampling
        int out_len = n_input / state->downsample_factor;
        polyphase_downsample_cpu(
            input_real, input_imag,
            output_real, output_imag,
            state->filter.data(),
            n_input, out_len,
            state->downsample_factor,
            state->filter_len
        );
        *n_output = out_len;

    } else if (state->upsample_factor > 1 && state->downsample_factor > 1) {
        // Rational resampling
        int temp_len = n_input * state->upsample_factor;
        int out_len = temp_len / state->downsample_factor;

        polyphase_upsample_cpu(
            input_real, input_imag,
            state->temp_real.data(), state->temp_imag.data(),
            state->filter.data(),
            n_input, temp_len,
            state->upsample_factor,
            state->filter_len,
            state->taps_per_phase
        );

        polyphase_downsample_cpu(
            state->temp_real.data(), state->temp_imag.data(),
            output_real, output_imag,
            state->filter.data(),
            temp_len, out_len,
            state->downsample_factor,
            state->filter_len
        );

        *n_output = out_len;

    } else {
        // No resampling
        memcpy(output_real, input_real, n_input * sizeof(float));
        memcpy(output_imag, input_imag, n_input * sizeof(float));
        *n_output = n_input;
    }

    return 0;
}

/**
 * Arbitrary rate resample.
 */
int resample_arbitrary_cpu(
    void* state_ptr,
    const float* input_real,
    const float* input_imag,
    int n_input,
    float* output_real,
    float* output_imag,
    int n_output
) {
    ResamplerCPUState* state = (ResamplerCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;

    float rate_ratio = (float)n_input / (float)n_output;
    cubic_resample_cpu(
        input_real, input_imag,
        output_real, output_imag,
        n_input, n_output, rate_ratio
    );

    return 0;
}

/**
 * Get expected output length.
 */
int get_output_length_cpu(void* state_ptr, int n_input) {
    ResamplerCPUState* state = (ResamplerCPUState*)state_ptr;
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
 * Reset resampler state.
 */
void reset_resampler_cpu(void* state_ptr) {
    ResamplerCPUState* state = (ResamplerCPUState*)state_ptr;
    if (!state || !state->initialized) return;

    std::fill(state->history_real.begin(), state->history_real.end(), 0.0f);
    std::fill(state->history_imag.begin(), state->history_imag.end(), 0.0f);
}

/**
 * Free resampler resources.
 */
void free_resampler_cpu(void* state_ptr) {
    ResamplerCPUState* state = (ResamplerCPUState*)state_ptr;
    if (!state) return;
    delete state;
}

} // extern "C"
