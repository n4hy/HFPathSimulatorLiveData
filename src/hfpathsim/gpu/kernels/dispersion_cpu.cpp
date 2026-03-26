/**
 * CPU fallback implementation for ionospheric dispersion filtering.
 *
 * Implements chirp all-pass filter using FFT-based overlap-save convolution.
 */

#include <cmath>
#include <cstring>
#include <complex>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#define PI 3.14159265358979323846
#define TWO_PI 6.28318530717958647692

typedef std::complex<float> Complex;

/**
 * Simple in-place FFT (Cooley-Tukey radix-2).
 */
static void fft_inplace(Complex* data, int N, bool inverse) {
    // Bit-reversal permutation
    for (int i = 1, j = 0; i < N; i++) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }

    // Cooley-Tukey FFT
    float sign = inverse ? 1.0f : -1.0f;
    for (int len = 2; len <= N; len <<= 1) {
        float ang = sign * TWO_PI / len;
        Complex wlen(cosf(ang), sinf(ang));
        for (int i = 0; i < N; i += len) {
            Complex w(1.0f, 0.0f);
            for (int j = 0; j < len / 2; j++) {
                Complex u = data[i + j];
                Complex v = data[i + j + len / 2] * w;
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        float scale = 1.0f / N;
        for (int i = 0; i < N; i++) {
            data[i] *= scale;
        }
    }
}

/**
 * Dispersion filter CPU state.
 */
struct DispersionCPUState {
    float sample_rate;
    float d_us_per_MHz;
    float d_chirp;

    std::vector<Complex> filter_time;
    std::vector<Complex> filter_freq;
    int filter_len;

    int fft_size;
    int overlap;
    int valid_per_block;

    std::vector<Complex> input_padded;
    std::vector<Complex> input_freq;
    std::vector<Complex> output_freq;
    std::vector<Complex> output_time;

    int max_samples;
    bool initialized;
    bool filter_valid;
};

/**
 * Generate chirp filter.
 */
static void generate_chirp_filter(
    Complex* filter,
    int filter_len,
    float d_chirp,
    float fs,
    float window_alpha
) {
    int half = filter_len / 2;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < filter_len; i++) {
        float t = (float)(i - half) / fs;
        float phase = PI * t * t / d_chirp;

        float re = cosf(phase);
        float im = sinf(phase);

        // Tukey window
        float win = 1.0f;
        float pos = (float)i / (float)(filter_len - 1);
        if (pos < window_alpha / 2.0f) {
            win = 0.5f * (1.0f + cosf(TWO_PI * (pos / window_alpha - 0.5f)));
        } else if (pos > 1.0f - window_alpha / 2.0f) {
            win = 0.5f * (1.0f + cosf(TWO_PI * ((pos - 1.0f) / window_alpha + 0.5f)));
        }

        filter[i] = Complex(re * win, im * win);
    }
}

extern "C" {

/**
 * Initialize dispersion filter.
 */
void* init_dispersion_cpu(
    float sample_rate,
    int max_samples
) {
    DispersionCPUState* state = new DispersionCPUState;
    state->sample_rate = sample_rate;
    state->max_samples = max_samples;
    state->d_us_per_MHz = 0.0f;
    state->filter_valid = false;
    state->initialized = false;

    // Compute FFT size
    int min_fft = 4096;
    state->fft_size = min_fft;
    while (state->fft_size < max_samples + 2048) {
        state->fft_size *= 2;
    }

    state->filter_len = 2001;
    state->overlap = state->filter_len - 1;
    state->valid_per_block = state->fft_size - state->overlap;

    // Allocate buffers
    state->filter_time.resize(state->fft_size, Complex(0, 0));
    state->filter_freq.resize(state->fft_size, Complex(0, 0));
    state->input_padded.resize(state->fft_size, Complex(0, 0));
    state->input_freq.resize(state->fft_size, Complex(0, 0));
    state->output_freq.resize(state->fft_size, Complex(0, 0));
    state->output_time.resize(state->fft_size, Complex(0, 0));

    state->initialized = true;
    return state;
}

/**
 * Set dispersion coefficient.
 */
int set_dispersion_coefficient_cpu(
    void* state_ptr,
    float d_us_per_MHz
) {
    DispersionCPUState* state = (DispersionCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;

    if (fabsf(d_us_per_MHz - state->d_us_per_MHz) < 0.001f && state->filter_valid) {
        return 0;
    }

    state->d_us_per_MHz = d_us_per_MHz;

    // Handle zero dispersion
    if (fabsf(d_us_per_MHz) < 0.01f) {
        std::fill(state->filter_time.begin(), state->filter_time.end(), Complex(0, 0));
        state->filter_time[state->filter_len / 2] = Complex(1.0f, 0.0f);

        std::copy(state->filter_time.begin(), state->filter_time.end(), state->filter_freq.begin());
        fft_inplace(state->filter_freq.data(), state->fft_size, false);

        state->filter_valid = true;
        return 0;
    }

    // Convert units
    state->d_chirp = d_us_per_MHz * 1e-12f;

    // Compute filter length
    float bandwidth = state->sample_rate / 2.0f;
    float spread_s = fabsf(d_us_per_MHz) * (bandwidth / 1e6f) * 1e-6f;
    int needed_len = (int)ceilf(spread_s * state->sample_rate * 4.0f);
    needed_len = (needed_len | 1);

    state->filter_len = std::min(needed_len, state->fft_size / 2);
    state->filter_len = std::max(state->filter_len, 17);
    if (state->filter_len % 2 == 0) state->filter_len++;

    state->overlap = state->filter_len - 1;
    state->valid_per_block = state->fft_size - state->overlap;

    // Generate chirp filter
    std::fill(state->filter_time.begin(), state->filter_time.end(), Complex(0, 0));
    generate_chirp_filter(state->filter_time.data(), state->filter_len,
                          state->d_chirp, state->sample_rate, 0.2f);

    // Normalize
    float energy = 0.0f;
    for (int i = 0; i < state->filter_len; i++) {
        energy += std::norm(state->filter_time[i]);
    }
    if (energy > 0) {
        float scale = 1.0f / sqrtf(energy);
        for (int i = 0; i < state->filter_len; i++) {
            state->filter_time[i] *= scale;
        }
    }

    // FFT of filter
    std::copy(state->filter_time.begin(), state->filter_time.end(), state->filter_freq.begin());
    fft_inplace(state->filter_freq.data(), state->fft_size, false);

    state->filter_valid = true;
    return 0;
}

/**
 * Apply dispersion.
 */
int apply_dispersion_cpu(
    void* state_ptr,
    const float* input_real,
    const float* input_imag,
    int n_samples,
    float* output_real,
    float* output_imag
) {
    DispersionCPUState* state = (DispersionCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (!state->filter_valid) return -2;
    if (n_samples > state->max_samples) return -3;

    // Trivial case
    if (fabsf(state->d_us_per_MHz) < 0.01f) {
        memcpy(output_real, input_real, n_samples * sizeof(float));
        memcpy(output_imag, input_imag, n_samples * sizeof(float));
        return 0;
    }

    // Single-block for short signals
    if (n_samples <= state->valid_per_block) {
        std::fill(state->input_padded.begin(), state->input_padded.end(), Complex(0, 0));

        for (int i = 0; i < n_samples; i++) {
            state->input_padded[state->overlap + i] = Complex(input_real[i], input_imag[i]);
        }

        // FFT
        std::copy(state->input_padded.begin(), state->input_padded.end(), state->input_freq.begin());
        fft_inplace(state->input_freq.data(), state->fft_size, false);

        // Multiply
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < state->fft_size; i++) {
            state->output_freq[i] = state->input_freq[i] * state->filter_freq[i];
        }

        // IFFT
        std::copy(state->output_freq.begin(), state->output_freq.end(), state->output_time.begin());
        fft_inplace(state->output_time.data(), state->fft_size, true);

        // Extract valid samples
        for (int i = 0; i < n_samples; i++) {
            output_real[i] = state->output_time[state->overlap + i].real();
            output_imag[i] = state->output_time[state->overlap + i].imag();
        }

        return 0;
    }

    // Overlap-save for longer signals
    std::vector<Complex> h_output(n_samples);
    int output_pos = 0;
    int input_pos = 0;

    while (output_pos < n_samples) {
        std::fill(state->input_padded.begin(), state->input_padded.end(), Complex(0, 0));

        int samples_to_copy = std::min(state->valid_per_block, n_samples - input_pos);
        for (int i = 0; i < samples_to_copy; i++) {
            state->input_padded[state->overlap + i] = Complex(input_real[input_pos + i],
                                                               input_imag[input_pos + i]);
        }

        // FFT
        std::copy(state->input_padded.begin(), state->input_padded.end(), state->input_freq.begin());
        fft_inplace(state->input_freq.data(), state->fft_size, false);

        // Multiply
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < state->fft_size; i++) {
            state->output_freq[i] = state->input_freq[i] * state->filter_freq[i];
        }

        // IFFT
        std::copy(state->output_freq.begin(), state->output_freq.end(), state->output_time.begin());
        fft_inplace(state->output_time.data(), state->fft_size, true);

        // Copy valid portion
        int valid_to_copy = std::min(state->valid_per_block, n_samples - output_pos);
        for (int i = 0; i < valid_to_copy; i++) {
            h_output[output_pos + i] = state->output_time[state->overlap + i];
        }

        input_pos += state->valid_per_block;
        output_pos += valid_to_copy;
    }

    // Copy to output
    for (int i = 0; i < n_samples; i++) {
        output_real[i] = h_output[i].real();
        output_imag[i] = h_output[i].imag();
    }

    return 0;
}

/**
 * Apply inverse dispersion.
 */
int apply_inverse_dispersion_cpu(
    void* state_ptr,
    const float* input_real,
    const float* input_imag,
    int n_samples,
    float* output_real,
    float* output_imag
) {
    DispersionCPUState* state = (DispersionCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;

    float orig_d = state->d_us_per_MHz;
    set_dispersion_coefficient_cpu(state_ptr, -orig_d);

    int ret = apply_dispersion_cpu(state_ptr, input_real, input_imag, n_samples,
                                   output_real, output_imag);

    set_dispersion_coefficient_cpu(state_ptr, orig_d);
    return ret;
}

/**
 * Get filter length.
 */
int get_dispersion_filter_length_cpu(void* state_ptr) {
    DispersionCPUState* state = (DispersionCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    return state->filter_len;
}

/**
 * Reset dispersion filter.
 */
void reset_dispersion_cpu(void* state_ptr) {
    DispersionCPUState* state = (DispersionCPUState*)state_ptr;
    if (!state || !state->initialized) return;

    state->filter_valid = false;
    state->d_us_per_MHz = 0.0f;
}

/**
 * Free dispersion filter.
 */
void free_dispersion_cpu(void* state_ptr) {
    DispersionCPUState* state = (DispersionCPUState*)state_ptr;
    if (!state) return;
    delete state;
}

} // extern "C"
