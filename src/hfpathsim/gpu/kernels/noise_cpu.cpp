/**
 * CPU fallback implementation for noise generation.
 *
 * Optimized C++ implementation using OpenMP for parallelization.
 */

#include <cmath>
#include <cstring>
#include <random>
#include <complex>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#define PI 3.14159265358979323846
#define TWO_PI 6.28318530717958647692
#define INV_SQRT2 0.7071067811865476

/**
 * Noise generator CPU state.
 */
struct NoiseGenCPUState {
    std::vector<std::mt19937> rngs;
    std::normal_distribution<float> normal_dist;
    std::uniform_real_distribution<float> uniform_dist;

    std::vector<float> spectrum_shape;
    std::vector<std::complex<float>> noise_buffer;

    int max_samples;
    float sample_rate;
    bool has_spectrum_shape;
    bool initialized;
};

extern "C" {

/**
 * Initialize noise generator.
 */
void* init_noise_gen_cpu(
    float sample_rate,
    int max_samples,
    unsigned long seed
) {
    NoiseGenCPUState* state = new NoiseGenCPUState;
    state->max_samples = max_samples;
    state->sample_rate = sample_rate;
    state->has_spectrum_shape = false;
    state->initialized = false;

    // Initialize RNGs (one per potential thread for thread safety)
    int n_threads = 1;
    #ifdef _OPENMP
    n_threads = omp_get_max_threads();
    #endif

    state->rngs.resize(n_threads);
    for (int i = 0; i < n_threads; i++) {
        state->rngs[i].seed(seed + i);
    }

    state->spectrum_shape.resize(max_samples, 1.0f);
    state->noise_buffer.resize(max_samples);

    state->initialized = true;
    return state;
}

/**
 * Generate AWGN samples.
 */
int generate_awgn_cpu(
    void* state_ptr,
    float noise_power,
    int n_samples,
    float* noise_real,
    float* noise_imag
) {
    NoiseGenCPUState* state = (NoiseGenCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_samples > state->max_samples) return -2;

    float sigma = sqrtf(noise_power / 2.0f);

    #pragma omp parallel
    {
        int thread_id = 0;
        #ifdef _OPENMP
        thread_id = omp_get_thread_num();
        #endif
        std::mt19937& rng = state->rngs[thread_id];
        std::normal_distribution<float>& dist = state->normal_dist;

        #pragma omp for schedule(static)
        for (int i = 0; i < n_samples; i++) {
            noise_real[i] = dist(rng) * sigma;
            noise_imag[i] = dist(rng) * sigma;
        }
    }

    return 0;
}

/**
 * Generate atmospheric noise samples.
 */
int generate_atmospheric_cpu(
    void* state_ptr,
    float noise_power,
    float vd,
    int n_samples,
    float* noise_real,
    float* noise_imag
) {
    NoiseGenCPUState* state = (NoiseGenCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_samples > state->max_samples) return -2;

    float p_impulse = vd * 0.1f;

    #pragma omp parallel
    {
        int thread_id = 0;
        #ifdef _OPENMP
        thread_id = omp_get_thread_num();
        #endif
        std::mt19937& rng = state->rngs[thread_id];
        std::normal_distribution<float>& normal = state->normal_dist;
        std::uniform_real_distribution<float>& uniform = state->uniform_dist;

        #pragma omp for schedule(static)
        for (int i = 0; i < n_samples; i++) {
            float u = uniform(rng);
            float sigma;

            if (u < p_impulse) {
                // Impulsive sample with log-normal envelope
                float log_sigma = 1.0f + 2.0f * vd;
                float log_mean = logf(sqrtf(noise_power)) + log_sigma * log_sigma;
                float log_val = normal(rng) * log_sigma + log_mean;
                sigma = expf(log_val);
            } else {
                // Gaussian background
                sigma = sqrtf(noise_power / 2.0f);
            }

            noise_real[i] = normal(rng) * sigma;
            noise_imag[i] = normal(rng) * sigma;
        }
    }

    return 0;
}

/**
 * Generate impulse noise samples.
 */
int generate_impulse_cpu(
    void* state_ptr,
    float impulse_rate,
    float impulse_amplitude,
    float noise_floor,
    int n_samples,
    float* noise_real,
    float* noise_imag
) {
    NoiseGenCPUState* state = (NoiseGenCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_samples > state->max_samples) return -2;

    float sigma = sqrtf(noise_floor / 2.0f);

    #pragma omp parallel
    {
        int thread_id = 0;
        #ifdef _OPENMP
        thread_id = omp_get_thread_num();
        #endif
        std::mt19937& rng = state->rngs[thread_id];
        std::normal_distribution<float>& normal = state->normal_dist;
        std::uniform_real_distribution<float>& uniform = state->uniform_dist;

        #pragma omp for schedule(static)
        for (int i = 0; i < n_samples; i++) {
            float re = normal(rng) * sigma;
            float im = normal(rng) * sigma;

            float u = uniform(rng);
            if (u < impulse_rate) {
                float phase = uniform(rng) * (float)TWO_PI;
                float amp = impulse_amplitude * (0.5f + uniform(rng));
                re += amp * cosf(phase);
                im += amp * sinf(phase);
            }

            noise_real[i] = re;
            noise_imag[i] = im;
        }
    }

    return 0;
}

/**
 * Set spectrum shape for colored noise.
 */
int set_noise_spectrum_shape_cpu(
    void* state_ptr,
    const float* spectrum_shape,
    int n_bins
) {
    NoiseGenCPUState* state = (NoiseGenCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_bins > state->max_samples) return -2;

    std::copy(spectrum_shape, spectrum_shape + n_bins, state->spectrum_shape.begin());
    state->has_spectrum_shape = true;

    return 0;
}

/**
 * Simple DFT for colored noise (no external FFT library dependency).
 */
static void simple_fft(
    std::complex<float>* data,
    int N,
    bool inverse
) {
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
        float ang = sign * (float)TWO_PI / len;
        std::complex<float> wlen(cosf(ang), sinf(ang));
        for (int i = 0; i < N; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (int j = 0; j < len / 2; j++) {
                std::complex<float> u = data[i + j];
                std::complex<float> v = data[i + j + len / 2] * w;
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
 * Generate colored (shaped spectrum) noise.
 */
int generate_colored_noise_cpu(
    void* state_ptr,
    float noise_power,
    int n_samples,
    float* noise_real,
    float* noise_imag
) {
    NoiseGenCPUState* state = (NoiseGenCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_samples > state->max_samples) return -2;
    if (!state->has_spectrum_shape) return -3;

    // Generate white noise
    std::mt19937& rng = state->rngs[0];
    std::normal_distribution<float>& normal = state->normal_dist;

    for (int i = 0; i < n_samples; i++) {
        float re = normal(rng) * (float)INV_SQRT2;
        float im = normal(rng) * (float)INV_SQRT2;
        state->noise_buffer[i] = std::complex<float>(re, im);
    }

    // FFT to frequency domain
    simple_fft(state->noise_buffer.data(), n_samples, false);

    // Apply spectrum shaping
    for (int i = 0; i < n_samples; i++) {
        state->noise_buffer[i] *= state->spectrum_shape[i];
    }

    // IFFT back to time domain
    simple_fft(state->noise_buffer.data(), n_samples, true);

    // Compute current power and normalize
    float power_sum = 0.0f;
    for (int i = 0; i < n_samples; i++) {
        float re = state->noise_buffer[i].real();
        float im = state->noise_buffer[i].imag();
        power_sum += re * re + im * im;
    }
    float current_power = power_sum / n_samples;
    float scale = sqrtf(noise_power / current_power);

    for (int i = 0; i < n_samples; i++) {
        noise_real[i] = state->noise_buffer[i].real() * scale;
        noise_imag[i] = state->noise_buffer[i].imag() * scale;
    }

    return 0;
}

/**
 * Add noise to signal.
 */
int add_noise_cpu(
    void* state_ptr,
    float* signal_real,
    float* signal_imag,
    int noise_type,
    float param1,
    float param2,
    float param3,
    int n_samples
) {
    NoiseGenCPUState* state = (NoiseGenCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (n_samples > state->max_samples) return -2;

    // Allocate temp buffers
    std::vector<float> noise_real(n_samples);
    std::vector<float> noise_imag(n_samples);

    int ret;
    switch (noise_type) {
        case 0:  // AWGN
            ret = generate_awgn_cpu(state_ptr, param1, n_samples,
                                   noise_real.data(), noise_imag.data());
            break;
        case 1:  // Atmospheric
            ret = generate_atmospheric_cpu(state_ptr, param1, param2, n_samples,
                                          noise_real.data(), noise_imag.data());
            break;
        case 2:  // Impulse
            ret = generate_impulse_cpu(state_ptr, param1, param2, param3, n_samples,
                                      noise_real.data(), noise_imag.data());
            break;
        default:
            return -3;
    }

    if (ret != 0) return ret;

    // Add to signal
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_samples; i++) {
        signal_real[i] += noise_real[i];
        signal_imag[i] += noise_imag[i];
    }

    return 0;
}

/**
 * Reset RNG state.
 */
void reset_noise_gen_cpu(void* state_ptr, unsigned long seed) {
    NoiseGenCPUState* state = (NoiseGenCPUState*)state_ptr;
    if (!state || !state->initialized) return;

    for (size_t i = 0; i < state->rngs.size(); i++) {
        state->rngs[i].seed(seed + i);
    }
}

/**
 * Free noise generator resources.
 */
void free_noise_gen_cpu(void* state_ptr) {
    NoiseGenCPUState* state = (NoiseGenCPUState*)state_ptr;
    if (!state) return;

    delete state;
}

} // extern "C"
