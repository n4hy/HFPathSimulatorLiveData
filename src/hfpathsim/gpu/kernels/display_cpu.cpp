/**
 * CPU fallback implementation for GUI display computations.
 *
 * Optimized C++ with OpenMP parallelization for:
 * - dB conversion
 * - FFT shift
 * - Moving average smoothing
 * - Peak hold with decay
 * - 2D normalization
 */

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cfloat>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// State Structure
// ============================================================================

struct DisplayCPUState {
    std::vector<float> spectrum_buf;
    std::vector<float> peak_hold_buf;
    std::vector<float> smooth_buf;
    std::vector<float> scatter_buf;

    int max_spectrum_size;
    int max_scatter_size;
    bool initialized;
};

extern "C" {

// ============================================================================
// Initialization
// ============================================================================

void* init_display_cpu(int max_spectrum_size, int max_scatter_rows, int max_scatter_cols) {
    DisplayCPUState* state = new DisplayCPUState;

    state->max_spectrum_size = max_spectrum_size;
    state->max_scatter_size = max_scatter_rows * max_scatter_cols;

    state->spectrum_buf.resize(max_spectrum_size, 0.0f);
    state->peak_hold_buf.resize(max_spectrum_size, -200.0f);
    state->smooth_buf.resize(max_spectrum_size, 0.0f);
    state->scatter_buf.resize(state->max_scatter_size, 0.0f);

    state->initialized = true;
    return state;
}

// ============================================================================
// dB Conversion
// ============================================================================

int magnitude_to_db_cpu(
    void* state_ptr,
    const float* mag,
    float* db,
    int N,
    float eps,
    float min_db
) {
    DisplayCPUState* state = (DisplayCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        float val = mag[i] + eps;
        float db_val = 20.0f * log10f(val);
        db[i] = std::max(db_val, min_db);
    }

    return 0;
}

int power_to_db_cpu(
    void* state_ptr,
    const float* power,
    float* db,
    int N,
    float eps,
    float min_db
) {
    DisplayCPUState* state = (DisplayCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        float val = power[i] + eps;
        float db_val = 10.0f * log10f(val);
        db[i] = std::max(db_val, min_db);
    }

    return 0;
}

// ============================================================================
// FFT Shift
// ============================================================================

int fftshift_cpu(
    void* state_ptr,
    const float* input,
    float* output,
    int N
) {
    DisplayCPUState* state = (DisplayCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;

    int half = N / 2;
    int offset = N & 1;  // 1 if odd, 0 if even

    // Copy second half to first half of output
    memcpy(output, input + half + offset, (half) * sizeof(float));

    // Copy first half to second half of output
    memcpy(output + half, input, (half + offset) * sizeof(float));

    return 0;
}

// ============================================================================
// Moving Average
// ============================================================================

int moving_average_cpu(
    void* state_ptr,
    const float* input,
    float* output,
    int N,
    int window_size
) {
    DisplayCPUState* state = (DisplayCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;

    int half_win = window_size / 2;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        int count = 0;

        for (int k = -half_win; k <= half_win; k++) {
            int idx = i + k;
            if (idx >= 0 && idx < N) {
                sum += input[idx];
                count++;
            }
        }

        output[i] = sum / (float)count;
    }

    return 0;
}

// ============================================================================
// Peak Hold
// ============================================================================

int peak_hold_cpu(
    void* state_ptr,
    const float* current,
    float* peak,
    int N,
    float decay_rate
) {
    DisplayCPUState* state = (DisplayCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        float decayed = peak[i] - decay_rate;
        peak[i] = std::max(decayed, current[i]);
    }

    return 0;
}

// ============================================================================
// Exponential Smoothing
// ============================================================================

int exponential_smooth_cpu(
    void* state_ptr,
    const float* current,
    float* smoothed,
    int N,
    float alpha
) {
    DisplayCPUState* state = (DisplayCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;

    float one_minus_alpha = 1.0f - alpha;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        smoothed[i] = alpha * current[i] + one_minus_alpha * smoothed[i];
    }

    return 0;
}

// ============================================================================
// Scattering Normalization
// ============================================================================

int normalize_scattering_cpu(
    void* state_ptr,
    const float* S,
    float* S_norm,
    int rows,
    int cols,
    float min_clip_db
) {
    DisplayCPUState* state = (DisplayCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;

    int total = rows * cols;
    float eps = 1e-10f;

    // Find max in dB
    float max_db = -FLT_MAX;
    for (int i = 0; i < total; i++) {
        float db = 10.0f * log10f(S[i] + eps);
        if (db > max_db) max_db = db;
    }

    // Normalize
    float range = -min_clip_db;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < total; i++) {
        float db_val = 10.0f * log10f(S[i] + eps);
        db_val = db_val - max_db;
        db_val = std::max(db_val, min_clip_db);
        db_val = std::min(db_val, 0.0f);
        S_norm[i] = (db_val - min_clip_db) / range;
    }

    return 0;
}

// ============================================================================
// 2D Transpose
// ============================================================================

int transpose_2d_cpu(
    void* state_ptr,
    const float* input,
    float* output,
    int rows,
    int cols
) {
    DisplayCPUState* state = (DisplayCPUState*)state_ptr;
    if (!state || !state->initialized) return -1;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            output[c * rows + r] = input[r * cols + c];
        }
    }

    return 0;
}

// ============================================================================
// Utility
// ============================================================================

void reset_peak_hold_cpu(void* state_ptr, int N, float initial_value) {
    DisplayCPUState* state = (DisplayCPUState*)state_ptr;
    if (!state || !state->initialized) return;

    std::fill(state->peak_hold_buf.begin(),
              state->peak_hold_buf.begin() + std::min(N, state->max_spectrum_size),
              initial_value);
}

void free_display_cpu(void* state_ptr) {
    DisplayCPUState* state = (DisplayCPUState*)state_ptr;
    if (!state) return;
    delete state;
}

} // extern "C"
