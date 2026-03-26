/**
 * CUDA kernels for signal processing operations.
 *
 * Implements GPU-accelerated overlap-save convolution and
 * related signal processing for real-time channel simulation.
 *
 * Phase 5 enhancements:
 * - Batched cuFFT for higher throughput
 * - GPU-accelerated Doppler fading generation
 * - Efficient spectrum computation
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <curand_kernel.h>
#include <math.h>

#define PI 3.14159265358979323846f

/**
 * Element-wise complex multiplication for filtering.
 */
__global__ void complex_multiply(
    const cufftComplex* a,
    const cufftComplex* b,
    cufftComplex* c,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float re = a[idx].x * b[idx].x - a[idx].y * b[idx].y;
    float im = a[idx].x * b[idx].y + a[idx].y * b[idx].x;

    c[idx].x = re;
    c[idx].y = im;
}

/**
 * Copy input block with zero-padding for overlap-save.
 */
__global__ void prepare_input_block(
    const cufftComplex* input,   // Full input signal
    cufftComplex* block,         // Output block (zero-padded)
    int input_offset,            // Offset in input
    int input_len,               // Total input length
    int block_size,              // FFT block size
    int overlap                  // Overlap samples
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= block_size) return;

    int src_idx = input_offset + idx - overlap;

    if (src_idx >= 0 && src_idx < input_len) {
        block[idx] = input[src_idx];
    } else {
        block[idx] = make_cuFloatComplex(0.0f, 0.0f);
    }
}

/**
 * Extract valid output from IFFT result, discarding overlap.
 */
__global__ void extract_output_block(
    const cufftComplex* ifft_result,   // IFFT output
    cufftComplex* output,               // Output buffer
    int output_offset,                  // Offset in output
    int overlap,                        // Samples to discard
    int output_size                     // Samples to keep
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;

    output[output_offset + idx] = ifft_result[overlap + idx];
}

/**
 * Scale FFT output (cuFFT doesn't normalize).
 */
__global__ void scale_ifft_output(
    cufftComplex* data,
    float scale,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    data[idx].x *= scale;
    data[idx].y *= scale;
}

/**
 * Compute power spectrum (magnitude squared).
 */
__global__ void compute_power_spectrum(
    const cufftComplex* spectrum,
    float* power,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    power[idx] = spectrum[idx].x * spectrum[idx].x +
                 spectrum[idx].y * spectrum[idx].y;
}

/**
 * Convert power to dB scale.
 */
__global__ void power_to_db(
    const float* power,
    float* db,
    float reference,
    float min_db,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float p = power[idx] / reference;
    if (p > 0.0f) {
        db[idx] = 10.0f * log10f(p);
        if (db[idx] < min_db) db[idx] = min_db;
    } else {
        db[idx] = min_db;
    }
}

/**
 * Apply window function to time-domain block.
 */
__global__ void apply_window(
    cufftComplex* data,
    const float* window,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    data[idx].x *= window[idx];
    data[idx].y *= window[idx];
}

/**
 * Generate Hann window.
 */
__global__ void generate_hann_window(
    float* window,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    window[idx] = 0.5f * (1.0f - cosf(2.0f * PI * idx / (N - 1)));
}

/**
 * Prepare multiple input blocks for batched overlap-save.
 * Each thread handles one sample across all batches.
 */
__global__ void prepare_input_blocks_batched(
    const cufftComplex* input,   // Full input signal
    cufftComplex* blocks,        // Output blocks [batch_size * block_size]
    int input_len,               // Total input length
    int block_size,              // FFT block size
    int overlap,                 // Overlap samples
    int output_size,             // block_size - overlap
    int batch_size               // Number of blocks to prepare
) {
    int block_idx = blockIdx.y;
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_idx >= batch_size || sample_idx >= block_size) return;

    int input_offset = block_idx * output_size;
    int src_idx = input_offset + sample_idx - overlap;

    cufftComplex value;
    if (src_idx >= 0 && src_idx < input_len) {
        value = input[src_idx];
    } else {
        value = make_cuFloatComplex(0.0f, 0.0f);
    }

    blocks[block_idx * block_size + sample_idx] = value;
}

/**
 * Element-wise complex multiply for batched processing.
 * Applies the same transfer function H to all batched blocks.
 */
__global__ void complex_multiply_batched(
    const cufftComplex* X,       // Batched input spectra [batch_size * block_size]
    const cufftComplex* H,       // Single transfer function [block_size]
    cufftComplex* Y,             // Batched output spectra [batch_size * block_size]
    int block_size,
    int batch_size
) {
    int block_idx = blockIdx.y;
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_idx >= batch_size || sample_idx >= block_size) return;

    int idx = block_idx * block_size + sample_idx;
    cufftComplex x = X[idx];
    cufftComplex h = H[sample_idx];

    float re = x.x * h.x - x.y * h.y;
    float im = x.x * h.y + x.y * h.x;

    Y[idx] = make_cuFloatComplex(re, im);
}

/**
 * Scale and extract output from batched IFFT results.
 */
__global__ void extract_output_batched(
    const cufftComplex* ifft_results,  // Batched IFFT outputs [batch_size * block_size]
    cufftComplex* output,              // Output buffer
    int block_size,
    int overlap,
    int output_size,
    int batch_size,
    float scale,
    int output_base                    // Starting offset in output buffer
) {
    int block_idx = blockIdx.y;
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_idx >= batch_size || sample_idx >= output_size) return;

    int src_idx = block_idx * block_size + overlap + sample_idx;
    int dst_idx = output_base + block_idx * output_size + sample_idx;

    cufftComplex val = ifft_results[src_idx];
    output[dst_idx] = make_cuFloatComplex(val.x * scale, val.y * scale);
}

/**
 * Apply Gaussian Doppler spectrum filter to noise.
 * This shapes complex Gaussian noise with a Doppler power spectrum.
 */
__global__ void apply_doppler_spectrum(
    cufftComplex* noise_spectrum,   // FFT of complex Gaussian noise [N]
    float doppler_spread_hz,        // Two-sided Doppler spread
    float sample_rate,              // Sample rate in Hz
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Compute frequency for this bin
    float freq;
    if (idx <= N / 2) {
        freq = (float)idx * sample_rate / (float)N;
    } else {
        freq = ((float)idx - (float)N) * sample_rate / (float)N;
    }

    // Gaussian Doppler filter
    // sigma_f = doppler_spread / 2.355 (FWHM to sigma)
    float sigma = doppler_spread_hz / 2.355f;
    float filter = expf(-0.5f * (freq * freq) / (sigma * sigma));

    noise_spectrum[idx].x *= filter;
    noise_spectrum[idx].y *= filter;
}

/**
 * Normalize complex array to unit variance.
 */
__global__ void normalize_complex(
    cufftComplex* data,
    float norm_factor,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    data[idx].x *= norm_factor;
    data[idx].y *= norm_factor;
}

/**
 * Initialize cuRAND states for Doppler fading generation.
 */
__global__ void init_doppler_rng(
    curandState* states,
    unsigned long seed,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    curand_init(seed, idx, 0, &states[idx]);
}

/**
 * Generate complex Gaussian noise samples.
 */
__global__ void generate_complex_gaussian(
    curandState* states,
    cufftComplex* noise,
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

// Host wrapper functions

extern "C" {

/**
 * Overlap-save processor state.
 */
struct OverlapSaveState {
    cufftHandle fft_plan;
    cufftHandle ifft_plan;
    cufftComplex* input_block;
    cufftComplex* output_block;
    cufftComplex* H_freq;
    cufftComplex* X_freq;
    cufftComplex* Y_freq;
    int block_size;
    int overlap;
    int output_size;
    bool initialized;
};

/**
 * Batched overlap-save processor state for high throughput.
 */
struct OverlapSaveBatchedState {
    cufftHandle fft_plan_batch;    // Batched forward FFT plan
    cufftHandle ifft_plan_batch;   // Batched inverse FFT plan
    cufftComplex* input_blocks;    // Batched input blocks [batch_size * block_size]
    cufftComplex* X_freq;          // Batched input spectra [batch_size * block_size]
    cufftComplex* Y_freq;          // Batched output spectra [batch_size * block_size]
    cufftComplex* output_blocks;   // Batched IFFT outputs [batch_size * block_size]
    cufftComplex* H_freq;          // Transfer function [block_size]
    int block_size;
    int overlap;
    int output_size;
    int batch_size;
    bool initialized;
};

/**
 * Doppler fading generator state.
 */
struct DopplerFadingState {
    curandState* rng_states;
    cufftComplex* noise_buffer;
    cufftComplex* spectrum_buffer;
    cufftHandle fft_plan;
    cufftHandle ifft_plan;
    int N;
    bool initialized;
};

/**
 * Initialize overlap-save processor.
 */
void* init_overlap_save(int block_size, int overlap) {
    OverlapSaveState* state = new OverlapSaveState;
    state->block_size = block_size;
    state->overlap = overlap;
    state->output_size = block_size - overlap;
    state->initialized = false;

    // Create FFT plans
    if (cufftPlan1d(&state->fft_plan, block_size, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
        delete state;
        return nullptr;
    }

    if (cufftPlan1d(&state->ifft_plan, block_size, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
        cufftDestroy(state->fft_plan);
        delete state;
        return nullptr;
    }

    // Allocate GPU memory
    cudaMalloc(&state->input_block, block_size * sizeof(cufftComplex));
    cudaMalloc(&state->output_block, block_size * sizeof(cufftComplex));
    cudaMalloc(&state->H_freq, block_size * sizeof(cufftComplex));
    cudaMalloc(&state->X_freq, block_size * sizeof(cufftComplex));
    cudaMalloc(&state->Y_freq, block_size * sizeof(cufftComplex));

    state->initialized = true;
    return state;
}

/**
 * Set transfer function for overlap-save processing.
 */
int set_transfer_function(
    void* state_ptr,
    const float* H_real,
    const float* H_imag,
    int N
) {
    OverlapSaveState* state = (OverlapSaveState*)state_ptr;
    if (!state || !state->initialized) return -1;

    // Pack into complex array
    cufftComplex* H_temp = new cufftComplex[N];
    for (int i = 0; i < N; i++) {
        H_temp[i].x = H_real[i];
        H_temp[i].y = H_imag[i];
    }

    cudaMemcpy(state->H_freq, H_temp, N * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    delete[] H_temp;

    return 0;
}

/**
 * Process signal through channel using overlap-save.
 */
int process_overlap_save(
    void* state_ptr,
    const float* input_real,
    const float* input_imag,
    int input_len,
    float* output_real,
    float* output_imag
) {
    OverlapSaveState* state = (OverlapSaveState*)state_ptr;
    if (!state || !state->initialized) return -1;

    int block_size = state->block_size;
    int overlap = state->overlap;
    int output_size = state->output_size;

    // Copy input to device
    cufftComplex* input_dev;
    cufftComplex* output_dev;

    int n_blocks = (input_len + output_size - 1) / output_size;
    int total_output = n_blocks * output_size;

    cudaMalloc(&input_dev, input_len * sizeof(cufftComplex));
    cudaMalloc(&output_dev, total_output * sizeof(cufftComplex));

    // Pack input
    cufftComplex* input_temp = new cufftComplex[input_len];
    for (int i = 0; i < input_len; i++) {
        input_temp[i].x = input_real[i];
        input_temp[i].y = input_imag[i];
    }
    cudaMemcpy(input_dev, input_temp, input_len * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    delete[] input_temp;

    int threads = 256;
    int blocks = (block_size + threads - 1) / threads;
    float scale = 1.0f / block_size;

    // Process blocks
    for (int b = 0; b < n_blocks; b++) {
        int input_offset = b * output_size;

        // Prepare input block
        prepare_input_block<<<blocks, threads>>>(
            input_dev, state->input_block,
            input_offset, input_len,
            block_size, overlap
        );

        // Forward FFT
        cufftExecC2C(state->fft_plan, state->input_block, state->X_freq, CUFFT_FORWARD);

        // Multiply by transfer function
        complex_multiply<<<blocks, threads>>>(
            state->X_freq, state->H_freq, state->Y_freq, block_size
        );

        // Inverse FFT
        cufftExecC2C(state->ifft_plan, state->Y_freq, state->output_block, CUFFT_INVERSE);

        // Scale IFFT output
        scale_ifft_output<<<blocks, threads>>>(state->output_block, scale, block_size);

        // Extract valid output
        int out_blocks = (output_size + threads - 1) / threads;
        extract_output_block<<<out_blocks, threads>>>(
            state->output_block, output_dev,
            b * output_size, overlap, output_size
        );
    }

    // Copy output back
    cufftComplex* output_temp = new cufftComplex[total_output];
    cudaMemcpy(output_temp, output_dev, total_output * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < input_len && i < total_output; i++) {
        output_real[i] = output_temp[i].x;
        output_imag[i] = output_temp[i].y;
    }

    delete[] output_temp;
    cudaFree(input_dev);
    cudaFree(output_dev);

    return 0;
}

/**
 * Free overlap-save processor resources.
 */
void free_overlap_save(void* state_ptr) {
    OverlapSaveState* state = (OverlapSaveState*)state_ptr;
    if (!state) return;

    if (state->initialized) {
        cufftDestroy(state->fft_plan);
        cufftDestroy(state->ifft_plan);
        cudaFree(state->input_block);
        cudaFree(state->output_block);
        cudaFree(state->H_freq);
        cudaFree(state->X_freq);
        cudaFree(state->Y_freq);
    }

    delete state;
}

/**
 * Compute power spectrum on GPU.
 */
int compute_spectrum_gpu(
    const float* signal_real,
    const float* signal_imag,
    int N,
    float* power_db,
    float reference
) {
    // Allocate GPU memory
    cufftComplex* signal_dev;
    float* power_dev;

    cudaError_t err;
    err = cudaMalloc(&signal_dev, N * sizeof(cufftComplex));
    if (err != cudaSuccess) return -1;
    err = cudaMalloc(&power_dev, N * sizeof(float));
    if (err != cudaSuccess) { cudaFree(signal_dev); return -2; }

    // Pack and copy signal
    cufftComplex* signal_temp = new cufftComplex[N];
    for (int i = 0; i < N; i++) {
        signal_temp[i].x = signal_real[i];
        signal_temp[i].y = signal_imag[i];
    }
    cudaMemcpy(signal_dev, signal_temp, N * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    delete[] signal_temp;

    // Create FFT plan and execute
    cufftHandle plan;
    cufftResult fft_result;
    fft_result = cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    if (fft_result != CUFFT_SUCCESS) {
        cudaFree(signal_dev); cudaFree(power_dev);
        return -4;
    }
    fft_result = cufftExecC2C(plan, signal_dev, signal_dev, CUFFT_FORWARD);
    if (fft_result != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        cudaFree(signal_dev); cudaFree(power_dev);
        return -5;
    }

    // Synchronize after FFT
    cudaDeviceSynchronize();

    // Compute power spectrum
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    compute_power_spectrum<<<blocks, threads>>>(signal_dev, power_dev, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cufftDestroy(plan);
        cudaFree(signal_dev); cudaFree(power_dev);
        return -6;
    }
    cudaDeviceSynchronize();

    // Copy power to CPU and convert to dB
    float* power_host = new float[N];
    cudaMemcpy(power_host, power_dev, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        float p = power_host[i] / reference;
        if (p > 0.0f) {
            power_db[i] = 10.0f * log10f(p);
            if (power_db[i] < -120.0f) power_db[i] = -120.0f;
        } else {
            power_db[i] = -120.0f;
        }
    }

    delete[] power_host;

    // Cleanup
    cufftDestroy(plan);
    cudaFree(signal_dev);
    cudaFree(power_dev);

    return 0;
}

// ============================================================================
// Batched Overlap-Save Functions for High Throughput
// ============================================================================

/**
 * Initialize batched overlap-save processor.
 *
 * Uses cufftPlanMany to create batched FFT plans for processing
 * multiple blocks simultaneously, achieving higher throughput.
 */
void* init_overlap_save_batched(int block_size, int overlap, int batch_size) {
    OverlapSaveBatchedState* state = new OverlapSaveBatchedState;
    state->block_size = block_size;
    state->overlap = overlap;
    state->output_size = block_size - overlap;
    state->batch_size = batch_size;
    state->initialized = false;

    // Create batched FFT plans using cufftPlanMany
    // This is significantly more efficient than calling single FFTs in a loop
    int n[] = {block_size};

    cufftResult result = cufftPlanMany(
        &state->fft_plan_batch,
        1,                    // rank
        n,                    // dimensions
        NULL,                 // inembed (NULL = contiguous)
        1,                    // istride
        block_size,           // idist (distance between batches)
        NULL,                 // onembed
        1,                    // ostride
        block_size,           // odist
        CUFFT_C2C,
        batch_size
    );

    if (result != CUFFT_SUCCESS) {
        delete state;
        return nullptr;
    }

    result = cufftPlanMany(
        &state->ifft_plan_batch,
        1, n,
        NULL, 1, block_size,
        NULL, 1, block_size,
        CUFFT_C2C,
        batch_size
    );

    if (result != CUFFT_SUCCESS) {
        cufftDestroy(state->fft_plan_batch);
        delete state;
        return nullptr;
    }

    // Allocate GPU memory for batched processing
    size_t batch_buffer_size = batch_size * block_size * sizeof(cufftComplex);

    cudaMalloc(&state->input_blocks, batch_buffer_size);
    cudaMalloc(&state->X_freq, batch_buffer_size);
    cudaMalloc(&state->Y_freq, batch_buffer_size);
    cudaMalloc(&state->output_blocks, batch_buffer_size);
    cudaMalloc(&state->H_freq, block_size * sizeof(cufftComplex));

    state->initialized = true;
    return state;
}

/**
 * Set transfer function for batched overlap-save processing.
 */
int set_transfer_function_batched(
    void* state_ptr,
    const float* H_real,
    const float* H_imag,
    int N
) {
    OverlapSaveBatchedState* state = (OverlapSaveBatchedState*)state_ptr;
    if (!state || !state->initialized) return -1;
    if (N != state->block_size) return -2;

    // Pack into complex array
    cufftComplex* H_temp = new cufftComplex[N];
    for (int i = 0; i < N; i++) {
        H_temp[i].x = H_real[i];
        H_temp[i].y = H_imag[i];
    }

    cudaMemcpy(state->H_freq, H_temp, N * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    delete[] H_temp;

    return 0;
}

/**
 * Process signal through channel using batched overlap-save.
 *
 * Processes batch_size blocks simultaneously using batched cuFFT,
 * achieving significantly higher throughput than sequential processing.
 */
int process_overlap_save_batched(
    void* state_ptr,
    const float* input_real,
    const float* input_imag,
    int input_len,
    float* output_real,
    float* output_imag
) {
    OverlapSaveBatchedState* state = (OverlapSaveBatchedState*)state_ptr;
    if (!state || !state->initialized) return -1;

    int block_size = state->block_size;
    int overlap = state->overlap;
    int output_size = state->output_size;
    int batch_size = state->batch_size;

    int n_blocks = (input_len + output_size - 1) / output_size;
    int total_output = n_blocks * output_size;

    // Allocate device memory for input and output
    cufftComplex* input_dev;
    cufftComplex* output_dev;

    cudaMalloc(&input_dev, input_len * sizeof(cufftComplex));
    cudaMalloc(&output_dev, total_output * sizeof(cufftComplex));

    // Pack and copy input
    cufftComplex* input_temp = new cufftComplex[input_len];
    for (int i = 0; i < input_len; i++) {
        input_temp[i].x = input_real[i];
        input_temp[i].y = input_imag[i];
    }
    cudaMemcpy(input_dev, input_temp, input_len * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    delete[] input_temp;

    int threads = 256;
    float scale = 1.0f / block_size;

    // Process in batches
    for (int batch_start = 0; batch_start < n_blocks; batch_start += batch_size) {
        int current_batch = min(batch_size, n_blocks - batch_start);

        // Prepare input blocks (2D grid: samples x batches)
        dim3 block_dim(threads);
        dim3 grid_dim((block_size + threads - 1) / threads, current_batch);

        prepare_input_blocks_batched<<<grid_dim, block_dim>>>(
            input_dev, state->input_blocks,
            input_len, block_size, overlap, output_size, current_batch
        );

        // Batched forward FFT
        cufftExecC2C(state->fft_plan_batch, state->input_blocks, state->X_freq, CUFFT_FORWARD);

        // Batched complex multiply with transfer function
        complex_multiply_batched<<<grid_dim, block_dim>>>(
            state->X_freq, state->H_freq, state->Y_freq,
            block_size, current_batch
        );

        // Batched inverse FFT
        cufftExecC2C(state->ifft_plan_batch, state->Y_freq, state->output_blocks, CUFFT_INVERSE);

        // Extract and scale output
        dim3 out_grid((output_size + threads - 1) / threads, current_batch);
        extract_output_batched<<<out_grid, block_dim>>>(
            state->output_blocks, output_dev,
            block_size, overlap, output_size, current_batch, scale,
            batch_start * output_size
        );
    }

    // Copy output back
    cufftComplex* output_temp = new cufftComplex[total_output];
    cudaMemcpy(output_temp, output_dev, total_output * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < input_len && i < total_output; i++) {
        output_real[i] = output_temp[i].x;
        output_imag[i] = output_temp[i].y;
    }

    delete[] output_temp;
    cudaFree(input_dev);
    cudaFree(output_dev);

    return 0;
}

/**
 * Free batched overlap-save processor resources.
 */
void free_overlap_save_batched(void* state_ptr) {
    OverlapSaveBatchedState* state = (OverlapSaveBatchedState*)state_ptr;
    if (!state) return;

    if (state->initialized) {
        cufftDestroy(state->fft_plan_batch);
        cufftDestroy(state->ifft_plan_batch);
        cudaFree(state->input_blocks);
        cudaFree(state->X_freq);
        cudaFree(state->Y_freq);
        cudaFree(state->output_blocks);
        cudaFree(state->H_freq);
    }

    delete state;
}

/**
 * Get batch size from batched state.
 */
int get_batch_size(void* state_ptr) {
    OverlapSaveBatchedState* state = (OverlapSaveBatchedState*)state_ptr;
    if (!state || !state->initialized) return -1;
    return state->batch_size;
}

// ============================================================================
// Doppler Fading Generation
// ============================================================================

/**
 * Initialize Doppler fading generator.
 */
void* init_doppler_fading(int N, unsigned long seed) {
    DopplerFadingState* state = new DopplerFadingState;
    state->N = N;
    state->initialized = false;

    // Allocate GPU memory
    cudaMalloc(&state->rng_states, N * sizeof(curandState));
    cudaMalloc(&state->noise_buffer, N * sizeof(cufftComplex));
    cudaMalloc(&state->spectrum_buffer, N * sizeof(cufftComplex));

    // Create FFT plans for shaping
    if (cufftPlan1d(&state->fft_plan, N, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
        cudaFree(state->rng_states);
        cudaFree(state->noise_buffer);
        cudaFree(state->spectrum_buffer);
        delete state;
        return nullptr;
    }

    if (cufftPlan1d(&state->ifft_plan, N, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
        cufftDestroy(state->fft_plan);
        cudaFree(state->rng_states);
        cudaFree(state->noise_buffer);
        cudaFree(state->spectrum_buffer);
        delete state;
        return nullptr;
    }

    // Initialize RNG states
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    init_doppler_rng<<<blocks, threads>>>(state->rng_states, seed, N);

    cudaDeviceSynchronize();
    state->initialized = true;
    return state;
}

/**
 * Generate Doppler-shaped fading samples.
 *
 * Generates complex fading coefficients with a Gaussian Doppler
 * power spectrum, suitable for HF channel simulation.
 */
int generate_doppler_fading_gpu(
    void* state_ptr,
    float doppler_spread_hz,
    float sample_rate,
    float* fading_real,
    float* fading_imag
) {
    DopplerFadingState* state = (DopplerFadingState*)state_ptr;
    if (!state || !state->initialized) return -1;

    int N = state->N;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Generate complex Gaussian noise
    generate_complex_gaussian<<<blocks, threads>>>(
        state->rng_states, state->noise_buffer, N
    );

    // FFT to frequency domain
    cufftExecC2C(state->fft_plan, state->noise_buffer, state->spectrum_buffer, CUFFT_FORWARD);

    // Apply Doppler spectrum shaping
    apply_doppler_spectrum<<<blocks, threads>>>(
        state->spectrum_buffer, doppler_spread_hz, sample_rate, N
    );

    // IFFT back to time domain
    cufftExecC2C(state->ifft_plan, state->spectrum_buffer, state->noise_buffer, CUFFT_INVERSE);

    // Normalize
    float scale = 1.0f / (float)N;
    normalize_complex<<<blocks, threads>>>(state->noise_buffer, scale, N);

    // Copy result back to host
    cufftComplex* result_temp = new cufftComplex[N];
    cudaMemcpy(result_temp, state->noise_buffer, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        fading_real[i] = result_temp[i].x;
        fading_imag[i] = result_temp[i].y;
    }

    delete[] result_temp;
    return 0;
}

/**
 * Free Doppler fading generator resources.
 */
void free_doppler_fading(void* state_ptr) {
    DopplerFadingState* state = (DopplerFadingState*)state_ptr;
    if (!state) return;

    if (state->initialized) {
        cufftDestroy(state->fft_plan);
        cufftDestroy(state->ifft_plan);
        cudaFree(state->rng_states);
        cudaFree(state->noise_buffer);
        cudaFree(state->spectrum_buffer);
    }

    delete state;
}

/**
 * Generate Doppler fading without persistent state (convenience function).
 *
 * For one-shot generation; less efficient than using persistent state
 * but simpler to use.
 */
int generate_doppler_fading_oneshot(
    float doppler_spread_hz,
    float sample_rate,
    int N,
    float* fading_real,
    float* fading_imag,
    unsigned long seed
) {
    void* state = init_doppler_fading(N, seed);
    if (!state) return -1;

    int result = generate_doppler_fading_gpu(
        state, doppler_spread_hz, sample_rate,
        fading_real, fading_imag
    );

    free_doppler_fading(state);
    return result;
}

} // extern "C"
