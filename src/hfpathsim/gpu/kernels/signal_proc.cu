/**
 * CUDA kernels for signal processing operations.
 *
 * Implements GPU-accelerated overlap-save convolution and
 * related signal processing for real-time channel simulation.
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

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

    window[idx] = 0.5f * (1.0f - cosf(2.0f * 3.14159265f * idx / (N - 1)));
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
    float* db_dev;

    cudaMalloc(&signal_dev, N * sizeof(cufftComplex));
    cudaMalloc(&power_dev, N * sizeof(float));
    cudaMalloc(&db_dev, N * sizeof(float));

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
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    cufftExecC2C(plan, signal_dev, signal_dev, CUFFT_FORWARD);

    // Compute power spectrum
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    compute_power_spectrum<<<blocks, threads>>>(signal_dev, power_dev, N);
    power_to_db<<<blocks, threads>>>(power_dev, db_dev, reference, -120.0f, N);

    // Copy result
    cudaMemcpy(power_db, db_dev, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cufftDestroy(plan);
    cudaFree(signal_dev);
    cudaFree(power_dev);
    cudaFree(db_dev);

    return 0;
}

} // extern "C"
