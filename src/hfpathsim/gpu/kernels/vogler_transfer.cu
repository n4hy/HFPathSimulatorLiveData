/**
 * CUDA kernels for Vogler-Hoffmeyer IPM transfer function computation.
 *
 * Based on NTIA TR-88-240: "A full-wave calculation of ionospheric
 * Doppler spread and its application to HF channel modeling."
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <math.h>

// Constants
#define PI 3.14159265358979323846
#define EULER_GAMMA 0.5772156649015329

/**
 * Complex gamma function approximation using Stirling's formula.
 *
 * For |z| > 10, uses asymptotic expansion.
 * For smaller |z|, uses reflection and recurrence.
 */
__device__ cuDoubleComplex cgamma(cuDoubleComplex z) {
    double x = cuCreal(z);
    double y = cuCimag(z);

    // Handle special cases
    if (x <= 0.0 && y == 0.0 && fmod(-x, 1.0) == 0.0) {
        // Poles at non-positive integers
        return make_cuDoubleComplex(INFINITY, 0.0);
    }

    // Use reflection formula for Re(z) < 0.5
    if (x < 0.5) {
        // gamma(z) = pi / (sin(pi*z) * gamma(1-z))
        cuDoubleComplex one_minus_z = make_cuDoubleComplex(1.0 - x, -y);
        cuDoubleComplex gamma_1mz = cgamma(one_minus_z);

        // sin(pi * z)
        double sin_re = sin(PI * x) * cosh(PI * y);
        double sin_im = cos(PI * x) * sinh(PI * y);
        cuDoubleComplex sin_piz = make_cuDoubleComplex(sin_re, sin_im);

        // pi / (sin(pi*z) * gamma(1-z))
        cuDoubleComplex denom = cuCmul(sin_piz, gamma_1mz);
        double denom_mag_sq = cuCreal(denom) * cuCreal(denom) +
                              cuCimag(denom) * cuCimag(denom);
        return make_cuDoubleComplex(
            PI * cuCreal(denom) / denom_mag_sq,
            -PI * cuCimag(denom) / denom_mag_sq
        );
    }

    // Lanczos approximation coefficients (g=7)
    const double p[] = {
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    };

    z = make_cuDoubleComplex(x - 1.0, y);

    cuDoubleComplex sum = make_cuDoubleComplex(p[0], 0.0);
    for (int i = 1; i < 9; i++) {
        cuDoubleComplex term = make_cuDoubleComplex(
            cuCreal(z) + i,
            cuCimag(z)
        );
        double mag_sq = cuCreal(term) * cuCreal(term) +
                        cuCimag(term) * cuCimag(term);
        cuDoubleComplex inv_term = make_cuDoubleComplex(
            cuCreal(term) / mag_sq,
            -cuCimag(term) / mag_sq
        );
        sum = cuCadd(sum, make_cuDoubleComplex(
            p[i] * cuCreal(inv_term),
            p[i] * cuCimag(inv_term)
        ));
    }

    cuDoubleComplex t = make_cuDoubleComplex(
        cuCreal(z) + 7.5,
        cuCimag(z)
    );

    // sqrt(2*pi)
    double sqrt_2pi = 2.5066282746310005;

    // t^(z+0.5)
    double t_mag = sqrt(cuCreal(t) * cuCreal(t) + cuCimag(t) * cuCimag(t));
    double t_arg = atan2(cuCimag(t), cuCreal(t));

    double exp_re = cuCreal(z) + 0.5;
    double exp_im = cuCimag(z);

    double pow_mag = pow(t_mag, exp_re) * exp(-exp_im * t_arg);
    double pow_arg = exp_re * t_arg + exp_im * log(t_mag);

    cuDoubleComplex t_pow = make_cuDoubleComplex(
        pow_mag * cos(pow_arg),
        pow_mag * sin(pow_arg)
    );

    // exp(-t)
    cuDoubleComplex exp_neg_t = make_cuDoubleComplex(
        exp(-cuCreal(t)) * cos(-cuCimag(t)),
        exp(-cuCreal(t)) * sin(-cuCimag(t))
    );

    // Combine
    cuDoubleComplex result = cuCmul(
        make_cuDoubleComplex(sqrt_2pi, 0.0),
        cuCmul(t_pow, cuCmul(exp_neg_t, sum))
    );

    return result;
}

/**
 * Compute Vogler reflection coefficient R(omega).
 *
 * R(ω) = Γ(1-iσω)Γ(1/2-χ+iσω)Γ(1/2+χ+iσω)e^(-iωt₀)
 *        ─────────────────────────────────────────────
 *        Γ(1+iσω)Γ(1/2-χ)Γ(1/2+χ)
 */
__global__ void compute_reflection_coefficient(
    const double* freq,          // Frequency array [N]
    double fc,                   // Critical frequency (Hz)
    double sigma,                // Layer thickness parameter
    double chi,                  // Penetration parameter
    double t0,                   // Base propagation delay (s)
    cuDoubleComplex* R_out,      // Reflection coefficient [N]
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    double omega = freq[idx] / fc;  // Normalized frequency

    // Numerator gamma functions
    cuDoubleComplex g1_arg = make_cuDoubleComplex(1.0, -sigma * omega);
    cuDoubleComplex g2_arg = make_cuDoubleComplex(0.5 - chi, sigma * omega);
    cuDoubleComplex g3_arg = make_cuDoubleComplex(0.5 + chi, sigma * omega);

    cuDoubleComplex g1 = cgamma(g1_arg);
    cuDoubleComplex g2 = cgamma(g2_arg);
    cuDoubleComplex g3 = cgamma(g3_arg);

    cuDoubleComplex num = cuCmul(g1, cuCmul(g2, g3));

    // Denominator gamma functions
    cuDoubleComplex g4_arg = make_cuDoubleComplex(1.0, sigma * omega);
    cuDoubleComplex g4 = cgamma(g4_arg);
    cuDoubleComplex g5 = cgamma(make_cuDoubleComplex(0.5 - chi, 0.0));
    cuDoubleComplex g6 = cgamma(make_cuDoubleComplex(0.5 + chi, 0.0));

    cuDoubleComplex den = cuCmul(g4, cuCmul(g5, g6));

    // R = num / den
    double den_mag_sq = cuCreal(den) * cuCreal(den) +
                        cuCimag(den) * cuCimag(den);
    cuDoubleComplex R = make_cuDoubleComplex(
        (cuCreal(num) * cuCreal(den) + cuCimag(num) * cuCimag(den)) / den_mag_sq,
        (cuCimag(num) * cuCreal(den) - cuCreal(num) * cuCimag(den)) / den_mag_sq
    );

    // Phase from propagation delay: e^(-i * 2pi * f * t0)
    double phase_angle = -2.0 * PI * freq[idx] * t0;
    cuDoubleComplex phase = make_cuDoubleComplex(
        cos(phase_angle),
        sin(phase_angle)
    );

    R_out[idx] = cuCmul(R, phase);
}

/**
 * Apply Gaussian fading to transfer function.
 */
__global__ void apply_gaussian_fading(
    cuDoubleComplex* H,          // Transfer function [N] (in/out)
    const double* freq,          // Frequency array [N]
    double doppler_spread,       // Doppler spread (Hz)
    double delay_spread,         // Delay spread (s)
    float* noise_real,           // Gaussian noise real part [N]
    float* noise_imag,           // Gaussian noise imag part [N]
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Doppler fading - modulate with filtered noise
    double doppler_filter = exp(-0.5 * pow(freq[idx] / doppler_spread, 2));

    cuDoubleComplex noise = make_cuDoubleComplex(
        noise_real[idx] * doppler_filter,
        noise_imag[idx] * doppler_filter
    );

    // Amplitude modulation
    double amp_mod = 1.0 + 0.3 * cuCreal(noise);
    H[idx] = make_cuDoubleComplex(
        cuCreal(H[idx]) * amp_mod,
        cuCimag(H[idx]) * amp_mod
    );
}

/**
 * Overlap-save convolution kernel.
 * Processes one block at a time.
 */
__global__ void overlap_save_multiply(
    const cufftComplex* X,       // Input spectrum [block_size]
    const cufftComplex* H,       // Transfer function [block_size]
    cufftComplex* Y,             // Output spectrum [block_size]
    int block_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= block_size) return;

    // Complex multiplication
    float re = X[idx].x * H[idx].x - X[idx].y * H[idx].y;
    float im = X[idx].x * H[idx].y + X[idx].y * H[idx].x;

    Y[idx].x = re;
    Y[idx].y = im;
}

/**
 * Copy overlap-save output, discarding overlap region.
 */
__global__ void copy_output_discard_overlap(
    const cufftComplex* y,       // IFFT result [block_size]
    cufftComplex* output,        // Output buffer
    int output_offset,           // Where to write in output
    int overlap,                 // Samples to discard
    int output_size              // Samples to copy
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;

    output[output_offset + idx] = y[overlap + idx];
}

// Host wrapper functions declared extern "C" for pybind11

extern "C" {

/**
 * Get CUDA device information.
 */
int get_cuda_device_info(
    char* name, int name_len,
    int* major, int* minor,
    size_t* total_mem,
    int* multiprocessors
) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);

    if (err != cudaSuccess) {
        return -1;
    }

    strncpy(name, prop.name, name_len - 1);
    name[name_len - 1] = '\0';

    *major = prop.major;
    *minor = prop.minor;
    *total_mem = prop.totalGlobalMem;
    *multiprocessors = prop.multiProcessorCount;

    return 0;
}

/**
 * Compute Vogler transfer function (host entry point).
 */
int vogler_transfer_function_cuda(
    const double* freq_host,
    int N,
    double fc,
    double sigma,
    double chi,
    double t0,
    double* R_real_out,
    double* R_imag_out
) {
    // Allocate device memory
    double* freq_dev;
    cuDoubleComplex* R_dev;

    cudaMalloc(&freq_dev, N * sizeof(double));
    cudaMalloc(&R_dev, N * sizeof(cuDoubleComplex));

    // Copy frequency array to device
    cudaMemcpy(freq_dev, freq_host, N * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    compute_reflection_coefficient<<<blocks, threads>>>(
        freq_dev, fc, sigma, chi, t0, R_dev, N
    );

    // Copy result back
    cuDoubleComplex* R_host = new cuDoubleComplex[N];
    cudaMemcpy(R_host, R_dev, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    // Extract real and imaginary parts
    for (int i = 0; i < N; i++) {
        R_real_out[i] = cuCreal(R_host[i]);
        R_imag_out[i] = cuCimag(R_host[i]);
    }

    // Cleanup
    delete[] R_host;
    cudaFree(freq_dev);
    cudaFree(R_dev);

    return 0;
}

} // extern "C"
