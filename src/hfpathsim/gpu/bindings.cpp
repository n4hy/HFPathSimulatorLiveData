/**
 * pybind11 bindings for HF Path Simulator GPU module.
 *
 * Exposes CUDA kernels to Python for:
 * - Vogler transfer function computation
 * - Channel application via overlap-save
 * - GPU device information
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <map>
#include <complex>

namespace py = pybind11;

// External CUDA functions
extern "C" {
    int get_cuda_device_info(
        char* name, int name_len,
        int* major, int* minor,
        size_t* total_mem,
        int* multiprocessors
    );

    int vogler_transfer_function_cuda(
        const double* freq_host,
        int N,
        double fc,
        double sigma,
        double chi,
        double t0,
        double* R_real_out,
        double* R_imag_out
    );

    // Overlap-save functions
    void* init_overlap_save(int block_size, int overlap);
    int set_transfer_function(void* state_ptr, const float* H_real, const float* H_imag, int N);
    int process_overlap_save(void* state_ptr,
                             const float* input_real, const float* input_imag, int input_len,
                             float* output_real, float* output_imag);
    void free_overlap_save(void* state_ptr);

    // Batched overlap-save functions (high throughput)
    void* init_overlap_save_batched(int block_size, int overlap, int batch_size);
    int set_transfer_function_batched(void* state_ptr, const float* H_real, const float* H_imag, int N);
    int process_overlap_save_batched(void* state_ptr,
                                     const float* input_real, const float* input_imag, int input_len,
                                     float* output_real, float* output_imag);
    void free_overlap_save_batched(void* state_ptr);
    int get_batch_size(void* state_ptr);

    // Fading generator functions
    void* init_fading_generator(int N, unsigned long seed);
    int generate_faded_channel(void* state_ptr,
                               const float* H_base_real, const float* H_base_imag,
                               float doppler_spread, float delay_spread, float sample_rate,
                               float* H_out_real, float* H_out_imag);
    void free_fading_generator(void* state_ptr);

    // Doppler fading generation
    void* init_doppler_fading(int N, unsigned long seed);
    int generate_doppler_fading_gpu(void* state_ptr,
                                    float doppler_spread_hz, float sample_rate,
                                    float* fading_real, float* fading_imag);
    void free_doppler_fading(void* state_ptr);
    int generate_doppler_fading_oneshot(float doppler_spread_hz, float sample_rate, int N,
                                        float* fading_real, float* fading_imag, unsigned long seed);

    // Spectrum computation
    int compute_spectrum_gpu(const float* signal_real, const float* signal_imag,
                             int N, float* power_db, float reference);

    // VH RF Chain - GPU implementation
    void* init_vh_rf_chain(int input_rate, int rf_rate, int max_input_samples,
                           float carrier_freq_hz, float coherence_time_sec,
                           float k_factor, unsigned long seed);
    int configure_vh_taps(void* state_ptr, const int* delays, const float* amplitudes,
                          const float* doppler_hz, int n_taps);
    int process_vh_rf_chain(void* state_ptr,
                            const float* input_real, const float* input_imag, int n_samples,
                            float* output_real, float* output_imag);
    void free_vh_rf_chain(void* state_ptr);
    void reset_vh_rf_chain(void* state_ptr);

    // VH RF Chain - CPU implementation (fallback)
    void* init_vh_rf_chain_cpu(int input_rate, int rf_rate, int max_input_samples,
                               float carrier_freq_hz, float coherence_time_sec,
                               float k_factor, unsigned long seed);
    int configure_vh_taps_cpu(void* state_ptr, const int* delays, const float* amplitudes,
                              const float* doppler_hz, int n_taps);
    int process_vh_rf_chain_cpu(void* state_ptr,
                                const float* input_real, const float* input_imag, int n_samples,
                                float* output_real, float* output_imag);
    void free_vh_rf_chain_cpu(void* state_ptr);
    void reset_vh_rf_chain_cpu(void* state_ptr);

    // Debug functions for GPU VH RF chain
    float debug_test_upsample(void* state_ptr, const float* input_real, const float* input_imag, int n_samples);
    float debug_test_resample_roundtrip(void* state_ptr, const float* input_real, const float* input_imag, int n_samples);
    int debug_get_filter(void* state_ptr, float* filter_out, int max_len);
    float debug_test_mixer_roundtrip(void* state_ptr, const float* input_real, const float* input_imag, int n_samples);
    float debug_test_tdl_unity(void* state_ptr, const float* input_real, const float* input_imag, int n_samples);
    void debug_get_fading_stats(void* state_ptr, int n_samples, float* mean_mag, float* mean_magsq);
    float debug_test_tdl_constant_fading(void* state_ptr, const float* input_real, const float* input_imag,
                                         int n_samples, float fading_re, float fading_im);
    void debug_power_stages(void* state_ptr, const float* input_real, const float* input_imag,
                            int n_samples, float* powers_out);

    // Watterson channel - GPU
    void* init_watterson_gpu(float sample_rate, int max_taps, int max_delay_samples,
                             int max_samples_per_block, unsigned long seed);
    int configure_watterson_taps_gpu(void* state_ptr, const int* delays, const float* amplitudes,
                                     const float* doppler_spreads, const int* spectrum_types,
                                     const int* is_rician, const float* k_factors,
                                     int n_taps, float update_rate);
    int process_watterson_gpu(void* state_ptr, const float* input_real, const float* input_imag,
                              int n_samples, float* output_real, float* output_imag);
    int get_watterson_gains_gpu(void* state_ptr, float* gains_real, float* gains_imag, int max_taps);
    void reset_watterson_gpu(void* state_ptr);
    void free_watterson_gpu(void* state_ptr);

    // Watterson channel - CPU fallback
    void* init_watterson_cpu(float sample_rate, int max_taps, int max_delay_samples,
                             int max_samples_per_block, unsigned long seed);
    int configure_watterson_taps_cpu(void* state_ptr, const int* delays, const float* amplitudes,
                                     const float* doppler_spreads, const int* spectrum_types,
                                     const int* is_rician, const float* k_factors,
                                     int n_taps, float update_rate);
    int process_watterson_cpu(void* state_ptr, const float* input_real, const float* input_imag,
                              int n_samples, float* output_real, float* output_imag);
    int get_watterson_gains_cpu(void* state_ptr, float* gains_real, float* gains_imag, int max_taps);
    void reset_watterson_cpu(void* state_ptr);
    void free_watterson_cpu(void* state_ptr);

    // AGC - GPU
    void* init_agc_gpu(float sample_rate, float attack_time_ms, float release_time_ms,
                       float hold_time_ms, bool hang_agc, float target_level_db,
                       float max_gain_db, float min_gain_db, float soft_knee_db, int max_samples);
    int process_agc_gpu(void* state_ptr, const float* input_real, const float* input_imag,
                        int n_samples, float* output_real, float* output_imag);
    float get_agc_gain_db(void* state_ptr);
    void reset_agc_gpu(void* state_ptr);
    void free_agc_gpu(void* state_ptr);

    // AGC - CPU fallback
    void* init_agc_cpu(float sample_rate, float attack_time_ms, float release_time_ms,
                       float hold_time_ms, bool hang_agc, float target_level_db,
                       float max_gain_db, float min_gain_db, float soft_knee_db, int max_samples);
    int process_agc_cpu(void* state_ptr, const float* input_real, const float* input_imag,
                        int n_samples, float* output_real, float* output_imag);
    float get_agc_gain_db_cpu(void* state_ptr);
    void reset_agc_cpu(void* state_ptr);
    void free_agc_cpu(void* state_ptr);

    // Limiter - GPU
    void* init_limiter_gpu(float threshold_db, int mode, int max_samples);
    int process_limiter_gpu(void* state_ptr, const float* input_real, const float* input_imag,
                            int n_samples, float* output_real, float* output_imag);
    void set_limiter_params_gpu(void* state_ptr, float threshold_db, int mode);
    void free_limiter_gpu(void* state_ptr);

    // Limiter - CPU fallback
    void* init_limiter_cpu(float threshold_db, int mode, int max_samples);
    int process_limiter_cpu(void* state_ptr, const float* input_real, const float* input_imag,
                            int n_samples, float* output_real, float* output_imag);
    void set_limiter_params_cpu(void* state_ptr, float threshold_db, int mode);
    void free_limiter_cpu(void* state_ptr);

    // Noise generator - GPU
    void* init_noise_gen_gpu(float sample_rate, int max_samples, unsigned long seed);
    int generate_awgn_gpu(void* state_ptr, float noise_power, int n_samples,
                          float* noise_real, float* noise_imag);
    int generate_atmospheric_gpu(void* state_ptr, float noise_power, float vd,
                                 int n_samples, float* noise_real, float* noise_imag);
    int generate_impulse_gpu(void* state_ptr, float impulse_rate, float impulse_amplitude,
                             float noise_floor, int n_samples, float* noise_real, float* noise_imag);
    int set_noise_spectrum_shape_gpu(void* state_ptr, const float* spectrum_shape, int n_bins);
    int generate_colored_noise_gpu(void* state_ptr, float noise_power, int n_samples,
                                   float* noise_real, float* noise_imag);
    int add_noise_gpu(void* state_ptr, float* signal_real, float* signal_imag,
                      int noise_type, float param1, float param2, float param3, int n_samples);
    void reset_noise_gen_gpu(void* state_ptr, unsigned long seed);
    void free_noise_gen_gpu(void* state_ptr);

    // Noise generator - CPU fallback
    void* init_noise_gen_cpu(float sample_rate, int max_samples, unsigned long seed);
    int generate_awgn_cpu(void* state_ptr, float noise_power, int n_samples,
                          float* noise_real, float* noise_imag);
    int generate_atmospheric_cpu(void* state_ptr, float noise_power, float vd,
                                 int n_samples, float* noise_real, float* noise_imag);
    int generate_impulse_cpu(void* state_ptr, float impulse_rate, float impulse_amplitude,
                             float noise_floor, int n_samples, float* noise_real, float* noise_imag);
    int set_noise_spectrum_shape_cpu(void* state_ptr, const float* spectrum_shape, int n_bins);
    int generate_colored_noise_cpu(void* state_ptr, float noise_power, int n_samples,
                                   float* noise_real, float* noise_imag);
    int add_noise_cpu(void* state_ptr, float* signal_real, float* signal_imag,
                      int noise_type, float param1, float param2, float param3, int n_samples);
    void reset_noise_gen_cpu(void* state_ptr, unsigned long seed);
    void free_noise_gen_cpu(void* state_ptr);
}

/**
 * Get GPU device information.
 */
std::map<std::string, py::object> get_device_info() {
    std::map<std::string, py::object> info;

    char name[256];
    int major, minor, multiprocessors;
    size_t total_mem;

    int ret = get_cuda_device_info(name, 256, &major, &minor, &total_mem, &multiprocessors);

    if (ret == 0) {
        info["name"] = py::str(name);
        info["compute_capability"] = py::str(std::to_string(major) + "." + std::to_string(minor));
        info["total_memory_gb"] = py::float_(total_mem / (1024.0 * 1024.0 * 1024.0));
        info["multiprocessors"] = py::int_(multiprocessors);
        info["backend"] = py::str("cuda");
    } else {
        info["name"] = py::str("No GPU detected");
        info["compute_capability"] = py::str("N/A");
        info["total_memory_gb"] = py::float_(0.0);
        info["multiprocessors"] = py::int_(0);
        info["backend"] = py::str("none");
    }

    return info;
}

/**
 * Compute Vogler transfer function.
 */
py::array_t<std::complex<float>> vogler_transfer_function(
    py::array_t<double> freq_hz,
    double foF2_mhz,
    double hmF2_km,
    double sigma,
    double chi,
    double t0_sec
) {
    py::buffer_info freq_buf = freq_hz.request();
    int N = freq_buf.size;
    double* freq_ptr = static_cast<double*>(freq_buf.ptr);

    // Critical frequency in Hz
    double fc = foF2_mhz * 1e6;

    // Allocate output arrays
    std::vector<double> R_real(N);
    std::vector<double> R_imag(N);

    // Call CUDA kernel
    int ret = vogler_transfer_function_cuda(
        freq_ptr, N, fc, sigma, chi, t0_sec,
        R_real.data(), R_imag.data()
    );

    if (ret != 0) {
        throw std::runtime_error("CUDA kernel execution failed");
    }

    // Create output array
    py::array_t<std::complex<float>> result(N);
    py::buffer_info result_buf = result.request();
    std::complex<float>* result_ptr = static_cast<std::complex<float>*>(result_buf.ptr);

    for (int i = 0; i < N; i++) {
        result_ptr[i] = std::complex<float>(R_real[i], R_imag[i]);
    }

    return result;
}

/**
 * Overlap-save processor wrapper class.
 */
class OverlapSaveProcessor {
public:
    OverlapSaveProcessor(int block_size, int overlap)
        : block_size_(block_size), overlap_(overlap), state_(nullptr)
    {
        state_ = init_overlap_save(block_size, overlap);
        if (!state_) {
            throw std::runtime_error("Failed to initialize overlap-save processor");
        }
    }

    ~OverlapSaveProcessor() {
        if (state_) {
            free_overlap_save(state_);
        }
    }

    void set_transfer_function(py::array_t<std::complex<float>> H) {
        py::buffer_info buf = H.request();
        int N = buf.size;
        std::complex<float>* H_ptr = static_cast<std::complex<float>*>(buf.ptr);

        std::vector<float> H_real(N), H_imag(N);
        for (int i = 0; i < N; i++) {
            H_real[i] = H_ptr[i].real();
            H_imag[i] = H_ptr[i].imag();
        }

        int ret = ::set_transfer_function(state_, H_real.data(), H_imag.data(), N);
        if (ret != 0) {
            throw std::runtime_error("Failed to set transfer function");
        }
    }

    py::array_t<std::complex<float>> process(py::array_t<std::complex<float>> input) {
        py::buffer_info buf = input.request();
        int N = buf.size;
        std::complex<float>* input_ptr = static_cast<std::complex<float>*>(buf.ptr);

        std::vector<float> in_real(N), in_imag(N);
        for (int i = 0; i < N; i++) {
            in_real[i] = input_ptr[i].real();
            in_imag[i] = input_ptr[i].imag();
        }

        std::vector<float> out_real(N), out_imag(N);

        int ret = process_overlap_save(
            state_,
            in_real.data(), in_imag.data(), N,
            out_real.data(), out_imag.data()
        );

        if (ret != 0) {
            throw std::runtime_error("Overlap-save processing failed");
        }

        py::array_t<std::complex<float>> result(N);
        py::buffer_info result_buf = result.request();
        std::complex<float>* result_ptr = static_cast<std::complex<float>*>(result_buf.ptr);

        for (int i = 0; i < N; i++) {
            result_ptr[i] = std::complex<float>(out_real[i], out_imag[i]);
        }

        return result;
    }

private:
    int block_size_;
    int overlap_;
    void* state_;
};

/**
 * Apply channel using overlap-save (standalone function).
 */
py::array_t<std::complex<float>> apply_channel(
    py::array_t<std::complex<float>> input,
    py::array_t<std::complex<float>> H,
    int block_size,
    int overlap
) {
    OverlapSaveProcessor proc(block_size, overlap);
    proc.set_transfer_function(H);
    return proc.process(input);
}

/**
 * Batched overlap-save processor wrapper class for high throughput.
 */
class OverlapSaveProcessorBatched {
public:
    OverlapSaveProcessorBatched(int block_size, int overlap, int batch_size)
        : block_size_(block_size), overlap_(overlap), batch_size_(batch_size), state_(nullptr)
    {
        state_ = init_overlap_save_batched(block_size, overlap, batch_size);
        if (!state_) {
            throw std::runtime_error("Failed to initialize batched overlap-save processor");
        }
    }

    ~OverlapSaveProcessorBatched() {
        if (state_) {
            free_overlap_save_batched(state_);
        }
    }

    void set_transfer_function(py::array_t<std::complex<float>> H) {
        py::buffer_info buf = H.request();
        int N = buf.size;
        std::complex<float>* H_ptr = static_cast<std::complex<float>*>(buf.ptr);

        std::vector<float> H_real(N), H_imag(N);
        for (int i = 0; i < N; i++) {
            H_real[i] = H_ptr[i].real();
            H_imag[i] = H_ptr[i].imag();
        }

        int ret = set_transfer_function_batched(state_, H_real.data(), H_imag.data(), N);
        if (ret != 0) {
            throw std::runtime_error("Failed to set transfer function");
        }
    }

    py::array_t<std::complex<float>> process(py::array_t<std::complex<float>> input) {
        py::buffer_info buf = input.request();
        int N = buf.size;
        std::complex<float>* input_ptr = static_cast<std::complex<float>*>(buf.ptr);

        std::vector<float> in_real(N), in_imag(N);
        for (int i = 0; i < N; i++) {
            in_real[i] = input_ptr[i].real();
            in_imag[i] = input_ptr[i].imag();
        }

        std::vector<float> out_real(N), out_imag(N);

        int ret = process_overlap_save_batched(
            state_,
            in_real.data(), in_imag.data(), N,
            out_real.data(), out_imag.data()
        );

        if (ret != 0) {
            throw std::runtime_error("Batched overlap-save processing failed");
        }

        py::array_t<std::complex<float>> result(N);
        py::buffer_info result_buf = result.request();
        std::complex<float>* result_ptr = static_cast<std::complex<float>*>(result_buf.ptr);

        for (int i = 0; i < N; i++) {
            result_ptr[i] = std::complex<float>(out_real[i], out_imag[i]);
        }

        return result;
    }

    int get_batch_size() const { return batch_size_; }
    int get_block_size() const { return block_size_; }
    int get_overlap() const { return overlap_; }

private:
    int block_size_;
    int overlap_;
    int batch_size_;
    void* state_;
};

/**
 * Doppler fading generator wrapper class.
 */
class DopplerFadingGenerator {
public:
    DopplerFadingGenerator(int n_samples, unsigned long seed = 42)
        : n_samples_(n_samples), state_(nullptr)
    {
        state_ = init_doppler_fading(n_samples, seed);
        if (!state_) {
            throw std::runtime_error("Failed to initialize Doppler fading generator");
        }
    }

    ~DopplerFadingGenerator() {
        if (state_) {
            free_doppler_fading(state_);
        }
    }

    py::array_t<std::complex<float>> generate(float doppler_spread_hz, float sample_rate) {
        std::vector<float> fading_real(n_samples_), fading_imag(n_samples_);

        int ret = generate_doppler_fading_gpu(
            state_, doppler_spread_hz, sample_rate,
            fading_real.data(), fading_imag.data()
        );

        if (ret != 0) {
            throw std::runtime_error("Doppler fading generation failed");
        }

        py::array_t<std::complex<float>> result(n_samples_);
        py::buffer_info result_buf = result.request();
        std::complex<float>* result_ptr = static_cast<std::complex<float>*>(result_buf.ptr);

        for (int i = 0; i < n_samples_; i++) {
            result_ptr[i] = std::complex<float>(fading_real[i], fading_imag[i]);
        }

        return result;
    }

    int get_n_samples() const { return n_samples_; }

private:
    int n_samples_;
    void* state_;
};

/**
 * Generate Doppler fading (standalone function).
 */
py::array_t<std::complex<float>> generate_doppler_fading(
    float doppler_spread_hz,
    float sample_rate,
    int n_samples,
    unsigned long seed = 42
) {
    std::vector<float> fading_real(n_samples), fading_imag(n_samples);

    int ret = generate_doppler_fading_oneshot(
        doppler_spread_hz, sample_rate, n_samples,
        fading_real.data(), fading_imag.data(), seed
    );

    if (ret != 0) {
        throw std::runtime_error("Doppler fading generation failed");
    }

    py::array_t<std::complex<float>> result(n_samples);
    py::buffer_info result_buf = result.request();
    std::complex<float>* result_ptr = static_cast<std::complex<float>*>(result_buf.ptr);

    for (int i = 0; i < n_samples; i++) {
        result_ptr[i] = std::complex<float>(fading_real[i], fading_imag[i]);
    }

    return result;
}

/**
 * Apply channel using batched overlap-save (standalone function).
 */
py::array_t<std::complex<float>> apply_channel_batched(
    py::array_t<std::complex<float>> input,
    py::array_t<std::complex<float>> H,
    int block_size,
    int overlap,
    int batch_size
) {
    OverlapSaveProcessorBatched proc(block_size, overlap, batch_size);
    proc.set_transfer_function(H);
    return proc.process(input);
}

/**
 * VH RF Chain processor wrapper class.
 *
 * Automatically selects GPU or CPU implementation based on availability.
 */
class VHRFChainProcessor {
public:
    VHRFChainProcessor(
        int input_rate,
        int rf_rate,
        int max_input_samples,
        float carrier_freq_hz,
        float coherence_time_sec,
        float k_factor = 0.0f,
        unsigned long seed = 42
    ) : input_rate_(input_rate), rf_rate_(rf_rate),
        max_input_samples_(max_input_samples),
        use_gpu_(false), state_(nullptr)
    {
        // Try GPU first
        state_ = init_vh_rf_chain(input_rate, rf_rate, max_input_samples,
                                  carrier_freq_hz, coherence_time_sec, k_factor, seed);
        if (state_) {
            use_gpu_ = true;
        } else {
            // Fall back to CPU
            state_ = init_vh_rf_chain_cpu(input_rate, rf_rate, max_input_samples,
                                          carrier_freq_hz, coherence_time_sec, k_factor, seed);
            if (!state_) {
                throw std::runtime_error("Failed to initialize VH RF Chain processor");
            }
            use_gpu_ = false;
        }
    }

    ~VHRFChainProcessor() {
        if (state_) {
            if (use_gpu_) {
                free_vh_rf_chain(state_);
            } else {
                free_vh_rf_chain_cpu(state_);
            }
        }
    }

    void configure_taps(
        py::array_t<int> delays,
        py::array_t<float> amplitudes,
        py::array_t<float> doppler_hz
    ) {
        py::buffer_info d_buf = delays.request();
        py::buffer_info a_buf = amplitudes.request();
        py::buffer_info f_buf = doppler_hz.request();

        int n_taps = d_buf.size;
        if (a_buf.size != n_taps || f_buf.size != n_taps) {
            throw std::runtime_error("Tap arrays must have same length");
        }

        int ret;
        if (use_gpu_) {
            ret = configure_vh_taps(state_,
                                    static_cast<int*>(d_buf.ptr),
                                    static_cast<float*>(a_buf.ptr),
                                    static_cast<float*>(f_buf.ptr),
                                    n_taps);
        } else {
            ret = configure_vh_taps_cpu(state_,
                                        static_cast<int*>(d_buf.ptr),
                                        static_cast<float*>(a_buf.ptr),
                                        static_cast<float*>(f_buf.ptr),
                                        n_taps);
        }

        if (ret != 0) {
            throw std::runtime_error("Failed to configure taps");
        }
    }

    py::array_t<std::complex<float>> process(py::array_t<std::complex<float>> input) {
        py::buffer_info buf = input.request();
        int N = buf.size;
        std::complex<float>* input_ptr = static_cast<std::complex<float>*>(buf.ptr);

        if (N > max_input_samples_) {
            throw std::runtime_error("Input exceeds maximum samples");
        }

        std::vector<float> in_real(N), in_imag(N);
        for (int i = 0; i < N; i++) {
            in_real[i] = input_ptr[i].real();
            in_imag[i] = input_ptr[i].imag();
        }

        std::vector<float> out_real(N), out_imag(N);

        int ret;
        if (use_gpu_) {
            ret = process_vh_rf_chain(state_,
                                      in_real.data(), in_imag.data(), N,
                                      out_real.data(), out_imag.data());
        } else {
            ret = process_vh_rf_chain_cpu(state_,
                                          in_real.data(), in_imag.data(), N,
                                          out_real.data(), out_imag.data());
        }

        if (ret != 0) {
            throw std::runtime_error("VH RF chain processing failed");
        }

        // Create result array with proper shape AND strides
        // The strides must be specified to avoid the zero-stride bug
        std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(N)};
        std::vector<py::ssize_t> strides_vec = {static_cast<py::ssize_t>(sizeof(std::complex<float>))};
        py::array_t<std::complex<float>> result(shape, strides_vec);
        py::buffer_info result_buf = result.request();
        std::complex<float>* result_ptr = static_cast<std::complex<float>*>(result_buf.ptr);

        for (int i = 0; i < N; i++) {
            result_ptr[i] = std::complex<float>(out_real[i], out_imag[i]);
        }

        return result;
    }

    void reset() {
        if (use_gpu_) {
            reset_vh_rf_chain(state_);
        } else {
            reset_vh_rf_chain_cpu(state_);
        }
    }

    bool is_using_gpu() const { return use_gpu_; }
    int get_input_rate() const { return input_rate_; }
    int get_rf_rate() const { return rf_rate_; }
    void* get_state_ptr() const { return state_; }

private:
    int input_rate_;
    int rf_rate_;
    int max_input_samples_;
    bool use_gpu_;
    void* state_;
};

/**
 * Watterson channel processor wrapper class.
 */
class WattersonProcessor {
public:
    WattersonProcessor(
        float sample_rate,
        int max_taps = 16,
        int max_delay_samples = 1024,
        int max_samples = 4096,
        unsigned long seed = 42
    ) : sample_rate_(sample_rate), max_taps_(max_taps),
        max_delay_(max_delay_samples), max_samples_(max_samples),
        use_gpu_(false), state_(nullptr)
    {
        // Try GPU first
        state_ = init_watterson_gpu(sample_rate, max_taps, max_delay_samples,
                                    max_samples, seed);
        if (state_) {
            use_gpu_ = true;
        } else {
            // Fall back to CPU
            state_ = init_watterson_cpu(sample_rate, max_taps, max_delay_samples,
                                        max_samples, seed);
            if (!state_) {
                throw std::runtime_error("Failed to initialize Watterson processor");
            }
            use_gpu_ = false;
        }
    }

    ~WattersonProcessor() {
        if (state_) {
            if (use_gpu_) {
                free_watterson_gpu(state_);
            } else {
                free_watterson_cpu(state_);
            }
        }
    }

    void configure_taps(
        py::array_t<int> delays,
        py::array_t<float> amplitudes,
        py::array_t<float> doppler_spreads,
        py::array_t<int> spectrum_types,
        py::array_t<int> is_rician,
        py::array_t<float> k_factors,
        float update_rate
    ) {
        py::buffer_info d_buf = delays.request();
        int n_taps = d_buf.size;

        int ret;
        if (use_gpu_) {
            ret = configure_watterson_taps_gpu(state_,
                static_cast<int*>(d_buf.ptr),
                static_cast<float*>(amplitudes.request().ptr),
                static_cast<float*>(doppler_spreads.request().ptr),
                static_cast<int*>(spectrum_types.request().ptr),
                static_cast<int*>(is_rician.request().ptr),
                static_cast<float*>(k_factors.request().ptr),
                n_taps, update_rate);
        } else {
            ret = configure_watterson_taps_cpu(state_,
                static_cast<int*>(d_buf.ptr),
                static_cast<float*>(amplitudes.request().ptr),
                static_cast<float*>(doppler_spreads.request().ptr),
                static_cast<int*>(spectrum_types.request().ptr),
                static_cast<int*>(is_rician.request().ptr),
                static_cast<float*>(k_factors.request().ptr),
                n_taps, update_rate);
        }
        if (ret != 0) {
            throw std::runtime_error("Failed to configure Watterson taps");
        }
    }

    py::array_t<std::complex<float>> process(py::array_t<std::complex<float>> input) {
        py::buffer_info buf = input.request();
        int N = buf.size;
        std::complex<float>* input_ptr = static_cast<std::complex<float>*>(buf.ptr);

        std::vector<float> in_real(N), in_imag(N);
        for (int i = 0; i < N; i++) {
            in_real[i] = input_ptr[i].real();
            in_imag[i] = input_ptr[i].imag();
        }

        std::vector<float> out_real(N), out_imag(N);

        int ret;
        if (use_gpu_) {
            ret = process_watterson_gpu(state_, in_real.data(), in_imag.data(), N,
                                        out_real.data(), out_imag.data());
        } else {
            ret = process_watterson_cpu(state_, in_real.data(), in_imag.data(), N,
                                        out_real.data(), out_imag.data());
        }
        if (ret != 0) {
            throw std::runtime_error("Watterson processing failed");
        }

        std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(N)};
        std::vector<py::ssize_t> strides = {static_cast<py::ssize_t>(sizeof(std::complex<float>))};
        py::array_t<std::complex<float>> result(shape, strides);
        auto result_ptr = static_cast<std::complex<float>*>(result.request().ptr);
        for (int i = 0; i < N; i++) {
            result_ptr[i] = std::complex<float>(out_real[i], out_imag[i]);
        }
        return result;
    }

    void reset() {
        if (use_gpu_) { reset_watterson_gpu(state_); }
        else { reset_watterson_cpu(state_); }
    }

    bool is_using_gpu() const { return use_gpu_; }

private:
    float sample_rate_;
    int max_taps_;
    int max_delay_;
    int max_samples_;
    bool use_gpu_;
    void* state_;
};

/**
 * AGC processor wrapper class.
 */
class AGCProcessor {
public:
    AGCProcessor(
        float sample_rate,
        float attack_time_ms = 50.0f,
        float release_time_ms = 500.0f,
        float hold_time_ms = 50.0f,
        bool hang_agc = true,
        float target_level_db = -10.0f,
        float max_gain_db = 60.0f,
        float min_gain_db = -20.0f,
        float soft_knee_db = 6.0f,
        int max_samples = 4096
    ) : use_gpu_(false), state_(nullptr)
    {
        state_ = init_agc_gpu(sample_rate, attack_time_ms, release_time_ms,
                              hold_time_ms, hang_agc, target_level_db,
                              max_gain_db, min_gain_db, soft_knee_db, max_samples);
        if (state_) {
            use_gpu_ = true;
        } else {
            state_ = init_agc_cpu(sample_rate, attack_time_ms, release_time_ms,
                                  hold_time_ms, hang_agc, target_level_db,
                                  max_gain_db, min_gain_db, soft_knee_db, max_samples);
            if (!state_) {
                throw std::runtime_error("Failed to initialize AGC processor");
            }
        }
    }

    ~AGCProcessor() {
        if (state_) {
            if (use_gpu_) { free_agc_gpu(state_); }
            else { free_agc_cpu(state_); }
        }
    }

    py::array_t<std::complex<float>> process(py::array_t<std::complex<float>> input) {
        py::buffer_info buf = input.request();
        int N = buf.size;
        std::complex<float>* input_ptr = static_cast<std::complex<float>*>(buf.ptr);

        std::vector<float> in_real(N), in_imag(N);
        for (int i = 0; i < N; i++) {
            in_real[i] = input_ptr[i].real();
            in_imag[i] = input_ptr[i].imag();
        }

        std::vector<float> out_real(N), out_imag(N);

        int ret;
        if (use_gpu_) {
            ret = process_agc_gpu(state_, in_real.data(), in_imag.data(), N,
                                  out_real.data(), out_imag.data());
        } else {
            ret = process_agc_cpu(state_, in_real.data(), in_imag.data(), N,
                                  out_real.data(), out_imag.data());
        }
        if (ret != 0) {
            throw std::runtime_error("AGC processing failed");
        }

        std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(N)};
        std::vector<py::ssize_t> strides = {static_cast<py::ssize_t>(sizeof(std::complex<float>))};
        py::array_t<std::complex<float>> result(shape, strides);
        auto result_ptr = static_cast<std::complex<float>*>(result.request().ptr);
        for (int i = 0; i < N; i++) {
            result_ptr[i] = std::complex<float>(out_real[i], out_imag[i]);
        }
        return result;
    }

    float get_gain_db() {
        if (use_gpu_) { return get_agc_gain_db(state_); }
        else { return get_agc_gain_db_cpu(state_); }
    }

    void reset() {
        if (use_gpu_) { reset_agc_gpu(state_); }
        else { reset_agc_cpu(state_); }
    }

    bool is_using_gpu() const { return use_gpu_; }

private:
    bool use_gpu_;
    void* state_;
};

/**
 * Limiter processor wrapper class.
 */
class LimiterProcessor {
public:
    LimiterProcessor(float threshold_db = -3.0f, int mode = 1, int max_samples = 4096)
        : use_gpu_(false), state_(nullptr)
    {
        state_ = init_limiter_gpu(threshold_db, mode, max_samples);
        if (state_) {
            use_gpu_ = true;
        } else {
            state_ = init_limiter_cpu(threshold_db, mode, max_samples);
            if (!state_) {
                throw std::runtime_error("Failed to initialize Limiter processor");
            }
        }
    }

    ~LimiterProcessor() {
        if (state_) {
            if (use_gpu_) { free_limiter_gpu(state_); }
            else { free_limiter_cpu(state_); }
        }
    }

    py::array_t<std::complex<float>> process(py::array_t<std::complex<float>> input) {
        py::buffer_info buf = input.request();
        int N = buf.size;
        std::complex<float>* input_ptr = static_cast<std::complex<float>*>(buf.ptr);

        std::vector<float> in_real(N), in_imag(N);
        for (int i = 0; i < N; i++) {
            in_real[i] = input_ptr[i].real();
            in_imag[i] = input_ptr[i].imag();
        }

        std::vector<float> out_real(N), out_imag(N);

        int ret;
        if (use_gpu_) {
            ret = process_limiter_gpu(state_, in_real.data(), in_imag.data(), N,
                                      out_real.data(), out_imag.data());
        } else {
            ret = process_limiter_cpu(state_, in_real.data(), in_imag.data(), N,
                                      out_real.data(), out_imag.data());
        }
        if (ret != 0) {
            throw std::runtime_error("Limiter processing failed");
        }

        std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(N)};
        std::vector<py::ssize_t> strides = {static_cast<py::ssize_t>(sizeof(std::complex<float>))};
        py::array_t<std::complex<float>> result(shape, strides);
        auto result_ptr = static_cast<std::complex<float>*>(result.request().ptr);
        for (int i = 0; i < N; i++) {
            result_ptr[i] = std::complex<float>(out_real[i], out_imag[i]);
        }
        return result;
    }

    void set_params(float threshold_db, int mode) {
        if (use_gpu_) { set_limiter_params_gpu(state_, threshold_db, mode); }
        else { set_limiter_params_cpu(state_, threshold_db, mode); }
    }

    bool is_using_gpu() const { return use_gpu_; }

private:
    bool use_gpu_;
    void* state_;
};

/**
 * Noise generator wrapper class.
 */
class NoiseGenerator {
public:
    NoiseGenerator(float sample_rate, int max_samples = 4096, unsigned long seed = 42)
        : max_samples_(max_samples), use_gpu_(false), state_(nullptr)
    {
        state_ = init_noise_gen_gpu(sample_rate, max_samples, seed);
        if (state_) {
            use_gpu_ = true;
        } else {
            state_ = init_noise_gen_cpu(sample_rate, max_samples, seed);
            if (!state_) {
                throw std::runtime_error("Failed to initialize Noise generator");
            }
        }
    }

    ~NoiseGenerator() {
        if (state_) {
            if (use_gpu_) { free_noise_gen_gpu(state_); }
            else { free_noise_gen_cpu(state_); }
        }
    }

    py::array_t<std::complex<float>> generate_awgn(float noise_power, int n_samples) {
        std::vector<float> noise_real(n_samples), noise_imag(n_samples);
        int ret;
        if (use_gpu_) {
            ret = generate_awgn_gpu(state_, noise_power, n_samples,
                                    noise_real.data(), noise_imag.data());
        } else {
            ret = generate_awgn_cpu(state_, noise_power, n_samples,
                                    noise_real.data(), noise_imag.data());
        }
        if (ret != 0) { throw std::runtime_error("AWGN generation failed"); }

        std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(n_samples)};
        std::vector<py::ssize_t> strides = {static_cast<py::ssize_t>(sizeof(std::complex<float>))};
        py::array_t<std::complex<float>> result(shape, strides);
        auto result_ptr = static_cast<std::complex<float>*>(result.request().ptr);
        for (int i = 0; i < n_samples; i++) {
            result_ptr[i] = std::complex<float>(noise_real[i], noise_imag[i]);
        }
        return result;
    }

    py::array_t<std::complex<float>> generate_atmospheric(float noise_power, float vd, int n_samples) {
        std::vector<float> noise_real(n_samples), noise_imag(n_samples);
        int ret;
        if (use_gpu_) {
            ret = generate_atmospheric_gpu(state_, noise_power, vd, n_samples,
                                           noise_real.data(), noise_imag.data());
        } else {
            ret = generate_atmospheric_cpu(state_, noise_power, vd, n_samples,
                                           noise_real.data(), noise_imag.data());
        }
        if (ret != 0) { throw std::runtime_error("Atmospheric noise generation failed"); }

        std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(n_samples)};
        std::vector<py::ssize_t> strides = {static_cast<py::ssize_t>(sizeof(std::complex<float>))};
        py::array_t<std::complex<float>> result(shape, strides);
        auto result_ptr = static_cast<std::complex<float>*>(result.request().ptr);
        for (int i = 0; i < n_samples; i++) {
            result_ptr[i] = std::complex<float>(noise_real[i], noise_imag[i]);
        }
        return result;
    }

    py::array_t<std::complex<float>> generate_impulse(float impulse_rate, float impulse_amplitude,
                                                      float noise_floor, int n_samples) {
        std::vector<float> noise_real(n_samples), noise_imag(n_samples);
        int ret;
        if (use_gpu_) {
            ret = generate_impulse_gpu(state_, impulse_rate, impulse_amplitude, noise_floor,
                                       n_samples, noise_real.data(), noise_imag.data());
        } else {
            ret = generate_impulse_cpu(state_, impulse_rate, impulse_amplitude, noise_floor,
                                       n_samples, noise_real.data(), noise_imag.data());
        }
        if (ret != 0) { throw std::runtime_error("Impulse noise generation failed"); }

        std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(n_samples)};
        std::vector<py::ssize_t> strides = {static_cast<py::ssize_t>(sizeof(std::complex<float>))};
        py::array_t<std::complex<float>> result(shape, strides);
        auto result_ptr = static_cast<std::complex<float>*>(result.request().ptr);
        for (int i = 0; i < n_samples; i++) {
            result_ptr[i] = std::complex<float>(noise_real[i], noise_imag[i]);
        }
        return result;
    }

    void reset(unsigned long seed) {
        if (use_gpu_) { reset_noise_gen_gpu(state_, seed); }
        else { reset_noise_gen_cpu(state_, seed); }
    }

    bool is_using_gpu() const { return use_gpu_; }

private:
    int max_samples_;
    bool use_gpu_;
    void* state_;
};

/**
 * Compute power spectrum.
 */
py::array_t<float> compute_spectrum(
    py::array_t<std::complex<float>> signal,
    float reference = 1.0f
) {
    py::buffer_info input_buf = signal.request();
    int N = input_buf.size;
    std::complex<float>* sig_ptr = static_cast<std::complex<float>*>(input_buf.ptr);

    std::vector<float> sig_real(N), sig_imag(N);
    for (int i = 0; i < N; i++) {
        sig_real[i] = sig_ptr[i].real();
        sig_imag[i] = sig_ptr[i].imag();
    }

    // Allocate our own result buffer
    std::vector<float> result_data(N);

    int ret = compute_spectrum_gpu(
        sig_real.data(), sig_imag.data(), N,
        result_data.data(), reference
    );

    if (ret != 0) {
        throw std::runtime_error("Spectrum computation failed");
    }

    // Create numpy array with proper shape and strides
    std::vector<ssize_t> shape = {static_cast<ssize_t>(N)};
    std::vector<ssize_t> strides = {static_cast<ssize_t>(sizeof(float))};
    py::array_t<float> result(shape, strides);

    // Get writable buffer and copy data
    py::buffer_info result_buf = result.request(true);
    float* result_ptr = static_cast<float*>(result_buf.ptr);
    memcpy(result_ptr, result_data.data(), N * sizeof(float));

    return result;
}

PYBIND11_MODULE(_hfpathsim_gpu, m) {
    m.doc() = "HF Path Simulator GPU acceleration module (Phase 5: cuFFT)";

    // Device info
    m.def("get_device_info", &get_device_info,
          "Get GPU device information");

    // Vogler transfer function
    m.def("vogler_transfer_function", &vogler_transfer_function,
          "Compute Vogler reflection coefficient",
          py::arg("freq_hz"),
          py::arg("foF2_mhz"),
          py::arg("hmF2_km"),
          py::arg("sigma"),
          py::arg("chi"),
          py::arg("t0_sec"));

    // Single-block overlap-save (original)
    m.def("apply_channel", &apply_channel,
          "Apply channel using overlap-save convolution",
          py::arg("input"),
          py::arg("H"),
          py::arg("block_size") = 4096,
          py::arg("overlap") = 1024);

    // Batched overlap-save (high throughput)
    m.def("apply_channel_batched", &apply_channel_batched,
          "Apply channel using batched overlap-save convolution for higher throughput",
          py::arg("input"),
          py::arg("H"),
          py::arg("block_size") = 4096,
          py::arg("overlap") = 1024,
          py::arg("batch_size") = 8);

    // Spectrum computation
    m.def("compute_spectrum", &compute_spectrum,
          "Compute power spectrum in dB",
          py::arg("signal"),
          py::arg("reference") = 1.0f);

    // Doppler fading generation
    m.def("generate_doppler_fading", &generate_doppler_fading,
          "Generate Doppler-shaped fading samples",
          py::arg("doppler_spread_hz"),
          py::arg("sample_rate"),
          py::arg("n_samples"),
          py::arg("seed") = 42);

    // Original overlap-save processor class
    py::class_<OverlapSaveProcessor>(m, "OverlapSaveProcessor")
        .def(py::init<int, int>(),
             py::arg("block_size") = 4096,
             py::arg("overlap") = 1024)
        .def("set_transfer_function", &OverlapSaveProcessor::set_transfer_function)
        .def("process", &OverlapSaveProcessor::process);

    // Batched overlap-save processor class (high throughput)
    py::class_<OverlapSaveProcessorBatched>(m, "OverlapSaveProcessorBatched")
        .def(py::init<int, int, int>(),
             py::arg("block_size") = 4096,
             py::arg("overlap") = 1024,
             py::arg("batch_size") = 8)
        .def("set_transfer_function", &OverlapSaveProcessorBatched::set_transfer_function)
        .def("process", &OverlapSaveProcessorBatched::process)
        .def("get_batch_size", &OverlapSaveProcessorBatched::get_batch_size)
        .def("get_block_size", &OverlapSaveProcessorBatched::get_block_size)
        .def("get_overlap", &OverlapSaveProcessorBatched::get_overlap);

    // Doppler fading generator class
    py::class_<DopplerFadingGenerator>(m, "DopplerFadingGenerator")
        .def(py::init<int, unsigned long>(),
             py::arg("n_samples"),
             py::arg("seed") = 42)
        .def("generate", &DopplerFadingGenerator::generate,
             py::arg("doppler_spread_hz"),
             py::arg("sample_rate"))
        .def("get_n_samples", &DopplerFadingGenerator::get_n_samples);

    // VH RF Chain processor class (auto-selects GPU/CPU)
    py::class_<VHRFChainProcessor>(m, "VHRFChainProcessor")
        .def(py::init<int, int, int, float, float, float, unsigned long>(),
             py::arg("input_rate"),
             py::arg("rf_rate"),
             py::arg("max_input_samples"),
             py::arg("carrier_freq_hz"),
             py::arg("coherence_time_sec"),
             py::arg("k_factor") = 0.0f,
             py::arg("seed") = 42,
             "Initialize VH RF Chain processor. Automatically uses GPU if available, "
             "otherwise falls back to optimized CPU implementation.")
        .def("configure_taps", &VHRFChainProcessor::configure_taps,
             py::arg("delays"),
             py::arg("amplitudes"),
             py::arg("doppler_hz"),
             "Configure TDL taps with delays (samples), amplitudes, and Doppler shifts")
        .def("process", &VHRFChainProcessor::process,
             py::arg("input"),
             "Process samples through VH RF chain")
        .def("reset", &VHRFChainProcessor::reset,
             "Reset processor state (fading, phase)")
        .def("is_using_gpu", &VHRFChainProcessor::is_using_gpu,
             "Returns True if using GPU, False if using CPU fallback")
        .def("get_input_rate", &VHRFChainProcessor::get_input_rate)
        .def("get_rf_rate", &VHRFChainProcessor::get_rf_rate)
        .def("debug_test_upsample", [](VHRFChainProcessor& self, py::array_t<std::complex<float>> input) {
            py::buffer_info buf = input.request();
            int N = buf.size;
            std::complex<float>* ptr = static_cast<std::complex<float>*>(buf.ptr);
            std::vector<float> real(N), imag(N);
            for (int i = 0; i < N; i++) {
                real[i] = ptr[i].real();
                imag[i] = ptr[i].imag();
            }
            // Only works for GPU backend
            if (self.is_using_gpu()) {
                return debug_test_upsample(self.get_state_ptr(), real.data(), imag.data(), N);
            }
            return -1.0f;
        }, "Debug: test upsample and return power")
        .def("debug_test_roundtrip", [](VHRFChainProcessor& self, py::array_t<std::complex<float>> input) {
            py::buffer_info buf = input.request();
            int N = buf.size;
            std::complex<float>* ptr = static_cast<std::complex<float>*>(buf.ptr);
            std::vector<float> real(N), imag(N);
            for (int i = 0; i < N; i++) {
                real[i] = ptr[i].real();
                imag[i] = ptr[i].imag();
            }
            if (self.is_using_gpu()) {
                return debug_test_resample_roundtrip(self.get_state_ptr(), real.data(), imag.data(), N);
            }
            return -1.0f;
        }, "Debug: test upsample+downsample roundtrip and return power")
        .def("debug_get_filter", [](VHRFChainProcessor& self) {
            std::vector<float> filter(2048);
            if (self.is_using_gpu()) {
                int len = debug_get_filter(self.get_state_ptr(), filter.data(), 2048);
                filter.resize(len);
            }
            return filter;
        }, "Debug: get filter coefficients")
        .def("debug_test_mixer", [](VHRFChainProcessor& self, py::array_t<std::complex<float>> input) {
            py::buffer_info buf = input.request();
            int N = buf.size;
            std::complex<float>* ptr = static_cast<std::complex<float>*>(buf.ptr);
            std::vector<float> real(N), imag(N);
            for (int i = 0; i < N; i++) {
                real[i] = ptr[i].real();
                imag[i] = ptr[i].imag();
            }
            if (self.is_using_gpu()) {
                return debug_test_mixer_roundtrip(self.get_state_ptr(), real.data(), imag.data(), N);
            }
            return -1.0f;
        }, "Debug: test upsample+mixer+downsample (no TDL)")
        .def("debug_test_tdl_unity", [](VHRFChainProcessor& self, py::array_t<std::complex<float>> input) {
            py::buffer_info buf = input.request();
            int N = buf.size;
            std::complex<float>* ptr = static_cast<std::complex<float>*>(buf.ptr);
            std::vector<float> real(N), imag(N);
            for (int i = 0; i < N; i++) {
                real[i] = ptr[i].real();
                imag[i] = ptr[i].imag();
            }
            if (self.is_using_gpu()) {
                return debug_test_tdl_unity(self.get_state_ptr(), real.data(), imag.data(), N);
            }
            return -1.0f;
        }, "Debug: test full chain with TDL bypassed (unity gain)")
        .def("debug_fading_stats", [](VHRFChainProcessor& self, int n_samples) {
            float mean_mag = -1.0f, mean_magsq = -1.0f;
            if (self.is_using_gpu()) {
                debug_get_fading_stats(self.get_state_ptr(), n_samples, &mean_mag, &mean_magsq);
            }
            return std::make_pair(mean_mag, mean_magsq);
        }, "Debug: get fading coefficient statistics")
        .def("debug_test_constant_fading", [](VHRFChainProcessor& self, py::array_t<std::complex<float>> input, float fading_re, float fading_im) {
            py::buffer_info buf = input.request();
            int N = buf.size;
            std::complex<float>* ptr = static_cast<std::complex<float>*>(buf.ptr);
            std::vector<float> real(N), imag(N);
            for (int i = 0; i < N; i++) {
                real[i] = ptr[i].real();
                imag[i] = ptr[i].imag();
            }
            if (self.is_using_gpu()) {
                return debug_test_tdl_constant_fading(self.get_state_ptr(), real.data(), imag.data(), N, fading_re, fading_im);
            }
            return -1.0f;
        }, "Debug: test TDL with constant fading value")
        .def("debug_power_stages", [](VHRFChainProcessor& self, py::array_t<std::complex<float>> input) {
            py::buffer_info buf = input.request();
            int N = buf.size;
            std::complex<float>* ptr = static_cast<std::complex<float>*>(buf.ptr);
            std::vector<float> real(N), imag(N);
            for (int i = 0; i < N; i++) {
                real[i] = ptr[i].real();
                imag[i] = ptr[i].imag();
            }
            std::vector<float> powers(7, -1.0f);
            if (self.is_using_gpu()) {
                debug_power_stages(self.get_state_ptr(), real.data(), imag.data(), N, powers.data());
            }
            // Return dict with stage names
            py::dict result;
            result["input"] = powers[0];
            result["after_upsample"] = powers[1];
            result["after_mixup"] = powers[2];
            result["fading_magsq"] = powers[3];
            result["after_tdl"] = powers[4];
            result["after_mixdown"] = powers[5];
            result["output"] = powers[6];
            return result;
        }, "Debug: report power at each stage of RF chain");

    // Watterson channel processor (auto-selects GPU/CPU)
    py::class_<WattersonProcessor>(m, "WattersonProcessor")
        .def(py::init<float, int, int, int, unsigned long>(),
             py::arg("sample_rate"),
             py::arg("max_taps") = 16,
             py::arg("max_delay_samples") = 1024,
             py::arg("max_samples") = 4096,
             py::arg("seed") = 42,
             "Initialize Watterson channel processor")
        .def("configure_taps", &WattersonProcessor::configure_taps,
             py::arg("delays"),
             py::arg("amplitudes"),
             py::arg("doppler_spreads"),
             py::arg("spectrum_types"),
             py::arg("is_rician"),
             py::arg("k_factors"),
             py::arg("update_rate"),
             "Configure TDL taps with Doppler spectrum parameters")
        .def("process", &WattersonProcessor::process,
             py::arg("input"),
             "Process samples through Watterson channel")
        .def("reset", &WattersonProcessor::reset,
             "Reset channel state")
        .def("is_using_gpu", &WattersonProcessor::is_using_gpu);

    // AGC processor (auto-selects GPU/CPU)
    py::class_<AGCProcessor>(m, "AGCProcessor")
        .def(py::init<float, float, float, float, bool, float, float, float, float, int>(),
             py::arg("sample_rate"),
             py::arg("attack_time_ms") = 50.0f,
             py::arg("release_time_ms") = 500.0f,
             py::arg("hold_time_ms") = 50.0f,
             py::arg("hang_agc") = true,
             py::arg("target_level_db") = -10.0f,
             py::arg("max_gain_db") = 60.0f,
             py::arg("min_gain_db") = -20.0f,
             py::arg("soft_knee_db") = 6.0f,
             py::arg("max_samples") = 4096,
             "Initialize AGC processor")
        .def("process", &AGCProcessor::process,
             py::arg("input"),
             "Process samples through AGC")
        .def("get_gain_db", &AGCProcessor::get_gain_db,
             "Get current gain in dB")
        .def("reset", &AGCProcessor::reset,
             "Reset AGC state")
        .def("is_using_gpu", &AGCProcessor::is_using_gpu);

    // Limiter processor (auto-selects GPU/CPU)
    py::class_<LimiterProcessor>(m, "LimiterProcessor")
        .def(py::init<float, int, int>(),
             py::arg("threshold_db") = -3.0f,
             py::arg("mode") = 1,  // 0=hard, 1=soft, 2=cubic
             py::arg("max_samples") = 4096,
             "Initialize limiter (mode: 0=hard, 1=soft/tanh, 2=cubic)")
        .def("process", &LimiterProcessor::process,
             py::arg("input"),
             "Process samples through limiter")
        .def("set_params", &LimiterProcessor::set_params,
             py::arg("threshold_db"),
             py::arg("mode"),
             "Update limiter parameters")
        .def("is_using_gpu", &LimiterProcessor::is_using_gpu);

    // Noise generator (auto-selects GPU/CPU)
    py::class_<NoiseGenerator>(m, "NoiseGenerator")
        .def(py::init<float, int, unsigned long>(),
             py::arg("sample_rate"),
             py::arg("max_samples") = 4096,
             py::arg("seed") = 42,
             "Initialize noise generator")
        .def("generate_awgn", &NoiseGenerator::generate_awgn,
             py::arg("noise_power"),
             py::arg("n_samples"),
             "Generate AWGN samples with specified power")
        .def("generate_atmospheric", &NoiseGenerator::generate_atmospheric,
             py::arg("noise_power"),
             py::arg("vd"),
             py::arg("n_samples"),
             "Generate atmospheric noise (vd=0 Gaussian, vd=1 very impulsive)")
        .def("generate_impulse", &NoiseGenerator::generate_impulse,
             py::arg("impulse_rate"),
             py::arg("impulse_amplitude"),
             py::arg("noise_floor"),
             py::arg("n_samples"),
             "Generate impulse noise with specified parameters")
        .def("reset", &NoiseGenerator::reset,
             py::arg("seed"),
             "Reset RNG with new seed")
        .def("is_using_gpu", &NoiseGenerator::is_using_gpu);
}
