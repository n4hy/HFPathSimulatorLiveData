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
 * Compute power spectrum.
 */
py::array_t<float> compute_spectrum(
    py::array_t<std::complex<float>> signal,
    float reference = 1.0f
) {
    py::buffer_info buf = signal.request();
    int N = buf.size;
    std::complex<float>* sig_ptr = static_cast<std::complex<float>*>(buf.ptr);

    std::vector<float> sig_real(N), sig_imag(N);
    for (int i = 0; i < N; i++) {
        sig_real[i] = sig_ptr[i].real();
        sig_imag[i] = sig_ptr[i].imag();
    }

    py::array_t<float> result(N);
    py::buffer_info result_buf = result.request();
    float* result_ptr = static_cast<float*>(result_buf.ptr);

    int ret = compute_spectrum_gpu(
        sig_real.data(), sig_imag.data(), N,
        result_ptr, reference
    );

    if (ret != 0) {
        throw std::runtime_error("Spectrum computation failed");
    }

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
}
