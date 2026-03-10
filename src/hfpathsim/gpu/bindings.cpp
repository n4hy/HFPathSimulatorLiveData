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

    // Fading generator functions
    void* init_fading_generator(int N, unsigned long seed);
    int generate_faded_channel(void* state_ptr,
                               const float* H_base_real, const float* H_base_imag,
                               float doppler_spread, float delay_spread, float sample_rate,
                               float* H_out_real, float* H_out_imag);
    void free_fading_generator(void* state_ptr);

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
    m.doc() = "HF Path Simulator GPU acceleration module";

    m.def("get_device_info", &get_device_info,
          "Get GPU device information");

    m.def("vogler_transfer_function", &vogler_transfer_function,
          "Compute Vogler reflection coefficient",
          py::arg("freq_hz"),
          py::arg("foF2_mhz"),
          py::arg("hmF2_km"),
          py::arg("sigma"),
          py::arg("chi"),
          py::arg("t0_sec"));

    m.def("apply_channel", &apply_channel,
          "Apply channel using overlap-save convolution",
          py::arg("input"),
          py::arg("H"),
          py::arg("block_size") = 4096,
          py::arg("overlap") = 1024);

    m.def("compute_spectrum", &compute_spectrum,
          "Compute power spectrum in dB",
          py::arg("signal"),
          py::arg("reference") = 1.0f);

    py::class_<OverlapSaveProcessor>(m, "OverlapSaveProcessor")
        .def(py::init<int, int>(),
             py::arg("block_size") = 4096,
             py::arg("overlap") = 1024)
        .def("set_transfer_function", &OverlapSaveProcessor::set_transfer_function)
        .def("process", &OverlapSaveProcessor::process);
}
