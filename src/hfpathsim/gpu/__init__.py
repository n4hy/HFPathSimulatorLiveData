"""GPU acceleration module for HF Path Simulator.

This module provides CUDA-accelerated implementations of:
- Vogler reflection coefficient computation
- Channel transfer function H(f,t)
- Overlap-save convolution
- Scattering function computation

Falls back to CuPy or NumPy if native CUDA module unavailable.
"""

from typing import Optional, Dict, Any
import numpy as np

# Try to import native CUDA module
_gpu_module = None
_cupy_available = False
_cupy_tested = False  # Have we tested if CuPy kernels work?
_cupy_works = False   # Do CuPy kernels actually compile/run?

try:
    from . import _hfpathsim_gpu

    _gpu_module = _hfpathsim_gpu
except ImportError:
    pass

# Fall back to CuPy
if _gpu_module is None:
    try:
        import cupy as cp

        _cupy_available = True
    except ImportError:
        pass


def _test_cupy_kernels() -> bool:
    """Test if CuPy kernels can compile for this GPU.

    New GPU architectures (like RTX 5090) may not have precompiled
    kernels in the CuPy package.
    """
    global _cupy_tested, _cupy_works

    if _cupy_tested:
        return _cupy_works

    _cupy_tested = True

    if not _cupy_available:
        _cupy_works = False
        return False

    try:
        import cupy as cp
        # Try a simple operation that requires kernel compilation
        test = cp.array([1.0, 2.0, 3.0])
        result = test * 2
        _ = cp.asnumpy(result)
        _cupy_works = True
        return True
    except Exception as e:
        print(f"CuPy kernel test failed: {e}")
        print("Falling back to NumPy for GPU operations")
        _cupy_works = False
        return False


def get_device_info() -> Dict[str, Any]:
    """Get GPU device information.

    Returns:
        Dictionary with device name, compute capability, memory, etc.
    """
    if _gpu_module is not None:
        return _gpu_module.get_device_info()

    if _cupy_available:
        import cupy as cp

        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)

        # Check if CuPy kernels actually work
        cupy_works = _test_cupy_kernels()
        backend = "cupy" if cupy_works else "numpy (cupy kernels unavailable)"

        return {
            "name": props["name"].decode(),
            "compute_capability": f"{props['major']}.{props['minor']}",
            "total_memory_gb": props["totalGlobalMem"] / (1024**3),
            "multiprocessors": props["multiProcessorCount"],
            "backend": backend,
        }

    return {
        "name": "CPU (No GPU)",
        "compute_capability": "N/A",
        "total_memory_gb": 0,
        "multiprocessors": 0,
        "backend": "numpy",
    }


def is_available() -> bool:
    """Check if GPU acceleration is available."""
    if _gpu_module is not None:
        return True
    if _cupy_available:
        return _test_cupy_kernels()
    return False


def vogler_transfer_function(
    freq_hz: np.ndarray,
    foF2_mhz: float,
    hmF2_km: float,
    sigma: float,
    chi: float,
    t0_sec: float,
) -> np.ndarray:
    """Compute Vogler reflection coefficient R(omega).

    GPU-accelerated computation of the ionospheric reflection coefficient
    based on NTIA TR-88-240.

    Args:
        freq_hz: Frequency array in Hz
        foF2_mhz: F2 layer critical frequency in MHz
        hmF2_km: F2 layer peak height in km
        sigma: Layer thickness parameter
        chi: Penetration parameter
        t0_sec: Base propagation delay in seconds

    Returns:
        Complex reflection coefficient array
    """
    if _gpu_module is not None:
        return _gpu_module.vogler_transfer_function(
            freq_hz, foF2_mhz, hmF2_km, sigma, chi, t0_sec
        )

    if _cupy_available and _test_cupy_kernels():
        try:
            return _vogler_cupy(freq_hz, foF2_mhz, sigma, chi, t0_sec)
        except Exception:
            pass  # Fall through to numpy

    return _vogler_numpy(freq_hz, foF2_mhz, sigma, chi, t0_sec)


def _vogler_cupy(
    freq_hz: np.ndarray,
    foF2_mhz: float,
    sigma: float,
    chi: float,
    t0_sec: float,
) -> np.ndarray:
    """CuPy implementation of Vogler reflection coefficient."""
    import cupy as cp
    from cupyx.scipy.special import gamma as cp_gamma

    # Transfer to GPU
    freq_gpu = cp.asarray(freq_hz, dtype=cp.float64)

    fc = foF2_mhz * 1e6
    omega_norm = freq_gpu / fc

    # Compute gamma function terms
    # Note: CuPy gamma doesn't support complex, so we use approximation
    # For full accuracy, use CPU scipy.special.gamma

    # Simplified amplitude model
    R_amp = cp.exp(-cp.abs(omega_norm) ** 2 / (2 * sigma**2))

    # Phase from propagation delay
    phase = cp.exp(-1j * 2 * cp.pi * freq_gpu * t0_sec)

    # Frequency-dependent phase shift
    group_delay_var = sigma * 1e-3  # Approximation
    phase_shift = cp.exp(-1j * cp.pi * omega_norm**2 * group_delay_var)

    R = R_amp * phase * phase_shift

    return cp.asnumpy(R).astype(np.complex64)


def _vogler_numpy(
    freq_hz: np.ndarray,
    foF2_mhz: float,
    sigma: float,
    chi: float,
    t0_sec: float,
) -> np.ndarray:
    """NumPy fallback implementation of Vogler reflection coefficient."""
    from scipy.special import gamma as scipy_gamma

    fc = foF2_mhz * 1e6
    omega_norm = freq_hz / fc

    R = np.zeros_like(omega_norm, dtype=np.complex128)

    for i, omega in enumerate(omega_norm):
        try:
            g1 = scipy_gamma(complex(1, -sigma * omega))
            g2 = scipy_gamma(complex(0.5 - chi, sigma * omega))
            g3 = scipy_gamma(complex(0.5 + chi, sigma * omega))
            num = g1 * g2 * g3

            g4 = scipy_gamma(complex(1, sigma * omega))
            g5 = scipy_gamma(0.5 - chi)
            g6 = scipy_gamma(0.5 + chi)
            den = g4 * g5 * g6

            phase = np.exp(-1j * 2 * np.pi * freq_hz[i] * t0_sec)
            R[i] = (num / den) * phase

        except (ValueError, ZeroDivisionError):
            R[i] = 0.0

    return R.astype(np.complex64)


def apply_channel(
    input_signal: np.ndarray,
    transfer_function: np.ndarray,
    block_size: int = 4096,
    overlap: int = 1024,
) -> np.ndarray:
    """Apply channel transfer function using overlap-save.

    GPU-accelerated overlap-save convolution for real-time processing.

    Args:
        input_signal: Complex input signal
        transfer_function: Channel transfer function H(f)
        block_size: FFT block size
        overlap: Overlap samples

    Returns:
        Complex output signal
    """
    if _gpu_module is not None:
        return _gpu_module.apply_channel(
            input_signal, transfer_function, block_size, overlap
        )

    if _cupy_available and _test_cupy_kernels():
        try:
            return _apply_channel_cupy(
                input_signal, transfer_function, block_size, overlap
            )
        except Exception:
            pass  # Fall through to numpy

    return _apply_channel_numpy(
        input_signal, transfer_function, block_size, overlap
    )


def _apply_channel_cupy(
    input_signal: np.ndarray,
    H: np.ndarray,
    block_size: int,
    overlap: int,
) -> np.ndarray:
    """CuPy implementation of overlap-save convolution."""
    import cupy as cp

    output_size = block_size - overlap
    n_samples = len(input_signal)
    n_blocks = (n_samples + output_size - 1) // output_size
    padded_length = n_blocks * output_size + overlap

    # Transfer to GPU
    H_gpu = cp.asarray(H, dtype=cp.complex64)
    padded_input = cp.zeros(padded_length, dtype=cp.complex64)
    padded_input[overlap : overlap + n_samples] = cp.asarray(input_signal)

    output = cp.zeros(n_blocks * output_size, dtype=cp.complex64)

    # Process blocks
    for i in range(n_blocks):
        start = i * output_size
        block = padded_input[start : start + block_size]

        X = cp.fft.fft(block)
        Y = X * H_gpu
        y = cp.fft.ifft(Y)

        output[i * output_size : (i + 1) * output_size] = y[overlap:]

    return cp.asnumpy(output[:n_samples])


def _apply_channel_numpy(
    input_signal: np.ndarray,
    H: np.ndarray,
    block_size: int,
    overlap: int,
) -> np.ndarray:
    """NumPy fallback implementation of overlap-save convolution."""
    output_size = block_size - overlap
    n_samples = len(input_signal)
    n_blocks = (n_samples + output_size - 1) // output_size
    padded_length = n_blocks * output_size + overlap

    padded_input = np.zeros(padded_length, dtype=np.complex64)
    padded_input[overlap : overlap + n_samples] = input_signal

    output = np.zeros(n_blocks * output_size, dtype=np.complex64)

    for i in range(n_blocks):
        start = i * output_size
        block = padded_input[start : start + block_size]

        X = np.fft.fft(block)
        Y = X * H
        y = np.fft.ifft(Y)

        output[i * output_size : (i + 1) * output_size] = y[overlap:]

    return output[:n_samples]


def compute_scattering_function(
    delay_axis_ms: np.ndarray,
    doppler_axis_hz: np.ndarray,
    delay_spread_ms: float,
    doppler_spread_hz: float,
) -> np.ndarray:
    """Compute scattering function S(tau, nu).

    Args:
        delay_axis_ms: Delay axis in milliseconds
        doppler_axis_hz: Doppler axis in Hz
        delay_spread_ms: Delay spread parameter
        doppler_spread_hz: Doppler spread parameter

    Returns:
        2D scattering function array
    """
    if _cupy_available and _test_cupy_kernels():
        try:
            import cupy as cp

            tau = cp.asarray(delay_axis_ms)
            nu = cp.asarray(doppler_axis_hz)
            TAU, NU = cp.meshgrid(tau, nu)

            S = cp.exp(-TAU / delay_spread_ms) * cp.exp(
                -(NU / doppler_spread_hz) ** 2
            )
            S = S / cp.max(S)

            return cp.asnumpy(S).astype(np.float32)
        except Exception:
            pass  # Fall through to numpy

    TAU, NU = np.meshgrid(delay_axis_ms, doppler_axis_hz)
    S = np.exp(-TAU / delay_spread_ms) * np.exp(-(NU / doppler_spread_hz) ** 2)
    S = S / np.max(S)

    return S.astype(np.float32)
