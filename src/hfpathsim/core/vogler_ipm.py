"""Python interface to Vogler-Hoffmeyer IPM GPU kernels."""

from typing import Optional, Tuple
import numpy as np

from .parameters import VoglerParameters


class VoglerIPM:
    """Vogler-Hoffmeyer Ionospheric Propagation Model.

    This class provides the interface between Python and the CUDA
    GPU kernels for computing the channel transfer function.
    """

    def __init__(self, use_gpu: bool = True):
        """Initialize the Vogler IPM interface.

        Args:
            use_gpu: Whether to use GPU acceleration
        """
        self.use_gpu = use_gpu
        self._gpu_available = False
        self._device_info = None

        if use_gpu:
            self._init_gpu()

    def _init_gpu(self):
        """Initialize GPU resources."""
        try:
            from hfpathsim import gpu

            self._device_info = gpu.get_device_info()
            self._gpu_available = True
            print(f"GPU initialized: {self._device_info['name']}")
        except ImportError as e:
            print(f"GPU module not available: {e}")
            print("Falling back to CPU implementation")
            self._gpu_available = False

    @property
    def gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self._gpu_available

    @property
    def device_info(self) -> Optional[dict]:
        """Get GPU device information."""
        return self._device_info

    def compute_reflection_coefficient(
        self,
        freq_hz: np.ndarray,
        params: VoglerParameters,
    ) -> np.ndarray:
        """Compute Vogler reflection coefficient R(omega).

        Args:
            freq_hz: Frequency array in Hz
            params: Vogler parameters

        Returns:
            Complex reflection coefficient array
        """
        if self._gpu_available:
            return self._compute_reflection_gpu(freq_hz, params)
        else:
            return self._compute_reflection_cpu(freq_hz, params)

    def _compute_reflection_gpu(
        self,
        freq_hz: np.ndarray,
        params: VoglerParameters,
    ) -> np.ndarray:
        """GPU implementation of reflection coefficient."""
        from hfpathsim import gpu

        return gpu.vogler_transfer_function(
            freq_hz.astype(np.float64),
            params.foF2,
            params.hmF2,
            params.sigma,
            params.chi,
            params.get_base_delay_ms() / 1000,
        )

    def _compute_reflection_cpu(
        self,
        freq_hz: np.ndarray,
        params: VoglerParameters,
    ) -> np.ndarray:
        """CPU fallback implementation of reflection coefficient."""
        from scipy.special import gamma as scipy_gamma

        fc = params.foF2 * 1e6
        omega_norm = freq_hz / fc
        sigma = params.sigma
        chi = params.chi
        t0 = params.get_base_delay_ms() / 1000

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

                phase = np.exp(-1j * 2 * np.pi * freq_hz[i] * t0)
                R[i] = (num / den) * phase

            except (ValueError, ZeroDivisionError):
                R[i] = 0.0

        return R.astype(np.complex64)

    def compute_transfer_function(
        self,
        freq_hz: np.ndarray,
        time_s: float,
        params: VoglerParameters,
        include_fading: bool = True,
    ) -> np.ndarray:
        """Compute time-varying transfer function H(f,t).

        Args:
            freq_hz: Frequency array in Hz
            time_s: Current time in seconds
            params: Vogler parameters
            include_fading: Whether to include stochastic fading

        Returns:
            Complex transfer function array
        """
        # Get base reflection coefficient
        R = self.compute_reflection_coefficient(freq_hz, params)

        if not include_fading:
            return R

        # Apply stochastic fading
        return self._apply_fading(R, freq_hz, params)

    def _apply_fading(
        self,
        R: np.ndarray,
        freq_hz: np.ndarray,
        params: VoglerParameters,
    ) -> np.ndarray:
        """Apply Gaussian scatter fading model."""
        N = len(R)

        # Doppler fading
        if params.doppler_spread_hz > 0:
            noise = (
                np.random.randn(N) + 1j * np.random.randn(N)
            ) / np.sqrt(2)

            doppler_filter = np.exp(
                -0.5 * (freq_hz / params.doppler_spread_hz) ** 2
            )
            doppler_filter /= np.sqrt(np.sum(doppler_filter**2) + 1e-10)

            fading = np.fft.ifft(np.fft.fft(noise) * doppler_filter)
            fading = fading / (np.std(fading) + 1e-10)

            R = R * (1 + 0.3 * fading)

        return R.astype(np.complex64)

    def apply_channel(
        self,
        input_signal: np.ndarray,
        H: np.ndarray,
        block_size: int = 4096,
        overlap: int = 1024,
    ) -> np.ndarray:
        """Apply channel transfer function to input signal.

        Uses overlap-save convolution for efficient block processing.

        Args:
            input_signal: Complex input signal
            H: Channel transfer function
            block_size: FFT block size
            overlap: Overlap samples

        Returns:
            Complex output signal
        """
        if self._gpu_available:
            return self._apply_channel_gpu(input_signal, H, block_size, overlap)
        else:
            return self._apply_channel_cpu(input_signal, H, block_size, overlap)

    def _apply_channel_gpu(
        self,
        input_signal: np.ndarray,
        H: np.ndarray,
        block_size: int,
        overlap: int,
    ) -> np.ndarray:
        """GPU implementation of channel application."""
        from hfpathsim import gpu

        return gpu.apply_channel(
            input_signal.astype(np.complex64),
            H.astype(np.complex64),
            block_size,
            overlap,
        )

    def _apply_channel_cpu(
        self,
        input_signal: np.ndarray,
        H: np.ndarray,
        block_size: int,
        overlap: int,
    ) -> np.ndarray:
        """CPU fallback implementation of channel application."""
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
        self,
        params: VoglerParameters,
        delay_axis_ms: np.ndarray,
        doppler_axis_hz: np.ndarray,
    ) -> np.ndarray:
        """Compute scattering function S(tau, nu).

        Args:
            params: Vogler parameters
            delay_axis_ms: Delay axis in milliseconds
            doppler_axis_hz: Doppler axis in Hz

        Returns:
            2D scattering function (delay x doppler)
        """
        TAU, NU = np.meshgrid(delay_axis_ms, doppler_axis_hz)

        delay_spread = params.delay_spread_ms
        doppler_spread = params.doppler_spread_hz

        S = np.exp(-TAU / delay_spread) * np.exp(-(NU / doppler_spread) ** 2)
        S = S / np.max(S)

        return S.astype(np.float32)
