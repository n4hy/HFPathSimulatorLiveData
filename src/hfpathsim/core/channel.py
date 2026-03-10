"""HF Channel abstraction and processing pipeline."""

from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np
from scipy import signal
from scipy.special import gamma as scipy_gamma

from .parameters import VoglerParameters, ChannelState


@dataclass
class ProcessingConfig:
    """Configuration for real-time channel processing."""

    sample_rate_hz: float = 2_000_000  # 2 Msps
    block_size: int = 4096  # FFT block size
    overlap: int = 1024  # Overlap for overlap-save
    channel_update_rate_hz: float = 10.0  # H(f,t) update rate


class HFChannel:
    """HF ionospheric channel simulator using Vogler-Hoffmeyer IPM.

    Implements the channel transfer function H(f,t) based on:
    - NTIA TR-88-240 reflection coefficient model
    - ITU-R F.1487 channel statistics
    - Gaussian scatter fading model
    """

    def __init__(
        self,
        params: Optional[VoglerParameters] = None,
        config: Optional[ProcessingConfig] = None,
        use_gpu: bool = True,
    ):
        """Initialize the HF channel.

        Args:
            params: Vogler ionospheric parameters
            config: Processing configuration
            use_gpu: Whether to use GPU acceleration
        """
        self.params = params or VoglerParameters()
        self.config = config or ProcessingConfig()
        self.use_gpu = use_gpu

        # State
        self._state = ChannelState()
        self._time = 0.0

        # GPU module (lazy load)
        self._gpu_module = None

        # Pre-compute frequency axis
        self._init_axes()

        # Overlap-save state
        self._overlap_buffer = np.zeros(self.config.overlap, dtype=np.complex64)

        # Callbacks for state updates
        self._state_callbacks: list[Callable[[ChannelState], None]] = []

    def _init_axes(self):
        """Initialize frequency and delay axes."""
        N = self.config.block_size
        fs = self.config.sample_rate_hz

        # Frequency axis (centered)
        self._freq_axis = np.fft.fftfreq(N, 1 / fs)

        # Delay axis for impulse response
        self._delay_axis = np.arange(N) / fs * 1000  # ms

        # Doppler axis for scattering function
        # Use same resolution as channel update rate
        doppler_max = 20.0  # Hz, covers most HF cases
        self._doppler_axis = np.linspace(-doppler_max, doppler_max, 64)

    def _load_gpu_module(self):
        """Lazy load GPU acceleration module."""
        if self._gpu_module is None and self.use_gpu:
            try:
                from hfpathsim.gpu import get_device_info, vogler_transfer_function

                info = get_device_info()
                print(f"GPU: {info['name']}, Compute {info['compute_capability']}")
                self._gpu_module = True
            except ImportError:
                print("GPU module not available, using CPU fallback")
                self._gpu_module = False
                self.use_gpu = False

    def update_parameters(self, params: VoglerParameters):
        """Update channel parameters and recompute transfer function."""
        self.params = params
        self._compute_transfer_function()

    def _compute_reflection_coefficient(self, freq_hz: np.ndarray) -> np.ndarray:
        """Compute Vogler reflection coefficient R(omega).

        Based on NTIA TR-88-240 equation for quasi-parabolic layer.

        R(ω) = Γ(1-iσω)Γ(1/2-χ+iσω)Γ(1/2+χ+iσω)e^(-iωt₀)
               ─────────────────────────────────────────────
               Γ(1+iσω)Γ(1/2-χ)Γ(1/2+χ)

        Args:
            freq_hz: Frequency array in Hz

        Returns:
            Complex reflection coefficient array
        """
        # Convert to normalized angular frequency
        # Normalize to critical frequency
        fc = self.params.foF2 * 1e6  # Convert MHz to Hz
        omega_norm = freq_hz / fc  # Normalized frequency

        sigma = self.params.sigma
        chi = self.params.chi

        # Base propagation delay
        t0 = self.params.get_base_delay_ms() / 1000  # Convert to seconds

        # Compute reflection coefficient using gamma functions
        # Use scipy's gamma function for complex arguments
        R = np.zeros_like(omega_norm, dtype=np.complex128)

        for i, omega in enumerate(omega_norm):
            try:
                # Numerator terms
                g1 = scipy_gamma(complex(1, -sigma * omega))
                g2 = scipy_gamma(complex(0.5 - chi, sigma * omega))
                g3 = scipy_gamma(complex(0.5 + chi, sigma * omega))
                num = g1 * g2 * g3

                # Denominator terms
                g4 = scipy_gamma(complex(1, sigma * omega))
                g5 = scipy_gamma(0.5 - chi)
                g6 = scipy_gamma(0.5 + chi)
                den = g4 * g5 * g6

                # Phase from propagation delay
                phase = np.exp(-1j * 2 * np.pi * freq_hz[i] * t0)

                R[i] = (num / den) * phase

            except (ValueError, ZeroDivisionError):
                # Handle edge cases (poles of gamma function)
                R[i] = 0.0

        return R.astype(np.complex64)

    def _apply_fading(self, R: np.ndarray) -> np.ndarray:
        """Apply stochastic fading to reflection coefficient.

        Uses Gaussian scatter model for delay-Doppler spreading.

        Args:
            R: Base reflection coefficient

        Returns:
            Faded reflection coefficient H(f)
        """
        N = len(R)

        # Doppler fading - complex Gaussian in time domain
        doppler_spread = self.params.doppler_spread_hz
        if doppler_spread > 0:
            # Generate complex Gaussian noise
            noise = (
                np.random.randn(N) + 1j * np.random.randn(N)
            ) / np.sqrt(2)

            # Shape with Gaussian Doppler spectrum
            doppler_filter = np.exp(
                -0.5 * (self._freq_axis / doppler_spread) ** 2
            )
            doppler_filter /= np.sqrt(np.sum(doppler_filter**2))

            # Apply in frequency domain
            fading = np.fft.ifft(np.fft.fft(noise) * doppler_filter)
            fading = fading / np.std(fading)  # Normalize

            R = R * (1 + 0.3 * fading)  # Modulate amplitude

        # Delay spreading - convolve with exponential decay
        delay_spread = self.params.delay_spread_ms
        if delay_spread > 0:
            # Create delay spread filter
            tau = np.arange(N) / self.config.sample_rate_hz * 1000  # ms
            delay_filter = np.exp(-tau / delay_spread)
            delay_filter = delay_filter / np.sum(delay_filter)

            # Apply in frequency domain (multiplication)
            H_delay = np.fft.fft(delay_filter)
            R = R * H_delay

        return R.astype(np.complex64)

    def _compute_transfer_function(self):
        """Compute complete channel transfer function H(f,t)."""
        # Get base reflection coefficient
        R = self._compute_reflection_coefficient(self._freq_axis)

        # Apply fading
        H = self._apply_fading(R)

        # Sum contributions from multiple modes
        H_total = np.zeros_like(H)
        for mode in self.params.modes:
            if not mode.enabled:
                continue

            # Apply mode-specific delay and amplitude
            delay_samples = int(
                mode.delay_offset_ms / 1000 * self.config.sample_rate_hz
            )
            phase_shift = np.exp(
                -1j * 2 * np.pi * self._freq_axis * mode.delay_offset_ms / 1000
            )
            H_total += mode.relative_amplitude * H * phase_shift

        # Normalize
        if np.max(np.abs(H_total)) > 0:
            H_total = H_total / np.max(np.abs(H_total))

        # Update state
        self._state.transfer_function = H_total
        self._state.freq_axis_hz = self._freq_axis
        self._state.timestamp = self._time

        # Compute impulse response
        h = np.fft.ifft(H_total)
        self._state.impulse_response = h
        self._state.delay_axis_ms = self._delay_axis

        # Compute scattering function (simplified)
        self._compute_scattering_function(H_total)

        # Notify callbacks
        for callback in self._state_callbacks:
            callback(self._state)

    def _compute_scattering_function(self, H: np.ndarray):
        """Compute scattering function S(τ, ν) from transfer function.

        The scattering function is the 2D Fourier transform of the
        time-frequency correlation function.
        """
        N_delay = self.config.block_size
        N_doppler = len(self._doppler_axis)

        # For now, use a simplified model based on parameters
        # Full computation requires time history of H(f,t)

        # Create delay-Doppler power distribution
        tau = self._delay_axis
        nu = self._doppler_axis

        # Gaussian in both dimensions
        delay_spread = self.params.delay_spread_ms
        doppler_spread = self.params.doppler_spread_hz

        # 2D meshgrid
        TAU, NU = np.meshgrid(tau[:128], nu)  # Truncate delay for display

        # Scattering function shape
        S = np.exp(-TAU / delay_spread) * np.exp(-(NU / doppler_spread) ** 2)
        S = S / np.max(S)

        self._state.scattering_function = S.astype(np.float32)
        self._state.doppler_axis_hz = nu

    def process(self, input_samples: np.ndarray) -> np.ndarray:
        """Process input samples through the channel.

        Uses overlap-save convolution for block processing.

        Args:
            input_samples: Complex input samples

        Returns:
            Complex output samples after channel
        """
        # Ensure transfer function is computed
        if self._state.transfer_function is None:
            self._compute_transfer_function()

        block_size = self.config.block_size
        overlap = self.config.overlap
        output_size = block_size - overlap

        # Pad input to block boundary
        n_samples = len(input_samples)
        n_blocks = (n_samples + output_size - 1) // output_size
        padded_length = n_blocks * output_size + overlap
        padded_input = np.zeros(padded_length, dtype=np.complex64)
        padded_input[overlap : overlap + n_samples] = input_samples

        # Initialize output
        output = np.zeros(n_blocks * output_size, dtype=np.complex64)

        # Process blocks
        for i in range(n_blocks):
            # Extract block with overlap
            start = i * output_size
            block = padded_input[start : start + block_size]

            # FFT
            X = np.fft.fft(block)

            # Apply channel
            Y = X * self._state.transfer_function

            # IFFT
            y = np.fft.ifft(Y)

            # Save non-overlapping portion
            output[i * output_size : (i + 1) * output_size] = y[overlap:]

        # Update time
        self._time += n_samples / self.config.sample_rate_hz

        # Periodically update channel
        if (
            self._time * self.config.channel_update_rate_hz
        ) % 1 < n_samples / self.config.sample_rate_hz:
            self._compute_transfer_function()

        return output[:n_samples]

    def get_state(self) -> ChannelState:
        """Get current channel state for visualization."""
        if self._state.transfer_function is None:
            self._compute_transfer_function()
        return self._state

    def add_state_callback(self, callback: Callable[[ChannelState], None]):
        """Register callback for state updates."""
        self._state_callbacks.append(callback)

    def remove_state_callback(self, callback: Callable[[ChannelState], None]):
        """Remove state update callback."""
        if callback in self._state_callbacks:
            self._state_callbacks.remove(callback)
