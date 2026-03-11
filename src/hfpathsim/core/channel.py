"""HF Channel abstraction and processing pipeline."""

from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
import numpy as np
from scipy import signal
from scipy.special import gamma as scipy_gamma

from .parameters import VoglerParameters, ChannelState, PropagationMode


@dataclass
class ProcessingConfig:
    """Configuration for real-time channel processing."""

    sample_rate_hz: float = 2_000_000  # 2 Msps
    block_size: int = 4096  # FFT block size
    overlap: int = 1024  # Overlap for overlap-save
    channel_update_rate_hz: float = 10.0  # H(f,t) update rate


@dataclass
class RayTracingConfig:
    """Configuration for ray-traced propagation modes."""

    enabled: bool = False  # Use ray tracing for mode discovery
    tx_lat: float = 0.0  # Transmitter latitude (degrees)
    tx_lon: float = 0.0  # Transmitter longitude (degrees)
    rx_lat: float = 0.0  # Receiver latitude (degrees)
    rx_lon: float = 0.0  # Receiver longitude (degrees)
    max_hops: int = 3  # Maximum number of hops to consider
    use_sporadic_e: bool = False  # Include sporadic-E modes
    use_geomagnetic: bool = False  # Apply geomagnetic modulation


class HFChannel:
    """HF ionospheric channel simulator using Vogler-Hoffmeyer IPM.

    Implements the channel transfer function H(f,t) based on:
    - NTIA TR-88-240 reflection coefficient model
    - ITU-R F.1487 channel statistics
    - Gaussian scatter fading model

    Supports optional physics-based ray tracing for accurate:
    - Oblique incidence angle (sec_phi) computation
    - Multi-hop propagation mode discovery
    - Sporadic-E layer effects
    - Geomagnetic modulation
    """

    def __init__(
        self,
        params: Optional[VoglerParameters] = None,
        config: Optional[ProcessingConfig] = None,
        use_gpu: bool = True,
        use_ray_tracing: bool = False,
        ray_config: Optional[RayTracingConfig] = None,
    ):
        """Initialize the HF channel.

        Args:
            params: Vogler ionospheric parameters
            config: Processing configuration
            use_gpu: Whether to use GPU acceleration
            use_ray_tracing: Enable physics-based ray tracing
            ray_config: Ray tracing configuration
        """
        self.params = params or VoglerParameters()
        self.config = config or ProcessingConfig()
        self.use_gpu = use_gpu
        self.use_ray_tracing = use_ray_tracing
        self.ray_config = ray_config or RayTracingConfig()

        # State
        self._state = ChannelState()
        self._time = 0.0

        # GPU module (lazy load)
        self._gpu_module = None

        # Ray tracing components (lazy load)
        self._ionosphere_profile = None
        self._path_finder = None
        self._sporadic_e = None
        self._geomag_modulator = None

        # Pre-compute frequency axis
        self._init_axes()

        # Overlap-save state
        self._overlap_buffer = np.zeros(self.config.overlap, dtype=np.complex64)

        # Callbacks for state updates
        self._state_callbacks: list[Callable[[ChannelState], None]] = []

        # Initialize ray tracing if enabled
        if self.use_ray_tracing:
            self._init_ray_tracing()

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

    def _init_ray_tracing(self):
        """Initialize ray tracing components."""
        from .raytracing import (
            create_simple_profile,
            PathFinder,
            IonosphereProfile,
        )
        from hfpathsim.iono.sporadic_e import SporadicELayer, SporadicEConfig
        from hfpathsim.iono.geomagnetic import GeomagneticModulator, GeomagneticIndices

        # Create ionosphere profile from Vogler parameters
        self._ionosphere_profile = create_simple_profile(
            foF2=self.params.foF2,
            hmF2=self.params.hmF2,
            foE=self.params.foE,
            hmE=self.params.hmE,
            ym_F2=self.params.ym_F2,
            ym_E=self.params.ym_E,
        )

        # Create path finder
        self._path_finder = PathFinder(self._ionosphere_profile)

        # Initialize sporadic-E if enabled
        if self.ray_config.use_sporadic_e:
            self._sporadic_e = SporadicELayer(SporadicEConfig(enabled=True))

        # Initialize geomagnetic modulator if enabled
        if self.ray_config.use_geomagnetic:
            self._geomag_modulator = GeomagneticModulator(GeomagneticIndices.quiet())

    def update_ionosphere(
        self,
        foF2: Optional[float] = None,
        hmF2: Optional[float] = None,
        foE: Optional[float] = None,
        hmE: Optional[float] = None,
    ):
        """Update ionospheric parameters and recompute ray-traced modes.

        Args:
            foF2: F2 critical frequency (MHz)
            hmF2: F2 peak height (km)
            foE: E critical frequency (MHz)
            hmE: E peak height (km)
        """
        # Update Vogler parameters
        if foF2 is not None:
            self.params.foF2 = foF2
        if hmF2 is not None:
            self.params.hmF2 = hmF2
        if foE is not None:
            self.params.foE = foE
        if hmE is not None:
            self.params.hmE = hmE

        # Re-initialize ray tracing if enabled
        if self.use_ray_tracing:
            self._init_ray_tracing()
            self._update_modes_from_ray_tracing()

        # Recompute transfer function
        self._compute_transfer_function()

    def _update_modes_from_ray_tracing(self):
        """Update propagation modes using ray tracing."""
        if not self.use_ray_tracing or self._path_finder is None:
            return

        # Apply sporadic-E if enabled
        profile = self._ionosphere_profile
        if self._sporadic_e is not None and self._sporadic_e.enabled:
            profile = self._sporadic_e.inject(profile)
            self._path_finder = type(self._path_finder)(profile)

        # Apply geomagnetic modulation if enabled
        if self._geomag_modulator is not None:
            latitude = (self.ray_config.tx_lat + self.ray_config.rx_lat) / 2
            profile = self._geomag_modulator.apply_to_profile(profile, latitude)
            self._path_finder = type(self._path_finder)(profile)

        # Find propagation modes
        modes = self._path_finder.find_modes(
            frequency_mhz=self.params.frequency_mhz,
            tx_lat=self.ray_config.tx_lat,
            tx_lon=self.ray_config.tx_lon,
            rx_lat=self.ray_config.rx_lat,
            rx_lon=self.ray_config.rx_lon,
            max_hops=self.ray_config.max_hops,
        )

        # Convert to PropagationMode objects
        self.params.modes = [
            PropagationMode(
                name=m.name,
                enabled=True,
                relative_amplitude=m.relative_amplitude,
                delay_offset_ms=m.delay_offset_ms,
                n_hops=m.n_hops,
                reflection_height_km=m.reflection_height_km,
                layer=m.layer,
                sec_phi=m.sec_phi,
                launch_angle_deg=m.launch_angle_deg,
            )
            for m in modes
        ]

    def set_path(
        self,
        tx_lat: float,
        tx_lon: float,
        rx_lat: float,
        rx_lon: float,
    ):
        """Set transmitter and receiver locations for ray tracing.

        Args:
            tx_lat: Transmitter latitude (degrees)
            tx_lon: Transmitter longitude (degrees)
            rx_lat: Receiver latitude (degrees)
            rx_lon: Receiver longitude (degrees)
        """
        self.ray_config.tx_lat = tx_lat
        self.ray_config.tx_lon = tx_lon
        self.ray_config.rx_lat = rx_lat
        self.ray_config.rx_lon = rx_lon

        # Update path length in Vogler parameters
        from .raytracing.geometry import great_circle_distance
        self.params.path_length_km = great_circle_distance(
            tx_lat, tx_lon, rx_lat, rx_lon
        )

        # Update modes if ray tracing enabled
        if self.use_ray_tracing:
            self._update_modes_from_ray_tracing()
            self._compute_transfer_function()

    def enable_sporadic_e(self, foEs: float = 5.0, hmEs: float = 105.0):
        """Enable sporadic-E layer.

        Args:
            foEs: Sporadic-E critical frequency (MHz)
            hmEs: Sporadic-E height (km)
        """
        from hfpathsim.iono.sporadic_e import SporadicELayer, SporadicEConfig

        self._sporadic_e = SporadicELayer(
            SporadicEConfig(enabled=True, foEs_mhz=foEs, hmEs_km=hmEs)
        )
        self.ray_config.use_sporadic_e = True

        if self.use_ray_tracing:
            self._update_modes_from_ray_tracing()
            self._compute_transfer_function()

    def disable_sporadic_e(self):
        """Disable sporadic-E layer."""
        if self._sporadic_e is not None:
            self._sporadic_e.disable()
        self.ray_config.use_sporadic_e = False

        if self.use_ray_tracing:
            self._update_modes_from_ray_tracing()
            self._compute_transfer_function()

    def set_geomagnetic_indices(
        self,
        f10_7: float = 100.0,
        kp: float = 2.0,
        dst: float = 0.0,
    ):
        """Set geomagnetic indices for ionospheric modulation.

        Args:
            f10_7: F10.7 solar flux (sfu)
            kp: Kp geomagnetic index (0-9)
            dst: Dst storm-time index (nT)
        """
        from hfpathsim.iono.geomagnetic import GeomagneticModulator, GeomagneticIndices

        indices = GeomagneticIndices(f10_7=f10_7, kp=kp, dst=dst)
        self._geomag_modulator = GeomagneticModulator(indices)
        self.ray_config.use_geomagnetic = True

        if self.use_ray_tracing:
            self._update_modes_from_ray_tracing()
            self._compute_transfer_function()

    def get_muf(self, layer: str = "F2") -> float:
        """Get Maximum Usable Frequency for current path.

        Args:
            layer: Ionospheric layer ("F2", "E", or "Es")

        Returns:
            MUF in MHz
        """
        if self.use_ray_tracing and self._path_finder is not None:
            from .raytracing.path_finder import estimate_muf
            return estimate_muf(
                self._ionosphere_profile,
                self.params.path_length_km,
                layer,
            )
        else:
            return self.params.get_muf(layer)

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

        # Check for gamma function poles:
        # gamma(0.5 + chi) has pole when chi = -0.5, -1.5, -2.5, ...
        # gamma(0.5 - chi) has pole when chi = 0.5, 1.5, 2.5, ...
        # When chi <= -0.5, frequency is above MUF - no reflection possible
        if chi is None or chi <= -0.49:
            # No reflection - return zero with proper shape
            return np.zeros_like(omega_norm, dtype=np.complex64)

        # Check for pole at chi near 0.5 (very close to critical frequency)
        if abs(chi - 0.5) < 0.01:
            chi = 0.51  # Nudge away from pole

        # Base propagation delay
        t0 = self.params.get_base_delay_ms() / 1000  # Convert to seconds

        # Compute reflection coefficient using gamma functions
        # Use scipy's gamma function for complex arguments
        R = np.zeros_like(omega_norm, dtype=np.complex128)

        # Pre-compute denominator real gamma values (same for all frequencies)
        try:
            g5 = scipy_gamma(0.5 - chi)
            g6 = scipy_gamma(0.5 + chi)
            den_real = g5 * g6
            if not np.isfinite(den_real) or abs(den_real) < 1e-30:
                return np.zeros_like(omega_norm, dtype=np.complex64)
        except (ValueError, OverflowError):
            return np.zeros_like(omega_norm, dtype=np.complex64)

        for i, omega in enumerate(omega_norm):
            try:
                # Numerator terms
                g1 = scipy_gamma(complex(1, -sigma * omega))
                g2 = scipy_gamma(complex(0.5 - chi, sigma * omega))
                g3 = scipy_gamma(complex(0.5 + chi, sigma * omega))
                num = g1 * g2 * g3

                # Denominator terms
                g4 = scipy_gamma(complex(1, sigma * omega))
                den = g4 * den_real

                # Check for invalid values
                if not (np.isfinite(num) and np.isfinite(den) and abs(den) > 1e-30):
                    R[i] = 0.0
                    continue

                # Phase from propagation delay
                phase = np.exp(-1j * 2 * np.pi * freq_hz[i] * t0)

                result = (num / den) * phase

                # Final sanity check
                if np.isfinite(result):
                    R[i] = result
                else:
                    R[i] = 0.0

            except (ValueError, ZeroDivisionError, OverflowError):
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
            std_fading = np.std(fading)
            if std_fading > 1e-10:
                fading = fading / std_fading  # Normalize
                R = R * (1 + 0.3 * fading)  # Modulate amplitude
            # else: skip fading if std is zero (degenerate case)

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
