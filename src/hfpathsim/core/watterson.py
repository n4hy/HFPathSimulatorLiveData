"""Watterson HF channel model implementation.

The Watterson model is a tapped delay line (TDL) model where each tap
represents a propagation mode with independent Rayleigh or Rician fading.

Uses CUDA/C++ compiled implementations when available with automatic
fallback to pure Python.

Reference: Watterson, C.C., Juroshek, J.R., and Bensema, W.D.,
"Experimental confirmation of an HF channel model," IEEE Trans.
Comm. Tech., vol. COM-18, pp. 792-803, Dec. 1970.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Callable
import numpy as np
from scipy import signal

from .parameters import ITUCondition

# Try to import compiled GPU implementations
try:
    from ..gpu import WattersonProcessor as _WattersonProcessor
    _HAS_COMPILED = _WattersonProcessor is not None
except ImportError:
    _HAS_COMPILED = False
    _WattersonProcessor = None


class DopplerSpectrum(Enum):
    """Doppler spectrum shape options."""

    GAUSSIAN = "gaussian"  # Gaussian (default for HF)
    FLAT = "flat"  # Flat (uniform)
    JAKES = "jakes"  # Jakes/Clarke (mobile)


@dataclass
class WattersonTap:
    """Single tap in Watterson tapped delay line model.

    Each tap represents one propagation mode (e.g., 1F2, 2F2).
    """

    delay_ms: float = 0.0  # Tap delay in milliseconds
    amplitude: float = 1.0  # Relative amplitude (linear)
    doppler_spread_hz: float = 1.0  # Two-sided Doppler spread
    doppler_spectrum: DopplerSpectrum = DopplerSpectrum.GAUSSIAN
    is_specular: bool = False  # True for Rician (has LOS component)
    k_factor_db: float = 0.0  # Rician K-factor (specular/scatter ratio)

    def __post_init__(self):
        """Validate parameters."""
        if self.delay_ms < 0:
            raise ValueError("Delay must be non-negative")
        if self.amplitude < 0:
            raise ValueError("Amplitude must be non-negative")
        if self.doppler_spread_hz < 0:
            raise ValueError("Doppler spread must be non-negative")


@dataclass
class WattersonConfig:
    """Configuration for Watterson channel model."""

    taps: List[WattersonTap] = field(default_factory=list)
    sample_rate_hz: float = 2_000_000
    block_size: int = 4096
    update_rate_hz: float = 100.0  # Fading coefficient update rate

    @classmethod
    def from_itu_condition(
        cls,
        condition: ITUCondition,
        sample_rate_hz: float = 2_000_000,
    ) -> "WattersonConfig":
        """Create Watterson config from ITU-R F.1487 condition.

        ITU-R F.1487 Table 1 defines channel conditions with
        delay spread and Doppler spread parameters.

        Args:
            condition: ITU channel condition
            sample_rate_hz: Sample rate

        Returns:
            WattersonConfig with appropriate tap configuration
        """
        if condition == ITUCondition.QUIET:
            # Good conditions: 2 paths, low spread
            taps = [
                WattersonTap(delay_ms=0.0, amplitude=1.0, doppler_spread_hz=0.1),
                WattersonTap(delay_ms=0.5, amplitude=0.5, doppler_spread_hz=0.1),
            ]

        elif condition == ITUCondition.MODERATE:
            # Moderate: 2 paths, typical spread
            taps = [
                WattersonTap(delay_ms=0.0, amplitude=1.0, doppler_spread_hz=1.0),
                WattersonTap(delay_ms=2.0, amplitude=0.7, doppler_spread_hz=1.0),
            ]

        elif condition == ITUCondition.DISTURBED:
            # Disturbed: 3 paths, high spread
            taps = [
                WattersonTap(delay_ms=0.0, amplitude=1.0, doppler_spread_hz=2.0),
                WattersonTap(delay_ms=2.0, amplitude=0.7, doppler_spread_hz=2.0),
                WattersonTap(delay_ms=4.0, amplitude=0.5, doppler_spread_hz=2.0),
            ]

        elif condition == ITUCondition.FLUTTER:
            # Flutter fading: 2 paths, very high Doppler
            taps = [
                WattersonTap(delay_ms=0.0, amplitude=1.0, doppler_spread_hz=10.0),
                WattersonTap(delay_ms=1.0, amplitude=0.8, doppler_spread_hz=10.0),
            ]

        else:
            # Default: single tap
            taps = [
                WattersonTap(delay_ms=0.0, amplitude=1.0, doppler_spread_hz=1.0),
            ]

        return cls(taps=taps, sample_rate_hz=sample_rate_hz)

    @classmethod
    def ccir_good(cls, sample_rate_hz: float = 2_000_000) -> "WattersonConfig":
        """CCIR 'Good' channel (legacy designation).

        2 paths, 0.5ms differential delay, 0.1Hz Doppler.
        """
        return cls(
            taps=[
                WattersonTap(delay_ms=0.0, amplitude=1.0, doppler_spread_hz=0.1),
                WattersonTap(delay_ms=0.5, amplitude=1.0, doppler_spread_hz=0.1),
            ],
            sample_rate_hz=sample_rate_hz,
        )

    @classmethod
    def ccir_moderate(cls, sample_rate_hz: float = 2_000_000) -> "WattersonConfig":
        """CCIR 'Moderate' channel.

        2 paths, 1.0ms differential delay, 0.5Hz Doppler.
        """
        return cls(
            taps=[
                WattersonTap(delay_ms=0.0, amplitude=1.0, doppler_spread_hz=0.5),
                WattersonTap(delay_ms=1.0, amplitude=1.0, doppler_spread_hz=0.5),
            ],
            sample_rate_hz=sample_rate_hz,
        )

    @classmethod
    def ccir_poor(cls, sample_rate_hz: float = 2_000_000) -> "WattersonConfig":
        """CCIR 'Poor' channel.

        2 paths, 2.0ms differential delay, 1.0Hz Doppler.
        """
        return cls(
            taps=[
                WattersonTap(delay_ms=0.0, amplitude=1.0, doppler_spread_hz=1.0),
                WattersonTap(delay_ms=2.0, amplitude=1.0, doppler_spread_hz=1.0),
            ],
            sample_rate_hz=sample_rate_hz,
        )


class WattersonChannel:
    """Watterson tapped delay line channel model.

    Implements the classic Watterson HF channel model with:
    - Multiple taps at configurable delays
    - Independent Rayleigh/Rician fading per tap
    - Gaussian Doppler spectrum (default)
    - Real-time fading coefficient generation

    Uses CUDA/C++ implementation when available for high performance.
    """

    def __init__(
        self,
        config: Optional[WattersonConfig] = None,
        seed: Optional[int] = None,
        use_compiled: bool = True,
        max_samples: int = 65536,
    ):
        """Initialize Watterson channel.

        Args:
            config: Channel configuration
            seed: Random seed for reproducibility
            use_compiled: Use compiled CUDA/C++ implementation if available
            max_samples: Maximum samples per block (for compiled backend)
        """
        self.config = config or WattersonConfig.from_itu_condition(
            ITUCondition.MODERATE
        )

        # Random state
        self._rng = np.random.default_rng(seed)
        self._seed = seed if seed is not None else 42

        # Fading state for each tap
        self._tap_states: List[dict] = []
        self._init_tap_states()

        # Processing state
        self._time = 0.0
        self._sample_count = 0

        # Callbacks
        self._state_callbacks: List[Callable] = []

        # Try to use compiled implementation
        # C++ signature: (sample_rate, max_taps, max_delay_samples, max_samples, seed)
        self._compiled_processor = None
        self._max_samples = max_samples
        if use_compiled and _HAS_COMPILED and len(self.config.taps) > 0:
            try:
                # Calculate max delay in samples
                max_delay_ms = max(tap.delay_ms for tap in self.config.taps)
                max_delay_samples = int(max_delay_ms / 1000.0 * self.config.sample_rate_hz) + 1024

                self._compiled_processor = _WattersonProcessor(
                    self.config.sample_rate_hz,
                    len(self.config.taps),
                    max_delay_samples,
                    max_samples,
                    self._seed,
                )
                # Configure taps
                self._configure_compiled_taps()
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._compiled_processor = None

    def _configure_compiled_taps(self):
        """Configure taps in the compiled processor."""
        if self._compiled_processor is None:
            return

        n_taps = len(self.config.taps)

        # Build arrays for all taps
        delays = np.zeros(n_taps, dtype=np.int32)
        amplitudes = np.zeros(n_taps, dtype=np.float32)
        doppler_spreads = np.zeros(n_taps, dtype=np.float32)
        spectrum_types = np.zeros(n_taps, dtype=np.int32)
        is_rician = np.zeros(n_taps, dtype=np.int32)
        k_factors = np.zeros(n_taps, dtype=np.float32)

        # Map DopplerSpectrum enum to int: GAUSSIAN=0, FLAT=1, JAKES=2
        spectrum_map = {
            DopplerSpectrum.GAUSSIAN: 0,
            DopplerSpectrum.FLAT: 1,
            DopplerSpectrum.JAKES: 2,
        }

        for i, tap in enumerate(self.config.taps):
            # Convert delay from ms to samples
            delays[i] = int(tap.delay_ms / 1000.0 * self.config.sample_rate_hz)
            amplitudes[i] = tap.amplitude
            doppler_spreads[i] = tap.doppler_spread_hz
            spectrum_types[i] = spectrum_map.get(tap.doppler_spectrum, 0)
            is_rician[i] = 1 if tap.is_specular else 0
            k_factors[i] = tap.k_factor_db

        try:
            self._compiled_processor.configure_taps(
                delays,
                amplitudes,
                doppler_spreads,
                spectrum_types,
                is_rician,
                k_factors,
                self.config.update_rate_hz,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._compiled_processor = None

    def _init_tap_states(self):
        """Initialize fading generators for each tap."""
        self._tap_states = []

        # Calculate actual block update rate (blocks per second)
        # With 50ms blocks at 8kHz, this is 20 Hz
        block_size = max(1, int(self.config.sample_rate_hz * 0.05))  # 50ms blocks
        actual_update_rate = self.config.sample_rate_hz / block_size

        for tap in self.config.taps:
            # Delay in samples
            delay_samples = int(
                tap.delay_ms / 1000.0 * self.config.sample_rate_hz
            )

            # Doppler filter design - filter length based on ACTUAL update rate
            # For Rayleigh fading, we need ~10 samples per fade cycle
            if tap.doppler_spread_hz > 0:
                # Coherence time in blocks
                coherence_blocks = actual_update_rate / tap.doppler_spread_hz
                # Filter needs to span a few coherence times
                filter_len = max(16, min(128, int(coherence_blocks * 2)))
            else:
                filter_len = 16

            # Create Doppler shaping filter
            doppler_filter = self._create_doppler_filter(
                tap.doppler_spread_hz,
                tap.doppler_spectrum,
                filter_len,
                actual_update_rate,
            )

            state = {
                "tap": tap,
                "delay_samples": delay_samples,
                "doppler_filter": doppler_filter,
                "filter_state": np.zeros(len(doppler_filter) - 1, dtype=np.complex128),
                "current_gain": complex(tap.amplitude, 0.0),
                # Noise buffer initialized with complex Gaussian
                "noise_buffer": (self._rng.standard_normal(filter_len)
                    + 1j * self._rng.standard_normal(filter_len)) / np.sqrt(2),
                # Delay line buffer for phase-continuous processing
                "delay_buffer": np.zeros(max(1, delay_samples), dtype=np.complex128),
                # Store actual update rate for diagnostics
                "actual_update_rate": actual_update_rate,
            }

            self._tap_states.append(state)

    def _create_doppler_filter(
        self,
        doppler_spread_hz: float,
        spectrum_type: DopplerSpectrum,
        length: int,
        actual_update_rate: float = None,
    ) -> np.ndarray:
        """Create FIR filter for Doppler spectrum shaping.

        The filter shapes white noise to have the desired Doppler spectrum.
        For Rayleigh fading, the output should have unit variance so that
        the magnitude follows a Rayleigh distribution.

        Args:
            doppler_spread_hz: Two-sided Doppler spread
            spectrum_type: Doppler spectrum shape
            length: Filter length
            actual_update_rate: Actual block update rate (Hz)

        Returns:
            FIR filter coefficients
        """
        if doppler_spread_hz <= 0:
            # No fading - return delta function
            h = np.zeros(length)
            h[length // 2] = 1.0
            return h

        # Use actual update rate if provided, otherwise config value
        update_rate = actual_update_rate or self.config.update_rate_hz

        # Normalized Doppler frequency (relative to update rate)
        f_norm = doppler_spread_hz / update_rate

        if spectrum_type == DopplerSpectrum.GAUSSIAN:
            # Gaussian spectrum -> Gaussian impulse response
            # The sigma controls the correlation time
            # Smaller sigma = faster fading = more variation per block
            t = np.arange(length) - length // 2
            # sigma in samples = 1/(2*pi*f_norm) to match Doppler spread
            sigma = max(1.0, 1.0 / (2 * np.pi * f_norm)) if f_norm > 0 else length / 4
            h = np.exp(-0.5 * (t / sigma) ** 2)

        elif spectrum_type == DopplerSpectrum.FLAT:
            # Flat spectrum -> sinc impulse response
            t = np.arange(length) - length // 2
            t = np.where(t == 0, 1e-10, t)
            h = np.sinc(2 * f_norm * t)

        elif spectrum_type == DopplerSpectrum.JAKES:
            # Jakes spectrum (U-shaped)
            t = np.arange(length) - length // 2
            t = np.where(t == 0, 1e-10, t)
            h = np.sinc(2 * f_norm * t) * np.cos(np.pi * f_norm * t)

        else:
            h = np.ones(length)

        # Normalize for unit variance output (Rayleigh fading)
        # This ensures |output|^2 has mean = 1
        h = h / np.sqrt(np.sum(h**2))

        return h.astype(np.complex128)

    def _update_fading_coefficients(self):
        """Update fading coefficients for all taps.

        Generates Rayleigh-distributed fading for each tap by filtering
        complex Gaussian noise through a Doppler-shaping filter.
        The filtered output is complex Gaussian, so its magnitude
        follows a Rayleigh distribution with deep fades.
        """
        for state in self._tap_states:
            tap = state["tap"]

            # Generate new complex Gaussian noise sample (unit variance per component)
            noise = (
                self._rng.standard_normal()
                + 1j * self._rng.standard_normal()
            ) / np.sqrt(2)

            # Shift noise buffer and add new sample
            state["noise_buffer"] = np.roll(state["noise_buffer"], -1)
            state["noise_buffer"][-1] = noise

            # Convolve with Doppler filter (matched to spectrum shape)
            # The filter is normalized so output has unit variance
            filtered = np.sum(state["noise_buffer"] * state["doppler_filter"])

            # For Rayleigh fading, the magnitude of 'filtered' follows Rayleigh distribution
            # Mean magnitude = sqrt(pi/2) ≈ 1.25, but with deep fades near zero
            # Scale by tap amplitude
            scatter_component = filtered * tap.amplitude

            # Add specular (LOS) component for Rician fading
            if tap.is_specular:
                k_linear = 10 ** (tap.k_factor_db / 10)
                # Specular component is constant (deterministic)
                specular = np.sqrt(k_linear / (1 + k_linear)) * tap.amplitude
                # Scale scatter component
                scatter_component *= np.sqrt(1 / (1 + k_linear))
                state["current_gain"] = specular + scatter_component
            else:
                # Pure Rayleigh (no specular component)
                state["current_gain"] = scatter_component

    def process(self, input_samples: np.ndarray) -> np.ndarray:
        """Process samples through the Watterson channel.

        Args:
            input_samples: Complex input samples

        Returns:
            Complex output samples after channel
        """
        # Use compiled implementation if available
        if self._compiled_processor is not None:
            try:
                input_real = np.ascontiguousarray(input_samples.real, dtype=np.float32)
                input_imag = np.ascontiguousarray(input_samples.imag, dtype=np.float32)
                output_real, output_imag = self._compiled_processor.process(
                    input_real, input_imag
                )
                n_samples = len(input_samples)
                self._sample_count += n_samples
                self._time += n_samples / self.config.sample_rate_hz
                # Notify callbacks
                for callback in self._state_callbacks:
                    callback(self.get_state())
                return (output_real + 1j * output_imag).astype(np.complex64)
            except Exception:
                pass  # Fall through to Python implementation

        # Python fallback implementation
        n_samples = len(input_samples)
        output = np.zeros(n_samples, dtype=np.complex128)

        # Find maximum delay for output buffer sizing
        max_delay = max(s["delay_samples"] for s in self._tap_states)

        # Process sample by sample (can be optimized with block processing)
        samples_per_update = int(
            self.config.sample_rate_hz / self.config.update_rate_hz
        )

        for i in range(n_samples):
            # Update fading coefficients periodically
            if self._sample_count % samples_per_update == 0:
                self._update_fading_coefficients()

            # Sum contributions from all taps
            for state in self._tap_states:
                delay = state["delay_samples"]
                src_idx = i - delay

                if 0 <= src_idx < n_samples:
                    output[i] += input_samples[src_idx] * state["current_gain"]

            self._sample_count += 1

        self._time += n_samples / self.config.sample_rate_hz

        # Notify callbacks
        for callback in self._state_callbacks:
            callback(self.get_state())

        return output.astype(np.complex64)

    def process_block(self, input_samples: np.ndarray) -> np.ndarray:
        """Process a block of samples with phase-continuous delay lines.

        Uses delay line buffers to maintain continuity across block boundaries.
        Gain is interpolated across the block to avoid discontinuities at
        block boundaries. Delay buffer stores RAW samples without gain;
        gain is applied at output time.

        Args:
            input_samples: Complex input samples

        Returns:
            Complex output samples
        """
        # Use compiled implementation if available
        if self._compiled_processor is not None:
            try:
                input_real = np.ascontiguousarray(input_samples.real, dtype=np.float32)
                input_imag = np.ascontiguousarray(input_samples.imag, dtype=np.float32)
                output_real, output_imag = self._compiled_processor.process(
                    input_real, input_imag
                )
                n_samples = len(input_samples)
                self._sample_count += n_samples
                self._time += n_samples / self.config.sample_rate_hz
                return (output_real + 1j * output_imag).astype(np.complex64)
            except Exception:
                pass  # Fall through to Python implementation

        # Python fallback implementation
        n_samples = len(input_samples)
        output = np.zeros(n_samples, dtype=np.complex128)

        # Save old gains BEFORE updating (for interpolation)
        old_gains = [state["current_gain"] for state in self._tap_states]

        # Update fading at block rate
        self._update_fading_coefficients()

        # Precompute interpolation weights (0 at start of block, 1 at end)
        t = np.linspace(0, 1, n_samples, endpoint=False, dtype=np.float64)

        # Process each tap with proper delay line and gain interpolation
        for idx, state in enumerate(self._tap_states):
            delay = state["delay_samples"]
            delay_buffer = state["delay_buffer"]

            # Interpolate gain smoothly across block to avoid clicks
            gain_old = old_gains[idx]
            gain_new = state["current_gain"]
            interpolated_gain = gain_old * (1 - t) + gain_new * t

            if delay == 0:
                # No delay - direct path with interpolated gain
                output += input_samples * interpolated_gain
            else:
                # Get raw samples that will be output this block
                # delay_buffer stores RAW samples (no gain applied)
                extended = np.concatenate([delay_buffer, input_samples])
                raw_output_samples = extended[:n_samples]

                # Apply interpolated gain at output time
                output += raw_output_samples * interpolated_gain

                # Update delay buffer with RAW samples (no gain)
                state["delay_buffer"] = extended[n_samples:n_samples + delay].astype(np.complex128)

        self._sample_count += n_samples
        self._time += n_samples / self.config.sample_rate_hz

        return output.astype(np.complex64)

    def get_state(self) -> dict:
        """Get current channel state.

        Returns:
            Dictionary with tap gains, delays, and statistics
        """
        tap_info = []
        for state in self._tap_states:
            tap_info.append({
                "delay_ms": state["tap"].delay_ms,
                "amplitude": state["tap"].amplitude,
                "doppler_spread_hz": state["tap"].doppler_spread_hz,
                "current_gain": state["current_gain"],
                "current_gain_db": 20 * np.log10(abs(state["current_gain"]) + 1e-10),
            })

        return {
            "time": self._time,
            "taps": tap_info,
            "num_taps": len(self.config.taps),
        }

    def get_impulse_response(self, length: int = 256) -> np.ndarray:
        """Get current channel impulse response.

        Args:
            length: Length of impulse response in samples

        Returns:
            Complex impulse response array
        """
        h = np.zeros(length, dtype=np.complex128)

        for state in self._tap_states:
            delay = state["delay_samples"]
            if delay < length:
                h[delay] = state["current_gain"]

        return h.astype(np.complex64)

    def get_frequency_response(self, n_points: int = 1024) -> tuple:
        """Get current channel frequency response.

        Args:
            n_points: Number of frequency points

        Returns:
            Tuple of (frequency_axis, H(f))
        """
        h = self.get_impulse_response(n_points)
        H = np.fft.fftshift(np.fft.fft(h))
        freq = np.fft.fftshift(
            np.fft.fftfreq(n_points, 1 / self.config.sample_rate_hz)
        )
        return freq, H.astype(np.complex64)

    def reset(self, seed: Optional[int] = None):
        """Reset channel state.

        Args:
            seed: New random seed (optional)
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._seed = seed

        self._init_tap_states()
        self._time = 0.0
        self._sample_count = 0

        # Reset compiled processor if available
        if self._compiled_processor is not None:
            try:
                self._compiled_processor.reset(self._seed)
                self._configure_compiled_taps()
            except Exception:
                pass

    def add_state_callback(self, callback: Callable):
        """Register callback for state updates."""
        self._state_callbacks.append(callback)

    def remove_state_callback(self, callback: Callable):
        """Remove state callback."""
        if callback in self._state_callbacks:
            self._state_callbacks.remove(callback)


def compare_models(
    vogler_params,
    watterson_config: WattersonConfig,
    duration_sec: float = 1.0,
    sample_rate_hz: float = 2_000_000,
) -> dict:
    """Compare Vogler-Hoffmeyer and Watterson model outputs.

    Args:
        vogler_params: VoglerParameters instance
        watterson_config: WattersonConfig instance
        duration_sec: Test signal duration
        sample_rate_hz: Sample rate

    Returns:
        Dictionary of comparison statistics
    """
    from .channel import HFChannel, ProcessingConfig

    n_samples = int(duration_sec * sample_rate_hz)

    # Generate test signal (wideband noise)
    rng = np.random.default_rng(42)
    test_signal = (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    ).astype(np.complex64) / np.sqrt(2)

    # Process through Vogler channel
    vogler_config = ProcessingConfig(sample_rate_hz=sample_rate_hz)
    vogler_channel = HFChannel(vogler_params, vogler_config, use_gpu=False)
    vogler_output = vogler_channel.process(test_signal)

    # Process through Watterson channel
    watterson_channel = WattersonChannel(watterson_config, seed=42)
    watterson_output = watterson_channel.process_block(test_signal)

    # Compute statistics
    def compute_stats(signal):
        power = np.mean(np.abs(signal) ** 2)
        envelope = np.abs(signal)
        return {
            "mean_power_db": 10 * np.log10(power + 1e-10),
            "envelope_std": np.std(envelope),
            "envelope_mean": np.mean(envelope),
            "fade_depth_db": 20 * np.log10(
                np.max(envelope) / (np.min(envelope) + 1e-10)
            ),
        }

    vogler_stats = compute_stats(vogler_output)
    watterson_stats = compute_stats(watterson_output)

    # Cross-correlation for similarity
    correlation = np.abs(np.corrcoef(
        np.abs(vogler_output[:10000]),
        np.abs(watterson_output[:10000])
    )[0, 1])

    return {
        "vogler": vogler_stats,
        "watterson": watterson_stats,
        "envelope_correlation": correlation,
    }
