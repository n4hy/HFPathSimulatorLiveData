"""Watterson HF channel model implementation.

The Watterson model is a tapped delay line (TDL) model where each tap
represents a propagation mode with independent Rayleigh or Rician fading.

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
    """

    def __init__(
        self,
        config: Optional[WattersonConfig] = None,
        seed: Optional[int] = None,
    ):
        """Initialize Watterson channel.

        Args:
            config: Channel configuration
            seed: Random seed for reproducibility
        """
        self.config = config or WattersonConfig.from_itu_condition(
            ITUCondition.MODERATE
        )

        # Random state
        self._rng = np.random.default_rng(seed)

        # Fading state for each tap
        self._tap_states: List[dict] = []
        self._init_tap_states()

        # Processing state
        self._time = 0.0
        self._sample_count = 0

        # Callbacks
        self._state_callbacks: List[Callable] = []

    def _init_tap_states(self):
        """Initialize fading generators for each tap."""
        self._tap_states = []

        for tap in self.config.taps:
            # Delay in samples
            delay_samples = int(
                tap.delay_ms / 1000.0 * self.config.sample_rate_hz
            )

            # Doppler filter design
            # Filter length based on coherence time
            if tap.doppler_spread_hz > 0:
                coherence_samples = int(
                    self.config.sample_rate_hz / tap.doppler_spread_hz / 4
                )
                filter_len = max(64, min(1024, coherence_samples))
            else:
                filter_len = 64

            # Create Doppler shaping filter
            doppler_filter = self._create_doppler_filter(
                tap.doppler_spread_hz,
                tap.doppler_spectrum,
                filter_len,
            )

            state = {
                "tap": tap,
                "delay_samples": delay_samples,
                "doppler_filter": doppler_filter,
                "filter_state": np.zeros(len(doppler_filter) - 1, dtype=np.complex128),
                "current_gain": complex(1.0, 0.0),
                "noise_buffer": self._rng.standard_normal(filter_len)
                + 1j * self._rng.standard_normal(filter_len),
            }

            self._tap_states.append(state)

    def _create_doppler_filter(
        self,
        doppler_spread_hz: float,
        spectrum_type: DopplerSpectrum,
        length: int,
    ) -> np.ndarray:
        """Create FIR filter for Doppler spectrum shaping.

        Args:
            doppler_spread_hz: Two-sided Doppler spread
            spectrum_type: Doppler spectrum shape
            length: Filter length

        Returns:
            FIR filter coefficients
        """
        if doppler_spread_hz <= 0:
            # No fading - return delta function
            h = np.zeros(length)
            h[length // 2] = 1.0
            return h

        # Normalized frequency (relative to sample rate)
        # We're filtering at the block update rate, not sample rate
        f_norm = doppler_spread_hz / self.config.update_rate_hz

        if spectrum_type == DopplerSpectrum.GAUSSIAN:
            # Gaussian spectrum -> Gaussian impulse response
            t = np.arange(length) - length // 2
            sigma = length / (4 * np.pi * f_norm) if f_norm > 0 else length / 4
            h = np.exp(-0.5 * (t / sigma) ** 2)

        elif spectrum_type == DopplerSpectrum.FLAT:
            # Flat spectrum -> sinc impulse response
            t = np.arange(length) - length // 2
            t = np.where(t == 0, 1e-10, t)
            h = np.sinc(2 * f_norm * t)

        elif spectrum_type == DopplerSpectrum.JAKES:
            # Jakes spectrum (U-shaped)
            # For mobile channels, not typical HF
            t = np.arange(length) - length // 2
            t = np.where(t == 0, 1e-10, t)
            h = np.sinc(2 * f_norm * t) * np.cos(np.pi * f_norm * t)

        else:
            h = np.ones(length)

        # Normalize
        h = h / np.sqrt(np.sum(h**2))

        return h.astype(np.complex128)

    def _update_fading_coefficients(self):
        """Update fading coefficients for all taps."""
        for state in self._tap_states:
            tap = state["tap"]

            # Generate new noise sample
            noise = (
                self._rng.standard_normal()
                + 1j * self._rng.standard_normal()
            ) / np.sqrt(2)

            # Filter through Doppler spectrum shaping filter
            state["noise_buffer"] = np.roll(state["noise_buffer"], -1)
            state["noise_buffer"][-1] = noise

            filtered = np.sum(state["noise_buffer"] * state["doppler_filter"])

            # Apply tap amplitude
            scatter_component = filtered * tap.amplitude

            # Add specular component for Rician fading
            if tap.is_specular:
                k_linear = 10 ** (tap.k_factor_db / 10)
                specular = np.sqrt(k_linear / (1 + k_linear)) * tap.amplitude
                scatter_component *= np.sqrt(1 / (1 + k_linear))
                state["current_gain"] = specular + scatter_component
            else:
                state["current_gain"] = scatter_component

    def process(self, input_samples: np.ndarray) -> np.ndarray:
        """Process samples through the Watterson channel.

        Args:
            input_samples: Complex input samples

        Returns:
            Complex output samples after channel
        """
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
        """Process a block of samples efficiently.

        Uses vectorized operations for better performance.

        Args:
            input_samples: Complex input samples

        Returns:
            Complex output samples
        """
        n_samples = len(input_samples)
        output = np.zeros(n_samples, dtype=np.complex128)

        # Update fading at block rate (approximation for efficiency)
        self._update_fading_coefficients()

        # Process each tap
        for state in self._tap_states:
            delay = state["delay_samples"]
            gain = state["current_gain"]

            if delay == 0:
                output += input_samples * gain
            elif delay < n_samples:
                # Delayed contribution
                output[delay:] += input_samples[:-delay] * gain

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

        self._init_tap_states()
        self._time = 0.0
        self._sample_count = 0

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
