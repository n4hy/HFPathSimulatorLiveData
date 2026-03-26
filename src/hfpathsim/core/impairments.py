"""Signal impairments for HF channel simulation.

Implements receiver front-end effects:
- Automatic Gain Control (AGC)
- Signal limiting/clipping
- Nonlinear distortion
- Frequency offset and phase noise

Uses CUDA/C++ compiled implementations when available with automatic
fallback to pure Python.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import numpy as np

# Try to import compiled GPU implementations
try:
    from ..gpu import AGCProcessor as _AGCProcessor, LimiterProcessor as _LimiterProcessor
    _HAS_COMPILED = _AGCProcessor is not None and _LimiterProcessor is not None
except ImportError:
    _HAS_COMPILED = False
    _AGCProcessor = None
    _LimiterProcessor = None


class AGCMode(Enum):
    """AGC operating modes."""

    SLOW = "slow"  # Slow AGC for steady signals
    MEDIUM = "medium"  # Medium AGC for typical voice/data
    FAST = "fast"  # Fast AGC for burst signals
    MANUAL = "manual"  # Fixed gain (no AGC)


@dataclass
class AGCConfig:
    """Configuration for AGC model."""

    mode: AGCMode = AGCMode.MEDIUM
    target_level_db: float = -10.0  # Target output level in dBFS
    max_gain_db: float = 60.0  # Maximum gain
    min_gain_db: float = -20.0  # Minimum gain (attenuation)

    # Time constants (mode-dependent defaults applied in __post_init__)
    attack_time_ms: Optional[float] = None  # Attack time constant
    release_time_ms: Optional[float] = None  # Release (decay) time constant
    hold_time_ms: float = 50.0  # Hold time before release

    # Behavior
    hang_agc: bool = True  # Use hang AGC (hold at peak before release)
    soft_knee_db: float = 6.0  # Soft knee transition range

    def __post_init__(self):
        """Apply mode-dependent defaults."""
        mode_defaults = {
            AGCMode.SLOW: (500.0, 2000.0),
            AGCMode.MEDIUM: (50.0, 500.0),
            AGCMode.FAST: (5.0, 50.0),
            AGCMode.MANUAL: (1.0, 1.0),
        }

        attack, release = mode_defaults[self.mode]

        if self.attack_time_ms is None:
            self.attack_time_ms = attack
        if self.release_time_ms is None:
            self.release_time_ms = release


class AGC:
    """Automatic Gain Control model.

    Implements a realistic AGC with:
    - Envelope detection
    - Attack/release time constants
    - Hang AGC for reduced pumping
    - Configurable gain range

    Uses CUDA/C++ implementation when available for high performance.
    """

    def __init__(
        self,
        config: Optional[AGCConfig] = None,
        sample_rate_hz: float = 2_000_000,
        use_compiled: bool = True,
        max_samples: int = 65536,
    ):
        """Initialize AGC.

        Args:
            config: AGC configuration
            sample_rate_hz: Sample rate
            use_compiled: Use compiled CUDA/C++ implementation if available
            max_samples: Maximum samples per block (for compiled backend)
        """
        self.config = config or AGCConfig()
        self.sample_rate = sample_rate_hz

        # State
        self._gain_db = 0.0  # Current gain in dB
        self._envelope = 0.0  # Smoothed envelope
        self._hold_counter = 0  # Hold timer
        self._peak_envelope = 0.0  # Peak for hang AGC

        # Try to use compiled implementation
        # C++ signature: (sample_rate, attack_time_ms, release_time_ms, hold_time_ms,
        #                 hang_agc, target_level_db, max_gain_db, min_gain_db,
        #                 soft_knee_db, max_samples)
        self._compiled_processor = None
        if use_compiled and _HAS_COMPILED:
            try:
                self._compiled_processor = _AGCProcessor(
                    sample_rate_hz,
                    self.config.attack_time_ms,
                    self.config.release_time_ms,
                    self.config.hold_time_ms,
                    self.config.hang_agc,
                    self.config.target_level_db,
                    self.config.max_gain_db,
                    self.config.min_gain_db,
                    self.config.soft_knee_db,
                    max_samples,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._compiled_processor = None

        # Pre-compute coefficients for Python fallback
        self._update_coefficients()

    def _update_coefficients(self):
        """Update filter coefficients from config."""
        # Convert time constants to filter coefficients
        # Using single-pole IIR: y[n] = alpha * x[n] + (1-alpha) * y[n-1]
        # Time constant tau: alpha = 1 - exp(-1/(fs*tau))

        tau_attack = self.config.attack_time_ms / 1000
        tau_release = self.config.release_time_ms / 1000

        self._alpha_attack = 1 - np.exp(-1 / (self.sample_rate * tau_attack))
        self._alpha_release = 1 - np.exp(-1 / (self.sample_rate * tau_release))

        self._hold_samples = int(
            self.config.hold_time_ms / 1000 * self.sample_rate
        )

    def process(self, signal: np.ndarray) -> np.ndarray:
        """Apply AGC to signal.

        Args:
            signal: Complex input signal

        Returns:
            Gain-adjusted signal
        """
        if self.config.mode == AGCMode.MANUAL:
            return signal * (10 ** (self._gain_db / 20))

        # Use compiled implementation if available
        if self._compiled_processor is not None:
            try:
                input_real = np.ascontiguousarray(signal.real, dtype=np.float32)
                input_imag = np.ascontiguousarray(signal.imag, dtype=np.float32)
                output_real, output_imag = self._compiled_processor.process(
                    input_real, input_imag
                )
                return (output_real + 1j * output_imag).astype(np.complex64)
            except Exception:
                pass  # Fall through to Python implementation

        # Python fallback implementation
        n_samples = len(signal)
        output = np.zeros(n_samples, dtype=np.complex64)
        envelope = np.abs(signal)

        for i in range(n_samples):
            env = envelope[i]

            # Update envelope estimate
            if env > self._envelope:
                # Attack
                self._envelope += self._alpha_attack * (env - self._envelope)
                self._peak_envelope = self._envelope
                self._hold_counter = self._hold_samples
            else:
                # Hold or release
                if self.config.hang_agc and self._hold_counter > 0:
                    self._hold_counter -= 1
                else:
                    # Release
                    self._envelope += self._alpha_release * (env - self._envelope)

            # Compute desired gain
            if self._envelope > 1e-10:
                target_linear = 10 ** (self.config.target_level_db / 20)
                desired_gain_db = 20 * np.log10(target_linear / self._envelope)
            else:
                desired_gain_db = self.config.max_gain_db

            # Apply soft knee
            knee = self.config.soft_knee_db
            if abs(desired_gain_db - self._gain_db) < knee:
                # In the knee region - smooth transition
                self._gain_db += 0.1 * (desired_gain_db - self._gain_db)
            else:
                self._gain_db = desired_gain_db

            # Clamp gain to range
            self._gain_db = np.clip(
                self._gain_db,
                self.config.min_gain_db,
                self.config.max_gain_db,
            )

            # Apply gain
            gain_linear = 10 ** (self._gain_db / 20)
            output[i] = signal[i] * gain_linear

        return output

    def process_block(self, signal: np.ndarray) -> np.ndarray:
        """Process a block with constant gain (faster).

        Uses block-average envelope for efficiency.

        Args:
            signal: Complex input signal

        Returns:
            Gain-adjusted signal
        """
        if self.config.mode == AGCMode.MANUAL:
            return signal * (10 ** (self._gain_db / 20))

        # Use compiled implementation if available (per-sample processing)
        if self._compiled_processor is not None:
            try:
                input_real = np.ascontiguousarray(signal.real, dtype=np.float32)
                input_imag = np.ascontiguousarray(signal.imag, dtype=np.float32)
                output_real, output_imag = self._compiled_processor.process(
                    input_real, input_imag
                )
                return (output_real + 1j * output_imag).astype(np.complex64)
            except Exception:
                pass  # Fall through to Python implementation

        # Python fallback - block-average implementation
        # Block envelope
        envelope = np.mean(np.abs(signal))

        # Update envelope with attack/release
        if envelope > self._envelope:
            alpha = min(1.0, self._alpha_attack * len(signal))
            self._envelope += alpha * (envelope - self._envelope)
        else:
            alpha = min(1.0, self._alpha_release * len(signal))
            self._envelope += alpha * (envelope - self._envelope)

        # Compute gain
        if self._envelope > 1e-10:
            target_linear = 10 ** (self.config.target_level_db / 20)
            desired_gain_db = 20 * np.log10(target_linear / self._envelope)
        else:
            desired_gain_db = self.config.max_gain_db

        # Smooth gain change
        self._gain_db += 0.3 * (desired_gain_db - self._gain_db)

        # Clamp
        self._gain_db = np.clip(
            self._gain_db,
            self.config.min_gain_db,
            self.config.max_gain_db,
        )

        # Apply
        gain_linear = 10 ** (self._gain_db / 20)
        return (signal * gain_linear).astype(np.complex64)

    def reset(self):
        """Reset AGC state."""
        self._gain_db = 0.0
        self._envelope = 0.0
        self._hold_counter = 0
        self._peak_envelope = 0.0
        if self._compiled_processor is not None:
            try:
                self._compiled_processor.reset()
            except Exception:
                pass

    @property
    def current_gain_db(self) -> float:
        """Get current gain in dB."""
        return self._gain_db

    def set_gain(self, gain_db: float):
        """Set manual gain (for MANUAL mode).

        Args:
            gain_db: Gain in dB
        """
        self._gain_db = np.clip(
            gain_db,
            self.config.min_gain_db,
            self.config.max_gain_db,
        )


@dataclass
class LimiterConfig:
    """Configuration for signal limiter."""

    threshold_db: float = -3.0  # Limiting threshold in dBFS
    mode: str = "soft"  # "hard", "soft", or "cubic"
    knee_db: float = 6.0  # Soft knee range
    attack_time_ms: float = 0.1  # Attack time
    release_time_ms: float = 10.0  # Release time


class Limiter:
    """Signal limiter/clipper.

    Implements various limiting modes:
    - Hard clipping
    - Soft limiting (tanh-like)
    - Cubic soft clipping

    Uses CUDA/C++ implementation when available for high performance.
    """

    def __init__(
        self,
        config: Optional[LimiterConfig] = None,
        sample_rate_hz: float = 2_000_000,
        use_compiled: bool = True,
        max_samples: int = 65536,
    ):
        """Initialize limiter.

        Args:
            config: Limiter configuration
            sample_rate_hz: Sample rate
            use_compiled: Use compiled CUDA/C++ implementation if available
            max_samples: Maximum samples per block (for compiled backend)
        """
        self.config = config or LimiterConfig()
        self.sample_rate = sample_rate_hz

        self._threshold = 10 ** (self.config.threshold_db / 20)
        self._gain_reduction = 0.0

        # Map mode string to integer for compiled backend
        mode_map = {"hard": 0, "soft": 1, "cubic": 2}
        mode_int = mode_map.get(self.config.mode, 1)

        # Try to use compiled implementation
        # C++ signature: (threshold_db, mode, max_samples)
        self._compiled_processor = None
        if use_compiled and _HAS_COMPILED:
            try:
                self._compiled_processor = _LimiterProcessor(
                    self.config.threshold_db,
                    mode_int,
                    max_samples,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._compiled_processor = None

    def process(self, signal: np.ndarray) -> np.ndarray:
        """Apply limiting to signal.

        Args:
            signal: Complex input signal

        Returns:
            Limited signal
        """
        # Use compiled implementation if available
        if self._compiled_processor is not None:
            try:
                input_real = np.ascontiguousarray(signal.real, dtype=np.float32)
                input_imag = np.ascontiguousarray(signal.imag, dtype=np.float32)
                output_real, output_imag = self._compiled_processor.process(
                    input_real, input_imag
                )
                output = (output_real + 1j * output_imag).astype(np.complex64)
                # Compute gain reduction for monitoring
                envelope = np.abs(signal)
                limited_env = np.abs(output)
                with np.errstate(divide='ignore', invalid='ignore'):
                    gr = np.where(
                        envelope > 0,
                        20 * np.log10(limited_env / envelope),
                        0,
                    )
                    self._gain_reduction = np.min(gr)
                return output
            except Exception:
                pass  # Fall through to Python implementation

        # Python fallback implementation
        envelope = np.abs(signal)
        phase = np.angle(signal)

        if self.config.mode == "hard":
            # Hard clipping
            limited_env = np.minimum(envelope, self._threshold)

        elif self.config.mode == "soft":
            # Soft limiting using tanh
            normalized = envelope / self._threshold
            limited_normalized = np.tanh(normalized)
            limited_env = limited_normalized * self._threshold

        elif self.config.mode == "cubic":
            # Cubic soft clipping
            normalized = envelope / self._threshold
            # Below threshold: linear, above: cubic compression
            limited_normalized = np.where(
                normalized < 1,
                normalized,
                1 + (normalized - 1) / (1 + (normalized - 1) ** 2),
            )
            limited_env = limited_normalized * self._threshold

        else:
            limited_env = envelope

        # Reconstruct complex signal
        output = limited_env * np.exp(1j * phase)

        # Track gain reduction
        with np.errstate(divide='ignore', invalid='ignore'):
            gr = np.where(
                envelope > 0,
                20 * np.log10(limited_env / envelope),
                0,
            )
            self._gain_reduction = np.min(gr)

        return output.astype(np.complex64)

    @property
    def gain_reduction_db(self) -> float:
        """Get current gain reduction in dB."""
        return self._gain_reduction


@dataclass
class FrequencyOffsetConfig:
    """Configuration for frequency offset simulation."""

    offset_hz: float = 0.0  # Static frequency offset
    drift_rate_hz_per_sec: float = 0.0  # Linear drift rate
    phase_noise_level_dbc: float = -80.0  # Phase noise level at 1kHz offset


class FrequencyOffset:
    """Frequency offset and phase noise model.

    Simulates:
    - Static carrier frequency offset
    - Linear frequency drift
    - Phase noise (optional)
    """

    def __init__(
        self,
        config: Optional[FrequencyOffsetConfig] = None,
        sample_rate_hz: float = 2_000_000,
        seed: Optional[int] = None,
    ):
        """Initialize frequency offset model.

        Args:
            config: Configuration
            sample_rate_hz: Sample rate
            seed: Random seed for phase noise
        """
        self.config = config or FrequencyOffsetConfig()
        self.sample_rate = sample_rate_hz
        self._rng = np.random.default_rng(seed)

        self._phase = 0.0  # Accumulated phase
        self._time = 0.0  # Elapsed time

    def process(self, signal: np.ndarray) -> np.ndarray:
        """Apply frequency offset to signal.

        Args:
            signal: Complex input signal

        Returns:
            Frequency-shifted signal
        """
        n_samples = len(signal)
        t = np.arange(n_samples) / self.sample_rate + self._time

        # Static offset + drift
        freq = self.config.offset_hz + self.config.drift_rate_hz_per_sec * t

        # Phase accumulation
        phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate + self._phase

        # Add phase noise if significant
        if self.config.phase_noise_level_dbc > -100:
            # Simple phase noise model
            noise_std = 10 ** (self.config.phase_noise_level_dbc / 20)
            phase_noise = np.cumsum(self._rng.standard_normal(n_samples)) * noise_std
            phase += phase_noise

        # Apply rotation
        output = signal * np.exp(1j * phase)

        # Update state
        self._phase = phase[-1] % (2 * np.pi)
        self._time += n_samples / self.sample_rate

        return output.astype(np.complex64)

    def reset(self):
        """Reset phase accumulator."""
        self._phase = 0.0
        self._time = 0.0

    def set_offset(self, offset_hz: float):
        """Set frequency offset.

        Args:
            offset_hz: Frequency offset in Hz
        """
        self.config.offset_hz = offset_hz


class ImpairmentChain:
    """Chain of signal impairments.

    Combines multiple impairment models in a processing chain.
    """

    def __init__(
        self,
        agc: Optional[AGC] = None,
        limiter: Optional[Limiter] = None,
        freq_offset: Optional[FrequencyOffset] = None,
        noise_generator=None,  # NoiseGenerator from noise.py
    ):
        """Initialize impairment chain.

        Args:
            agc: AGC model (or None to disable)
            limiter: Limiter model (or None to disable)
            freq_offset: Frequency offset model (or None to disable)
            noise_generator: Noise generator (or None to disable)
        """
        self.agc = agc
        self.limiter = limiter
        self.freq_offset = freq_offset
        self.noise_generator = noise_generator

    def process(self, signal: np.ndarray) -> np.ndarray:
        """Process signal through impairment chain.

        Order: noise -> freq_offset -> agc -> limiter

        Args:
            signal: Complex input signal

        Returns:
            Impaired signal
        """
        output = signal.copy()

        # Add noise first (before receiver)
        if self.noise_generator is not None:
            output = self.noise_generator.add_noise(output)

        # Frequency offset (oscillator error)
        if self.freq_offset is not None:
            output = self.freq_offset.process(output)

        # AGC
        if self.agc is not None:
            output = self.agc.process_block(output)

        # Limiter (after AGC)
        if self.limiter is not None:
            output = self.limiter.process(output)

        return output

    def reset(self):
        """Reset all impairment states."""
        if self.agc is not None:
            self.agc.reset()
        if self.freq_offset is not None:
            self.freq_offset.reset()

    def get_status(self) -> dict:
        """Get current status of all impairments.

        Returns:
            Dictionary with impairment states
        """
        status = {}

        if self.agc is not None:
            status["agc_gain_db"] = self.agc.current_gain_db

        if self.limiter is not None:
            status["limiter_gr_db"] = self.limiter.gain_reduction_db

        if self.freq_offset is not None:
            status["freq_offset_hz"] = self.freq_offset.config.offset_hz

        return status
