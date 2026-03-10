"""Noise models for HF channel simulation.

Implements various noise sources:
- AWGN (Additive White Gaussian Noise)
- Atmospheric noise per ITU-R P.372
- Man-made noise
- Impulse noise

Reference: ITU-R P.372-16, "Radio noise"
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class NoiseType(Enum):
    """Types of noise sources."""

    AWGN = "awgn"  # Additive White Gaussian Noise
    ATMOSPHERIC = "atmospheric"  # Atmospheric/galactic noise
    MANMADE = "manmade"  # Man-made noise
    IMPULSE = "impulse"  # Impulsive interference


class ManMadeEnvironment(Enum):
    """Man-made noise environment categories per ITU-R P.372."""

    CITY = "city"  # City/industrial
    RESIDENTIAL = "residential"  # Residential
    RURAL = "rural"  # Rural
    QUIET_RURAL = "quiet_rural"  # Quiet rural


@dataclass
class NoiseConfig:
    """Configuration for noise generation."""

    # AWGN parameters
    snr_db: float = 20.0  # Signal-to-noise ratio in dB
    noise_bandwidth_hz: float = 3000.0  # Noise bandwidth

    # Atmospheric noise (ITU-R P.372)
    enable_atmospheric: bool = False
    frequency_mhz: float = 10.0  # Operating frequency for noise model
    season: str = "summer"  # "summer" or "winter"
    time_of_day: str = "day"  # "day" or "night"
    latitude: float = 45.0  # Geographic latitude

    # Man-made noise
    enable_manmade: bool = False
    environment: ManMadeEnvironment = ManMadeEnvironment.RESIDENTIAL

    # Impulse noise
    enable_impulse: bool = False
    impulse_rate_hz: float = 10.0  # Average impulses per second
    impulse_amplitude_db: float = 20.0  # Impulse amplitude above noise floor
    impulse_duration_us: float = 100.0  # Impulse duration in microseconds


class NoiseGenerator:
    """Generator for various noise types.

    Combines multiple noise sources into a realistic HF noise environment.
    """

    def __init__(
        self,
        config: Optional[NoiseConfig] = None,
        sample_rate_hz: float = 2_000_000,
        seed: Optional[int] = None,
    ):
        """Initialize noise generator.

        Args:
            config: Noise configuration
            sample_rate_hz: Sample rate in Hz
            seed: Random seed for reproducibility
        """
        self.config = config or NoiseConfig()
        self.sample_rate = sample_rate_hz
        self._rng = np.random.default_rng(seed)

        # Pre-compute noise power levels
        self._update_noise_levels()

    def _update_noise_levels(self):
        """Update noise power levels from configuration."""
        # AWGN noise variance (assuming unit signal power)
        snr_linear = 10 ** (self.config.snr_db / 10)
        self._awgn_variance = 1.0 / snr_linear

        # Atmospheric noise figure (ITU-R P.372)
        if self.config.enable_atmospheric:
            self._fa_atmospheric = self._compute_atmospheric_noise_figure()
        else:
            self._fa_atmospheric = 0.0

        # Man-made noise figure
        if self.config.enable_manmade:
            self._fa_manmade = self._compute_manmade_noise_figure()
        else:
            self._fa_manmade = 0.0

    def _compute_atmospheric_noise_figure(self) -> float:
        """Compute atmospheric noise figure per ITU-R P.372.

        Returns:
            Noise figure Fa in dB above kTB
        """
        f_mhz = self.config.frequency_mhz

        # Simplified model based on ITU-R P.372 Figure 2
        # Fa = a - b*log10(f) where f is in MHz
        # Coefficients vary with season, time, latitude

        # Base coefficients for mid-latitude summer daytime
        if self.config.season == "summer":
            if self.config.time_of_day == "day":
                a, b = 53.6, 28.6
            else:  # night
                a, b = 76.8, 27.7
        else:  # winter
            if self.config.time_of_day == "day":
                a, b = 45.2, 28.0
            else:  # night
                a, b = 60.3, 27.0

        # Latitude adjustment (simplified)
        lat_factor = 1.0 + 0.1 * (abs(self.config.latitude) - 45) / 45

        fa = (a - b * np.log10(f_mhz)) * lat_factor

        # Clamp to reasonable range
        return max(0, min(fa, 120))

    def _compute_manmade_noise_figure(self) -> float:
        """Compute man-made noise figure per ITU-R P.372.

        Returns:
            Noise figure Fa in dB above kTB
        """
        f_mhz = self.config.frequency_mhz

        # ITU-R P.372 Table 2 coefficients
        # Fa = c - d*log10(f)
        coefficients = {
            ManMadeEnvironment.CITY: (76.8, 27.7),
            ManMadeEnvironment.RESIDENTIAL: (72.5, 27.7),
            ManMadeEnvironment.RURAL: (67.2, 27.7),
            ManMadeEnvironment.QUIET_RURAL: (53.6, 28.6),
        }

        c, d = coefficients[self.config.environment]
        fa = c - d * np.log10(f_mhz)

        return max(0, min(fa, 120))

    def _noise_figure_to_power(self, fa_db: float) -> float:
        """Convert noise figure to noise power.

        Args:
            fa_db: Noise figure in dB above kTB

        Returns:
            Relative noise power (linear)
        """
        # Reference: kTB at 290K, B = 1 Hz
        # Fa is dB above this reference
        # Scale by actual bandwidth
        bw_factor = self.config.noise_bandwidth_hz

        # Convert to linear and normalize
        return 10 ** (fa_db / 10) * bw_factor / 1e12  # Arbitrary scaling

    def generate_awgn(self, n_samples: int) -> np.ndarray:
        """Generate AWGN samples.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Complex noise samples
        """
        std = np.sqrt(self._awgn_variance / 2)
        noise = (
            self._rng.standard_normal(n_samples)
            + 1j * self._rng.standard_normal(n_samples)
        ) * std

        return noise.astype(np.complex64)

    def generate_atmospheric(self, n_samples: int) -> np.ndarray:
        """Generate atmospheric noise.

        Atmospheric noise has impulsive characteristics and
        non-Gaussian amplitude distribution.

        Args:
            n_samples: Number of samples

        Returns:
            Complex noise samples
        """
        if not self.config.enable_atmospheric or self._fa_atmospheric <= 0:
            return np.zeros(n_samples, dtype=np.complex64)

        power = self._noise_figure_to_power(self._fa_atmospheric)
        std = np.sqrt(power / 2)

        # Atmospheric noise is non-Gaussian - use mixture model
        # Mix of Gaussian background and impulsive component
        gaussian = (
            self._rng.standard_normal(n_samples)
            + 1j * self._rng.standard_normal(n_samples)
        ) * std * 0.7

        # Impulsive component (sparse, high amplitude)
        impulse_prob = 0.01  # 1% of samples have impulses
        impulse_mask = self._rng.random(n_samples) < impulse_prob
        impulses = np.zeros(n_samples, dtype=np.complex64)
        n_impulses = np.sum(impulse_mask)
        if n_impulses > 0:
            impulses[impulse_mask] = (
                self._rng.standard_normal(n_impulses)
                + 1j * self._rng.standard_normal(n_impulses)
            ) * std * 5

        return (gaussian + impulses).astype(np.complex64)

    def generate_manmade(self, n_samples: int) -> np.ndarray:
        """Generate man-made noise.

        Man-made noise has line spectrum components and
        time-varying characteristics.

        Args:
            n_samples: Number of samples

        Returns:
            Complex noise samples
        """
        if not self.config.enable_manmade or self._fa_manmade <= 0:
            return np.zeros(n_samples, dtype=np.complex64)

        power = self._noise_figure_to_power(self._fa_manmade)
        std = np.sqrt(power / 2)

        # Wideband component
        wideband = (
            self._rng.standard_normal(n_samples)
            + 1j * self._rng.standard_normal(n_samples)
        ) * std * 0.5

        # Add some line components (power line harmonics, etc.)
        t = np.arange(n_samples) / self.sample_rate
        # 50/60 Hz harmonics (audible in HF)
        line_freq = 60.0  # Hz (use 50 for Europe)
        harmonics = np.zeros(n_samples, dtype=np.complex64)
        for h in range(1, 5):
            phase = self._rng.random() * 2 * np.pi
            harmonics += (
                std * 0.1 / h *
                np.exp(1j * (2 * np.pi * line_freq * h * t + phase))
            )

        return (wideband + harmonics).astype(np.complex64)

    def generate_impulse(self, n_samples: int) -> np.ndarray:
        """Generate impulse noise.

        Models interference from lightning, switching transients,
        and other impulsive sources.

        Args:
            n_samples: Number of samples

        Returns:
            Complex noise samples
        """
        if not self.config.enable_impulse:
            return np.zeros(n_samples, dtype=np.complex64)

        noise = np.zeros(n_samples, dtype=np.complex64)

        # Average time between impulses
        mean_interval = self.sample_rate / self.config.impulse_rate_hz

        # Impulse duration in samples
        duration = int(
            self.config.impulse_duration_us * 1e-6 * self.sample_rate
        )
        duration = max(1, duration)

        # Generate Poisson-distributed impulse times
        position = 0
        while position < n_samples:
            # Next impulse position (exponential inter-arrival)
            interval = int(self._rng.exponential(mean_interval))
            position += interval

            if position >= n_samples:
                break

            # Impulse amplitude
            amp_db = self.config.impulse_amplitude_db
            amp_linear = 10 ** (amp_db / 20)

            # Random phase and small frequency offset
            phase = self._rng.random() * 2 * np.pi
            freq_offset = self._rng.standard_normal() * 100  # Hz

            # Generate impulse waveform (damped sinusoid)
            end_pos = min(position + duration, n_samples)
            t = np.arange(end_pos - position) / self.sample_rate

            impulse = (
                amp_linear *
                np.exp(-t / (self.config.impulse_duration_us * 1e-6 / 3)) *
                np.exp(1j * (2 * np.pi * freq_offset * t + phase))
            )

            noise[position:end_pos] += impulse

        return noise.astype(np.complex64)

    def generate(self, n_samples: int) -> np.ndarray:
        """Generate combined noise.

        Combines all enabled noise sources.

        Args:
            n_samples: Number of samples

        Returns:
            Complex noise samples
        """
        noise = self.generate_awgn(n_samples)

        if self.config.enable_atmospheric:
            noise += self.generate_atmospheric(n_samples)

        if self.config.enable_manmade:
            noise += self.generate_manmade(n_samples)

        if self.config.enable_impulse:
            noise += self.generate_impulse(n_samples)

        return noise

    def add_noise(
        self,
        signal: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """Add noise to a signal.

        Args:
            signal: Complex input signal
            normalize: If True, scale noise to achieve target SNR

        Returns:
            Signal with noise added
        """
        n_samples = len(signal)
        noise = self.generate(n_samples)

        if normalize:
            # Scale noise to achieve target SNR
            signal_power = np.mean(np.abs(signal) ** 2)
            noise_power = np.mean(np.abs(noise) ** 2)

            target_noise_power = signal_power / (10 ** (self.config.snr_db / 10))

            if noise_power > 0:
                scale = np.sqrt(target_noise_power / noise_power)
                noise = noise * scale

        return signal + noise

    def set_snr(self, snr_db: float):
        """Set SNR and update noise levels.

        Args:
            snr_db: Signal-to-noise ratio in dB
        """
        self.config.snr_db = snr_db
        self._update_noise_levels()

    def set_frequency(self, frequency_mhz: float):
        """Set operating frequency and update atmospheric/man-made noise.

        Args:
            frequency_mhz: Operating frequency in MHz
        """
        self.config.frequency_mhz = frequency_mhz
        self._update_noise_levels()


@dataclass
class NoiseFloorEstimate:
    """Estimated noise floor parameters."""

    noise_power_dbm: float
    snr_estimate_db: float
    noise_figure_db: float


def estimate_noise_floor(
    signal: np.ndarray,
    sample_rate_hz: float,
    percentile: float = 10.0,
) -> NoiseFloorEstimate:
    """Estimate noise floor from signal.

    Uses percentile method to estimate noise in presence of signal.

    Args:
        signal: Complex signal samples
        sample_rate_hz: Sample rate
        percentile: Percentile for noise estimate (lower = less signal leakage)

    Returns:
        NoiseFloorEstimate with noise parameters
    """
    # Compute power in small windows
    window_size = int(sample_rate_hz * 0.001)  # 1ms windows
    n_windows = len(signal) // window_size

    powers = np.zeros(n_windows)
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        powers[i] = np.mean(np.abs(signal[start:end]) ** 2)

    # Noise floor is low percentile of powers
    noise_power = np.percentile(powers, percentile)
    signal_power = np.mean(powers)

    # Convert to dBm (assuming 50 ohm, but relative values)
    noise_power_dbm = 10 * np.log10(noise_power + 1e-20) + 30

    snr_estimate = 10 * np.log10((signal_power / noise_power) - 1 + 1e-10)

    # Approximate noise figure (relative to thermal at 290K)
    # NF = 10*log10(Pn/(k*T*B))
    k = 1.380649e-23  # Boltzmann
    T = 290  # Standard temperature
    B = sample_rate_hz
    thermal_power = k * T * B
    noise_figure = 10 * np.log10(noise_power / thermal_power + 1e-20)

    return NoiseFloorEstimate(
        noise_power_dbm=noise_power_dbm,
        snr_estimate_db=snr_estimate,
        noise_figure_db=noise_figure,
    )
