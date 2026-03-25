"""ITU-R standardized HF channel models.

Implements channel models from ITU-R recommendations:
- ITU-R F.520-2 (formerly CCIR 520): Standard HF channel simulation models
- ITU-R F.1289: Wideband HF channel model
- ITU-R F.1487: HF channel simulation parameters

Reference: ITU-R F.520-2 "Use of high frequency ionospheric channel simulators"
Reference: ITU-R F.1289 "HF radio channel simulator performance requirements"
Reference: ITU-R F.1487 "Testing of HF modems with bandwidths up to about 12 kHz"
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Callable, Dict, Tuple
import numpy as np
from scipy import signal

from .watterson import WattersonTap, WattersonConfig, WattersonChannel, DopplerSpectrum


class CCIR520Condition(Enum):
    """CCIR 520 / ITU-R F.520 channel condition presets.

    These are the classic standardized HF channel conditions
    used for modem testing and development.
    """

    # Good conditions (stable ionosphere)
    GOOD_LOW_LATENCY = "good_low_latency"
    GOOD_HIGH_LATENCY = "good_high_latency"

    # Moderate conditions (typical operation)
    MODERATE = "moderate"
    MODERATE_MULTIPATH = "moderate_multipath"

    # Poor conditions (disturbed ionosphere)
    POOR = "poor"
    POOR_MULTIPATH = "poor_multipath"

    # Flutter fading (high-latitude auroral)
    FLUTTER = "flutter"

    # Doppler and multipath test cases
    DOPPLER_TEST = "doppler_test"
    MULTIPATH_TEST = "multipath_test"


class ITURF1289Condition(Enum):
    """ITU-R F.1289 wideband HF channel conditions.

    Wideband channel model conditions for testing HF modems
    with bandwidths up to 24 kHz.
    """

    # Low-latitude (equatorial)
    LOW_LATITUDE_QUIET = "low_latitude_quiet"
    LOW_LATITUDE_MODERATE = "low_latitude_moderate"
    LOW_LATITUDE_DISTURBED = "low_latitude_disturbed"

    # Mid-latitude
    MID_LATITUDE_QUIET = "mid_latitude_quiet"
    MID_LATITUDE_MODERATE = "mid_latitude_moderate"
    MID_LATITUDE_DISTURBED = "mid_latitude_disturbed"

    # High-latitude (polar/auroral)
    HIGH_LATITUDE_QUIET = "high_latitude_quiet"
    HIGH_LATITUDE_MODERATE = "high_latitude_moderate"
    HIGH_LATITUDE_DISTURBED = "high_latitude_disturbed"
    HIGH_LATITUDE_FLUTTER = "high_latitude_flutter"


class ITURF1487Condition(Enum):
    """ITU-R F.1487 channel conditions for modem testing.

    Used for testing HF modems with bandwidths up to 12 kHz.
    """

    QUIET = "quiet"
    MODERATE = "moderate"
    DISTURBED = "disturbed"
    FLUTTER = "flutter"


@dataclass
class CCIR520ChannelSpec:
    """CCIR 520 / ITU-R F.520 channel specification.

    Defines the complete channel characteristics per the standard.

    Attributes:
        condition: Channel condition preset
        delay_spread_ms: Multipath delay spread (τ)
        doppler_spread_hz: Doppler frequency spread (2σ)
        num_paths: Number of propagation paths
        path_delays_ms: Delay for each path
        path_amplitudes: Relative amplitude for each path
        path_doppler_spreads_hz: Doppler spread per path
        doppler_spectrum: Shape of Doppler spectrum
        k_factor_db: Rician K-factor for specular paths
    """

    condition: CCIR520Condition
    delay_spread_ms: float
    doppler_spread_hz: float
    num_paths: int
    path_delays_ms: List[float]
    path_amplitudes: List[float]
    path_doppler_spreads_hz: List[float]
    doppler_spectrum: DopplerSpectrum = DopplerSpectrum.GAUSSIAN
    k_factor_db: Optional[float] = None
    description: str = ""


# CCIR 520 / ITU-R F.520-2 Standard Channel Specifications
CCIR520_PRESETS: Dict[CCIR520Condition, CCIR520ChannelSpec] = {
    CCIR520Condition.GOOD_LOW_LATENCY: CCIR520ChannelSpec(
        condition=CCIR520Condition.GOOD_LOW_LATENCY,
        delay_spread_ms=0.5,
        doppler_spread_hz=0.1,
        num_paths=2,
        path_delays_ms=[0.0, 0.5],
        path_amplitudes=[1.0, 1.0],
        path_doppler_spreads_hz=[0.1, 0.1],
        description="Good conditions, low multipath delay (CCIR Good)",
    ),
    CCIR520Condition.GOOD_HIGH_LATENCY: CCIR520ChannelSpec(
        condition=CCIR520Condition.GOOD_HIGH_LATENCY,
        delay_spread_ms=1.0,
        doppler_spread_hz=0.1,
        num_paths=2,
        path_delays_ms=[0.0, 1.0],
        path_amplitudes=[1.0, 1.0],
        path_doppler_spreads_hz=[0.1, 0.1],
        description="Good conditions, higher multipath delay",
    ),
    CCIR520Condition.MODERATE: CCIR520ChannelSpec(
        condition=CCIR520Condition.MODERATE,
        delay_spread_ms=1.0,
        doppler_spread_hz=0.5,
        num_paths=2,
        path_delays_ms=[0.0, 1.0],
        path_amplitudes=[1.0, 1.0],
        path_doppler_spreads_hz=[0.5, 0.5],
        description="Moderate conditions (CCIR Moderate)",
    ),
    CCIR520Condition.MODERATE_MULTIPATH: CCIR520ChannelSpec(
        condition=CCIR520Condition.MODERATE_MULTIPATH,
        delay_spread_ms=2.0,
        doppler_spread_hz=0.5,
        num_paths=3,
        path_delays_ms=[0.0, 1.0, 2.0],
        path_amplitudes=[1.0, 0.8, 0.6],
        path_doppler_spreads_hz=[0.5, 0.5, 0.5],
        description="Moderate conditions with 3-path multipath",
    ),
    CCIR520Condition.POOR: CCIR520ChannelSpec(
        condition=CCIR520Condition.POOR,
        delay_spread_ms=2.0,
        doppler_spread_hz=1.0,
        num_paths=2,
        path_delays_ms=[0.0, 2.0],
        path_amplitudes=[1.0, 1.0],
        path_doppler_spreads_hz=[1.0, 1.0],
        description="Poor conditions (CCIR Poor)",
    ),
    CCIR520Condition.POOR_MULTIPATH: CCIR520ChannelSpec(
        condition=CCIR520Condition.POOR_MULTIPATH,
        delay_spread_ms=4.0,
        doppler_spread_hz=1.0,
        num_paths=4,
        path_delays_ms=[0.0, 1.5, 3.0, 4.0],
        path_amplitudes=[1.0, 0.8, 0.6, 0.4],
        path_doppler_spreads_hz=[1.0, 1.0, 1.0, 1.0],
        description="Poor conditions with severe multipath",
    ),
    CCIR520Condition.FLUTTER: CCIR520ChannelSpec(
        condition=CCIR520Condition.FLUTTER,
        delay_spread_ms=1.0,
        doppler_spread_hz=10.0,
        num_paths=2,
        path_delays_ms=[0.0, 1.0],
        path_amplitudes=[1.0, 0.8],
        path_doppler_spreads_hz=[10.0, 10.0],
        description="Flutter fading (high-latitude auroral)",
    ),
    CCIR520Condition.DOPPLER_TEST: CCIR520ChannelSpec(
        condition=CCIR520Condition.DOPPLER_TEST,
        delay_spread_ms=0.0,
        doppler_spread_hz=5.0,
        num_paths=1,
        path_delays_ms=[0.0],
        path_amplitudes=[1.0],
        path_doppler_spreads_hz=[5.0],
        description="Single path with Doppler for testing",
    ),
    CCIR520Condition.MULTIPATH_TEST: CCIR520ChannelSpec(
        condition=CCIR520Condition.MULTIPATH_TEST,
        delay_spread_ms=3.0,
        doppler_spread_hz=0.2,
        num_paths=3,
        path_delays_ms=[0.0, 1.5, 3.0],
        path_amplitudes=[1.0, 0.9, 0.7],
        path_doppler_spreads_hz=[0.2, 0.2, 0.2],
        description="Multipath test with low Doppler",
    ),
}


@dataclass
class ITURF1289ChannelSpec:
    """ITU-R F.1289 wideband HF channel specification.

    Defines wideband channel characteristics for testing modems
    with bandwidths up to 24 kHz.

    Attributes:
        condition: Channel condition preset
        delay_spread_ms: RMS delay spread
        doppler_spread_hz: Two-sided Doppler spread (2σ)
        coherence_bandwidth_khz: Frequency correlation bandwidth
        num_paths: Number of propagation paths
        path_delays_ms: Path delay values
        path_amplitudes: Path relative amplitudes
        frequency_selective: Whether channel is frequency-selective
        dispersion_factor: Group delay variation factor
    """

    condition: ITURF1289Condition
    delay_spread_ms: float
    doppler_spread_hz: float
    coherence_bandwidth_khz: float
    num_paths: int
    path_delays_ms: List[float]
    path_amplitudes: List[float]
    frequency_selective: bool
    dispersion_factor: float = 0.0
    description: str = ""


# ITU-R F.1289 Wideband Channel Specifications
ITURF1289_PRESETS: Dict[ITURF1289Condition, ITURF1289ChannelSpec] = {
    ITURF1289Condition.LOW_LATITUDE_QUIET: ITURF1289ChannelSpec(
        condition=ITURF1289Condition.LOW_LATITUDE_QUIET,
        delay_spread_ms=0.3,
        doppler_spread_hz=0.05,
        coherence_bandwidth_khz=500.0,
        num_paths=2,
        path_delays_ms=[0.0, 0.3],
        path_amplitudes=[1.0, 0.5],
        frequency_selective=False,
        description="Equatorial quiet conditions",
    ),
    ITURF1289Condition.LOW_LATITUDE_MODERATE: ITURF1289ChannelSpec(
        condition=ITURF1289Condition.LOW_LATITUDE_MODERATE,
        delay_spread_ms=0.8,
        doppler_spread_hz=0.2,
        coherence_bandwidth_khz=200.0,
        num_paths=2,
        path_delays_ms=[0.0, 0.8],
        path_amplitudes=[1.0, 0.7],
        frequency_selective=True,
        description="Equatorial moderate conditions",
    ),
    ITURF1289Condition.LOW_LATITUDE_DISTURBED: ITURF1289ChannelSpec(
        condition=ITURF1289Condition.LOW_LATITUDE_DISTURBED,
        delay_spread_ms=1.5,
        doppler_spread_hz=0.5,
        coherence_bandwidth_khz=100.0,
        num_paths=3,
        path_delays_ms=[0.0, 0.8, 1.5],
        path_amplitudes=[1.0, 0.7, 0.5],
        frequency_selective=True,
        dispersion_factor=0.1,
        description="Equatorial disturbed with spread-F",
    ),
    ITURF1289Condition.MID_LATITUDE_QUIET: ITURF1289ChannelSpec(
        condition=ITURF1289Condition.MID_LATITUDE_QUIET,
        delay_spread_ms=0.5,
        doppler_spread_hz=0.1,
        coherence_bandwidth_khz=300.0,
        num_paths=2,
        path_delays_ms=[0.0, 0.5],
        path_amplitudes=[1.0, 1.0],
        frequency_selective=False,
        description="Mid-latitude quiet daytime",
    ),
    ITURF1289Condition.MID_LATITUDE_MODERATE: ITURF1289ChannelSpec(
        condition=ITURF1289Condition.MID_LATITUDE_MODERATE,
        delay_spread_ms=2.0,
        doppler_spread_hz=1.0,
        coherence_bandwidth_khz=80.0,
        num_paths=2,
        path_delays_ms=[0.0, 2.0],
        path_amplitudes=[1.0, 1.0],
        frequency_selective=True,
        description="Mid-latitude moderate conditions",
    ),
    ITURF1289Condition.MID_LATITUDE_DISTURBED: ITURF1289ChannelSpec(
        condition=ITURF1289Condition.MID_LATITUDE_DISTURBED,
        delay_spread_ms=4.0,
        doppler_spread_hz=2.0,
        coherence_bandwidth_khz=40.0,
        num_paths=3,
        path_delays_ms=[0.0, 2.0, 4.0],
        path_amplitudes=[1.0, 0.8, 0.5],
        frequency_selective=True,
        dispersion_factor=0.2,
        description="Mid-latitude disturbed storm",
    ),
    ITURF1289Condition.HIGH_LATITUDE_QUIET: ITURF1289ChannelSpec(
        condition=ITURF1289Condition.HIGH_LATITUDE_QUIET,
        delay_spread_ms=1.0,
        doppler_spread_hz=0.5,
        coherence_bandwidth_khz=160.0,
        num_paths=2,
        path_delays_ms=[0.0, 1.0],
        path_amplitudes=[1.0, 0.8],
        frequency_selective=True,
        description="High-latitude quiet conditions",
    ),
    ITURF1289Condition.HIGH_LATITUDE_MODERATE: ITURF1289ChannelSpec(
        condition=ITURF1289Condition.HIGH_LATITUDE_MODERATE,
        delay_spread_ms=3.0,
        doppler_spread_hz=3.0,
        coherence_bandwidth_khz=50.0,
        num_paths=3,
        path_delays_ms=[0.0, 1.5, 3.0],
        path_amplitudes=[1.0, 0.8, 0.6],
        frequency_selective=True,
        dispersion_factor=0.15,
        description="High-latitude moderate auroral",
    ),
    ITURF1289Condition.HIGH_LATITUDE_DISTURBED: ITURF1289ChannelSpec(
        condition=ITURF1289Condition.HIGH_LATITUDE_DISTURBED,
        delay_spread_ms=7.0,
        doppler_spread_hz=5.0,
        coherence_bandwidth_khz=20.0,
        num_paths=4,
        path_delays_ms=[0.0, 2.5, 5.0, 7.0],
        path_amplitudes=[1.0, 0.8, 0.6, 0.4],
        frequency_selective=True,
        dispersion_factor=0.3,
        description="High-latitude severe storm",
    ),
    ITURF1289Condition.HIGH_LATITUDE_FLUTTER: ITURF1289ChannelSpec(
        condition=ITURF1289Condition.HIGH_LATITUDE_FLUTTER,
        delay_spread_ms=2.0,
        doppler_spread_hz=10.0,
        coherence_bandwidth_khz=80.0,
        num_paths=2,
        path_delays_ms=[0.0, 2.0],
        path_amplitudes=[1.0, 0.8],
        frequency_selective=True,
        dispersion_factor=0.1,
        description="High-latitude flutter fading",
    ),
}


@dataclass
class ITURF1487ChannelSpec:
    """ITU-R F.1487 channel specification for modem testing.

    Defines channel parameters for testing HF modems with
    bandwidths up to about 12 kHz.
    """

    condition: ITURF1487Condition
    delay_spread_ms: float
    doppler_spread_hz: float
    num_paths: int
    description: str


# ITU-R F.1487 Specifications (Table 1)
ITURF1487_PRESETS: Dict[ITURF1487Condition, ITURF1487ChannelSpec] = {
    ITURF1487Condition.QUIET: ITURF1487ChannelSpec(
        condition=ITURF1487Condition.QUIET,
        delay_spread_ms=0.5,
        doppler_spread_hz=0.1,
        num_paths=2,
        description="Quiet mid-latitude, good SNR",
    ),
    ITURF1487Condition.MODERATE: ITURF1487ChannelSpec(
        condition=ITURF1487Condition.MODERATE,
        delay_spread_ms=2.0,
        doppler_spread_hz=1.0,
        num_paths=2,
        description="Moderate conditions, typical daytime",
    ),
    ITURF1487Condition.DISTURBED: ITURF1487ChannelSpec(
        condition=ITURF1487Condition.DISTURBED,
        delay_spread_ms=4.0,
        doppler_spread_hz=2.0,
        num_paths=3,
        description="Disturbed, magnetic storm",
    ),
    ITURF1487Condition.FLUTTER: ITURF1487ChannelSpec(
        condition=ITURF1487Condition.FLUTTER,
        delay_spread_ms=7.0,
        doppler_spread_hz=10.0,
        num_paths=2,
        description="Flutter fading, high-latitude auroral",
    ),
}


class CCIR520Channel(WattersonChannel):
    """CCIR 520 / ITU-R F.520 standardized HF channel.

    Implements the classic CCIR 520 channel model (now ITU-R F.520-2)
    which defines standardized HF channel conditions for modem testing.

    Example:
        channel = CCIR520Channel.from_preset(CCIR520Condition.MODERATE)
        output = channel.process(input_signal)
    """

    def __init__(
        self,
        spec: CCIR520ChannelSpec,
        sample_rate_hz: float = 2_000_000,
        seed: Optional[int] = None,
    ):
        """Initialize CCIR 520 channel from specification.

        Args:
            spec: Channel specification
            sample_rate_hz: Sample rate
            seed: Random seed for reproducibility
        """
        self.spec = spec

        # Convert spec to Watterson config
        taps = []
        for i in range(spec.num_paths):
            taps.append(WattersonTap(
                delay_ms=spec.path_delays_ms[i],
                amplitude=spec.path_amplitudes[i],
                doppler_spread_hz=spec.path_doppler_spreads_hz[i],
                doppler_spectrum=spec.doppler_spectrum,
                is_specular=(spec.k_factor_db is not None and i == 0),
                k_factor_db=spec.k_factor_db or 0.0,
            ))

        config = WattersonConfig(
            taps=taps,
            sample_rate_hz=sample_rate_hz,
        )

        super().__init__(config, seed)

    @classmethod
    def from_preset(
        cls,
        condition: CCIR520Condition,
        sample_rate_hz: float = 2_000_000,
        seed: Optional[int] = None,
    ) -> "CCIR520Channel":
        """Create channel from preset condition.

        Args:
            condition: CCIR 520 condition preset
            sample_rate_hz: Sample rate
            seed: Random seed

        Returns:
            CCIR520Channel instance
        """
        spec = CCIR520_PRESETS[condition]
        return cls(spec, sample_rate_hz, seed)

    @classmethod
    def good(cls, sample_rate_hz: float = 2_000_000, seed: Optional[int] = None) -> "CCIR520Channel":
        """Create CCIR 'Good' channel (0.5ms delay, 0.1Hz Doppler)."""
        return cls.from_preset(CCIR520Condition.GOOD_LOW_LATENCY, sample_rate_hz, seed)

    @classmethod
    def moderate(cls, sample_rate_hz: float = 2_000_000, seed: Optional[int] = None) -> "CCIR520Channel":
        """Create CCIR 'Moderate' channel (1ms delay, 0.5Hz Doppler)."""
        return cls.from_preset(CCIR520Condition.MODERATE, sample_rate_hz, seed)

    @classmethod
    def poor(cls, sample_rate_hz: float = 2_000_000, seed: Optional[int] = None) -> "CCIR520Channel":
        """Create CCIR 'Poor' channel (2ms delay, 1Hz Doppler)."""
        return cls.from_preset(CCIR520Condition.POOR, sample_rate_hz, seed)

    @classmethod
    def flutter(cls, sample_rate_hz: float = 2_000_000, seed: Optional[int] = None) -> "CCIR520Channel":
        """Create flutter fading channel (1ms delay, 10Hz Doppler)."""
        return cls.from_preset(CCIR520Condition.FLUTTER, sample_rate_hz, seed)


class ITURF1289Channel(WattersonChannel):
    """ITU-R F.1289 wideband HF channel model.

    Implements the ITU-R F.1289 wideband channel model for testing
    HF modems with bandwidths up to 24 kHz.

    Features frequency-selective fading and group delay dispersion
    for wideband operation.
    """

    def __init__(
        self,
        spec: ITURF1289ChannelSpec,
        sample_rate_hz: float = 2_000_000,
        bandwidth_khz: float = 24.0,
        seed: Optional[int] = None,
    ):
        """Initialize ITU-R F.1289 wideband channel.

        Args:
            spec: Channel specification
            sample_rate_hz: Sample rate
            bandwidth_khz: Signal bandwidth in kHz
            seed: Random seed
        """
        self.spec = spec
        self.bandwidth_khz = bandwidth_khz
        self._dispersion_enabled = spec.frequency_selective
        self._dispersion_factor = spec.dispersion_factor

        # Create taps from spec
        taps = []
        for i in range(spec.num_paths):
            # Doppler spread may vary slightly per path in wideband
            doppler = spec.doppler_spread_hz

            taps.append(WattersonTap(
                delay_ms=spec.path_delays_ms[i],
                amplitude=spec.path_amplitudes[i],
                doppler_spread_hz=doppler,
                doppler_spectrum=DopplerSpectrum.GAUSSIAN,
            ))

        config = WattersonConfig(
            taps=taps,
            sample_rate_hz=sample_rate_hz,
        )

        super().__init__(config, seed)

        # Dispersion filter for frequency-selective fading
        self._dispersion_filter = self._create_dispersion_filter()

    def _create_dispersion_filter(self) -> Optional[np.ndarray]:
        """Create group delay dispersion filter.

        Returns:
            FIR filter coefficients or None if disabled
        """
        if not self._dispersion_enabled or self._dispersion_factor <= 0:
            return None

        # Create group delay variation filter
        # Filter length based on dispersion factor and sample rate
        n_taps = 64
        freq_response = np.ones(n_taps, dtype=np.complex128)

        # Add frequency-dependent phase (group delay variation)
        freqs = np.fft.fftfreq(n_taps)
        phase_variation = self._dispersion_factor * np.pi * freqs ** 2
        freq_response *= np.exp(1j * phase_variation)

        # Convert to time domain
        h = np.fft.ifft(freq_response)

        return h.astype(np.complex64)

    def process_block(self, input_samples: np.ndarray) -> np.ndarray:
        """Process samples with wideband effects.

        Args:
            input_samples: Complex input samples

        Returns:
            Complex output with multipath and dispersion
        """
        # Apply basic Watterson processing
        output = super().process_block(input_samples)

        # Apply dispersion if enabled
        if self._dispersion_filter is not None:
            output = signal.lfilter(self._dispersion_filter, [1.0], output)

        return output.astype(np.complex64)

    @classmethod
    def from_preset(
        cls,
        condition: ITURF1289Condition,
        sample_rate_hz: float = 2_000_000,
        bandwidth_khz: float = 24.0,
        seed: Optional[int] = None,
    ) -> "ITURF1289Channel":
        """Create channel from preset condition.

        Args:
            condition: ITU-R F.1289 condition
            sample_rate_hz: Sample rate
            bandwidth_khz: Signal bandwidth
            seed: Random seed

        Returns:
            ITURF1289Channel instance
        """
        spec = ITURF1289_PRESETS[condition]
        return cls(spec, sample_rate_hz, bandwidth_khz, seed)

    def get_coherence_bandwidth(self) -> float:
        """Get channel coherence bandwidth in kHz."""
        return self.spec.coherence_bandwidth_khz

    def is_frequency_selective(self) -> bool:
        """Check if channel is frequency-selective for current bandwidth."""
        return self.bandwidth_khz > self.spec.coherence_bandwidth_khz


class ITURF1487Channel(CCIR520Channel):
    """ITU-R F.1487 channel for HF modem testing.

    Simplified channel model based on ITU-R F.1487 Table 1
    for testing modems with bandwidths up to 12 kHz.
    """

    def __init__(
        self,
        condition: ITURF1487Condition,
        sample_rate_hz: float = 2_000_000,
        seed: Optional[int] = None,
    ):
        """Initialize ITU-R F.1487 channel.

        Args:
            condition: Channel condition
            sample_rate_hz: Sample rate
            seed: Random seed
        """
        spec_1487 = ITURF1487_PRESETS[condition]

        # Convert to CCIR520 spec format
        num_paths = spec_1487.num_paths

        if num_paths == 2:
            path_delays = [0.0, spec_1487.delay_spread_ms]
            path_amplitudes = [1.0, 1.0]
        else:  # 3 paths
            path_delays = [0.0, spec_1487.delay_spread_ms / 2, spec_1487.delay_spread_ms]
            path_amplitudes = [1.0, 0.8, 0.5]

        spec = CCIR520ChannelSpec(
            condition=CCIR520Condition.MODERATE,  # Placeholder
            delay_spread_ms=spec_1487.delay_spread_ms,
            doppler_spread_hz=spec_1487.doppler_spread_hz,
            num_paths=num_paths,
            path_delays_ms=path_delays,
            path_amplitudes=path_amplitudes,
            path_doppler_spreads_hz=[spec_1487.doppler_spread_hz] * num_paths,
            description=spec_1487.description,
        )

        super().__init__(spec, sample_rate_hz, seed)
        self.itu_condition = condition

    @classmethod
    def quiet(cls, sample_rate_hz: float = 2_000_000, seed: Optional[int] = None) -> "ITURF1487Channel":
        """Create quiet channel (τ=0.5ms, ν=0.1Hz)."""
        return cls(ITURF1487Condition.QUIET, sample_rate_hz, seed)

    @classmethod
    def moderate(cls, sample_rate_hz: float = 2_000_000, seed: Optional[int] = None) -> "ITURF1487Channel":
        """Create moderate channel (τ=2ms, ν=1Hz)."""
        return cls(ITURF1487Condition.MODERATE, sample_rate_hz, seed)

    @classmethod
    def disturbed(cls, sample_rate_hz: float = 2_000_000, seed: Optional[int] = None) -> "ITURF1487Channel":
        """Create disturbed channel (τ=4ms, ν=2Hz)."""
        return cls(ITURF1487Condition.DISTURBED, sample_rate_hz, seed)

    @classmethod
    def flutter(cls, sample_rate_hz: float = 2_000_000, seed: Optional[int] = None) -> "ITURF1487Channel":
        """Create flutter channel (τ=7ms, ν=10Hz)."""
        return cls(ITURF1487Condition.FLUTTER, sample_rate_hz, seed)


def list_ccir520_presets() -> List[str]:
    """List available CCIR 520 preset names."""
    return [c.value for c in CCIR520Condition]


def list_iturf1289_presets() -> List[str]:
    """List available ITU-R F.1289 preset names."""
    return [c.value for c in ITURF1289Condition]


def list_iturf1487_presets() -> List[str]:
    """List available ITU-R F.1487 preset names."""
    return [c.value for c in ITURF1487Condition]


def get_preset_info(preset_name: str) -> Optional[Dict]:
    """Get information about a preset by name.

    Args:
        preset_name: Preset name string

    Returns:
        Dictionary with preset parameters or None
    """
    # Check CCIR520
    for cond in CCIR520Condition:
        if cond.value == preset_name:
            spec = CCIR520_PRESETS[cond]
            return {
                "type": "CCIR520",
                "condition": cond.value,
                "delay_spread_ms": spec.delay_spread_ms,
                "doppler_spread_hz": spec.doppler_spread_hz,
                "num_paths": spec.num_paths,
                "description": spec.description,
            }

    # Check ITU-R F.1289
    for cond in ITURF1289Condition:
        if cond.value == preset_name:
            spec = ITURF1289_PRESETS[cond]
            return {
                "type": "ITURF1289",
                "condition": cond.value,
                "delay_spread_ms": spec.delay_spread_ms,
                "doppler_spread_hz": spec.doppler_spread_hz,
                "coherence_bandwidth_khz": spec.coherence_bandwidth_khz,
                "num_paths": spec.num_paths,
                "description": spec.description,
            }

    # Check ITU-R F.1487
    for cond in ITURF1487Condition:
        if cond.value == preset_name:
            spec = ITURF1487_PRESETS[cond]
            return {
                "type": "ITURF1487",
                "condition": cond.value,
                "delay_spread_ms": spec.delay_spread_ms,
                "doppler_spread_hz": spec.doppler_spread_hz,
                "num_paths": spec.num_paths,
                "description": spec.description,
            }

    return None


def create_channel(
    preset_name: str,
    sample_rate_hz: float = 2_000_000,
    seed: Optional[int] = None,
) -> WattersonChannel:
    """Create a channel model from any preset name.

    Args:
        preset_name: Name of the preset (from any ITU-R recommendation)
        sample_rate_hz: Sample rate
        seed: Random seed

    Returns:
        Appropriate channel model instance

    Raises:
        ValueError: If preset name not found
    """
    # Try CCIR520
    for cond in CCIR520Condition:
        if cond.value == preset_name:
            return CCIR520Channel.from_preset(cond, sample_rate_hz, seed)

    # Try ITU-R F.1289
    for cond in ITURF1289Condition:
        if cond.value == preset_name:
            return ITURF1289Channel.from_preset(cond, sample_rate_hz, seed=seed)

    # Try ITU-R F.1487
    for cond in ITURF1487Condition:
        if cond.value == preset_name:
            return ITURF1487Channel(cond, sample_rate_hz, seed)

    raise ValueError(f"Unknown preset: {preset_name}")
