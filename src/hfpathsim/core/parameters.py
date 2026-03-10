"""Ionospheric and channel parameters for Vogler-Hoffmeyer IPM."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class ITUCondition(Enum):
    """ITU-R F.1487 channel condition presets."""

    QUIET = "quiet"
    MODERATE = "moderate"
    DISTURBED = "disturbed"
    FLUTTER = "flutter"  # High-latitude flutter fading


# ITU-R F.1487 parameter presets (Table 1)
ITU_PRESETS = {
    ITUCondition.QUIET: {
        "delay_spread_ms": 0.5,
        "doppler_spread_hz": 0.1,
        "num_paths": 2,
        "description": "Quiet mid-latitude conditions",
    },
    ITUCondition.MODERATE: {
        "delay_spread_ms": 2.0,
        "doppler_spread_hz": 1.0,
        "num_paths": 2,
        "description": "Moderate conditions - typical daytime",
    },
    ITUCondition.DISTURBED: {
        "delay_spread_ms": 4.0,
        "doppler_spread_hz": 2.0,
        "num_paths": 3,
        "description": "Disturbed conditions - magnetic storm",
    },
    ITUCondition.FLUTTER: {
        "delay_spread_ms": 7.0,
        "doppler_spread_hz": 10.0,
        "num_paths": 2,
        "description": "High-latitude flutter fading",
    },
}


@dataclass
class PropagationMode:
    """Single ionospheric propagation mode (e.g., 1F2, 2F2, 1E)."""

    name: str  # e.g., "1F2", "2F2", "1E"
    enabled: bool = True
    relative_amplitude: float = 1.0  # Relative to strongest mode
    delay_offset_ms: float = 0.0  # Additional delay relative to first mode


@dataclass
class VoglerParameters:
    """Parameters for Vogler-Hoffmeyer Ionospheric Propagation Model.

    Based on NTIA TR-88-240 "A full-wave calculation of ionospheric Doppler
    spread and its application to HF channel modeling."
    """

    # Critical ionospheric parameters
    foF2: float = 7.5  # F2 layer critical frequency (MHz)
    hmF2: float = 300.0  # F2 layer peak height (km)
    foE: float = 3.0  # E layer critical frequency (MHz)
    hmE: float = 110.0  # E layer peak height (km)

    # Layer shape parameters (quasi-parabolic model)
    ym_F2: float = 100.0  # F2 layer semi-thickness (km)
    ym_E: float = 20.0  # E layer semi-thickness (km)

    # Vogler model parameters
    sigma: float = 0.1  # Layer thickness parameter (dimensionless)
    chi: Optional[float] = None  # Penetration parameter (computed if None)

    # Stochastic fading parameters
    doppler_spread_hz: float = 1.0  # Two-sided Doppler spread (Hz)
    delay_spread_ms: float = 2.0  # Delay spread (ms)

    # Path geometry
    path_length_km: float = 1000.0  # Great circle path length
    frequency_mhz: float = 10.0  # Operating frequency (MHz)

    # Propagation modes
    modes: list[PropagationMode] = field(default_factory=lambda: [
        PropagationMode("1F2", True, 1.0, 0.0),
        PropagationMode("2F2", True, 0.7, 1.5),
    ])

    def __post_init__(self):
        """Compute derived parameters."""
        if self.chi is None:
            self.chi = self._compute_chi()

    def _compute_chi(self) -> float:
        """Compute penetration parameter chi from ionospheric parameters.

        chi determines the character of reflection:
        - chi > 0.5: partial reflection (below critical frequency)
        - chi < 0.5: total reflection (above critical frequency)
        """
        # Simplified calculation - full version requires ray tracing
        fc = self.foF2  # Critical frequency
        f = self.frequency_mhz
        if f <= fc:
            # Below critical frequency - partial penetration
            return 0.5 * (1 - (f / fc) ** 2)
        else:
            # Above critical frequency for vertical incidence
            # Oblique incidence can still reflect via secant law
            # MUF = foF2 * sec(phi) where phi is angle of incidence
            # Assume typical oblique geometry
            sec_phi = 3.0  # Typical for 1000 km path
            muf = fc * sec_phi
            if f <= muf:
                return 0.5 * (1 - (f / muf) ** 2)
            else:
                return -0.5  # No reflection possible

    @classmethod
    def from_itu_condition(
        cls,
        condition: ITUCondition,
        frequency_mhz: float = 10.0,
        path_length_km: float = 1000.0,
    ) -> "VoglerParameters":
        """Create parameters from ITU-R F.1487 condition preset.

        Args:
            condition: ITU condition enum
            frequency_mhz: Operating frequency in MHz
            path_length_km: Path length in km

        Returns:
            VoglerParameters instance with preset values
        """
        preset = ITU_PRESETS[condition]

        # Estimate ionospheric parameters from condition
        # These are typical daytime mid-latitude values
        if condition == ITUCondition.QUIET:
            foF2, hmF2 = 8.0, 280.0
        elif condition == ITUCondition.MODERATE:
            foF2, hmF2 = 7.0, 300.0
        elif condition == ITUCondition.DISTURBED:
            foF2, hmF2 = 5.0, 350.0
        else:  # FLUTTER
            foF2, hmF2 = 6.0, 320.0

        return cls(
            foF2=foF2,
            hmF2=hmF2,
            doppler_spread_hz=preset["doppler_spread_hz"],
            delay_spread_ms=preset["delay_spread_ms"],
            frequency_mhz=frequency_mhz,
            path_length_km=path_length_km,
        )

    def get_base_delay_ms(self) -> float:
        """Calculate base propagation delay in milliseconds.

        Uses simplified geometry assuming single-hop reflection.
        """
        # Speed of light
        c = 299792.458  # km/s

        # Simple geometric calculation for hop length
        # Assume reflection at hmF2
        d = self.path_length_km / 2  # Half path
        h = self.hmF2
        hop_length = 2 * np.sqrt(d**2 + h**2)

        # Delay in ms
        return hop_length / c * 1000

    def get_coherence_time_ms(self) -> float:
        """Calculate channel coherence time from Doppler spread."""
        if self.doppler_spread_hz <= 0:
            return float("inf")
        # Coherence time ~ 1 / (2 * pi * doppler_spread)
        return 1000 / (2 * np.pi * self.doppler_spread_hz)

    def get_coherence_bandwidth_khz(self) -> float:
        """Calculate channel coherence bandwidth from delay spread."""
        if self.delay_spread_ms <= 0:
            return float("inf")
        # Coherence bandwidth ~ 1 / (2 * pi * delay_spread)
        return 1 / (2 * np.pi * self.delay_spread_ms)


@dataclass
class ChannelState:
    """Current state of the time-varying channel."""

    timestamp: float = 0.0  # Current time in seconds
    transfer_function: Optional[np.ndarray] = None  # H(f) complex array
    impulse_response: Optional[np.ndarray] = None  # h(t) complex array
    scattering_function: Optional[np.ndarray] = None  # S(tau, nu) real array

    # Axis arrays for plotting
    freq_axis_hz: Optional[np.ndarray] = None
    time_axis_ms: Optional[np.ndarray] = None
    delay_axis_ms: Optional[np.ndarray] = None
    doppler_axis_hz: Optional[np.ndarray] = None
