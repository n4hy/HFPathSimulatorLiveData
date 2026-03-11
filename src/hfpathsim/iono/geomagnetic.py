"""Geomagnetic effects on ionospheric propagation.

Provides modulation of ionospheric parameters based on:
- Solar flux (F10.7) index: affects background ionization
- Kp geomagnetic index: affects auroral activity and spread
- Dst storm-time index: affects main phase depression

These effects modify the base ionospheric parameters from IRI
or other sources to account for space weather conditions.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class GeomagneticIndices:
    """Space weather indices for ionospheric modulation.

    Attributes:
        f10_7: F10.7 solar radio flux (sfu, 10^-22 W/m^2/Hz)
               Quiet: ~70, Moderate: 100-150, Active: >200
        kp: Kp geomagnetic index (0-9)
            Quiet: 0-2, Unsettled: 3-4, Storm: 5+
        dst: Dst storm-time index (nT)
             Quiet: ~0, Moderate storm: -50 to -100
             Intense storm: < -100
        ap: Ap geomagnetic index (daily average, 0-400)
        ssn: Sunspot number (optional)
    """

    f10_7: float = 100.0  # Moderate solar activity
    kp: float = 2.0  # Quiet conditions
    dst: float = 0.0  # No storm
    ap: float = 10.0  # Quiet
    ssn: Optional[float] = None  # Sunspot number

    def __post_init__(self):
        """Validate indices."""
        self.f10_7 = max(65.0, min(350.0, self.f10_7))
        self.kp = max(0.0, min(9.0, self.kp))
        self.dst = max(-500.0, min(50.0, self.dst))
        self.ap = max(0.0, min(400.0, self.ap))

    @classmethod
    def quiet(cls) -> "GeomagneticIndices":
        """Create quiet geomagnetic conditions."""
        return cls(f10_7=70.0, kp=1.0, dst=0.0, ap=5.0)

    @classmethod
    def moderate(cls) -> "GeomagneticIndices":
        """Create moderate activity conditions."""
        return cls(f10_7=120.0, kp=3.0, dst=-20.0, ap=20.0)

    @classmethod
    def disturbed(cls) -> "GeomagneticIndices":
        """Create disturbed/storm conditions."""
        return cls(f10_7=100.0, kp=5.0, dst=-80.0, ap=50.0)

    @classmethod
    def severe_storm(cls) -> "GeomagneticIndices":
        """Create severe storm conditions."""
        return cls(f10_7=100.0, kp=7.0, dst=-200.0, ap=100.0)

    @classmethod
    def solar_maximum(cls) -> "GeomagneticIndices":
        """Create solar maximum conditions."""
        return cls(f10_7=200.0, kp=2.0, dst=0.0, ap=10.0, ssn=150)

    @classmethod
    def solar_minimum(cls) -> "GeomagneticIndices":
        """Create solar minimum conditions."""
        return cls(f10_7=70.0, kp=1.0, dst=0.0, ap=5.0, ssn=10)


class GeomagneticModulator:
    """Modulates ionospheric parameters based on space weather.

    Applies empirical corrections to foF2, hmF2, and channel
    fading parameters based on geomagnetic indices.
    """

    def __init__(self, indices: Optional[GeomagneticIndices] = None):
        """Initialize modulator.

        Args:
            indices: Geomagnetic indices (default: quiet conditions)
        """
        self.indices = indices or GeomagneticIndices.quiet()

    def set_indices(self, indices: GeomagneticIndices):
        """Update geomagnetic indices.

        Args:
            indices: New indices
        """
        self.indices = indices

    def scale_foF2(
        self,
        foF2_base: float,
        latitude: float = 45.0,
    ) -> float:
        """Scale foF2 based on geomagnetic conditions.

        Applies:
        1. F10.7 solar flux scaling (higher flux -> higher foF2)
        2. Dst storm-time depression (negative Dst -> lower foF2)

        The empirical relations are:
            foF2 ~ sqrt(1 + 0.014 * (F10.7 - 100))  for solar flux
            delta_foF2 ~ 0.05 * Dst * cos^2(lat)    for storm depression

        Note: Dst is negative during storms, so the effect is:
        - Dst < 0 (storm) -> delta_foF2 < 0 -> foF2 decreases

        Args:
            foF2_base: Base foF2 from IRI or model (MHz)
            latitude: Geographic latitude (degrees)

        Returns:
            Scaled foF2 in MHz
        """
        # F10.7 scaling
        # Reference flux is 100 sfu
        flux_factor = np.sqrt(max(0.5, 1.0 + 0.014 * (self.indices.f10_7 - 100)))

        # Dst storm depression
        # During main phase of storm (Dst negative), foF2 decreases
        # The 0.05 coefficient means Dst=-100 causes ~5 MHz depression at equator
        cos_lat_sq = np.cos(np.radians(latitude)) ** 2
        dst_change = 0.05 * self.indices.dst * cos_lat_sq  # MHz (negative during storms)

        # Apply both effects
        foF2_scaled = foF2_base * flux_factor + dst_change

        # Ensure reasonable bounds
        return max(1.0, min(20.0, foF2_scaled))

    def scale_hmF2(
        self,
        hmF2_base: float,
        latitude: float = 45.0,
    ) -> float:
        """Scale hmF2 based on geomagnetic conditions.

        Storm conditions generally raise hmF2 (thermal expansion).

        Args:
            hmF2_base: Base hmF2 (km)
            latitude: Geographic latitude (degrees)

        Returns:
            Scaled hmF2 in km
        """
        # Dst effect - negative Dst raises hmF2
        # Effect is stronger at equator
        cos_lat_sq = np.cos(np.radians(latitude)) ** 2
        dst_rise = -0.2 * self.indices.dst * cos_lat_sq  # km

        # Kp effect - increased Kp raises hmF2
        kp_rise = 5.0 * max(0, self.indices.kp - 3)  # km above Kp=3

        hmF2_scaled = hmF2_base + dst_rise + kp_rise

        # Ensure reasonable bounds
        return max(200.0, min(500.0, hmF2_scaled))

    def scale_doppler_spread(
        self,
        doppler_base_hz: float,
        latitude: float = 45.0,
    ) -> float:
        """Scale Doppler spread based on geomagnetic activity.

        Higher Kp increases Doppler spread due to:
        - Enhanced ionospheric irregularities
        - Auroral zone expansion at high latitudes

        Empirical relation:
            doppler_enhanced = doppler_base * (1 + 0.1 * Kp)

        At high latitudes during storms, can get much more enhancement.

        Args:
            doppler_base_hz: Base Doppler spread (Hz)
            latitude: Geographic latitude (degrees)

        Returns:
            Enhanced Doppler spread in Hz
        """
        # Base Kp scaling
        kp_factor = 1.0 + 0.1 * self.indices.kp

        # Additional enhancement at high latitudes during storms
        abs_lat = abs(latitude)
        if abs_lat > 50 and self.indices.kp > 4:
            # Auroral zone enhancement
            auroral_factor = 1.0 + 0.2 * (self.indices.kp - 4) * (abs_lat - 50) / 20
            kp_factor *= auroral_factor

        doppler_enhanced = doppler_base_hz * kp_factor

        return min(20.0, doppler_enhanced)  # Cap at 20 Hz

    def scale_delay_spread(
        self,
        delay_base_ms: float,
        latitude: float = 45.0,
    ) -> float:
        """Scale delay spread based on geomagnetic activity.

        Increased Kp can cause multipath enhancement.

        Args:
            delay_base_ms: Base delay spread (ms)
            latitude: Geographic latitude (degrees)

        Returns:
            Enhanced delay spread in ms
        """
        # Moderate Kp effect on delay spread
        kp_factor = 1.0 + 0.05 * self.indices.kp

        # High latitude enhancement during storms
        abs_lat = abs(latitude)
        if abs_lat > 55 and self.indices.kp > 4:
            kp_factor *= 1.0 + 0.1 * (self.indices.kp - 4)

        delay_enhanced = delay_base_ms * kp_factor

        return min(10.0, delay_enhanced)  # Cap at 10 ms

    def get_absorption_factor(
        self,
        frequency_mhz: float,
        latitude: float = 45.0,
    ) -> float:
        """Estimate D-region absorption enhancement factor.

        During disturbed conditions, especially polar cap absorption
        events (PCAs), HF absorption increases dramatically.

        Args:
            frequency_mhz: Operating frequency (MHz)
            latitude: Geographic latitude (degrees)

        Returns:
            Absorption multiplier (1.0 = normal, >1 = enhanced)
        """
        # Kp-dependent absorption at high latitudes
        abs_lat = abs(latitude)

        if abs_lat < 55:
            # Mid-latitude - moderate enhancement
            return 1.0 + 0.02 * max(0, self.indices.kp - 2)

        # High latitude - significant enhancement possible
        base_enhancement = 1.0 + 0.1 * max(0, self.indices.kp - 2)

        # Frequency dependence - lower frequencies more affected
        # Absorption ~ f^-2
        freq_factor = (10.0 / max(3.0, frequency_mhz)) ** 2

        return base_enhancement * (0.5 + 0.5 * freq_factor)

    def is_blackout(
        self,
        frequency_mhz: float,
        latitude: float = 45.0,
    ) -> bool:
        """Check if polar blackout conditions exist.

        During severe storms (Kp > 7), high latitudes may experience
        complete HF blackout.

        Args:
            frequency_mhz: Operating frequency (MHz)
            latitude: Geographic latitude (degrees)

        Returns:
            True if blackout conditions likely
        """
        abs_lat = abs(latitude)

        # Polar blackout during severe storms
        if abs_lat > 60 and self.indices.kp >= 7:
            # Lower frequencies black out first
            if frequency_mhz < 10:
                return True
            elif frequency_mhz < 15 and self.indices.kp >= 8:
                return True

        return False

    def apply_to_profile(
        self,
        profile,
        latitude: float = 45.0,
    ):
        """Apply geomagnetic modulation to ionosphere profile.

        Args:
            profile: IonosphereProfile to modify
            latitude: Geographic latitude (degrees)

        Returns:
            Modified IonosphereProfile
        """
        from hfpathsim.core.raytracing.ionosphere import IonosphereProfile

        # Scale critical frequencies and heights
        new_foF2 = self.scale_foF2(profile.foF2, latitude)
        new_hmF2 = self.scale_hmF2(profile.hmF2, latitude)

        # E layer is less affected by storms
        new_foE = profile.foE  # Keep base value
        new_hmE = profile.hmE

        # Scale electron density array proportionally
        foF2_ratio = new_foF2 / profile.foF2 if profile.foF2 > 0 else 1.0
        new_ne = profile.electron_density * (foF2_ratio ** 2)

        return IonosphereProfile(
            altitude_km=profile.altitude_km.copy(),
            electron_density=new_ne,
            foF2=new_foF2,
            hmF2=new_hmF2,
            foE=new_foE,
            hmE=new_hmE,
            foEs=profile.foEs,
            hmEs=profile.hmEs,
            foF1=profile.foF1,
            hmF1=profile.hmF1,
            ym_F2=profile.ym_F2,
            ym_E=profile.ym_E,
        )


def kp_from_ap(ap: float) -> float:
    """Convert Ap index to Kp index.

    Uses standard IAGA table.

    Args:
        ap: Ap index (0-400)

    Returns:
        Kp index (0-9)
    """
    # IAGA conversion table (approximate)
    ap_thresholds = [0, 2, 3, 4, 5, 6, 7, 9, 12, 15, 18, 22, 27, 32, 39,
                     48, 56, 67, 80, 94, 111, 132, 154, 179, 207, 236,
                     300, 400]
    kp_values = [0.0, 0.33, 0.67, 1.0, 1.33, 1.67, 2.0, 2.33, 2.67, 3.0,
                 3.33, 3.67, 4.0, 4.33, 4.67, 5.0, 5.33, 5.67, 6.0, 6.33,
                 6.67, 7.0, 7.33, 7.67, 8.0, 8.33, 8.67, 9.0]

    for i, threshold in enumerate(ap_thresholds):
        if ap < threshold:
            return kp_values[max(0, i - 1)]

    return 9.0


def ap_from_kp(kp: float) -> float:
    """Convert Kp index to Ap index.

    Args:
        kp: Kp index (0-9)

    Returns:
        Ap index
    """
    # Approximate inverse conversion
    if kp < 0:
        return 0
    elif kp <= 2:
        return 3 * kp
    elif kp <= 4:
        return 5 + 10 * (kp - 2)
    elif kp <= 6:
        return 25 + 20 * (kp - 4)
    elif kp <= 8:
        return 65 + 50 * (kp - 6)
    else:
        return 165 + 100 * (kp - 8)


def estimate_ssn_from_f10_7(f10_7: float) -> float:
    """Estimate sunspot number from F10.7 flux.

    Uses empirical relation: F10.7 = 67 + 0.57 * SSN + 0.0012 * SSN^2

    Args:
        f10_7: F10.7 solar flux (sfu)

    Returns:
        Estimated sunspot number
    """
    # Solve quadratic for SSN
    # 0.0012 * SSN^2 + 0.57 * SSN + (67 - F10.7) = 0
    a = 0.0012
    b = 0.57
    c = 67 - f10_7

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return 0

    ssn = (-b + np.sqrt(discriminant)) / (2 * a)
    return max(0, ssn)


@dataclass
class StormPhase:
    """Ionospheric storm phase characterization."""

    name: str
    description: str
    foF2_change: str  # "increase", "decrease", "normal"
    hmF2_change: str
    typical_duration_hours: float


# Storm phase definitions
STORM_PHASES = {
    "initial": StormPhase(
        name="initial",
        description="Initial positive phase with foF2 enhancement",
        foF2_change="increase",
        hmF2_change="increase",
        typical_duration_hours=6,
    ),
    "main": StormPhase(
        name="main",
        description="Main phase with foF2 depression",
        foF2_change="decrease",
        hmF2_change="increase",
        typical_duration_hours=12,
    ),
    "recovery": StormPhase(
        name="recovery",
        description="Recovery phase returning to normal",
        foF2_change="normal",
        hmF2_change="normal",
        typical_duration_hours=24,
    ),
}


def classify_storm_phase(
    dst: float,
    dst_rate: float = 0.0,
) -> str:
    """Classify ionospheric storm phase from Dst.

    Args:
        dst: Current Dst value (nT)
        dst_rate: Rate of Dst change (nT/hour)

    Returns:
        Storm phase name: "quiet", "initial", "main", "recovery"
    """
    if dst > -20:
        return "quiet"
    elif dst_rate < -10:
        # Rapidly decreasing Dst = main phase onset
        return "main"
    elif dst_rate > 5:
        # Recovering (Dst becoming less negative)
        return "recovery"
    elif dst > -50:
        return "initial"
    else:
        return "main"
