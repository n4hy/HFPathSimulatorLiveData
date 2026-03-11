"""Sporadic-E layer model for HF propagation.

Provides time-varying sporadic-E (Es) layer modeling including
occurrence probability, critical frequency variations, and
injection into ionosphere profiles.

Sporadic-E characteristics:
- Thin (~5 km), intense ionization patches at ~100-120 km
- Can support frequencies well above normal E layer MUF
- Highly variable in time and space
- More common in summer daytime at mid-latitudes
- Affects frequencies up to ~150 MHz in extreme cases
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
from datetime import datetime
import numpy as np


@dataclass
class SporadicEConfig:
    """Configuration for sporadic-E layer simulation.

    Attributes:
        enabled: Whether Es layer is active
        foEs_mhz: Sporadic-E critical frequency (MHz)
        hmEs_km: Es layer height (km), typically 100-120 km
        thickness_km: Layer semi-thickness (km), typically 2-10 km
        blanketing_freq_mhz: fbEs, frequency below which Es is opaque
        variability_period_s: Period of foEs variation (seconds)
        variability_amplitude: Amplitude of foEs variation (0-1)
    """

    enabled: bool = False
    foEs_mhz: float = 5.0  # Typical daytime value
    hmEs_km: float = 105.0  # Typical height
    thickness_km: float = 5.0  # Semi-thickness
    blanketing_freq_mhz: Optional[float] = None  # fbEs
    variability_period_s: float = 300.0  # 5 minute variations
    variability_amplitude: float = 0.2  # 20% variation

    def __post_init__(self):
        """Validate configuration."""
        if self.foEs_mhz < 0:
            raise ValueError("foEs must be non-negative")
        if self.hmEs_km < 80 or self.hmEs_km > 150:
            raise ValueError("hmEs should be between 80-150 km")
        if self.thickness_km <= 0 or self.thickness_km > 20:
            raise ValueError("thickness should be between 0-20 km")


class SporadicELayer:
    """Time-varying sporadic-E layer model.

    Simulates realistic Es layer behavior including:
    - Critical frequency variations
    - Seasonal and diurnal occurrence probability
    - Patch-like spatial structure (simplified)
    """

    def __init__(self, config: Optional[SporadicEConfig] = None):
        """Initialize Es layer model.

        Args:
            config: Es layer configuration
        """
        self.config = config or SporadicEConfig()
        self._base_foEs = self.config.foEs_mhz
        self._current_foEs = self._base_foEs
        self._phase = 0.0
        self._rng = np.random.default_rng()

    @property
    def foEs(self) -> float:
        """Get current sporadic-E critical frequency."""
        return self._current_foEs if self.config.enabled else 0.0

    @property
    def hmEs(self) -> float:
        """Get sporadic-E layer height."""
        return self.config.hmEs_km

    @property
    def enabled(self) -> bool:
        """Check if Es layer is enabled."""
        return self.config.enabled

    def update(self, time_s: float):
        """Update Es layer state for given time.

        Applies time-varying fluctuations to foEs.

        Args:
            time_s: Simulation time in seconds
        """
        if not self.config.enabled:
            return

        # Sinusoidal variation with some randomness
        period = self.config.variability_period_s
        amplitude = self.config.variability_amplitude

        # Smooth sinusoidal component
        phase = 2 * np.pi * time_s / period
        sinusoidal = np.sin(phase)

        # Add some random flutter
        flutter = self._rng.normal(0, 0.05)

        # Compute current foEs
        variation = amplitude * (sinusoidal + flutter)
        self._current_foEs = self._base_foEs * (1 + variation)

        # Clamp to reasonable range
        self._current_foEs = max(0.5, min(15.0, self._current_foEs))

    def set_foEs(self, foEs_mhz: float):
        """Directly set foEs value.

        Args:
            foEs_mhz: New critical frequency in MHz
        """
        self._base_foEs = foEs_mhz
        self._current_foEs = foEs_mhz

    def enable(self, foEs_mhz: Optional[float] = None):
        """Enable Es layer.

        Args:
            foEs_mhz: Optional new foEs value
        """
        self.config.enabled = True
        if foEs_mhz is not None:
            self.set_foEs(foEs_mhz)

    def disable(self):
        """Disable Es layer."""
        self.config.enabled = False

    def inject(self, profile) -> "IonosphereProfile":
        """Inject Es layer into an ionosphere profile.

        Creates a new profile with the Es layer added.

        Args:
            profile: IonosphereProfile to modify

        Returns:
            New IonosphereProfile with Es layer
        """
        if not self.config.enabled:
            return profile

        from hfpathsim.core.raytracing.ionosphere import (
            IonosphereProfile,
            ne_from_plasma_frequency,
        )

        # Create copy of profile data
        new_ne = profile.electron_density.copy()

        # Add Es layer contribution
        ne_Es = ne_from_plasma_frequency(self._current_foEs)
        ym_Es = self.config.thickness_km

        for i, h in enumerate(profile.altitude_km):
            delta_h = h - self.config.hmEs_km
            if abs(delta_h) < ym_Es:
                # Parabolic layer shape
                contribution = ne_Es * (1 - (delta_h / ym_Es) ** 2)
                new_ne[i] += contribution

        # Create new profile with Es parameters
        return IonosphereProfile(
            altitude_km=profile.altitude_km.copy(),
            electron_density=new_ne,
            foF2=profile.foF2,
            hmF2=profile.hmF2,
            foE=profile.foE,
            hmE=profile.hmE,
            foEs=self._current_foEs,
            hmEs=self.config.hmEs_km,
            foF1=profile.foF1,
            hmF1=profile.hmF1,
            ym_F2=profile.ym_F2,
            ym_E=profile.ym_E,
        )

    def get_muf(self, path_km: float) -> float:
        """Calculate MUF for Es layer.

        Args:
            path_km: Path length in km

        Returns:
            Es MUF in MHz
        """
        if not self.config.enabled:
            return 0.0

        from hfpathsim.core.raytracing.geometry import sec_phi_spherical

        sec_phi = sec_phi_spherical(path_km, self.config.hmEs_km)
        return self._current_foEs * sec_phi


def estimate_es_occurrence(
    latitude: float,
    month: int,
    hour_utc: int,
) -> float:
    """Estimate sporadic-E occurrence probability.

    Based on climatological patterns:
    - Higher in summer months at mid-latitudes
    - Peaks in late morning/afternoon
    - Lower at high latitudes

    Args:
        latitude: Geographic latitude (degrees)
        month: Month (1-12)
        hour_utc: Hour of day (0-23)

    Returns:
        Occurrence probability (0-1)
    """
    abs_lat = abs(latitude)

    # Latitude factor - peak around 40-50 degrees
    if abs_lat < 20:
        lat_factor = 0.3
    elif abs_lat < 60:
        # Peak around 45 degrees
        lat_factor = 1.0 - abs(abs_lat - 45) / 30
    else:
        lat_factor = 0.2

    # Seasonal factor - summer peak (different for hemispheres)
    if latitude >= 0:
        # Northern hemisphere - June/July peak
        season_factor = 0.5 + 0.5 * np.cos(2 * np.pi * (month - 6.5) / 12)
    else:
        # Southern hemisphere - December/January peak
        season_factor = 0.5 + 0.5 * np.cos(2 * np.pi * (month - 0.5) / 12)

    # Diurnal factor - afternoon peak
    # Local time estimate (simplified)
    diurnal_factor = 0.5 + 0.5 * np.cos(2 * np.pi * (hour_utc - 14) / 24)

    # Combined probability
    probability = lat_factor * season_factor * (0.3 + 0.7 * diurnal_factor)

    return np.clip(probability, 0, 1)


def estimate_foEs(
    latitude: float,
    month: int,
    hour_utc: int,
    solar_flux: float = 100.0,
) -> float:
    """Estimate typical foEs when Es layer is present.

    Args:
        latitude: Geographic latitude (degrees)
        month: Month (1-12)
        hour_utc: Hour of day (0-23)
        solar_flux: F10.7 solar flux index

    Returns:
        Estimated foEs in MHz
    """
    # Base value around 5 MHz
    base_foEs = 5.0

    # Increase with latitude (up to a point)
    abs_lat = abs(latitude)
    if abs_lat < 60:
        lat_adjustment = 1.0 + 0.02 * abs_lat
    else:
        lat_adjustment = 1.0

    # Seasonal adjustment
    if latitude >= 0:
        season_adjustment = 1.0 + 0.3 * np.cos(2 * np.pi * (month - 6.5) / 12)
    else:
        season_adjustment = 1.0 + 0.3 * np.cos(2 * np.pi * (month - 0.5) / 12)

    # Solar flux adjustment (weak effect)
    flux_adjustment = 1.0 + 0.001 * (solar_flux - 100)

    foEs = base_foEs * lat_adjustment * season_adjustment * flux_adjustment

    return np.clip(foEs, 2.0, 15.0)


@dataclass
class SporadicEPreset:
    """Preset Es layer configurations for common scenarios."""

    name: str
    foEs_mhz: float
    hmEs_km: float
    description: str


# Common Es presets
ES_PRESETS = {
    "weak": SporadicEPreset(
        name="weak",
        foEs_mhz=3.0,
        hmEs_km=105.0,
        description="Weak Es, marginal propagation",
    ),
    "moderate": SporadicEPreset(
        name="moderate",
        foEs_mhz=6.0,
        hmEs_km=105.0,
        description="Moderate Es, good 6m propagation",
    ),
    "strong": SporadicEPreset(
        name="strong",
        foEs_mhz=10.0,
        hmEs_km=100.0,
        description="Strong Es, 2m propagation possible",
    ),
    "intense": SporadicEPreset(
        name="intense",
        foEs_mhz=15.0,
        hmEs_km=100.0,
        description="Intense Es, 2m DX conditions",
    ),
}


def create_es_from_preset(preset_name: str) -> SporadicEConfig:
    """Create Es configuration from preset name.

    Args:
        preset_name: One of "weak", "moderate", "strong", "intense"

    Returns:
        SporadicEConfig instance
    """
    preset = ES_PRESETS.get(preset_name.lower())
    if preset is None:
        raise ValueError(f"Unknown preset: {preset_name}")

    return SporadicEConfig(
        enabled=True,
        foEs_mhz=preset.foEs_mhz,
        hmEs_km=preset.hmEs_km,
    )
