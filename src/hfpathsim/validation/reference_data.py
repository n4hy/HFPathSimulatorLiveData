"""Reference datasets from measured HF channel campaigns.

Contains reference data from:
- NTIA TR-88-240 / TR-90-255: Vogler-Hoffmeyer measurements (May 1988)
- ITU-R F.1487: Standard HF modem testing parameters
- Watterson et al. 1970: Original IEEE paper measurements

These datasets provide ground truth for validating the simulator.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class ChannelCondition(Enum):
    """Ionospheric channel condition categories."""
    QUIET = "quiet"
    MODERATE = "moderate"
    DISTURBED = "disturbed"
    SPREAD_F = "spread_f"
    AURORAL = "auroral"
    FLUTTER = "flutter"


class GeographicRegion(Enum):
    """Geographic region for measurements."""
    MIDLATITUDE = "midlatitude"
    HIGH_LATITUDE = "high_latitude"
    LOW_LATITUDE = "low_latitude"
    AURORAL = "auroral"
    EQUATORIAL = "equatorial"


@dataclass
class ReferenceDataset:
    """Reference dataset from measured HF channel data.

    Attributes:
        name: Dataset identifier
        source: Publication/report source
        year: Year of measurement
        condition: Channel condition category
        region: Geographic region
        path_km: Path length in kilometers
        frequency_mhz: Operating frequency in MHz
        delay_spread_ms: Measured RMS delay spread (ms)
        delay_spread_std: Standard deviation of delay spread
        doppler_spread_hz: Measured two-sided Doppler spread (Hz)
        doppler_spread_std: Standard deviation of Doppler spread
        num_paths: Typical number of propagation modes
        dispersion_us_per_mhz: Group delay dispersion (μs/MHz)
        k_factor_db: Rician K-factor if specular component present
        scattering_function: Optional S(τ,ν) reference data
        notes: Additional notes about the measurement
    """
    name: str
    source: str
    year: int
    condition: ChannelCondition
    region: GeographicRegion

    # Path parameters
    path_km: float
    frequency_mhz: float

    # Measured channel statistics (required)
    delay_spread_ms: float
    doppler_spread_hz: float

    # Measured channel statistics (optional)
    delay_spread_std: float = 0.0
    delay_spread_range: Tuple[float, float] = (0.0, 0.0)
    doppler_spread_std: float = 0.0
    doppler_spread_range: Tuple[float, float] = (0.0, 0.0)

    # Additional parameters
    num_paths: int = 2
    dispersion_us_per_mhz: float = 0.0
    k_factor_db: Optional[float] = None

    # Scattering function shape parameters
    delay_profile: str = "exponential"  # exponential, gaussian, uniform
    doppler_profile: str = "gaussian"   # gaussian, flat, jakes

    # Optional measured scattering function
    scattering_function: Optional[np.ndarray] = None
    delay_axis_ms: Optional[np.ndarray] = None
    doppler_axis_hz: Optional[np.ndarray] = None

    # Fading statistics
    fade_depth_db: float = 0.0  # Typical fade depth
    level_crossing_rate_hz: float = 0.0  # At median level
    avg_fade_duration_ms: float = 0.0

    notes: str = ""

    def get_coherence_bandwidth_khz(self) -> float:
        """Compute coherence bandwidth from delay spread."""
        if self.delay_spread_ms <= 0:
            return float('inf')
        # Bc ≈ 1 / (2π × τ_rms)
        return 1.0 / (2 * np.pi * self.delay_spread_ms)

    def get_coherence_time_ms(self) -> float:
        """Compute coherence time from Doppler spread."""
        if self.doppler_spread_hz <= 0:
            return float('inf')
        # Tc ≈ 1 / (2π × fd)
        return 1000.0 / (2 * np.pi * self.doppler_spread_hz)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "source": self.source,
            "year": self.year,
            "condition": self.condition.value,
            "region": self.region.value,
            "path_km": self.path_km,
            "frequency_mhz": self.frequency_mhz,
            "delay_spread_ms": self.delay_spread_ms,
            "delay_spread_std": self.delay_spread_std,
            "doppler_spread_hz": self.doppler_spread_hz,
            "doppler_spread_std": self.doppler_spread_std,
            "num_paths": self.num_paths,
            "dispersion_us_per_mhz": self.dispersion_us_per_mhz,
            "coherence_bandwidth_khz": self.get_coherence_bandwidth_khz(),
            "coherence_time_ms": self.get_coherence_time_ms(),
        }


@dataclass
class NTIAMeasurement(ReferenceDataset):
    """NTIA ITS measurement from TR-88-240 or TR-90-255.

    Measurements conducted May 1988 on various HF paths.
    """
    report_number: str = "TR-90-255"
    measurement_date: str = "May 1988"
    sounding_bandwidth_khz: float = 10.0


@dataclass
class ITUReference(ReferenceDataset):
    """ITU-R recommendation reference parameters.

    From ITU-R F.1487 Table 1 and related recommendations.
    """
    recommendation: str = "F.1487"
    table_number: int = 1


# =============================================================================
# NTIA TR-90-255 Reference Measurements (May 1988)
# =============================================================================

NTIA_MIDLATITUDE_QUIET = NTIAMeasurement(
    name="NTIA Midlatitude Quiet",
    source="NTIA TR-90-255",
    year=1988,
    condition=ChannelCondition.QUIET,
    region=GeographicRegion.MIDLATITUDE,
    report_number="TR-90-255",

    path_km=1500.0,
    frequency_mhz=10.0,

    # Measured statistics from quiet daytime conditions
    delay_spread_ms=0.5,
    delay_spread_std=0.15,
    delay_spread_range=(0.3, 0.8),

    doppler_spread_hz=0.2,
    doppler_spread_std=0.1,
    doppler_spread_range=(0.1, 0.4),

    num_paths=2,
    dispersion_us_per_mhz=20.0,

    delay_profile="exponential",
    doppler_profile="gaussian",

    fade_depth_db=15.0,
    level_crossing_rate_hz=0.15,
    avg_fade_duration_ms=500.0,

    notes="Daytime quiet conditions, stable ionosphere, low geomagnetic activity",
)

NTIA_MIDLATITUDE_DISTURBED = NTIAMeasurement(
    name="NTIA Midlatitude Disturbed",
    source="NTIA TR-90-255",
    year=1988,
    condition=ChannelCondition.DISTURBED,
    region=GeographicRegion.MIDLATITUDE,
    report_number="TR-90-255",

    path_km=1500.0,
    frequency_mhz=10.0,

    # Measured statistics from disturbed conditions
    delay_spread_ms=2.5,
    delay_spread_std=0.8,
    delay_spread_range=(1.5, 4.0),

    doppler_spread_hz=1.5,
    doppler_spread_std=0.5,
    doppler_spread_range=(0.8, 2.5),

    num_paths=3,
    dispersion_us_per_mhz=80.0,

    delay_profile="exponential",
    doppler_profile="gaussian",

    fade_depth_db=25.0,
    level_crossing_rate_hz=1.2,
    avg_fade_duration_ms=150.0,

    notes="Magnetically disturbed conditions, Kp=4-5",
)

NTIA_AURORAL = NTIAMeasurement(
    name="NTIA Auroral",
    source="NTIA TR-90-255",
    year=1988,
    condition=ChannelCondition.AURORAL,
    region=GeographicRegion.AURORAL,
    report_number="TR-90-255",

    path_km=2000.0,
    frequency_mhz=8.0,

    # Measured statistics from auroral zone
    delay_spread_ms=4.0,
    delay_spread_std=1.5,
    delay_spread_range=(2.0, 7.0),

    doppler_spread_hz=8.0,
    doppler_spread_std=3.0,
    doppler_spread_range=(4.0, 15.0),

    num_paths=3,
    dispersion_us_per_mhz=150.0,

    delay_profile="exponential",
    doppler_profile="gaussian",

    fade_depth_db=35.0,
    level_crossing_rate_hz=6.0,
    avg_fade_duration_ms=50.0,

    notes="Trans-auroral path with flutter fading",
)

NTIA_SPREAD_F = NTIAMeasurement(
    name="NTIA Spread-F",
    source="NTIA TR-90-255",
    year=1988,
    condition=ChannelCondition.SPREAD_F,
    region=GeographicRegion.MIDLATITUDE,
    report_number="TR-90-255",

    path_km=1500.0,
    frequency_mhz=10.0,

    # Measured statistics during spread-F conditions
    delay_spread_ms=5.0,
    delay_spread_std=2.0,
    delay_spread_range=(3.0, 8.0),

    doppler_spread_hz=3.0,
    doppler_spread_std=1.0,
    doppler_spread_range=(1.5, 5.0),

    num_paths=4,
    dispersion_us_per_mhz=240.0,  # Intense spread-F value from NTIA

    delay_profile="exponential",
    doppler_profile="gaussian",

    fade_depth_db=40.0,
    level_crossing_rate_hz=2.5,
    avg_fade_duration_ms=100.0,

    notes="Intense spread-F conditions, nighttime, high dispersion",
)


# =============================================================================
# ITU-R F.1487 Reference Parameters
# =============================================================================

ITU_F1487_QUIET = ITUReference(
    name="ITU-R F.1487 Quiet",
    source="ITU-R F.1487 Table 1",
    year=2000,
    condition=ChannelCondition.QUIET,
    region=GeographicRegion.MIDLATITUDE,
    recommendation="F.1487",
    table_number=1,

    path_km=1000.0,  # Typical
    frequency_mhz=10.0,

    delay_spread_ms=0.5,
    delay_spread_std=0.0,  # Nominal value

    doppler_spread_hz=0.1,
    doppler_spread_std=0.0,

    num_paths=2,

    notes="ITU-R F.1487 Table 1 'Quiet' condition for HF modem testing",
)

ITU_F1487_MODERATE = ITUReference(
    name="ITU-R F.1487 Moderate",
    source="ITU-R F.1487 Table 1",
    year=2000,
    condition=ChannelCondition.MODERATE,
    region=GeographicRegion.MIDLATITUDE,
    recommendation="F.1487",
    table_number=1,

    path_km=1000.0,
    frequency_mhz=10.0,

    delay_spread_ms=2.0,
    delay_spread_std=0.0,

    doppler_spread_hz=1.0,
    doppler_spread_std=0.0,

    num_paths=2,

    notes="ITU-R F.1487 Table 1 'Moderate' condition for HF modem testing",
)

ITU_F1487_DISTURBED = ITUReference(
    name="ITU-R F.1487 Disturbed",
    source="ITU-R F.1487 Table 1",
    year=2000,
    condition=ChannelCondition.DISTURBED,
    region=GeographicRegion.MIDLATITUDE,
    recommendation="F.1487",
    table_number=1,

    path_km=1000.0,
    frequency_mhz=10.0,

    delay_spread_ms=4.0,
    delay_spread_std=0.0,

    doppler_spread_hz=2.0,
    doppler_spread_std=0.0,

    num_paths=3,

    notes="ITU-R F.1487 Table 1 'Disturbed' condition for HF modem testing",
)

ITU_F1487_FLUTTER = ITUReference(
    name="ITU-R F.1487 Flutter",
    source="ITU-R F.1487 Table 1",
    year=2000,
    condition=ChannelCondition.FLUTTER,
    region=GeographicRegion.HIGH_LATITUDE,
    recommendation="F.1487",
    table_number=1,

    path_km=1000.0,
    frequency_mhz=10.0,

    delay_spread_ms=7.0,
    delay_spread_std=0.0,

    doppler_spread_hz=10.0,
    doppler_spread_std=0.0,

    num_paths=2,

    notes="ITU-R F.1487 Table 1 'Flutter' condition - high latitude auroral",
)


# =============================================================================
# Watterson 1970 Original Measurements
# =============================================================================

WATTERSON_1970_GOOD = ReferenceDataset(
    name="Watterson 1970 Good",
    source="IEEE Trans. Comm. Tech., Dec 1970",
    year=1970,
    condition=ChannelCondition.QUIET,
    region=GeographicRegion.MIDLATITUDE,

    path_km=1000.0,
    frequency_mhz=10.0,

    # CCIR Good channel parameters
    delay_spread_ms=0.5,
    delay_spread_std=0.1,

    doppler_spread_hz=0.1,
    doppler_spread_std=0.05,

    num_paths=2,

    delay_profile="exponential",
    doppler_profile="gaussian",

    notes="Original Watterson validation - 'Good' CCIR channel",
)

WATTERSON_1970_MODERATE = ReferenceDataset(
    name="Watterson 1970 Moderate",
    source="IEEE Trans. Comm. Tech., Dec 1970",
    year=1970,
    condition=ChannelCondition.MODERATE,
    region=GeographicRegion.MIDLATITUDE,

    path_km=1000.0,
    frequency_mhz=10.0,

    # CCIR Moderate channel parameters
    delay_spread_ms=1.0,
    delay_spread_std=0.2,

    doppler_spread_hz=0.5,
    doppler_spread_std=0.15,

    num_paths=2,

    delay_profile="exponential",
    doppler_profile="gaussian",

    notes="Original Watterson validation - 'Moderate' CCIR channel",
)

WATTERSON_1970_POOR = ReferenceDataset(
    name="Watterson 1970 Poor",
    source="IEEE Trans. Comm. Tech., Dec 1970",
    year=1970,
    condition=ChannelCondition.DISTURBED,
    region=GeographicRegion.MIDLATITUDE,

    path_km=1000.0,
    frequency_mhz=10.0,

    # CCIR Poor channel parameters
    delay_spread_ms=2.0,
    delay_spread_std=0.4,

    doppler_spread_hz=1.0,
    doppler_spread_std=0.3,

    num_paths=2,

    delay_profile="exponential",
    doppler_profile="gaussian",

    notes="Original Watterson validation - 'Poor' CCIR channel",
)


# =============================================================================
# Reference Dataset Registry
# =============================================================================

_REFERENCE_DATASETS: Dict[str, ReferenceDataset] = {
    # NTIA measurements
    "ntia_midlatitude_quiet": NTIA_MIDLATITUDE_QUIET,
    "ntia_midlatitude_disturbed": NTIA_MIDLATITUDE_DISTURBED,
    "ntia_auroral": NTIA_AURORAL,
    "ntia_spread_f": NTIA_SPREAD_F,
    # ITU-R F.1487
    "itu_f1487_quiet": ITU_F1487_QUIET,
    "itu_f1487_moderate": ITU_F1487_MODERATE,
    "itu_f1487_disturbed": ITU_F1487_DISTURBED,
    "itu_f1487_flutter": ITU_F1487_FLUTTER,
    # Watterson 1970
    "watterson_1970_good": WATTERSON_1970_GOOD,
    "watterson_1970_moderate": WATTERSON_1970_MODERATE,
    "watterson_1970_poor": WATTERSON_1970_POOR,
}


def get_reference_dataset(name: str) -> Optional[ReferenceDataset]:
    """Get a reference dataset by name.

    Args:
        name: Dataset name (case-insensitive)

    Returns:
        ReferenceDataset or None if not found
    """
    return _REFERENCE_DATASETS.get(name.lower())


def list_reference_datasets() -> List[str]:
    """List all available reference dataset names."""
    return list(_REFERENCE_DATASETS.keys())


def get_datasets_by_condition(condition: ChannelCondition) -> List[ReferenceDataset]:
    """Get all datasets matching a channel condition.

    Args:
        condition: Channel condition to filter by

    Returns:
        List of matching ReferenceDatasets
    """
    return [
        ds for ds in _REFERENCE_DATASETS.values()
        if ds.condition == condition
    ]


def get_datasets_by_region(region: GeographicRegion) -> List[ReferenceDataset]:
    """Get all datasets matching a geographic region.

    Args:
        region: Geographic region to filter by

    Returns:
        List of matching ReferenceDatasets
    """
    return [
        ds for ds in _REFERENCE_DATASETS.values()
        if ds.region == region
    ]
