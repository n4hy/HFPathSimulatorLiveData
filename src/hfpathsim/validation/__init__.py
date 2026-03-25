"""Real-world validation module for HF channel simulator.

This module provides tools for validating the HF channel simulator against
measured data from actual ionospheric propagation campaigns.

Reference sources:
- NTIA TR-88-240: Vogler-Hoffmeyer deterministic model measurements
- NTIA TR-90-255: Wideband HF stochastic model validation
- ITU-R F.1487: Standard HF modem testing parameters
- Watterson et al. 1970: Original HF channel model validation

Validation capabilities:
- Delay spread comparison
- Doppler spread comparison
- Scattering function correlation
- Fading statistics (Rayleigh fit, level crossing, fade duration)
- Validation report generation
"""

from .reference_data import (
    # Reference datasets
    ReferenceDataset,
    NTIAMeasurement,
    ITUReference,
    # Preset references
    NTIA_MIDLATITUDE_QUIET,
    NTIA_MIDLATITUDE_DISTURBED,
    NTIA_AURORAL,
    NTIA_SPREAD_F,
    ITU_F1487_QUIET,
    ITU_F1487_MODERATE,
    ITU_F1487_DISTURBED,
    ITU_F1487_FLUTTER,
    WATTERSON_1970_GOOD,
    WATTERSON_1970_MODERATE,
    WATTERSON_1970_POOR,
    # Utilities
    get_reference_dataset,
    list_reference_datasets,
)

from .statistics import (
    # Delay/Doppler analysis
    compute_delay_spread,
    compute_doppler_spread,
    compute_coherence_bandwidth,
    compute_coherence_time,
    # Scattering function
    compute_scattering_function,
    compare_scattering_functions,
    # Fading statistics
    compute_fading_statistics,
    rayleigh_fit_test,
    compute_level_crossing_rate,
    compute_average_fade_duration,
    compute_fade_depth,
)

from .validator import (
    ChannelValidator,
    ValidationResult,
    ValidationReport,
    validate_channel,
    generate_validation_report,
)

__all__ = [
    # Reference data
    "ReferenceDataset",
    "NTIAMeasurement",
    "ITUReference",
    "NTIA_MIDLATITUDE_QUIET",
    "NTIA_MIDLATITUDE_DISTURBED",
    "NTIA_AURORAL",
    "NTIA_SPREAD_F",
    "ITU_F1487_QUIET",
    "ITU_F1487_MODERATE",
    "ITU_F1487_DISTURBED",
    "ITU_F1487_FLUTTER",
    "WATTERSON_1970_GOOD",
    "WATTERSON_1970_MODERATE",
    "WATTERSON_1970_POOR",
    "get_reference_dataset",
    "list_reference_datasets",
    # Statistics
    "compute_delay_spread",
    "compute_doppler_spread",
    "compute_coherence_bandwidth",
    "compute_coherence_time",
    "compute_scattering_function",
    "compare_scattering_functions",
    "compute_fading_statistics",
    "rayleigh_fit_test",
    "compute_level_crossing_rate",
    "compute_average_fade_duration",
    "compute_fade_depth",
    # Validator
    "ChannelValidator",
    "ValidationResult",
    "ValidationReport",
    "validate_channel",
    "generate_validation_report",
]
