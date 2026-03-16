"""Core channel simulation components."""

from .parameters import VoglerParameters, ITUCondition, PropagationMode
from .channel import HFChannel, RayTracingConfig
from .watterson import WattersonChannel, WattersonConfig, WattersonTap
from .vogler_hoffmeyer import (
    VoglerHoffmeyerChannel,
    VoglerHoffmeyerConfig,
    ModeParameters,
    CorrelationType,
    VOGLER_HOFFMEYER_PRESETS,
    get_vogler_hoffmeyer_preset,
    list_vogler_hoffmeyer_presets,
)
from .dispersion import DispersionModel, DispersionParameters, compute_d_from_qp
from .noise import NoiseGenerator, NoiseConfig, NoiseType
from .impairments import (
    AGC,
    AGCConfig,
    AGCMode,
    Limiter,
    LimiterConfig,
    FrequencyOffset,
    FrequencyOffsetConfig,
    ImpairmentChain,
)
from .recording import (
    ChannelRecorder,
    ChannelPlayer,
    ChannelSnapshot,
    RecordingMetadata,
)

# Ray tracing components
from .raytracing import (
    # Geometry
    EARTH_RADIUS_KM,
    great_circle_distance,
    sec_phi_spherical,
    compute_launch_angle,
    group_delay_ms,
    # Ionosphere
    IonosphereProfile,
    create_simple_profile,
    create_chapman_profile,
    # Ray engine
    RayPath,
    RayEngine,
    # Path finder
    PathFinder,
    PropagationModeResult,
    find_propagation_modes,
    estimate_muf,
)

__all__ = [
    # Channel models
    "VoglerParameters",
    "ITUCondition",
    "PropagationMode",
    "HFChannel",
    "RayTracingConfig",
    "WattersonChannel",
    "WattersonConfig",
    "WattersonTap",
    # Vogler-Hoffmeyer wideband stochastic model
    "VoglerHoffmeyerChannel",
    "VoglerHoffmeyerConfig",
    "ModeParameters",
    "CorrelationType",
    "VOGLER_HOFFMEYER_PRESETS",
    "get_vogler_hoffmeyer_preset",
    "list_vogler_hoffmeyer_presets",
    # Dispersion
    "DispersionModel",
    "DispersionParameters",
    "compute_d_from_qp",
    # Noise
    "NoiseGenerator",
    "NoiseConfig",
    "NoiseType",
    # Impairments
    "AGC",
    "AGCConfig",
    "AGCMode",
    "Limiter",
    "LimiterConfig",
    "FrequencyOffset",
    "FrequencyOffsetConfig",
    "ImpairmentChain",
    # Recording
    "ChannelRecorder",
    "ChannelPlayer",
    "ChannelSnapshot",
    "RecordingMetadata",
    # Ray tracing - Geometry
    "EARTH_RADIUS_KM",
    "great_circle_distance",
    "sec_phi_spherical",
    "compute_launch_angle",
    "group_delay_ms",
    # Ray tracing - Ionosphere
    "IonosphereProfile",
    "create_simple_profile",
    "create_chapman_profile",
    # Ray tracing - Engine
    "RayPath",
    "RayEngine",
    # Ray tracing - Path finder
    "PathFinder",
    "PropagationModeResult",
    "find_propagation_modes",
    "estimate_muf",
]
