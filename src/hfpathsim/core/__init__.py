"""Core channel simulation components."""

from .parameters import VoglerParameters, ITUCondition
from .channel import HFChannel
from .watterson import WattersonChannel, WattersonConfig, WattersonTap
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

__all__ = [
    # Channel models
    "VoglerParameters",
    "ITUCondition",
    "HFChannel",
    "WattersonChannel",
    "WattersonConfig",
    "WattersonTap",
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
]
