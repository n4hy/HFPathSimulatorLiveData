"""Pydantic models for API request/response validation."""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# Enums

class ChannelModelType(str, Enum):
    """Channel model selection."""

    VOGLER = "vogler"
    WATTERSON = "watterson"
    VOGLER_HOFFMEYER = "vogler_hoffmeyer"
    PASSTHROUGH = "passthrough"


class ITUConditionType(str, Enum):
    """ITU-R F.1487 channel conditions."""

    QUIET = "quiet"
    MODERATE = "moderate"
    DISTURBED = "disturbed"
    FLUTTER = "flutter"


class AGCModeType(str, Enum):
    """AGC operating modes."""

    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"
    MANUAL = "manual"


class ManMadeEnvironmentType(str, Enum):
    """Man-made noise environment categories."""

    CITY = "city"
    RESIDENTIAL = "residential"
    RURAL = "rural"
    QUIET_RURAL = "quiet_rural"


# Request Models

class VoglerConfig(BaseModel):
    """Vogler channel configuration."""

    foF2: Optional[float] = Field(
        None, ge=1.0, le=20.0,
        description="F2 critical frequency (MHz)"
    )
    hmF2: Optional[float] = Field(
        None, ge=150.0, le=500.0,
        description="F2 peak height (km)"
    )
    foE: Optional[float] = Field(
        None, ge=0.5, le=10.0,
        description="E critical frequency (MHz)"
    )
    hmE: Optional[float] = Field(
        None, ge=80.0, le=150.0,
        description="E peak height (km)"
    )
    doppler_spread_hz: Optional[float] = Field(
        None, ge=0.0, le=20.0,
        description="Two-sided Doppler spread (Hz)"
    )
    delay_spread_ms: Optional[float] = Field(
        None, ge=0.0, le=10.0,
        description="Delay spread (ms)"
    )
    frequency_mhz: Optional[float] = Field(
        None, ge=2.0, le=30.0,
        description="Operating frequency (MHz)"
    )
    path_length_km: Optional[float] = Field(
        None, ge=0.0, le=20000.0,
        description="Path length (km)"
    )


class WattersonConfig(BaseModel):
    """Watterson channel configuration."""

    condition: ITUConditionType = Field(
        ITUConditionType.MODERATE,
        description="ITU-R F.1487 condition"
    )


class VoglerHoffmeyerConfig(BaseModel):
    """Vogler-Hoffmeyer channel configuration."""

    condition: ITUConditionType = Field(
        ITUConditionType.MODERATE,
        description="ITU-R F.1487 condition"
    )
    spread_f_enabled: Optional[bool] = Field(
        None,
        description="Enable spread-F ionospheric conditions"
    )
    spread_f_intensity: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Spread-F intensity (0-1)"
    )


class NoiseConfig(BaseModel):
    """Noise configuration."""

    snr_db: Optional[float] = Field(
        None, ge=-20.0, le=60.0,
        description="Signal-to-noise ratio (dB)"
    )
    enable_atmospheric: Optional[bool] = Field(
        None,
        description="Enable atmospheric noise"
    )
    enable_manmade: Optional[bool] = Field(
        None,
        description="Enable man-made noise"
    )
    environment: Optional[ManMadeEnvironmentType] = Field(
        None,
        description="Man-made noise environment"
    )
    enable_impulse: Optional[bool] = Field(
        None,
        description="Enable impulse noise"
    )
    impulse_rate_hz: Optional[float] = Field(
        None, ge=0.0, le=1000.0,
        description="Impulse rate (Hz)"
    )


class AGCConfigRequest(BaseModel):
    """AGC configuration request."""

    enabled: bool = Field(True, description="Enable AGC")
    mode: Optional[AGCModeType] = Field(None, description="AGC mode")
    target_level_db: Optional[float] = Field(
        None, ge=-40.0, le=0.0,
        description="Target output level (dBFS)"
    )
    max_gain_db: Optional[float] = Field(
        None, ge=0.0, le=80.0,
        description="Maximum gain (dB)"
    )
    min_gain_db: Optional[float] = Field(
        None, ge=-40.0, le=0.0,
        description="Minimum gain (dB)"
    )


class LimiterConfigRequest(BaseModel):
    """Limiter configuration request."""

    enabled: bool = Field(True, description="Enable limiter")
    threshold_db: Optional[float] = Field(
        None, ge=-20.0, le=0.0,
        description="Limiting threshold (dBFS)"
    )
    knee_db: Optional[float] = Field(
        None, ge=0.0, le=10.0,
        description="Soft knee width (dB)"
    )


class FreqOffsetConfigRequest(BaseModel):
    """Frequency offset configuration request."""

    enabled: bool = Field(True, description="Enable frequency offset")
    offset_hz: Optional[float] = Field(
        None, ge=-10000.0, le=10000.0,
        description="Frequency offset (Hz)"
    )
    drift_hz_per_sec: Optional[float] = Field(
        None, ge=-100.0, le=100.0,
        description="Drift rate (Hz/s)"
    )


class ProcessSamplesRequest(BaseModel):
    """Request to process samples."""

    samples_base64: str = Field(
        ...,
        description="Base64-encoded complex64 samples"
    )
    format: str = Field(
        "complex64",
        description="Sample format (complex64, complex128)"
    )


class SessionCreateRequest(BaseModel):
    """Session creation request."""

    channel_model: Optional[ChannelModelType] = Field(
        None,
        description="Channel model to use"
    )
    sample_rate_hz: Optional[float] = Field(
        None, ge=1000.0, le=50_000_000.0,
        description="Sample rate (Hz)"
    )
    use_gpu: Optional[bool] = Field(
        None,
        description="Enable GPU acceleration"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Session metadata"
    )


# Response Models

class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field("ok", description="Service status")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Uptime in seconds")


class GPUInfoResponse(BaseModel):
    """GPU information response."""

    available: bool = Field(..., description="GPU available")
    name: Optional[str] = Field(None, description="GPU name")
    compute_capability: Optional[str] = Field(
        None,
        description="CUDA compute capability"
    )
    total_memory_gb: Optional[float] = Field(
        None,
        description="Total GPU memory (GB)"
    )
    multiprocessors: Optional[int] = Field(
        None,
        description="Number of multiprocessors"
    )
    backend: Optional[str] = Field(
        None,
        description="GPU backend (cuda, cupy, cpu)"
    )


class ChannelStateResponse(BaseModel):
    """Channel state response."""

    model: ChannelModelType = Field(..., description="Active channel model")
    running: bool = Field(..., description="Processing running")
    total_samples_processed: int = Field(
        ...,
        description="Total samples processed"
    )
    blocks_processed: int = Field(..., description="Blocks processed")

    # Vogler parameters (if applicable)
    vogler: Optional[VoglerConfig] = Field(None, description="Vogler parameters")

    # Impairment states
    agc_enabled: bool = Field(False, description="AGC enabled")
    agc_gain_db: float = Field(0.0, description="Current AGC gain (dB)")
    limiter_enabled: bool = Field(False, description="Limiter enabled")
    limiter_reduction_db: float = Field(
        0.0,
        description="Current limiter reduction (dB)"
    )
    freq_offset_enabled: bool = Field(False, description="Freq offset enabled")
    current_freq_offset_hz: float = Field(
        0.0,
        description="Current frequency offset (Hz)"
    )


class ProcessSamplesResponse(BaseModel):
    """Response from sample processing."""

    samples_base64: str = Field(
        ...,
        description="Base64-encoded processed samples"
    )
    samples_count: int = Field(..., description="Number of samples")
    blocks_processed: int = Field(..., description="Total blocks processed")


class SessionResponse(BaseModel):
    """Session information response."""

    session_id: str = Field(..., description="Session ID")
    created_at: str = Field(..., description="Creation timestamp (ISO)")
    last_accessed: str = Field(..., description="Last access timestamp (ISO)")
    age_seconds: float = Field(..., description="Session age (seconds)")
    idle_seconds: float = Field(..., description="Idle time (seconds)")
    engine_running: bool = Field(..., description="Engine running")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Session metadata"
    )


class SessionListResponse(BaseModel):
    """Session list response."""

    sessions: List[SessionResponse] = Field(..., description="Active sessions")
    count: int = Field(..., description="Number of sessions")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    code: Optional[str] = Field(None, description="Error code")


# WebSocket Messages

class WSMessage(BaseModel):
    """Base WebSocket message."""

    type: str = Field(..., description="Message type")
    timestamp: float = Field(..., description="Unix timestamp")


class WSStateMessage(WSMessage):
    """WebSocket state update message."""

    type: str = Field("state", description="Message type")
    state: ChannelStateResponse = Field(..., description="Channel state")


class WSSpectrumMessage(WSMessage):
    """WebSocket spectrum data message."""

    type: str = Field("spectrum", description="Message type")
    spectrum_db: List[float] = Field(..., description="Spectrum in dB")
    freq_axis_hz: List[float] = Field(..., description="Frequency axis (Hz)")


class WSSamplesMessage(WSMessage):
    """WebSocket samples message."""

    type: str = Field("samples", description="Message type")
    samples_base64: str = Field(..., description="Base64-encoded samples")
    count: int = Field(..., description="Sample count")


class WSErrorMessage(WSMessage):
    """WebSocket error message."""

    type: str = Field("error", description="Message type")
    error: str = Field(..., description="Error message")
