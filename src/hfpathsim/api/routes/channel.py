"""Channel configuration routes."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from ..models import (
    VoglerConfig,
    WattersonConfig,
    VoglerHoffmeyerConfig,
    NoiseConfig,
    AGCConfigRequest,
    LimiterConfigRequest,
    FreqOffsetConfigRequest,
    ChannelStateResponse,
    ChannelModelType,
    ErrorResponse,
)
from ...engine import SimulationEngine, ChannelModel
from ...engine.session import Session, get_session_manager
from ...core.parameters import ITUCondition

router = APIRouter(prefix="/channel", tags=["channel"])


def get_engine(session_id: Optional[str] = None) -> SimulationEngine:
    """Get engine from session or create default."""
    if session_id:
        manager = get_session_manager()
        session = manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session.engine

    # Use global engine if no session
    from ..app import get_global_engine
    return get_global_engine()


@router.get(
    "/state",
    response_model=ChannelStateResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_channel_state(session_id: Optional[str] = None):
    """Get current channel state.

    Returns the active channel model, processing statistics,
    and impairment states.
    """
    engine = get_engine(session_id)
    state = engine.get_state()

    # Get Vogler parameters if applicable
    vogler_config = None
    if engine.config.channel_model == ChannelModel.VOGLER:
        params = engine._vogler_params
        vogler_config = VoglerConfig(
            foF2=params.foF2,
            hmF2=params.hmF2,
            foE=params.foE,
            hmE=params.hmE,
            doppler_spread_hz=params.doppler_spread_hz,
            delay_spread_ms=params.delay_spread_ms,
            frequency_mhz=params.frequency_mhz,
            path_length_km=params.path_length_km,
        )

    return ChannelStateResponse(
        model=ChannelModelType(engine.config.channel_model.value),
        running=state["running"],
        total_samples_processed=state["total_samples_processed"],
        blocks_processed=state["blocks_processed"],
        vogler=vogler_config,
        agc_enabled=engine.config.agc_enabled,
        agc_gain_db=state["agc_gain_db"],
        limiter_enabled=engine.config.limiter_enabled,
        limiter_reduction_db=state["limiter_reduction_db"],
        freq_offset_enabled=engine.config.freq_offset_enabled,
        current_freq_offset_hz=state["current_freq_offset_hz"],
    )


@router.post(
    "/vogler",
    response_model=ChannelStateResponse,
    responses={400: {"model": ErrorResponse}},
)
async def configure_vogler(
    config: VoglerConfig,
    session_id: Optional[str] = None,
):
    """Configure Vogler channel model.

    Sets ionospheric parameters for the Vogler IPM model.
    """
    engine = get_engine(session_id)

    # Switch to Vogler model if needed
    if engine.config.channel_model != ChannelModel.VOGLER:
        engine.configure({"channel_model": "vogler"})
        engine._init_components()

    # Apply configuration
    params = config.model_dump(exclude_none=True)
    if params:
        engine.configure({"vogler": params})

    return await get_channel_state(session_id)


@router.post(
    "/watterson",
    response_model=ChannelStateResponse,
    responses={400: {"model": ErrorResponse}},
)
async def configure_watterson(
    config: WattersonConfig,
    session_id: Optional[str] = None,
):
    """Configure Watterson channel model.

    Sets ITU-R F.1487 condition for tapped delay line model.
    """
    engine = get_engine(session_id)

    # Switch to Watterson model
    engine.config.channel_model = ChannelModel.WATTERSON

    # Map condition
    condition_map = {
        "quiet": ITUCondition.QUIET,
        "moderate": ITUCondition.MODERATE,
        "disturbed": ITUCondition.DISTURBED,
        "flutter": ITUCondition.FLUTTER,
    }
    condition = condition_map.get(config.condition.value, ITUCondition.MODERATE)
    engine.configure_watterson(condition)

    return await get_channel_state(session_id)


@router.post(
    "/vh",
    response_model=ChannelStateResponse,
    responses={400: {"model": ErrorResponse}},
)
async def configure_vogler_hoffmeyer(
    config: VoglerHoffmeyerConfig,
    session_id: Optional[str] = None,
):
    """Configure Vogler-Hoffmeyer wideband channel model.

    Sets parameters for the wideband stochastic model including
    optional spread-F conditions.
    """
    engine = get_engine(session_id)

    # Switch to VH model
    engine.config.channel_model = ChannelModel.VOGLER_HOFFMEYER
    engine._init_components()

    # Configure spread-F if applicable
    if engine._vh_channel and config.spread_f_enabled is not None:
        engine._vh_channel.config.spread_f_enabled = config.spread_f_enabled
        if config.spread_f_intensity is not None:
            engine._vh_channel.config.spread_f_intensity = config.spread_f_intensity

    return await get_channel_state(session_id)


@router.post(
    "/noise",
    response_model=ChannelStateResponse,
    responses={400: {"model": ErrorResponse}},
)
async def configure_noise(
    config: NoiseConfig,
    session_id: Optional[str] = None,
):
    """Configure noise generation.

    Sets AWGN, atmospheric, man-made, and impulse noise parameters.
    """
    engine = get_engine(session_id)

    params = config.model_dump(exclude_none=True)

    # Handle environment conversion
    if "environment" in params:
        params["environment"] = params["environment"].value

    engine.configure({"noise": params, "noise_enabled": True})

    return await get_channel_state(session_id)


@router.post(
    "/noise/disable",
    response_model=ChannelStateResponse,
)
async def disable_noise(session_id: Optional[str] = None):
    """Disable noise generation."""
    engine = get_engine(session_id)
    engine.configure({"noise_enabled": False})
    return await get_channel_state(session_id)


@router.post(
    "/impairments/agc",
    response_model=ChannelStateResponse,
    responses={400: {"model": ErrorResponse}},
)
async def configure_agc(
    config: AGCConfigRequest,
    session_id: Optional[str] = None,
):
    """Configure Automatic Gain Control.

    Sets AGC mode, target level, and gain limits.
    """
    engine = get_engine(session_id)

    params = {}
    if config.target_level_db is not None:
        params["target_level_db"] = config.target_level_db
    if config.max_gain_db is not None:
        params["max_gain_db"] = config.max_gain_db
    if config.min_gain_db is not None:
        params["min_gain_db"] = config.min_gain_db

    engine.configure({"agc": params, "agc_enabled": config.enabled})

    return await get_channel_state(session_id)


@router.post(
    "/impairments/limiter",
    response_model=ChannelStateResponse,
    responses={400: {"model": ErrorResponse}},
)
async def configure_limiter(
    config: LimiterConfigRequest,
    session_id: Optional[str] = None,
):
    """Configure signal limiter.

    Sets limiting threshold and knee width.
    """
    engine = get_engine(session_id)

    params = {}
    if config.threshold_db is not None:
        params["threshold_db"] = config.threshold_db

    engine.configure({"limiter": params, "limiter_enabled": config.enabled})

    return await get_channel_state(session_id)


@router.post(
    "/impairments/offset",
    response_model=ChannelStateResponse,
    responses={400: {"model": ErrorResponse}},
)
async def configure_freq_offset(
    config: FreqOffsetConfigRequest,
    session_id: Optional[str] = None,
):
    """Configure frequency offset.

    Sets fixed offset and drift rate.
    """
    engine = get_engine(session_id)

    params = {}
    if config.offset_hz is not None:
        params["offset_hz"] = config.offset_hz
    if config.drift_hz_per_sec is not None:
        params["drift_hz_per_sec"] = config.drift_hz_per_sec

    engine.configure({
        "freq_offset": params,
        "freq_offset_enabled": config.enabled,
    })

    return await get_channel_state(session_id)


@router.post(
    "/reset",
    response_model=ChannelStateResponse,
)
async def reset_channel(session_id: Optional[str] = None):
    """Reset channel to default state.

    Reinitializes all processing components.
    """
    engine = get_engine(session_id)
    engine.reset()
    return await get_channel_state(session_id)
