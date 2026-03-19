"""Processing and session routes."""

import base64
import numpy as np
from fastapi import APIRouter, HTTPException
from typing import Optional

from ..models import (
    ProcessSamplesRequest,
    ProcessSamplesResponse,
    SessionCreateRequest,
    SessionResponse,
    SessionListResponse,
    ChannelModelType,
    ErrorResponse,
)
from ...engine import SimulationEngine, EngineConfig, ChannelModel
from ...engine.session import get_session_manager

router = APIRouter(prefix="/processing", tags=["processing"])


@router.post(
    "/samples",
    response_model=ProcessSamplesResponse,
    responses={400: {"model": ErrorResponse}},
)
async def process_samples(
    request: ProcessSamplesRequest,
    session_id: Optional[str] = None,
):
    """Process a block of samples through the channel.

    Accepts base64-encoded complex samples, processes them through
    the configured channel model and impairments, and returns the
    processed samples.
    """
    # Get engine
    if session_id:
        manager = get_session_manager()
        session = manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        engine = session.engine
    else:
        from ..app import get_global_engine
        engine = get_global_engine()

    # Decode samples
    try:
        raw_bytes = base64.b64decode(request.samples_base64)

        if request.format == "complex64":
            samples = np.frombuffer(raw_bytes, dtype=np.complex64)
        elif request.format == "complex128":
            samples = np.frombuffer(raw_bytes, dtype=np.complex128)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {request.format}",
            )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode samples: {str(e)}",
        )

    # Process
    try:
        output = engine.process(samples)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}",
        )

    # Encode output
    output_bytes = output.astype(np.complex64).tobytes()
    output_base64 = base64.b64encode(output_bytes).decode("ascii")

    return ProcessSamplesResponse(
        samples_base64=output_base64,
        samples_count=len(output),
        blocks_processed=engine._state.blocks_processed,
    )


@router.post("/start")
async def start_processing(session_id: Optional[str] = None):
    """Start streaming processing.

    Requires configured input/output sources (via streaming endpoints).
    """
    if session_id:
        manager = get_session_manager()
        session = manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        engine = session.engine
    else:
        from ..app import get_global_engine
        engine = get_global_engine()

    if engine._state.running:
        raise HTTPException(status_code=400, detail="Processing already running")

    # Note: Actual streaming requires input source configuration
    # This endpoint is for future use with configured sources
    return {"status": "ok", "message": "Processing started"}


@router.post("/stop")
async def stop_processing(session_id: Optional[str] = None):
    """Stop streaming processing."""
    if session_id:
        manager = get_session_manager()
        session = manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        engine = session.engine
    else:
        from ..app import get_global_engine
        engine = get_global_engine()

    if engine._state.running:
        engine.stop_streaming()

    return {"status": "ok", "message": "Processing stopped"}


# Session management

@router.post(
    "/sessions",
    response_model=SessionResponse,
    responses={400: {"model": ErrorResponse}},
)
async def create_session(request: SessionCreateRequest):
    """Create a new simulation session.

    Creates an isolated simulation engine for multi-user scenarios.
    """
    manager = get_session_manager()

    # Build config
    config = EngineConfig()
    if request.channel_model:
        config.channel_model = ChannelModel(request.channel_model.value)
    if request.sample_rate_hz:
        config.sample_rate_hz = request.sample_rate_hz
    if request.use_gpu is not None:
        config.use_gpu = request.use_gpu

    try:
        session = manager.create_session(
            config=config,
            metadata=request.metadata,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return SessionResponse(
        session_id=session.session_id,
        created_at=session.created_at.isoformat(),
        last_accessed=session.last_accessed.isoformat(),
        age_seconds=session.age_seconds,
        idle_seconds=session.idle_seconds,
        engine_running=session.engine._state.running,
        metadata=session.metadata,
    )


@router.get(
    "/sessions",
    response_model=SessionListResponse,
)
async def list_sessions():
    """List all active sessions."""
    manager = get_session_manager()
    sessions = manager.list_sessions()

    return SessionListResponse(
        sessions=[
            SessionResponse(**s) for s in sessions
        ],
        count=len(sessions),
    )


@router.get(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_session(session_id: str):
    """Get session details."""
    manager = get_session_manager()
    session = manager.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionResponse(
        session_id=session.session_id,
        created_at=session.created_at.isoformat(),
        last_accessed=session.last_accessed.isoformat(),
        age_seconds=session.age_seconds,
        idle_seconds=session.idle_seconds,
        engine_running=session.engine._state.running,
        metadata=session.metadata,
    )


@router.delete(
    "/sessions/{session_id}",
    responses={404: {"model": ErrorResponse}},
)
async def delete_session(session_id: str):
    """Delete a session."""
    manager = get_session_manager()

    if not manager.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "ok", "message": "Session deleted"}
