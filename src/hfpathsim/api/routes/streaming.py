"""WebSocket streaming routes."""

import asyncio
import base64
import json
import time
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Optional, Dict, Set
import logging

from ...engine import SimulationEngine
from ...engine.session import get_session_manager

router = APIRouter(prefix="/stream", tags=["streaming"])
logger = logging.getLogger(__name__)


# Track active WebSocket connections
class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, channel: str):
        """Accept and register connection."""
        await websocket.accept()
        if channel not in self.active_connections:
            self.active_connections[channel] = set()
        self.active_connections[channel].add(websocket)

    def disconnect(self, websocket: WebSocket, channel: str):
        """Remove connection."""
        if channel in self.active_connections:
            self.active_connections[channel].discard(websocket)

    async def broadcast(self, channel: str, message: dict):
        """Broadcast to all connections on channel."""
        if channel not in self.active_connections:
            return

        dead_connections = set()
        for connection in self.active_connections[channel]:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.add(connection)

        # Clean up dead connections
        for conn in dead_connections:
            self.active_connections[channel].discard(conn)


manager = ConnectionManager()


def get_engine(session_id: Optional[str] = None) -> SimulationEngine:
    """Get engine from session or global."""
    if session_id:
        sm = get_session_manager()
        session = sm.get_session(session_id)
        if not session:
            raise ValueError("Session not found")
        return session.engine

    from ..app import get_global_engine
    return get_global_engine()


@router.websocket("/input")
async def stream_input(
    websocket: WebSocket,
    session_id: Optional[str] = None,
):
    """WebSocket endpoint for streaming input samples.

    Clients send base64-encoded complex64 samples.
    Server processes and returns processed samples.

    Message format (client → server):
    {
        "type": "samples",
        "samples_base64": "<base64 data>",
        "format": "complex64"
    }

    Response format (server → client):
    {
        "type": "processed",
        "samples_base64": "<base64 data>",
        "count": 4096,
        "timestamp": 1234567890.123
    }
    """
    try:
        engine = get_engine(session_id)
    except ValueError as e:
        await websocket.close(code=4004, reason=str(e))
        return

    channel = f"input:{session_id or 'global'}"
    await manager.connect(websocket, channel)

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()

            if data.get("type") == "samples":
                # Decode samples
                try:
                    raw_bytes = base64.b64decode(data["samples_base64"])
                    fmt = data.get("format", "complex64")

                    if fmt == "complex64":
                        samples = np.frombuffer(raw_bytes, dtype=np.complex64)
                    elif fmt == "complex128":
                        samples = np.frombuffer(raw_bytes, dtype=np.complex128)
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "error": f"Unsupported format: {fmt}",
                            "timestamp": time.time(),
                        })
                        continue

                    # Process
                    output = engine.process(samples)

                    # Encode and send
                    output_bytes = output.astype(np.complex64).tobytes()
                    output_base64 = base64.b64encode(output_bytes).decode("ascii")

                    await websocket.send_json({
                        "type": "processed",
                        "samples_base64": output_base64,
                        "count": len(output),
                        "timestamp": time.time(),
                    })

                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e),
                        "timestamp": time.time(),
                    })

            elif data.get("type") == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": time.time(),
                })

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from input stream")
    finally:
        manager.disconnect(websocket, channel)


@router.websocket("/output")
async def stream_output(
    websocket: WebSocket,
    session_id: Optional[str] = None,
):
    """WebSocket endpoint for receiving processed output.

    Server pushes processed samples when engine is running.
    Client receives base64-encoded samples.

    Message format (server → client):
    {
        "type": "samples",
        "samples_base64": "<base64 data>",
        "count": 4096,
        "timestamp": 1234567890.123
    }
    """
    try:
        engine = get_engine(session_id)
    except ValueError as e:
        await websocket.close(code=4004, reason=str(e))
        return

    channel = f"output:{session_id or 'global'}"
    await manager.connect(websocket, channel)

    # Track output queue
    output_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

    def on_output(samples: np.ndarray):
        """Callback for engine output."""
        try:
            output_queue.put_nowait(samples.copy())
        except asyncio.QueueFull:
            pass  # Drop if queue full

    engine.add_output_callback(on_output)

    try:
        while True:
            # Wait for output or timeout
            try:
                samples = await asyncio.wait_for(
                    output_queue.get(),
                    timeout=1.0,
                )

                # Encode and send
                output_bytes = samples.astype(np.complex64).tobytes()
                output_base64 = base64.b64encode(output_bytes).decode("ascii")

                await websocket.send_json({
                    "type": "samples",
                    "samples_base64": output_base64,
                    "count": len(samples),
                    "timestamp": time.time(),
                })

            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({
                    "type": "keepalive",
                    "timestamp": time.time(),
                })

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from output stream")
    finally:
        engine.remove_output_callback(on_output)
        manager.disconnect(websocket, channel)


@router.websocket("/state")
async def stream_state(
    websocket: WebSocket,
    session_id: Optional[str] = None,
    interval_ms: int = 100,
):
    """WebSocket endpoint for channel state updates.

    Periodically sends channel state and meter values.

    Message format (server → client):
    {
        "type": "state",
        "timestamp": 1234567890.123,
        "running": true,
        "blocks_processed": 1234,
        "total_samples_processed": 5062656,
        "agc_gain_db": -5.2,
        "limiter_reduction_db": 0.0,
        "current_freq_offset_hz": 100.5
    }
    """
    try:
        engine = get_engine(session_id)
    except ValueError as e:
        await websocket.close(code=4004, reason=str(e))
        return

    channel = f"state:{session_id or 'global'}"
    await manager.connect(websocket, channel)

    interval = max(10, min(interval_ms, 1000)) / 1000.0  # 10ms to 1s

    try:
        while True:
            # Get state
            state = engine.get_state()

            # Send update
            await websocket.send_json({
                "type": "state",
                "timestamp": time.time(),
                **state,
            })

            await asyncio.sleep(interval)

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from state stream")
    finally:
        manager.disconnect(websocket, channel)


@router.websocket("/spectrum")
async def stream_spectrum(
    websocket: WebSocket,
    session_id: Optional[str] = None,
    interval_ms: int = 100,
):
    """WebSocket endpoint for spectrum data.

    Sends spectrum magnitude in dB computed from recent samples.

    Message format (server → client):
    {
        "type": "spectrum",
        "timestamp": 1234567890.123,
        "spectrum_db": [-80.0, -75.2, ...],
        "freq_axis_hz": [-1000000, -999512, ...],
        "fft_size": 4096
    }
    """
    try:
        engine = get_engine(session_id)
    except ValueError as e:
        await websocket.close(code=4004, reason=str(e))
        return

    channel = f"spectrum:{session_id or 'global'}"
    await manager.connect(websocket, channel)

    interval = max(50, min(interval_ms, 1000)) / 1000.0  # 50ms to 1s

    # Sample buffer for spectrum
    sample_buffer: asyncio.Queue = asyncio.Queue(maxsize=10)

    def on_output(samples: np.ndarray):
        """Capture samples for spectrum."""
        try:
            sample_buffer.put_nowait(samples.copy())
        except asyncio.QueueFull:
            pass

    engine.add_output_callback(on_output)

    try:
        fft_size = engine.config.block_size
        sample_rate = engine.config.sample_rate_hz
        freq_axis = np.fft.fftshift(
            np.fft.fftfreq(fft_size, 1 / sample_rate)
        ).tolist()

        while True:
            # Get latest samples
            samples = None
            try:
                while True:
                    samples = sample_buffer.get_nowait()
            except asyncio.QueueEmpty:
                pass

            if samples is not None and len(samples) >= fft_size:
                # Compute spectrum
                spectrum = np.fft.fftshift(np.fft.fft(samples[:fft_size]))
                spectrum_db = 20 * np.log10(
                    np.abs(spectrum) + 1e-10
                ).tolist()

                await websocket.send_json({
                    "type": "spectrum",
                    "timestamp": time.time(),
                    "spectrum_db": spectrum_db,
                    "freq_axis_hz": freq_axis,
                    "fft_size": fft_size,
                })
            else:
                # Send keepalive if no samples
                await websocket.send_json({
                    "type": "keepalive",
                    "timestamp": time.time(),
                })

            await asyncio.sleep(interval)

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from spectrum stream")
    finally:
        engine.remove_output_callback(on_output)
        manager.disconnect(websocket, channel)
