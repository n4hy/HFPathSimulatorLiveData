"""FastAPI REST API for HF Path Simulator.

Provides HTTP/JSON/WebSocket interface for remote control and streaming.
"""

from .app import app, create_app, run_server

__all__ = ["app", "create_app", "run_server"]
