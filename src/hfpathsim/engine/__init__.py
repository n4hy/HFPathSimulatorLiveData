"""Headless simulation engine for server/container deployment.

This module provides a GUI-independent API for HF channel simulation,
enabling deployment in servers, containers, and REST API backends.
"""

from .simulation_engine import SimulationEngine, EngineConfig, ChannelModel
from .session import SessionManager, Session

__all__ = [
    "SimulationEngine",
    "EngineConfig",
    "ChannelModel",
    "SessionManager",
    "Session",
]
