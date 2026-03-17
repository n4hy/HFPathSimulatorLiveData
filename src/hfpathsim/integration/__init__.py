"""Integration modules for HF Path Simulator.

Provides integration with external tools:
- GNU Radio via ZMQ bridge
- MATLAB via MAT-file I/O and optional Engine API
"""

from .gnuradio_zmq import GNURadioZMQBridge, create_gr_flowgraph_snippet
from .matlab_interface import MATFileInterface

__all__ = [
    "GNURadioZMQBridge",
    "create_gr_flowgraph_snippet",
    "MATFileInterface",
]
