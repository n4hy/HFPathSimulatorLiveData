"""Output data sinks for HF Path Simulator."""

from .base import OutputSink, OutputFormat
from .file import FileOutputSink
from .network import NetworkOutputSink, NetworkProtocol
from .audio import AudioOutputSink
from .sdr import SDROutputSink
from .multiplex import MultiplexOutputSink, TeeOutputSink

__all__ = [
    "OutputSink",
    "OutputFormat",
    "FileOutputSink",
    "NetworkOutputSink",
    "NetworkProtocol",
    "AudioOutputSink",
    "SDROutputSink",
    "MultiplexOutputSink",
    "TeeOutputSink",
]
