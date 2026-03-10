"""GUI widgets for HF Path Simulator."""

from .channel_display import ChannelDisplayWidget
from .scattering import ScatteringWidget
from .spectrum import SpectrumWidget
from .parameters import ParameterPanel
from .input_config import InputConfigWidget

__all__ = [
    "ChannelDisplayWidget",
    "ScatteringWidget",
    "SpectrumWidget",
    "ParameterPanel",
    "InputConfigWidget",
]
