"""GUI widgets for HF Path Simulator."""

from .channel_display import ChannelDisplayWidget
from .scattering import ScatteringWidget
from .spectrum import SpectrumWidget
from .parameters import ParameterPanel
from .input_config import InputConfigWidget
from .control_tabs import ControlTabWidget
from .channel_panel import ChannelPanel, TapWidget
from .noise_panel import NoisePanel
from .impairments_panel import ImpairmentsPanel, AGCMeter, GainReductionMeter
from .ionosphere_panel import IonospherePanel
from .recording_panel import RecordingPanel

__all__ = [
    # Display widgets
    "ChannelDisplayWidget",
    "ScatteringWidget",
    "SpectrumWidget",
    # Legacy control widgets
    "ParameterPanel",
    "InputConfigWidget",
    # New tabbed control system
    "ControlTabWidget",
    "ChannelPanel",
    "TapWidget",
    "NoisePanel",
    "ImpairmentsPanel",
    "AGCMeter",
    "GainReductionMeter",
    "IonospherePanel",
    "RecordingPanel",
]
