"""Tabbed control container widget for all control panels."""

from PyQt6.QtWidgets import QTabWidget, QWidget
from PyQt6.QtCore import pyqtSignal

from hfpathsim.core.parameters import VoglerParameters
from hfpathsim.core.watterson import WattersonConfig
from hfpathsim.core.vogler_hoffmeyer import VoglerHoffmeyerConfig
from hfpathsim.core.noise import NoiseConfig
from hfpathsim.core.impairments import AGCConfig, LimiterConfig, FrequencyOffsetConfig
from hfpathsim.core.raytracing.ionosphere import IonosphereProfile
from hfpathsim.iono.sporadic_e import SporadicEConfig
from hfpathsim.iono.geomagnetic import GeomagneticIndices


class ControlTabWidget(QTabWidget):
    """Tabbed container holding all control panels.

    Tabs:
    - Input: Input source configuration (file/network/SDR)
    - Channel: Vogler + Watterson channel model controls
    - Noise: AWGN, atmospheric, man-made, impulse noise
    - Impairments: AGC, limiter, frequency offset
    - Ionosphere: Ray tracing, sporadic-E, geomagnetic, GIRO/IRI
    - Recording: Channel state recording and playback
    """

    # Forward signals from child panels
    parameters_changed = pyqtSignal(VoglerParameters)
    watterson_config_changed = pyqtSignal(WattersonConfig)
    vogler_hoffmeyer_config_changed = pyqtSignal(VoglerHoffmeyerConfig)
    model_changed = pyqtSignal(str)  # "vogler", "watterson", or "vogler_hoffmeyer"
    noise_config_changed = pyqtSignal(NoiseConfig)
    agc_config_changed = pyqtSignal(AGCConfig)
    limiter_config_changed = pyqtSignal(LimiterConfig)
    freq_offset_config_changed = pyqtSignal(FrequencyOffsetConfig)
    ionosphere_profile_changed = pyqtSignal(IonosphereProfile)
    sporadic_e_changed = pyqtSignal(SporadicEConfig)
    geomagnetic_changed = pyqtSignal(GeomagneticIndices)
    ray_tracing_requested = pyqtSignal(dict)  # TX/RX coords, frequency
    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal(str)  # filename
    playback_started = pyqtSignal(str)  # filename
    playback_stopped = pyqtSignal()

    # Input source signals
    source_changed = pyqtSignal(object)  # InputSource
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Child panel references (set by setup_panels)
        self._input_panel = None
        self._channel_panel = None
        self._noise_panel = None
        self._impairments_panel = None
        self._ionosphere_panel = None
        self._recording_panel = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup the tabbed interface."""
        self.setTabPosition(QTabWidget.TabPosition.North)
        self.setDocumentMode(True)

    def setup_panels(
        self,
        input_panel: QWidget,
        channel_panel: QWidget,
        noise_panel: QWidget,
        impairments_panel: QWidget,
        ionosphere_panel: QWidget,
        recording_panel: QWidget,
    ):
        """Add all control panels to tabs and connect signals.

        Args:
            input_panel: InputConfigWidget for input source configuration
            channel_panel: ChannelPanel for Vogler/Watterson controls
            noise_panel: NoisePanel for noise configuration
            impairments_panel: ImpairmentsPanel for AGC/limiter/freq offset
            ionosphere_panel: IonospherePanel for ray tracing/Es/geomag
            recording_panel: RecordingPanel for record/playback
        """
        self._input_panel = input_panel
        self._channel_panel = channel_panel
        self._noise_panel = noise_panel
        self._impairments_panel = impairments_panel
        self._ionosphere_panel = ionosphere_panel
        self._recording_panel = recording_panel

        # Add tabs
        self.addTab(input_panel, "Input")
        self.addTab(channel_panel, "Channel")
        self.addTab(noise_panel, "Noise")
        self.addTab(impairments_panel, "Impairments")
        self.addTab(ionosphere_panel, "Ionosphere")
        self.addTab(recording_panel, "Recording")

        # Connect signals from child panels
        self._connect_panel_signals()

    def _connect_panel_signals(self):
        """Connect signals from child panels to forwarded signals."""
        # Input panel signals
        if hasattr(self._input_panel, 'source_changed'):
            self._input_panel.source_changed.connect(self.source_changed)
        if hasattr(self._input_panel, 'start_requested'):
            self._input_panel.start_requested.connect(self.start_requested)
        if hasattr(self._input_panel, 'stop_requested'):
            self._input_panel.stop_requested.connect(self.stop_requested)

        # Channel panel signals
        if hasattr(self._channel_panel, 'parameters_changed'):
            self._channel_panel.parameters_changed.connect(self.parameters_changed)
        if hasattr(self._channel_panel, 'watterson_config_changed'):
            self._channel_panel.watterson_config_changed.connect(self.watterson_config_changed)
        if hasattr(self._channel_panel, 'vogler_hoffmeyer_config_changed'):
            self._channel_panel.vogler_hoffmeyer_config_changed.connect(self.vogler_hoffmeyer_config_changed)
        if hasattr(self._channel_panel, 'model_changed'):
            self._channel_panel.model_changed.connect(self.model_changed)

        # Noise panel signals
        if hasattr(self._noise_panel, 'noise_config_changed'):
            self._noise_panel.noise_config_changed.connect(self.noise_config_changed)

        # Impairments panel signals
        if hasattr(self._impairments_panel, 'agc_config_changed'):
            self._impairments_panel.agc_config_changed.connect(self.agc_config_changed)
        if hasattr(self._impairments_panel, 'limiter_config_changed'):
            self._impairments_panel.limiter_config_changed.connect(self.limiter_config_changed)
        if hasattr(self._impairments_panel, 'freq_offset_config_changed'):
            self._impairments_panel.freq_offset_config_changed.connect(self.freq_offset_config_changed)

        # Ionosphere panel signals
        if hasattr(self._ionosphere_panel, 'ionosphere_profile_changed'):
            self._ionosphere_panel.ionosphere_profile_changed.connect(self.ionosphere_profile_changed)
        if hasattr(self._ionosphere_panel, 'sporadic_e_changed'):
            self._ionosphere_panel.sporadic_e_changed.connect(self.sporadic_e_changed)
        if hasattr(self._ionosphere_panel, 'geomagnetic_changed'):
            self._ionosphere_panel.geomagnetic_changed.connect(self.geomagnetic_changed)
        if hasattr(self._ionosphere_panel, 'ray_tracing_requested'):
            self._ionosphere_panel.ray_tracing_requested.connect(self.ray_tracing_requested)

        # Recording panel signals
        if hasattr(self._recording_panel, 'recording_started'):
            self._recording_panel.recording_started.connect(self.recording_started)
        if hasattr(self._recording_panel, 'recording_stopped'):
            self._recording_panel.recording_stopped.connect(self.recording_stopped)
        if hasattr(self._recording_panel, 'playback_started'):
            self._recording_panel.playback_started.connect(self.playback_started)
        if hasattr(self._recording_panel, 'playback_stopped'):
            self._recording_panel.playback_stopped.connect(self.playback_stopped)

    # Accessor methods for child panels
    @property
    def input_panel(self) -> QWidget:
        """Get the input configuration panel."""
        return self._input_panel

    @property
    def channel_panel(self) -> QWidget:
        """Get the channel control panel."""
        return self._channel_panel

    @property
    def noise_panel(self) -> QWidget:
        """Get the noise configuration panel."""
        return self._noise_panel

    @property
    def impairments_panel(self) -> QWidget:
        """Get the impairments panel."""
        return self._impairments_panel

    @property
    def ionosphere_panel(self) -> QWidget:
        """Get the ionosphere panel."""
        return self._ionosphere_panel

    @property
    def recording_panel(self) -> QWidget:
        """Get the recording panel."""
        return self._recording_panel
