"""Main application window for HF Path Simulator."""

from typing import Optional
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QStatusBar,
    QToolBar,
    QLabel,
    QComboBox,
    QPushButton,
    QFileDialog,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction

from hfpathsim.core.channel import HFChannel, ProcessingConfig
from hfpathsim.core.parameters import VoglerParameters, ITUCondition
from hfpathsim.core.watterson import WattersonChannel, WattersonConfig
from hfpathsim.core.vogler_hoffmeyer import VoglerHoffmeyerChannel, VoglerHoffmeyerConfig
from hfpathsim.core.noise import NoiseGenerator, NoiseConfig
from hfpathsim.core.impairments import (
    ImpairmentChain,
    AGC,
    Limiter,
    FrequencyOffset,
    AGCConfig,
    LimiterConfig,
    FrequencyOffsetConfig,
)
from hfpathsim.core.recording import ChannelRecorder, ChannelPlayer
from hfpathsim.input.base import InputSource
from hfpathsim.input.file import FileInputSource
from hfpathsim.output.base import OutputSink

from .widgets.channel_display import ChannelDisplayWidget
from .widgets.scattering import ScatteringWidget
from .widgets.spectrum import SpectrumWidget
from .widgets.control_tabs import ControlTabWidget
from .widgets.input_config import InputConfigWidget
from .widgets.output_config import OutputConfigWidget
from .widgets.channel_panel import ChannelPanel
from .widgets.noise_panel import NoisePanel
from .widgets.impairments_panel import ImpairmentsPanel
from .widgets.ionosphere_panel import IonospherePanel
from .widgets.recording_panel import RecordingPanel


class MainWindow(QMainWindow):
    """Main application window for HF Path Simulator.

    Integrates all backend features:
    - Vogler IPM and Watterson TDL channel models
    - Noise injection (AWGN, atmospheric, man-made, impulse)
    - Signal impairments (AGC, limiter, frequency offset)
    - Ray tracing and ionospheric modeling
    - Channel state recording and playback
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("HF Path Simulator")
        self.setMinimumSize(1400, 900)

        # Core components
        self._channel: Optional[HFChannel] = None
        self._watterson_channel: Optional[WattersonChannel] = None
        self._vh_channel: Optional[VoglerHoffmeyerChannel] = None
        self._input_source: Optional[InputSource] = None
        self._output_sink: Optional[OutputSink] = None
        self._output_enabled = False
        self._running = False

        # Current channel model selection: "vogler", "watterson", or "vogler_hoffmeyer"
        self._current_model = "vogler"

        # Impairment chain components
        self._noise_generator: Optional[NoiseGenerator] = None
        self._agc: Optional[AGC] = None
        self._limiter: Optional[Limiter] = None
        self._freq_offset: Optional[FrequencyOffset] = None

        # Recording/playback
        self._recorder: Optional[ChannelRecorder] = None
        self._player: Optional[ChannelPlayer] = None

        # Processing timer
        self._process_timer = QTimer()
        self._process_timer.timeout.connect(self._process_block)

        # Meter update timer
        self._meter_timer = QTimer()
        self._meter_timer.timeout.connect(self._update_meters)

        # Setup UI
        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()

        # Initialize channel and impairments
        self._init_channel()
        self._init_impairments()

        # Apply stylesheet
        self._apply_style()

    def _setup_ui(self):
        """Setup the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Top section: Display widgets
        display_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side: Channel response and input spectrum
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self._channel_display = ChannelDisplayWidget()
        self._input_spectrum = SpectrumWidget(title="Input Spectrum")

        left_layout.addWidget(self._channel_display, stretch=1)
        left_layout.addWidget(self._input_spectrum, stretch=1)

        # Right side: Scattering function and output spectrum
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._scattering = ScatteringWidget()
        self._output_spectrum = SpectrumWidget(title="Output Spectrum")

        right_layout.addWidget(self._scattering, stretch=1)
        right_layout.addWidget(self._output_spectrum, stretch=1)

        display_splitter.addWidget(left_panel)
        display_splitter.addWidget(right_panel)
        display_splitter.setSizes([700, 700])

        main_layout.addWidget(display_splitter, stretch=3)

        # Bottom section: Tabbed controls
        self._control_tabs = ControlTabWidget()

        # Create all control panels
        self._input_config = InputConfigWidget()
        self._output_config = OutputConfigWidget()
        self._channel_panel = ChannelPanel()
        self._noise_panel = NoisePanel()
        self._impairments_panel = ImpairmentsPanel()
        self._ionosphere_panel = IonospherePanel()
        self._recording_panel = RecordingPanel()

        # Setup tabs with panels
        self._control_tabs.setup_panels(
            input_panel=self._input_config,
            channel_panel=self._channel_panel,
            noise_panel=self._noise_panel,
            impairments_panel=self._impairments_panel,
            ionosphere_panel=self._ionosphere_panel,
            recording_panel=self._recording_panel,
            output_panel=self._output_config,
        )

        # Connect control tab signals
        self._connect_control_signals()

        main_layout.addWidget(self._control_tabs, stretch=0)

    def _connect_control_signals(self):
        """Connect signals from control tabs to handlers."""
        # Input source signals
        self._control_tabs.source_changed.connect(self._on_input_changed)
        self._control_tabs.start_requested.connect(self._start_processing)
        self._control_tabs.stop_requested.connect(self._stop_processing)

        # Channel model signals
        self._control_tabs.parameters_changed.connect(self._on_parameters_changed)
        self._control_tabs.watterson_config_changed.connect(self._on_watterson_config_changed)
        self._control_tabs.vogler_hoffmeyer_config_changed.connect(self._on_vh_config_changed)
        self._control_tabs.model_changed.connect(self._on_model_changed)

        # Noise signals
        self._control_tabs.noise_config_changed.connect(self._on_noise_config_changed)

        # Impairment signals
        self._control_tabs.agc_config_changed.connect(self._on_agc_config_changed)
        self._control_tabs.limiter_config_changed.connect(self._on_limiter_config_changed)
        self._control_tabs.freq_offset_config_changed.connect(self._on_freq_offset_config_changed)

        # Ionosphere signals
        self._control_tabs.ray_tracing_requested.connect(self._on_ray_tracing_requested)
        self._control_tabs.sporadic_e_changed.connect(self._on_sporadic_e_changed)
        self._control_tabs.geomagnetic_changed.connect(self._on_geomagnetic_changed)

        # Recording signals
        self._control_tabs.recording_started.connect(self._on_recording_started)
        self._control_tabs.recording_stopped.connect(self._on_recording_stopped)
        self._control_tabs.playback_started.connect(self._on_playback_started)
        self._control_tabs.playback_stopped.connect(self._on_playback_stopped)

        # Output sink signals
        self._control_tabs.output_sink_changed.connect(self._on_output_sink_changed)
        self._control_tabs.output_enabled_changed.connect(self._on_output_enabled_changed)

    def _setup_menu(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open IQ File...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        reset_view = QAction("&Reset Views", self)
        reset_view.triggered.connect(self._reset_views)
        view_menu.addAction(reset_view)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        gpu_info = QAction("&GPU Info", self)
        gpu_info.triggered.connect(self._show_gpu_info)
        tools_menu.addAction(gpu_info)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_toolbar(self):
        """Setup toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Start/Stop buttons
        self._start_btn = QPushButton("Start")
        self._start_btn.clicked.connect(self._start_processing)
        toolbar.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._stop_processing)
        self._stop_btn.setEnabled(False)
        toolbar.addWidget(self._stop_btn)

        toolbar.addSeparator()

        # ITU condition presets
        toolbar.addWidget(QLabel("Preset: "))
        self._preset_combo = QComboBox()
        self._preset_combo.addItems([
            "Quiet",
            "Moderate",
            "Disturbed",
            "Flutter",
        ])
        self._preset_combo.setCurrentIndex(1)  # Moderate
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        toolbar.addWidget(self._preset_combo)

        toolbar.addSeparator()

        # GPU status indicator
        self._gpu_indicator = QLabel("GPU: --")
        toolbar.addWidget(self._gpu_indicator)

    def _setup_statusbar(self):
        """Setup status bar."""
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)

        # Permanent widgets
        self._gpu_label = QLabel("GPU: Checking...")
        self._statusbar.addPermanentWidget(self._gpu_label)

        self._rate_label = QLabel("Rate: 0 Msps")
        self._statusbar.addPermanentWidget(self._rate_label)

        self._snr_label = QLabel("SNR: -- dB")
        self._statusbar.addPermanentWidget(self._snr_label)

        self._mode_count_label = QLabel("Modes: 0")
        self._statusbar.addPermanentWidget(self._mode_count_label)

        self._statusbar.showMessage("Ready")

        # Check GPU
        self._check_gpu()

    def _check_gpu(self):
        """Check GPU availability and update status."""
        try:
            from hfpathsim.gpu import get_device_info, is_available

            if is_available():
                info = get_device_info()
                self._gpu_label.setText(
                    f"GPU: {info['name']} ({info['backend']})"
                )
                self._gpu_indicator.setText(f"GPU: {info['name'][:20]}")
                self._gpu_indicator.setStyleSheet("color: #4EC9B0;")
            else:
                self._gpu_label.setText("GPU: Not available (CPU mode)")
                self._gpu_indicator.setText("GPU: CPU mode")
                self._gpu_indicator.setStyleSheet("color: #CE9178;")
        except Exception as e:
            self._gpu_label.setText(f"GPU: Error - {e}")
            self._gpu_indicator.setText("GPU: Error")
            self._gpu_indicator.setStyleSheet("color: #F14C4C;")

    def _apply_style(self):
        """Apply stylesheet to the application."""
        style = """
        QMainWindow {
            background-color: #1e1e1e;
        }
        QWidget {
            background-color: #252526;
            color: #cccccc;
            font-family: "Segoe UI", Arial, sans-serif;
            font-size: 12px;
        }
        QGroupBox {
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 8px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
        }
        QPushButton {
            background-color: #0e639c;
            border: none;
            border-radius: 4px;
            padding: 6px 16px;
            color: white;
        }
        QPushButton:hover {
            background-color: #1177bb;
        }
        QPushButton:pressed {
            background-color: #0d5a8c;
        }
        QPushButton:disabled {
            background-color: #3c3c3c;
            color: #666666;
        }
        QComboBox {
            background-color: #3c3c3c;
            border: 1px solid #555555;
            border-radius: 4px;
            padding: 4px 8px;
        }
        QComboBox::drop-down {
            border: none;
        }
        QDoubleSpinBox, QSpinBox {
            background-color: #3c3c3c;
            border: 1px solid #555555;
            border-radius: 4px;
            padding: 4px;
        }
        QCheckBox {
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
        }
        QRadioButton {
            spacing: 8px;
        }
        QRadioButton::indicator {
            width: 16px;
            height: 16px;
        }
        QSlider::groove:horizontal {
            height: 6px;
            background: #3c3c3c;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            width: 16px;
            margin: -5px 0;
            background: #0e639c;
            border-radius: 8px;
        }
        QSlider::handle:horizontal:hover {
            background: #1177bb;
        }
        QProgressBar {
            border: 1px solid #3c3c3c;
            border-radius: 3px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #0e639c;
            border-radius: 2px;
        }
        QTabWidget::pane {
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            padding: 4px;
        }
        QTabBar::tab {
            background-color: #2d2d2d;
            border: 1px solid #3c3c3c;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            padding: 6px 16px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #252526;
            border-bottom: 1px solid #252526;
        }
        QTabBar::tab:hover:!selected {
            background-color: #333333;
        }
        QTableWidget {
            background-color: #1e1e1e;
            gridline-color: #3c3c3c;
        }
        QTableWidget::item {
            padding: 4px;
        }
        QHeaderView::section {
            background-color: #2d2d2d;
            border: 1px solid #3c3c3c;
            padding: 4px;
        }
        QStatusBar {
            background-color: #007acc;
            color: white;
        }
        QToolBar {
            background-color: #333333;
            border: none;
            spacing: 8px;
            padding: 4px;
        }
        QMenuBar {
            background-color: #333333;
        }
        QMenuBar::item:selected {
            background-color: #094771;
        }
        QMenu {
            background-color: #252526;
            border: 1px solid #3c3c3c;
        }
        QMenu::item:selected {
            background-color: #094771;
        }
        QLineEdit {
            background-color: #3c3c3c;
            border: 1px solid #555555;
            border-radius: 4px;
            padding: 4px;
        }
        QScrollArea {
            border: none;
        }
        QFrame[frameShape="4"], QFrame[frameShape="5"] {
            background-color: #2d2d2d;
        }
        """
        self.setStyleSheet(style)

    def _init_channel(self):
        """Initialize the HF channel."""
        params = VoglerParameters.from_itu_condition(ITUCondition.MODERATE)
        config = ProcessingConfig()

        self._channel = HFChannel(params, config, use_gpu=True)
        self._channel.add_state_callback(self._on_channel_state)

        # Also initialize Watterson channel
        watterson_config = WattersonConfig.from_itu_condition(ITUCondition.MODERATE)
        self._watterson_channel = WattersonChannel(watterson_config)

        # Initialize Vogler-Hoffmeyer channel
        vh_config = VoglerHoffmeyerConfig.from_itu_condition(ITUCondition.MODERATE)
        self._vh_channel = VoglerHoffmeyerChannel(vh_config)
        self._vh_channel.add_state_callback(self._on_vh_channel_state)

        # Trigger initial state update
        state = self._channel.get_state()
        self._on_channel_state(state)

        # Update mode count in status bar
        self._mode_count_label.setText(f"Modes: {len(params.modes)}")

    def _init_impairments(self):
        """Initialize impairment chain components."""
        # Noise generator
        self._noise_generator = NoiseGenerator(NoiseConfig())

        # AGC
        self._agc = AGC(AGCConfig())

        # Limiter
        self._limiter = Limiter(LimiterConfig())

        # Frequency offset
        self._freq_offset = FrequencyOffset(FrequencyOffsetConfig())

    def _on_channel_state(self, state):
        """Handle channel state updates."""
        # Update displays
        if state.transfer_function is not None:
            self._channel_display.update_transfer_function(
                state.freq_axis_hz,
                state.transfer_function,
            )

        if state.impulse_response is not None:
            self._channel_display.update_impulse_response(
                state.delay_axis_ms,
                state.impulse_response,
            )

        if state.scattering_function is not None:
            self._scattering.update_data(
                state.scattering_function,
                state.delay_axis_ms[:state.scattering_function.shape[1]],
                state.doppler_axis_hz,
            )

        # Record snapshot if recording
        if self._recorder and self._recording_panel.is_recording():
            self._recorder.capture(state.timestamp)
            self._recording_panel.update_snapshot_count(self._recorder.num_snapshots)

    def _on_input_changed(self, source: InputSource):
        """Handle input source change."""
        self._input_source = source
        self._statusbar.showMessage(
            f"Input: {type(source).__name__} @ {source.sample_rate/1e6:.1f} Msps"
        )

    def _on_parameters_changed(self, params: VoglerParameters):
        """Handle Vogler parameter changes."""
        if self._channel and self._current_model == "vogler":
            self._channel.update_parameters(params)
            self._mode_count_label.setText(f"Modes: {len(params.modes)}")

    def _on_watterson_config_changed(self, config: WattersonConfig):
        """Handle Watterson config changes."""
        if self._watterson_channel and self._current_model == "watterson":
            self._watterson_channel = WattersonChannel(config)
            self._mode_count_label.setText(f"Taps: {len(config.taps)}")

    def _on_vh_config_changed(self, config: VoglerHoffmeyerConfig):
        """Handle Vogler-Hoffmeyer config changes."""
        if self._current_model == "vogler_hoffmeyer":
            self._vh_channel = VoglerHoffmeyerChannel(config)
            self._vh_channel.add_state_callback(self._on_vh_channel_state)
            self._mode_count_label.setText(f"Modes: {len(config.modes)}")

    def _on_vh_channel_state(self, state):
        """Handle Vogler-Hoffmeyer channel state updates."""
        # Update scattering function display
        if self._vh_channel and self._current_model == "vogler_hoffmeyer":
            delay_axis, doppler_axis, S = self._vh_channel.compute_scattering_function()
            self._scattering.update_data(S.T, delay_axis / 1000.0, doppler_axis)  # Convert us to ms

    def _on_model_changed(self, model_id: str):
        """Handle channel model selection change."""
        self._current_model = model_id
        model_names = {
            "vogler": "Vogler IPM",
            "watterson": "Watterson TDL",
            "vogler_hoffmeyer": "Vogler-Hoffmeyer Stochastic"
        }
        self._statusbar.showMessage(
            f"Channel model: {model_names.get(model_id, model_id)}"
        )

        # Update mode count for current model
        if model_id == "vogler_hoffmeyer" and self._vh_channel:
            self._mode_count_label.setText(f"Modes: {len(self._vh_channel.config.modes)}")
            # Trigger scattering function update
            self._on_vh_channel_state(self._vh_channel.get_state())

    def _on_noise_config_changed(self, config: NoiseConfig):
        """Handle noise configuration change."""
        if self._noise_generator:
            # Preserve the sample rate when recreating
            sample_rate = self._noise_generator.sample_rate
            self._noise_generator = NoiseGenerator(config, sample_rate_hz=sample_rate)
            self._snr_label.setText(f"SNR: {config.snr_db:.1f} dB")

    def _on_agc_config_changed(self, config: AGCConfig):
        """Handle AGC configuration change."""
        if self._agc:
            # Preserve the sample rate when recreating
            sample_rate = self._agc.sample_rate
            self._agc = AGC(config, sample_rate_hz=sample_rate)

    def _on_limiter_config_changed(self, config: LimiterConfig):
        """Handle limiter configuration change."""
        if self._limiter:
            # Preserve the sample rate when recreating
            sample_rate = self._limiter.sample_rate
            self._limiter = Limiter(config, sample_rate_hz=sample_rate)

    def _on_freq_offset_config_changed(self, config: FrequencyOffsetConfig):
        """Handle frequency offset configuration change."""
        if self._freq_offset:
            # Preserve the sample rate when recreating
            sample_rate = self._freq_offset.sample_rate
            self._freq_offset = FrequencyOffset(config, sample_rate_hz=sample_rate)

    def _on_ray_tracing_requested(self, request: dict):
        """Handle ray tracing request."""
        try:
            from hfpathsim.core.raytracing.path_finder import PathFinder
            from hfpathsim.core.raytracing.ionosphere import create_simple_profile

            # Get current ionospheric parameters from channel panel
            params = self._channel_panel.get_vogler_parameters()

            profile = create_simple_profile(
                foF2=params.foF2,
                hmF2=params.hmF2,
                foE=params.foE,
                hmE=params.hmE,
            )

            finder = PathFinder(profile)
            modes = finder.find_modes(
                frequency_mhz=params.frequency_mhz,
                tx_lat=request["tx_lat"],
                tx_lon=request["tx_lon"],
                rx_lat=request["rx_lat"],
                rx_lon=request["rx_lon"],
                max_hops=request.get("max_hops", 3),
            )

            # Convert to display format
            display_modes = []
            for mode in modes:
                display_modes.append({
                    "name": mode.name,
                    "delay_ms": mode.group_delay_ms,
                    "muf_mhz": mode.reflection_height_km * 0.05 + params.foF2,  # Approx
                    "angle_deg": mode.launch_angle_deg,
                })

            self._ionosphere_panel.set_discovered_modes(display_modes)
            self._mode_count_label.setText(f"Modes: {len(modes)}")

        except Exception as e:
            self._statusbar.showMessage(f"Ray tracing error: {e}")

    def _on_sporadic_e_changed(self, config):
        """Handle Sporadic-E configuration change."""
        # Could inject Es layer into profile
        pass

    def _on_geomagnetic_changed(self, indices):
        """Handle geomagnetic indices change."""
        # Could apply to channel model
        pass

    def _on_recording_started(self):
        """Handle recording start."""
        if self._channel:
            rate = self._recording_panel.get_snapshot_rate()
            max_duration = self._recording_panel.get_max_duration()

            self._recorder = ChannelRecorder(
                self._channel,
                snapshot_rate_hz=rate,
                max_duration_sec=max_duration,
            )
            self._recorder.start()
            self._statusbar.showMessage("Recording started")

    def _on_recording_stopped(self, filename: str):
        """Handle recording stop."""
        if self._recorder:
            self._recorder.stop()

            # Get metadata from panel
            metadata = self._recording_panel.get_metadata()

            # Save with selected format
            fmt = self._recording_panel.get_format()
            filepath = f"./{filename}"

            try:
                self._recorder.save(filepath, format=fmt)
                self._statusbar.showMessage(f"Recording saved: {filepath}")
            except Exception as e:
                self._statusbar.showMessage(f"Save error: {e}")

            self._recorder = None

    def _on_playback_started(self, filepath: str):
        """Handle playback start."""
        try:
            self._player = ChannelPlayer()
            self._player.load(filepath)
            self._statusbar.showMessage(f"Playing: {filepath}")

            # Start playback timer
            rate = self._recording_panel.get_playback_rate()
            interval_ms = int(1000 / (self._player.snapshot_rate_hz * rate))
            # Playback would be handled by a separate timer

        except Exception as e:
            self._statusbar.showMessage(f"Playback error: {e}")

    def _on_playback_stopped(self):
        """Handle playback stop."""
        self._player = None
        self._statusbar.showMessage("Playback stopped")

    def _on_output_sink_changed(self, sink: OutputSink):
        """Handle output sink change."""
        # Close existing sink if open
        if self._output_sink and self._output_sink.is_open:
            self._output_sink.close()

        self._output_sink = sink
        self._statusbar.showMessage(
            f"Output: {type(sink).__name__} @ {sink.sample_rate/1e6:.1f} Msps"
        )

        # If output is already enabled, open the new sink immediately
        if self._output_enabled and not sink.is_open:
            if sink.open():
                self._statusbar.showMessage("Output enabled")
                self._output_config.set_status_message("Streaming")
            else:
                self._statusbar.showMessage("Failed to open output sink")
                self._output_config.set_status_message("Failed to open")
                self._output_enabled = False

    def _on_output_enabled_changed(self, enabled: bool):
        """Handle output enable/disable."""
        self._output_enabled = enabled

        if enabled and self._output_sink and not self._output_sink.is_open:
            print(f"DEBUG: Attempting to open sink: {type(self._output_sink).__name__}")
            result = self._output_sink.open()
            print(f"DEBUG: Sink open() returned: {result}, is_open={self._output_sink.is_open}")
            if result:
                self._statusbar.showMessage("Output enabled")
                self._output_config.set_status_message("Streaming")
            else:
                self._statusbar.showMessage("Failed to open output sink")
                self._output_config.set_status_message("Failed to open")
                self._output_enabled = False
        elif not enabled and self._output_sink and self._output_sink.is_open:
            self._output_sink.close()
            self._statusbar.showMessage("Output disabled")
            self._output_config.set_status_message("Stopped")

    def _on_preset_changed(self, preset_name: str):
        """Handle ITU preset selection."""
        presets = {
            "Quiet": ITUCondition.QUIET,
            "Moderate": ITUCondition.MODERATE,
            "Disturbed": ITUCondition.DISTURBED,
            "Flutter": ITUCondition.FLUTTER,
        }

        if preset_name in presets:
            params = VoglerParameters.from_itu_condition(presets[preset_name])
            self._channel_panel.set_vogler_parameters(params)

            if self._channel:
                self._channel.update_parameters(params)

    def _start_processing(self):
        """Start real-time processing."""
        if self._running:
            return

        if self._input_source is None:
            QMessageBox.warning(
                self, "No Input",
                "Please configure an input source first."
            )
            return

        if not self._input_source.is_open:
            if not self._input_source.open():
                QMessageBox.warning(
                    self, "Input Error",
                    "Failed to open input source."
                )
                return

        # Recreate ALL channels and impairments with correct sample rate from input
        input_rate = self._input_source.sample_rate
        print(f"=" * 60)
        print(f"DEBUG _start_processing: Input sample rate = {input_rate}Hz")
        print(f"DEBUG: Current channel model = {self._current_model}")

        # Reset debug flags so we log again
        for attr in ['_debug_logged', '_debug_logged2', '_debug_noise', '_debug_agc']:
            if hasattr(self, attr):
                delattr(self, attr)

        # Recreate Watterson channel
        if self._watterson_channel:
            old_config = self._watterson_channel.config
            new_config = WattersonConfig(
                taps=old_config.taps,
                sample_rate_hz=input_rate,
                block_size=int(input_rate * 0.05),  # 50ms blocks
                update_rate_hz=old_config.update_rate_hz,
            )
            self._watterson_channel = WattersonChannel(new_config)
            print(f"DEBUG: Watterson recreated at {input_rate}Hz")

        # Recreate Vogler HFChannel at 1 MHz (RF rate)
        # Vogler requires MHz-rate processing for valid ionospheric physics
        # We upsample before Vogler and downsample after in _process_vogler_with_resampling
        vogler_rate = 1_000_000  # 1 MHz
        if self._channel:
            old_params = self._channel.params
            new_config = ProcessingConfig(
                sample_rate_hz=vogler_rate,
                block_size=int(vogler_rate * 0.05),  # 50ms blocks at 1 MHz = 50000 samples
                overlap=int(vogler_rate * 0.0125),
            )
            self._channel = HFChannel(old_params, new_config, use_gpu=self._channel.use_gpu)
            self._channel.add_state_callback(self._on_channel_state)
            print(f"DEBUG: Vogler HFChannel at {vogler_rate/1e6:.0f}MHz (input will be resampled)")

        # Recreate impairments with correct sample rate
        # This is CRITICAL - defaults are 2MHz but input may be 8kHz!
        print(f"DEBUG: Recreating impairments at {input_rate}Hz sample rate")
        if self._noise_generator:
            old_noise_rate = self._noise_generator.sample_rate
            self._noise_generator = NoiseGenerator(
                self._noise_generator.config,
                sample_rate_hz=input_rate,
            )
            print(f"DEBUG: NoiseGenerator: {old_noise_rate}Hz -> {input_rate}Hz")
        if self._agc:
            old_agc_rate = self._agc.sample_rate
            self._agc = AGC(
                self._agc.config,
                sample_rate_hz=input_rate,
            )
            print(f"DEBUG: AGC: {old_agc_rate}Hz -> {input_rate}Hz")
        if self._limiter:
            old_limiter_rate = self._limiter.sample_rate
            self._limiter = Limiter(
                self._limiter.config,
                sample_rate_hz=input_rate,
            )
            print(f"DEBUG: Limiter: {old_limiter_rate}Hz -> {input_rate}Hz")
        if self._freq_offset:
            old_freq_rate = self._freq_offset.sample_rate
            self._freq_offset = FrequencyOffset(
                self._freq_offset.config,
                sample_rate_hz=input_rate,
            )
            print(f"DEBUG: FreqOffset: {old_freq_rate}Hz -> {input_rate}Hz")
        # Warn if using Vogler model with low sample rate
        if self._current_model == "vogler" and input_rate < 100000:
            print("=" * 60)
            print("NOTE: Vogler IPM with multipath Rayleigh fading")
            print(f"      Modes: {len(self._channel.params.modes)}, "
                  f"Doppler: {self._channel.params.doppler_spread_hz} Hz, "
                  f"Delay spread: {self._channel.params.delay_spread_ms} ms")
            print("=" * 60)

        # Reset block counter for debug output
        self._block_count = 0
        print(f"=" * 60)

        # Recreate Vogler-Hoffmeyer channel with correct sample rate
        if self._vh_channel:
            old_vh_config = self._vh_channel.config
            new_vh_config = VoglerHoffmeyerConfig(
                modes=old_vh_config.modes,
                sample_rate=input_rate,
                spread_f_enabled=old_vh_config.spread_f_enabled,
                random_seed=old_vh_config.random_seed,
            )
            self._vh_channel = VoglerHoffmeyerChannel(new_vh_config)
            self._vh_channel.add_state_callback(self._on_vh_channel_state)

        self._running = True
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._statusbar.showMessage(f"Processing at {input_rate/1000:.1f} kHz...")

        # Start processing timer (process blocks every 50ms)
        self._process_timer.start(50)

        # Start meter update timer (update meters every 100ms)
        self._meter_timer.start(100)

    def _stop_processing(self):
        """Stop real-time processing."""
        if not self._running:
            return

        self._running = False
        self._process_timer.stop()
        self._meter_timer.stop()

        # Clear the audio buffer immediately to stop playback
        if self._output_sink and hasattr(self._output_sink, 'clear'):
            self._output_sink.clear()

        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._statusbar.showMessage("Stopped")

    def _process_vogler_with_resampling(self, samples: np.ndarray) -> np.ndarray:
        """Process samples through Vogler ionospheric channel model.

        Full RF processing chain:
        1. Upsample from baseband rate (8 kHz) to RF rate (1 MHz)
        2. Mix up to RF carrier frequency
        3. Apply Vogler channel model at RF rate
        4. Mix back down to baseband
        5. Lowpass filter
        6. Downsample back to baseband rate

        Args:
            samples: Input samples at baseband rate (e.g., 8 kHz)

        Returns:
            Processed samples with ionospheric fading applied
        """
        from scipy import signal as scipy_signal

        if self._channel is None:
            return samples

        params = self._channel.params
        input_rate = self._input_source.sample_rate
        n_samples = len(samples)

        # RF processing parameters — must be > input_rate so the anti-alias
        # cutoff normalizes to < 1.  Use 4× input_rate with a 1 MHz floor.
        rf_rate = max(1_000_000, int(input_rate) * 4)
        upsample_factor = int(rf_rate / input_rate)
        rf_carrier_hz = 100_000  # 100 kHz carrier within the RF band

        # Initialize RF processing state if needed
        if not hasattr(self, '_vogler_rf_state'):
            self._init_vogler_rf_state(input_rate, rf_rate, rf_carrier_hz)

        # === Step 1: Upsample to RF rate ===
        n_rf_samples = n_samples * upsample_factor
        rf_samples = scipy_signal.resample(samples, n_rf_samples).astype(np.complex128)

        # === Step 2: Mix up to RF carrier ===
        state = self._vogler_rf_state
        t_rf = np.arange(n_rf_samples) / rf_rate + state['rf_time']
        state['rf_time'] += n_rf_samples / rf_rate
        rf_signal = rf_samples * np.exp(1j * 2 * np.pi * rf_carrier_hz * t_rf)

        # === Step 3: Apply Vogler channel at RF rate ===
        # Use the HFChannel.process() method at RF rate
        rf_processed = self._apply_vogler_rf(rf_signal, rf_rate, params)

        # === Step 4: Mix back down to baseband ===
        baseband_signal = rf_processed * np.exp(-1j * 2 * np.pi * rf_carrier_hz * t_rf)

        # === Step 5: Lowpass filter (anti-aliasing) ===
        # Filter bandwidth = input_rate / 2 (Nyquist of original signal)
        cutoff_normalized = (input_rate / 2) / (rf_rate / 2)  # Normalized to Nyquist
        filtered = scipy_signal.sosfilt(state['lpf_sos'], baseband_signal)

        # === Step 6: Downsample back to input rate ===
        output = scipy_signal.resample(filtered, n_samples).astype(np.complex64)

        if not hasattr(self, '_vogler_rf_debug'):
            self._vogler_rf_debug = True
            print(f"DEBUG Vogler RF: {input_rate/1000:.0f}kHz -> {rf_rate/1e6:.0f}MHz -> "
                  f"{input_rate/1000:.0f}kHz, carrier={rf_carrier_hz/1000:.0f}kHz")
            print(f"DEBUG Vogler RF: in={np.max(np.abs(samples)):.4f}, "
                  f"rf={np.max(np.abs(rf_processed)):.4f}, "
                  f"out={np.max(np.abs(output)):.4f}")

        return output

    def _init_vogler_rf_state(self, input_rate: float, rf_rate: float, rf_carrier_hz: float):
        """Initialize state for RF processing chain."""
        from scipy import signal as scipy_signal

        # Design lowpass filter for anti-aliasing before downsampling
        # Cutoff at input_rate/2 (Nyquist of original signal)
        cutoff_normalized = (input_rate / 2) / (rf_rate / 2)
        lpf_sos = scipy_signal.butter(8, cutoff_normalized, btype='low', output='sos')

        # State for the Vogler RF channel processing
        self._vogler_rf_state = {
            'rf_time': 0.0,  # Running time for mixer phase continuity
            'lpf_sos': lpf_sos,
            'rf_rate': rf_rate,
            'input_rate': input_rate,
            'rf_carrier_hz': rf_carrier_hz,
            # Fading state per mode
            'mode_states': [],
        }

        # Initialize fading state for each mode
        if self._channel is not None:
            params = self._channel.params
            block_duration = 0.05  # 50ms blocks
            update_rate = 1.0 / block_duration

            for mode in params.modes:
                if not mode.enabled:
                    continue

                # Doppler filter for Rayleigh fading
                doppler = params.doppler_spread_hz
                if doppler > 0:
                    coherence_blocks = update_rate / doppler
                    filter_len = max(16, min(64, int(coherence_blocks * 2)))
                else:
                    filter_len = 16

                # Gaussian Doppler shaping filter
                t = np.arange(filter_len) - filter_len // 2
                f_norm = doppler / update_rate if update_rate > 0 else 0.05
                sigma = max(1.0, 1.0 / (2 * np.pi * f_norm)) if f_norm > 0 else filter_len / 4
                doppler_filter = np.exp(-0.5 * (t / sigma) ** 2)
                doppler_filter = doppler_filter / np.sqrt(np.sum(doppler_filter**2))

                # Delay samples at RF rate
                delay_samples_rf = int(mode.delay_offset_ms / 1000 * rf_rate)

                mode_state = {
                    'mode': mode,
                    'doppler_filter': doppler_filter.astype(np.complex128),
                    'noise_buffer': (np.random.randn(filter_len)
                        + 1j * np.random.randn(filter_len)) / np.sqrt(2),
                    'current_gain': complex(mode.relative_amplitude, 0),
                    'prev_gain': complex(mode.relative_amplitude, 0),
                    'delay_buffer': np.zeros(max(1, delay_samples_rf), dtype=np.complex128),
                    'delay_samples': delay_samples_rf,
                }
                self._vogler_rf_state['mode_states'].append(mode_state)

            print(f"DEBUG: Vogler RF initialized: {len(self._vogler_rf_state['mode_states'])} modes, "
                  f"rf_rate={rf_rate/1e6:.0f}MHz")

    def _apply_vogler_rf(self, rf_signal: np.ndarray, rf_rate: float,
                         params) -> np.ndarray:
        """Apply Vogler channel effects at RF rate.

        Implements multipath propagation with time-varying Rayleigh fading
        for each propagation mode.
        """
        n_samples = len(rf_signal)
        output = np.zeros(n_samples, dtype=np.complex128)

        # Update fading coefficients once per block
        self._update_vogler_rf_fading()

        for state in self._vogler_rf_state['mode_states']:
            mode = state['mode']
            fading_gain = state['current_gain']
            delay_samples = state['delay_samples']

            # Apply delay for this mode
            if delay_samples == 0:
                delayed = rf_signal
            else:
                buf = state['delay_buffer']
                buf_len = len(buf)
                if buf_len >= n_samples:
                    # Buffer is large enough
                    extended = np.concatenate([buf[-delay_samples:], rf_signal])
                    delayed = extended[:n_samples]
                    # Update buffer: keep last delay_samples from extended
                    state['delay_buffer'] = extended[-(delay_samples):]
                else:
                    # Small delay - simpler handling
                    extended = np.concatenate([buf, rf_signal])
                    delayed = extended[:n_samples]
                    state['delay_buffer'] = rf_signal[-delay_samples:] if delay_samples <= n_samples else \
                        np.concatenate([state['delay_buffer'][-(delay_samples-n_samples):], rf_signal])

            # Apply fading gain with interpolation for smoothness
            old_gain = state['prev_gain']
            t = np.linspace(0, 1, n_samples, dtype=np.float64)
            interp_gain = old_gain * (1 - t) + fading_gain * t
            state['prev_gain'] = fading_gain

            output += delayed * interp_gain

        return output

    def _update_vogler_rf_fading(self):
        """Update Rayleigh fading coefficients for Vogler RF modes."""
        if not hasattr(self, '_vogler_rf_state'):
            return

        for state in self._vogler_rf_state['mode_states']:
            mode = state['mode']

            # Generate new complex Gaussian noise
            noise = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)

            # Shift buffer and filter
            state['noise_buffer'] = np.roll(state['noise_buffer'], -1)
            state['noise_buffer'][-1] = noise

            # Convolve with Doppler shaping filter
            filtered = np.sum(state['noise_buffer'] * state['doppler_filter'])

            # Apply mode amplitude (Rayleigh fading)
            state['current_gain'] = filtered * mode.relative_amplitude

    def _process_vh_with_resampling(self, samples: np.ndarray) -> np.ndarray:
        """Process samples through Vogler-Hoffmeyer channel model.

        Full RF processing chain:
        1. Upsample from baseband rate (8 kHz) to RF rate (1 MHz)
        2. Mix up to RF carrier frequency
        3. Apply Vogler-Hoffmeyer channel model at RF rate
        4. Mix back down to baseband
        5. Lowpass filter
        6. Downsample back to baseband rate

        Args:
            samples: Input samples at baseband rate (e.g., 8 kHz)

        Returns:
            Processed samples with ionospheric fading applied
        """
        from scipy import signal as scipy_signal

        if self._vh_channel is None:
            return samples

        input_rate = self._input_source.sample_rate
        n_samples = len(samples)

        # RF processing parameters — must be > input_rate so the anti-alias
        # cutoff normalizes to < 1.  Use 4× input_rate with a 1 MHz floor.
        rf_rate = max(1_000_000, int(input_rate) * 4)
        upsample_factor = int(rf_rate / input_rate)
        rf_carrier_hz = 100_000  # 100 kHz carrier within the RF band

        # Initialize RF processing state if needed
        if not hasattr(self, '_vh_rf_state'):
            self._init_vh_rf_state(input_rate, rf_rate, rf_carrier_hz)

        # === Step 1: Upsample to RF rate ===
        n_rf_samples = n_samples * upsample_factor
        rf_samples = scipy_signal.resample(samples, n_rf_samples).astype(np.complex128)

        # === Step 2: Mix up to RF carrier ===
        state = self._vh_rf_state
        t_rf = np.arange(n_rf_samples) / rf_rate + state['rf_time']
        state['rf_time'] += n_rf_samples / rf_rate
        rf_signal = rf_samples * np.exp(1j * 2 * np.pi * rf_carrier_hz * t_rf)

        # === Step 3: Apply Vogler-Hoffmeyer channel at RF rate ===
        # Temporarily update VH channel sample rate to RF rate
        old_config = self._vh_channel.config
        if old_config.sample_rate != rf_rate:
            # Create new config at RF rate
            from hfpathsim.core.vogler_hoffmeyer import VoglerHoffmeyerConfig
            rf_config = VoglerHoffmeyerConfig(
                modes=old_config.modes,
                sample_rate=rf_rate,
                spread_f_enabled=old_config.spread_f_enabled,
                random_seed=old_config.random_seed,
            )
            self._vh_rf_channel = VoglerHoffmeyerChannel(rf_config)
        elif not hasattr(self, '_vh_rf_channel'):
            self._vh_rf_channel = self._vh_channel

        rf_processed = self._vh_rf_channel.process(rf_signal)

        # === Step 4: Mix back down to baseband ===
        baseband_signal = rf_processed * np.exp(-1j * 2 * np.pi * rf_carrier_hz * t_rf)

        # === Step 5: Lowpass filter (anti-aliasing) ===
        filtered = scipy_signal.sosfilt(state['lpf_sos'], baseband_signal)

        # === Step 6: Downsample back to input rate ===
        output = scipy_signal.resample(filtered, n_samples).astype(np.complex64)

        if not hasattr(self, '_vh_rf_debug'):
            self._vh_rf_debug = True
            print(f"DEBUG VH RF: {input_rate/1000:.0f}kHz -> {rf_rate/1e6:.0f}MHz -> "
                  f"{input_rate/1000:.0f}kHz, carrier={rf_carrier_hz/1000:.0f}kHz")
            print(f"DEBUG VH RF: in={np.max(np.abs(samples)):.4f}, "
                  f"rf={np.max(np.abs(rf_processed)):.4f}, "
                  f"out={np.max(np.abs(output)):.4f}")

        return output

    def _init_vh_rf_state(self, input_rate: float, rf_rate: float, rf_carrier_hz: float):
        """Initialize state for Vogler-Hoffmeyer RF processing chain."""
        from scipy import signal as scipy_signal

        # Design lowpass filter for anti-aliasing before downsampling
        # Cutoff at input_rate/2 (Nyquist of original signal)
        cutoff_normalized = (input_rate / 2) / (rf_rate / 2)
        lpf_sos = scipy_signal.butter(8, cutoff_normalized, btype='low', output='sos')

        self._vh_rf_state = {
            'rf_time': 0.0,  # Running time for mixer phase continuity
            'lpf_sos': lpf_sos,
            'rf_rate': rf_rate,
            'input_rate': input_rate,
            'rf_carrier_hz': rf_carrier_hz,
        }

        print(f"DEBUG: VH RF initialized: rf_rate={rf_rate/1e6:.0f}MHz, "
              f"carrier={rf_carrier_hz/1000:.0f}kHz")

    def _process_block(self):
        """Process one block of samples through the full chain.

        Processing order:
        Input → Channel → Noise → AGC → Limiter → Freq Offset → Output
        """
        if not self._running or self._input_source is None:
            return

        # Calculate block size to match real-time rate
        # Timer fires every 50ms, so read 50ms worth of samples
        # This prevents buffer over/underflow and maintains phase continuity
        timer_interval_sec = 0.05  # 50ms timer
        sample_rate = self._input_source.sample_rate
        block_size = int(sample_rate * timer_interval_sec)
        block_size = max(256, min(block_size, 8192))  # Clamp to reasonable range

        samples = self._input_source.read(block_size)

        if samples is None or len(samples) == 0:
            return

        # Update input spectrum
        self._input_spectrum.update_data(samples)

        # Process through channel model
        if not hasattr(self, '_debug_logged'):
            self._debug_logged = True
            print(f"DEBUG _process_block: model={self._current_model}, "
                  f"watterson={self._watterson_channel is not None}, "
                  f"input_mag={np.max(np.abs(samples)):.4f}")

        if self._current_model == "watterson" and self._watterson_channel:
            channel_output = self._watterson_channel.process_block(samples)
            if not hasattr(self, '_debug_logged2'):
                self._debug_logged2 = True
                print(f"DEBUG watterson output: max_mag={np.max(np.abs(channel_output)):.4f}")
        elif self._current_model == "vogler_hoffmeyer" and self._vh_channel:
            # Vogler-Hoffmeyer requires RF-rate processing - upsample, process, downsample
            channel_output = self._process_vh_with_resampling(samples)
        elif self._current_model == "vogler" and self._channel:
            # Vogler requires RF-rate processing - upsample, process, downsample
            channel_output = self._process_vogler_with_resampling(samples)
        else:
            channel_output = samples

        after_channel_mag = np.max(np.abs(channel_output))

        # Apply noise if enabled
        if self._noise_panel.is_noise_enabled() and self._noise_generator:
            channel_output = self._noise_generator.add_noise(channel_output)
            if not hasattr(self, '_debug_noise'):
                self._debug_noise = True
                print(f"DEBUG noise: enabled=True, sample_rate={self._noise_generator.sample_rate}, "
                      f"snr={self._noise_generator.config.snr_db}dB, "
                      f"after_noise_mag={np.max(np.abs(channel_output)):.4f}")

        # Apply AGC if enabled
        if self._impairments_panel.is_agc_enabled() and self._agc:
            before_agc = np.max(np.abs(channel_output))
            channel_output = self._agc.process_block(channel_output)
            if not hasattr(self, '_debug_agc'):
                self._debug_agc = True
                print(f"DEBUG agc: enabled=True, sample_rate={self._agc.sample_rate}, "
                      f"before={before_agc:.4f}, after={np.max(np.abs(channel_output)):.4f}, "
                      f"gain={self._agc.current_gain_db:.1f}dB")

        # Apply limiter if enabled
        if self._impairments_panel.is_limiter_enabled() and self._limiter:
            channel_output = self._limiter.process(channel_output)

        # Apply frequency offset if enabled
        if self._impairments_panel.is_freq_offset_enabled() and self._freq_offset:
            channel_output = self._freq_offset.process(channel_output)

        # Debug output - first 5 blocks show detailed trace
        if not hasattr(self, '_block_count'):
            self._block_count = 0
        self._block_count += 1

        if self._block_count <= 5:
            noise_rate = self._noise_generator.sample_rate if self._noise_generator else 'N/A'
            agc_rate = self._agc.sample_rate if self._agc else 'N/A'
            agc_gain = self._agc.current_gain_db if self._agc else 0
            print(f"Block {self._block_count}: in={np.max(np.abs(samples)):.3f} -> "
                  f"out={np.max(np.abs(channel_output)):.3f}, "
                  f"noise_sr={noise_rate}, agc_sr={agc_rate}, agc={agc_gain:.1f}dB")
        elif self._block_count % 100 == 0:
            print(f"Block {self._block_count}: input_mag={np.max(np.abs(samples)):.3f}, "
                  f"output_mag={np.max(np.abs(channel_output)):.3f}, "
                  f"agc_gain={self._agc.current_gain_db if self._agc else 'N/A'}dB")

        # Update output spectrum
        self._output_spectrum.update_data(channel_output)

        # Write to output sink if enabled
        if self._output_enabled and self._output_sink and self._output_sink.is_open:
            written = self._output_sink.write(channel_output)
            if self._block_count <= 5:
                fill = getattr(self._output_sink, 'buffer_fill', None)
                fill_str = f", buffer_fill={fill:.1f}%" if fill is not None else ""
                print(f"  Audio write: {written}/{len(channel_output)} samples{fill_str}")
        else:
            # Debug: show why output isn't working
            if not hasattr(self, '_debug_output_shown'):
                self._debug_output_shown = True
                print(f"DEBUG: output_enabled={self._output_enabled}, sink={self._output_sink is not None}, is_open={self._output_sink.is_open if self._output_sink else 'N/A'}")
            # Update output status periodically
            if hasattr(self._output_sink, 'buffer_fill'):
                self._output_config.update_status(
                    self._output_sink.total_samples_written,
                    self._output_sink.buffer_fill,
                )

        # Update rate display
        rate = self._input_source.sample_rate / 1e6
        self._rate_label.setText(f"Rate: {rate:.1f} Msps")

    def _update_meters(self):
        """Update AGC and limiter meters."""
        if self._agc and self._impairments_panel.is_agc_enabled():
            self._impairments_panel.update_agc_meter(self._agc.current_gain_db)

        if self._limiter and self._impairments_panel.is_limiter_enabled():
            self._impairments_panel.update_limiter_meter(self._limiter.gain_reduction_db)

        if self._freq_offset and self._impairments_panel.is_freq_offset_enabled():
            # Get current offset including drift
            config = self._impairments_panel.get_freq_offset_config()
            self._impairments_panel.update_current_offset(config.offset_hz)

    def _open_file(self):
        """Open an IQ file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open IQ File",
            "",
            "IQ Files (*.wav *.sigmf-data *.raw *.bin *.cf32);;All Files (*)",
        )

        if filepath:
            source = FileInputSource(filepath, loop=True)
            if source.open():
                self._input_source = source
                self._input_config.set_file_source(filepath)
                self._statusbar.showMessage(f"Loaded: {filepath}")
            else:
                QMessageBox.warning(
                    self, "File Error",
                    f"Could not open file: {filepath}"
                )

    def _reset_views(self):
        """Reset all display views."""
        self._channel_display.reset_view()
        self._scattering.reset_view()
        self._input_spectrum.reset_view()
        self._output_spectrum.reset_view()

    def _show_gpu_info(self):
        """Show GPU information dialog."""
        try:
            from hfpathsim.gpu import get_device_info

            info = get_device_info()
            msg = (
                f"GPU: {info['name']}\n"
                f"Compute Capability: {info['compute_capability']}\n"
                f"Memory: {info['total_memory_gb']:.1f} GB\n"
                f"Multiprocessors: {info['multiprocessors']}\n"
                f"Backend: {info['backend']}"
            )
        except Exception as e:
            msg = f"Error getting GPU info: {e}"

        QMessageBox.information(self, "GPU Information", msg)

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About HF Path Simulator",
            "HF Path Simulator v0.3.0\n\n"
            "Multiple HF Channel Models\n"
            "with RTX GPU acceleration\n\n"
            "Features:\n"
            "- Vogler IPM (ray-based) channel model\n"
            "- Watterson TDL channel model\n"
            "- Vogler-Hoffmeyer wideband stochastic model\n"
            "- Noise injection (AWGN, atmospheric, man-made)\n"
            "- AGC, limiter, frequency offset\n"
            "- Ray tracing and ionospheric modeling\n"
            "- Channel state recording/playback\n\n"
            "Based on NTIA TR-88-240, TR-90-255, and ITU-R F.1487"
        )

    def closeEvent(self, event):
        """Handle window close."""
        self._stop_processing()

        if self._input_source and self._input_source.is_open:
            self._input_source.close()

        if self._output_sink and self._output_sink.is_open:
            self._output_sink.close()

        # Stop any timers
        self._process_timer.stop()
        self._meter_timer.stop()

        # Stop GIRO auto-update if running
        if hasattr(self._ionosphere_panel, '_giro_timer'):
            self._ionosphere_panel._giro_timer.stop()

        event.accept()
