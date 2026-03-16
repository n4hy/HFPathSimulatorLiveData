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

from .widgets.channel_display import ChannelDisplayWidget
from .widgets.scattering import ScatteringWidget
from .widgets.spectrum import SpectrumWidget
from .widgets.control_tabs import ControlTabWidget
from .widgets.input_config import InputConfigWidget
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
            self._recorder.capture()
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
            self._noise_generator = NoiseGenerator(config)
            self._snr_label.setText(f"SNR: {config.snr_db:.1f} dB")

    def _on_agc_config_changed(self, config: AGCConfig):
        """Handle AGC configuration change."""
        if self._agc:
            self._agc = AGC(config)

    def _on_limiter_config_changed(self, config: LimiterConfig):
        """Handle limiter configuration change."""
        if self._limiter:
            self._limiter = Limiter(config)

    def _on_freq_offset_config_changed(self, config: FrequencyOffsetConfig):
        """Handle frequency offset configuration change."""
        if self._freq_offset:
            self._freq_offset = FrequencyOffset(config)

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

        self._running = True
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._statusbar.showMessage("Processing...")

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

        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._statusbar.showMessage("Stopped")

    def _process_block(self):
        """Process one block of samples through the full chain.

        Processing order:
        Input → Channel → Noise → AGC → Limiter → Freq Offset → Output
        """
        if not self._running or self._input_source is None:
            return

        # Read samples
        block_size = 4096
        samples = self._input_source.read(block_size)

        if samples is None or len(samples) == 0:
            return

        # Update input spectrum
        self._input_spectrum.update_data(samples)

        # Process through channel model
        if self._current_model == "watterson" and self._watterson_channel:
            channel_output = self._watterson_channel.process_block(samples)
        elif self._current_model == "vogler_hoffmeyer" and self._vh_channel:
            channel_output = self._vh_channel.process(samples)
        elif self._current_model == "vogler" and self._channel:
            channel_output = self._channel.process(samples)
        else:
            channel_output = samples

        # Apply noise if enabled
        if self._noise_panel.is_noise_enabled() and self._noise_generator:
            channel_output = self._noise_generator.add_noise(channel_output)

        # Apply AGC if enabled
        if self._impairments_panel.is_agc_enabled() and self._agc:
            channel_output = self._agc.process_block(channel_output)

        # Apply limiter if enabled
        if self._impairments_panel.is_limiter_enabled() and self._limiter:
            channel_output = self._limiter.process(channel_output)

        # Apply frequency offset if enabled
        if self._impairments_panel.is_freq_offset_enabled() and self._freq_offset:
            channel_output = self._freq_offset.process(channel_output)

        # Update output spectrum
        self._output_spectrum.update_data(channel_output)

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

        # Stop any timers
        self._process_timer.stop()
        self._meter_timer.stop()

        # Stop GIRO auto-update if running
        if hasattr(self._ionosphere_panel, '_giro_timer'):
            self._ionosphere_panel._giro_timer.stop()

        event.accept()
