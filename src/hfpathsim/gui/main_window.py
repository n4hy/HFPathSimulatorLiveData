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
    QGroupBox,
    QGridLayout,
    QDoubleSpinBox,
    QCheckBox,
    QFileDialog,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QIcon

from hfpathsim.core.channel import HFChannel, ProcessingConfig
from hfpathsim.core.parameters import VoglerParameters, ITUCondition
from hfpathsim.input.base import InputSource
from hfpathsim.input.file import FileInputSource

from .widgets.channel_display import ChannelDisplayWidget
from .widgets.scattering import ScatteringWidget
from .widgets.spectrum import SpectrumWidget
from .widgets.parameters import ParameterPanel
from .widgets.input_config import InputConfigWidget


class MainWindow(QMainWindow):
    """Main application window for HF Path Simulator."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("HF Path Simulator")
        self.setMinimumSize(1280, 800)

        # Core components
        self._channel: Optional[HFChannel] = None
        self._input_source: Optional[InputSource] = None
        self._running = False

        # Processing timer
        self._process_timer = QTimer()
        self._process_timer.timeout.connect(self._process_block)

        # Setup UI
        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()

        # Initialize channel
        self._init_channel()

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
        display_splitter.setSizes([640, 640])

        main_layout.addWidget(display_splitter, stretch=3)

        # Bottom section: Controls
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        # Input configuration
        self._input_config = InputConfigWidget()
        self._input_config.source_changed.connect(self._on_input_changed)
        self._input_config.start_requested.connect(self._start_processing)
        self._input_config.stop_requested.connect(self._stop_processing)

        # Parameter panel
        self._param_panel = ParameterPanel()
        self._param_panel.parameters_changed.connect(self._on_parameters_changed)

        controls_layout.addWidget(self._input_config, stretch=1)
        controls_layout.addWidget(self._param_panel, stretch=2)

        main_layout.addWidget(controls_widget, stretch=0)

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

    def _setup_statusbar(self):
        """Setup status bar."""
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)

        # Permanent widgets
        self._gpu_label = QLabel("GPU: Checking...")
        self._statusbar.addPermanentWidget(self._gpu_label)

        self._rate_label = QLabel("Rate: 0 Msps")
        self._statusbar.addPermanentWidget(self._rate_label)

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
            else:
                self._gpu_label.setText("GPU: Not available (CPU mode)")
        except Exception as e:
            self._gpu_label.setText(f"GPU: Error - {e}")

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
        """
        self.setStyleSheet(style)

    def _init_channel(self):
        """Initialize the HF channel."""
        params = VoglerParameters.from_itu_condition(ITUCondition.MODERATE)
        config = ProcessingConfig()

        self._channel = HFChannel(params, config, use_gpu=True)
        self._channel.add_state_callback(self._on_channel_state)

        # Trigger initial state update
        state = self._channel.get_state()
        self._on_channel_state(state)

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

    def _on_input_changed(self, source: InputSource):
        """Handle input source change."""
        self._input_source = source
        self._statusbar.showMessage(
            f"Input: {type(source).__name__} @ {source.sample_rate/1e6:.1f} Msps"
        )

    def _on_parameters_changed(self, params: VoglerParameters):
        """Handle parameter changes."""
        if self._channel:
            self._channel.update_parameters(params)

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
            self._param_panel.set_parameters(params)

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

    def _stop_processing(self):
        """Stop real-time processing."""
        if not self._running:
            return

        self._running = False
        self._process_timer.stop()

        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._statusbar.showMessage("Stopped")

    def _process_block(self):
        """Process one block of samples."""
        if not self._running or self._input_source is None:
            return

        # Read samples
        block_size = 4096
        samples = self._input_source.read(block_size)

        if samples is None or len(samples) == 0:
            return

        # Update input spectrum
        self._input_spectrum.update_data(samples)

        # Process through channel
        if self._channel:
            output = self._channel.process(samples)
            self._output_spectrum.update_data(output)

            # Update rate display
            rate = self._input_source.sample_rate / 1e6
            self._rate_label.setText(f"Rate: {rate:.1f} Msps")

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
            "HF Path Simulator v0.1.0\n\n"
            "Vogler-Hoffmeyer Ionospheric Propagation Model\n"
            "with RTX GPU acceleration\n\n"
            "Based on NTIA TR-88-240 and ITU-R F.1487"
        )

    def closeEvent(self, event):
        """Handle window close."""
        self._stop_processing()

        if self._input_source and self._input_source.is_open:
            self._input_source.close()

        event.accept()
