"""Input source configuration widget."""

from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QStackedWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal

from hfpathsim.input.base import InputSource, InputFormat
from hfpathsim.input.file import FileInputSource
from hfpathsim.input.network import NetworkInputSource, NetworkProtocol


class InputConfigWidget(QWidget):
    """Widget for configuring input sources."""

    source_changed = pyqtSignal(object)  # InputSource
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._current_source: Optional[InputSource] = None
        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Input source group
        group = QGroupBox("Input Source")
        group_layout = QVBoxLayout(group)

        # Source type selector
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))

        self._type_combo = QComboBox()
        self._type_combo.addItems(["File", "Network", "SDR"])
        self._type_combo.currentTextChanged.connect(self._on_type_changed)
        type_row.addWidget(self._type_combo)

        type_row.addStretch()

        # Start/Stop buttons
        self._start_btn = QPushButton("Start")
        self._start_btn.clicked.connect(self._on_start)
        type_row.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setEnabled(False)
        type_row.addWidget(self._stop_btn)

        group_layout.addLayout(type_row)

        # Stacked widget for different source configs
        self._stack = QStackedWidget()
        group_layout.addWidget(self._stack)

        # File config
        self._file_widget = self._create_file_config()
        self._stack.addWidget(self._file_widget)

        # Network config
        self._network_widget = self._create_network_config()
        self._stack.addWidget(self._network_widget)

        # SDR config
        self._sdr_widget = self._create_sdr_config()
        self._stack.addWidget(self._sdr_widget)

        layout.addWidget(group)

    def _create_file_config(self) -> QWidget:
        """Create file source configuration panel."""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)

        # File path
        layout.addWidget(QLabel("File:"), 0, 0)
        self._file_path = QLineEdit()
        self._file_path.setReadOnly(True)
        layout.addWidget(self._file_path, 0, 1)

        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.clicked.connect(self._browse_file)
        layout.addWidget(self._browse_btn, 0, 2)

        # Sample rate (for raw files)
        layout.addWidget(QLabel("Sample Rate:"), 1, 0)
        self._file_rate = QDoubleSpinBox()
        self._file_rate.setRange(0.001, 100.0)
        self._file_rate.setValue(2.0)
        self._file_rate.setSuffix(" Msps")
        self._file_rate.setSingleStep(0.1)
        layout.addWidget(self._file_rate, 1, 1)

        # Format (for raw files)
        layout.addWidget(QLabel("Format:"), 2, 0)
        self._file_format = QComboBox()
        self._file_format.addItems([
            "Auto",
            "Complex64",
            "Int16 I/Q",
            "Int8 I/Q",
            "Float32 I/Q",
        ])
        layout.addWidget(self._file_format, 2, 1)

        # Loop
        self._file_loop = QPushButton("Loop")
        self._file_loop.setCheckable(True)
        self._file_loop.setChecked(True)
        layout.addWidget(self._file_loop, 2, 2)

        return widget

    def _create_network_config(self) -> QWidget:
        """Create network source configuration panel."""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)

        # Protocol
        layout.addWidget(QLabel("Protocol:"), 0, 0)
        self._net_protocol = QComboBox()
        self._net_protocol.addItems(["TCP", "UDP", "ZMQ"])
        layout.addWidget(self._net_protocol, 0, 1)

        # Host
        layout.addWidget(QLabel("Host:"), 0, 2)
        self._net_host = QLineEdit("127.0.0.1")
        layout.addWidget(self._net_host, 0, 3)

        # Port
        layout.addWidget(QLabel("Port:"), 1, 0)
        self._net_port = QSpinBox()
        self._net_port.setRange(1, 65535)
        self._net_port.setValue(5555)
        layout.addWidget(self._net_port, 1, 1)

        # Sample rate
        layout.addWidget(QLabel("Sample Rate:"), 1, 2)
        self._net_rate = QDoubleSpinBox()
        self._net_rate.setRange(0.001, 100.0)
        self._net_rate.setValue(2.0)
        self._net_rate.setSuffix(" Msps")
        layout.addWidget(self._net_rate, 1, 3)

        return widget

    def _create_sdr_config(self) -> QWidget:
        """Create SDR source configuration panel."""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)

        # Device
        layout.addWidget(QLabel("Device:"), 0, 0)
        self._sdr_device = QComboBox()
        self._sdr_device.addItems(["(Scan for devices)"])
        layout.addWidget(self._sdr_device, 0, 1)

        self._sdr_scan_btn = QPushButton("Scan")
        self._sdr_scan_btn.clicked.connect(self._scan_sdr)
        layout.addWidget(self._sdr_scan_btn, 0, 2)

        # Sample rate
        layout.addWidget(QLabel("Sample Rate:"), 1, 0)
        self._sdr_rate = QComboBox()
        self._sdr_rate.addItems(["250 ksps", "1 Msps", "2 Msps", "2.4 Msps"])
        self._sdr_rate.setCurrentText("2 Msps")
        layout.addWidget(self._sdr_rate, 1, 1)

        # Center frequency
        layout.addWidget(QLabel("Center Freq:"), 1, 2)
        self._sdr_freq = QDoubleSpinBox()
        self._sdr_freq.setRange(0.1, 6000.0)
        self._sdr_freq.setValue(10.0)
        self._sdr_freq.setSuffix(" MHz")
        layout.addWidget(self._sdr_freq, 1, 3)

        # Gain
        layout.addWidget(QLabel("Gain:"), 2, 0)
        self._sdr_gain = QDoubleSpinBox()
        self._sdr_gain.setRange(0.0, 50.0)
        self._sdr_gain.setValue(40.0)
        self._sdr_gain.setSuffix(" dB")
        layout.addWidget(self._sdr_gain, 2, 1)

        return widget

    def _on_type_changed(self, type_name: str):
        """Handle source type change."""
        index = {"File": 0, "Network": 1, "SDR": 2}.get(type_name, 0)
        self._stack.setCurrentIndex(index)

    def _browse_file(self):
        """Browse for input file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select IQ File",
            "",
            "IQ Files (*.wav *.sigmf-data *.raw *.bin *.cf32);;All Files (*)",
        )

        if filepath:
            self._file_path.setText(filepath)
            self._create_file_source()

    def _create_file_source(self) -> Optional[InputSource]:
        """Create file input source from current settings."""
        filepath = self._file_path.text()
        if not filepath:
            return None

        # Determine format
        format_map = {
            "Auto": None,
            "Complex64": InputFormat.COMPLEX64,
            "Int16 I/Q": InputFormat.INT16_IQ,
            "Int8 I/Q": InputFormat.INT8_IQ,
            "Float32 I/Q": InputFormat.FLOAT32_IQ,
        }
        fmt = format_map.get(self._file_format.currentText())

        # Create source
        source = FileInputSource(
            filepath,
            sample_rate_hz=self._file_rate.value() * 1e6,
            input_format=fmt or InputFormat.COMPLEX64,
            loop=self._file_loop.isChecked(),
        )

        return source

    def _create_network_source(self) -> Optional[InputSource]:
        """Create network input source from current settings."""
        protocol_map = {
            "TCP": NetworkProtocol.TCP,
            "UDP": NetworkProtocol.UDP,
            "ZMQ": NetworkProtocol.ZMQ_SUB,
        }

        source = NetworkInputSource(
            host=self._net_host.text(),
            port=self._net_port.value(),
            protocol=protocol_map[self._net_protocol.currentText()],
            sample_rate_hz=self._net_rate.value() * 1e6,
        )

        return source

    def _scan_sdr(self):
        """Scan for SDR devices."""
        try:
            from hfpathsim.input.sdr import SDRInputSource

            devices = SDRInputSource.enumerate_devices()

            self._sdr_device.clear()
            if devices:
                for dev in devices:
                    label = f"{dev['driver']}: {dev['label']}"
                    self._sdr_device.addItem(label, dev)
            else:
                self._sdr_device.addItem("(No devices found)")

        except ImportError:
            self._sdr_device.clear()
            self._sdr_device.addItem("(SoapySDR not installed)")

    def _on_start(self):
        """Handle start button click."""
        # Create source based on current type
        source_type = self._type_combo.currentText()

        if source_type == "File":
            self._current_source = self._create_file_source()
        elif source_type == "Network":
            self._current_source = self._create_network_source()
        else:
            # SDR would go here
            pass

        if self._current_source:
            self.source_changed.emit(self._current_source)
            self.start_requested.emit()

            self._start_btn.setEnabled(False)
            self._stop_btn.setEnabled(True)

    def _on_stop(self):
        """Handle stop button click."""
        self.stop_requested.emit()

        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    def set_file_source(self, filepath: str):
        """Set file source path externally.

        Args:
            filepath: Path to IQ file
        """
        self._type_combo.setCurrentText("File")
        self._file_path.setText(filepath)

    def get_current_source(self) -> Optional[InputSource]:
        """Get current input source.

        Returns:
            Current InputSource or None
        """
        return self._current_source
