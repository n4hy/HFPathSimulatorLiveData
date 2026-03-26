"""Output sink configuration widget."""

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
    QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal

from hfpathsim.output.base import OutputSink, OutputFormat
from hfpathsim.output.file import FileOutputSink
from hfpathsim.output.network import NetworkOutputSink, NetworkProtocol
from hfpathsim.output.audio import AudioOutputSink
from hfpathsim.output.sdr import SDROutputSink
from hfpathsim.output.multiplex import MultiplexOutputSink


class OutputConfigWidget(QWidget):
    """Widget for configuring output sinks."""

    sink_changed = pyqtSignal(object)  # OutputSink
    output_enabled = pyqtSignal(bool)
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._current_sink: Optional[OutputSink] = None
        self._enabled = False
        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Output sink group
        group = QGroupBox("Output Sink")
        group_layout = QVBoxLayout(group)

        # Enable checkbox and type selector
        top_row = QHBoxLayout()

        self._enable_check = QCheckBox("Enable Output")
        self._enable_check.setStyleSheet("""
            QCheckBox {
                color: #ffffff;
                font-weight: bold;
                padding: 4px 8px;
                background-color: #2d2d2d;
                border: 1px solid #555555;
                border-radius: 4px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #888888;
                border-radius: 4px;
                background-color: #3c3c3c;
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
                border-color: #4CAF50;
            }
            QCheckBox::indicator:hover {
                border-color: #4CAF50;
            }
        """)
        self._enable_check.stateChanged.connect(self._on_enable_changed)
        top_row.addWidget(self._enable_check)

        top_row.addWidget(QLabel("Type:"))

        self._type_combo = QComboBox()
        self._type_combo.addItems(["Network", "File", "Audio", "SDR"])
        self._type_combo.currentTextChanged.connect(self._on_type_changed)
        top_row.addWidget(self._type_combo)

        top_row.addStretch()

        # Apply button
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._on_apply)
        top_row.addWidget(self._apply_btn)

        group_layout.addLayout(top_row)

        # Stacked widget for different sink configs
        self._stack = QStackedWidget()
        group_layout.addWidget(self._stack)

        # Network config
        self._network_widget = self._create_network_config()
        self._stack.addWidget(self._network_widget)

        # File config
        self._file_widget = self._create_file_config()
        self._stack.addWidget(self._file_widget)

        # Audio config
        self._audio_widget = self._create_audio_config()
        self._stack.addWidget(self._audio_widget)

        # SDR config
        self._sdr_widget = self._create_sdr_config()
        self._stack.addWidget(self._sdr_widget)

        layout.addWidget(group)

        # Status section
        status_group = QGroupBox("Output Status")
        status_layout = QGridLayout(status_group)

        status_layout.addWidget(QLabel("Samples Written:"), 0, 0)
        self._samples_label = QLabel("0")
        status_layout.addWidget(self._samples_label, 0, 1)

        status_layout.addWidget(QLabel("Buffer Fill:"), 0, 2)
        self._buffer_label = QLabel("0%")
        status_layout.addWidget(self._buffer_label, 0, 3)

        status_layout.addWidget(QLabel("Status:"), 1, 0)
        self._status_label = QLabel("Not configured")
        status_layout.addWidget(self._status_label, 1, 1, 1, 3)

        layout.addWidget(status_group)

    def _create_network_config(self) -> QWidget:
        """Create network sink configuration panel."""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)

        # Protocol
        layout.addWidget(QLabel("Protocol:"), 0, 0)
        self._net_protocol = QComboBox()
        self._net_protocol.addItems(["ZMQ (GNU Radio)", "TCP Server", "UDP"])
        layout.addWidget(self._net_protocol, 0, 1)

        # Host (bind address)
        layout.addWidget(QLabel("Bind Address:"), 0, 2)
        self._net_host = QLineEdit("0.0.0.0")
        layout.addWidget(self._net_host, 0, 3)

        # Port
        layout.addWidget(QLabel("Port:"), 1, 0)
        self._net_port = QSpinBox()
        self._net_port.setRange(1, 65535)
        self._net_port.setValue(5556)
        layout.addWidget(self._net_port, 1, 1)

        # Sample rate
        layout.addWidget(QLabel("Sample Rate:"), 1, 2)
        self._net_rate = QDoubleSpinBox()
        self._net_rate.setRange(0.001, 100.0)
        self._net_rate.setValue(2.0)
        self._net_rate.setSuffix(" Msps")
        layout.addWidget(self._net_rate, 1, 3)

        # Format
        layout.addWidget(QLabel("Format:"), 2, 0)
        self._net_format = QComboBox()
        self._net_format.addItems([
            "Complex64 (GR default)",
            "Int16 I/Q",
            "Float32 I/Q",
        ])
        layout.addWidget(self._net_format, 2, 1)

        # ZMQ connection string display
        layout.addWidget(QLabel("Connect:"), 2, 2)
        self._zmq_connect = QLineEdit()
        self._zmq_connect.setReadOnly(True)
        self._zmq_connect.setText("tcp://127.0.0.1:5556")
        layout.addWidget(self._zmq_connect, 2, 3)

        # Update connect string when port changes
        self._net_port.valueChanged.connect(self._update_zmq_connect)
        self._update_zmq_connect()

        return widget

    def _update_zmq_connect(self):
        """Update ZMQ connection string display."""
        port = self._net_port.value()
        self._zmq_connect.setText(f"tcp://127.0.0.1:{port}")

    def _create_file_config(self) -> QWidget:
        """Create file sink configuration panel."""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)

        # File path
        layout.addWidget(QLabel("File:"), 0, 0)
        self._file_path = QLineEdit()
        self._file_path.setPlaceholderText("output.sigmf-data")
        layout.addWidget(self._file_path, 0, 1)

        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.clicked.connect(self._browse_output_file)
        layout.addWidget(self._browse_btn, 0, 2)

        # Sample rate
        layout.addWidget(QLabel("Sample Rate:"), 1, 0)
        self._file_rate = QDoubleSpinBox()
        self._file_rate.setRange(0.001, 100.0)
        self._file_rate.setValue(2.0)
        self._file_rate.setSuffix(" Msps")
        layout.addWidget(self._file_rate, 1, 1)

        # Format
        layout.addWidget(QLabel("Format:"), 1, 2)
        self._file_format = QComboBox()
        self._file_format.addItems([
            "SigMF (recommended)",
            "WAV (stereo I/Q)",
            "Raw Complex64",
            "Raw Int16 I/Q",
        ])
        layout.addWidget(self._file_format, 1, 3)

        # Center frequency (for metadata)
        layout.addWidget(QLabel("Center Freq:"), 2, 0)
        self._file_freq = QDoubleSpinBox()
        self._file_freq.setRange(0.0, 30000.0)
        self._file_freq.setValue(10.0)
        self._file_freq.setSuffix(" MHz")
        layout.addWidget(self._file_freq, 2, 1)

        return widget

    def _browse_output_file(self):
        """Browse for output file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Select Output File",
            "",
            "SigMF (*.sigmf-data);;WAV (*.wav);;Raw (*.raw *.bin *.cf32);;All Files (*)",
        )

        if filepath:
            self._file_path.setText(filepath)

    def _create_audio_config(self) -> QWidget:
        """Create audio sink configuration panel."""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)

        # Device - prominent selection
        device_label = QLabel("Audio Device:")
        device_label.setStyleSheet("font-weight: bold; color: #ffffff;")
        layout.addWidget(device_label, 0, 0)

        self._audio_device = QComboBox()
        self._audio_device.setMinimumWidth(300)
        self._audio_device.setStyleSheet("""
            QComboBox {
                padding: 6px;
                font-size: 12px;
                background-color: #3c3c3c;
                border: 2px solid #0e639c;
                border-radius: 4px;
            }
            QComboBox:hover {
                border-color: #4CAF50;
            }
            QComboBox::drop-down {
                width: 30px;
            }
        """)
        layout.addWidget(self._audio_device, 0, 1, 1, 2)

        self._audio_scan_btn = QPushButton("Refresh")
        self._audio_scan_btn.clicked.connect(self._scan_audio_devices)
        layout.addWidget(self._audio_scan_btn, 0, 3)

        # Sample rate
        layout.addWidget(QLabel("Sample Rate:"), 1, 0)
        self._audio_rate = QComboBox()
        self._audio_rate.addItems(["8000 Hz", "11025 Hz", "16000 Hz", "22050 Hz", "44100 Hz", "48000 Hz", "96000 Hz", "192000 Hz"])
        self._audio_rate.setCurrentText("8000 Hz")  # Default to match signal generator
        layout.addWidget(self._audio_rate, 1, 1)

        # Latency
        layout.addWidget(QLabel("Latency:"), 1, 2)
        self._audio_latency = QComboBox()
        self._audio_latency.addItems(["Low", "High"])
        layout.addWidget(self._audio_latency, 1, 3)

        return widget

    def _scan_audio_devices(self):
        """Scan for audio output devices."""
        self._audio_device.clear()
        self._audio_device.addItem("(Default)", None)

        try:
            devices = AudioOutputSink.list_devices()
            for dev in devices:
                label = f"{dev['name']} ({dev['hostapi']})"
                self._audio_device.addItem(label, dev["index"])
        except Exception:
            pass

    def _create_sdr_config(self) -> QWidget:
        """Create SDR TX sink configuration panel."""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)

        # Device
        layout.addWidget(QLabel("Device:"), 0, 0)
        self._sdr_device = QComboBox()
        self._sdr_device.addItem("(Scan for devices)")
        layout.addWidget(self._sdr_device, 0, 1)

        self._sdr_scan_btn = QPushButton("Scan")
        self._sdr_scan_btn.clicked.connect(self._scan_sdr_devices)
        layout.addWidget(self._sdr_scan_btn, 0, 2)

        # Sample rate
        layout.addWidget(QLabel("Sample Rate:"), 1, 0)
        self._sdr_rate = QComboBox()
        self._sdr_rate.addItems(["250 ksps", "1 Msps", "2 Msps", "2.4 Msps"])
        self._sdr_rate.setCurrentText("2 Msps")
        layout.addWidget(self._sdr_rate, 1, 1)

        # Center frequency
        layout.addWidget(QLabel("TX Freq:"), 1, 2)
        self._sdr_freq = QDoubleSpinBox()
        self._sdr_freq.setRange(0.1, 6000.0)
        self._sdr_freq.setValue(10.0)
        self._sdr_freq.setSuffix(" MHz")
        layout.addWidget(self._sdr_freq, 1, 3)

        # TX Gain
        layout.addWidget(QLabel("TX Gain:"), 2, 0)
        self._sdr_gain = QDoubleSpinBox()
        self._sdr_gain.setRange(0.0, 60.0)
        self._sdr_gain.setValue(40.0)
        self._sdr_gain.setSuffix(" dB")
        layout.addWidget(self._sdr_gain, 2, 1)

        return widget

    def _scan_sdr_devices(self):
        """Scan for SDR TX devices."""
        self._sdr_device.clear()

        try:
            devices = SDROutputSink.enumerate_devices()
            if devices:
                for dev in devices:
                    label = f"{dev['driver']}: {dev['label']}"
                    self._sdr_device.addItem(label, dev)
            else:
                self._sdr_device.addItem("(No TX devices found)")
        except Exception:
            self._sdr_device.addItem("(SoapySDR not installed)")

    def _on_type_changed(self, type_name: str):
        """Handle sink type change."""
        index = {"Network": 0, "File": 1, "Audio": 2, "SDR": 3}.get(type_name, 0)
        self._stack.setCurrentIndex(index)

        # Auto-scan audio devices when Audio is selected
        if type_name == "Audio":
            self._scan_audio_devices()

    def _on_enable_changed(self, state):
        """Handle enable checkbox change."""
        self._enabled = state == Qt.CheckState.Checked.value
        self.output_enabled.emit(self._enabled)

    def _on_apply(self):
        """Apply current configuration and create sink."""
        sink_type = self._type_combo.currentText()

        if sink_type == "Network":
            self._current_sink = self._create_network_sink()
        elif sink_type == "File":
            self._current_sink = self._create_file_sink()
        elif sink_type == "Audio":
            self._current_sink = self._create_audio_sink()
        elif sink_type == "SDR":
            self._current_sink = self._create_sdr_sink()

        if self._current_sink:
            self.sink_changed.emit(self._current_sink)
            self._status_label.setText(f"Configured: {sink_type}")

    def _create_network_sink(self) -> Optional[OutputSink]:
        """Create network output sink from current settings."""
        protocol_map = {
            "ZMQ (GNU Radio)": NetworkProtocol.ZMQ_PUB,
            "TCP Server": NetworkProtocol.TCP,
            "UDP": NetworkProtocol.UDP,
        }

        format_map = {
            "Complex64 (GR default)": OutputFormat.COMPLEX64,
            "Int16 I/Q": OutputFormat.INT16_IQ,
            "Float32 I/Q": OutputFormat.FLOAT32_IQ,
        }

        return NetworkOutputSink(
            host=self._net_host.text(),
            port=self._net_port.value(),
            protocol=protocol_map[self._net_protocol.currentText()],
            sample_rate_hz=self._net_rate.value() * 1e6,
            output_format=format_map[self._net_format.currentText()],
        )

    def _create_file_sink(self) -> Optional[OutputSink]:
        """Create file output sink from current settings."""
        filepath = self._file_path.text()
        if not filepath:
            return None

        format_map = {
            "SigMF (recommended)": OutputFormat.COMPLEX64,
            "WAV (stereo I/Q)": OutputFormat.INT16_IQ,
            "Raw Complex64": OutputFormat.COMPLEX64,
            "Raw Int16 I/Q": OutputFormat.INT16_IQ,
        }

        # Adjust extension if needed
        fmt_text = self._file_format.currentText()
        if fmt_text == "SigMF (recommended)" and not filepath.endswith(".sigmf-data"):
            filepath = filepath.rsplit(".", 1)[0] + ".sigmf-data"
        elif fmt_text == "WAV (stereo I/Q)" and not filepath.endswith(".wav"):
            filepath = filepath.rsplit(".", 1)[0] + ".wav"

        return FileOutputSink(
            filepath=filepath,
            sample_rate_hz=self._file_rate.value() * 1e6,
            center_freq_hz=self._file_freq.value() * 1e6,
            output_format=format_map[fmt_text],
        )

    def _create_audio_sink(self) -> Optional[OutputSink]:
        """Create audio output sink from current settings."""
        device = self._audio_device.currentData()

        rate_map = {
            "8000 Hz": 8000,
            "11025 Hz": 11025,
            "16000 Hz": 16000,
            "22050 Hz": 22050,
            "44100 Hz": 44100,
            "48000 Hz": 48000,
            "96000 Hz": 96000,
            "192000 Hz": 192000,
        }

        return AudioOutputSink(
            device=device,
            sample_rate_hz=rate_map[self._audio_rate.currentText()],
            latency=self._audio_latency.currentText().lower(),
        )

    def _create_sdr_sink(self) -> Optional[OutputSink]:
        """Create SDR TX output sink from current settings."""
        device_data = self._sdr_device.currentData()
        if not device_data:
            return None

        rate_map = {
            "250 ksps": 250000,
            "1 Msps": 1000000,
            "2 Msps": 2000000,
            "2.4 Msps": 2400000,
        }

        device_args = device_data.get("args", {})
        if isinstance(device_args, dict):
            device_args = ",".join(f"{k}={v}" for k, v in device_args.items())

        return SDROutputSink(
            device_args=device_args,
            sample_rate_hz=rate_map[self._sdr_rate.currentText()],
            center_freq_hz=self._sdr_freq.value() * 1e6,
            tx_gain=self._sdr_gain.value(),
        )

    def is_output_enabled(self) -> bool:
        """Return whether output is enabled."""
        return self._enabled

    def get_current_sink(self) -> Optional[OutputSink]:
        """Get current output sink."""
        return self._current_sink

    def update_status(self, samples_written: int, buffer_fill: float):
        """Update status display.

        Args:
            samples_written: Total samples written
            buffer_fill: Buffer fill percentage
        """
        self._samples_label.setText(f"{samples_written:,}")
        self._buffer_label.setText(f"{buffer_fill:.1f}%")

    def set_status_message(self, message: str):
        """Set status message.

        Args:
            message: Status message to display
        """
        self._status_label.setText(message)
