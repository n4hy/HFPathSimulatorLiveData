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
from hfpathsim.input.flexradio import FlexRadioInputSource
from hfpathsim.input.siggen import SignalGenerator, WaveformType


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
        self._type_combo.addItems(["File", "Network", "SDR", "Flex Radio", "Signal Generator"])
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

        # Flex Radio config
        self._flex_widget = self._create_flex_config()
        self._stack.addWidget(self._flex_widget)

        # Signal Generator config
        self._siggen_widget = self._create_siggen_config()
        self._stack.addWidget(self._siggen_widget)

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

    def _create_flex_config(self) -> QWidget:
        """Create Flex Radio DAX IQ source configuration panel."""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)

        # Radio host/IP
        layout.addWidget(QLabel("Radio IP:"), 0, 0)
        self._flex_host = QLineEdit("192.168.1.100")
        self._flex_host.setPlaceholderText("e.g., 192.168.1.100")
        layout.addWidget(self._flex_host, 0, 1)

        # Discover button
        self._flex_discover_btn = QPushButton("Discover")
        self._flex_discover_btn.clicked.connect(self._discover_flex)
        layout.addWidget(self._flex_discover_btn, 0, 2)

        # Discovered radios combo
        layout.addWidget(QLabel("Radio:"), 0, 3)
        self._flex_radio_combo = QComboBox()
        self._flex_radio_combo.addItem("(Enter IP or Discover)")
        self._flex_radio_combo.currentIndexChanged.connect(self._on_flex_radio_selected)
        layout.addWidget(self._flex_radio_combo, 0, 4)

        # DAX channel
        layout.addWidget(QLabel("DAX Channel:"), 1, 0)
        self._flex_dax = QSpinBox()
        self._flex_dax.setRange(1, 8)
        self._flex_dax.setValue(1)
        layout.addWidget(self._flex_dax, 1, 1)

        # Sample rate
        layout.addWidget(QLabel("Sample Rate:"), 1, 2)
        self._flex_rate = QComboBox()
        self._flex_rate.addItems(["24 kHz", "48 kHz", "96 kHz", "192 kHz"])
        self._flex_rate.setCurrentText("48 kHz")
        layout.addWidget(self._flex_rate, 1, 3, 1, 2)

        # Slice number (for frequency control)
        layout.addWidget(QLabel("Slice:"), 2, 0)
        self._flex_slice = QSpinBox()
        self._flex_slice.setRange(0, 7)
        self._flex_slice.setValue(0)
        layout.addWidget(self._flex_slice, 2, 1)

        # Center frequency
        layout.addWidget(QLabel("Frequency:"), 2, 2)
        self._flex_freq = QDoubleSpinBox()
        self._flex_freq.setRange(0.1, 54.0)
        self._flex_freq.setValue(14.0)
        self._flex_freq.setDecimals(6)
        self._flex_freq.setSuffix(" MHz")
        layout.addWidget(self._flex_freq, 2, 3, 1, 2)

        # Status label
        self._flex_status = QLabel("Not connected")
        layout.addWidget(self._flex_status, 3, 0, 1, 5)

        return widget

    def _create_siggen_config(self) -> QWidget:
        """Create signal generator configuration panel."""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)

        # Waveform type
        layout.addWidget(QLabel("Waveform:"), 0, 0)
        self._siggen_waveform = QComboBox()
        self._siggen_waveform.addItems(["RTTY", "SSB Voice", "PSK31"])
        layout.addWidget(self._siggen_waveform, 0, 1)

        # Sample rate
        layout.addWidget(QLabel("Sample Rate:"), 0, 2)
        self._siggen_rate = QDoubleSpinBox()
        self._siggen_rate.setRange(1.0, 96.0)
        self._siggen_rate.setValue(8.0)
        self._siggen_rate.setSuffix(" kHz")
        self._siggen_rate.setDecimals(1)
        layout.addWidget(self._siggen_rate, 0, 3)

        # Center frequency
        layout.addWidget(QLabel("Center Freq:"), 1, 0)
        self._siggen_freq = QDoubleSpinBox()
        self._siggen_freq.setRange(100.0, 5000.0)
        self._siggen_freq.setValue(1500.0)
        self._siggen_freq.setSuffix(" Hz")
        self._siggen_freq.setDecimals(1)
        layout.addWidget(self._siggen_freq, 1, 1)

        # Duration
        layout.addWidget(QLabel("Duration:"), 1, 2)
        self._siggen_duration = QDoubleSpinBox()
        self._siggen_duration.setRange(10.0, 300.0)
        self._siggen_duration.setValue(60.0)
        self._siggen_duration.setSuffix(" sec")
        self._siggen_duration.setDecimals(0)
        layout.addWidget(self._siggen_duration, 1, 3)

        return widget

    def _discover_flex(self):
        """Discover Flex Radio devices on the network."""
        self._flex_radio_combo.clear()
        self._flex_radio_combo.addItem("(Searching...)")
        self._flex_discover_btn.setEnabled(False)

        try:
            radios = FlexRadioInputSource.discover_radios(timeout=3.0)
            self._flex_radio_combo.clear()

            if radios:
                for radio in radios:
                    label = f"{radio.get('nickname', radio.get('model', 'Unknown'))} ({radio.get('ip', '')})"
                    self._flex_radio_combo.addItem(label, radio)
                self._flex_status.setText(f"Found {len(radios)} radio(s)")
            else:
                self._flex_radio_combo.addItem("(No radios found)")
                self._flex_status.setText("No Flex Radios found on network")
        except Exception as e:
            self._flex_radio_combo.clear()
            self._flex_radio_combo.addItem("(Discovery failed)")
            self._flex_status.setText(f"Discovery error: {e}")
        finally:
            self._flex_discover_btn.setEnabled(True)

    def _on_flex_radio_selected(self, index: int):
        """Handle Flex Radio selection from discovered list."""
        radio_data = self._flex_radio_combo.currentData()
        if radio_data and isinstance(radio_data, dict):
            ip = radio_data.get("ip", "")
            if ip:
                self._flex_host.setText(ip)
                self._flex_status.setText(f"Selected: {radio_data.get('nickname', radio_data.get('model', 'Unknown'))}")

    def _create_flex_source(self) -> Optional[InputSource]:
        """Create Flex Radio input source from current settings."""
        # Parse sample rate
        rate_text = self._flex_rate.currentText()
        rate_map = {
            "24 kHz": 24000,
            "48 kHz": 48000,
            "96 kHz": 96000,
            "192 kHz": 192000,
        }
        sample_rate = rate_map.get(rate_text, 48000)

        source = FlexRadioInputSource(
            host=self._flex_host.text(),
            dax_channel=self._flex_dax.value(),
            sample_rate_hz=sample_rate,
            center_freq_hz=self._flex_freq.value() * 1e6,
        )

        return source

    def _on_type_changed(self, type_name: str):
        """Handle source type change."""
        index = {"File": 0, "Network": 1, "SDR": 2, "Flex Radio": 3, "Signal Generator": 4}.get(type_name, 0)
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

    def _create_siggen_source(self) -> Optional[InputSource]:
        """Create signal generator source from current settings."""
        waveform_map = {
            "RTTY": WaveformType.RTTY,
            "SSB Voice": WaveformType.SSB_VOICE,
            "PSK31": WaveformType.PSK31,
        }

        waveform = waveform_map.get(self._siggen_waveform.currentText(), WaveformType.RTTY)

        source = SignalGenerator(
            waveform=waveform,
            sample_rate_hz=self._siggen_rate.value() * 1000.0,  # kHz to Hz
            center_freq_hz=self._siggen_freq.value(),
            duration_sec=self._siggen_duration.value(),
        )

        return source

    def _on_start(self):
        """Handle start button click."""
        # Create source based on current type
        source_type = self._type_combo.currentText()

        if source_type == "File":
            self._current_source = self._create_file_source()
        elif source_type == "Network":
            self._current_source = self._create_network_source()
        elif source_type == "Flex Radio":
            self._current_source = self._create_flex_source()
            if hasattr(self, '_flex_status'):
                self._flex_status.setText("Connecting...")
        elif source_type == "Signal Generator":
            self._current_source = self._create_siggen_source()
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
