"""Noise configuration panel widget."""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QComboBox,
    QSlider,
    QPushButton,
    QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal

from hfpathsim.core.noise import NoiseConfig, ManMadeEnvironment


class NoisePanel(QWidget):
    """Panel for configuring noise injection.

    Supports:
    - AWGN with SNR control
    - Atmospheric noise (ITU-R P.372)
    - Man-made noise by environment
    - Impulse noise
    """

    noise_config_changed = pyqtSignal(NoiseConfig)
    snr_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._config = NoiseConfig()
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the noise panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Top row: Master enable and SNR
        top_row = QHBoxLayout()

        self._enable_noise_check = QCheckBox("Enable Noise")
        self._enable_noise_check.setChecked(True)
        top_row.addWidget(self._enable_noise_check)

        top_row.addSpacing(20)

        top_row.addWidget(QLabel("SNR:"))
        self._snr_slider = QSlider(Qt.Orientation.Horizontal)
        self._snr_slider.setRange(-20, 60)
        self._snr_slider.setValue(20)
        self._snr_slider.setFixedWidth(200)
        top_row.addWidget(self._snr_slider)

        self._snr_label = QLabel("20.0 dB")
        self._snr_label.setFixedWidth(60)
        top_row.addWidget(self._snr_label)

        top_row.addWidget(QLabel("Bandwidth:"))
        self._bandwidth_spin = QDoubleSpinBox()
        self._bandwidth_spin.setRange(100.0, 100000.0)
        self._bandwidth_spin.setValue(3000.0)
        self._bandwidth_spin.setSuffix(" Hz")
        self._bandwidth_spin.setSingleStep(100.0)
        top_row.addWidget(self._bandwidth_spin)

        top_row.addStretch()

        self._apply_btn = QPushButton("Apply")
        top_row.addWidget(self._apply_btn)

        layout.addLayout(top_row)

        # Main content: Two-column layout
        content_layout = QHBoxLayout()

        # Left column: AWGN and Atmospheric
        left_column = QVBoxLayout()

        # AWGN group
        awgn_group = QGroupBox("AWGN")
        awgn_layout = QVBoxLayout(awgn_group)

        self._awgn_check = QCheckBox("Enable AWGN")
        self._awgn_check.setChecked(True)
        awgn_layout.addWidget(self._awgn_check)

        awgn_layout.addWidget(QLabel("White Gaussian noise at specified SNR"))

        left_column.addWidget(awgn_group)

        # Atmospheric noise group
        atmos_group = QGroupBox("Atmospheric (ITU-R P.372)")
        atmos_layout = QGridLayout(atmos_group)

        self._atmos_check = QCheckBox("Enable")
        atmos_layout.addWidget(self._atmos_check, 0, 0, 1, 2)

        atmos_layout.addWidget(QLabel("Frequency:"), 1, 0)
        self._atmos_freq_spin = QDoubleSpinBox()
        self._atmos_freq_spin.setRange(0.1, 30.0)
        self._atmos_freq_spin.setValue(10.0)
        self._atmos_freq_spin.setSuffix(" MHz")
        self._atmos_freq_spin.setSingleStep(0.5)
        atmos_layout.addWidget(self._atmos_freq_spin, 1, 1)

        atmos_layout.addWidget(QLabel("Season:"), 2, 0)
        self._season_combo = QComboBox()
        self._season_combo.addItems(["Summer", "Winter"])
        atmos_layout.addWidget(self._season_combo, 2, 1)

        atmos_layout.addWidget(QLabel("Time:"), 3, 0)
        self._time_combo = QComboBox()
        self._time_combo.addItems(["Day", "Night"])
        atmos_layout.addWidget(self._time_combo, 3, 1)

        atmos_layout.addWidget(QLabel("Latitude:"), 4, 0)
        self._latitude_spin = QDoubleSpinBox()
        self._latitude_spin.setRange(-90.0, 90.0)
        self._latitude_spin.setValue(45.0)
        self._latitude_spin.setSuffix(" °")
        self._latitude_spin.setSingleStep(5.0)
        atmos_layout.addWidget(self._latitude_spin, 4, 1)

        left_column.addWidget(atmos_group)
        left_column.addStretch()

        content_layout.addLayout(left_column)

        # Right column: Man-made and Impulse
        right_column = QVBoxLayout()

        # Man-made noise group
        manmade_group = QGroupBox("Man-Made Noise")
        manmade_layout = QGridLayout(manmade_group)

        self._manmade_check = QCheckBox("Enable")
        manmade_layout.addWidget(self._manmade_check, 0, 0, 1, 2)

        manmade_layout.addWidget(QLabel("Environment:"), 1, 0)
        self._environment_combo = QComboBox()
        self._environment_combo.addItems([
            "City",
            "Residential",
            "Rural",
            "Quiet Rural",
        ])
        self._environment_combo.setCurrentIndex(1)  # Residential
        manmade_layout.addWidget(self._environment_combo, 1, 1)

        right_column.addWidget(manmade_group)

        # Impulse noise group
        impulse_group = QGroupBox("Impulse Noise")
        impulse_layout = QGridLayout(impulse_group)

        self._impulse_check = QCheckBox("Enable")
        impulse_layout.addWidget(self._impulse_check, 0, 0, 1, 2)

        impulse_layout.addWidget(QLabel("Rate:"), 1, 0)
        self._impulse_rate_spin = QDoubleSpinBox()
        self._impulse_rate_spin.setRange(0.1, 100.0)
        self._impulse_rate_spin.setValue(10.0)
        self._impulse_rate_spin.setSuffix(" Hz")
        self._impulse_rate_spin.setSingleStep(1.0)
        impulse_layout.addWidget(self._impulse_rate_spin, 1, 1)

        impulse_layout.addWidget(QLabel("Amplitude:"), 2, 0)
        self._impulse_amp_spin = QDoubleSpinBox()
        self._impulse_amp_spin.setRange(0.0, 40.0)
        self._impulse_amp_spin.setValue(20.0)
        self._impulse_amp_spin.setSuffix(" dB")
        self._impulse_amp_spin.setSingleStep(1.0)
        impulse_layout.addWidget(self._impulse_amp_spin, 2, 1)

        impulse_layout.addWidget(QLabel("Duration:"), 3, 0)
        self._impulse_dur_spin = QDoubleSpinBox()
        self._impulse_dur_spin.setRange(1.0, 1000.0)
        self._impulse_dur_spin.setValue(100.0)
        self._impulse_dur_spin.setSuffix(" μs")
        self._impulse_dur_spin.setSingleStep(10.0)
        impulse_layout.addWidget(self._impulse_dur_spin, 3, 1)

        right_column.addWidget(impulse_group)
        right_column.addStretch()

        content_layout.addLayout(right_column)

        layout.addLayout(content_layout)

        # Bottom status row
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(8, 4, 8, 4)

        self._noise_floor_label = QLabel("Noise Floor: -95 dBm")
        status_layout.addWidget(self._noise_floor_label)

        status_layout.addWidget(QLabel("|"))

        self._est_snr_label = QLabel("Estimated SNR: 18.5 dB")
        status_layout.addWidget(self._est_snr_label)

        status_layout.addWidget(QLabel("|"))

        self._nf_label = QLabel("NF: 8.2 dB")
        status_layout.addWidget(self._nf_label)

        status_layout.addStretch()

        layout.addWidget(status_frame)

    def _connect_signals(self):
        """Connect widget signals."""
        self._snr_slider.valueChanged.connect(self._on_snr_changed)
        self._apply_btn.clicked.connect(self._on_apply)

        # Enable/disable groups
        self._enable_noise_check.toggled.connect(self._update_enabled_state)
        self._atmos_check.toggled.connect(self._update_atmos_enabled)
        self._manmade_check.toggled.connect(self._update_manmade_enabled)
        self._impulse_check.toggled.connect(self._update_impulse_enabled)

        # Initial state
        self._update_enabled_state(self._enable_noise_check.isChecked())

    def _on_snr_changed(self, value: int):
        """Handle SNR slider change."""
        snr_db = float(value)
        self._snr_label.setText(f"{snr_db:.1f} dB")
        self.snr_changed.emit(snr_db)

    def _on_apply(self):
        """Apply current settings."""
        self._update_config()
        self.noise_config_changed.emit(self._config)

    def _update_config(self):
        """Update internal config from widget values."""
        env_map = {
            "City": ManMadeEnvironment.CITY,
            "Residential": ManMadeEnvironment.RESIDENTIAL,
            "Rural": ManMadeEnvironment.RURAL,
            "Quiet Rural": ManMadeEnvironment.QUIET_RURAL,
        }

        self._config = NoiseConfig(
            snr_db=float(self._snr_slider.value()),
            noise_bandwidth_hz=self._bandwidth_spin.value(),
            enable_atmospheric=self._atmos_check.isChecked(),
            frequency_mhz=self._atmos_freq_spin.value(),
            season=self._season_combo.currentText().lower(),
            time_of_day=self._time_combo.currentText().lower(),
            latitude=self._latitude_spin.value(),
            enable_manmade=self._manmade_check.isChecked(),
            environment=env_map[self._environment_combo.currentText()],
            enable_impulse=self._impulse_check.isChecked(),
            impulse_rate_hz=self._impulse_rate_spin.value(),
            impulse_amplitude_db=self._impulse_amp_spin.value(),
            impulse_duration_us=self._impulse_dur_spin.value(),
        )

    def _update_enabled_state(self, enabled: bool):
        """Update enabled state of all controls."""
        for widget in [self._awgn_check, self._atmos_check, self._manmade_check,
                       self._impulse_check, self._snr_slider, self._bandwidth_spin]:
            widget.setEnabled(enabled)

        if enabled:
            self._update_atmos_enabled(self._atmos_check.isChecked())
            self._update_manmade_enabled(self._manmade_check.isChecked())
            self._update_impulse_enabled(self._impulse_check.isChecked())

    def _update_atmos_enabled(self, enabled: bool):
        """Update atmospheric controls enabled state."""
        for widget in [self._atmos_freq_spin, self._season_combo,
                       self._time_combo, self._latitude_spin]:
            widget.setEnabled(enabled and self._enable_noise_check.isChecked())

    def _update_manmade_enabled(self, enabled: bool):
        """Update man-made controls enabled state."""
        self._environment_combo.setEnabled(enabled and self._enable_noise_check.isChecked())

    def _update_impulse_enabled(self, enabled: bool):
        """Update impulse controls enabled state."""
        for widget in [self._impulse_rate_spin, self._impulse_amp_spin,
                       self._impulse_dur_spin]:
            widget.setEnabled(enabled and self._enable_noise_check.isChecked())

    def set_config(self, config: NoiseConfig):
        """Set widget values from config."""
        self._config = config

        # Block signals during update
        self._snr_slider.blockSignals(True)

        self._snr_slider.setValue(int(config.snr_db))
        self._snr_label.setText(f"{config.snr_db:.1f} dB")
        self._bandwidth_spin.setValue(config.noise_bandwidth_hz)

        self._atmos_check.setChecked(config.enable_atmospheric)
        self._atmos_freq_spin.setValue(config.frequency_mhz)
        self._season_combo.setCurrentText(config.season.capitalize())
        self._time_combo.setCurrentText(config.time_of_day.capitalize())
        self._latitude_spin.setValue(config.latitude)

        self._manmade_check.setChecked(config.enable_manmade)
        env_names = {
            ManMadeEnvironment.CITY: "City",
            ManMadeEnvironment.RESIDENTIAL: "Residential",
            ManMadeEnvironment.RURAL: "Rural",
            ManMadeEnvironment.QUIET_RURAL: "Quiet Rural",
        }
        self._environment_combo.setCurrentText(env_names.get(config.environment, "Residential"))

        self._impulse_check.setChecked(config.enable_impulse)
        self._impulse_rate_spin.setValue(config.impulse_rate_hz)
        self._impulse_amp_spin.setValue(config.impulse_amplitude_db)
        self._impulse_dur_spin.setValue(config.impulse_duration_us)

        self._snr_slider.blockSignals(False)

    def get_config(self) -> NoiseConfig:
        """Get current noise configuration."""
        self._update_config()
        return self._config

    def is_noise_enabled(self) -> bool:
        """Check if noise is enabled."""
        return self._enable_noise_check.isChecked()

    def update_status(self, noise_floor_dbm: float, est_snr_db: float, nf_db: float):
        """Update the status display."""
        self._noise_floor_label.setText(f"Noise Floor: {noise_floor_dbm:.1f} dBm")
        self._est_snr_label.setText(f"Estimated SNR: {est_snr_db:.1f} dB")
        self._nf_label.setText(f"NF: {nf_db:.1f} dB")
