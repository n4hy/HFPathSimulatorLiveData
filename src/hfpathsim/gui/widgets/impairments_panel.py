"""Signal impairments panel widget."""

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
    QProgressBar,
    QRadioButton,
    QButtonGroup,
    QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal

from hfpathsim.core.impairments import (
    AGCConfig,
    AGCMode,
    LimiterConfig,
    FrequencyOffsetConfig,
)


class AGCMeter(QProgressBar):
    """Gain meter display for AGC."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(-20, 60)
        self.setValue(0)
        self.setTextVisible(True)
        self.setFormat("%v dB")
        self.setFixedHeight(20)

    def set_gain(self, gain_db: float):
        """Set current gain value."""
        self.setValue(int(gain_db))


class GainReductionMeter(QProgressBar):
    """Gain reduction meter for limiter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(0, 30)
        self.setValue(0)
        self.setTextVisible(True)
        self.setFormat("-%v dB")
        self.setFixedHeight(20)
        self.setInvertedAppearance(True)

    def set_reduction(self, reduction_db: float):
        """Set gain reduction value (positive dB)."""
        self.setValue(int(abs(reduction_db)))


class ImpairmentsPanel(QWidget):
    """Panel for configuring signal impairments.

    Supports:
    - AGC with multiple modes
    - Limiter with hard/soft/cubic modes
    - Frequency offset with drift and phase noise
    """

    agc_config_changed = pyqtSignal(AGCConfig)
    limiter_config_changed = pyqtSignal(LimiterConfig)
    freq_offset_config_changed = pyqtSignal(FrequencyOffsetConfig)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._agc_config = AGCConfig()
        self._limiter_config = LimiterConfig()
        self._freq_offset_config = FrequencyOffsetConfig()

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the impairments panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Main content: Three groups
        content_layout = QHBoxLayout()

        # AGC Group
        agc_group = QGroupBox("AGC")
        agc_layout = QGridLayout(agc_group)

        self._agc_enable_check = QCheckBox("Enable")
        self._agc_enable_check.setChecked(True)
        agc_layout.addWidget(self._agc_enable_check, 0, 0, 1, 2)

        agc_layout.addWidget(QLabel("Mode:"), 1, 0)
        self._agc_mode_combo = QComboBox()
        self._agc_mode_combo.addItems(["Slow", "Medium", "Fast", "Manual"])
        self._agc_mode_combo.setCurrentIndex(1)  # Medium
        agc_layout.addWidget(self._agc_mode_combo, 1, 1)

        agc_layout.addWidget(QLabel("Target:"), 2, 0)
        self._agc_target_slider = QSlider(Qt.Orientation.Horizontal)
        self._agc_target_slider.setRange(-30, 0)
        self._agc_target_slider.setValue(-10)
        agc_layout.addWidget(self._agc_target_slider, 2, 1)

        self._agc_target_label = QLabel("-10 dB")
        self._agc_target_label.setFixedWidth(50)
        agc_layout.addWidget(self._agc_target_label, 2, 2)

        agc_layout.addWidget(QLabel("Max Gain:"), 3, 0)
        self._agc_max_gain_spin = QDoubleSpinBox()
        self._agc_max_gain_spin.setRange(0.0, 80.0)
        self._agc_max_gain_spin.setValue(60.0)
        self._agc_max_gain_spin.setSuffix(" dB")
        agc_layout.addWidget(self._agc_max_gain_spin, 3, 1)

        agc_layout.addWidget(QLabel("Attack:"), 4, 0)
        self._agc_attack_spin = QDoubleSpinBox()
        self._agc_attack_spin.setRange(1.0, 1000.0)
        self._agc_attack_spin.setValue(50.0)
        self._agc_attack_spin.setSuffix(" ms")
        agc_layout.addWidget(self._agc_attack_spin, 4, 1)

        agc_layout.addWidget(QLabel("Release:"), 5, 0)
        self._agc_release_spin = QDoubleSpinBox()
        self._agc_release_spin.setRange(10.0, 5000.0)
        self._agc_release_spin.setValue(500.0)
        self._agc_release_spin.setSuffix(" ms")
        agc_layout.addWidget(self._agc_release_spin, 5, 1)

        self._agc_hang_check = QCheckBox("Hang AGC")
        self._agc_hang_check.setChecked(True)
        agc_layout.addWidget(self._agc_hang_check, 6, 0, 1, 2)

        agc_layout.addWidget(QLabel("Current Gain:"), 7, 0)
        self._agc_meter = AGCMeter()
        agc_layout.addWidget(self._agc_meter, 7, 1, 1, 2)

        content_layout.addWidget(agc_group)

        # Limiter Group
        limiter_group = QGroupBox("Limiter")
        limiter_layout = QGridLayout(limiter_group)

        self._limiter_enable_check = QCheckBox("Enable")
        limiter_layout.addWidget(self._limiter_enable_check, 0, 0, 1, 2)

        limiter_layout.addWidget(QLabel("Threshold:"), 1, 0)
        self._limiter_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._limiter_threshold_slider.setRange(-20, 0)
        self._limiter_threshold_slider.setValue(-3)
        limiter_layout.addWidget(self._limiter_threshold_slider, 1, 1)

        self._limiter_threshold_label = QLabel("-3 dB")
        self._limiter_threshold_label.setFixedWidth(50)
        limiter_layout.addWidget(self._limiter_threshold_label, 1, 2)

        limiter_layout.addWidget(QLabel("Mode:"), 2, 0)

        mode_widget = QWidget()
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)

        self._limiter_mode_group = QButtonGroup(self)
        self._limiter_hard_radio = QRadioButton("Hard")
        self._limiter_soft_radio = QRadioButton("Soft")
        self._limiter_soft_radio.setChecked(True)
        self._limiter_cubic_radio = QRadioButton("Cubic")

        self._limiter_mode_group.addButton(self._limiter_hard_radio, 0)
        self._limiter_mode_group.addButton(self._limiter_soft_radio, 1)
        self._limiter_mode_group.addButton(self._limiter_cubic_radio, 2)

        mode_layout.addWidget(self._limiter_hard_radio)
        mode_layout.addWidget(self._limiter_soft_radio)
        mode_layout.addWidget(self._limiter_cubic_radio)

        limiter_layout.addWidget(mode_widget, 2, 1, 1, 2)

        limiter_layout.addWidget(QLabel("Attack:"), 3, 0)
        self._limiter_attack_spin = QDoubleSpinBox()
        self._limiter_attack_spin.setRange(0.01, 10.0)
        self._limiter_attack_spin.setValue(0.1)
        self._limiter_attack_spin.setSuffix(" ms")
        self._limiter_attack_spin.setDecimals(2)
        limiter_layout.addWidget(self._limiter_attack_spin, 3, 1)

        limiter_layout.addWidget(QLabel("Release:"), 4, 0)
        self._limiter_release_spin = QDoubleSpinBox()
        self._limiter_release_spin.setRange(1.0, 500.0)
        self._limiter_release_spin.setValue(10.0)
        self._limiter_release_spin.setSuffix(" ms")
        limiter_layout.addWidget(self._limiter_release_spin, 4, 1)

        limiter_layout.addWidget(QLabel("Gain Reduction:"), 5, 0)
        self._limiter_meter = GainReductionMeter()
        limiter_layout.addWidget(self._limiter_meter, 5, 1, 1, 2)

        # Add stretch to push content up
        limiter_layout.setRowStretch(6, 1)

        content_layout.addWidget(limiter_group)

        # Frequency Offset Group
        freq_group = QGroupBox("Frequency Offset")
        freq_layout = QGridLayout(freq_group)

        self._freq_enable_check = QCheckBox("Enable")
        freq_layout.addWidget(self._freq_enable_check, 0, 0, 1, 2)

        freq_layout.addWidget(QLabel("Offset:"), 1, 0)
        self._freq_offset_slider = QSlider(Qt.Orientation.Horizontal)
        self._freq_offset_slider.setRange(-500, 500)
        self._freq_offset_slider.setValue(0)
        freq_layout.addWidget(self._freq_offset_slider, 1, 1)

        self._freq_offset_label = QLabel("0 Hz")
        self._freq_offset_label.setFixedWidth(60)
        freq_layout.addWidget(self._freq_offset_label, 1, 2)

        freq_layout.addWidget(QLabel("Drift:"), 2, 0)
        self._freq_drift_spin = QDoubleSpinBox()
        self._freq_drift_spin.setRange(-10.0, 10.0)
        self._freq_drift_spin.setValue(0.0)
        self._freq_drift_spin.setSuffix(" Hz/s")
        self._freq_drift_spin.setSingleStep(0.1)
        freq_layout.addWidget(self._freq_drift_spin, 2, 1)

        freq_layout.addWidget(QLabel("Phase Noise:"), 3, 0)
        self._phase_noise_spin = QDoubleSpinBox()
        self._phase_noise_spin.setRange(-120.0, -40.0)
        self._phase_noise_spin.setValue(-80.0)
        self._phase_noise_spin.setSuffix(" dBc/Hz")
        self._phase_noise_spin.setSingleStep(5.0)
        freq_layout.addWidget(self._phase_noise_spin, 3, 1)

        freq_layout.addWidget(QLabel("@ 1kHz offset"))

        # Current offset display
        current_frame = QFrame()
        current_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        current_layout = QHBoxLayout(current_frame)
        current_layout.setContentsMargins(4, 2, 4, 2)
        current_layout.addWidget(QLabel("Current:"))
        self._current_offset_label = QLabel("+0.0 Hz")
        self._current_offset_label.setStyleSheet("font-weight: bold;")
        current_layout.addWidget(self._current_offset_label)
        current_layout.addStretch()

        freq_layout.addWidget(current_frame, 4, 0, 1, 3)

        # Add stretch
        freq_layout.setRowStretch(5, 1)

        content_layout.addWidget(freq_group)

        layout.addLayout(content_layout)

        # Apply button row
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._apply_btn = QPushButton("Apply All")
        btn_row.addWidget(self._apply_btn)

        layout.addLayout(btn_row)

    def _connect_signals(self):
        """Connect widget signals."""
        # Sliders with labels
        self._agc_target_slider.valueChanged.connect(
            lambda v: self._agc_target_label.setText(f"{v} dB")
        )
        self._limiter_threshold_slider.valueChanged.connect(
            lambda v: self._limiter_threshold_label.setText(f"{v} dB")
        )
        self._freq_offset_slider.valueChanged.connect(
            lambda v: self._freq_offset_label.setText(f"{v:+d} Hz")
        )

        # Enable/disable groups
        self._agc_enable_check.toggled.connect(self._update_agc_enabled)
        self._limiter_enable_check.toggled.connect(self._update_limiter_enabled)
        self._freq_enable_check.toggled.connect(self._update_freq_enabled)

        # AGC mode presets
        self._agc_mode_combo.currentTextChanged.connect(self._on_agc_mode_changed)

        # Apply button
        self._apply_btn.clicked.connect(self._on_apply)

        # Initial state
        self._update_agc_enabled(self._agc_enable_check.isChecked())
        self._update_limiter_enabled(self._limiter_enable_check.isChecked())
        self._update_freq_enabled(self._freq_enable_check.isChecked())

    def _on_agc_mode_changed(self, mode_name: str):
        """Apply AGC mode presets."""
        presets = {
            "Slow": (500.0, 2000.0),
            "Medium": (50.0, 500.0),
            "Fast": (5.0, 50.0),
            "Manual": (1.0, 1.0),
        }

        if mode_name in presets:
            attack, release = presets[mode_name]
            self._agc_attack_spin.setValue(attack)
            self._agc_release_spin.setValue(release)

    def _update_agc_enabled(self, enabled: bool):
        """Update AGC controls enabled state."""
        for widget in [self._agc_mode_combo, self._agc_target_slider,
                       self._agc_max_gain_spin, self._agc_attack_spin,
                       self._agc_release_spin, self._agc_hang_check]:
            widget.setEnabled(enabled)

    def _update_limiter_enabled(self, enabled: bool):
        """Update limiter controls enabled state."""
        for widget in [self._limiter_threshold_slider, self._limiter_hard_radio,
                       self._limiter_soft_radio, self._limiter_cubic_radio,
                       self._limiter_attack_spin, self._limiter_release_spin]:
            widget.setEnabled(enabled)

    def _update_freq_enabled(self, enabled: bool):
        """Update frequency offset controls enabled state."""
        for widget in [self._freq_offset_slider, self._freq_drift_spin,
                       self._phase_noise_spin]:
            widget.setEnabled(enabled)

    def _on_apply(self):
        """Apply all settings."""
        self._update_configs()
        self.agc_config_changed.emit(self._agc_config)
        self.limiter_config_changed.emit(self._limiter_config)
        self.freq_offset_config_changed.emit(self._freq_offset_config)

    def _update_configs(self):
        """Update internal configs from widget values."""
        # AGC config
        mode_map = {
            "Slow": AGCMode.SLOW,
            "Medium": AGCMode.MEDIUM,
            "Fast": AGCMode.FAST,
            "Manual": AGCMode.MANUAL,
        }

        self._agc_config = AGCConfig(
            mode=mode_map[self._agc_mode_combo.currentText()],
            target_level_db=float(self._agc_target_slider.value()),
            max_gain_db=self._agc_max_gain_spin.value(),
            attack_time_ms=self._agc_attack_spin.value(),
            release_time_ms=self._agc_release_spin.value(),
            hang_agc=self._agc_hang_check.isChecked(),
        )

        # Limiter config
        mode_names = {0: "hard", 1: "soft", 2: "cubic"}
        limiter_mode = mode_names[self._limiter_mode_group.checkedId()]

        self._limiter_config = LimiterConfig(
            threshold_db=float(self._limiter_threshold_slider.value()),
            mode=limiter_mode,
            attack_time_ms=self._limiter_attack_spin.value(),
            release_time_ms=self._limiter_release_spin.value(),
        )

        # Frequency offset config
        self._freq_offset_config = FrequencyOffsetConfig(
            offset_hz=float(self._freq_offset_slider.value()),
            drift_rate_hz_per_sec=self._freq_drift_spin.value(),
            phase_noise_level_dbc=self._phase_noise_spin.value(),
        )

    def set_agc_config(self, config: AGCConfig):
        """Set AGC configuration."""
        self._agc_config = config

        mode_names = {
            AGCMode.SLOW: "Slow",
            AGCMode.MEDIUM: "Medium",
            AGCMode.FAST: "Fast",
            AGCMode.MANUAL: "Manual",
        }

        self._agc_mode_combo.setCurrentText(mode_names.get(config.mode, "Medium"))
        self._agc_target_slider.setValue(int(config.target_level_db))
        self._agc_max_gain_spin.setValue(config.max_gain_db)
        self._agc_attack_spin.setValue(config.attack_time_ms)
        self._agc_release_spin.setValue(config.release_time_ms)
        self._agc_hang_check.setChecked(config.hang_agc)

    def set_limiter_config(self, config: LimiterConfig):
        """Set limiter configuration."""
        self._limiter_config = config

        self._limiter_threshold_slider.setValue(int(config.threshold_db))

        mode_buttons = {"hard": self._limiter_hard_radio,
                        "soft": self._limiter_soft_radio,
                        "cubic": self._limiter_cubic_radio}
        if config.mode in mode_buttons:
            mode_buttons[config.mode].setChecked(True)

        self._limiter_attack_spin.setValue(config.attack_time_ms)
        self._limiter_release_spin.setValue(config.release_time_ms)

    def set_freq_offset_config(self, config: FrequencyOffsetConfig):
        """Set frequency offset configuration."""
        self._freq_offset_config = config

        self._freq_offset_slider.setValue(int(config.offset_hz))
        self._freq_drift_spin.setValue(config.drift_rate_hz_per_sec)
        self._phase_noise_spin.setValue(config.phase_noise_level_dbc)

    def get_agc_config(self) -> AGCConfig:
        """Get current AGC configuration."""
        self._update_configs()
        return self._agc_config

    def get_limiter_config(self) -> LimiterConfig:
        """Get current limiter configuration."""
        self._update_configs()
        return self._limiter_config

    def get_freq_offset_config(self) -> FrequencyOffsetConfig:
        """Get current frequency offset configuration."""
        self._update_configs()
        return self._freq_offset_config

    def is_agc_enabled(self) -> bool:
        """Check if AGC is enabled."""
        return self._agc_enable_check.isChecked()

    def is_limiter_enabled(self) -> bool:
        """Check if limiter is enabled."""
        return self._limiter_enable_check.isChecked()

    def is_freq_offset_enabled(self) -> bool:
        """Check if frequency offset is enabled."""
        return self._freq_enable_check.isChecked()

    def update_agc_meter(self, gain_db: float):
        """Update AGC gain meter."""
        self._agc_meter.set_gain(gain_db)

    def update_limiter_meter(self, reduction_db: float):
        """Update limiter gain reduction meter."""
        self._limiter_meter.set_reduction(reduction_db)

    def update_current_offset(self, offset_hz: float):
        """Update current frequency offset display."""
        self._current_offset_label.setText(f"{offset_hz:+.1f} Hz")
