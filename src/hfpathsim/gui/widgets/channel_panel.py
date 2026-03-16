"""Channel control panel widget with Vogler, Watterson, and Vogler-Hoffmeyer models."""

from typing import Optional, List

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
    QPushButton,
    QScrollArea,
    QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal

from hfpathsim.core.parameters import VoglerParameters, ITUCondition, PropagationMode
from hfpathsim.core.watterson import (
    WattersonConfig,
    WattersonTap,
    DopplerSpectrum,
)
from hfpathsim.core.vogler_hoffmeyer import (
    VoglerHoffmeyerConfig,
    ModeParameters,
    CorrelationType,
    VOGLER_HOFFMEYER_PRESETS,
)


class TapWidget(QFrame):
    """Widget for configuring a single Watterson tap."""

    tap_changed = pyqtSignal()
    remove_requested = pyqtSignal(object)  # self

    def __init__(self, tap_index: int, parent=None):
        super().__init__(parent)
        self._tap_index = tap_index
        self._setup_ui()

    def _setup_ui(self):
        """Setup the tap configuration UI."""
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(8)

        # Tap label
        layout.addWidget(QLabel(f"Tap {self._tap_index + 1}:"))

        # Delay
        layout.addWidget(QLabel("τ:"))
        self._delay_spin = QDoubleSpinBox()
        self._delay_spin.setRange(0.0, 20.0)
        self._delay_spin.setValue(0.0 if self._tap_index == 0 else 2.0)
        self._delay_spin.setSuffix(" ms")
        self._delay_spin.setSingleStep(0.1)
        self._delay_spin.setFixedWidth(80)
        self._delay_spin.valueChanged.connect(self.tap_changed)
        layout.addWidget(self._delay_spin)

        # Amplitude
        layout.addWidget(QLabel("A:"))
        self._amplitude_spin = QDoubleSpinBox()
        self._amplitude_spin.setRange(0.0, 2.0)
        self._amplitude_spin.setValue(1.0 if self._tap_index == 0 else 0.7)
        self._amplitude_spin.setSingleStep(0.1)
        self._amplitude_spin.setDecimals(2)
        self._amplitude_spin.setFixedWidth(60)
        self._amplitude_spin.valueChanged.connect(self.tap_changed)
        layout.addWidget(self._amplitude_spin)

        # Doppler spread
        layout.addWidget(QLabel("ν:"))
        self._doppler_spin = QDoubleSpinBox()
        self._doppler_spin.setRange(0.01, 20.0)
        self._doppler_spin.setValue(1.0)
        self._doppler_spin.setSuffix(" Hz")
        self._doppler_spin.setSingleStep(0.1)
        self._doppler_spin.setFixedWidth(80)
        self._doppler_spin.valueChanged.connect(self.tap_changed)
        layout.addWidget(self._doppler_spin)

        # Doppler spectrum type
        self._spectrum_combo = QComboBox()
        self._spectrum_combo.addItems(["Gaussian", "Flat", "Jakes"])
        self._spectrum_combo.setFixedWidth(80)
        self._spectrum_combo.currentIndexChanged.connect(lambda: self.tap_changed.emit())
        layout.addWidget(self._spectrum_combo)

        # Remove button
        self._remove_btn = QPushButton("×")
        self._remove_btn.setFixedSize(24, 24)
        self._remove_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        layout.addWidget(self._remove_btn)

    def get_tap(self) -> WattersonTap:
        """Get the current tap configuration."""
        spectrum_map = {
            "Gaussian": DopplerSpectrum.GAUSSIAN,
            "Flat": DopplerSpectrum.FLAT,
            "Jakes": DopplerSpectrum.JAKES,
        }
        return WattersonTap(
            delay_ms=self._delay_spin.value(),
            amplitude=self._amplitude_spin.value(),
            doppler_spread_hz=self._doppler_spin.value(),
            doppler_spectrum=spectrum_map[self._spectrum_combo.currentText()],
        )

    def set_tap(self, tap: WattersonTap):
        """Set tap configuration from WattersonTap."""
        self._delay_spin.blockSignals(True)
        self._amplitude_spin.blockSignals(True)
        self._doppler_spin.blockSignals(True)
        self._spectrum_combo.blockSignals(True)

        self._delay_spin.setValue(tap.delay_ms)
        self._amplitude_spin.setValue(tap.amplitude)
        self._doppler_spin.setValue(tap.doppler_spread_hz)

        spectrum_names = {
            DopplerSpectrum.GAUSSIAN: "Gaussian",
            DopplerSpectrum.FLAT: "Flat",
            DopplerSpectrum.JAKES: "Jakes",
        }
        self._spectrum_combo.setCurrentText(spectrum_names.get(tap.doppler_spectrum, "Gaussian"))

        self._delay_spin.blockSignals(False)
        self._amplitude_spin.blockSignals(False)
        self._doppler_spin.blockSignals(False)
        self._spectrum_combo.blockSignals(False)


class ChannelPanel(QWidget):
    """Panel for controlling channel model parameters.

    Combines Vogler IPM, Watterson TDL, and Vogler-Hoffmeyer stochastic models with ITU presets.
    """

    parameters_changed = pyqtSignal(VoglerParameters)
    watterson_config_changed = pyqtSignal(WattersonConfig)
    vogler_hoffmeyer_config_changed = pyqtSignal(VoglerHoffmeyerConfig)
    model_changed = pyqtSignal(str)  # "vogler", "watterson", or "vogler_hoffmeyer"

    def __init__(self, parent=None):
        super().__init__(parent)

        self._params = VoglerParameters()
        self._watterson_config = WattersonConfig.from_itu_condition(ITUCondition.MODERATE)
        self._vh_config = VoglerHoffmeyerConfig.from_itu_condition(ITUCondition.MODERATE)
        self._tap_widgets: List[TapWidget] = []

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the channel panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Model selection row
        model_row = QHBoxLayout()

        model_row.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        self._model_combo.addItems(["Vogler IPM", "Watterson TDL", "Vogler-Hoffmeyer"])
        model_row.addWidget(self._model_combo)

        model_row.addWidget(QLabel("ITU Preset:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItems(["Quiet", "Moderate", "Disturbed", "Flutter"])
        self._preset_combo.setCurrentIndex(1)
        model_row.addWidget(self._preset_combo)

        model_row.addStretch()

        self._apply_btn = QPushButton("Apply")
        model_row.addWidget(self._apply_btn)

        layout.addLayout(model_row)

        # Main content area with three columns
        content_layout = QHBoxLayout()

        # Column 1: Ionospheric parameters
        iono_group = QGroupBox("Ionospheric")
        iono_layout = QGridLayout(iono_group)

        iono_layout.addWidget(QLabel("foF2:"), 0, 0)
        self._foF2_spin = QDoubleSpinBox()
        self._foF2_spin.setRange(1.0, 20.0)
        self._foF2_spin.setValue(7.5)
        self._foF2_spin.setSuffix(" MHz")
        self._foF2_spin.setSingleStep(0.1)
        iono_layout.addWidget(self._foF2_spin, 0, 1)

        iono_layout.addWidget(QLabel("hmF2:"), 1, 0)
        self._hmF2_spin = QDoubleSpinBox()
        self._hmF2_spin.setRange(150.0, 500.0)
        self._hmF2_spin.setValue(300.0)
        self._hmF2_spin.setSuffix(" km")
        self._hmF2_spin.setSingleStep(10.0)
        iono_layout.addWidget(self._hmF2_spin, 1, 1)

        iono_layout.addWidget(QLabel("foE:"), 2, 0)
        self._foE_spin = QDoubleSpinBox()
        self._foE_spin.setRange(0.5, 5.0)
        self._foE_spin.setValue(3.0)
        self._foE_spin.setSuffix(" MHz")
        self._foE_spin.setSingleStep(0.1)
        iono_layout.addWidget(self._foE_spin, 2, 1)

        iono_layout.addWidget(QLabel("hmE:"), 3, 0)
        self._hmE_spin = QDoubleSpinBox()
        self._hmE_spin.setRange(90.0, 130.0)
        self._hmE_spin.setValue(110.0)
        self._hmE_spin.setSuffix(" km")
        self._hmE_spin.setSingleStep(5.0)
        iono_layout.addWidget(self._hmE_spin, 3, 1)

        content_layout.addWidget(iono_group)

        # Column 2: Channel statistics
        stats_group = QGroupBox("Channel Stats")
        stats_layout = QGridLayout(stats_group)

        stats_layout.addWidget(QLabel("Delay:"), 0, 0)
        self._delay_spin = QDoubleSpinBox()
        self._delay_spin.setRange(0.1, 10.0)
        self._delay_spin.setValue(2.0)
        self._delay_spin.setSuffix(" ms")
        self._delay_spin.setSingleStep(0.1)
        stats_layout.addWidget(self._delay_spin, 0, 1)

        stats_layout.addWidget(QLabel("Doppler:"), 1, 0)
        self._doppler_spin = QDoubleSpinBox()
        self._doppler_spin.setRange(0.01, 20.0)
        self._doppler_spin.setValue(1.0)
        self._doppler_spin.setSuffix(" Hz")
        self._doppler_spin.setSingleStep(0.1)
        stats_layout.addWidget(self._doppler_spin, 1, 1)

        stats_layout.addWidget(QLabel("Frequency:"), 2, 0)
        self._freq_spin = QDoubleSpinBox()
        self._freq_spin.setRange(2.0, 30.0)
        self._freq_spin.setValue(10.0)
        self._freq_spin.setSuffix(" MHz")
        self._freq_spin.setSingleStep(0.5)
        stats_layout.addWidget(self._freq_spin, 2, 1)

        stats_layout.addWidget(QLabel("Path:"), 3, 0)
        self._path_spin = QDoubleSpinBox()
        self._path_spin.setRange(100.0, 10000.0)
        self._path_spin.setValue(1000.0)
        self._path_spin.setSuffix(" km")
        self._path_spin.setSingleStep(100.0)
        stats_layout.addWidget(self._path_spin, 3, 1)

        content_layout.addWidget(stats_group)

        # Column 3: Watterson taps / Mode selection / Vogler-Hoffmeyer presets
        self._taps_group = QGroupBox("Watterson Taps / Modes / VH Presets")
        taps_layout = QVBoxLayout(self._taps_group)

        # Mode checkboxes (for Vogler)
        self._modes_widget = QWidget()
        modes_layout = QGridLayout(self._modes_widget)
        modes_layout.setContentsMargins(0, 0, 0, 0)

        self._mode_1F2_check = QCheckBox("1F2")
        self._mode_1F2_check.setChecked(True)
        modes_layout.addWidget(self._mode_1F2_check, 0, 0)

        self._mode_2F2_check = QCheckBox("2F2")
        self._mode_2F2_check.setChecked(True)
        modes_layout.addWidget(self._mode_2F2_check, 0, 1)

        self._mode_1E_check = QCheckBox("1E")
        modes_layout.addWidget(self._mode_1E_check, 1, 0)

        self._mode_Es_check = QCheckBox("Es")
        modes_layout.addWidget(self._mode_Es_check, 1, 1)

        taps_layout.addWidget(self._modes_widget)

        # Tap widgets container (for Watterson)
        self._taps_container = QWidget()
        self._taps_layout = QVBoxLayout(self._taps_container)
        self._taps_layout.setContentsMargins(0, 0, 0, 0)
        self._taps_layout.setSpacing(2)

        # Scroll area for taps
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._taps_container)
        scroll.setMaximumHeight(120)
        taps_layout.addWidget(scroll)

        # Vogler-Hoffmeyer configuration widget
        self._vh_widget = QWidget()
        vh_layout = QVBoxLayout(self._vh_widget)
        vh_layout.setContentsMargins(0, 0, 0, 0)

        # VH Preset selection
        vh_preset_row = QHBoxLayout()
        vh_preset_row.addWidget(QLabel("VH Preset:"))
        self._vh_preset_combo = QComboBox()
        self._vh_preset_combo.addItems(["equatorial", "polar", "midlatitude", "auroral_spread_f"])
        self._vh_preset_combo.currentTextChanged.connect(self._on_vh_preset_changed)
        vh_preset_row.addWidget(self._vh_preset_combo)
        vh_preset_row.addStretch()
        vh_layout.addLayout(vh_preset_row)

        # VH parameters grid
        vh_params_grid = QGridLayout()

        vh_params_grid.addWidget(QLabel("Delay Spread:"), 0, 0)
        self._vh_delay_spin = QDoubleSpinBox()
        self._vh_delay_spin.setRange(10.0, 5000.0)
        self._vh_delay_spin.setValue(100.0)
        self._vh_delay_spin.setSuffix(" us")
        self._vh_delay_spin.setSingleStep(10.0)
        vh_params_grid.addWidget(self._vh_delay_spin, 0, 1)

        vh_params_grid.addWidget(QLabel("Doppler Spread:"), 1, 0)
        self._vh_doppler_spin = QDoubleSpinBox()
        self._vh_doppler_spin.setRange(0.01, 50.0)
        self._vh_doppler_spin.setValue(1.0)
        self._vh_doppler_spin.setSuffix(" Hz")
        self._vh_doppler_spin.setSingleStep(0.1)
        vh_params_grid.addWidget(self._vh_doppler_spin, 1, 1)

        vh_params_grid.addWidget(QLabel("Correlation:"), 2, 0)
        self._vh_correlation_combo = QComboBox()
        self._vh_correlation_combo.addItems(["Gaussian", "Exponential"])
        vh_params_grid.addWidget(self._vh_correlation_combo, 2, 1)

        self._vh_spread_f_check = QCheckBox("Spread-F")
        vh_params_grid.addWidget(self._vh_spread_f_check, 3, 0, 1, 2)

        vh_layout.addLayout(vh_params_grid)

        taps_layout.addWidget(self._vh_widget)
        self._vh_widget.setVisible(False)  # Hidden by default

        # Add/Remove tap buttons
        tap_btn_row = QHBoxLayout()
        self._add_tap_btn = QPushButton("Add Tap")
        self._add_tap_btn.clicked.connect(self._add_tap)
        tap_btn_row.addWidget(self._add_tap_btn)

        tap_btn_row.addStretch()

        # Rician fading options
        self._rician_check = QCheckBox("Rician")
        tap_btn_row.addWidget(self._rician_check)

        tap_btn_row.addWidget(QLabel("K:"))
        self._k_factor_spin = QDoubleSpinBox()
        self._k_factor_spin.setRange(0.0, 20.0)
        self._k_factor_spin.setValue(6.0)
        self._k_factor_spin.setSuffix(" dB")
        self._k_factor_spin.setFixedWidth(70)
        self._k_factor_spin.setEnabled(False)
        tap_btn_row.addWidget(self._k_factor_spin)

        taps_layout.addLayout(tap_btn_row)

        content_layout.addWidget(self._taps_group)

        layout.addLayout(content_layout)

        # Initialize with default taps
        self._init_default_taps()

        # Set initial visibility based on model
        self._update_model_visibility()

    def _connect_signals(self):
        """Connect widget signals."""
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        self._apply_btn.clicked.connect(self._on_apply)
        self._rician_check.toggled.connect(self._k_factor_spin.setEnabled)

    def _init_default_taps(self):
        """Initialize with default Watterson taps."""
        self._clear_taps()
        for i in range(2):
            self._add_tap()

    def _add_tap(self):
        """Add a new tap widget."""
        tap_widget = TapWidget(len(self._tap_widgets))
        tap_widget.tap_changed.connect(self._on_taps_changed)
        tap_widget.remove_requested.connect(self._remove_tap)
        self._tap_widgets.append(tap_widget)
        self._taps_layout.addWidget(tap_widget)

    def _remove_tap(self, tap_widget: TapWidget):
        """Remove a tap widget."""
        if len(self._tap_widgets) <= 1:
            return  # Keep at least one tap

        self._tap_widgets.remove(tap_widget)
        self._taps_layout.removeWidget(tap_widget)
        tap_widget.deleteLater()

        # Renumber remaining taps
        for i, tw in enumerate(self._tap_widgets):
            tw._tap_index = i

        self._on_taps_changed()

    def _clear_taps(self):
        """Clear all tap widgets."""
        for tw in self._tap_widgets:
            self._taps_layout.removeWidget(tw)
            tw.deleteLater()
        self._tap_widgets.clear()

    def _update_model_visibility(self):
        """Update visibility based on selected model."""
        model_text = self._model_combo.currentText()
        is_watterson = model_text == "Watterson TDL"
        is_vh = model_text == "Vogler-Hoffmeyer"
        is_vogler = model_text == "Vogler IPM"

        # Show/hide mode checkboxes vs tap controls vs VH settings
        self._modes_widget.setVisible(is_vogler)
        self._taps_container.setVisible(is_watterson)
        self._add_tap_btn.setVisible(is_watterson)
        self._rician_check.setVisible(is_watterson)
        self._k_factor_spin.setVisible(is_watterson)
        self._vh_widget.setVisible(is_vh)

    def _on_model_changed(self, model_name: str):
        """Handle model selection change."""
        self._update_model_visibility()
        if model_name == "Watterson TDL":
            model_id = "watterson"
        elif model_name == "Vogler-Hoffmeyer":
            model_id = "vogler_hoffmeyer"
        else:
            model_id = "vogler"
        self.model_changed.emit(model_id)

    def _on_preset_changed(self, preset_name: str):
        """Handle ITU preset change."""
        presets = {
            "Quiet": ITUCondition.QUIET,
            "Moderate": ITUCondition.MODERATE,
            "Disturbed": ITUCondition.DISTURBED,
            "Flutter": ITUCondition.FLUTTER,
        }

        if preset_name in presets:
            condition = presets[preset_name]

            # Update Vogler parameters
            params = VoglerParameters.from_itu_condition(condition)
            self.set_vogler_parameters(params)

            # Update Watterson config
            self._watterson_config = WattersonConfig.from_itu_condition(condition)
            self._update_taps_from_config()

    def _on_taps_changed(self):
        """Handle tap configuration changes."""
        # Build new config from tap widgets
        taps = [tw.get_tap() for tw in self._tap_widgets]
        self._watterson_config = WattersonConfig(
            taps=taps,
            sample_rate_hz=self._watterson_config.sample_rate_hz,
            block_size=self._watterson_config.block_size,
            update_rate_hz=self._watterson_config.update_rate_hz,
        )

    def _on_apply(self):
        """Apply current settings."""
        model_text = self._model_combo.currentText()

        if model_text == "Watterson TDL":
            self._on_taps_changed()
            self.watterson_config_changed.emit(self._watterson_config)
        elif model_text == "Vogler-Hoffmeyer":
            self._update_vh_config()
            self.vogler_hoffmeyer_config_changed.emit(self._vh_config)
        else:
            self._update_vogler_params()
            self.parameters_changed.emit(self._params)

    def _on_vh_preset_changed(self, preset_name: str):
        """Handle Vogler-Hoffmeyer preset change."""
        if preset_name in VOGLER_HOFFMEYER_PRESETS:
            self._vh_config = VOGLER_HOFFMEYER_PRESETS[preset_name]()
            self._update_vh_widgets_from_config()

    def _update_vh_config(self):
        """Update internal VH config from widget values."""
        correlation = (CorrelationType.GAUSSIAN
                      if self._vh_correlation_combo.currentText() == "Gaussian"
                      else CorrelationType.EXPONENTIAL)

        mode = ModeParameters(
            name="Custom",
            amplitude=1.0,
            floor_amplitude=0.01,
            tau_L=0.0,
            sigma_tau=self._vh_delay_spin.value(),
            sigma_c=self._vh_delay_spin.value() / 4,
            sigma_D=self._vh_doppler_spin.value(),
            doppler_shift=0.0,
            doppler_shift_min_delay=0.0,
            correlation_type=correlation
        )

        self._vh_config = VoglerHoffmeyerConfig(
            sample_rate=self._vh_config.sample_rate,
            modes=[mode],
            spread_f_enabled=self._vh_spread_f_check.isChecked()
        )

    def _update_vh_widgets_from_config(self):
        """Update VH widgets from current config."""
        if self._vh_config.modes:
            mode = self._vh_config.modes[0]

            self._vh_delay_spin.blockSignals(True)
            self._vh_doppler_spin.blockSignals(True)
            self._vh_correlation_combo.blockSignals(True)
            self._vh_spread_f_check.blockSignals(True)

            self._vh_delay_spin.setValue(mode.sigma_tau)
            self._vh_doppler_spin.setValue(mode.sigma_D)
            self._vh_correlation_combo.setCurrentText(
                "Gaussian" if mode.correlation_type == CorrelationType.GAUSSIAN else "Exponential"
            )
            self._vh_spread_f_check.setChecked(self._vh_config.spread_f_enabled)

            self._vh_delay_spin.blockSignals(False)
            self._vh_doppler_spin.blockSignals(False)
            self._vh_correlation_combo.blockSignals(False)
            self._vh_spread_f_check.blockSignals(False)

    def _update_vogler_params(self):
        """Update internal Vogler parameters from widget values."""
        modes = []
        if self._mode_1F2_check.isChecked():
            modes.append(PropagationMode("1F2", True, 1.0, 0.0))
        if self._mode_2F2_check.isChecked():
            modes.append(PropagationMode("2F2", True, 0.7, 1.5))
        if self._mode_1E_check.isChecked():
            modes.append(PropagationMode("1E", True, 0.3, -0.5))
        if self._mode_Es_check.isChecked():
            modes.append(PropagationMode("Es", True, 0.5, -1.0))

        self._params = VoglerParameters(
            foF2=self._foF2_spin.value(),
            hmF2=self._hmF2_spin.value(),
            foE=self._foE_spin.value(),
            hmE=self._hmE_spin.value(),
            doppler_spread_hz=self._doppler_spin.value(),
            delay_spread_ms=self._delay_spin.value(),
            frequency_mhz=self._freq_spin.value(),
            path_length_km=self._path_spin.value(),
            modes=modes,
        )

    def _update_taps_from_config(self):
        """Update tap widgets from current Watterson config."""
        self._clear_taps()
        for i, tap in enumerate(self._watterson_config.taps):
            tap_widget = TapWidget(i)
            tap_widget.set_tap(tap)
            tap_widget.tap_changed.connect(self._on_taps_changed)
            tap_widget.remove_requested.connect(self._remove_tap)
            self._tap_widgets.append(tap_widget)
            self._taps_layout.addWidget(tap_widget)

    def set_vogler_parameters(self, params: VoglerParameters):
        """Set widget values from Vogler parameters."""
        self._params = params

        # Block signals during update
        for spin in [self._foF2_spin, self._hmF2_spin, self._foE_spin, self._hmE_spin,
                     self._delay_spin, self._doppler_spin, self._freq_spin, self._path_spin]:
            spin.blockSignals(True)

        self._foF2_spin.setValue(params.foF2)
        self._hmF2_spin.setValue(params.hmF2)
        self._foE_spin.setValue(params.foE)
        self._hmE_spin.setValue(params.hmE)
        self._delay_spin.setValue(params.delay_spread_ms)
        self._doppler_spin.setValue(params.doppler_spread_hz)
        self._freq_spin.setValue(params.frequency_mhz)
        self._path_spin.setValue(params.path_length_km)

        # Update mode checkboxes
        mode_names = [m.name for m in params.modes if m.enabled]
        self._mode_1F2_check.setChecked("1F2" in mode_names)
        self._mode_2F2_check.setChecked("2F2" in mode_names)
        self._mode_1E_check.setChecked("1E" in mode_names)
        self._mode_Es_check.setChecked("Es" in mode_names)

        for spin in [self._foF2_spin, self._hmF2_spin, self._foE_spin, self._hmE_spin,
                     self._delay_spin, self._doppler_spin, self._freq_spin, self._path_spin]:
            spin.blockSignals(False)

    def get_vogler_parameters(self) -> VoglerParameters:
        """Get current Vogler parameters."""
        self._update_vogler_params()
        return self._params

    def set_watterson_config(self, config: WattersonConfig):
        """Set Watterson configuration."""
        self._watterson_config = config
        self._update_taps_from_config()

    def get_watterson_config(self) -> WattersonConfig:
        """Get current Watterson configuration."""
        self._on_taps_changed()
        return self._watterson_config

    def get_current_model(self) -> str:
        """Get current model selection."""
        model_text = self._model_combo.currentText()
        if model_text == "Watterson TDL":
            return "watterson"
        elif model_text == "Vogler-Hoffmeyer":
            return "vogler_hoffmeyer"
        else:
            return "vogler"

    def set_vogler_hoffmeyer_config(self, config: VoglerHoffmeyerConfig):
        """Set Vogler-Hoffmeyer configuration."""
        self._vh_config = config
        self._update_vh_widgets_from_config()

    def get_vogler_hoffmeyer_config(self) -> VoglerHoffmeyerConfig:
        """Get current Vogler-Hoffmeyer configuration."""
        self._update_vh_config()
        return self._vh_config
