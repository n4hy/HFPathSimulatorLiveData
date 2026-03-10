"""Parameter control panel widget."""

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QPushButton,
)
from PyQt6.QtCore import Qt, pyqtSignal

from hfpathsim.core.parameters import VoglerParameters, ITUCondition, PropagationMode


class ParameterPanel(QWidget):
    """Panel for controlling ionospheric and channel parameters."""

    parameters_changed = pyqtSignal(VoglerParameters)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._params = VoglerParameters()
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Ionospheric parameters group
        iono_group = QGroupBox("Ionospheric Parameters")
        iono_layout = QGridLayout(iono_group)

        # foF2
        iono_layout.addWidget(QLabel("foF2:"), 0, 0)
        self._foF2_spin = QDoubleSpinBox()
        self._foF2_spin.setRange(1.0, 20.0)
        self._foF2_spin.setValue(7.5)
        self._foF2_spin.setSuffix(" MHz")
        self._foF2_spin.setSingleStep(0.1)
        iono_layout.addWidget(self._foF2_spin, 0, 1)

        # hmF2
        iono_layout.addWidget(QLabel("hmF2:"), 0, 2)
        self._hmF2_spin = QDoubleSpinBox()
        self._hmF2_spin.setRange(150.0, 500.0)
        self._hmF2_spin.setValue(300.0)
        self._hmF2_spin.setSuffix(" km")
        self._hmF2_spin.setSingleStep(10.0)
        iono_layout.addWidget(self._hmF2_spin, 0, 3)

        # foE
        iono_layout.addWidget(QLabel("foE:"), 1, 0)
        self._foE_spin = QDoubleSpinBox()
        self._foE_spin.setRange(0.5, 5.0)
        self._foE_spin.setValue(3.0)
        self._foE_spin.setSuffix(" MHz")
        self._foE_spin.setSingleStep(0.1)
        iono_layout.addWidget(self._foE_spin, 1, 1)

        # hmE
        iono_layout.addWidget(QLabel("hmE:"), 1, 2)
        self._hmE_spin = QDoubleSpinBox()
        self._hmE_spin.setRange(90.0, 130.0)
        self._hmE_spin.setValue(110.0)
        self._hmE_spin.setSuffix(" km")
        self._hmE_spin.setSingleStep(5.0)
        iono_layout.addWidget(self._hmE_spin, 1, 3)

        layout.addWidget(iono_group)

        # Channel parameters group
        channel_group = QGroupBox("Channel Parameters")
        channel_layout = QGridLayout(channel_group)

        # Delay spread
        channel_layout.addWidget(QLabel("Delay Spread:"), 0, 0)
        self._delay_spin = QDoubleSpinBox()
        self._delay_spin.setRange(0.1, 10.0)
        self._delay_spin.setValue(2.0)
        self._delay_spin.setSuffix(" ms")
        self._delay_spin.setSingleStep(0.1)
        channel_layout.addWidget(self._delay_spin, 0, 1)

        # Doppler spread
        channel_layout.addWidget(QLabel("Doppler Spread:"), 0, 2)
        self._doppler_spin = QDoubleSpinBox()
        self._doppler_spin.setRange(0.01, 20.0)
        self._doppler_spin.setValue(1.0)
        self._doppler_spin.setSuffix(" Hz")
        self._doppler_spin.setSingleStep(0.1)
        channel_layout.addWidget(self._doppler_spin, 0, 3)

        # Operating frequency
        channel_layout.addWidget(QLabel("Frequency:"), 1, 0)
        self._freq_spin = QDoubleSpinBox()
        self._freq_spin.setRange(2.0, 30.0)
        self._freq_spin.setValue(10.0)
        self._freq_spin.setSuffix(" MHz")
        self._freq_spin.setSingleStep(0.5)
        channel_layout.addWidget(self._freq_spin, 1, 1)

        # Path length
        channel_layout.addWidget(QLabel("Path Length:"), 1, 2)
        self._path_spin = QDoubleSpinBox()
        self._path_spin.setRange(100.0, 10000.0)
        self._path_spin.setValue(1000.0)
        self._path_spin.setSuffix(" km")
        self._path_spin.setSingleStep(100.0)
        channel_layout.addWidget(self._path_spin, 1, 3)

        layout.addWidget(channel_group)

        # Propagation modes group
        modes_group = QGroupBox("Propagation Modes")
        modes_layout = QVBoxLayout(modes_group)

        modes_row = QHBoxLayout()
        self._mode_1F_check = QCheckBox("1F")
        self._mode_1F_check.setChecked(True)
        modes_row.addWidget(self._mode_1F_check)

        self._mode_2F_check = QCheckBox("2F")
        self._mode_2F_check.setChecked(True)
        modes_row.addWidget(self._mode_2F_check)

        self._mode_E_check = QCheckBox("E")
        self._mode_E_check.setChecked(False)
        modes_row.addWidget(self._mode_E_check)

        modes_layout.addLayout(modes_row)

        # Apply button
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._on_apply)
        modes_layout.addWidget(self._apply_btn)

        layout.addWidget(modes_group)

    def _connect_signals(self):
        """Connect widget signals."""
        # Auto-apply on value change (optional)
        # self._foF2_spin.valueChanged.connect(self._on_value_changed)
        pass

    def _on_apply(self):
        """Apply current parameters."""
        self._update_params()
        self.parameters_changed.emit(self._params)

    def _on_value_changed(self):
        """Handle value change (for auto-apply mode)."""
        self._update_params()
        self.parameters_changed.emit(self._params)

    def _update_params(self):
        """Update internal parameters from widget values."""
        # Build mode list
        modes = []
        if self._mode_1F_check.isChecked():
            modes.append(PropagationMode("1F2", True, 1.0, 0.0))
        if self._mode_2F_check.isChecked():
            modes.append(PropagationMode("2F2", True, 0.7, 1.5))
        if self._mode_E_check.isChecked():
            modes.append(PropagationMode("1E", True, 0.3, -0.5))

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

    def set_parameters(self, params: VoglerParameters):
        """Set widget values from parameters.

        Args:
            params: VoglerParameters instance
        """
        self._params = params

        # Block signals during update
        self._foF2_spin.blockSignals(True)
        self._hmF2_spin.blockSignals(True)
        self._foE_spin.blockSignals(True)
        self._hmE_spin.blockSignals(True)
        self._delay_spin.blockSignals(True)
        self._doppler_spin.blockSignals(True)
        self._freq_spin.blockSignals(True)
        self._path_spin.blockSignals(True)

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
        self._mode_1F_check.setChecked("1F2" in mode_names)
        self._mode_2F_check.setChecked("2F2" in mode_names)
        self._mode_E_check.setChecked("1E" in mode_names)

        # Unblock signals
        self._foF2_spin.blockSignals(False)
        self._hmF2_spin.blockSignals(False)
        self._foE_spin.blockSignals(False)
        self._hmE_spin.blockSignals(False)
        self._delay_spin.blockSignals(False)
        self._doppler_spin.blockSignals(False)
        self._freq_spin.blockSignals(False)
        self._path_spin.blockSignals(False)

    def get_parameters(self) -> VoglerParameters:
        """Get current parameters.

        Returns:
            VoglerParameters instance
        """
        self._update_params()
        return self._params
