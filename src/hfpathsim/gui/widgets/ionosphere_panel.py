"""Ionosphere configuration panel widget."""

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
    QSlider,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QFrame,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QLineEdit,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

from hfpathsim.core.raytracing.ionosphere import IonosphereProfile, create_simple_profile
from hfpathsim.iono.sporadic_e import SporadicEConfig, SporadicEPreset
from hfpathsim.iono.geomagnetic import GeomagneticIndices


class IonospherePanel(QWidget):
    """Panel for ionospheric modeling and ray tracing.

    Features:
    - Data source selection (Manual, GIRO, IRI-2020)
    - Path geometry (TX/RX coordinates)
    - Ray tracing controls and results
    - Sporadic-E layer configuration
    - Geomagnetic indices and presets
    - GIRO station selection and auto-update
    """

    ionosphere_profile_changed = pyqtSignal(IonosphereProfile)
    sporadic_e_changed = pyqtSignal(SporadicEConfig)
    geomagnetic_changed = pyqtSignal(GeomagneticIndices)
    ray_tracing_requested = pyqtSignal(dict)  # TX/RX coords, frequency
    data_source_changed = pyqtSignal(str)
    giro_station_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._profile = create_simple_profile(foF2=7.5, hmF2=300.0)
        self._es_config = SporadicEConfig()
        self._geomag = GeomagneticIndices.moderate()
        self._discovered_modes: List[dict] = []

        # GIRO auto-update timer
        self._giro_timer = QTimer()
        self._giro_timer.timeout.connect(self._fetch_giro_data)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the ionosphere panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Top row: Data source selection
        source_row = QHBoxLayout()

        source_row.addWidget(QLabel("Data Source:"))

        self._source_group = QButtonGroup(self)
        self._manual_radio = QRadioButton("Manual")
        self._manual_radio.setChecked(True)
        self._giro_radio = QRadioButton("GIRO")
        self._iri_radio = QRadioButton("IRI-2020")

        self._source_group.addButton(self._manual_radio, 0)
        self._source_group.addButton(self._giro_radio, 1)
        self._source_group.addButton(self._iri_radio, 2)

        source_row.addWidget(self._manual_radio)
        source_row.addWidget(self._giro_radio)
        source_row.addWidget(self._iri_radio)

        source_row.addStretch()

        self._fetch_btn = QPushButton("Fetch Now")
        self._fetch_btn.setEnabled(False)
        source_row.addWidget(self._fetch_btn)

        layout.addLayout(source_row)

        # Main content: 2x2 grid of groups
        content_layout = QGridLayout()

        # Top-left: Path Geometry
        path_group = QGroupBox("Path Geometry")
        path_layout = QGridLayout(path_group)

        path_layout.addWidget(QLabel("TX:"), 0, 0)
        self._tx_lat_spin = QDoubleSpinBox()
        self._tx_lat_spin.setRange(-90.0, 90.0)
        self._tx_lat_spin.setValue(38.9)
        self._tx_lat_spin.setSuffix("°N")
        self._tx_lat_spin.setSingleStep(1.0)
        path_layout.addWidget(self._tx_lat_spin, 0, 1)

        self._tx_lon_spin = QDoubleSpinBox()
        self._tx_lon_spin.setRange(-180.0, 180.0)
        self._tx_lon_spin.setValue(-77.0)
        self._tx_lon_spin.setSuffix("°W")
        self._tx_lon_spin.setSingleStep(1.0)
        path_layout.addWidget(self._tx_lon_spin, 0, 2)

        path_layout.addWidget(QLabel("RX:"), 1, 0)
        self._rx_lat_spin = QDoubleSpinBox()
        self._rx_lat_spin.setRange(-90.0, 90.0)
        self._rx_lat_spin.setValue(51.5)
        self._rx_lat_spin.setSuffix("°N")
        self._rx_lat_spin.setSingleStep(1.0)
        path_layout.addWidget(self._rx_lat_spin, 1, 1)

        self._rx_lon_spin = QDoubleSpinBox()
        self._rx_lon_spin.setRange(-180.0, 180.0)
        self._rx_lon_spin.setValue(-0.1)
        self._rx_lon_spin.setSuffix("°E")
        self._rx_lon_spin.setSingleStep(1.0)
        path_layout.addWidget(self._rx_lon_spin, 1, 2)

        self._distance_label = QLabel("Distance: 5892 km")
        path_layout.addWidget(self._distance_label, 2, 0, 1, 2)

        self._bearing_label = QLabel("Bearing: 51.2°")
        path_layout.addWidget(self._bearing_label, 2, 2)

        content_layout.addWidget(path_group, 0, 0)

        # Top-right: Ray Tracing
        ray_group = QGroupBox("Ray Tracing")
        ray_layout = QVBoxLayout(ray_group)

        ray_enable_row = QHBoxLayout()
        self._ray_enable_check = QCheckBox("Enable Ray Tracing")
        ray_enable_row.addWidget(self._ray_enable_check)

        ray_enable_row.addWidget(QLabel("Max Hops:"))
        self._max_hops_spin = QSpinBox()
        self._max_hops_spin.setRange(1, 5)
        self._max_hops_spin.setValue(3)
        ray_enable_row.addWidget(self._max_hops_spin)

        ray_enable_row.addStretch()

        self._trace_btn = QPushButton("Trace")
        ray_enable_row.addWidget(self._trace_btn)

        ray_layout.addLayout(ray_enable_row)

        ray_layout.addWidget(QLabel("Discovered Modes:"))

        # Mode results table
        self._modes_table = QTableWidget()
        self._modes_table.setColumnCount(4)
        self._modes_table.setHorizontalHeaderLabels(["Mode", "Delay", "MUF", "Angle"])
        self._modes_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._modes_table.setMaximumHeight(100)
        self._modes_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        ray_layout.addWidget(self._modes_table)

        content_layout.addWidget(ray_group, 0, 1)

        # Bottom-left: Sporadic-E
        es_group = QGroupBox("Sporadic-E")
        es_layout = QGridLayout(es_group)

        self._es_enable_check = QCheckBox("Enable Es Layer")
        es_layout.addWidget(self._es_enable_check, 0, 0, 1, 2)

        es_layout.addWidget(QLabel("Preset:"), 1, 0)
        self._es_preset_combo = QComboBox()
        self._es_preset_combo.addItems(["Weak", "Moderate", "Strong", "Intense"])
        self._es_preset_combo.setCurrentIndex(1)
        es_layout.addWidget(self._es_preset_combo, 1, 1)

        es_layout.addWidget(QLabel("foEs:"), 2, 0)
        self._foEs_spin = QDoubleSpinBox()
        self._foEs_spin.setRange(2.0, 20.0)
        self._foEs_spin.setValue(6.0)
        self._foEs_spin.setSuffix(" MHz")
        self._foEs_spin.setSingleStep(0.5)
        es_layout.addWidget(self._foEs_spin, 2, 1)

        es_layout.addWidget(QLabel("hmEs:"), 3, 0)
        self._hmEs_spin = QDoubleSpinBox()
        self._hmEs_spin.setRange(80.0, 150.0)
        self._hmEs_spin.setValue(105.0)
        self._hmEs_spin.setSuffix(" km")
        self._hmEs_spin.setSingleStep(5.0)
        es_layout.addWidget(self._hmEs_spin, 3, 1)

        self._es_muf_label = QLabel("Es MUF: 18.4 MHz")
        es_layout.addWidget(self._es_muf_label, 4, 0, 1, 2)

        content_layout.addWidget(es_group, 1, 0)

        # Bottom-right: Geomagnetic
        geomag_group = QGroupBox("Geomagnetic")
        geomag_layout = QGridLayout(geomag_group)

        geomag_layout.addWidget(QLabel("Preset:"), 0, 0)
        self._geomag_preset_combo = QComboBox()
        self._geomag_preset_combo.addItems([
            "Quiet", "Moderate", "Disturbed", "Severe Storm",
            "Solar Maximum", "Solar Minimum"
        ])
        self._geomag_preset_combo.setCurrentIndex(1)
        geomag_layout.addWidget(self._geomag_preset_combo, 0, 1)

        geomag_layout.addWidget(QLabel("F10.7:"), 1, 0)
        self._f107_spin = QDoubleSpinBox()
        self._f107_spin.setRange(65.0, 350.0)
        self._f107_spin.setValue(120.0)
        self._f107_spin.setSuffix(" sfu")
        self._f107_spin.setSingleStep(10.0)
        geomag_layout.addWidget(self._f107_spin, 1, 1)

        geomag_layout.addWidget(QLabel("Kp:"), 2, 0)
        self._kp_slider = QSlider(Qt.Orientation.Horizontal)
        self._kp_slider.setRange(0, 9)
        self._kp_slider.setValue(3)
        geomag_layout.addWidget(self._kp_slider, 2, 1)

        self._kp_label = QLabel("3")
        self._kp_label.setFixedWidth(20)
        geomag_layout.addWidget(self._kp_label, 2, 2)

        geomag_layout.addWidget(QLabel("Dst:"), 3, 0)
        self._dst_spin = QDoubleSpinBox()
        self._dst_spin.setRange(-500.0, 50.0)
        self._dst_spin.setValue(-20.0)
        self._dst_spin.setSuffix(" nT")
        self._dst_spin.setSingleStep(10.0)
        geomag_layout.addWidget(self._dst_spin, 3, 1)

        self._storm_phase_label = QLabel("Storm Phase: Quiet")
        geomag_layout.addWidget(self._storm_phase_label, 4, 0, 1, 2)

        content_layout.addWidget(geomag_group, 1, 1)

        layout.addLayout(content_layout)

        # GIRO Station row
        giro_group = QGroupBox("GIRO Station")
        giro_layout = QHBoxLayout(giro_group)

        giro_layout.addWidget(QLabel("Station:"))
        self._station_combo = QComboBox()
        self._station_combo.addItems([
            "Boulder (BC840)",
            "Wallops Island (WP937)",
            "Eglin (EG931)",
            "Juliusruh (JR055)",
            "Dourbes (DB049)",
            "Rome (RO041)",
            "Millstone Hill (MH453)",
            "Ascension Island (AS00Q)",
        ])
        giro_layout.addWidget(self._station_combo)

        self._auto_update_check = QCheckBox("Auto-Update:")
        giro_layout.addWidget(self._auto_update_check)

        giro_layout.addWidget(QLabel("Every"))
        self._update_interval_spin = QSpinBox()
        self._update_interval_spin.setRange(1, 60)
        self._update_interval_spin.setValue(15)
        self._update_interval_spin.setSuffix(" min")
        giro_layout.addWidget(self._update_interval_spin)

        giro_layout.addStretch()

        self._giro_status_label = QLabel("Last: --")
        giro_layout.addWidget(self._giro_status_label)

        layout.addWidget(giro_group)

    def _connect_signals(self):
        """Connect widget signals."""
        # Data source selection
        self._source_group.buttonClicked.connect(self._on_source_changed)

        # Fetch button
        self._fetch_btn.clicked.connect(self._fetch_data)

        # Ray tracing
        self._trace_btn.clicked.connect(self._on_trace_requested)
        self._ray_enable_check.toggled.connect(self._update_ray_enabled)

        # Sporadic-E preset
        self._es_preset_combo.currentTextChanged.connect(self._on_es_preset_changed)
        self._es_enable_check.toggled.connect(self._update_es_enabled)

        # Geomagnetic preset
        self._geomag_preset_combo.currentTextChanged.connect(self._on_geomag_preset_changed)
        self._kp_slider.valueChanged.connect(lambda v: self._kp_label.setText(str(v)))

        # Path geometry changes
        for spin in [self._tx_lat_spin, self._tx_lon_spin, self._rx_lat_spin, self._rx_lon_spin]:
            spin.valueChanged.connect(self._update_path_info)

        # GIRO auto-update
        self._auto_update_check.toggled.connect(self._on_auto_update_changed)

        # Initial state
        self._update_ray_enabled(self._ray_enable_check.isChecked())
        self._update_es_enabled(self._es_enable_check.isChecked())
        self._update_path_info()

    def _on_source_changed(self, button):
        """Handle data source change."""
        source_id = self._source_group.id(button)
        source_names = {0: "manual", 1: "giro", 2: "iri"}
        source = source_names.get(source_id, "manual")

        # Enable/disable fetch button
        self._fetch_btn.setEnabled(source != "manual")

        self.data_source_changed.emit(source)

    def _fetch_data(self):
        """Fetch data from selected source."""
        source_id = self._source_group.checkedId()

        if source_id == 1:  # GIRO
            self._fetch_giro_data()
        elif source_id == 2:  # IRI
            self._fetch_iri_data()

    def _fetch_giro_data(self):
        """Fetch data from GIRO."""
        station_text = self._station_combo.currentText()
        # Extract station code from "(CODE)" format
        station_code = station_text.split("(")[-1].rstrip(")")

        try:
            from hfpathsim.iono.giro import GIROClient

            client = GIROClient(station_code)
            ionogram = client.fetch_latest()

            if ionogram:
                # Update UI with fetched values
                self._foEs_spin.setValue(ionogram.foF2 if ionogram.foF2 else 7.5)
                # Note: GIRO provides foF2, not foEs directly

                self._giro_status_label.setText(
                    f"Last: foF2={ionogram.foF2:.1f} MHz"
                )

        except Exception as e:
            self._giro_status_label.setText(f"Error: {str(e)[:30]}")

    def _fetch_iri_data(self):
        """Fetch data from IRI model."""
        try:
            from hfpathsim.iono.iri import IRIModel

            model = IRIModel()
            if model.available:
                # Use midpoint of path for IRI query
                lat = (self._tx_lat_spin.value() + self._rx_lat_spin.value()) / 2
                lon = (self._tx_lon_spin.value() + self._rx_lon_spin.value()) / 2

                from datetime import datetime
                params = model.get_parameters(lat, lon, datetime.utcnow())

                if params:
                    self._giro_status_label.setText(
                        f"IRI: foF2={params.get('foF2', 0):.1f} MHz"
                    )

        except Exception as e:
            self._giro_status_label.setText(f"IRI Error: {str(e)[:25]}")

    def _update_path_info(self):
        """Update path distance and bearing display."""
        try:
            from hfpathsim.core.raytracing.geometry import great_circle_distance, initial_bearing

            tx_lat = self._tx_lat_spin.value()
            tx_lon = self._tx_lon_spin.value()
            rx_lat = self._rx_lat_spin.value()
            rx_lon = self._rx_lon_spin.value()

            distance = great_circle_distance(tx_lat, tx_lon, rx_lat, rx_lon)
            bearing = initial_bearing(tx_lat, tx_lon, rx_lat, rx_lon)

            self._distance_label.setText(f"Distance: {distance:.0f} km")
            self._bearing_label.setText(f"Bearing: {bearing:.1f}°")

        except ImportError:
            # Fallback if geometry module not available
            pass

    def _update_ray_enabled(self, enabled: bool):
        """Update ray tracing controls enabled state."""
        self._max_hops_spin.setEnabled(enabled)
        self._trace_btn.setEnabled(enabled)
        self._modes_table.setEnabled(enabled)

    def _update_es_enabled(self, enabled: bool):
        """Update Es controls enabled state."""
        for widget in [self._es_preset_combo, self._foEs_spin, self._hmEs_spin]:
            widget.setEnabled(enabled)

    def _on_es_preset_changed(self, preset_name: str):
        """Apply Es preset values."""
        presets = {
            "Weak": (3.0, 105.0),
            "Moderate": (6.0, 105.0),
            "Strong": (10.0, 100.0),
            "Intense": (15.0, 100.0),
        }

        if preset_name in presets:
            foEs, hmEs = presets[preset_name]
            self._foEs_spin.setValue(foEs)
            self._hmEs_spin.setValue(hmEs)
            self._update_es_muf()

    def _update_es_muf(self):
        """Update Es MUF display."""
        foEs = self._foEs_spin.value()
        # Simple approximation: Es MUF ~ foEs * 3 for typical paths
        es_muf = foEs * 3.0
        self._es_muf_label.setText(f"Es MUF: {es_muf:.1f} MHz")

    def _on_geomag_preset_changed(self, preset_name: str):
        """Apply geomagnetic preset values."""
        presets = {
            "Quiet": GeomagneticIndices.quiet(),
            "Moderate": GeomagneticIndices.moderate(),
            "Disturbed": GeomagneticIndices.disturbed(),
            "Severe Storm": GeomagneticIndices.severe_storm(),
            "Solar Maximum": GeomagneticIndices.solar_maximum(),
            "Solar Minimum": GeomagneticIndices.solar_minimum(),
        }

        if preset_name in presets:
            indices = presets[preset_name]
            self._f107_spin.setValue(indices.f10_7)
            self._kp_slider.setValue(int(indices.kp))
            self._dst_spin.setValue(indices.dst)
            self._update_storm_phase()

    def _update_storm_phase(self):
        """Update storm phase display."""
        dst = self._dst_spin.value()

        if dst > -20:
            phase = "Quiet"
        elif dst > -50:
            phase = "Minor Storm"
        elif dst > -100:
            phase = "Moderate Storm"
        elif dst > -200:
            phase = "Severe Storm"
        else:
            phase = "Extreme Storm"

        self._storm_phase_label.setText(f"Storm Phase: {phase}")

    def _on_trace_requested(self):
        """Handle ray tracing request."""
        request = {
            "tx_lat": self._tx_lat_spin.value(),
            "tx_lon": self._tx_lon_spin.value(),
            "rx_lat": self._rx_lat_spin.value(),
            "rx_lon": self._rx_lon_spin.value(),
            "max_hops": self._max_hops_spin.value(),
        }
        self.ray_tracing_requested.emit(request)

    def _on_auto_update_changed(self, enabled: bool):
        """Handle auto-update toggle."""
        if enabled:
            interval_ms = self._update_interval_spin.value() * 60 * 1000
            self._giro_timer.start(interval_ms)
            self._fetch_giro_data()  # Initial fetch
        else:
            self._giro_timer.stop()

    def set_discovered_modes(self, modes: List[dict]):
        """Set the discovered propagation modes table."""
        self._discovered_modes = modes
        self._modes_table.setRowCount(len(modes))

        for i, mode in enumerate(modes):
            self._modes_table.setItem(i, 0, QTableWidgetItem(mode.get("name", "")))
            self._modes_table.setItem(i, 1, QTableWidgetItem(f"{mode.get('delay_ms', 0):.1f} ms"))
            self._modes_table.setItem(i, 2, QTableWidgetItem(f"{mode.get('muf_mhz', 0):.1f} MHz"))
            self._modes_table.setItem(i, 3, QTableWidgetItem(f"{mode.get('angle_deg', 0):.1f}°"))

    def get_sporadic_e_config(self) -> SporadicEConfig:
        """Get current Sporadic-E configuration."""
        return SporadicEConfig(
            enabled=self._es_enable_check.isChecked(),
            foEs_mhz=self._foEs_spin.value(),
            hmEs_km=self._hmEs_spin.value(),
        )

    def get_geomagnetic_indices(self) -> GeomagneticIndices:
        """Get current geomagnetic indices."""
        return GeomagneticIndices(
            f10_7=self._f107_spin.value(),
            kp=float(self._kp_slider.value()),
            dst=self._dst_spin.value(),
        )

    def get_path_geometry(self) -> dict:
        """Get current path geometry."""
        return {
            "tx_lat": self._tx_lat_spin.value(),
            "tx_lon": self._tx_lon_spin.value(),
            "rx_lat": self._rx_lat_spin.value(),
            "rx_lon": self._rx_lon_spin.value(),
        }

    def set_ionosphere_profile(self, profile: IonosphereProfile):
        """Set ionosphere profile and emit signal."""
        self._profile = profile
        self.ionosphere_profile_changed.emit(profile)

    def get_data_source(self) -> str:
        """Get current data source selection."""
        source_names = {0: "manual", 1: "giro", 2: "iri"}
        return source_names.get(self._source_group.checkedId(), "manual")
