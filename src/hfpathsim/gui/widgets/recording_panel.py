"""Channel recording panel widget."""

from pathlib import Path
from typing import Optional

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
    QLineEdit,
    QFileDialog,
    QFrame,
    QTextEdit,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer


class RecordingPanel(QWidget):
    """Panel for channel state recording and playback.

    Features:
    - Record channel snapshots at configurable rate
    - Playback recorded channel states
    - Metadata entry (description, tags)
    - Multiple output formats (NPZ, HDF5, JSON)
    """

    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal(str)  # filename
    playback_started = pyqtSignal(str)  # filename
    playback_stopped = pyqtSignal()
    playback_position_changed = pyqtSignal(float)  # position 0-1

    def __init__(self, parent=None):
        super().__init__(parent)

        self._recording = False
        self._playing = False
        self._paused = False
        self._current_file: Optional[str] = None
        self._snapshot_count = 0
        self._elapsed_time = 0.0

        # Timer for updating elapsed time
        self._elapsed_timer = QTimer()
        self._elapsed_timer.timeout.connect(self._update_elapsed)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the recording panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Main content: Record and Playback groups side by side
        content_layout = QHBoxLayout()

        # Record group
        record_group = QGroupBox("Record")
        record_layout = QVBoxLayout(record_group)

        # Record controls row
        record_btn_row = QHBoxLayout()

        self._record_btn = QPushButton("● Record")
        self._record_btn.setStyleSheet("QPushButton { color: red; }")
        record_btn_row.addWidget(self._record_btn)

        self._stop_record_btn = QPushButton("■ Stop")
        self._stop_record_btn.setEnabled(False)
        record_btn_row.addWidget(self._stop_record_btn)

        record_layout.addLayout(record_btn_row)

        # Record settings
        settings_layout = QGridLayout()

        settings_layout.addWidget(QLabel("Snapshot Rate:"), 0, 0)
        self._snapshot_rate_spin = QSpinBox()
        self._snapshot_rate_spin.setRange(1, 100)
        self._snapshot_rate_spin.setValue(10)
        self._snapshot_rate_spin.setSuffix(" Hz")
        settings_layout.addWidget(self._snapshot_rate_spin, 0, 1)

        settings_layout.addWidget(QLabel("Max Duration:"), 1, 0)
        self._max_duration_spin = QSpinBox()
        self._max_duration_spin.setRange(1, 7200)
        self._max_duration_spin.setValue(3600)
        self._max_duration_spin.setSuffix(" sec")
        settings_layout.addWidget(self._max_duration_spin, 1, 1)

        settings_layout.addWidget(QLabel("Format:"), 2, 0)
        self._format_combo = QComboBox()
        self._format_combo.addItems(["NPZ", "HDF5", "JSON"])
        settings_layout.addWidget(self._format_combo, 2, 1)

        record_layout.addLayout(settings_layout)

        # Recording status
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        status_layout = QGridLayout(status_frame)
        status_layout.setContentsMargins(4, 4, 4, 4)

        self._record_status_label = QLabel("Status: Idle")
        status_layout.addWidget(self._record_status_label, 0, 0, 1, 2)

        status_layout.addWidget(QLabel("Snapshots:"), 1, 0)
        self._snapshot_count_label = QLabel("0")
        status_layout.addWidget(self._snapshot_count_label, 1, 1)

        status_layout.addWidget(QLabel("Elapsed:"), 2, 0)
        self._elapsed_label = QLabel("0.0 sec")
        status_layout.addWidget(self._elapsed_label, 2, 1)

        record_layout.addWidget(status_frame)

        record_layout.addStretch()

        content_layout.addWidget(record_group)

        # Playback group
        playback_group = QGroupBox("Playback")
        playback_layout = QVBoxLayout(playback_group)

        # Playback controls row
        play_btn_row = QHBoxLayout()

        self._play_btn = QPushButton("▶ Play")
        play_btn_row.addWidget(self._play_btn)

        self._pause_btn = QPushButton("⏸ Pause")
        self._pause_btn.setEnabled(False)
        play_btn_row.addWidget(self._pause_btn)

        self._stop_play_btn = QPushButton("■ Stop")
        self._stop_play_btn.setEnabled(False)
        play_btn_row.addWidget(self._stop_play_btn)

        playback_layout.addLayout(play_btn_row)

        # File selection
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("File:"))
        self._file_edit = QLineEdit()
        self._file_edit.setReadOnly(True)
        self._file_edit.setPlaceholderText("No file loaded")
        file_row.addWidget(self._file_edit)

        self._browse_btn = QPushButton("...")
        self._browse_btn.setFixedWidth(30)
        file_row.addWidget(self._browse_btn)

        playback_layout.addLayout(file_row)

        # Playback info
        info_layout = QGridLayout()

        info_layout.addWidget(QLabel("Duration:"), 0, 0)
        self._duration_label = QLabel("--")
        info_layout.addWidget(self._duration_label, 0, 1)

        info_layout.addWidget(QLabel("Progress:"), 1, 0)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        info_layout.addWidget(self._progress_bar, 1, 1)

        playback_layout.addLayout(info_layout)

        # Playback options
        options_row = QHBoxLayout()

        options_row.addWidget(QLabel("Rate:"))
        self._playback_rate_combo = QComboBox()
        self._playback_rate_combo.addItems(["0.25x", "0.5x", "1.0x", "2.0x", "4.0x"])
        self._playback_rate_combo.setCurrentIndex(2)  # 1.0x
        options_row.addWidget(self._playback_rate_combo)

        self._loop_check = QCheckBox("Loop")
        options_row.addWidget(self._loop_check)

        options_row.addStretch()

        playback_layout.addLayout(options_row)

        # Loaded file info
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        info_frame_layout = QVBoxLayout(info_frame)
        info_frame_layout.setContentsMargins(4, 4, 4, 4)

        self._model_label = QLabel("Model: --")
        info_frame_layout.addWidget(self._model_label)

        self._params_label = QLabel("Parameters: --")
        info_frame_layout.addWidget(self._params_label)

        playback_layout.addWidget(info_frame)

        playback_layout.addStretch()

        content_layout.addWidget(playback_group)

        layout.addLayout(content_layout)

        # Metadata group
        meta_group = QGroupBox("Metadata")
        meta_layout = QGridLayout(meta_group)

        meta_layout.addWidget(QLabel("Description:"), 0, 0, Qt.AlignmentFlag.AlignTop)
        self._description_edit = QLineEdit()
        self._description_edit.setPlaceholderText("Recording description...")
        meta_layout.addWidget(self._description_edit, 0, 1)

        meta_layout.addWidget(QLabel("Tags:"), 1, 0)
        self._tags_edit = QLineEdit()
        self._tags_edit.setPlaceholderText("tag1, tag2, tag3...")
        meta_layout.addWidget(self._tags_edit, 1, 1)

        layout.addWidget(meta_group)

    def _connect_signals(self):
        """Connect widget signals."""
        self._record_btn.clicked.connect(self._on_record)
        self._stop_record_btn.clicked.connect(self._on_stop_record)

        self._play_btn.clicked.connect(self._on_play)
        self._pause_btn.clicked.connect(self._on_pause)
        self._stop_play_btn.clicked.connect(self._on_stop_play)

        self._browse_btn.clicked.connect(self._browse_file)

    def _on_record(self):
        """Start recording."""
        if self._recording:
            return

        self._recording = True
        self._snapshot_count = 0
        self._elapsed_time = 0.0

        self._record_btn.setEnabled(False)
        self._stop_record_btn.setEnabled(True)
        self._record_status_label.setText("Status: Recording...")
        self._record_status_label.setStyleSheet("color: red;")

        self._elapsed_timer.start(100)  # Update every 100ms

        self.recording_started.emit()

    def _on_stop_record(self):
        """Stop recording."""
        if not self._recording:
            return

        self._recording = False
        self._elapsed_timer.stop()

        self._record_btn.setEnabled(True)
        self._stop_record_btn.setEnabled(False)
        self._record_status_label.setText("Status: Idle")
        self._record_status_label.setStyleSheet("")

        # Generate filename based on format
        format_ext = {
            "NPZ": ".npz",
            "HDF5": ".h5",
            "JSON": ".json",
        }
        ext = format_ext.get(self._format_combo.currentText(), ".npz")

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"channel_rec_{timestamp}{ext}"

        self._current_file = filename
        self.recording_stopped.emit(filename)

    def _update_elapsed(self):
        """Update elapsed time display."""
        self._elapsed_time += 0.1
        self._elapsed_label.setText(f"{self._elapsed_time:.1f} sec")

        # Check max duration
        if self._elapsed_time >= self._max_duration_spin.value():
            self._on_stop_record()

    def _browse_file(self):
        """Browse for playback file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Recording File",
            "",
            "Recording Files (*.npz *.h5 *.json);;All Files (*)",
        )

        if filepath:
            self._load_file(filepath)

    def _load_file(self, filepath: str):
        """Load a recording file."""
        self._current_file = filepath
        self._file_edit.setText(Path(filepath).name)

        try:
            from hfpathsim.core.recording import ChannelPlayer

            player = ChannelPlayer()
            player.load(filepath)

            # Update info display
            self._duration_label.setText(f"{player.duration:.1f} sec")
            self._model_label.setText(f"Model: {player.metadata.channel_model}")
            self._params_label.setText(
                f"foF2={player.metadata.foF2_mhz:.1f} MHz, "
                f"τ={player.metadata.delay_spread_ms:.1f} ms, "
                f"ν={player.metadata.doppler_spread_hz:.1f} Hz"
            )

            self._play_btn.setEnabled(True)

        except Exception as e:
            self._duration_label.setText("Error loading")
            self._model_label.setText(f"Error: {str(e)[:50]}")
            self._params_label.setText("--")
            self._play_btn.setEnabled(False)

    def _on_play(self):
        """Start playback."""
        if not self._current_file:
            return

        if self._paused:
            # Resume from pause
            self._paused = False
            self._pause_btn.setText("⏸ Pause")
        else:
            # Start new playback
            self._playing = True

        self._play_btn.setEnabled(False)
        self._pause_btn.setEnabled(True)
        self._stop_play_btn.setEnabled(True)

        self.playback_started.emit(self._current_file)

    def _on_pause(self):
        """Pause/resume playback."""
        if self._paused:
            # Resume
            self._paused = False
            self._pause_btn.setText("⏸ Pause")
        else:
            # Pause
            self._paused = True
            self._pause_btn.setText("▶ Resume")

        self._play_btn.setEnabled(self._paused)

    def _on_stop_play(self):
        """Stop playback."""
        self._playing = False
        self._paused = False

        self._play_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._pause_btn.setText("⏸ Pause")
        self._stop_play_btn.setEnabled(False)

        self._progress_bar.setValue(0)

        self.playback_stopped.emit()

    def update_snapshot_count(self, count: int):
        """Update snapshot count display."""
        self._snapshot_count = count
        self._snapshot_count_label.setText(str(count))

    def update_playback_progress(self, progress: float):
        """Update playback progress (0.0 to 1.0)."""
        self._progress_bar.setValue(int(progress * 100))
        self.playback_position_changed.emit(progress)

    def get_snapshot_rate(self) -> float:
        """Get snapshot rate in Hz."""
        return float(self._snapshot_rate_spin.value())

    def get_max_duration(self) -> float:
        """Get max duration in seconds."""
        return float(self._max_duration_spin.value())

    def get_format(self) -> str:
        """Get output format (npz, h5, json)."""
        return self._format_combo.currentText().lower()

    def get_playback_rate(self) -> float:
        """Get playback rate multiplier."""
        rate_text = self._playback_rate_combo.currentText()
        return float(rate_text.rstrip("x"))

    def is_loop_enabled(self) -> bool:
        """Check if loop is enabled."""
        return self._loop_check.isChecked()

    def get_metadata(self) -> dict:
        """Get recording metadata."""
        return {
            "description": self._description_edit.text(),
            "tags": [t.strip() for t in self._tags_edit.text().split(",") if t.strip()],
        }

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self._playing and not self._paused

    def is_paused(self) -> bool:
        """Check if playback is paused."""
        return self._paused
