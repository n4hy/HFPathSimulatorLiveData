"""Spectrum analyzer widget."""

from typing import Optional
import numpy as np

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from PyQt6.QtCore import Qt

import pyqtgraph as pg

# Try to import GPU acceleration
_gpu_available = False
_compute_spectrum_gpu = None

try:
    from hfpathsim.gpu import compute_spectrum_db, is_available
    if is_available():
        _gpu_available = True
        _compute_spectrum_gpu = compute_spectrum_db
except ImportError:
    pass


class SpectrumWidget(QWidget):
    """Widget for displaying signal spectrum.

    Shows real-time FFT spectrum with configurable averaging
    and display options. Uses GPU acceleration when available.
    """

    def __init__(self, title: str = "Spectrum", parent=None, use_gpu: bool = True):
        super().__init__(parent)

        self._title = title
        self._averaging = 4
        self._avg_buffer = []
        self._fft_size = 4096
        self._sample_rate = 2e6
        self._use_gpu = use_gpu and _gpu_available

        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Controls bar
        controls = QHBoxLayout()
        controls.setSpacing(8)

        controls.addWidget(QLabel(self._title))
        controls.addStretch()

        controls.addWidget(QLabel("Avg:"))
        self._avg_combo = QComboBox()
        self._avg_combo.addItems(["1", "2", "4", "8", "16"])
        self._avg_combo.setCurrentText("4")
        self._avg_combo.currentTextChanged.connect(self._on_avg_changed)
        controls.addWidget(self._avg_combo)

        controls.addWidget(QLabel("FFT:"))
        self._fft_combo = QComboBox()
        self._fft_combo.addItems(["1024", "2048", "4096", "8192"])
        self._fft_combo.setCurrentText("4096")
        self._fft_combo.currentTextChanged.connect(self._on_fft_changed)
        controls.addWidget(self._fft_combo)

        layout.addLayout(controls)

        # Spectrum plot
        self._plot = pg.PlotWidget()
        self._plot.setLabel("left", "Power", units="dB")
        self._plot.setLabel("bottom", "Frequency", units="Hz")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setBackground("#1e1e1e")
        self._plot.setYRange(-100, 0)

        layout.addWidget(self._plot)

        # Spectrum curve with fill
        self._curve = self._plot.plot(
            pen=pg.mkPen("#00aaff", width=1),
            fillLevel=-120,
            fillBrush=pg.mkBrush("#00aaff40"),
        )

        # Peak hold curve
        self._peak_curve = self._plot.plot(
            pen=pg.mkPen("#ff5555", width=0.5)
        )
        self._peak_data = None

        # Window function (Blackman-Harris for good dynamic range)
        self._window = np.blackman(self._fft_size)

    def _on_avg_changed(self, value: str):
        """Handle averaging change."""
        self._averaging = int(value)
        self._avg_buffer.clear()

    def _on_fft_changed(self, value: str):
        """Handle FFT size change."""
        self._fft_size = int(value)
        self._window = np.blackman(self._fft_size)
        self._avg_buffer.clear()
        self._peak_data = None

    def set_sample_rate(self, sample_rate: float):
        """Set the sample rate for frequency axis scaling.

        Args:
            sample_rate: Sample rate in Hz
        """
        self._sample_rate = sample_rate

    def set_use_gpu(self, use_gpu: bool):
        """Enable or disable GPU acceleration.

        Args:
            use_gpu: Whether to use GPU for spectrum computation
        """
        self._use_gpu = use_gpu and _gpu_available

    def is_using_gpu(self) -> bool:
        """Check if GPU acceleration is being used."""
        return self._use_gpu

    def update_data(self, samples: np.ndarray):
        """Update spectrum with new samples.

        Args:
            samples: Complex samples (at least fft_size)
        """
        if len(samples) < self._fft_size:
            # Pad with zeros
            padded = np.zeros(self._fft_size, dtype=np.complex64)
            padded[: len(samples)] = samples
            samples = padded

        # Apply window
        windowed = samples[: self._fft_size] * self._window

        # Compute spectrum (GPU or CPU)
        if self._use_gpu and _compute_spectrum_gpu is not None:
            # GPU-accelerated spectrum computation
            # Note: GPU compute_spectrum returns un-shifted spectrum
            power_db = _compute_spectrum_gpu(windowed.astype(np.complex64), 1.0)
            power_db = np.fft.fftshift(power_db)
        else:
            # CPU fallback
            spectrum = np.fft.fftshift(np.fft.fft(windowed))
            power_db = 20 * np.log10(np.abs(spectrum) + 1e-10)

        # Normalize by window power
        window_power = np.sum(self._window**2)
        power_db -= 10 * np.log10(window_power)

        # Averaging
        self._avg_buffer.append(power_db)
        if len(self._avg_buffer) > self._averaging:
            self._avg_buffer.pop(0)

        avg_power = np.mean(self._avg_buffer, axis=0)

        # Update peak hold
        if self._peak_data is None:
            self._peak_data = avg_power.copy()
        else:
            self._peak_data = np.maximum(self._peak_data, avg_power)
            # Slow decay
            self._peak_data -= 0.1

        # Frequency axis
        freq_axis = np.fft.fftshift(
            np.fft.fftfreq(self._fft_size, 1 / self._sample_rate)
        )

        # Update plots
        self._curve.setData(freq_axis, avg_power)
        self._peak_curve.setData(freq_axis, self._peak_data)

    def reset_view(self):
        """Reset view to auto-range."""
        self._plot.autoRange()
        self._peak_data = None
        self._avg_buffer.clear()

    def clear_peak_hold(self):
        """Clear peak hold data."""
        self._peak_data = None
