"""Scattering function S(tau, nu) display widget."""

from typing import Optional
import numpy as np

from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt

import pyqtgraph as pg


class ScatteringWidget(QWidget):
    """Widget for displaying the scattering function S(tau, nu).

    Shows the 2D power distribution in the delay-Doppler domain,
    which characterizes the channel's time-frequency selectivity.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create graphics layout for plot + colorbar
        self._graphics_layout = pg.GraphicsLayoutWidget()
        self._graphics_layout.setBackground("#1e1e1e")
        layout.addWidget(self._graphics_layout)

        # Create plot item
        self._plot = self._graphics_layout.addPlot(
            title="Scattering Function S(\u03c4, \u03bd)"
        )
        self._plot.setLabel("left", "Doppler Shift", units="Hz")
        self._plot.setLabel("bottom", "Delay", units="ms")

        # Create image item for 2D display
        self._image = pg.ImageItem()
        self._plot.addItem(self._image)

        # Color map (viridis-like)
        cmap = pg.colormap.get("viridis")
        self._image.setColorMap(cmap)

        # Color bar - add to layout next to plot
        self._colorbar = pg.ColorBarItem(
            values=(0, 1),
            colorMap=cmap,
            label="Power (dB)",
        )
        self._colorbar.setImageItem(self._image)
        self._graphics_layout.addItem(self._colorbar)

        # Store axes for coordinate transform
        self._delay_axis = None
        self._doppler_axis = None

    def update_data(
        self,
        S: np.ndarray,
        delay_axis_ms: np.ndarray,
        doppler_axis_hz: np.ndarray,
    ):
        """Update the scattering function display.

        Args:
            S: 2D scattering function (doppler x delay)
            delay_axis_ms: Delay axis in milliseconds
            doppler_axis_hz: Doppler axis in Hz
        """
        self._delay_axis = delay_axis_ms
        self._doppler_axis = doppler_axis_hz

        # Convert to dB scale for better visualization
        S_db = 10 * np.log10(S + 1e-10)
        S_db = S_db - np.max(S_db)  # Normalize to 0 dB peak
        S_db = np.clip(S_db, -40, 0)  # Clip at -40 dB

        # Normalize to 0-1 for colormap
        S_norm = (S_db + 40) / 40

        # Set image data
        self._image.setImage(S_norm.T)  # Transpose for correct orientation

        # Set coordinate transform
        if len(delay_axis_ms) > 1 and len(doppler_axis_hz) > 1:
            d_delay = delay_axis_ms[1] - delay_axis_ms[0]
            d_doppler = doppler_axis_hz[1] - doppler_axis_hz[0]

            # Transform: scale and translate
            tr = pg.QtGui.QTransform()
            tr.translate(delay_axis_ms[0], doppler_axis_hz[0])
            tr.scale(d_delay, d_doppler)
            self._image.setTransform(tr)

    def reset_view(self):
        """Reset view to auto-range."""
        self._plot.getViewBox().autoRange()

    def get_value_at(self, delay_ms: float, doppler_hz: float) -> Optional[float]:
        """Get scattering function value at given point.

        Args:
            delay_ms: Delay in milliseconds
            doppler_hz: Doppler shift in Hz

        Returns:
            Normalized power value or None if out of bounds
        """
        if self._delay_axis is None or self._doppler_axis is None:
            return None

        # Find nearest indices
        delay_idx = np.argmin(np.abs(self._delay_axis - delay_ms))
        doppler_idx = np.argmin(np.abs(self._doppler_axis - doppler_hz))

        # Get image data
        img = self._image.image
        if img is None:
            return None

        if 0 <= doppler_idx < img.shape[0] and 0 <= delay_idx < img.shape[1]:
            return float(img[doppler_idx, delay_idx])

        return None
