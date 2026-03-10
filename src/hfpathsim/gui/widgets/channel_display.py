"""Channel impulse/frequency response display widget."""

from typing import Optional
import numpy as np

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTabWidget
from PyQt6.QtCore import Qt

import pyqtgraph as pg


class ChannelDisplayWidget(QWidget):
    """Widget for displaying channel response.

    Shows both frequency response H(f) and impulse response h(t).
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Tab widget for different views
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        # Frequency response plot
        self._freq_widget = pg.PlotWidget(title="Channel Frequency Response |H(f)|")
        self._freq_widget.setLabel("left", "Magnitude", units="dB")
        self._freq_widget.setLabel("bottom", "Frequency", units="Hz")
        self._freq_widget.showGrid(x=True, y=True, alpha=0.3)
        self._freq_widget.setBackground("#1e1e1e")

        self._freq_curve = self._freq_widget.plot(
            pen=pg.mkPen("#00aaff", width=1.5)
        )

        self._tabs.addTab(self._freq_widget, "Frequency Response")

        # Impulse response plot
        self._impulse_widget = pg.PlotWidget(title="Channel Impulse Response |h(t)|")
        self._impulse_widget.setLabel("left", "Magnitude")
        self._impulse_widget.setLabel("bottom", "Delay", units="ms")
        self._impulse_widget.showGrid(x=True, y=True, alpha=0.3)
        self._impulse_widget.setBackground("#1e1e1e")

        self._impulse_curve = self._impulse_widget.plot(
            pen=pg.mkPen("#00ff88", width=1.5)
        )

        self._tabs.addTab(self._impulse_widget, "Impulse Response")

        # Phase response plot
        self._phase_widget = pg.PlotWidget(title="Channel Phase Response")
        self._phase_widget.setLabel("left", "Phase", units="rad")
        self._phase_widget.setLabel("bottom", "Frequency", units="Hz")
        self._phase_widget.showGrid(x=True, y=True, alpha=0.3)
        self._phase_widget.setBackground("#1e1e1e")

        self._phase_curve = self._phase_widget.plot(
            pen=pg.mkPen("#ffaa00", width=1.5)
        )

        self._tabs.addTab(self._phase_widget, "Phase Response")

        # Group delay plot
        self._delay_widget = pg.PlotWidget(title="Group Delay")
        self._delay_widget.setLabel("left", "Delay", units="ms")
        self._delay_widget.setLabel("bottom", "Frequency", units="Hz")
        self._delay_widget.showGrid(x=True, y=True, alpha=0.3)
        self._delay_widget.setBackground("#1e1e1e")

        self._delay_curve = self._delay_widget.plot(
            pen=pg.mkPen("#ff5588", width=1.5)
        )

        self._tabs.addTab(self._delay_widget, "Group Delay")

    def update_transfer_function(
        self,
        freq_axis: np.ndarray,
        H: np.ndarray,
    ):
        """Update the frequency response display.

        Args:
            freq_axis: Frequency axis in Hz
            H: Complex transfer function
        """
        # Sort by frequency for proper plotting
        sort_idx = np.argsort(freq_axis)
        freq_sorted = freq_axis[sort_idx]
        H_sorted = H[sort_idx]

        # Magnitude in dB
        mag_db = 20 * np.log10(np.abs(H_sorted) + 1e-10)
        self._freq_curve.setData(freq_sorted, mag_db)

        # Phase (unwrapped)
        phase = np.unwrap(np.angle(H_sorted))
        self._phase_curve.setData(freq_sorted, phase)

        # Group delay (negative derivative of phase)
        if len(freq_sorted) > 2:
            df = np.diff(freq_sorted)
            dphi = np.diff(phase)
            group_delay = -dphi / (2 * np.pi * df) * 1000  # Convert to ms
            # Smooth the group delay
            if len(group_delay) > 5:
                group_delay = np.convolve(
                    group_delay, np.ones(5) / 5, mode="same"
                )
            self._delay_curve.setData(
                freq_sorted[:-1], group_delay
            )

    def update_impulse_response(
        self,
        delay_axis: np.ndarray,
        h: np.ndarray,
    ):
        """Update the impulse response display.

        Args:
            delay_axis: Delay axis in ms
            h: Complex impulse response
        """
        # Only show positive delays (causal system)
        # and truncate to significant portion
        mag = np.abs(h)
        threshold = np.max(mag) * 0.001

        # Find last significant sample
        significant = np.where(mag > threshold)[0]
        if len(significant) > 0:
            last_sig = min(significant[-1] + 10, len(mag))
        else:
            last_sig = len(mag) // 4

        self._impulse_curve.setData(
            delay_axis[:last_sig],
            mag[:last_sig],
        )

    def reset_view(self):
        """Reset all plot views to auto-range."""
        self._freq_widget.autoRange()
        self._impulse_widget.autoRange()
        self._phase_widget.autoRange()
        self._delay_widget.autoRange()
