#!/usr/bin/env python3
"""Test dashboard with automatic start to capture debug output."""

import sys
import time
sys.path.insert(0, '/home/n4hy/HFPathSimulatorLiveData/src')

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from hfpathsim.gui.main_window import MainWindow
from hfpathsim.input.siggen import SignalGenerator, WaveformType


def main():
    print("=" * 60)
    print("DASHBOARD DEBUG TEST")
    print("=" * 60)
    print("This will start the dashboard with RTTY signal generator")
    print("and automatically click Start to capture debug output.")
    print("Watch for sample rate debug messages.")
    print("=" * 60)
    print()

    app = QApplication(sys.argv)

    # Create signal generator
    siggen = SignalGenerator(WaveformType.RTTY, sample_rate_hz=8000.0)
    if not siggen.open():
        print("ERROR: Failed to open signal generator")
        return 1

    print(f"Signal generator: {siggen.waveform_type.value} at {siggen.sample_rate}Hz")

    # Create window
    window = MainWindow()
    window._input_source = siggen

    # Set Watterson model for audio-rate signals
    window._current_model = "watterson"
    window._channel_panel._model_combo.setCurrentText("Watterson TDL")
    print(f"Channel model: Watterson TDL (optimal for audio-rate signals)")

    # Auto-start processing after 500ms
    def auto_start():
        print("\n*** AUTO-STARTING PROCESSING ***\n")
        window._start_processing()

    # Auto-stop after 3 seconds
    def auto_stop():
        print("\n*** AUTO-STOPPING PROCESSING ***\n")
        window._stop_processing()
        print("\nTest complete - check debug output above")
        app.quit()

    QTimer.singleShot(500, auto_start)
    QTimer.singleShot(3500, auto_stop)

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
