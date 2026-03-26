#!/usr/bin/env python3
"""Test dashboard with audio output enabled."""

import sys
import time
sys.path.insert(0, '/home/n4hy/HFPathSimulatorLiveData/src')

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from hfpathsim.gui.main_window import MainWindow
from hfpathsim.input.siggen import SignalGenerator, WaveformType
from hfpathsim.output.audio import AudioOutputSink


def main():
    print("=" * 60)
    print("DASHBOARD AUDIO TEST")
    print("=" * 60)
    print("Testing with RTTY signal generator -> Watterson -> Audio Output")
    print("Listen for clean RTTY tones (should NOT be just clicks/noise)")
    print("=" * 60)
    print()

    app = QApplication(sys.argv)

    # Create signal generator
    siggen = SignalGenerator(WaveformType.RTTY, sample_rate_hz=8000.0)
    if not siggen.open():
        print("ERROR: Failed to open signal generator")
        return 1

    print(f"Signal generator: {siggen.waveform_type.value} at {siggen.sample_rate}Hz")

    # Create audio output sink
    audio_sink = AudioOutputSink(
        sample_rate_hz=8000.0,
        buffer_size=1048576,
        blocksize=256,
        latency="high",
    )

    # Create window
    window = MainWindow()
    window._input_source = siggen

    # Set Watterson model for audio-rate signals
    window._current_model = "watterson"
    window._channel_panel._model_combo.setCurrentText("Watterson TDL")
    print(f"Channel model: Watterson TDL")

    # Configure and enable audio output
    window._output_sink = audio_sink
    window._output_enabled = True
    if audio_sink.open():
        print(f"Audio output: Opened successfully")
    else:
        print("ERROR: Failed to open audio output")
        return 1

    # Auto-start processing after 500ms
    def auto_start():
        print("\n*** STARTING PROCESSING - LISTEN FOR RTTY TONES ***\n")
        window._start_processing()

    # Auto-stop after 5 seconds
    def auto_stop():
        print("\n*** STOPPING PROCESSING ***\n")
        window._stop_processing()
        print(f"Audio underruns: {audio_sink.underruns}")
        print(f"Final buffer fill: {audio_sink.buffer_fill:.1f}%")
        print("\nDid you hear clean RTTY tones? (not just clicks/noise)")
        app.quit()

    QTimer.singleShot(500, auto_start)
    QTimer.singleShot(5500, auto_stop)

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
