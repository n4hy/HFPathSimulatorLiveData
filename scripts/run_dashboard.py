#!/usr/bin/env python3
"""Convenience launcher for HF Path Simulator dashboard.

Usage:
    python run_dashboard.py                    # Launch with no input
    python run_dashboard.py siggen rtty        # Launch with RTTY signal generator
    python run_dashboard.py siggen ssb         # Launch with SSB voice signal
    python run_dashboard.py siggen psk31       # Launch with PSK31 signal

Signal Generator Options:
    rtty      - 45.45 baud FSK with 170 Hz shift
    ssb       - Simulated SSB voice (filtered noise with formants)
    psk31     - 31.25 baud BPSK with raised cosine shaping
"""

import sys
import argparse
from pathlib import Path

# Add src to path if running from repo
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))


WAVEFORM_DESCRIPTIONS = {
    "rtty": "RTTY (45.45 baud, 170 Hz shift)",
    "ssb": "SSB Voice (simulated speech)",
    "psk31": "PSK31 (31.25 baud BPSK)",
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HF Path Simulator Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Input source command")

    # Signal generator subcommand
    siggen_parser = subparsers.add_parser(
        "siggen", help="Use built-in signal generator"
    )
    siggen_parser.add_argument(
        "waveform",
        choices=["rtty", "ssb", "psk31"],
        help="Waveform type to generate",
    )
    siggen_parser.add_argument(
        "--sample-rate",
        type=float,
        default=8000.0,
        help="Sample rate in Hz (default: 8000)",
    )
    siggen_parser.add_argument(
        "--center-freq",
        type=float,
        default=1500.0,
        help="Center frequency in Hz (default: 1500)",
    )
    siggen_parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Signal duration in seconds (default: 60)",
    )

    return parser.parse_args()


def main():
    """Launch the HF Path Simulator dashboard."""
    args = parse_args()

    # Prepare signal generator if requested
    input_source = None
    if args.command == "siggen":
        from hfpathsim.input import create_signal_generator

        input_source = create_signal_generator(
            waveform_name=args.waveform,
            sample_rate_hz=args.sample_rate,
            center_freq_hz=args.center_freq,
            duration_sec=args.duration,
        )
        print(f"Signal generator: {WAVEFORM_DESCRIPTIONS[args.waveform]}")
        print(f"  Sample rate: {args.sample_rate} Hz")
        print(f"  Center freq: {args.center_freq} Hz")
        print(f"  Duration: {args.duration} seconds")

    # Launch the dashboard
    from PyQt6.QtWidgets import QApplication
    from hfpathsim.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("HF Path Simulator")
    app.setOrganizationName("HFPathSim")

    window = MainWindow()

    # Configure input source if provided
    if input_source is not None:
        if input_source.open():
            window._input_source = input_source
            window._statusbar.showMessage(
                f"Input: {WAVEFORM_DESCRIPTIONS[args.waveform]} @ "
                f"{input_source.sample_rate/1000:.1f} kHz"
            )
            # Set Watterson model for signal generator (audio-rate signals)
            # Vogler model is designed for wideband MHz signals, not narrowband audio
            window._current_model = "watterson"
            window._channel_panel._model_combo.setCurrentText("Watterson TDL")
            print(f"Channel model: Watterson TDL (optimal for audio-rate signals)")
        else:
            print("Warning: Failed to open signal generator")

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
