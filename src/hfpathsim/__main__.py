"""Entry point for HF Path Simulator."""

import sys


def main():
    """Launch the HF Path Simulator dashboard."""
    from PyQt6.QtWidgets import QApplication
    from hfpathsim.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("HF Path Simulator")
    app.setOrganizationName("HFPathSim")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
