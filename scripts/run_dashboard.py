#!/usr/bin/env python3
"""Convenience launcher for HF Path Simulator dashboard."""

import sys
from pathlib import Path

# Add src to path if running from repo
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))


def main():
    """Launch the HF Path Simulator dashboard."""
    from hfpathsim.__main__ import main as app_main

    app_main()


if __name__ == "__main__":
    main()
