"""File-based input sources for HF Path Simulator."""

import json
from pathlib import Path
from typing import Optional, Union
import numpy as np
import wave

from .base import InputSource, InputFormat


class FileInputSource(InputSource):
    """Input source from file (WAV, raw binary, SigMF).

    Supports:
    - WAV files (mono or stereo I/Q)
    - Raw binary files with configurable format
    - SigMF recordings (metadata + data)
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        sample_rate_hz: float = 0.0,  # Auto-detect for WAV/SigMF
        center_freq_hz: float = 0.0,
        input_format: InputFormat = InputFormat.COMPLEX64,
        loop: bool = False,
    ):
        """Initialize file input source.

        Args:
            filepath: Path to input file
            sample_rate_hz: Sample rate (auto-detected for WAV/SigMF)
            center_freq_hz: Center frequency
            input_format: Data format for raw files
            loop: Whether to loop the file
        """
        super().__init__(sample_rate_hz, center_freq_hz, input_format)

        self._filepath = Path(filepath)
        self._loop = loop

        # File handles
        self._file = None
        self._wav_file = None

        # File info
        self._file_samples = 0
        self._current_position = 0

        # SigMF metadata
        self._sigmf_meta = None

    def open(self) -> bool:
        """Open the file for reading."""
        if not self._filepath.exists():
            print(f"File not found: {self._filepath}")
            return False

        suffix = self._filepath.suffix.lower()

        try:
            if suffix == ".wav":
                return self._open_wav()
            elif suffix in [".sigmf-data", ".sigmf-meta"]:
                return self._open_sigmf()
            else:
                return self._open_raw()

        except Exception as e:
            print(f"Error opening file: {e}")
            return False

    def _open_wav(self) -> bool:
        """Open WAV file."""
        self._wav_file = wave.open(str(self._filepath), "rb")

        # Get parameters
        channels = self._wav_file.getnchannels()
        sample_width = self._wav_file.getsampwidth()
        frame_rate = self._wav_file.getframerate()
        n_frames = self._wav_file.getnframes()

        # Auto-detect sample rate
        if self._sample_rate == 0:
            self._sample_rate = float(frame_rate)

        # Determine format
        if channels == 2 and sample_width == 2:
            self._input_format = InputFormat.INT16_IQ
        elif channels == 2 and sample_width == 4:
            self._input_format = InputFormat.FLOAT32_IQ
        elif channels == 1 and sample_width == 4:
            # Mono float - could be magnitude only
            self._input_format = InputFormat.FLOAT32_IQ
        else:
            print(f"Unsupported WAV format: {channels}ch, {sample_width*8}bit")
            return False

        self._file_samples = n_frames
        self._is_open = True
        return True

    def _open_sigmf(self) -> bool:
        """Open SigMF recording."""
        # Handle both .sigmf-meta and .sigmf-data
        base = str(self._filepath).rsplit(".sigmf-", 1)[0]
        meta_path = Path(f"{base}.sigmf-meta")
        data_path = Path(f"{base}.sigmf-data")

        if not meta_path.exists() or not data_path.exists():
            print(f"SigMF files not found: {base}")
            return False

        # Read metadata
        with open(meta_path, "r") as f:
            self._sigmf_meta = json.load(f)

        global_meta = self._sigmf_meta.get("global", {})
        captures = self._sigmf_meta.get("captures", [{}])

        # Extract parameters
        self._sample_rate = float(
            global_meta.get("core:sample_rate", self._sample_rate)
        )
        self._center_freq = float(
            captures[0].get("core:frequency", self._center_freq)
        )

        # Determine data type
        datatype = global_meta.get("core:datatype", "cf32_le")
        if datatype in ["cf32_le", "cf32"]:
            self._input_format = InputFormat.COMPLEX64
            bytes_per_sample = 8
        elif datatype in ["ci16_le", "ci16"]:
            self._input_format = InputFormat.INT16_IQ
            bytes_per_sample = 4
        elif datatype in ["ci8", "cu8"]:
            self._input_format = InputFormat.INT8_IQ
            bytes_per_sample = 2
        else:
            print(f"Unsupported SigMF datatype: {datatype}")
            return False

        # Open data file
        self._file = open(data_path, "rb")
        file_size = data_path.stat().st_size
        self._file_samples = file_size // bytes_per_sample

        self._is_open = True
        return True

    def _open_raw(self) -> bool:
        """Open raw binary file."""
        if self._sample_rate == 0:
            print("Sample rate required for raw files")
            return False

        self._file = open(self._filepath, "rb")
        file_size = self._filepath.stat().st_size

        # Calculate samples based on format
        fmt = self._input_format
        if fmt == InputFormat.COMPLEX64:
            bytes_per_sample = 8
        elif fmt == InputFormat.COMPLEX128:
            bytes_per_sample = 16
        elif fmt == InputFormat.INT16_IQ:
            bytes_per_sample = 4
        elif fmt == InputFormat.INT8_IQ:
            bytes_per_sample = 2
        elif fmt == InputFormat.FLOAT32_IQ:
            bytes_per_sample = 8
        else:
            bytes_per_sample = 8

        self._file_samples = file_size // bytes_per_sample
        self._is_open = True
        return True

    def close(self):
        """Close file handles."""
        if self._wav_file:
            self._wav_file.close()
            self._wav_file = None

        if self._file:
            self._file.close()
            self._file = None

        self._is_open = False

    def read(self, num_samples: int) -> Optional[np.ndarray]:
        """Read samples from file."""
        if not self._is_open:
            return None

        if self._wav_file:
            return self._read_wav(num_samples)
        else:
            return self._read_raw(num_samples)

    def _read_wav(self, num_samples: int) -> Optional[np.ndarray]:
        """Read from WAV file."""
        frames = self._wav_file.readframes(num_samples)

        if len(frames) == 0:
            if self._loop:
                self._wav_file.rewind()
                self._current_position = 0
                frames = self._wav_file.readframes(num_samples)
            else:
                return None

        # Convert based on format
        sample_width = self._wav_file.getsampwidth()
        channels = self._wav_file.getnchannels()

        if sample_width == 2:
            data = np.frombuffer(frames, dtype=np.int16)
        elif sample_width == 4:
            data = np.frombuffer(frames, dtype=np.float32)
        else:
            data = np.frombuffer(frames, dtype=np.int16)

        # Reshape for stereo I/Q
        if channels == 2:
            data = data.reshape(-1, 2)
            if sample_width == 2:
                samples = (
                    data[:, 0].astype(np.float32) / 32768.0
                    + 1j * data[:, 1].astype(np.float32) / 32768.0
                )
            else:
                samples = data[:, 0] + 1j * data[:, 1]
        else:
            samples = data.astype(np.complex64)

        self._current_position += len(samples)
        self._total_samples_read += len(samples)

        return samples.astype(np.complex64)

    def _read_raw(self, num_samples: int) -> Optional[np.ndarray]:
        """Read from raw binary file."""
        fmt = self._input_format

        # Determine bytes to read
        if fmt == InputFormat.COMPLEX64:
            dtype = np.complex64
            bytes_to_read = num_samples * 8
        elif fmt == InputFormat.COMPLEX128:
            dtype = np.complex128
            bytes_to_read = num_samples * 16
        elif fmt == InputFormat.INT16_IQ:
            dtype = np.int16
            bytes_to_read = num_samples * 4
        elif fmt == InputFormat.INT8_IQ:
            dtype = np.uint8
            bytes_to_read = num_samples * 2
        elif fmt == InputFormat.FLOAT32_IQ:
            dtype = np.float32
            bytes_to_read = num_samples * 8
        else:
            return None

        data = self._file.read(bytes_to_read)

        if len(data) == 0:
            if self._loop:
                self._file.seek(0)
                self._current_position = 0
                data = self._file.read(bytes_to_read)
            else:
                return None

        raw = np.frombuffer(data, dtype=dtype)
        samples = self._convert_format(raw)

        self._current_position += len(samples)
        self._total_samples_read += len(samples)

        return samples

    def available(self) -> int:
        """Return samples available without blocking."""
        if not self._is_open:
            return 0

        remaining = self._file_samples - self._current_position
        if self._loop and remaining == 0:
            return self._file_samples
        return remaining

    def seek(self, sample_position: int):
        """Seek to a sample position in the file."""
        if not self._is_open:
            return

        if self._wav_file:
            self._wav_file.setpos(sample_position)
        elif self._file:
            # Calculate byte offset
            fmt = self._input_format
            if fmt == InputFormat.COMPLEX64:
                byte_offset = sample_position * 8
            elif fmt == InputFormat.COMPLEX128:
                byte_offset = sample_position * 16
            elif fmt == InputFormat.INT16_IQ:
                byte_offset = sample_position * 4
            elif fmt == InputFormat.INT8_IQ:
                byte_offset = sample_position * 2
            elif fmt == InputFormat.FLOAT32_IQ:
                byte_offset = sample_position * 8
            else:
                byte_offset = sample_position * 8

            self._file.seek(byte_offset)

        self._current_position = sample_position

    @property
    def duration_seconds(self) -> float:
        """Return file duration in seconds."""
        if self._sample_rate > 0:
            return self._file_samples / self._sample_rate
        return 0.0

    @property
    def position_seconds(self) -> float:
        """Return current position in seconds."""
        if self._sample_rate > 0:
            return self._current_position / self._sample_rate
        return 0.0
