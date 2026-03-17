"""File-based output sinks for HF Path Simulator."""

import json
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np

from .base import OutputSink, OutputFormat


class FileOutputSink(OutputSink):
    """Output sink to file (WAV, raw binary, SigMF).

    Supports:
    - WAV files (stereo I/Q)
    - Raw binary files with configurable format
    - SigMF recordings (metadata + data)
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        sample_rate_hz: float = 48000.0,
        center_freq_hz: float = 0.0,
        output_format: OutputFormat = OutputFormat.COMPLEX64,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize file output sink.

        Args:
            filepath: Path to output file
            sample_rate_hz: Sample rate
            center_freq_hz: Center frequency (stored in SigMF metadata)
            output_format: Data format
            metadata: Additional metadata for SigMF
        """
        super().__init__(sample_rate_hz, center_freq_hz, output_format)

        self._filepath = Path(filepath)
        self._metadata = metadata or {}

        # File handles
        self._file = None
        self._wav_file = None

        # SigMF metadata accumulation
        self._sigmf_annotations = []
        self._start_time = None

    @property
    def filepath(self) -> Path:
        """Return output file path."""
        return self._filepath

    def open(self) -> bool:
        """Open the file for writing."""
        suffix = self._filepath.suffix.lower()

        try:
            # Ensure parent directory exists
            self._filepath.parent.mkdir(parents=True, exist_ok=True)

            if suffix == ".wav":
                return self._open_wav()
            elif suffix in [".sigmf-data", ".sigmf"]:
                return self._open_sigmf()
            else:
                return self._open_raw()

        except Exception as e:
            print(f"Error opening output file: {e}")
            return False

    def _open_wav(self) -> bool:
        """Open WAV file for writing."""
        self._wav_file = wave.open(str(self._filepath), "wb")

        # Stereo I/Q
        self._wav_file.setnchannels(2)

        # Set format based on output format
        if self._output_format in [OutputFormat.INT16_IQ, OutputFormat.COMPLEX64]:
            self._wav_file.setsampwidth(2)  # 16-bit
        elif self._output_format == OutputFormat.FLOAT32_IQ:
            self._wav_file.setsampwidth(4)  # 32-bit float
        elif self._output_format == OutputFormat.INT8_IQ:
            self._wav_file.setsampwidth(1)  # 8-bit
        else:
            self._wav_file.setsampwidth(2)  # Default 16-bit

        self._wav_file.setframerate(int(self._sample_rate))

        self._start_time = datetime.now(timezone.utc)
        self._is_open = True
        return True

    def _open_sigmf(self) -> bool:
        """Open SigMF recording for writing."""
        # Ensure proper extension
        base = str(self._filepath).replace(".sigmf-data", "").replace(".sigmf", "")
        data_path = Path(f"{base}.sigmf-data")

        self._filepath = data_path
        self._file = open(data_path, "wb")
        self._start_time = datetime.now(timezone.utc)
        self._is_open = True
        return True

    def _open_raw(self) -> bool:
        """Open raw binary file for writing."""
        self._file = open(self._filepath, "wb")
        self._start_time = datetime.now(timezone.utc)
        self._is_open = True
        return True

    def close(self):
        """Close file handles and finalize."""
        if self._wav_file:
            self._wav_file.close()
            self._wav_file = None

        if self._file:
            self._file.close()
            self._file = None

            # Write SigMF metadata if applicable
            if self._filepath.suffix == ".sigmf-data":
                self._write_sigmf_meta()

        self._is_open = False

    def _write_sigmf_meta(self):
        """Write SigMF metadata file."""
        base = str(self._filepath).replace(".sigmf-data", "")
        meta_path = Path(f"{base}.sigmf-meta")

        # Map format to SigMF datatype
        datatype_map = {
            OutputFormat.COMPLEX64: "cf32_le",
            OutputFormat.COMPLEX128: "cf64_le",
            OutputFormat.INT16_IQ: "ci16_le",
            OutputFormat.INT8_IQ: "cu8",
            OutputFormat.FLOAT32_IQ: "cf32_le",
        }

        metadata = {
            "global": {
                "core:datatype": datatype_map.get(self._output_format, "cf32_le"),
                "core:sample_rate": self._sample_rate,
                "core:version": "1.0.0",
                "core:description": self._metadata.get("description", "HF Path Simulator output"),
                "core:author": self._metadata.get("author", "HF Path Simulator"),
                "core:recorder": "hfpathsim",
                **{k: v for k, v in self._metadata.items() if k not in ["description", "author"]},
            },
            "captures": [
                {
                    "core:sample_start": 0,
                    "core:frequency": self._center_freq,
                    "core:datetime": self._start_time.isoformat() + "Z" if self._start_time else None,
                }
            ],
            "annotations": self._sigmf_annotations,
        }

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def write(self, samples: np.ndarray) -> int:
        """Write samples to file."""
        if not self._is_open:
            return 0

        samples = samples.astype(np.complex64)

        if self._wav_file:
            return self._write_wav(samples)
        elif self._file:
            return self._write_raw(samples)

        return 0

    def _write_wav(self, samples: np.ndarray) -> int:
        """Write samples to WAV file."""
        sample_width = self._wav_file.getsampwidth()

        if sample_width == 2:
            # 16-bit stereo I/Q
            i = (np.real(samples) * 32767).astype(np.int16)
            q = (np.imag(samples) * 32767).astype(np.int16)
            interleaved = np.zeros(len(samples) * 2, dtype=np.int16)
            interleaved[0::2] = i
            interleaved[1::2] = q
            self._wav_file.writeframes(interleaved.tobytes())

        elif sample_width == 4:
            # 32-bit float stereo I/Q
            i = np.real(samples).astype(np.float32)
            q = np.imag(samples).astype(np.float32)
            interleaved = np.zeros(len(samples) * 2, dtype=np.float32)
            interleaved[0::2] = i
            interleaved[1::2] = q
            self._wav_file.writeframes(interleaved.tobytes())

        elif sample_width == 1:
            # 8-bit unsigned stereo I/Q
            i = ((np.real(samples) + 1.0) * 127.5).astype(np.uint8)
            q = ((np.imag(samples) + 1.0) * 127.5).astype(np.uint8)
            interleaved = np.zeros(len(samples) * 2, dtype=np.uint8)
            interleaved[0::2] = i
            interleaved[1::2] = q
            self._wav_file.writeframes(interleaved.tobytes())

        self._total_samples_written += len(samples)
        return len(samples)

    def _write_raw(self, samples: np.ndarray) -> int:
        """Write samples to raw binary file."""
        data = self._convert_to_format(samples)
        self._file.write(data.tobytes())
        self._total_samples_written += len(samples)
        return len(samples)

    def available(self) -> int:
        """Return samples that can be written without blocking.

        Files can accept unlimited samples (until disk full).
        """
        return self._buffer_size

    def flush(self):
        """Flush buffered data to disk."""
        if self._wav_file:
            # wave module doesn't have explicit flush
            pass
        if self._file:
            self._file.flush()

    def add_annotation(
        self,
        sample_start: int,
        sample_count: int,
        label: str,
        comment: str = "",
    ):
        """Add SigMF annotation (only for SigMF files).

        Args:
            sample_start: Start sample of annotation
            sample_count: Number of samples
            label: Annotation label
            comment: Optional comment
        """
        annotation = {
            "core:sample_start": sample_start,
            "core:sample_count": sample_count,
            "core:label": label,
        }
        if comment:
            annotation["core:comment"] = comment

        self._sigmf_annotations.append(annotation)

    @property
    def duration_seconds(self) -> float:
        """Return duration of recorded data in seconds."""
        if self._sample_rate > 0:
            return self._total_samples_written / self._sample_rate
        return 0.0

    @property
    def file_size_bytes(self) -> int:
        """Return current file size in bytes."""
        return self._total_samples_written * self._bytes_per_sample()
