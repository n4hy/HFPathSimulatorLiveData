"""Abstract base class for output data sinks."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional
import numpy as np


class OutputFormat(Enum):
    """Supported output data formats."""

    COMPLEX64 = "complex64"  # numpy complex64 (float32 I/Q pairs)
    COMPLEX128 = "complex128"  # numpy complex128 (float64 I/Q pairs)
    INT16_IQ = "int16_iq"  # Interleaved 16-bit I/Q
    INT8_IQ = "int8_iq"  # Interleaved 8-bit I/Q
    FLOAT32_IQ = "float32_iq"  # Interleaved 32-bit float I/Q


class OutputSink(ABC):
    """Abstract base class for output data sinks.

    All output sinks accept complex samples at a known sample rate
    and write them to a destination (network, file, audio, SDR).
    """

    def __init__(
        self,
        sample_rate_hz: float,
        center_freq_hz: float = 0.0,
        output_format: OutputFormat = OutputFormat.COMPLEX64,
        buffer_size: int = 65536,
    ):
        """Initialize output sink.

        Args:
            sample_rate_hz: Sample rate in Hz
            center_freq_hz: Center frequency in Hz (for RF sinks)
            output_format: Output data format
            buffer_size: Internal buffer size in samples
        """
        self._sample_rate = sample_rate_hz
        self._center_freq = center_freq_hz
        self._output_format = output_format
        self._buffer_size = buffer_size
        self._is_open = False
        self._total_samples_written = 0

    @property
    def sample_rate(self) -> float:
        """Return sample rate in Hz."""
        return self._sample_rate

    @property
    def center_frequency(self) -> float:
        """Return center frequency in Hz."""
        return self._center_freq

    @property
    def output_format(self) -> OutputFormat:
        """Return output data format."""
        return self._output_format

    @property
    def is_open(self) -> bool:
        """Return whether the sink is open and ready."""
        return self._is_open

    @property
    def total_samples_written(self) -> int:
        """Return total number of samples written."""
        return self._total_samples_written

    @abstractmethod
    def open(self) -> bool:
        """Open the output sink.

        Returns:
            True if successfully opened
        """
        pass

    @abstractmethod
    def close(self):
        """Close the output sink and release resources."""
        pass

    @abstractmethod
    def write(self, samples: np.ndarray) -> int:
        """Write samples to the sink.

        Args:
            samples: Complex64 numpy array of samples

        Returns:
            Number of samples actually written
        """
        pass

    @abstractmethod
    def available(self) -> int:
        """Return number of samples that can be written without blocking."""
        pass

    def _convert_to_format(self, samples: np.ndarray) -> np.ndarray:
        """Convert complex64 samples to output format.

        Args:
            samples: Complex64 numpy array

        Returns:
            Data in output format
        """
        fmt = self._output_format

        if fmt == OutputFormat.COMPLEX64:
            return samples.astype(np.complex64)

        elif fmt == OutputFormat.COMPLEX128:
            return samples.astype(np.complex128)

        elif fmt == OutputFormat.INT16_IQ:
            # Interleaved I/Q as int16
            i = np.real(samples) * 32767.0
            q = np.imag(samples) * 32767.0
            interleaved = np.zeros(len(samples) * 2, dtype=np.int16)
            interleaved[0::2] = np.clip(i, -32768, 32767).astype(np.int16)
            interleaved[1::2] = np.clip(q, -32768, 32767).astype(np.int16)
            return interleaved

        elif fmt == OutputFormat.INT8_IQ:
            # Interleaved I/Q as uint8 (RTL-SDR style, centered at 127.5)
            i = np.real(samples) * 127.5 + 127.5
            q = np.imag(samples) * 127.5 + 127.5
            interleaved = np.zeros(len(samples) * 2, dtype=np.uint8)
            interleaved[0::2] = np.clip(i, 0, 255).astype(np.uint8)
            interleaved[1::2] = np.clip(q, 0, 255).astype(np.uint8)
            return interleaved

        elif fmt == OutputFormat.FLOAT32_IQ:
            # Interleaved I/Q as float32
            interleaved = np.zeros(len(samples) * 2, dtype=np.float32)
            interleaved[0::2] = np.real(samples).astype(np.float32)
            interleaved[1::2] = np.imag(samples).astype(np.float32)
            return interleaved

        else:
            raise ValueError(f"Unknown format: {fmt}")

    def _bytes_per_sample(self) -> int:
        """Return bytes per complex sample for current format."""
        fmt = self._output_format
        if fmt == OutputFormat.COMPLEX64:
            return 8
        elif fmt == OutputFormat.COMPLEX128:
            return 16
        elif fmt == OutputFormat.INT16_IQ:
            return 4
        elif fmt == OutputFormat.INT8_IQ:
            return 2
        elif fmt == OutputFormat.FLOAT32_IQ:
            return 8
        else:
            return 8

    def flush(self):
        """Flush any buffered data. Override in subclasses if needed."""
        pass

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.flush()
        self.close()
        return False
